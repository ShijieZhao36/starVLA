#!/usr/bin/env python3
"""Read-only paired-noise tracing with optional frozen-feature export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['repo', 'config', 'cache', 'checkpoint', 'output', 'helper-dir']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--readout-rounds', type=int, default=96)
    p.add_argument('--times', default='0.1,0.5,0.9')
    p.add_argument('--seed', type=int, default=202611040)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo)); sys.path.insert(1, str(a.helper_dir))
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from train_ab import load_cache, prepare_batch, velocity_forward, seed_everything
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device); torch.cuda.set_device(device)
    cfg = OmegaConf.load(a.config)
    data = load_cache(a.cache, cfg)
    n, horizon = data['actions'].shape[:2]
    assert n == 8 and horizon == 8
    a.output.mkdir(parents=True, exist_ok=False)
    torch.distributions.Distribution.set_default_validate_args(False)
    state = torch.load(a.checkpoint, mmap=True, map_location='cpu', weights_only=True)
    with torch.device('meta'):
        head = LayerwiseFlowmatchingActionHead(cfg)
    head.load_state_dict(state, assign=True, strict=True)
    del state
    head = head.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    versions = {k: (v.data_ptr(), v._version) for k, v in head.named_parameters()}
    checkpoint_stat = (a.checkpoint.stat().st_size, a.checkpoint.stat().st_mtime_ns)
    vl, action, mask, attention = prepare_batch(data, torch.arange(n), device, repeat=2)
    # Match prior training input quantization; FP32 operator path prevents small
    # feature differences being hidden by additional BF16 rounding in this probe.
    vl = [x.float() for x in vl]
    assert all(torch.equal(x[:n], x[n:]) for x in vl)
    assert torch.equal(attention[:n], attention[n:])
    valid = mask[0, 0].bool()
    assert torch.equal(mask.bool(), valid[None, None].expand_as(mask))
    valid_indices = valid.nonzero().flatten()
    assert len(valid_indices) == 48
    times = [float(t) for t in a.times.split(',')]
    feature_stages = ['action_encoder', 'block_05.output', 'block_09.output',
        'block_11.output', 'block_15.output', 'block_23.output', 'block_35.output',
        'decoder_after_input_norm', 'decoder_pre_relu', 'decoder_post_relu', 'decoder_output']
    features = {k: [] for k in feature_stages}
    targets, native_deltas = [], []
    stat_names = ['rms', 'delta_rms', 'delta_relative_rms', 'centered_pair_cosine',
                  'common_rms', 'positive_fraction', 'paired_exactly_equal_fraction', 'max_abs', 'finite']
    context = {'collect': False, 'stats': {}, 'features': {}, 'branch': {}, 'diagnostics': {}}
    handles = []

    def rms(x):
        return x.float().square().mean().sqrt()

    def part_stats(x):
        x = x.detach().float()
        first, second = x[:n], x[n:]
        delta = second - first
        xc = first - first.mean(-1, keepdim=True)
        yc = second - second.mean(-1, keepdim=True)
        scale, ds = rms(x), rms(delta)
        return torch.stack([scale, ds, ds / scale.clamp_min(1e-12),
            F.cosine_similarity(xc, yc, dim=-1).mean(), rms((first + second) / 2),
            (x > 0).float().mean(), (first == second).float().mean(),
            x.abs().max(), torch.isfinite(x).all().float()])

    def record(name, value, **meta):
        if isinstance(value, tuple): value = value[0]
        assert value.ndim == 3 and value.shape[0] == 2*n, (name, value.shape)
        parts = {'action_tokens': part_stats(value[:, -horizon:])}
        if value.shape[1] > horizon:
            parts['prefix_tokens'] = part_stats(value[:, :-horizon])
        context['stats'][name] = ({'shape': list(value.shape), 'dtype': str(value.dtype), **meta}, parts)
        if name in ['decoder_after_input_norm','decoder_pre_relu','decoder_post_relu','decoder_output']:
            context['branch'][name] = value[:, -horizon:].detach().float()
        if context['collect'] and name in features:
            x = value[:, -horizon:].detach().float()
            context['features'][name] = (x[n:] - x[:n]).reshape(n*horizon, -1).cpu()

    def branch_relation(name, before, update):
        before, update = before[:, -horizon:].float(), update[:, -horizon:].float()
        db, du = before[n:] - before[:n], update[n:] - update[:n]
        cb, cu = (before[n:] + before[:n])/2, (update[n:] + update[:n])/2
        cosine = lambda x,y: F.cosine_similarity(x.reshape(1,-1), y.reshape(1,-1)).squeeze()
        context['diagnostics'][name] = torch.stack([rms(update)/rms(before).clamp_min(1e-12),
            rms(du)/rms(db).clamp_min(1e-12), cosine(db, du), cosine(cb, cu)])

    handles.append(head.action_encoder.register_forward_hook(lambda m,i,o: record('action_encoder', o)))
    for idx, block in enumerate(head.model.transformer_blocks):
        name = f'block_{idx:02d}'
        kind = 'self' if idx % 2 and head.model._interleave_self_attention else 'cross'
        meta = {'index': idx, 'attention_type': kind}
        def before(m, inputs, kwargs, name=name, meta=meta):
            x = inputs[0] if inputs else kwargs['hidden_states']
            context['branch'][name] = {'input': x.detach()}
            record(name+'.input', x, component='input', **meta)
        def attn(m, inputs, out, name=name, meta=meta):
            context['branch'][name]['attn'] = out.detach()
            record(name+'.attn_output', out, component='attn_output', **meta)
            branch_relation(name+'.attn', context['branch'][name]['input'], out)
        def after_attn(m, inputs, name=name, meta=meta):
            context['branch'][name]['after_attn'] = inputs[0].detach()
            record(name+'.after_attn', inputs[0], component='after_attn', **meta)
        def ff(m, inputs, out, name=name, meta=meta):
            record(name+'.ff_output', out, component='ff_output', **meta)
            branch_relation(name+'.ff', context['branch'][name]['after_attn'], out)
        handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(block.attn1.register_forward_hook(attn))
        handles.append(block.norm3.register_forward_pre_hook(after_attn))
        handles.append(block.ff.register_forward_hook(ff))
        handles.append(block.register_forward_hook(lambda m,i,o,name=name,meta=meta:
            record(name+'.output', o, component='output', **meta)))
    handles.append(head.action_decoder.register_forward_pre_hook(lambda m,i: record('decoder_input', i[0])))
    handles.append(head.action_decoder.layer1.register_forward_pre_hook(lambda m,i: record('decoder_after_input_norm', i[0])))
    handles.append(head.action_decoder.layer1.register_forward_hook(lambda m,i,o: record('decoder_pre_relu', o)))
    handles.append(head.action_decoder.layer2.register_forward_pre_hook(lambda m,i: record('decoder_post_relu', i[0])))
    handles.append(head.action_decoder.layer2.register_forward_hook(lambda m,i,o: record('decoder_output', o)))

    def serialize_stats():
        entries, names = [], []
        for name, (_, parts) in context['stats'].items():
            for part, values in parts.items():
                names.append((name, part)); entries.append(values)
        values = torch.stack(entries).cpu().tolist()
        out = {name: dict(meta) for name, (meta,_) in context['stats'].items()}
        for (name, part), values in zip(names, values):
            out[name][part] = dict(zip(stat_names, values))
        return out

    def output_metrics(pred, target, noise):
        pred, target = pred[..., valid].float(), target[..., valid].float()
        dp = pred[n:] - pred[:n]
        dt = -(noise[n:, :, valid].float() - noise[:n, :, valid].float())
        mse = (pred-target).square().mean()
        energy = dt.square().mean()
        cos = F.cosine_similarity(dp.reshape(1,-1),dt.reshape(1,-1)).squeeze()
        dot = (dp*dt).sum(); pp = dp.square().sum().clamp_min(1e-20)
        scale = dot/pp
        metrics = {'velocity_masked_mse': mse.item(),
            'noise_delta_relative_mse': ((dp-dt).square().mean()/energy).item(),
            'noise_delta_cosine': cos.item(), 'noise_delta_pred_rms': rms(dp).item(),
            'noise_delta_target_rms': rms(dt).item(), 'best_scalar_gain': scale.item(),
            'best_scalar_relative_mse': ((scale*dp-dt).square().mean()/energy).item(),
            'per_observation_noise_relative_mse': (((dp-dt).square().mean((1,2))) / dt.square().mean((1,2))).cpu().tolist()}
        metrics['native_target_delta_rounding_rms'] = rms((target[n:]-target[:n])-dt).item()
        z = context['branch']['decoder_pre_relu']
        z0,z1=z[:n],z[n:]
        act=context['branch']['decoder_post_relu']
        da=act[n:]-act[:n]
        both=(z0>0)&(z1>0); switch=(z0>0)^(z1>0)
        w2=head.action_decoder.layer2.weight[valid]
        for label, selection in [('both_active',both),('switching',switch)]:
            contrib=F.linear(da*selection,w2)
            metrics['decoder_'+label+'_fraction']=selection.float().mean().item()
            metrics['decoder_'+label+'_output_delta_rms']=rms(contrib).item()
            metrics['decoder_'+label+'_target_dot_over_energy']=((contrib*dt).mean()/energy).item()
        metrics['decoder_both_inactive_fraction']=((z0<=0)&(z1<=0)).float().mean().item()
        metrics['decoder_linear2_delta_identity_max_error']=(F.linear(da,w2)-dp).abs().max().item()
        ln=context['branch']['decoder_after_input_norm']
        metrics['decoder_linear1_delta_identity_max_error']=(F.linear(ln[n:]-ln[:n],head.action_decoder.layer1.weight)-(z1-z0)).abs().max().item()
        return metrics, dp.reshape(n*horizon,-1).cpu(), dt.reshape(n*horizon,-1).cpu()

    metadata = {'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'checkpoint_stat_before': checkpoint_stat, 'parameter_dtype':'float32', 'compute':'FP32; TF32 disabled',
        'input_policy':'native BF16 noise/actions/time and xt arithmetic, then exact cast to FP32',
        'num_observations':n,'action_horizon':horizon,'valid_indices':valid_indices.cpu().tolist(),
        'feature_stages':feature_stages,'times':times,'readout_split':'rounds 0:64 train, 64:80 validation, 80:96 test',
        'readonly':True,'torch_version':str(torch.__version__),
        'git_commit':subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()}
    metadata['condition_valid_token_rms_by_layer'] = [rms(x[:n][attention[:n]]).item() for x in vl]
    metadata['condition_all_token_rms_by_layer'] = [rms(x[:n]).item() for x in vl]
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    start = time.monotonic()
    rows = []
    try:
        with torch.inference_mode(), (a.output/'metrics.jsonl').open('w', buffering=1) as log:
            for time_index,t in enumerate(times):
                rounds = max(a.rounds,a.readout_rounds) if t == .5 else a.rounds
                for rnd in range(-1,rounds):
                    seed = a.seed + time_index*100000 + max(rnd,0)
                    seed_everything(seed)
                    clean = action*mask
                    noise = torch.randn_like(action)*mask
                    if rnd == -1: noise[n:] = noise[:n]
                    time_q = torch.full((2*n,1,1),t,device=device,dtype=torch.bfloat16)
                    xt = ((1-time_q)*noise + time_q*clean).float()
                    target = clean-noise
                    bucket = (time_q[:,0,0]*head.num_timestep_buckets).long()
                    context.update(collect=(t==.5 and rnd>=0 and a.readout_rounds>0),stats={},features={},branch={},diagnostics={})
                    pred = velocity_forward(head,vl,xt,bucket,attention)
                    if rnd == -1:
                        acts = serialize_stats()
                        max_delta = max(v['action_tokens']['delta_rms'] for v in acts.values())
                        prior = pred.clone()
                        pred2 = velocity_forward(head,vl,xt,bucket,attention)
                        repeat_error = (pred2-prior).abs().max().item()
                        record_row = {'kind':'duplicate_control','checkpoint':a.name,'requested_time':t,
                            'max_action_stage_delta_rms':max_delta,'repeat_prediction_max_abs':repeat_error}
                        log.write(json.dumps(record_row)+'\n'); print(json.dumps(record_row),flush=True)
                        assert max_delta<1e-6 and repeat_error<1e-6, record_row
                        continue
                    met, dp, dt = output_metrics(pred,target,noise)
                    diagnostics = {k:dict(zip(['update_to_residual_rms','delta_update_to_residual_rms',
                        'delta_update_residual_cosine','common_update_residual_cosine'],v.cpu().tolist()))
                        for k,v in context['diagnostics'].items()}
                    row = {'kind':'paired_probe','checkpoint':a.name,'branch':'fp32','requested_time':t,
                        'quantized_time':time_q[0,0,0].item(),'round':rnd,'seed':seed,'num_observations':n,
                        'metrics':met,'activations':serialize_stats(),'diagnostics':diagnostics}
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    rows.append({k:v for k,v in row.items() if k not in ['activations','diagnostics']})
                    if context['collect']:
                        assert set(context['features'])==set(feature_stages)
                        for k in features: features[k].append(context['features'][k])
                        targets.append(dt); native_deltas.append(dp)
                    if rnd%8==0 or rnd==rounds-1:
                        print(json.dumps({'time':t,'round':rnd,**met,'elapsed_seconds':time.monotonic()-start}),flush=True)
            if targets:
                torch.save({'features':{k:torch.stack(v) for k,v in features.items()},'target':torch.stack(targets),
                    'native_prediction_delta':torch.stack(native_deltas),'metadata':metadata},a.output/'paired_features.pt')
    finally:
        for h in handles: h.remove()
    assert versions == {k:(v.data_ptr(),v._version) for k,v in head.named_parameters()}
    assert checkpoint_stat == (a.checkpoint.stat().st_size,a.checkpoint.stat().st_mtime_ns)
    summary = {'status':'complete','model_parameters_unchanged':True,'checkpoint_file_unchanged':True,
        'num_probe_forwards':len(rows),'readout_rounds':len(targets),'elapsed_seconds':time.monotonic()-start,
        'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/2**30,'results':rows}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__ == '__main__':
    main()
