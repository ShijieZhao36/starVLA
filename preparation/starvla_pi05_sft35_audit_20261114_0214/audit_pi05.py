"""Diagnostic only: StarVLA PI05 + the repository's full SFT35 dataloader."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors import safe_open

REPO = Path('/mnt/vla-pretrain/zh1112/workspace/starVLA_socks_sft35_abs_h50_action5e5_40k_20261112')
OUT = Path(__file__).resolve().parent
OLD_RUN = Path('/mnt/vla-pretrain/zh1112/checkpoints/starVLA/socks_sft35_abs_h50_letterbox_pi05_ln_a5e5_save500_8gpu_40k_20261113')
ASSETS = Path('/mnt/vla-pretrain/zh1112/pi05_assets')
WEIGHTS = ASSETS / 'pi05_base_modelscope/model.safetensors'
sys.path.insert(0, str(REPO))
os.chdir(REPO)

REPORT = {'diagnostic_only': True, 'repo': str(REPO), 'python': sys.executable,
          'formal_training_started': False, 'source_code_modified': False}


def save(stage, **items):
    REPORT.update(items)
    REPORT['stage'] = stage
    (OUT / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'stage': stage, **items}, ensure_ascii=False), flush=True)


def shape_summary(example):
    return {'keys': sorted(example), 'action_shape': list(example['action'].shape),
            'state_shape': list(example['state'].shape) if 'state' in example else None,
            'image_sizes': [list(im.size) for im in example['image']],
            'action_min': float(example['action'].min()),
            'action_max': float(example['action'].max()), 'language': example['lang']}


def main():
    Accelerator()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    from starVLA.dataloader import build_dataloader
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
    from starVLA.training.training_telemetry import GROUP_LABELS
    from starVLA.training.action_metrics import compute_action_metrics

    files = ['starVLA/model/framework/VLM4A/PI05.py', 'starVLA/model/framework/VLM4A/PI0.py',
             'starVLA/model/modules/action_model/OpenPI_ActionHead.py',
             'starVLA/model/modules/vlm/OpenPIPaliGemma.py', 'starVLA/dataloader/__init__.py',
             'starVLA/dataloader/gr00t_sharded.py', 'starVLA/dataloader/gr00t_sharded_processor.py',
             'starVLA/training/train_starvla.py', 'starVLA/training/training_telemetry.py']
    save('imports_ok', source_hashes={p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in files})
    old = OmegaConf.load(OLD_RUN / 'config.full.yaml')
    old.output_dir = str(OUT / 'old_loader')
    old.datasets.vla_data.per_device_batch_size = 1
    old.datasets.vla_data.num_workers = 0
    Path(old.output_dir).mkdir(exist_ok=True)
    baseline = build_dataloader(old, dataset_py='gr00t_sharded')
    old_iter = iter(baseline)
    old_example = next(old_iter)[0]
    old_summary = shape_summary(old_example)
    del old_iter, baseline
    gc.collect()
    save('old_repository_dataloader_ok', old_loader=old_summary)

    cfg = OmegaConf.create(OmegaConf.to_container(old, resolve=True))
    cfg.output_dir = str(OUT / 'pi05_loader')
    Path(cfg.output_dir).mkdir(exist_ok=True)
    cfg.framework = {'name': 'PI05', 'precision': 'bfloat16', 'action_dim': 32,
                     'action_horizon': 50, 'max_state_dim': 32,
                     'discrete_state_input': True, 'max_token_len': 200,
                     'num_inference_steps': 10, 'image_resolution': [224, 224],
                     'image_keys': ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'],
                     'tokenizer': {'model_path': str(ASSETS / 'paligemma_tokenizer.model'), 'pad_state_value': -2.0},
                     'paligemma': {'model_name_or_path': ''}}
    cfg.datasets.vla_data.canonical_layout = None
    cfg.datasets.vla_data.canonical_robots = None
    cfg.datasets.vla_data.include_state = True
    cfg.trainer.pretrained_checkpoint = str(WEIGHTS)
    cfg.trainer.is_resume = False
    cfg.trainer.learning_rate = {'base': 1e-5, 'vlm': 1e-5, 'action_head': 5e-5}
    OmegaConf.save(cfg, OUT / 'diagnostic_config.yaml')
    loader = build_dataloader(cfg, dataset_py='gr00t_sharded')
    mixture = loader.dataset.mixture
    entries = DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix]
    expected = (REPO / 'examples/pretrain/SocksSFT35H50/train_files/data_registry/requested_datasets.txt').read_text().splitlines()
    aliases = [row[0].split('/')[0].removeprefix('socks_new_v1__') for row in entries]
    assert aliases == expected and len(entries) == 35
    iterator = iter(loader)
    examples = next(iterator)
    assert examples[0]['action'].shape == (50, 14)
    assert examples[0]['state'].shape == (1, 14)
    assert np.isfinite(examples[0]['action']).all() and np.isfinite(examples[0]['state']).all()
    assert len(examples[0]['image']) == 3
    old_stats = json.loads((OLD_RUN / 'dataset_statistics.json').read_text())
    new_stats = json.loads((Path(cfg.output_dir) / 'dataset_statistics.json').read_text())
    old_action = next(v['action'] for v in old_stats.values() if 'action' in v)
    new_action = next(v['action'] for v in new_stats.values() if 'action' in v)
    stats_diff = {k: float(np.max(np.abs(np.asarray(old_action[k]) - np.asarray(new_action[k])))) for k in ['q01', 'q99']}
    save('pi05_repository_dataloader_ok', dataset_count=len(entries), datasets=aliases,
         sampling_weights=[r[1] for r in entries], valid_sample_mass=float(sum(r[1] for r in entries)),
         virtual_length=len(loader.dataset), loader=shape_summary(examples[0]),
         stats_action_bounds_max_diff_from_old=stats_diff, statistics_tags=list(new_stats))

    with torch.device('meta'):
        meta_model = build_framework(cfg)
    meta_state = meta_model.state_dict()
    with safe_open(str(WEIGHTS), framework='pt', device='cpu') as f:
        raw = {k: torch.empty(tuple(f.get_slice(k).get_shape()), device='meta') for k in f.keys()}
    normalized = meta_model._normalize_state_dict(raw)
    missing = [k for k in meta_state if k not in normalized]
    mismatch = {k: [list(v.shape), list(meta_state[k].shape)] for k,v in normalized.items() if k in meta_state and v.shape != meta_state[k].shape}
    unexpected = [k for k in normalized if k not in meta_state]
    real_missing = [k for k in missing if not meta_model._is_expected_openpi_missing_key(k)]
    old_groups = build_param_lr_groups(meta_model, old)
    new_groups = build_param_lr_groups(meta_model, cfg)
    groups = lambda gs: [{'name': g['name'], 'lr': g['lr'], 'numel': sum(p.numel() for p in g['params'])} for g in gs]
    save('weight_shapes_audited', checkpoint_bytes=WEIGHTS.stat().st_size,
         checkpoint_tensors=len(raw), normalized_tensors=len(normalized), model_tensors=len(meta_state),
         real_missing=real_missing, expected_missing=missing, shape_mismatch=mismatch,
         unexpected_checkpoint_keys=unexpected, old_lr_groups_on_pi05=groups(old_groups),
         diagnostic_pi05_lr_groups=groups(new_groups), telemetry_group_labels=GROUP_LABELS,
         current_trainer_incompatible_groups=[g['name'] for g in new_groups if g['name'] not in GROUP_LABELS],
         current_trainer_missing_action_model_config=not hasattr(cfg.framework, 'action_model'))
    assert not real_missing and not mismatch
    try:
        meta_model._prepare_examples([old_example], include_actions=True)
    except ValueError as exc:
        REPORT['old_90d_loader_rejected_by_pi05'] = str(exc)
    else:
        raise AssertionError('Expected old 90D sample to fail PI05 shape contract')
    try:
        compute_action_metrics(np.zeros((1, 50, 32), dtype=np.float32), examples)
    except ValueError as exc:
        REPORT['current_trainer_probe_dimension_error'] = str(exc)
    del meta_model, meta_state, raw, normalized, old_groups, new_groups
    gc.collect()

    active = os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').read().strip()
    if active:
        raise RuntimeError('GPU now occupied; diagnostic aborted: ' + active)
    start = time.monotonic()
    model = build_framework(cfg)
    model.load_pretrained_checkpoint(str(WEIGHTS))
    model.to('cuda:0')
    prepared = model._prepare_examples(examples, include_actions=True)
    assert prepared['action'].shape == (1, 50, 32)
    assert prepared['state'].shape == (1, 1, 32)
    assert torch.all(prepared['action'][..., 14:] == 0)
    assert torch.all(prepared['state'][..., 14:] == -2)
    token_state = prepared['state'][0].squeeze(0)
    ta, _ = model.tokenizer.tokenize(examples[0]['lang'], token_state)
    tb, _ = model.tokenizer.tokenize(examples[0]['lang'], token_state + 0.1)
    assert not np.array_equal(ta, tb)
    save('model_loaded', loading_seconds=time.monotonic()-start,
         prepared_image_shape=list(prepared['image'].shape),
         prepared_state_shape=list(prepared['state'].shape), prepared_action_shape=list(prepared['action'].shape),
         normalized_image_range=[float(prepared['image'].min()), float(prepared['image'].max())],
         state_changes_prompt_tokens=True, model_parameter_count=sum(p.numel() for p in model.parameters()))
    del prepared
    optimizer = torch.optim.AdamW(build_param_lr_groups(model, cfg), betas=(0.9,0.95), eps=1e-8, weight_decay=1e-8, fused=True)
    results = []
    for step in range(1,4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        t0=time.monotonic()
        before=model.action_head.action_out_proj.weight.detach().float().clone()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output=model(examples=examples, return_debug=True)
            loss=output['action_loss']
        assert torch.isfinite(loss)
        squared=(output['v_t'].detach()-output['u_t'].detach()).square()
        loss.backward()
        grad={}
        for name,module in [('VLM', model.vlm),('Action', model.action_head)]:
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
            norm=torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]).norm()
            grad[name]=float(norm)
            assert grad[name]>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        torch.cuda.synchronize()
        changed=bool(torch.any(before != model.action_head.action_out_proj.weight.detach().float()))
        assert changed
        row={'step':step,'loss':float(loss.detach()),'valid14_velocity_mse':float(squared[...,:14].mean()),
             'padding18_velocity_mse':float(squared[...,14:].mean()),'gradient_norms':grad,
             'action_weights_changed':changed,'elapsed_seconds':time.monotonic()-t0,
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
             'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        results.append(row)
        save('single_gpu_diagnostic_update_passed', updates=results)
        del output,loss,squared,before
        optimizer.zero_grad(set_to_none=True)
        if step<3:
            examples=next(iterator)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        predicted=model.predict_action(examples=examples)['normalized_actions']
    assert predicted.shape==(1,50,32) and np.isfinite(predicted).all()
    save('complete', inference_shape=list(predicted.shape), inference_finite=True,
         inference_first14_minmax=[float(predicted[...,:14].min()),float(predicted[...,:14].max())],
         diagnostic_passed=True, full_8gpu_trainer_passed=False)


if __name__=='__main__':
    try:
        main()
    except Exception:
        save('failed', error=traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)
