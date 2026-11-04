#!/usr/bin/env python3
"""Small fixed-observation A/B validation of Xperience decoder input LayerNorm.

Run one process per GPU from a starVLA checkout containing the opt-in
framework.action_model.decoder_input_norm implementation. Training calls the
repository's original LayerwiseFlowmatchingActionHead.forward directly.

Cache schema (all tensors on CPU, no autograd graph):
  actions: [N,8,90], action_mask: [N,8,90], attention_mask: [N,S]
  vl_embs_list: list of 36 tensors shaped [N,S,2560]
The frozen VLM is never instantiated by this program.
"""

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

FAILURE_CONTEXT = {}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'ln'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=1e-8)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dead-patience', type=int, default=20)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--no-save-model', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='CPU/meta-only validation; no CUDA allocation or training')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.repeat, args.eval_every) < 1:
        parser.error('steps, batch-size, repeat, and eval-every must be positive')
    if args.warmup < 0 or args.lr <= 0:
        parser.error('warmup must be nonnegative and lr positive')
    if args.dead_patience < 1:
        parser.error('dead-patience must be positive')
    if args.save_model and args.no_save_model:
        parser.error('--save-model and --no-save-model conflict')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def load_cache(path, cfg):
    data = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    required = ['actions', 'action_mask', 'attention_mask', 'vl_embs_list']
    for key in required:
        if key not in data:
            raise ValueError(f'Cache is missing {key}; expected {required}')
    actions = data['actions']
    mask = data['action_mask']
    attn = data['attention_mask']
    hidden = data['vl_embs_list']
    horizon = int(cfg.framework.action_model.action_horizon)
    action_dim = int(cfg.framework.action_model.action_dim)
    layers = int(cfg.framework.action_model.diffusion_model_cfg.num_layers)
    width = int(cfg.framework.action_model.diffusion_model_cfg.input_embedding_dim)
    if actions.ndim != 3 or tuple(actions.shape[1:]) != (horizon, action_dim):
        raise ValueError(f'Expected actions [N,{horizon},{action_dim}], got {tuple(actions.shape)}')
    if actions.shape[0] == 0 or mask.shape != actions.shape:
        raise ValueError('Empty cache or action_mask shape does not match actions')
    if len(hidden) != layers:
        raise ValueError(f'Expected {layers} cached hidden layers, got {len(hidden)}')
    n = actions.shape[0]
    if attn.ndim != 2 or attn.shape[0] != n:
        raise ValueError('attention_mask must have shape [N,S]')
    for index, tensor in enumerate(hidden):
        if tuple(tensor.shape) != (n, attn.shape[1], width):
            raise ValueError(f'Bad hidden layer {index} shape: {tuple(tensor.shape)}')
        if tensor.device.type != 'cpu' or tensor.requires_grad:
            raise ValueError('Frozen hidden states must be CPU tensors without gradients')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite cached hidden layer {index}')
    if not torch.isfinite(actions).all():
        raise ValueError('Nonfinite actions')
    if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any():
        raise ValueError('action_mask must be nonempty and binary')
    if not torch.all((attn == 0) | (attn == 1)) or not attn.bool().any(dim=1).all():
        raise ValueError('Each observation needs at least one valid VLM token')
    return data


def prepare_batch(data, indices, device, repeat=1):
    # Selection happens on CPU, so only this batch of frozen features reaches GPU.
    indices = torch.as_tensor(indices, dtype=torch.long, device='cpu').repeat(repeat)
    action = data['actions'].index_select(0, indices).to(device=device, dtype=torch.bfloat16)
    mask = data['action_mask'].index_select(0, indices).to(device=device, dtype=torch.bfloat16)
    attention = data['attention_mask'].index_select(0, indices).to(device=device, dtype=torch.bool)
    vl = [item.index_select(0, indices).to(device=device, dtype=torch.bfloat16)
          for item in data['vl_embs_list']]
    return vl, action, mask, attention


def enable_block_checkpointing(head):
    # DiT.gradient_checkpointing is only a stored flag in the original code.
    # Non-reentrant checkpointing supports kwargs and preserves dropout RNG and
    # the surrounding autocast mode; do not checkpoint the decoder metric hooks.
    for block in head.model.transformer_blocks:
        original = block.forward

        @functools.wraps(original)
        def wrapped(*args, _original=original, **kwargs):
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return checkpoint(_original, *args, use_reentrant=False,
                              preserve_rng_state=True, **kwargs)

        block.forward = wrapped


class DecoderMetrics:
    def __init__(self, head):
        self.horizon = head.action_horizon
        self.latest = {}
        self.handles = [
            head.action_decoder.register_forward_pre_hook(self.raw_input),
            head.action_decoder.layer1.register_forward_pre_hook(self.normalized_input),
            head.action_decoder.layer1.register_forward_hook(self.pre_relu),
        ]

    @staticmethod
    def rms(tensor):
        return tensor.detach().float().square().mean().sqrt().item()

    def raw_input(self, module, inputs):
        self.latest['decoder_raw_input_rms'] = self.rms(inputs[0][:, -self.horizon:])

    def normalized_input(self, module, inputs):
        self.latest['decoder_layer1_input_rms'] = self.rms(inputs[0][:, -self.horizon:])

    def pre_relu(self, module, inputs, output):
        value = output.detach()[:, -self.horizon:]
        positive = value > 0
        self.latest.update(
            relu_positive_fraction=positive.float().mean().item(),
            relu_ever_active_neurons=int(positive.any(dim=0).any(dim=0).sum().item()),
            relu_preactivation_min=value.min().item(),
            relu_preactivation_max=value.max().item(),
        )


def velocity_forward(head, vl, noisy_action, bucket, attention):
    """Diagnostic-only deterministic module path, matching native head.forward."""
    action_features = head.action_encoder(noisy_action, bucket)
    if head.config.add_pos_embed:
        positions = torch.arange(action_features.shape[1], device=noisy_action.device)
        action_features = action_features + head.position_embedding(positions).unsqueeze(0)
    prefix = head.future_tokens.weight.unsqueeze(0).expand(noisy_action.shape[0], -1, -1)
    features = torch.cat((prefix, action_features), dim=1)
    output = head.model(hidden_states=features, encoder_hidden_states=vl,
                        timestep=bucket, encoder_attention_mask=attention,
                        return_pre_output=True)
    return head.action_decoder(output)[:, -noisy_action.shape[1]:]


def assert_native_forward_equivalence(head, batch, seed, device):
    vl, action, mask, attention = batch
    old_mode = head.training
    head.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            seed_everything(seed)
            native = head(vl, action, state=None, encoder_attention_mask=attention,
                          action_mask=mask)
            seed_everything(seed)
            noise = torch.randn(action.shape, device=action.device, dtype=action.dtype) * mask
            clean = action * mask
            time_sample = head.sample_time(action.shape[0], action.device, action.dtype)[:, None, None]
            noisy = (1 - time_sample) * noise + time_sample * clean
            velocity = clean - noise
            bucket = (time_sample[:, 0, 0] * head.num_timestep_buckets).long()
            pred = velocity_forward(head, vl, noisy, bucket, attention)
            reproduced = (((pred - velocity) ** 2) * mask).sum() / mask.sum().clamp_min(1)
        torch.testing.assert_close(native.float(), reproduced.float(), rtol=1e-4, atol=1e-5)
        return {'native_loss': native.item(), 'manual_loss': reproduced.item(),
                'absolute_difference': (native - reproduced).abs().item()}
    finally:
        head.train(old_mode)


def masked_mean(value, mask):
    return (value.float() * mask.float()).sum() / mask.float().sum().clamp_min(1)


def fixed_probe(head, data, device, metrics, seed):
    """Independent, repeatable evaluation noise; no parameter updates."""
    old_mode = head.training
    head.eval()
    n = data['actions'].shape[0]
    indices = torch.arange(n).repeat(3)
    # Estimate a constant action/velocity predictor from the fixed training
    # observations, independently of the newly sampled evaluation noise.
    source_mask = data['action_mask'].float()
    source_action = data['actions'].float()
    constant = ((source_action * source_mask).sum(dim=(0, 1)) /
                source_mask.sum(dim=(0, 1)).clamp_min(1)).to(device)
    constant_action_mse = masked_mean(
        (source_action - constant.cpu()[None, None]).square(), source_mask).item()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            seed_everything(seed)
            vl, action, mask, attention = prepare_batch(data, indices, device, repeat=2)
            half = n * 3
            time_sample = torch.tensor([.1, .5, .9], device=device, dtype=action.dtype)
            time_sample = time_sample.repeat_interleave(n).repeat(2)[:, None, None]
            noise = torch.randn_like(action) * mask
            clean = action * mask
            noisy = (1 - time_sample) * noise + time_sample * clean
            bucket = (time_sample[:, 0, 0] * head.num_timestep_buckets).long()
            pred = velocity_forward(head, vl, noisy, bucket, attention)
            error = (pred.float() - (clean - noise).float()).square()
            delta_pred = pred[half:].float() - pred[:half].float()
            delta_noisy = noisy[half:].float() - noisy[:half].float()
            delta_target = -(noise[half:].float() - noise[:half].float())
            pair_mask = mask[:half]
            out_diff = masked_mean(delta_pred.square(), pair_mask).sqrt()
            in_diff = masked_mean(delta_noisy.square(), pair_mask).sqrt()
            d_pred = (delta_pred * pair_mask).reshape(-1)
            d_target = (delta_target * pair_mask).reshape(-1)
            correlation = torch.dot(d_pred, d_target) / (d_pred.norm() * d_target.norm()).clamp_min(1e-12)
            row = dict(metrics.latest)
            row.update(
                fixed_velocity_mse=masked_mean(error, mask).item(),
                paired_output_diff_rms=out_diff.item(),
                paired_input_diff_rms=in_diff.item(),
                noise_sensitivity_rms_ratio=(out_diff / in_diff.clamp_min(1e-12)).item(),
                noise_delta_cosine=correlation.item(),
                noise_delta_relative_mse=(masked_mean((delta_pred-delta_target).square(), pair_mask) /
                    masked_mean(delta_target.square(), pair_mask).clamp_min(1e-12)).item(),
                constant_velocity_baseline_mse=masked_mean(
                    (constant[None, None] - (clean-noise).float()).square(), mask).item(),
                zero_velocity_baseline_mse=masked_mean((clean-noise).float().square(), mask).item(),
                constant_action_baseline_mse=constant_action_mse,
                theoretical_constant_velocity_baseline_mse=1.0+constant_action_mse,
                output_minus_bias_rms=masked_mean(
                    (pred.float() - head.action_decoder.layer2.bias.float()[None, None]).square(), mask).sqrt().item(),
            )
            for j, t in enumerate([.1, .5, .9]):
                selection = torch.cat([torch.arange(j*n, (j+1)*n, device=device),
                                       torch.arange(half+j*n, half+(j+1)*n, device=device)])
                row[f'velocity_mse_t{t}'] = masked_mean(error[selection], mask[selection]).item()
                pair_selection = slice(j*n, (j+1)*n)
                row[f'noise_delta_relative_mse_t{t}'] = (
                    masked_mean((delta_pred[pair_selection]-delta_target[pair_selection]).square(), pair_mask[pair_selection]) /
                    masked_mean(delta_target[pair_selection].square(), pair_mask[pair_selection]).clamp_min(1e-12)).item()
                row[f'constant_velocity_baseline_mse_t{t}'] = masked_mean(
                    (constant[None, None]-(clean[selection]-noise[selection]).float()).square(), mask[selection]).item()
            return row
    finally:
        head.train(old_mode)


def native_euler_probe(head, data, device, seed):
    old_mode = head.training
    head.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            seed_everything(seed)
            vl, action, mask, attention = prepare_batch(data, torch.arange(data['actions'].shape[0]), device)
            pred = head.predict_action(vl, state=None, encoder_attention_mask=attention, action_mask=mask)
            error = pred.float() - action.float()
            row = {'euler_steps': int(head.num_inference_timesteps),
                   'native_action_masked_mse': masked_mean(error.square(), mask).item(),
                   'native_action_masked_mae': masked_mean(error.abs(), mask).item()}
            return row, pred.detach().float().cpu().numpy()
    finally:
        head.train(old_mode)


def gradient_metrics(head):
    selected = {
        'action_encoder_layer1': head.action_encoder.layer1.weight,
        'decoder_layer1': head.action_decoder.layer1.weight,
        'decoder_layer2': head.action_decoder.layer2.weight,
        'dit_first_q': head.model.transformer_blocks[0].attn1.to_q.weight,
        'dit_last_ff': head.model.transformer_blocks[-1].ff.net[2].weight,
    }
    result = {}
    for name, parameter in selected.items():
        grad = parameter.grad
        result[f'grad_rms/{name}'] = None if grad is None else grad.detach().float().square().mean().sqrt().item()
    return result


def fingerprint(head):
    """Hash complete initialized model state, not sampled weight entries.

    Decoder input LayerNorm has affine=False, so baseline and LN arms have the
    same state keys and should have identical hashes under the same init seed.
    A uint8 memoryview avoids allocating a full-model Python bytes copy.
    """
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        digest.update(json.dumps({'key': name, 'shape': list(value.shape),
                                 'dtype': str(value.dtype)}, sort_keys=True).encode())
        digest.update(b'\0')
        raw = value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def current_lr(args, step):
    if args.warmup and step <= args.warmup:
        return args.lr * step / args.warmup
    if args.schedule == 'constant':
        return args.lr
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress)) / 2


def main():
    args = parse_args()
    FAILURE_CONTEXT.update(output_dir=str(args.output_dir), variant=args.variant,
                           step=0, phase='setup', output_created=False)
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.repo.resolve()))
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead

    cfg = OmegaConf.load(args.config)
    cfg.framework.action_model.decoder_input_norm = args.variant == 'ln'
    data = load_cache(args.cache, cfg)
    seed_everything(args.seed)
    if args.dry_run:
        torch.distributions.Distribution.set_default_validate_args(False)
        with torch.device('meta'):
            head = LayerwiseFlowmatchingActionHead(cfg)
    else:
        head = LayerwiseFlowmatchingActionHead(cfg).float()
    norm = getattr(head.action_decoder, 'input_norm', None)
    expected_norm = torch.nn.LayerNorm if args.variant == 'ln' else torch.nn.Identity
    if not isinstance(norm, expected_norm):
        raise RuntimeError(f'Variant {args.variant} requires decoder.input_norm={expected_norm.__name__}; got {norm}. Check checkout patch.')
    # These modules are constructed for compatibility but absent from this path.
    for module in [head.state_encoder, head.model.norm_out,
                   head.model.proj_out_1, head.model.proj_out_2]:
        if module is not None:
            module.requires_grad_(False)
    trainable = [p for p in head.parameters() if p.requires_grad]
    parameter_count = sum(p.numel() for p in trainable)
    metadata = {
        'variant': args.variant,
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'num_observations': int(data['actions'].shape[0]),
        'action_shape': list(data['actions'].shape),
        'mask_valid_per_step': data['action_mask'].sum(-1).unique().tolist(),
        'vl_layer_shape': list(data['vl_embs_list'][0].shape),
        'trainable_parameters': parameter_count,
        'fp32_weights_grad_adam_estimated_gib': parameter_count * 16 / 2**30,
        'training_loss_path': 'unmodified native head.forward, original BF16 action/noise/time/masked loss',
        'parameter_dtype': 'float32', 'compute_autocast': 'bfloat16',
        'fixed_probe_note': 'Fixed held-out noise on same observations; diagnostic float32 reductions; no generalization claim',
        'torch_version': torch.__version__,
    }
    try:
        metadata['git_commit'] = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata['git_commit'] = None
    if args.dry_run:
        print(json.dumps(metadata, indent=2), flush=True)
        return
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Training harness requires a dedicated CUDA GPU; use --dry-run for CPU/meta checks')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected CUDA environment does not support BF16')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    FAILURE_CONTEXT['output_created'] = True
    metadata['initial_weight_fingerprint'] = fingerprint(head)
    metadata['initial_fingerprint_scope'] = 'SHA256 of every complete state_dict tensor, including key, shape, and dtype; no sampling'
    head = head.to(device)
    # Build the optimizer only from the final GPU Parameter objects.
    trainable = [p for p in head.parameters() if p.requires_grad]
    if not args.no_checkpoint:
        enable_block_checkpointing(head)
    metrics = DecoderMetrics(head)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(.9, .95), eps=1e-8,
                                  weight_decay=args.weight_decay, fused=True)
    # Module.to() preserves Parameter objects for same-type moves in supported
    # PyTorch; explicitly guard the optimizer references rather than assume it.
    model_ids = {id(p) for p in head.parameters() if p.requires_grad}
    if any(id(p) not in model_ids or p.device != device for group in optimizer.param_groups for p in group['params']):
        raise RuntimeError('Optimizer parameter references do not match model GPU parameters')
    OmegaConf.save(cfg, args.output_dir / 'config.yaml')
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    train_batch = prepare_batch(data, torch.arange(min(args.batch_size, data['actions'].shape[0])), device, args.repeat)
    FAILURE_CONTEXT['phase'] = 'native_forward_equivalence'
    comparison = assert_native_forward_equivalence(head, train_batch, args.seed + 765432, device)
    (args.output_dir / 'forward_equivalence.json').write_text(json.dumps(comparison, indent=2))
    del train_batch
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with (args.output_dir / 'metrics.jsonl').open('w', buffering=1) as handle:
        def record(row):
            row.update(elapsed_seconds=time.monotonic()-started,
                       gpu_allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                       gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row, allow_nan=False), flush=True)

        FAILURE_CONTEXT['phase'] = 'initial_probe'
        last_probe = fixed_probe(head, data, device, metrics, args.seed + 987654)
        record({'kind': 'probe', 'step': 0, **last_probe})
        head.train()
        completed_steps = 0
        dead_streak = 0
        stopped_early = False
        stop_reason = None
        for step in range(1, args.steps + 1):
            FAILURE_CONTEXT.update(step=step, phase='train_forward')
            # Re-seed after every evaluation. A/B see identical Beta draws,
            # action noise, minibatches and dropout RNG for a given step.
            seed_everything(args.seed + step)
            n = data['actions'].shape[0]
            selection = torch.arange(n) if args.batch_size >= n else torch.randperm(n)[:args.batch_size]
            batch = prepare_batch(data, selection, device, args.repeat)
            vl, action, mask, attention = batch
            optimizer.zero_grad(set_to_none=True)
            lr = current_lr(args, step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            step_start = time.monotonic()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = head(vl, action, state=None, encoder_attention_mask=attention, action_mask=mask)
            FAILURE_CONTEXT['last_loss'] = str(loss.item())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}: {loss.item()}')
            activation = dict(metrics.latest)
            FAILURE_CONTEXT['phase'] = 'backward'
            loss.backward()
            grads = gradient_metrics(head)
            dead_streak = dead_streak + 1 if (
                activation['relu_ever_active_neurons'] == 0 and
                grads['grad_rms/decoder_layer1'] == 0.0) else 0
            FAILURE_CONTEXT['phase'] = 'clip_gradients'
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip, error_if_nonfinite=True)
            FAILURE_CONTEXT['phase'] = 'optimizer_step'
            optimizer.step()
            completed_steps = step
            record({'kind': 'train', 'step': step, 'loss': loss.item(), 'loss_dtype': str(loss.dtype),
                    'lr': lr, 'global_grad_norm_before_clip': grad_norm.item(),
                    'dead_decoder_consecutive_steps': dead_streak,
                    'step_seconds': time.monotonic()-step_start, **activation, **grads})
            optimizer.zero_grad(set_to_none=True)
            del batch, vl, action, mask, attention, loss
            should_stop = dead_streak >= args.dead_patience
            if step in {1, 10, args.steps} or step % args.eval_every == 0 or should_stop:
                FAILURE_CONTEXT['phase'] = 'fixed_probe'
                last_probe = fixed_probe(head, data, device, metrics, args.seed + 987654)
                record({'kind': 'probe', 'step': step, **last_probe})
            if should_stop:
                stopped_early = True
                stop_reason = f'ReLU activation count and decoder.layer1 gradient RMS both zero for {dead_streak} consecutive training steps'
                record({'kind': 'early_stop', 'step': step, 'reason': stop_reason})
                break
        FAILURE_CONTEXT['phase'] = 'native_euler_probe'
        native_row, native_prediction = native_euler_probe(head, data, device, args.seed + 135790)
        np.save(args.output_dir / 'native_normalized_action_predictions.npy', native_prediction)
        record({'kind': 'native_euler', 'step': completed_steps, **native_row})
        should_save = args.save_model or (args.variant == 'ln' and not args.no_save_model)
        if should_save:
            FAILURE_CONTEXT['phase'] = 'save_model'
            save_path = args.output_dir / 'action_head_model.pt'
            torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, save_path)
            record({'kind': 'checkpoint', 'step': completed_steps, 'path': str(save_path),
                    'optimizer_saved': False})
        summary = {'status': 'early_stopped_dead_decoder' if stopped_early else 'completed',
                   'completed_steps': completed_steps, 'requested_steps': args.steps,
                   'stop_reason': stop_reason, 'final_probe': last_probe,
                   'native_euler': native_row, 'model_saved': should_save}
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        FAILURE_CONTEXT['phase'] = 'complete'
        record({'kind': 'complete', 'step': completed_steps,
                'status': summary['status'], 'requested_steps': args.steps})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if FAILURE_CONTEXT.get('output_created'):
            failure = {**FAILURE_CONTEXT, 'error_type': type(error).__name__,
                       'error': str(error), 'traceback': traceback.format_exc()}
            (Path(FAILURE_CONTEXT['output_dir']) / 'failure.json').write_text(
                json.dumps(failure, indent=2, allow_nan=False))
        raise
