#!/usr/bin/env python3
"""Read-only action-head BF16/FP32 paired-noise propagation probe.

This program performs inference only: no optimizer, gradients or weight changes.
Both branches use identical FP32 parameters and exactly the same BF16-quantized
input values. Only operator precision differs. --output is a fresh directory.
Place beside train_ab.py and run against the same patched repository/config.
"""
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-observations", type=int, default=8)
    parser.add_argument("--times", default="0.1,0.5,0.9")
    parser.add_argument("--seed", type=int, default=987696)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from train_ab import load_cache, prepare_batch, velocity_forward, seed_everything
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("This inference probe requires CUDA; writing/importing the script does not run it")
    torch.cuda.set_device(device)
    times = [float(value) for value in args.times.split(",")]
    if not times or not all(0 < value < 1 for value in times):
        parser.error("Every probe time must be strictly between 0 and 1")
    if args.num_observations < 1:
        parser.error("--num-observations must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.config)
    data = load_cache(args.cache, cfg)
    n = min(args.num_observations, int(data["actions"].shape[0]))
    horizon = int(cfg.framework.action_model.action_horizon)

    # mmap + meta + assign avoids constructing another full initialized head.
    torch.distributions.Distribution.set_default_validate_args(False)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    checkpoint_dtypes = sorted({str(value.dtype) for value in state.values() if torch.is_tensor(value)})
    with torch.device("meta"):
        head = LayerwiseFlowmatchingActionHead(cfg)
    head.load_state_dict(state, strict=True, assign=True)
    del state
    if any(value.is_meta for value in list(head.parameters()) + list(head.buffers())):
        raise RuntimeError("Checkpoint did not materialize every model parameter/buffer")
    head = head.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if any(parameter.dtype != torch.float32 for parameter in head.parameters() if parameter.is_floating_point()):
        raise RuntimeError("Both precision branches must share FP32 model weights")
    parameters_before = {name: (p.data_ptr(), p._version) for name, p in head.named_parameters()}
    torch.cuda.reset_peak_memory_stats(device)

    def rms(value):
        return value.float().square().mean().sqrt()

    def activation_stats(value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not torch.is_tensor(value) or value.shape[0] != 2 * n:
            raise RuntimeError(f"Unexpected paired activation: {type(value)}")
        record = {"shape": list(value.shape), "dtype": str(value.dtype)}
        for label, part in (("all_tokens", value), ("action_tokens", value[:, -horizon:])):
            part32 = part.detach().float()
            delta = part32[n:] - part32[:n]
            absolute = rms(part32)
            delta_rms = rms(delta)
            record[label] = {"rms": absolute.item(), "delta_rms": delta_rms.item(),
                "delta_relative_rms": (delta_rms / absolute.clamp_min(1e-12)).item(),
                "paired_exactly_equal_fraction": (part[n:] == part[:n]).float().mean().item(),
                "max_abs": part32.abs().max().item(), "finite": bool(torch.isfinite(part32).all())}
        return record

    context = {"branch": None, "t": None, "records": None}
    handles = []

    def add_record(name, value):
        context["records"][name] = activation_stats(value)

    def output_hook(name):
        return lambda module, inputs, output: add_record(name, output)

    def input_hook(name):
        return lambda module, inputs: add_record(name, inputs[0])

    handles.append(head.action_encoder.register_forward_hook(output_hook("action_encoder")))
    handles.append(head.model.register_forward_pre_hook(
        lambda module, inputs, kwargs: add_record("dit_input", kwargs["hidden_states"]), with_kwargs=True))
    for index, block in enumerate(head.model.transformer_blocks):
        handles.append(block.register_forward_hook(output_hook(f"block_{index:02d}")))
    handles.append(head.action_decoder.register_forward_pre_hook(input_hook("decoder_input")))
    handles.append(head.action_decoder.layer1.register_forward_pre_hook(input_hook("decoder_after_input_norm")))
    handles.append(head.action_decoder.layer1.register_forward_hook(output_hook("decoder_layer1_pre_relu")))

    def masked_mean(value, mask):
        return (value.float() * mask.float()).sum() / mask.float().sum().clamp_min(1)

    def velocity_metrics(pred, target, noise, noisy, mask):
        pred = pred.float()
        pair_mask = mask[:n].float()
        delta_pred = pred[n:] - pred[:n]
        # This matches train_ab.fixed_probe's noiseDelta target convention.
        delta_target = -(noise[n:].float() - noise[:n].float())
        delta_native_target = target[n:].float() - target[:n].float()
        target_energy = masked_mean(delta_target.square(), pair_mask).clamp_min(1e-12)
        p = (delta_pred * pair_mask).reshape(-1)
        t = (delta_target * pair_mask).reshape(-1)
        noise_delta_cosine = F.cosine_similarity(p[None], t[None], dim=1, eps=1e-12).item()
        return {"velocity_masked_mse": masked_mean((pred - target.float()).square(), mask).item(),
            "noise_delta_cosine": noise_delta_cosine,
            "noise_delta_relative_mse": (masked_mean((delta_pred - delta_target).square(), pair_mask) / target_energy).item(),
            "noise_delta_pred_rms": masked_mean(delta_pred.square(), pair_mask).sqrt().item(),
            "noise_delta_target_rms": target_energy.sqrt().item(),
            "input_noise_delta_rms": masked_mean((noisy[n:].float() - noisy[:n].float()).square(), pair_mask).sqrt().item(),
            "quantized_velocity_delta_target_difference_rms": masked_mean((delta_native_target - delta_target).square(), pair_mask).sqrt().item(),
            "prediction_rms": masked_mean(pred.square(), mask).sqrt().item(),
            "prediction_finite": bool(torch.isfinite(pred).all())}

    metadata = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "num_observations": n, "paired_batch": 2 * n, "torch_version": str(torch.__version__),
        "checkpoint_dtypes": checkpoint_dtypes, "parameter_dtype_both_branches": "torch.float32",
        "input_policy": "Generate clean/noise/xt/time bucket/native velocity target once in training BF16; FP32 branch casts these quantized values and quantized VLM features to FP32 without regenerating values",
        "comparison": "eval/inference only; BF16 autocast versus FP32 autocast disabled; same weights and input values; TF32 off",
        "decoder_input_norm": bool(cfg.framework.action_model.get("decoder_input_norm", False)),
        "git_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results, prediction_cache = [], {}
    started = time.monotonic()
    try:
        with torch.inference_mode(), (args.output / "metrics.jsonl").open("w", buffering=1) as log:
            for time_index, requested_time in enumerate(times):
                seed_everything(args.seed + time_index)
                vl_q, action_q, mask_q, attention = prepare_batch(data, torch.arange(n), device, repeat=2)
                # BF16 tensor operations intentionally match the native training inputs.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    clean_q = action_q * mask_q
                    noise_q = torch.randn_like(action_q) * mask_q
                    time_q = torch.full((2 * n, 1, 1), requested_time, device=device, dtype=torch.bfloat16)
                    noisy_q = (1 - time_q) * noise_q + time_q * clean_q
                    target_q = clean_q - noise_q
                    bucket = (time_q[:, 0, 0] * head.num_timestep_buckets).long()
                assert torch.equal(clean_q[:n], clean_q[n:])
                time_outputs = {}
                for branch in ("bf16", "fp32"):
                    context.update(branch=branch, t=requested_time, records={})
                    vl = vl_q if branch == "bf16" else [value.float() for value in vl_q]
                    noisy = noisy_q if branch == "bf16" else noisy_q.float()
                    # Casting BF16 to FP32 is exact; assert value identity explicitly.
                    assert torch.equal(noisy.float(), noisy_q.float())
                    if branch == "fp32":
                        assert all(torch.equal(a, b.float()) for a, b in zip(vl, vl_q))
                    torch.cuda.synchronize(device)
                    branch_start = time.monotonic()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(branch == "bf16")):
                        pred = velocity_forward(head, vl, noisy, bucket, attention)
                    torch.cuda.synchronize(device)
                    row = {"branch": branch, "requested_time": requested_time,
                        "quantized_time": float(time_q[0, 0, 0]), "time_bucket": int(bucket[0]),
                        "forward_seconds": time.monotonic() - branch_start,
                        "metrics": velocity_metrics(pred, target_q, noise_q, noisy_q, mask_q),
                        "activations": context["records"]}
                    assert len([key for key in row["activations"] if key.startswith("block_")]) == len(head.model.transformer_blocks)
                    results.append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps({k: v for k, v in row.items() if k != "activations"}, allow_nan=False), flush=True)
                    time_outputs[branch] = pred.detach().float().cpu()
                    del pred, vl, noisy
                prediction_cache[str(requested_time)] = {**time_outputs,
                    "native_quantized_target": target_q.float().cpu(), "noise_quantized": noise_q.float().cpu(),
                    "noisy_action_quantized": noisy_q.float().cpu(), "mask": mask_q.float().cpu(), "bucket": bucket.cpu()}
                diff = time_outputs["fp32"] - time_outputs["bf16"]
                difference = {"kind": "precision_output_difference", "requested_time": requested_time,
                    "masked_rms": masked_mean(diff.square(), mask_q.cpu()).sqrt().item(),
                    "masked_max_abs": (diff.abs() * mask_q.cpu()).max().item()}
                log.write(json.dumps(difference, allow_nan=False) + "\n")
                print(json.dumps(difference, allow_nan=False), flush=True)
                del vl_q, action_q, mask_q, attention, clean_q, noise_q, time_q, noisy_q, target_q, bucket
    finally:
        for handle in handles:
            handle.remove()
    assert parameters_before == {name: (p.data_ptr(), p._version) for name, p in head.named_parameters()}, "Probe changed model parameters"
    summary = {"status": "complete", "metadata": metadata, "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "model_parameters_unchanged": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    torch.save(prediction_cache, args.output / "predictions.pt")
    print(json.dumps({"summary": str(args.output / "summary.json"), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
