#!/usr/bin/env python3
"""Fit independent linear readouts to saved paired-noise activations.

This diagnostic reads a CPU tensor cache and writes a new report directory. It
does not load, modify, or optimize any action-head/VLM checkpoint. Generalization
here means unseen noise rounds for the same eight cached observations, not new
observations. Input schema: features[stage] [96,64,D], target [96,64,48], and
native_prediction_delta [96,64,48]. All values must be float32 CPU tensors.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import warnings


def metrics(prediction, target):
    import torch
    prediction, target = prediction.double(), target.double()
    target_energy = target.square().sum()
    if target_energy <= 0:
        raise ValueError("A split has zero target energy")
    pred_energy = prediction.square().sum()
    dot = (prediction * target).sum()
    gain = (pred_energy / target_energy).sqrt()
    parallel_gain = dot / target_energy
    cosine = dot / (pred_energy * target_energy).sqrt().clamp_min(1e-30)
    return {
        "relative_mse": ((prediction - target).square().sum() / target_energy).item(),
        "cosine": cosine.item(), "gain": gain.item(),
        "parallel_gain": parallel_gain.item(),
        "orthogonal_relative_energy": (pred_energy / target_energy - parallel_gain.square()).clamp_min(0).item(),
        "prediction_rms": prediction.square().mean().sqrt().item(),
        "target_rms": target.square().mean().sqrt().item(),
        "rows": int(target.shape[0]),
    }


def spectrum_stats(eigenvalues):
    import torch
    eigenvalues = eigenvalues.detach().double().cpu()
    raw_min = eigenvalues.min().item()
    positive = eigenvalues.clamp_min(0).flip(0)  # eigh/eigvalsh return ascending order.
    energy = positive.sum()
    if energy <= 0:
        return {"participation_ratio": 0.0, "entropy_effective_rank": 0.0,
                "top1_energy": 0.0, "top5_energy": 0.0, "top10_energy": 0.0,
                "trace": 0.0, "max_eigenvalue": 0.0, "min_raw_eigenvalue": raw_min}
    p = positive / energy
    nonzero = p > 0
    return {
        "participation_ratio": (energy.square() / positive.square().sum()).item(),
        "entropy_effective_rank": (-(p[nonzero] * p[nonzero].log()).sum()).exp().item(),
        "top1_energy": p[:1].sum().item(), "top5_energy": p[:5].sum().item(),
        "top10_energy": p[:10].sum().item(), "trace": energy.item(),
        "max_eigenvalue": positive[0].item(), "min_raw_eigenvalue": raw_min,
    }


def centered_covariance_report(value):
    import torch
    value = value.reshape(-1, value.shape[-1]).double()
    centered = value - value.mean(dim=0, keepdim=True)
    eig = torch.linalg.eigvalsh(centered.T @ centered / value.shape[0])
    return {"definition": "centered test-row covariance, denominator N",
            **spectrum_stats(eig), "eigenvalues_descending": eig.flip(0).tolist()}


def fit_stage(feature, target, device, lambdas, evaluate_test=True):
    import torch
    started = time.monotonic()
    train = feature[:64].reshape(-1, feature.shape[-1]).to(device=device, dtype=torch.float64)
    val = feature[64:80].reshape(-1, feature.shape[-1]).to(device=device, dtype=torch.float64)
    y_train = target[:64].reshape(-1, 48).to(device=device, dtype=torch.float64)
    y_val = target[64:80].reshape(-1, 48).to(device=device, dtype=torch.float64)
    raw_rms = train.square().mean().sqrt().item()
    scale = raw_rms if raw_rms > 0 else 1.0
    train.div_(scale)
    val.div_(scale)
    # Objective: ||X_scaled W - Y||_F^2 / N + lambda * ||W||_F^2.
    # No centering, intercept, per-channel standardization, or rank truncation.
    gram = train.T @ train / train.shape[0]
    eig, basis = torch.linalg.eigh(gram)
    gram_spectrum = spectrum_stats(eig)
    negative_tolerance = 1e-10 * max(1.0, float(eig[-1]))
    if float(eig[0]) < -negative_tolerance:
        raise RuntimeError("Training Gram has unexpectedly negative eigenvalues")
    eig = eig.clamp_min(0)  # Roundoff only; all D eigenvectors are retained.
    rhs = basis.T @ (train.T @ y_train / train.shape[0])
    val_projected = val @ basis
    candidates = []
    for regularizer in lambdas:
        coefficients = rhs / (eig[:, None] + regularizer)
        val_pred = val_projected @ coefficients
        candidates.append({"lambda": regularizer, "val": metrics(val_pred, y_val)})
    best_error = min(row["val"]["relative_mse"] for row in candidates)
    # Prefer stronger regularization for numerical ties, using validation only.
    best = max((row for row in candidates if row["val"]["relative_mse"] <= best_error + 1e-12),
               key=lambda row: row["lambda"])
    chosen = best["lambda"]
    weight = basis @ (rhs / (eig[:, None] + chosen))
    final_metrics = {"train": metrics(train @ weight, y_train),
                     "val": metrics(val @ weight, y_val)}
    # Optional: inspect a wider validation grid without evaluating the old test
    # split again. A repeated/adaptive invocation is not a new held-out test.
    if evaluate_test:
        test = feature[80:96].reshape(-1, feature.shape[-1]).to(device=device, dtype=torch.float64) / scale
        y_test = target[80:96].reshape(-1, 48).to(device=device, dtype=torch.float64)
        final_metrics["test"] = metrics(test @ weight, y_test)
    return {"device": str(device), "feature_dim": int(feature.shape[-1]),
            "train_feature_global_rms": raw_rms, "feature_scale_used": scale,
            "selected_lambda": chosen,
            "lambda_at_grid_boundary": chosen in (lambdas[0], lambdas[-1]),
            "regularized_gram_condition_estimate": ((eig[-1] + chosen) / (eig[0] + chosen)).item(),
            "metrics": final_metrics, "validation_grid": candidates,
            "train_scaled_feature_gram_spectrum": {
                "definition": "uncentered X_scaled.T @ X_scaled / N; no rank truncation",
                **gram_spectrum},
            "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh report directory")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]; unavailable CUDA/OOM falls back to CPU")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lambda-grid", default=",".join(f"1e{power}" for power in range(-12, 2)),
                        help="Comma-separated positive ridge penalties on X.T @ X / N; default 1e-12,...,10")
    parser.add_argument("--stages", help="Comma-separated feature names to fit; default all")
    parser.add_argument("--validation-only", action="store_true",
                        help="Fit and select using train/val only; omit all test metrics/covariance")
    args = parser.parse_args()
    try:
        lambdas = sorted(set(float(value.strip()) for value in args.lambda_grid.split(",")))
    except ValueError:
        parser.error("--lambda-grid must contain comma-separated numbers")
    if not lambdas or not all(math.isfinite(value) and value > 0 for value in lambdas):
        parser.error("Every --lambda-grid value must be finite and positive")
    import torch
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    fallback_reason = None
    if device.type == "cuda" and (not torch.cuda.is_available() or
            (device.index is not None and device.index >= torch.cuda.device_count())):
        fallback_reason = f"Requested CUDA device unavailable: {device}"
        warnings.warn(fallback_reason)
        device = torch.device("cpu")
    data = torch.load(args.input, map_location="cpu", mmap=True, weights_only=True)
    target, native, features = data["target"], data["native_prediction_delta"], data["features"]
    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a nonempty dictionary")
    if args.stages:
        selected_stages = list(dict.fromkeys(name.strip() for name in args.stages.split(",")))
        missing = [name for name in selected_stages if name not in features]
        if missing:
            parser.error(f"Unknown --stages names: {missing}; available: {list(features)}")
        features = {name: features[name] for name in selected_stages}
    for name, value in [("target", target), ("native_prediction_delta", native), *features.items()]:
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError(f"Expected string key and tensor for {name!r}")
        if value.dtype != torch.float32 or value.device.type != "cpu" or value.ndim != 3:
            raise ValueError(f"Expected float32 CPU [round,row,dim] for {name}")
        if tuple(value.shape[:2]) != (96, 64) or value.shape[-1] < 1 or not torch.isfinite(value).all():
            raise ValueError(f"Bad shape or nonfinite values in {name}: {tuple(value.shape)}")
    if tuple(target.shape) != (96, 64, 48) or native.shape != target.shape:
        raise ValueError("target/native_prediction_delta must both be [96,64,48]")
    args.output.mkdir(parents=True, exist_ok=False)
    splits = {"train": (0, 64), "val": (64, 80), "test": (80, 96)}
    evaluated_splits = {name: bounds for name, bounds in splits.items()
                        if name != "test" or not args.validation_only}
    native_metrics = {name: metrics(native[start:end].reshape(-1, 48), target[start:end].reshape(-1, 48))
                      for name, (start, end) in evaluated_splits.items()}
    report = {
        "status": "running", "input": str(args.input.resolve()),
        "torch_version": str(torch.__version__), "requested_device": args.device,
        "initial_device": str(device), "fallback_reason": fallback_reason,
        "precision": "float64 Gram/eigendecomposition/solve/predictions/metrics",
        "split_rounds_half_open": splits, "lambda_grid": lambdas,
        "selected_stages": list(features), "validation_only": args.validation_only,
        "test_policy": "This invocation uses validation only to choose lambda; prior adaptive reuse of these rounds is exploratory, not an independent confirmation. Use fresh noise rounds for confirmatory evaluation.",
        "fit_policy": "train-only global RMS, no intercept/centering/channel scaling/rank truncation; validation chooses lambda; no refit on validation",
        "scope": "Independent diagnostic linear readouts only; no original model/checkpoint loaded or modified; fixed observations, held-out noise rounds",
        "rank_interpretation": "Energy concentration is not proof of information loss; low linear-readout quality does not exclude nonlinear information",
        "native_prediction_delta": native_metrics,
        "stages": {},
    }
    if not args.validation_only:
        report["test_native_prediction_covariance"] = centered_covariance_report(native[80:96])
        report["test_target_covariance"] = centered_covariance_report(target[80:96])
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    started = time.monotonic()
    with (args.output / "stages.jsonl").open("w", buffering=1) as log:
        for name, feature in features.items():
            try:
                row = fit_stage(feature, target, device, lambdas, evaluate_test=not args.validation_only)
            except torch.cuda.OutOfMemoryError as error:
                if device.type != "cuda":
                    raise
                reason = str(error)
                error.__traceback__ = None
                gc.collect()
                torch.cuda.empty_cache()
                warnings.warn(f"{name}: CUDA OOM; retrying this and remaining stages on CPU")
                device = torch.device("cpu")
                row = fit_stage(feature, target, device, lambdas, evaluate_test=not args.validation_only)
                row["fallback_reason"] = reason
            if not args.validation_only:
                row["test_relative_mse_minus_native"] = row["metrics"]["test"]["relative_mse"] - native_metrics["test"]["relative_mse"]
            report["stages"][name] = row
            log.write(json.dumps({"stage": name, **row}, allow_nan=False) + "\n")
            summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"stage": name, "lambda": row["selected_lambda"],
                              "metrics": row["metrics"], "seconds": row["elapsed_seconds"]}), flush=True)
    report.update(status="complete", elapsed_seconds=time.monotonic() - started)
    summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(summary), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
