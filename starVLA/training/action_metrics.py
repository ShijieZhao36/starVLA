"""Action reconstruction metrics over valid normalized action elements."""

import numpy as np


def compute_action_metrics(normalized_actions, examples):
    """Return per-element MAE/MSE, aligned to the training target's final chunk.

    Each example contains a ``[T, D]`` action and optionally an equally shaped
    binary ``action_mask``. A batch must either provide masks for every example
    or omit them for every example. Invalid elements are selected out before
    subtraction so unused canonical dimensions cannot contaminate the metrics.
    """
    predicted = np.asarray(normalized_actions)
    targets = np.asarray([example["action"] for example in examples])
    if predicted.ndim != 3 or any(size == 0 for size in predicted.shape):
        raise ValueError("normalized_actions must have nonempty shape [B, T, D]")
    if (
        targets.ndim != 3
        or targets.shape[0] != predicted.shape[0]
        or targets.shape[2] != predicted.shape[2]
        or targets.shape[1] < predicted.shape[1]
    ):
        raise ValueError(
            f"action targets {targets.shape} must match prediction batch/action "
            f"dimensions and cover its horizon {predicted.shape}"
        )

    has_mask = ["action_mask" in example for example in examples]
    if any(has_mask) and not all(has_mask):
        raise ValueError("action_mask must be present on every sample in the batch")
    if all(has_mask):
        mask = np.asarray([example["action_mask"] for example in examples])
        if mask.shape != targets.shape:
            raise ValueError(f"action_mask shape {mask.shape} must match action targets {targets.shape}")
        if not np.all(np.isfinite(mask) & ((mask == 0) | (mask == 1))):
            raise ValueError("action_mask must contain only finite binary values")
        valid = mask[:, -predicted.shape[1] :, :].astype(bool)
    else:
        valid = np.ones(predicted.shape, dtype=bool)
    if not np.any(valid):
        raise ValueError("action metrics require at least one valid action element")

    # QwenPI.forward likewise supervises only the final action_horizon timesteps.
    targets = targets[:, -predicted.shape[1] :, :]
    errors = predicted[valid].astype(np.float64) - targets[valid].astype(np.float64)
    return {
        "mae_score": float(np.mean(np.abs(errors))),
        "mse_score": float(np.mean(np.square(errors))),
    }
