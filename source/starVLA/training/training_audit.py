"""Opt-in training provenance; no sampling or random-number operations."""

from collections import Counter
import hashlib
import json

import torch
import torch.distributed as dist


class SourceExposureAudit:
    """Count examples that completed backward, scoped to this process lifetime.

    All ranks must call ``global_metrics`` at the same optimization-step boundary.
    Resume starts new counters; the caller records the starting optimizer step.
    """

    def __init__(self):
        self.cumulative = Counter()
        self.window = Counter()

    def observe_backward_batch(self, examples):
        counts = Counter(str(example["robot_tag"]) for example in examples)
        self.cumulative.update(counts)
        self.window.update(counts)

    def global_metrics(self, device):
        tags = sorted(self.cumulative)
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            rank_tags = [None] * dist.get_world_size()
            # A source may be absent on some ranks or first appear in a later window.
            dist.all_gather_object(rank_tags, tags)
            tags = sorted({tag for names in rank_tags for tag in names})
        if not tags:
            return {}
        counts = torch.tensor(
            [[self.cumulative[tag] for tag in tags], [self.window[tag] for tag in tags]],
            dtype=torch.int64,
            device=device,
        )
        if distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        cumulative, window = counts.cpu().tolist()
        cumulative_total, window_total = sum(cumulative), sum(window)
        metrics = {
            "source_exposure/cumulative_examples": cumulative_total,
            "source_exposure/window_examples": window_total,
        }
        for tag, total, recent in zip(tags, cumulative, window):
            prefix = f"source_exposure/{tag}"
            metrics[f"{prefix}/cumulative_examples"] = total
            metrics[f"{prefix}/window_examples"] = recent
            metrics[f"{prefix}/cumulative_fraction"] = total / cumulative_total if cumulative_total else 0.0
            metrics[f"{prefix}/window_fraction"] = recent / window_total if window_total else 0.0
        self.window.clear()
        return metrics


def model_parameter_identity(model):
    """Hash all distinct named parameters, including frozen ones, without RNG use.

    Names, shapes and dtypes are framed with a newline before contiguous raw bytes.
    Tied parameters follow ``named_parameters``' default deduplication. Only one
    parameter is copied to CPU at a time; no model weights are saved to disk.
    """
    digests = {name: hashlib.sha256() for name in ("model", "action_head", "vlm")}
    numel = Counter()
    tensors = Counter()
    for name, parameter in sorted(model.named_parameters()):
        metadata = json.dumps(
            [name, list(parameter.shape), str(parameter.dtype)], separators=(",", ":")
        ).encode("utf-8") + b"\n"
        raw = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        scopes = ["model"]
        if name.startswith("action_model."):
            scopes.append("action_head")
        if name.startswith("qwen_vl_interface."):
            scopes.append("vlm")
        for scope in scopes:
            digests[scope].update(metadata)
            digests[scope].update(memoryview(raw))
            numel[scope] += parameter.numel()
            tensors[scope] += 1
    result = {
        "hash_format": "starvla-named-parameters-sha256-v1",
        "parameters": {
            scope: {"sha256": digest.hexdigest(), "numel": numel[scope], "tensors": tensors[scope]}
            for scope, digest in digests.items()
        },
    }
    vlm = getattr(getattr(model, "qwen_vl_interface", None), "model", None)
    vlm_config = getattr(vlm, "config", None)
    if vlm_config is not None:
        config_dict = vlm_config.to_dict()
        encoded = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        result["vlm_config"] = config_dict
        result["vlm_config_sha256"] = hashlib.sha256(encoded).hexdigest()
        result["vlm_checkpoint_identity"] = {
            "name_or_path": getattr(vlm_config, "_name_or_path", None),
            "commit_hash": getattr(vlm_config, "_commit_hash", None),
        }
    return result
