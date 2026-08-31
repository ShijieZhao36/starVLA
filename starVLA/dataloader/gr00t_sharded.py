# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""starVLA adapter around unmodified GR00T sharded LeRobot I/O.

``dataset_py: gr00t_sharded`` keeps the starVLA sample contract and mix registry,
but loads via GR00T ``ShardedSingleStepDataset`` / ``ShardedMixtureDataset``.
Video decoding uses GR00T ``video_utils`` (torchcodec); YAML ``video_backend``
is ignored on this path.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import IterableDataset

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_stats
from gr00t.data.types import ModalityConfig as Gr00tModalityConfig
from gr00t.utils.dist_utils import run_or_wait_on_rank0
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_sharded_processor import StarVLAPackProcessor, _tag_value

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return batch


class StatsTag:
    """Duck-typed GR00T ``EmbodimentTag`` so stats merge by starVLA robot tag."""

    def __init__(self, value: str):
        self.value = str(value)
        self.name = self.value.upper().replace("-", "_")

    def __repr__(self) -> str:
        return f"StatsTag({self.value!r})"


class TaggedShardedSingleStepDataset(ShardedSingleStepDataset):
    def __init__(self, *args, stats_tag: str, **kwargs):
        super().__init__(*args, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, **kwargs)
        self.embodiment_tag = StatsTag(stats_tag)


class Gr00tShardedMixture(IterableDataset):
    """Iterable wrapper that exposes ``__len__`` and starVLA stats dump."""

    def __init__(
        self,
        mixture: ShardedMixtureDataset,
        processor: StarVLAPackProcessor,
        virtual_length: int,
    ):
        super().__init__()
        self.mixture = mixture
        self.processor = processor
        self._virtual_length = max(int(virtual_length), 1)

    def __iter__(self):
        return iter(self.mixture)

    def __len__(self) -> int:
        return self._virtual_length

    def get_dataset_statistics(self) -> dict:
        return self.mixture.get_dataset_statistics()

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        if format != "json":
            raise ValueError(f"Unsupported stats format: {format}")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = freeze_concat_statistics(
            self.get_dataset_statistics(),
            self.processor.modality_configs,
            use_percentiles=bool(getattr(self.processor, "use_percentiles", True)),
        )
        with open(save_path, "w") as f:
            json.dump(_jsonify(payload), f, indent=2)
        logger.info("Saved dataset statistics file at path %s", save_path)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _as_list(value: Any) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_as_list(item))
            else:
                out.append(float(item) if isinstance(item, (np.floating, np.integer, float, int)) else item)
        return out
    return [value]


def freeze_concat_statistics(
    global_stats: dict,
    modality_configs: dict[str, dict[str, Gr00tModalityConfig]],
    use_percentiles: bool = True,
) -> dict:
    """Flatten per-group GR00T stats into concat vectors keyed by embodiment tag."""
    frozen = {}
    for tag, tag_stats in (global_stats or {}).items():
        if tag == "__fingerprints__" or not isinstance(tag_stats, dict):
            continue
        tag_out = {}
        configs = modality_configs.get(tag, {})
        for modality, out_name in (("action", "action"), ("state", "proprio")):
            if modality not in tag_stats:
                continue
            keys = list(configs[modality].modality_keys) if modality in configs else list(tag_stats[modality])
            combined = {name: [] for name in ("min", "max", "mean", "std", "q01", "q99")}
            for key in keys:
                if key not in tag_stats[modality]:
                    continue
                group = tag_stats[modality][key]
                for name in combined:
                    if name in group:
                        combined[name].extend(_as_list(group[name]))
            if combined["min"]:
                combined["mask"] = [True] * len(combined["min"])
                combined["use_percentiles"] = bool(use_percentiles)
                tag_out[out_name] = combined
        if tag_out:
            frozen[tag] = tag_out
    return frozen


def _strip_modality_prefix(key: str, modality: str) -> str:
    prefix = f"{modality}."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def data_config_to_gr00t_modality_configs(data_config) -> dict[str, Gr00tModalityConfig]:
    """Convert starVLA ``DataConfig.modality_config()`` into GR00T key layout."""
    src = data_config.modality_config()
    out: dict[str, Gr00tModalityConfig] = {}
    for modality, cfg in src.items():
        keys = list(cfg.modality_keys)
        if modality in ("video", "state", "action"):
            keys = [_strip_modality_prefix(k, modality) for k in keys]
        if not keys:
            continue
        out[modality] = Gr00tModalityConfig(
            delta_indices=list(cfg.delta_indices),
            modality_keys=keys,
            action_configs=None,
        )
    return out


def stats_tag_for(data_config, robot_type: str) -> str:
    tag = getattr(data_config, "embodiment_tag", None)
    if tag is not None:
        return _tag_value(tag)
    return str(robot_type)


def load_modality_config_path(modality_config_path: str) -> None:
    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        logger.info("Loaded modality config: %s", path)
        return
    raise FileNotFoundError(
        f"Modality config path does not exist or is not a .py file: {modality_config_path}"
    )


def resolve_modality_configs(
    data_config,
    robot_type: str,
    prefer_registry: bool = False,
) -> dict[str, Gr00tModalityConfig]:
    tag = stats_tag_for(data_config, robot_type)
    if prefer_registry and tag in MODALITY_CONFIGS:
        return MODALITY_CONFIGS[tag]
    return data_config_to_gr00t_modality_configs(data_config)


def _cfg_get(data_cfg, key: str, default=None):
    if data_cfg is None:
        return default
    getter = getattr(data_cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return data_cfg[key] if key in data_cfg else default


def _warn_ignored_yaml(data_cfg) -> None:
    action_mode = _cfg_get(data_cfg, "action_mode", None)
    action_type = _cfg_get(data_cfg, "action_type", None)
    ignored = [v for v in (action_mode, action_type) if v not in (None, "abs", "abs_only")]
    if ignored:
        logger.warning(
            "gr00t_sharded uses abs_only min-max; ignoring action_mode/action_type=%r",
            ignored,
        )
    video_backend = _cfg_get(data_cfg, "video_backend", None)
    if video_backend is not None:
        logger.warning(
            "gr00t_sharded uses torchcodec via GR00T video_utils; ignoring video_backend=%r",
            video_backend,
        )


def _include_state(data_cfg) -> bool:
    return _cfg_get(data_cfg, "include_state", False) not in ["False", False]


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def get_vla_dataset(
    data_cfg,
    mode: str = "train",
    seed: int = 42,
    **kwargs,
) -> Gr00tShardedMixture:
    """Build a GR00T-sharded iterable mixture for the named ``data_mix``."""
    _ = kwargs
    _warn_ignored_yaml(data_cfg)

    modality_config_path = _cfg_get(data_cfg, "modality_config_path", None)
    if modality_config_path:
        load_modality_config_path(str(modality_config_path))

    data_root_dir = Path(_cfg_get(data_cfg, "data_root_dir"))
    data_mix = _cfg_get(data_cfg, "data_mix")
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(
        "[gr00t_sharded] Using mixture '%s': %s",
        data_mix,
        [(d, w, r) for d, w, r in mixture_spec],
    )

    shard_size = int(_cfg_get(data_cfg, "shard_size", 1024))
    episode_sampling_rate = float(_cfg_get(data_cfg, "episode_sampling_rate", 1.0))
    allow_padding = bool(_cfg_get(data_cfg, "allow_padding", False))
    num_shards_per_epoch = int(_cfg_get(data_cfg, "num_shards_per_epoch", int(1e5)))
    include_state = _include_state(data_cfg)
    use_percentiles = _as_bool(_cfg_get(data_cfg, "use_percentiles", True), default=True)
    logger.info(
        "[gr00t_sharded] action/state bounds=%s (YAML use_percentiles)",
        "q01/q99" if use_percentiles else "min/max",
    )

    included = set()
    filtered = []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included:
            logger.warning("Skipping Duplicate Dataset: %s", (d_name, d_weight, robot_type))
            continue
        included.add(dataset_key)
        filtered.append((d_name, d_weight, robot_type))

    datasets = []
    weights = []
    processor_modality_configs: dict[str, dict[str, Gr00tModalityConfig]] = {}

    for d_name, d_weight, robot_type in filtered:
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(
                f"Unknown robot_type={robot_type!r} for dataset {d_name!r}. "
                f"Available: {sorted(ROBOT_TYPE_CONFIG_MAP)}"
            )
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        tag = stats_tag_for(data_config, robot_type)
        modality_configs = resolve_modality_configs(
            data_config, robot_type, prefer_registry=bool(modality_config_path)
        )
        if tag not in processor_modality_configs:
            processor_modality_configs[tag] = modality_configs
        dataset_path = data_root_dir / d_name
        with run_or_wait_on_rank0(label=f"generate_stats({dataset_path})") as is_rank0:
            if is_rank0:
                generate_stats(dataset_path)
        dataset = TaggedShardedSingleStepDataset(
            dataset_path=dataset_path,
            stats_tag=tag,
            modality_configs=modality_configs,
            shard_size=shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=seed,
            allow_padding=allow_padding,
        )
        datasets.append(dataset)
        weights.append(float(d_weight))

    processor = StarVLAPackProcessor(
        modality_configs=processor_modality_configs,
        include_state=include_state,
        use_percentiles=use_percentiles,
        clip_outliers=True,
    )
    mixture = ShardedMixtureDataset(
        datasets=datasets,
        weights=weights,
        processor=processor,
        seed=seed,
        training=(mode == "train"),
        num_shards_per_epoch=num_shards_per_epoch,
        override_pretraining_statistics=True,
    )
    total_steps = 0
    for dataset in datasets:
        total_steps += int(np.sum(dataset.shard_lengths))
    return Gr00tShardedMixture(
        mixture=mixture,
        processor=processor,
        virtual_length=max(total_steps, 1),
    )
