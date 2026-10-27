# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Pack GR00T ``VLAStepData`` into the starVLA sample dict.

Applies q01/q99 (or min-max) normalization via GR00T ``StateActionProcessor``.
Relative EEF or joint conversion runs only when ``use_relative_action`` is
true and the key's ``ActionConfig.rep`` is ``RELATIVE``. Qwen tokenization
stays in ``forward()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from gr00t.data.interfaces import BaseProcessor
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionRepresentation, EmbodimentTag, ModalityConfig, VLAStepData
from gr00t.data.utils import parse_modality_configs
from starVLA.dataloader.canonical_layout import (
    CanonicalLayout,
    RobotCanonicalMap,
    gather,
    scatter,
)


def _tag_value(tag: Any) -> str:
    if isinstance(tag, str):
        return tag
    return getattr(tag, "value", str(tag))


def _to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr[..., :3])


def _concat_groups(group_dict: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    pieces = []
    for key in keys:
        if key not in group_dict:
            raise KeyError(f"Missing joint group {key!r} in {list(group_dict)}")
        value = np.asarray(group_dict[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[None, :]
        pieces.append(value)
    return np.concatenate(pieces, axis=-1).astype(np.float32)


class _IdentityCollator:
    def __call__(self, batch):
        return batch


class StarVLAPackProcessor(BaseProcessor):
    """GR00T processor hook that emits starVLA ``{image, lang, action, ...}`` dicts."""

    attributes = []

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict | None = None,
        include_state: bool = False,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        use_relative_action: bool = False,
        canonical_layout: CanonicalLayout | None = None,
        robot_maps: dict[str, RobotCanonicalMap] | None = None,
    ):
        self.modality_configs = parse_modality_configs(modality_configs)
        self.include_state = include_state
        self.use_percentiles = use_percentiles
        self.use_relative_action = bool(use_relative_action)
        self.canonical_layout = canonical_layout
        self.robot_maps = robot_maps or {}
        self.state_action_processor = StateActionProcessor(
            modality_configs=self.modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=False,
            use_relative_action=self.use_relative_action,
        )
        self.training = True

    @property
    def collator(self):
        return _IdentityCollator()

    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        self.state_action_processor.set_statistics(statistics, override=override)
        # GR00T replaces a relative key's norm params with the raw relative-stats
        # dict, whose "min"/"max" are the full range. Training asked for q01/q99
        # via use_percentiles, so put those back into the bounds that normalize().
        if not (self.use_percentiles and self.use_relative_action):
            return
        norm_params = self.state_action_processor.norm_params
        for tag, configs in self.modality_configs.items():
            action = configs.get("action")
            if action is None or tag not in norm_params or "action" not in norm_params[tag]:
                continue
            for key in action.modality_keys:
                params = norm_params[tag]["action"].get(key)
                if params is None or "min" not in params:
                    continue
                # Bounds may be (D,) or (T, D). The split width is D, not T.
                params["dim"] = np.array(np.asarray(params["min"]).shape[-1])
            if not action.action_configs:
                continue
            for key, action_config in zip(action.modality_keys, action.action_configs):
                if action_config.rep != ActionRepresentation.RELATIVE:
                    continue
                params = norm_params[tag]["action"][key]
                if "q01" not in params or "q99" not in params:
                    continue
                params["min"] = np.asarray(params["q01"], dtype=np.float64)
                params["max"] = np.asarray(params["q99"], dtype=np.float64)
                params["dim"] = np.array(np.asarray(params["min"]).shape[-1])

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        step = None
        for message in messages:
            if message.get("type") == "episode_step":
                step = message["content"]
                break
        if step is None:
            raise ValueError("StarVLAPackProcessor expected an episode_step message")
        return self._pack_step(step)

    def _pack_step(self, step: VLAStepData) -> dict[str, Any]:
        tag = _tag_value(step.embodiment)
        configs = self.modality_configs[tag]

        raw_state = step.states or {}
        raw_action = step.actions or {}
        if self.include_state and raw_state and "state" in configs:
            norm_state = self.state_action_processor.apply_state(raw_state, tag)
        else:
            norm_state = raw_state
        norm_action = self.state_action_processor.apply_action(
            raw_action, tag, state=raw_state if raw_state else None
        )

        action_mask = None
        if self.canonical_layout is not None:
            robot_map = self._robot_map(tag)
            action, action_mask = scatter(norm_action, robot_map.action, self.canonical_layout.dim)
        else:
            action = _concat_groups(norm_action, list(configs["action"].modality_keys))

        images = []
        video_cfg = configs.get("video")
        if video_cfg is not None:
            for key in video_cfg.modality_keys:
                frames = (step.images or {}).get(key, [])
                if not frames:
                    raise KeyError(f"No frames for video key {key!r} (tag={tag})")
                images.append(_to_pil(frames[0]))

        sample = {
            "image": images,
            "lang": "" if step.text is None else str(step.text),
            "action": action,
            "robot_tag": tag,
        }
        if action_mask is not None:
            sample["action_mask"] = action_mask
        if self.include_state and "state" in configs and norm_state:
            if self.canonical_layout is not None:
                robot_map = self._robot_map(tag)
                state, state_mask = scatter(norm_state, robot_map.state, self.canonical_layout.dim)
                sample["state"] = state
                sample["state_mask"] = state_mask
            else:
                sample["state"] = _concat_groups(norm_state, list(configs["state"].modality_keys))
        return sample

    def _robot_map(self, tag: str) -> RobotCanonicalMap:
        if tag not in self.robot_maps:
            raise KeyError(
                f"No canonical robot map for {tag!r}. Available: {sorted(self.robot_maps)}"
            )
        return self.robot_maps[tag]

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag | str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        tag = _tag_value(embodiment_tag)
        action = np.asarray(action)
        if self.canonical_layout is not None:
            grouped = gather(action, self._robot_map(tag).action)
            return self.state_action_processor.unapply_action(grouped, tag, state=state)
        keys = list(self.modality_configs[tag]["action"].modality_keys)
        grouped = {}
        offset = 0
        for key in keys:
            dim = int(self.state_action_processor.norm_params[tag]["action"][key]["dim"])
            grouped[key] = action[..., offset : offset + dim]
            offset += dim
        return self.state_action_processor.unapply_action(grouped, tag, state=state)
