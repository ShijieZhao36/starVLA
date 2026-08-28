# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Pack GR00T ``VLAStepData`` into the starVLA sample dict.

Applies absolute min-max normalization via GR00T ``StateActionProcessor``
(no relative EEF conversion) and leaves Qwen tokenization to ``forward()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from gr00t.data.interfaces import BaseProcessor
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import EmbodimentTag, ModalityConfig, VLAStepData
from gr00t.data.utils import parse_modality_configs


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
        use_percentiles: bool = False,
        clip_outliers: bool = True,
    ):
        self.modality_configs = parse_modality_configs(modality_configs)
        self.include_state = include_state
        self.state_action_processor = StateActionProcessor(
            modality_configs=self.modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=False,
            use_relative_action=False,
        )
        self.training = True

    @property
    def collator(self):
        return _IdentityCollator()

    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        self.state_action_processor.set_statistics(statistics, override=override)

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
        if self.include_state and "state" in configs and norm_state:
            sample["state"] = _concat_groups(norm_state, list(configs["state"].modality_keys))
        return sample

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag | str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        tag = _tag_value(embodiment_tag)
        keys = list(self.modality_configs[tag]["action"].modality_keys)
        grouped = {}
        offset = 0
        action = np.asarray(action)
        for key in keys:
            dim = int(self.state_action_processor.norm_params[tag]["action"][key]["dim"])
            grouped[key] = action[..., offset : offset + dim]
            offset += dim
        return self.state_action_processor.unapply_action(grouped, tag, state=state)
