# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""LIBERO-plus eval policy for the 90-D canonical checkpoint.

The model emits a relative right EEF pose. At replan time that chunk is
converted to absolute targets with the observed end-effector. Each control
step then sends an OSC delta from the live end-effector to the stored target.
The gripper stays an absolute [0, 1] value; the rollout binarizes it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from gr00t.data.state_action.pose import EndEffectorPose
from starVLA.dataloader.canonical_layout import load_layout, load_robot_maps
from starVLA.dataloader.gr00t_sharded import data_config_to_gr00t_modality_configs
from starVLA.dataloader.gr00t_sharded_processor import StarVLAPackProcessor
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import read_mode_config

_ROT6D_POSITION_SCALE_M = 0.05
_ROT6D_ROTATION_SCALE_RAD = 0.5
_ROBOT_TYPE = "libero_plus_rot6d"
_LAYOUT = "examples/pretrain/canonical/bimanual_v1.yaml"
_ROBOTS = "examples/pretrain/canonical/robots"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch


def _libero_plus_data_config():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[3]
        / "simBenchmarks"
        / "LIBERO-plus"
        / "train_files"
        / "data_registry"
        / "data_config.py"
    )
    spec = importlib.util.spec_from_file_location("libero_plus_data_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LIBERO-plus data config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LiberoPlusRot6dDataConfig()


def _split_native_action_stats(flat: dict, keys: list[str], dims: list[int]) -> dict[str, dict[str, np.ndarray]]:
    grouped = {key: {} for key in keys}
    names = [name for name in ("min", "max", "mean", "std", "q01", "q99") if name in flat]
    if not names:
        raise KeyError(f"action stats are missing bound fields: {sorted(flat)}")
    for name in names:
        values = np.asarray(flat[name], dtype=np.float64)
        offset = 0
        for key, dim in zip(keys, dims):
            grouped[key][name] = values[..., offset : offset + dim]
            offset += dim
        if offset != values.shape[-1]:
            raise ValueError(
                f"Action stat {name} last dimension {values.shape[-1]} does not match native dims {dims}. "
                "Expected the concatenated right_eef + right_gripper vector, not the 90-D canonical vector."
            )
    return grouped


def osc_delta(target_eef: np.ndarray, live_eef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World xyz delta and relative rotvec, both in controller units."""
    target = np.asarray(target_eef, dtype=np.float64).reshape(9)
    live = np.asarray(live_eef, dtype=np.float64).reshape(9)
    delta_xyz = (target[:3] - live[:3]) / _ROT6D_POSITION_SCALE_M
    target_pose = EndEffectorPose(translation=target[:3], rotation=target[3:], rotation_type="rot6d")
    live_pose = EndEffectorPose(translation=live[:3], rotation=live[3:], rotation_type="rot6d")
    relative = target_pose - live_pose
    rotvec = Rotation.from_matrix(relative.homogeneous[:3, :3]).as_rotvec() / _ROT6D_ROTATION_SCALE_RAD
    return delta_xyz.astype(np.float32), rotvec.astype(np.float32)


class PolicyModel:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
        use_bf16=True,
        use_canonical_forward: bool = False,
        gripper_open_sign: str = "positive",
    ) -> None:
        del horizon, action_ensemble, action_ensemble_horizon, adaptive_ensemble_alpha, host, port, gripper_open_sign
        if image_size is None:
            image_size = [224, 224]
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        canonical_override = (
            "framework.action_model.diffusion_model_cfg.use_canonical_forward="
            + str(bool(use_canonical_forward)).lower()
        )
        print(f"Applying config override: {canonical_override}")
        vla = baseframework.from_pretrained(policy_ckpt_path, config_overrides=[canonical_override])
        if use_bf16:
            vla = vla.to(torch.bfloat16)
        self.vla = vla.to("cuda").eval()
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.task_description = None
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path)
        self._processor = self._build_processor()
        self._targets = None
        self._grippers = None

    def _build_processor(self) -> StarVLAPackProcessor:
        data_config = _libero_plus_data_config()
        modality_configs = data_config_to_gr00t_modality_configs(data_config)
        keys = [key.split(".", 1)[-1] for key in data_config.action_keys]
        dims = [int(data_config.action_key_dims[key]) for key in data_config.action_keys]
        grouped = _split_native_action_stats(self.action_norm_stats, keys, dims)
        action_configs = modality_configs["action"].action_configs or []
        relative = {
            key: grouped[key]
            for key, action_config in zip(keys, action_configs)
            if action_config.rep.name == "RELATIVE"
        }
        layout = load_layout(_LAYOUT)
        robot_maps = load_robot_maps(_ROBOTS, layout)
        processor = StarVLAPackProcessor(
            modality_configs={_ROBOT_TYPE: modality_configs},
            include_state=False,
            use_percentiles=bool(self.action_norm_stats.get("use_percentiles", True)),
            clip_outliers=True,
            use_relative_action=True,
            canonical_layout=layout,
            robot_maps=robot_maps,
        )
        processor.set_statistics(
            {_ROBOT_TYPE: {"action": grouped, "relative_action": relative}},
            override=True,
        )
        return processor

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self._targets = None
        self._grippers = None

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        import cv2 as cv

        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def _remember_chunk(self, example: dict, start_eef: np.ndarray) -> None:
        response = self.vla.predict_action(
            example,
            do_sample=False,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )
        normalized = np.asarray(response["normalized_actions"][0], dtype=np.float32)
        if normalized.shape[-1] != 90:
            raise ValueError(f"Expected a 90-D canonical action, got {normalized.shape}")
        decoded = self._processor.decode_action(
            normalized,
            _ROBOT_TYPE,
            state={"right_eef": np.asarray(start_eef, dtype=np.float32)[None, :]},
        )
        self._targets = np.asarray(decoded["right_eef"], dtype=np.float32)
        self._grippers = np.asarray(decoded["right_gripper"], dtype=np.float32).reshape(-1)
        if self._targets.shape[0] < self.action_chunk_size:
            raise ValueError(
                f"Decoded chunk length {self._targets.shape[0]} is shorter than {self.action_chunk_size}"
            )

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        del kwargs
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)
        images = [self._resize_image(image) for image in example["image"]]
        example = dict(example)
        example["image"] = images
        live = np.asarray(example["raw_right_eef"], dtype=np.float32).reshape(9)
        if step % self.action_chunk_size == 0 or self._targets is None:
            self._remember_chunk(example, live)
        index = step % self.action_chunk_size
        delta_xyz, rotvec = osc_delta(self._targets[index], live)
        return {
            "raw_action": {
                "world_vector": delta_xyz,
                "rotation_delta": rotvec,
                "open_gripper": np.asarray([self._grippers[index]], dtype=np.float32),
            }
        }

    @staticmethod
    def get_action_stats(unnorm_key: str | None, policy_ckpt_path) -> dict:
        _model_config, norm_stats = read_mode_config(policy_ckpt_path)
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    f"Pass unnorm_key. Available dataset stats: {sorted(norm_stats)}"
                )
            unnorm_key = next(iter(norm_stats))
        if unnorm_key not in norm_stats:
            raise KeyError(f"unnorm_key {unnorm_key!r} not in {sorted(norm_stats)}")
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path) -> int:
        model_config, _ = read_mode_config(policy_ckpt_path)
        return int(model_config["framework"]["action_model"]["future_action_window_size"]) + 1
