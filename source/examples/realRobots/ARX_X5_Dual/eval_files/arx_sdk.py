#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Evo-RL `lerobot.robots.arx5_follower.arx5_client` to load the
# official ARX X5 Python SDK (`bimanual.SingleArm`) without depending on Evo-RL.

"""Thin wrapper around the official ARX X5 Python SDK."""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ARX_SDK_ROOT = Path(
    os.path.expanduser(os.environ.get("ARX_SDK_ROOT", "~/workspace/ARX_X5/py/arx_x5_python"))
)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 5.0


def _setup_sdk_search_path(sdk_root: Path) -> None:
    if sdk_root.is_dir() and str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))


def _preload_sdk_shared_objects(sdk_root: Path) -> None:
    if not sdk_root.is_dir():
        return
    shared_object_dirs = [
        sdk_root / "bimanual" / "api" / "arx_x5_src",
        sdk_root / "bimanual" / "api",
    ]
    for directory in shared_object_dirs:
        if not directory.is_dir():
            continue
        for shared_object in sorted(directory.glob("*.so")):
            if shared_object.name.endswith("-arm64.so"):
                continue
            try:
                ctypes.cdll.LoadLibrary(str(shared_object))
            except OSError as error:
                logger.debug("Skipping optional ARX5 shared object %s: %s", shared_object, error)


def load_single_arm_class(sdk_root: Path | None = None):
    """Return vendor `SingleArm`, or None if the SDK cannot be imported."""
    root = Path(sdk_root).expanduser().resolve() if sdk_root is not None else DEFAULT_ARX_SDK_ROOT
    _setup_sdk_search_path(root)
    _preload_sdk_shared_objects(root)
    try:
        return importlib.import_module("bimanual").SingleArm
    except ImportError:
        logger.warning("ARX5 SDK not found under %s. Using stub arm.", root)
        return None


class _StubArm:
    """Fallback arm used when the vendor SDK or hardware is unavailable."""

    def __init__(self, config: dict[str, Any]):
        self._joint_positions = np.zeros(7, dtype=np.float64)
        self._ee_pose = np.zeros(7, dtype=np.float64)
        logger.info("Using ARX5 stub arm with config=%s", config)

    def go_home(self) -> None:
        self._joint_positions[:] = 0.0
        self._ee_pose[:] = 0.0

    def protect_mode(self) -> None:
        return

    def set_joint_positions(self, joints: list[float]) -> None:
        joints_array = np.asarray(joints, dtype=np.float64)
        self._joint_positions[: min(6, joints_array.size)] = joints_array[:6]

    def set_catch_pos(self, value: float) -> None:
        self._ee_pose[6] = float(value)
        self._joint_positions[6] = float(value)

    def get_joint_positions(self) -> list[float]:
        return self._joint_positions.tolist()


class ARX5ArmClient:
    """Thin wrapper around the ARX5 vendor SDK with a deterministic stub fallback."""

    def __init__(
        self,
        *,
        can_port: str,
        arm_type: int = 0,
        use_stub: bool = False,
        sdk_root: Path | str | None = None,
    ):
        self._last_sent_pose = [0.0] * 7
        single_arm_cls = None if use_stub else load_single_arm_class(sdk_root)
        if use_stub or single_arm_cls is None:
            self.arm = _StubArm({"can_port": can_port, "type": arm_type})
        else:
            self.arm = single_arm_cls({"can_port": can_port, "type": arm_type})
        self._last_sent_pose = self.get_state()

    def _read_gripper_position(self) -> float:
        if hasattr(self.arm, "get_catch_pos"):
            value = self.arm.get_catch_pos()
            if isinstance(value, (list, tuple, np.ndarray)):
                return float(value[0]) if len(value) > 0 else 0.0
            return float(value)
        if len(self._last_sent_pose) >= 7:
            return float(self._last_sent_pose[6])
        return 0.0

    def get_state(self) -> list[float]:
        joint_positions = np.asarray(self.arm.get_joint_positions(), dtype=np.float64)
        gripper = self._read_gripper_position()
        state = np.zeros(7, dtype=np.float64)
        state[: min(6, joint_positions.size)] = joint_positions[:6]
        state[6] = gripper
        return state.tolist()

    def send_joint(self, joint: list[float]) -> list[float]:
        full_joint = list(joint[:7])
        if len(full_joint) < 7:
            full_joint.extend([0.0] * (7 - len(full_joint)))
        self.arm.set_joint_positions(full_joint[:6])
        self.arm.set_catch_pos(float(full_joint[6]))
        self._last_sent_pose = full_joint
        return full_joint

    def hold_position(self) -> None:
        self.send_joint(self.get_state())
        time.sleep(0.2)

    def go_home(self) -> None:
        self.arm.go_home()

    def protect_mode(self) -> None:
        self.arm.protect_mode()
