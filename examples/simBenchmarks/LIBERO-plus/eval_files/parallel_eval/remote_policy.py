"""Remote policy with the same ``step`` contract as ``PolicyModel``.

Images are resized here, before the request, so the server does not resize a
second time. The unnormalized chunk is cached on this client.
"""

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from gr00t.data.state_action.pose import EndEffectorPose

_ROT6D_POSITION_SCALE_M = 0.05
_ROT6D_ROTATION_SCALE_RAD = 0.5


def osc_delta(target_eef: np.ndarray, live_eef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World xyz delta and world-frame rotvec, both in controller units.

    OSC_POSE left-multiplies, so ``R_delta = R_target @ R_live.T``.
    """
    target = np.asarray(target_eef, dtype=np.float64).reshape(9)
    live = np.asarray(live_eef, dtype=np.float64).reshape(9)
    delta_xyz = (target[:3] - live[:3]) / _ROT6D_POSITION_SCALE_M
    target_pose = EndEffectorPose(translation=target[:3], rotation=target[3:], rotation_type="rot6d")
    live_pose = EndEffectorPose(translation=live[:3], rotation=live[3:], rotation_type="rot6d")
    rotation_delta = target_pose.homogeneous[:3, :3] @ live_pose.homogeneous[:3, :3].T
    rotvec = Rotation.from_matrix(rotation_delta).as_rotvec() / _ROT6D_ROTATION_SCALE_RAD
    return delta_xyz.astype(np.float32), rotvec.astype(np.float32)


class RemoteChunkPolicy:
    def __init__(self, host: str, port: int, image_size=None) -> None:
        if image_size is None:
            image_size = [224, 224]
        self.client = WebsocketClientPolicy(host, port)
        metadata = self.client.get_server_metadata()
        self.action_chunk_size = int(metadata["action_chunk_size"])
        self.image_size = image_size
        self.task_description = None
        self.raw_actions = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.raw_actions = None

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        # Match PolicyModel._resize_image. cv2.resize takes (width, height).
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def step(self, example: dict, step: int = 0, **kwargs):
        del kwargs
        task_description = example.get("lang", None)
        if example is not None and task_description != self.task_description:
            self.reset(task_description)

        example["image"] = [self._resize_image(image) for image in example["image"]]
        if step % self.action_chunk_size == 0:
            response = self.client.predict_action({"example": example})
            if not response.get("ok", False):
                raise RuntimeError(f"policy server error: {response}")
            self.raw_actions = np.asarray(response["data"]["actions"])

        if self.raw_actions is None:
            raise RuntimeError("policy step requested an action before the first chunk")

        chunk = self.raw_actions[0] if self.raw_actions.ndim == 3 else self.raw_actions
        row = chunk[step % self.action_chunk_size]
        if row.shape[-1] == 7:
            raw_action = {
                "world_vector": np.array(row[:3]),
                "rotation_delta": np.array(row[3:6]),
                "open_gripper": np.array(row[6:7]),  # range [0, 1]; 1 = open; 0 = close
            }
            return {"raw_action": raw_action}
        if row.shape[-1] == 10:
            live = example.get("raw_right_eef")
            if live is None:
                raise RuntimeError("90-d absolute targets require raw_right_eef on every step")
            delta_xyz, rotvec = osc_delta(row[:9], live)
            raw_action = {
                "world_vector": delta_xyz,
                "rotation_delta": rotvec,
                "open_gripper": np.array(row[9:10], dtype=np.float32),
            }
            return {"raw_action": raw_action}
        raise RuntimeError(f"Unexpected action width {row.shape[-1]}; expected 7 or 10")
