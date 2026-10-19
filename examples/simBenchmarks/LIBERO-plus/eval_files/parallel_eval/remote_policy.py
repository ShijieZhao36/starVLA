"""Remote policy with the same ``step`` contract as ``PolicyModel``.

Images are resized here, before the request, so the server does not resize a
second time. The unnormalized chunk is cached on this client.
"""

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


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

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        return {"raw_action": raw_action}
