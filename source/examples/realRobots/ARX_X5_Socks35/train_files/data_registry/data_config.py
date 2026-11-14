"""Three-camera ARX5 SOCKS SFT35 data: absolute joint targets and grippers.

The source files and copied metadata use the order left6, left_gripper,
right6, right_gripper in action/observation.state. joint_action/joint_state
are deliberately not used: their gripper ordering differs.
"""
import json
from pathlib import Path

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class SocksSFT35AbsoluteJointDataConfig:
    embodiment_tag = EmbodimentTag.ARX_X5
    video_keys = ["video.front", "video.left_wrist", "video.right_wrist"]
    state_keys = ["state.left_arm_joints", "state.left_gripper", "state.right_arm_joints", "state.right_gripper"]
    action_keys = ["action.left_arm_joints", "action.left_gripper", "action.right_arm_joints", "action.right_gripper"]
    state_key_dims = {"state.left_arm_joints": 6, "state.left_gripper": 1, "state.right_arm_joints": 6, "state.right_gripper": 1}
    action_key_dims = {"action.left_arm_joints": 6, "action.left_gripper": 1, "action.right_arm_joints": 6, "action.right_gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))
    action_configs = [
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                     format=ActionFormat.DEFAULT, state_key=key)
        for key in ("left_arm_joints", "left_gripper", "right_arm_joints", "right_gripper")
    ]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # gr00t_sharded uses StarVLAPackProcessor; retain the registration contract.
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={key: "q99" for key in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={key: "q99" for key in self.action_keys}),
        ])


ROBOT_TYPE = "arx5_socks_sft35_h50"
ROBOT_TYPE_CONFIG_MAP = {ROBOT_TYPE: SocksSFT35AbsoluteJointDataConfig()}
_weights = json.loads(Path(__file__).with_name("mix_weights.json").read_text())
assert _weights["horizon"] == 50 and _weights["dataset_count"] == 35
assert len(_weights["datasets"]) == 35
assert sum(item["weight"] for item in _weights["datasets"]) == _weights["valid_starts"]
assert all(item["group"] == "socks_new_v1" and item["weight"] == item["valid_starts_h50"] for item in _weights["datasets"])
assert [item["subset"] for item in _weights["datasets"]] == Path(__file__).with_name("requested_datasets.txt").read_text().splitlines()
# A weight of N_i is the desired effective-sample mass. The scheduler already
# divides it by N_i / number_of_shards_i; multiplying by size again is incorrect.
DATASET_NAMED_MIXTURES = {
    "socks_sft35_abs_h50": [(item["alias"], float(item["weight"]), ROBOT_TYPE) for item in _weights["datasets"]]
}
