"""ARX X5 dual-arm LeRobot v2.1 registration for PI05 + gr00t_sharded.

Keys match converted ``meta/modality.json`` (front / left_wrist / right_wrist,
left_arm + gripper + right_arm + gripper), not Robotwin/RoboDojo cam_high /
left_joints names.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class ArxX5DualPI05DataConfig:
    """Three-camera, 14D joint+gripper layout used by converted ARX X5 dual datasets."""

    embodiment_tag = EmbodimentTag.ARX_X5
    video_keys = [
        "video.front",
        "video.left_wrist",
        "video.right_wrist",
    ]
    state_keys = [
        "state.left_arm",
        "state.left_gripper",
        "state.right_arm",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]
    state_key_dims = {
        "state.left_arm": 6,
        "state.left_gripper": 1,
        "state.right_arm": 6,
        "state.right_gripper": 1,
    }
    action_key_dims = {
        "action.left_arm": 6,
        "action.left_gripper": 1,
        "action.right_arm": 6,
        "action.right_gripper": 1,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # Unused by gr00t_sharded (StarVLAPackProcessor + use_percentiles).
        # Kept so the same robot_type can switch back to lerobot_datasets.
        state_modes = {key: "q99" for key in self.state_keys}
        action_modes = {key: "q99" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


_ROBOT_TYPE = "arx_x5_dual_pi05"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TYPE: ArxX5DualPI05DataConfig(),
}

# Append sibling LeRobot roots under data_root_dir to train more tasks together.
DATASET_NAMED_MIXTURES = {
    "arx_x5_dual": [
        ("lerobot_subtask_coffee_v10_0721day_815checked", 1.0, _ROBOT_TYPE),
        ("lerobot_subtask_coffee_v10_0721night_815checked", 1.0, _ROBOT_TYPE),
        ("lerobot_subtask_coffee_v10_0722day_815checked", 1.0, _ROBOT_TYPE),
        ("lerobot_subtask_coffee_v10_0722night_815checked", 1.0, _ROBOT_TYPE),
        ("lerobot_subtask_coffee_v10_0723day_815checked", 1.0, _ROBOT_TYPE),
    ],
}
