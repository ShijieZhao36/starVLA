"""LIBERO-plus stage1 registration for QwenPI + gr00t_sharded.

Keys match ``meta/modality.json`` of
``/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1/dataset``:
front / right_wrist, right_eef (rotation_6d, 9) + right_gripper (1).
This is not the 7-D ``libero_franka`` layout.
"""

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class LiberoPlusRot6dDataConfig:
    """Two-camera, 10D EEF rotation_6d + gripper layout of converted LIBERO-plus stage1."""

    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.front",
        "video.right_wrist",
    ]
    state_keys = [
        "state.right_eef",
        "state.right_gripper",
    ]
    action_keys = [
        "action.right_eef",
        "action.right_gripper",
    ]
    state_key_dims = {
        "state.right_eef": 9,
        "state.right_gripper": 1,
    }
    action_key_dims = {
        "action.right_eef": 9,
        "action.right_gripper": 1,
    }
    # Parallel to action_keys. Consumed only by gr00t_sharded. Keys are unprefixed
    # because extract_step_data stores groups as "right_eef", not "action.right_eef".
    # right_eef is an absolute xyz+rot6d pose on disk; the processor subtracts the
    # chunk-start state when use_relative_action is true. The gripper stays absolute.
    action_configs = [
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.EEF,
            format=ActionFormat.XYZ_ROT6D,
            state_key="right_eef",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="right_gripper",
        ),
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

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


_ROBOT_TYPE = "libero_plus_rot6d"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TYPE: LiberoPlusRot6dDataConfig(),
}

# Directory name under data_root_dir.
DATASET_NAMED_MIXTURES = {
    "libero_plus_stage1": [
        ("dataset_absolute", 1.0, _ROBOT_TYPE),
    ],
}
