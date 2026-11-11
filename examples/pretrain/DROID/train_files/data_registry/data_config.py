"""DROID stage3-unmasked registration for the 90-D QwenPI canonical layout.

The source vectors are 17-D: absolute xyz+row-major rot6d (9), absolute
open-positive gripper (1), and absolute commanded joint positions (7).
Only the EEF pose becomes relative to the chunk-start observation during
training; gripper and joint targets remain absolute. The canonical robot map
places all 17 supervised dimensions in the right arm, aligned with LIBERO.

All three RGB cameras are used in the stable order front, exterior_2, wrist.
The mixture's parent data_root_dir is .../job/full, which contains dataset/.
"""

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


ROBOT_TYPE = "droid_franka_rot6d"
DATA_MIX = "droid_stage3_unmasked"
CAMERA_NAMES = ("front", "exterior_2", "wrist")


class DroidFrankaDataConfig:
    """Relative EEF + absolute gripper/joints; eight future action targets."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [f"video.{name}" for name in CAMERA_NAMES]
    # State is needed as the EEF reference even when include_state is false.
    # Keeping all groups also makes the canonical state definition complete.
    state_keys = ["state.arm_eef", "state.arm_gripper", "state.arm_joints"]
    action_keys = ["action.arm_eef", "action.arm_gripper", "action.arm_joints"]
    state_key_dims = {"state.arm_eef": 9, "state.arm_gripper": 1, "state.arm_joints": 7}
    action_key_dims = {"action.arm_eef": 9, "action.arm_gripper": 1, "action.arm_joints": 7}
    action_configs = [
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.EEF,
            format=ActionFormat.XYZ_ROT6D,
            state_key="arm_eef",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="arm_gripper",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="arm_joints",
        ),
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices, modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # gr00t_sharded uses StarVLAPackProcessor instead. This matches the
        # existing registries' fallback normalization interface.
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {ROBOT_TYPE: DroidFrankaDataConfig()}
DATASET_NAMED_MIXTURES = {DATA_MIX: [("dataset", 1.0, ROBOT_TYPE)]}
