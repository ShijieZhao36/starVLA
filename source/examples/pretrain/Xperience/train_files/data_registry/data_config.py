"""Xperience human hours100 registration for QwenPI + gr00t_sharded.

Disk state and action are 48-D and absolute, in the segment-first left stereo
camera frame. Training rewrites ``left_eef`` and ``right_eef`` relative to the
chunk-start state. Fingertips stay absolute. ``joint_pos``, ``gripper``, and
``reserved`` are absent on disk and stay zero in the 90-D canonical vector.

The only camera is ``front``. Language is ``task_index`` via ``tasks.jsonl``.

Paths are relative to ``data_root_dir``
``/mnt/zhaoshijie/dataset/xperience_runs/stage1``.
"""

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class XperienceHumanDataConfig:
    """One-camera human layout: 48-D action, relative dual EEF, absolute fingertips."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front",
    ]
    # Only the reference poses for relative EEF. include_state stays false.
    state_keys = [
        "state.left_eef",
        "state.right_eef",
    ]
    action_keys = [
        "action.left_eef",
        "action.left_fingertips",
        "action.right_eef",
        "action.right_fingertips",
    ]
    state_key_dims = {
        "state.left_eef": 9,
        "state.right_eef": 9,
    }
    action_key_dims = {
        "action.left_eef": 9,
        "action.left_fingertips": 15,
        "action.right_eef": 9,
        "action.right_fingertips": 15,
    }
    # Parallel to action_keys. Consumed only by gr00t_sharded. Keys are unprefixed
    # because extract_step_data stores groups as "left_eef", not "action.left_eef".
    # Both EEFs are absolute xyz+rot6d on disk. Fingertips are absolute xyz.
    action_configs = [
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.EEF,
            format=ActionFormat.XYZ_ROT6D,
            state_key="left_eef",
        ),
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
            state_key="left_fingertips",
        ),
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
            state_key="right_fingertips",
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


_ROBOT_TYPE = "xperience_human"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TYPE: XperienceHumanDataConfig(),
}

DATASET_NAMED_MIXTURES = {
    "xperience_hours100": [
        ("hours100", 1.0, _ROBOT_TYPE),
    ],
}
