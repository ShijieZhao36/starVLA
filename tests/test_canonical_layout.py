import unittest
from pathlib import Path

import numpy as np

from starVLA.dataloader.canonical_layout import (
    gather,
    load_layout,
    load_robot_maps,
    parse_layout,
    parse_robot_map,
    scatter,
)

_REPO = Path(__file__).resolve().parents[1]
_LAYOUT = _REPO / "examples/pretrain/canonical/bimanual_v1.yaml"
_ROBOTS = _REPO / "examples/pretrain/canonical/robots"


def _block(width: int, fill: float, steps: int = 2) -> np.ndarray:
    return np.full((steps, width), fill, dtype=np.float32)


class CanonicalLayoutTest(unittest.TestCase):
    def setUp(self):
        self.layout = load_layout(_LAYOUT)
        self.maps = load_robot_maps(_ROBOTS, self.layout)
        self.robot = self.maps["galaxea_r1lite_stage1"]

    def test_widths_sum_to_declared_dim(self):
        self.assertEqual(self.layout.dim, 90)
        covered = sum(field.dim for field in self.layout.fields.values())
        self.assertEqual(covered, 90)
        self.assertEqual(self.layout.fields["left.joint_pos"].dim, 7)
        self.assertEqual(self.layout.fields["left.fingertips"].dim, 15)
        self.assertEqual(self.layout.fields["reserved"].dim, 26)

    def test_galaxea_action_mask_and_joint_padding(self):
        groups = {
            "left_arm_joints": np.arange(6, dtype=np.float32)[None, :],
            "left_eef": _block(9, 0.1, steps=1),
            "left_gripper": _block(1, 0.2, steps=1),
            "right_arm_joints": _block(6, 0.3, steps=1),
            "right_eef": _block(9, 0.4, steps=1),
            "right_gripper": _block(1, 0.5, steps=1),
            "torso_twist": _block(6, 0.6, steps=1),
            "chassis_velocity": _block(3, 0.7, steps=1),
        }
        vector, mask = scatter(groups, self.robot.action, self.layout.dim)
        self.assertEqual(vector.shape, (1, 90))
        self.assertEqual(int(mask.sum()), 41)

        joints = self.layout.fields["left.joint_pos"]
        self.assertTrue(np.all(mask[0, joints.start : joints.start + 6] == 1))
        self.assertEqual(mask[0, joints.start + 6], 0)
        self.assertTrue(np.allclose(vector[0, joints.start : joints.start + 6], np.arange(6)))
        self.assertEqual(vector[0, joints.start + 6], 0)

        fingers = self.layout.fields["left.fingertips"]
        self.assertTrue(np.all(mask[0, fingers.start : fingers.start + fingers.dim] == 0))

        reserved = self.layout.fields["reserved"]
        self.assertTrue(np.all(mask[0, reserved.start : reserved.start + 6] == 1))
        self.assertTrue(np.allclose(vector[0, reserved.start : reserved.start + 6], 0.6))
        self.assertTrue(np.all(mask[0, reserved.start + 6 : reserved.start + 9] == 1))
        self.assertTrue(np.allclose(vector[0, reserved.start + 6 : reserved.start + 9], 0.7))
        self.assertTrue(np.all(mask[0, reserved.start + 9 : reserved.start + reserved.dim] == 0))

        restored = gather(vector, self.robot.action)
        self.assertEqual(restored["left_arm_joints"].shape[-1], 6)
        self.assertTrue(np.allclose(restored["left_arm_joints"], groups["left_arm_joints"]))
        self.assertTrue(np.allclose(restored["torso_twist"], groups["torso_twist"]))
        self.assertTrue(np.allclose(restored["chassis_velocity"], groups["chassis_velocity"]))

    def test_galaxea_state_aligns_torso_and_leaves_chassis_empty(self):
        groups = {
            "left_arm_joints": _block(6, 1),
            "left_eef": _block(9, 2),
            "left_gripper": _block(1, 3),
            "right_arm_joints": _block(6, 4),
            "right_eef": _block(9, 5),
            "right_gripper": _block(1, 6),
            "torso": _block(3, 7),
        }
        vector, mask = scatter(groups, self.robot.state, self.layout.dim)
        self.assertEqual(int(mask[0].sum()), 35)
        reserved = self.layout.fields["reserved"]
        self.assertTrue(np.allclose(vector[0, reserved.start : reserved.start + 3], 7))
        self.assertTrue(np.all(mask[0, reserved.start : reserved.start + 3] == 1))
        self.assertTrue(np.all(mask[0, reserved.start + 3 : reserved.start + 9] == 0))

    def test_xperience_mask_covers_eef_and_fingertips(self):
        robot = self.maps["xperience_human"]
        groups = {
            "left_eef": _block(9, 0.1, steps=1),
            "left_fingertips": np.arange(15, dtype=np.float32)[None, :],
            "right_eef": _block(9, 0.4, steps=1),
            "right_fingertips": _block(15, 0.5, steps=1),
        }
        empty = (
            "left.joint_pos",
            "left.gripper",
            "right.joint_pos",
            "right.gripper",
            "reserved",
        )
        for placements in (robot.action, robot.state):
            vector, mask = scatter(groups, placements, self.layout.dim)
            self.assertEqual(vector.shape, (1, 90))
            self.assertEqual(int(mask.sum()), 48)
            for name in empty:
                field = self.layout.fields[name]
                span = slice(field.start, field.start + field.dim)
                self.assertTrue(np.all(mask[0, span] == 0))
                self.assertTrue(np.all(vector[0, span] == 0))
            left_eef = self.layout.fields["left.eef_pose"]
            left_fingers = self.layout.fields["left.fingertips"]
            right_fingers = self.layout.fields["right.fingertips"]
            self.assertTrue(np.allclose(vector[0, left_eef.start : left_eef.start + 9], 0.1))
            self.assertTrue(np.all(mask[0, left_fingers.start : left_fingers.start + 15] == 1))
            self.assertTrue(
                np.allclose(vector[0, left_fingers.start : left_fingers.start + 15], np.arange(15))
            )
            self.assertTrue(np.all(mask[0, right_fingers.start : right_fingers.start + 15] == 1))
            self.assertTrue(np.allclose(vector[0, right_fingers.start : right_fingers.start + 15], 0.5))
            restored = gather(vector, placements)
            self.assertTrue(np.allclose(restored["left_fingertips"], groups["left_fingertips"]))
            self.assertTrue(np.allclose(restored["right_eef"], groups["right_eef"]))

    def test_overlapping_write_is_rejected(self):
        data = {
            "robot_type": "overlap",
            "action": [
                {"field": "reserved", "source": "a", "start": 0, "length": 6},
                {"field": "reserved", "source": "b", "start": 4, "length": 4},
            ],
        }
        with self.assertRaises(ValueError):
            parse_robot_map(data, self.layout)

    def test_field_sum_must_match_dim(self):
        data = {
            "dim": 10,
            "arms": ["left"],
            "arm_fields": [{"name": "joint_pos", "dim": 7}],
            "shared": [{"name": "reserved", "dim": 2}],
        }
        with self.assertRaises(ValueError):
            parse_layout(data)


if __name__ == "__main__":
    unittest.main()
