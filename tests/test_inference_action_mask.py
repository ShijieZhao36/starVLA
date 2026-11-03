import unittest
from pathlib import Path

import numpy as np

from starVLA.dataloader.canonical_layout import (
    build_inference_action_mask,
    load_layout,
    load_robot_maps,
    scatter,
)


class InferenceActionMaskTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        layout = load_layout(root / "examples/pretrain/canonical/bimanual_v1.yaml")
        self.maps = load_robot_maps(root / "examples/pretrain/canonical/robots", layout)

    def test_each_robot_matches_training_scatter(self):
        tags = list(self.maps)
        masks = build_inference_action_mask([{"robot_tag": tag} for tag in tags], 8, 90, self.maps)
        for index, tag in enumerate(tags):
            placements = self.maps[tag].action
            groups = {p.source: np.ones((8, p.length)) for p in placements}
            _, expected = scatter(groups, placements, 90)
            np.testing.assert_array_equal(masks[index], expected)
        libero = masks[tags.index("libero_plus_rot6d")]
        np.testing.assert_array_equal(np.flatnonzero(libero[0]), np.arange(39, 49))

    def test_explicit_training_masks_align_to_target_horizon(self):
        full = np.zeros((12, 90))
        full[:, 39:49] = 1
        full[-1, 48] = 0
        actual = build_inference_action_mask([{"action_mask": full}], 8, 90, self.maps)
        np.testing.assert_array_equal(actual[0], full[-8:])

    def test_static_mask_broadcasts(self):
        mask = np.array([1, 0, 1])
        for static in [mask, mask[None]]:
            actual = build_inference_action_mask([{"action_mask": static}], 2, 3)
            np.testing.assert_array_equal(actual, [[[1, 0, 1], [1, 0, 1]]])

    def test_legacy_unmasked_path(self):
        self.assertIsNone(build_inference_action_mask([{}], 8, 10))

    def test_canonical_unknown_or_missing_tag_fails(self):
        for sample in [{}, {"robot_tag": "unknown"}]:
            with self.assertRaisesRegex(ValueError, "known robot_tag"):
                build_inference_action_mask([sample], 8, 90, self.maps)

    def test_invalid_mask_fails(self):
        for mask in [np.zeros((8, 90)), np.ones((7, 90)), np.ones((8, 89)), np.full((8, 90), np.nan), np.full((8, 90), 0.5)]:
            with self.assertRaises(ValueError):
                build_inference_action_mask([{"action_mask": mask}], 8, 90)
        with self.assertRaises(ValueError):
            build_inference_action_mask([{"action_mask": np.ones(90)}, {}], 8, 90)


if __name__ == "__main__":
    unittest.main()
