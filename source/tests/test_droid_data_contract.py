"""CPU contract tests for DROID's 17-D source and shared frame-cache layout."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gr00t.data.types import ActionRepresentation, VLAStepData
from starVLA.dataloader.canonical_layout import gather, load_layout, load_robot_maps, scatter
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_sharded import _dataset_frame_cache_dir, data_config_to_gr00t_modality_configs
from starVLA.dataloader.precompute_galaxea_frame_cache import build_jobs
from starVLA.dataloader.gr00t_sharded_processor import StarVLAPackProcessor


REPO = Path(__file__).resolve().parents[1]
ROBOT_TYPE = "droid_franka_rot6d"
DATA_MIX = "droid_stage3_unmasked"


class DroidDataContractTest(unittest.TestCase):
    def test_auto_discovery_and_relative_eef_only(self):
        self.assertEqual(DATASET_NAMED_MIXTURES[DATA_MIX], [("dataset", 1.0, ROBOT_TYPE)])
        config = ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE]
        self.assertEqual(config.video_keys, ["video.front", "video.exterior_2", "video.wrist"])
        self.assertEqual(sum(config.action_key_dims.values()), 17)
        self.assertEqual(sum(config.state_key_dims.values()), 17)
        modalities = data_config_to_gr00t_modality_configs(config)
        self.assertEqual(modalities["action"].delta_indices, list(range(8)))
        self.assertEqual(modalities["state"].delta_indices, [0])
        self.assertEqual(
            [item.rep for item in modalities["action"].action_configs],
            [ActionRepresentation.RELATIVE, ActionRepresentation.ABSOLUTE, ActionRepresentation.ABSOLUTE],
        )
        self.assertEqual(modalities["action"].action_configs[0].state_key, "arm_eef")

    def test_every_native_dimension_survives_right_arm_packing(self):
        layout = load_layout(REPO / "examples/pretrain/canonical/bimanual_v1.yaml")
        robot = load_robot_maps(REPO / "examples/pretrain/canonical/robots", layout)[ROBOT_TYPE]
        native = np.arange(8 * 17, dtype=np.float32).reshape(8, 17)
        groups = {"arm_eef": native[:, :9], "arm_gripper": native[:, 9:10], "arm_joints": native[:, 10:]}
        for placements in (robot.action, robot.state):
            packed, mask = scatter(groups, placements, layout.dim)
            self.assertEqual(packed.shape, (8, 90))
            np.testing.assert_array_equal(mask.sum(axis=-1), np.full(8, 17))
            # Right arm occupies canonical [32:64]: joints7, EEF9, gripper1,
            # fingertips15. Only its first 17 dimensions may be supervised.
            np.testing.assert_array_equal(packed[:, 32:39], native[:, 10:17])
            np.testing.assert_array_equal(packed[:, 39:48], native[:, :9])
            np.testing.assert_array_equal(packed[:, 48:49], native[:, 9:10])
            np.testing.assert_array_equal(mask[:, :32], 0)
            np.testing.assert_array_equal(mask[:, 32:49], 1)
            np.testing.assert_array_equal(mask[:, 49:], 0)
            restored = gather(packed, placements)
            np.testing.assert_array_equal(
                np.concatenate([restored["arm_eef"], restored["arm_gripper"], restored["arm_joints"]], axis=-1),
                native,
            )

    def test_processor_body_frame_relative_eef_and_absolute_gripper_joints(self):
        layout = load_layout(REPO / "examples/pretrain/canonical/bimanual_v1.yaml")
        maps = load_robot_maps(REPO / "examples/pretrain/canonical/robots", layout)
        modalities = data_config_to_gr00t_modality_configs(ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE])
        processor = StarVLAPackProcessor(
            {ROBOT_TYPE: modalities}, include_state=False, use_percentiles=True,
            clip_outliers=False, use_relative_action=True, canonical_layout=layout, robot_maps=maps,
        )

        def bounds(shape, lo, hi):
            return {"min": np.full(shape, lo * 10), "max": np.full(shape, hi * 10),
                    "q01": np.full(shape, lo), "q99": np.full(shape, hi),
                    "mean": np.zeros(shape), "std": np.ones(shape)}

        processor.set_statistics({ROBOT_TYPE: {
            "action": {"arm_eef": bounds(9, -100, 100), "arm_gripper": bounds(1, 0, 1),
                       "arm_joints": bounds(7, -np.pi, np.pi)},
            "relative_action": {"arm_eef": bounds((8, 9), -2, 2)},
        }})
        # The current tool frame is rotated +90 degrees around base Z.
        # A +0.1m base-X target must therefore become -0.1m tool-Y.
        state_eef = np.array([[0.4, 0.2, 0.6, 0, -1, 0, 1, 0, 0]], dtype=np.float32)
        target_eef = np.repeat(state_eef, 8, axis=0)
        target_eef[:, 0] += 0.1
        gripper = np.linspace(0, 1, 8, dtype=np.float32)[:, None]
        joints = np.repeat(np.linspace(-1, 1, 7, dtype=np.float32)[None, :], 8, axis=0)
        states = {"arm_eef": state_eef, "arm_gripper": np.ones((1, 1), dtype=np.float32),
                  "arm_joints": np.zeros((1, 7), dtype=np.float32)}
        step = VLAStepData(
            images={name: [np.full((4, 6, 3), value, dtype=np.uint8)]
                    for name, value in zip(("front", "exterior_2", "wrist"), (10, 70, 140))},
            states=states, actions={"arm_eef": target_eef, "arm_gripper": gripper, "arm_joints": joints},
            text="pick up the block", embodiment=ROBOT_TYPE,
        )
        sample = processor([{"type": "episode_step", "content": step}])
        self.assertEqual(sample["action"].shape, (8, 90))
        self.assertNotIn("state", sample)
        self.assertEqual([np.asarray(frame)[0, 0, 0] for frame in sample["image"]], [10, 70, 140])
        np.testing.assert_allclose(sample["action"][:, 39:42], np.tile([0, -0.05, 0], (8, 1)), atol=1e-6)
        np.testing.assert_allclose(sample["action"][:, 42:48], np.tile([0.5, 0, 0, 0, 0.5, 0], (8, 1)), atol=1e-6)
        np.testing.assert_allclose(sample["action"][:, 48:49], 2 * gripper - 1, atol=1e-6)
        np.testing.assert_allclose(sample["action"][:, 32:39], joints / np.pi, atol=1e-6)
        np.testing.assert_array_equal(sample["action_mask"].sum(axis=-1), np.full(8, 17))
        restored = processor.decode_action(sample["action"], ROBOT_TYPE, state=states)
        np.testing.assert_allclose(restored["arm_eef"], target_eef, atol=1e-6)
        np.testing.assert_allclose(restored["arm_gripper"], gripper, atol=1e-6)
        np.testing.assert_allclose(restored["arm_joints"], joints, atol=1e-6)

    def test_cache_jobs_resolve_original_camera_keys_across_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "dataset/meta"
            meta.mkdir(parents=True)
            info = {"chunks_size": 1000, "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
            names = ("front", "exterior_2", "wrist")
            modality = {"video": {name: {"original_key": f"observation.images.{name}"} for name in names}}
            (meta / "info.json").write_text(json.dumps(info))
            (meta / "modality.json").write_text(json.dumps(modality))
            (meta / "episodes.jsonl").write_text('\n'.join(json.dumps({"episode_index": episode, "length": 11}) for episode in (999, 1000)))
            with patch.dict(os.environ, {"CACHE_CAMERAS": ','.join(names)}):
                jobs = build_jobs(root, DATASET_NAMED_MIXTURES[DATA_MIX], root / "cache", 256, 32)
            self.assertEqual(len(jobs), 6)
            self.assertEqual(Path(jobs[-1][0]), root / "dataset/videos/chunk-001/observation.images.wrist/episode_001000.mp4")
            self.assertEqual(Path(jobs[-1][1]), root / "cache/dataset/observation.images.wrist/episode_001000.npy")
            self.assertEqual(jobs[-1][2:], (11, 256, 32))
            self.assertEqual(_dataset_frame_cache_dir(root / "cache", "dataset"), root / "cache/dataset")


if __name__ == "__main__":
    unittest.main()
