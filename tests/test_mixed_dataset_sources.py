import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from starVLA.dataloader.gr00t_lerobot.mixtures import resolve_dataset_mixture
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_sharded import (
    build_sampling_manifest,
    resolve_dataset_paths,
    validate_dataset_paths,
)


class MixedDatasetSourcesTest(unittest.TestCase):
    def test_real_mix_retains_all_sources_and_frame_weights(self):
        mixed = resolve_dataset_mixture("galaxea_libero_plus_stage1_80_20", DATASET_NAMED_MIXTURES)
        galaxea = [row for row in mixed if row[2] == "galaxea_r1lite_stage1"]
        libero = [row for row in mixed if row[2] == "libero_plus_rot6d"]
        self.assertEqual(len(galaxea), 218)
        self.assertEqual([row[0] for row in galaxea], [row[0] for row in DATASET_NAMED_MIXTURES["galaxea_open_world_stage1"]])
        self.assertAlmostEqual(sum(row[1] for row in galaxea), 0.8)
        self.assertEqual(libero, [("dataset_absolute", 0.2, "libero_plus_rot6d")])
        self.assertTrue(all(abs(row[1] - 0.8 / 218) < 1e-15 for row in galaxea))

    def test_legacy_roots_and_mixture_are_unchanged(self):
        cfg = {"data_root_dir": "/data", "frame_cache_dir": "/cache"}
        self.assertEqual(resolve_dataset_paths(cfg, "subset/stage1/dataset", "g"),
                         (Path("/data/subset/stage1/dataset"), Path("/cache/subset")))
        self.assertEqual(resolve_dataset_mixture("libero_plus_stage1", DATASET_NAMED_MIXTURES),
                         DATASET_NAMED_MIXTURES["libero_plus_stage1"])

    def test_per_robot_paths_and_explicit_cache_disable(self):
        cfg = {"data_root_dir": "/g", "frame_cache_dir": "/gc",
               "data_root_dirs_by_robot": {"l": "/l"},
               "frame_cache_dirs_by_robot": {"l": "/lc", "disabled": None}}
        self.assertEqual(resolve_dataset_paths(cfg, "dataset_absolute", "l"),
                         (Path("/l/dataset_absolute"), Path("/lc/dataset_absolute")))
        self.assertEqual(resolve_dataset_paths(cfg, "subset/stage1/dataset", "g"),
                         (Path("/g/subset/stage1/dataset"), Path("/gc/subset")))
        self.assertIsNone(resolve_dataset_paths(cfg, "dataset", "disabled")[1])

    def test_missing_path_errors_name_robot_and_path(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = Path(root) / "dataset"
            cache = Path(root) / "cache"
            with self.assertRaisesRegex(FileNotFoundError, "robot_type='l'.*info.json"):
                validate_dataset_paths(dataset, cache, "l")
            (dataset / "meta").mkdir(parents=True)
            (dataset / "meta/info.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "robot_type='l'.*cache"):
                validate_dataset_paths(dataset, cache, "l")
            cache.mkdir()
            validate_dataset_paths(dataset, cache, "l")

    def test_manifest_distinguishes_sample_weights_from_shard_probabilities(self):
        datasets = [SimpleNamespace(shard_lengths=np.array([100, 100])),
                    SimpleNamespace(shard_lengths=np.array([50, 50]))]
        mixture = SimpleNamespace(weights=[0.8, 0.2], seed=42, epoch=0,
                                  shard_sampling_schedule=[(0, 0), (1, 1), (0, 1)])
        entries = [("g", 0.8, "g", Path("/g"), None),
                   ("l", 0.2, "l", Path("/l"), Path("/lc"))]
        manifest = build_sampling_manifest("mixed", entries, datasets, mixture)
        self.assertAlmostEqual(manifest["datasets"][0]["shard_sampling_probability"], 2 / 3)
        self.assertAlmostEqual(manifest["by_robot"]["g"]["expected_frame_fraction"], 0.8)
        self.assertEqual(manifest["by_robot"]["g"]["planned_frames"], 200)
        self.assertEqual(manifest["by_robot"]["l"]["planned_frames"], 50)


if __name__ == "__main__":
    unittest.main()
