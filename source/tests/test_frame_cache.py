import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gr00t.data.dataset.frame_cache import (
    cache_is_complete,
    frame_cache_path,
    read_cached_frames,
)
from gr00t.data.dataset.sharded_single_step_dataset import video_indices_for_steps
from starVLA.dataloader.gr00t_sharded import _dataset_frame_cache_dir
from starVLA.dataloader.precompute_galaxea_frame_cache import (
    _as_uint8_hwc,
    build_jobs,
    cache_one,
)


class FrameCacheTest(unittest.TestCase):
    def test_path_matches_episode_mp4_name(self):
        path = frame_cache_path(
            "/mnt/zhaoshijie/frame_cache/galaxea_stage1_256/Organize_The_Bread_Shelf_20250630_003",
            "observation.images.front",
            0,
        )
        self.assertEqual(
            path,
            Path(
                "/mnt/zhaoshijie/frame_cache/galaxea_stage1_256/"
                "Organize_The_Bread_Shelf_20250630_003/"
                "observation.images.front/episode_000000.npy"
            ),
        )

    def test_read_copies_requested_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = frame_cache_path(tmp, "observation.images.front", 3)
            path.parent.mkdir(parents=True)
            frames = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
            np.save(path, frames)
            got = read_cached_frames(path, np.array([4, 1, 4]))
            self.assertEqual(got.shape, (3, 2, 2, 3))
            np.testing.assert_array_equal(got[0], frames[4])
            np.testing.assert_array_equal(got[1], frames[1])
            np.testing.assert_array_equal(got[2], frames[4])

    def test_missing_cache_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_cached_frames("/tmp/does_not_exist_frame_cache.npy", np.array([0]))

    def test_complete_requires_length_and_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_000000.npy"
            np.save(path, np.zeros((4, 8, 8, 3), dtype=np.uint8))
            self.assertTrue(cache_is_complete(path, 4, 8))
            self.assertFalse(cache_is_complete(path, 5, 8))
            self.assertFalse(cache_is_complete(path, 4, 16))
            np.save(path, np.zeros((4, 8, 8, 3), dtype=np.float32))
            self.assertFalse(cache_is_complete(path, 4, 8))

    def test_index_outside_cache_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_000000.npy"
            np.save(path, np.zeros((3, 2, 2, 3), dtype=np.uint8))
            with self.assertRaises(IndexError):
                read_cached_frames(path, np.array([0, 3]))

    def test_video_indices_follow_deltas(self):
        got = video_indices_for_steps([1, 4], [0], episode_length=10)
        np.testing.assert_array_equal(got, np.array([1, 4]))
        got = video_indices_for_steps([8], [0, 1], episode_length=10)
        np.testing.assert_array_equal(got, np.array([8, 9]))
        got = video_indices_for_steps([9], [0, 1], episode_length=10, allow_padding=True)
        np.testing.assert_array_equal(got, np.array([9]))

    def test_dataset_cache_dir_uses_subset_id(self):
        path = _dataset_frame_cache_dir(
            "/mnt/zhaoshijie/frame_cache/galaxea_stage1_256",
            "Organize_The_Bread_Shelf_20250630_003/stage1/dataset",
        )
        self.assertEqual(
            path,
            Path("/mnt/zhaoshijie/frame_cache/galaxea_stage1_256/Organize_The_Bread_Shelf_20250630_003"),
        )
        self.assertIsNone(_dataset_frame_cache_dir(None, "subset/stage1/dataset"))
        self.assertIsNone(_dataset_frame_cache_dir("", "subset/stage1/dataset"))

    def test_hwc_uint8_conversion(self):
        chw = np.zeros((3, 4, 5), dtype=np.float32)
        chw[0] = 1.0
        got = _as_uint8_hwc(chw)
        self.assertEqual(got.shape, (4, 5, 3))
        self.assertEqual(got.dtype, np.uint8)
        self.assertEqual(int(got[0, 0, 0]), 255)

    def test_cache_one_skips_finished_file_without_reading_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = frame_cache_path(tmp, "observation.images.front", 0)
            path.parent.mkdir(parents=True)
            np.save(path, np.zeros((4, 8, 8, 3), dtype=np.uint8))
            status = cache_one(("/no/such/video.mp4", str(path), 4, 8, 32))
            self.assertTrue(status.startswith("skip "))

    def test_build_jobs_uses_three_cameras_and_episode_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root / "SubsetA" / "stage1" / "dataset" / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text(
                json.dumps(
                    {
                        "chunks_size": 1000,
                        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                    }
                )
            )
            (meta / "modality.json").write_text(
                json.dumps(
                    {
                        "video": {
                            "front": {"original_key": "observation.images.front"},
                            "left_wrist": {"original_key": "observation.images.left_wrist"},
                            "right_wrist": {"original_key": "observation.images.right_wrist"},
                            "head_right": {"original_key": "observation.images.head_right"},
                        }
                    }
                )
            )
            (meta / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 7, "length": 12}) + "\n"
            )
            jobs = build_jobs(
                root,
                [("SubsetA/stage1/dataset", 1.0, "galaxea_r1lite_stage1")],
                Path("/cache"),
                256,
                32,
            )
            self.assertEqual(len(jobs), 3)
            video_paths = {job[0] for job in jobs}
            self.assertIn(
                str(root / "SubsetA/stage1/dataset/videos/chunk-000/observation.images.front/episode_000007.mp4"),
                video_paths,
            )
            self.assertFalse(any("head_right" in job[0] for job in jobs))
            out_paths = {job[1] for job in jobs}
            self.assertIn(
                "/cache/SubsetA/observation.images.left_wrist/episode_000007.npy",
                out_paths,
            )
            self.assertTrue(all(job[2] == 12 and job[3] == 256 and job[4] == 32 for job in jobs))


if __name__ == "__main__":
    unittest.main()
