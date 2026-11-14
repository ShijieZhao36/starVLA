"""Small CPU tests: real Parquet, multiprocessing, exact upstream math/contracts.

Run in the training environment with numpy, scipy and pyarrow installed:
    python -m unittest discover -s examples/pretrain/DROID/tests -v

The reference helpers are compiled directly from their unchanged source to
avoid importing the whole model/video stack for these CPU-only checks.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = Path(os.environ.get("DROID_REFERENCE_ROOT", ROOT))
MODULE_PATH = ROOT / "examples/pretrain/DROID/train_files/precompute_droid_stats.py"
spec = importlib.util.spec_from_file_location("droid_stats_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def reference_functions(path, names, namespace):
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing reference helpers in {path}")
    source = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)


def runtime_from_reference():
    namespace = dict(np=np, json=json, hashlib=hashlib, os=os, Path=Path, tempfile=tempfile,
                     logger=logging.getLogger("stats-test"), STATS_FINGERPRINTS_KEY="__fingerprints__")
    reference_functions(REFERENCE / "gr00t/data/stats.py", [
        "_stale_features", "_compute_stats_fingerprint", "_load_stats_cache", "_dump_stats_cache_atomic",
    ], namespace)
    eef = SimpleNamespace(name="EEF")
    format_6d = SimpleNamespace(name="XYZ_ROT6D")
    namespace.update(ActionType=SimpleNamespace(EEF=eef, NON_EEF=object()),
                     ActionFormat=SimpleNamespace(XYZ_ROT6D=format_6d))
    reference_functions(REFERENCE / "starVLA/dataloader/gr00t_sharded.py", [
        "_relative_fingerprint", "_rot6d_to_matrix", "_scipy_rotation_matrix",
        "_relative_xyz_rot6d", "_relative_windows",
    ], namespace)
    config = SimpleNamespace(type=eef, format=format_6d, rep=SimpleNamespace(name="RELATIVE"), state_key="arm_eef")
    action = SimpleNamespace(delta_indices=list(range(8)))
    state = SimpleNamespace(delta_indices=[0])
    return dict(
        stale=namespace["_stale_features"], feature_fingerprint=namespace["_compute_stats_fingerprint"],
        load=namespace["_load_stats_cache"], dump=namespace["_dump_stats_cache_atomic"],
        relative_windows=namespace["_relative_windows"], relative_config=config,
        relative_fingerprint=namespace["_relative_fingerprint"](action, state, "arm_eef", config),
        action_deltas=np.arange(8), state_offset=0,
    )


def make_dataset(root):
    meta = root / "meta"
    meta.mkdir()
    lengths = [5, 8, 11, 14, 29]  # 67 frames; unequal episodes and one too short for H8
    features = {
        "observation.state": dict(dtype="float32", shape=[17]),
        "action": dict(dtype="float32", shape=[17]),
        "source.raw_scalar": dict(dtype="float64", shape=[1]),
        "source.raw_vector": dict(dtype="float32", shape=[2]),
        "source.text": dict(dtype="string", shape=[1]),
    }
    info = dict(features=features, total_episodes=len(lengths), total_frames=sum(lengths),
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", chunks_size=2)
    (meta / "info.json").write_text(json.dumps(info))
    groups = dict(arm_eef=dict(start=0, end=9, rotation_type="rotation_6d"),
                  arm_gripper=dict(start=9, end=10), arm_joints=dict(start=10, end=17))
    (meta / "modality.json").write_text(json.dumps(dict(state=groups, action=groups)))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(dict(episode_index=i, length=n)) + "\n" for i, n in enumerate(lengths)))
    episodes = []
    rng = np.random.default_rng(583)
    for index, length in enumerate(lengths):
        state = rng.normal(size=(length, 17)).astype(np.float32)
        action = rng.normal(size=(length, 17)).astype(np.float32)
        state[:, 3:9] = Rotation.from_rotvec(rng.normal(size=(length, 3))).as_matrix()[:, :2].reshape(length, 6)
        action[:, 3:9] = Rotation.from_rotvec(rng.normal(size=(length, 3))).as_matrix()[:, :2].reshape(length, 6)
        # Large separation makes an average of episode quantiles visibly wrong.
        action[:, 0] += index * 10
        numeric = {"observation.state": state, "action": action,
                   "source.raw_scalar": (np.arange(length) + index * 0.01).reshape(length, 1),
                   "source.raw_vector": rng.normal(size=(length, 2)).astype(np.float32)}
        columns = {key: pa.array(value.tolist()) for key, value in numeric.items() if key != "source.raw_scalar"}
        columns["source.raw_scalar"] = pa.array(numeric["source.raw_scalar"][:, 0])
        columns["source.text"] = pa.array(["not needed for numeric statistics"] * length)
        path = root / info["data_path"].format(episode_chunk=index // 2, episode_index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), path)
        episodes.append(numeric)
    return episodes


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episodes = make_dataset(self.root)
        self.runtime = runtime_from_reference()
        self.args = module.parse_args(["--dataset-dir", str(self.root), "--workers", "2",
                                      "--summary-workers", "2", "--episodes-per-task", "2"])

    def tearDown(self):
        self.temporary.cleanup()

    def execute(self):
        with redirect_stdout(io.StringIO()):
            module.run(self.args, runtime=self.runtime)

    def assert_summary(self, actual, expected):
        expected = np.asarray(expected, dtype=np.float32)
        refs = dict(mean=np.mean(expected, axis=0), std=np.std(expected, axis=0),
                    min=np.min(expected, axis=0), max=np.max(expected, axis=0),
                    q01=np.quantile(expected, .01, axis=0), q99=np.quantile(expected, .99, axis=0))
        for key, value in refs.items():
            np.testing.assert_allclose(actual[key], value, rtol=2e-6, atol=2e-6, err_msg=key)
        np.testing.assert_array_equal(actual["q01"], refs["q01"])
        np.testing.assert_array_equal(actual["q99"], refs["q99"])

    def test_parallel_exact_stats_relative_windows_and_fingerprints(self):
        self.execute()
        actual = json.loads((self.root / "meta/stats.json").read_text())
        for key in self.episodes[0]:
            self.assert_summary(actual[key], np.concatenate([ep[key] for ep in self.episodes]))
        self.assertNotIn("source.text", actual)
        relative = json.loads((self.root / "meta/relative_stats.json").read_text())
        blocks = [self.runtime["relative_windows"](
            ep["action"][:, :9], ep["observation.state"][:, :9], np.arange(8), 0,
            self.runtime["relative_config"],
        ) for ep in self.episodes]
        blocks = [block for block in blocks if block is not None]
        self.assertEqual(sum(len(block) for block in blocks), 34)
        self.assert_summary(relative["arm_eef"], np.concatenate(blocks))
        self.assertEqual(np.shape(relative["arm_eef"]["q01"]), (8, 9))
        info = json.loads((self.root / "meta/info.json").read_text())
        numeric = {key: value for key, value in info["features"].items() if "float" in value["dtype"]}
        self.assertFalse(self.runtime["stale"](actual, numeric, list(numeric)))
        self.assertEqual(relative["__fingerprints__"]["arm_eef"], self.runtime["relative_fingerprint"])
        self.assertFalse(list((self.root / "meta").glob(".droid_stats_*")))

    def test_second_run_skips_all_parquet_reads(self):
        self.execute()
        original = (self.root / "meta/stats.json").read_bytes()
        for path in self.root.glob("data/*/*.parquet"):
            path.unlink()
        self.execute()  # valid fingerprints must make this succeed without Parquet
        self.assertEqual(original, (self.root / "meta/stats.json").read_bytes())

    def test_check_only_is_strictly_read_only_and_detects_wrong_shapes(self):
        args = module.parse_args(["--dataset-dir", str(self.root), "--check-only"])
        before = {str(path) for path in self.root.rglob("*")}
        with self.assertRaisesRegex(RuntimeError, "cached_not_ready"):
            module.run(args, runtime=self.runtime)
        self.assertEqual(before, {str(path) for path in self.root.rglob("*")})
        self.execute()
        output = io.StringIO()
        stats_file = self.root / "meta/stats.json"
        rel_file = self.root / "meta/relative_stats.json"
        timestamps = (stats_file.stat().st_mtime_ns, rel_file.stat().st_mtime_ns)
        with redirect_stdout(output):
            module.run(args, runtime=self.runtime)
        self.assertEqual(output.getvalue().strip(), "cached_ready")
        self.assertEqual(timestamps, (stats_file.stat().st_mtime_ns, rel_file.stat().st_mtime_ns))
        relative = json.loads(rel_file.read_text())
        relative["arm_eef"]["q01"] = [0.0] * 9  # stale shape with a valid fingerprint
        rel_file.write_text(json.dumps(relative))
        with self.assertRaisesRegex(RuntimeError, "cached_not_ready"):
            module.run(args, runtime=self.runtime)

    def test_failure_does_not_publish_or_leave_temporary_maps(self):
        path = next(self.root.glob("data/*/*.parquet"))
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute()
        self.assertFalse((self.root / "meta/stats.json").exists())
        self.assertFalse((self.root / "meta/relative_stats.json").exists())
        self.assertFalse(list((self.root / "meta").glob(".droid_stats_*")))

    def test_lock_excludes_concurrent_writer(self):
        with module.exclusive_lock(self.root / "meta/.droid_stats.lock"):
            with self.assertRaisesRegex(RuntimeError, "Another DROID stats process"):
                self.execute()

    def test_dry_run_never_writes_and_needs_no_worker_count(self):
        before = {str(path) for path in self.root.rglob("*")}
        args = module.parse_args(["--dataset-dir", str(self.root), "--dry-run"])
        output = io.StringIO()
        with redirect_stdout(output):
            module.run(args)
        self.assertEqual(before, {str(path) for path in self.root.rglob("*")})
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["frames"], 67)
        self.assertEqual(plan["relative_windows"], 34)


if __name__ == "__main__":
    unittest.main()
