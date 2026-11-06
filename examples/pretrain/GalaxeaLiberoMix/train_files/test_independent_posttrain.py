"""CPU-only independent-start/source-isolation and common-contract tests."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("posttrain_validator", HERE / "validate_libero_pair_60k.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ForbiddenPeer:
    """A peer value that fails if even converted to a filesystem path."""
    def __fspath__(self):
        raise AssertionError("Independent start accessed a peer path")


class IndependentPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        code_paths = (
            "starVLA/training/train_starvla.py", "starVLA/training/training_audit.py",
            "starVLA/training/trainer_utils/trainer_tools.py", "starVLA/model/framework/VLM4A/QwenPI.py",
            "starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py", "starVLA/model/modules/vlm/QWen3.py",
            "starVLA/dataloader/gr00t_sharded.py", "starVLA/dataloader/gr00t_sharded_processor.py",
            "starVLA/dataloader/canonical_layout.py", "gr00t/data/dataset/sharded_mixture_dataset.py",
            "examples/pretrain/canonical/bimanual_v1.yaml", "examples/pretrain/canonical/robots/libero_plus.yaml",
            "examples/simBenchmarks/LIBERO-plus/train_files/data_registry/data_config.py",
            "examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py",
        )
        for name in code_paths:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("strict_pretrained_load\n")
        self.data = self.root / "data"
        for name in validator.TRAIN_META:
            path = self.data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"splits": {"train": "0:14347"}}))
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.overlay = self.root / "overlay"
        self.overlay.mkdir()
        (self.root / "sim_overlay_manifest.json").write_text("{}")
        self.config = self.root / "config.yaml"
        self.config.write_text((HERE / "starvla_libero_pair_ln_60k.yaml").read_text())
        self.version_patch = patch.object(validator.importlib.metadata, "version", return_value="same-version")
        self.commit_patch = patch.object(validator.subprocess, "check_output", return_value="a" * 40 + "\n")
        self.version_patch.start()
        self.commit_patch.start()
        self.addCleanup(self.version_patch.stop)
        self.addCleanup(self.commit_patch.stop)

    def source(self, arm):
        run = self.root / f"pretrain_{arm}"
        launcher = self.root / f"launcher_{arm}"
        launcher.mkdir()
        ckpt = run / "final_model/pytorch_model.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"final model")
        cfg = yaml.safe_load(self.config.read_text())
        cfg["trainer"]["max_train_steps"] = 80000
        cfg["datasets"]["vla_data"]["data_mix"] = validator.SOURCE_MIX[arm]
        (run / "config.full.yaml").write_text(yaml.safe_dump(cfg))
        tags = {validator.GALAXEA: {}}
        if arm == "mixed":
            tags[validator.LIBERO] = {}
        (run / "dataset_statistics.json").write_text(json.dumps(tags))
        (run / "pretrain_success.json").write_text(json.dumps({
            "arm": arm, "steps": 80000, "checkpoint": str(ckpt), "bytes": ckpt.stat().st_size}))
        (run / "exit_code").write_text("0\n")
        (launcher / "exit_code").write_text("0\n")
        return run, launcher

    def args(self, arm="control", independent=True):
        run, launcher = self.source(arm)
        out = self.root / f"posttrain_{arm}"
        (out / "launch").mkdir(parents=True)
        return SimpleNamespace(
            arm=arm, source=str(run), source_launcher=str(launcher),
            peer=ForbiddenPeer(), peer_launcher=ForbiddenPeer(), independent_start=independent,
            output=str(out), repo=str(self.repo), config=str(self.config),
            runner=str(HERE / "run_libero_pair_60k_eval.sh"),
            accelerate=str(self.config), deepspeed=str(self.config), overlay=str(self.overlay),
            data=str(self.data), cache=str(self.cache), contract=str(self.root / "contract"),
            dry_run=False, check_only=False, smoke=False, report=str(self.root / f"report_{arm}.json"))

    def test_independent_never_touches_peer_and_pins_only_own_source(self):
        args = self.args()
        validator.preflight(args)
        contract = Path(args.contract)
        common = json.loads((contract / "protocol_manifest.json").read_text())
        self.assertNotIn("sources", common)
        self.assertTrue((contract / "source_provenance_control.json").is_file())
        self.assertFalse((contract / "source_provenance_mixed.json").exists())
        report = json.loads(Path(args.report).read_text())
        self.assertTrue(report["ready"])
        self.assertEqual(report["source_gate_scope"], "self_only")
        self.assertIsNone(report["paired_pretrain_run_dir"])

    def test_later_other_arm_matches_common_contract_with_different_source(self):
        control = self.args()
        mixed = self.args("mixed")
        validator.preflight(control)
        validator.preflight(mixed)
        left = json.loads(Path(control.report).read_text())
        right = json.loads(Path(mixed.report).read_text())
        self.assertEqual(left["common_protocol_sha256"], right["common_protocol_sha256"])
        self.assertNotEqual(left["pretrain_run_dir"], right["pretrain_run_dir"])
        self.assertTrue((Path(mixed.contract) / "source_provenance_mixed.json").is_file())

    def test_later_arm_cannot_change_shared_training_data(self):
        control = self.args()
        mixed = self.args("mixed")
        validator.preflight(control)
        (self.data / "meta/stats.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "Paired protocol mismatch"):
            validator.preflight(mixed)

    def test_same_arm_cannot_substitute_source_after_contract_pin(self):
        args = self.args()
        validator.preflight(args)
        marker = Path(args.source) / "pretrain_success.json"
        content = json.loads(marker.read_text())
        content["changed"] = "provenance is immutable"
        marker.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "source_provenance_control"):
            validator.preflight(args)

    def test_independent_still_requires_own_final_80k_success(self):
        args = self.args()
        for path, text in (
            (Path(args.source) / "exit_code", "1"),
            (Path(args.source_launcher) / "exit_code", "1"),
            (Path(args.source) / "pretrain_success.json", None),
        ):
            original = path.read_text()
            if text is None:
                marker = json.loads(original)
                marker["steps"] = 10000
                text = json.dumps(marker)
            path.write_text(text)
            with self.subTest(path=path), self.assertRaises(ValueError):
                validator.preflight(args)
            path.write_text(original)
        self.assertFalse((Path(args.contract) / "protocol_manifest.json").exists())

    def test_dry_run_reports_pending_own_source_without_touching_peer(self):
        args = self.args()
        args.dry_run = True
        (Path(args.source_launcher) / "exit_code").unlink()
        validator.preflight(args)
        report = json.loads(Path(args.report).read_text())
        self.assertFalse(report["ready"])
        self.assertEqual(len(report["source_gate_errors"]), 1)
        self.assertTrue(report["source_gate_errors"][0].startswith("control:"))
        self.assertFalse(Path(args.contract).exists())

    def test_default_paired_mode_still_requires_peer_success(self):
        args = self.args(independent=False)
        args.peer = None
        args.peer_launcher = None
        with self.assertRaisesRegex(ValueError, "requires --peer"):
            validator.preflight(args)
        args.peer, args.peer_launcher = map(str, self.source("mixed"))
        (Path(args.peer_launcher) / "exit_code").write_text("1")
        with self.assertRaisesRegex(ValueError, "outer launcher"):
            validator.preflight(args)

    def test_check_only_neither_creates_contract_nor_output_artifacts(self):
        args = self.args()
        args.check_only = True
        validator.preflight(args)
        self.assertFalse(Path(args.contract).exists())
        self.assertFalse((Path(args.output) / "launch/source_provenance.json").exists())


class RunnerIsolationTests(unittest.TestCase):
    def test_independent_shell_does_not_require_validate_or_resolve_peer(self):
        # Stop at the first local prerequisite after resolving the selected source.
        # A poisoned peer path would trigger the former absolute-path check or the
        # readlink spy; neither is allowed before the mixed checking window.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakebin = root / "bin"
            fakebin.mkdir()
            trace = root / "readlink_calls"
            spy = fakebin / "readlink"
            spy.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$READLINK_TRACE"\nexec /usr/bin/readlink "$@"\n')
            spy.chmod(0o755)
            env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}", "READLINK_TRACE": str(trace),
                   "ARM": "control", "PRETRAIN_RUN_DIR": str(root / "own"),
                   "REQUIRE_PAIRED_SOURCE": "0", "PAIRED_PRETRAIN_RUN_DIR": "FORBIDDEN_PEER",
                   "PAIRED_PRETRAIN_LAUNCHER_DIR": "FORBIDDEN_PEER_LAUNCHER",
                   "REPO_DIR": str(root / "missing_repo"), "PYTHON_BIN": "/usr/bin/python3",
                   "EVAL_PYTHON": "/usr/bin/python3"}
            result = subprocess.run(["bash", str(HERE / "run_libero_pair_60k_eval.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing file:", result.stderr)
            self.assertNotIn("FORBIDDEN_PEER", trace.read_text())
            self.assertNotIn("peer", result.stderr.lower())
            env.pop("PAIRED_PRETRAIN_RUN_DIR")
            env.pop("PAIRED_PRETRAIN_LAUNCHER_DIR")
            result = subprocess.run(["bash", str(HERE / "run_libero_pair_60k_eval.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertIn("Missing file:", result.stderr)
            self.assertNotIn("PAIRED_PRETRAIN", result.stderr)


if __name__ == "__main__":
    unittest.main()
