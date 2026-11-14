"""CPU-only readiness tests using temporary metadata and dummy checkpoint bytes."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "train_files" / "wait_for_xperience_final.py"
SPEC = importlib.util.spec_from_file_location("wait_for_xperience_final", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.control = self.root / "control"
        self.run.mkdir()
        self.control.mkdir()

    def completed_fixture(self, norm=True):
        for rank in (0, 1):
            (self.control / f"node{rank}.exitcode").write_text("0\n")
        (self.run / "final_model").mkdir()
        (self.run / "final_model/pytorch_model.pt").write_bytes(b"not-a-real-checkpoint")
        # JSON is a YAML subset, so the fixture needs no serialization dependency.
        (self.run / "config.full.yaml").write_text(json.dumps({
            "trainer": {"max_train_steps": 30000},
            "framework": {"name": "QwenPI", "action_model": {
                "action_dim": 90, "state_dim": 90, "action_horizon": 8,
                "decoder_input_norm": norm,
            }},
        }))
        (self.run / "summary.jsonl").write_text('{"steps": 25000}\n{"steps": 30000}\n')

    def check(self):
        return MODULE.check_ready(self.run, self.control, 30000)

    def test_unfinished(self):
        self.assertFalse(self.check().ready)

    def test_partial_and_empty_sentinel(self):
        (self.control / "node0.exitcode").write_text("0\n")
        self.assertFalse(self.check().ready)
        (self.control / "node1.exitcode").write_text("")
        self.assertFalse(self.check().ready)

    def test_failure_detected_without_other_node_sentinel(self):
        (self.control / "node1.exitcode").write_text("17\n")
        with self.assertRaisesRegex(RuntimeError, "exit code 17"):
            self.check()

    def test_success_reads_metadata_only(self):
        self.completed_fixture()
        state = self.check()
        self.assertTrue(state.ready)
        self.assertEqual(state.checkpoint, self.run / "final_model/pytorch_model.pt")

    def test_missing_checkpoint_after_success_is_error(self):
        self.completed_fixture()
        (self.run / "final_model/pytorch_model.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "checkpoint is missing or empty"):
            self.check()

    def test_empty_checkpoint_after_success_is_error(self):
        self.completed_fixture()
        (self.run / "final_model/pytorch_model.pt").write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "checkpoint is missing or empty"):
            self.check()

    def test_layer_norm_false_is_error(self):
        self.completed_fixture(norm=False)
        with self.assertRaisesRegex(RuntimeError, "decoder_input_norm must be true"):
            self.check()

    def test_missing_final_summary_is_error(self):
        self.completed_fixture()
        (self.run / "summary.jsonl").write_text('{"steps": 25000}\n')
        with self.assertRaisesRegex(RuntimeError, "lacks steps=30000"):
            self.check()

    def test_check_only_returns_three_and_creates_nothing(self):
        result = subprocess.run([
            sys.executable, str(HELPER), "--run-dir", str(self.run),
            "--control-dir", str(self.control), "--check-only",
        ], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(list(self.run.iterdir()), [])
        self.assertEqual(list(self.control.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
