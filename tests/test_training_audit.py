"""CPU-only checks: python -m unittest discover -s tests -p test_training_audit.py."""

import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from starVLA.training.training_audit import SourceExposureAudit, model_parameter_identity


def _distributed_worker(rank, rendezvous, output_dir):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42 + rank)
        rng_before = torch.get_rng_state().clone()
        audit = SourceExposureAudit()
        batches = (
            (["galaxea", "galaxea"], ["libero"]),
            (["libero"], ["galaxea", "galaxea"]),
            (["new_source"], ["galaxea"]),
        )
        records = []
        for by_rank in batches:
            audit.observe_backward_batch([{"robot_tag": tag} for tag in by_rank[rank]])
            records.append(audit.global_metrics("cpu"))
        assert torch.equal(rng_before, torch.get_rng_state())
        Path(output_dir, f"rank_{rank}.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


class TrainingAuditTest(unittest.TestCase):
    def test_accumulated_microbatches_and_window_reset(self):
        audit = SourceExposureAudit()
        audit.observe_backward_batch([{"robot_tag": "galaxea"}] * 3)
        audit.observe_backward_batch([{"robot_tag": "libero"}])
        metrics = audit.global_metrics("cpu")
        self.assertEqual(metrics["source_exposure/cumulative_examples"], 4)
        self.assertEqual(metrics["source_exposure/galaxea/cumulative_fraction"], 0.75)
        audit.observe_backward_batch([{"robot_tag": "libero"}])
        metrics = audit.global_metrics("cpu")
        self.assertEqual(metrics["source_exposure/cumulative_examples"], 5)
        self.assertEqual(metrics["source_exposure/window_examples"], 1)
        self.assertEqual(metrics["source_exposure/galaxea/window_examples"], 0)

    def test_hash_all_parameters_bfloat16_scalar_and_rng(self):
        model = torch.nn.Module()
        model.action_model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        model.qwen_vl_interface = torch.nn.Linear(2, 2)
        model.register_parameter("scalar", torch.nn.Parameter(torch.tensor(1.0)))
        model.qwen_vl_interface.requires_grad_(False)
        torch_rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        first = model_parameter_identity(model)
        second = model_parameter_identity(model)
        self.assertEqual(first, second)
        self.assertEqual(first["parameters"]["model"]["numel"], 13)
        self.assertEqual(first["parameters"]["action_head"]["numel"], 6)
        self.assertEqual(first["parameters"]["vlm"]["numel"], 6)
        with torch.no_grad():
            model.qwen_vl_interface.weight[0, 0].add_(1)
        changed = model_parameter_identity(model)
        self.assertNotEqual(first["parameters"]["model"], changed["parameters"]["model"])
        self.assertNotEqual(first["parameters"]["vlm"], changed["parameters"]["vlm"])
        self.assertEqual(first["parameters"]["action_head"], changed["parameters"]["action_head"])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())

    def test_two_rank_three_step_global_counts_and_late_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = str(Path(tmp, "rendezvous"))
            mp.spawn(_distributed_worker, args=(rendezvous, tmp), nprocs=2, join=True)
            records = json.loads(Path(tmp, "rank_0.json").read_text())
            self.assertEqual(records, json.loads(Path(tmp, "rank_1.json").read_text()))
        self.assertEqual([row["source_exposure/cumulative_examples"] for row in records], [3, 6, 8])
        self.assertEqual([row["source_exposure/window_examples"] for row in records], [3, 3, 2])
        self.assertEqual(records[-1]["source_exposure/galaxea/cumulative_examples"], 5)
        self.assertEqual(records[-1]["source_exposure/libero/cumulative_examples"], 2)
        self.assertEqual(records[-1]["source_exposure/new_source/cumulative_examples"], 1)
        self.assertEqual(records[-1]["source_exposure/libero/window_examples"], 0)


if __name__ == "__main__":
    unittest.main()
