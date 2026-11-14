"""CPU check of rank-owned streams and the actual patched prepare branch.

Run from the starVLA checkout with the CRA Python after applying the patch.
The test reads/parses the trainer instead of importing its distributed entrypoint.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from accelerate.data_loader import IterableDatasetShard
from torch.utils.data import DataLoader, IterableDataset


class OwnedStream(IterableDataset):
    def __init__(self, rank, world_size=16):
        self.rank = rank
        self.world_size = world_size
        self.reads = 0

    def __iter__(self):
        i = 0
        while True:
            self.reads += 1
            yield {"rank": self.rank, "sample_id": self.rank + self.world_size * i}
            i += 1


batch_size = 8
world_size = 16
rank_batches = []
for rank in range(world_size):
    original = OwnedStream(rank)
    double_shard = IterableDatasetShard(
        original, batch_size=batch_size, num_processes=world_size,
        process_index=rank, split_batches=False,
    )
    old_batch = next(iter(DataLoader(double_shard, batch_size=batch_size)))
    assert original.reads == batch_size * world_size
    single = OwnedStream(rank)
    new_batch = next(iter(DataLoader(single, batch_size=batch_size)))
    assert single.reads == batch_size
    assert new_batch["rank"].tolist() == [rank] * batch_size
    rank_batches.extend(new_batch["sample_id"].tolist())
assert len(rank_batches) == len(set(rank_batches)) == batch_size * world_size

source = Path("starVLA/training/train_starvla.py").read_text()
tree = ast.parse(source)
trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLATrainer")
prepare = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_training")
branch = next(
    n for n in prepare.body
    if isinstance(n, ast.If) and "gr00t_sharded" in ast.unparse(n.test)
)
branch_code = compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), "prepare_branch", "exec")


def verify_branch(dataset_name):
    ds_config = {"train_micro_batch_size_per_gpu": "auto", "train_batch_size": "auto"}
    state = SimpleNamespace(deepspeed_plugin=SimpleNamespace(deepspeed_config=ds_config))
    accel = SimpleNamespace(state=state)
    model, optimizer, loader = object(), object(), object()
    calls = []

    def prepare_components(received_accelerator, *components):
        assert received_accelerator is accel
        calls.append(components)
        if dataset_name == "gr00t_sharded":
            assert ds_config["train_micro_batch_size_per_gpu"] == batch_size
        return components

    self = SimpleNamespace(
        config=SimpleNamespace(datasets=SimpleNamespace(vla_data=SimpleNamespace(
            dataset_py=dataset_name, per_device_batch_size=batch_size))),
        accelerator=accel, model=model, optimizer=optimizer,
        vla_train_dataloader=loader, setup_distributed_training=prepare_components,
    )
    exec(branch_code, {"self": self})
    assert self.vla_train_dataloader is loader
    assert len(calls) == 1
    if dataset_name == "gr00t_sharded":
        assert calls[0] == (model, optimizer)
        assert ds_config["train_micro_batch_size_per_gpu"] == batch_size
    else:
        assert calls[0] == (model, optimizer, loader)
        assert ds_config["train_micro_batch_size_per_gpu"] == "auto"


verify_branch("gr00t_sharded")
verify_branch("lerobot_datasets")
print(json.dumps({
    "status": "passed", "world_size": world_size, "batch_size": batch_size,
    "old_reads_per_rank_batch": batch_size * world_size,
    "new_reads_per_rank_batch": batch_size,
    "unique_global_sample_ids": len(set(rank_batches)),
    "gr00t_loader_preserved": True, "other_dataset_path_unchanged": True,
}, indent=2))
