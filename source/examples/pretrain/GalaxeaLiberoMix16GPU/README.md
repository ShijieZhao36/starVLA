# Two-node Galaxea/LIBERO pretraining

This is the versioned launcher for the completed neo-bj5 + neo-bj6 experiment.
It uses Galaxea/LIBERO 80:20, fresh Qwen3-VL-4B/QwenPI initialization, LayerwiseFM
with input LayerNorm, canonical 90D actions, horizon 8, and seed 42.

- Pretraining: 2 nodes × 8 GPUs × batch 4 = global 64, 80,000 updates.
- Posttraining: rank-0 node only, 8 GPUs × batch 8 = global 64, pure LIBERO,
  60,000 updates; strict full-model transfer with a fresh optimizer.
- Evaluation: the final posttraining checkpoint, 10,030 cases, environment seed 7,
  8 GPUs × 4 simulations, BF16/canonical forward/positive-open gripper.
- An explicit full smoke mode runs only pretraining 3 updates → posttraining
  3 updates → one simulation case. It uses `_smoke3` run/output suffixes and a
  separate control subtree. Formal mode requires this smoke to have succeeded.

`historical_spec.json` records the original immutable repo paths, commits and run
IDs. It is provenance, **not a launch template**: those runs already exist. The
original artifacts/checkpoints remain unchanged. New runs pin the current clean
integration commit and use one checkout for every stage. Config/DeepSpeed,
initial-weight, reference-statistics and reference-sampling hashes remain those
of the original single-node control. Fresh-training code additionally pins the
current commit instead of requiring the obsolete pretraining-only commit.

## Run a new experiment

Use a shared checkout and shared checkpoint/control filesystem visible at the
same absolute paths on both hosts. The defaults describe this cluster's data,
MRA runtime, immutable baseline, simulation overlay and LIBERO installation;
pass the corresponding arguments when the paths differ. Hosts must each have
8 idle healthy GPUs and passwordless SSH. `MASTER_ADDR` must be reachable from
both nodes. Network variables in `run_mixed_16gpu_pretrain.sh` retain this
cluster's `eth0`/`com0_*` fabric settings. The runner explicitly defaults
`NCCL_IB_GID_INDEX=5` (the cluster's IPv4 RoCEv2 GID) and
`NCCL_CROSS_NIC=0`, even when the login shell has no NCCL settings. The pipeline
continues to discard inherited NCCL variables. The actual transport environment
is recorded in each node's preflight protocol and launch log; cross-node
comparison rejects a mismatch. Adapt these cluster defaults before using a
different fabric.

Generate the spec **after committing all code**, outside the checkout:

```bash
python examples/pretrain/GalaxeaLiberoMix16GPU/make_spec.py \
  --repo /absolute/path/to/checkout \
  --hosts neo-bj5 neo-bj6 --master-addr RANK0_IP \
  --run-id galaxea_libero_mixed_16gpu_FRESH_TIMESTAMP \
  --output /absolute/path/outside/checkout/spec.json
```

The controller never writes reports into the Git checkout. Commands below use
`SPEC=/absolute/path/outside/checkout/spec.json`; set it in your shell first.

```bash
python examples/pretrain/GalaxeaLiberoMix16GPU/deploy.py prepare --spec "$SPEC"
python examples/pretrain/GalaxeaLiberoMix16GPU/deploy.py dry-run --spec "$SPEC" --mode smoke
python examples/pretrain/GalaxeaLiberoMix16GPU/deploy.py launch --spec "$SPEC" --mode smoke
python examples/pretrain/GalaxeaLiberoMix16GPU/deploy.py probe --spec "$SPEC"
# After smoke_success.json is present and both nodes have released their GPUs:
python examples/pretrain/GalaxeaLiberoMix16GPU/deploy.py dry-run --spec "$SPEC" --mode formal
python examples/pretrain/GalaxeaLiberoMix16GPU/deploy.py launch --spec "$SPEC" --mode formal
```

The controller snapshots only an explicit source-file allowlist under a fresh
control directory, pins both checkout commits and the snapshot hashes, then
launches node 1 before node 0 in detached screen sessions. Source preflight checks
metadata and baseline provenance; GPU checks reject active jobs, uncorrectable
ECC and pending recovery. Node barriers reject a peer's failure and time out.
Cleanup signals only the process groups created by this launcher. Existing run,
control, training and evaluation directories are never overwritten.

Formal mode automatically runs posttraining/evaluation on rank 0 after both
pretraining nodes validate and exit. Rank 1 is released at that point. Success
is published only after the fixed final checkpoint and complete evaluation
validate. Any stage failure stops downstream work.

The shared posttraining runner accepts a 3-update source only when both
`SMOKE_TEST=1` and `SOURCE_SMOKE_TEST=1` are explicit. It checks the real source
marker/config/metrics, finite losses and 192 global64 examples. Formal source
validation continues to require 80,000 updates. No metadata is fabricated.

## CPU checks

```bash
python -m unittest discover -s examples/pretrain/GalaxeaLiberoMix16GPU -p 'test_*.py' -v
python -m unittest discover -s examples/pretrain/GalaxeaLiberoMix16GPU/train_files -p 'test_*.py' -v
bash -n examples/pretrain/GalaxeaLiberoMix16GPU/train_files/run_mixed_16gpu_pretrain.sh
```

These tests cover command/run isolation, invalid batch/seed/resume rejection,
source smoke gating, cross-node protocol mismatch, barrier timeout, failure
propagation and whole-group cleanup after a group leader exits. GPU training,
full-model inheritance and simulation are separate end-to-end checks.
