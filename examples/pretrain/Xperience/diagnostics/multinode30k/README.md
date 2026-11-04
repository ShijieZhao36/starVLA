# Xperience hours100: decoder LayerNorm, 2 nodes, 16 GPUs, 30k updates

Deploy the launcher and three configuration files under
`/mnt/zhaoshijie/workspace/starVLA/examples/pretrain/Xperience/train_files/`
and this README under
`/mnt/zhaoshijie/workspace/starVLA/examples/pretrain/Xperience/diagnostics/multinode30k/`.
neo-bj1 and neo-bj4 share this GPFS path, so deploy once. The checkout needs the
decoder-input-LayerNorm patch and the gr00t self-sharded-dataloader fix.

## Experiment

- neo-bj1 is node rank 0; neo-bj4 is node rank 1; 8 H100 GPUs per node.
- DeepSpeed ZeRO-2, BF16, no optimizer/parameter CPU offload. BF16 is configured
  in the DeepSpeed JSON; do not duplicate `mixed_precision` in the Accelerate
  YAML when using this external DeepSpeed configuration.
- Per-device batch 8, gradient accumulation 1, global batch 128.
- 30,000 optimizer updates; 2,500 warmup updates; cosine decay to 1e-6.
- Action-head learning rate 1e-4; Qwen3-VL learning rate 1e-5.
- Original Qwen3-VL-4B-Instruct weights; entire action head newly initialized.
- Decoder input LN enabled; VLM and action head both train (`freeze_modules: ''`).
- Same Xperience hours100 data, action processor, masks and frame cache as the
  existing single-node script. Action horizon 8; canonical dimension 90 with
  Xperience's existing 48-dimensional mask.
- Save model weights every 5,000 updates. Evaluate every 100 updates by the
  existing QwenPI's 4-step Euler action prediction (the trainer's requested
  `num_ddim_steps=20` is ignored by this framework); this consumes an additional
  training-data batch. This path does not switch the model to eval mode, so
  dropout remains active. Its metrics are not held-out validation results.
- `gradient_checkpointing: true` is retained in the configuration, but the
  existing QwenPI path does not actually enable activation checkpointing.
- The original trainer saves model weights, not full optimizer/RNG state.
  This launcher is intentionally for fresh runs, not exact resumption.

## Included training fixes

- For `gr00t_sharded` only, `train_starvla.py` keeps the already self-sharded
  dataloader out of Accelerate's dataloader preparation, preventing a second
  rank-based split. Model and optimizer preparation still use Accelerate.
  The DeepSpeed microbatch size is set explicitly because the dataloader is
  no longer passed to its preparation step. Other dataset paths are unchanged.
- QwenPI converts predicted actions to FP32 with `.float()` before NumPy
  conversion, allowing BF16 inference outputs to be evaluated.
- `test_single_shard.py` provides CPU regression checks for the dataloader
  preparation change. Distributed communication and the actual GPU training
  loop still require the two-node smoke run below.

## Start on both machines

Generate a RUN_ID once on the controller:

```bash
RUN_ID="xperience_hours100_ln_16gpu_30k_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
```

Copy that exact value to both commands below. MASTER_ADDR defaults to neo-bj1's
verified private IPv4 address `172.27.6.128`. Run on each host:

```bash
# neo-bj1
NODE_RANK=0 MASTER_ADDR=172.27.6.128 MASTER_PORT=29604 \
RUN_ID=THE_SAME_TIMESTAMPED_RUN_ID \
bash /mnt/zhaoshijie/workspace/starVLA/examples/pretrain/Xperience/train_files/run_xperience_hours100_ln_2nodes_30k.sh
```

```bash
# neo-bj4
NODE_RANK=1 MASTER_ADDR=172.27.6.128 MASTER_PORT=29604 \
RUN_ID=THE_SAME_TIMESTAMPED_RUN_ID \
bash /mnt/zhaoshijie/workspace/starVLA/examples/pretrain/Xperience/train_files/run_xperience_hours100_ln_2nodes_30k.sh
```

Use tmux or nohup externally to keep each launcher alive. Store launcher logs
outside the experiment output directory, for example under
`/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/`. Rank 0 atomically creates
the output directory and rejects an existing directory. Only rank 0 writes
the source/config snapshots under `RUN_ROOT_DIR/RUN_ID/launch/`.

## Checks and short run

`DRY_RUN=1` validates arguments and required paths, and prints the command;
it does not initialize CUDA, test communication or create an output directory.

For a short end-to-end two-node run, use a separate RUN_ID and set the same
overrides on both nodes:

```bash
MAX_TRAIN_STEPS=10 WARMUP_STEPS=2 SAVE_INTERVAL=10 \
EVAL_INTERVAL=100 LOGGING_FREQUENCY=1 WANDB_MODE=disabled
```

The larger eval interval skips the expensive action evaluation during this
short run. To exercise that path too, use `EVAL_INTERVAL=10`. The trainer
also saves a final model at the end, so a smoke run still writes model files.
Use a new RUN_ID and the default 30k settings for the actual experiment.

Environment overrides: `REPO_DIR`, `PYTHON_BIN`, `RUN_ROOT_DIR`, `NODE_RANK`,
`MASTER_ADDR`, `MASTER_PORT`, `RUN_ID`, `MAX_TRAIN_STEPS`, `WARMUP_STEPS`,
`SAVE_INTERVAL`, `EVAL_INTERVAL`, `LOGGING_FREQUENCY`, `DRY_RUN`,
`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `WANDB_PROJECT`, `WANDB_ENTITY`,
and normal W&B/NCCL environment variables. `PRETRAINED_CHECKPOINT` must be
unset or empty. Exactly eight GPU IDs are required per host.

The defaults `NCCL_SOCKET_IFNAME=eth0`, `GLOO_SOCKET_IFNAME=eth0`, and
`NCCL_IB_HCA='=com0_0,com0_1,com0_2,com0_3,com0_4,com0_5,com0_6,com0_7'`
passed a 16-rank communication test on these hosts using NET/IB/GDRDMA.
They remain overridable; use identical settings on both hosts.
`NCCL_DEBUG=WARN` keeps regular training logs compact.
