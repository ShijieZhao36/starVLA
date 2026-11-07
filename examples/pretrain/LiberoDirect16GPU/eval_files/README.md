# LIBERO Plus 16-GPU final-checkpoint evaluation

Run `run_libero_plus_16gpu.py` once on each host in its existing background
pipeline. Set the established `LIBERO_HOME`, `LIBERO_CONFIG_PATH`, `PYTHONPATH`
(simulation overlay, repository and LIBERO checkout), `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl` first. Both hosts must see the same checkpoint and fresh
shared output directory. Each host uses its own Python environment.

```bash
python run_libero_plus_16gpu.py \
  --node-rank 0 \
  --repo /mnt/zhaoshijie/workspace/starVLA_libero_direct_ln_16gpu_30k_20261107 \
  --pretrained-path /absolute/run/final_model/pytorch_model.pt \
  --output-dir /absolute/shared/evaluation
```

Use `--node-rank 1` on the second host. `--dry-run` checks the task plan and
protocol on CPU without creating output or launching processes. A future final
checkpoint may be absent during `--dry-run`; its size/time fields are then null
and its existence check is explicitly deferred until real launch. An existing
empty file is rejected, and a real launch always requires a non-empty final
checkpoint. `--smoke`
evaluates 16 distinct tasks, one per GPU and one simulator per GPU, using a
separate smoke output directory. The formal mode starts 16 policy servers and
64 simulator workers: 8 GPUs × 4 simulators on each host.

The policy server, simulation worker, rollout code and final result verifier
are imported from the specified repository without modifying them. Protocol:
canonical forward, BF16, positive gripper sign, environment seed 7, one trial
per task, 10 warmup environment steps, original unseeded policy noise. Execution
errors fail validation and are not automatically retried. Policy failure is a
valid score of zero.

Each suite's original `task_id % 16` selects its global GPU. This covers all
10,030 tasks exactly once, with 5,023 on node 0 and 5,007 on node 1. Each GPU has
an independent queue and uniquely named logs/video output. Node 0 merges both
completed result files and invokes the existing strict result verifier. A
successful full run must have suite counts 2,402 / 2,518 / 2,591 / 2,519 for
Spatial / Object / Goal / LIBERO-10, with no missing tasks, duplicates or
execution errors.

## Isolated native-crash diagnostic

`diagnose_one_task.py` is separate from the formal 16-GPU evaluator. It runs one
original policy server, one original worker with observation-only wrappers,
and the original task queue with one attempt. It never writes a formal
`eval_success.json`. Example for neo-bj6 local GPU 4, using a completed reference
model only to check evaluation infrastructure:

```bash
python diagnose_one_task.py \
  --repo /mnt/zhaoshijie/workspace/starVLA_libero_direct_ln_16gpu_30k_20261107 \
  --pretrained-path /mnt/zhaoshijie/checkpoints/starVLA/libero_from_galaxea_control_ln_8gpu_60k_20261105_pair/final_model/pytorch_model.pt \
  --output-dir /absolute/fresh/diagnostic_control_task3 \
  --suite libero_10 --task-id 3 --local-gpu 4
```

The environment variables listed above must be set exactly as for the formal
run. `--dry-run` checks the plan without using a GPU. The observer logs every
environment step's action shape, finite count and finite min/max; it forwards
the exact original action object, does not clip or replace values, and does not
change RNG state. It logs rollout return and environment-close boundaries.
Both policy and worker enable Python's native fatal-signal traceback handler.
Worker errors include worker ID, PID, exit code, signal name and log path.

Output includes `worker.stdout`, `policy_server.log`, `processes.json`, the
one-episode results and a `diagnostic_success.json` or `failure.json`. A
reference policy that finishes successfully before the 520-step limit does not
establish that a full-length random-action trajectory is stable. This tool does
not modify the training initialization or the formal 10030-task selection.

Shared output contains `node0/` and `node1/` with immutable run claims,
protocol plans, GPU logs, node results and exit markers. Final aggregate files
are `results.jsonl`, `summary.json`, `summary.txt`,
`completion_validation.json`, `eval_success.json` and `exit_code`.
`eval_success.json` is written only after aggregate validation; it contains
`smoke: true` for a smoke run. Both node processes return success only after
the aggregate success marker exists. Existing node directories reject repeat
submission. Peer and evaluation timeouts default to 12 hours; failures preserve
evidence and terminate only child process groups owned by that node.

`test_eval16.py` runs local CPU integration tests using the original queue and
verifier sources fetched read-only from neo-bj5. It substitutes small fake
policy/rollout processes. The tests verify two-node coordination for the
16-task smoke and complete 10,030-task formal plans, every GPU's participation,
cross-node failure propagation, and rejection of missing, duplicate, erroneous
or repeated-attempt records. They do not test actual GPU inference.
