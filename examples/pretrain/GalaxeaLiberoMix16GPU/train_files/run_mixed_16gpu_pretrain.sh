#!/usr/bin/env bash
# Run on BOTH nodes using one RUN_ID and one fresh shared LAUNCHER_DIR.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
: "${NODE_RANK:?Set NODE_RANK=0 on neo-bj5 or 1 on neo-bj6}"
: "${RUN_ID:?Use the same fresh RUN_ID on both nodes}"
: "${LAUNCHER_DIR:?Use the same fresh shared launcher directory on both nodes}"
[[ "$NODE_RANK" =~ ^[01]$ ]] || die 'Invalid NODE_RANK'
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 'Invalid RUN_ID'
[[ "$LAUNCHER_DIR" == /* ]] || die 'LAUNCHER_DIR must be absolute'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/MRA/bin/python}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
: "${MASTER_ADDR:?Set the rank-0 address reachable from both nodes}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
[[ "$DRY_RUN" =~ ^[01]$ && "$SMOKE_TEST" =~ ^[01]$ ]] || die 'Invalid mode'
if [[ -z "${PER_DEVICE_BATCH_SIZE:-}" ]]; then
 if [[ -n "${GLOBAL_BATCH:-}" ]]; then
  [[ "$GLOBAL_BATCH" == 64 || "$GLOBAL_BATCH" == 128 ]] || die 'GLOBAL_BATCH must be 64 or 128'
  PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH / 16))
 else PER_DEVICE_BATCH_SIZE=4; fi
fi
[[ "$PER_DEVICE_BATCH_SIZE" == 4 || "$PER_DEVICE_BATCH_SIZE" == 8 ]] || die 'PER_DEVICE_BATCH_SIZE must be 4 or 8'
GLOBAL_BATCH="${GLOBAL_BATCH:-$((16 * PER_DEVICE_BATCH_SIZE))}"
[[ "$GLOBAL_BATCH" == "$((16 * PER_DEVICE_BATCH_SIZE))" ]] || die 'GLOBAL_BATCH does not match 16 ranks'
MASTER_PORT="${MASTER_PORT:-}"
MAX_TRAIN_STEPS=80000 WARMUP_STEPS=5000 SAVE_INTERVAL=10000 LOG_INTERVAL=100
if [[ "$SMOKE_TEST" == 1 ]]; then
 [[ "$RUN_ID" == *_smoke3 ]] || die 'Smoke RUN_ID needs _smoke3 suffix'
 MASTER_PORT="${MASTER_PORT:-${SMOKE_PORT:-29715}}"
 MAX_TRAIN_STEPS=3 WARMUP_STEPS=1 SAVE_INTERVAL=3 LOG_INTERVAL=1
else
 [[ "$RUN_ID" != *_smoke3 ]] || die 'Formal run cannot use smoke suffix'
 MASTER_PORT="${MASTER_PORT:-29716}"
fi
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] && ((MASTER_PORT > 1024 && MASTER_PORT < 65536)) || die 'Invalid master port'
[[ "$MASTER_ADDR" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'Invalid master address'
CONFIG_YAML="$REPO_DIR/examples/pretrain/GalaxeaLiberoMix/train_files/starvla_mixed_ln_80k.yaml"
DEEPSPEED_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json"
ACCELERATE_CONFIG="$SCRIPT_DIR/accelerate_mixed_2nodes.yaml"
VALIDATOR="$SCRIPT_DIR/validate_mixed16.py"
COORDINATOR="$SCRIPT_DIR/coordinate_nodes.py"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NO_ALBUMENTATIONS_UPDATE=1
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='=com0_0,com0_1,com0_2,com0_3,com0_4,com0_5,com0_6,com0_7'
unset NCCL_IB_DISABLE PRETRAINED_CHECKPOINT
export WANDB_MODE=offline
[[ "$SMOKE_TEST" == 0 ]] || export WANDB_MODE=disabled
train_cmd=("$PYTHON_BIN" -m accelerate.commands.launch
 --config_file "$ACCELERATE_CONFIG" --num_machines 2 --num_processes 16 --machine_rank "$NODE_RANK"
 --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT"
 --same_network --rdzv_backend static --max_restarts 0 --deepspeed_multinode_launcher standard
 --deepspeed_config_file "$DEEPSPEED_CONFIG"
 "$REPO_DIR/starVLA/training/train_starvla.py" --config_yaml "$CONFIG_YAML"
 --datasets.vla_data.per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
 --trainer.max_train_steps "$MAX_TRAIN_STEPS" --trainer.num_warmup_steps "$WARMUP_STEPS"
 --trainer.save_interval "$SAVE_INTERVAL" --trainer.logging_frequency "$LOG_INTERVAL" --trainer.eval_interval "$LOG_INTERVAL"
 --trainer.expected_initial_model_sha256 530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89
 --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID")
gate_args=(--repo "$REPO_DIR" --run "$OUTPUT_DIR" --launcher "$LAUNCHER_DIR" --node-rank "$NODE_RANK"
 --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" --global-batch "$GLOBAL_BATCH")
[[ "$SMOKE_TEST" == 0 ]] || gate_args+=(--smoke)
[[ -z "${EXPECTED_COMMIT:-}" ]] || gate_args+=(--expected-commit "$EXPECTED_COMMIT")
[[ -z "${REFERENCE_RUN:-}" ]] || gate_args+=(--reference-run "$REFERENCE_RUN")
coord_args=(--run "$OUTPUT_DIR" --launcher "$LAUNCHER_DIR" --node-rank "$NODE_RANK")
printf 'run=%s node_rank=%s world_size=16 global_batch=%s dry_run=%s smoke=%s\n' "$RUN_ID" "$NODE_RANK" "$GLOBAL_BATCH" "$DRY_RUN" "$SMOKE_TEST"
printf 'Command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
 # Print-only mode is deliberately usable from a machine without the training mounts.
 printf 'Preflight command: '; printf '%q ' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}"; printf '\n'
 exit 0
fi
[[ -x "$PYTHON_BIN" ]] || die 'Python runtime missing'
[[ ! -e "$LAUNCHER_DIR/node$NODE_RANK" ]] || die 'Node launcher directory exists; choose a fresh launcher directory'
mkdir -p "$LAUNCHER_DIR"
mkdir "$LAUNCHER_DIR/node$NODE_RANK"
phase=preflight
owns_run=0
finish() {
 rc=$?
 trap - EXIT
 "$PYTHON_BIN" "$COORDINATOR" mark "${coord_args[@]}" --path "$LAUNCHER_DIR/node$NODE_RANK/process_exit_code" --value "$rc" || true
 printf 'time=%s phase=%s exit_code=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" "$rc" >> "$LAUNCHER_DIR/node$NODE_RANK/stages.txt"
 if [[ "$rc" != 0 ]]; then
  "$PYTHON_BIN" - "$LAUNCHER_DIR/node$NODE_RANK/failure.json" "$NODE_RANK" "$phase" "$rc" <<'PYFAIL'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_name(p.name+f'.{os.getpid()}.tmp')
t.write_text(json.dumps(dict(node_rank=int(sys.argv[2]),phase=sys.argv[3],exit_code=int(sys.argv[4])))+'\n'); os.replace(t,p)
PYFAIL
  "$PYTHON_BIN" "$COORDINATOR" mark "${coord_args[@]}" --path "$LAUNCHER_DIR/node$NODE_RANK/exit_code" --value "$rc" || true
  if [[ "$NODE_RANK" == 0 && "$owns_run" == 1 ]]; then
   "$PYTHON_BIN" "$COORDINATOR" mark "${coord_args[@]}" --path "$OUTPUT_DIR/exit_code" --value "$rc" || true
  fi
 fi
 exit "$rc"
}
trap finish EXIT
exec 9>"/tmp/starvla_mix_pair_8gpu_$(id -u).lock"
flock -n 9 || die 'A mixed/pretraining pipeline holds this host lock'
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" --report "$LAUNCHER_DIR/node$NODE_RANK/preflight.json"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" gpu "${gate_args[@]}" > "$LAUNCHER_DIR/node$NODE_RANK/gpu_preflight.json"
"$PYTHON_BIN" "$COORDINATOR" mark "${coord_args[@]}" --path "$LAUNCHER_DIR/node$NODE_RANK/ready" --value 1
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 1200 --path "$LAUNCHER_DIR/node0/ready" --path "$LAUNCHER_DIR/node1/ready"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" compare "${gate_args[@]}"
cd "$REPO_DIR"
if [[ "$NODE_RANK" == 0 ]]; then
 [[ ! -e "$OUTPUT_DIR" ]] || die 'Output exists; choose a fresh RUN_ID'
 mkdir "$OUTPUT_DIR"; mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb"
 owns_run=1
 cp "$SCRIPT_DIR/run_mixed_16gpu_pretrain.sh" "$VALIDATOR" "$COORDINATOR" "$ACCELERATE_CONFIG" "$CONFIG_YAML" "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/launch/"
 cp "$LAUNCHER_DIR/node0/preflight.json" "$OUTPUT_DIR/launch/preflight_node0.json"
 cp "$LAUNCHER_DIR/node1/preflight.json" "$OUTPUT_DIR/launch/preflight_node1.json"
 git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
 git status --short > "$OUTPUT_DIR/launch/git_status.txt"
 "$PYTHON_BIN" "$COORDINATOR" mark "${coord_args[@]}" --path "$OUTPUT_DIR/launch/created" --value 1
else
 "$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$OUTPUT_DIR/launch/created"
 owns_run=1
fi
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch/train_command_node$NODE_RANK.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
phase=training
train_log="$OUTPUT_DIR/train_node$NODE_RANK.log"
[[ "$NODE_RANK" == 1 ]] || train_log="$OUTPUT_DIR/train.log"
"$PYTHON_BIN" "$COORDINATOR" run "${coord_args[@]}" --timeout 172800 --log "$train_log" -- "${train_cmd[@]}"
"$PYTHON_BIN" "$COORDINATOR" mark "${coord_args[@]}" --path "$LAUNCHER_DIR/node$NODE_RANK/train_exit" --value 0
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 1200 --path "$LAUNCHER_DIR/node0/train_exit" --path "$LAUNCHER_DIR/node1/train_exit"
phase=validation
if [[ "$NODE_RANK" == 0 ]]; then
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" trained "${gate_args[@]}"
fi
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 1200 --path "$OUTPUT_DIR/launch/pretrain_validation.json"
"$PYTHON_BIN" "$COORDINATOR" mark "${coord_args[@]}" --path "$LAUNCHER_DIR/node$NODE_RANK/exit_code" --value 0
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$LAUNCHER_DIR/node0/exit_code" --path "$LAUNCHER_DIR/node1/exit_code"
if [[ "$NODE_RANK" == 0 ]]; then
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" complete "${gate_args[@]}"
fi
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$OUTPUT_DIR/pretrain_success.json" --path "$LAUNCHER_DIR/exit_code"
phase=completed
printf 'Completed mixed16 pretraining: %s node %s\n' "$RUN_ID" "$NODE_RANK"
