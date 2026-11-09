#!/usr/bin/env bash
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
: "${REPO_DIR:?}" "${RUN_ID:?}" "${ARM:?}"
[[ "$ARM" == A || "$ARM" == B ]] || die 'ARM must be A or B'
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/MRA/bin/python}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
MASTER_PORT="${MASTER_PORT:-29821}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 'Invalid run id'
[[ "$SMOKE_TEST" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || die 'Invalid flags'
TRAIN_FILES="$REPO_DIR/examples/pretrain/Ego50AB/train_files"
CONFIG_YAML="$TRAIN_FILES/pretrain_${ARM}.yaml"
ACCELERATE_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/accelerate_libero_plus_ln_8gpu.yaml"
DEEPSPEED_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json"
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
MAX_TRAIN_STEPS=80000 WARMUP_STEPS=5000 SAVE_INTERVAL=10000 LOG_INTERVAL=100
if [[ "$SMOKE_TEST" == 1 ]]; then MAX_TRAIN_STEPS=3 WARMUP_STEPS=1 SAVE_INTERVAL=3 LOG_INTERVAL=1; fi
cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$(dirname "$PYTHON_BIN"):$PATH" PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=offline
[[ "$SMOKE_TEST" == 0 ]] || export WANDB_MODE=disabled
train_cmd=("$PYTHON_BIN" -m accelerate.commands.launch
 --config_file "$ACCELERATE_CONFIG" --num_machines 1 --num_processes 8 --machine_rank 0
 --main_process_port "$MASTER_PORT" --max_restarts 0 --deepspeed_config_file "$DEEPSPEED_CONFIG"
 "$REPO_DIR/starVLA/training/train_starvla.py" --config_yaml "$CONFIG_YAML"
 --trainer.max_train_steps "$MAX_TRAIN_STEPS" --trainer.num_warmup_steps "$WARMUP_STEPS"
 --trainer.save_interval "$SAVE_INTERVAL" --trainer.logging_frequency "$LOG_INTERVAL"
 --trainer.eval_interval "$LOG_INTERVAL"
 --trainer.expected_initial_model_sha256 530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89
 --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID")
printf 'ARM=%s RUN_ID=%s SMOKE=%s global_batch=64\n' "$ARM" "$RUN_ID" "$SMOKE_TEST"
printf 'Command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
[[ "$DRY_RUN" == 0 ]] || exit 0
exec 9>"/tmp/starvla_mix_pair_8gpu_$(id -u).lock"
flock -n 9 || die 'Another training pipeline holds the host lock'
gpu_pids="$(timeout 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${gpu_pids//[[:space:]]/}" ]] || die 'GPU compute processes are active'
mkdir -p "$RUN_ROOT_DIR"
mkdir "$OUTPUT_DIR" || die 'Output already exists'
mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb"
phase=preflight
finish() { rc=$?; printf '%s\n' "$rc" > "$OUTPUT_DIR/exit_code"; printf 'phase=%s exit_code=%s\n' "$phase" "$rc" >> "$OUTPUT_DIR/pipeline_status.txt"; }
trap finish EXIT
cp "${BASH_SOURCE[0]}" "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/launch/"
git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
git status --short > "$OUTPUT_DIR/launch/git_status.txt"
git diff --binary HEAD > "$OUTPUT_DIR/launch/tracked_changes.patch"
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch/train_command.txt"
printf '\n' >> "$OUTPUT_DIR/launch/train_command.txt"
phase=pretraining
"${train_cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
phase=validating
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$TRAIN_FILES/validate_pretrain.py" --run "$OUTPUT_DIR" --arm "$ARM" --steps "$MAX_TRAIN_STEPS"
phase=completed
