#!/usr/bin/env bash
# Shared paths and validation for the single-node, eight-GPU DROID pipeline.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }
TRAIN_FILES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$TRAIN_FILES/../../../.." && pwd)}"
EXPERIMENT_ENV="${EXPERIMENT_ENV:-$TRAIN_FILES/experiment.env}"
[[ -f "$EXPERIMENT_ENV" ]] || die "Missing experiment settings: $EXPERIMENT_ENV"
# This is a user-editable shell configuration, never generated from data text.
source "$EXPERIMENT_ENV"
SMOKE_TEST="${SMOKE_TEST:-0}"
[[ "$SMOKE_TEST" =~ ^[01]$ ]] || die 'SMOKE_TEST must be 0 or 1'
if [[ "$SMOKE_TEST" == 1 ]]; then
  # Integration checks have their own budgets and output names. Never modify
  # the formal YAML defaults or reuse a formal run directory.
  PRETRAIN_STEPS=3 POSTTRAIN_STEPS=3 PRETRAIN_WARMUP=1 POSTTRAIN_WARMUP=1
  PRETRAIN_SAVE_INTERVAL=3 POSTTRAIN_SAVE_INTERVAL=3 LOG_INTERVAL=1
fi
export SMOKE_TEST PRETRAIN_STEPS POSTTRAIN_STEPS PRETRAIN_WARMUP POSTTRAIN_WARMUP
export PRETRAIN_SAVE_INTERVAL POSTTRAIN_SAVE_INTERVAL LOG_INTERVAL
[[ "$DRY_RUN" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
[[ -x "$PYTHON_BIN" ]] || die "Missing Python: $PYTHON_BIN"
for name in RUN_ROOT_DIR BASE_VLM DROID_DATASET_DIR DROID_FRAME_CACHE LIBERO_DATA_ROOT LIBERO_FRAME_CACHE; do
  [[ "${!name}" == /* ]] || die "$name must be an absolute path"
done
[[ "$(basename "$DROID_DATASET_DIR")" == dataset ]] || die 'DROID dataset directory must be named dataset'
export REPO_DIR PYTHON_BIN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT WANDB_ENTITY WANDB_MODE
export STATS_WORKERS STATS_SUMMARY_WORKERS STATS_TEMP_DIR CACHE_WORKERS CACHE_CHUNK CACHE_SIZE
CHECKS="$TRAIN_FILES/pipeline_checks.py"

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 'Run IDs may contain letters, numbers, dot, underscore and hyphen'
}

smoke_run_id() {
  if [[ "$SMOKE_TEST" == 1 && "$1" != *_smoke ]]; then
    printf '%s_smoke' "$1"
  else
    printf '%s' "$1"
  fi
}

require_eight_idle_gpus() {
  [[ "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" == '0,1,2,3,4,5,6,7' ]] || die 'This protocol uses physical GPUs 0-7'
  local count jobs
  count="$(timeout 15 nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  (( count == 8 )) || die "Expected eight GPUs, found $count"
  jobs="$(timeout 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${jobs//[[:space:]]/}" ]] || die "GPU processes are already active: $jobs"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
}

record_launch() {
  local output="$1"
  cp "$TRAIN_FILES"/*.sh "$TRAIN_FILES"/*.yaml "$TRAIN_FILES"/*.json "$TRAIN_FILES"/*.py "$EXPERIMENT_ENV" "$output/launch/"
  cp "$TRAIN_FILES/data_registry/data_config.py" "$output/launch/droid_data_config.py"
  cp "$REPO_DIR/examples/pretrain/canonical/robots/droid_franka.yaml" "$output/launch/"
  cp "$REPO_DIR/examples/pretrain/canonical/bimanual_v1.yaml" "$output/launch/"
  git -C "$REPO_DIR" rev-parse HEAD > "$output/launch/git_commit.txt"
  git -C "$REPO_DIR" status --short > "$output/launch/git_status.txt"
  git -C "$REPO_DIR" diff --binary HEAD > "$output/launch/tracked_changes.patch"
  printf 'host=%s\nstarted_at=%s\n' "$(hostname)" "$(TZ=Asia/Shanghai date --iso-8601=seconds)" > "$output/launch/runtime.txt"
}
