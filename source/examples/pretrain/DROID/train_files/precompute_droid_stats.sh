#!/usr/bin/env bash
# Exact full-dataset absolute + chunk-relative action statistics for DROID.
# CPU only. DRY_RUN=1 previews the work and never writes dataset metadata.
# Example after selecting concurrency:
#   DRY_RUN=0 STATS_WORKERS=160 bash examples/pretrain/DROID/train_files/precompute_droid_stats.sh
set -Eeuo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/CRA/bin/python}"
DROID_DATASET_DIR="${DROID_DATASET_DIR:-/mnt/zhaoshijie/dataset/oxe_franka_lerobot/droid_1.0.1/stage3_unmasked_20261021_142915/job/full/dataset}"
DRY_RUN="${DRY_RUN:-1}"
STATS_FORCE="${STATS_FORCE:-0}"
STATS_WORKERS="${STATS_WORKERS:-160}"
STATS_EPISODES_PER_TASK="${STATS_EPISODES_PER_TASK:-32}"
for flag in DRY_RUN STATS_FORCE; do [[ "${!flag}" =~ ^[01]$ ]] || die "$flag must be 0 or 1"; done
positive_int STATS_EPISODES_PER_TASK "$STATS_EPISODES_PER_TASK"
[[ -x "$PYTHON_BIN" ]] || die "Missing Python interpreter: $PYTHON_BIN"
[[ -d "$DROID_DATASET_DIR/meta" ]] || die "Missing dataset metadata: $DROID_DATASET_DIR/meta"
HELPER="$SCRIPT_DIR/precompute_droid_stats.py"
[[ -f "$HELPER" ]] || die "Missing statistics helper: $HELPER"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1

available_cpus="$("$PYTHON_BIN" -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)')"
cmd=("$PYTHON_BIN" "$HELPER" --dataset-dir "$DROID_DATASET_DIR" --robot-type droid_franka_rot6d
     --episodes-per-task "$STATS_EPISODES_PER_TASK")
if [[ -n "${STATS_WORKERS:-}" ]]; then
  positive_int STATS_WORKERS "$STATS_WORKERS"
  (( STATS_WORKERS <= available_cpus )) || die "STATS_WORKERS=$STATS_WORKERS exceeds CPU affinity ($available_cpus)"
  cmd+=(--workers "$STATS_WORKERS")
elif [[ "$DRY_RUN" == 0 ]]; then
  die 'Set STATS_WORKERS explicitly before computing statistics'
fi
if [[ -n "${STATS_SUMMARY_WORKERS:-}" ]]; then
  positive_int STATS_SUMMARY_WORKERS "$STATS_SUMMARY_WORKERS"
  (( STATS_SUMMARY_WORKERS <= available_cpus )) || die "STATS_SUMMARY_WORKERS exceeds CPU affinity ($available_cpus)"
  cmd+=(--summary-workers "$STATS_SUMMARY_WORKERS")
fi
[[ -z "${STATS_TEMP_DIR:-}" ]] || cmd+=(--temp-dir "$STATS_TEMP_DIR")
[[ "$STATS_FORCE" == 0 ]] || cmd+=(--force)
[[ "$DRY_RUN" == 0 ]] || cmd+=(--dry-run)
printf 'DROID statistics: dry_run=%s cpus=%s workers=%s summary_workers=%s\n' \
  "$DRY_RUN" "$available_cpus" "${STATS_WORKERS:-not-selected}" "${STATS_SUMMARY_WORKERS:-helper-default}"
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
# The helper owns the dataset-scoped lock and atomic metadata replacement.
exec "${cmd[@]}"
