#!/usr/bin/env bash
# Run DROID pretraining -> LIBERO Plus posttraining -> full evaluation in order.
# Precomputation is explicit: PREPARE_DATA=1 adds statistics then frame caching.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PIPELINE_ID="${PIPELINE_ID:-droid_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
PIPELINE_ID="$(smoke_run_id "$PIPELINE_ID")"
validate_run_id "$PIPELINE_ID"
PREPARE_DATA="${PREPARE_DATA:-0}"
[[ "$PREPARE_DATA" =~ ^[01]$ ]] || die 'PREPARE_DATA must be 0 or 1'
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-${PIPELINE_ID}_pretrain}"
POSTTRAIN_RUN_ID="${POSTTRAIN_RUN_ID:-${PIPELINE_ID}_libero_plus}"
PRETRAIN_RUN_ID="$(smoke_run_id "$PRETRAIN_RUN_ID")"
POSTTRAIN_RUN_ID="$(smoke_run_id "$POSTTRAIN_RUN_ID")"
validate_run_id "$PRETRAIN_RUN_ID"; validate_run_id "$POSTTRAIN_RUN_ID"
[[ "$PRETRAIN_RUN_ID" != "$POSTTRAIN_RUN_ID" ]] || die 'Training stage run IDs must differ'
export PRETRAIN_RUN_DIR="$RUN_ROOT_DIR/$PRETRAIN_RUN_ID"
export POSTTRAIN_RUN_DIR="$RUN_ROOT_DIR/$POSTTRAIN_RUN_ID"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_ROOT_DIR/${POSTTRAIN_RUN_ID}_final$( [[ "$SMOKE_TEST" == 0 ]] || printf _smoke )}"
[[ "$SMOKE_TEST" == 0 || "$EVAL_OUTPUT_DIR" == *_smoke ]] || die 'Smoke EVAL_OUTPUT_DIR must end with _smoke'
export DRY_RUN EXPERIMENT_ENV
PIPELINE_DIR="$RUN_ROOT_DIR/launcher_logs/$PIPELINE_ID"
printf 'Pipeline %s on %s: DROID -> LIBERO Plus -> eval (formal=10030, smoke=1); prepare=%s dry_run=%s\n' "$PIPELINE_ID" "$(hostname)" "$PREPARE_DATA" "$DRY_RUN"
if [[ "$DRY_RUN" == 0 ]]; then
  for path in "$PIPELINE_DIR" "$PRETRAIN_RUN_DIR" "$POSTTRAIN_RUN_DIR" "$EVAL_OUTPUT_DIR"; do
    [[ ! -e "$path" ]] || die "Pipeline path already exists: $path"
  done
  exec 8>"/tmp/starvla_droid_pipeline_$(id -u).lock"
  flock -n 8 || die 'Another DROID pipeline is active on this host'
  mkdir -p "$(dirname "$PIPELINE_DIR")"
  mkdir "$PIPELINE_DIR"
  phase=starting
  trap 'rc=$?; printf "%s\n" "$rc" > "$PIPELINE_DIR/exit_code"; printf "phase=%s exit_code=%s\n" "$phase" "$rc" >> "$PIPELINE_DIR/status.txt"' EXIT
  printf 'pretrain=%s\nposttrain=%s\neval=%s\n' "$PRETRAIN_RUN_DIR" "$POSTTRAIN_RUN_DIR" "$EVAL_OUTPUT_DIR" > "$PIPELINE_DIR/paths.txt"
fi
run_step() {
  phase="$1"
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    "$@"
  else
    printf 'time=%s phase=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" >> "$PIPELINE_DIR/status.txt"
    "$@" 2>&1 | tee "$PIPELINE_DIR/$phase.log"
  fi
}
if [[ "$PREPARE_DATA" == 1 ]]; then
  run_step statistics env DROID_DATASET_DIR="$DROID_DATASET_DIR" bash "$TRAIN_FILES/precompute_droid_stats.sh"
  run_step cache env DROID_DATASET_DIR="$DROID_DATASET_DIR" FRAME_CACHE_DIR="$DROID_FRAME_CACHE" bash "$TRAIN_FILES/precompute_droid_frame_cache.sh"
fi
run_step pretraining env RUN_ID="$PRETRAIN_RUN_ID" bash "$TRAIN_FILES/run_droid_pretrain.sh"
run_step posttraining env RUN_ID="$POSTTRAIN_RUN_ID" bash "$TRAIN_FILES/run_libero_plus_posttrain.sh"
run_step evaluating bash "$TRAIN_FILES/run_libero_plus_eval.sh"
phase=completed
printf 'Pipeline finished successfully (dry_run=%s).\n' "$DRY_RUN"
