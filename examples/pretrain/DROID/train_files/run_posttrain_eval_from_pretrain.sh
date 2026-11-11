#!/usr/bin/env bash
# Continue a validated DROID checkpoint with new LIBERO Plus training and eval.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${PRETRAIN_RUN_DIR:?Set PRETRAIN_RUN_DIR to the completed DROID pretraining run}"
[[ "$PRETRAIN_RUN_DIR" == /* && -d "$PRETRAIN_RUN_DIR" ]] || die 'PRETRAIN_RUN_DIR must be an existing absolute directory'
RECOVERY_ID="${RECOVERY_ID:-droid_posttrain_recovery_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
POSTTRAIN_RUN_ID="${POSTTRAIN_RUN_ID:-${RECOVERY_ID}_libero_plus}"
RECOVERY_ID="$(smoke_run_id "$RECOVERY_ID")"
POSTTRAIN_RUN_ID="$(smoke_run_id "$POSTTRAIN_RUN_ID")"
validate_run_id "$RECOVERY_ID"; validate_run_id "$POSTTRAIN_RUN_ID"
export PRETRAIN_RUN_DIR
export POSTTRAIN_RUN_DIR="$RUN_ROOT_DIR/$POSTTRAIN_RUN_ID"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_ROOT_DIR/${POSTTRAIN_RUN_ID}_final$( [[ "$SMOKE_TEST" == 0 ]] || printf _smoke )}"
[[ "$EVAL_OUTPUT_DIR" == /* ]] || die 'EVAL_OUTPUT_DIR must be absolute'
[[ "$SMOKE_TEST" == 0 || "$EVAL_OUTPUT_DIR" == *_smoke ]] || die 'Smoke EVAL_OUTPUT_DIR must end with _smoke'
export DRY_RUN EXPERIMENT_ENV
RECOVERY_DIR="$RUN_ROOT_DIR/launcher_logs/$RECOVERY_ID"
printf 'Recovery %s on %s: existing DROID -> LIBERO Plus -> full 10030 eval; dry_run=%s\n' "$RECOVERY_ID" "$(hostname)" "$DRY_RUN"
if [[ "$DRY_RUN" == 0 ]]; then
  exec 8>"/tmp/starvla_droid_pipeline_$(id -u).lock"
  flock -n 8 || die 'Another DROID pipeline is active on this host'
  for path in "$RECOVERY_DIR" "$POSTTRAIN_RUN_DIR" "$EVAL_OUTPUT_DIR"; do
    [[ ! -e "$path" && ! -L "$path" ]] || die "Recovery path already exists: $path"
  done
  mkdir -p "$(dirname "$RECOVERY_DIR")"
  mkdir "$RECOVERY_DIR"
  phase=starting
  trap 'rc=$?; printf "%s\n" "$rc" > "$RECOVERY_DIR/exit_code"; printf "time=%s phase=%s exit_code=%s\n" "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" "$rc" >> "$RECOVERY_DIR/status.txt"' EXIT
  printf 'pretrain=%s\nposttrain=%s\neval=%s\n' "$PRETRAIN_RUN_DIR" "$POSTTRAIN_RUN_DIR" "$EVAL_OUTPUT_DIR" > "$RECOVERY_DIR/paths.txt"
fi
run_step() {
  phase="$1"
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    "$@"
  else
    printf 'time=%s phase=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" >> "$RECOVERY_DIR/status.txt"
    "$@" 2>&1 | tee "$RECOVERY_DIR/$phase.log"
  fi
}
source_args=()
[[ "$SMOKE_TEST" == 0 ]] || source_args+=(--smoke)
if [[ "$DRY_RUN" == 0 ]]; then
  run_step validating_pretrain env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CHECKS" source --stage pretrain --run "$PRETRAIN_RUN_DIR" --steps "$PRETRAIN_STEPS" "${source_args[@]}"
fi
run_step posttraining env RUN_ID="$POSTTRAIN_RUN_ID" bash "$TRAIN_FILES/run_libero_plus_posttrain.sh"
run_step evaluating bash "$TRAIN_FILES/run_libero_plus_eval.sh"
phase=completed
printf 'Recovery finished successfully (dry_run=%s).\n' "$DRY_RUN"
