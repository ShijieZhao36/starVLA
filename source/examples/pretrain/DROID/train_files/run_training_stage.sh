#!/usr/bin/env bash
# Internal stage runner. Use run_droid_pretrain.sh or run_libero_plus_posttrain.sh.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGE="${1:?Set pretrain or posttrain}"
[[ "$STAGE" == pretrain || "$STAGE" == posttrain ]] || die 'Unknown training stage'
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
if [[ "$STAGE" == pretrain ]]; then
  CONFIG="$TRAIN_FILES/starvla_droid_pretrain.yaml"
  RUN_ID="${RUN_ID:-droid_pretrain_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
  STEPS="$PRETRAIN_STEPS" WARMUP="$PRETRAIN_WARMUP" BATCH="$PRETRAIN_BATCH" SAVE="$PRETRAIN_SAVE_INTERVAL"
  DATA_ROOT="$(dirname "$DROID_DATASET_DIR")" CACHE="$DROID_FRAME_CACHE"
  MASTER_PORT="$PRETRAIN_MASTER_PORT"
  init_args=(--trainer.pretrained_checkpoint=null --trainer.reload_modules=null --trainer.is_resume=false)
else
  CONFIG="$TRAIN_FILES/starvla_libero_plus_posttrain.yaml"
  : "${PRETRAIN_RUN_DIR:?Set PRETRAIN_RUN_DIR to the completed DROID pretraining run}"
  [[ "$PRETRAIN_RUN_DIR" == /* ]] || die 'PRETRAIN_RUN_DIR must be absolute'
  RUN_ID="${RUN_ID:-droid_libero_posttrain_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
  STEPS="$POSTTRAIN_STEPS" WARMUP="$POSTTRAIN_WARMUP" BATCH="$POSTTRAIN_BATCH" SAVE="$POSTTRAIN_SAVE_INTERVAL"
  DATA_ROOT="$LIBERO_DATA_ROOT" CACHE="$LIBERO_FRAME_CACHE"
  MASTER_PORT="$POSTTRAIN_MASTER_PORT"
  init_args=(--trainer.pretrained_checkpoint "$PRETRAIN_RUN_DIR/final_model/pytorch_model.pt"
             --trainer.reload_modules=null --trainer.is_resume=false --trainer.strict_pretrained_load=true)
fi
RUN_ID="$(smoke_run_id "$RUN_ID")"
validate_run_id "$RUN_ID"
for name in STEPS WARMUP BATCH SAVE MASTER_PORT LOG_INTERVAL; do positive_int "$name" "${!name}"; done
(( WARMUP < STEPS && MASTER_PORT > 1024 && MASTER_PORT <= 65535 )) || die 'Invalid warmup or port'
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
for file in "$CONFIG" "$TRAIN_FILES/accelerate_8gpu.yaml" "$TRAIN_FILES/deepspeed_zero2.json" "$BASE_VLM/config.json"; do
  [[ -f "$file" ]] || die "Missing file: $file"
done
cmd=("$PYTHON_BIN" -m accelerate.commands.launch
  --config_file "$TRAIN_FILES/accelerate_8gpu.yaml" --num_machines 1 --num_processes 8
  --machine_rank 0 --main_process_port "$MASTER_PORT" --max_restarts 0
  --deepspeed_config_file "$TRAIN_FILES/deepspeed_zero2.json"
  "$REPO_DIR/starVLA/training/train_starvla.py" --config_yaml "$CONFIG"
  --framework.qwenvl.base_vlm "$BASE_VLM"
  --datasets.vla_data.data_root_dir "$DATA_ROOT" --datasets.vla_data.frame_cache_dir "$CACHE"
  --datasets.vla_data.per_device_batch_size "$BATCH"
  --trainer.max_train_steps "$STEPS" --trainer.num_warmup_steps "$WARMUP"
  --trainer.save_interval "$SAVE" --trainer.eval_interval "$LOG_INTERVAL" --trainer.logging_frequency "$LOG_INTERVAL"
  --trainer.gradient_accumulation_steps 1 --trainer.save_format pt
  "${init_args[@]}" --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID"
  --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY")
printf 'stage=%s host=%s GPUs=8 batch_per_GPU=%s global_batch=%s steps=%s dry_run=%s\n' \
  "$STAGE" "$(hostname)" "$BATCH" "$((BATCH * 8))" "$STEPS" "$DRY_RUN"
printf 'Output: %s\nCommand: ' "$OUTPUT_DIR"; printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Command preview only; no output directory, statistics, cache or GPU job is created.\n'
  exit 0
fi
[[ ! -e "$OUTPUT_DIR" ]] || die "Output exists; use a new RUN_ID: $OUTPUT_DIR"
mode_args=()
[[ "$SMOKE_TEST" == 0 ]] || mode_args+=(--smoke)
if [[ "$STAGE" == posttrain ]]; then
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CHECKS" source --run "$PRETRAIN_RUN_DIR" --steps "$PRETRAIN_STEPS" "${mode_args[@]}"
fi
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$TRAIN_FILES/check_prepared_data.py" --stage "$STAGE" --dataset "$DATA_ROOT/$( [[ "$STAGE" == pretrain ]] && printf dataset || printf dataset_absolute )" --cache "$CACHE"
command -v flock >/dev/null || die 'flock is required'
exec 9>"/tmp/starvla_droid_8gpu_$(id -u).lock"
flock -n 9 || die 'Another DROID training/evaluation stage is active on this host'
require_eight_idle_gpus
mkdir -p "$RUN_ROOT_DIR"
mkdir "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUTPUT_DIR/stage_exit_code"' EXIT
record_launch "$OUTPUT_DIR"
printf '%q ' "${cmd[@]}" > "$OUTPUT_DIR/launch/train_command.txt"
printf '\n' >> "$OUTPUT_DIR/launch/train_command.txt"
if [[ "$STAGE" == posttrain ]]; then
  cp "$PRETRAIN_RUN_DIR/pretrain_success.json" "$OUTPUT_DIR/launch/source_pretrain_success.json"
fi
set +e
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
statuses=("${PIPESTATUS[@]}")
set -e
rc="${statuses[0]}"
(( rc != 0 || statuses[1] == 0 )) || rc="${statuses[1]}"
printf '%s\n' "$rc" > "$OUTPUT_DIR/train_exit_code"
(( rc == 0 )) || exit "$rc"
check_args=(trained --run "$OUTPUT_DIR" --stage "$STAGE" --steps "$STEPS" --batch "$BATCH" "${mode_args[@]}")
[[ "$STAGE" == pretrain ]] || check_args+=(--source "$PRETRAIN_RUN_DIR")
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CHECKS" "${check_args[@]}"
printf 'Completed and validated %s: %s\n' "$STAGE" "$OUTPUT_DIR"
