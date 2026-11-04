#!/usr/bin/env bash
# Direct LIBERO Plus baseline: original Qwen3-VL + fresh complete action head.
# Runs 30k updates, then evaluates THIS run's final checkpoint. Run inside screen.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }

REPO_DIR="${REPO_DIR:-/mnt/zhaoshijie/workspace/starVLA}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/MRA/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON_BIN}"
LIBERO_HOME="${LIBERO_HOME:-/root/workspace/LIBERO-plus}"
ENV_OVERLAY="${ENV_OVERLAY:-/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/libero_ln_bj5_env/sim_overlay}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$SMOKE_TEST" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || die "SMOKE_TEST and DRY_RUN must be 0 or 1"
prefix=libero_plus_direct_ln_8gpu_30k
[[ "$SMOKE_TEST" == 0 ]] || prefix=libero_plus_direct_ln_smoke
RUN_ID="${RUN_ID:-${prefix}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "Invalid RUN_ID"
[[ -z "${PRETRAINED_CHECKPOINT:-}" ]] || die "Direct baseline requires PRETRAINED_CHECKPOINT to be empty"
MAX_TRAIN_STEPS=30000
WARMUP_STEPS=5000
SAVE_INTERVAL=5000
EVAL_INTERVAL=100
LOGGING_FREQUENCY=100
if [[ "$SMOKE_TEST" == 1 ]]; then
  MAX_TRAIN_STEPS=3
  WARMUP_STEPS=1
  SAVE_INTERVAL=3
  EVAL_INTERVAL=1
  LOGGING_FREQUENCY=1
fi
MASTER_PORT="${MASTER_PORT:-29605}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-10083}"
EVAL_SIMS_PER_GPU="${EVAL_SIMS_PER_GPU:-4}"
for name in MASTER_PORT EVAL_BASE_PORT EVAL_SIMS_PER_GPU; do positive_int "$name" "${!name}"; done
(( MASTER_PORT <= 65535 && EVAL_BASE_PORT <= 65527 )) || die "Invalid port range"
[[ -x "$PYTHON_BIN" && -x "$EVAL_PYTHON" ]] || die "Missing Python interpreter"
cd "$REPO_DIR"
TRAIN_FILES="$REPO_DIR/examples/pretrain/LiberoPlus/train_files"
CONFIG_YAML="$TRAIN_FILES/starvla_libero_plus_direct_ln_30k.yaml"
ACCELERATE_CONFIG="$TRAIN_FILES/accelerate_libero_plus_ln_8gpu.yaml"
DEEPSPEED_CONFIG="$TRAIN_FILES/deepspeed_libero_plus_ln_zero2.json"
BASE_VLM="$REPO_DIR/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
DATA_ROOT=/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1
FRAME_CACHE=/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256
EVAL_LAUNCHER="$REPO_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_libero_plus_canonical_qwen3vl_pi_gr00t_60k_1029_8gpu.sh"
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/mnt/zhaoshijie/eval/starVLA/libero_plus/${RUN_ID}_final}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
for path in "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$BASE_VLM/config.json" "$EVAL_LAUNCHER"; do
  [[ -f "$path" ]] || die "Missing file: $path"
done
for path in "$DATA_ROOT/dataset_absolute" "$FRAME_CACHE/dataset_absolute" "$LIBERO_HOME/libero"; do
  [[ -d "$path" ]] || die "Missing directory: $path"
done
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ENV_OVERLAY:$REPO_DIR:$LIBERO_HOME${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_HOME LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-pretrain}"
export WANDB_ENTITY="${WANDB_ENTITY:-jasperz-georgia-institute-of-technology}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1

train_cmd=(
  "$PYTHON_BIN" -m accelerate.commands.launch
  --config_file "$ACCELERATE_CONFIG" --num_machines 1 --num_processes 8
  --machine_rank 0 --main_process_port "$MASTER_PORT" --max_restarts 0
  --deepspeed_config_file "$DEEPSPEED_CONFIG"
  "$REPO_DIR/starVLA/training/train_starvla.py" --config_yaml "$CONFIG_YAML"
  --framework.qwenvl.base_vlm "$BASE_VLM"
  --framework.action_model.decoder_input_norm true
  --datasets.vla_data.data_root_dir "$DATA_ROOT"
  --datasets.vla_data.frame_cache_dir "$FRAME_CACHE"
  --datasets.vla_data.per_device_batch_size 8
  --trainer.pretrained_checkpoint=null --trainer.is_resume=false
  --trainer.max_train_steps "$MAX_TRAIN_STEPS"
  --trainer.num_warmup_steps "$WARMUP_STEPS"
  --trainer.save_interval "$SAVE_INTERVAL" --trainer.eval_interval "$EVAL_INTERVAL"
  --trainer.logging_frequency "$LOGGING_FREQUENCY"
  --trainer.gradient_accumulation_steps 1
  --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID"
  --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
)
printf 'Host=%s run_id=%s smoke=%s global_batch=64\nOutput=%s\n' "$(hostname)" "$RUN_ID" "$SMOKE_TEST" "$OUTPUT_DIR"
printf 'Training command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
printf 'After successful training: final_model/pytorch_model.pt -> %s\n' "$EVAL_OUTPUT_DIR"
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi

# Hold the host-local lock through both training and evaluation.
exec 9>"/tmp/starvla_libero_ln_8gpu_$(id -u).lock"
flock -n 9 || die "Another LIBERO LN pipeline is active on this host"
[[ ! -e "$EVAL_OUTPUT_DIR" ]] || die "Evaluation directory exists: $EVAL_OUTPUT_DIR"
mkdir -p "$RUN_ROOT_DIR"
mkdir "$OUTPUT_DIR" || die "Output exists; choose a fresh RUN_ID"
mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb"
cp "$SCRIPT_PATH" "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/launch/"
git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
git status --short > "$OUTPUT_DIR/launch/git_status.txt"
git diff --binary HEAD > "$OUTPUT_DIR/launch/tracked_changes.patch"
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch/train_command.txt"
printf '\n' >> "$OUTPUT_DIR/launch/train_command.txt"
phase=preflight
finish() {
  rc=$?
  printf 'time=%s phase=%s exit_code=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" "$rc" >> "$OUTPUT_DIR/pipeline_status.txt"
}
trap finish EXIT
printf 'start_time=%s\nrun_id=%s\npython=%s\neval_python=%s\nenv_overlay=%s\nwandb_mode=%s\nsmoke=%s\n' \
  "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$RUN_ID" "$PYTHON_BIN" "$EVAL_PYTHON" "$ENV_OVERLAY" "$WANDB_MODE" "$SMOKE_TEST" > "$OUTPUT_DIR/launch/experiment.txt"

# Fail before committing to a long run if either environment cannot import.
"$PYTHON_BIN" -c 'import torch, deepspeed, transformers; assert torch.cuda.device_count() == 8; print("Training imports and 8 GPUs OK (Qwen3 uses SDPA)")'
"$EVAL_PYTHON" -c 'import mujoco, robosuite, pandas; from libero.libero import benchmark; print("Simulation imports OK")'
phase=training
printf 'time=%s phase=training\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" >> "$OUTPUT_DIR/pipeline_status.txt"
"${train_cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"

# Successful accelerate exit occurs after final saving and the all-rank barrier.
FINAL_CHECKPOINT="$OUTPUT_DIR/final_model/pytorch_model.pt"
[[ -s "$FINAL_CHECKPOINT" ]] || die "Training returned success without final checkpoint"
"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
run = Path(sys.argv[1])
cfg = OmegaConf.load(run / "config.yaml")
assert cfg.framework.action_model.decoder_input_norm is True
assert cfg.trainer.pretrained_checkpoint is None
assert cfg.trainer.max_train_steps in (3, 30000)
assert json.loads((run / "dataset_statistics.json").read_text())
print("Final checkpoint metadata: decoder LayerNorm enabled; LIBERO statistics present")
PY
mkdir "$EVAL_OUTPUT_DIR"
printf 'checkpoint=%s\noutput_dir=%s\nsmoke=%s\n' "$FINAL_CHECKPOINT" "$EVAL_OUTPUT_DIR" "$SMOKE_TEST" > "$OUTPUT_DIR/posttrain_eval_info.txt"
phase=evaluating
printf 'time=%s phase=evaluating checkpoint=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$FINAL_CHECKPOINT" >> "$OUTPUT_DIR/pipeline_status.txt"
(
  unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
  if [[ "$SMOKE_TEST" == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=0
    "$EVAL_PYTHON" "$REPO_DIR/examples/pretrain/LiberoPlus/eval_files/eval_libero_model.py" \
      --pretrained_path "$FINAL_CHECKPOINT" --output_dir "$EVAL_OUTPUT_DIR" \
      --task_suite_name libero_goal --start_idx 0 --end_idx 1 \
      --num_trials_per_task 1 --use_bf16 true --use_canonical_forward true
  else
    export STARVLA_DIR="$REPO_DIR" LIBERO_PYTHON="$EVAL_PYTHON"
    export your_ckpt="$FINAL_CHECKPOINT" output_dir="$EVAL_OUTPUT_DIR"
    export USE_CANONICAL_FORWARD=true GRIPPER_OPEN_SIGN=positive USE_BF16=1
    export NUM_GPUS=8 SIMS_PER_GPU="$EVAL_SIMS_PER_GPU" NUM_TRIALS_PER_TASK=1
    export BASE_PORT="$EVAL_BASE_PORT" LAUNCH_STAGGER_SEC=0
    bash "$EVAL_LAUNCHER"
  fi
) 2>&1 | tee "$EVAL_OUTPUT_DIR/driver.log"
if [[ "$SMOKE_TEST" == 0 ]]; then
  "$EVAL_PYTHON" "$REPO_DIR/examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py" \
    "$EVAL_OUTPUT_DIR" --num-trials-per-task 1 | tee -a "$EVAL_OUTPUT_DIR/driver.log"
else
  "$EVAL_PYTHON" - "$EVAL_OUTPUT_DIR/logs/libero_goal/0_to_1.json" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))
assert sum(v['total_count'] for v in results.values()) == 1, results
print('Smoke evaluation: exactly one episode completed')
PY
fi
phase=completed
printf 'Completed training and evaluation: %s\n' "$RUN_ID"
