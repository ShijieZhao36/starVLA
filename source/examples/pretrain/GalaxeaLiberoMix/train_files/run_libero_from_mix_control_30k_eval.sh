#!/usr/bin/env bash
# Load all weights from a completed paired pretraining run, then LIBERO 30k + eval.
# Preparing this script does not schedule or start it. PRETRAIN_RUN_DIR is required.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }
REPO_DIR="${REPO_DIR:-/mnt/zhaoshijie/workspace/starVLA_galaxea_libero_mix_20261105}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/CRA/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON_BIN}"
LIBERO_HOME="${LIBERO_HOME:-/root/workspace/LIBERO-plus}"
ENV_OVERLAY="${ENV_OVERLAY:-/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_pair_validation_20261105/sim_overlay}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
: "${PRETRAIN_RUN_DIR:?Set PRETRAIN_RUN_DIR to a completed paired pretraining run}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$SMOKE_TEST" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || die "SMOKE_TEST and DRY_RUN must be 0 or 1"
[[ "$PRETRAIN_RUN_DIR" == /* ]] || die "PRETRAIN_RUN_DIR must be absolute"
[[ -x "$PYTHON_BIN" && -x "$EVAL_PYTHON" ]] || die "Missing Python interpreter"
PRETRAIN_RUN_DIR="$(readlink -f "$PRETRAIN_RUN_DIR")"
PRETRAINED_CHECKPOINT="$PRETRAIN_RUN_DIR/final_model/pytorch_model.pt"
# The parent writes this marker only after accelerate has exited successfully.
"$PYTHON_BIN" - "$PRETRAIN_RUN_DIR" "$SMOKE_TEST" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
run = Path(sys.argv[1])
marker = json.loads((run / 'pretrain_success.json').read_text())
cfg = OmegaConf.load(run / 'config.full.yaml')
checkpoint = (run / 'final_model/pytorch_model.pt').resolve()
assert marker['steps'] in (3, 80000), marker
assert sys.argv[2] == '1' or marker['steps'] == 80000, 'Formal posttraining requires completed 80k pretraining'
assert Path(marker['checkpoint']).is_absolute()
assert Path(marker['checkpoint']).resolve() == checkpoint
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
assert checkpoint.stat().st_size == marker['bytes'], 'Checkpoint size changed since success marker'
assert cfg.framework.action_model.decoder_input_norm is True
assert int(cfg.trainer.max_train_steps) == marker['steps']
assert cfg.framework.name == 'QwenPI'
assert cfg.framework.action_model.action_dim == 90
assert cfg.framework.action_model.action_horizon == 8
assert json.loads((run / 'dataset_statistics.json').read_text())
assert isinstance(marker['arm'], str) and marker['arm']
print(f"Validated source arm={marker['arm']} steps={marker['steps']} checkpoint={checkpoint}")
PY

prefix="$(basename "$PRETRAIN_RUN_DIR")_libero_ln_30k"
[[ "$SMOKE_TEST" == 0 ]] || prefix="$(basename "$PRETRAIN_RUN_DIR")_libero_ln_smoke"
RUN_ID="${RUN_ID:-${prefix}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "Invalid RUN_ID"
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
MASTER_PORT="${MASTER_PORT:-29615}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-10183}"
EVAL_SIMS_PER_GPU="${EVAL_SIMS_PER_GPU:-4}"
for name in MASTER_PORT EVAL_BASE_PORT EVAL_SIMS_PER_GPU; do positive_int "$name" "${!name}"; done
(( MASTER_PORT <= 65535 && EVAL_BASE_PORT <= 65527 )) || die "Invalid port range"
cd "$REPO_DIR"
TRAIN_FILES="$REPO_DIR/examples/pretrain/LiberoPlus/train_files"
CONFIG_YAML="$TRAIN_FILES/starvla_libero_plus_direct_ln_30k.yaml"
ACCELERATE_CONFIG="$TRAIN_FILES/accelerate_libero_plus_ln_8gpu.yaml"
DEEPSPEED_CONFIG="$TRAIN_FILES/deepspeed_libero_plus_ln_zero2.json"
BASE_VLM=/mnt/zhaoshijie/workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
DATA_ROOT=/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1
FRAME_CACHE=/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256
EVAL_LAUNCHER="$REPO_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_libero_plus_canonical_qwen3vl_pi_gr00t_60k_1029_8gpu.sh"
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/mnt/zhaoshijie/eval/starVLA/libero_plus/${RUN_ID}_final}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
for path in "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$BASE_VLM/config.json" "$EVAL_LAUNCHER" "$ENV_OVERLAY/../sim_overlay_manifest.json"; do
  [[ -f "$path" ]] || die "Missing file: $path"
done
for path in "$DATA_ROOT/dataset_absolute" "$FRAME_CACHE/dataset_absolute" "$LIBERO_HOME/libero" "$ENV_OVERLAY"; do
  [[ -d "$path" ]] || die "Missing directory: $path"
done
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ENV_OVERLAY:$REPO_DIR:$LIBERO_HOME${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_HOME LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-pretrain}" WANDB_ENTITY="${WANDB_ENTITY:-jasperz-georgia-institute-of-technology}"
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
  --trainer.pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
  --trainer.reload_modules=null --trainer.is_resume=false
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" --trainer.num_warmup_steps "$WARMUP_STEPS"
  --trainer.save_interval "$SAVE_INTERVAL" --trainer.eval_interval "$EVAL_INTERVAL"
  --trainer.logging_frequency "$LOGGING_FREQUENCY" --trainer.gradient_accumulation_steps 1
  --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID"
  --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
)
printf 'Host=%s run_id=%s smoke=%s global_batch=64\nOutput=%s\n' "$(hostname)" "$RUN_ID" "$SMOKE_TEST" "$OUTPUT_DIR"
printf 'Training command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
printf 'After successful training: final_model/pytorch_model.pt -> %s\n' "$EVAL_OUTPUT_DIR"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Dry run: no output directory, GPU process, or evaluation started.\n'
  exit 0
fi

# Reuse the parent's inherited fd9 when called after pretraining; standalone
# calls take the same pair lock themselves. fd8 also excludes older LIBERO jobs.
pair_lock="/tmp/starvla_mix_pair_8gpu_$(id -u).lock"
if [[ "$(readlink "/proc/$$/fd/9" 2>/dev/null || true)" != "$pair_lock" ]]; then
  exec 9>"$pair_lock"
fi
flock -n 9 || die "Another paired eight-GPU pipeline is active on this host"
exec 8>"/tmp/starvla_libero_ln_8gpu_$(id -u).lock"
flock -n 8 || die "Another LIBERO LN pipeline is active on this host"
gpu_jobs="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${gpu_jobs//[[:space:]]/}" ]] || die "GPU compute processes are active: $gpu_jobs"
[[ ! -e "$EVAL_OUTPUT_DIR" && ! -e "$OUTPUT_DIR" ]] || die "Output exists; choose fresh RUN_ID/EVAL_OUTPUT_DIR"
mkdir -p "$RUN_ROOT_DIR"
mkdir "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb" "$OUTPUT_DIR/runtime_cache"
export NUMBA_CACHE_DIR="$OUTPUT_DIR/runtime_cache/numba"
cp "$SCRIPT_PATH" "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/launch/"
cp "$PRETRAIN_RUN_DIR/pretrain_success.json" "$OUTPUT_DIR/launch/source_pretrain_success.json"
cp "$PRETRAIN_RUN_DIR/config.full.yaml" "$OUTPUT_DIR/launch/source_config.full.yaml"
cp "$PRETRAIN_RUN_DIR/dataset_statistics.json" "$OUTPUT_DIR/launch/source_dataset_statistics.json"
cp "$ENV_OVERLAY/../sim_overlay_manifest.json" "$OUTPUT_DIR/launch/"
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
printf 'start_time=%s\nrun_id=%s\npython=%s\neval_python=%s\nenv_overlay=%s\nsource_run=%s\nsmoke=%s\n' \
  "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$RUN_ID" "$PYTHON_BIN" "$EVAL_PYTHON" "$ENV_OVERLAY" "$PRETRAIN_RUN_DIR" "$SMOKE_TEST" > "$OUTPUT_DIR/launch/experiment.txt"
"$PYTHON_BIN" -c 'import torch, deepspeed, transformers; assert torch.cuda.device_count() == 8; print("Training imports and 8 GPUs OK")'
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" -c 'import mujoco, robosuite, pandas; from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; print("CPU simulation imports OK")'
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$REPO_DIR/examples/pretrain/LiberoPlus/eval_files/eval_libero_model.py" --help > "$OUTPUT_DIR/launch/eval_cli_help.txt"
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$REPO_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_queue_eval.py" --help > "$OUTPUT_DIR/launch/queue_cli_help.txt"
"$PYTHON_BIN" - "$OUTPUT_DIR" "$DATA_ROOT/dataset_absolute" <<'PY'
import hashlib,json,sys
from pathlib import Path
run,data=map(Path,sys.argv[1:])
info=json.loads((data/'meta/info.json').read_text())
assert info['splits']=={'train':'0:14347'}
paths=[data/'meta/info.json',data/'conversion_report.json',data/'meta/tasks.jsonl',data/'meta/stats.json',data/'meta/relative_stats.json']
report={'train_root':str(data),'splits':info['splits'],'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(run/'launch/training_data_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
PY
phase=training
printf 'time=%s phase=training\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" >> "$OUTPUT_DIR/pipeline_status.txt"
"${train_cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
FINAL_CHECKPOINT="$OUTPUT_DIR/final_model/pytorch_model.pt"
[[ -s "$FINAL_CHECKPOINT" ]] || die "Training returned success without final checkpoint"
"$PYTHON_BIN" - "$OUTPUT_DIR" "$PRETRAINED_CHECKPOINT" "$MAX_TRAIN_STEPS" <<'PY'
import json,sys
from pathlib import Path
from omegaconf import OmegaConf
run=Path(sys.argv[1]);cfg=OmegaConf.load(run/'config.full.yaml')
assert cfg.framework.action_model.decoder_input_norm is True
assert Path(cfg.trainer.pretrained_checkpoint).resolve()==Path(sys.argv[2]).resolve()
assert cfg.trainer.is_resume is False and cfg.trainer.reload_modules is None
assert cfg.trainer.max_train_steps==int(sys.argv[3])
assert cfg.datasets.vla_data.data_mix=='libero_plus_stage1'
assert json.loads((run/'dataset_statistics.json').read_text())
(run/'posttrain_success.json').write_text(json.dumps({'steps':int(sys.argv[3]),'checkpoint':str(run/'final_model/pytorch_model.pt'),'bytes':(run/'final_model/pytorch_model.pt').stat().st_size,'source_checkpoint':sys.argv[2]},indent=2)+'\n')
PY
mkdir -p "$(dirname "$EVAL_OUTPUT_DIR")"
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
import json,sys
results=json.load(open(sys.argv[1]))
assert sum(v['total_count'] for v in results.values())==1,results
print('Smoke evaluation: exactly one episode completed')
PY
fi
phase=completed
printf 'Completed training and evaluation: %s\n' "$RUN_ID"
