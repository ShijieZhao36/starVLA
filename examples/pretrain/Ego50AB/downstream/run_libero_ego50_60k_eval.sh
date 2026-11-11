#!/usr/bin/env bash
# One Ego A/B arm: own successful pretrain80k -> LIBERO60k -> one final10030 eval.
# Invoke separately inside screen on each host. DRY_RUN defaults to 1.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }
: "${ARM:?Set ARM to A or B}"
: "${PRETRAIN_RUN_DIR:?Set the completed 80k run directory for this arm}"
REQUIRE_PAIRED_SOURCE="${REQUIRE_PAIRED_SOURCE:-0}"
[[ "$REQUIRE_PAIRED_SOURCE" =~ ^[01]$ ]] || die 'REQUIRE_PAIRED_SOURCE must be 0 or 1'
if [[ "$REQUIRE_PAIRED_SOURCE" == 1 ]]; then
  : "${PAIRED_PRETRAIN_RUN_DIR:?Set the completed 80k run directory for the other arm}"
fi
[[ "$ARM" == A || "$ARM" == B ]] || die 'ARM must be A or B'
REPO_DIR="${REPO_DIR:-/mnt/zhaoshijie/workspace/starVLA_ego50_ab_20261110_0237}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/CRA/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON_BIN}"
LIBERO_HOME="${LIBERO_HOME:-/root/workspace/LIBERO-plus}"
ENV_OVERLAY="${ENV_OVERLAY:-/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_pair_validation_20261105/sim_overlay}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
PAIR_PROTOCOL_DIR="${PAIR_PROTOCOL_DIR:-$RUN_ROOT_DIR/launcher_logs/ego50_ab_posttrain60k_20261110_0237}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
for flag in DRY_RUN SMOKE_TEST; do [[ "${!flag}" =~ ^[01]$ ]] || die "$flag must be 0 or 1"; done
for path in "$REPO_DIR" "$PRETRAIN_RUN_DIR" "$RUN_ROOT_DIR" "$PAIR_PROTOCOL_DIR"; do
  [[ "$path" == /* ]] || die "Path must be absolute: $path"
done
[[ -x "$PYTHON_BIN" && -x "$EVAL_PYTHON" ]] || die 'Missing Python interpreter'
PRETRAIN_RUN_DIR="$(readlink -m "$PRETRAIN_RUN_DIR")"
PRETRAIN_LAUNCHER_DIR="${PRETRAIN_LAUNCHER_DIR:-$(dirname "$PRETRAIN_RUN_DIR")/launcher_logs/$(basename "$PRETRAIN_RUN_DIR")}"
# Independent starts must not inspect the other source, even through symlinks.
if [[ "$REQUIRE_PAIRED_SOURCE" == 1 ]]; then
  [[ "$PAIRED_PRETRAIN_RUN_DIR" == /* ]] || die 'Peer path must be absolute'
  PAIRED_PRETRAIN_RUN_DIR="$(readlink -m "$PAIRED_PRETRAIN_RUN_DIR")"
  PAIRED_PRETRAIN_LAUNCHER_DIR="${PAIRED_PRETRAIN_LAUNCHER_DIR:-$(dirname "$PAIRED_PRETRAIN_RUN_DIR")/launcher_logs/$(basename "$PAIRED_PRETRAIN_RUN_DIR")}"
fi
prefix="libero_plus_${ARM}_ln_8gpu_60k"
MAX_TRAIN_STEPS=60000 WARMUP_STEPS=5000 SAVE_INTERVAL=10000 LOG_INTERVAL=100
if [[ "$SMOKE_TEST" == 1 ]]; then
  prefix="libero_plus_${ARM}_ln_smoke3"
  MAX_TRAIN_STEPS=3 WARMUP_STEPS=1 SAVE_INTERVAL=3 LOG_INTERVAL=1
  PAIR_PROTOCOL_DIR="${PAIR_PROTOCOL_DIR}_smoke3"
fi
RUN_ID="${RUN_ID:-${prefix}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 'Invalid RUN_ID'
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/mnt/zhaoshijie/eval/starVLA/libero_plus/${RUN_ID}_final}"
[[ "$EVAL_OUTPUT_DIR" == /* ]] || die 'EVAL_OUTPUT_DIR must be absolute'
MASTER_PORT="${MASTER_PORT:-29665}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-10283}"
for name in MASTER_PORT EVAL_BASE_PORT; do positive_int "$name" "${!name}"; done
(( MASTER_PORT > 1024 && MASTER_PORT <= 65535 && EVAL_BASE_PORT > 1024 && EVAL_BASE_PORT <= 65527 )) || die 'Invalid port range'
# Same physical/data/model protocol on both hosts; only paths and init checkpoint vary.
BASE_VLM=/mnt/zhaoshijie/workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
DATA_ROOT=/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1
FRAME_CACHE=/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
TRAIN_FILES="$(dirname "$SCRIPT_PATH")"
CONFIG_YAML="$TRAIN_FILES/starvla_libero_ego50_ln_60k.yaml"
VALIDATOR="$TRAIN_FILES/validate_libero_ego50_60k.py"
ACCELERATE_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/accelerate_libero_plus_ln_8gpu.yaml"
DEEPSPEED_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json"
EVAL_LAUNCHER="$REPO_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_libero_plus_canonical_qwen3vl_pi_gr00t_60k_1029_8gpu.sh"
PRETRAINED_CHECKPOINT="$PRETRAIN_RUN_DIR/final_model/pytorch_model.pt"
FINAL_CHECKPOINT="$OUTPUT_DIR/final_model/pytorch_model.pt"
for path in "$CONFIG_YAML" "$VALIDATOR" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$BASE_VLM/config.json" "$EVAL_LAUNCHER" "$ENV_OVERLAY/../sim_overlay_manifest.json"; do
  [[ -f "$path" ]] || die "Missing file: $path"
done
for path in "$DATA_ROOT/dataset_absolute" "$FRAME_CACHE/dataset_absolute" "$LIBERO_HOME/libero" "$ENV_OVERLAY"; do
  [[ -d "$path" ]] || die "Missing directory: $path"
done
cd "$REPO_DIR"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
# Avoid inheriting another checkout or package overlay from the calling shell.
export PYTHONPATH="$ENV_OVERLAY:$REPO_DIR:$LIBERO_HOME"
export LIBERO_HOME LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_PROJECT=pretrain WANDB_ENTITY=jasperz-georgia-institute-of-technology
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
train_cmd=(
  "$PYTHON_BIN" -m accelerate.commands.launch
  --config_file "$ACCELERATE_CONFIG" --num_machines 1 --num_processes 8
  --machine_rank 0 --main_process_port "$MASTER_PORT" --max_restarts 0
  --deepspeed_config_file "$DEEPSPEED_CONFIG"
  "$REPO_DIR/starVLA/training/train_starvla.py" --config_yaml "$CONFIG_YAML"
  --framework.qwenvl.base_vlm "$BASE_VLM" --seed 42
  --trainer.pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
  --trainer.reload_modules=null --trainer.is_resume=false --trainer.strict_pretrained_load=true
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" --trainer.num_warmup_steps "$WARMUP_STEPS"
  --trainer.save_interval "$SAVE_INTERVAL" --trainer.eval_interval "$LOG_INTERVAL"
  --trainer.logging_frequency "$LOG_INTERVAL"
  --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID"
)
gate_args=(
  --arm "$ARM" --source "$PRETRAIN_RUN_DIR" --source-launcher "$PRETRAIN_LAUNCHER_DIR"
  --output "$OUTPUT_DIR" --repo "$REPO_DIR" --config "$CONFIG_YAML" --runner "$SCRIPT_PATH"
  --accelerate "$ACCELERATE_CONFIG" --deepspeed "$DEEPSPEED_CONFIG" --overlay "$ENV_OVERLAY"
  --data "$DATA_ROOT/dataset_absolute" --cache "$FRAME_CACHE" --contract "$PAIR_PROTOCOL_DIR"
)
if [[ "$REQUIRE_PAIRED_SOURCE" == 1 ]]; then
  gate_args+=(--peer "$PAIRED_PRETRAIN_RUN_DIR" --peer-launcher "$PAIRED_PRETRAIN_LAUNCHER_DIR")
else
  gate_args+=(--independent-start)
fi
[[ "$SMOKE_TEST" == 0 ]] || gate_args+=(--smoke)
report_args=()
[[ -z "${CPU_PREFLIGHT_REPORT:-}" ]] || report_args=(--report "$CPU_PREFLIGHT_REPORT")
printf 'Host=%s arm=%s run=%s dry_run=%s smoke=%s require_paired_source=%s global_batch=64\n' "$(hostname)" "$ARM" "$RUN_ID" "$DRY_RUN" "$SMOKE_TEST" "$REQUIRE_PAIRED_SOURCE"
printf 'Source=%s\nOutput=%s\nEval=%s\n' "$PRETRAIN_RUN_DIR" "$OUTPUT_DIR" "$EVAL_OUTPUT_DIR"
if [[ "$REQUIRE_PAIRED_SOURCE" == 1 ]]; then printf 'Peer=%s\n' "$PAIRED_PRETRAIN_RUN_DIR"; fi
printf 'Training command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
printf 'Evaluation: fixed final-%s checkpoint, 8 GPUs x 4 sims, env seed 7, 10030 episodes (smoke: 1).\n' "$MAX_TRAIN_STEPS"
if [[ "$DRY_RUN" == 1 ]]; then
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" --dry-run "${report_args[@]}"
  CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$VALIDATOR" eval-protocol \
    --output "$OUTPUT_DIR" --contract "$PAIR_PROTOCOL_DIR" --libero-home "$LIBERO_HOME" --dry-run "${report_args[@]}"
  printf 'Dry run complete: no training output directory or GPU process was created.\n'
  exit 0
fi
# Strict source validation before acquiring GPUs or creating this run's output.
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" --check-only
exec 9>"/tmp/starvla_mix_pair_8gpu_$(id -u).lock"
flock -n 9 || die 'A pretraining/paired pipeline still holds the host lock'
exec 8>"/tmp/starvla_libero_ln_8gpu_$(id -u).lock"
flock -n 8 || die 'Another LIBERO LN pipeline holds the host lock'
gpu_jobs="$(timeout 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${gpu_jobs//[[:space:]]/}" ]] || die "GPU compute processes are active: $gpu_jobs"
[[ ! -e "$OUTPUT_DIR" && ! -e "$EVAL_OUTPUT_DIR" ]] || die 'Output exists; choose a fresh run ID and evaluation directory'
mkdir -p "$RUN_ROOT_DIR"
mkdir "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb" "$OUTPUT_DIR/runtime_cache"
phase=preflight
finish() {
  rc=$?
  printf '%s\n' "$rc" > "$OUTPUT_DIR/exit_code"
  printf 'time=%s phase=%s exit_code=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" "$rc" >> "$OUTPUT_DIR/pipeline_status.txt"
}
trap finish EXIT
cp "$SCRIPT_PATH" "$CONFIG_YAML" "$VALIDATOR" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/launch/"
cp "$ENV_OVERLAY/../sim_overlay_manifest.json" "$OUTPUT_DIR/launch/"
for source_file in pretrain_success.json config.full.yaml dataset_statistics.json exit_code; do
  cp "$PRETRAIN_RUN_DIR/$source_file" "$OUTPUT_DIR/launch/source_${source_file}"
done
git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
git status --short > "$OUTPUT_DIR/launch/git_status.txt"
git diff --binary HEAD > "$OUTPUT_DIR/launch/tracked_changes.patch"
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch/train_command.txt"
printf '\n' >> "$OUTPUT_DIR/launch/train_command.txt"
printf 'start_time=%s\nhost=%s\narm=%s\npython=%s\neval_python=%s\nsource=%s\nrequire_paired_source=%s\nsmoke=%s\n' \
  "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$(hostname)" "$ARM" "$PYTHON_BIN" "$EVAL_PYTHON" "$PRETRAIN_RUN_DIR" "$REQUIRE_PAIRED_SOURCE" "$SMOKE_TEST" > "$OUTPUT_DIR/launch/experiment.txt"
if [[ "$REQUIRE_PAIRED_SOURCE" == 1 ]]; then printf 'peer=%s\n' "$PAIRED_PRETRAIN_RUN_DIR" >> "$OUTPUT_DIR/launch/experiment.txt"; fi
export NUMBA_CACHE_DIR="$OUTPUT_DIR/runtime_cache/numba"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" "${report_args[@]}"
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$VALIDATOR" eval-protocol \
  --output "$OUTPUT_DIR" --contract "$PAIR_PROTOCOL_DIR" --libero-home "$LIBERO_HOME" "${report_args[@]}"
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" -c 'import mujoco, robosuite, pandas; from libero.libero.envs import OffScreenRenderEnv; print("Simulation CPU imports OK")'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$PYTHON_BIN" -c 'import torch, deepspeed, transformers; assert torch.cuda.device_count() == 8; print("Training imports and 8 GPUs OK")'
phase=training
printf 'time=%s phase=training\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" >> "$OUTPUT_DIR/pipeline_status.txt"
"${train_cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
trained_args=(--arm "$ARM" --source "$PRETRAIN_RUN_DIR" --output "$OUTPUT_DIR" --data "$DATA_ROOT/dataset_absolute" --contract "$PAIR_PROTOCOL_DIR")
[[ "$SMOKE_TEST" == 0 ]] || trained_args+=(--smoke)
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" trained "${trained_args[@]}"
# Recheck the frozen simulation inputs after the long training run.
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$VALIDATOR" eval-protocol \
  --output "$OUTPUT_DIR" --contract "$PAIR_PROTOCOL_DIR" --libero-home "$LIBERO_HOME"
[[ ! -e "$EVAL_OUTPUT_DIR" ]] || die 'Evaluation output appeared while training; refusing overwrite'
mkdir -p "$(dirname "$EVAL_OUTPUT_DIR")"
mkdir "$EVAL_OUTPUT_DIR"
printf 'checkpoint=%s\noutput_dir=%s\nsteps=%s\n' "$FINAL_CHECKPOINT" "$EVAL_OUTPUT_DIR" "$MAX_TRAIN_STEPS" > "$OUTPUT_DIR/posttrain_eval_info.txt"
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
    export NUM_GPUS=8 SIMS_PER_GPU=4 NUM_TRIALS_PER_TASK=1
    export BASE_PORT="$EVAL_BASE_PORT" LAUNCH_STAGGER_SEC=0
    bash "$EVAL_LAUNCHER"
  fi
) 2>&1 | tee "$EVAL_OUTPUT_DIR/driver.log"
phase=validating_evaluation
if [[ "$SMOKE_TEST" == 0 ]]; then
  # Reuse the existing verifier while keeping benchmark task-order prints out
  # of the machine-readable JSON report.
  CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" - "$REPO_DIR" "$EVAL_OUTPUT_DIR" <<'PY'
import contextlib, importlib.util, io, json, sys
from pathlib import Path
repo, out = map(Path, sys.argv[1:])
path = repo / 'examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py'
spec = importlib.util.spec_from_file_location('paired_result_verifier', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with contextlib.redirect_stdout(io.StringIO()):
    expected = module.expected_episode_keys(1)
records = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines() if line.strip()]
report = module.validate_records(records, json.loads((out / 'summary.json').read_text()), expected)
encoded = json.dumps(report, indent=2, sort_keys=True) + '\n'
(out / 'completion_validation.json').write_text(encoded)
print(encoded, end='')
PY
fi
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" - "$OUTPUT_DIR" "$EVAL_OUTPUT_DIR" "$SMOKE_TEST" <<'PY'
import json, sys
from pathlib import Path
run, out = map(Path, sys.argv[1:3])
if sys.argv[3] == '1':
    records = json.loads((out / 'logs/libero_goal/0_to_1.json').read_text())
    assert sum(item['total_count'] for item in records.values()) == 1
    report = {'validated': True, 'episode_count': 1, 'smoke': True}
else:
    report = json.loads((out / 'completion_validation.json').read_text())
    assert report['validated'] is True and report['episode_count'] == 10030
    assert report['suite_episode_counts'] == {'libero_spatial': 2402, 'libero_object': 2518, 'libero_goal': 2591, 'libero_10': 2519}
    assert report['execution_errors'] == 0
report.update(checkpoint=str(run / 'final_model/pytorch_model.pt'), output_dir=str(out))
(run / 'eval_success.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
(out / 'exit_code').write_text('0\n')
PY
phase=completed
printf 'Completed paired LIBERO training and final-checkpoint evaluation: %s\n' "$RUN_ID"
