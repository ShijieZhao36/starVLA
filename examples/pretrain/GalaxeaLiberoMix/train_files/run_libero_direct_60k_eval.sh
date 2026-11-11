#!/usr/bin/env bash
# Direct original-Qwen + seed42 fresh complete action head -> LIBERO 60k -> final eval.
# Same training config/core/evaluation code as the completed paired experiment.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
PYTHON_BIN=/root/miniconda3/envs/CRA/bin/python
EVAL_PYTHON="$PYTHON_BIN"
LIBERO_HOME=/root/workspace/LIBERO-plus
ENV_OVERLAY=/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_pair_validation_20261105/sim_overlay
RUN_ROOT_DIR=/mnt/zhaoshijie/checkpoints/starVLA
REFERENCE_CONTRACT=/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_posttrain60k_20261105_contract
DRY_RUN="${DRY_RUN:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
for flag in DRY_RUN SMOKE_TEST; do [[ "${!flag}" =~ ^[01]$ ]] || die "$flag must be 0 or 1"; done
: "${RUN_ID:?Provide a unique RUN_ID}"
[[ "$RUN_ID" =~ ^libero_plus_direct_ln_8gpu_60k_[A-Za-z0-9_.-]+$ ]] || die 'Invalid direct RUN_ID'
MAX_TRAIN_STEPS=60000 WARMUP_STEPS=5000 SAVE_INTERVAL=10000 LOG_INTERVAL=100
if [[ "$SMOKE_TEST" == 1 ]]; then
  [[ "$RUN_ID" == *_smoke3 ]] || die 'Smoke run ID must end in _smoke3'
  MAX_TRAIN_STEPS=3 WARMUP_STEPS=1 SAVE_INTERVAL=3 LOG_INTERVAL=1
else
  [[ "$RUN_ID" != *_smoke3 ]] || die 'Formal run must not use smoke suffix'
fi
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
EVAL_OUTPUT_DIR="/mnt/zhaoshijie/eval/starVLA/libero_plus/${RUN_ID}_final"
MASTER_PORT=29675
EVAL_BASE_PORT=10383
BASE_VLM=/mnt/zhaoshijie/workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
DATA_ROOT=/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1
FRAME_CACHE=/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
TRAIN_FILES="$(dirname "$SCRIPT_PATH")"
CONFIG_YAML="$TRAIN_FILES/starvla_libero_pair_ln_60k.yaml"
VALIDATOR="$TRAIN_FILES/validate_libero_direct_60k.py"
ACCELERATE_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/accelerate_libero_plus_ln_8gpu.yaml"
DEEPSPEED_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json"
EVAL_LAUNCHER="$REPO_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_libero_plus_canonical_qwen3vl_pi_gr00t_60k_1029_8gpu.sh"
FINAL_CHECKPOINT="$OUTPUT_DIR/final_model/pytorch_model.pt"
EXPECTED_INITIAL=530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89
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
  --trainer.pretrained_checkpoint=null
  --trainer.expected_initial_model_sha256 "$EXPECTED_INITIAL"
  --trainer.reload_modules=null --trainer.is_resume=false --trainer.strict_pretrained_load=true
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" --trainer.num_warmup_steps "$WARMUP_STEPS"
  --trainer.save_interval "$SAVE_INTERVAL" --trainer.eval_interval "$LOG_INTERVAL"
  --trainer.logging_frequency "$LOG_INTERVAL"
  --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID"
)
gate_args=(
  --output "$OUTPUT_DIR" --repo "$REPO_DIR" --config "$CONFIG_YAML" --runner "$SCRIPT_PATH"
  --accelerate "$ACCELERATE_CONFIG" --deepspeed "$DEEPSPEED_CONFIG" --overlay "$ENV_OVERLAY"
  --data "$DATA_ROOT/dataset_absolute" --cache "$FRAME_CACHE" --reference-contract "$REFERENCE_CONTRACT"
  --base-vlm "$BASE_VLM"
)
[[ "$SMOKE_TEST" == 0 ]] || gate_args+=(--smoke)
report_args=()
[[ -z "${CPU_PREFLIGHT_REPORT:-}" ]] || report_args=(--report "$CPU_PREFLIGHT_REPORT")
printf 'Host=%s run=%s dry_run=%s smoke=%s global_batch=64 original_Qwen_fresh_head=true\n' "$(hostname)" "$RUN_ID" "$DRY_RUN" "$SMOKE_TEST"
printf 'Training command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" --dry-run "${report_args[@]}"
  CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$VALIDATOR" eval-protocol --output "$OUTPUT_DIR" --reference-contract "$REFERENCE_CONTRACT" --libero-home "$LIBERO_HOME" --dry-run "${report_args[@]}"
  exit 0
fi
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" --check-only
exec 9>"/tmp/starvla_mix_pair_8gpu_$(id -u).lock"
flock -n 9 || die 'Another pipeline holds host lock'
exec 8>"/tmp/starvla_libero_ln_8gpu_$(id -u).lock"
flock -n 8 || die 'Another LIBERO pipeline holds host lock'
# Inspect hardware state without starting a GPU workload.
"$PYTHON_BIN" - <<'PYGPU'
import json, subprocess, xml.etree.ElementTree as ET
root = ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'], timeout=20))
cards = root.findall('gpu')
assert len(cards) == 8, 'Expected eight GPUs'
for g in cards:
    state = (g.findtext('gpu_recovery_action'), g.findtext('remapped_rows/remapped_row_pending'), g.findtext('remapped_rows/remapped_row_failure'))
    assert state == ('None','No','No'), (g.findtext('uuid'), state)
    assert g.findtext('ecc_errors/volatile/dram_uncorrectable') == '0', 'Uncorrectable DRAM ECC'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], timeout=20).strip(), 'GPU processes active'
print('Eight idle GPUs passed recovery, ECC and row-remap gates')
PYGPU
[[ ! -e "$OUTPUT_DIR" && ! -e "$EVAL_OUTPUT_DIR" ]] || die 'Output exists; refusing duplicate run'
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
git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
git status --short > "$OUTPUT_DIR/launch/git_status.txt"
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch/train_command.txt"
printf '\n' >> "$OUTPUT_DIR/launch/train_command.txt"
printf 'start_time=%s\nhost=%s\npython=%s\ninitialization=original_Qwen_and_fresh_complete_action_head\nsmoke=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$(hostname)" "$PYTHON_BIN" "$SMOKE_TEST" > "$OUTPUT_DIR/launch/experiment.txt"
export NUMBA_CACHE_DIR="$OUTPUT_DIR/runtime_cache/numba"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" "${report_args[@]}"
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$VALIDATOR" eval-protocol --output "$OUTPUT_DIR" --reference-contract "$REFERENCE_CONTRACT" --libero-home "$LIBERO_HOME" "${report_args[@]}"
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" -c 'import mujoco, robosuite, pandas; from libero.libero.envs import OffScreenRenderEnv; print("Simulation CPU imports OK")'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$PYTHON_BIN" -c 'import torch, deepspeed, transformers; assert torch.cuda.device_count() == 8; print("Training imports and 8 GPUs OK")'
phase=training
printf 'time=%s phase=training\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" >> "$OUTPUT_DIR/pipeline_status.txt"
"${train_cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
trained_args=(--output "$OUTPUT_DIR" --data "$DATA_ROOT/dataset_absolute" --reference-contract "$REFERENCE_CONTRACT")
[[ "$SMOKE_TEST" == 0 ]] || trained_args+=(--smoke)
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" trained "${trained_args[@]}"
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$VALIDATOR" eval-protocol --output "$OUTPUT_DIR" --reference-contract "$REFERENCE_CONTRACT" --libero-home "$LIBERO_HOME"
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
printf 'Completed direct LIBERO training and final-checkpoint evaluation: %s\n' "$RUN_ID"
