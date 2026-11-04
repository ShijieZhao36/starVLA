#!/usr/bin/env bash
# One independent arm of the matched Galaxea / Galaxea+LIBERO experiment.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
REPO_DIR="${REPO_DIR:-/mnt/zhaoshijie/workspace/starVLA_galaxea_libero_mix_20261105}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/CRA/bin/python}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
ARM="${ARM:-mixed}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-0}"
[[ "$ARM" == mixed || "$ARM" == control ]] || die 'ARM must be mixed or control'
for flag in SMOKE_TEST DRY_RUN RUN_DOWNSTREAM; do
  [[ "${!flag}" =~ ^[01]$ ]] || die "$flag must be 0 or 1"
done
prefix="galaxea_libero_${ARM}_ln_8gpu_80k"
[[ "$SMOKE_TEST" == 0 ]] || prefix="galaxea_libero_${ARM}_ln_smoke"
RUN_ID="${RUN_ID:-${prefix}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 'Invalid RUN_ID'
MASTER_PORT="${MASTER_PORT:-29615}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] && ((MASTER_PORT > 1024 && MASTER_PORT < 65536)) || die 'Invalid port'
cd "$REPO_DIR"
TRAIN_FILES="$REPO_DIR/examples/pretrain/GalaxeaLiberoMix/train_files"
CONFIG_YAML="$TRAIN_FILES/starvla_${ARM}_ln_80k.yaml"
ACCELERATE_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/accelerate_libero_plus_ln_8gpu.yaml"
DEEPSPEED_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json"
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
MAX_TRAIN_STEPS=80000
WARMUP_STEPS=5000
SAVE_INTERVAL=10000
LOG_INTERVAL=100
if [[ "$SMOKE_TEST" == 1 ]]; then
  MAX_TRAIN_STEPS=3
  WARMUP_STEPS=1
  SAVE_INTERVAL=3
  LOG_INTERVAL=1
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
[[ "$SMOKE_TEST" == 0 ]] || export WANDB_MODE=disabled
train_cmd=(
  "$PYTHON_BIN" -m accelerate.commands.launch
  --config_file "$ACCELERATE_CONFIG" --num_machines 1 --num_processes 8
  --machine_rank 0 --main_process_port "$MASTER_PORT" --max_restarts 0
  --deepspeed_config_file "$DEEPSPEED_CONFIG"
  "$REPO_DIR/starVLA/training/train_starvla.py" --config_yaml "$CONFIG_YAML"
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" --trainer.num_warmup_steps "$WARMUP_STEPS"
  --trainer.save_interval "$SAVE_INTERVAL" --trainer.logging_frequency "$LOG_INTERVAL"
  --trainer.eval_interval "$LOG_INTERVAL"
  --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID"
)
printf 'Host=%s ARM=%s RUN_ID=%s SMOKE=%s global_batch=64\n' "$(hostname)" "$ARM" "$RUN_ID" "$SMOKE_TEST"
printf 'Command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
[[ "$DRY_RUN" == 0 ]] || exit 0
[[ -x "$PYTHON_BIN" && -f "$CONFIG_YAML" ]] || die 'Missing runtime or config'
exec 9>"/tmp/starvla_mix_pair_8gpu_$(id -u).lock"
flock -n 9 || die 'A mixed/control experiment already holds this host lock'
gpu_pids="$(timeout 10 nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "${gpu_pids//[[:space:]]/}" ]] || die 'GPU compute processes already running; refusing overlap'
mkdir -p "$RUN_ROOT_DIR"
mkdir "$OUTPUT_DIR" || die 'Output exists: choose a fresh RUN_ID'
mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb"
phase=preflight
finish() {
  rc=$?
  printf 'time=%s phase=%s exit_code=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" "$rc" >> "$OUTPUT_DIR/pipeline_status.txt"
  printf '%s\n' "$rc" > "$OUTPUT_DIR/exit_code"
}
trap finish EXIT
cp "$(readlink -f "${BASH_SOURCE[0]}")" "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/launch/"
git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
git status --short > "$OUTPUT_DIR/launch/git_status.txt"
git diff --binary HEAD > "$OUTPUT_DIR/launch/tracked_changes.patch"
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch/train_command.txt"
printf '\n' >> "$OUTPUT_DIR/launch/train_command.txt"
printf 'start_time=%s\nhost=%s\narm=%s\nsmoke=%s\nrun_downstream=%s\npython=%s\n' \
  "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$(hostname)" "$ARM" "$SMOKE_TEST" "$RUN_DOWNSTREAM" "$PYTHON_BIN" > "$OUTPUT_DIR/launch/experiment.txt"
"$PYTHON_BIN" -c 'import torch, deepspeed, transformers, accelerate; assert torch.cuda.device_count() == 8; print({"torch":torch.__version__, "transformers":transformers.__version__, "accelerate":accelerate.__version__, "deepspeed":deepspeed.__version__})' | tee "$OUTPUT_DIR/launch/environment.txt"
phase=pretraining
printf 'time=%s phase=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" >> "$OUTPUT_DIR/pipeline_status.txt"
"${train_cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
"$PYTHON_BIN" - "$OUTPUT_DIR" "$MAX_TRAIN_STEPS" "$ARM" <<'PY'
import datetime, json, math, sys
from pathlib import Path
from omegaconf import OmegaConf
p = Path(sys.argv[1])
cfg = OmegaConf.load(p/'config.full.yaml')
ckpt = p/'final_model/pytorch_model.pt'
assert ckpt.stat().st_size > 1_000_000
assert cfg.framework.action_model.decoder_input_norm is True
assert cfg.trainer.pretrained_checkpoint is None
assert cfg.trainer.max_train_steps == int(sys.argv[2])
assert json.loads((p/'dataset_statistics.json').read_text())
metrics = [json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
assert metrics[-1]['step'] == int(sys.argv[2])
assert metrics[-1]['source_exposure/cumulative_examples'] == int(sys.argv[2]) * 64
assert all(math.isfinite(row['action_dit_loss']) for row in metrics)
marker = dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              steps=int(sys.argv[2]), arm=sys.argv[3], checkpoint=str(ckpt), bytes=ckpt.stat().st_size)
(p/'pretrain_success.json').write_text(json.dumps(marker, indent=2)+'\n')
print(marker)
PY
phase=pretraining_completed
printf 'time=%s phase=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" >> "$OUTPUT_DIR/pipeline_status.txt"
if [[ "$RUN_DOWNSTREAM" == 1 ]]; then
  phase=downstream
  PRETRAIN_RUN_DIR="$OUTPUT_DIR" RUN_ID="${RUN_ID}_libero30k" \
    SMOKE_TEST="$SMOKE_TEST" REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
    bash "$TRAIN_FILES/run_libero_from_mix_control_30k_eval.sh" 2>&1 | tee "$OUTPUT_DIR/downstream.log"
fi
phase=completed
