#!/usr/bin/env bash
# Run this on BOTH hosts, with NODE_RANK=0 on neo-bj1 and NODE_RANK=1 on neo-bj4.
# Generate RUN_ID once, then pass the SAME value and MASTER_ADDR/PORT to both.
# The caller owns nohup/tmux and logs. Put launcher logs outside OUTPUT_DIR:
# rank 0 requires OUTPUT_DIR not to exist, to prevent overwriting a previous run.
set -Eeuo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }

REPO_DIR="${REPO_DIR:-/mnt/zhaoshijie/workspace/starVLA}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/CRA/bin/python}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
: "${NODE_RANK:?Set NODE_RANK=0 on neo-bj1 or NODE_RANK=1 on neo-bj4}"
MASTER_ADDR="${MASTER_ADDR:-172.27.6.128}"
: "${RUN_ID:?Set one timestamped RUN_ID and reuse it on both hosts}"
MASTER_PORT="${MASTER_PORT:-29604}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
WARMUP_STEPS="${WARMUP_STEPS:-2500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || die "NODE_RANK must be 0 or 1"
[[ "$MASTER_ADDR" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Use an IPv4 address or hostname for MASTER_ADDR"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "RUN_ID may contain letters, numbers, dot, underscore and hyphen"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ -z "${PRETRAINED_CHECKPOINT:-}" ]] || die "This experiment requires a fresh action head; PRETRAINED_CHECKPOINT must be empty"
for name in MASTER_PORT MAX_TRAIN_STEPS WARMUP_STEPS SAVE_INTERVAL EVAL_INTERVAL LOGGING_FREQUENCY; do
  positive_int "$name" "${!name}"
done
(( MASTER_PORT <= 65535 )) || die "MASTER_PORT must be <= 65535"
(( WARMUP_STEPS < MAX_TRAIN_STEPS )) || die "WARMUP_STEPS must be less than MAX_TRAIN_STEPS (override it for a short smoke run)"
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
[[ -d "$REPO_DIR/.git" || -f "$REPO_DIR/.git" ]] || die "Not a Git checkout: $REPO_DIR"
cd "$REPO_DIR"

TRAIN_FILES="$REPO_DIR/examples/pretrain/Xperience/train_files"
CONFIG_YAML="$TRAIN_FILES/starvla_xperience_hours100_ln_16gpu_30k.yaml"
ACCELERATE_CONFIG="$TRAIN_FILES/accelerate_xperience_2nodes_zero2.yaml"
DEEPSPEED_CONFIG="$TRAIN_FILES/deepspeed_xperience_zero2.json"
BASE_VLM="$REPO_DIR/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
DATA_ROOT=/mnt/zhaoshijie/dataset/xperience_runs/stage1
FRAME_CACHE=/mnt/zhaoshijie/frame_cache/xperience_hours100_256
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
for path in "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$BASE_VLM/config.json" "$REPO_DIR/starVLA/training/train_starvla.py"; do
  [[ -f "$path" ]] || die "Required file missing: $path"
done
[[ -d "$DATA_ROOT/hours100" ]] || die "Dataset missing: $DATA_ROOT/hours100"
[[ -d "$FRAME_CACHE" ]] || die "Frame cache missing: $FRAME_CACHE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
(( ${#visible_gpus[@]} == 8 )) || die "Exactly 8 visible GPUs are required on each host"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-=com0_0,com0_1,com0_2,com0_3,com0_4,com0_5,com0_6,com0_7}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT="${WANDB_PROJECT:-pretrain}"
export WANDB_ENTITY="${WANDB_ENTITY:-jasperz-georgia-institute-of-technology}"

cmd=(
  "$PYTHON_BIN" -m accelerate.commands.launch
  --config_file "$ACCELERATE_CONFIG"
  --num_machines 2 --num_processes 16 --machine_rank "$NODE_RANK"
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT"
  --same_network --rdzv_backend static --max_restarts 0
  --deepspeed_multinode_launcher standard
  --deepspeed_config_file "$DEEPSPEED_CONFIG"
  "$REPO_DIR/starVLA/training/train_starvla.py"
  --config_yaml "$CONFIG_YAML"
  --framework.qwenvl.base_vlm "$BASE_VLM"
  --framework.action_model.decoder_input_norm true
  --datasets.vla_data.data_root_dir "$DATA_ROOT"
  --datasets.vla_data.frame_cache_dir "$FRAME_CACHE"
  --datasets.vla_data.per_device_batch_size 8
  --trainer.pretrained_checkpoint=null --trainer.is_resume=false
  --trainer.max_train_steps "$MAX_TRAIN_STEPS"
  --trainer.num_warmup_steps "$WARMUP_STEPS"
  --trainer.save_interval "$SAVE_INTERVAL"
  --trainer.eval_interval "$EVAL_INTERVAL"
  --trainer.logging_frequency "$LOGGING_FREQUENCY"
  --trainer.gradient_accumulation_steps 1
  --trainer.learning_rate.action_model 1e-4
  --trainer.learning_rate.qwen_vl_interface 1e-5
  --trainer.lr_scheduler_type cosine_with_min_lr
  --trainer.scheduler_specific_kwargs.min_lr 1e-6
  --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID"
  --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
)

printf 'Host=%s node_rank=%s world_size=16 global_batch=128 run_id=%s\n' "$(hostname)" "$NODE_RANK" "$RUN_ID"
printf 'Output: %s\nCommand: ' "$OUTPUT_DIR"
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Dry run: no output directory created; no training launched.\n'
  exit 0
fi

# Hold this host-local lock for the launcher's entire lifetime (including exec).
# It prevents another copy of this all-eight-GPU launcher from overlapping here.
command -v flock >/dev/null || die "flock is required for duplicate-launch protection"
exec 9>"/tmp/starvla_xperience_16gpu_$(id -u).lock"
flock -n 9 || die "Another Xperience 16-GPU launcher is already active on this host"

# Only rank 0 writes shared launch metadata. Rank 1 does not touch these files.
if [[ "$NODE_RANK" == 0 ]]; then
  mkdir -p "$RUN_ROOT_DIR"
  mkdir "$OUTPUT_DIR" || die "Output directory already exists; choose a new RUN_ID: $OUTPUT_DIR"
  mkdir "$OUTPUT_DIR/launch"
  cp "$SCRIPT_PATH" "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/launch/"
  git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
  git status --short > "$OUTPUT_DIR/launch/git_status.txt"
  git diff --binary HEAD > "$OUTPUT_DIR/launch/tracked_changes.patch"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$OUTPUT_DIR/launch/rank0_command.sh"
  {
    printf 'created_at=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
    printf 'run_id=%s\nmaster_addr=%s\nmaster_port=%s\n' "$RUN_ID" "$MASTER_ADDR" "$MASTER_PORT"
    printf 'nodes=2\ngpus_per_node=8\nworld_size=16\nglobal_batch=128\n'
    printf 'max_train_steps=%s\nwarmup_steps=%s\nsave_interval=%s\neval_interval=%s\nlogging_frequency=%s\n' "$MAX_TRAIN_STEPS" "$WARMUP_STEPS" "$SAVE_INTERVAL" "$EVAL_INTERVAL" "$LOGGING_FREQUENCY"
    printf 'nccl_socket_ifname=%s\ngloo_socket_ifname=%s\nnccl_ib_hca=%s\nnccl_ib_disable=%s\n' "$NCCL_SOCKET_IFNAME" "$GLOO_SOCKET_IFNAME" "$NCCL_IB_HCA" "${NCCL_IB_DISABLE:-auto}"
  } > "$OUTPUT_DIR/launch/experiment.txt"
fi

exec "${cmd[@]}"
