#!/usr/bin/env bash
# Two-node direct LIBERO LN: each node runs eight GPUs; rank0=bj5, rank1=bj6.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
: "${NODE_RANK:?Set NODE_RANK=0 or 1}"
: "${RUN_ID:?Use the same unique RUN_ID on both nodes}"
: "${LAUNCHER_DIR:?Set the same shared launcher directory}"
[[ "$NODE_RANK" =~ ^[01]$ ]] || die 'Invalid NODE_RANK'
[[ "$RUN_ID" =~ ^libero_plus_direct_ln_16gpu_30k_[A-Za-z0-9_.-]+$ ]] || die 'Invalid RUN_ID'
REPO_DIR=/mnt/zhaoshijie/workspace/starVLA_libero_direct_ln_16gpu_30k_20261107
PYTHON_BIN=/root/miniconda3/envs/MRA/bin/python
DRY_RUN="${DRY_RUN:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
[[ "$DRY_RUN" =~ ^[01]$ && "$SMOKE_TEST" =~ ^[01]$ ]] || die 'Invalid mode'
RUN_ROOT_DIR=/mnt/zhaoshijie/checkpoints/starVLA
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
EVAL_OUTPUT_DIR="/mnt/zhaoshijie/eval/starVLA/libero_plus/${RUN_ID}_final16"
MASTER_ADDR=172.27.13.188
MASTER_PORT=29685
BASE_VLM=/mnt/zhaoshijie/workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
REFERENCE_CONTRACT=/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_posttrain60k_20261105_contract
ENV_OVERLAY=/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_pair_validation_20261105/sim_overlay
LIBERO_HOME=/root/workspace/LIBERO-plus
DATA_ROOT=/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1
FRAME_CACHE=/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
TRAIN_FILES="$(dirname "$SCRIPT_PATH")"
CONFIG_YAML="$REPO_DIR/examples/pretrain/GalaxeaLiberoMix/train_files/starvla_libero_pair_ln_60k.yaml"
ACCELERATE_CONFIG="$TRAIN_FILES/accelerate_libero_direct_2nodes.yaml"
DEEPSPEED_CONFIG="$REPO_DIR/examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json"
VALIDATOR="$TRAIN_FILES/validate_libero_direct_16gpu.py"
COORDINATOR="$TRAIN_FILES/coordinate_nodes.py"
EVAL_SCRIPT="$TRAIN_FILES/../eval_files/run_libero_plus_16gpu.py"
MAX_TRAIN_STEPS=30000 WARMUP_STEPS=5000 SAVE_INTERVAL=10000 LOG_INTERVAL=100
if [[ "$SMOKE_TEST" == 1 ]]; then
 [[ "$RUN_ID" == *_smoke3 ]] || die 'Smoke needs _smoke3 suffix'
 MAX_TRAIN_STEPS=3 WARMUP_STEPS=1 SAVE_INTERVAL=3 LOG_INTERVAL=1
else
 [[ "$RUN_ID" != *_smoke3 ]] || die 'Formal run cannot use smoke suffix'
fi
cd "$REPO_DIR"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ENV_OVERLAY:$REPO_DIR:$LIBERO_HOME"
export LIBERO_HOME LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='=com0_0,com0_1,com0_2,com0_3,com0_4,com0_5,com0_6,com0_7'
unset NCCL_IB_DISABLE
export WANDB_MODE=offline WANDB_PROJECT=pretrain WANDB_ENTITY=jasperz-georgia-institute-of-technology
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
train_cmd=("$PYTHON_BIN" -m accelerate.commands.launch
 --config_file "$ACCELERATE_CONFIG" --num_machines 2 --num_processes 16 --machine_rank "$NODE_RANK"
 --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT"
 --same_network --rdzv_backend static --max_restarts 0 --deepspeed_multinode_launcher standard
 --deepspeed_config_file "$DEEPSPEED_CONFIG"
 "$REPO_DIR/starVLA/training/train_starvla.py" --config_yaml "$CONFIG_YAML"
 --framework.qwenvl.base_vlm "$BASE_VLM" --seed 42
 --trainer.pretrained_checkpoint=null --trainer.reload_modules=null --trainer.is_resume=false
 --trainer.strict_pretrained_load=true
 --trainer.expected_initial_model_sha256 530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89
 --datasets.vla_data.per_device_batch_size 8 --trainer.gradient_accumulation_steps 1
 --trainer.max_train_steps "$MAX_TRAIN_STEPS" --trainer.num_warmup_steps "$WARMUP_STEPS"
 --trainer.save_interval "$SAVE_INTERVAL" --trainer.eval_interval "$LOG_INTERVAL" --trainer.logging_frequency "$LOG_INTERVAL"
 --run_root_dir "$RUN_ROOT_DIR" --run_id "$RUN_ID")
gate_args=(--output "$OUTPUT_DIR" --repo "$REPO_DIR" --config "$CONFIG_YAML" --runner "$SCRIPT_PATH"
 --accelerate "$ACCELERATE_CONFIG" --deepspeed "$DEEPSPEED_CONFIG" --overlay "$ENV_OVERLAY"
 --data "$DATA_ROOT/dataset_absolute" --cache "$FRAME_CACHE" --reference-contract "$REFERENCE_CONTRACT"
 --base-vlm "$BASE_VLM" --node-rank "$NODE_RANK")
smoke_args=(); [[ "$SMOKE_TEST" == 0 ]] || smoke_args=(--smoke)
gate_args+=("${smoke_args[@]}")
report_args=(); [[ -z "${CPU_PREFLIGHT_REPORT:-}" ]] || report_args=(--report "$CPU_PREFLIGHT_REPORT")
coord_args=(--run "$OUTPUT_DIR" --launcher "$LAUNCHER_DIR" --node-rank "$NODE_RANK")
eval_args=(--node-rank "$NODE_RANK" --repo "$REPO_DIR" --pretrained-path "$OUTPUT_DIR/final_model/pytorch_model.pt" --output-dir "$EVAL_OUTPUT_DIR" --renderer-map 0,1,2,3,4,5,6,7,0,1,6,1,5,5,6,7 "${smoke_args[@]}")
printf 'run=%s node_rank=%s world_size=16 global_batch=128 dry_run=%s smoke=%s\n' "$RUN_ID" "$NODE_RANK" "$DRY_RUN" "$SMOKE_TEST"
printf 'Training command: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" --dry-run "${report_args[@]}"
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" eval-protocol --output "$OUTPUT_DIR" --reference-contract "$REFERENCE_CONTRACT" --libero-home "$LIBERO_HOME" --dry-run "${report_args[@]}"
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$EVAL_SCRIPT" "${eval_args[@]}" --dry-run
 exit 0
fi
eval_record_args=(); [[ "$NODE_RANK" == 0 ]] || eval_record_args=(--dry-run)
phase=preflight
owns_run=0
mkdir -p "$LAUNCHER_DIR/node$NODE_RANK"
finish() {
 rc=$?
 mkdir -p "$LAUNCHER_DIR/node$NODE_RANK"
 printf 'time=%s phase=%s exit_code=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" "$rc" >> "$LAUNCHER_DIR/node$NODE_RANK/stages.txt"
 if [[ "$owns_run" == 1 ]]; then
  printf '%s\n' "$rc" > "$OUTPUT_DIR/launch/exit_code_node$NODE_RANK"
  if [[ "$rc" != 0 ]]; then printf '{"node_rank":%s,"exit_code":%s,"phase":"%s"}\n' "$NODE_RANK" "$rc" "$phase" > "$OUTPUT_DIR/launch/failure_node$NODE_RANK.json"; fi
  if [[ "$NODE_RANK" == 0 ]]; then printf '%s\n' "$rc" > "$OUTPUT_DIR/exit_code"; fi
 fi
}
trap finish EXIT
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" --check-only
exec 9>"/tmp/starvla_mix_pair_8gpu_$(id -u).lock"; flock -n 9 || die 'Host pipeline lock occupied'
exec 8>"/tmp/starvla_libero_ln_8gpu_$(id -u).lock"; flock -n 8 || die 'LIBERO host lock occupied'
"$PYTHON_BIN" - <<'PYGPU'
import subprocess,xml.etree.ElementTree as E
cards=E.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],timeout=20)).findall('gpu')
assert len(cards)==8
for g in cards:
 assert (g.findtext('gpu_recovery_action'),g.findtext('remapped_rows/remapped_row_pending'),g.findtext('remapped_rows/remapped_row_failure'))==('None','No','No')
 assert g.findtext('ecc_errors/volatile/dram_uncorrectable')=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],timeout=20).strip()
PYGPU
if [[ "$NODE_RANK" == 0 ]]; then
 [[ ! -e "$OUTPUT_DIR" && ! -e "$EVAL_OUTPUT_DIR" ]] || die 'Refusing existing output'
 mkdir "$OUTPUT_DIR"; mkdir "$OUTPUT_DIR/launch" "$OUTPUT_DIR/wandb" "$OUTPUT_DIR/runtime_cache"
 cp "$SCRIPT_PATH" "$VALIDATOR" "$CONFIG_YAML" "$ACCELERATE_CONFIG" "$DEEPSPEED_CONFIG" "$COORDINATOR" "$OUTPUT_DIR/launch/"
 git rev-parse HEAD > "$OUTPUT_DIR/launch/git_commit.txt"
 git status --short > "$OUTPUT_DIR/launch/git_status.txt"
 printf '{"run_id":"%s"}\n' "$RUN_ID" > "$OUTPUT_DIR/launch/created.json"
else
 "$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$OUTPUT_DIR/launch/created.json"
fi
mkdir "$OUTPUT_DIR/launch/owner_node$NODE_RANK" || die 'This node already owns the run'
owns_run=1
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch/train_command_node$NODE_RANK.txt"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" preflight "${gate_args[@]}" "${report_args[@]}"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" eval-protocol --output "$OUTPUT_DIR" --reference-contract "$REFERENCE_CONTRACT" --libero-home "$LIBERO_HOME" "${eval_record_args[@]}" "${report_args[@]}"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -c 'import mujoco,robosuite; from libero.libero.envs import OffScreenRenderEnv; print("Simulation imports passed")'
printf '{"ready":true}\n' > "$OUTPUT_DIR/launch/ready_node$NODE_RANK.json"
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$OUTPUT_DIR/launch/ready_node0.json" --path "$OUTPUT_DIR/launch/ready_node1.json"
"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PYPAIR'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/'launch'
a=json.loads((root/'direct_protocol_node0.json').read_text())
b=json.loads((root/'direct_protocol_node1.json').read_text())
assert a.pop('node_rank')==0 and b.pop('node_rank')==1
assert a==b,'Two-node protocol mismatch before training'
PYPAIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUMBA_CACHE_DIR="$OUTPUT_DIR/runtime_cache/node$NODE_RANK/numba"
phase=training
printf 'time=%s phase=training node_rank=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$NODE_RANK" >> "$LAUNCHER_DIR/node$NODE_RANK/stages.txt"
train_log="$OUTPUT_DIR/train_node$NODE_RANK.log"
[[ "$NODE_RANK" == 1 ]] || train_log="$OUTPUT_DIR/train.log"
"$PYTHON_BIN" "$COORDINATOR" run "${coord_args[@]}" --timeout 86400 --log "$train_log" -- "${train_cmd[@]}"
printf '0\n' > "$OUTPUT_DIR/launch/train_exit_node$NODE_RANK"
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$OUTPUT_DIR/launch/train_exit_node0" --path "$OUTPUT_DIR/launch/train_exit_node1"
phase=validating_training
if [[ "$NODE_RANK" == 0 ]]; then
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" trained --output "$OUTPUT_DIR" --data "$DATA_ROOT/dataset_absolute" --reference-contract "$REFERENCE_CONTRACT" "${smoke_args[@]}"
fi
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$OUTPUT_DIR/posttrain_success.json"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$VALIDATOR" eval-protocol --output "$OUTPUT_DIR" --reference-contract "$REFERENCE_CONTRACT" --libero-home "$LIBERO_HOME" "${eval_record_args[@]}"
phase=evaluating
printf 'time=%s phase=evaluating node_rank=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$NODE_RANK" >> "$LAUNCHER_DIR/node$NODE_RANK/stages.txt"
"$PYTHON_BIN" "$COORDINATOR" run "${coord_args[@]}" --timeout 21600 --log "$OUTPUT_DIR/eval_driver_node$NODE_RANK.log" -- "$PYTHON_BIN" "$EVAL_SCRIPT" "${eval_args[@]}"
printf '0\n' > "$OUTPUT_DIR/launch/eval_exit_node$NODE_RANK"
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$OUTPUT_DIR/launch/eval_exit_node0" --path "$OUTPUT_DIR/launch/eval_exit_node1"
if [[ "$NODE_RANK" == 0 ]]; then
 cp "$EVAL_OUTPUT_DIR/eval_success.json" "$OUTPUT_DIR/eval_success.json"
fi
phase=completed
printf 'Completed %s on node %s\n' "$RUN_ID" "$NODE_RANK"
