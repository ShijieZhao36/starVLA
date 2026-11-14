#!/usr/bin/env bash
set -euo pipefail

# 10-d QwenPI LIBERO-plus stage1 run (gr00t_sharded, xyz + rot6d + gripper).
# PolicyModel decodes that chunk to the 7-d OSC command before the sim worker.
# Trained with the interleaved DiT, so USE_CANONICAL_FORWARD must stay true.
# Stage1 stores gripper as (1 - raw) / 2, so raw -1 open becomes 1. Keep
# GRIPPER_OPEN_SIGN=positive; the 10-d path does not use the 7-d threshold.
#
# parallel_eval/ is five levels below the repo root:
# parallel_eval -> eval_files -> LIBERO-plus -> simBenchmarks -> examples -> starVLA
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../../../" && pwd)}"
export LIBERO_HOME="${LIBERO_HOME:-/root/workspace/LIBERO-plus}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
your_ckpt="${your_ckpt:-/mnt/zhaoshijie/checkpoints/starVLA/libero_plus_qwen3vl_pi_gr00t_120k/checkpoints/steps_120000_pytorch_model.pt}"
output_dir="${output_dir:-/mnt/zhaoshijie/eval/starVLA/libero_plus/libero_plus_qwen3vl_pi_gr00t_120k}"
USE_CANONICAL_FORWARD="${USE_CANONICAL_FORWARD:-true}"
GRIPPER_OPEN_SIGN="${GRIPPER_OPEN_SIGN:-positive}"
SIMS_PER_GPU="${SIMS_PER_GPU:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-9883}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-90}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"

mkdir -p "${output_dir}"

if [[ "${USE_CANONICAL_FORWARD}" != "true" && "${USE_CANONICAL_FORWARD}" != "false" ]]; then
  echo "USE_CANONICAL_FORWARD must be 'true' or 'false'; got '${USE_CANONICAL_FORWARD}'." >&2
  exit 2
fi
if [[ "${GRIPPER_OPEN_SIGN}" != "positive" && "${GRIPPER_OPEN_SIGN}" != "negative" ]]; then
  echo "GRIPPER_OPEN_SIGN must be 'positive' or 'negative'; got '${GRIPPER_OPEN_SIGN}'." >&2
  exit 2
fi

echo "One policy server per GPU, ${SIMS_PER_GPU} sim workers per GPU, dynamic episode queue."
echo "Checkpoint: ${your_ckpt}"
echo "Applying config override: framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}"
echo "Gripper open sign: ${GRIPPER_OPEN_SIGN}"

cmd=(
  "${LIBERO_PYTHON}"
  examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_queue_eval.py
  --pretrained-path "${your_ckpt}"
  --output-dir "${output_dir}"
  --use-canonical-forward "${USE_CANONICAL_FORWARD}"
  --gripper-open-sign "${GRIPPER_OPEN_SIGN}"
  --sims-per-gpu "${SIMS_PER_GPU}"
  --num-gpus "${NUM_GPUS}"
  --base-port "${BASE_PORT}"
  --launch-stagger-sec "${LAUNCH_STAGGER_SEC}"
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}"
)
if [[ "${USE_BF16}" == "0" ]]; then
  cmd+=(--no-bf16)
fi
"${cmd[@]}"
