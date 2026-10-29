#!/usr/bin/env bash
# In-process LIBERO-plus eval for the 90-D canonical checkpoint.
# The policy turns relative right.eef_pose into an OSC delta from the live
# end-effector. Gripper binarization stays in libero_rollout.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export LIBERO_HOME="${LIBERO_HOME:-/root/workspace/LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}:${LIBERO_HOME}"

pretrained_path="${PRETRAINED_PATH:-/mnt/zhaoshijie/checkpoints/starVLA/libero_plus_canonical_qwen3vl_pi_gr00t_80k/checkpoints/steps_80000_pytorch_model.pt}"
output_dir="${OUTPUT_DIR:-/mnt/zhaoshijie/eval/starVLA/libero_plus/libero_plus_canonical_qwen3vl_pi_gr00t_80k}"
task_suite="${TASK_SUITE:-libero_goal}"

python examples/pretrain/LiberoPlus/eval_files/eval_libero_model.py \
  --args.pretrained-path "${pretrained_path}" \
  --args.output-dir "${output_dir}" \
  --args.task-suite-name "${task_suite}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK:-1}" \
  --args.use-bf16 "${USE_BF16:-true}" \
  --args.use-canonical-forward "${USE_CANONICAL_FORWARD:-true}"
