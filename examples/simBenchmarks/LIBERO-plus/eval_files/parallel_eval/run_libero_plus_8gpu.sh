#!/usr/bin/env bash
set -euo pipefail

# parallel_eval/ is five levels below the repo root:
# parallel_eval -> eval_files -> LIBERO-plus -> simBenchmarks -> examples -> starVLA
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../../../" && pwd)}"
export LIBERO_HOME="${LIBERO_HOME:-/root/workspace/LIBERO-plus}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
your_ckpt="${your_ckpt:-/mnt/zhaoshijie/models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_8gpu_2env}"
# This released checkpoint was trained with historical LayerwiseFM. Set true only
# for a checkpoint trained with the canonical interleaved DiT.
USE_CANONICAL_FORWARD="${USE_CANONICAL_FORWARD:-false}"

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"

mkdir -p "${output_dir}"

if [[ "${USE_CANONICAL_FORWARD}" != "true" && "${USE_CANONICAL_FORWARD}" != "false" ]]; then
  echo "USE_CANONICAL_FORWARD must be 'true' or 'false'; got '${USE_CANONICAL_FORWARD}'." >&2
  exit 2
fi
if [[ "${USE_CANONICAL_FORWARD}" == "true" ]]; then
  canonical_arg=True
else
  canonical_arg=False
fi
echo "Applying config override: framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}"

# end_idx is exclusive. Two processes share each GPU and each loads the checkpoint.
# Shards are paired so the two control-step budgets on one GPU sum to about 420k–445k.
# libero_10 is 530 steps/task, goal 310, object 290, spatial 230.
launch() {
  local gpu_id="$1"
  local task_suite_name="$2"
  local start_idx="$3"
  local end_idx="$4"
  echo "GPU ${gpu_id}: ${task_suite_name} [${start_idx}, ${end_idx})"
  # EGL device ids follow the physical GPU list, not CUDA_VISIBLE_DEVICES.
  CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" "${LIBERO_PYTHON}" \
    examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
    --pretrained_path "${your_ckpt}" \
    --task_suite_name "${task_suite_name}" \
    --start_idx "${start_idx}" \
    --end_idx "${end_idx}" \
    --num_trials_per_task 1 \
    --output_dir "${output_dir}" \
    --use_bf16 True \
    --use_canonical_forward "${canonical_arg}"
}

run_pair() {
  local gpu_id="$1"
  shift
  launch "${gpu_id}" "$1" "$2" "$3" &
  launch "${gpu_id}" "$4" "$5" "$6" &
}

# Each pair loads the checkpoint twice. Space the pairs so only one pair is
# reading weights at a time. Earlier GPUs keep evaluating during the wait.
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-90}"

run_pair 0 libero_10 0 420 libero_10 420 840
sleep "${LAUNCH_STAGGER_SEC}"
run_pair 1 libero_10 840 1260 libero_goal 0 648
sleep "${LAUNCH_STAGGER_SEC}"
run_pair 2 libero_10 1260 1680 libero_goal 648 1296
sleep "${LAUNCH_STAGGER_SEC}"
run_pair 3 libero_10 1680 2100 libero_goal 1296 1944
sleep "${LAUNCH_STAGGER_SEC}"
run_pair 4 libero_10 2100 2519 libero_goal 1944 2591
sleep "${LAUNCH_STAGGER_SEC}"
run_pair 5 libero_object 0 840 libero_spatial 0 801
sleep "${LAUNCH_STAGGER_SEC}"
run_pair 6 libero_object 840 1679 libero_spatial 801 1602
sleep "${LAUNCH_STAGGER_SEC}"
run_pair 7 libero_object 1679 2518 libero_spatial 1602 2402

wait
echo "All 16 eval jobs finished. Logs: ${output_dir}/logs/<suite>/<start>_<end>.log"
