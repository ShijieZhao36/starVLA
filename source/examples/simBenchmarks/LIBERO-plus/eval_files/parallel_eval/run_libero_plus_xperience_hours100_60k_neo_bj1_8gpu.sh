#!/usr/bin/env bash
set -euo pipefail

# LIBERO fine-tuning step 60000, initialized from Xperience hours100 step 80000.
# Environment variables may override the checkpoint, paths, and worker settings.
export STARVLA_DIR="${STARVLA_DIR:-/mnt/zhaoshijie/workspace/starVLA}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/root/miniconda3/envs/libero_plus/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/root/workspace/LIBERO-plus}"
export your_ckpt="${your_ckpt:-/mnt/zhaoshijie/checkpoints/starVLA/libero_plus_60k_xperience_hours100_80k/checkpoints/steps_60000_pytorch_model.pt}"
export output_dir="${output_dir:-/mnt/zhaoshijie/eval/starVLA/libero_plus/libero_plus_60k_xperience_hours100_80k_neo_bj1_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
export USE_CANONICAL_FORWARD="${USE_CANONICAL_FORWARD:-true}"
export GRIPPER_OPEN_SIGN="${GRIPPER_OPEN_SIGN:-positive}"
export USE_BF16="${USE_BF16:-1}"
export NUM_GPUS="${NUM_GPUS:-8}" SIMS_PER_GPU="${SIMS_PER_GPU:-4}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
export BASE_PORT="${BASE_PORT:-9963}" LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-0}"
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID

[[ -d "$STARVLA_DIR" ]] || { echo "StarVLA checkout missing: $STARVLA_DIR" >&2; exit 2; }
[[ -f "$your_ckpt" ]] || { echo "Checkpoint missing: $your_ckpt" >&2; exit 2; }
[[ -x "$LIBERO_PYTHON" ]] || { echo "Python executable missing: $LIBERO_PYTHON" >&2; exit 2; }
[[ -d "$LIBERO_HOME/libero" ]] || { echo "LIBERO checkout missing: $LIBERO_HOME" >&2; exit 2; }

# Resolve paths before changing directory so explicit relative overrides work.
STARVLA_DIR="$(cd "$STARVLA_DIR" && pwd)"
LIBERO_HOME="$(cd "$LIBERO_HOME" && pwd)"
your_ckpt="$(cd "$(dirname "$your_ckpt")" && pwd)/$(basename "$your_ckpt")"
LIBERO_PYTHON="$(cd "$(dirname "$LIBERO_PYTHON")" && pwd)/$(basename "$LIBERO_PYTHON")"
launcher="$STARVLA_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_libero_plus_canonical_qwen3vl_pi_gr00t_60k_1029_8gpu.sh"
[[ -f "$launcher" ]] || { echo "Shared canonical launcher missing: $launcher" >&2; exit 2; }
checkpoint_root="$(dirname "$(dirname "$your_ckpt")")"
for metadata in config.yaml dataset_statistics.json; do
    [[ -f "$checkpoint_root/$metadata" ]] || {
        echo "Checkpoint metadata missing: $checkpoint_root/$metadata" >&2
        exit 2
    }
done
[[ ! -e "$output_dir/results.jsonl" && ! -e "$output_dir/driver.log" ]] || {
    echo "Output directory already contains an evaluation; choose a fresh output_dir: $output_dir" >&2
    exit 2
}
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
export PYTHONPATH="$STARVLA_DIR:$LIBERO_HOME"

{
    printf 'started_at=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
    printf 'host=%s\n' "$(hostname)"
    printf 'checkpoint=%s\npython=%s\nstarvla_dir=%s\nlibero_home=%s\noutput_dir=%s\n' \
        "$your_ckpt" "$LIBERO_PYTHON" "$STARVLA_DIR" "$LIBERO_HOME" "$output_dir"
    printf 'num_gpus=%s\nsims_per_gpu=%s\ntrials_per_task=%s\nbase_port=%s\nstagger_sec=%s\n' \
        "$NUM_GPUS" "$SIMS_PER_GPU" "$NUM_TRIALS_PER_TASK" "$BASE_PORT" "$LAUNCH_STAGGER_SEC"
    printf 'bf16=%s\ncanonical_forward=%s\ngripper_open_sign=%s\n' \
        "$USE_BF16" "$USE_CANONICAL_FORWARD" "$GRIPPER_OPEN_SIGN"
} > "$output_dir/run_info.txt"

cd "$STARVLA_DIR"
set +e
bash "$launcher" 2>&1 | tee "$output_dir/driver.log"
pipeline_status=("${PIPESTATUS[@]}")
eval_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
set -e
printf '%s\n' "$eval_status" > "$output_dir/exit_status.txt"
printf '%s\n' "EVAL_EXIT=$eval_status TEE_EXIT=$tee_status $(TZ=Asia/Shanghai date --iso-8601=seconds)" | tee -a "$output_dir/driver.log"
if [[ "$eval_status" -ne 0 ]]; then
    exit "$eval_status"
fi
exit "$tee_status"
