#!/usr/bin/env bash
# Full 10,030-episode final-checkpoint evaluation, eight policy servers x four sims.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${POSTTRAIN_RUN_DIR:?Set POSTTRAIN_RUN_DIR to the completed LIBERO Plus run}"
[[ "$POSTTRAIN_RUN_DIR" == /* ]] || die 'POSTTRAIN_RUN_DIR must be absolute'
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_ROOT_DIR/$(basename "$POSTTRAIN_RUN_DIR")_final$( [[ "$SMOKE_TEST" == 0 ]] || printf _smoke )}"
[[ "$SMOKE_TEST" == 0 || "$EVAL_OUTPUT_DIR" == *_smoke ]] || die 'Smoke EVAL_OUTPUT_DIR must end with _smoke'
[[ "$EVAL_OUTPUT_DIR" == /* ]] || die 'EVAL_OUTPUT_DIR must be absolute'
[[ -x "$EVAL_PYTHON" ]] || die "Missing evaluation interpreter: $EVAL_PYTHON"
positive_int EVAL_BASE_PORT "$EVAL_BASE_PORT"
(( EVAL_BASE_PORT > 1024 && EVAL_BASE_PORT <= 65526 )) || die 'Invalid evaluation port range'
EVAL_LAUNCHER="$REPO_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_libero_plus_canonical_qwen3vl_pi_gr00t_60k_1029_8gpu.sh"
[[ -f "$EVAL_LAUNCHER" ]] || die "Missing evaluation launcher: $EVAL_LAUNCHER"
export LIBERO_HOME LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${SIM_ENV_OVERLAY:+$SIM_ENV_OVERLAY:}$REPO_DIR:$LIBERO_HOME"
export STARVLA_DIR="$REPO_DIR" LIBERO_PYTHON="$EVAL_PYTHON"
export your_ckpt="$POSTTRAIN_RUN_DIR/final_model/pytorch_model.pt" output_dir="$EVAL_OUTPUT_DIR"
export USE_CANONICAL_FORWARD=true GRIPPER_OPEN_SIGN=positive USE_BF16=1
export NUM_GPUS=8 SIMS_PER_GPU=4 NUM_TRIALS_PER_TASK=1 BASE_PORT="$EVAL_BASE_PORT" LAUNCH_STAGGER_SEC=0
printf 'LIBERO Plus eval: smoke=%s checkpoint=%s output=%s GPUs=8 sims=32 formal_episodes=10030 smoke_episodes=1 dry_run=%s\n' "$SMOKE_TEST" "$your_ckpt" "$output_dir" "$DRY_RUN"
if [[ "$SMOKE_TEST" == 1 ]]; then
  eval_cmd=("$EVAL_PYTHON" "$REPO_DIR/examples/pretrain/LiberoPlus/eval_files/eval_libero_model.py"
    --pretrained_path "$your_ckpt" --output_dir "$EVAL_OUTPUT_DIR"
    --task_suite_name libero_goal --start_idx 0 --end_idx 1
    --num_trials_per_task 1 --use_bf16 true --use_canonical_forward true)
else
  eval_cmd=(bash "$EVAL_LAUNCHER")
fi
printf 'Command: '; printf '%q ' "${eval_cmd[@]}"; printf '\n' 
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Command preview only; no model, simulator or GPU is started.\n'
  exit 0
fi
mode_args=()
[[ "$SMOKE_TEST" == 0 ]] || mode_args+=(--smoke)
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CHECKS" source --stage posttrain --run "$POSTTRAIN_RUN_DIR" --steps "$POSTTRAIN_STEPS" "${mode_args[@]}"
[[ ! -e "$EVAL_OUTPUT_DIR" ]] || die "Evaluation output exists; refusing overwrite: $EVAL_OUTPUT_DIR"
[[ -z "$SIM_ENV_OVERLAY" || -d "$SIM_ENV_OVERLAY" ]] || die "Missing simulation overlay: $SIM_ENV_OVERLAY"
CUDA_VISIBLE_DEVICES='' NUMBA_CACHE_DIR="/tmp/droid_eval_cpu_$$" "$EVAL_PYTHON" -c 'import mujoco, robosuite, pandas; from libero.libero.envs import OffScreenRenderEnv; print("Simulation CPU imports OK")'
exec 9>"/tmp/starvla_droid_8gpu_$(id -u).lock"
flock -n 9 || die 'Another DROID training/evaluation stage is active on this host'
require_eight_idle_gpus
mkdir -p "$(dirname "$EVAL_OUTPUT_DIR")"
mkdir "$EVAL_OUTPUT_DIR"
export NUMBA_CACHE_DIR="$EVAL_OUTPUT_DIR/runtime_cache/numba"
trap 'rc=$?; printf "%s\n" "$rc" > "$EVAL_OUTPUT_DIR/exit_code"' EXIT
printf 'checkpoint=%s\nenv_seed=7\npolicy_noise_seed=existing_server_behavior\n' "$your_ckpt" > "$EVAL_OUTPUT_DIR/protocol.txt"
cd "$REPO_DIR"
(
  unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
  [[ "$SMOKE_TEST" == 0 ]] || export CUDA_VISIBLE_DEVICES=0
  "${eval_cmd[@]}"
) 2>&1 | tee "$EVAL_OUTPUT_DIR/driver.log"
CUDA_VISIBLE_DEVICES='' "$EVAL_PYTHON" "$CHECKS" evaluated --repo "$REPO_DIR" --run "$POSTTRAIN_RUN_DIR" --output "$EVAL_OUTPUT_DIR" "${mode_args[@]}"
printf 'Completed validated LIBERO Plus evaluation: %s\n' "$EVAL_OUTPUT_DIR"
