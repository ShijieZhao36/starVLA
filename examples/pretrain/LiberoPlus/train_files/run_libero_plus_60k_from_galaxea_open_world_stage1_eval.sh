#!/usr/bin/env bash
set -euo pipefail

# LIBERO-plus post-training from the complete Galaxea Open World stage1 model.
# Load Galaxea step 80000 (VLM + action head), then train LIBERO for 60000
# new steps. LIBERO supplies its own data, normalization stats, and robot mask.
# After successful training, evaluate this run's final model on 8 GPUs.
# Usage: conda activate CRA; bash examples/pretrain/LiberoPlus/train_files/run_libero_plus_60k_from_galaxea_open_world_stage1_eval.sh
# Set DRY_RUN=1 to preview training and evaluation without starting either.

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
repo_root="${STARVLA_DIR:-$(cd "$(dirname "$script_path")/../../../.." && pwd)}"
cd "$repo_root"

base_vlm="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
config_yaml=examples/pretrain/LiberoPlus/train_files/starvla_libero_plus_stage1.yaml
libero_plus_data_root="${LIBERO_PLUS_DATA_ROOT:-/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1}"
frame_cache_dir="${FRAME_CACHE_DIR:-/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256}"
run_root_dir="${RUN_ROOT_DIR:-/mnt/zhaoshijie/checkpoints/starVLA}"
# Append Beijing time on every invocation; RUN_ID overrides the name prefix.
run_timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
run_id="${RUN_ID:-libero_plus_60k_galaxea_open_world_stage1_80k}_${run_timestamp}"
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-/mnt/zhaoshijie/checkpoints/starVLA/galaxea_open_world_stage1_qwen3vl_pi_gr00t_80k/checkpoints/steps_80000_pytorch_model.pt}"
num_processes="${NUM_PROCESSES:-8}"
accelerate_bin="${ACCELERATE_BIN:-accelerate}"
freeze_module_list="${FREEZE_MODULES:-}"
output_dir="$run_root_dir/$run_id"
training_output_dir="$output_dir"
eval_launcher="$repo_root/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_libero_plus_canonical_qwen3vl_pi_gr00t_60k_1029_8gpu.sh"
eval_python="${EVAL_PYTHON:-/root/miniconda3/envs/libero_plus/bin/python}"
eval_libero_home="${EVAL_LIBERO_HOME:-/root/workspace/LIBERO-plus}"
eval_root_dir="${EVAL_OUTPUT_ROOT:-/mnt/zhaoshijie/eval/starVLA/libero_plus}"
eval_output_dir="${EVAL_OUTPUT_DIR:-$eval_root_dir/${run_id}_final}"
eval_base_port="${EVAL_BASE_PORT:-9983}"
eval_sims_per_gpu="${EVAL_SIMS_PER_GPU:-4}"

[[ -f "$pretrained_checkpoint" ]] || { echo "Missing initial checkpoint: $pretrained_checkpoint" >&2; exit 2; }
[[ -f "$config_yaml" ]] || { echo "Missing LIBERO config: $config_yaml" >&2; exit 2; }
[[ -d "$libero_plus_data_root/dataset_absolute" ]] || { echo "Missing LIBERO dataset: $libero_plus_data_root/dataset_absolute" >&2; exit 2; }
[[ -d "$frame_cache_dir/dataset_absolute" ]] || { echo "Missing LIBERO frame cache: $frame_cache_dir/dataset_absolute" >&2; exit 2; }
[[ -f "$eval_launcher" ]] || { echo "Missing evaluation launcher: $eval_launcher" >&2; exit 2; }
[[ -x "$eval_python" ]] || { echo "Missing evaluation Python: $eval_python" >&2; exit 2; }
[[ -d "$eval_libero_home/libero" ]] || { echo "Missing LIBERO checkout: $eval_libero_home" >&2; exit 2; }
[[ ! -e "$eval_output_dir/results.jsonl" && ! -e "$eval_output_dir/driver.log" ]] || {
    echo "Evaluation output already exists; choose a fresh EVAL_OUTPUT_DIR: $eval_output_dir" >&2
    exit 2
}

cmd=(
    "$accelerate_bin" launch
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
    --num_processes "$num_processes"
    starVLA/training/train_starvla.py
    --config_yaml "$config_yaml"
    --framework.name QwenPI
    --framework.qwenvl.base_vlm "$base_vlm"
    --framework.action_model.action_dim 90
    --framework.action_model.state_dim 90
    --framework.action_model.action_horizon 8
    --datasets.vla_data.dataset_py gr00t_sharded
    --datasets.vla_data.data_root_dir "$libero_plus_data_root"
    --datasets.vla_data.data_mix libero_plus_stage1
    --datasets.vla_data.canonical_layout examples/pretrain/canonical/bimanual_v1.yaml
    --datasets.vla_data.canonical_robots examples/pretrain/canonical/robots
    --datasets.vla_data.frame_cache_dir "$frame_cache_dir"
    --datasets.vla_data.use_relative_action true
    --datasets.vla_data.per_device_batch_size 8
    --trainer.pretrained_checkpoint "$pretrained_checkpoint"
    --trainer.is_resume false
    --trainer.max_train_steps 60000
    --trainer.save_interval 10000
    --trainer.logging_frequency 100
    --trainer.eval_interval 100
    --run_root_dir "$run_root_dir"
    --run_id "$run_id"
    --wandb_project "${WANDB_PROJECT:-pretrain}"
    --wandb_entity "${WANDB_ENTITY:-jasperz-georgia-institute-of-technology}"
)
# The LIBERO YAML defaults to no frozen modules. Do not pass an empty,
# unquoted option: the CLI would interpret that bare flag as boolean true.
if [[ -n "$freeze_module_list" ]]; then
    cmd+=(--trainer.freeze_modules "$freeze_module_list")
fi

printf 'Initial checkpoint: %s\nOutput directory: %s\n' "$pretrained_checkpoint" "$output_dir"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    printf 'After training succeeds: evaluate %s/final_model/pytorch_model.pt (or model.safetensors)\n' "$training_output_dir"
    printf 'Evaluation: 8 GPUs x %s sims, base port %s, Python %s\n' "$eval_sims_per_gpu" "$eval_base_port" "$eval_python"
    printf 'Evaluation launcher: %s\nEvaluation output: %s\n' "$eval_launcher" "$eval_output_dir"
    exit 0
fi
command -v "$accelerate_bin" >/dev/null || { echo "Activate the training environment or set ACCELERATE_BIN." >&2; exit 2; }
if [[ -e "$output_dir/config.yaml" || -d "$output_dir/checkpoints" || -d "$output_dir/final_model" ]]; then
    echo "Existing training run at $output_dir; set a fresh RUN_ID." >&2
    exit 2
fi
mkdir -p "$output_dir"
cp "$script_path" "$output_dir/"
if "${cmd[@]}"; then
    printf 'Training completed successfully: %s\n' "$training_output_dir"
else
    training_status=$?
    printf 'Training failed (exit %s); evaluation was not started.\n' "$training_status" >&2
    exit "$training_status"
fi

# The trainer writes final_model after its last periodic checkpoint and waits
# for all ranks. A successful accelerate exit means the final save is complete.
if [[ -s "$training_output_dir/final_model/pytorch_model.pt" ]]; then
    final_checkpoint="$training_output_dir/final_model/pytorch_model.pt"
elif [[ -s "$training_output_dir/final_model/model.safetensors" ]]; then
    final_checkpoint="$training_output_dir/final_model/model.safetensors"
else
    echo "Training returned success but its final checkpoint is missing; evaluation was not started." >&2
    exit 2
fi
for metadata in config.yaml dataset_statistics.json; do
    [[ -s "$training_output_dir/$metadata" ]] || {
        echo "Missing evaluation metadata: $training_output_dir/$metadata" >&2
        exit 2
    }
done
# Recheck after the long training run before writing evaluation output.
[[ ! -e "$eval_output_dir/results.jsonl" && ! -e "$eval_output_dir/driver.log" ]] || {
    echo "Evaluation output already exists: $eval_output_dir" >&2
    exit 2
}
mkdir -p "$eval_output_dir"
printf 'checkpoint=%s\noutput_dir=%s\nlauncher=%s\n' \
    "$final_checkpoint" "$eval_output_dir" "$eval_launcher" > "$training_output_dir/posttrain_eval_info.txt"
printf 'Starting 8-GPU evaluation of final checkpoint: %s\n' "$final_checkpoint"

set +e
(
    unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
    export STARVLA_DIR="$repo_root"
    export LIBERO_PYTHON="$eval_python" LIBERO_HOME="$eval_libero_home"
    export PYTHONPATH="$repo_root:$eval_libero_home"
    export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export your_ckpt="$final_checkpoint" output_dir="$eval_output_dir"
    export USE_CANONICAL_FORWARD=true GRIPPER_OPEN_SIGN=positive USE_BF16=1
    export NUM_GPUS=8 SIMS_PER_GPU="$eval_sims_per_gpu" NUM_TRIALS_PER_TASK=1
    export BASE_PORT="$eval_base_port" LAUNCH_STAGGER_SEC=0
    bash "$eval_launcher"
) 2>&1 | tee "$eval_output_dir/driver.log"
pipeline_status=("${PIPESTATUS[@]}")
eval_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
set -e
printf '%s\n' "$eval_status" > "$eval_output_dir/exit_status.txt"
printf 'EVAL_EXIT=%s TEE_EXIT=%s\n' "$eval_status" "$tee_status" | tee -a "$eval_output_dir/driver.log"
if [[ "$eval_status" -ne 0 ]]; then
    exit "$eval_status"
fi
exit "$tee_status"
