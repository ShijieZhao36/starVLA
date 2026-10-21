#!/usr/bin/env bash
# Post-train LIBERO-plus with starVLA's original dataloader (lerobot_datasets).
# Model: Qwen3-VL-PI, framework name QwenPI, backbone Qwen3-VL-4B-Instruct.
#
# data_mix "libero_plus" resolves to:
#   ${libero_plus_data_root}/libero_plus_lerobot_starvla
# which must contain meta/modality.json (video keys front / wrist).

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_plus_data_root=/mnt/zhaoshijie/dataset
data_mix=libero_plus
run_root_dir=/mnt/zhaoshijie/checkpoints/starVLA
run_id=libero_plus_qwen3vl_pi_60k
# Optional: continue from the released LIBERO-4in1 Qwen3-VL-PI weights.
# Leave empty to train the current QwenPI head from the base VLM.
# The released checkpoint was trained with the historical 16-layer DiT; the
# current QwenPI rebuilds the action head to the VLM depth, so loading that
# file can fail on shape mismatch.
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-}"
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/


# This host has 8x H100 80GB. Set NUM_PROCESSES to override.
num_processes=${NUM_PROCESSES:-8}

extra_args=()
if [[ -n "${pretrained_checkpoint}" ]]; then
  extra_args+=(--trainer.pretrained_checkpoint "${pretrained_checkpoint}")
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_dim 7 \
  --framework.action_model.state_dim 7 \
  --framework.action_model.action_horizon 8 \
  --datasets.vla_data.dataset_py lerobot_datasets \
  --datasets.vla_data.data_root_dir ${libero_plus_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 60000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project pretrain \
  --wandb_entity jasperz-georgia-institute-of-technology \
  "${extra_args[@]}"
  # --is_debug True
