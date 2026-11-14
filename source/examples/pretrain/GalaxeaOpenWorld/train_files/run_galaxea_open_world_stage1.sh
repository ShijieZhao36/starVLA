#!/usr/bin/env bash
# Pretrain on Galaxea Open World R1 Lite stage1 with gr00t_sharded.
# Model: Qwen3-VL-PI, framework name QwenPI, backbone Qwen3-VL-4B-Instruct.
#
# data_mix "galaxea_open_world_stage1" resolves each subset to:
#   ${galaxea_data_root}/<subset_id>/stage1/dataset
# Cameras are front / left_wrist / right_wrist.
# Actions are scattered into the 90-D canonical vector. Galaxea supervises 41
# of those dims; the rest stay zero and are masked out of the loss.
# gr00t_sharded decodes video with torchcodec; video_backend is not passed.

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/pretrain/GalaxeaOpenWorld/train_files/starvla_galaxea_open_world_stage1.yaml
galaxea_data_root=/mnt/zhanghao/Galaxea-Open-World-Dataset/_processed
data_mix=galaxea_open_world_stage1
run_root_dir=/mnt/zhaoshijie/checkpoints/starVLA
run_id=galaxea_open_world_stage1_qwen3vl_pi_gr00t_80k
# Leave empty to train the QwenPI head from the base VLM.
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
  --framework.action_model.action_dim 90 \
  --framework.action_model.state_dim 90 \
  --framework.action_model.action_horizon 8 \
  --datasets.vla_data.dataset_py gr00t_sharded \
  --datasets.vla_data.data_root_dir ${galaxea_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.use_relative_action true \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project pretrain \
  --wandb_entity jasperz-georgia-institute-of-technology \
  "${extra_args[@]}"
  # --is_debug True
