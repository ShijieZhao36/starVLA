#!/usr/bin/env bash
# Pretrain on LIBERO-plus stage1 with the 90-D canonical layout.
# Model: Qwen3-VL-PI, framework name QwenPI, backbone Qwen3-VL-4B-Instruct.
#
# data_mix "libero_plus_stage1" resolves to:
#   ${libero_plus_data_root}/dataset_absolute
# Only right.eef_pose and right.gripper are supervised. right_eef is relative
# to the chunk-start state. The gripper stays absolute.
# gr00t_sharded reads the frame cache; video_backend is not passed.

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/pretrain/LiberoPlus/train_files/starvla_libero_plus_stage1.yaml
libero_plus_data_root=/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1
data_mix=libero_plus_stage1
run_root_dir=/mnt/zhaoshijie/checkpoints/starVLA
run_id=libero_plus_canonical_qwen3vl_pi_gr00t_80k
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
  --datasets.vla_data.data_root_dir ${libero_plus_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.canonical_layout examples/pretrain/canonical/bimanual_v1.yaml \
  --datasets.vla_data.canonical_robots examples/pretrain/canonical/robots \
  --datasets.vla_data.frame_cache_dir /mnt/zhaoshijie/frame_cache/libero_plus_stage1_256 \
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
