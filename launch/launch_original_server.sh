#!/usr/bin/env bash
set -euo pipefail
# Recorded original server paths; this script was not executed by the archive task.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
export PYTHONPATH=/mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114
export PYTHONUNBUFFERED=1
cd /mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114
exec /root/miniconda3/envs/starvla_pi05/bin/python -m accelerate.commands.launch --config_file /mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114/starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 --main_process_port 29872 --deepspeed_config_file /mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114/starVLA/config/deepseeds/ds_config.yaml /mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114/starVLA/training/train_starvla.py --config_yaml /mnt/vla-pretrain/zh1112/preparation/starvla_pi05_socks35_bs8_100k_20261114_0417/train_100k_bs8.yaml --run_root_dir /mnt/vla-pretrain/zh1112/checkpoints/starVLA --run_id pi05_socks35_abs_h50_state14_mask14_pil224_lr5e5_bs8_8gpu_100k_20261114
