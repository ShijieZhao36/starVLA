#!/usr/bin/env bash
# Precompute 256x256 uint8 frame caches for Galaxea Open World stage1.
#
# One uncompressed .npy per episode per camera, shape (T, 256, 256, 3):
#   ${FRAME_CACHE_DIR}/<subset_id>/<video_key>/episode_XXXXXX.npy
# video_key matches the mp4 name, for example observation.images.front.
# Cameras are front, left_wrist, and right_wrist. T is the episodes.jsonl
# length. A file that already has that length is skipped.
#
# Each process decodes 32 frames of 720p at a time and is pinned to one
# thread. CACHE_WORKERS defaults to available CPUs minus 8. Training
# num_workers stays 4; this script does not change it.
#
#   conda activate MRA
#   bash examples/pretrain/GalaxeaOpenWorld/train_files/precompute_galaxea_frame_cache.sh
#
#   CACHE_WORKERS=32 bash examples/pretrain/GalaxeaOpenWorld/train_files/precompute_galaxea_frame_cache.sh

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export GALAXEA_DATA_ROOT="${GALAXEA_DATA_ROOT:-/mnt/zhanghao/Galaxea-Open-World-Dataset/_processed}"
export DATA_MIX="${DATA_MIX:-galaxea_open_world_stage1}"
export FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-/mnt/zhaoshijie/frame_cache/galaxea_stage1_256}"
export CACHE_SIZE="${CACHE_SIZE:-256}"
export CACHE_CHUNK="${CACHE_CHUNK:-32}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

python -m starVLA.dataloader.precompute_galaxea_frame_cache
