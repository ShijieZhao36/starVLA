#!/usr/bin/env bash
# Precompute native-resolution uint8 frame caches for LIBERO-plus stage1.
#
# One uncompressed .npy per episode per camera. Videos are already 256x256,
# so the cache is not resized:
#   ${FRAME_CACHE_DIR}/dataset_absolute/<video_key>/episode_XXXXXX.npy
# video_key is observation.images.front or observation.images.right_wrist.
# T is the episodes.jsonl length. A file that already has that length is skipped.
#
#   conda activate MRA
#   bash examples/pretrain/LiberoPlus/train_files/precompute_libero_plus_frame_cache.sh

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export LIBERO_PLUS_DATA_ROOT="${LIBERO_PLUS_DATA_ROOT:-/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1}"
export DATA_MIX="${DATA_MIX:-libero_plus_stage1}"
export FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256}"
export CACHE_CHUNK="${CACHE_CHUNK:-32}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

python -m starVLA.dataloader.precompute_libero_frame_cache
