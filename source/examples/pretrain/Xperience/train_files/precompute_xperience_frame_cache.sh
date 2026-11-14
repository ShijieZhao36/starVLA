#!/usr/bin/env bash
# Precompute 256x256 uint8 frame caches for Xperience hours100.
#
# One uncompressed .npy per episode, shape (T, 256, 256, 3):
#   ${FRAME_CACHE_DIR}/hours100/observation.images.front/episode_XXXXXX.npy
# The only camera is front. T is the episodes.jsonl length. A file that
# already has that length is skipped. 7.2M frames at 256x256x3 is about 1.3 TiB.
#
# Each process decodes 32 frames of 512x512 AV1 at a time and is pinned to one
# thread. CACHE_WORKERS defaults to available CPUs minus 8. Training
# num_workers stays 4; this script does not change it.
#
#   conda activate MRA
#   bash examples/pretrain/Xperience/train_files/precompute_xperience_frame_cache.sh
#
#   CACHE_WORKERS=32 bash examples/pretrain/Xperience/train_files/precompute_xperience_frame_cache.sh

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export DATA_ROOT="${DATA_ROOT:-/mnt/zhaoshijie/dataset/xperience_runs/stage1}"
export DATA_MIX="${DATA_MIX:-xperience_hours100}"
export FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-/mnt/zhaoshijie/frame_cache/xperience_hours100_256}"
export CACHE_CAMERAS="${CACHE_CAMERAS:-front}"
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
