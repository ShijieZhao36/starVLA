#!/usr/bin/env bash
# Precompute dataset stats before LIBERO-plus canonical training.
#
# Rank 0 otherwise builds meta/stats.json and meta/relative_stats.json inside
# an NCCL barrier. Run this once, with the training env active, before
# run_libero_plus_stage1.sh:
#
#   conda activate MRA
#   bash examples/pretrain/LiberoPlus/train_files/precompute_libero_plus_stage1_stats.sh
#
# data_mix "libero_plus_stage1" resolves to:
#   ${LIBERO_PLUS_DATA_ROOT}/dataset_absolute
# The 90-D canonical layout is applied later, during packing.
# A finished run writes:
#   ${LIBERO_PLUS_DATA_ROOT}/dataset_absolute/meta/stats.json
#   ${LIBERO_PLUS_DATA_ROOT}/dataset_absolute/meta/relative_stats.json

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export GALAXEA_DATA_ROOT="${LIBERO_PLUS_DATA_ROOT:-/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1}"
export DATA_MIX="${DATA_MIX:-libero_plus_stage1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

python -m starVLA.dataloader.precompute_dataset_stats
