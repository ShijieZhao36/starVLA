#!/usr/bin/env bash
# Precompute dataset stats before Galaxea Open World stage1 training.
#
# Rank 0 otherwise builds meta/stats.json and meta/relative_stats.json inside
# an NCCL barrier. This mixture is 219 subsets, so the other ranks can hit the
# 30-minute process-group timeout. Run this once, with the training env
# active, before run_galaxea_open_world_stage1.sh:
#
#   conda activate MRA
#   bash examples/pretrain/GalaxeaOpenWorld/train_files/precompute_galaxea_open_world_stage1_stats.sh
#
# Paths match run_galaxea_open_world_stage1.sh. Override if needed:
#   GALAXEA_DATA_ROOT=/path/to/_processed DATA_MIX=galaxea_open_world_stage1 bash ...
#   WORKERS=32 bash ...
#
# Subsets run in parallel, one process per available CPU (this host: 168).
# Each process is pinned to one thread. Left and right EEF relative stats are
# computed in one pass. The 90-D canonical layout is applied later, during packing.
# A finished subset writes:
#   ${GALAXEA_DATA_ROOT}/<subset_id>/stage1/dataset/meta/stats.json
#   ${GALAXEA_DATA_ROOT}/<subset_id>/stage1/dataset/meta/relative_stats.json

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export GALAXEA_DATA_ROOT="${GALAXEA_DATA_ROOT:-/mnt/zhanghao/Galaxea-Open-World-Dataset/_processed}"
export DATA_MIX="${DATA_MIX:-galaxea_open_world_stage1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

python -m starVLA.dataloader.precompute_dataset_stats
