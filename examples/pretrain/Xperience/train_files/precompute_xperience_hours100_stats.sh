#!/usr/bin/env bash
# Precompute dataset stats before Xperience hours100 pretraining.
#
# Rank 0 otherwise builds meta/stats.json and meta/relative_stats.json inside
# an NCCL barrier. This mixture is one dataset, so the work is a single
# process. generate_stats rewrites meta/stats.json when __fingerprints__ is
# missing. Relative stats cover left_eef and right_eef only. Fingertips stay
# absolute. The 90-D canonical layout is applied later, during packing.
#
#   conda activate MRA
#   bash examples/pretrain/Xperience/train_files/precompute_xperience_hours100_stats.sh
#
# Paths match run_xperience_hours100.sh. Override if needed:
#   DATA_ROOT=/path/to/stage1 DATA_MIX=xperience_hours100 bash ...
#
# A finished dataset writes:
#   ${DATA_ROOT}/hours100/meta/stats.json
#   ${DATA_ROOT}/hours100/meta/relative_stats.json

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export DATA_ROOT="${DATA_ROOT:-/mnt/zhaoshijie/dataset/xperience_runs/stage1}"
export DATA_MIX="${DATA_MIX:-xperience_hours100}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

python -m starVLA.dataloader.precompute_dataset_stats
