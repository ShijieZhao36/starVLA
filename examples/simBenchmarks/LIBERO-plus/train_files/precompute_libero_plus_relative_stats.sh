#!/usr/bin/env bash
# Precompute dataset stats before LIBERO-plus gr00t_sharded training.
#
# Rank 0 otherwise builds meta/relative_stats.json inside an NCCL barrier.
# This mixture is 14347 episodes / ~2.2M frames, so the other ranks hit the
# 30-minute process-group timeout. Run this once, with the training env
# active, before run_libero_plus_train_gr00t.sh:
#
#   bash examples/simBenchmarks/LIBERO-plus/train_files/precompute_libero_plus_relative_stats.sh
#
# Paths match run_libero_plus_train_gr00t.sh. Override if needed:
#   LIBERO_PLUS_DATA_ROOT=/path/to/stage1 DATA_MIX=libero_plus_stage1 bash ...
#
# The relative-stats pass is single-process and CPU-only. It can take hours.
# A finished run writes:
#   ${LIBERO_PLUS_DATA_ROOT}/dataset_absolute/meta/relative_stats.json

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${repo_root}"

export LIBERO_PLUS_DATA_ROOT="${LIBERO_PLUS_DATA_ROOT:-/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1}"
export DATA_MIX="${DATA_MIX:-libero_plus_stage1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

python - <<'PY'
import logging
import os
from pathlib import Path

from gr00t.data.stats import generate_stats
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_sharded import (
    resolve_modality_configs,
    write_relative_action_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

data_root = Path(os.environ["LIBERO_PLUS_DATA_ROOT"])
data_mix = os.environ["DATA_MIX"]
if data_mix not in DATASET_NAMED_MIXTURES:
    raise KeyError(
        f"Unknown data_mix={data_mix!r}. Available: {sorted(DATASET_NAMED_MIXTURES)}"
    )

print(f"mixture {data_mix}: {DATASET_NAMED_MIXTURES[data_mix]}")
seen = set()
for dataset_name, _weight, robot_type in DATASET_NAMED_MIXTURES[data_mix]:
    key = (dataset_name, robot_type)
    if key in seen:
        continue
    seen.add(key)
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise KeyError(
            f"Unknown robot_type={robot_type!r} for dataset {dataset_name!r}. "
            f"Available: {sorted(ROBOT_TYPE_CONFIG_MAP)}"
        )
    dataset_path = data_root / dataset_name
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Same config training uses when modality_config_path is unset.
    modality_configs = resolve_modality_configs(
        ROBOT_TYPE_CONFIG_MAP[robot_type],
        robot_type,
        prefer_registry=False,
    )
    print(f"dataset: {dataset_path}")
    generate_stats(dataset_path)
    write_relative_action_stats(dataset_path, modality_configs)

    stats_path = dataset_path / "meta" / "relative_stats.json"
    if not stats_path.is_file():
        raise SystemExit(f"expected relative stats at {stats_path}")
    print(f"wrote {stats_path}")
PY
