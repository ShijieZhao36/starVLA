# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Precompute per-dataset stats outside the training process group.

Each subset writes only its own ``meta/stats.json`` and ``meta/relative_stats.json``,
so subsets run in parallel. One worker process is used per available CPU.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        return


def precompute_one(dataset_path: str, robot_type: str) -> str:
    """Write absolute and relative stats for one LeRobot dataset."""
    from gr00t.data.stats import generate_stats
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.gr00t_sharded import (
        resolve_modality_configs,
        write_relative_action_stats,
    )

    path = Path(dataset_path)
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise KeyError(
            f"Unknown robot_type={robot_type!r} for dataset {path}. "
            f"Available: {sorted(ROBOT_TYPE_CONFIG_MAP)}"
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    # Same config training uses when include_state is false.
    modality_configs = resolve_modality_configs(
        ROBOT_TYPE_CONFIG_MAP[robot_type],
        robot_type,
        prefer_registry=False,
    )
    generate_stats(path)
    write_relative_action_stats(path, modality_configs)
    relative_path = path / "meta" / "relative_stats.json"
    if not relative_path.is_file():
        raise FileNotFoundError(f"expected relative stats at {relative_path}")
    return str(path)


def _data_root_from_env() -> Path:
    """Prefer ``DATA_ROOT``. ``GALAXEA_DATA_ROOT`` remains the Galaxea default."""
    raw = os.environ.get("DATA_ROOT", "").strip() or os.environ.get("GALAXEA_DATA_ROOT", "").strip()
    if not raw:
        raise KeyError("Set DATA_ROOT or GALAXEA_DATA_ROOT")
    return Path(raw)


def main() -> None:
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

    data_root = _data_root_from_env()
    data_mix = os.environ["DATA_MIX"]
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"Unknown data_mix={data_mix!r}. Available: {sorted(DATASET_NAMED_MIXTURES)}")

    jobs: list[tuple[str, str]] = []
    seen = set()
    for dataset_name, _weight, robot_type in DATASET_NAMED_MIXTURES[data_mix]:
        key = (dataset_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        jobs.append((str(data_root / dataset_name), robot_type))

    requested = int(os.environ.get("WORKERS", "0"))
    workers = requested if requested > 0 else min(len(jobs), _available_cpus())
    workers = max(1, min(workers, len(jobs)))
    print(
        f"mixture {data_mix}: {len(jobs)} datasets under {data_root} "
        f"with {workers} workers ({_available_cpus()} cpus)",
        flush=True,
    )

    context = multiprocessing.get_context("fork")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        futures = [pool.submit(precompute_one, dataset_path, robot_type) for dataset_path, robot_type in jobs]
        for future in as_completed(futures):
            dataset_path = future.result()
            done += 1
            print(f"[{done}/{len(jobs)}] wrote {dataset_path}/meta/relative_stats.json", flush=True)


if __name__ == "__main__":
    main()
