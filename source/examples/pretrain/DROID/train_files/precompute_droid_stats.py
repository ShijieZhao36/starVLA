#!/usr/bin/env python3
"""Exact, episode-parallel DROID statistics compatible with gr00t_sharded.

Only numeric columns are read. Workers write disjoint episode ranges into
temporary float32 memmaps, then independent coordinate reductions compute the
global percentiles exactly. Relative statistics retain their (horizon, 9)
shape and use the training loader's SE(3) conversion and fingerprints.

Run --dry-run first. Real execution requires an explicit --workers value.
Do not run another statistics writer or start training on this dataset until
this command succeeds. The lock serializes invocations of this command; the
older, generic training-time writer does not acquire this lock.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

# Set these before importing numpy or any repository/Arrow modules.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_WORKER = None
_RUNTIME = None


def load_runtime(robot_type):
    """Import the exact implementations that the training loader uses."""
    from gr00t.data import stats as stats_module
    from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType
    from starVLA.dataloader import gr00t_sharded as sharded
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP

    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise KeyError(f"Unknown robot type {robot_type!r}; install the DROID registry first")
    modalities = sharded.resolve_modality_configs(
        ROBOT_TYPE_CONFIG_MAP[robot_type], robot_type, prefer_registry=False
    )
    action, state = modalities["action"], modalities["state"]
    if list(action.delta_indices) != list(range(8)) or list(state.delta_indices) != [0]:
        raise ValueError("DROID requires the approved action horizon 8 and current-state index [0]")
    if list(action.modality_keys) != ["arm_eef", "arm_gripper", "arm_joints"]:
        raise ValueError("DROID statistics require all 17 approved action dimensions")
    pairs = [
        (key, config)
        for key, config in zip(action.modality_keys, action.action_configs or [])
        if config.rep == ActionRepresentation.RELATIVE
    ]
    if len(pairs) != 1 or pairs[0][0] != "arm_eef":
        raise ValueError("Only DROID arm_eef must use relative actions")
    key, config = pairs[0]
    if config.type != ActionType.EEF or config.format != ActionFormat.XYZ_ROT6D:
        raise ValueError("DROID arm_eef must be EEF XYZ_ROT6D")
    sharded._check_relative_alignment(modalities, key)
    return dict(
        stale=stats_module._stale_features,
        feature_fingerprint=stats_module._compute_stats_fingerprint,
        load=stats_module._load_stats_cache,
        dump=stats_module._dump_stats_cache_atomic,
        relative_windows=sharded._relative_windows,
        relative_config=config,
        relative_fingerprint=sharded._relative_fingerprint(action, state, key, config),
        action_deltas=np.asarray(action.delta_indices, dtype=np.int64),
        state_offset=int(state.delta_indices[-1]),
    )


def read_metadata(dataset):
    info = json.loads((dataset / "meta/info.json").read_text())
    modality = json.loads((dataset / "meta/modality.json").read_text())
    for kind in ("state", "action"):
        expected = {"arm_eef": (0, 9), "arm_gripper": (9, 10), "arm_joints": (10, 17)}
        for name, bounds in expected.items():
            spec = modality[kind][name]
            if (spec["start"], spec["end"]) != bounds:
                raise ValueError(f"Unexpected {kind}.{name} slice: {spec}")
        if modality[kind]["arm_eef"].get("rotation_type") != "rotation_6d":
            raise ValueError(f"{kind}.arm_eef must use rotation_6d")
    features = {key: value for key, value in info["features"].items() if "float" in value["dtype"]}
    for key, spec in features.items():
        if len(spec["shape"]) != 1 or int(spec["shape"][0]) < 1:
            raise ValueError(f"Only scalar/vector numeric features are supported: {key}: {spec}")
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [17]:
            raise ValueError(f"{key} must have shape [17]")
    episodes = []
    with (dataset / "meta/episodes.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                episode = json.loads(line)
                episodes.append((int(episode["episode_index"]), int(episode["length"])))
    episodes.sort()
    if len({index for index, _ in episodes}) != len(episodes):
        raise ValueError("Duplicate episode indices")
    if any(length <= 0 for _, length in episodes):
        raise ValueError("Episode lengths must be positive")
    if len(episodes) != int(info["total_episodes"]) or sum(length for _, length in episodes) != int(info["total_frames"]):
        raise ValueError("episodes.jsonl totals disagree with info.json")
    return info, modality, features, episodes


@contextmanager
def exclusive_lock(path):
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another DROID stats process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def bounded_results(pool, function, jobs, max_pending):
    """Bound submitted jobs so a 64k-episode dataset does not queue huge futures."""
    iterator = iter(jobs)
    pending = set()
    for _ in range(max_pending):
        job = next(iterator, None)
        if job is None:
            break
        pending.add(pool.submit(function, job))
    while pending:
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            yield future.result()
            job = next(iterator, None)
            if job is not None:
                pending.add(pool.submit(function, job))


def column_float32(column, rows, width, name):
    """Arrow lists/scalars -> float32, matching upstream np.vstack coercion."""
    import pyarrow as pa

    values = column.combine_chunks()
    if values.null_count:
        raise ValueError(f"Null values in {name}")
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        offsets = np.asarray(values.offsets)
        if not np.all(np.diff(offsets) == width):
            raise ValueError(f"Inconsistent vector lengths in {name}; expected {width}")
        array = np.asarray(values.flatten(), dtype=np.float32).reshape(rows, width)
    elif pa.types.is_fixed_size_list(values.type):
        if values.type.list_size != width:
            raise ValueError(f"Wrong vector width in {name}")
        array = np.asarray(values.flatten(), dtype=np.float32).reshape(rows, width)
    else:
        array = np.asarray(values, dtype=np.float32).reshape(rows, width)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {name}")
    return array


def init_worker(config):
    global _WORKER
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    _WORKER = dict(config)
    _WORKER["arrays"] = {
        key: np.load(spec["path"], mmap_mode="r+") for key, spec in config["maps"].items()
    }


def scan_episode_batch(episodes):
    import pyarrow.parquet as pq

    config = _WORKER
    for episode_index, length, frame_start, window_start in episodes:
        relative_path = config["pattern"].format(
            episode_chunk=episode_index // config["chunks_size"], episode_index=episode_index
        )
        path = Path(config["dataset"]) / relative_path
        table = pq.read_table(path, columns=config["columns"], use_threads=False)
        if table.num_rows != length:
            raise ValueError(f"{path}: expected {length} rows, found {table.num_rows}")
        arrays = {
            name: column_float32(table[name], length, config["widths"][name], name)
            for name in config["columns"]
        }
        for name in config["absolute"]:
            config["arrays"][name][:, frame_start : frame_start + length] = arrays[name].T
        if config["relative"]:
            block = _RUNTIME["relative_windows"](
                arrays["action"][:, :9], arrays["observation.state"][:, :9],
                _RUNTIME["action_deltas"], _RUNTIME["state_offset"], _RUNTIME["relative_config"],
            )
            if block is not None:
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite relative poses in {path}")
                expected = max(0, length - int(_RUNTIME["action_deltas"][-1]))
                if block.shape != (expected, 8, 9):
                    raise ValueError(f"Unexpected relative block shape: {block.shape}")
                config["arrays"]["__relative__"][:, window_start : window_start + expected] = block.reshape(expected, -1).T
    return len(episodes)


def summarize_coordinate(job):
    """Exact global linear percentiles; working RAM is one float32 coordinate."""
    key, path, coordinate = job
    matrix = np.load(path, mmap_mode="r")
    # A private contiguous copy lets quantile partition in place without
    # modifying the shared memmap or allocating another whole coordinate.
    values = np.array(matrix[coordinate], dtype=np.float32, copy=True)
    result = dict(
        min=float(np.min(values)), max=float(np.max(values)),
        # Float32 inputs match training. Float64 accumulation avoids drift
        # when reducing 16.8 million rows; std is population std (ddof=0).
        mean=float(np.mean(values, dtype=np.float64)),
        std=float(np.std(values, dtype=np.float64)),
    )
    quantiles = np.quantile(values, [0.01, 0.99], overwrite_input=True)
    result.update(q01=float(quantiles[0]), q99=float(quantiles[1]))
    return key, coordinate, result


def valid_summary(entry, shape):
    if not isinstance(entry, dict):
        return False
    for name in ("min", "max", "mean", "std", "q01", "q99"):
        try:
            value = np.asarray(entry[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        if value.shape != tuple(shape) or not np.isfinite(value).all():
            return False
    return True


def cache_status(runtime, features, absolute, relative):
    """Read-only training gate: fingerprints, dimensions and finite summaries."""
    stale = set(runtime["stale"](absolute, features, list(features)))
    stale.update(key for key, spec in features.items() if not valid_summary(absolute.get(key), spec["shape"]))
    fingerprints = relative.get("__fingerprints__", {})
    relative_stale = (
        not isinstance(fingerprints, dict)
        or fingerprints.get("arm_eef") != runtime["relative_fingerprint"]
        or not valid_summary(relative.get("arm_eef"), (8, 9))
    )
    obsolete = [key for key, value in absolute.items() if key != "__fingerprints__" and key not in features and isinstance(value, dict)]
    abs_fps = absolute.get("__fingerprints__", {})
    if isinstance(abs_fps, dict):
        obsolete += [f"__fingerprints__.{key}" for key in abs_fps if key not in features]
    return [key for key in features if key in stale], relative_stale, obsolete


def allocate_maps(directory, features, stale, frames, windows, relative):
    specs = {}
    for index, name in enumerate(stale):
        specs[name] = dict(path=str(directory / f"absolute_{index:03d}.npy"), shape=(int(features[name]["shape"][0]), frames), output_shape=(int(features[name]["shape"][0]),))
    if relative:
        if windows <= 0:
            raise ValueError("No full eight-step relative-action windows")
        specs["__relative__"] = dict(path=str(directory / "relative.npy"), shape=(8 * 9, windows), output_shape=(8, 9))
    for spec in specs.values():
        mapped = np.lib.format.open_memmap(spec["path"], mode="w+", dtype=np.float32, shape=spec["shape"])
        mapped.flush()
        del mapped
    return specs


def run(args, runtime=None):
    global _RUNTIME
    dataset = args.dataset_dir.resolve()
    info, _modality, features, episodes = read_metadata(dataset)
    frames = sum(length for _, length in episodes)
    windows = sum(max(0, length - 7) for _, length in episodes)
    upper_bytes = 4 * (frames * sum(int(spec["shape"][0]) for spec in features.values()) + windows * 8 * 9)
    if args.dry_run:
        print(json.dumps(dict(
            dry_run=True, dataset=str(dataset), robot_type=args.robot_type,
            episodes=len(episodes), frames=frames, relative_windows=windows, horizon=8,
            numeric_features=list(features), workers=args.workers,
            summary_workers=args.summary_workers,
            temporary_disk_upper_gib=round(upper_bytes / 2**30, 3),
            ram_per_summary_worker_mib=round(3 * max(frames, windows) * 4 / 2**20, 1),
            temp_dir=str(args.temp_dir or dataset / "meta"),
            outputs=[str(dataset / "meta/stats.json"), str(dataset / "meta/relative_stats.json")],
            note="Exact global q01/q99; runtime checks existing fingerprints and skips fresh entries. No writes in dry-run.",
        ), indent=2))
        return
    runtime = runtime or load_runtime(args.robot_type)
    meta = dataset / "meta"
    if args.check_only:
        absolute = runtime["load"](meta / "stats.json")
        relative = runtime["load"](meta / "relative_stats.json")
        stale, relative_stale, obsolete = cache_status(runtime, features, absolute, relative)
        if stale or relative_stale or obsolete:
            raise RuntimeError(
                f"cached_not_ready: stale_absolute={stale}, stale_relative={relative_stale}, "
                f"obsolete_entries={obsolete}; run the DROID statistics preparation script first"
            )
        print("cached_ready", flush=True)
        return
    if args.workers is None or args.workers < 1:
        raise ValueError("Real execution requires --workers N with N >= 1")
    if args.episodes_per_task < 1 or (args.summary_workers is not None and args.summary_workers < 1):
        raise ValueError("Batch size and summary workers must be positive")
    _RUNTIME = runtime  # inherited by fork workers; no heavy imports per episode
    with exclusive_lock(meta / ".droid_stats.lock"):
        existing = runtime["load"](meta / "stats.json")
        relative_stats = runtime["load"](meta / "relative_stats.json")
        stale, relative, _obsolete = cache_status(runtime, features, existing, relative_stats)
        if args.force:
            stale, relative = list(features), True
        relative_fps = relative_stats.get("__fingerprints__", {})
        if not isinstance(relative_fps, dict):
            relative_fps = {}
        # Match generate_stats cleanup so it will neither recompute nor rewrite.
        cleaned = {key: value for key, value in existing.items() if key == "__fingerprints__" or key in features or not isinstance(value, dict)}
        fingerprints = existing.get("__fingerprints__", {})
        if not isinstance(fingerprints, dict):
            fingerprints = {}
        cleaned["__fingerprints__"] = {key: value for key, value in fingerprints.items() if key in features}
        if not stale and not relative:
            if cleaned != existing:
                runtime["dump"](meta / "stats.json", cleaned)
            print("All absolute and relative fingerprints are current; no scan needed.", flush=True)
            return
        temp_parent = (args.temp_dir or meta).resolve()
        temp_parent.mkdir(parents=True, exist_ok=True)
        needed = 4 * (frames * sum(int(features[key]["shape"][0]) for key in stale) + (windows * 8 * 9 if relative else 0))
        if shutil.disk_usage(temp_parent).free < needed + 1024**3:
            raise RuntimeError(f"Need {needed / 2**30:.2f} GiB plus 1 GiB free under {temp_parent}")
        with tempfile.TemporaryDirectory(prefix=".droid_stats_", dir=temp_parent) as temporary:
            maps = allocate_maps(Path(temporary), features, stale, frames, windows, relative)
            columns = list(dict.fromkeys(stale + (["action", "observation.state"] if relative else [])))
            config = dict(
                dataset=str(dataset), pattern=info["data_path"], chunks_size=int(info["chunks_size"]),
                columns=columns, widths={key: int(features[key]["shape"][0]) for key in columns},
                absolute=stale, relative=relative, maps=maps,
            )
            offsets = []
            frame_offset = window_offset = 0
            for index, length in episodes:
                offsets.append((index, length, frame_offset, window_offset))
                frame_offset += length
                window_offset += max(0, length - 7)
            batches = [offsets[i : i + args.episodes_per_task] for i in range(0, len(offsets), args.episodes_per_task)]
            workers = min(args.workers, len(batches))
            context = multiprocessing.get_context("fork")
            print(f"Scanning {len(episodes)} episodes with {workers} processes; numeric columns={len(columns)}, temp={needed / 2**30:.2f} GiB", flush=True)
            done = 0
            last_log = time.monotonic()
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(config,)) as pool:
                for count in bounded_results(pool, scan_episode_batch, batches, max_pending=2 * workers):
                    done += count
                    now = time.monotonic()
                    if now - last_log > 10 or done == len(episodes):
                        print(f"Scanned {done}/{len(episodes)} episodes", flush=True)
                        last_log = now
            jobs = [(key, spec["path"], coordinate) for key, spec in maps.items() for coordinate in range(spec["shape"][0])]
            summary_workers = min(args.summary_workers or args.workers, len(jobs))
            results = {key: {name: np.empty(spec["shape"][0], dtype=np.float64) for name in ("min", "max", "mean", "std", "q01", "q99")} for key, spec in maps.items()}
            print(f"Computing {len(jobs)} exact coordinate summaries with {summary_workers} processes", flush=True)
            with ProcessPoolExecutor(max_workers=summary_workers, mp_context=context) as pool:
                for key, coordinate, values in bounded_results(pool, summarize_coordinate, jobs, max_pending=2 * summary_workers):
                    for stat_name, value in values.items():
                        results[key][stat_name][coordinate] = value
            for key, values in results.items():
                reshaped = {name: value.reshape(maps[key]["output_shape"]).tolist() for name, value in values.items()}
                if key == "__relative__":
                    relative_stats["arm_eef"] = reshaped
                    relative_fps["arm_eef"] = runtime["relative_fingerprint"]
                else:
                    cleaned[key] = reshaped
                    cleaned["__fingerprints__"][key] = runtime["feature_fingerprint"](key, features[key])
            # Publish only after all scans and exact reductions have succeeded.
            # Each JSON is atomic; a later restart reuses a published file if the
            # process stops between these two independent cache replacements.
            if stale or cleaned != existing:
                runtime["dump"](meta / "stats.json", cleaned)
            if relative:
                relative_stats["__fingerprints__"] = relative_fps
                runtime["dump"](meta / "relative_stats.json", relative_stats)
        stale, relative_stale, obsolete = cache_status(
            runtime, features, runtime["load"](meta / "stats.json"), runtime["load"](meta / "relative_stats.json")
        )
        if stale or relative_stale or obsolete:
            raise RuntimeError("Published statistics failed fingerprint/shape verification")
        print(f"Finished exact statistics: {meta / 'stats.json'} and {meta / 'relative_stats.json'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-type", default="droid_franka_rot6d")
    parser.add_argument("--workers", type=int, help="Episode scan processes; required unless --dry-run")
    parser.add_argument("--summary-workers", type=int, help="Exact coordinate reducers; default workers, capped by coordinate count")
    parser.add_argument("--episodes-per-task", type=int, default=32)
    parser.add_argument("--temp-dir", type=Path, help="Temporary mmap parent; default dataset/meta")
    parser.add_argument("--force", action="store_true", help="Recompute even when fingerprints match")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read metadata and print an upper-bound plan without writing")
    mode.add_argument("--check-only", action="store_true", help="Strictly read-only readiness check; print cached_ready and exit 0 only on valid caches")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
