# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Precompute Galaxea stage1 256x256 frame caches.

One worker handles one episode of one camera. It decodes the 720p video in
small chunks, resizes into a preallocated uint8 array, and writes an
uncompressed ``.npy``. A file that already has the right shape is skipped.

Training ``num_workers`` is unrelated. This process count is ``CACHE_WORKERS``.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from gr00t.data.dataset.frame_cache import cache_is_complete, frame_cache_path


CAMERA_NAMES = ("front", "left_wrist", "right_wrist")
DEFAULT_SIZE = 256
DEFAULT_CHUNK = 32


def camera_names() -> tuple[str, ...]:
    """Cameras to cache. ``CACHE_CAMERAS`` is a comma-separated override."""
    raw = os.environ.get("CACHE_CAMERAS", "").strip()
    if not raw:
        return CAMERA_NAMES
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("CACHE_CAMERAS did not contain a camera name")
    return names


def _data_root_from_env() -> Path:
    """Prefer ``DATA_ROOT``. ``GALAXEA_DATA_ROOT`` remains the Galaxea default."""
    raw = os.environ.get("DATA_ROOT", "").strip() or os.environ.get("GALAXEA_DATA_ROOT", "").strip()
    if not raw:
        raise KeyError("Set DATA_ROOT or GALAXEA_DATA_ROOT")
    return Path(raw)


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
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        return


def _as_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        peak = float(frame.max()) if frame.size else 0.0
        if peak <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame


def _write_npy(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, frames)
    os.replace(temporary, path)


def cache_one(job: tuple) -> str:
    """Decode one episode camera into ``(T, size, size, 3)`` uint8."""
    video_path, out_path, length, size, chunk = job
    out_path = Path(out_path)
    length = int(length)
    size = int(size)
    chunk = int(chunk)
    if cache_is_complete(out_path, length, size):
        return f"skip {out_path}"

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if length <= 0:
        _write_npy(out_path, np.empty((0, size, size, 3), dtype=np.uint8))
        return f"write {out_path}"

    import cv2
    from torchcodec.decoders import VideoDecoder

    cv2.setNumThreads(1)
    decoder = VideoDecoder(
        str(video_path),
        device="cpu",
        dimension_order="NHWC",
        num_ffmpeg_threads=1,
    )
    n_frames = int(decoder.metadata.num_frames or 0)
    if n_frames < length:
        raise RuntimeError(
            f"{video_path} has {n_frames} frames, episode length is {length}"
        )

    cached = np.empty((length, size, size, 3), dtype=np.uint8)
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        decoded = decoder.get_frames_at(indices=list(range(start, stop))).data.numpy()
        if decoded.shape[0] != stop - start:
            raise RuntimeError(
                f"{video_path} returned {decoded.shape[0]} frames for {start}:{stop}"
            )
        for offset, frame in enumerate(decoded):
            resized = cv2.resize(
                _as_uint8_hwc(frame),
                (size, size),
                interpolation=cv2.INTER_AREA,
            )
            cached[start + offset] = resized
        del decoded
    del decoder
    _write_npy(out_path, cached)
    return f"write {out_path}"


def _original_video_keys(modality: dict, names: tuple[str, ...]) -> list[str]:
    video = modality.get("video", {})
    keys = []
    for name in names:
        if name not in video:
            raise KeyError(f"modality.json is missing video key {name!r}")
        keys.append(video[name].get("original_key", f"observation.images.{name}"))
    return keys


def build_jobs(data_root: Path, mixture: list, cache_root: Path, size: int, chunk: int) -> list[tuple]:
    import json

    jobs = []
    seen = set()
    for dataset_name, _weight, _robot_type in mixture:
        if dataset_name in seen:
            continue
        seen.add(dataset_name)
        dataset_path = data_root / dataset_name
        meta = dataset_path / "meta"
        with (meta / "info.json").open() as handle:
            info = json.load(handle)
        with (meta / "modality.json").open() as handle:
            modality = json.load(handle)
        video_keys = _original_video_keys(modality, camera_names())
        chunks_size = int(info["chunks_size"])
        pattern = info["video_path"]
        subset_id = Path(dataset_name).parts[0]
        with (meta / "episodes.jsonl").open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                episode = json.loads(line)
                episode_index = int(episode["episode_index"])
                length = int(episode["length"])
                episode_chunk = episode_index // chunks_size
                for video_key in video_keys:
                    video_path = dataset_path / pattern.format(
                        episode_chunk=episode_chunk,
                        video_key=video_key,
                        episode_index=episode_index,
                    )
                    out_path = frame_cache_path(cache_root / subset_id, video_key, episode_index)
                    jobs.append((str(video_path), str(out_path), length, size, chunk))
    return jobs


def main() -> None:
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

    data_root = _data_root_from_env()
    data_mix = os.environ["DATA_MIX"]
    names = camera_names()
    cache_root = Path(os.environ["FRAME_CACHE_DIR"])
    size = int(os.environ.get("CACHE_SIZE", DEFAULT_SIZE))
    chunk = int(os.environ.get("CACHE_CHUNK", DEFAULT_CHUNK))
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"Unknown data_mix={data_mix!r}. Available: {sorted(DATASET_NAMED_MIXTURES)}")

    jobs = build_jobs(data_root, DATASET_NAMED_MIXTURES[data_mix], cache_root, size, chunk)
    if not jobs:
        raise RuntimeError(f"No frame-cache jobs for mixture {data_mix}")

    requested = os.environ.get("CACHE_WORKERS", "").strip()
    if requested:
        workers = int(requested)
    else:
        workers = max(1, _available_cpus() - 8)
    workers = max(1, min(workers, len(jobs)))
    print(
        f"mixture {data_mix}: {len(jobs)} episode-cameras under {cache_root} "
        f"cameras={','.join(names)} with {workers} workers "
        f"({_available_cpus()} cpus, chunk {chunk})",
        flush=True,
    )

    context = multiprocessing.get_context("fork")
    done = 0
    written = 0
    skipped = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
    ) as pool:
        futures = [pool.submit(cache_one, job) for job in jobs]
        for future in as_completed(futures):
            status = future.result()
            done += 1
            if status.startswith("skip "):
                skipped += 1
            else:
                written += 1
            print(f"[{done}/{len(jobs)}] {status}", flush=True)
    print(f"finished {len(jobs)} jobs, wrote {written}, skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
