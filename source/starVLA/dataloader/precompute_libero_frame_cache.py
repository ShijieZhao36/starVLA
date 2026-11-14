# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Precompute native-resolution frame caches for LIBERO-plus stage1.

Videos are already 256x256. Each file is uint8 ``(T, H, W, 3)`` and is not
resized. A file that already has the episode length and that spatial size
is skipped.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from gr00t.data.dataset.frame_cache import cache_is_complete, frame_cache_path
from starVLA.dataloader.precompute_galaxea_frame_cache import (
    _as_uint8_hwc,
    _available_cpus,
    _init_worker,
    _write_npy,
)


CAMERA_NAMES = ("front", "right_wrist")
DEFAULT_CHUNK = 32


def cache_one(job: tuple) -> str:
    """Decode one episode camera at its native square resolution."""
    video_path, out_path, length, chunk = job
    out_path = Path(out_path)
    length = int(length)
    chunk = int(chunk)
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

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
        raise RuntimeError(f"{video_path} has {n_frames} frames, episode length is {length}")
    probe = _as_uint8_hwc(decoder.get_frames_at(indices=[0]).data.numpy()[0])
    height, width = int(probe.shape[0]), int(probe.shape[1])
    if height != width:
        raise ValueError(f"{video_path} frame is {height}x{width}; cache expects a square frame")
    size = height
    if cache_is_complete(out_path, length, size):
        del decoder
        return f"skip {out_path}"
    if length <= 0:
        _write_npy(out_path, np.empty((0, size, size, 3), dtype=np.uint8))
        del decoder
        return f"write {out_path}"

    cached = np.empty((length, size, size, 3), dtype=np.uint8)
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        decoded = decoder.get_frames_at(indices=list(range(start, stop))).data.numpy()
        if decoded.shape[0] != stop - start:
            raise RuntimeError(f"{video_path} returned {decoded.shape[0]} frames for {start}:{stop}")
        for offset, frame in enumerate(decoded):
            image = _as_uint8_hwc(frame)
            if image.shape[0] != size or image.shape[1] != size:
                raise ValueError(
                    f"{video_path} frame {start + offset} is {image.shape[0]}x{image.shape[1]}, expected {size}"
                )
            cached[start + offset] = image
        del decoded
    del decoder
    _write_npy(out_path, cached)
    return f"write {out_path}"


def _original_video_keys(modality: dict) -> list[str]:
    video = modality.get("video", {})
    keys = []
    for name in CAMERA_NAMES:
        if name not in video:
            raise KeyError(f"modality.json is missing video key {name!r}")
        keys.append(video[name].get("original_key", f"observation.images.{name}"))
    return keys


def build_jobs(data_root: Path, mixture: list, cache_root: Path, chunk: int) -> list[tuple]:
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
        video_keys = _original_video_keys(modality)
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
                    jobs.append((str(video_path), str(out_path), length, chunk))
    return jobs


def main() -> None:
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

    data_root = Path(os.environ["LIBERO_PLUS_DATA_ROOT"])
    data_mix = os.environ["DATA_MIX"]
    cache_root = Path(os.environ["FRAME_CACHE_DIR"])
    chunk = int(os.environ.get("CACHE_CHUNK", DEFAULT_CHUNK))
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"Unknown data_mix={data_mix!r}. Available: {sorted(DATASET_NAMED_MIXTURES)}")

    jobs = build_jobs(data_root, DATASET_NAMED_MIXTURES[data_mix], cache_root, chunk)
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
        f"with {workers} workers ({_available_cpus()} cpus, chunk {chunk})",
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
