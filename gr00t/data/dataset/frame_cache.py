# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uncompressed per-episode frame cache.

One ``uint8`` ``.npy`` per episode per camera, shape ``(T, H, W, 3)``.
``np.load(..., mmap_mode="r")`` then integer indexing copies only the
requested rows. A missing file is an error: callers must not fall back
to decoding the source video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def frame_cache_path(cache_dir: str | Path, video_key: str, episode_index: int) -> Path:
    """``{cache_dir}/{video_key}/episode_{episode_index:06d}.npy``."""
    return Path(cache_dir) / video_key / f"episode_{episode_index:06d}.npy"


def cache_is_complete(path: str | Path, length: int, size: int = 256) -> bool:
    """True when ``path`` is a uint8 array of shape ``(length, size, size, 3)``."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        frames = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (
        frames.ndim == 4
        and frames.shape == (int(length), int(size), int(size), 3)
        and frames.dtype == np.uint8
    )


def read_cached_frames(path: str | Path, indices: np.ndarray) -> np.ndarray:
    """Copy ``indices`` out of a memory-mapped episode cache.

    Raises:
        FileNotFoundError: the cache file is missing.
        IndexError: an index is outside ``[0, T)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Frame cache is missing: {path}. "
            "Precompute it before training; this path does not decode the source video."
        )
    frames = np.load(path, mmap_mode="r")
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size:
        lowest = int(indices.min())
        highest = int(indices.max())
        if lowest < 0 or highest >= frames.shape[0]:
            raise IndexError(
                f"Frame index {lowest}..{highest} is outside cache length "
                f"{frames.shape[0]} in {path}"
            )
    return np.ascontiguousarray(frames[indices])
