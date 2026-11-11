#!/usr/bin/env python3
"""Read-only loader-compatible statistics and frame-cache checks before training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pretrain", "posttrain"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    from gr00t.data.stats import _stale_features
    from gr00t.data.types import ActionRepresentation
    from gr00t.data.dataset.frame_cache import frame_cache_path, cache_is_complete
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.gr00t_sharded import resolve_modality_configs, _relative_fingerprint

    root = args.dataset.resolve(strict=True)
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    absolute = json.loads((root / "meta/stats.json").read_text())
    relative = json.loads((root / "meta/relative_stats.json").read_text())
    fields = {key: value for key, value in info["features"].items() if "float" in value["dtype"]}
    stale = _stale_features(absolute, info["features"], list(fields))
    require(not stale, f"Statistics are missing/stale: {stale}. Run the precompute script first.")
    for key, feature in fields.items():
        for name in ["min", "max", "mean", "std", "q01", "q99"]:
            value = np.asarray(absolute[key][name])
            require(value.shape == tuple(feature["shape"]) and np.isfinite(value).all(), f"Invalid {key}.{name} statistics")
    tag = "droid_franka_rot6d" if args.stage == "pretrain" else "libero_plus_rot6d"
    cfg = resolve_modality_configs(ROBOT_TYPE_CONFIG_MAP[tag], tag, prefer_registry=False)
    action, state = cfg["action"], cfg["state"]
    for key, settings in zip(action.modality_keys, action.action_configs):
        if settings.rep != ActionRepresentation.RELATIVE:
            continue
        expected = _relative_fingerprint(action, state, key, settings)
        require(relative.get("__fingerprints__", {}).get(key) == expected, f"Stale relative statistics: {key}")
        width = modality["action"][key]["end"] - modality["action"][key]["start"]
        for name in ["min", "max", "mean", "std", "q01", "q99"]:
            value = np.asarray(relative[key][name])
            require(value.shape == (8, width) and np.isfinite(value).all(), f"Invalid relative {key}.{name}")

    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    require(len(episodes) == info["total_episodes"], "Episode count mismatch")
    cameras = [key.removeprefix("video.") for key in cfg["video"].modality_keys]
    if args.stage == "pretrain":
        marker_path = args.cache / "droid_cache_complete.json"
        require(marker_path.is_file(), f"Missing {marker_path}; run precompute_droid_frame_cache.sh to completion")
        marker = json.loads(marker_path.read_text())
        require(marker.get("validated") is True, "Cache is not marked complete")
        require(Path(marker["dataset"]).resolve() == root, "Cache dataset differs")
        require(marker["cameras"] == cameras and marker["size"] == 256, "Cache cameras/size differ from training")
        require(marker["episodes"] == len(episodes) and marker["frames"] == info["total_frames"], "Cache count mismatch")
        require(marker["job_count"] == len(episodes) * len(cameras), "Incomplete cache job count")
        for name in ["info", "episodes", "modality"]:
            suffix = "jsonl" if name == "episodes" else "json"
            require(marker[f"{name}_sha256"] == digest(root / f"meta/{name}.{suffix}"), f"Dataset {name} changed after caching")
        # The marker is only published after every writer/skip check succeeds.
        # Check endpoints again to catch common accidental path/copy mistakes.
        selected = [episodes[0], episodes[-1]]
    else:
        # Existing LIBERO caches predate the DROID completion marker. Inspect all
        # headers without decoding videos or loading pixel arrays into RAM.
        selected = episodes
    jobs = []
    for episode in selected:
        for camera in cameras:
            video_key = modality["video"][camera].get("original_key", f"observation.images.{camera}")
            path = frame_cache_path(args.cache / root.name, video_key, episode["episode_index"])
            jobs.append((path, episode["length"]))
    def check_one(job):
        path, length = job
        require(cache_is_complete(path, length, 256), f"Missing or invalid frame cache: {path}")
    with ThreadPoolExecutor(max_workers=32) as pool:
        for _ in pool.map(check_one, jobs):
            pass
    print(json.dumps({"prepared": True, "stage": args.stage, "dataset": str(root), "robot_type": tag,
                      "stats_features": len(fields), "cameras": cameras, "cache_headers_checked": len(jobs)}))


if __name__ == "__main__":
    main()
