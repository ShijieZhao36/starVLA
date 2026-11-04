#!/usr/bin/env python3
"""Cache a fixed real Xperience batch using production GR00T preprocessing.

CPU only; existing dataset, statistics, frame cache and repository are read-only.
No full sharded dataset construction and no statistics generation are performed.
Run with the repository's CRA Python and an isolated --output directory.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/mnt/zhaoshijie/workspace/starVLA")
    parser.add_argument("--config", default="examples/pretrain/Xperience/train_files/starvla_xperience_hours100.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--episodes", help="Optional comma-separated dataset episode IDs, exactly --count entries")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.gr00t_sharded import StatsTag, resolve_modality_configs, _dataset_frame_cache_dir, _relative_fingerprint
    from starVLA.dataloader.gr00t_sharded_processor import StarVLAPackProcessor
    from starVLA.dataloader.canonical_layout import load_layout, load_robot_maps
    torch.set_num_threads(1)

    config_path = Path(args.config).resolve()
    cfg = OmegaConf.load(config_path)
    dc = cfg.datasets.vla_data
    assert args.count in (4, 8), "Use a fixed batch of 4 or 8 samples"
    assert dc.include_state is False and dc.use_relative_action is True and dc.use_percentiles is True
    mixture = DATASET_NAMED_MIXTURES[dc.data_mix]
    assert len(mixture) == 1, "This preparation script intentionally matches the single Xperience mixture"
    subset, weight, robot_type = mixture[0]
    dataset_path = (Path(dc.data_root_dir) / subset).resolve()
    assert output != dataset_path and dataset_path not in output.parents, "Never write into the source dataset"
    assert not (output / "batch.pt").exists(), "Use a fresh output directory to preserve prior experiment caches"
    modalities = resolve_modality_configs(ROBOT_TYPE_CONFIG_MAP[robot_type], robot_type)
    assert list(modalities["action"].delta_indices) == list(range(8))
    assert list(modalities["video"].delta_indices) == [0]
    assert list(modalities["video"].modality_keys) == ["front"]
    cache = Path(_dataset_frame_cache_dir(dc.frame_cache_dir, subset)).resolve()
    loader = LeRobotEpisodeLoader(dataset_path, modalities, frame_cache_dir=cache)
    relative_file = dataset_path / "meta/relative_stats.json"
    relative_raw = json.loads(relative_file.read_text())
    for key, action_config in zip(modalities["action"].modality_keys, modalities["action"].action_configs):
        if str(action_config.rep.value).lower() != "relative":
            continue
        expected = _relative_fingerprint(modalities["action"], modalities["state"], key, action_config)
        recorded = relative_raw.get("__fingerprints__", {}).get(key)
        assert recorded == expected, f"Relative statistics fingerprint mismatch for {key}: {recorded} != {expected}"

    # Exactly the production mixture's statistics path for its sole dataset.
    per_dataset = loader.get_dataset_statistics()
    merged = {robot_type: {k: merge_statistics([v], [float(weight)], is_relative_stats=(k == "relative_action"))
                          for k, v in per_dataset.items() if k in ("state", "action", "relative_action")}}
    layout = load_layout(dc.canonical_layout)
    robot_maps = load_robot_maps(dc.canonical_robots, layout)
    processor = StarVLAPackProcessor({robot_type: modalities}, include_state=False,
        use_percentiles=True, clip_outliers=True, use_relative_action=True,
        canonical_layout=layout, robot_maps=robot_maps)
    processor.set_statistics(merged, override=True)
    expected_mask = np.zeros(90, dtype=np.float32)
    for start, end in ((7, 16), (17, 32), (39, 48), (49, 64)):
        expected_mask[start:end] = 1
    metadata_hashes = {str(dataset_path / "meta" / name): sha256(dataset_path / "meta" / name)
                       for name in ("info.json", "modality.json", "tasks.jsonl", "stats.json", "relative_stats.json")}
    rng = np.random.default_rng(args.seed)
    eligible = np.array([i for i, n in enumerate(loader.episode_lengths) if n >= 16])
    if args.episodes:
        wanted = [int(v) for v in args.episodes.split(",")]
        assert len(wanted) == args.count and len(set(wanted)) == args.count
        id_to_row = {int(m["episode_index"]): i for i, m in enumerate(loader.episodes_metadata)}
        rows = [id_to_row[eid] for eid in wanted]
    else:
        rows = rng.choice(eligible, args.count, replace=False).tolist()
    selection_path = dataset_path / "selection.json"
    selection = json.loads(selection_path.read_text())["episodes"] if selection_path.exists() else []
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    arrays, masks, images, instructions, provenance = [], [], [], [], []
    for index, row in enumerate(rows):
        episode_meta = loader.episodes_metadata[row]
        eid = int(episode_meta["episode_index"])
        length = int(loader.episode_lengths[row])
        # Same no-padding admissible range as ShardedSingleStepDataset.
        step_index = int(rng.integers(0, length - 8 + 1))
        episode = loader.load_episode(row, video_indices=np.array([step_index]))
        step = extract_step_data(episode, step_index, modalities, StatsTag(robot_type), allow_padding=False)
        sample = processor._pack_step(step)
        action = np.asarray(sample["action"], dtype=np.float32)
        mask = np.asarray(sample["action_mask"], dtype=np.float32)
        image = np.asarray(sample["image"][0].convert("RGB"), dtype=np.uint8)
        assert action.shape == mask.shape == (8, 90), (action.shape, mask.shape)
        assert np.isfinite(action).all() and np.isfinite(mask).all()
        assert np.array_equal(mask, np.broadcast_to(expected_mask, mask.shape))
        assert np.all(action[mask == 0] == 0)
        assert np.all(np.abs(action) <= 1.000001)
        assert image.shape == (256, 256, 3), image.shape
        assert len(sample["image"]) == 1 and "state" not in sample
        assert isinstance(sample["lang"], str) and sample["lang"].strip()
        image_path = output / "images" / f"sample_{index:02d}_episode_{eid:06d}_step_{step_index:04d}.png"
        Image.fromarray(image).save(image_path)
        chunk = eid // int(loader.chunk_size)
        parquet = dataset_path / loader.data_path_pattern.format(episode_chunk=chunk, episode_index=eid)
        original_video_key = loader.modality_meta["video"]["front"].get("original_key", "observation.images.front")
        cache_path = cache / original_video_key / f"episode_{eid:06d}.npy"
        assert cache_path.is_file(), cache_path
        source = selection[eid] if eid < len(selection) else {}
        provenance.append({"sample_index": index, "episode_index": eid, "episode_row": int(row),
            "step_index": step_index, "episode_length": length, "action_indices": list(range(step_index, step_index + 8)),
            "parquet_path": str(parquet), "parquet_sha256": sha256(parquet),
            "frame_cache_path": str(cache_path), "cached_image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "image_path": str(image_path), "instruction": sample["lang"], "source_selection": source,
            "source_local_frame_index": int(source.get("local_start", 0)) + step_index,
            "valid_action_dims_per_step": 48, "action_min": float(action.min()), "action_max": float(action.max())})
        arrays.append(action); masks.append(mask); images.append(image); instructions.append(sample["lang"])
    batch = {"actions": torch.from_numpy(np.stack(arrays)), "action_mask": torch.from_numpy(np.stack(masks)),
             "images": torch.from_numpy(np.stack(images)), "instructions": instructions, "provenance": provenance}
    torch.save(batch, output / "batch.pt")
    assert all(sha256(p) == h for p, h in metadata_hashes.items()), "Source metadata changed during preparation"
    manifest = {"repository": str(repo), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": str(config_path), "config_sha256": sha256(config_path), "seed": args.seed,
        "count": args.count, "dataset_path": str(dataset_path), "frame_cache_dir": str(cache),
        "normalization": "production StarVLAPackProcessor: relative dual EEF, absolute fingertips, q01/q99, clip=True",
        "metadata_sha256": metadata_hashes, "batch_sha256": sha256(output / "batch.pt"),
        "tensor_shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
        "samples": provenance, "cpu_only": True}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"batch": str(output / "batch.pt"), "manifest": str(output / "manifest.json"),
        "tensor_shapes": manifest["tensor_shapes"], "samples": [{"episode": p["episode_index"], "step": p["step_index"]} for p in provenance]}, indent=2))


if __name__ == "__main__":
    main()
