#!/usr/bin/env python3
"""Read-only data preflight using production metadata, stats merging and processor.

Requires prepared source stats and frame caches. Does not regenerate source stats,
start distributed training, initialize CUDA, or alter any source dataset.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path
sys.dont_write_bytecode = True

EXPECTED = {
    "galaxea_libero_ego50_A": {"galaxea_r1lite_stage1": 0.3, "libero_plus_rot6d": 0.2, "xperience_human": 0.5},
    "galaxea_libero_ego50_B": {"galaxea_r1lite_stage1": 0.4, "libero_plus_rot6d": 0.1, "xperience_human": 0.5},
}
DIMENSIONS = {"galaxea_r1lite_stage1": 41, "libero_plus_rot6d": 10, "xperience_human": 48}
VIEWS = {"galaxea_r1lite_stage1": 3, "libero_plus_rot6d": 2, "xperience_human": 1}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples-output", required=True)
    p.add_argument("--samples-per-source", type=int, default=3)
    args = p.parse_args()
    repo = Path(args.repo).resolve()
    config = Path(args.config).resolve()
    output = Path(args.output).resolve()
    samples_output = Path(args.samples_output).resolve()
    assert args.samples_per_source > 0
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
    from starVLA.dataloader.canonical_layout import load_layout, load_robot_maps
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.gr00t_lerobot.mixtures import resolve_dataset_mixture
    from starVLA.dataloader.gr00t_sharded import StatsTag, resolve_dataset_paths, validate_dataset_paths, resolve_modality_configs, _relative_fingerprint, freeze_concat_statistics
    from starVLA.dataloader.gr00t_sharded_processor import StarVLAPackProcessor
    torch.set_num_threads(1)
    cfg = OmegaConf.load(config)
    dc = cfg.datasets.vla_data
    assert dc.include_state is False and dc.use_relative_action is True and dc.use_percentiles is True
    assert cfg.trainer.max_train_steps == 80000 and dc.per_device_batch_size == 8
    spec = resolve_dataset_mixture(dc.data_mix, DATASET_NAMED_MIXTURES)
    expected = EXPECTED[dc.data_mix]
    weights = collections.defaultdict(float)
    counts = collections.Counter()
    modalities_by_tag, loaders_by_tag, stats_by_tag = {}, collections.defaultdict(list), collections.defaultdict(list)
    hashes, paths = {}, []
    for name, weight, tag in spec:
        assert weight > 0 and np.isfinite(weight)
        weights[tag] += weight
        counts[tag] += 1
        dataset, cache = resolve_dataset_paths(dc, name, tag)
        validate_dataset_paths(dataset, cache, tag)
        assert cache is not None
        modalities = resolve_modality_configs(ROBOT_TYPE_CONFIG_MAP[tag], tag)
        assert list(modalities["action"].delta_indices) == list(range(8))
        assert list(modalities["video"].delta_indices) == [0]
        modalities_by_tag.setdefault(tag, modalities)
        relative_file = dataset / "meta/relative_stats.json"
        relative = json.loads(relative_file.read_text())
        for key, action_config in zip(modalities["action"].modality_keys, modalities["action"].action_configs):
            if str(action_config.rep.value).lower() == "relative":
                assert relative.get("__fingerprints__", {}).get(key) == _relative_fingerprint(modalities["action"], modalities["state"], key, action_config), (dataset, key, "relative statistics mismatch")
        for filename in ("info.json", "modality.json", "stats.json", "relative_stats.json"):
            path = dataset / "meta" / filename
            hashes[str(path)] = sha256(path)
        loader = LeRobotEpisodeLoader(dataset, modalities, frame_cache_dir=cache)
        stats = loader.get_dataset_statistics()
        loaders_by_tag[tag].append((loader, dataset, cache, float(weight)))
        stats_by_tag[tag].append(stats)
        paths.append({"name": name, "tag": tag, "weight": weight, "dataset": str(dataset), "cache": str(cache)})
    assert set(weights) == set(expected)
    for tag, fraction in expected.items():
        assert abs(weights[tag] - fraction) < 1e-12, (tag, weights[tag], fraction)
    assert counts == {"galaxea_r1lite_stage1": 218, "libero_plus_rot6d": 1, "xperience_human": 1}, counts
    # All original source entries and their relative weights must be retained.
    for source, tag in (("galaxea_open_world_stage1", "galaxea_r1lite_stage1"), ("libero_plus_stage1", "libero_plus_rot6d"), ("xperience_hours100", "xperience_human")):
        original = DATASET_NAMED_MIXTURES[source]
        actual = [row for row in spec if row[2] == tag]
        assert [row[0] for row in actual] == [row[0] for row in original]
        np.testing.assert_allclose([row[1] / expected[tag] for row in actual], np.array([row[1] for row in original]) / sum(row[1] for row in original), rtol=1e-12, atol=1e-15)
    merged = {}
    for tag, stats in stats_by_tag.items():
        source_weights = [entry[3] for entry in loaders_by_tag[tag]]
        merged[tag] = {key: merge_statistics([s[key] for s in stats], source_weights, is_relative_stats=(key == "relative_action"))
                       for key in ("state", "action", "relative_action") if key in stats[0]}
    layout = load_layout(dc.canonical_layout)
    robot_maps = load_robot_maps(dc.canonical_robots, layout)
    assert layout.dim == 90
    processor = StarVLAPackProcessor(modalities_by_tag, include_state=False, use_percentiles=True, clip_outliers=True,
                                    use_relative_action=True, canonical_layout=layout, robot_maps=robot_maps)
    processor.set_statistics(merged, override=True)
    frozen = freeze_concat_statistics(merged, modalities_by_tag, use_percentiles=True, use_relative_action=True)
    assert set(frozen) == set(expected)
    for tag in expected:
        for stat in ("min", "max", "mean", "std", "q01", "q99"):
            array = np.asarray(frozen[tag]["action"][stat])
            assert array.shape[-1] == DIMENSIONS[tag] and np.isfinite(array).all(), (tag, stat, array.shape)
    rng = np.random.default_rng(int(cfg.seed))
    samples, sample_metadata = [], []
    for tag, choices in loaders_by_tag.items():
        for i in range(args.samples_per_source):
            loader, dataset, cache, _ = choices[(i * 73) % len(choices)]
            rows = [row for row, length in enumerate(loader.episode_lengths) if int(length) >= 8]
            assert rows
            row = rows[int(rng.integers(0, len(rows)))]
            length = int(loader.episode_lengths[row])
            step_index = int(rng.integers(0, length - 8 + 1))
            episode = loader.load_episode(row, video_indices=np.array([step_index]))
            step = extract_step_data(episode, step_index, modalities_by_tag[tag], StatsTag(tag), allow_padding=False)
            sample = processor._pack_step(step)
            action, mask = np.asarray(sample["action"]), np.asarray(sample["action_mask"])
            assert action.shape == mask.shape == (8, 90)
            assert np.isfinite(action).all() and np.isfinite(mask).all()
            assert np.all((mask == 0) | (mask == 1))
            assert np.all(mask.sum(axis=-1) == DIMENSIONS[tag])
            assert np.all(action[mask == 0] == 0)
            assert np.abs(action).max() <= 1.00001
            assert sample["robot_tag"] == tag and "state" not in sample
            assert len(sample["image"]) == VIEWS[tag]
            for image in sample["image"]:
                assert image.mode == "RGB" and image.size == (256, 256), (tag, image.mode, image.size)
            assert isinstance(sample["lang"], str) and sample["lang"].strip()
            if tag == "xperience_human":
                ego_mask = np.zeros(90)
                for begin, end in ((7, 16), (17, 32), (39, 48), (49, 64)):
                    ego_mask[begin:end] = 1
                np.testing.assert_array_equal(mask, np.broadcast_to(ego_mask, mask.shape))
            samples.append(sample)
            sample_metadata.append({"tag": tag, "dataset": str(dataset), "episode_row": row, "step_index": step_index,
                                    "views": len(sample["image"]), "valid_dims": DIMENSIONS[tag], "shape": list(action.shape),
                                    "action_min": float(action.min()), "action_max": float(action.max()), "instruction": sample["lang"]})
    assert not torch.cuda.is_initialized()
    assert all(sha256(path) == old for path, old in hashes.items()), "Source metadata changed during preflight"
    output.parent.mkdir(parents=True, exist_ok=True)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    with samples_output.open("wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    report = {"passed": True, "config": str(config), "config_sha256": sha256(config), "mix": dc.data_mix,
              "source_fractions": dict(weights), "source_dataset_counts": dict(counts), "stats_tags": sorted(frozen),
              "unchanged_source_metadata": True, "cuda_initialized": False, "samples_file": str(samples_output),
              "samples_sha256": sha256(samples_output), "samples": sample_metadata, "datasets": paths,
              "source_metadata_sha256": hashes,
              "normalization": "separate robot-tag statistics; relative dual EEF and absolute fingertips for Ego; q01/q99 clip",
              "sampling": "baseline unchanged: per-source normalized subset weights; shard probability corrected by average shard length",
              "loss": "baseline unchanged: local-batch mean over valid action elements; sample fraction is not loss weight"}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "mix", "source_fractions", "source_dataset_counts", "stats_tags", "samples_file", "samples")}, indent=2))


if __name__ == "__main__":
    main()
