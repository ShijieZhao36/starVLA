#!/usr/bin/env python3
"""Append only experiment mixture declarations and derive A/B pretrain configs.

Run in an isolated checkout of the completed Galaxea/LIBERO baseline. Existing
sampling, normalization, model, loss and training implementations are untouched.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import yaml


MIXES = {
    "galaxea_libero_ego50_A": [
        ("galaxea_open_world_stage1", 0.3),
        ("libero_plus_stage1", 0.2),
        ("xperience_hours100", 0.5),
    ],
    "galaxea_libero_ego50_B": [
        ("galaxea_open_world_stage1", 0.4),
        ("libero_plus_stage1", 0.1),
        ("xperience_hours100", 0.5),
    ],
}
BEGIN = "# BEGIN Ego50 A/B composed dataset declarations (20261110)"
END = "# END Ego50 A/B composed dataset declarations (20261110)"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def patch_mixtures(path):
    original = path.read_text()
    block = BEGIN + "\nCOMPOSED_DATASET_MIXTURES.update(" + repr(MIXES) + ")\n" + END
    if BEGIN in original:
        assert block in original, "Existing Ego50 declarations conflict; inspect before updating"
        return {"already_applied": True, "before_sha256": digest(original), "after_sha256": digest(original)}
    tree = ast.parse(original)
    definitions = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "COMPOSED_DATASET_MIXTURES" for t in n.targets)]
    assert len(definitions) == 1, "Expected existing baseline composed-mixture mechanism"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "resolve_dataset_mixture" for n in tree.body)
    existing = ast.literal_eval(definitions[0].value)
    assert not (set(existing) & set(MIXES)), "Mixture names are already used"
    # Insert below the existing dictionary, leaving its resolver byte-for-byte intact.
    lines = original.splitlines(keepends=True)
    index = definitions[0].end_lineno
    updated = "".join(lines[:index]) + "\n\n" + block + "\n" + "".join(lines[index:])
    compile(updated, str(path), "exec")
    path.write_text(updated)
    return {"already_applied": False, "before_sha256": digest(original), "after_sha256": digest(updated)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--run-suffix", default="20261110_0237")
    parser.add_argument("--run-root", default="/mnt/zhaoshijie/checkpoints/starVLA")
    parser.add_argument("--base-vlm", default="/mnt/zhaoshijie/workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--baseline-config")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else repo / "examples/pretrain/Ego50AB/train_files"
    base = Path(args.baseline_config).resolve() if args.baseline_config else repo / "examples/pretrain/GalaxeaLiberoMix/train_files/starvla_mixed_ln_80k.yaml"
    cfg = yaml.safe_load(base.read_text())
    assert cfg["datasets"]["vla_data"]["data_mix"] == "galaxea_libero_plus_stage1_80_20"
    assert cfg["trainer"]["max_train_steps"] == 80000
    assert cfg["datasets"]["vla_data"]["per_device_batch_size"] == 8
    assert cfg["framework"]["action_model"]["decoder_input_norm"] is True
    assert cfg["trainer"]["pretrained_checkpoint"] is None
    out.mkdir(parents=True, exist_ok=True)
    patch = patch_mixtures(repo / "starVLA/dataloader/gr00t_lerobot/mixtures.py")
    outputs = {}
    for variant in ("A", "B"):
        current = copy.deepcopy(cfg)
        current["run_id"] = f"ego50_{variant.lower()}_ln_8gpu_pre80k_{args.run_suffix}"
        current["run_root_dir"] = args.run_root
        current["framework"]["qwenvl"]["base_vlm"] = args.base_vlm
        dc = current["datasets"]["vla_data"]
        dc["data_mix"] = f"galaxea_libero_ego50_{variant}"
        dc.setdefault("data_root_dirs_by_robot", {})["xperience_human"] = "/mnt/zhaoshijie/dataset/xperience_runs/stage1"
        dc.setdefault("frame_cache_dirs_by_robot", {})["xperience_human"] = "/mnt/zhaoshijie/frame_cache/xperience_hours100_256"
        dc["audit_sampling"] = True
        current["trainer"]["audit_source_exposure"] = True
        current["trainer"]["audit_initial_model"] = True
        target = out / f"pretrain_{variant}.yaml"
        encoded = yaml.safe_dump(current, sort_keys=False)
        if target.exists():
            assert target.read_text() == encoded, f"Refusing to overwrite differing config: {target}"
        else:
            target.write_text(encoded)
        outputs[variant] = {"config": str(target), "sha256": digest(encoded), "run_id": current["run_id"],
                            "mix": dc["data_mix"], "source_fractions": dict(MIXES[dc["data_mix"]])}
    manifest = {"baseline_config": str(base), "baseline_sha256": digest(base.read_text()),
                "mixture_patch": patch, "outputs": outputs,
                "preserved": "baseline sampler, source-internal weights, statistics, processor, model, loss and trainer"}
    (out / "data_patch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
