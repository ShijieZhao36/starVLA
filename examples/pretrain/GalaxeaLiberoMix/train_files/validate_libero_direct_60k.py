#!/usr/bin/env python3
"""Fail-closed parity gates for direct LIBERO LN 60k, against the completed pair."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import validate_libero_pair_60k as pair


REFERENCE_PROTOCOL = "8d967b0b221b604718bffd814598f31ad4d9997722e945f2b15f411ca3cde417"
REFERENCE_EVAL = "28775d2ff8734c4798976611792071420f7d88dabb78e36d16a4fe0b0096eb48"
REFERENCE_STATS = "28493dba8beffeb8abaef0254a1e8ef81d1c917005e2c3362237b5add036c939"
INITIAL_MODEL = "530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89"
REFERENCE_RUN = "/mnt/zhaoshijie/checkpoints/starVLA/libero_from_galaxea_control_ln_8gpu_60k_20261105_pair"
TRAIN_FILES = "examples/pretrain/GalaxeaLiberoMix/train_files/"
ALLOWED_ADDITIONS = {TRAIN_FILES + "run_libero_direct_60k_eval.sh", TRAIN_FILES + "validate_libero_direct_60k.py"}
require = pair.require


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reference(args):
    root = Path(args.reference_contract).resolve(strict=True)
    protocol = pair.load_json(root / "protocol_manifest.json")
    require(object_hash(protocol) == REFERENCE_PROTOCOL, "Reference training protocol fingerprint changed")
    evaluation = pair.load_json(root / "evaluation_protocol_manifest.json")
    require(object_hash(evaluation) == REFERENCE_EVAL, "Reference evaluation protocol fingerprint changed")
    parity = pair.load_json(root / "posttrain_statistics_contract.json")
    require(parity["dataset_statistics_sha256"] == REFERENCE_STATS, "Reference statistics fingerprint changed")
    require(parity["initial_parameters"]["model"]["sha256"] == INITIAL_MODEL, "Reference fresh initial model fingerprint changed")
    return protocol, parity


def base_identity(path):
    root = Path(path).resolve(strict=True)
    index = pair.load_json(root / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    require(shards, "Base Qwen source has no weight shards")
    for name in shards:
        require(Path(name).name == name, "Base weight index has an unsafe shard path")
        require((root / name).is_file() and (root / name).stat().st_size > 0, "Missing base Qwen shard: " + name)
    return {"root": str(root), "config_sha256": pair.digest(root / "config.json"),
            "weight_index_sha256": pair.digest(root / "model.safetensors.index.json"),
            "weight_shard_bytes": {name: (root / name).stat().st_size for name in shards}}


def compare_config(actual, expected, smoke):
    """Compare the resolved model/trainer settings, allowing only documented run fields."""
    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    require(actual["trainer"]["pretrained_checkpoint"] is None, "Direct baseline cannot load a robot checkpoint")
    require(actual["trainer"]["reload_modules"] is None and actual["trainer"]["is_resume"] is False,
            "Direct baseline cannot resume or partially reload")
    require(actual["trainer"].get("expected_initial_model_sha256") == INITIAL_MODEL,
            "Missing the trainer's hard gate on original Qwen plus fresh complete action head")
    actual["trainer"].pop("expected_initial_model_sha256")
    expected["trainer"]["pretrained_checkpoint"] = None
    for cfg in (actual, expected):
        for key in ("run_id", "output_dir", "config_yaml"):
            cfg.pop(key, None)
    if smoke:
        expected["trainer"].update(max_train_steps=3, num_warmup_steps=1, save_interval=3,
                                   eval_interval=1, logging_frequency=1)
    require(actual == expected, "Resolved config differs from matched control beyond allowed initialization/run/smoke changes")


def preflight(args):
    protocol, parity = reference(args)
    repo = Path(args.repo).resolve(strict=True)
    cfg = pair.load_yaml(args.config)
    pair.assert_architecture(cfg)
    require(cfg == protocol["common_config"], "Common YAML differs from the completed matched pair")
    require(Path(args.base_vlm).resolve(strict=True) == Path(cfg["framework"]["qwenvl"]["base_vlm"]).resolve(strict=True),
            "Base VLM must be the same original Qwen3-VL source")
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True)
    require(not status.strip(), "Repository must be committed and clean: " + status)
    changed = set(subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only", protocol["repo_commit"], "HEAD"], text=True).splitlines())
    require(changed == ALLOWED_ADDITIONS, "Only the new direct runner and validator may differ from the matched commit: " + str(sorted(changed)))
    for name, expected in protocol["code_sha256"].items():
        require(pair.digest(repo / name) == expected, "Matched training/evaluation code changed: " + name)
    for path in (args.config, args.accelerate, args.deepspeed, pair.__file__):
        name = Path(path).name
        require(pair.digest(path) == protocol["launcher_sha256"][name], "Matched configuration/helper changed: " + name)
    require(Path(args.runner).resolve() == repo / (TRAIN_FILES + "run_libero_direct_60k_eval.sh"), "Wrong direct runner path")
    require(Path(__file__).resolve() == repo / (TRAIN_FILES + "validate_libero_direct_60k.py"), "Validator must run from the committed direct checkout")
    data = pair.data_manifest(args.data)
    require(data == protocol["training_data"], "LIBERO training dataset/statistics metadata differs from matched pair")
    require(str(Path(args.cache).resolve(strict=True)) == protocol["frame_cache"], "Frame cache differs from matched pair")
    require(pair.digest(Path(args.overlay).parent / "sim_overlay_manifest.json") == protocol["overlay_manifest_sha256"], "Runtime overlay changed")
    versions = {name: importlib.metadata.version(name) for name in protocol["runtime_versions"]}
    require(versions == protocol["runtime_versions"], "Training runtime package versions differ from matched pair")
    identity = base_identity(args.base_vlm)
    evidence = {"status": "training_preflight_passed", "ready": True, "arm": "direct", "smoke": args.smoke,
                "reference_common_protocol_sha256": REFERENCE_PROTOCOL, "reference_evaluation_protocol_sha256": REFERENCE_EVAL,
                "configuration_sha256": pair.digest(args.config), "training_data": data, "runtime_versions": versions,
                "reference_statistics_sha256": REFERENCE_STATS, "expected_initial_parameters": parity["initial_parameters"],
                "repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "reference_commit": protocol["repo_commit"], "base_vlm": identity,
                "runner_sha256": pair.digest(args.runner), "validator_sha256": pair.digest(__file__),
                "initialization": "original Qwen3-VL plus seed42 fresh complete action head; no robot-pretrained checkpoint"}
    if args.report:
        save(args.report, evidence)
    if not (args.dry_run or args.check_only):
        out = Path(args.output) / "launch"
        save(out / "training_data_manifest.json", data)
        save(out / "direct_protocol_manifest.json", evidence)
        save(out / "reference_protocol_manifest.json", protocol)
    print("Validated unchanged 60k protocol, training data, code, runtime and original Qwen source")


def initial(args):
    protocol, parity = reference(args)
    run = Path(args.output).resolve(strict=True)
    cfg = pair.load_yaml(run / "config.full.yaml")
    compare_config(cfg, pair.load_yaml(Path(args.reference_run) / "config.full.yaml"), args.smoke)
    require(cfg["run_id"] == run.name and Path(cfg["output_dir"]).resolve() == run, "Resolved output/run identity mismatch")
    audit = pair.load_json(run / "initial_model_sha256.json")
    require(audit["phase"] == "initial" and audit["seed"] == 42, "Unexpected initial audit phase or seed")
    require(audit["configured_pretrained_checkpoint"] is None, "Initial audit reports a robot checkpoint")
    require(audit["configured_base_vlm"] == protocol["common_config"]["framework"]["qwenvl"]["base_vlm"], "Initial audit has wrong base Qwen")
    require(audit["parameters"] == parity["initial_parameters"], "Original Qwen / fresh complete action-head parameter hashes differ")
    require(not (run / "training_start_model_sha256.json").exists(), "Unexpected checkpoint-loaded/resumed model audit in direct baseline")
    require(set(pair.load_json(run / "dataset_statistics.json")) == {pair.LIBERO}, "Only LIBERO training statistics are permitted")
    require(pair.digest(run / "dataset_statistics.json") == REFERENCE_STATS, "LIBERO statistics differ from matched pair")
    recorded = pair.load_json(run / "launch/training_data_manifest.json")
    require(pair.data_manifest(args.data) == recorded == protocol["training_data"], "Training data metadata changed")
    launch = pair.load_json(run / "launch/direct_protocol_manifest.json")
    require(base_identity(audit["configured_base_vlm"]) == launch["base_vlm"], "Base Qwen files changed after preflight")
    text = (run / "train.log").read_text(errors="replace")
    # Rich wraps this message across lines and inserts a source-location column.
    # The null config, absent reload audit and exact initial hashes above are the
    # authoritative checks; this stable log prefix confirms the expected branch.
    require("No pretrained checkpoint" in text, "Trainer has not confirmed fresh direct initialization")
    require("loaded <full_model> model parameters" not in text and "Loaded pretrained checkpoint:" not in text,
            "Unexpected robot checkpoint load reported")
    report = {"status": "initial_passed", "arm": "direct", "statistics_sha256": REFERENCE_STATS,
              "initial_parameters": audit["parameters"], "output": str(run), "smoke": args.smoke}
    if args.report:
        save(args.report, report)
    print("Validated original Qwen, seed42 fresh complete action head and identical LIBERO statistics")
    return report


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def trained(args):
    initial(args)
    run = Path(args.output).resolve(strict=True)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    require(metrics and all(finite(row) for row in metrics), "Metrics are empty or contain NaN/Inf")
    steps = 3 if args.smoke else 60000
    require(metrics[-1]["step"] == steps, "Final metrics do not reach the required step target")
    require(all(a["step"] < b["step"] for a, b in zip(metrics, metrics[1:])), "Metrics steps are not strictly increasing")
    for row in metrics:
        require(row["source_exposure/cumulative_examples"] == row["step"] * 64, "Effective global batch differs from 64")
        require(row[f"source_exposure/{pair.LIBERO}/cumulative_examples"] == row["step"] * 64, "Non-LIBERO exposure detected")
        for key, value in row.items():
            allowed = {"source_exposure/cumulative_examples", f"source_exposure/{pair.LIBERO}/cumulative_examples"}
            if key.startswith("source_exposure/") and key.endswith("/cumulative_examples") and key not in allowed:
                require(value == 0, "Unexpected source exposure: " + key)
    checkpoint = run / "final_model/pytorch_model.pt"
    reference_checkpoint = Path(args.reference_run) / "final_model/pytorch_model.pt"
    require(checkpoint.is_file() and checkpoint.stat().st_size == reference_checkpoint.stat().st_size > 0,
            "Final checkpoint missing or bytes differ from the full matched architecture")
    marker = {"arm": "direct", "steps": steps, "checkpoint": str(checkpoint), "bytes": checkpoint.stat().st_size,
              "source_checkpoint": None, "statistics_sha256": REFERENCE_STATS,
              "initial_model_sha256": INITIAL_MODEL, "reference_common_protocol_sha256": REFERENCE_PROTOCOL}
    save(run / "posttrain_success.json", marker)
    print("Validated final direct checkpoint, exact update/exposure count and finite metrics")


def evaluation_protocol(args):
    reference(args)
    # Existing helper compares against an already-present contract. Never let it
    # create or alter the completed experiments' reference files.
    original_dry_run = args.dry_run
    args.contract = args.reference_contract
    args.dry_run = True
    pair.evaluation_protocol(args)
    if not original_dry_run:
        save(Path(args.output) / "launch/evaluation_protocol_manifest.json",
             pair.load_json(Path(args.reference_contract) / "evaluation_protocol_manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("preflight", "initial", "trained", "eval-protocol"):
        p = sub.add_parser(name)
        for key in ("output", "reference-contract"):
            p.add_argument("--" + key, required=True)
        p.add_argument("--report")
        if name == "eval-protocol":
            p.add_argument("--libero-home", required=True)
            p.add_argument("--dry-run", action="store_true")
        else:
            p.add_argument("--data", required=True)
            p.add_argument("--smoke", action="store_true")
            if name == "preflight":
                for key in ("repo", "config", "runner", "accelerate", "deepspeed", "overlay", "cache", "base-vlm"):
                    p.add_argument("--" + key, required=True)
                p.add_argument("--dry-run", action="store_true")
                p.add_argument("--check-only", action="store_true")
            else:
                p.add_argument("--reference-run", default=REFERENCE_RUN)
    args = parser.parse_args()
    {"preflight": preflight, "initial": initial, "trained": trained, "eval-protocol": evaluation_protocol}[args.mode](args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Direct LIBERO parity gate failed: {exc}", file=sys.stderr)
        sys.exit(1)
