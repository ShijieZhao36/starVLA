#!/usr/bin/env python3
"""CPU-only provenance gates for the paired LIBERO Plus posttraining runner."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml


GALAXEA = "galaxea_r1lite_stage1"
LIBERO = "libero_plus_rot6d"
SOURCE_MIX = {"control": "galaxea_open_world_stage1", "mixed": "galaxea_libero_plus_stage1_80_20"}
TRAIN_META = ("meta/info.json", "conversion_report.json", "meta/tasks.jsonl", "meta/modality.json", "meta/stats.json", "meta/relative_stats.json")
SUITE_COUNTS = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_architecture(cfg):
    require(cfg["framework"]["name"] == "QwenPI", "Expected QwenPI")
    action = cfg["framework"]["action_model"]
    for key, value in {"action_dim": 90, "state_dim": 90, "action_horizon": 8, "decoder_input_norm": True}.items():
        require(action[key] == value, f"Unexpected action_model.{key}: {action[key]!r}")
    require(cfg["datasets"]["vla_data"]["include_state"] is False, "Expected include_state=false")
    require(cfg["seed"] == 42, "Expected seed=42")


def validate_source(run, arm, launcher_dir, source_smoke=False):
    run = Path(run).resolve(strict=True)
    marker = load_json(run / "pretrain_success.json")
    cfg = load_yaml(run / "config.full.yaml")
    checkpoint = run / "final_model/pytorch_model.pt"
    require(marker["arm"] == arm, f"Source arm mismatch: expected {arm}, got {marker['arm']!r}")
    source_steps = 3 if source_smoke else 80000
    require(marker["steps"] == source_steps and cfg["trainer"]["max_train_steps"] == source_steps,
            f"Source must have completed {source_steps} updates")
    if source_smoke:
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        require(bool(metrics) and metrics[-1]["step"] == 3, "Smoke source must have real 3-step metrics")
        require(all(math.isfinite(row["action_dit_loss"]) and
                    row["source_exposure/cumulative_examples"] == row["step"] * 64 for row in metrics),
                "Smoke source must have finite loss and global64 sample exposure")
        require(metrics[-1]["source_exposure/cumulative_examples"] == 192,
                "Smoke source must have 192 cumulative examples")
    require(Path(marker["checkpoint"]).is_absolute(), "Source marker checkpoint must be absolute")
    require(Path(marker["checkpoint"]).resolve() == checkpoint, "Source marker must select final_model/pytorch_model.pt")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Missing final checkpoint")
    require(checkpoint.stat().st_size == marker["bytes"], "Source checkpoint size changed after success marker")
    require((run / "exit_code").read_text().strip() == "0", "Pretraining run has not exited successfully")
    require((Path(launcher_dir) / "exit_code").read_text().strip() == "0", "Pretraining outer launcher has not exited successfully")
    assert_architecture(cfg)
    require(cfg["datasets"]["vla_data"]["data_mix"] == SOURCE_MIX[arm], "Wrong pretraining mixture for this arm")
    stats = load_json(run / "dataset_statistics.json")
    expected_tags = {GALAXEA} if arm == "control" else {GALAXEA, LIBERO}
    require(set(stats) == expected_tags, f"Unexpected source statistics tags: {set(stats)}")
    return {"run_dir": str(run), "checkpoint": str(checkpoint), "bytes": marker["bytes"],
            "marker_sha256": digest(run / "pretrain_success.json"),
            "config_sha256": digest(run / "config.full.yaml"),
            "statistics_sha256": digest(run / "dataset_statistics.json")}


def data_manifest(data):
    data = Path(data).resolve(strict=True)
    info = load_json(data / "meta/info.json")
    require(info["splits"] == {"train": "0:14347"}, f"Unexpected LIBERO training split: {info['splits']}")
    return {"root": str(data), "splits": info["splits"],
            "files": {name: digest(data / name) for name in TRAIN_META}}


def pinned_contract(path, value, create):
    """Atomically pin a shared contract, then reject any difference on either host."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        require(load_json(path) == value, f"Paired protocol mismatch: {path}")
        return
    if not create:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".protocol-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        require(load_json(path) == value, f"Concurrent paired protocol mismatch: {path}")
    finally:
        Path(tmp).unlink(missing_ok=True)


def preflight(args):
    source_smoke = getattr(args, "source_smoke", False)
    require(not source_smoke or args.smoke, "--source-smoke requires --smoke")
    peer_arm = "mixed" if args.arm == "control" else "control"
    independent = args.independent_start
    selected_sources = [(args.arm, args.source, args.source_launcher)]
    if not independent:
        require(args.peer and args.peer_launcher, "Paired start requires --peer and --peer-launcher")
        require(Path(args.source).resolve() != Path(args.peer).resolve(), "Source runs must differ")
        selected_sources.append((peer_arm, args.peer, args.peer_launcher))
    sources, readiness_errors = {}, []
    for arm, run, launcher in selected_sources:
        try:
            sources[arm] = validate_source(run, arm, launcher, source_smoke=source_smoke)
        except Exception as exc:
            if not args.dry_run:
                raise
            readiness_errors.append(f"{arm}: {exc}")
    cfg = load_yaml(args.config)
    assert_architecture(cfg)
    require(cfg["trainer"]["max_train_steps"] == 60000, "Common YAML must specify 60000 steps")
    require(cfg["trainer"]["is_resume"] is False and cfg["trainer"]["reload_modules"] is None and cfg["trainer"]["strict_pretrained_load"] is True, "Expected fresh strict full-model transfer")
    require(cfg["datasets"]["vla_data"]["data_mix"] == "libero_plus_stage1", "Wrong posttraining data mix")
    repo = Path(args.repo).resolve(strict=True)
    require("strict_pretrained_load" in (repo / "starVLA/training/train_starvla.py").read_text(), "Selected checkout lacks the strict checkpoint-load gate")
    manifest = data_manifest(args.data)
    code_paths = [
        "starVLA/training/train_starvla.py", "starVLA/training/training_audit.py",
        "starVLA/training/trainer_utils/trainer_tools.py", "starVLA/model/framework/VLM4A/QwenPI.py",
        "starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py", "starVLA/model/modules/vlm/QWen3.py",
        "starVLA/dataloader/gr00t_sharded.py", "starVLA/dataloader/gr00t_sharded_processor.py",
        "starVLA/dataloader/canonical_layout.py", "gr00t/data/dataset/sharded_mixture_dataset.py",
        "examples/pretrain/canonical/bimanual_v1.yaml", "examples/pretrain/canonical/robots/libero_plus.yaml",
        "examples/simBenchmarks/LIBERO-plus/train_files/data_registry/data_config.py",
        "examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py",
    ]
    eval_dir = repo / "examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval"
    code_paths += [str(p.relative_to(repo)) for p in sorted(eval_dir.glob("*.py"))]
    code_paths += [str(p.relative_to(repo)) for p in sorted(eval_dir.glob("run_libero_plus_canonical*8gpu.sh"))]
    protocol = {
        # The common contract is identical even when the arms start at different
        # times. Source provenance is pinned separately below for each arm.
        "protocol": "galaxea-libero-paired-posttrain60k-v2", "smoke": args.smoke,
        "training_data": manifest,
        "common_config": cfg, "frame_cache": str(Path(args.cache).resolve(strict=True)),
        "repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {name: digest(repo / name) for name in code_paths},
        "launcher_sha256": {Path(p).name: digest(p) for p in (args.runner, __file__, args.config, args.accelerate, args.deepspeed)},
        "overlay_manifest_sha256": digest(Path(args.overlay).parent / "sim_overlay_manifest.json"),
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "accelerate", "deepspeed", "numpy", "scipy", "Pillow", "torchcodec")},
        "eval": {"episodes": 10030, "suite_counts": SUITE_COUNTS, "environment_seed": 7,
                 "policy_noise_seed": "unchanged-existing-unseeded-policy-server", "num_trials_per_task": 1, "num_gpus": 8, "sims_per_gpu": 4,
                 "canonical_forward": True, "bf16": True, "gripper_open_sign": "positive"},
    }
    if args.report:
        report = {"status": "training_preflight_passed", "ready": not readiness_errors,
                  "source_gate_errors": readiness_errors, "arm": args.arm,
                  "pretrain_run_dir": str(Path(args.source).resolve()),
                  "paired_pretrain_run_dir": None if independent else str(Path(args.peer).resolve()),
                  "source_gate_scope": "self_only" if independent else "paired",
                  "repo_commit": protocol["repo_commit"],
                  "common_protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                  "configuration_sha256": digest(args.config), "training_data": manifest}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if readiness_errors:
        print("DRY RUN — NOT READY: neither training nor evaluation was started.")
        for error in readiness_errors:
            print(error)
        print("Common configuration/data/code checks passed; source success gates remain pending.")
        return
    create = not (args.dry_run or args.check_only)
    pinned_contract(Path(args.contract) / "protocol_manifest.json", protocol, create)
    for arm, provenance in sources.items():
        pinned_contract(Path(args.contract) / f"source_provenance_{arm}.json", provenance, create)
    if not (args.dry_run or args.check_only):
        out = Path(args.output) / "launch"
        (out / "training_data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (out / "paired_protocol_manifest.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        (out / "source_provenance.json").write_text(json.dumps(sources[args.arm], indent=2, sort_keys=True) + "\n")
    scope = "this arm's completed 80k source" if independent else "both completed 80k sources"
    print(f"Validated {scope}, training-only LIBERO split and shared protocol ({args.arm})")


def trained(args):
    run = Path(args.output)
    cfg = load_yaml(run / "config.full.yaml")
    assert_architecture(cfg)
    require(Path(cfg["trainer"]["pretrained_checkpoint"]).resolve() == Path(args.source).resolve() / "final_model/pytorch_model.pt", "Wrong initial model")
    require(cfg["trainer"]["is_resume"] is False and cfg["trainer"]["reload_modules"] is None and cfg["trainer"]["strict_pretrained_load"] is True, "Resume/partial/non-strict reload is forbidden")
    require(cfg["trainer"]["max_train_steps"] == (3 if args.smoke else 60000), "Wrong final step target")
    require(cfg["datasets"]["vla_data"]["data_mix"] == "libero_plus_stage1", "Wrong posttraining data")
    require(cfg["datasets"]["vla_data"]["per_device_batch_size"] == 8, "Wrong batch size")
    require(cfg["trainer"]["gradient_accumulation_steps"] == 1, "Wrong accumulation")
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    steps = 3 if args.smoke else 60000
    require(metrics[-1]["step"] == steps, "Final metrics do not reach the required update count")
    require(metrics[-1]["source_exposure/cumulative_examples"] == steps * 64, "Effective sample exposure differs from 60k x global64")
    require(metrics[-1][f"source_exposure/{LIBERO}/cumulative_examples"] == steps * 64, "Non-LIBERO training exposure detected")
    require(set(load_json(run / "dataset_statistics.json")) == {LIBERO}, "Posttraining must use LIBERO-only statistics")
    recorded_data = load_json(run / "launch/training_data_manifest.json")
    require(data_manifest(args.data) == recorded_data, "Training metadata/statistics changed during the run")
    initial = load_json(run / "initial_model_sha256.json")
    training_start = load_json(run / "training_start_model_sha256.json")
    require(training_start["phase"] == "training_start", "Missing checkpoint-load parameter audit")
    require(Path(training_start["configured_pretrained_checkpoint"]).resolve() == Path(cfg["trainer"]["pretrained_checkpoint"]).resolve(), "Initial-weights audit selects wrong checkpoint")
    parity = {"initial_parameters": initial["parameters"], "dataset_statistics_sha256": digest(run / "dataset_statistics.json")}
    pinned_contract(Path(args.contract) / "posttrain_statistics_contract.json", parity, True)
    checkpoint = run / "final_model/pytorch_model.pt"
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Missing final checkpoint")
    require("loaded <full_model> model parameters" in (run / "train.log").read_text(errors="replace"), "Trainer did not report complete-model loading")
    marker = {"arm": args.arm, "steps": 3 if args.smoke else 60000, "checkpoint": str(checkpoint.resolve()),
              "bytes": checkpoint.stat().st_size, "source_checkpoint": cfg["trainer"]["pretrained_checkpoint"],
              "statistics_sha256": parity["dataset_statistics_sha256"]}
    (run / "posttrain_success.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print("Validated final checkpoint, complete-model transfer record and identical LIBERO statistics")


def evaluation_protocol(args):
    # These imports and file hashes are CPU-only; no simulation environment is built.
    from libero.libero import benchmark, get_libero_path
    with contextlib.redirect_stdout(io.StringIO()):
        suites = {name: benchmark.get_benchmark_dict()[name]() for name in SUITE_COUNTS}
    counts = {name: int(suite.n_tasks) for name, suite in suites.items()}
    require(counts == SUITE_COUNTS, f"Evaluation suite count mismatch: {counts}")
    tasks = {name: [list(suite.get_task(i)) for i in range(suite.n_tasks)] for name, suite in suites.items()}
    roots = {key: Path(get_libero_path(key)).resolve(strict=True) for key in ("bddl_files", "init_states")}
    # Hash all task and init-state files. Several task variants share source files;
    # hashing these roots avoids reimplementing LIBERO's special-name resolver.
    fingerprints = {}
    for key, root in roots.items():
        files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix not in {".pyc", ".pyo"}}
        require(files, f"No evaluation files under {root}")
        fingerprints[key] = files
    home = Path(args.libero_home).resolve(strict=True)
    code = {str(p.relative_to(home)): digest(p) for p in sorted((home / "libero").rglob("*.py"))}
    report = {"suite_counts": counts, "episode_count": sum(counts.values()), "num_trials_per_task": 1,
              "environment_seed": 7, "policy_noise_seed": "unchanged-existing-unseeded-policy-server",
              "libero_commit": subprocess.check_output(["git", "-C", str(home), "rev-parse", "HEAD"], text=True).strip(),
              "tasks": tasks, "file_sha256": fingerprints, "code_sha256": code,
              "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "mujoco", "robosuite", "pandas")}}
    pinned_contract(Path(args.contract) / "evaluation_protocol_manifest.json", report, not args.dry_run)
    if args.report:
        preflight = load_json(args.report)
        preflight.update(status="passed", evaluation_protocol_sha256=hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(), suite_counts=counts)
        Path(args.report).write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    if not args.dry_run:
        (Path(args.output) / "launch/evaluation_protocol_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Validated fixed LIBERO commit/task/init-state fingerprints and suite counts: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("preflight", "trained"):
        p = sub.add_parser(name)
        for key in ("arm", "source", "output", "data", "contract"):
            p.add_argument("--" + key, required=True)
        p.add_argument("--smoke", action="store_true")
        if name == "preflight":
            for key in ("source-launcher", "repo", "config", "runner", "accelerate", "deepspeed", "overlay", "cache"):
                p.add_argument("--" + key, required=True)
            p.add_argument("--source-smoke", action="store_true",
                           help="Only with --smoke: accept a real 3-step global64 source")
            p.add_argument("--peer")
            p.add_argument("--peer-launcher")
            p.add_argument("--independent-start", action="store_true",
                           help="Validate only this source; never inspect or resolve the peer source")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--check-only", action="store_true")
            p.add_argument("--report")
    p = sub.add_parser("eval-protocol")
    for key in ("output", "contract", "libero-home"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report")
    args = parser.parse_args()
    if args.mode == "eval-protocol":
        evaluation_protocol(args)
        return
    require(args.arm in SOURCE_MIX, "ARM must be control or mixed")
    (preflight if args.mode == "preflight" else trained)(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Paired LIBERO gate failed: {exc}", file=sys.stderr)
        sys.exit(1)
