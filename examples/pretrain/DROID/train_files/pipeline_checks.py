#!/usr/bin/env python3
"""CPU-only completion checks for DROID -> LIBERO Plus training."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
from pathlib import Path
import tempfile

import yaml

TAGS = {"pretrain": "droid_franka_rot6d", "posttrain": "libero_plus_rot6d"}
SUITES = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".droid-check-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def validate_source(run, stage="pretrain", expected_steps=None, smoke=False):
    run = Path(run).resolve(strict=True)
    marker = read_json(run / f"{stage}_success.json")
    ckpt = run / "final_model/pytorch_model.pt"
    require(marker["stage"] == stage, "Source stage mismatch")
    require(marker["validated"] is True, "Source is not validated")
    require(marker.get("smoke", False) is smoke, "Source smoke/formal mode mismatch")
    if expected_steps is not None:
        require(marker["steps"] == expected_steps, "Source step count mismatch")
    require((run / "train_exit_code").read_text().strip() == "0", "Source training has not exited successfully")
    require(ckpt.is_file() and ckpt.stat().st_size > 0, "Missing complete final checkpoint")
    require(Path(marker["checkpoint"]).resolve() == ckpt, "Wrong final checkpoint in source marker")
    require(ckpt.stat().st_size == marker["checkpoint_bytes"], "Source checkpoint size changed")
    require(ckpt.stat().st_mtime_ns == marker["checkpoint_mtime_ns"], "Source checkpoint was modified")
    for name, key in [("config.full.yaml", "config_sha256"), ("dataset_statistics.json", "statistics_sha256")]:
        require(digest(run / name) == marker[key], f"Source {name} changed")
    return marker


def validate_training(run, stage, steps, batch, source=None, smoke=False):
    run = Path(run).resolve(strict=True)
    require((run / "train_exit_code").read_text().strip() == "0", "Training process failed or has not exited")
    cfg = yaml.safe_load((run / "config.full.yaml").read_text())
    trainer, data = cfg["trainer"], cfg["datasets"]["vla_data"]
    require(not smoke or steps == 3, "Smoke training must use exactly three steps")
    action = cfg["framework"]["action_model"]
    require(cfg["framework"]["name"] == "QwenPI", "Expected QwenPI architecture")
    require(action["action_dim"] == action["state_dim"] == 90, "Expected canonical width 90")
    require(action["action_horizon"] == 8 and action["decoder_input_norm"] is True, "Horizon or LayerNorm mismatch")
    require(data["use_relative_action"] is True, "Expected relative EEF training")
    require(trainer["max_train_steps"] == steps, "Configured final step differs")
    require(data["per_device_batch_size"] == batch, "Per-GPU batch mismatch")
    require(trainer["gradient_accumulation_steps"] == 1, "Expected one accumulation step")
    require(trainer["is_resume"] is False, "Each stage must start a fresh optimizer/schedule")
    expected_mix = "droid_stage3_unmasked" if stage == "pretrain" else "libero_plus_stage1"
    require(data["data_mix"] == expected_mix, "Wrong training dataset")
    tag = TAGS[stage]
    stats = read_json(run / "dataset_statistics.json")
    require(set(stats) == {tag}, "Training used unexpected statistics tags")
    require(len(stats[tag]["action"]["mask"]) == (17 if stage == "pretrain" else 10), "Action group width mismatch")
    last = None
    with (run / "metrics.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    require(last is not None and last["step"] == steps, "Final metrics do not reach requested steps")
    losses = [float(v) for k, v in last.items() if "loss" in k.lower() and isinstance(v, (int, float))]
    require(losses and all(math.isfinite(v) for v in losses), "Final loss is absent or non-finite")
    exposure = steps * batch * 8
    require(last["source_exposure/cumulative_examples"] == exposure, "Global sample count differs from 8-GPU protocol")
    require(last[f"source_exposure/{tag}/cumulative_examples"] == exposure, "Wrong source exposure")
    log = (run / "train.log").read_text(errors="replace")
    # The trainer emits this event only after torch.save returns. Rich may wrap
    # the rest of the message and insert a source-location column, so matching
    # the full rendered sentence incorrectly rejects successfully saved runs.
    completion_log = ANSI_ESCAPE.sub("", log)
    require(re.search(r"\bTraining\s+complete\.", completion_log) is not None,
            "Trainer did not finish checkpoint saving")
    source_marker = None
    if stage == "posttrain":
        require(source is not None, "Posttraining requires the DROID source run")
        source_marker = validate_source(source, smoke=smoke)
        require(Path(trainer["pretrained_checkpoint"]).resolve() == Path(source_marker["checkpoint"]), "Wrong inherited checkpoint")
        require(trainer.get("reload_modules") is None and trainer["strict_pretrained_load"] is True, "Expected strict full-model transfer")
        require("loaded <full_model> model parameters" in log, "Missing full-model load confirmation")
        start = read_json(run / "training_start_model_sha256.json")
        require(start["phase"] == "training_start", "Missing post-load weight audit")
        require(Path(start["configured_pretrained_checkpoint"]).resolve() == Path(source_marker["checkpoint"]), "Post-load audit source mismatch")
    else:
        require(trainer.get("pretrained_checkpoint") is None, "Pretraining must initialize a fresh action head")
    ckpt = run / "final_model/pytorch_model.pt"
    require(ckpt.is_file() and ckpt.stat().st_size > 0, "Final model is missing or empty")
    marker = {
        "validated": True, "stage": stage, "steps": steps, "smoke": smoke,
        "global_batch": batch * 8, "source_exposure": exposure,
        "checkpoint": str(ckpt), "checkpoint_bytes": ckpt.stat().st_size,
        "checkpoint_mtime_ns": ckpt.stat().st_mtime_ns,
        "config_sha256": digest(run / "config.full.yaml"),
        "statistics_sha256": digest(run / "dataset_statistics.json"),
        "source_checkpoint": source_marker["checkpoint"] if source_marker else None,
    }
    write_json(run / f"{stage}_success.json", marker)
    return marker


def validate_evaluation(repo, run, output, smoke=False):
    source = validate_source(run, "posttrain", smoke=smoke)
    output = Path(output)
    if smoke:
        records = read_json(output / "logs/libero_goal/0_to_1.json")
        require(isinstance(records, dict) and records, "Smoke evaluation records are absent")
        total = sum(item["total_count"] for item in records.values())
        successes = sum(item["success_count"] for item in records.values())
        require(total == 1 and 0 <= successes <= 1, "Smoke evaluation must finish exactly one episode")
        report = {"validated": True, "smoke": True, "episode_count": total,
                  "successful_episodes": successes, "checkpoint": source["checkpoint"],
                  "output_dir": str(output.resolve())}
        write_json(output / "completion_validation.json", report)
        write_json(Path(run) / "eval_success.json", report)
        return report
    script = Path(repo) / "examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py"
    spec = importlib.util.spec_from_file_location("droid_libero_result_verifier", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        expected = module.expected_episode_keys(1)
    output = Path(output)
    with (output / "results.jsonl").open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    report = module.validate_records(records, read_json(output / "summary.json"), expected)
    require(report["episode_count"] == 10030 and report["suite_episode_counts"] == SUITES, "Unexpected LIBERO Plus benchmark coverage")
    report.update(checkpoint=source["checkpoint"], output_dir=str(output.resolve()))
    write_json(output / "completion_validation.json", report)
    write_json(Path(run) / "eval_success.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source")
    source.add_argument("--run", type=Path, required=True)
    source.add_argument("--stage", choices=TAGS, default="pretrain")
    source.add_argument("--steps", type=int)
    source.add_argument("--smoke", action="store_true")
    trained = sub.add_parser("trained")
    trained.add_argument("--run", type=Path, required=True)
    trained.add_argument("--stage", choices=TAGS, required=True)
    trained.add_argument("--steps", type=int, required=True)
    trained.add_argument("--batch", type=int, required=True)
    trained.add_argument("--source", type=Path)
    trained.add_argument("--smoke", action="store_true")
    evaluated = sub.add_parser("evaluated")
    evaluated.add_argument("--smoke", action="store_true")
    for key in ["repo", "run", "output"]:
        evaluated.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source":
        result = validate_source(args.run, args.stage, args.steps, args.smoke)
    elif args.command == "trained":
        result = validate_training(args.run, args.stage, args.steps, args.batch, args.source, args.smoke)
    else:
        result = validate_evaluation(args.repo, args.run, args.output, args.smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
