#!/usr/bin/env python3
"""Wait for both Xperience launchers to finish and validate final-model metadata.

This helper uses only CPU metadata reads. It neither imports torch nor opens a
checkpoint payload, and never writes files. Progress goes to stderr; stdout is
the final checkpoint path on success, suitable for shell command substitution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str
    checkpoint: Optional[Path] = None


def _load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    else:
        config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise RuntimeError(f"Expected a YAML mapping in {path}")
    return config


def check_ready(run_dir: Path, control_dir: Path, expected_steps: int) -> Readiness:
    """Inspect completion without side effects; fail on a failed/invalid run.

    Missing or empty exit-code files mean 'not ready'. An empty file is allowed
    because the launch wrappers use a non-atomic printf redirection. Once both
    wrappers report zero, all metadata and the final checkpoint must exist.
    """
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    run_dir, control_dir = Path(run_dir), Path(control_dir)
    states = []
    for rank in (0, 1):
        sentinel = control_dir / f"node{rank}.exitcode"
        try:
            value = sentinel.read_text().strip()
        except FileNotFoundError:
            states.append(f"node{rank}=running (no exit code)")
            continue
        if not value:
            states.append(f"node{rank}=exit code being written")
            continue
        try:
            exit_code = int(value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid exit code in {sentinel}: {value!r}") from exc
        if exit_code != 0:
            raise RuntimeError(f"Xperience node{rank} failed with exit code {exit_code}: {sentinel}")
        states.append(f"node{rank}=success")

    if states != ["node0=success", "node1=success"]:
        return Readiness(False, "; ".join(states))

    checkpoint = run_dir / "final_model" / "pytorch_model.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"Both launchers succeeded but final checkpoint is missing or empty: {checkpoint}")
    config_path = run_dir / "config.full.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"Missing source config: {config_path}")
    config = _load_config(config_path)
    trainer = config.get("trainer", {})
    framework = config.get("framework", {})
    action = framework.get("action_model", {})
    requirements = {
        "trainer.max_train_steps": (trainer.get("max_train_steps"), expected_steps),
        "framework.name": (framework.get("name"), "QwenPI"),
        "framework.action_model.action_dim": (action.get("action_dim"), 90),
        "framework.action_model.state_dim": (action.get("state_dim"), 90),
        "framework.action_model.action_horizon": (action.get("action_horizon"), 8),
    }
    for key, (actual, expected) in requirements.items():
        if actual != expected:
            raise RuntimeError(f"Source config {key} must be {expected!r}, found {actual!r}: {config_path}")
    if action.get("decoder_input_norm") is not True:
        raise RuntimeError(f"Source config decoder_input_norm must be true: {config_path}")

    summary_path = run_dir / "summary.jsonl"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing training summary: {summary_path}")
    try:
        records = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid training summary JSON: {summary_path}") from exc
    if not any(isinstance(row, dict) and row.get("steps") == expected_steps for row in records):
        raise RuntimeError(f"Training summary lacks steps={expected_steps}: {summary_path}")
    return Readiness(True, f"Both nodes succeeded; final checkpoint and {expected_steps}-step metadata verified", checkpoint)


def _log(message: str) -> None:
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=30000)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=43200)
    parser.add_argument("--check-only", action="store_true", help="Inspect once; exit 3 if not ready")
    args = parser.parse_args(argv)
    if args.expected_steps <= 0 or args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("expected-steps, poll-seconds and timeout-seconds must be positive")

    started = time.monotonic()
    last_reason = None
    last_log = started
    try:
        while True:
            state = check_ready(args.run_dir, args.control_dir, args.expected_steps)
            now = time.monotonic()
            if state.reason != last_reason or now - last_log >= 60:
                _log(f"{state.reason}; elapsed={now - started:.0f}s")
                last_reason, last_log = state.reason, now
            if state.ready:
                print(state.checkpoint, flush=True)
                return 0
            if args.check_only:
                return 3
            remaining = args.timeout_seconds - (now - started)
            if remaining <= 0:
                raise RuntimeError(f"Timed out after {args.timeout_seconds:g}s waiting for Xperience completion")
            time.sleep(min(args.poll_seconds, remaining, 60))
    except KeyboardInterrupt:
        _log("Interrupted; no training was started")
        return 130
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        _log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
