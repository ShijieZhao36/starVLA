#!/usr/bin/env python3
"""Reject incomplete LIBERO Plus queue runs, while allowing policy failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


EVAL_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
EpisodeKey = tuple[str, int, int]


def expected_episode_keys(num_trials: int) -> set[EpisodeKey]:
    from libero.libero import benchmark

    benchmarks = benchmark.get_benchmark_dict()
    return {
        (suite, task_id, episode_idx)
        for suite in EVAL_SUITES
        for task_id in range(int(benchmarks[suite]().n_tasks))
        for episode_idx in range(num_trials)
    }


def validate_records(
    records: list[dict], summary: dict, expected: set[EpisodeKey]
) -> dict:
    """Validate queue records and its summary; success=False is a valid result."""
    if not expected:
        raise ValueError("Expected episode set is empty")
    keys = []
    for line_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Result record {line_number} is not an object")
        suite, task_id, episode_idx = (
            record.get("suite"), record.get("task_id"), record.get("episode_idx")
        )
        if not isinstance(suite, str) or type(task_id) is not int or type(episode_idx) is not int:
            raise ValueError(f"Result record {line_number} has invalid episode identifiers")
        if type(record.get("success")) is not bool:
            raise ValueError(f"Result record {line_number} must have a boolean success field")
        key = (suite, task_id, episode_idx)
        if record.get("error"):
            raise ValueError(f"Episode {key!r} has an execution error: {record['error']}")
        keys.append(key)

    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count != 1]
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            f"Episode coverage mismatch: duplicates={duplicates[:10]!r}, "
            f"missing={missing[:10]!r} ({len(missing)} total), "
            f"unexpected={unexpected[:10]!r} ({len(unexpected)} total)"
        )
    if not isinstance(summary, dict) or summary.get("incomplete") != []:
        raise ValueError("summary.json must contain an empty incomplete list")
    successes = sum(record["success"] for record in records)
    total = summary.get("total", {})
    if total.get("total_count") != len(expected) or total.get("success_count") != successes:
        raise ValueError("summary.json total counts do not match results.jsonl")

    expected_per_suite = Counter(key[0] for key in expected)
    success_per_suite = Counter(record["suite"] for record in records if record["success"])
    per_suite = summary.get("per_suite", {})
    if set(per_suite) != set(expected_per_suite):
        raise ValueError("summary.json suite coverage does not match the benchmark")
    for suite, count in expected_per_suite.items():
        bucket = per_suite[suite]
        if bucket.get("total_count") != count or bucket.get("success_count") != success_per_suite[suite]:
            raise ValueError(f"summary.json counts for {suite} do not match results.jsonl")
    return {
        "validated": True,
        "episode_count": len(expected),
        "success_count": successes,
        "success_rate": successes / len(expected),
        "suite_episode_counts": dict(expected_per_suite),
        "execution_errors": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    args = parser.parse_args()
    if args.num_trials_per_task < 1:
        parser.error("--num-trials-per-task must be >= 1")
    try:
        expected = expected_episode_keys(args.num_trials_per_task)
        with (args.output_dir / "results.jsonl").open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
        report = validate_records(records, summary, expected)
    except Exception as exc:
        print(f"LIBERO Plus completion validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
