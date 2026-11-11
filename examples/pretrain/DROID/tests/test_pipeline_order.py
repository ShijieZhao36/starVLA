"""Subprocess failure injection for the one-command script; no GPUs involved."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / "train_files"


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("failure,prepare,expected", [
    ("", False, ["pretrain", "posttrain", "eval"]),
    ("pretrain", False, ["pretrain"]),
    ("posttrain", False, ["pretrain", "posttrain"]),
    ("eval", False, ["pretrain", "posttrain", "eval"]),
    ("statistics", True, ["statistics"]),
    ("cache", True, ["statistics", "cache"]),
    ("", True, ["statistics", "cache", "pretrain", "posttrain", "eval"]),
])
def test_stages_stop_at_first_failure(tmp_path, failure, prepare, expected, smoke):
    repo = tmp_path / "repo"
    scripts = repo / "examples/pretrain/DROID/train_files"
    scripts.mkdir(parents=True)
    for name in ["run_pipeline.sh", "common.sh"]:
        shutil.copy2(SCRIPTS / name, scripts / name)
    (scripts / "experiment.env").write_text(f'''
DRY_RUN=0
PYTHON_BIN={sys.executable}
RUN_ROOT_DIR={tmp_path / 'runs'}
BASE_VLM={tmp_path / 'vlm'}
DROID_DATASET_DIR={tmp_path / 'dataset'}
DROID_FRAME_CACHE={tmp_path / 'droid_cache'}
LIBERO_DATA_ROOT={tmp_path / 'libero'}
LIBERO_FRAME_CACHE={tmp_path / 'libero_cache'}
EVAL_ROOT_DIR={tmp_path / 'eval'}
WANDB_PROJECT=test
WANDB_ENTITY=test
WANDB_MODE=disabled
''')
    for name, phase in {
        "precompute_droid_stats.sh": "statistics", "precompute_droid_frame_cache.sh": "cache",
        "run_droid_pretrain.sh": "pretrain", "run_libero_plus_posttrain.sh": "posttrain", "run_libero_plus_eval.sh": "eval",
    }.items():
        (scripts / name).write_text(f'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' {phase} >> "$TEST_TRACE"
if [[ "$SMOKE_TEST" == 1 ]]; then
  [[ "$PRETRAIN_STEPS" == 3 && "$POSTTRAIN_STEPS" == 3 && "$PRETRAIN_WARMUP" == 1 && "$LOG_INTERVAL" == 1 ]]
  [[ "$PRETRAIN_RUN_DIR" == *_smoke && "$POSTTRAIN_RUN_DIR" == *_smoke && "$EVAL_OUTPUT_DIR" == *_smoke ]]
fi
[[ "$FAIL_PHASE" != {phase} ]] || exit 23
''')
    # Isolate locking from any real user pipeline; the orchestration behavior is
    # under test, not the operating system flock implementation.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_flock = bindir / "flock"
    fake_flock.write_text("#!/bin/sh\nexit 0\n")
    fake_flock.chmod(0o755)
    trace = tmp_path / "trace"
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", REPO_DIR=str(repo),
               PIPELINE_ID="test_order", SMOKE_TEST=str(int(smoke)), FAIL_PHASE=failure, TEST_TRACE=str(trace), PREPARE_DATA=str(int(prepare)))
    process = subprocess.run(["bash", str(scripts / "run_pipeline.sh")], env=env, text=True, capture_output=True)
    assert process.returncode == (23 if failure else 0), process.stdout + process.stderr
    assert trace.read_text().splitlines() == expected
    assert (tmp_path / ("runs/launcher_logs/test_order_smoke/exit_code" if smoke else "runs/launcher_logs/test_order/exit_code")).read_text().strip() == str(process.returncode)
    if not failure:
        again = subprocess.run(["bash", str(scripts / "run_pipeline.sh")], env=env, text=True, capture_output=True)
        assert again.returncode != 0 and "already exists" in again.stderr
        assert trace.read_text().splitlines() == expected
        assert (tmp_path / ("runs/launcher_logs/test_order_smoke/exit_code" if smoke else "runs/launcher_logs/test_order/exit_code")).read_text().strip() == "0"
