"""CPU subprocess tests for continuing an existing pretrain without rerunning it."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / "train_files"


@pytest.fixture
def recovery(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "examples/pretrain/DROID/train_files"
    scripts.mkdir(parents=True)
    for name in ("run_posttrain_eval_from_pretrain.sh", "common.sh"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    source = tmp_path / "completed_pretrain"
    source.mkdir()
    (source / "stage_exit_code").write_text("1\n")
    (source / "checkpoint_sentinel").write_text("existing checkpoint bytes\n")
    old_launcher = tmp_path / "runs/launcher_logs/original_pipeline"
    old_launcher.mkdir(parents=True)
    (old_launcher / "status.txt").write_text("phase=pretraining exit_code=1\n")
    # common.sh prepends the interpreter directory, so place the fake flock in
    # that directory too; otherwise the real /usr/bin/flock can win lookup.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    settings = {
        "PYTHON_BIN": bindir / "python", "RUN_ROOT_DIR": tmp_path / "runs",
        "BASE_VLM": tmp_path / "vlm", "DROID_DATASET_DIR": tmp_path / "dataset",
        "DROID_FRAME_CACHE": tmp_path / "droid_cache", "LIBERO_DATA_ROOT": tmp_path / "libero",
        "LIBERO_FRAME_CACHE": tmp_path / "libero_cache", "EVAL_ROOT_DIR": tmp_path / "eval",
        "PRETRAIN_STEPS": "30000", "WANDB_PROJECT": "test", "WANDB_ENTITY": "test",
        "WANDB_MODE": "disabled",
    }
    (scripts / "experiment.env").write_text(
        'DRY_RUN="${DRY_RUN:-0}"\n'
        + "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in settings.items())
    )
    (scripts / "pipeline_checks.py").write_text('''
import os, sys
assert sys.argv[1:] == ["source", "--stage", "pretrain", "--run", os.environ["PRETRAIN_RUN_DIR"], "--steps", "30000"]
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
with open(os.environ["TEST_TRACE"], "a") as stream:
    stream.write("source\\n")
print("source validation output", flush=True)
sys.exit(23 if os.environ["FAIL_PHASE"] == "source" else 0)
''')
    for name, phase in {
        "run_droid_pretrain.sh": "pretrain", "precompute_droid_stats.sh": "statistics",
        "precompute_droid_frame_cache.sh": "cache", "run_libero_plus_posttrain.sh": "posttrain",
        "run_libero_plus_eval.sh": "eval",
    }.items():
        (scripts / name).write_text(f'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' {phase} >> "$TEST_TRACE"
printf '%s output\\n' {phase}
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Command preview only: {phase}\\n'
  exit 0
fi
[[ "$FAIL_PHASE" != {phase} ]] || exit 23
if [[ {phase} == posttrain ]]; then
  [[ "$RUN_ID" == "$POSTTRAIN_RUN_ID" ]]
  [[ "$POSTTRAIN_RUN_DIR" == "$RUN_ROOT_DIR/$RUN_ID" ]]
  mkdir "$POSTTRAIN_RUN_DIR"
  printf '%s\\n' "$PRETRAIN_RUN_DIR" > "$POSTTRAIN_RUN_DIR/source.txt"
elif [[ {phase} == eval ]]; then
  [[ -f "$POSTTRAIN_RUN_DIR/source.txt" ]]
  mkdir -p "$EVAL_OUTPUT_DIR"
  printf '%s\\n' "$POSTTRAIN_RUN_DIR" > "$EVAL_OUTPUT_DIR/source.txt"
else
  exit 91
fi
''')
    fake_flock = bindir / "flock"
    fake_flock.write_text('''#!/bin/sh
[ "$*" = "-n 8" ] || exit 92
[ -e /proc/$$/fd/8 ] || exit 93
printf 'flock\\n' >> "$LOCK_TRACE"
exit "${LOCK_FAIL:-0}"
''')
    fake_flock.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", REPO_DIR=str(repo),
               PRETRAIN_RUN_DIR=str(source), RECOVERY_ID="recovery_run", POSTTRAIN_RUN_ID="new_posttrain",
               FAIL_PHASE="", TEST_TRACE=str(tmp_path / "trace"), LOCK_TRACE=str(tmp_path / "lock_trace"),
               PREPARE_DATA="1", DRY_RUN="0", RUN_ROOT_DIR=str(tmp_path / "runs"))
    for key in ("EVAL_OUTPUT_DIR", "EXPERIMENT_ENV", "LOCK_FAIL"):
        env.pop(key, None)
    return {"root": tmp_path, "scripts": scripts, "source": source, "env": env,
            "recovery_dir": tmp_path / "runs/launcher_logs/recovery_run",
            "posttrain": tmp_path / "runs/new_posttrain", "eval": tmp_path / "eval/new_posttrain_final"}


def run(recovery, **overrides):
    return subprocess.run(
        ["bash", str(recovery["scripts"] / "run_posttrain_eval_from_pretrain.sh")],
        env=dict(recovery["env"], **overrides), text=True, capture_output=True,
    )


def trace(recovery):
    path = Path(recovery["env"]["TEST_TRACE"])
    return path.read_text().splitlines() if path.exists() else []


@pytest.mark.parametrize("failure,expected,phase", [
    ("", ["source", "posttrain", "eval"], "completed"),
    ("source", ["source"], "validating_pretrain"),
    ("posttrain", ["source", "posttrain"], "posttraining"),
    ("eval", ["source", "posttrain", "eval"], "evaluating"),
])
def test_only_remaining_stages_run_and_failure_stops(recovery, failure, expected, phase):
    result = run(recovery, FAIL_PHASE=failure)
    assert result.returncode == (23 if failure else 0), result.stdout + result.stderr
    assert trace(recovery) == expected
    logdir = recovery["recovery_dir"]
    assert (logdir / "exit_code").read_text().strip() == str(result.returncode)
    assert f"phase={phase} exit_code={result.returncode}" in (logdir / "status.txt").read_text()
    assert "source validation output" in (logdir / "validating_pretrain.log").read_text()
    assert (recovery["source"] / "stage_exit_code").read_text() == "1\n"
    assert (recovery["source"] / "checkpoint_sentinel").read_text() == "existing checkpoint bytes\n"
    assert (recovery["root"] / "runs/launcher_logs/original_pipeline/status.txt").read_text() == "phase=pretraining exit_code=1\n"
    if not failure:
        assert (recovery["posttrain"] / "source.txt").read_text().strip() == str(recovery["source"])
        assert (recovery["eval"] / "source.txt").read_text().strip() == str(recovery["posttrain"])
    elif failure in ("source", "posttrain"):
        assert not recovery["posttrain"].exists()
        assert not recovery["eval"].exists()


@pytest.mark.parametrize("existing", ["recovery_dir", "posttrain", "eval"])
def test_existing_output_is_rejected_without_overwrite(recovery, existing):
    target = recovery[existing]
    target.mkdir(parents=True)
    sentinel = target / "sentinel"
    sentinel.write_text("preserve me\n")
    result = run(recovery)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert trace(recovery) == []
    assert list(target.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve me\n"
    if existing != "recovery_dir":
        assert not recovery["recovery_dir"].exists()


def test_rerun_preserves_completed_recovery(recovery):
    assert run(recovery).returncode == 0
    before = {p: p.read_bytes() for p in recovery["recovery_dir"].iterdir()}
    result = run(recovery)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert trace(recovery) == ["source", "posttrain", "eval"]
    assert {p: p.read_bytes() for p in recovery["recovery_dir"].iterdir()} == before


def test_dry_run_only_previews_wrappers_without_creating_outputs(recovery):
    result = run(recovery, DRY_RUN="1", FAIL_PHASE="source")
    assert result.returncode == 0, result.stdout + result.stderr
    assert trace(recovery) == ["posttrain", "eval"]
    assert "Command preview only: posttrain" in result.stdout
    assert "Command preview only: eval" in result.stdout
    for key in ("recovery_dir", "posttrain", "eval"):
        assert not recovery[key].exists()
    assert not Path(recovery["env"]["LOCK_TRACE"]).exists()


@pytest.mark.parametrize("source", ["", "relative/path", "/nonexistent/pretrain/source"])
def test_source_must_exist_before_any_stage(recovery, source):
    result = run(recovery, PRETRAIN_RUN_DIR=source)
    assert result.returncode != 0
    assert trace(recovery) == []
    assert not recovery["recovery_dir"].exists()


def test_pipeline_lock_failure_prevents_all_stages(recovery):
    result = run(recovery, LOCK_FAIL="1")
    assert result.returncode != 0 and "Another DROID pipeline" in result.stderr
    assert trace(recovery) == []
    assert not recovery["recovery_dir"].exists()
