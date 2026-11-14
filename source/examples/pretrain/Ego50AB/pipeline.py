#!/usr/bin/env python3
"""Finite, fail-closed pretraining -> posttraining -> evaluation pipeline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def terminate_owned_group(child):
    """Drain this pipeline's whole process group, even if its leader exited."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=10)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)

def stage_commands(spec, mode):
    smoke = mode == 'smoke'
    suffix = '_smoke3' if smoke else ''
    pre_id = spec['pretrain_run_id'] + suffix
    post_id = spec['posttrain_run_id'] + suffix
    pre_dir = str(Path(spec['run_root']) / pre_id)
    common = dict(REPO_DIR=spec['repo'], PYTHON_BIN=spec['python_bin'], RUN_ROOT_DIR=spec['run_root'],
                  ARM=spec['arm'], SMOKE_TEST=str(int(smoke)), DRY_RUN='0')
    pre_env = dict(common, RUN_ID=pre_id, MASTER_PORT=str(spec['pretrain_port']))
    post_env = dict(common, RUN_ID=post_id, EVAL_PYTHON=spec['python_bin'], REQUIRE_PAIRED_SOURCE='0',
                    PRETRAIN_RUN_DIR=pre_dir, PRETRAIN_LAUNCHER_DIR=pre_dir,
                    PAIR_PROTOCOL_DIR=spec['posttrain_contract'] + ('_smoke' if smoke else ''),
                    EVAL_OUTPUT_DIR=spec['eval_output_dir'] + suffix,
                    MASTER_PORT=str(spec['posttrain_port']), EVAL_BASE_PORT=str(spec['eval_base_port']),
                    ENV_OVERLAY=spec['env_overlay'], LIBERO_HOME=spec['libero_home'],
                    WANDB_MODE='disabled' if smoke else 'offline')
    root = Path(spec['repo']) / 'examples/pretrain/Ego50AB'
    return [('pretraining', ['bash', str(root / 'train_files/run_pretrain.sh')], pre_env),
            ('posttraining_and_evaluation', ['bash', str(root / 'downstream/run_libero_ego50_60k_eval.sh')], post_env)]

def run(spec, mode):
    root = Path(spec['control_dir'])
    if mode == 'formal':
        assert json.loads((root / 'smoke_success.json').read_text())['validated'] is True
    directory = root / mode
    directory.mkdir(exist_ok=False)
    child = None
    phase = 'starting'
    base_env = os.environ.copy()
    for key in list(base_env):
        if key.startswith(('ACCELERATE_', 'NCCL_')) or key in {
            'PRETRAINED_CHECKPOINT','PAIRED_PRETRAIN_RUN_DIR','PYTHONPATH','CUDA_VISIBLE_DEVICES',
            'SMOKE_TEST','RUN_ID','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE',
            'MASTER_ADDR','MASTER_PORT','PYTHONOPTIMIZE','MUJOCO_EGL_DEVICE_ID'}:
            base_env.pop(key, None)
    base_env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    def stopped(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        for phase, command, overrides in stage_commands(spec, mode):
            record = dict(phase=phase, mode=mode, arm=spec['arm'], host=spec['host'], command=command,
                          stage_env=overrides, updated_at=datetime.now(timezone.utc).isoformat())
            write_json(directory / 'status.json', record)
            print(json.dumps(record), flush=True)
            child = subprocess.Popen(command, env=dict(base_env, **overrides), start_new_session=True)
            rc = child.wait()
            if rc:
                raise RuntimeError(f'{phase} exited with status {rc}')
            child = None
        suffix = '_smoke3' if mode == 'smoke' else ''
        postrun = Path(spec['run_root']) / (spec['posttrain_run_id'] + suffix)
        report = json.loads((postrun / 'eval_success.json').read_text())
        assert report['validated'] is True
        assert report['episode_count'] == (1 if mode == 'smoke' else 10030)
        assert (postrun / 'exit_code').read_text().strip() == '0'
        success = dict(validated=True, arm=spec['arm'], mode=mode, evaluation=report,
                       completed_at=datetime.now(timezone.utc).isoformat())
        write_json(root / f'{mode}_success.json', success)
        write_json(directory / 'status.json', dict(phase='completed', **success))
        (directory / 'exit_code').write_text('0\n')
    except BaseException as exc:
        if child is not None:
            terminate_owned_group(child)
        write_json(directory / 'failure.json', dict(phase=phase, error=str(exc), arm=spec['arm'],
                   updated_at=datetime.now(timezone.utc).isoformat()))
        (directory / 'exit_code').write_text('1\n')
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--mode', choices=['smoke','formal'], required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    if args.dry_run:
        print(json.dumps(stage_commands(spec, args.mode), indent=2))
    else:
        run(spec, args.mode)
