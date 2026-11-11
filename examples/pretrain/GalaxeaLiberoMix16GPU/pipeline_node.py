#!/usr/bin/env python3
"""Finite two-node pretraining and rank-0 posttraining/evaluation pipeline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def terminate_owned_group(child, grace_seconds=30):
    """Drain the owned process group even when its leader has already exited."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(.2)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=10)


def selected_paths(spec, mode):
    smoke = mode == 'smoke'
    suffix = '_smoke3' if smoke else ''
    return dict(control=Path(spec['control_dir']) / mode,
                pretrain=Path(spec['run_root']) / (spec['pretrain_run_id'] + suffix),
                posttrain=Path(spec['run_root']) / (spec['posttrain_run_id'] + suffix),
                evaluation=Path(spec['eval_output_dir'] + suffix))


def stage_commands(spec, rank, mode='formal'):
    smoke = mode == 'smoke'
    paths = selected_paths(spec, mode)
    common = {
        'NODE_RANK': str(rank), 'PYTHON_BIN': spec['python_bin'],
        'REPO_DIR': spec['pretrain_repo'], 'RUN_ROOT_DIR': spec['run_root'],
        'EXPECTED_COMMIT': spec['pretrain_commit'],
        'REFERENCE_RUN': spec['reference_run'],
        'GLOBAL_BATCH': str(spec['pretrain_global_batch']),
        'PER_DEVICE_BATCH_SIZE': str(spec['pretrain_per_device_batch']),
        'MASTER_ADDR': spec['master_addr'], 'DRY_RUN': '0',
    }
    launcher = str(Path(spec['artifact_dir']) / 'train_files/run_mixed_16gpu_pretrain.sh')
    pretrain = dict(common, RUN_ID=paths['pretrain'].name,
                    LAUNCHER_DIR=str(paths['control'] / 'pretrain'),
                    MASTER_PORT=str(spec['smoke_port'] if smoke else spec['pretrain_port']),
                    SMOKE_TEST=str(int(smoke)))
    commands = [('pretraining', ['bash', launcher], pretrain)]
    if rank == 0:
        downstream = dict(
            REPO_DIR=spec['posttrain_repo'], PYTHON_BIN=spec['python_bin'],
            EVAL_PYTHON=spec['python_bin'], ARM='mixed', REQUIRE_PAIRED_SOURCE='0',
            PRETRAIN_RUN_DIR=str(paths['pretrain']),
            PRETRAIN_LAUNCHER_DIR=str(paths['control'] / 'pretrain'),
            PAIR_PROTOCOL_DIR=str(paths['control'] / 'posttrain_contract'),
            RUN_ID=paths['posttrain'].name, RUN_ROOT_DIR=spec['run_root'],
            EVAL_OUTPUT_DIR=str(paths['evaluation']), MASTER_PORT=str(spec['posttrain_port']),
            EVAL_BASE_PORT=str(spec['eval_base_port']), ENV_OVERLAY=spec['env_overlay'],
            LIBERO_HOME=spec['libero_home'], SMOKE_TEST=str(int(smoke)),
            SOURCE_SMOKE_TEST=str(int(smoke)), DRY_RUN='0',
            WANDB_MODE='disabled' if smoke else 'offline',
        )
        commands.append(('posttraining_and_evaluation', ['bash', spec['posttrain_runner']], downstream))
    return commands


def check_pretraining(spec, mode='formal'):
    paths = selected_paths(spec, mode)
    control, run = paths['control'] / 'pretrain', paths['pretrain']
    assert (control / 'exit_code').read_text().strip() == '0'
    for rank in (0, 1):
        assert (control / f'node{rank}/exit_code').read_text().strip() == '0'
    assert (run / 'exit_code').read_text().strip() == '0'
    marker = json.loads((run / 'pretrain_success.json').read_text())
    assert marker['arm'] == 'mixed' and marker['steps'] == (3 if mode == 'smoke' else 80000)
    checkpoint = run / 'final_model/pytorch_model.pt'
    assert Path(marker['checkpoint']).resolve() == checkpoint.resolve()
    assert checkpoint.stat().st_size == marker['bytes'] > 1_000_000


def run(spec, rank, mode='formal'):
    control_root = Path(spec['control_dir'])
    if mode == 'formal':
        assert json.loads((control_root / 'smoke_success.json').read_text())['validated'] is True
    paths = selected_paths(spec, mode)
    root = paths['control']
    root.mkdir(parents=True, exist_ok=True)
    node = root / f'pipeline_node{rank}'
    node.mkdir(exist_ok=False)
    phase = 'starting'
    child = None
    started = datetime.now(timezone.utc).isoformat()
    base_env = os.environ.copy()
    for key in list(base_env):
        if key.startswith(('ACCELERATE_', 'NCCL_')) or key in {
            'PRETRAINED_CHECKPOINT', 'PAIRED_PRETRAIN_RUN_DIR', 'PYTHONPATH',
            'CUDA_VISIBLE_DEVICES', 'GLOBAL_BATCH', 'PER_DEVICE_BATCH_SIZE',
            'SMOKE_TEST', 'SOURCE_SMOKE_TEST', 'RUN_ID', 'LAUNCHER_DIR',
            'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
            'MASTER_ADDR', 'MASTER_PORT', 'PYTHONOPTIMIZE', 'MUJOCO_EGL_DEVICE_ID'}:
            base_env.pop(key, None)
    base_env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='offline')

    def stopped(sig, frame):
        raise RuntimeError(f'Pipeline interrupted by signal {sig}')

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        for phase, command, overrides in stage_commands(spec, rank, mode):
            if phase == 'posttraining_and_evaluation':
                check_pretraining(spec, mode)
            status = dict(phase=phase, mode=mode, node_rank=rank, started_at=started,
                          updated_at=datetime.now(timezone.utc).isoformat(), command=command, stage_env=overrides)
            atomic_json(node / 'status.json', status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            child = subprocess.Popen(command, env=dict(base_env, **overrides), start_new_session=True)
            rc = child.wait()
            if rc:
                raise RuntimeError(f'{phase} exited with status {rc}')
            child = None
        if rank == 0:
            postrun, evaluation = paths['posttrain'], paths['evaluation']
            assert (postrun / 'exit_code').read_text().strip() == '0'
            report = json.loads((postrun / 'eval_success.json').read_text())
            assert report['validated'] is True and report['episode_count'] == (1 if mode == 'smoke' else 10030)
            if mode == 'formal':
                assert report['execution_errors'] == 0
            assert (evaluation / 'exit_code').read_text().strip() == '0'
            success = dict(validated=True, mode=mode, completed_at=datetime.now(timezone.utc).isoformat(),
                           pretrain_run_id=paths['pretrain'].name, posttrain_run_id=paths['posttrain'].name,
                           validation=report)
            if (evaluation / 'summary.json').is_file():
                success['evaluation'] = json.loads((evaluation / 'summary.json').read_text())
            atomic_json(root / 'experiment_success.json', success)
            atomic_json(control_root / f'{mode}_success.json', success)
            (root / 'exit_code').write_text('0\n')
        (node / 'exit_code').write_text('0\n')
        atomic_json(node / 'status.json', dict(phase='completed', mode=mode, node_rank=rank,
                    updated_at=datetime.now(timezone.utc).isoformat()))
    except BaseException as exc:
        if child is not None:
            terminate_owned_group(child)
        atomic_json(node / 'failure.json', dict(phase=phase, mode=mode, node_rank=rank,
                    error=str(exc), updated_at=datetime.now(timezone.utc).isoformat()))
        (node / 'exit_code').write_text('1\n')
        if rank == 0:
            (root / 'exit_code').write_text('1\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--node-rank', type=int, choices=[0, 1], required=True)
    parser.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    assert spec['pretrain_global_batch'] == 16 * spec['pretrain_per_device_batch'] == 64
    assert spec['posttrain_global_batch'] == 64
    if args.dry_run:
        print(json.dumps(stage_commands(spec, args.node_rank, args.mode), indent=2))
    else:
        run(spec, args.node_rank, args.mode)


if __name__ == '__main__':
    main()
