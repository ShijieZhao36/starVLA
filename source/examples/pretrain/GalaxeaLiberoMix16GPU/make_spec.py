#!/usr/bin/env python3
"""Create a fresh, same-checkout two-node run spec outside the repository."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
REFERENCE = '/mnt/zhaoshijie/checkpoints/starVLA/galaxea_libero_mixed_ln_8gpu_80k_bj5_20261105_153705'


def build_spec(repo, commit, run_id, hosts, master_addr, run_root, eval_root, python_bin,
               env_overlay, libero_home, reference_run, base_port=29715, eval_base_port=10483):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id) or run_id.endswith('_smoke3'):
        raise ValueError('Use a fresh safe run ID without the reserved _smoke3 suffix')
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        raise ValueError('Specify two different hosts in node-rank order')
    control = str(Path(run_root) / 'launcher_logs' / run_id)
    post_id = 'libero_from_' + run_id + '_60k'
    return dict(experiment='Galaxea/LIBERO80:20 16-GPU pre80k -> 8-GPU LIBERO post60k -> 10030 eval',
                created_at=datetime.now(timezone.utc).isoformat(), hosts=hosts, python_bin=python_bin,
                pretrain_repo=repo, posttrain_repo=repo, pretrain_commit=commit, posttrain_commit=commit,
                posttrain_runner=str(Path(repo) / 'examples/pretrain/GalaxeaLiberoMix/train_files/run_libero_pair_60k_eval.sh'),
                run_root=run_root, pretrain_run_id=run_id, posttrain_run_id=post_id,
                control_dir=control, artifact_dir=str(Path(control) / 'artifacts'),
                eval_output_dir=str(Path(eval_root) / (post_id + '_final')),
                env_overlay=env_overlay, libero_home=libero_home, reference_run=reference_run,
                master_addr=master_addr, smoke_port=base_port, pretrain_port=base_port + 1,
                posttrain_port=base_port + 2, eval_base_port=eval_base_port,
                pretrain_world_size=16, pretrain_per_device_batch=4, pretrain_global_batch=64,
                pretrain_steps=80000, posttrain_world_size=8, posttrain_per_device_batch=8,
                posttrain_global_batch=64, posttrain_steps=60000, seed=42, eval_seed=7,
                eval_world_size=8, eval_trials_per_task=1, eval_episodes=10030,
                screen_names=[f'{run_id}_n0', f'{run_id}_n1'],
                initial_model_sha256='530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(HERE.parents[2]))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--hosts', nargs=2, required=True)
    parser.add_argument('--master-addr', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--python-bin', default='/root/miniconda3/envs/MRA/bin/python')
    parser.add_argument('--run-root', default='/mnt/zhaoshijie/checkpoints/starVLA')
    parser.add_argument('--eval-root', default='/mnt/zhaoshijie/eval/starVLA/libero_plus')
    parser.add_argument('--env-overlay', default='/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_pair_validation_20261105/sim_overlay')
    parser.add_argument('--libero-home', default='/root/workspace/LIBERO-plus')
    parser.add_argument('--reference-run', default=REFERENCE)
    parser.add_argument('--base-port', type=int, default=29715)
    parser.add_argument('--eval-base-port', type=int, default=10483)
    args = parser.parse_args()
    repo = str(Path(args.repo).resolve(strict=True))
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(repo)):
        parser.error('--output must be outside the Git checkout')
    if output.exists():
        parser.error('Refusing to overwrite an existing spec')
    commit = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', repo, 'status', '--porcelain'], text=True).strip():
        parser.error('Commit changes before generating the pinned spec')
    spec = build_spec(repo, commit, args.run_id, args.hosts, args.master_addr,
                      str(Path(args.run_root).resolve()), str(Path(args.eval_root).resolve()),
                      args.python_bin, args.env_overlay, args.libero_home, args.reference_run,
                      args.base_port, args.eval_base_port)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(spec=str(output), commit=commit, control_dir=spec['control_dir'])))


if __name__ == '__main__':
    main()
