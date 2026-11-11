#!/usr/bin/env python3
"""CPU preflight and post-training gates for the unchanged mixed 80k recipe."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml

INITIAL = '530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89'
STATS = 'd5a2bc19752172416377bc8a1de3b030f2b419930fdf08a8660158cc77189e00'
SAMPLING = '0776bbb4fbf7ec0cffdfb3d907c8352d3932e8002fba0c9db698249f75a1870f'
GALAXEA = 'galaxea_r1lite_stage1'
LIBERO = 'libero_plus_rot6d'
MIX = 'galaxea_libero_plus_stage1_80_20'
REFERENCE = Path('/mnt/zhaoshijie/checkpoints/starVLA/galaxea_libero_mixed_ln_8gpu_80k_bj5_20261105_153705')
CONFIG_REL = 'examples/pretrain/GalaxeaLiberoMix/train_files/starvla_mixed_ln_80k.yaml'
DS_REL = 'examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json'
EXPECTED_FILES = {
    CONFIG_REL: 'a99029484931a2792c80b0902feeecc335bd925591988d03fa9aa8cdc9301ec1',
    DS_REL: 'e0d64ea23a733ca53fa86af037452f6d4362feb2cba9df331bc2ad2a66c6744d',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(value)
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n')


def batch_size(per_device, global_batch=None):
    require(per_device in (4, 8), 'per-device batch must be 4 or 8')
    expected = per_device * 16
    require(global_batch is None or global_batch == expected, 'Wrong global batch for 16 ranks')
    return expected


def require_source(cfg, per_device=None, smoke=False):
    data, trainer = cfg['datasets']['vla_data'], cfg['trainer']
    require(cfg['seed'] == 42 and cfg['framework']['name'] == 'QwenPI', 'Wrong seed/framework')
    action = cfg['framework']['action_model']
    require(all(action[key] == value for key, value in dict(action_dim=90, state_dim=90,
            action_horizon=8, decoder_input_norm=True).items()), 'Wrong action architecture')
    require(data['data_mix'] == MIX and data['dataset_py'] == 'gr00t_sharded', 'Wrong source mixture')
    require(data['include_state'] is False and data['num_workers'] == 4, 'Wrong state/worker config')
    require(data['use_relative_action'] is True and data['use_percentiles'] is True, 'Wrong action normalization')
    require(data['shard_size'] == 1024 and data['episode_sampling_rate'] == 1.0, 'Wrong data sampling')
    require(trainer['pretrained_checkpoint'] is None and trainer['is_resume'] is False,
            'Pretraining must initialize fresh')
    require(trainer['gradient_accumulation_steps'] == 1, 'Wrong gradient accumulation')
    require(trainer['max_train_steps'] == (3 if smoke else 80000), 'Wrong pretraining length')
    require(trainer['num_warmup_steps'] == (1 if smoke else 5000), 'Wrong warmup')
    require(trainer['save_interval'] == (3 if smoke else 10000), 'Wrong save interval')
    require(trainer['eval_interval'] == trainer['logging_frequency'] == (1 if smoke else 100),
            'Wrong logging/evaluation interval')
    if per_device is not None:
        require(data['per_device_batch_size'] == per_device, 'Wrong per-device batch')


def gpu_gate():
    cards = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], timeout=20)).findall('gpu')
    require(len(cards) == 8, 'Expected 8 local GPUs')
    for card in cards:
        recovery = (card.findtext('gpu_recovery_action'),
                    card.findtext('remapped_rows/remapped_row_pending'),
                    card.findtext('remapped_rows/remapped_row_failure'))
        require(recovery == ('None', 'No', 'No'), f'GPU recovery pending: {recovery}')
        require(card.findtext('ecc_errors/volatile/dram_uncorrectable') == '0', 'Uncorrectable GPU ECC')
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], timeout=20).strip(), 'GPU processes already active')
    return [card.findtext('uuid') for card in cards]


def preflight(args):
    repo = Path(args.repo).resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(not args.expected_commit or commit == args.expected_commit, 'Wrong pinned pretraining commit')
    reference_run = Path(args.reference_run).resolve(strict=True)
    require(not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip(),
            'Pinned pretraining checkout is dirty')
    for relative, sha in EXPECTED_FILES.items():
        require(digest(repo / relative) == sha, f'Pinned file changed: {relative}')
    cfg = yaml.safe_load((repo / CONFIG_REL).read_text())
    require_source(cfg, per_device=8)
    require(digest(reference_run / 'dataset_statistics.json') == STATS, 'Reference statistics changed')
    require(digest(reference_run / 'data_sampling_manifest.json') == SAMPLING, 'Reference sampling changed')
    sampling = load_json(reference_run / 'data_sampling_manifest.json')
    require(sampling['data_mix'] == MIX and sampling['seed'] == 42, 'Wrong reference sampling')
    require(set(sampling['by_robot']) == {GALAXEA, LIBERO}, 'Wrong reference source tags')
    data_hashes = {}
    for row in sampling['datasets']:
        root = Path(row['dataset_path'])
        cache = Path(row['frame_cache_path'])
        require(root.is_dir() and cache.is_dir(), f'Missing source/cache: {root}, {cache}')
        for relative in ('meta/info.json', 'meta/stats.json', 'meta/relative_stats.json',
                         'meta/modality.json', 'meta/tasks.jsonl'):
            path = root / relative
            require(path.is_file(), f'Missing source metadata: {path}')
            data_hashes[str(path)] = digest(path)
    base = Path(cfg['framework']['qwenvl']['base_vlm'])
    require((base / 'config.json').is_file(), 'Base VLM missing')
    weight_files = sorted(base.glob('*.safetensors')) + sorted(base.glob('pytorch_model*.bin'))
    require(bool(weight_files), 'Base VLM weight shards missing')
    base_hashes = {str(p.relative_to(base)): digest(p) for p in sorted(base.glob('*.json'))}
    base_weights = {p.name: p.stat().st_size for p in weight_files}
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', 'starVLA', 'gr00t',
                                      'examples/pretrain/canonical'], text=False).decode().split('\0')
    code_hashes = {name: digest(repo / name) for name in tracked
                   if name and Path(name).suffix in ('.py', '.yaml', '.json')}
    launch_files = ('run_mixed_16gpu_pretrain.sh', 'coordinate_nodes.py', 'validate_mixed16.py',
                    'accelerate_mixed_2nodes.yaml')
    launcher = Path(__file__).resolve().parent
    payload = dict(protocol='galaxea-libero-mixed16-pretrain-v1', repo=str(repo), commit=commit,
                   source_config_sha256=digest(repo / CONFIG_REL), deepspeed_sha256=digest(repo / DS_REL),
                   seed=42, nodes=2, world_size=16, per_device_batch_size=args.per_device_batch_size,
                   global_batch=batch_size(args.per_device_batch_size, args.global_batch),
                   smoke=args.smoke, code_sha256=code_hashes,
                   launcher_sha256={name: digest(launcher / name) for name in launch_files},
                   metadata_sha256=data_hashes, reference_statistics_sha256=STATS,
                   reference_sampling_sha256=SAMPLING, base_vlm_json_sha256=base_hashes,
                   base_vlm_weight_bytes=base_weights, expected_initial_model_sha256=INITIAL,
                   runtime_versions={name: importlib.metadata.version(name) for name in
                    ('torch', 'transformers', 'accelerate', 'deepspeed', 'numpy', 'scipy', 'Pillow', 'torchcodec')})
    if args.report:
        atomic_json(args.report, dict(node_rank=args.node_rank, protocol=payload))
    print(json.dumps(dict(status='preflight_passed', node_rank=args.node_rank,
                         protocol_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                         global_batch=payload['global_batch']), sort_keys=True))


def compare(args):
    root = Path(args.launcher)
    reports = [load_json(root / f'node{rank}/preflight.json') for rank in (0, 1)]
    require(reports[0]['node_rank'] == 0 and reports[1]['node_rank'] == 1, 'Wrong node reports')
    require(reports[0]['protocol'] == reports[1]['protocol'], 'Two-node protocol mismatch')
    print('Two-node code, runtime, source metadata, base VLM and launch configuration match')


def trained(args):
    run = Path(args.run)
    reference_run = Path(getattr(args, 'reference_run', REFERENCE))
    cfg = yaml.safe_load((run / 'config.full.yaml').read_text())
    require_source(cfg, per_device=args.per_device_batch_size, smoke=args.smoke)
    global_batch = batch_size(args.per_device_batch_size, args.global_batch)
    initial = load_json(run / 'initial_model_sha256.json')
    require(initial['parameters']['model']['sha256'] == INITIAL, 'Initial weights differ from original experiment')
    require(initial['seed'] == 42 and initial['configured_pretrained_checkpoint'] is None,
            'Wrong initialization provenance')
    require(digest(run / 'dataset_statistics.json') == STATS, 'Training normalization differs from original')
    require(digest(run / 'data_sampling_manifest.json') == SAMPLING, 'Initial data schedule differs from original')
    require(set(load_json(run / 'dataset_statistics.json')) == {GALAXEA, LIBERO}, 'Wrong training statistics tags')
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    steps = 3 if args.smoke else 80000
    require(bool(records) and records[-1]['step'] == steps, 'Training has not reached final update')
    require(all(math.isfinite(row['action_dit_loss']) for row in records), 'Nonfinite training loss')
    require(all(row['source_exposure/cumulative_examples'] == row['step'] * global_batch for row in records),
            'Wrong cumulative global sample exposure')
    last = records[-1]
    require(last[f'source_exposure/{GALAXEA}/cumulative_examples'] +
            last[f'source_exposure/{LIBERO}/cumulative_examples'] == steps * global_batch,
            'Source exposure contains missing/extra source tags')
    require(last[f'source_exposure/{GALAXEA}/cumulative_examples'] > 0, 'No Galaxea exposure')
    if not args.smoke:
        require(last[f'source_exposure/{LIBERO}/cumulative_examples'] > 0, 'No LIBERO exposure')
    checkpoint = run / 'final_model/pytorch_model.pt'
    require(checkpoint.is_file() and checkpoint.stat().st_size == 16358897228,
            'Incomplete final model checkpoint')
    import torch
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    reference = torch.load(reference_run / 'final_model/pytorch_model.pt', map_location='cpu', mmap=True, weights_only=True)
    require(isinstance(state, dict) and isinstance(reference, dict), 'Checkpoint must be a full model state dict')
    require(set(state) == set(reference), 'Final checkpoint keys differ from the original full architecture')
    require(all(torch.is_tensor(value) and torch.is_tensor(reference[name]) and
                value.shape == reference[name].shape and value.dtype == reference[name].dtype
                for name, value in state.items()), 'Final checkpoint tensor schema differs from original')
    tensor_count = len(state)
    tensor_numel = sum(value.numel() for value in state.values())
    del state, reference
    report = dict(validated=True, steps=steps, global_batch=global_batch,
                  per_device_batch_size=args.per_device_batch_size, world_size=16,
                  checkpoint=str(checkpoint.resolve()), bytes=checkpoint.stat().st_size,
                  initial_model_sha256=INITIAL, statistics_sha256=STATS, sampling_sha256=SAMPLING,
                  final_metrics=last, checkpoint_tensors=tensor_count, checkpoint_numel=tensor_numel)
    atomic_json(run / 'launch/pretrain_validation.json', report)
    print(json.dumps({key: report[key] for key in ('validated', 'steps', 'global_batch', 'bytes')}))


def complete(args):
    run, launcher = Path(args.run), Path(args.launcher)
    for rank in (0, 1):
        require((launcher / f'node{rank}/train_exit').read_text().strip() == '0', 'Node training did not exit successfully')
        require((launcher / f'node{rank}/exit_code').read_text().strip() == '0', 'Node did not validate successfully')
        require(not (launcher / f'node{rank}/failure.json').exists(), 'Node failure marker present')
    report = load_json(run / 'launch/pretrain_validation.json')
    require(report['validated'] is True, 'Training validation missing')
    checkpoint = run / 'final_model/pytorch_model.pt'
    marker = dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), arm='mixed',
                  steps=report['steps'], checkpoint=str(checkpoint.resolve()), bytes=checkpoint.stat().st_size)
    atomic_json(run / 'pretrain_success.json', marker)
    atomic_text(run / 'exit_code', '0\n')
    atomic_text(launcher / 'exit_code', '0\n')
    print(json.dumps(marker, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'gpu', 'compare', 'trained', 'complete'))
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[4]))
    parser.add_argument('--expected-commit', help='Exact current code revision recorded by the run spec')
    parser.add_argument('--reference-run', default=str(REFERENCE), help='Immutable single-node baseline artifacts')
    parser.add_argument('--run')
    parser.add_argument('--launcher')
    parser.add_argument('--node-rank', type=int, choices=(0, 1), default=0)
    parser.add_argument('--per-device-batch-size', type=int, choices=(4, 8), default=4)
    parser.add_argument('--global-batch', type=int, choices=(64, 128))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.mode == 'gpu':
        print(json.dumps(dict(status='gpu_gate_passed', gpu_uuids=gpu_gate())))
    else:
        globals()[args.mode](args)


if __name__ == '__main__':
    main()
