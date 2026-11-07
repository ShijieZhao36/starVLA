#!/usr/bin/env python3
"""Read-only reuse of the completed training smoke and separate reference eval gates."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
SOURCE_RUN = Path('/mnt/zhaoshijie/checkpoints/starVLA/libero_plus_direct_ln_16gpu_30k_20261107_184512_smoke3')
SOURCE_COMMIT = '980a2d1cf62f186686c6d07acfcd83bfbf029fc8'
REFERENCE_RUN = Path('/mnt/zhaoshijie/checkpoints/starVLA/libero_from_galaxea_control_ln_8gpu_60k_20261105_pair')
REFERENCE_CHECKPOINT = REFERENCE_RUN / 'final_model/pytorch_model.pt'
REFERENCE_CONTRACT = Path('/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_posttrain60k_20261105_contract')
TRAINING_DATA = Path('/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1/dataset_absolute')
EXPERIMENT = Path('examples/pretrain/LiberoDirect16GPU')
RENDERER_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 5, 5, 6, 7]


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def report(path, value):
    if path:
        path = Path(path).resolve()
        require(not path.is_relative_to(SOURCE_RUN), 'Cannot write into old smoke evidence')
        require(not path.is_relative_to(REFERENCE_RUN), 'Cannot write into reference experiment')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)
    print(json.dumps(value, indent=2, sort_keys=True))


def reference_identity():
    marker = load(REFERENCE_RUN / 'posttrain_success.json')
    require(marker['steps'] == 60000, 'Reference did not complete 60k')
    require(Path(marker['checkpoint']).resolve() == REFERENCE_CHECKPOINT.resolve(), 'Wrong reference checkpoint')
    require(REFERENCE_CHECKPOINT.is_file() and REFERENCE_CHECKPOINT.stat().st_size == marker['bytes'] == 16358897228,
            'Reference checkpoint bytes differ')
    require((REFERENCE_RUN / 'exit_code').read_text().strip() == '0', 'Reference pipeline did not exit successfully')
    statistics = digest(REFERENCE_RUN / 'dataset_statistics.json')
    require(statistics == '28493dba8beffeb8abaef0254a1e8ef81d1c917005e2c3362237b5add036c939',
            'Reference normalization differs')
    return {'checkpoint': str(REFERENCE_CHECKPOINT), 'bytes': REFERENCE_CHECKPOINT.stat().st_size,
            'mtime_ns': REFERENCE_CHECKPOINT.stat().st_mtime_ns, 'statistics_sha256': statistics,
            'purpose': 'diagnostic_only; infrastructure acceptance, never a training initialization'}


def training(args):
    repo = args.repo.resolve(strict=True)
    train = repo / EXPERIMENT / 'train_files'
    old = SOURCE_RUN / 'launch'
    protocols = [load(old / f'direct_protocol_node{rank}.json') for rank in (0, 1)]
    for rank, value in enumerate(protocols):
        require(value.pop('node_rank') == rank, 'Wrong source node rank')
        require(value['repo_commit'] == SOURCE_COMMIT, 'Unknown source training commit')
        require((old / f'train_exit_node{rank}').read_text().strip() == '0', 'Source training node failed')
    require(protocols[0] == protocols[1], 'Source node protocols differ')
    protocol = protocols[0]
    require(protocol['world_size'] == 16 and protocol['global_batch'] == 128 and protocol['steps'] == 3,
            'Wrong source training topology or budget')
    snapshots = {
        'run_libero_direct_16gpu_30k.sh': train / 'run_libero_direct_16gpu_30k.sh',
        'validate_libero_direct_16gpu.py': train / 'validate_libero_direct_16gpu.py',
        'accelerate_libero_direct_2nodes.yaml': train / 'accelerate_libero_direct_2nodes.yaml',
        'coordinate_nodes.py': train / 'coordinate_nodes.py',
        'deepspeed_libero_plus_ln_zero2.json': repo / 'examples/pretrain/LiberoPlus/train_files/deepspeed_libero_plus_ln_zero2.json',
        'starvla_libero_pair_ln_60k.yaml': repo / 'examples/pretrain/GalaxeaLiberoMix/train_files/starvla_libero_pair_ln_60k.yaml',
    }
    snapshot_hashes = {}
    for name, current in snapshots.items():
        snapshot_hashes[name] = digest(old / name)
        if name == 'run_libero_direct_16gpu_30k.sh':
            previous = (old / name).read_text()
            original = ' --output-dir "$EVAL_OUTPUT_DIR" "${smoke_args[@]}")'
            replacement = ' --output-dir "$EVAL_OUTPUT_DIR" --renderer-map ' + ','.join(map(str, RENDERER_MAP)) + ' "${smoke_args[@]}")'
            require(previous.count(original) == 1, 'Cannot identify the one original eval_args line')
            require(current.read_text() == previous.replace(original, replacement),
                    'Runner differs beyond the exact approved renderer-map argument')
        else:
            require(snapshot_hashes[name] == digest(current), 'Training smoke cannot be reused after changing ' + name)
    for name, key in [('run_libero_direct_16gpu_30k.sh', 'runner_sha256'),
                      ('validate_libero_direct_16gpu.py', 'validator_sha256'),
                      ('accelerate_libero_direct_2nodes.yaml', 'accelerate_sha256'),
                      ('starvla_libero_pair_ln_60k.yaml', 'configuration_sha256')]:
        require(snapshot_hashes[name] == protocol[key], 'Source snapshot does not match recorded hash: ' + name)
    validator = module('reuse_current_direct_validator', train / 'validate_libero_direct_16gpu.py')
    validation_args = SimpleNamespace(output=str(SOURCE_RUN), reference_contract=str(REFERENCE_CONTRACT),
                                     reference_run=str(REFERENCE_RUN), data=str(TRAINING_DATA), smoke=True, report=None)
    # initial() is read-only when report=None. trained() is deliberately not
    # called: it overwrites the old posttrain_success.json even in dry-run mode.
    initial = validator.initial(validation_args)
    recorded_core = load(old / 'reference_protocol_manifest.json')
    for name, expected in recorded_core['code_sha256'].items():
        require(digest(repo / name) == expected, 'Numerical/data code changed since training smoke: ' + name)
    metrics = [json.loads(line) for line in (SOURCE_RUN / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    require(metrics and all(validator.finite(row) for row in metrics), 'Source metrics empty or nonfinite')
    require(metrics[-1]['step'] == 3, 'Source training did not reach three steps')
    require(all(a['step'] < b['step'] for a, b in zip(metrics, metrics[1:])), 'Source steps are not increasing')
    source_key = f'source_exposure/{validator.pair.LIBERO}/cumulative_examples'
    for row in metrics:
        require(row['source_exposure/cumulative_examples'] == row['step'] * 128, 'Wrong effective source batch')
        require(row[source_key] == row['step'] * 128, 'Source includes non-LIBERO data')
        for key, value in row.items():
            if key.startswith('source_exposure/') and key.endswith('/cumulative_examples'):
                require(key in ('source_exposure/cumulative_examples', source_key) or value == 0,
                        'Unexpected source exposure: ' + key)
    checkpoint = SOURCE_RUN / 'final_model/pytorch_model.pt'
    require(checkpoint.is_file() and checkpoint.stat().st_size == validator.FINAL_BYTES, 'Source final weight is incomplete')
    expected_marker = {'arm': 'direct_16gpu', 'steps': 3, 'global_batch': 128,
                       'checkpoint': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                       'source_checkpoint': None, 'statistics_sha256': validator.REFERENCE_STATS,
                       'initial_model_sha256': validator.INITIAL_MODEL,
                       'reference_common_protocol_sha256': validator.REFERENCE_PROTOCOL}
    require(load(SOURCE_RUN / 'posttrain_success.json') == expected_marker, 'Source training marker mismatch')
    evidence_files = ['initial_model_sha256.json', 'posttrain_success.json', 'metrics.jsonl', 'dataset_statistics.json',
                      'config.full.yaml', 'launch/train_exit_node0', 'launch/train_exit_node1']
    return {'validated': True, 'training_smoke_reused_only': True, 'source_run': str(SOURCE_RUN),
            'source_commit': SOURCE_COMMIT, 'source_overall_pipeline': 'failed_during_evaluation; not reclassified',
            'permitted_runner_difference': {'evaluation_renderer_map_only': RENDERER_MAP,
                                            'training_command_unchanged': True},
            'snapshot_sha256': snapshot_hashes,
            'evidence_sha256': {name: digest(SOURCE_RUN / name) for name in evidence_files},
            'training_marker': expected_marker, 'initial_validation': initial,
            'reference_eval_identity': reference_identity()}


def reference_eval(args):
    repo, out, launcher = args.repo.resolve(strict=True), args.reference_eval_dir.resolve(strict=True), args.launcher.resolve(strict=True)
    identity = reference_identity()
    require(not out.is_relative_to(SOURCE_RUN) and not out.is_relative_to(REFERENCE_RUN), 'Diagnostic output overlaps prior experiment')
    require(not (out / 'failure.json').exists() and not list(out.glob('node*/failure.json')), 'Reference smoke has failure evidence')
    for rank in (0, 1):
        require((out / f'node{rank}/exit_code').read_text().strip() == '0', 'Reference eval node did not exit zero')
        require((launcher / f'node{rank}/reference_eval_exit_code').read_text().strip() == '0', 'Reference process did not finish cleanup')
    require((out / 'exit_code').read_text().strip() == '0', 'Reference combined eval did not exit zero')
    evaluator = module('reuse_reference_evaluator', repo / EXPERIMENT / 'eval_files/run_libero_plus_16gpu.py')
    parallel = repo / 'examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval'
    sys.path.insert(0, str(parallel))
    import task_queue
    verifier = module('reuse_reference_verifier', repo / 'examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py')
    shards = evaluator.split_tasks(evaluator.EXPECTED_COUNTS, smoke=True, smoke_sims_per_gpu=4)
    expected = {key for shard in shards for key in shard}
    require(len(expected) == 64 and all(len(shard) == 4 for shard in shards), 'Reference task split changed')
    records = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines() if line.strip()]
    require(all(record.get('attempt') == 1 for record in records), 'Reference smoke retried an episode')
    summary = task_queue.summarize_results(out / 'results.jsonl')
    validation = verifier.validate_records(records, summary, expected)
    require(validation['validated'] and validation['episode_count'] == 64, 'Reference smoke is incomplete')
    success = load(out / 'eval_success.json')
    require(success['validated'] and success['smoke'] and success['episode_count'] == 64, 'Reference success marker mismatch')
    contract = success['contract']
    require(contract['checkpoint'] == identity['checkpoint'] and contract['checkpoint_bytes'] == identity['bytes']
            and contract['checkpoint_mtime_ns'] == identity['mtime_ns'], 'Reference checkpoint identity changed')
    expected_flags = {'node_count': 2, 'gpu_count': 16, 'sims_per_gpu': 4, 'num_trials_per_task': 1,
                      'environment_seed': 7, 'num_steps_wait': 10, 'use_bf16': True,
                      'use_canonical_forward': True, 'gripper_open_sign': 'positive',
                      'policy_noise': 'original_unseeded', 'max_episode_attempts': 1, 'smoke': True,
                      'renderer_map': RENDERER_MAP}
    require(all(contract[key] == value for key, value in expected_flags.items()), 'Reference eval protocol differs')
    require(contract['task_assignment_sha256'] == hashlib.sha256(json.dumps(shards).encode()).hexdigest(), 'Reference task assignment differs')
    for name, expected_hash in contract['code_sha256'].items():
        require(digest(repo / name) == expected_hash, 'Evaluated code changed before formal launch: ' + name)
    for rank in (0, 1):
        completion = load(out / f'node{rank}/completion.json')
        require(completion['validated'] and completion['episode_count'] == 32 and completion['contract'] == contract,
                'Node reference eval validation or protocol differs')
    return {'validated': True, 'diagnostic_only': True, 'purpose': '16 policy GPUs with four simulators each; not formal benchmark results',
            'source_checkpoint': identity, 'episode_count': 64, 'reference_eval_dir': str(out),
            'validation': validation, 'contract': contract,
            'does_not_establish_failed_smoke_root_cause_fixed': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    for name in ('training', 'reference-eval'):
        child = sub.add_parser(name)
        child.add_argument('--repo', type=Path, required=True)
        child.add_argument('--report', type=Path)
        if name == 'reference-eval':
            child.add_argument('--reference-eval-dir', type=Path, required=True)
            child.add_argument('--launcher', type=Path, required=True)
    args = p.parse_args()
    value = training(args) if args.mode == 'training' else reference_eval(args)
    report(args.report, value)


if __name__ == '__main__':
    main()
