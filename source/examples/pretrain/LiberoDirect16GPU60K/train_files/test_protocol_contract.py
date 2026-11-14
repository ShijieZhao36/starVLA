#!/usr/bin/env python3
"""CPU-only regression checks for batch/step and fresh-init validation gates."""
import ast
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'validate_libero_direct_16gpu_60k_b4.py'
TREE = ast.parse(SOURCE.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extracted(*names):
    selected = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(copy=copy, json=json, math=math, os=os, Path=Path, require=require,
                     INITIAL_MODEL='fresh-model', REFERENCE_STATS='stats', REFERENCE_PROTOCOL='protocol',
                     FINAL_BYTES=4, pair=SimpleNamespace(LIBERO='libero_plus_rot6d'), initial=lambda args: None,
                     save=lambda path, obj: Path(path).write_text(json.dumps(obj)))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class ProtocolTests(unittest.TestCase):
    def config(self):
        expected = dict(trainer=dict(pretrained_checkpoint='prior.pt', reload_modules=None, is_resume=False,
                                    max_train_steps=60000, num_warmup_steps=5000, save_interval=10000,
                                    eval_interval=100, logging_frequency=100, learning_rate=1e-4),
                        datasets=dict(vla_data=dict(per_device_batch_size=8)), seed=42)
        actual = copy.deepcopy(expected)
        actual['trainer'].update(pretrained_checkpoint=None, expected_initial_model_sha256='fresh-model')
        actual['datasets']['vla_data']['per_device_batch_size'] = 4
        return actual, expected

    def test_only_authorized_resolved_config_passes(self):
        compare = extracted('compare_config')['compare_config']
        actual, expected = self.config()
        compare(actual, expected, False)
        for path, value in [(('trainer', 'max_train_steps'), 30000),
                            (('datasets', 'vla_data', 'per_device_batch_size'), 8),
                            (('trainer', 'pretrained_checkpoint'), '30k-final.pt'),
                            (('trainer', 'is_resume'), True),
                            (('trainer', 'learning_rate'), 2e-4),
                            (('trainer', 'expected_initial_model_sha256'), 'wrong')]:
            mutated = copy.deepcopy(actual)
            target = mutated
            for name in path[:-1]:
                target = target[name]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                compare(mutated, expected, False)

    def test_output_root_can_move_without_changing_training(self):
        compare = extracted('compare_config')['compare_config']
        actual, expected = self.config()
        actual['run_root_dir'] = '/isolated/integration/checkpoints'
        expected['run_root_dir'] = '/historical/checkpoints'
        compare(actual, expected, False)
        actual['trainer']['learning_rate'] = 2e-4
        with self.assertRaises(ValueError):
            compare(actual, expected, False)

    def test_reviewed_helper_and_mixture_cannot_drift(self):
        ns = extracted('assert_integrated_source_pins')
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            helper = repo / 'helper.py'
            mixture = repo / 'starVLA/dataloader/gr00t_lerobot/mixtures.py'
            mixture.parent.mkdir(parents=True)
            helper.write_text('reviewed helper')
            mixture.write_text('reviewed mixture')
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            ns.update(pair=SimpleNamespace(__file__=str(helper), digest=digest),
                      APPROVED_HELPER=digest(helper), APPROVED_MIXTURES=digest(mixture))
            check = ns['assert_integrated_source_pins']
            check(repo)
            helper.write_text('unreviewed helper')
            with self.assertRaises(ValueError):
                check(repo)
            helper.write_text('reviewed helper')
            mixture.write_text('unreviewed mixture')
            with self.assertRaises(ValueError):
                check(repo)

    def test_smoke_still_requires_batch4(self):
        compare = extracted('compare_config')['compare_config']
        actual, expected = self.config()
        actual['trainer'].update(max_train_steps=3, num_warmup_steps=1, save_interval=3,
                                 eval_interval=1, logging_frequency=1)
        compare(actual, expected, True)
        actual['datasets']['vla_data']['per_device_batch_size'] = 8
        with self.assertRaises(ValueError):
            compare(actual, expected, True)

    def test_completion_rejects_old_protocol_nonlibero_and_nonfinite(self):
        trained = extracted('finite', 'trained')['trained']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'final_model').mkdir()
            (root / 'final_model/pytorch_model.pt').write_bytes(b'test')
            row = {'step':60000, 'loss':.1, 'source_exposure/cumulative_examples':3840000,
                   'source_exposure/libero_plus_rot6d/cumulative_examples':3840000}
            args = SimpleNamespace(output=temp, smoke=False)
            (root / 'metrics.jsonl').write_text(json.dumps(row)+'\n')
            trained(args)
            marker = json.loads((root / 'posttrain_success.json').read_text())
            self.assertEqual((marker['steps'], marker['global_batch']), (60000, 64))
            for key, value in [('step', 30000), ('source_exposure/cumulative_examples', 7680000),
                               ('source_exposure/libero_plus_rot6d/cumulative_examples', 7680000),
                               ('source_exposure/galaxea/cumulative_examples', 1), ('loss', float('nan'))]:
                mutated = dict(row, **{key:value})
                (root / 'metrics.jsonl').write_text(json.dumps(mutated)+'\n')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    trained(args)
            (root / 'metrics.jsonl').write_text(json.dumps(row)+'\n')
            (root / 'final_model/pytorch_model.pt').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                trained(args)


class TransportTests(unittest.TestCase):
    def check_runner_environment(self, overrides):
        repo = ROOT.parents[3]
        runners = [
            (repo / 'examples/pretrain/LiberoDirect16GPU/train_files/run_libero_direct_16gpu_30k.sh',
             'libero_plus_direct_ln_16gpu_30k_transport_smoke3'),
            (ROOT / 'run_libero_direct_16gpu_60k_b4.sh',
             'libero_plus_direct_ln_16gpu_b4_60k_transport_smoke3'),
        ]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            fake = directory / 'fake_python'
            fake.write_text('#!' + sys.executable + "\nimport json,os\nfrom pathlib import Path\n" +
                            "with Path(os.environ['ENV_TRACE']).open('a') as f:\n" +
                            " f.write(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('NCCL_', 'GLOO_'))})+'\\n')\n")
            fake.chmod(0o755)
            for index, (runner, run_id) in enumerate(runners):
                trace = directory / ('trace%s.jsonl' % index)
                env = {name: value for name, value in os.environ.items()
                       if not name.startswith(('NCCL_', 'GLOO_'))}
                env.update(REPO_DIR=str(repo), PYTHON_BIN=str(fake), NODE_RANK='0',
                           RUN_ID=run_id, LAUNCHER_DIR=str(directory / ('control%s' % index)),
                           SMOKE_TEST='1', DRY_RUN='1', ENV_TRACE=str(trace),
                           PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
                env.update(overrides)
                result = subprocess.run(['bash', str(runner)], env=env, capture_output=True,
                                        text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertEqual(row['NCCL_IB_GID_INDEX'], overrides.get('NCCL_IB_GID_INDEX', '5'))
                    self.assertEqual(row['NCCL_CROSS_NIC'], overrides.get('NCCL_CROSS_NIC', '0'))
                    self.assertEqual(row['NCCL_SOCKET_IFNAME'], 'eth0')
                    self.assertEqual(row['GLOO_SOCKET_IFNAME'], 'eth0')
                    self.assertTrue(row['NCCL_IB_HCA'].startswith('=com0_0,'))
                    self.assertNotIn('NCCL_IB_DISABLE', row)

    def test_sanitized_parent_gets_explicit_roce_defaults(self):
        self.check_runner_environment({})

    def test_explicit_roce_overrides_remain_effective(self):
        self.check_runner_environment({'NCCL_IB_GID_INDEX': '7', 'NCCL_CROSS_NIC': '1',
                                       'NCCL_IB_DISABLE': '1'})

    def test_transport_evidence_reports_effective_settings(self):
        function = extracted('transport_environment')['transport_environment']
        values = dict(NCCL_IB_GID_INDEX='5', NCCL_CROSS_NIC='0', NCCL_SOCKET_IFNAME='eth0',
                      GLOO_SOCKET_IFNAME='eth0', NCCL_IB_HCA='=com0_0,com0_1')
        with patch.dict(os.environ, values, clear=True):
            report = function()
        self.assertEqual(report, dict(values, NCCL_IB_DISABLE=None))
        with patch.dict(os.environ, dict(values, NCCL_IB_GID_INDEX='7'), clear=True):
            self.assertEqual(function()['NCCL_IB_GID_INDEX'], '7')


if __name__ == '__main__':
    unittest.main()
