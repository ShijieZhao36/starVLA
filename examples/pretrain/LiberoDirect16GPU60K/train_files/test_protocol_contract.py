#!/usr/bin/env python3
"""CPU-only regression checks for batch/step and fresh-init validation gates."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'validate_libero_direct_16gpu_60k_b4.py'
TREE = ast.parse(SOURCE.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extracted(*names):
    selected = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(copy=copy, json=json, math=math, Path=Path, require=require,
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


if __name__ == '__main__':
    unittest.main()
