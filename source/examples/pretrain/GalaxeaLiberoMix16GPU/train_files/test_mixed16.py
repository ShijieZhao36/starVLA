#!/usr/bin/env python3
"""CPU tests for unsafe-launch rejection and cross-node child supervision."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mixed_gate', HERE / 'validate_mixed16.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
SOURCE = HERE.parents[3] / 'examples/pretrain/GalaxeaLiberoMix/train_files/starvla_mixed_ln_80k.yaml'


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2] != 'Z'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(SOURCE.read_text())

    def test_source_original_accepted_and_wrong_mix_rejected(self):
        gate.require_source(self.cfg, 8)
        self.cfg['datasets']['vla_data']['data_mix'] = 'galaxea_open_world_stage1'
        with self.assertRaisesRegex(ValueError, 'Wrong source mixture'):
            gate.require_source(self.cfg, 8)

    def test_wrong_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Wrong per-device batch'):
            gate.require_source(self.cfg, 4)
        self.assertEqual(gate.batch_size(4, 64), 64)
        self.assertEqual(gate.batch_size(8, 128), 128)
        with self.assertRaisesRegex(ValueError, 'Wrong global batch'):
            gate.batch_size(4, 128)

    def test_seed_and_resume_rejected(self):
        self.cfg['seed'] = 43
        with self.assertRaises(ValueError):
            gate.require_source(self.cfg, 8)
        self.cfg['seed'] = 42
        self.cfg['trainer']['is_resume'] = True
        with self.assertRaisesRegex(ValueError, 'initialize fresh'):
            gate.require_source(self.cfg, 8)

    def test_dry_run_commands_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name) / 'not_created'
            for smoke, batch in ((False, 64), (True, 128)):
                env = dict(os.environ, NODE_RANK='1', RUN_ID='audit_smoke3' if smoke else 'audit',
                           LAUNCHER_DIR=str(directory), DRY_RUN='1', SMOKE_TEST=str(int(smoke)),
                           GLOBAL_BATCH=str(batch), MASTER_ADDR='127.0.0.1')
                env.pop('PER_DEVICE_BATCH_SIZE', None)
                out = subprocess.check_output(['bash', str(HERE / 'run_mixed_16gpu_pretrain.sh')],
                                              env=env, text=True)
                self.assertIn('--num_machines 2 --num_processes 16 --machine_rank 1', out)
                self.assertIn(f'--datasets.vla_data.per_device_batch_size {batch // 16}', out)
                self.assertIn('--trainer.max_train_steps 3' if smoke else '--trainer.max_train_steps 80000', out)
                self.assertIn('--main_process_port 29715' if smoke else '--main_process_port 29716', out)
                self.assertFalse(directory.exists())

    def test_transport_defaults_do_not_depend_on_login_environment(self):
        env = {key:value for key,value in os.environ.items() if not key.startswith(('NCCL_', 'GLOO_'))}
        env.update(NODE_RANK='0', RUN_ID='transport_audit', LAUNCHER_DIR='/tmp/not-created',
                   DRY_RUN='1', MASTER_ADDR='127.0.0.1')
        command = ['bash', str(HERE / 'run_mixed_16gpu_pretrain.sh')]
        result = subprocess.check_output(command, env=env, text=True)
        self.assertIn('NCCL transport: gid_index=5 cross_nic=0 socket=eth0', result)
        env.update(NCCL_IB_GID_INDEX='3', NCCL_CROSS_NIC='2')
        overridden = subprocess.check_output(command, env=env, text=True)
        self.assertIn('NCCL transport: gid_index=3 cross_nic=2 socket=eth0', overridden)

    def test_actual_transport_environment_is_recorded_and_compared(self):
        values = dict(NCCL_SOCKET_IFNAME='eth0', GLOO_SOCKET_IFNAME='eth0',
                      NCCL_IB_HCA='=com0_0', NCCL_IB_GID_INDEX='5', NCCL_CROSS_NIC='0')
        with patch.dict(os.environ, values, clear=True):
            transport = gate.transport_environment()
        self.assertEqual(transport, dict(values, NCCL_IB_DISABLE=None))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for rank in (0,1):
                gate.atomic_json(root/f'node{rank}/preflight.json',
                                 dict(node_rank=rank,protocol=dict(transport_env=transport)))
            args=argparse.Namespace(launcher=str(root))
            gate.compare(args)
            wrong=dict(transport,NCCL_IB_GID_INDEX='4')
            gate.atomic_json(root/'node1/preflight.json',dict(node_rank=1,protocol=dict(transport_env=wrong)))
            with self.assertRaisesRegex(ValueError,'protocol mismatch'):
                gate.compare(args)

    def test_inconsistent_cli_batch_fails_before_launch(self):
        env = dict(os.environ, NODE_RANK='0', RUN_ID='audit', LAUNCHER_DIR='/tmp/no-mutation',
                   DRY_RUN='1', PER_DEVICE_BATCH_SIZE='4', GLOBAL_BATCH='128', MASTER_ADDR='127.0.0.1')
        proc = subprocess.run(['bash', str(HERE / 'run_mixed_16gpu_pretrain.sh')], env=env,
                              text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('GLOBAL_BATCH does not match', proc.stderr)

    def test_cross_node_runtime_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for rank in (0, 1):
                gate.atomic_json(root / f'node{rank}/preflight.json',
                                 dict(node_rank=rank, protocol=dict(runtime='same')))
            args = argparse.Namespace(launcher=str(root))
            gate.compare(args)
            gate.atomic_json(root / 'node1/preflight.json',
                             dict(node_rank=1, protocol=dict(runtime='different')))
            with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                gate.compare(args)

    def test_failure_prevents_success_publication(self):
        with tempfile.TemporaryDirectory() as name:
            run, launcher = Path(name) / 'run', Path(name) / 'launcher'
            for rank in (0, 1):
                gate.atomic_text(launcher / f'node{rank}/train_exit', '0\n')
                gate.atomic_text(launcher / f'node{rank}/exit_code', '0\n')
            gate.atomic_json(launcher / 'node1/failure.json', dict(error='failed'))
            with self.assertRaisesRegex(ValueError, 'failure marker'):
                gate.complete(argparse.Namespace(run=str(run), launcher=str(launcher)))
            self.assertFalse((run / 'pretrain_success.json').exists())
            self.assertFalse((launcher / 'exit_code').exists())

    def test_bad_loss_and_exposure_rejected_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            cfg = copy.deepcopy(self.cfg)
            cfg['datasets']['vla_data']['per_device_batch_size'] = 4
            (run / 'config.full.yaml').write_text(yaml.safe_dump(cfg))
            gate.atomic_json(run / 'initial_model_sha256.json', dict(seed=42,
                configured_pretrained_checkpoint=None, parameters=dict(model=dict(sha256=gate.INITIAL))))
            gate.atomic_json(run / 'dataset_statistics.json', {gate.GALAXEA: {}, gate.LIBERO: {}})
            rows = [dict(step=80000, action_dit_loss=float('nan'),
                         **{'source_exposure/cumulative_examples': 80000 * 64})]
            (run / 'metrics.jsonl').write_text(json.dumps(rows[0]) + '\n')
            args = argparse.Namespace(run=str(run), smoke=False, per_device_batch_size=4, global_batch=64)
            with patch.object(gate, 'digest', side_effect=lambda p: gate.STATS if Path(p).name == 'dataset_statistics.json' else gate.SAMPLING):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    gate.trained(args)
                rows[0]['action_dit_loss'] = .5
                rows[0]['source_exposure/cumulative_examples'] = 80000 * 128
                (run / 'metrics.jsonl').write_text(json.dumps(rows[0]) + '\n')
                with self.assertRaisesRegex(ValueError, 'global sample exposure'):
                    gate.trained(args)


class SupervisionTests(unittest.TestCase):
    def test_peer_failure_stops_owned_group_and_keeps_unrelated_process(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            launcher, run = root / 'launcher', root / 'run'
            pidfile = root / 'owned.json'
            outsider = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                        start_new_session=True)
            childcode = ('import json,os,subprocess,sys,time; '
                         'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                         'open(sys.argv[1],"w").write(json.dumps([os.getpid(),child.pid])); '
                         'time.sleep(60)')
            supervisor = subprocess.Popen([sys.executable, str(HERE / 'coordinate_nodes.py'), 'run',
                         '--run', str(run), '--launcher', str(launcher), '--node-rank', '0',
                         '--timeout', '30', '--log', str(root / 'child.log'), '--',
                         sys.executable, '-c', childcode, str(pidfile)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 8
                while not pidfile.is_file() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(pidfile.is_file())
                owned = json.loads(pidfile.read_text())
                gate.atomic_json(launcher / 'node1/failure.json', dict(error='synthetic peer failure'))
                supervisor.communicate(timeout=30)
                self.assertNotEqual(supervisor.returncode, 0)
                self.assertTrue((launcher / 'node0/failure.json').is_file())
                self.assertTrue(alive(outsider.pid))
                self.assertTrue(all(not alive(pid) for pid in owned))
            finally:
                if supervisor.poll() is None:
                    supervisor.terminate()
                    supervisor.communicate(timeout=30)
                outsider.terminate()
                outsider.wait(timeout=5)

    def test_missing_peer_barrier_times_out_and_records_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            proc = subprocess.run([sys.executable, str(HERE / 'coordinate_nodes.py'), 'wait',
                         '--run', str(root / 'run'), '--launcher', str(root / 'launcher'),
                         '--node-rank', '0', '--timeout', '.1', '--path', str(root / 'absent')],
                         capture_output=True, text=True, timeout=10)
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue((root / 'launcher/node0/failure.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
