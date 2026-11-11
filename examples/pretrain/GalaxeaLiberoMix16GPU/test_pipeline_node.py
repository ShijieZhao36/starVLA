"""Isolation, source acceptance and owned-process cleanup tests; CPU only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_module('mixed16_pipeline', HERE / 'pipeline_node.py')
gate = load_module('mixed16_post_gate', REPO / 'examples/pretrain/GalaxeaLiberoMix/train_files/validate_libero_pair_60k.py')
make_spec = load_module('mixed16_spec', HERE / 'make_spec.py')


def recipe(root):
    return dict(python_bin='/python', pretrain_repo='/same_checkout', posttrain_repo='/same_checkout',
                pretrain_commit='new_commit', posttrain_commit='new_commit', reference_run='/reference',
                run_root=str(root/'runs'), control_dir=str(root/'control'), artifact_dir='/artifacts',
                pretrain_run_id='fresh80k', posttrain_run_id='fresh60k', pretrain_global_batch=64,
                pretrain_per_device_batch=4, master_addr='127.0.0.1', smoke_port=29715,
                pretrain_port=29716, posttrain_port=29717, eval_base_port=10483,
                posttrain_runner='/same_checkout/posttrain.sh', env_overlay='/original_overlay',
                libero_home='/original_libero', eval_output_dir=str(root/'eval'), posttrain_global_batch=64)


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2] != 'Z'


class PipelineTests(unittest.TestCase):
    def test_formal_stages_and_downstream_stays_eight_gpus(self):
        cfg = recipe(Path('/tmp/unused'))
        zero = pipeline.stage_commands(cfg, 0, 'formal')
        one = pipeline.stage_commands(cfg, 1, 'formal')
        self.assertEqual([stage[0] for stage in zero], ['pretraining','posttraining_and_evaluation'])
        self.assertEqual(len(one), 1)
        self.assertEqual(zero[0][2]['GLOBAL_BATCH'], '64')
        self.assertEqual(zero[0][2]['PER_DEVICE_BATCH_SIZE'], '4')
        self.assertEqual(zero[0][2]['EXPECTED_COMMIT'], 'new_commit')
        downstream = zero[-1][2]
        self.assertNotIn('PER_DEVICE_BATCH_SIZE', downstream)
        self.assertNotIn('GLOBAL_BATCH', downstream)
        self.assertEqual(downstream['SMOKE_TEST'], '0')
        self.assertEqual(downstream['SOURCE_SMOKE_TEST'], '0')
        self.assertEqual(downstream['PRETRAIN_LAUNCHER_DIR'], '/tmp/unused/control/formal/pretrain')
        self.assertEqual(downstream['REQUIRE_PAIRED_SOURCE'], '0')

    def test_smoke_stages_only_use_isolated_short_runs(self):
        cfg = recipe(Path('/tmp/unused'))
        short = pipeline.stage_commands(cfg, 0, 'smoke')
        formal = pipeline.stage_commands(cfg, 0, 'formal')
        self.assertEqual(len(short), 2)
        for index in range(2):
            self.assertEqual(short[index][2]['SMOKE_TEST'], '1')
            self.assertTrue(short[index][2]['RUN_ID'].endswith('_smoke3'))
            self.assertNotEqual(short[index][2]['RUN_ID'], formal[index][2]['RUN_ID'])
        self.assertEqual(short[-1][2]['SOURCE_SMOKE_TEST'], '1')
        self.assertTrue(short[-1][2]['EVAL_OUTPUT_DIR'].endswith('_smoke3'))
        self.assertNotEqual(short[-1][2]['PAIR_PROTOCOL_DIR'], formal[-1][2]['PAIR_PROTOCOL_DIR'])

    def test_failed_pretrain_cannot_start_posttrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = recipe(Path(tmp))
            child = unittest.mock.Mock()
            child.wait.return_value = 7
            with patch.object(pipeline.subprocess, 'Popen', return_value=child) as spawn, \
                 patch.object(pipeline, 'terminate_owned_group') as cleanup:
                with self.assertRaisesRegex(RuntimeError, 'pretraining exited'):
                    pipeline.run(cfg, 0, 'smoke')
            self.assertEqual(spawn.call_count, 1)
            cleanup.assert_called_once_with(child)
            self.assertEqual((Path(cfg['control_dir'])/'smoke/exit_code').read_text().strip(), '1')

    def test_formal_cannot_start_without_full_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = recipe(Path(tmp))
            with patch.object(pipeline.subprocess, 'Popen') as spawn:
                with self.assertRaises(FileNotFoundError):
                    pipeline.run(cfg, 0, 'formal')
            spawn.assert_not_called()

    def test_missing_peer_success_cannot_pass_to_posttraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = recipe(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                pipeline.check_pretraining(cfg, 'smoke')

    def test_owned_group_is_drained_after_its_leader_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pidfile = root/'grandchild.pid'
            outsider = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
            childcode = ('import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                         'open(sys.argv[1],"w").write(str(os.getpid())); time.sleep(60)')
            leadercode = ('import os,subprocess,sys,time; '
                          'subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]]); '
                          'deadline=time.monotonic()+5; '
                          '\nwhile not os.path.exists(sys.argv[2]) and time.monotonic()<deadline: time.sleep(.01)\n'
                          'sys.exit(4)')
            leader = subprocess.Popen([sys.executable, '-c', leadercode, childcode, str(pidfile)], start_new_session=True)
            grandchild = None
            try:
                self.assertEqual(leader.wait(timeout=8), 4)
                grandchild = int(pidfile.read_text())
                self.assertTrue(alive(grandchild))
                pipeline.terminate_owned_group(leader, grace_seconds=.3)
                deadline = time.monotonic()+3
                while alive(grandchild) and time.monotonic()<deadline:
                    time.sleep(.05)
                self.assertFalse(alive(grandchild))
                self.assertTrue(alive(outsider.pid))
            finally:
                if grandchild and alive(grandchild):
                    os.kill(grandchild, signal.SIGKILL)
                outsider.terminate(); outsider.wait(timeout=5)


class SmokeSourceTests(unittest.TestCase):
    def source_fixture(self, root, steps=3):
        run, launcher = root/'pre', root/'launcher'
        (run/'final_model').mkdir(parents=True); launcher.mkdir()
        checkpoint = run/'final_model/pytorch_model.pt'; checkpoint.write_bytes(b'fixture')
        cfg = yaml.safe_load((REPO/'examples/pretrain/GalaxeaLiberoMix/train_files/starvla_mixed_ln_80k.yaml').read_text())
        cfg['trainer']['max_train_steps'] = steps
        (run/'config.full.yaml').write_text(yaml.safe_dump(cfg))
        (run/'pretrain_success.json').write_text(json.dumps(dict(arm='mixed', steps=steps,
                checkpoint=str(checkpoint), bytes=checkpoint.stat().st_size)))
        (run/'dataset_statistics.json').write_text(json.dumps({gate.GALAXEA:{}, gate.LIBERO:{}}))
        (run/'exit_code').write_text('0'); (launcher/'exit_code').write_text('0')
        (run/'metrics.jsonl').write_text(json.dumps(dict(step=steps, action_dit_loss=.5,
                **{'source_exposure/cumulative_examples': steps*64}))+'\n')
        return run, launcher

    def test_real_three_step_source_only_accepted_when_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, launcher = self.source_fixture(Path(tmp))
            self.assertEqual(gate.validate_source(run,'mixed',launcher,source_smoke=True)['bytes'], 7)
            with self.assertRaisesRegex(ValueError, '80000 updates'):
                gate.validate_source(run,'mixed',launcher)

    def test_wrong_sample_exposure_or_nonfinite_loss_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, launcher = self.source_fixture(Path(tmp))
            for count, loss in [(384,.5), (192,float('nan'))]:
                (run/'metrics.jsonl').write_text(json.dumps(dict(step=3,action_dit_loss=loss,
                     **{'source_exposure/cumulative_examples':count}))+'\n')
                with self.assertRaisesRegex(ValueError, 'finite loss and global64'):
                    gate.validate_source(run,'mixed',launcher,source_smoke=True)

    def test_marker_cannot_forge_short_source_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, launcher = self.source_fixture(Path(tmp))
            marker=json.loads((run/'pretrain_success.json').read_text());marker['steps']=80000
            (run/'pretrain_success.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,'3 updates'):
                gate.validate_source(run,'mixed',launcher,source_smoke=True)

    def test_formal_80000_source_still_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, launcher = self.source_fixture(Path(tmp), steps=80000)
            gate.validate_source(run,'mixed',launcher)

    def test_source_smoke_flag_rejected_without_post_smoke(self):
        args=argparse.Namespace(source_smoke=True,smoke=False)
        with self.assertRaisesRegex(ValueError,'requires --smoke'):
            gate.preflight(args)
        env=dict(os.environ,ARM='mixed',PRETRAIN_RUN_DIR='/tmp/unused',REQUIRE_PAIRED_SOURCE='0',
                 SMOKE_TEST='0',SOURCE_SMOKE_TEST='1',DRY_RUN='1')
        proc=subprocess.run(['bash',str(REPO/'examples/pretrain/GalaxeaLiberoMix/train_files/run_libero_pair_60k_eval.sh')],
                            env=env,capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('SOURCE_SMOKE_TEST requires SMOKE_TEST=1',proc.stderr)


class SpecTests(unittest.TestCase):
    def test_fresh_spec_uses_one_checkout_and_fresh_outputs(self):
        spec=make_spec.build_spec('/repo','commit','fresh',['node0','node1'],'rank0','/runs','/eval',
                                  '/python','/overlay','/libero','/reference')
        self.assertEqual(spec['pretrain_repo'],spec['posttrain_repo'])
        self.assertEqual(spec['pretrain_commit'],spec['posttrain_commit'])
        self.assertEqual(spec['pretrain_global_batch'],64)
        self.assertEqual(spec['posttrain_steps'],60000)
        self.assertNotIn('20261110',json.dumps(spec))
        for bad in ['../bad','bad/run','already_smoke3']:
            with self.assertRaises(ValueError):
                make_spec.build_spec('/repo','commit',bad,['node0','node1'],'rank0','/runs','/eval',
                                     '/python','/overlay','/libero','/reference')


if __name__ == '__main__':
    unittest.main(verbosity=2)
