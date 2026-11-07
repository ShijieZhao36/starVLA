#!/usr/bin/env python3
"""CPU integration test of 16-GPU scheduling, using real queue/verifier sources.

Policy servers and rollouts are deliberately replaced by tiny CPU fixtures;
this test verifies coordination/coverage/failure handling, not GPU inference.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import signal

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_libero_plus_16gpu.py"
PINNED = "/mnt/zhaoshijie/workspace/starVLA_galaxea_libero_posttrain60k_20261105"
PARALLEL = "examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval"
VERIFIER = "examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip())


def run():
    spec = importlib.util.spec_from_file_location("eval16", SCRIPT)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    retrieval = "from pathlib import Path\nimport json\nr=Path(" + repr(PINNED) + ")\nprint(json.dumps({n:(r/n).read_text() for n in " + repr([PARALLEL + "/task_queue.py", VERIFIER]) + "}))\n"
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "neo-bj5",
                             "/root/miniconda3/envs/MRA/bin/python -"],
                            input=retrieval, text=True, capture_output=True, check=True, timeout=45)
    sources = json.loads(result.stdout)
    cases = []
    with tempfile.TemporaryDirectory(prefix="libero_eval16_cpu_") as temp:
        root = Path(temp)
        # Real process groups: the owned leader exits before its stubborn child.
        # An unrelated group must remain alive throughout owned cleanup.
        descendant_pid_file = root / "descendant_pid"
        descendant_code = "import os,signal,time;from pathlib import Path;" \
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);" \
            f"Path({str(descendant_pid_file)!r}).write_text(str(os.getpid()));time.sleep(300)"
        leader_code = "import subprocess,sys,time;from pathlib import Path;" \
            f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}]);" \
            f"p=Path({str(descendant_pid_file)!r});" \
            "\nwhile not p.exists():time.sleep(.01)"
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(300)"],
                                     start_new_session=True)
        leader = subprocess.Popen([sys.executable, "-c", leader_code], start_new_session=True)
        try:
            assert leader.wait(timeout=5) == 0
            child_pid = int(descendant_pid_file.read_text())
            evaluator.stop_children([leader], grace_seconds=.3)
            assert unrelated.poll() is None
            for _ in range(100):
                state_file = Path(f"/proc/{child_pid}/stat")
                if not state_file.exists() or state_file.read_text().split()[2] == "Z":
                    break
                time.sleep(.01)
            else:
                raise AssertionError("Orphaned descendant survived owned group cleanup")
            assert leader.group_cleanup_complete is True
            evaluator.stop_children([leader], grace_seconds=.3)
            assert unrelated.poll() is None
            cases.append({"case": "cleanup_exited_leader_descendant_and_preserve_unrelated_group",
                          "passed": True})
        finally:
            for owned in [leader, unrelated]:
                try:
                    os.killpg(owned.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                owned.wait(timeout=5)
        repo = root / "repo"
        parallel = repo / PARALLEL
        for name, source in sources.items():
            write(repo / name, source)
        write(parallel / "policy_chunk_server.py", '''
            import argparse,json,os,socket,time
            print('FIXTURE_GPU_ENV '+json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID']}),flush=True)
            p=argparse.ArgumentParser();p.add_argument('--port',type=int);a,_=p.parse_known_args()
            s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind(('127.0.0.1',a.port));s.listen()
            while True:
                c,_=s.accept();c.close()
        ''')
        write(parallel / "eval_worker.py", '''
            import argparse,json,os
            from task_queue import TaskQueueClient
            class FakeEnv:
                def step(self,action):return None
                def close(self):return None
            def _get_libero_env(*a,**k):return FakeEnv(),'CPU fixture'
            def rollout_episode(env,**kwargs):
                env.step([0.0]*7);return False,[]
            def main():
                print('FIXTURE_GPU_ENV '+json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID']}),flush=True)
                p=argparse.ArgumentParser();p.add_argument('--queue-port',type=int)
                p.add_argument('--worker-id');a,_=p.parse_known_args()
                q=TaskQueueClient('127.0.0.1',a.queue_port)
                while True:
                    t=q.next_task()
                    if t is None:break
                    if os.environ.get('INJECT_ABORT')=='1':os.abort()
                    if os.environ.get('INJECT_ERROR')=='1' and a.worker_id=='node1_gpu08_0':
                        q.release();continue
                    env,_=_get_libero_env()
                    rollout_episode(env=env,max_steps=520,num_steps_wait=10)
                    env.close()
                    t.update(success=t['task_id']%2==0,category='CPU fixture',name=str(t['task_id']),worker=a.worker_id)
                    q.report(t)
                q.close()
            if __name__=='__main__':main()
        ''')
        write(parallel / "libero_rollout.py", "# CPU fixture: no rollout executed\n")
        write(parallel / "remote_policy.py", "# CPU fixture: no model loaded\n")
        write(parallel / "run_queue_eval.py", '''
            import os,socket,time
            def _child_env(gpu_id,for_sim):
                env=os.environ.copy()
                env['CUDA_VISIBLE_DEVICES']='' if for_sim else str(gpu_id)
                if for_sim:env['MUJOCO_EGL_DEVICE_ID']=str(gpu_id)
                else:env.pop('MUJOCO_EGL_DEVICE_ID',None)
                return env
            def _wait_for_port(port,proc,timeout,log):
                deadline=time.monotonic()+timeout
                while time.monotonic()<deadline:
                    if proc.poll() is not None:raise RuntimeError('fixture exited')
                    try:
                        with socket.create_connection(('127.0.0.1',port),timeout=.1):return
                    except OSError:time.sleep(.02)
                raise TimeoutError('fixture failed to listen')
        ''')
        write(repo / "libero/__init__.py", "")
        write(repo / "libero/libero/__init__.py", "")
        write(repo / "libero/libero/benchmark.py", '''
            from types import SimpleNamespace
            def get_benchmark_dict():
                counts={'libero_spatial':2402,'libero_object':2518,'libero_goal':2591,'libero_10':2519}
                return {s:(lambda n=n:SimpleNamespace(n_tasks=n)) for s,n in counts.items()}
        ''')
        wrapper = root / "node.py"
        write(wrapper, '''
            import os,runpy,socket,sys
            socket.gethostname=lambda:'fixture-host-'+os.environ['FIXTURE_RANK']
            script=sys.argv.pop(1)
            runpy.run_path(script,run_name='__main__')
        ''')
        ckpt = root / "train/final_model/pytorch_model.pt"
        write(ckpt, "CPU fixture checkpoint")
        env = dict(os.environ, PYTHONPATH=str(repo), LIBERO_HOME=str(repo), PYTHONDONTWRITEBYTECODE="1")
        future_ckpt = root / "future/final_model/pytorch_model.pt"
        future_output = root / "future_eval"
        future_command = [sys.executable, str(SCRIPT), "--repo", str(repo),
                          "--pretrained-path", str(future_ckpt), "--output-dir", str(future_output),
                          "--node-rank", "0"]
        dry = subprocess.run(future_command + ["--dry-run"], env=env, text=True,
                             capture_output=True, check=True, timeout=15)
        future_plan = json.loads(dry.stdout)
        assert future_plan["contract"]["checkpoint_bytes"] is None
        assert future_plan["contract"]["checkpoint_mtime_ns"] is None
        assert future_plan["contract"]["checkpoint_check"] == "deferred until real launch"
        assert not future_output.exists()
        cases.append({"case": "dry_run_allows_missing_future_checkpoint_without_output", "passed": True})
        missing = subprocess.run(future_command, env=env, text=True, capture_output=True, timeout=15)
        assert missing.returncode != 0 and "Final checkpoint is missing" in missing.stderr
        assert not future_output.exists()
        cases.append({"case": "real_launch_rejects_missing_checkpoint", "passed": True})
        write(future_ckpt, "")
        empty = subprocess.run(future_command + ["--dry-run"], env=env, text=True,
                               capture_output=True, timeout=15)
        assert empty.returncode != 0 and "not a non-empty file" in empty.stderr
        assert not future_output.exists()
        cases.append({"case": "dry_run_rejects_existing_empty_checkpoint", "passed": True})
        for name, smoke, inject in [("smoke16", True, False), ("full10030", False, False),
                                    ("episode_error", True, True)]:
            out = root / name
            procs, logs = [], []
            for rank in range(2):
                command = [sys.executable, str(wrapper), str(SCRIPT), "--repo", str(repo),
                           "--pretrained-path", str(ckpt), "--output-dir", str(out),
                           "--node-rank", str(rank), "--base-port", str(15300 + rank * 32),
                           "--server-timeout-sec", "15", "--peer-timeout-sec", "30",
                           "--evaluation-timeout-sec", "30"]
                if smoke:
                    command.append("--smoke")
                log = (root / f"{name}_node{rank}.log").open("w")
                logs.append(log)
                procs.append(subprocess.Popen(command, env=dict(env, FIXTURE_RANK=str(rank),
                                                                INJECT_ERROR="1" if inject else "0"),
                                              stdout=log, stderr=subprocess.STDOUT))
            try:
                exits = [proc.wait(timeout=60) for proc in procs]
            finally:
                for proc in procs:
                    if proc.poll() is None: proc.kill()
                for log in logs: log.close()
            if inject:
                assert all(code != 0 for code in exits), exits
                assert not (out / "eval_success.json").exists()
                assert (out / "node0/failure.json").exists()
                assert (out / "node1/failure.json").exists()
                cases.append({"case": name, "passed": True, "exit_codes": exits})
            else:
                if exits != [0, 0]:
                    for rank in range(2): print((root / f"{name}_node{rank}.log").read_text())
                    raise AssertionError(exits)
                report = json.loads((out / "eval_success.json").read_text())
                assert report["episode_count"] == (16 if smoke else 10030)
                assert report["validated"] and report["execution_errors"] == 0
                for rank in range(2):
                    for gpu in range(rank * 8, (rank + 1) * 8):
                        records = evaluator.records_from(out / f"node{rank}/gpu{gpu:02d}/results.jsonl")
                        assert records and all(r["worker"].startswith(f"node{rank}_gpu{gpu:02d}_") for r in records)
                cases.append({"case": name, "passed": True, "exit_codes": exits,
                              "episodes": report["episode_count"], "suite_counts": report["suite_episode_counts"]})
        # A single-task helper uses the same policy/worker/queue paths, and the
        # observer leaves original actions and NumPy RNG state unchanged.
        observer = evaluator.load_module("observer_test", HERE / "diagnostic_worker.py")
        import numpy as np
        from types import SimpleNamespace
        action = [0.25, -0.5, 0.0]
        seen = []
        class ObservedEnv:
            def step(self, actual):
                assert actual is action
                seen.append(actual)
                return ("unchanged",)
            def close(self):
                return "closed"
        observed = ObservedEnv()
        module = SimpleNamespace(_get_libero_env=lambda:(observed,"test"),
                                 rollout_episode=lambda **kwargs:(False, []))
        observer.attach_observers(module)
        state = np.random.get_state()
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            observed_env,_ = module._get_libero_env()
            assert observed_env.step(action) == ("unchanged",)
            assert observed_env.close() == "closed"
            assert module.rollout_episode(max_steps=520,num_steps_wait=10) == (False, [])
        after = np.random.get_state()
        assert state[0] == after[0] and np.array_equal(state[1],after[1]) and state[2:] == after[2:]
        assert seen == [action] and '"action_finite": true' in captured.getvalue()
        cases.append({"case":"diagnostic_observer_preserves_action_identity_return_values_and_rng","passed":True})
        for abort, renderer_gpu in [(False, None), (True, None), (False, 5)]:
            label = "single_task_abort" if abort else ("single_task_ok" if renderer_gpu is None else "single_task_split_gpu")
            diag_out = root / label
            command = [sys.executable, str(HERE / "diagnose_one_task.py"), "--repo", str(repo),
                       "--pretrained-path", str(ckpt), "--output-dir", str(diag_out),
                       "--suite", "libero_10", "--task-id", "3", "--local-gpu", "4",
                       "--base-port", "15400", "--server-timeout-sec", "15", "--timeout-sec", "30"]
            if renderer_gpu is not None:
                command += ["--renderer-gpu", str(renderer_gpu)]
            result = subprocess.run(command, env=dict(env,INJECT_ABORT="1" if abort else "0"),
                                    capture_output=True, text=True, timeout=45)
            if abort:
                failure = json.loads((diag_out / "failure.json").read_text())
                assert result.returncode != 0 and "SIGABRT" in failure["error"]
                assert "worker_id=diagnostic_gpu4_libero_10_3" in failure["error"]
                assert "pid=" in failure["error"] and "log=" in failure["error"]
                assert "Fatal Python error: Aborted" in (diag_out / "worker.stdout").read_text()
                assert not (diag_out / "diagnostic_success.json").exists()
                cases.append({"case":"diagnostic_captures_native_abort_and_named_worker_evidence","passed":True})
            else:
                if result.returncode != 0:print(result.stdout,result.stderr)
                assert result.returncode == 0
                report = json.loads((diag_out / "diagnostic_success.json").read_text())
                assert report["episode_count"] == 1 and report["diagnostic_only"] is True
                expected_renderer = 4 if renderer_gpu is None else renderer_gpu
                assert report["local_gpu"] == report["policy_gpu"] == 4
                assert report["renderer_gpu"] == expected_renderer
                def child_gpu_env(path):
                    return json.loads(next(line[len("FIXTURE_GPU_ENV "):]
                                           for line in path.read_text().splitlines()
                                           if line.startswith("FIXTURE_GPU_ENV ")))
                assert child_gpu_env(diag_out / "policy_server.log") == {
                    "CUDA_VISIBLE_DEVICES": "4", "MUJOCO_EGL_DEVICE_ID": None}
                assert child_gpu_env(diag_out / "worker.stdout") == {
                    "CUDA_VISIBLE_DEVICES": "", "MUJOCO_EGL_DEVICE_ID": str(expected_renderer)}
                assert '"env_step": 1' in (diag_out / "worker.stdout").read_text()
                assert '"event": "env_close_end"' in (diag_out / "worker.stdout").read_text()
                cases.append({"case":("single_task_diagnostic_original_queue_worker_integration"
                                      if renderer_gpu is None else "diagnostic_renderer_override_preserves_policy_gpu"),
                              "passed":True})
        # Validation failures are checked against the original verifier.
        sys.path.insert(0, str(parallel))
        import task_queue
        verifier = evaluator.load_module("cpu_original_verifier", repo / VERIFIER)
        good = evaluator.records_from(root / "full10030/results.jsonl")
        expected = {key for shard in evaluator.split_tasks(evaluator.EXPECTED_COUNTS) for key in shard}
        for label, records in [("missing", good[:-1]), ("duplicate", good + [good[0]]),
                               ("execution_error", [dict(good[0], error="fixture")] + good[1:]),
                               ("repeated_attempt", [dict(good[0], attempt=2)] + good[1:])]:
            target = root / label
            target.mkdir()
            source = target / "input.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in records))
            try:
                evaluator.merge_validate([source], target, expected, task_queue, verifier)
            except ValueError:
                cases.append({"case": "reject_" + label, "passed": True})
            else:
                raise AssertionError(f"Did not reject {label}")
    report = {"validated": True, "test_type": "CPU scheduler integration; no GPU inference", "cases": cases}
    (HERE / "cpu_integration_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
