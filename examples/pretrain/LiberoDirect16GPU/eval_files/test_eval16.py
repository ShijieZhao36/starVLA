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
            import argparse,socket,time
            p=argparse.ArgumentParser();p.add_argument('--port',type=int);a,_=p.parse_known_args()
            s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind(('127.0.0.1',a.port));s.listen()
            while True:
                c,_=s.accept();c.close()
        ''')
        write(parallel / "eval_worker.py", '''
            import argparse,os
            from task_queue import TaskQueueClient
            p=argparse.ArgumentParser();p.add_argument('--queue-port',type=int)
            p.add_argument('--worker-id');a,_=p.parse_known_args()
            q=TaskQueueClient('127.0.0.1',a.queue_port)
            while True:
                t=q.next_task()
                if t is None:break
                if os.environ.get('INJECT_ERROR')=='1' and a.worker_id=='node1_gpu08_0':
                    q.release();continue
                t.update(success=t['task_id']%2==0,category='CPU fixture',name=str(t['task_id']),worker=a.worker_id)
                q.report(t)
            q.close()
        ''')
        write(parallel / "libero_rollout.py", "# CPU fixture: no rollout executed\n")
        write(parallel / "remote_policy.py", "# CPU fixture: no model loaded\n")
        write(parallel / "run_queue_eval.py", '''
            import os,socket,time
            def _child_env(gpu_id,for_sim):return os.environ.copy()
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
