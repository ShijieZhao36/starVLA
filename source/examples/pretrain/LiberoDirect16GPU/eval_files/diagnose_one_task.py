#!/usr/bin/env python3
"""One isolated LIBERO Plus task using the original policy, worker and queue.

This is diagnostic evidence only, never a replacement for the 10030-task result.
One attempt, unchanged environment/policy settings, fresh output required.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_libero_plus_16gpu import (
    SUITES, atomic_json, load_module, merge_validate, process_failure,
    sha256, stop_children, utc_now,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--pretrained-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--task-id", required=True, type=int)
    p.add_argument("--local-gpu", required=True, type=int, choices=range(8))
    p.add_argument("--renderer-gpu", type=int, choices=range(8),
                   help="Diagnostic EGL device; defaults to the policy's --local-gpu")
    p.add_argument("--base-port", type=int, default=11483)
    p.add_argument("--server-timeout-sec", type=int, default=600)
    p.add_argument("--timeout-sec", type=int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    renderer_gpu = args.local_gpu if args.renderer_gpu is None else args.renderer_gpu
    repo, ckpt, out = args.repo.resolve(), args.pretrained_path.resolve(), args.output_dir.resolve()
    if ckpt.name != "pytorch_model.pt" or ckpt.parent.name != "final_model":
        raise ValueError("Only a final_model/pytorch_model.pt checkpoint is accepted")
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        raise ValueError("Diagnostic checkpoint must exist and be non-empty")
    if not 1024 < args.base_port < 65535 or min(args.timeout_sec, args.server_timeout_sec) <= 0:
        raise ValueError("Invalid diagnostic port or timeout")
    if not os.environ.get("LIBERO_HOME"):
        raise ValueError("LIBERO_HOME is required")
    parallel = repo / "examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval"
    sys.path.insert(0, str(parallel))
    import run_queue_eval
    import task_queue
    from libero.libero import benchmark
    with contextlib.redirect_stdout(io.StringIO()):
        count = int(benchmark.get_benchmark_dict()[args.suite]().n_tasks)
    if not 0 <= args.task_id < count:
        raise ValueError(f"Task id outside suite range [0,{count})")
    verifier_path = repo / "examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py"
    verifier = load_module("diagnostic_result_verifier", verifier_path)
    observer = Path(__file__).with_name("diagnostic_worker.py")
    plan = {"diagnostic_only": True, "checkpoint": str(ckpt), "checkpoint_bytes": ckpt.stat().st_size,
            "suite": args.suite, "task_id": args.task_id, "episode_idx": 0,
            "local_gpu": args.local_gpu, "policy_gpu": args.local_gpu,
            "renderer_gpu": renderer_gpu, "environment_seed": 7, "num_steps_wait": 10,
            "canonical_forward": True, "bf16": True, "gripper_open_sign": "positive",
            "policy_noise": "original_unseeded", "max_episode_attempts": 1,
            "observer_sha256": sha256(observer), "time": utc_now()}
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "diagnostic_plan.json", plan)
    procs, handles = [], []
    queue = None

    def interrupted(signum, _frame):
        raise RuntimeError(f"Diagnostic interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        key = (args.suite, args.task_id, 0)
        queue = task_queue.TaskQueueServer([key], out / "results.jsonl", max_attempts=1)
        queue.start("127.0.0.1", args.base_port + 1)
        log = out / "policy_server.log"
        handle = log.open("w")
        handles.append(handle)
        command = [sys.executable, "-X", "faulthandler", "-u", str(parallel / "policy_chunk_server.py"),
                   "--pretrained-path", str(ckpt), "--port", str(args.base_port),
                   "--use-canonical-forward", "true", "--gripper-open-sign", "positive",
                   "--idle-timeout", "-1", "--use-bf16"]
        server = subprocess.Popen(command, cwd=repo, env=run_queue_eval._child_env(args.local_gpu, for_sim=False),
                                  stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        server.worker_id, server.log_path = "diagnostic_policy", str(log)
        procs.append(server)
        run_queue_eval._wait_for_port(args.base_port, server, args.server_timeout_sec, log)
        log = out / "worker.stdout"
        handle = log.open("w")
        handles.append(handle)
        worker_id = f"diagnostic_gpu{args.local_gpu}_{args.suite}_{args.task_id}"
        command = [sys.executable, "-X", "faulthandler", "-u", str(observer), "--repo", str(repo),
                   "--host", "127.0.0.1", "--port", str(args.base_port),
                   "--queue-host", "127.0.0.1", "--queue-port", str(args.base_port + 1),
                   "--output-dir", str(out), "--worker-id", worker_id,
                   "--seed", "7", "--num-steps-wait", "10"]
        worker = subprocess.Popen(command, cwd=repo, env=run_queue_eval._child_env(renderer_gpu, for_sim=True),
                                  stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        worker.worker_id, worker.log_path = worker_id, str(log)
        procs.append(worker)
        atomic_json(out / "processes.json", {"policy_pid": server.pid, "worker_pid": worker.pid,
                                             "worker_id": worker_id, "worker_command": command})
        deadline = time.monotonic() + args.timeout_sec
        while True:
            if server.poll() is not None:
                raise RuntimeError(process_failure(server, "Diagnostic policy exited"))
            if worker.poll() not in (None, 0):
                raise RuntimeError(process_failure(worker, "Diagnostic worker exited"))
            done, total, pending, inflight = queue.progress()
            if done == total and pending == 0 and inflight == 0:
                break
            if worker.poll() == 0:
                current = queue.progress()
                if current[0] < current[1]:
                    raise RuntimeError("Diagnostic worker exited without a result")
            if time.monotonic() >= deadline:
                raise TimeoutError("Single-task diagnostic exceeded its bounded runtime")
            time.sleep(1)
        if worker.wait(timeout=600) != 0:
            raise RuntimeError(process_failure(worker, "Diagnostic worker exited"))
        stop_children(procs)
        queue.stop()
        queue = None
        report = merge_validate([out / "results.jsonl"], out, {key}, task_queue, verifier)
        report.update(plan)
        report["completed_at"] = utc_now()
        atomic_json(out / "diagnostic_success.json", report)
        (out / "exit_code").write_text("0\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        atomic_json(out / "failure.json", {"diagnostic_only": True, "error": repr(exc), "time": utc_now()})
        (out / "exit_code").write_text("1\n")
        raise
    finally:
        stop_children(procs)
        if queue is not None:
            queue.stop()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
