#!/usr/bin/env python3
"""Evaluate one final checkpoint on two hosts, eight GPUs on each host.

Invoke this script once per host, with the same shared --output-dir and distinct
--node-rank 0/1. It reuses the pinned policy, rollout, queue and result verifier;
only task assignment and two-host coordination are new. Existing output/claims
are never resumed or overwritten. A failed episode is not automatically retried.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
EXPECTED_COUNTS = {
    "libero_spatial": 2402, "libero_object": 2518,
    "libero_goal": 2591, "libero_10": 2519,
}
NODES, GPUS_PER_NODE = 2, 8
WORLD_GPUS = NODES * GPUS_PER_NODE


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_tasks(counts, smoke=False):
    """Balance every suite over GPUs, preserving its original task identifiers."""
    shards = [[] for _ in range(WORLD_GPUS)]
    if smoke:
        # One distinct original task per GPU; exercise all four suites.
        for gpu in range(WORLD_GPUS):
            shards[gpu].append((SUITES[gpu % len(SUITES)], gpu // len(SUITES), 0))
    else:
        for task_id in range(max(counts.values())):
            for suite in SUITES:
                if task_id < counts[suite]:
                    shards[task_id % WORLD_GPUS].append((suite, task_id, 0))
    flat = [key for shard in shards for key in shard]
    if len(flat) != len(set(flat)) or any(not shard for shard in shards):
        raise ValueError("Task splitting produced duplicates or an empty GPU shard")
    expected = ({(s, i, 0) for s in SUITES for i in range(counts[s])}
                if not smoke else set(flat))
    if set(flat) != expected or len(flat) != (16 if smoke else 10030):
        raise ValueError("Task splitting does not cover the intended episode set")
    return shards


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_peer_failure(out):
    for rank in range(NODES):
        path = out / f"node{rank}" / "failure.json"
        if path.exists():
            raise RuntimeError(f"Node {rank} failed: {path.read_text()}")


def wait_for(path, out, timeout):
    deadline = time.monotonic() + timeout
    while not path.exists():
        check_peer_failure(out)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(5)
    check_peer_failure(out)
    return json.loads(path.read_text())


def stop_children(procs, grace_seconds=20):
    """Stop owned groups even when their Popen leaders have already exited.

    Every process passed here was created with start_new_session=True, so its
    original PID is its owned PGID. A dead leader does not imply its descendants
    exited. Completed cleanups are marked to avoid signaling an old PGID again
    after a potentially long peer wait.
    """
    pending = []
    for proc in procs:
        if getattr(proc, "group_cleanup_complete", False):
            continue
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
            os.killpg(proc.pid, signal.SIGTERM)
            pending.append(proc)
        except ProcessLookupError:
            proc.group_cleanup_complete = True
    deadline = time.monotonic() + grace_seconds
    while pending and time.monotonic() < deadline:
        remaining = []
        for proc in pending:
            proc.poll()
            try:
                os.killpg(proc.pid, 0)
                remaining.append(proc)
            except ProcessLookupError:
                proc.group_cleanup_complete = True
        pending = remaining
        if pending:
            time.sleep(.1)
    for proc in pending:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
        proc.group_cleanup_complete = True


def records_from(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def process_failure(proc, kind):
    code = proc.returncode
    signal_name = ""
    if code is not None and code < 0:
        try:
            signal_name = f" signal={signal.Signals(-code).name}"
        except ValueError:
            signal_name = f" signal={-code}"
    return (f"{kind} worker_id={getattr(proc, 'worker_id', 'unknown')} "
            f"pid={proc.pid} exit_code={code}{signal_name} "
            f"log={getattr(proc, 'log_path', 'unknown')}")


def merge_validate(paths, destination, expected, queue_module, verifier):
    records = [record for path in paths for record in records_from(path)]
    # No retries: multiple attempts could alter stochastic policy outcomes.
    if any(record.get("attempt") != 1 for record in records):
        raise ValueError("An episode has a missing or non-first attempt")
    encoded = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    result_path = destination / "results.jsonl"
    result_path.write_text(encoded)
    summary = queue_module.summarize_results(result_path)
    report = verifier.validate_records(records, summary, expected)
    atomic_json(destination / "summary.json", summary)
    (destination / "summary.txt").write_text(queue_module.format_summary(summary) + "\n")
    atomic_json(destination / "completion_validation.json", report)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--pretrained-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Fresh shared directory, identical on both hosts")
    p.add_argument("--node-rank", required=True, type=int, choices=(0, 1))
    p.add_argument("--base-port", type=int, default=11283,
                   help="Local policy ports base..base+7; queue ports base+8..base+15")
    p.add_argument("--server-timeout-sec", type=int, default=600)
    p.add_argument("--peer-timeout-sec", type=int, default=43200)
    p.add_argument("--evaluation-timeout-sec", type=int, default=43200)
    p.add_argument("--smoke", action="store_true",
                   help="16 episodes, one per GPU, one simulator per GPU")
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-only task/protocol validation; creates no output")
    return p.parse_args()


def main():
    args = parse_args()
    repo, ckpt, out = args.repo.resolve(), args.pretrained_path.resolve(), args.output_dir.resolve()
    if not 1024 < args.base_port <= 65520:
        raise ValueError("Invalid local policy/queue port range")
    if min(args.server_timeout_sec, args.peer_timeout_sec, args.evaluation_timeout_sec) <= 0:
        raise ValueError("Timeouts must be positive")
    if ckpt.name != "pytorch_model.pt" or ckpt.parent.name != "final_model":
        raise ValueError("Only final_model/pytorch_model.pt is accepted")
    checkpoint_stat = None
    if ckpt.exists():
        if not ckpt.is_file() or ckpt.stat().st_size <= 0:
            raise ValueError("Final checkpoint is not a non-empty file")
        checkpoint_stat = ckpt.stat()
    elif not args.dry_run:
        raise ValueError("Final checkpoint is missing")
    if not os.environ.get("LIBERO_HOME"):
        raise ValueError("LIBERO_HOME must point to the matched LIBERO Plus checkout")
    parallel = repo / "examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval"
    verifier_path = repo / "examples/pretrain/LiberoPlus/eval_files/verify_libero_plus_results.py"
    sys.path.insert(0, str(parallel))
    import task_queue
    import run_queue_eval
    from libero.libero import benchmark
    verifier = load_module("final_result_verifier", verifier_path)
    with contextlib.redirect_stdout(io.StringIO()):
        benchmarks = benchmark.get_benchmark_dict()
        counts = {name: int(benchmarks[name]().n_tasks) for name in SUITES}
        full_expected = verifier.expected_episode_keys(1)
    if counts != EXPECTED_COUNTS or len(full_expected) != 10030:
        raise ValueError(f"Unexpected benchmark suite counts: {counts}")
    shards = split_tasks(counts, args.smoke)
    expected = {key for shard in shards for key in shard}
    if not args.smoke and expected != full_expected:
        raise ValueError("Tasks differ from the existing final-result verifier")
    component_paths = [parallel / name for name in (
        "policy_chunk_server.py", "eval_worker.py", "libero_rollout.py",
        "remote_policy.py", "task_queue.py", "run_queue_eval.py")]
    component_paths += [verifier_path, Path(__file__).resolve()]
    contract = {
        "checkpoint": str(ckpt),
        "checkpoint_bytes": checkpoint_stat.st_size if checkpoint_stat else None,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns if checkpoint_stat else None,
        "checkpoint_check": "non-empty final checkpoint verified" if checkpoint_stat
                            else "deferred until real launch",
        "suite_counts": counts, "node_count": NODES, "gpu_count": WORLD_GPUS,
        "sims_per_gpu": 1 if args.smoke else 4, "num_trials_per_task": 1,
        "environment_seed": 7, "num_steps_wait": 10, "use_bf16": True,
        "use_canonical_forward": True, "gripper_open_sign": "positive",
        "policy_noise": "original_unseeded", "max_episode_attempts": 1,
        "smoke": args.smoke,
        "task_assignment_sha256": hashlib.sha256(json.dumps(shards).encode()).hexdigest(),
        "code_sha256": {str(path.relative_to(repo)) if path.is_relative_to(repo)
                        else path.name: sha256(path) for path in component_paths},
    }
    plan = {"node_rank": args.node_rank, "contract": contract,
            "global_gpu_episode_counts": [len(shard) for shard in shards],
            "local_global_gpus": list(range(args.node_rank * 8, (args.node_rank + 1) * 8))}
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    # Atomic claim prevents duplicate submissions, including failed/partial runs.
    node_dir = out / f"node{args.node_rank}"
    node_dir.mkdir()
    procs, queues, handles = [], [], []

    def interrupted(signum, _frame):
        raise RuntimeError(f"Evaluation interrupted by signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        atomic_json(node_dir / "plan.json", plan)
        atomic_json(node_dir / "ready.json", {
            "host": socket.gethostname(), "node_rank": args.node_rank,
            "time": utc_now(), "contract": contract,
        })
        peer = wait_for(out / f"node{1 - args.node_rank}" / "ready.json",
                        out, args.peer_timeout_sec)
        if peer["contract"] != contract or peer["host"] == socket.gethostname():
            raise ValueError("Peer protocol/checkpoint differs or both ranks are on one host")
        for local_gpu in range(GPUS_PER_NODE):
            global_gpu = args.node_rank * GPUS_PER_NODE + local_gpu
            gpu_dir = node_dir / f"gpu{global_gpu:02d}"
            gpu_dir.mkdir()
            queue_port = args.base_port + GPUS_PER_NODE + local_gpu
            queue = task_queue.TaskQueueServer(shards[global_gpu], gpu_dir / "results.jsonl", max_attempts=1)
            queue.start("127.0.0.1", queue_port)
            queues.append(queue)
        for local_gpu, queue in enumerate(queues):
            check_peer_failure(out)
            if any(proc.poll() is not None for proc in procs if getattr(proc, "is_policy", False)):
                raise RuntimeError("A policy server exited during startup")
            global_gpu = args.node_rank * GPUS_PER_NODE + local_gpu
            gpu_dir = node_dir / f"gpu{global_gpu:02d}"
            port = args.base_port + local_gpu
            log = gpu_dir / "policy_server.log"
            handle = log.open("w")
            handles.append(handle)
            command = [sys.executable, "-X", "faulthandler", "-u", str(parallel / "policy_chunk_server.py"),
                       "--pretrained-path", str(ckpt), "--port", str(port),
                       "--use-canonical-forward", "true", "--gripper-open-sign", "positive",
                       "--idle-timeout", "-1", "--use-bf16"]
            proc = subprocess.Popen(command, cwd=repo,
                                    env=run_queue_eval._child_env(local_gpu, for_sim=False),
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            proc.is_policy = True
            proc.worker_id = f"node{args.node_rank}_gpu{global_gpu:02d}_policy"
            proc.log_path = str(log)
            procs.append(proc)
            run_queue_eval._wait_for_port(port, proc, args.server_timeout_sec, log)
            for sim in range(contract["sims_per_gpu"]):
                worker_id = f"node{args.node_rank}_gpu{global_gpu:02d}_{sim}"
                worker_log = gpu_dir / f"{worker_id}.stdout"
                handle = worker_log.open("w")
                handles.append(handle)
                command = [sys.executable, "-X", "faulthandler", "-u", str(parallel / "eval_worker.py"),
                           "--host", "127.0.0.1", "--port", str(port),
                           "--queue-host", "127.0.0.1", "--queue-port",
                           str(args.base_port + GPUS_PER_NODE + local_gpu),
                           "--output-dir", str(gpu_dir), "--worker-id", worker_id,
                           "--seed", "7", "--num-steps-wait", "10"]
                proc = subprocess.Popen(command, cwd=repo,
                                        env=run_queue_eval._child_env(local_gpu, for_sim=True),
                                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                proc.is_policy = False
                proc.local_gpu = local_gpu
                proc.worker_id = worker_id
                proc.log_path = str(worker_log)
                procs.append(proc)
            print(f"Global GPU {global_gpu}: {len(shards[global_gpu])} episodes, "
                  f"{contract['sims_per_gpu']} simulators started", flush=True)
        deadline = time.monotonic() + args.evaluation_timeout_sec
        last = None
        while True:
            check_peer_failure(out)
            progress = [queue.progress() for queue in queues]
            completed = sum(item[0] for item in progress)
            total = sum(item[1] for item in progress)
            if completed != last:
                print(f"Node {args.node_rank}: {completed}/{total} episodes", flush=True)
                last = completed
            if all(done == count and pending == 0 and inflight == 0
                   for done, count, pending, inflight in progress):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Node evaluation exceeded its bounded runtime")
            for proc in procs:
                if getattr(proc, "is_policy", False) and proc.poll() is not None:
                    raise RuntimeError(process_failure(proc, "Policy server exited"))
                if not getattr(proc, "is_policy", False) and proc.poll() not in (None, 0):
                    raise RuntimeError(process_failure(proc, "Simulation worker exited"))
            for gpu, state in enumerate(progress):
                workers = [p for p in procs if getattr(p, "local_gpu", None) == gpu]
                if state[0] < state[1] and all(p.poll() is not None for p in workers):
                    # Workers may have reported the final result and exited
                    # after the progress snapshot above. Recheck under the
                    # queue's lock before classifying a normal exit as failure.
                    current = queues[gpu].progress()
                    if current[0] < current[1]:
                        raise RuntimeError(f"All workers on GPU {gpu} exited before completion")
            time.sleep(5)
        for proc in procs:
            if not getattr(proc, "is_policy", False):
                if proc.wait(timeout=600) != 0:
                    raise RuntimeError(process_failure(proc, "Simulation worker exited"))
        # Stop this node's inference servers before waiting for the peer.
        stop_children(procs)
        for queue in queues:
            queue.stop()
        queues.clear()
        paths = [node_dir / f"gpu{gpu:02d}" / "results.jsonl"
                 for gpu in plan["local_global_gpus"]]
        local_expected = {key for gpu in plan["local_global_gpus"] for key in shards[gpu]}
        report = merge_validate(paths, node_dir, local_expected, task_queue, verifier)
        report.update(node_rank=args.node_rank, time=utc_now(), contract=contract)
        atomic_json(node_dir / "completion.json", report)
        if args.node_rank == 0:
            peer_done = wait_for(out / "node1/completion.json", out, args.peer_timeout_sec)
            if peer_done["contract"] != contract or not peer_done["validated"]:
                raise ValueError("Peer completion protocol mismatch")
            report = merge_validate([out / "node0/results.jsonl", out / "node1/results.jsonl"],
                                    out, expected, task_queue, verifier)
            if not args.smoke and report["suite_episode_counts"] != EXPECTED_COUNTS:
                raise ValueError("Final suite counts do not match all 10030 tasks")
            report.update(checkpoint=str(ckpt), output_dir=str(out), time=utc_now(),
                          smoke=args.smoke, contract=contract)
            atomic_json(out / "eval_success.json", report)
            (out / "exit_code").write_text("0\n")
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        else:
            wait_for(out / "eval_success.json", out, args.peer_timeout_sec)
        (node_dir / "exit_code").write_text("0\n")
        return 0
    except BaseException as exc:
        atomic_json(node_dir / "failure.json", {
            "node_rank": args.node_rank, "host": socket.gethostname(),
            "time": utc_now(), "error": repr(exc),
        })
        (node_dir / "exit_code").write_text("1\n")
        if args.node_rank == 0:
            (out / "exit_code").write_text("1\n")
        raise
    finally:
        stop_children(procs)
        for queue in queues:
            queue.stop()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
