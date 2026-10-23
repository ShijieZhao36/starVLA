"""Launch one policy server per GPU and a pool of sim workers fed by a task queue."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from libero.libero import benchmark

from libero_rollout import EVAL_SUITES
from task_queue import TaskQueueServer, format_summary, interleave_tasks, summarize_results

REPO_ROOT = Path(__file__).resolve().parents[5]
PARALLEL_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = PARALLEL_DIR / "policy_chunk_server.py"
WORKER_SCRIPT = PARALLEL_DIR / "eval_worker.py"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-canonical-forward", choices=("true", "false"), default="false")
    parser.add_argument("--gripper-open-sign", choices=("positive", "negative"), default="positive")
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16")
    parser.add_argument("--sims-per-gpu", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--base-port", type=int, default=9883)
    parser.add_argument("--queue-port", type=int, default=0)
    parser.add_argument("--launch-stagger-sec", type=float, default=90.0)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--server-timeout-sec", type=int, default=1800)
    return parser.parse_args()


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float, log_path: Path) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited with {proc.returncode} before port {port} opened. log: {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"timed out waiting for port {port}. log: {log_path}")


def _stop_processes(procs) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 15
    for proc in procs:
        if proc.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", encoding="utf-8")


def _child_env(gpu_id: int, for_sim: bool) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pythonpath = env.get("PYTHONPATH", "")
    root = str(REPO_ROOT)
    if root not in pythonpath.split(":"):
        env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    if for_sim:
        # EGL follows the physical GPU id, not CUDA_VISIBLE_DEVICES. Hide CUDA from
        # sim workers so importing LIBERO (which imports torch) cannot allocate a
        # context next to the policy server.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
    else:
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    return env


def main() -> None:
    args = _parse_args()
    if args.num_gpus < 1 or args.sims_per_gpu < 1 or args.num_trials_per_task < 1:
        raise SystemExit("num-gpus, sims-per-gpu, and num-trials-per-task must be >= 1")
    if not os.environ.get("LIBERO_HOME"):
        raise SystemExit("LIBERO_HOME is required.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_port = args.queue_port or (args.base_port + args.num_gpus)
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_counts = {name: int(benchmark_dict[name]().n_tasks) for name in EVAL_SUITES}
    tasks = interleave_tasks(suite_counts, EVAL_SUITES, args.num_trials_per_task)
    print(
        f"Queued {len(tasks)} episodes across {suite_counts} "
        f"on {args.num_gpus} GPUs x {args.sims_per_gpu} sims.",
        flush=True,
    )

    results_path = output_dir / "results.jsonl"
    queue = TaskQueueServer(tasks, results_path)
    queue.start("127.0.0.1", queue_port)

    servers = []
    workers = []
    log_handles = []
    try:
        for gpu_id in range(args.num_gpus):
            if any(proc.poll() is not None for proc in servers):
                raise RuntimeError("a policy server exited during startup; see logs/server_gpu*.log")
            started = time.time()
            port = args.base_port + gpu_id
            server_log = output_dir / "logs" / f"server_gpu{gpu_id}.log"
            server_cmd = [
                sys.executable,
                "-u",
                str(SERVER_SCRIPT),
                "--pretrained-path",
                args.pretrained_path,
                "--port",
                str(port),
                "--use-canonical-forward",
                args.use_canonical_forward,
                "--gripper-open-sign",
                args.gripper_open_sign,
                "--idle-timeout",
                "-1",
            ]
            if args.use_bf16:
                server_cmd.append("--use-bf16")
            server_handle = _open_log(server_log)
            log_handles.append(server_handle)
            server_proc = subprocess.Popen(
                server_cmd,
                cwd=str(REPO_ROOT),
                env=_child_env(gpu_id, for_sim=False),
                stdout=server_handle,
                stderr=subprocess.STDOUT,
            )
            servers.append(server_proc)
            print(f"GPU {gpu_id}: loading policy server on port {port}", flush=True)
            _wait_for_port(port, server_proc, args.server_timeout_sec, server_log)
            print(f"GPU {gpu_id}: policy server ready, starting {args.sims_per_gpu} sim workers", flush=True)
            for worker_index in range(args.sims_per_gpu):
                worker_id = f"gpu{gpu_id}_{worker_index}"
                worker_log = output_dir / "logs" / f"{worker_id}.stdout"
                worker_cmd = [
                    sys.executable,
                    "-u",
                    str(WORKER_SCRIPT),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--queue-host",
                    "127.0.0.1",
                    "--queue-port",
                    str(queue_port),
                    "--output-dir",
                    str(output_dir),
                    "--worker-id",
                    worker_id,
                    "--seed",
                    str(args.seed),
                    "--num-steps-wait",
                    str(args.num_steps_wait),
                ]
                worker_handle = _open_log(worker_log)
                log_handles.append(worker_handle)
                worker_proc = subprocess.Popen(
                    worker_cmd,
                    cwd=str(REPO_ROOT),
                    env=_child_env(gpu_id, for_sim=True),
                    stdout=worker_handle,
                    stderr=subprocess.STDOUT,
                )
                workers.append(worker_proc)
                time.sleep(1)
                if worker_proc.poll() is not None:
                    raise RuntimeError(f"{worker_id} exited immediately. log: {worker_log}")
            if gpu_id != args.num_gpus - 1:
                remaining = args.launch_stagger_sec - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)

        last_completed = -1
        while True:
            completed, total, pending, inflight = queue.progress()
            if completed != last_completed:
                print(f"episodes {completed}/{total} pending {pending} inflight {inflight}", flush=True)
                last_completed = completed
            if completed >= total and pending == 0 and inflight == 0:
                break
            if any(proc.poll() is not None for proc in servers):
                raise RuntimeError("a policy server exited; see logs/server_gpu*.log")
            if workers and all(proc.poll() is not None for proc in workers):
                completed, total, pending, inflight = queue.progress()
                raise RuntimeError(
                    f"all sim workers exited early: completed {completed}/{total} "
                    f"pending {pending} inflight {inflight}"
                )
            time.sleep(5)

        summary = summarize_results(results_path)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        text = format_summary(summary)
        print(text, flush=True)
        with open(output_dir / "summary.txt", "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"Summary: {summary_path}", flush=True)
        for proc in workers:
            proc.wait(timeout=600)
    finally:
        queue.stop()
        _stop_processes(workers + servers)
        for handle in log_handles:
            handle.close()


if __name__ == "__main__":
    main()
