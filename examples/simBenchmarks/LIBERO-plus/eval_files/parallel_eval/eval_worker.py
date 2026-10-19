"""Lightweight LIBERO-plus worker. Pulls episodes from the task queue and calls a remote policy."""

import argparse
import os
import pathlib
import sys

import numpy as np
# libero.benchmark imports torch. This process still does not load the checkpoint.
from libero.libero import benchmark

from libero_rollout import (
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _start_video_writer,
    get_logger,
    load_id_to_category,
    max_steps_for_suite,
    rollout_episode,
)
from remote_policy import RemoteChunkPolicy
from task_queue import TaskQueueClient


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--queue-host", default="127.0.0.1")
    parser.add_argument("--queue-port", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    libero_home = os.environ.get("LIBERO_HOME")
    if not libero_home:
        print("LIBERO_HOME is required.", file=sys.stderr)
        sys.exit(2)

    log_dir = pathlib.Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_dir / f"{args.worker_id}.log", name=f"worker.{args.worker_id}")
    np.random.seed(args.seed)

    policy = RemoteChunkPolicy(args.host, args.port, image_size=list(args.image_size))
    queue_client = TaskQueueClient(args.queue_host, args.queue_port)
    video_queue, video_thread = _start_video_writer(logger)
    benchmark_dict = benchmark.get_benchmark_dict()
    suites = {}
    categories = {}
    completed = 0
    successes = 0

    try:
        while True:
            task = queue_client.next_task()
            if task is None:
                logger.info("No more episodes.")
                break
            try:
                result = _run_episode(
                    task=task,
                    policy=policy,
                    benchmark_dict=benchmark_dict,
                    suites=suites,
                    categories=categories,
                    libero_home=libero_home,
                    output_dir=pathlib.Path(args.output_dir),
                    seed=args.seed,
                    num_steps_wait=args.num_steps_wait,
                    video_queue=video_queue,
                    logger=logger,
                    worker_id=args.worker_id,
                )
            except Exception:
                logger.exception(
                    "Episode failed before a score was recorded: %s task %s episode %s",
                    task.get("suite"),
                    task.get("task_id"),
                    task.get("episode_idx"),
                )
                queue_client.release()
                continue
            queue_client.report(result)
            completed += 1
            successes += int(result["success"])
            logger.info(
                "Success: %s | worker episodes %s successes %s (%.1f%%)",
                result["success"],
                completed,
                successes,
                100.0 * successes / completed,
            )
    finally:
        video_thread._replay_shutdown = True
        video_queue.put(None)
        video_thread.join()
        queue_client.close()


def _run_episode(
    task,
    policy,
    benchmark_dict,
    suites,
    categories,
    libero_home,
    output_dir,
    seed,
    num_steps_wait,
    video_queue,
    logger,
    worker_id,
):
    suite_name = task["suite"]
    task_id = int(task["task_id"])
    episode_idx = int(task["episode_idx"])
    if suite_name not in suites:
        suites[suite_name] = benchmark_dict[suite_name]()
        categories[suite_name] = load_id_to_category(libero_home, suite_name)
    task_suite = suites[suite_name]
    id_to_category = categories[suite_name]
    category, item_name = id_to_category[task_id + 1]

    task_obj = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task_obj, LIBERO_ENV_RESOLUTION, seed)
    try:
        done, replay_images = rollout_episode(
            env=env,
            policy=policy,
            initial_state=initial_states[episode_idx],
            task_description=task_description,
            max_steps=max_steps_for_suite(suite_name),
            num_steps_wait=num_steps_wait,
            logger=logger,
            episode_number=episode_idx + 1,
        )
        suffix = "success" if done else "failure"
        video_dir = output_dir / suite_name
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"rollout_{item_name}_episode{episode_idx}_{suffix}.mp4"
        video_queue.put((video_path, replay_images))
        return {
            "suite": suite_name,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "success": bool(done),
            "category": category,
            "name": item_name,
            "worker": worker_id,
        }
    finally:
        env.close()


if __name__ == "__main__":
    main()
