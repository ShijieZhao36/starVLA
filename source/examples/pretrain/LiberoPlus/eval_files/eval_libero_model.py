import dataclasses
import json
import os
import pathlib
import sys

import draccus
import numpy as np
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

from libero.libero import benchmark

from policy_model import PolicyModel

_PARALLEL_EVAL = (
    pathlib.Path(__file__).resolve().parents[3]
    / "simBenchmarks"
    / "LIBERO-plus"
    / "eval_files"
    / "parallel_eval"
)
sys.path.insert(0, str(_PARALLEL_EVAL))

from libero_rollout import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _start_video_writer,
    empty_disturb_res,
    get_logger,
    load_id_to_category,
    max_steps_for_suite,
    rollout_episode,
)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "experiments/libero/logs"
    log_path: str = "experiments/libero/logs"
    seed: int = 7
    pretrained_path: str = ""
    post_process_action: bool = True
    job_name: str = "test"
    use_bf16: bool = True
    use_canonical_forward: bool = False
    start_idx: int = -1
    end_idx: int = -1
    output_dir: str = "./output"


@draccus.wrap()
def eval_libero(args: Args) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    print(f"Rank {rank}/{world_size} | GPU: {local_rank}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.start_idx == -1:
        args.start_idx = 0
        args.end_idx = num_tasks_in_suite
    log_path = os.path.join(args.output_dir, f"logs/{args.task_suite_name}")
    log_file = os.path.join(log_path, f"{args.start_idx}_{args.end_idx}.log")
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file)
    logger.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    video_out_path = os.path.join(args.output_dir, args.task_suite_name)
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)
    video_queue, video_thread = _start_video_writer(logger)
    max_steps = max_steps_for_suite(args.task_suite_name)

    client_model = PolicyModel(
        policy_ckpt_path=args.pretrained_path,
        image_size=args.resize_size,
        use_bf16=args.use_bf16,
        use_canonical_forward=args.use_canonical_forward,
    )

    libero_home = os.environ.get("LIBERO_HOME", "path_to_LIBERO-plus")
    id_to_category = load_id_to_category(libero_home, args.task_suite_name)
    disturb_res = empty_disturb_res(id_to_category)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(args.start_idx, args.end_idx)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                done, replay_images = rollout_episode(
                    env=env,
                    policy=client_model,
                    initial_state=initial_states[episode_idx],
                    task_description=task_description,
                    max_steps=max_steps,
                    num_steps_wait=args.num_steps_wait,
                    logger=logger,
                    episode_number=task_episodes + 1,
                )
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res[id_to_category[task_id + 1][0]]["success_count"] += 1
                task_episodes += 1
                total_episodes += 1
                disturb_res[id_to_category[task_id + 1][0]]["total_count"] += 1
                suffix = "success" if done else "failure"
                video_path = (
                    pathlib.Path(video_out_path)
                    / f"rollout_{id_to_category[task_id+1][1]}_episode{episode_idx}_{suffix}.mp4"
                )
                video_queue.put((video_path, replay_images))
                logger.info(f"Success: {done}")
                logger.info(f"# episodes completed so far: {total_episodes}")
                logger.info(
                    f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
                )
            logger.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logger.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        finally:
            env.close()
    with open(os.path.join(log_path, f"{args.start_idx}_to_{args.end_idx}.json"), "w", encoding="utf-8") as handle:
        json.dump(disturb_res, handle)
    logger.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logger.info(f"Total episodes: {total_episodes}")
    video_thread._replay_shutdown = True
    video_queue.put(None)
    video_thread.join()


if __name__ == "__main__":
    eval_libero()
