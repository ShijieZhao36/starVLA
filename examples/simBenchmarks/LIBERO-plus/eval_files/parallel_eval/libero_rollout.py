"""LIBERO-plus rollout loop. This module does not import torch."""

import atexit
import json
import logging
import math
import os
import pathlib
import queue
import threading

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

SUITE_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}

EVAL_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def max_steps_for_suite(task_suite_name: str) -> int:
    try:
        return SUITE_MAX_STEPS[task_suite_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task suite: {task_suite_name}") from exc


def _obs_eef_quat_to_xyz_rot6d(pos, quat_xyzw) -> np.ndarray:
    """xyz plus the first two rows of the rotation matrix. Quat is robosuite xyzw."""
    position = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    matrix = Rotation.from_quat(quat).as_matrix()
    return np.concatenate([position, matrix[:2, :].reshape(6)]).astype(np.float32)


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def get_logger(file, name="dual_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_id_to_category(libero_home: str, task_suite_name: str) -> dict:
    classification_path = os.path.join(libero_home, "libero/libero/benchmark/task_classification.json")
    with open(classification_path, encoding="utf-8") as f:
        task_mapping = json.load(f)[task_suite_name]
    id_to_category = {}
    for item in task_mapping:
        id_to_category[item["id"]] = (item["category"], item["name"])
    return id_to_category


def empty_disturb_res(id_to_category: dict) -> dict:
    disturb_res = {}
    for category, _item_name in id_to_category.values():
        if category not in disturb_res:
            disturb_res[category] = {"total_count": 0, "success_count": 0}
    return disturb_res


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def rollout_episode(
    env,
    policy,
    initial_state,
    task_description,
    max_steps,
    num_steps_wait,
    logger,
    episode_number: int,
):
    """Run one episode. Returns ``(success, replay_images)``.

    Control flow matches the in-process evaluator: dummy steps, 180-degree image
    flip, no proprioceptive state sent to the policy, gripper binarized after unnorm.
    """
    logger.info(f"\nTask: {task_description}")

    # Reset environment
    policy.reset(task_description=task_description)
    env.reset()

    # Set initial states
    obs = env.set_init_state(initial_state)

    # Setup
    t = 0
    replay_images = []
    full_actions = []
    done = False

    logger.info(f"Starting episode {episode_number}...")
    step = 0

    while t < max_steps + num_steps_wait:
        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
        # and we need to wait for them to fall
        if t < num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # IMPORTANT: rotate 180 degrees to match train preprocessing
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        replay_images.append(img)

        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )

        observation = {  #
            "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
            "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
            "observation.state": np.expand_dims(state, axis=0),
            "instruction": [str(task_description)],
        }

        example_dict = {
            "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
            "lang": observation["instruction"][0],
            # Absolute xyz+rot6d at this control step. decode_action uses it as the
            # chunk-start reference; it is not the normalized proprio sent to the VLM.
            "raw_right_eef": _obs_eef_quat_to_xyz_rot6d(obs["robot0_eef_pos"], obs["robot0_eef_quat"]),
        }

        response = policy.step(example=example_dict, step=step)
        raw_action = response["raw_action"]

        world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
        gripper = _binarize_gripper_open(open_gripper)

        if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
            logger.warning(
                f"Unexpected action sizes: "
                f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                f"Falling back to LIBERO_DUMMY_ACTION."
            )
            raise ValueError(
                f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
            )
        else:
            delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

        full_actions.append(delta_action)

        # see ../robosuite/controllers/controller_factory.py
        obs, reward, done, info = env.step(delta_action.tolist())
        if done:
            break
        t += 1
        step += 1

    if full_actions:
        np.stack(full_actions)
    return bool(done), replay_images


def _encode_replay_video(video_path, frames) -> None:
    """Write every replay frame. Prefer one-thread ultrafast H.264; fall back to mpeg4."""
    import av

    arrays = [np.asarray(frame) for frame in frames]
    if not arrays:
        return
    height, width = arrays[0].shape[:2]
    codec = "libx264"
    try:
        av.Codec(codec, "w")
    except Exception:
        codec = "mpeg4"
    container = av.open(str(video_path), mode="w")
    try:
        stream = container.add_stream(codec, rate=25)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            stream.options = {"preset": "ultrafast", "threads": "1"}
        stream.codec_context.thread_count = 1
        stream.codec_context.thread_type = av.codec.context.ThreadType.NONE
        for array in arrays:
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _start_video_writer(logger):
    """Encode replay mp4s on one background thread so the next episode can start."""
    video_queue = queue.Queue(maxsize=4)

    def _loop():
        while True:
            item = video_queue.get()
            if item is None:
                break
            video_path, frames = item
            try:
                _encode_replay_video(video_path, frames)
            except Exception as exc:
                logger.warning(f"Failed to write replay video {video_path}: {exc}")

    thread = threading.Thread(target=_loop, name="replay-video-writer")
    thread.start()

    def _shutdown():
        if thread._replay_shutdown:
            return
        thread._replay_shutdown = True
        video_queue.put(None)
        thread.join()

    thread._replay_shutdown = False
    atexit.register(_shutdown)
    return video_queue, thread
