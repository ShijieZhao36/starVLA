#!/usr/bin/env python
"""Synchronous StarVLA PI05 inference on two ARX X5 arms.

Control loop (pull / blocking, not RTC):
  1. Capture 3 cameras + 14D proprio (left arm then right arm).
  2. q99-normalize state with the checkpoint's training transform.
  3. Block on the Policy Server until a full action chunk returns.
  4. Execute the chunk step-by-step with dual-arm send_joint.

Hardware clients come from Evo-RL (set EVO_RL_ROOT or --evo-rl-root). Keyboard
hotkeys: [R] start, [Space] e-stop, [H] home, [I] next chunk in --safe-mode, [Q] quit.

State / action layout matches training DataConfig (NOT Evo-RL's right-then-left):
  [left_j1..j6, left_grip, right_j1..j6, right_grip]
Camera order matches PI05 image_keys: front, left_wrist, right_wrist.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import logging
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

DUAL_STATE_DIM = 14
JOINT_STEP_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
CAMERA_SLOT_NAMES = ("front", "left_wrist", "right_wrist")
DEFAULT_CAMERAS = (
    "front:254522071216",
    "left_wrist:150622073629",
    "right_wrist:409122272986",
)
DEFAULT_LEFT_CAN_PORT = "can0"
DEFAULT_RIGHT_CAN_PORT = "can1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_starvla_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_evo_rl_src(evo_rl_root: Path | None) -> Path | None:
    if evo_rl_root is not None:
        src = evo_rl_root.expanduser().resolve()
        if (src / "lerobot").is_dir():
            return src
        if (src / "src" / "lerobot").is_dir():
            return src / "src"
        raise FileNotFoundError(f"No lerobot package under --evo-rl-root={evo_rl_root}")

    env_root = os.environ.get("EVO_RL_ROOT")
    if env_root:
        src = Path(env_root).expanduser().resolve()
        if (src / "lerobot").is_dir():
            return src
        if (src / "src" / "lerobot").is_dir():
            return src / "src"

    sibling = Path.home() / "workspace" / "Evo-RL" / "src"
    if (sibling / "lerobot").is_dir():
        return sibling
    return None


def _ensure_evo_rl_on_path(evo_rl_root: Path | None) -> Path:
    src = _resolve_evo_rl_src(evo_rl_root)
    if src is None:
        raise FileNotFoundError(
            "Evo-RL not found. Set EVO_RL_ROOT or pass --evo-rl-root pointing at "
            "the Evo-RL repo (or its src/ directory)."
        )
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


class LoopState(Enum):
    STOPPED = auto()
    RUNNING = auto()


class KeyboardListener:
    """Non-blocking keyboard input for Linux terminals (same idea as Evo-RL)."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old_settings: list[int] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        atexit.register(self.restore)
        self.resume()

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (ValueError, OSError):
                break
            if self._stop.is_set():
                break
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char:
                self._queue.put(char.lower())

    def get_key(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def pause(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except termios.error:
                pass
            self._old_settings = None

    def resume(self) -> None:
        self.pause()
        self._stop.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except termios.error:
            self._old_settings = None
            LOGGER.warning("stdin is not a TTY; keyboard hotkeys are disabled.")
            return
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def restore(self) -> None:
        self.pause()


def _precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _parse_camera_specs(specs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid camera spec '{spec}', expected name:value")
        name, value = spec.split(":", 1)
        mapping[name.strip()] = value.strip()
    return mapping


def _make_camera_configs(
    *,
    camera_specs: dict[str, str],
    use_usb_cams: bool,
    width: int,
    height: int,
    fps: int,
    flipped_cameras: set[str],
) -> dict[str, Any]:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    configs: dict[str, Any] = {}
    for name in CAMERA_SLOT_NAMES:
        if name not in camera_specs:
            raise ValueError(
                f"Missing camera spec for slot '{name}'. Required: {list(CAMERA_SLOT_NAMES)}. "
                f"Pass e.g. --cameras {name}:<serial>."
            )
        source = camera_specs[name]
        rotation = 180 if name in flipped_cameras else 0
        if use_usb_cams:
            configs[name] = OpenCVCameraConfig(
                index_or_path=int(source),
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
        else:
            configs[name] = RealSenseCameraConfig(
                serial_number_or_name=source,
                width=width,
                height=height,
                fps=fps,
                use_depth=False,
                rotation=rotation,
            )
    return configs


def _read_dual_state(left_arm: Any, right_arm: Any) -> np.ndarray:
    """Training order: left 7D then right 7D."""
    left_state = list(left_arm.get_state())[:7]
    right_state = list(right_arm.get_state())[:7]
    state = np.asarray(left_state + right_state, dtype=np.float32)
    if state.shape[0] != DUAL_STATE_DIM:
        raise RuntimeError(f"Expected {DUAL_STATE_DIM}-dim state, got {state.shape[0]}.")
    return state


def _dummy_image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _build_observation(
    *,
    left_arm: Any,
    right_arm: Any,
    cameras: dict[str, Any] | None,
    cam_height: int,
    cam_width: int,
) -> dict[str, Any]:
    state = _read_dual_state(left_arm, right_arm)
    images: list[np.ndarray] = []
    if cameras is None:
        images = [_dummy_image(cam_height, cam_width) for _ in CAMERA_SLOT_NAMES]
    else:
        for name in CAMERA_SLOT_NAMES:
            frame = cameras[name].async_read()
            if frame is None:
                raise RuntimeError(f"Camera '{name}' returned no frame.")
            images.append(np.asarray(frame))
    return {"image": images, "state": state}


def _split_dual_action(action: np.ndarray) -> tuple[list[float], list[float]]:
    row = np.asarray(action, dtype=np.float64).reshape(-1)
    if row.size < DUAL_STATE_DIM:
        raise ValueError(f"Expected at least {DUAL_STATE_DIM} action dims, got {row.size}")
    left = [float(v) for v in row[:7]]
    right = [float(v) for v in row[7:14]]
    return left, right


def _clip_safe_actions(
    actions: np.ndarray,
    current_state: np.ndarray,
    max_joint_step: float,
) -> np.ndarray:
    previous = np.asarray(current_state, dtype=np.float64).reshape(-1)
    safe: list[np.ndarray] = []
    for row in np.asarray(actions, dtype=np.float64):
        target = np.asarray(row, dtype=np.float64).copy()
        delta = target[JOINT_STEP_INDICES] - previous[JOINT_STEP_INDICES]
        clipped = np.clip(delta, -max_joint_step, max_joint_step)
        if not np.allclose(delta, clipped):
            target[JOINT_STEP_INDICES] = previous[JOINT_STEP_INDICES] + clipped
            LOGGER.warning(
                "Safe mode: clipped dual-arm joint step, max_joint_step=%.4f",
                max_joint_step,
            )
        previous = target
        safe.append(target.astype(np.float32))
    return np.stack(safe, axis=0)


def _log_keyboard_help(safe_mode: bool) -> None:
    message = (
        "Keyboard: [Space] e-stop | [H] home | [R] start inference | [Q] quit"
    )
    if safe_mode:
        message += " | [I] next chunk"
    LOGGER.info(message)


def _run_keyboard_command(
    *,
    key: str | None,
    left_arm: Any,
    right_arm: Any,
    state: LoopState,
    safe_mode: bool,
    request_next_chunk: bool,
) -> tuple[LoopState, bool, bool]:
    """Returns (loop_state, request_next_chunk, running)."""
    if key == " ":
        left_arm.hold_position()
        right_arm.hold_position()
        LOGGER.info("E-stop: holding current pose.")
        return LoopState.STOPPED, False, True
    if key == "q":
        LOGGER.info("Quit requested.")
        return state, request_next_chunk, False
    if state == LoopState.STOPPED:
        if key == "h":
            LOGGER.info("Homing both ARX5 arms.")
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            left_arm.go_home()
            right_arm.go_home()
            time.sleep(2.0)
            left_arm.hold_position()
            right_arm.hold_position()
        elif key == "r":
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            LOGGER.info("Starting synchronous inference.")
            return LoopState.RUNNING, not safe_mode, True
    elif state == LoopState.RUNNING and safe_mode and key == "i":
        LOGGER.info("Safe mode: requested next chunk.")
        return state, True, True
    return state, request_next_chunk, True


def _execute_dual_chunk(
    *,
    left_arm: Any,
    right_arm: Any,
    actions: np.ndarray,
    step_duration_s: float,
    keyboard: KeyboardListener | None,
    safe_mode: bool,
) -> tuple[LoopState, bool, bool]:
    loop_state = LoopState.RUNNING
    request_next_chunk = not safe_mode
    running = True
    for index, action in enumerate(actions):
        step_start = time.perf_counter()
        if keyboard is not None:
            key = keyboard.get_key()
            loop_state, request_next_chunk, running = _run_keyboard_command(
                key=key,
                left_arm=left_arm,
                right_arm=right_arm,
                state=loop_state,
                safe_mode=safe_mode,
                request_next_chunk=request_next_chunk,
            )
            if loop_state != LoopState.RUNNING or not running:
                break

        left_joint, right_joint = _split_dual_action(action)
        left_thread = threading.Thread(target=left_arm.send_joint, args=(left_joint,))
        right_thread = threading.Thread(target=right_arm.send_joint, args=(right_joint,))
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()

        elapsed = time.perf_counter() - step_start
        if elapsed > step_duration_s:
            LOGGER.warning(
                "Control step overrun: step=%d elapsed=%.4fs > duration=%.4fs",
                index,
                elapsed,
                step_duration_s,
            )
        _precise_sleep(max(step_duration_s - elapsed, 0.0))
    return loop_state, request_next_chunk, running


def _predict_chunk(
    *,
    client: Any,
    images: list[np.ndarray],
    normalized_state: np.ndarray,
    task: str,
    unnorm_key: str | None,
) -> np.ndarray:
    payload: dict[str, Any] = {
        "examples": [
            {
                "image": images,
                "lang": task,
                "state": np.asarray(normalized_state, dtype=np.float32).reshape(1, -1),
            }
        ],
    }
    if unnorm_key is not None:
        payload["unnorm_key"] = unnorm_key
    response = client.predict_action(payload)
    if response.get("status") != "ok" and not response.get("ok", False):
        raise RuntimeError(f"Policy server error: {response}")
    data = response.get("data", response)
    if "actions" not in data:
        raise KeyError(f"No 'actions' in server response. keys={list(data.keys())}")
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim == 3:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError(f"Expected actions (T, D), got {actions.shape}")
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronous StarVLA PI05 control for ARX X5 dual arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, required=True, help="Language instruction.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Checkpoint used to rebuild PolicyNormProcessor. Defaults to server metadata ckpt_path.",
    )
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--duration", type=float, default=0.1, help="Seconds per executed action.")
    parser.add_argument("--evo-rl-root", type=Path, default=None)
    parser.add_argument("--left-can-port", type=str, default=DEFAULT_LEFT_CAN_PORT)
    parser.add_argument("--right-can-port", type=str, default=DEFAULT_RIGHT_CAN_PORT)
    parser.add_argument("--arm-type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use-stub", action="store_true")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--use-usb-cams", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-cameras", nargs="*", default=[])
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--max-joint-step", type=float, default=0.02)
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--protect-on-disconnect", action="store_true", default=True)
    parser.add_argument("--no-protect-on-disconnect", dest="protect_on_disconnect", action="store_false")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    args = parse_args()
    if args.execution_horizon is not None and args.execution_horizon <= 0:
        raise SystemExit("--execution-horizon must be positive.")
    if args.safe_mode and args.no_keyboard:
        raise SystemExit("--safe-mode requires keyboard control.")

    _ensure_starvla_on_path()
    evo_src = _ensure_evo_rl_on_path(args.evo_rl_root)
    LOGGER.info("Using Evo-RL package from %s", evo_src)

    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.robots.arx5_follower.arx5_client import ARX5ArmClient
    from lerobot.robots.arx5_follower.config_arx5_follower import ARX5FollowerConfigBase

    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    gripper_min = float(next(f.default for f in dataclasses.fields(ARX5FollowerConfigBase) if f.name == "gripper_min"))
    gripper_max = float(next(f.default for f in dataclasses.fields(ARX5FollowerConfigBase) if f.name == "gripper_max"))
    LOGGER.info("ARX5 gripper range: [%.3f, %.3f]", gripper_min, gripper_max)

    cameras: dict[str, Any] | None = None
    camera_configs = None
    if not args.use_stub:
        camera_specs = _parse_camera_specs(args.cameras)
        camera_configs = _make_camera_configs(
            camera_specs=camera_specs,
            use_usb_cams=args.use_usb_cams,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            flipped_cameras=set(args.flip_cameras),
        )
        cameras = make_cameras_from_configs(camera_configs)

    left_arm = ARX5ArmClient(
        can_port=args.left_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=Path("checkpoints/left_recorded_pose.json"),
    )
    right_arm = ARX5ArmClient(
        can_port=args.right_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=Path("checkpoints/right_recorded_pose.json"),
    )

    LOGGER.info("Connecting to StarVLA policy server %s:%s", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    server_meta = client.get_server_metadata()
    ckpt_path = args.ckpt_path or server_meta.get("ckpt_path")
    if not ckpt_path:
        raise SystemExit("Need --ckpt-path or server metadata ckpt_path to build PolicyNormProcessor.")
    unnorm_key = args.unnorm_key or server_meta.get("default_unnorm_key")
    chunk_size = int(server_meta.get("action_chunk_size", 50))
    execution_horizon = args.execution_horizon or chunk_size
    LOGGER.info(
        "Server metadata: action_chunk_size=%s default_unnorm_key=%s ckpt=%s",
        chunk_size,
        unnorm_key,
        ckpt_path,
    )

    norm_processor = PolicyNormProcessor(str(ckpt_path), unnorm_key=unnorm_key)
    LOGGER.info(
        "State keys=%s action keys=%s (left-then-right 14D)",
        norm_processor.state_keys,
        norm_processor.action_keys,
    )

    keyboard = None if args.no_keyboard else KeyboardListener()
    loop_state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    request_next_chunk = keyboard is None or not args.safe_mode
    running = True

    try:
        if cameras is not None:
            for name, camera in cameras.items():
                LOGGER.info("Connecting camera '%s'", name)
                camera.connect()

        if keyboard is not None:
            LOGGER.info("Homing both arms before keyboard control.")
            left_arm.go_home()
            right_arm.go_home()
            time.sleep(2.0)
            left_arm.hold_position()
            right_arm.hold_position()
            _log_keyboard_help(args.safe_mode)
            if args.safe_mode:
                LOGGER.info("Safe mode on. Press [R] to run, [I] after each chunk.")
        else:
            LOGGER.info("Running without keyboard: continuous sync infer loop.")

        while running:
            if keyboard is not None:
                key = keyboard.get_key()
                loop_state, request_next_chunk, running = _run_keyboard_command(
                    key=key,
                    left_arm=left_arm,
                    right_arm=right_arm,
                    state=loop_state,
                    safe_mode=args.safe_mode,
                    request_next_chunk=request_next_chunk,
                )
            if not running:
                break
            if loop_state != LoopState.RUNNING:
                time.sleep(0.05)
                continue
            if args.safe_mode and not request_next_chunk:
                time.sleep(0.05)
                continue

            try:
                observation = _build_observation(
                    left_arm=left_arm,
                    right_arm=right_arm,
                    cameras=cameras,
                    cam_height=args.cam_height,
                    cam_width=args.cam_width,
                )
            except Exception as error:
                LOGGER.error("Failed to read observation: %s", error)
                left_arm.hold_position()
                right_arm.hold_position()
                loop_state = LoopState.STOPPED
                request_next_chunk = False
                continue

            raw_state = np.asarray(observation["state"], dtype=np.float32)
            normalized_state = norm_processor.apply_state(raw_state)

            infer_start = time.perf_counter()
            actions = _predict_chunk(
                client=client,
                images=observation["image"],
                normalized_state=normalized_state,
                task=args.task,
                unnorm_key=unnorm_key,
            )
            LOGGER.info("Sync infer took %.4fs, chunk shape=%s", time.perf_counter() - infer_start, actions.shape)

            actions = actions[:execution_horizon]
            if actions.shape[-1] > DUAL_STATE_DIM:
                actions = actions[:, :DUAL_STATE_DIM]
            if args.safe_mode:
                actions = _clip_safe_actions(actions, raw_state, args.max_joint_step)
            if actions.shape[0] == 0:
                LOGGER.warning("Empty action chunk; retrying.")
                time.sleep(0.1)
                continue

            LOGGER.info(
                "Executing %d actions. left_grip=%s right_grip=%s",
                actions.shape[0],
                np.round(actions[:, 6], 4).tolist()[:8],
                np.round(actions[:, 13], 4).tolist()[:8],
            )
            request_next_chunk = False
            loop_state, request_next_chunk, running = _execute_dual_chunk(
                left_arm=left_arm,
                right_arm=right_arm,
                actions=actions,
                step_duration_s=args.duration,
                keyboard=keyboard,
                safe_mode=args.safe_mode,
            )
            if args.safe_mode and loop_state == LoopState.RUNNING:
                LOGGER.info("Safe mode: press [I] for the next chunk.")
    finally:
        if keyboard is not None:
            keyboard.restore()
        if cameras is not None:
            for camera in cameras.values():
                if getattr(camera, "is_connected", False):
                    camera.disconnect()
        client.close()
        if args.protect_on_disconnect:
            left_arm.protect_mode()
            right_arm.protect_mode()


if __name__ == "__main__":
    main()
