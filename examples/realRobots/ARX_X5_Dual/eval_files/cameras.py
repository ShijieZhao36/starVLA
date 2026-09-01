#!/usr/bin/env python
"""Minimal RGB cameras for ARX X5 dual-arm inference.

Default: Intel RealSense color stream via pyrealsense2, selected by serial.
Optional: OpenCV USB cameras via ``cv2.VideoCapture(index)``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CAMERA_SLOT_NAMES = ("front", "left_wrist", "right_wrist")


def _rotate_180(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame[::-1, ::-1])


class RealSenseCamera:
    """Color-only RealSense camera with a background latest-frame thread."""

    def __init__(
        self,
        *,
        serial_number: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip: bool = False,
    ):
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip
        self.is_connected = False
        self._pipeline = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise ImportError(
                "pyrealsense2 is required for RealSense cameras. "
                "Install it or pass --use-usb-cams."
            ) from error

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        try:
            pipeline.start(config)
        except RuntimeError as error:
            raise ConnectionError(
                f"Failed to open RealSense serial={self.serial_number}."
            ) from error

        self._pipeline = pipeline
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self.is_connected = True

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.async_read(timeout_s=0.2) is not None:
                logger.info("RealSense %s connected.", self.serial_number)
                return
        raise ConnectionError(f"RealSense {self.serial_number} produced no color frames.")

    def _read_loop(self) -> None:
        while not self._stop.is_set() and self._pipeline is not None:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data())
            if image.ndim == 2:
                continue
            if image.shape[-1] == 4:
                image = image[..., :3]
            if self.flip:
                image = _rotate_180(image)
            with self._lock:
                self._latest = np.ascontiguousarray(image)

    def async_read(self, timeout_s: float = 0.5) -> np.ndarray | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.01)
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None
        self.is_connected = False


class OpenCVCamera:
    """USB camera via OpenCV VideoCapture. Frames are converted BGR → RGB."""

    def __init__(
        self,
        *,
        index: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip: bool = False,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip
        self.is_connected = False
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None

    def connect(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise ImportError("opencv-python is required for --use-usb-cams.") from error

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise ConnectionError(f"Failed to open OpenCV camera index={self.index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self.is_connected = True

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.async_read(timeout_s=0.2) is not None:
                logger.info("OpenCV camera index=%s connected.", self.index)
                return
        raise ConnectionError(f"OpenCV camera index={self.index} produced no frames.")

    def _read_loop(self) -> None:
        import cv2

        while not self._stop.is_set() and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.flip:
                image = _rotate_180(image)
            with self._lock:
                self._latest = np.ascontiguousarray(image)

    def async_read(self, timeout_s: float = 0.5) -> np.ndarray | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.01)
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.is_connected = False


def make_cameras(
    slot_specs: dict[str, str],
    *,
    use_usb: bool,
    width: int,
    height: int,
    fps: int,
    flipped: set[str],
) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for name in CAMERA_SLOT_NAMES:
        if name not in slot_specs:
            raise ValueError(
                f"Missing camera spec for slot '{name}'. Required: {list(CAMERA_SLOT_NAMES)}."
            )
        source = slot_specs[name]
        flip = name in flipped
        if use_usb:
            cameras[name] = OpenCVCamera(
                index=int(source),
                width=width,
                height=height,
                fps=fps,
                flip=flip,
            )
        else:
            cameras[name] = RealSenseCamera(
                serial_number=source,
                width=width,
                height=height,
                fps=fps,
                flip=flip,
            )
    return cameras
