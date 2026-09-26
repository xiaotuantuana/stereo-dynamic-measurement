from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    frame_size: tuple[int, int]
    view_size: tuple[int, int]
    frame_count: int
    fps: float
    duration_s: float


def validate_stereo_frame_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) not in (2, 3) or len(shape) < 2:
        raise ValueError("视频帧格式无效；需要左右并排的灰度或彩色视频")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0 or width % 2 != 0:
        raise ValueError("视频必须是左右并排格式，且总宽度必须为偶数")
    return width // 2, height


def inspect_stereo_video(path: str | Path) -> VideoMetadata:
    video_path = Path(path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("视频中没有可读取的帧")
        view_size = validate_stereo_frame_shape(frame.shape)
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        return VideoMetadata(
            path=video_path,
            frame_size=(int(frame.shape[1]), int(frame.shape[0])),
            view_size=view_size,
            frame_count=frame_count,
            fps=fps,
            duration_s=(frame_count / fps if frame_count > 0 else 0.0),
        )
    finally:
        capture.release()


def normalize_disparity_image(disparity: np.ndarray) -> np.ndarray:
    values = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(values[valid], (2, 98))
        if high <= low:
            high = low + 1.0
        normalized[valid] = np.clip(
            (values[valid] - low) * 255.0 / (high - low),
            0,
            255,
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored
