from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from ..calibration import StereoCalibration
from ..global_matching import GlobalStereoMatcher
from ..models import FramePointResult, MatcherConfig, MethodName, PointSpec, method_profile
from ..pipeline import TemporalStereoPipeline
from ..runner import CSV_FIELDS, split_side_by_side
from .video import normalize_disparity_image


@dataclass(frozen=True)
class ProcessingRequest:
    video_path: Path
    output_path: Path
    method: MethodName
    start_frame: int
    end_frame: int
    points: tuple[PointSpec, ...]
    calibration: StereoCalibration
    config: MatcherConfig
    result_method_label: str | None = None

    def __post_init__(self) -> None:
        method_profile(self.method)
        if self.start_frame < 0:
            raise ValueError("起始帧不能小于 0")
        if self.end_frame < self.start_frame:
            raise ValueError("结束帧不能早于起始帧")
        if not self.points:
            raise ValueError("至少需要一个测点")


@dataclass(frozen=True)
class FramePreview:
    frame: int
    progress_pct: int
    left: np.ndarray
    right: np.ndarray
    disparity_bgr: np.ndarray
    measurement_bgr: np.ndarray
    results: tuple[FramePointResult, ...]
    frame_total_ms: float


@dataclass(frozen=True)
class ProcessingSummary:
    output_path: Path
    frames_processed: int
    rows_written: int
    valid_rows: int
    cancelled: bool
    elapsed_s: float


@dataclass(frozen=True)
class BatchProcessingSummary:
    summaries: tuple[ProcessingSummary, ...]
    cancelled: bool


class StereoVideoProcessor:
    def __init__(self, request: ProcessingRequest) -> None:
        self.request = request

    def run(
        self,
        on_frame: Callable[[FramePreview], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        wait_if_paused: Callable[[], None] | None = None,
    ) -> ProcessingSummary:
        request = self.request
        if not request.video_path.exists():
            raise FileNotFoundError(f"视频不存在：{request.video_path}")
        capture = cv2.VideoCapture(str(request.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{request.video_path}")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        capture.set(cv2.CAP_PROP_POS_FRAMES, request.start_frame)
        cv2.setNumThreads(1)
        pipeline = TemporalStereoPipeline(
            request.method,
            request.calibration.rectification().q,
            request.calibration.unit,
            request.config,
        )
        display_matcher = GlobalStereoMatcher(request.config)
        histories: dict[str, list[tuple[int, int]]] = {
            point.point_id: [] for point in request.points
        }
        frames_processed = 0
        rows_written = 0
        valid_rows = 0
        cancelled = False
        started = time.perf_counter()
        total_frames = request.end_frame - request.start_frame + 1
        try:
            with request.output_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for frame_index in range(request.start_frame, request.end_frame + 1):
                    if wait_if_paused is not None:
                        wait_if_paused()
                    if should_stop is not None and should_stop():
                        cancelled = True
                        break
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    left_raw, right_raw = split_side_by_side(frame)
                    left, right = request.calibration.rectify_pair(left_raw, right_raw)
                    algorithm_started = time.perf_counter()
                    if frames_processed == 0:
                        results = pipeline.initialize(left, right, request.points, frame_index)
                    else:
                        results = pipeline.step(left, right, frame_index)
                    frame_total_ms = (time.perf_counter() - algorithm_started) * 1000.0
                    for result in results:
                        row = {"repeat": 0, **result.as_csv_row(), "frame_total_ms": frame_total_ms}
                        if request.result_method_label:
                            row["method"] = request.result_method_label
                        writer.writerow(row)
                        rows_written += 1
                        valid_rows += int(result.status == "valid")
                    handle.flush()

                    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
                    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
                    dense = display_matcher.compute(left_gray, right_gray)
                    disparity_bgr = normalize_disparity_image(dense.left_disparity)
                    measurement = _render_measurement(left, results, histories)
                    frames_processed += 1
                    progress = int(round(frames_processed * 100.0 / total_frames))
                    if on_frame is not None:
                        on_frame(
                            FramePreview(
                                frame=frame_index,
                                progress_pct=min(progress, 100),
                                left=left,
                                right=right,
                                disparity_bgr=disparity_bgr,
                                measurement_bgr=measurement,
                                results=tuple(results),
                                frame_total_ms=frame_total_ms,
                            )
                        )
        finally:
            capture.release()
        return ProcessingSummary(
            output_path=request.output_path,
            frames_processed=frames_processed,
            rows_written=rows_written,
            valid_rows=valid_rows,
            cancelled=cancelled,
            elapsed_s=time.perf_counter() - started,
        )


class ProcessingWorker(QObject):
    frameReady = Signal(object)
    completed = Signal(object)
    failed = Signal(str)
    statusChanged = Signal(str)

    def __init__(self, request: ProcessingRequest) -> None:
        super().__init__()
        self.request = request
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False

    @Slot()
    def run(self) -> None:
        try:
            self.statusChanged.emit("正在处理")
            summary = StereoVideoProcessor(self.request).run(
                on_frame=self.frameReady.emit,
                should_stop=self._should_stop,
                wait_if_paused=self._wait_if_paused,
            )
            self.completed.emit(summary)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    @Slot(bool)
    def set_paused(self, paused: bool) -> None:
        with self._condition:
            self._paused = bool(paused)
            if not self._paused:
                self._condition.notify_all()
        self.statusChanged.emit("已暂停" if paused else "正在处理")

    @Slot()
    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()

    def _should_stop(self) -> bool:
        with self._condition:
            return self._stopped

    def _wait_if_paused(self) -> None:
        with self._condition:
            while self._paused and not self._stopped:
                self._condition.wait(timeout=0.2)


class BatchProcessingWorker(QObject):
    """Sequential GUI runner for a predefined thesis experiment group."""
    frameReady = Signal(object)
    completed = Signal(object)
    failed = Signal(str)
    statusChanged = Signal(str)

    def __init__(self, requests: tuple[ProcessingRequest, ...]) -> None:
        super().__init__()
        self.requests = requests
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False

    @Slot()
    def run(self) -> None:
        summaries: list[ProcessingSummary] = []
        try:
            for index, request in enumerate(self.requests, start=1):
                if self._should_stop():
                    break
                self.statusChanged.emit(
                    f"实验 {index}/{len(self.requests)}：{request.result_method_label or request.method}"
                )
                summary = StereoVideoProcessor(request).run(
                    on_frame=self.frameReady.emit,
                    should_stop=self._should_stop,
                    wait_if_paused=self._wait_if_paused,
                )
                summaries.append(summary)
                if summary.cancelled:
                    break
            self.completed.emit(BatchProcessingSummary(tuple(summaries), self._should_stop()))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    @Slot(bool)
    def set_paused(self, paused: bool) -> None:
        with self._condition:
            self._paused = bool(paused)
            if not self._paused:
                self._condition.notify_all()
        self.statusChanged.emit("已暂停" if paused else "正在处理")

    @Slot()
    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()

    def _should_stop(self) -> bool:
        with self._condition:
            return self._stopped

    def _wait_if_paused(self) -> None:
        with self._condition:
            while self._paused and not self._stopped:
                self._condition.wait(timeout=0.2)


def _render_measurement(
    left: np.ndarray,
    results: list[FramePointResult],
    histories: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    display = left.copy()
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], 34), (7, 15, 27), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    valid_count = sum(result.status == "valid" for result in results)
    cv2.putText(
        display,
        f"VALID {valid_count}/{len(results)}",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (226, 236, 247),
        1,
        cv2.LINE_AA,
    )
    for result in results:
        if result.left_x is None or result.left_y is None:
            continue
        center = (int(round(result.left_x)), int(round(result.left_y)))
        history = histories.setdefault(result.point_id, [])
        if result.status == "valid":
            history.append(center)
            del history[:-120]
        if len(history) >= 2:
            cv2.polylines(
                display,
                [np.asarray(history, dtype=np.int32)],
                False,
                (255, 180, 60),
                1,
                cv2.LINE_AA,
            )
        color = (75, 220, 135) if result.status == "valid" else (80, 90, 230)
        cv2.circle(display, center, 5, color, 2, cv2.LINE_AA)
        depth = f" Z={result.z_m:.3f}m" if result.z_m is not None else ""
        cv2.putText(
            display,
            f"{result.point_id}{depth}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )
    return display
