from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectification:
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray
    q: np.ndarray
    focal_px: float
    baseline_m: float
    valid_roi_left: tuple[int, int, int, int]
    valid_roi_right: tuple[int, int, int, int]


@dataclass(frozen=True)
class StereoCalibration:
    name: str
    image_size: tuple[int, int]
    left_camera_matrix: np.ndarray
    right_camera_matrix: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    unit: str = "mm"

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain positive width and height")
        if self.left_camera_matrix.shape != (3, 3) or self.right_camera_matrix.shape != (3, 3):
            raise ValueError("Camera matrices must have shape (3, 3)")
        if self.rotation.shape != (3, 3):
            raise ValueError("Stereo rotation must have shape (3, 3)")
        if self.translation.size != 3:
            raise ValueError("Stereo translation must contain three values")
        if self.unit not in {"mm", "m"}:
            raise ValueError("Calibration unit must be 'mm' or 'm'")

    @cached_property
    def _rectification(self) -> Rectification:
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            self.left_camera_matrix,
            self.left_distortion,
            self.right_camera_matrix,
            self.right_distortion,
            self.image_size,
            self.rotation,
            self.translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
        )
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix,
            self.left_distortion,
            r1,
            p1,
            self.image_size,
            cv2.CV_16SC2,
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix,
            self.right_distortion,
            r2,
            p2,
            self.image_size,
            cv2.CV_16SC2,
        )
        raw_baseline = abs(1.0 / float(q[3, 2]))
        baseline_m = raw_baseline / 1000.0 if self.unit == "mm" else raw_baseline
        return Rectification(
            left_map1=left_map1,
            left_map2=left_map2,
            right_map1=right_map1,
            right_map2=right_map2,
            q=q.astype(np.float64),
            focal_px=abs(float(q[2, 3])),
            baseline_m=baseline_m,
            valid_roi_left=tuple(int(v) for v in roi1),
            valid_roi_right=tuple(int(v) for v in roi2),
        )

    def rectification(self) -> Rectification:
        return self._rectification

    def rectify_pair(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expected = (self.image_size[1], self.image_size[0])
        if left.shape[:2] != expected or right.shape[:2] != expected:
            width, height = self.image_size
            raise ValueError(
                f"Calibration {self.name} requires {width}x{height} images; "
                f"received left={left.shape[1]}x{left.shape[0]} and "
                f"right={right.shape[1]}x{right.shape[0]}"
            )
        rectification = self.rectification()
        left_rectified = cv2.remap(
            left,
            rectification.left_map1,
            rectification.left_map2,
            cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            right,
            rectification.right_map1,
            rectification.right_map2,
            cv2.INTER_LINEAR,
        )
        return left_rectified, right_rectified

    def to_json(self, path: str | Path) -> Path:
        output = Path(path)
        payload = {
            "name": self.name,
            "image_size": list(self.image_size),
            "left_camera_matrix": self.left_camera_matrix.tolist(),
            "right_camera_matrix": self.right_camera_matrix.tolist(),
            "left_distortion": self.left_distortion.reshape(-1).tolist(),
            "right_distortion": self.right_distortion.reshape(-1).tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.reshape(-1).tolist(),
            "unit": self.unit,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_json(cls, path: str | Path) -> "StereoCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(payload["name"]),
            image_size=tuple(int(value) for value in payload["image_size"]),
            left_camera_matrix=np.asarray(payload["left_camera_matrix"], dtype=np.float64),
            right_camera_matrix=np.asarray(payload["right_camera_matrix"], dtype=np.float64),
            left_distortion=np.asarray(payload["left_distortion"], dtype=np.float64),
            right_distortion=np.asarray(payload["right_distortion"], dtype=np.float64),
            rotation=np.asarray(payload["rotation"], dtype=np.float64),
            translation=np.asarray(payload["translation"], dtype=np.float64),
            unit=str(payload.get("unit", "mm")),
        )


def builtin_640x480() -> StereoCalibration:
    return StereoCalibration(
        name="original_640x480_sgbm",
        image_size=(640, 480),
        left_camera_matrix=np.array(
            [
                [516.5066236, -1.444673028, 320.2950423],
                [0.0, 516.5816117, 270.7881873],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        right_camera_matrix=np.array(
            [
                [511.8428182, 1.295112628, 317.310253],
                [0.0, 513.0748795, 269.5885026],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        left_distortion=np.array(
            [-0.046645194, 0.077595167, 0.012476819, -0.000711358, 0.0],
            dtype=np.float64,
        ),
        right_distortion=np.array(
            [-0.061588946, 0.122384376, 0.011081232, -0.000750439, 0.0],
            dtype=np.float64,
        ),
        rotation=np.array(
            [
                [0.999911333, -0.004351508, 0.012585312],
                [0.004184066, 0.999902792, 0.013300386],
                [-0.012641965, -0.013246549, 0.999832341],
            ],
            dtype=np.float64,
        ),
        translation=np.array(
            [-120.3559901, -0.188953775, -0.662073075],
            dtype=np.float64,
        ),
        unit="mm",
    )
