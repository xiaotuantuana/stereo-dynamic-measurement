from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"Expected array shape {shape}, received {result.shape}")
    return result


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True)
class StereoCameraModel:
    """Stereo calibration in the left-camera coordinate system and mm units."""

    name: str
    image_size: tuple[int, int]
    K_left: np.ndarray
    D_left: np.ndarray
    K_right: np.ndarray
    D_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    unit: str = "mm"

    def __post_init__(self) -> None:
        if self.unit != "mm":
            raise ValueError("StereoCameraModel requires calibration translation in mm")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must be positive (width, height)")
        for name, matrix, shape in (
            ("K_left", self.K_left, (3, 3)), ("K_right", self.K_right, (3, 3)),
            ("R", self.R, (3, 3)), ("P1", self.P1, (3, 4)),
            ("P2", self.P2, (3, 4)), ("Q", self.Q, (4, 4)),
        ):
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must be a finite {shape} matrix")
        if self.T.shape not in {(3,), (3, 1)} or not np.isfinite(self.T).all():
            raise ValueError("T must contain three finite mm values")

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StereoCameraModel":
        unit = str(mapping.get("unit", "mm"))
        K_left = _array(_first(mapping, "K_left", "left_camera_matrix"), (3, 3))
        K_right = _array(_first(mapping, "K_right", "right_camera_matrix"), (3, 3))
        D_left = _array(_first(mapping, "D_left", "left_distortion", default=np.zeros(5))).reshape(-1)
        D_right = _array(_first(mapping, "D_right", "right_distortion", default=np.zeros(5))).reshape(-1)
        R = _array(_first(mapping, "R", "rotation", default=np.eye(3)), (3, 3))
        T = _array(_first(mapping, "T", "translation")).reshape(3)
        size = _first(mapping, "image_size", "imageSize")
        if size is None:
            raise ValueError("Calibration is missing image_size")
        image_size = tuple(int(v) for v in np.asarray(size).reshape(-1)[:2])
        P1 = _first(mapping, "P1")
        P2 = _first(mapping, "P2")
        if P1 is None:
            P1 = K_left @ np.column_stack((np.eye(3), np.zeros(3)))
        if P2 is None:
            P2 = K_right @ np.column_stack((R, T))
        Q = _first(mapping, "Q")
        if Q is None:
            _, _, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(
                K_left, D_left, K_right, D_right, image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY,
            )
            # Rectified matrices make disparity/Q internally consistent while
            # original P1/P2 retain the physical camera projection convention.
            del P1_rect, P2_rect
        return cls(
            name=str(mapping.get("name", "stereo_camera")), image_size=image_size,
            K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right,
            R=R, T=T, P1=_array(P1, (3, 4)), P2=_array(P2, (3, 4)),
            Q=_array(Q, (4, 4)), unit=unit,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StereoCameraModel":
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        elif suffix == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                payload = {key: archive[key].tolist() for key in archive.files}
        elif suffix == ".xml":
            storage = cv2.FileStorage(str(source), cv2.FILE_STORAGE_READ)
            if not storage.isOpened():
                raise ValueError(f"Unable to open OpenCV XML calibration: {source}")
            try:
                keys = ("name", "image_size", "K_left", "D_left", "K_right", "D_right", "R", "T", "P1", "P2", "Q", "unit")
                payload = {}
                for key in keys:
                    node = storage.getNode(key)
                    if node.empty():
                        continue
                    if key in {"name", "unit"}:
                        payload[key] = node.string()
                    else:
                        matrix = node.mat()
                        if matrix is not None:
                            payload[key] = matrix.tolist()
            finally:
                storage.release()
        else:
            raise ValueError("Calibration format must be .yaml/.yml, .json, .xml, or .npz")
        if not isinstance(payload, Mapping):
            raise ValueError("Calibration file must contain a mapping")
        return cls.from_mapping(payload)

    def project(self, xyz_mm: np.ndarray, camera: str) -> np.ndarray:
        """Project Nx3 left-world points to Nx2 pixels without adding noise."""
        points = _array(xyz_mm)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz_mm must have shape (N, 3)")
        matrix = self.P1 if camera == "left" else self.P2 if camera == "right" else None
        if matrix is None:
            raise ValueError("camera must be 'left' or 'right'")
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
        if np.any(homogeneous[:, 2] <= 0):
            raise ValueError("All projected points must have positive camera depth")
        return homogeneous[:, :2] / homogeneous[:, 2:3]
