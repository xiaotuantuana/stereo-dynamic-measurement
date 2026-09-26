from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .models import MatcherConfig


@dataclass(frozen=True)
class CameraCompensationResult:
    status: str
    rotation: np.ndarray | None
    translation_m: np.ndarray | None
    inlier_ids: tuple[str, ...]
    residual_rmse_mm: float | None


def estimate_rigid_transform_weighted(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    current, reference, normalized_weights = _validated_points(
        current_points,
        reference_points,
        weights,
        minimum=3,
    )
    if _is_degenerate(current) or _is_degenerate(reference):
        raise ValueError("Rigid transform requires at least three non-collinear points")
    current_centroid = np.sum(current * normalized_weights[:, None], axis=0)
    reference_centroid = np.sum(reference * normalized_weights[:, None], axis=0)
    current_centered = current - current_centroid
    reference_centered = reference - reference_centroid
    covariance = (current_centered * normalized_weights[:, None]).T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = reference_centroid - rotation @ current_centroid
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Rigid transform estimation produced non-finite values")
    return rotation, translation


def estimate_camera_compensation(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    point_ids: tuple[str, ...],
    weights: np.ndarray | None,
    config: MatcherConfig,
) -> CameraCompensationResult:
    current = np.asarray(current_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    count = current.shape[0] if current.ndim == 2 else 0
    if count < config.camera_compensation_min_points:
        return CameraCompensationResult(
            "insufficient_references", None, None, (), None
        )
    if len(point_ids) != count:
        raise ValueError("point_ids must match the number of reference points")
    try:
        current, reference, normalized_weights = _validated_points(
            current,
            reference,
            weights,
            minimum=config.camera_compensation_min_points,
        )
    except ValueError:
        return CameraCompensationResult("invalid_references", None, None, (), None)
    if _is_degenerate(current) or _is_degenerate(reference):
        return CameraCompensationResult("degenerate_references", None, None, (), None)

    minimum_inliers = max(
        config.camera_compensation_min_points,
        int(np.ceil(config.camera_compensation_min_inlier_ratio * count)),
    )
    threshold_m = config.camera_compensation_inlier_threshold_mm / 1000.0
    triplets = list(combinations(range(count), 3))
    if len(triplets) > config.camera_compensation_ransac_iterations:
        rng = np.random.default_rng(0)
        indices = rng.choice(
            len(triplets),
            size=config.camera_compensation_ransac_iterations,
            replace=False,
        )
        triplets = [triplets[int(index)] for index in sorted(indices)]

    best_inliers: np.ndarray | None = None
    best_rmse = float("inf")
    found_nondegenerate = False
    for sample_indices in triplets:
        sample = np.asarray(sample_indices, dtype=np.int64)
        if _is_degenerate(current[sample]) or _is_degenerate(reference[sample]):
            continue
        found_nondegenerate = True
        try:
            rotation, translation = estimate_rigid_transform_weighted(
                current[sample],
                reference[sample],
                normalized_weights[sample],
            )
        except ValueError:
            continue
        residuals = np.linalg.norm(
            apply_rigid_transform(current, rotation, translation) - reference,
            axis=1,
        )
        inliers = np.flatnonzero(residuals <= threshold_m)
        if inliers.size < minimum_inliers:
            continue
        rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        if (
            best_inliers is None
            or inliers.size > best_inliers.size
            or (inliers.size == best_inliers.size and rmse < best_rmse)
        ):
            best_inliers = inliers
            best_rmse = rmse
    if best_inliers is None:
        status = "ransac_failed" if found_nondegenerate else "degenerate_references"
        return CameraCompensationResult(status, None, None, (), None)

    rotation, translation = estimate_rigid_transform_weighted(
        current[best_inliers],
        reference[best_inliers],
        normalized_weights[best_inliers],
    )
    residuals = np.linalg.norm(
        apply_rigid_transform(current[best_inliers], rotation, translation)
        - reference[best_inliers],
        axis=1,
    )
    rmse_mm = float(np.sqrt(np.mean(residuals**2)) * 1000.0)
    if not np.isfinite(rmse_mm) or rmse_mm > config.camera_compensation_max_rmse_mm:
        return CameraCompensationResult("rmse_exceeded", None, None, (), rmse_mm)
    return CameraCompensationResult(
        status="valid",
        rotation=rotation,
        translation_m=translation,
        inlier_ids=tuple(point_ids[int(index)] for index in best_inliers),
        residual_rmse_mm=rmse_mm,
    )


def apply_rigid_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rotation and translation have invalid shapes")
    transformed = (rotation @ values.T).T + translation
    if not np.isfinite(transformed).all():
        raise ValueError("Rigid transform produced non-finite values")
    return transformed


def rotation_matrix_to_euler_xyz_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return intrinsic XYZ Euler angles in degrees for the logged rigid transform."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    sy = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if sy > 1e-8:
        x, y, z = np.arctan2(matrix[2, 1], matrix[2, 2]), np.arctan2(-matrix[2, 0], sy), np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x, y, z = np.arctan2(-matrix[1, 2], matrix[1, 1]), np.arctan2(-matrix[2, 0], sy), 0.0
    return tuple(float(value) for value in np.degrees([x, y, z]))


def _validated_points(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    weights: np.ndarray | None,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = np.asarray(current_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    if (
        current.ndim != 2
        or current.shape[1:] != (3,)
        or reference.shape != current.shape
        or current.shape[0] < minimum
        or not np.isfinite(current).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Point arrays must be finite matching arrays with shape (N, 3)")
    if weights is None:
        normalized = np.full(current.shape[0], 1.0 / current.shape[0], dtype=np.float64)
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        if (
            normalized.shape != (current.shape[0],)
            or not np.isfinite(normalized).all()
            or np.any(normalized <= 0)
        ):
            raise ValueError("weights must be finite and positive")
        normalized = normalized / float(normalized.sum())
    return current, reference, normalized


def _is_degenerate(points: np.ndarray) -> bool:
    centered = points - np.mean(points, axis=0)
    return int(np.linalg.matrix_rank(centered, tol=1e-10)) < 2
