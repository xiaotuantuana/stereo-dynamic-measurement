from __future__ import annotations

import numpy as np

from stereo_research.camera_compensation import (
    apply_rigid_transform,
    estimate_camera_compensation,
    estimate_rigid_transform_weighted,
)
from stereo_research.models import FramePointResult, MatcherConfig, PointState
from stereo_research.pipeline import TemporalStereoPipeline


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]],
        dtype=np.float64,
    )


def _reference_points() -> np.ndarray:
    return np.array(
        [
            [-0.3, -0.2, 2.0],
            [0.4, -0.1, 2.2],
            [-0.2, 0.5, 2.4],
            [0.5, 0.4, 2.7],
            [0.1, 0.2, 3.0],
        ],
        dtype=np.float64,
    )


def test_weighted_kabsch_recovers_known_transform() -> None:
    reference = _reference_points()
    expected_rotation = _rotation_z(2.5)
    expected_translation = np.array([0.012, -0.008, 0.02])
    current = (expected_rotation.T @ (reference - expected_translation).T).T

    rotation, translation = estimate_rigid_transform_weighted(
        current,
        reference,
        weights=np.array([1.0, 2.0, 1.5, 0.8, 1.2]),
    )

    assert np.allclose(rotation, expected_rotation, atol=1e-10)
    assert np.allclose(translation, expected_translation, atol=1e-10)
    assert np.sqrt(np.mean(np.sum((apply_rigid_transform(current, rotation, translation) - reference) ** 2, axis=1))) < 1e-10


def test_ransac_rejects_one_bad_reference_point() -> None:
    reference = _reference_points()
    expected_rotation = _rotation_z(-1.8)
    expected_translation = np.array([-0.01, 0.006, 0.015])
    current = (expected_rotation.T @ (reference - expected_translation).T).T
    current[4] += np.array([0.08, -0.06, 0.1])

    result = estimate_camera_compensation(
        current,
        reference,
        point_ids=("R1", "R2", "R3", "R4", "BAD"),
        weights=np.ones(5),
        config=MatcherConfig(
            camera_compensation_inlier_threshold_mm=2.0,
            camera_compensation_max_rmse_mm=2.0,
        ),
    )

    assert result.status == "valid"
    assert "BAD" not in result.inlier_ids
    assert len(result.inlier_ids) == 4
    assert result.residual_rmse_mm is not None and result.residual_rmse_mm < 0.01
    assert result.rotation is not None
    assert result.translation_m is not None
    assert np.all(np.isfinite(result.rotation))
    assert np.all(np.isfinite(result.translation_m))


def test_camera_compensation_reports_insufficient_references() -> None:
    points = _reference_points()[:2]

    result = estimate_camera_compensation(
        points,
        points,
        point_ids=("R1", "R2"),
        weights=None,
        config=MatcherConfig(),
    )

    assert result.status == "insufficient_references"
    assert result.rotation is None
    assert result.translation_m is None


def test_camera_compensation_rejects_collinear_references() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0]])

    result = estimate_camera_compensation(
        points,
        points,
        point_ids=("R1", "R2", "R3"),
        weights=None,
        config=MatcherConfig(),
    )

    assert result.status == "degenerate_references"


def test_pipeline_frame_postprocess_compensates_measurement_point() -> None:
    reference = _reference_points()[:4]
    measurement_reference = np.array([0.2, -0.15, 2.5])
    rotation = _rotation_z(2.0)
    translation = np.array([0.01, -0.006, 0.02])
    current_reference = (rotation.T @ (reference - translation).T).T
    current_measurement = rotation.T @ (measurement_reference - translation)
    pipeline = TemporalStereoPipeline(
        "research_full",
        q=np.eye(4),
        calibration_unit="m",
        config=MatcherConfig(camera_compensation_max_rmse_mm=1.0),
    )
    results: list[FramePointResult] = []
    for index, point_id in enumerate(("R1", "R2", "R3", "R4")):
        state = PointState(point_id, (0.0, 0.0), (0.0, 0.0), point_role="reference")
        state.reference_estimated_xyz = tuple(reference[index])
        state.last_disparity_variance_px2 = 0.01
        state.confidence = 0.9
        pipeline.states[point_id] = state
        results.append(
            FramePointResult(
                "research_full", 1, point_id, "valid",
                estimated_x_m=float(current_reference[index, 0]),
                estimated_y_m=float(current_reference[index, 1]),
                estimated_z_m=float(current_reference[index, 2]),
                point_role="reference",
            )
        )
    measurement = PointState("P1", (0.0, 0.0), (0.0, 0.0), point_role="measurement")
    measurement.reference_estimated_xyz = tuple(measurement_reference)
    pipeline.states["P1"] = measurement
    results.append(
        FramePointResult(
            "research_full", 1, "P1", "valid",
            estimated_x_m=float(current_measurement[0]),
            estimated_y_m=float(current_measurement[1]),
            estimated_z_m=float(current_measurement[2]),
            point_role="measurement",
        )
    )

    compensated = pipeline._apply_camera_compensation(results)
    point = next(result for result in compensated if result.point_id == "P1")

    assert point.compensation_applied is True
    assert point.compensation_status == "valid"
    assert np.allclose(
        [point.final_x_m, point.final_y_m, point.final_z_m],
        measurement_reference,
        atol=1e-9,
    )
    assert np.allclose(
        [point.delta_x_mm, point.delta_y_mm, point.delta_z_mm],
        np.zeros(3),
        atol=1e-6,
    )
    assert measurement.reference_compensated_xyz == tuple(measurement_reference)
