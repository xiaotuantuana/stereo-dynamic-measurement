from __future__ import annotations

import cv2
import numpy as np

from stereo_research.models import FramePointResult, MatcherConfig, PointSpec
from stereo_research.pipeline import TemporalStereoPipeline
from stereo_research.runner import ACCURACY_POLICY_CSV_FIELDS, CSV_FIELDS, SHADOW_CSV_FIELDS


def _stereo_sequence() -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[PointSpec, ...]]:
    rng = np.random.default_rng(20260820)
    base = cv2.GaussianBlur(
        rng.integers(0, 256, (160, 280), dtype=np.uint8), (3, 3), 0.5
    )
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for dx, dy in ((0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5)):
        motion = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        left = cv2.warpAffine(
            base, motion, (280, 160), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        right = cv2.warpAffine(
            left,
            np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (280, 160),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        frames.append((left, right))
    points = (
        PointSpec("R1", (90.0, 50.0), role="reference"),
        PointSpec("R2", (190.0, 50.0), role="reference"),
        PointSpec("R3", (90.0, 115.0), role="reference"),
        PointSpec("R4", (190.0, 115.0), role="reference"),
        PointSpec("P1", (140.0, 80.0)),
    )
    return frames, points


def _q() -> np.ndarray:
    return np.array(
        [[1.0, 0.0, 0.0, -140.0], [0.0, 1.0, 0.0, -80.0],
         [0.0, 0.0, 0.0, 100.0], [0.0, 0.0, 10.0, 0.0]],
        dtype=float,
    )


def _run(enable_shadow: bool):
    images, points = _stereo_sequence()
    pipeline = TemporalStereoPipeline(
        "research_full",
        _q(),
        "m",
        MatcherConfig(
            uniqueness_margin=0.005,
            enable_physics_shadow=enable_shadow,
            enable_fault_shadow=enable_shadow,
        ),
    )
    results = [pipeline.initialize(*images[0], points, frame=0)]
    for frame, pair in enumerate(images[1:], start=1):
        results.append(pipeline.step(*pair, frame=frame))
    return pipeline, results


def _assert_value_equal(left, right, path: str) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True,
                                   err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_value_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (l_value, r_value) in enumerate(zip(left, right, strict=True)):
            _assert_value_equal(l_value, r_value, f"{path}[{index}]")
    elif hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        _assert_value_equal(vars(left), vars(right), path)
    elif isinstance(left, float) or isinstance(right, float):
        assert np.isclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True), path
    else:
        assert left == right, path


def test_shadow_on_off_full_sequence_preserves_every_legacy_result_and_pipeline_state() -> None:
    off_pipeline, off_frames = _run(False)
    on_pipeline, on_frames = _run(True)
    first_round_end = len(CSV_FIELDS) - len(ACCURACY_POLICY_CSV_FIELDS)
    legacy_fields = CSV_FIELDS[:first_round_end - len(SHADOW_CSV_FIELDS)]
    timing_fields = {"flow_ms", "matching_ms", "total_ms"}

    assert len(off_frames) == len(on_frames)
    for frame_index, (off_results, on_results) in enumerate(
        zip(off_frames, on_frames, strict=True)
    ):
        assert [item.point_id for item in off_results] == [item.point_id for item in on_results]
        for off, on in zip(off_results, on_results, strict=True):
            off_row, on_row = off.as_csv_row(), on.as_csv_row()
            for name in legacy_fields:
                if name not in off_row or name not in on_row:
                    # Runner-level sequence metadata (for example ``repeat``)
                    # is not produced by FramePointResult and is identical input
                    # for the paired runs.
                    assert name not in off_row and name not in on_row
                    continue
                if name in timing_fields:
                    # Wall-clock telemetry is nondeterministic, but Shadow runs only
                    # after these legacy timers are finalized; validate its domain.
                    assert float(off_row[name]) >= 0.0
                    assert float(on_row[name]) >= 0.0
                    continue
                _assert_value_equal(
                    off_row[name], on_row[name],
                    f"frame={frame_index}, point={off.point_id}, field={name}",
                )

            # Explicitly name the thesis-critical stages in addition to the full
            # legacy CSV comparison above.
            for name in (
                "measured_x_m", "measured_y_m", "measured_z_m",
                "estimated_x_m", "estimated_y_m", "estimated_z_m",
                "compensated_x_m", "compensated_y_m", "compensated_z_m",
                "disparity", "predicted_disparity", "used_search_radius",
                "status", "confidence", "recovery_stage", "final_x_m",
                "final_y_m", "final_z_m",
            ):
                _assert_value_equal(getattr(off, name), getattr(on, name), name)

    # ShadowAnalyzer owns a separate copied history.  Core mutable pipeline state,
    # Kalman filters, flow references, and reacquisition bookkeeping stay identical.
    _assert_value_equal(off_pipeline.states, on_pipeline.states, "states")
    assert off_pipeline.filters.keys() == on_pipeline.filters.keys()
    for point_id in off_pipeline.filters:
        off_filter = off_pipeline.filters[point_id]
        on_filter = on_pipeline.filters[point_id]
        _assert_value_equal(off_filter.state, on_filter.state, f"filters.{point_id}.state")
        _assert_value_equal(
            off_filter.covariance,
            on_filter.covariance,
            f"filters.{point_id}.covariance",
        )
        assert off_filter.predict_only_frames == on_filter.predict_only_frames
    _assert_value_equal(
        off_pipeline.flow_reference_gray,
        on_pipeline.flow_reference_gray,
        "flow_reference_gray",
    )
    _assert_value_equal(
        off_pipeline.right_flow_reference_gray,
        on_pipeline.right_flow_reference_gray,
        "right_flow_reference_gray",
    )
    assert off_pipeline.shadow_analyzer is None
    assert on_pipeline.shadow_analyzer is not None
    assert on_pipeline.shadow_analyzer._history_mm
