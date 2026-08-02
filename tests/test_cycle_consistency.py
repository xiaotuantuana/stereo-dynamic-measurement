from __future__ import annotations

import cv2
import numpy as np

import stereo_research.pipeline as pipeline_module
from stereo_research.cycle_consistency import evaluate_cycle_consistency
from stereo_research.local_matching import LocalMatcher
from stereo_research.models import MatcherConfig, PointState, method_profile
from stereo_research.models import PointSpec
from stereo_research.pipeline import TemporalStereoPipeline
from stereo_research.tracking import FlowResult, track_xy_lk


def _texture() -> np.ndarray:
    rng = np.random.default_rng(51)
    image = rng.integers(0, 256, (120, 200), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0.5)


def test_generic_lk_tracks_a_right_image_point() -> None:
    previous = _texture()
    matrix = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.25]], dtype=np.float32)
    current = cv2.warpAffine(
        previous,
        matrix,
        (previous.shape[1], previous.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    result = track_xy_lk(
        previous,
        current,
        previous_xy=(120.0, 65.0),
        initial_velocity=(2.0, -1.0),
        config=MatcherConfig(),
        fb_threshold=1.0,
    )

    assert result.status == "valid"
    assert result.left_xy is not None
    assert abs(result.left_xy[0] - 122.5) < 0.15
    assert abs(result.left_xy[1] - 63.75) < 0.15


def test_correct_four_view_cycle_has_near_zero_error() -> None:
    result = evaluate_cycle_consistency(
        stereo_right_xy=(110.1, 60.0),
        temporal_right_xy=(110.0, 60.0),
        config=MatcherConfig(),
    )

    assert result.status == "cycle_valid"
    assert result.error_px < 0.11
    assert result.cost < 0.1


def test_large_four_view_error_is_rejected() -> None:
    result = evaluate_cycle_consistency(
        stereo_right_xy=(114.0, 60.0),
        temporal_right_xy=(110.0, 60.0),
        config=MatcherConfig(),
    )

    assert result.status == "cycle_failed"
    assert result.should_recover is False


def test_medium_error_can_be_marked_cycle_recovered() -> None:
    result = evaluate_cycle_consistency(
        stereo_right_xy=(112.0, 60.0),
        temporal_right_xy=(110.0, 60.0),
        recovered_right_xy=(110.4, 60.0),
        config=MatcherConfig(),
    )

    assert result.status == "cycle_recovered"
    assert result.error_px < 0.5


def test_soft_cycle_error_reduces_confidence_without_rejection() -> None:
    result = evaluate_cycle_consistency(
        stereo_right_xy=(110.9, 60.0),
        temporal_right_xy=(110.0, 60.0),
        config=MatcherConfig(),
    )

    assert result.status == "cycle_soft"
    assert 0.0 < result.confidence_scale < 1.0


def test_local_matcher_exports_cycle_raw_quality_values() -> None:
    left = _texture()
    transform = np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    right = cv2.warpAffine(
        left,
        transform,
        (left.shape[1], left.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    matcher = LocalMatcher(MatcherConfig(uniqueness_margin=0.005, icgn_max_residual=0.5))

    result = matcher.match(
        left,
        right,
        left_xy=(120.0, 65.0),
        predicted_disparity=8.0,
        profile=method_profile("research_full"),
        latest_disparity=8.0,
        temporal_right_xy=(112.0, 65.0),
        right_flow_fb_error_px=0.2,
    )

    assert result.status == "valid"
    assert result.temporal_right_xy == (112.0, 65.0)
    assert result.right_flow_fb_error_px == 0.2
    assert result.cycle_error_px is not None and result.cycle_error_px < 0.1
    assert result.cycle_cost is not None and result.cycle_cost < 0.1


def test_research_pipeline_tracks_both_right_frames_and_reports_cycle() -> None:
    left0 = _texture()
    right0 = cv2.warpAffine(
        left0,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (left0.shape[1], left0.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    motion = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    left1 = cv2.warpAffine(
        left0,
        motion,
        (left0.shape[1], left0.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    right1 = cv2.warpAffine(
        right0,
        motion,
        (right0.shape[1], right0.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    q = np.array(
        [[1, 0, 0, -100], [0, 1, 0, -60], [0, 0, 0, 100], [0, 0, 10, 0]],
        dtype=np.float64,
    )
    pipeline = TemporalStereoPipeline(
        "research_full",
        q=q,
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005, icgn_max_residual=0.5),
    )
    initial = pipeline.initialize(left0, right0, (PointSpec("P1", (120.0, 65.0)),), 0)[0]

    current = pipeline.step(left1, right1, 1)[0]

    assert initial.status == "valid"
    assert current.status == "valid"
    assert current.temporal_right_x is not None
    assert abs(current.temporal_right_x - 114.0) < 0.2
    assert current.right_flow_fb_error_px is not None
    assert current.cycle_error_px is not None
    assert current.cycle_error_px < 0.5
    assert current.cycle_status == "cycle_valid"


def test_research_sgbm_recapture_restores_right_flow_reference() -> None:
    left = _texture()
    right = cv2.warpAffine(
        left,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (left.shape[1], left.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    q = np.array(
        [[1, 0, 0, -100], [0, 1, 0, -60], [0, 0, 0, 100], [0, 0, 10, 0]],
        dtype=np.float64,
    )
    pipeline = TemporalStereoPipeline(
        "research_full",
        q=q,
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005, recovery_interval_frames=1),
    )
    pipeline.initialize(left, right, (PointSpec("P1", (120.0, 65.0)),), 0)
    state = pipeline.states["P1"]
    state.disparity_history.clear()
    state.estimated_disparity_history.clear()
    state.right_xy = None
    state.right_flow_reference_xy = None
    state.status = "lost"
    pipeline.right_flow_reference_gray.clear()

    recovered = pipeline.step(left, right, 1)[0]

    assert recovered.status == "valid"
    assert "P1" in pipeline.right_flow_reference_gray
    assert state.right_flow_reference_xy is not None
    assert state.right_flow_reference_frame == 1


def test_right_reference_image_and_coordinate_are_synchronized() -> None:
    q = np.eye(4, dtype=np.float64)
    pipeline = TemporalStereoPipeline("research_full", q, "m", MatcherConfig())
    state = PointState("P1", (30.0, 20.0), (30.0, 20.0))
    pipeline.states["P1"] = state
    reference_image = np.full((40, 60), 17, dtype=np.uint8)

    pipeline._update_right_flow_reference("P1", 3, reference_image, (21.0, 20.0))
    reference_image[:] = 99

    assert state.right_flow_reference_frame == 3
    assert state.right_flow_reference_xy == (21.0, 20.0)
    assert np.all(pipeline.right_flow_reference_gray["P1"] == 17)
    state.record_stereo_measurement((5.0, 5.0), 25.0, 1.0)
    assert state.right_flow_reference_frame == 3
    assert state.right_flow_reference_xy == (21.0, 20.0)

    image, xy, status = pipeline._right_flow_reference("P1", current_frame=4)
    assert status == "valid"
    assert image is not None and xy == (21.0, 20.0)
    state.right_flow_reference_frame = 4
    _, _, status = pipeline._right_flow_reference("P1", current_frame=4)
    assert status == "invalid_right_reference"


def test_kalman_rejected_right_measurement_is_not_used_as_reference() -> None:
    q = np.array(
        [[1, 0, 0, -100], [0, 1, 0, -60], [0, 0, 0, 100], [0, 0, 10, 0]],
        dtype=np.float64,
    )
    pipeline = TemporalStereoPipeline("research_full", q, "m", MatcherConfig())
    state = PointState("P1", (120.0, 65.0), (120.0, 65.0))
    pipeline.states["P1"] = state
    right_gray = np.zeros((120, 200), dtype=np.uint8)
    first = pipeline._valid_result(
        0, state, (110.0, 65.0), 10.0, 0.02, 0.02, 0.95, 0, 0, 0,
        texture_std=30.0, uniqueness_margin_value=0.3, cost_curvature=2.0,
    )
    pipeline._finalize_right_flow_reference(first, 0, right_gray)
    for frame in range(1, 4):
        result = pipeline._valid_result(
            frame, state, (110.0, 65.0), 10.0, 0.02, 0.02, 0.95, 0, 0, 0,
            texture_std=30.0, uniqueness_margin_value=0.3, cost_curvature=2.0,
        )
        pipeline._finalize_right_flow_reference(result, frame, right_gray)

    rejected = pipeline._valid_result(
        4, state, (95.0, 65.0), 25.0, 0.02, 0.02, 0.95, 0, 0, 0,
        texture_std=30.0, uniqueness_margin_value=0.3, cost_curvature=2.0,
    )
    finalized = pipeline._finalize_right_flow_reference(rejected, 4, right_gray)

    assert rejected.kalman_update_status == "predict_only_outlier"
    assert finalized.right_reference_source == "estimated"
    assert state.right_flow_reference_frame == 4
    assert state.right_flow_reference_xy == (
        rejected.estimated_right_x,
        rejected.estimated_right_y,
    )
    assert state.right_flow_reference_xy != (95.0, 65.0)


def test_failed_right_flow_then_stereo_success_refreshes_entire_reference(
    monkeypatch,
) -> None:
    left = _texture()
    right = cv2.warpAffine(
        left,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (left.shape[1], left.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    q = np.array(
        [[1, 0, 0, -100], [0, 1, 0, -60], [0, 0, 0, 100], [0, 0, 10, 0]],
        dtype=np.float64,
    )
    pipeline = TemporalStereoPipeline(
        "research_full",
        q,
        "m",
        MatcherConfig(uniqueness_margin=0.005),
    )
    initial = pipeline.initialize(left, right, (PointSpec("P1", (120.0, 65.0)),), 0)[0]
    assert initial.status == "valid"
    monkeypatch.setattr(
        pipeline_module,
        "track_xy_lk",
        lambda *args, **kwargs: FlowResult(status="flow_failed"),
    )

    current = pipeline.step(left, right, 1)[0]
    state = pipeline.states["P1"]

    assert current.status == "valid"
    assert current.cycle_status == "right_flow_failed"
    assert current.right_reference_source == "measured"
    assert state.right_flow_reference_frame == 1
    assert state.right_flow_reference_xy == (
        current.measured_right_x,
        current.measured_right_y,
    )
    assert np.array_equal(pipeline.right_flow_reference_gray["P1"], right)
