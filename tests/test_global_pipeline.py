from __future__ import annotations

import cv2
import numpy as np

from stereo_research.global_matching import GlobalStereoMatcher
from stereo_research.models import MatcherConfig, PointSpec, PointState
from stereo_research.pipeline import TemporalStereoPipeline


def _shift_right_image(left: np.ndarray, disparity: float) -> np.ndarray:
    transform = np.array([[1.0, 0.0, -disparity], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(
        left,
        transform,
        (left.shape[1], left.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _textured_image(width: int = 240, height: int = 140) -> np.ndarray:
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0.5)


def _synthetic_q(focal: float = 100.0, baseline_m: float = 0.1) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, -120.0],
            [0.0, 1.0, 0.0, -70.0],
            [0.0, 0.0, 0.0, focal],
            [0.0, 0.0, 1.0 / baseline_m, 0.0],
        ],
        dtype=np.float64,
    )


def test_global_sgbm_initialization_recovers_known_disparity() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 12.0)
    matcher = GlobalStereoMatcher(MatcherConfig())

    result = matcher.compute(left, right)
    sample = matcher.sample_initial(result, (150.0, 70.0))

    assert sample.status == "valid"
    assert abs(sample.disparity - 12.0) <= 0.75
    assert sample.lr_error_px is not None
    assert sample.lr_error_px <= 1.0


def test_pipeline_expands_global_initialization_beyond_64_pixels() -> None:
    left = _textured_image(width=360, height=160)
    right = _shift_right_image(left, 72.0)
    pipeline = TemporalStereoPipeline(
        method="local",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(),
    )

    result = pipeline.initialize(
        left,
        right,
        (PointSpec("P1", (250.0, 80.0)),),
        frame=0,
    )[0]

    assert result.status == "valid"
    assert abs(result.disparity - 72.0) <= 1.0


def test_pipeline_local_flow_tracks_left_point_and_keeps_depth() -> None:
    left0 = _textured_image()
    right0 = _shift_right_image(left0, 8.0)
    transform = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    left1 = cv2.warpAffine(
        left0,
        transform,
        (left0.shape[1], left0.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    right1 = _shift_right_image(left1, 8.0)
    pipeline = TemporalStereoPipeline(
        method="local_flow",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005),
    )

    initial = pipeline.initialize(
        left0,
        right0,
        (PointSpec("P1", (150.0, 70.0)),),
        frame=0,
    )
    current = pipeline.step(left1, right1, frame=1)

    assert initial[0].status == "valid"
    assert current[0].status == "valid"
    assert abs(current[0].left_x - 152.0) <= 0.2
    assert abs(current[0].left_y - 71.0) <= 0.2
    assert abs(current[0].disparity - 8.0) <= 0.2
    assert abs(current[0].z_m - 1.25) <= 0.05


def test_pipeline_marks_point_lost_after_three_consecutive_flow_failures() -> None:
    left0 = _textured_image()
    right0 = _shift_right_image(left0, 8.0)
    flat = np.full_like(left0, 100)
    pipeline = TemporalStereoPipeline(
        method="full",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005, max_failures=3),
    )
    pipeline.initialize(left0, right0, (PointSpec("P1", (150.0, 70.0)),), frame=0)

    statuses = [
        pipeline.step(flat, flat, frame=frame)[0].status
        for frame in (1, 2, 3)
    ]

    assert statuses[-1] == "lost"


def test_pipeline_recovers_after_one_flow_failure_from_last_valid_reference() -> None:
    left0 = _textured_image()
    right0 = _shift_right_image(left0, 8.0)
    flat = np.full_like(left0, 100)
    transform = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    left2 = cv2.warpAffine(
        left0,
        transform,
        (left0.shape[1], left0.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    right2 = _shift_right_image(left2, 8.0)
    pipeline = TemporalStereoPipeline(
        method="local_flow",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005, max_failures=3),
    )
    pipeline.initialize(left0, right0, (PointSpec("P1", (150.0, 70.0)),), frame=0)

    failed = pipeline.step(flat, flat, frame=1)[0]
    recovered = pipeline.step(left2, right2, frame=2)[0]

    assert failed.status == "flow_failed"
    assert recovered.status == "valid"
    assert abs(recovered.left_x - 154.0) <= 0.3
    assert abs(recovered.left_y - 72.0) <= 0.3


def test_pipeline_initialization_rejects_low_texture_patch() -> None:
    left = _textured_image()
    left[65:76, 145:156] = 128
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="full",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005),
    )

    result = pipeline.initialize(
        left,
        right,
        (PointSpec("P1", (150.0, 70.0)),),
        frame=0,
    )[0]

    assert result.status == "low_texture"


def test_pipeline_initialization_rejects_point_without_complete_patch() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="full",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005),
    )

    result = pipeline.initialize(
        left,
        right,
        (PointSpec("P1", (150.0, 2.0)),),
        frame=0,
    )[0]

    assert result.status == "out_of_bounds"


def test_pipeline_rejects_finite_point_behind_camera() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="full",
        q=_synthetic_q(baseline_m=-0.1),
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005),
    )

    result = pipeline.initialize(
        left,
        right,
        (PointSpec("P1", (150.0, 70.0)),),
        frame=0,
    )[0]

    assert result.status == "depth_out_of_range"


def test_pipeline_applies_configured_depth_range() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="full",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(
            uniqueness_margin=0.005,
            min_depth_m=0.2,
            max_depth_m=1.0,
        ),
    )

    result = pipeline.initialize(
        left,
        right,
        (PointSpec("P1", (150.0, 70.0)),),
        frame=0,
    )[0]

    assert result.status == "depth_out_of_range"
    state = pipeline.states["P1"]
    assert state.disparity_history == []
    assert state.estimated_disparity_history == []
    assert state.right_xy is None
    assert state.confidence == 0.0


def test_temporal_estimation_runs_only_after_raw_measurement_is_validated() -> None:
    pipeline = TemporalStereoPipeline(
        method="full_quality",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(quality_disparity_alpha=0.2),
    )
    state = PointState("P1", (150.0, 70.0), (150.0, 70.0))
    state.disparity_history = [5.5]
    state.estimated_disparity_history = [5.5]
    pipeline.states[state.point_id] = state

    result = pipeline._valid_result(
        frame=1,
        state=state,
        right_xy=(145.0, 70.0),
        disparity=5.0,
        measured_disparity=5.0,
        estimated_disparity=5.0,
        match_cost=0.1,
        lr_error=0.1,
        confidence=0.9,
        flow_ms=0.0,
        matching_ms=1.0,
        total_ms=1.0,
        quality_stage="primary",
    )

    assert result.status == "valid"
    assert result.measured_disparity == 5.0
    assert abs(result.estimated_disparity - 5.4) < 1e-12
    assert state.disparity_history == [5.5, 5.0]
    assert state.estimated_disparity_history == [5.5, 5.4]


def test_rejected_candidate_preserves_last_valid_stereo_state() -> None:
    pipeline = TemporalStereoPipeline(
        method="full",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(max_depth_m=1.0),
    )
    state = PointState("P1", (150.0, 70.0), (150.0, 70.0))
    state.right_xy = (142.0, 70.0)
    state.disparity_history = [8.0]
    state.estimated_disparity_history = [8.0]
    state.xyz = (0.0, 0.0, 2.0)
    state.estimated_xyz = (0.0, 0.0, 2.0)
    state.confidence = 0.8
    state.status = "tracking"

    result = pipeline._valid_result(
        frame=2,
        state=state,
        right_xy=(145.0, 70.0),
        disparity=5.0,
        match_cost=0.1,
        lr_error=0.1,
        confidence=0.2,
        flow_ms=0.0,
        matching_ms=1.0,
        total_ms=1.0,
    )

    assert result.status == "depth_out_of_range"
    assert state.right_xy == (142.0, 70.0)
    assert state.disparity_history == [8.0]
    assert state.estimated_disparity_history == [8.0]
    assert state.xyz == (0.0, 0.0, 2.0)
    assert state.confidence == 0.8


def test_consecutive_depth_range_failures_eventually_mark_point_lost() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="local_flow",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(
            uniqueness_margin=0.005,
            max_failures=3,
            max_depth_m=1.0,
        ),
    )

    statuses = [
        pipeline.initialize(
            left,
            right,
            (PointSpec("P1", (150.0, 70.0)),),
            frame=0,
        )[0].status,
        pipeline.step(left, right, frame=1)[0].status,
        pipeline.step(left, right, frame=2)[0].status,
    ]

    assert statuses == ["depth_out_of_range", "ambiguous", "lost"]


def test_full_quality_pipeline_uses_context_recovery_after_primary_lr_failure() -> None:
    rng = np.random.default_rng(5)
    repeating_tile = rng.integers(0, 256, (140, 4), dtype=np.uint8)
    left = np.tile(repeating_tile, (1, 60))
    left[:, 156:158] = np.tile(np.array([[20, 240]], dtype=np.uint8), (140, 1))
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="full_quality",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(max_photo_cost=0.6, uniqueness_margin=0.05),
    )

    initial = pipeline.initialize(
        left,
        right,
        (PointSpec("P1", (150.0, 70.0)),),
        frame=0,
    )[0]
    recovered = pipeline.step(left, right, frame=1)[0]

    assert initial.status == "valid"
    assert recovered.status == "valid"
    assert abs(recovered.disparity - 8.0) <= 0.1
    assert recovered.quality_stage == "context_recovery"


def test_dense_sgbm_result_charges_shared_matching_time_to_frame_runtime() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="sgbm",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(),
    )
    pipeline.initialize(left, right, (PointSpec("P1", (150.0, 70.0)),), frame=0)

    result = pipeline.step(left, right, frame=1)[0]

    assert result.matching_ms > 0
    assert result.total_ms >= result.matching_ms


def test_lost_point_is_periodically_recaptured_from_last_valid_frame() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 8.0)
    flat = np.full_like(left, 100)
    pipeline = TemporalStereoPipeline(
        method="full_quality",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(
            uniqueness_margin=0.005,
            max_failures=2,
            recovery_interval_frames=1,
        ),
    )
    pipeline.initialize(left, right, (PointSpec("P1", (150.0, 70.0)),), frame=0)
    pipeline.step(flat, flat, frame=1)
    lost = pipeline.step(flat, flat, frame=2)[0]

    recovered = pipeline.step(left, right, frame=3)[0]

    assert lost.status == "lost"
    assert recovered.status == "valid"
    assert recovered.recovery_stage == "recaptured"
    assert recovered.recovery_attempt_count == 2
    assert recovered.recovery_success_count == 1
    assert recovered.mean_recovery_frames == 1.0
    assert pipeline.states["P1"].status == "tracking"


def test_recovery_reinitializes_missing_disparity_from_sgbm() -> None:
    left = _textured_image()
    right = _shift_right_image(left, 8.0)
    pipeline = TemporalStereoPipeline(
        method="full_quality",
        q=_synthetic_q(),
        calibration_unit="m",
        config=MatcherConfig(uniqueness_margin=0.005, recovery_interval_frames=1),
    )
    state = PointState("P1", (150.0, 70.0), (150.0, 70.0), status="lost")
    pipeline.states["P1"] = state
    pipeline.flow_reference_gray["P1"] = left.copy()
    pipeline.previous_left_gray = left.copy()
    pipeline.initialized = True

    result = pipeline.step(left, right, frame=5)[0]

    assert result.status == "valid"
    assert result.recovery_stage == "sgbm_reinitialized"
    assert state.disparity_history
