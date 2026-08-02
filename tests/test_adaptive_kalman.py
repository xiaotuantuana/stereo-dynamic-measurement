from __future__ import annotations

import numpy as np

from stereo_research.filtering import AdaptivePointKalman
from stereo_research.models import MatcherConfig
from stereo_research.models import PointState
from stereo_research.pipeline import TemporalStereoPipeline
from stereo_research.uncertainty import estimate_match_uncertainty


def _run_disparity_series(measurements: np.ndarray, variance: float = 0.04) -> np.ndarray:
    filter_ = AdaptivePointKalman(MatcherConfig())
    filter_.initialize((100.0, 50.0), float(measurements[0]))
    estimates = [float(measurements[0])]
    for value in measurements[1:]:
        filter_.predict()
        update = filter_.update((100.0, 50.0), float(value), 0.01, variance)
        estimates.append(update.estimated_disparity)
    return np.asarray(estimates)


def test_static_point_noise_is_reduced() -> None:
    rng = np.random.default_rng(3)
    measured = 8.0 + rng.normal(0.0, 0.18, 180)

    estimated = _run_disparity_series(measured)

    assert np.std(estimated[30:]) < 0.65 * np.std(measured[30:])


def test_constant_velocity_has_small_steady_state_lag() -> None:
    measured = 5.0 + 0.04 * np.arange(160, dtype=np.float64)

    estimated = _run_disparity_series(measured, variance=0.01)

    assert abs(estimated[-1] - measured[-1]) < 0.05


def test_sinusoidal_amplitude_is_preserved() -> None:
    frames = np.arange(300, dtype=np.float64)
    measured = 8.0 + 0.6 * np.sin(2.0 * np.pi * frames / 80.0)

    estimated = _run_disparity_series(measured, variance=0.01)
    measured_amplitude = 0.5 * np.ptp(measured[80:])
    estimated_amplitude = 0.5 * np.ptp(estimated[80:])

    assert estimated_amplitude / measured_amplitude > 0.95


def test_single_outlier_is_rejected_by_nis() -> None:
    config = MatcherConfig()
    filter_ = AdaptivePointKalman(config)
    filter_.initialize((100.0, 50.0), 8.0)
    for _ in range(10):
        filter_.predict()
        filter_.update((100.0, 50.0), 8.0, 0.01, 0.01)

    filter_.predict()
    update = filter_.update((100.0, 50.0), 20.0, 0.01, 0.01)

    assert update.status == "predict_only_outlier"
    assert abs(update.estimated_disparity - 8.0) < 0.2


def test_measurement_variance_controls_disparity_gain() -> None:
    low = AdaptivePointKalman(MatcherConfig())
    high = AdaptivePointKalman(MatcherConfig())
    low.initialize((100.0, 50.0), 8.0)
    high.initialize((100.0, 50.0), 8.0)
    low.predict()
    high.predict()

    low_update = low.update((100.0, 50.0), 8.2, 0.01, 0.005)
    high_update = high.update((100.0, 50.0), 8.2, 1.0, 2.0)

    assert low_update.kalman_gain_disparity > high_update.kalman_gain_disparity


def test_match_uncertainty_increases_for_poor_quality() -> None:
    config = MatcherConfig()
    good = estimate_match_uncertainty(
        texture_std=35.0,
        photo_cost=0.04,
        uniqueness_margin=0.3,
        cost_curvature=2.0,
        flow_fb_error_px=0.05,
        right_flow_fb_error_px=0.05,
        lr_error_px=0.05,
        cycle_error_px=0.05,
        icgn_residual=0.03,
        icgn_hessian=100.0,
        config=config,
    )
    poor = estimate_match_uncertainty(
        texture_std=5.0,
        photo_cost=0.4,
        uniqueness_margin=0.02,
        cost_curvature=0.02,
        flow_fb_error_px=0.9,
        right_flow_fb_error_px=0.9,
        lr_error_px=0.9,
        cycle_error_px=1.4,
        icgn_residual=0.3,
        icgn_hessian=0.01,
        config=config,
    )

    assert poor.disparity_variance_px2 > good.disparity_variance_px2
    assert poor.left_position_variance_px2 > good.left_position_variance_px2
    assert poor.quality_score < good.quality_score
    assert set(good.components) >= {"texture", "photo", "margin", "flow", "lr", "cycle", "icgn"}


def test_research_pipeline_keeps_measurement_and_adaptive_estimate_separate() -> None:
    q = np.array(
        [[1, 0, 0, -100], [0, 1, 0, -60], [0, 0, 0, 100], [0, 0, 10, 0]],
        dtype=np.float64,
    )
    pipeline = TemporalStereoPipeline(
        "research_full",
        q=q,
        calibration_unit="m",
        config=MatcherConfig(),
    )
    state = PointState("P1", (120.0, 65.0), (120.0, 65.0))
    pipeline.states["P1"] = state
    first = pipeline._valid_result(
        0, state, (112.0, 65.0), 8.0, 0.05, 0.05, 0.95, 0.0, 1.0, 1.0,
        texture_std=30.0,
        uniqueness_margin_value=0.3,
        cycle_error_px=0.05,
        icgn_residual=0.03,
        icgn_hessian=100.0,
    )
    state.left_xy = (120.2, 65.0)
    second = pipeline._valid_result(
        1, state, (111.7, 65.0), 8.5, 0.35, 0.8, 0.4, 0.8, 1.0, 1.8,
        texture_std=6.0,
        uniqueness_margin_value=0.02,
        cycle_error_px=1.2,
        right_flow_fb_error_px=0.8,
        icgn_residual=0.3,
        icgn_hessian=0.1,
    )

    assert first.kalman_update_status == "initialized"
    assert second.measured_disparity == 8.5
    assert second.estimated_disparity is not None
    assert second.estimated_disparity != second.measured_disparity
    assert second.disparity_variance_px2 is not None
    assert second.left_variance_px2 is not None
    assert second.measurement_quality_score is not None
    assert second.kalman_gain_disparity is not None
    assert second.kalman_nis is not None
    assert second.measured_z_m != second.estimated_z_m


def test_repeated_predict_only_updates_put_point_into_recovery() -> None:
    q = np.array(
        [[1, 0, 0, -100], [0, 1, 0, -60], [0, 0, 0, 100], [0, 0, 10, 0]],
        dtype=np.float64,
    )
    pipeline = TemporalStereoPipeline(
        "research_full",
        q=q,
        calibration_unit="m",
        config=MatcherConfig(kalman_max_predict_only_frames=2),
    )
    state = PointState("P1", (120.0, 65.0), (120.0, 65.0))
    pipeline.states["P1"] = state
    pipeline._valid_result(0, state, (112.0, 65.0), 8.0, 0.05, 0.05, 0.9, 0, 0, 0)
    pipeline._valid_result(1, state, (100.0, 65.0), 20.0, 0.05, 0.05, 0.9, 0, 0, 0)
    last = pipeline._valid_result(2, state, (100.0, 65.0), 20.0, 0.05, 0.05, 0.9, 0, 0, 0)

    assert last.kalman_update_status == "predict_only_outlier"
    assert state.status == "recovering"
