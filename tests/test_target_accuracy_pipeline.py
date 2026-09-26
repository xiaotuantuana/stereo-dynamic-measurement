from __future__ import annotations

import cv2
import numpy as np

from stereo_research.models import MatcherConfig, PointSpec
from stereo_research.pipeline import TemporalStereoPipeline


def _images() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(818)
    base = cv2.GaussianBlur(rng.integers(0, 256, (140, 240), dtype=np.uint8), (3, 3), 0.5)
    output = []
    for dx in (0.0, 1.0, 2.0):
        left = cv2.warpAffine(
            base, np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], np.float32),
            (240, 140), borderMode=cv2.BORDER_REFLECT101,
        )
        right = cv2.warpAffine(
            left, np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], np.float32),
            (240, 140), borderMode=cv2.BORDER_REFLECT101,
        )
        output.append((left, right))
    return output


def _q() -> np.ndarray:
    return np.array(
        [[1, 0, 0, -120], [0, 1, 0, -70], [0, 0, 0, 100], [0, 0, 10, 0]],
        dtype=float,
    )


def _run(target_z_mm: float, max_precision_retry: int = 1):
    images = _images()
    pipeline = TemporalStereoPipeline(
        "research_full", _q(), "m",
        MatcherConfig(
            uniqueness_margin=0.005,
            enable_target_accuracy_policy=True,
            target_sigma_z_mm=target_z_mm,
            max_precision_retry=max_precision_retry,
            enable_camera_compensation=False,
        ),
    )
    first = pipeline.initialize(images[0][0], images[0][1], (PointSpec("P1", (150.0, 70.0)),), frame=0)[0]
    second = pipeline.step(images[1][0], images[1][1], frame=1)[0]
    return pipeline, first, second


def test_pipeline_uses_legacy_initialization_as_accuracy_policy_warmup() -> None:
    _, first, _ = _run(20.0)

    assert first.status == "valid"
    assert first.accuracy_policy_enabled
    assert first.precision_policy_warmup
    assert first.precision_status == "WARMUP"
    assert first.policy_precision_retry_count == 0


def test_different_targets_change_real_refinement_policy_not_motion_radius() -> None:
    _, _, loose = _run(20.0)
    _, _, strict = _run(10.0)

    assert loose.status == strict.status == "valid"
    assert loose.required_sigma_d_px > strict.required_sigma_d_px
    assert loose.policy_refinement_level < strict.policy_refinement_level
    assert loose.policy_base_search_radius_px == strict.policy_base_search_radius_px
    assert loose.policy_final_search_radius_px == strict.policy_final_search_radius_px
    assert loose.precision_status == "MET"
    assert strict.precision_status == "VALID_BUT_PRECISION_UNMET"
    assert strict.policy_precision_retry_count == 1
    assert strict.policy_refinement_level >= 1
    assert strict.policy_acceptance_reason == "valid_stereo_precision_unmet_retry_exhausted"
    assert strict.final_z_m is not None


def test_impossible_target_keeps_valid_stereo_measurement_and_skips_retry() -> None:
    _, _, result = _run(0.001)

    assert result.status == "valid"
    assert result.precision_status == "INFEASIBLE"
    assert result.precision_feasible is False
    assert result.policy_retry_budget == 0
    assert result.policy_precision_retry_count == 0
    assert result.measured_z_m is not None and result.final_z_m is not None


def test_pipeline_honors_configured_retry_bound_and_then_stops() -> None:
    _, _, result = _run(10.0, max_precision_retry=2)

    assert result.status == "valid"
    assert result.policy_precision_retry_count == 2
    assert result.precision_status == "VALID_BUT_PRECISION_UNMET"
    assert result.policy_retry_budget == 2


def test_feature_off_leaves_policy_fields_empty_and_shadow_safety_active() -> None:
    images = _images()
    pipeline = TemporalStereoPipeline(
        "research_full", _q(), "m",
        MatcherConfig(uniqueness_margin=0.005, enable_target_accuracy_policy=False),
    )
    pipeline.initialize(images[0][0], images[0][1], (PointSpec("P1", (150.0, 70.0)),), frame=0)
    result = pipeline.step(images[1][0], images[1][1], frame=1)[0]

    assert not result.accuracy_policy_enabled
    assert result.precision_policy_warmup is None
    assert result.precision_status == ""
    assert result.shadow_input_stage in {"estimated", "compensated"}


def test_feature_off_complete_sequence_matches_first_round_default_results_and_state() -> None:
    images = _images()

    def execute(config: MatcherConfig):
        pipeline = TemporalStereoPipeline("research_full", _q(), "m", config)
        frames = [pipeline.initialize(
            images[0][0], images[0][1], (PointSpec("P1", (150.0, 70.0)),), frame=0
        )]
        frames.append(pipeline.step(images[1][0], images[1][1], frame=1))
        frames.append(pipeline.step(images[2][0], images[2][1], frame=2))
        return pipeline, frames

    baseline, baseline_frames = execute(MatcherConfig(uniqueness_margin=0.005))
    explicit_off, off_frames = execute(MatcherConfig(
        uniqueness_margin=0.005, enable_target_accuracy_policy=False
    ))
    timing = {"flow_ms", "matching_ms", "total_ms"}

    for baseline_results, off_results in zip(baseline_frames, off_frames, strict=True):
        for baseline_result, off_result in zip(baseline_results, off_results, strict=True):
            baseline_row, off_row = baseline_result.as_csv_row(), off_result.as_csv_row()
            assert baseline_row.keys() == off_row.keys()
            for key in baseline_row:
                if key not in timing:
                    assert baseline_row[key] == off_row[key], key
    assert vars(baseline.states["P1"]) == vars(explicit_off.states["P1"])
    np.testing.assert_allclose(
        baseline.filters["P1"].state, explicit_off.filters["P1"].state,
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        baseline.filters["P1"].covariance, explicit_off.filters["P1"].covariance,
        rtol=0.0, atol=0.0,
    )
