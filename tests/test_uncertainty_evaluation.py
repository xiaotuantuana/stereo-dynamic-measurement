from __future__ import annotations

import inspect

import numpy as np
import pytest

from stereo_research.uncertainty import estimate_match_uncertainty
from stereo_research.models import MatcherConfig
from stereo_research.uncertainty_evaluation import (
    GroundTruthRecord,
    UncertaintyEvaluationRecord,
    align_ground_truth,
    evaluate_records,
    fit_scale_calibration,
)


def _measurement(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "frame": 10, "point_id": "p0", "method": "M3",
        "estimated_disparity": 10.5, "estimated_sigma_d_px": 0.5,
        "estimated_Z_m": 2.02, "estimated_sigma_z_mm": 10.0,
        "estimated_X_m": 0.001, "estimated_Y_m": -0.002,
        "policy_refinement_level": 1, "policy_precision_retry_count": 0,
        "adaptive_search_radius": 8, "confidence_state": "HIGH",
        "match_cost": 0.1, "uniqueness_margin_value": 0.2,
        "lr_error_px": 0.05, "flow_fb_error_px": 0.1,
        "texture_std": 12.0, "precision_status": "MET",
    }
    row.update(overrides)
    return row


def test_runtime_uncertainty_api_has_no_ground_truth_parameters() -> None:
    forbidden = {"gt", "ground_truth"}
    names = {parameter.name.lower() for parameter in inspect.signature(estimate_match_uncertainty).parameters.values()}
    assert not any(any(token in name for token in forbidden) for name in names)


def test_gt_variation_does_not_change_runtime_sigma() -> None:
    config = MatcherConfig()
    kwargs = dict(texture_std=10.0, photo_cost=0.1, uniqueness_margin=0.2,
                  cost_curvature=0.4, flow_fb_error_px=0.1, right_flow_fb_error_px=0.1,
                  lr_error_px=0.1, cycle_error_px=0.1, icgn_residual=0.1,
                  icgn_hessian=10.0, config=config)
    assert estimate_match_uncertainty(**kwargs).disparity_variance_px2 == estimate_match_uncertainty(**kwargs).disparity_variance_px2


def test_evaluation_record_computes_actual_errors_outside_runtime() -> None:
    record = UncertaintyEvaluationRecord.from_measurement_and_gt(
        _measurement(), GroundTruthRecord(frame=10, point_id="p0", x_mm=0.0, y_mm=0.0, z_mm=2000.0, disparity_px=10.0)
    )
    assert record.disparity_error_px == pytest.approx(0.5)
    assert record.z_error_mm == pytest.approx(20.0)
    assert record.x_error_mm == pytest.approx(1.0)


def test_coverage_ratio_and_ranking_are_calculated_from_independent_values() -> None:
    records = [
        UncertaintyEvaluationRecord.from_measurement_and_gt(_measurement(estimated_disparity=10.5, estimated_sigma_d_px=0.5, estimated_Z_m=2.005, estimated_sigma_z_mm=5.0), GroundTruthRecord(frame=10, point_id="p0", z_mm=2000.0, disparity_px=10.0)),
        UncertaintyEvaluationRecord.from_measurement_and_gt(_measurement(frame=11, estimated_disparity=12.0, estimated_sigma_d_px=1.0, estimated_Z_m=2.020, estimated_sigma_z_mm=10.0), GroundTruthRecord(frame=11, point_id="p0", z_mm=2000.0, disparity_px=10.0)),
    ]
    stats = evaluate_records(records)
    assert stats["disparity_coverage_1sigma"] == pytest.approx(0.5)
    assert stats["disparity_coverage_2sigma"] == pytest.approx(1.0)
    assert stats["disparity_calibration_ratio"] == pytest.approx(np.sqrt((0.5**2 + 2.0**2) / 2) / np.sqrt((0.5**2 + 1.0**2) / 2))
    assert stats["disparity_spearman"] == pytest.approx(1.0)


def test_frame_and_timestamp_alignment_reject_distant_samples() -> None:
    gt = [GroundTruthRecord(frame=10, point_id="p0", z_mm=2000.0), GroundTruthRecord(timestamp_s=1.0, point_id="p0", z_mm=2000.0)]
    assert align_ground_truth(_measurement(), gt, alignment_mode="frame") is not None
    assert align_ground_truth(_measurement(timestamp_s=1.015), gt, alignment_mode="timestamp", max_time_offset_ms=20) is not None
    assert align_ground_truth(_measurement(timestamp_s=1.100), gt, alignment_mode="timestamp", max_time_offset_ms=20) is None


def test_scale_calibration_is_explicit_and_uses_calibration_records_only() -> None:
    records = [
        UncertaintyEvaluationRecord.from_measurement_and_gt(_measurement(estimated_disparity=11.0, estimated_sigma_d_px=0.5, estimated_Z_m=2.020, estimated_sigma_z_mm=10.0), GroundTruthRecord(frame=10, point_id="p0", z_mm=2000.0, disparity_px=10.0)),
    ]
    model = fit_scale_calibration(records, dataset_id="sequence_A")
    assert model.scale_factor_d == pytest.approx(2.0)
    assert model.scale_factor_z == pytest.approx(2.0)
    assert model.calibrate_d(0.5) == pytest.approx(1.0)
