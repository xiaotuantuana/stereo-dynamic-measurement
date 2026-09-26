from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import inspect

from stereo_dynamic_measurement.innovation2.physics_confidence import (
    EvidenceTerm,
    PhysicsConfidenceConfig,
    compute_physics_confidence,
)
from stereo_dynamic_measurement.innovation2.transient_gate import decide_transient
from stereo_dynamic_measurement.innovation2.trajectory_corrector import (
    CorrectionConfig,
    correct_trajectory_point,
)
from stereo_dynamic_measurement.innovation2.physics_validation import (
    GroundTruthEvaluation,
    evaluate_runtime_against_ground_truth,
    run_runtime_physics_analysis,
)


def test_missing_physics_evidence_is_unknown_not_healthy() -> None:
    config = PhysicsConfidenceConfig(weights={"temporal": 0.6, "flow_3d": 0.4})

    result = compute_physics_confidence(
        {
            "temporal": EvidenceTerm(value=0.8, valid=True, reliability=0.5),
            "flow_3d": EvidenceTerm.unavailable("independent 2D-3D projection unavailable"),
        },
        config,
    )

    assert result.valid
    assert result.c_phy == pytest.approx(0.8)
    assert result.num_valid_evidence == 1
    assert result.available_evidence_names == ("temporal",)
    assert result.missing_evidence_names == ("flow_3d",)
    assert result.effective_weights == {"temporal": pytest.approx(1.0)}
    assert result.components["flow_3d"].valid is False
    assert result.components["flow_3d"].value is None


def test_no_valid_physics_evidence_produces_no_confidence_claim() -> None:
    result = compute_physics_confidence(
        {"flow_3d": EvidenceTerm.unavailable("not measured")},
        PhysicsConfidenceConfig(weights={"flow_3d": 1.0}),
    )

    assert not result.valid
    assert result.c_phy is None
    assert result.r_phy is None
    assert result.num_valid_evidence == 0


def test_synchronous_supported_fast_motion_is_transient_protected() -> None:
    decision = decide_transient(
        temporal_residual_mm=30.0,
        flow_3d=EvidenceTerm(0.9, True),
        spatial=EvidenceTerm(0.85, True),
        visual_consistency=EvidenceTerm(0.9, True),
        synchronous_motion_ratio=0.9,
    )
    raw = np.array([0.0, 30.0, 2000.0])
    result = correct_trajectory_point(
        raw,
        np.array([0.0, 2.0, 2000.0]),
        np.array([[0.0, 29.0, 2000.0], [0.0, 31.0, 2000.0]]),
        c_phy=0.2,
        transient_decision=decision,
        config=CorrectionConfig(max_correction_mm=20.0),
    )

    assert decision.is_possible_real_transient
    assert not decision.allow_correction
    assert result.transient_protected
    assert not result.correction_applied_candidate
    assert np.array_equal(result.candidate_corrected_xyz_mm, raw)


def test_isolated_multisource_outlier_allows_bounded_candidate_correction() -> None:
    decision = decide_transient(
        temporal_residual_mm=75.0,
        flow_3d=EvidenceTerm(0.1, True),
        spatial=EvidenceTerm(0.1, True),
        visual_consistency=EvidenceTerm(0.2, True),
        synchronous_motion_ratio=0.0,
    )
    raw = np.array([0.0, 80.0, 2000.0])
    prediction = np.array([0.0, 5.0, 2000.0])
    neighbors = np.array([[0.0, 3.0, 2000.0], [0.0, 6.0, 2000.0]])
    result = correct_trajectory_point(
        raw,
        prediction,
        neighbors,
        c_phy=0.15,
        transient_decision=decision,
        config=CorrectionConfig(max_correction_mm=25.0),
    )

    assert decision.possible_measurement_error
    assert decision.allow_correction
    assert result.correction_applied_candidate
    assert result.correction_delta_mm == pytest.approx(25.0)
    assert np.linalg.norm(result.candidate_corrected_xyz_mm - prediction) < np.linalg.norm(raw - prediction)


def test_ground_truth_changes_evaluation_only_not_runtime_physics() -> None:
    rows = []
    for frame in range(24):
        for index, point_id in enumerate(("P1", "P2", "P3")):
            rows.append({
                "frame": frame,
                "timestamp_s": frame / 20.0,
                "point_id": point_id,
                "raw_X_mm": index * 100.0,
                "raw_Y_mm": float(np.sin(frame / 3.0) * (index + 1)),
                "raw_Z_mm": 2000.0,
            })
    raw = pd.DataFrame(rows)
    assert all("gt" not in name.lower() and "truth" not in name.lower() for name in inspect.signature(run_runtime_physics_analysis).parameters)

    runtime = run_runtime_physics_analysis(raw, fs_hz=20.0, threshold=0.65)
    trajectories_before = runtime.trajectories.copy(deep=True)
    confidence_before = runtime.confidence.copy(deep=True)
    gt_a = GroundTruthEvaluation(xyz_mm=raw[["frame", "point_id"]].assign(X_gt_mm=0.0, Y_gt_mm=0.0, Z_gt_mm=2000.0), phase_rad={"P2": 0.0}, frequency_hz=2.0, injected_fault="none")
    gt_b = GroundTruthEvaluation(xyz_mm=raw[["frame", "point_id"]].assign(X_gt_mm=50.0, Y_gt_mm=50.0, Z_gt_mm=2050.0), phase_rad={"P2": 1.2}, frequency_hz=8.0, injected_fault="different")

    evaluation_a = evaluate_runtime_against_ground_truth(runtime, gt_a)
    evaluation_b = evaluate_runtime_against_ground_truth(runtime, gt_b)

    assert_frame_equal(runtime.trajectories, trajectories_before)
    assert_frame_equal(runtime.confidence, confidence_before)
    assert evaluation_a != evaluation_b
