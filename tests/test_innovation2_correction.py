from __future__ import annotations

import numpy as np

from stereo_dynamic_measurement.innovation2.physics_confidence import EvidenceTerm
from stereo_dynamic_measurement.innovation2.transient_gate import decide_transient
from stereo_dynamic_measurement.innovation2.trajectory_corrector import CorrectionConfig, correct_trajectory_point


def test_low_physics_confidence_creates_separate_corrected_xyz_with_audit_metadata() -> None:
    raw = np.array([0.0, 80.0, 2000.0])
    predicted = np.array([0.0, 5.0, 2000.0])
    neighbors = np.array([[0.0, 3.0, 2000.0], [0.0, 6.0, 2000.0], [0.0, 4.0, 2000.0]])

    decision = decide_transient(
        temporal_residual_mm=75.0,
        flow_3d=EvidenceTerm(0.1, True),
        spatial=EvidenceTerm(0.1, True),
        visual_consistency=EvidenceTerm(0.2, True),
        synchronous_motion_ratio=0.0,
    )
    result = correct_trajectory_point(
        raw,
        predicted,
        neighbors,
        c_phy=0.15,
        transient_decision=decision,
        config=CorrectionConfig(threshold=0.55, max_correction_mm=25.0),
    )

    assert result.correction_applied_candidate
    assert result.correction_method in {"history_neighbor_weighted", "history_prediction"}
    assert np.array_equal(result.raw_xyz_mm, raw)
    assert result.candidate_corrected_xyz_mm[1] < raw[1]
    assert result.correction_delta_mm == 25.0


def test_high_physics_confidence_preserves_raw_trajectory() -> None:
    raw = np.array([1.0, 2.0, 2000.0])

    result = correct_trajectory_point(raw, np.array([0.0, 3.0, 2000.0]), np.empty((0, 3)), c_phy=0.9, config=CorrectionConfig())

    assert not result.correction_applied
    assert result.correction_method == "none"
    assert np.array_equal(result.corrected_xyz_mm, raw)
