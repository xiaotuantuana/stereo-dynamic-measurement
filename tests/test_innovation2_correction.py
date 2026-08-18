from __future__ import annotations

import numpy as np

from stereo_dynamic_measurement.innovation2.trajectory_corrector import CorrectionConfig, correct_trajectory_point


def test_low_physics_confidence_creates_separate_corrected_xyz_with_audit_metadata() -> None:
    raw = np.array([0.0, 80.0, 2000.0])
    predicted = np.array([0.0, 5.0, 2000.0])
    neighbors = np.array([[0.0, 3.0, 2000.0], [0.0, 6.0, 2000.0], [0.0, 4.0, 2000.0]])

    result = correct_trajectory_point(raw, predicted, neighbors, c_phy=0.15, config=CorrectionConfig(threshold=0.55))

    assert result.correction_applied
    assert result.correction_method in {"history_neighbor_weighted", "history_prediction"}
    assert np.array_equal(result.raw_xyz_mm, raw)
    assert result.corrected_xyz_mm[1] < 10.0
    assert result.correction_magnitude_mm > 60.0


def test_high_physics_confidence_preserves_raw_trajectory() -> None:
    raw = np.array([1.0, 2.0, 2000.0])

    result = correct_trajectory_point(raw, np.array([0.0, 3.0, 2000.0]), np.empty((0, 3)), c_phy=0.9, config=CorrectionConfig())

    assert not result.correction_applied
    assert result.correction_method == "none"
    assert np.array_equal(result.corrected_xyz_mm, raw)
