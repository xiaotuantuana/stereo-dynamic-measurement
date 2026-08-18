from __future__ import annotations

import numpy as np
import pytest

from stereo_dynamic_measurement.innovation3.fault_classifier import FaultType, RuleDiagnosticEngine
from stereo_dynamic_measurement.innovation3.fault_fingerprint import FaultFingerprint
from stereo_dynamic_measurement.innovation3.geometry_health import geometry_health


def test_rule_engine_explains_stereo_mismatch_and_camera_motion() -> None:
    engine = RuleDiagnosticEngine()
    mismatch = engine.diagnose(FaultFingerprint(lr_residual=3.0, neighbor_residual=2.0, reference_motion_residual=0.1))
    camera = engine.diagnose(FaultFingerprint(reference_motion_residual=3.0, common_target_motion_ratio=0.9))

    assert mismatch.fault_type is FaultType.STEREO_MISMATCH
    assert camera.fault_type is FaultType.CAMERA_MOTION
    assert mismatch.evidence and camera.evidence


def test_geometry_health_uses_vertical_disparity_statistics() -> None:
    result = geometry_health(np.array([0.1, -0.1, 0.2, 0.0]), np.array([0.0, 0.2, -0.1, 0.1]))
    drift = geometry_health(np.array([3.0, 2.0, 4.0, 3.0]), np.zeros(4))

    assert result.health_score > drift.health_score
    assert drift.epipolar_rms_px > 2.0
