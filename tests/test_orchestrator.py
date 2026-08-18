from __future__ import annotations

import numpy as np

from stereo_dynamic_measurement.innovation3.fault_classifier import FaultType
from stereo_dynamic_measurement.pipeline.data_types import MeasurementResult, PhysicsValidationResult
from stereo_dynamic_measurement.pipeline.orchestrator import StereoMeasurementOrchestrator


def _measurement(*, lr_residual: float) -> MeasurementResult:
    return MeasurementResult(
        frame_id=2, timestamp=0.2, point_id="P1", left_xy=(100.0, 100.0), right_xy=(90.0, 100.0),
        disparity_raw=10.0, disparity_subpixel=10.0, xyz_raw=np.array([0.0, 0.0, 2_000.0]),
        gradient_score=0.9, texture_score=0.9, blur_score=0.9, lr_residual=lr_residual,
        neighbor_residual=lr_residual, measurement_confidence=0.3,
    )


def _physics(measurement: MeasurementResult, confidence: float) -> PhysicsValidationResult:
    return PhysicsValidationResult(
        frame_id=measurement.frame_id, timestamp=measurement.timestamp, point_id=measurement.point_id,
        xyz_raw=measurement.xyz_raw, xyz_corrected=measurement.xyz_raw, physics_confidence=confidence,
    )


def test_orchestrator_remeasures_and_revalidates_after_fault_recovery() -> None:
    calls = {"remeasure": 0, "validate": 0}

    def validate(measurement: MeasurementResult) -> PhysicsValidationResult:
        calls["validate"] += 1
        return _physics(measurement, 0.2 if calls["validate"] == 1 else 0.9)

    def remeasure(_action) -> MeasurementResult:
        calls["remeasure"] += 1
        return _measurement(lr_residual=0.1)

    result = StereoMeasurementOrchestrator().process(
        measurement=_measurement(lr_residual=1.5), validate=validate, remeasure=remeasure
    )

    assert result.diagnosis.fault_type == FaultType.STEREO_MISMATCH.value
    assert calls == {"remeasure": 1, "validate": 2}
    assert result.diagnosis.recovery_success is True
    assert result.physics_after.physics_confidence == 0.9
