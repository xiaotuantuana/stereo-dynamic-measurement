from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..innovation3.fault_classifier import RuleDiagnosticEngine
from ..innovation3.fault_fingerprint import FaultFingerprint
from ..innovation3.recovery_manager import RecoveryManager
from ..innovation3.verifier import verify_recovery
from .data_types import DiagnosisResult, MeasurementResult, PhysicsValidationResult, RecoveryAction


@dataclass(frozen=True)
class OrchestratedFrameResult:
    measurement_before: MeasurementResult
    physics_before: PhysicsValidationResult
    measurement_after: MeasurementResult
    physics_after: PhysicsValidationResult
    diagnosis: DiagnosisResult

    @property
    def final_xyz(self):
        return self.physics_after.xyz_corrected


class StereoMeasurementOrchestrator:
    """Runs the innovation layers in order and re-runs them after recoverable faults.

    The callbacks are hardware-agnostic boundaries: a video/camera runner supplies a
    remeasurement implementation, while this class owns diagnosis and verification.
    No ground-truth values or injected labels are accepted by this runtime path.
    """

    def __init__(self, diagnostic_engine: RuleDiagnosticEngine | None = None, recovery_manager: RecoveryManager | None = None) -> None:
        self.diagnostic_engine = diagnostic_engine or RuleDiagnosticEngine()
        self.recovery_manager = recovery_manager or RecoveryManager()

    @staticmethod
    def _fingerprint(measurement: MeasurementResult, physics: PhysicsValidationResult) -> FaultFingerprint:
        return FaultFingerprint(
            gradient_quality=measurement.gradient_score,
            blur_score=measurement.blur_score,
            lr_residual=measurement.lr_residual,
            epipolar_residual=measurement.epipolar_residual,
            matching_cost_residual=measurement.matching_cost,
            neighbor_residual=measurement.neighbor_residual,
            flow_residual=max(abs(measurement.flow_u), abs(measurement.flow_v)),
            fb_error=measurement.flow_fb_error,
            temporal_residual=measurement.temporal_residual,
            physics_residual=1.0 - physics.physics_confidence,
        )

    def process(
        self,
        *,
        measurement: MeasurementResult,
        validate: Callable[[MeasurementResult], PhysicsValidationResult],
        remeasure: Callable[[RecoveryAction], MeasurementResult] | None = None,
    ) -> OrchestratedFrameResult:
        """Validate one frame and, for a fault, execute and verify a remeasurement."""
        physics_before = validate(measurement)
        fingerprint = self._fingerprint(measurement, physics_before)
        fault = self.diagnostic_engine.diagnose(fingerprint)
        plan = self.recovery_manager.plan(fault.fault_type)
        action = RecoveryAction(plan.action, dict(plan.parameter_updates))
        after_measurement, after_physics = measurement, physics_before
        if fault.fault_type.value != "NORMAL" and remeasure is not None:
            after_measurement = remeasure(action)
            after_physics = validate(after_measurement)
        residual_before = max(measurement.lr_residual, measurement.epipolar_residual, measurement.flow_fb_error)
        residual_after = max(after_measurement.lr_residual, after_measurement.epipolar_residual, after_measurement.flow_fb_error)
        verification = verify_recovery(
            c_phy_before=physics_before.physics_confidence,
            c_phy_after=after_physics.physics_confidence,
            primary_residual_before=residual_before,
            primary_residual_after=residual_after,
        )
        diagnosis = DiagnosisResult(
            frame_id=measurement.frame_id, timestamp=measurement.timestamp, point_id=measurement.point_id,
            fault_type=fault.fault_type.value, fault_score=fault.score, recovery_action=action,
            recovery_success=verification.recovery_success if fault.fault_type.value != "NORMAL" else True,
            c_phy_before=physics_before.physics_confidence, c_phy_after=after_physics.physics_confidence,
            residuals_before={"primary": residual_before, "physics": 1.0 - physics_before.physics_confidence},
            residuals_after={"primary": residual_after, "physics": 1.0 - after_physics.physics_confidence},
        )
        return OrchestratedFrameResult(measurement, physics_before, after_measurement, after_physics, diagnosis)
