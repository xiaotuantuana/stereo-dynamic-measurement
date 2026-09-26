from __future__ import annotations

from dataclasses import replace

import numpy as np

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
from stereo_dynamic_measurement.innovation3.fault_classifier import RuleDiagnosticEngine
from stereo_dynamic_measurement.innovation3.fault_fingerprint import FaultFingerprint
from stereo_dynamic_measurement.innovation3.recovery_manager import RecoveryManager

from .models import FramePointResult


class ShadowAnalyzer:
    """Read-only observer over completed frame results.

    Stored history is copied and never shares mutable arrays with the stereo pipeline.
    """

    def __init__(
        self,
        *,
        physics_config: PhysicsConfidenceConfig | None = None,
        correction_config: CorrectionConfig | None = None,
    ) -> None:
        self.physics_config = physics_config or PhysicsConfidenceConfig()
        self.correction_config = correction_config or CorrectionConfig()
        self._history_mm: dict[str, list[np.ndarray]] = {}
        self._diagnostic_engine = RuleDiagnosticEngine()
        self._recovery_manager = RecoveryManager()

    @staticmethod
    def _input_xyz(result: FramePointResult) -> tuple[np.ndarray, str] | None:
        stages = (
            ("compensated", (result.compensated_x_m, result.compensated_y_m, result.compensated_z_m)),
            ("estimated", (result.estimated_x_m, result.estimated_y_m, result.estimated_z_m)),
            ("measured", (result.measured_x_m, result.measured_y_m, result.measured_z_m)),
        )
        for name, values in stages:
            if all(value is not None for value in values):
                xyz = np.asarray(values, dtype=np.float64)
                if np.isfinite(xyz).all():
                    return xyz.copy() * 1000.0, name
        return None

    @staticmethod
    def _visual_evidence(result: FramePointResult) -> EvidenceTerm:
        residuals = [
            value
            for value in (result.lr_error_px, result.flow_fb_error_px)
            if value is not None and np.isfinite(value)
        ]
        if result.left_y is not None and result.right_y is not None:
            residuals.append(abs(float(result.left_y) - float(result.right_y)))
        if not residuals:
            return EvidenceTerm.unavailable("stereo/tracking residuals unavailable")
        return EvidenceTerm(float(np.exp(-max(residuals))), True)

    def process_frame(self, results: list[FramePointResult]) -> list[FramePointResult]:
        selected = {
            result.point_id: value
            for result in results
            if result.status == "valid" and (value := self._input_xyz(result)) is not None
        }
        current = {point_id: value[0] for point_id, value in selected.items()}
        displacements = {
            point_id: xyz - self._history_mm[point_id][-1]
            for point_id, xyz in current.items()
            if self._history_mm.get(point_id)
        }
        median_displacement = (
            np.median(np.stack(tuple(displacements.values())), axis=0)
            if displacements
            else None
        )

        processed: list[FramePointResult] = []
        for result in results:
            selected_input = selected.get(result.point_id)
            if selected_input is None:
                processed.append(result)
                continue
            xyz_mm, stage = selected_input
            history = self._history_mm.get(result.point_id, [])
            if len(history) >= 2:
                prediction = 2.0 * history[-1] - history[-2]
                temporal_residual = float(np.linalg.norm(xyz_mm - prediction))
                temporal = EvidenceTerm(float(np.exp(-temporal_residual / 15.0)), True)
            else:
                prediction = None
                temporal_residual = None
                temporal = EvidenceTerm.unavailable("two prior measurements required")

            displacement = displacements.get(result.point_id)
            if displacement is not None and median_displacement is not None:
                spatial_residual = float(np.linalg.norm(displacement - median_displacement))
                spatial = EvidenceTerm(float(np.exp(-spatial_residual / 15.0)), True)
                synchronous_ratio = 1.0 if spatial_residual <= 5.0 else 0.0
            else:
                spatial_residual = None
                spatial = EvidenceTerm.unavailable("previous same-frame multi-point motion required")
                synchronous_ratio = None

            visual = self._visual_evidence(result)
            evidence = {
                "flow_3d": EvidenceTerm.unavailable("independent 2D-3D prediction unavailable"),
                "temporal": temporal,
                "spatial": spatial,
                "spectral": EvidenceTerm.unavailable("runtime history window insufficient"),
                "phase": EvidenceTerm.unavailable("runtime history window insufficient"),
                "coherence": EvidenceTerm.unavailable("runtime history window insufficient"),
            }
            physics = compute_physics_confidence(evidence, self.physics_config)
            transient = decide_transient(
                temporal_residual_mm=(temporal_residual if temporal_residual is not None else float("nan")),
                flow_3d=evidence["flow_3d"],
                spatial=spatial,
                visual_consistency=visual,
                synchronous_motion_ratio=synchronous_ratio,
            )
            neighbors = np.stack(
                [other for point_id, other in current.items() if point_id != result.point_id]
            ) if len(current) > 1 else np.empty((0, 3), dtype=np.float64)
            if physics.valid and physics.c_phy is not None:
                correction = correct_trajectory_point(
                    xyz_mm,
                    prediction,
                    neighbors,
                    c_phy=physics.c_phy,
                    config=self.correction_config,
                    transient_decision=transient,
                )
                candidate_m = correction.candidate_corrected_xyz_mm / 1000.0
                transient_protected = correction.transient_protected
            else:
                candidate_m = xyz_mm / 1000.0
                transient_protected = False

            epipolar = (
                abs(float(result.left_y) - float(result.right_y))
                if result.left_y is not None and result.right_y is not None
                else None
            )
            reference_residual = result.compensation_rmse_mm
            fingerprint = FaultFingerprint(
                gradient_quality=(float(np.clip(result.texture_std / 10.0, 0.0, 1.0)) if result.texture_std is not None else None),
                lr_residual=result.lr_error_px,
                epipolar_residual=epipolar,
                matching_cost_residual=(result.match_cost / 0.45 if result.match_cost is not None else None),
                neighbor_residual=result.neighbor_disparity_mad,
                fb_error=result.flow_fb_error_px,
                temporal_residual=temporal_residual,
                reference_motion_residual=reference_residual,
                geometry_health_residual=(float(np.clip(epipolar, 0.0, 1.0)) if epipolar is not None else None),
                physics_residual=(physics.r_phy if physics.valid else None),
                common_target_motion_ratio=synchronous_ratio,
                tracking_loss_residual=0.0,
            )
            diagnosis = self._diagnostic_engine.diagnose(fingerprint)
            plan = self._recovery_manager.plan(diagnosis.fault_type)
            processed.append(
                replace(
                    result,
                    shadow_input_x_m=float(xyz_mm[0] / 1000.0),
                    shadow_input_y_m=float(xyz_mm[1] / 1000.0),
                    shadow_input_z_m=float(xyz_mm[2] / 1000.0),
                    shadow_input_stage=stage,
                    candidate_corrected_x_m=float(candidate_m[0]),
                    candidate_corrected_y_m=float(candidate_m[1]),
                    candidate_corrected_z_m=float(candidate_m[2]),
                    c_phy=physics.c_phy,
                    c_phy_valid=physics.valid,
                    c_phy_valid_terms=";".join(physics.available_evidence_names),
                    c_phy_missing_terms=";".join(physics.missing_evidence_names),
                    transient_protected=transient_protected,
                    fault_class=diagnosis.fault_type.value,
                    fault_confidence=diagnosis.score,
                    recommended_recovery=plan.action,
                    r_2d3d=None,
                    r_temporal=temporal_residual,
                    r_spatial=spatial_residual,
                    r_frequency=None,
                    r_phase=None,
                    r_coherence=None,
                    r_lr=result.lr_error_px,
                    r_epi=epipolar,
                    r_fb=result.flow_fb_error_px,
                    r_ref=reference_residual,
                    r_calib=None,
                )
            )

        for point_id, xyz in current.items():
            history = self._history_mm.setdefault(point_id, [])
            history.append(xyz.copy())
            if len(history) > 256:
                del history[:-256]
        return processed
