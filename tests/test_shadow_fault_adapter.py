from __future__ import annotations

from stereo_dynamic_measurement.innovation3.fault_classifier import FaultType, RuleDiagnosticEngine
from stereo_dynamic_measurement.innovation3.fault_fingerprint import FaultFingerprint
from stereo_dynamic_measurement.innovation3.recovery_manager import RecoveryManager


def test_missing_fault_evidence_is_unavailable_and_does_not_trigger_a_rule() -> None:
    fingerprint = FaultFingerprint(lr_residual=3.0)

    diagnosis = RuleDiagnosticEngine().diagnose(fingerprint)

    assert fingerprint.gradient_quality is None
    assert fingerprint.neighbor_residual is None
    assert diagnosis.fault_type is FaultType.NORMAL
    assert diagnosis.evidence == ()


def test_recovery_plan_is_recommendation_data_without_execution_interface() -> None:
    plan = RecoveryManager().plan(FaultType.SEARCH_RANGE_FAILURE)

    assert plan.action == "EXPAND_SEARCH"
    assert plan.parameter_updates == {"search_scale": 2.0}
    assert not hasattr(plan, "execute")
