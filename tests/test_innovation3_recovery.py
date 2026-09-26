from __future__ import annotations
from stereo_dynamic_measurement.innovation3.fault_classifier import FaultType
from stereo_dynamic_measurement.innovation3.recovery_manager import RecoveryManager
from stereo_dynamic_measurement.innovation3.verifier import verify_recovery

def test_recovery_actions_are_fault_specific_and_verification_requires_both_improvements() -> None:
    manager = RecoveryManager()
    assert manager.plan(FaultType.STEREO_MISMATCH).action == "LOCAL_REMATCH"
    assert manager.plan(FaultType.CAMERA_MOTION).action == "REFERENCE_MOTION_COMPENSATION"
    success = verify_recovery(c_phy_before=.3, c_phy_after=.8, primary_residual_before=4.0, primary_residual_after=1.0)
    failed = verify_recovery(c_phy_before=.3, c_phy_after=.8, primary_residual_before=4.0, primary_residual_after=5.0)
    assert success.recovery_success and not failed.recovery_success
