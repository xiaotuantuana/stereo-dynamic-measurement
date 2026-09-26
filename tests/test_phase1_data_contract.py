from __future__ import annotations

import pytest

from stereo_research.models import (
    FINAL_RESULT_OWNER,
    FramePointResult,
    Innovation2Mode,
    Innovation3Mode,
    MatcherConfig,
    SystemMode,
    resolve_system_mode,
)
from stereo_research.runner import CSV_FIELDS


def _result(**overrides: object) -> FramePointResult:
    values: dict[str, object] = {
        "method": "research_full",
        "frame": 7,
        "point_id": "P1",
        "status": "valid",
        "measured_x_m": 1.0,
        "measured_y_m": 2.0,
        "measured_z_m": 3.0,
        "estimated_x_m": 1.1,
        "estimated_y_m": 2.1,
        "estimated_z_m": 3.1,
        "compensated_x_m": 1.2,
        "compensated_y_m": 2.2,
        "compensated_z_m": 3.2,
        "candidate_corrected_x_m": 1.3,
        "candidate_corrected_y_m": 2.3,
        "candidate_corrected_z_m": 3.3,
        "final_x_m": 1.2,
        "final_y_m": 2.2,
        "final_z_m": 3.2,
    }
    values.update(overrides)
    return FramePointResult(**values)


def test_baseline_mode_strictly_reproduces_frozen_default_configuration() -> None:
    resolved = resolve_system_mode(SystemMode.BASELINE)

    assert resolved.matcher_config == MatcherConfig()
    assert resolved.enable_unified_innovation1 is False
    assert resolved.innovation2_mode is Innovation2Mode.SHADOW
    assert resolved.innovation3_mode is Innovation3Mode.DIAGNOSE
    assert resolved.matcher_config.enable_prediction is True
    assert resolved.matcher_config.enable_flow is True
    assert resolved.matcher_config.enable_target_accuracy_policy is False
    assert resolved.matcher_config.enable_physics_shadow is True
    assert resolved.matcher_config.enable_fault_shadow is True
    assert resolved.matcher_config.allow_physics_correction_to_final is False
    assert resolved.matcher_config.allow_fault_recovery_to_final is False


def test_baseline_rejects_legacy_flag_conflicts_instead_of_silently_disabling_shadow() -> None:
    with pytest.raises(ValueError, match="BASELINE"):
        resolve_system_mode(
            SystemMode.BASELINE,
            matcher_config=MatcherConfig(enable_physics_shadow=False),
        )


def test_mode_resolution_is_deterministic_and_full_enhanced_is_unauthorized() -> None:
    assert resolve_system_mode("BASELINE") == resolve_system_mode(SystemMode.BASELINE)
    with pytest.raises(PermissionError, match="FULL_ENHANCED"):
        resolve_system_mode(SystemMode.FULL_ENHANCED)


def test_innovation1_mode_preserves_existing_shadow_and_diagnose_behavior() -> None:
    resolved = resolve_system_mode(SystemMode.INNOVATION1)

    assert resolved.enable_unified_innovation1 is True
    assert resolved.innovation2_mode is Innovation2Mode.SHADOW
    assert resolved.innovation3_mode is Innovation3Mode.DIAGNOSE
    assert resolved.matcher_config.enable_physics_shadow is True
    assert resolved.matcher_config.enable_fault_shadow is True


def test_lifecycle_exposes_unambiguous_coordinate_stages() -> None:
    result = _result()

    assert result.triangulated_xyz_m == (1.0, 2.0, 3.0)
    assert result.pipeline_baseline_xyz_m == (1.2, 2.2, 3.2)
    assert result.pipeline_baseline_source == "COMPENSATED"
    assert result.candidate_xyz_m == (1.3, 2.3, 3.3)
    assert result.final_xyz_m == (1.2, 2.2, 3.2)


def test_pipeline_baseline_falls_back_to_estimated_result_without_relabeling_raw() -> None:
    result = _result(
        compensated_x_m=None,
        compensated_y_m=None,
        compensated_z_m=None,
    )

    assert result.pipeline_baseline_xyz_m == (1.1, 2.1, 3.1)
    assert result.pipeline_baseline_source == "ESTIMATED"


def test_final_contract_names_temporal_pipeline_as_the_only_owner() -> None:
    result = _result()

    assert result.final_owner == FINAL_RESULT_OWNER == "TemporalStereoPipeline"


def test_csv_serialization_preserves_legacy_schema_while_contract_stays_in_memory() -> None:
    row = _result().as_csv_row()
    assert set(row) <= set(CSV_FIELDS)
    assert row["measured_X_m"] == 1.0
    assert row["compensated_X_m"] == 1.2
    assert row["candidate_corrected_x_m"] == 1.3
    assert row["final_X_m"] == 1.2


def test_shadow_candidate_does_not_change_final_contract_value() -> None:
    result = _result(candidate_corrected_z_m=0.5)

    assert result.candidate_xyz_m == (1.3, 2.3, 0.5)
    assert result.final_xyz_m == (1.2, 2.2, 3.2)
