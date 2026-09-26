from __future__ import annotations

import pytest

from stereo_research.models import FramePointResult, MatcherConfig
from stereo_research.runner import (
    ACCURACY_POLICY_CSV_FIELDS,
    CSV_FIELDS,
    SHADOW_CSV_FIELDS,
)


EXPECTED_ACCURACY_FIELDS = [
    "accuracy_policy_enabled", "precision_policy_warmup", "target_metric",
    "target_x_mm", "target_y_mm", "target_z_mm",
    "required_sigma_d_px", "estimated_sigma_d_px",
    "estimated_sigma_x_mm", "estimated_sigma_y_mm", "estimated_sigma_z_mm",
    "precision_ratio", "precision_feasible", "precision_status", "limiting_axis",
    "policy_base_search_radius_px", "policy_final_search_radius_px",
    "policy_refinement_level", "policy_retry_budget", "policy_precision_retry_count",
    "policy_acceptance_reason", "policy_reason",
    "retry_triggered", "retry_candidate_accepted", "retry_sigma_before_px",
    "retry_sigma_after_px", "retry_match_cost_before", "retry_match_cost_after",
    "retry_lr_error_before_px", "retry_lr_error_after_px",
]


def test_accuracy_fields_append_after_unchanged_first_round_169_column_prefix() -> None:
    assert len(CSV_FIELDS) == 169 + len(EXPECTED_ACCURACY_FIELDS)
    assert ACCURACY_POLICY_CSV_FIELDS == EXPECTED_ACCURACY_FIELDS
    assert CSV_FIELDS[-len(EXPECTED_ACCURACY_FIELDS):] == EXPECTED_ACCURACY_FIELDS
    shadow_start = 169 - len(SHADOW_CSV_FIELDS)
    assert CSV_FIELDS[shadow_start:169] == SHADOW_CSV_FIELDS


def test_feature_defaults_off_and_disabled_result_serializes_empty_policy_values() -> None:
    config = MatcherConfig()
    result = FramePointResult(method="M3", frame=0, point_id="P1", status="valid")
    row = result.as_csv_row()

    assert not config.enable_target_accuracy_policy
    assert row["accuracy_policy_enabled"] is False
    assert row["target_metric"] == ""
    assert row["required_sigma_d_px"] == ""
    assert row["precision_status"] == ""


def test_enabled_policy_requires_explicit_sigma_target_and_safe_bounds() -> None:
    with pytest.raises(ValueError, match="target sigma"):
        MatcherConfig(enable_target_accuracy_policy=True)
    with pytest.raises(ValueError, match="sigma"):
        MatcherConfig(
            enable_target_accuracy_policy=True,
            target_metric="rmse",
            target_sigma_z_mm=1.0,
        )
    with pytest.raises(ValueError, match="retry"):
        MatcherConfig(
            enable_target_accuracy_policy=True,
            target_sigma_z_mm=1.0,
            max_precision_retry=-1,
        )
