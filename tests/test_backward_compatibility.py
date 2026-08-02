from __future__ import annotations

import json
from pathlib import Path

import pytest

from stereo_research.models import MatcherConfig, SequenceManifest, method_profile
from stereo_research.runner import CSV_FIELDS


OLD_METHOD_EXPECTATIONS = {
    "sgbm": (True, False, False, False, False, False, False),
    "local": (False, False, False, False, False, False, False),
    "local_flow": (False, True, False, False, False, False, False),
    "full": (False, True, True, True, True, True, True),
    "full_quality": (False, True, True, True, True, True, True),
}


def test_old_method_component_profiles_remain_unchanged() -> None:
    for method, expected in OLD_METHOD_EXPECTATIONS.items():
        profile = method_profile(method)
        actual = (
            profile.dense_sgbm_each_frame,
            profile.use_flow,
            profile.use_prediction,
            profile.use_epipolar,
            profile.use_neighborhood,
            profile.use_subpixel,
            profile.use_lr_check,
        )
        assert actual == expected
        assert profile.use_cycle_consistency is False
        assert profile.use_icgn is False
        assert profile.use_adaptive_filter is False
        assert profile.use_camera_compensation is False


def test_research_full_enables_all_new_components() -> None:
    profile = method_profile("research_full")

    assert profile.use_cycle_consistency is True
    assert profile.use_icgn is True
    assert profile.use_adaptive_filter is True
    assert profile.use_camera_compensation is True


def test_old_point_json_defaults_to_measurement_role(tmp_path: Path) -> None:
    (tmp_path / "points.json").write_text(
        json.dumps({"frame": 0, "points": [{"id": "P1", "x": 10, "y": 20}]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "old-format",
                "video": "stereo.avi",
                "start_frame": 0,
                "calibration": "builtin_640x480",
                "points": "points.json",
            }
        ),
        encoding="utf-8",
    )

    manifest = SequenceManifest.from_json(manifest_path)

    assert manifest.point_specs[0].role == "measurement"


def test_point_json_rejects_unknown_role(tmp_path: Path) -> None:
    (tmp_path / "points.json").write_text(
        json.dumps(
            {"frame": 0, "points": [{"id": "P1", "x": 10, "y": 20, "role": "moving"}]}
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "bad-role",
                "video": "stereo.avi",
                "start_frame": 0,
                "points": "points.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="role"):
        SequenceManifest.from_json(manifest_path)


def test_old_csv_fields_keep_their_original_order_as_prefix() -> None:
    old_fields = [
        "repeat", "method", "frame", "point_id", "status", "left_x", "left_y",
        "right_x", "right_y", "disparity", "X_m", "Y_m", "Z_m", "distance_m",
        "match_cost", "flow_fb_error_px", "lr_error_px", "confidence",
        "raw_disparity", "integer_disparity", "measured_disparity",
        "estimated_disparity", "measured_right_x", "measured_right_y",
        "estimated_right_x", "estimated_right_y", "measured_X_m", "measured_Y_m",
        "measured_Z_m", "estimated_X_m", "estimated_Y_m", "estimated_Z_m",
        "delta_X_mm", "delta_Y_mm", "delta_Z_mm", "subpixel_offset",
        "neighbor_disparity", "used_search_radius", "quality_stage", "recovery_stage",
        "recovery_attempt_count", "recovery_success_count", "mean_recovery_frames",
        "flow_ms", "matching_ms", "total_ms", "frame_total_ms",
    ]

    assert CSV_FIELDS[: len(old_fields)] == old_fields


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"cycle_soft_threshold_px": 0.0}, "cycle"),
        ({"cycle_soft_threshold_px": 1.0, "cycle_hard_threshold_px": 0.5}, "cycle"),
        ({"cycle_prediction_blend": 1.1}, "blend"),
        ({"icgn_patch_size": 14}, "icgn_patch_size"),
        ({"icgn_max_iterations": 0}, "icgn_max_iterations"),
        ({"icgn_fallback_method": "bad"}, "fallback"),
        ({"kalman_nis_soft_threshold": 10.0, "kalman_nis_hard_threshold": 5.0}, "NIS"),
        ({"camera_compensation_min_points": 2}, "camera_compensation_min_points"),
    ],
)
def test_new_configuration_ranges_are_validated(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MatcherConfig(**kwargs)
