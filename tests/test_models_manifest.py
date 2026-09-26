from __future__ import annotations

import json
from pathlib import Path

import pytest

from stereo_research.models import (
    FramePointResult,
    MatcherConfig,
    PointState,
    SequenceManifest,
    method_profile,
)


def test_manifest_resolves_relative_paths_and_loads_multiple_points(tmp_path: Path) -> None:
    points_path = tmp_path / "points.json"
    points_path.write_text(
        json.dumps(
            {
                "frame": 250,
                "points": [
                    {"id": "P1", "x": 120.5, "y": 80.0},
                    {"id": "P2", "x": 250.0, "y": 180.25},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "sequence-a",
                "video": "car.avi",
                "start_frame": 250,
                "end_frame": 500,
                "calibration": "builtin_640x480",
                "points": "points.json",
            }
        ),
        encoding="utf-8",
    )

    manifest = SequenceManifest.from_json(manifest_path)

    assert manifest.video == (tmp_path / "car.avi").resolve()
    assert manifest.start_frame == 250
    assert manifest.end_frame == 500
    assert [point.point_id for point in manifest.point_specs] == ["P1", "P2"]
    assert manifest.point_specs[1].xy == (250.0, 180.25)


def test_manifest_rejects_points_from_a_different_initial_frame(tmp_path: Path) -> None:
    (tmp_path / "points.json").write_text(
        json.dumps({"frame": 10, "points": [{"id": "P1", "x": 20, "y": 20}]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "bad",
                "video": "car.avi",
                "start_frame": 20,
                "end_frame": 30,
                "calibration": "builtin_640x480",
                "points": "points.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="start frame"):
        SequenceManifest.from_json(manifest_path)


def test_method_profiles_isolate_the_planned_components() -> None:
    assert method_profile("sgbm").dense_sgbm_each_frame is True
    assert method_profile("local").use_flow is False
    assert method_profile("local_flow").use_flow is True
    assert method_profile("local_flow").use_prediction is False
    full = method_profile("full")
    assert full.use_prediction is True
    assert full.use_neighborhood is True
    assert full.use_subpixel is True
    assert full.use_lr_check is True
    quality = method_profile("full_quality")
    assert quality == full
    assert method_profile("sgbm_flow").dense_sgbm_each_frame is True
    assert method_profile("sgbm_flow").use_flow is True


def test_matcher_config_rejects_even_patch_size() -> None:
    with pytest.raises(ValueError, match="odd"):
        MatcherConfig(patch_size=10)


def test_regular_matcher_can_use_patch_larger_than_quality_context_default() -> None:
    config = MatcherConfig(patch_size=21)

    assert config.patch_size == 21


def test_frame_result_serializes_invalid_values_as_empty_csv_fields() -> None:
    state = PointState(point_id="P1", initial_left_xy=(10.0, 20.0), left_xy=(10.0, 20.0))
    result = FramePointResult.invalid(
        method="full",
        frame=251,
        point_state=state,
        status="flow_failed",
        flow_ms=0.4,
        total_ms=0.5,
    )

    row = result.as_csv_row()

    assert row["point_id"] == "P1"
    assert row["status"] == "flow_failed"
    assert row["disparity"] == ""
    assert row["X_m"] == ""
    assert row["flow_ms"] == 0.4
