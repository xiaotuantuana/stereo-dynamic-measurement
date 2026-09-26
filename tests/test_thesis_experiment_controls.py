from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import cv2

from stereo_research.ground_truth import ExternalGroundTruthLoader
from stereo_research.confidence import decide_confidence
from stereo_research.uncertainty import estimate_match_uncertainty
from stereo_research.models import MatcherConfig, PointState, method_profile
from stereo_research.pipeline import TemporalStereoPipeline
from stereo_research.annotate import write_points_file
from stereo_research.runner import _load_timestamps, run_confidence_suite, run_manifest


def test_thesis_method_profiles_keep_m1_m2_fixed_and_m3_adaptive() -> None:
    assert method_profile("M0").dense_sgbm_each_frame
    assert method_profile("M1").use_flow and not method_profile("M1").use_prediction
    assert method_profile("M2").use_flow and method_profile("M2").use_prediction
    assert method_profile("M3").adaptive_search


def test_adaptive_window_changes_only_for_m3() -> None:
    config = MatcherConfig()
    state = PointState("P1", (0, 0), (0, 0), confidence_state="HIGH")
    m3 = TemporalStereoPipeline("M3", np.eye(4), "m", config)
    assert m3._determine_search_radius(state, 0.0, False) == (4, "stable_prediction")
    state.last_prediction_residual_px = 2.0
    assert m3._determine_search_radius(state, 0.0, False) == (16, "prediction_error")
    assert TemporalStereoPipeline("M2", np.eye(4), "m", config)._determine_search_radius(state, 5.0, False) == (8, "fixed")


def test_external_ground_truth_uses_linear_timestamp_interpolation(tmp_path: Path) -> None:
    path = tmp_path / "external_gt.csv"
    path.write_text("timestamp_s,Z_gt\n0,1\n2,3\n", encoding="utf-8")
    result = ExternalGroundTruthLoader.from_csv(path).interpolate(0.5)
    assert result["gt_valid"] is True
    assert result["Z_gt"] == 1.5


def test_confidence_thresholds_and_lost_transition() -> None:
    config = MatcherConfig(confidence_low_to_lost_frames=2)
    frames = 0
    assert decide_confidence(.85, frames, high=.75, medium=.45, low=.25, lost_after=2, mode="closed_loop")[0].state == "HIGH"
    assert decide_confidence(.60, frames, high=.75, medium=.45, low=.25, lost_after=2, mode="closed_loop")[0].state == "MEDIUM"
    assert decide_confidence(.35, frames, high=.75, medium=.45, low=.25, lost_after=2, mode="closed_loop")[0].state == "LOW"
    decision, frames = decide_confidence(.10, frames, high=.75, medium=.45, low=.25, lost_after=2, mode="closed_loop")
    assert decision.state == "LOW"
    assert decide_confidence(.10, frames, high=.75, medium=.45, low=.25, lost_after=2, mode="closed_loop")[0].state == "LOST"


def test_unavailable_optional_evidence_does_not_prevent_high_confidence() -> None:
    quality = estimate_match_uncertainty(40, .01, .5, None, .01, None, .01, None, None, None, MatcherConfig())
    assert quality.quality_score >= .75


def test_m0_to_m3_run_on_identical_video_and_export_thesis_fields(tmp_path: Path) -> None:
    video = tmp_path / "stereo.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 480))
    rng = np.random.default_rng(17)
    for _ in range(3):
        left = cv2.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (3, 3), 0.5)
        right = cv2.warpAffine(left, np.asarray([[1, 0, -8], [0, 1, 0]], np.float32), (640, 480), borderMode=cv2.BORDER_REFLECT101)
        writer.write(np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)]))
    writer.release()
    write_points_file(tmp_path / "points.json", 0, [(400.0, 250.0)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"name": "thesis", "video": video.name, "start_frame": 0, "end_frame": 2, "calibration": "builtin_640x480", "points": "points.json", "output_dir": "results"}), encoding="utf-8")
    outputs = run_manifest(manifest, methods=("M0", "M1", "M2", "M3"), warmup_frames=0)
    assert set(outputs) == {"M0", "M1", "M2", "M3"}
    for method, output in outputs.items():
        with output.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3 and all(row["timestamp_source"] == "video_fps" for row in rows)
        if method in {"M1", "M2"}:
            assert {row["search_radius_reason"] for row in rows[1:]} <= {"fixed", ""}


def test_manifest_separates_ground_truth_and_hardware_timestamps(tmp_path: Path) -> None:
    (tmp_path / "points.json").write_text('{"frame":0,"points":[{"id":"P1","x":1,"y":1}]}', encoding="utf-8")
    (tmp_path / "stereo.csv").write_text("frame,left_x,right_x\n0,1,1\n", encoding="utf-8")
    (tmp_path / "external.csv").write_text("timestamp_s,Z_gt\n0,1\n1,2\n", encoding="utf-8")
    (tmp_path / "timestamps.csv").write_text("frame,left_timestamp_s,right_timestamp_s\n0,1.0,1.003\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"name":"x", "video":"video.avi", "points":"points.json", "stereo_ground_truth":"stereo.csv", "external_ground_truth":"external.csv", "timestamps":"timestamps.csv"}), encoding="utf-8")
    from stereo_research.models import SequenceManifest
    loaded = SequenceManifest.from_json(manifest)
    assert loaded.stereo_ground_truth.name == "stereo.csv" and loaded.external_ground_truth.name == "external.csv"
    assert _load_timestamps(loaded.timestamps)[0] == (1.0, 1.003)
