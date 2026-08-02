from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo_research.annotate import annotate_video, write_points_file
from stereo_research.annotate_gt import write_ground_truth_csv
from stereo_research.calibration import builtin_640x480
from stereo_research.report import evaluate_experiment, trajectory_series
from stereo_research.runner import run_ablation_suite, run_manifest


def _write_stereo_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (1280, 480),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(9)
    for frame_index in range(frame_count):
        left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (3, 3), 0.5)
        transform = np.array(
            [[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        right = cv2.warpAffine(
            left,
            transform,
            (640, 480),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        stereo = np.hstack(
            [
                cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(right, cv2.COLOR_GRAY2BGR),
            ]
        )
        writer.write(stereo)
    writer.release()


def test_noninteractive_annotation_writes_reusable_point_schema(tmp_path: Path) -> None:
    output = tmp_path / "points.json"

    write_points_file(output, frame=250, points=[(100.5, 80.0), (200.0, 150.25)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["frame"] == 250
    assert payload["points"][0] == {"id": "P1", "x": 100.5, "y": 80.0}
    assert payload["points"][1]["id"] == "P2"


def test_annotation_uses_requested_calibration_file(tmp_path: Path) -> None:
    video = tmp_path / "stereo.avi"
    _write_stereo_video(video, frame_count=1)
    calibration_path = tmp_path / "wrong_size_calibration.json"
    replace(builtin_640x480(), image_size=(320, 480)).to_json(calibration_path)

    with pytest.raises(ValueError, match="640x480"):
        annotate_video(
            video,
            start_frame=0,
            output=tmp_path / "points.json",
            calibration=str(calibration_path),
        )


def test_sparse_ground_truth_writer_uses_evaluator_schema(tmp_path: Path) -> None:
    output = tmp_path / "ground_truth.csv"

    write_ground_truth_csv(
        output,
        [
            {
                "frame": 10,
                "point_id": "P1",
                "left_x": 100.0,
                "left_y": 80.0,
                "right_x": 92.0,
                "right_y": 80.0,
            }
        ],
    )

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["frame"] == "10"
    assert rows[0]["right_x"] == "92.0"
    assert "distance_m" in rows[0]


def test_trajectory_series_never_connects_different_point_ids() -> None:
    rows = [
        {"repeat": "0", "frame": "0", "point_id": "P1", "status": "valid", "Z_m": "2.0"},
        {"repeat": "0", "frame": "0", "point_id": "P2", "status": "valid", "Z_m": "8.0"},
        {"repeat": "0", "frame": "1", "point_id": "P1", "status": "valid", "Z_m": "2.1"},
        {"repeat": "0", "frame": "1", "point_id": "P2", "status": "valid", "Z_m": "8.1"},
    ]

    series = trajectory_series(rows)

    assert series == {
        "P1": ([0, 1], [2.0, 2.1]),
        "P2": ([0, 1], [8.0, 8.1]),
    }


def test_run_manifest_writes_comparable_csv_for_each_method(tmp_path: Path) -> None:
    video = tmp_path / "stereo.avi"
    _write_stereo_video(video)
    write_points_file(tmp_path / "points.json", frame=0, points=[(400.0, 250.0)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "integration",
                "video": video.name,
                "start_frame": 0,
                "end_frame": 2,
                "calibration": "builtin_640x480",
                "points": "points.json",
                "output_dir": "results",
            }
        ),
        encoding="utf-8",
    )

    outputs = run_manifest(manifest, methods=("sgbm", "local"), warmup_frames=0)

    assert set(outputs) == {"sgbm", "local"}
    for path in outputs.values():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert {row["point_id"] for row in rows} == {"P1"}
        assert all("total_ms" in row for row in rows)
        assert all("measured_disparity" in row for row in rows)
        assert all("estimated_disparity" in row for row in rows)
        assert all("delta_X_mm" in row for row in rows)
        assert all(float(row["frame_total_ms"]) > 0 for row in rows)


def test_run_manifest_excludes_warmup_frames_from_runtime_samples(tmp_path: Path) -> None:
    video = tmp_path / "stereo.avi"
    _write_stereo_video(video, frame_count=4)
    write_points_file(tmp_path / "points.json", frame=0, points=[(400.0, 250.0)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "warmup",
                "video": video.name,
                "start_frame": 0,
                "end_frame": 3,
                "calibration": "builtin_640x480",
                "points": "points.json",
                "output_dir": "results",
            }
        ),
        encoding="utf-8",
    )

    output = run_manifest(
        manifest,
        methods=("sgbm",),
        warmup_frames=2,
    )["sgbm"]

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["frame_total_ms"] == "" for row in rows] == [True, True, False, False]
    metadata = json.loads((tmp_path / "results" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["warmup_frames"] == 2


def test_evaluate_experiment_writes_summary_without_fabricating_truth_metrics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    with (results / "full.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "frame",
                "point_id",
                "status",
                "X_m",
                "Y_m",
                "Z_m",
                "distance_m",
                "total_ms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "full",
                "frame": 0,
                "point_id": "P1",
                "status": "valid",
                "X_m": 0,
                "Y_m": 0,
                "Z_m": 2,
                "distance_m": 2,
                "total_ms": 5,
            }
        )

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "PLOTS_SKIPPED.txt").write_text("stale", encoding="utf-8")
    summary_path = evaluate_experiment(tmp_path, report_dir)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["methods"]["full"]["xyz_metric_available"] is False
    assert summary["methods"]["full"]["distance_metric_available"] is False
    assert (tmp_path / "report" / "summary.csv").exists()
    assert (tmp_path / "report" / "runtime_boxplot.png").exists()
    assert (tmp_path / "report" / "depth_trajectories.png").exists()
    for name in (
        "disparity_error.png",
        "delta_x_trajectories.png",
        "delta_y_trajectories.png",
        "delta_z_trajectories.png",
        "raw_vs_filtered_displacement.png",
        "coverage_comparison.png",
        "false_match_comparison.png",
        "recovery_comparison.png",
    ):
        assert (tmp_path / "report" / name).exists()
    assert not (tmp_path / "report" / "PLOTS_SKIPPED.txt").exists()


def test_evaluate_experiment_reports_speedup_relative_to_sgbm(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    fieldnames = [
        "method",
        "repeat",
        "frame",
        "point_id",
        "status",
        "frame_total_ms",
    ]
    for method, runtime in (("sgbm", 10.0), ("full", 2.0)):
        with (results / f"{method}.csv").open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "method": method,
                    "repeat": 0,
                    "frame": 0,
                    "point_id": "P1",
                    "status": "valid",
                    "frame_total_ms": runtime,
                }
            )

    summary_path = evaluate_experiment(tmp_path, tmp_path / "report")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["methods"]["sgbm"]["speedup_vs_sgbm"] == 1.0
    assert summary["methods"]["full"]["speedup_vs_sgbm"] == 5.0


def test_evaluate_experiment_adds_ground_truth_only_figures(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    fields = ["method", "repeat", "frame", "point_id", "status", "X_m", "Y_m", "Z_m", "measured_disparity"]
    with (results / "full.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"method": "full", "repeat": 0, "frame": 0, "point_id": "P1", "status": "valid", "X_m": 0, "Y_m": 0, "Z_m": 2, "measured_disparity": 8})
    gt = tmp_path / "ground_truth.csv"
    write_ground_truth_csv(gt, [{"frame": 0, "point_id": "P1", "left_x": 10, "left_y": 20, "right_x": 2, "right_y": 20, "gt_disparity": 8, "X_m": 0, "Y_m": 0, "Z_m": 2}])

    evaluate_experiment(tmp_path, tmp_path / "report", ground_truth=gt)

    assert (tmp_path / "report" / "predicted_vs_ground_truth.png").exists()
    assert (tmp_path / "report" / "displacement_error_trajectories.png").exists()


def test_ablation_suite_keeps_each_full_method_deletion_separate(tmp_path: Path) -> None:
    video = tmp_path / "stereo.avi"
    _write_stereo_video(video, frame_count=1)
    write_points_file(tmp_path / "points.json", frame=0, points=[(400.0, 250.0)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "ablation",
                "video": video.name,
                "start_frame": 0,
                "end_frame": 0,
                "calibration": "builtin_640x480",
                "points": "points.json",
                "output_dir": "results",
            }
        ),
        encoding="utf-8",
    )

    outputs = run_ablation_suite(manifest)

    assert {path.parent.name for path in outputs.values()} == {"ablations"}
    assert set(outputs) == {
        "full",
        "full_no_flow",
        "full_no_prediction",
        "full_no_epipolar",
        "full_no_neighborhood",
        "full_no_subpixel",
        "full_no_lr",
        "full_no_pyramid",
        "full_no_recovery",
        "full_no_temporal_estimation",
    }
    with outputs["full_no_lr"].open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["method"] == "full_no_lr"
