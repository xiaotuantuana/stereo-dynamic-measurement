from __future__ import annotations

import csv

import numpy as np

from stereo_research.annotate_gt import enrich_annotation_with_geometry, write_ground_truth_csv
from stereo_research.models import FramePointResult


def _q() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, -100.0],
            [0.0, 1.0, 0.0, -50.0],
            [0.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_pixel_ground_truth_is_written_with_reprojected_xyz(tmp_path) -> None:
    annotation = enrich_annotation_with_geometry(
        {
            "frame": 3,
            "point_id": "P1",
            "left_x": 110.0,
            "left_y": 50.0,
            "right_x": 100.0,
            "right_y": 50.0,
        },
        _q(),
        "m",
    )
    output = write_ground_truth_csv(tmp_path / "gt.csv", [annotation])
    with output.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))

    assert float(row["gt_disparity"]) == 10.0
    assert abs(float(row["X_m"]) - 0.1) < 1e-12
    assert abs(float(row["Y_m"])) < 1e-12
    assert abs(float(row["Z_m"]) - 1.0) < 1e-12


def test_csv_row_preserves_measured_estimated_and_displacement_fields() -> None:
    row = FramePointResult(
        method="full_quality",
        frame=2,
        point_id="P1",
        status="valid",
        integer_disparity=8.0,
        measured_disparity=8.4,
        estimated_disparity=8.2,
        measured_x_m=0.1,
        estimated_x_m=0.11,
        delta_x_mm=10.0,
    ).as_csv_row()

    assert row["integer_disparity"] == 8.0
    assert row["measured_disparity"] == 8.4
    assert row["estimated_disparity"] == 8.2
    assert row["measured_X_m"] == 0.1
    assert row["estimated_X_m"] == 0.11
    assert row["delta_X_mm"] == 10.0
