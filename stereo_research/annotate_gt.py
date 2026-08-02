from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .geometry import reproject_point_m
from .models import SequenceManifest
from .runner import load_calibration, split_side_by_side


GROUND_TRUTH_FIELDS = [
    "frame",
    "point_id",
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "gt_disparity",
    "X_m",
    "Y_m",
    "Z_m",
    "distance_m",
]


def enrich_annotation_with_geometry(
    annotation: dict[str, Any],
    q: np.ndarray,
    calibration_unit: str,
) -> dict[str, Any]:
    result = dict(annotation)
    disparity = float(result["left_x"]) - float(result["right_x"])
    if disparity <= 0:
        raise ValueError("Ground-truth disparity must be positive")
    xyz = reproject_point_m(
        float(result["left_x"]),
        float(result["left_y"]),
        disparity,
        q,
        calibration_unit,
    )
    result.update(
        {
            "gt_disparity": disparity,
            "X_m": float(xyz[0]),
            "Y_m": float(xyz[1]),
            "Z_m": float(xyz[2]),
            "distance_m": float(np.linalg.norm(xyz)),
        }
    )
    return result


def write_ground_truth_csv(
    output: str | Path,
    annotations: Iterable[dict[str, Any]],
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUND_TRUTH_FIELDS)
        writer.writeheader()
        for annotation in annotations:
            writer.writerow(
                {
                    field: annotation.get(field, "")
                    for field in GROUND_TRUTH_FIELDS
                }
            )
    return output_path


def annotate_sparse_correspondences(
    manifest_path: str | Path,
    interval: int,
    output: str | Path,
) -> Path:
    if interval < 1:
        raise ValueError("interval must be positive")
    manifest = SequenceManifest.from_json(manifest_path)
    calibration_value = manifest.calibration
    if calibration_value != "builtin_640x480":
        calibration_path = Path(calibration_value)
        if not calibration_path.is_absolute():
            calibration_value = str((Path(manifest_path).resolve().parent / calibration_path).resolve())
    calibration = load_calibration(calibration_value)
    capture = cv2.VideoCapture(str(manifest.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {manifest.video}")
    end_frame = manifest.end_frame
    if end_frame is None:
        end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    annotations: list[dict[str, Any]] = []
    window = "Stereo GT - left then right; Enter accept, S skip, Backspace clear, Esc finish"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for frame_index in range(manifest.start_frame, end_frame + 1, interval):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            left_raw, right_raw = split_side_by_side(frame)
            left, right = calibration.rectify_pair(left_raw, right_raw)
            width = left.shape[1]
            for point in manifest.point_specs:
                selections: list[tuple[float, float]] = []

                def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
                    if event != cv2.EVENT_LBUTTONDOWN:
                        return
                    if len(selections) == 0 and x < width:
                        selections.append((float(x), float(y)))
                    elif len(selections) == 1 and x >= width:
                        selections.append((float(x - width), float(y)))

                cv2.setMouseCallback(window, on_mouse)
                while True:
                    canvas = np.hstack([left, right]).copy()
                    cv2.putText(
                        canvas,
                        f"Frame {frame_index} {point.point_id}: click LEFT then RIGHT",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if selections:
                        cv2.drawMarker(
                            canvas,
                            tuple(int(round(v)) for v in selections[0]),
                            (0, 255, 255),
                            cv2.MARKER_CROSS,
                            18,
                            2,
                        )
                    if len(selections) == 2:
                        right_display = (
                            int(round(selections[1][0] + width)),
                            int(round(selections[1][1])),
                        )
                        cv2.drawMarker(
                            canvas,
                            right_display,
                            (0, 255, 255),
                            cv2.MARKER_CROSS,
                            18,
                            2,
                        )
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (8, 127):
                        selections.clear()
                    elif key in (10, 13) and len(selections) == 2:
                        annotations.append(
                            enrich_annotation_with_geometry(
                            {
                                "frame": frame_index,
                                "point_id": point.point_id,
                                "left_x": selections[0][0],
                                "left_y": selections[0][1],
                                "right_x": selections[1][0],
                                "right_y": selections[1][1],
                            },
                            calibration.rectification().q,
                            calibration.unit,
                            )
                        )
                        break
                    elif key in (ord("s"), ord("S")):
                        break
                    elif key == 27:
                        return write_ground_truth_csv(output, annotations)
    finally:
        capture.release()
        cv2.destroyWindow(window)
    return write_ground_truth_csv(output, annotations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate sparse rectified left/right correspondences"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(annotate_sparse_correspondences(args.manifest, args.interval, args.output))


if __name__ == "__main__":
    main()
