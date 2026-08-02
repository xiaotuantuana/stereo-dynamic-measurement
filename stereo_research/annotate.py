from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2

from .runner import load_calibration, split_side_by_side


def write_points_file(
    output: str | Path,
    frame: int,
    points: Iterable[tuple[float, float]],
) -> Path:
    output_path = Path(output)
    coordinates = [(float(x), float(y)) for x, y in points]
    if not coordinates:
        raise ValueError("At least one point is required")
    payload = {
        "frame": int(frame),
        "points": [
            {"id": f"P{index}", "x": x, "y": y}
            for index, (x, y) in enumerate(coordinates, start=1)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def annotate_video(
    video: str | Path,
    start_frame: int,
    output: str | Path,
    calibration: str = "builtin_640x480",
) -> Path:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {start_frame} from {video}")
    left, right = split_side_by_side(frame)
    left_rectified, _ = load_calibration(calibration).rectify_pair(left, right)
    points: list[tuple[float, float]] = []
    window_name = "Rectified left image - click points, Backspace undo, Enter save, Esc cancel"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        display = left_rectified.copy()
        for index, (x, y) in enumerate(points, start=1):
            center = (int(round(x)), int(round(y)))
            cv2.drawMarker(display, center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                display,
                f"P{index}",
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (8, 127) and points:
            points.pop()
        elif key in (10, 13):
            if points:
                break
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Annotation cancelled")
    cv2.destroyWindow(window_name)
    return write_points_file(output, start_frame, points)


def _parse_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in value.split(";"):
        x_text, y_text = item.split(",", 1)
        points.append((float(x_text), float(y_text)))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate initial points on a rectified left frame")
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, default=250)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calibration",
        default="builtin_640x480",
        help="builtin_640x480 or a calibration JSON file",
    )
    parser.add_argument(
        "--points",
        help='Non-interactive coordinates, for example "100,80;200,150"',
    )
    args = parser.parse_args()
    if args.points:
        path = write_points_file(args.output, args.start_frame, _parse_points(args.points))
    else:
        path = annotate_video(
            args.video,
            args.start_frame,
            args.output,
            calibration=args.calibration,
        )
    print(path)


if __name__ == "__main__":
    main()
