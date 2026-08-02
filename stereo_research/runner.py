from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, cast

import cv2
import numpy as np

from .calibration import StereoCalibration, builtin_640x480
from .models import METHOD_NAMES, MatcherConfig, MethodName, SequenceManifest, method_profile
from .pipeline import TemporalStereoPipeline


CSV_FIELDS = [
    "repeat",
    "method",
    "frame",
    "point_id",
    "status",
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "disparity",
    "X_m",
    "Y_m",
    "Z_m",
    "distance_m",
    "match_cost",
    "flow_fb_error_px",
    "lr_error_px",
    "confidence",
    "raw_disparity",
    "integer_disparity",
    "measured_disparity",
    "estimated_disparity",
    "measured_right_x",
    "measured_right_y",
    "estimated_right_x",
    "estimated_right_y",
    "measured_X_m",
    "measured_Y_m",
    "measured_Z_m",
    "estimated_X_m",
    "estimated_Y_m",
    "estimated_Z_m",
    "delta_X_mm",
    "delta_Y_mm",
    "delta_Z_mm",
    "subpixel_offset",
    "neighbor_disparity",
    "used_search_radius",
    "quality_stage",
    "recovery_stage",
    "recovery_attempt_count",
    "recovery_success_count",
    "mean_recovery_frames",
    "flow_ms",
    "matching_ms",
    "total_ms",
    "frame_total_ms",
]


def split_side_by_side(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame.ndim not in (2, 3) or frame.shape[1] % 2 != 0:
        raise ValueError("Stereo frame must have an even width and contain side-by-side views")
    half_width = frame.shape[1] // 2
    return frame[:, :half_width].copy(), frame[:, half_width:].copy()


def load_calibration(value: str) -> StereoCalibration:
    if value == "builtin_640x480":
        return builtin_640x480()
    path = Path(value)
    if path.exists():
        return StereoCalibration.from_json(path)
    raise ValueError(
        f"Unsupported calibration {value!r}; use builtin_640x480 or a calibration JSON file"
    )


def run_manifest(
    manifest_path: str | Path,
    methods: Iterable[str] = ("sgbm", "local", "local_flow", "full"),
    config: MatcherConfig | None = None,
    repeats: int = 1,
    warmup_frames: int = 30,
) -> dict[str, Path]:
    cv2.setNumThreads(1)
    manifest = SequenceManifest.from_json(manifest_path)
    if not manifest.video.exists():
        raise FileNotFoundError(f"Video does not exist: {manifest.video}")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    method_names: list[MethodName] = []
    for method in methods:
        method_profile(method)
        method_names.append(cast(MethodName, method))
    calibration_value = manifest.calibration
    if calibration_value != "builtin_640x480":
        calibration_path = Path(calibration_value)
        if not calibration_path.is_absolute():
            calibration_value = str((Path(manifest_path).resolve().parent / calibration_path).resolve())
    calibration = load_calibration(calibration_value)
    matcher_config = config or MatcherConfig()
    output_dir = manifest.output_dir or manifest.points_path.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    run_metadata: dict[str, object] = {
        "sequence": manifest.name,
        "video": str(manifest.video),
        "start_frame": manifest.start_frame,
        "end_frame": manifest.end_frame,
        "calibration": calibration.name,
        "methods": method_names,
        "repeats": repeats,
        "warmup_frames": warmup_frames,
        "config": asdict(matcher_config),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "opencv_threads": 1,
    }
    for method in method_names:
        rows: list[dict[str, object]] = []
        for repeat in range(repeats):
            rows.extend(
                _run_method_once(
                    manifest,
                    calibration,
                    matcher_config,
                    method,
                    repeat,
                    warmup_frames,
                )
            )
        output_path = output_dir / f"{method}.csv"
        with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        outputs[method] = output_path
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outputs


def run_ablation_suite(
    manifest_path: str | Path,
    config: MatcherConfig | None = None,
    repeats: int = 1,
    warmup_frames: int = 30,
) -> dict[str, Path]:
    """Run the complete method and one-component deletion variants."""
    cv2.setNumThreads(1)
    manifest = SequenceManifest.from_json(manifest_path)
    if not manifest.video.exists():
        raise FileNotFoundError(f"Video does not exist: {manifest.video}")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    calibration_value = manifest.calibration
    if calibration_value != "builtin_640x480":
        calibration_path = Path(calibration_value)
        if not calibration_path.is_absolute():
            calibration_value = str(
                (Path(manifest_path).resolve().parent / calibration_path).resolve()
            )
    calibration = load_calibration(calibration_value)
    base = config or MatcherConfig()
    variants = {
        "full": base,
        "full_no_flow": replace(base, enable_flow=False),
        "full_no_prediction": replace(base, enable_prediction=False),
        "full_no_epipolar": replace(base, enable_epipolar=False),
        "full_no_neighborhood": replace(base, enable_neighborhood=False),
        "full_no_subpixel": replace(base, enable_subpixel=False),
        "full_no_lr": replace(base, enable_lr_check=False),
        "full_no_pyramid": replace(base, enable_pyramid=False),
        "full_no_recovery": replace(base, enable_recovery=False),
        "full_no_temporal_estimation": replace(base, enable_temporal_estimation=False),
    }
    results_root = manifest.output_dir or manifest.points_path.parent / "results"
    output_dir = results_root / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for label, variant_config in variants.items():
        rows: list[dict[str, object]] = []
        for repeat_index in range(repeats):
            variant_rows = _run_method_once(
                manifest,
                calibration,
                variant_config,
                "full_quality",
                repeat_index,
                warmup_frames,
            )
            for row in variant_rows:
                row["method"] = label
            rows.extend(variant_rows)
        output_path = output_dir / f"{label}.csv"
        with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        outputs[label] = output_path
    metadata = {
        "sequence": manifest.name,
        "engine_method": "full_quality",
        "variants": {label: asdict(value) for label, value in variants.items()},
        "repeats": repeats,
        "warmup_frames": warmup_frames,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "ablation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outputs


def _run_method_once(
    manifest: SequenceManifest,
    calibration: StereoCalibration,
    config: MatcherConfig,
    method: MethodName,
    repeat: int,
    warmup_frames: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(manifest.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {manifest.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, manifest.start_frame)
    pipeline: TemporalStereoPipeline | None = None
    rows: list[dict[str, object]] = []
    frame_index = manifest.start_frame
    try:
        while manifest.end_frame is None or frame_index <= manifest.end_frame:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            algorithm_start = time.perf_counter()
            left_raw, right_raw = split_side_by_side(frame)
            left, right = calibration.rectify_pair(left_raw, right_raw)
            if pipeline is None:
                pipeline = TemporalStereoPipeline(
                    method=method,
                    q=calibration.rectification().q,
                    calibration_unit=calibration.unit,
                    config=config,
                )
                results = pipeline.initialize(
                    left,
                    right,
                    manifest.point_specs,
                    frame_index,
                )
            else:
                results = pipeline.step(left, right, frame_index)
            frame_total_ms = (time.perf_counter() - algorithm_start) * 1000.0
            for result in results:
                row = result.as_csv_row()
                row["repeat"] = repeat
                row["frame_total_ms"] = (
                    ""
                    if frame_index < manifest.start_frame + warmup_frames
                    else frame_total_ms
                )
                rows.append(row)
            frame_index += 1
    finally:
        capture.release()
    if pipeline is None:
        raise RuntimeError(
            f"No frames were read from {manifest.video} at frame {manifest.start_frame}"
        )
    return rows
