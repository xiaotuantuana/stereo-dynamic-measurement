from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stereo_research.models import FramePointResult, MatcherConfig, MethodName, PointSpec
from stereo_research.pipeline import TemporalStereoPipeline

from ..datasets.io import read_image, read_pfm
from .metrics import summarize_sequence_rows
from .models import DatasetSequence


SEQUENCE_RESULT_FIELDS = [
    "provenance", "sequence_id", "sequence_frame_index", "sample_id", "frame_id",
    "timestamp", "point_id", "method", "valid", "status", "evaluable",
    "raw_disparity", "predicted_disparity", "refined_disparity", "final_disparity",
    "gt_disparity", "disparity_error", "confidence", "confidence_source",
    "search_range", "candidate_count", "texture_score", "match_score",
    "uniqueness_score", "lr_consistency", "fb_consistency", "temporal_prediction",
    "local_disparity_gradient", "initial_disparity_disagreement",
    "c_phy", "c_phy_valid", "r_frequency", "r_phase", "r_coherence",
    "raw_depth", "corrected_depth", "diagnosis", "failure_reason", "runtime_ms",
    "left_x", "left_y", "left_path", "right_path",
]


def _gt_at(result: FramePointResult, frame_path: Path | None, scale: float) -> float | None:
    if frame_path is None or result.left_x is None or result.left_y is None:
        return None
    gt = read_pfm(frame_path) * scale
    x, y = int(round(result.left_x)), int(round(result.left_y))
    if not (0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]):
        return None
    value = float(gt[y, x])
    return value if np.isfinite(value) and value > 0 else None


def _result_row(
    sequence: DatasetSequence,
    frame_index: int,
    result: FramePointResult,
    runtime_ms: float,
) -> dict[str, object]:
    frame = sequence.frames[frame_index]
    final = result.disparity
    if final is None:
        final = result.measured_disparity
    gt = _gt_at(result, frame.disparity_gt_path, frame.disparity_scale)
    valid = result.status == "valid" and final is not None
    evaluable = valid and gt is not None
    return {
        "provenance": sequence.provenance,
        "sequence_id": sequence.sequence_id,
        "sequence_frame_index": frame_index,
        "sample_id": frame.sample_id,
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "point_id": result.point_id,
        "method": result.method,
        "valid": valid,
        "status": result.status,
        "evaluable": evaluable,
        "raw_disparity": result.raw_disparity,
        "predicted_disparity": result.predicted_disparity,
        "refined_disparity": result.measured_disparity,
        "final_disparity": final,
        "gt_disparity": gt,
        "disparity_error": abs(float(final) - gt) if evaluable else None,
        "confidence": result.confidence,
        "confidence_source": result.confidence_source,
        "search_range": result.used_search_radius,
        "candidate_count": result.candidate_count,
        "texture_score": result.texture_std,
        "match_score": result.match_cost,
        "uniqueness_score": result.uniqueness_margin_value,
        "lr_consistency": result.lr_error_px,
        "fb_consistency": result.flow_fb_error_px,
        "temporal_prediction": result.predicted_disparity,
        "local_disparity_gradient": result.neighbor_disparity_mad,
        "initial_disparity_disagreement": result.initial_disparity_disagreement,
        "c_phy": result.c_phy,
        "c_phy_valid": result.c_phy_valid,
        "r_frequency": result.r_frequency,
        "r_phase": result.r_phase,
        "r_coherence": result.r_coherence,
        "raw_depth": result.measured_z_m,
        "corrected_depth": result.candidate_corrected_z_m,
        "diagnosis": result.fault_class,
        "failure_reason": "" if valid else result.status,
        "runtime_ms": runtime_ms,
        "left_x": result.left_x,
        "left_y": result.left_y,
        "left_path": str(frame.left_path),
        "right_path": str(frame.right_path),
    }


def run_sequences(
    sequences: Iterable[DatasetSequence],
    run_dir: str | Path,
    *,
    method: MethodName,
    q: np.ndarray,
    calibration_unit: str,
    points: tuple[PointSpec, ...],
    matcher_config: MatcherConfig | None = None,
    pipeline_factory: Callable[..., Any] = TemporalStereoPipeline,
    image_loader: Callable[[Path], np.ndarray] = read_image,
) -> dict[str, object]:
    chosen = list(sequences)
    if not chosen:
        raise ValueError("at least one sequence is required")
    if not points:
        raise ValueError("at least one point is required")
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    resolved_config = matcher_config or MatcherConfig()
    metadata = {
        "method": method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "calibration_unit": calibration_unit,
        "q": np.asarray(q, dtype=float).tolist(),
        "points": [asdict(point) for point in points],
        "matcher_config": asdict(resolved_config),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "sequence_count": len(chosen),
        "sequences": [
            {
                "sequence_id": sequence.sequence_id,
                "provenance": sequence.provenance,
                "metadata": sequence.metadata,
                "frames": [
                    {
                        "frame_id": frame.frame_id, "timestamp": frame.timestamp,
                        "left_path": str(frame.left_path), "right_path": str(frame.right_path),
                        "disparity_gt_path": None if frame.disparity_gt_path is None else str(frame.disparity_gt_path),
                    }
                    for frame in sequence.frames
                ],
            }
            for sequence in chosen
        ],
    }
    (output / "sequence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, object]] = []
    lifecycle_rows: list[dict[str, object]] = []
    initialize_calls = step_calls = 0
    for sequence in chosen:
        pipeline = pipeline_factory(method, q, calibration_unit, resolved_config)
        sequence_initialize = sequence_steps = 0
        for frame_index, frame in enumerate(sequence.frames):
            left, right = image_loader(frame.left_path), image_loader(frame.right_path)
            started = time.perf_counter()
            if frame_index == 0:
                results = pipeline.initialize(left, right, points, frame_index)
                sequence_initialize += 1
                initialize_calls += 1
            else:
                results = pipeline.step(left, right, frame_index)
                sequence_steps += 1
                step_calls += 1
            runtime_ms = (time.perf_counter() - started) * 1000.0
            rows.extend(_result_row(sequence, frame_index, result, runtime_ms) for result in results)
        lifecycle_rows.append({
            "sequence_id": sequence.sequence_id,
            "frames": len(sequence.frames),
            "initialize_calls": sequence_initialize,
            "step_calls": sequence_steps,
            "reset_equivalent": True,
        })
        del pipeline
    lifecycle = {
        "sequences": len(chosen),
        "initialize_calls": initialize_calls,
        "step_calls": step_calls,
        "reset_equivalent_calls": len(chosen),
    }
    with (output / "frame_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEQUENCE_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "frame_results.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lifecycle_payload = {
        "reset_strategy": "new_pipeline_per_sequence",
        "aggregate": lifecycle,
        "per_sequence": lifecycle_rows,
    }
    (output / "lifecycle.json").write_text(
        json.dumps(lifecycle_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary: dict[str, object] = summarize_sequence_rows(rows)
    summary.update({
        "method": method,
        "sequence_count": len(chosen),
        "provenance": sorted({sequence.provenance for sequence in chosen}),
        "lifecycle": lifecycle,
        "matcher_config": asdict(resolved_config),
    })
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
