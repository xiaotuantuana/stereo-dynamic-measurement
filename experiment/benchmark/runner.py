from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import yaml

from stereo_research.models import MatcherConfig, MethodName, PointSpec, method_profile
from stereo_research.pipeline import TemporalStereoPipeline

from ..datasets.io import read_image, read_pfm
from ..datasets.models import DatasetSample
from .metrics import summarize_rows
from .result_writer import ResultWriter


BenchmarkMode = Literal["smoke", "development", "full"]


@dataclass(frozen=True)
class BenchmarkFilters:
    dataset: str | None = None
    sequence: str | None = None
    max_frames: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    sample_rate: int = 1

    def __post_init__(self) -> None:
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if self.start_frame is not None and self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError("end_frame must not precede start_frame")


def _frame_number(sample: DatasetSample) -> int | None:
    try:
        return int(sample.frame_id)
    except ValueError:
        return None


def _evenly_spaced(items: list[DatasetSample], count: int) -> list[DatasetSample]:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(index)] for index in indices]


def select_samples(
    samples: Iterable[DatasetSample],
    mode: BenchmarkMode,
    filters: BenchmarkFilters,
    *,
    smoke_per_dataset: int = 10,
    development_fraction: float = 0.05,
    development_max_frames: int = 500,
) -> Iterator[DatasetSample]:
    if mode not in {"smoke", "development", "full"}:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    groups: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        if filters.dataset and sample.dataset_name != filters.dataset:
            continue
        if filters.sequence and sample.sequence_name != filters.sequence:
            continue
        number = _frame_number(sample)
        if filters.start_frame is not None and number is not None and number < filters.start_frame:
            continue
        if filters.end_frame is not None and number is not None and number > filters.end_frame:
            continue
        groups[sample.dataset_name].append(sample)
    selected: list[DatasetSample] = []
    for dataset_name in sorted(groups):
        items = sorted(groups[dataset_name], key=lambda item: (item.sequence_name, item.frame_id))
        items = items[::filters.sample_rate]
        if mode == "smoke":
            items = _evenly_spaced(items, smoke_per_dataset)
        elif mode == "development":
            count = min(development_max_frames, max(1, int(np.ceil(len(items) * development_fraction))))
            items = _evenly_spaced(items, count)
        selected.extend(items)
    if filters.max_frames is not None:
        selected = selected[:filters.max_frames]
    yield from selected


def _benchmark_q() -> np.ndarray:
    """Synthetic Q used only to satisfy the pipeline on disparity-only datasets."""

    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)


def _grid_points(width: int, height: int) -> tuple[PointSpec, ...]:
    xs = (0.35, 0.55, 0.75)
    ys = (0.25, 0.50, 0.75)
    return tuple(
        PointSpec(f"grid_{row}_{column}", (width * x, height * y))
        for row, y in enumerate(ys)
        for column, x in enumerate(xs)
    )


def evaluate_sample(
    sample: DatasetSample,
    method: str,
    config: MatcherConfig | None = None,
) -> list[dict[str, object]]:
    method_profile(method)
    method_name = cast(MethodName, method)
    left = read_image(sample.left_path)
    right = read_image(sample.right_path)
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"Stereo dimensions differ for {sample.sample_id}")
    gt = (
        read_pfm(sample.disparity_gt_path) * sample.disparity_scale
        if sample.disparity_gt_path else None
    )
    matcher_config = config or MatcherConfig()
    matcher_config = replace(matcher_config, min_depth_m=1e-6, max_depth_m=1e6)
    pipeline = TemporalStereoPipeline(method_name, _benchmark_q(), "m", matcher_config)
    started = time.perf_counter()
    results = pipeline.initialize(left, right, _grid_points(left.shape[1], left.shape[0]), 0)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    rows: list[dict[str, object]] = []
    for result in results:
        gt_value: float | None = None
        if gt is not None and result.left_x is not None and result.left_y is not None:
            x = int(round(result.left_x))
            y = int(round(result.left_y))
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                candidate = float(gt[y, x])
                if np.isfinite(candidate) and candidate > 0:
                    gt_value = candidate
        predicted = result.measured_disparity if result.measured_disparity is not None else result.disparity
        valid = result.status == "valid" and predicted is not None
        evaluable = valid and gt_value is not None
        rows.append({
            "sample_id": sample.sample_id,
            "dataset": sample.dataset_name,
            "sequence": sample.sequence_name,
            "frame_id": sample.frame_id,
            "point_id": result.point_id,
            "method": method,
            "valid": valid,
            "evaluable": evaluable,
            "status": result.status,
            "predicted_disparity": predicted,
            "gt_disparity": gt_value,
            "raw_disparity": result.raw_disparity,
            "refined_disparity": result.measured_disparity,
            "final_disparity": predicted,
            "disparity_error": abs(float(predicted) - gt_value) if evaluable else None,
            "confidence": result.confidence,
            "confidence_source": result.confidence_source,
            "texture_std": result.texture_std,
            "match_cost": result.match_cost,
            "uniqueness_margin": result.uniqueness_margin_value,
            "lr_error_px": result.lr_error_px,
            "subpixel_offset": result.subpixel_offset,
            "search_range": result.used_search_radius,
            "candidate_count": result.candidate_count,
            "initial_best_disparity": result.initial_best_disparity,
            "initial_disparity_disagreement": result.initial_disparity_disagreement,
            "left_x": result.left_x,
            "left_y": result.left_y,
            "runtime_ms": runtime_ms,
            "failure_reason": "" if valid else result.status,
            "left_path": str(sample.left_path),
            "right_path": str(sample.right_path),
        })
    return rows


def _git_hash(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_benchmark(
    samples: Iterable[DatasetSample],
    run_dir: str | Path,
    *,
    mode: BenchmarkMode = "smoke",
    method: str = "M3",
    filters: BenchmarkFilters | None = None,
    smoke_per_dataset: int = 10,
    development_fraction: float = 0.05,
    development_max_frames: int = 500,
    resume: bool = False,
    project_root: str | Path | None = None,
    matcher_config: MatcherConfig | None = None,
) -> dict[str, object]:
    output = Path(run_dir)
    root = Path(project_root or Path.cwd()).resolve()
    chosen = list(select_samples(
        samples, mode, filters or BenchmarkFilters(),
        smoke_per_dataset=smoke_per_dataset,
        development_fraction=development_fraction,
        development_max_frames=development_max_frames,
    ))
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": mode,
        "method": method,
        "selected_samples": len(chosen),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_hash(root),
        "matcher_config": asdict(matcher_config or MatcherConfig()),
    }
    (output / "config.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    dataset_info = {
        "sample_count": len(chosen),
        "datasets": sorted({sample.dataset_name for sample in chosen}),
        "sequences": sorted({f"{sample.dataset_name}/{sample.sequence_name}" for sample in chosen}),
        "disparity_gt_samples": sum(sample.disparity_gt_path is not None for sample in chosen),
        "depth_gt_samples": sum(sample.depth_gt_path is not None for sample in chosen),
        "calibrated_samples": sum(sample.calibration_path is not None for sample in chosen),
    }
    (output / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    log_path = output / "log.txt"
    with log_path.open("a" if resume else "w", encoding="utf-8", newline="\n") as log, ResultWriter(output, resume=resume) as writer:
        log.write(f"{datetime.now(timezone.utc).isoformat()} start mode={mode} method={method} samples={len(chosen)}\n")
        log.flush()
        for index, sample in enumerate(chosen, start=1):
            if writer.is_completed(sample.sample_id):
                log.write(f"skip completed {sample.sample_id}\n")
                continue
            try:
                rows = evaluate_sample(sample, method, matcher_config)
            except Exception as exc:
                rows = [{
                    "sample_id": sample.sample_id, "dataset": sample.dataset_name,
                    "sequence": sample.sequence_name, "frame_id": sample.frame_id,
                    "point_id": "", "method": method, "valid": False, "status": "exception",
                    "predicted_disparity": None, "gt_disparity": None, "disparity_error": None,
                    "confidence": 0.0, "left_x": None, "left_y": None, "runtime_ms": 0.0,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "left_path": str(sample.left_path), "right_path": str(sample.right_path),
                }]
            writer.write_sample(sample.sample_id, rows)
            valid_count = sum(bool(row.get("valid")) for row in rows)
            log.write(f"complete {sample.sample_id} valid={valid_count}/{len(rows)}\n")
            log.flush()
            print(f"[{index}/{len(chosen)}] {sample.sample_id}")
    all_rows = _read_rows(output / "metrics.jsonl")
    summary = summarize_rows(all_rows)
    summary.update(metadata)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures = sorted(
        (row for row in all_rows if not row.get("valid")),
        key=lambda row: float(row.get("disparity_error") or -1.0),
        reverse=True,
    )
    failure_fields = [
        "dataset", "sequence", "frame_id", "point_id", "disparity_error", "confidence",
        "status", "failure_reason", "left_path", "right_path",
    ]
    with (output / "failure_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures[:300])
    error_rows = sorted(
        (row for row in all_rows if row.get("disparity_error") is not None),
        key=lambda row: float(row["disparity_error"]),
        reverse=True,
    )
    error_fields = [
        "sample_id", "dataset", "sequence", "frame_id", "point_id",
        "predicted_disparity", "gt_disparity", "disparity_error", "confidence",
        "status", "left_path", "right_path",
    ]
    with (output / "errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(error_rows)
    return summary
