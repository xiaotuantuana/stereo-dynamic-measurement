from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RESULT_FIELDS = [
    "sample_id", "dataset", "sequence", "frame_id", "point_id", "method", "valid",
    "status", "evaluable", "predicted_disparity", "raw_disparity", "refined_disparity",
    "final_disparity", "gt_disparity", "disparity_error", "confidence", "confidence_source",
    "texture_std", "match_cost", "uniqueness_margin", "lr_error_px", "subpixel_offset",
    "search_range", "candidate_count", "initial_best_disparity", "initial_disparity_disagreement",
    "left_x", "left_y", "runtime_ms", "failure_reason", "left_path", "right_path",
]


class ResultWriter:
    def __init__(self, run_dir: str | Path, *, resume: bool) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self.checkpoint_path = self.run_dir / "completed_samples.txt"
        if not resume:
            for path in (self.jsonl_path, self.csv_path, self.checkpoint_path):
                if path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing run artifact: {path}")
        self.completed = set()
        if resume and self.checkpoint_path.is_file():
            self.completed = set(self.checkpoint_path.read_text(encoding="utf-8").splitlines())
        csv_exists = self.csv_path.is_file() and self.csv_path.stat().st_size > 0
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8", newline="\n")
        self._csv = self.csv_path.open("a", encoding="utf-8-sig", newline="")
        self._checkpoint = self.checkpoint_path.open("a", encoding="utf-8", newline="\n")
        self._csv_writer = csv.DictWriter(self._csv, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not csv_exists:
            self._csv_writer.writeheader()
            self._csv.flush()

    def is_completed(self, sample_id: str) -> bool:
        return sample_id in self.completed

    def write(self, row: dict[str, Any]) -> None:
        self.write_sample(str(row["sample_id"]), [row])

    def write_sample(self, sample_id: str, rows: list[dict[str, Any]]) -> None:
        if sample_id in self.completed:
            return
        for row in rows:
            self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._csv_writer.writerow(row)
        self._jsonl.flush()
        self._csv.flush()
        self._checkpoint.write(sample_id + "\n")
        self._checkpoint.flush()
        self.completed.add(sample_id)

    def close(self) -> None:
        self._jsonl.close()
        self._csv.close()
        self._checkpoint.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
