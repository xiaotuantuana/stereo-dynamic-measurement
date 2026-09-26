from __future__ import annotations

from pathlib import Path

from experiment.benchmark import runner
from experiment.benchmark.runner import run_benchmark
from experiment.datasets.models import DatasetSample


def test_run_writes_reproducible_artifacts_and_failure_csv(tmp_path: Path, monkeypatch) -> None:
    sample = DatasetSample("set-a", "static", "1", tmp_path / "left.png", tmp_path / "right.png")

    def fake_evaluate(_sample, method, config=None):
        return [{
            "sample_id": sample.sample_id, "dataset": "set-a", "sequence": "static",
            "frame_id": "1", "point_id": "p", "method": method, "valid": False,
            "status": "low_texture", "predicted_disparity": None, "gt_disparity": 4.0,
            "disparity_error": None, "confidence": 0.1, "left_x": 10.0, "left_y": 10.0,
            "runtime_ms": 5.0, "failure_reason": "low_texture",
            "left_path": str(sample.left_path), "right_path": str(sample.right_path),
        }]

    monkeypatch.setattr(runner, "evaluate_sample", fake_evaluate)
    run_dir = tmp_path / "run"
    summary = run_benchmark([sample], run_dir, project_root=tmp_path)

    assert summary["failure_rate"] == 1.0
    for name in (
        "config.yaml", "environment.json", "dataset_info.json", "metrics.csv",
        "summary.json", "errors.csv", "failure_cases.csv", "log.txt",
    ):
        assert (run_dir / name).is_file(), name
