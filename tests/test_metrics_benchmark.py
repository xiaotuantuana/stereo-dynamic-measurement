from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment.benchmark.metrics import disparity_metrics, summarize_rows
from experiment.benchmark.runner import BenchmarkFilters, select_samples
from experiment.datasets.models import DatasetSample


def _sample(frame: int, dataset: str = "set-a") -> DatasetSample:
    return DatasetSample(
        dataset_name=dataset,
        sequence_name="static",
        frame_id=f"{frame:07d}",
        left_path=Path(f"left-{frame}.png"),
        right_path=Path(f"right-{frame}.png"),
    )


def test_disparity_metrics_include_bad_pixel_rates() -> None:
    metrics = disparity_metrics(
        np.array([10.0, 10.0, 10.0, 10.0]),
        np.array([10.0, 11.5, 12.5, 14.0]),
    )

    assert metrics["disparity_mae"] == 2.0
    assert metrics["disparity_rmse"] == np.sqrt(6.125)
    assert metrics["bad_1"] == 0.75
    assert metrics["bad_2"] == 0.5
    assert metrics["bad_3"] == 0.25


def test_smoke_selection_is_deterministic_and_bounded_per_dataset() -> None:
    samples = [_sample(index, dataset) for dataset in ("a", "b") for index in range(40)]

    selected = list(select_samples(samples, "smoke", BenchmarkFilters(), smoke_per_dataset=10))

    assert len(selected) == 20
    assert [item.sample_id for item in selected] == [
        item.sample_id
        for item in select_samples(samples, "smoke", BenchmarkFilters(), smoke_per_dataset=10)
    ]


def test_summary_counts_invalid_results_without_hiding_them() -> None:
    summary = summarize_rows([
        {"valid": True, "disparity_error": 1.0, "runtime_ms": 10.0},
        {"valid": False, "disparity_error": None, "runtime_ms": 30.0},
    ])

    assert summary["valid_rate"] == 0.5
    assert summary["failure_rate"] == 0.5
    assert summary["disparity_mae"] == 1.0
    assert summary["runtime_median_ms"] == 20.0

