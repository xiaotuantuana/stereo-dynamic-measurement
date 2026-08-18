from __future__ import annotations

from stereo_dynamic_measurement.benchmark_faults import run_fault_benchmark


def test_fault_benchmark_writes_metrics_without_using_labels_at_runtime(tmp_path) -> None:
    summary = run_fault_benchmark(output_dir=tmp_path, seeds=2)

    assert summary["case_count"] == 16
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "recovery_metrics.csv").exists()
    assert (tmp_path / "per_case_results.csv").exists()
    assert 0.0 <= summary["macro_f1"] <= 1.0
