from __future__ import annotations

import pandas as pd

from stereo_dynamic_measurement.innovation1.baseline_experiment import BaselineExperimentConfig, run_baseline_distance_experiment


def test_baseline_distance_experiment_exports_all_requested_combinations(tmp_path) -> None:
    result = run_baseline_distance_experiment(
        BaselineExperimentConfig(samples_per_condition=100, random_seed=5), tmp_path
    )
    table = pd.read_csv(result.csv_path)

    assert len(table) == 7 * 5 * 4
    assert {"baseline_mm", "distance_mm", "disparity_noise_px", "depth_rmse_mm", "theoretical_depth_sigma_mm"} <= set(table.columns)
    assert result.plot_path.exists()
    assert table.loc[table["distance_mm"] == 5000, "depth_rmse_mm"].mean() > table.loc[table["distance_mm"] == 1000, "depth_rmse_mm"].mean()
