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


def test_existing_innovation1_experiment_exports_target_accuracy_ablation(tmp_path) -> None:
    result = run_baseline_distance_experiment(
        BaselineExperimentConfig(samples_per_condition=100, random_seed=5), tmp_path
    )
    policy = pd.read_csv(result.policy_csv_path)

    assert len(policy) == 4 * 4
    assert set(policy["ablation_group"]) == {"A0_FIXED", "A1_VISION_ONLY", "A2_ACCURACY_ONLY", "A3_FULL"}
    assert set(policy["target_level"]) == {"LOOSE", "MEDIUM", "STRICT", "INFEASIBLE"}
    assert {
        "target_sigma_z_mm", "required_sigma_d_px", "estimated_sigma_d_px",
        "estimated_sigma_z_mm", "search_radius_px", "refinement_level",
        "precision_retry_count", "precision_status", "runtime_ms",
        "valid_measurement_rate", "estimated_target_attainment_rate",
        "gt_depth_rmse_mm", "gt_depth_mae_mm", "gt_target_attainment_rate",
    } <= set(policy.columns)
    assert policy.loc[policy.target_level == "INFEASIBLE", "precision_status"].isin({"INFEASIBLE", "LEGACY_DISABLED"}).all()
    assert policy.loc[policy.ablation_group == "A0_FIXED", "required_sigma_d_px"].isna().all()
    assert policy.loc[policy.ablation_group == "A1_VISION_ONLY", "required_sigma_d_px"].isna().all()
    accuracy_rows = policy[policy.ablation_group.isin(["A2_ACCURACY_ONLY", "A3_FULL"])]
    assert accuracy_rows["required_sigma_d_px"].notna().all()
    assert result.policy_csv_path.exists()
