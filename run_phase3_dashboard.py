from pathlib import Path

from experiment.reporting.phase3_dashboard import build_phase3_dashboard


if __name__ == "__main__":
    build_phase3_dashboard(
        output_path="results/phase3/experiment_summary.csv",
        innovation1_csv=Path("results/phase3/stateful_innovation1_smoke/ablation_innovation1.csv"),
        innovation2_csv=Path("results/phase3/innovation2_controlled/scenario_metrics.csv"),
        innovation3_summary=Path("results/phase3/innovation3_controlled/summary.json"),
    )
