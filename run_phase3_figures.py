from pathlib import Path

from experiment.reporting.phase3_figures import generate_phase3_figures


if __name__ == "__main__":
    generate_phase3_figures(
        output_dir="results/phase3/figures",
        innovation1_csv=Path("results/phase3/stateful_innovation1_500/ablation_innovation1.csv"),
        innovation2_csv=Path("results/phase3/innovation2_controlled/raw_vs_corrected.csv"),
        innovation3_confusion=Path("results/phase3/innovation3_controlled/confusion_matrix.csv"),
        occlusion_csv=Path("results/phase3/occlusion_diagnostics/occlusion_discontinuity_diagnostics.csv"),
    )
