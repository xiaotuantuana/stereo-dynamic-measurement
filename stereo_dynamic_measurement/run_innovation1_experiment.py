from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .innovation1.baseline_experiment import BaselineExperimentConfig, run_baseline_distance_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run innovation-1 baseline-distance precision experiment.")
    parser.add_argument("--config", default="configs/innovation1_experiment.yaml")
    args = parser.parse_args()
    source = Path(args.config).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    values = dict(payload.get("baseline_experiment", payload))
    output = Path(values.pop("output_dir", "../outputs/innovation1"))
    if not output.is_absolute(): output = source.parent / output
    for key in ("baselines_mm", "distances_mm", "disparity_noises_px"):
        if key in values: values[key] = tuple(float(value) for value in values[key])
    result = run_baseline_distance_experiment(BaselineExperimentConfig(**values), output)
    print(f"csv: {result.csv_path}")
    print(f"plot: {result.plot_path}")
    print(f"policy_csv: {result.policy_csv_path}")


if __name__ == "__main__":
    main()
