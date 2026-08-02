from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .metrics import paired_bootstrap_ci


DEFAULT_METRICS = (
    "false_match_rate_pct",
    "jump_rate_pct",
    "runtime_median_ms",
    "xyz_rmse_m",
    "distance_rmse_m",
    "effective_tracking_rate_pct",
)


def aggregate_reports(
    summary_paths: Iterable[str | Path],
    baseline: str = "local_flow",
    improved: str = "full",
    metric_names: Iterable[str] = DEFAULT_METRICS,
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in summary_paths
    ]
    if not reports:
        raise ValueError("At least one summary report is required")
    metrics: dict[str, object] = {}
    for metric_name in metric_names:
        baseline_values: list[float] = []
        improved_values: list[float] = []
        sequences: list[str] = []
        for report in reports:
            methods = report.get("methods", {})
            baseline_value = methods.get(baseline, {}).get(metric_name)
            improved_value = methods.get(improved, {}).get(metric_name)
            if baseline_value is None or improved_value is None:
                continue
            baseline_values.append(float(baseline_value))
            improved_values.append(float(improved_value))
            sequences.append(str(report.get("experiment", "")))
        if len(baseline_values) < 2:
            metrics[metric_name] = {
                "available": False,
                "sequence_count": len(baseline_values),
                "reason": "At least two paired sequence summaries are required",
            }
            continue
        lower, estimate, upper = paired_bootstrap_ci(
            baseline_values,
            improved_values,
            samples=bootstrap_samples,
        )
        metrics[metric_name] = {
            "available": True,
            "sequence_count": len(baseline_values),
            "sequences": sequences,
            "baseline_mean": sum(baseline_values) / len(baseline_values),
            "improved_mean": sum(improved_values) / len(improved_values),
            "paired_difference": estimate,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "difference_definition": f"{baseline} - {improved}",
        }
    return {
        "baseline": baseline,
        "improved": improved,
        "report_count": len(reports),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate sequence reports with paired bootstrap confidence intervals"
    )
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--baseline", default="local_flow")
    parser.add_argument("--improved", default="full")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate = aggregate_reports(
        args.summaries,
        baseline=args.baseline,
        improved=args.improved,
        metric_names=args.metrics,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
