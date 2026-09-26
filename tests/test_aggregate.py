from __future__ import annotations

import json
from pathlib import Path

from stereo_research.aggregate import aggregate_reports


def test_aggregate_reports_computes_sequence_level_paired_intervals(tmp_path: Path) -> None:
    summaries = []
    for index, (baseline, improved) in enumerate(((20.0, 15.0), (18.0, 12.0), (22.0, 16.0))):
        path = tmp_path / f"sequence_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "experiment": f"sequence-{index}",
                    "methods": {
                        "local_flow": {"false_match_rate_pct": baseline},
                        "full": {"false_match_rate_pct": improved},
                    },
                }
            ),
            encoding="utf-8",
        )
        summaries.append(path)

    aggregate = aggregate_reports(
        summaries,
        baseline="local_flow",
        improved="full",
        metric_names=("false_match_rate_pct",),
        bootstrap_samples=1000,
    )

    result = aggregate["metrics"]["false_match_rate_pct"]
    assert result["sequence_count"] == 3
    assert result["paired_difference"] > 0
    assert result["ci95_lower"] > 0
