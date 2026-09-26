from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from stereo_dynamic_measurement.benchmark_faults import run_fault_benchmark


def run_innovation3_controlled_validation(
    output_dir: str | Path,
    *,
    seeds: int = 20,
) -> dict[str, float | int]:
    output = Path(output_dir)
    summary = run_fault_benchmark(
        output_dir=output_dir,
        seeds=seeds,
        severities=("mild", "medium", "severe"),
    )
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    (output / "run_metadata.json").write_text(json.dumps({
        "experiment": "Innovation3 CONTROLLED severity validation",
        "generated_at": datetime.now(timezone.utc).isoformat(), "seeds": seeds,
        "severities": ["mild", "medium", "severe"], "git_commit": git_commit,
        "git_dirty": git_dirty, "python": sys.version, "platform": platform.platform(),
        "runtime_labels_visible": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
