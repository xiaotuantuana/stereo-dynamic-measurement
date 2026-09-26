from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiment.benchmark.runner import BenchmarkFilters, run_benchmark
from experiment.config import load_dataset_config
from experiment.datasets.index import build_index, iter_index
from experiment.datasets.manifests import read_manifest
from stereo_research.models import MatcherConfig


PROJECT_ROOT = Path(__file__).resolve().parent


def _resume_dir(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("run_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("--resume requested but no prior run_* directory exists")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset benchmark using TemporalStereoPipeline.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--mode", choices=("smoke", "development", "full"), default="smoke")
    parser.add_argument("--method")
    parser.add_argument("--dataset")
    parser.add_argument("--sequence")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--sample-rate", type=int, default=1)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--initial-confidence-calibration", action="store_true")
    parser.add_argument("--uniqueness-margin", type=float)
    parser.add_argument("--confidence-medium-threshold", type=float)
    parser.add_argument("--confidence-low-threshold", type=float)
    parser.add_argument("--manifest", help="Fixed JSONL manifest; bypasses index sampling.")
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    if args.rebuild_index or not config.index_path.is_file():
        build_index(config.dataset_root, config.index_path)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.resume:
        run_dir = _resume_dir(config.results_root)
    else:
        run_dir = config.results_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    samples = (
        read_manifest(args.manifest, config.dataset_root)
        if args.manifest else iter_index(config.index_path, config.dataset_root)
    )
    matcher_defaults = MatcherConfig()
    matcher_config = MatcherConfig(
        enable_initial_confidence_calibration=args.initial_confidence_calibration,
        uniqueness_margin=(
            args.uniqueness_margin
            if args.uniqueness_margin is not None
            else matcher_defaults.uniqueness_margin
        ),
        confidence_medium_threshold=(
            args.confidence_medium_threshold
            if args.confidence_medium_threshold is not None
            else matcher_defaults.confidence_medium_threshold
        ),
        confidence_low_threshold=(
            args.confidence_low_threshold
            if args.confidence_low_threshold is not None
            else matcher_defaults.confidence_low_threshold
        ),
    )
    summary = run_benchmark(
        samples,
        run_dir,
        mode="full" if args.manifest else args.mode,
        method=args.method or config.default_method,
        filters=BenchmarkFilters(
            dataset=args.dataset, sequence=args.sequence, max_frames=args.max_frames,
            start_frame=args.start_frame, end_frame=args.end_frame, sample_rate=args.sample_rate,
        ),
        smoke_per_dataset=config.smoke_per_dataset,
        development_fraction=config.development_fraction,
        development_max_frames=config.development_max_frames,
        resume=args.resume,
        project_root=PROJECT_ROOT,
        matcher_config=matcher_config,
    )
    print(f"Run directory: {run_dir}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
