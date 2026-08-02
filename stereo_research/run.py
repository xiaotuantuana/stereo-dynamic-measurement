from __future__ import annotations

import argparse
from dataclasses import replace

from .models import METHOD_NAMES, MatcherConfig
from .runner import run_ablation_suite, run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal stereo ablation methods")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_NAMES,
        default=["sgbm", "local", "local_flow", "full"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--disable-prediction", action="store_true")
    parser.add_argument("--disable-flow", action="store_true")
    parser.add_argument("--disable-epipolar", action="store_true")
    parser.add_argument("--disable-neighborhood", action="store_true")
    parser.add_argument("--disable-subpixel", action="store_true")
    parser.add_argument("--disable-lr-check", action="store_true")
    parser.add_argument("--disable-pyramid", action="store_true")
    parser.add_argument("--disable-recovery", action="store_true")
    parser.add_argument("--disable-temporal-estimation", action="store_true")
    parser.add_argument(
        "--subpixel-method",
        choices=("continuous", "parabolic"),
        default="continuous",
    )
    parser.add_argument("--subpixel-step", type=float, default=0.1)
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Run the complete quality method and all one-component deletion variants",
    )
    args = parser.parse_args()
    config = MatcherConfig()
    config = replace(
        config,
        enable_prediction=not args.disable_prediction,
        enable_flow=not args.disable_flow,
        enable_epipolar=not args.disable_epipolar,
        enable_neighborhood=not args.disable_neighborhood,
        enable_subpixel=not args.disable_subpixel,
        enable_lr_check=not args.disable_lr_check,
        enable_pyramid=not args.disable_pyramid,
        enable_recovery=not args.disable_recovery,
        enable_temporal_estimation=not args.disable_temporal_estimation,
        subpixel_method=args.subpixel_method,
        subpixel_step=args.subpixel_step,
    )
    if args.ablation_suite:
        outputs = run_ablation_suite(
            args.manifest,
            config=config,
            repeats=args.repeats,
            warmup_frames=args.warmup_frames,
        )
    else:
        outputs = run_manifest(
            args.manifest,
            methods=args.methods,
            config=config,
            repeats=args.repeats,
            warmup_frames=args.warmup_frames,
        )
    for method, path in outputs.items():
        print(f"{method}: {path}")


if __name__ == "__main__":
    main()
