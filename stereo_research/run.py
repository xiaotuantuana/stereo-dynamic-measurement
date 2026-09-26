from __future__ import annotations

import argparse
from dataclasses import replace

from .models import METHOD_NAMES, MatcherConfig
from .runner import run_ablation_suite, run_compensation_suite, run_confidence_suite, run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal stereo ablation methods")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_NAMES,
        default=["M0", "M1", "M2", "M3"],
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
    parser.add_argument("--disable-cycle-consistency", action="store_true")
    parser.add_argument("--disable-icgn", action="store_true")
    parser.add_argument("--disable-adaptive-filter", action="store_true")
    parser.add_argument("--disable-camera-compensation", action="store_true")
    parser.add_argument(
        "--subpixel-method",
        choices=("continuous", "parabolic", "icgn"),
        default=None,
    )
    parser.add_argument("--subpixel-step", type=float, default=0.1)
    parser.add_argument("--cycle-soft-threshold", type=float)
    parser.add_argument("--cycle-hard-threshold", type=float)
    parser.add_argument("--cycle-weight", type=float)
    parser.add_argument("--icgn-patch-size", type=int)
    parser.add_argument("--icgn-max-iterations", type=int)
    parser.add_argument("--icgn-epsilon", type=float)
    parser.add_argument("--camera-compensation-inlier-threshold-mm", type=float)
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Run the complete quality method and all one-component deletion variants",
    )
    parser.add_argument("--confidence-suite", action="store_true", help="Run M3_C0 through M3_C3")
    parser.add_argument("--compensation-suite", action="store_true", help="Run THESIS_FULL_R0 through R2")
    parser.add_argument("--camera-compensation-mode", choices=("none", "single_reference", "multi_reference_rigid"))
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
        subpixel_step=args.subpixel_step,
        enable_cycle_consistency=not args.disable_cycle_consistency,
        enable_icgn=not args.disable_icgn,
        enable_adaptive_filter=not args.disable_adaptive_filter,
        enable_camera_compensation=not args.disable_camera_compensation,
        **({"camera_compensation_mode": args.camera_compensation_mode} if args.camera_compensation_mode else {}),
    )
    optional_updates = {
        "cycle_soft_threshold_px": args.cycle_soft_threshold,
        "cycle_hard_threshold_px": args.cycle_hard_threshold,
        "cycle_weight": args.cycle_weight,
        "icgn_patch_size": args.icgn_patch_size,
        "icgn_max_iterations": args.icgn_max_iterations,
        "icgn_epsilon": args.icgn_epsilon,
        "camera_compensation_inlier_threshold_mm": (
            args.camera_compensation_inlier_threshold_mm
        ),
    }
    if args.subpixel_method is not None:
        optional_updates["subpixel_method"] = args.subpixel_method
        optional_updates["research_subpixel_method"] = args.subpixel_method
    config = replace(
        config,
        **{key: value for key, value in optional_updates.items() if value is not None},
    )
    if args.compensation_suite:
        outputs = run_compensation_suite(args.manifest, config=config, repeats=args.repeats, warmup_frames=args.warmup_frames)
    elif args.confidence_suite:
        outputs = run_confidence_suite(args.manifest, config=config, repeats=args.repeats, warmup_frames=args.warmup_frames)
    elif args.ablation_suite:
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
