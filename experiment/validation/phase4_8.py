"""Phase 4.8A: one fixed offline P0/P1 predictor comparison only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stereo_research.final_arbitration import ExperimentAuthority, I1BaselineView
from stereo_research.pipeline import TemporalStereoPipeline

from ..datasets.io import read_image
from ..datasets.models import DatasetSample
from ..sequence.ablation import _controlled_q
from ..simulation.image_anomaly_injector import ImageAnomalyInjector
from .phase4_5 import _POINTS, _load_verified_phase3_source, build_phase4_5_case_windows
from .phase4_6 import phase4_6_subset
from .phase4_7 import _confirmed_anomaly, _distance, _gt_xyz_m, _phase, classify_candidate_outcome


XYZ = tuple[float, float, float]


def persistence_prediction_xyz_mm(trusted_xyz_mm: list[XYZ]) -> XYZ | None:
    """P1: the latest trusted coordinate, with no velocity extrapolation."""

    return None if not trusted_xyz_mm else tuple(float(value) for value in trusted_xyz_mm[-1])


def evaluate_persistence_acceptance(trace: pd.DataFrame) -> dict[str, object]:
    """Evaluate the four user-frozen P1 adoption conditions."""

    injected = trace.loc[trace["arm"] == "injected"]
    clean = trace.loc[trace["arm"] == "clean"]
    p0_harmful = int((injected["p0_outcome"] == "HARMFUL").sum())
    p1_harmful = int((injected["p1_outcome"] == "HARMFUL").sum())
    p0_beneficial = int((injected["p0_outcome"] == "BENEFICIAL").sum())
    p1_beneficial = int((injected["p1_outcome"] == "BENEFICIAL").sum())
    p0_clean_harmful = int((clean["p0_outcome"] == "HARMFUL").sum())
    p1_clean_harmful = int((clean["p1_outcome"] == "HARMFUL").sum())
    p0_median = float(injected["p0_improvement_mm"].median())
    p1_median = float(injected["p1_improvement_mm"].median())
    harmful_reduction = p0_harmful - p1_harmful
    conditions = {
        "injected_harmful_reduced_by_two": harmful_reduction >= 2,
        "beneficial_not_lower": p1_beneficial >= p0_beneficial,
        "median_improvement_better": p1_median > p0_median,
        "clean_harmful_not_increased": p1_clean_harmful <= p0_clean_harmful,
    }
    return {
        "accepted": bool(all(conditions.values())),
        "injected_harmful_reduction": harmful_reduction,
        "p0_harmful": p0_harmful,
        "p1_harmful": p1_harmful,
        "p0_beneficial": p0_beneficial,
        "p1_beneficial": p1_beneficial,
        "p0_median_improvement_mm": p0_median,
        "p1_median_improvement_mm": p1_median,
        "p0_clean_harmful": p0_clean_harmful,
        "p1_clean_harmful": p1_clean_harmful,
        **conditions,
    }


def _run_arm(case: object, *, arm: str, frames: dict[str, DatasetSample]) -> list[dict[str, object]]:
    injector = None if arm == "clean" else ImageAnomalyInjector(case.injected)
    pipeline = TemporalStereoPipeline(
        "M3", _controlled_q(), "m", experiment_authority=ExperimentAuthority.shadow()
    )
    q = _controlled_q()
    disparity_cache: dict[Path, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for local_index, frame_id in enumerate(case.window_frame_ids):
        sample = frames[frame_id]
        source_left, source_right = read_image(sample.left_path), read_image(sample.right_path)
        left, right = source_left.copy(), source_right.copy()
        if injector is not None:
            left = injector.transform(source_left, frame_id=frame_id, side="left")
            right = injector.transform(source_right, frame_id=frame_id, side="right")
        results = (
            pipeline.initialize(left, right, _POINTS, int(frame_id))
            if local_index == 0 else pipeline.step(left, right, int(frame_id))
        )
        for result in results:
            baseline = I1BaselineView.from_result(result)
            prediction_m = (
                None if result.i2_prediction_x_m is None or result.i2_prediction_y_m is None or result.i2_prediction_z_m is None
                else (float(result.i2_prediction_x_m), float(result.i2_prediction_y_m), float(result.i2_prediction_z_m))
            )
            if (
                baseline.xyz_m is None
                or prediction_m is None
                or not result.i2_correction_applied
                or not result.candidate_safe
            ):
                continue
            processor = pipeline.enhanced_processors[result.point_id]
            p1_mm = persistence_prediction_xyz_mm([item.xyz_mm for item in processor.trusted_history])
            if p1_mm is None:
                continue
            gt_m = _gt_xyz_m(result, sample, q, disparity_cache)
            if gt_m is None:
                continue
            raw_mm = tuple(value * 1000.0 for value in baseline.xyz_m)
            p0_mm = tuple(value * 1000.0 for value in prediction_m)
            gt_mm = tuple(value * 1000.0 for value in gt_m)
            raw_error = _distance(raw_mm, gt_mm)
            p0_error = _distance(p0_mm, gt_mm)
            p1_error = _distance(p1_mm, gt_mm)
            rows.append({
                "case_id": case.injected.case_id,
                "pair_id": case.injected.pair_id,
                "subset": phase4_6_subset(case.injected.case_id),
                "arm": arm,
                "anomaly_type": case.injected.anomaly_type,
                "frame_index": local_index,
                "frame_phase": _phase(case, frame_id),
                "point_id": result.point_id,
                "raw_i1_xyz_mm": json.dumps(raw_mm),
                "p0_constant_velocity_xyz_mm": json.dumps(p0_mm),
                "p1_persistence_xyz_mm": json.dumps(p1_mm),
                "gt_xyz_mm": json.dumps(gt_mm),
                "raw_baseline_error_mm": raw_error,
                "p0_error_mm": p0_error,
                "p1_error_mm": p1_error,
                "p0_improvement_mm": raw_error - p0_error,
                "p1_improvement_mm": raw_error - p1_error,
                "p0_outcome": classify_candidate_outcome(raw_error_mm=raw_error, candidate_error_mm=p0_error),
                "p1_outcome": classify_candidate_outcome(raw_error_mm=raw_error, candidate_error_mm=p1_error),
            })
    return rows


def run_phase4_8a_offline_analysis(
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
    tests_passed: bool,
) -> dict[str, object]:
    """Run development-only P0/P1 comparison without changing runtime logic."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = _load_verified_phase3_source(manifest_path)
    frames = {frame.frame_id: frame for frame in source.frames}
    rows: list[dict[str, object]] = []
    for case in build_phase4_5_case_windows(manifest_path):
        if phase4_6_subset(case.injected.case_id) != "development":
            continue
        for arm in ("clean", "injected"):
            rows.extend(_run_arm(case, arm=arm, frames=frames))
    trace = pd.DataFrame(rows)
    trace.to_csv(output / "development_p0_p1_predictor_trace.csv", index=False, encoding="utf-8-sig")
    decision = evaluate_persistence_acceptance(trace)
    pd.DataFrame([decision]).to_csv(output / "p0_p1_acceptance_summary.csv", index=False, encoding="utf-8-sig")
    full_write = "NO"
    shadow = "YES" if tests_passed else "NO"
    predictor_status = "P1_ACCEPTED" if decision["accepted"] else "I2_SIMPLE_TEMPORAL_PREDICTOR_INSUFFICIENT"
    report = "\n".join([
        "# Phase 4.8A Simple Predictor Decision",
        "",
        f"`P1_ACCEPTED = {'YES' if decision['accepted'] else 'NO'}`",
        f"`{predictor_status}`",
        f"`ALLOW_2000_FULL_WRITE = {full_write}`",
        f"`ALLOW_2000_SHADOW_BENCHMARK = {shadow}`",
        "",
        "## Frozen adoption conditions",
        "",
        f"- injected harmful reduction: {decision['injected_harmful_reduction']} (requires >= 2): {decision['injected_harmful_reduced_by_two']}",
        f"- beneficial P0/P1: {decision['p0_beneficial']}/{decision['p1_beneficial']}: {decision['beneficial_not_lower']}",
        f"- injected median improvement P0/P1: {decision['p0_median_improvement_mm']:.6f}/{decision['p1_median_improvement_mm']:.6f}: {decision['median_improvement_better']}",
        f"- clean harmful P0/P1: {decision['p0_clean_harmful']}/{decision['p1_clean_harmful']}: {decision['clean_harmful_not_increased']}",
        "",
        "No P2/P3, window search, threshold tuning, holdout selection, or production predictor change was performed.",
        "",
        "## Verdict",
        "",
        predictor_status,
    ])
    (output / "PHASE4_8_SIMPLE_PREDICTOR_REPORT.md").write_text(report + "\n", encoding="utf-8")
    return {
        "p1_accepted": bool(decision["accepted"]),
        "allow_2000_full_write": full_write,
        "allow_2000_shadow_benchmark": shadow,
        "predictor_status": predictor_status,
        **decision,
    }
