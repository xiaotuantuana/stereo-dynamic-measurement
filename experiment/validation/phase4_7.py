"""Phase 4.7A offline-only I2 candidate and recovery analysis.

No function in this module participates in the production pipeline or has
authority to alter a final result.  Ground truth is evaluation-only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stereo_research.final_arbitration import ExperimentAuthority, I1BaselineView
from stereo_research.models import PointSpec
from stereo_research.pipeline import TemporalStereoPipeline

from ..datasets.io import read_image, read_pfm
from ..datasets.models import DatasetSample
from ..sequence.ablation import _controlled_q
from ..simulation.image_anomaly_injector import ImageAnomalyInjector
from .phase4_5 import _POINTS, _load_verified_phase3_source, build_phase4_5_case_windows
from .phase4_6 import classify_i2_recovery, phase4_6_subset

XYZ = tuple[float, float, float]


def _distance(left: XYZ, right: XYZ) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def bounded_candidate_xyz_mm(
    *,
    raw_xyz_mm: XYZ,
    prediction_xyz_mm: XYZ,
    cap_mm: float = 25.0,
) -> tuple[XYZ, float]:
    """C1: move from raw toward prediction by at most the fixed 25 mm cap."""

    if cap_mm <= 0:
        raise ValueError("cap_mm must be positive")
    distance = _distance(raw_xyz_mm, prediction_xyz_mm)
    if distance == 0.0:
        return raw_xyz_mm, 0.0
    magnitude = min(float(cap_mm), distance)
    candidate = tuple(
        raw_xyz_mm[index]
        + (prediction_xyz_mm[index] - raw_xyz_mm[index]) * magnitude / distance
        for index in range(3)
    )
    return candidate, magnitude


def classify_candidate_outcome(*, raw_error_mm: float, candidate_error_mm: float) -> str:
    """Offline GT label; never a runtime input."""

    if candidate_error_mm < raw_error_mm - 1e-6:
        return "BENEFICIAL"
    if candidate_error_mm > raw_error_mm + 1e-6:
        return "HARMFUL"
    return "NEUTRAL"


def _xyz_from_json(value: object) -> XYZ:
    values = json.loads(str(value))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("expected a three-dimensional JSON coordinate")
    xyz = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in xyz):
        raise ValueError("coordinate must be finite")
    return xyz  # type: ignore[return-value]


def classify_stale_prediction_recovery(
    trace: pd.DataFrame,
    *,
    consistency_mm: float = 10.0,
) -> pd.DataFrame:
    """Classify two-observation recovery traces without changing I2 state.

    A stale predictor pattern requires two I1-valid, clean observations that
    agree with each other within the existing 10 mm consistency bound while
    both remain farther than that bound from the old trusted prediction.
    """

    required = {
        "case_id", "point_id", "frame_index", "i1_valid", "hard_failure",
        "confirmed_anomaly", "raw_xyz_mm", "prediction_xyz_mm",
    }
    missing = required.difference(trace.columns)
    if missing:
        raise ValueError(f"stale-recovery trace missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for (case_id, point_id), group in trace.groupby(["case_id", "point_id"], sort=True):
        group = group.sort_values("frame_index")
        if "frame_phase" in group.columns:
            group = group.loc[group["frame_phase"].isin(["recovery", "stability"])]
        clean = group.loc[
            group["i1_valid"].astype(bool)
            & ~group["hard_failure"].astype(bool)
            & ~group["confirmed_anomaly"].astype(bool)
        ].iloc[:2]
        base = {"case_id": case_id, "point_id": point_id}
        if len(clean) < 2:
            records.append({**base, "recovery_pattern": "INSUFFICIENT_CLEAN_OBSERVATIONS"})
            continue
        first, second = clean.iloc[0], clean.iloc[1]
        raw_first, raw_second = _xyz_from_json(first["raw_xyz_mm"]), _xyz_from_json(second["raw_xyz_mm"])
        old_prediction = _xyz_from_json(first["prediction_xyz_mm"])
        mutual_distance = _distance(raw_first, raw_second)
        first_old_distance = _distance(raw_first, old_prediction)
        second_old_distance = _distance(raw_second, old_prediction)
        stale = (
            mutual_distance <= consistency_mm
            and first_old_distance > consistency_mm
            and second_old_distance > consistency_mm
        )
        records.append({
            **base,
            "first_frame_index": int(first["frame_index"]),
            "second_frame_index": int(second["frame_index"]),
            "pair_distance_mm": mutual_distance,
            "first_old_prediction_distance_mm": first_old_distance,
            "second_old_prediction_distance_mm": second_old_distance,
            "recovery_pattern": "STALE_PREDICTION" if stale else "NOT_STALE",
        })
    return pd.DataFrame(records)


def _json_xyz_mm(xyz_m: tuple[float, float, float] | None) -> str | None:
    if xyz_m is None:
        return None
    values = tuple(float(value) * 1000.0 for value in xyz_m)
    return json.dumps(values)


def _prediction_xyz_m(result: object) -> tuple[float, float, float] | None:
    values = (
        result.i2_prediction_x_m,
        result.i2_prediction_y_m,
        result.i2_prediction_z_m,
    )
    if any(value is None for value in values):
        return None
    xyz = tuple(float(value) for value in values)
    return xyz if np.isfinite(np.asarray(xyz, dtype=float)).all() else None


def _gt_xyz_m(
    result: object,
    sample: DatasetSample,
    q: np.ndarray,
    disparity_cache: dict[Path, np.ndarray],
) -> tuple[float, float, float] | None:
    if result.left_x is None or result.left_y is None or sample.disparity_gt_path is None:
        return None
    gt = disparity_cache.setdefault(sample.disparity_gt_path, read_pfm(sample.disparity_gt_path))
    x, y = int(round(result.left_x)), int(round(result.left_y))
    if not (0 <= x < gt.shape[1] and 0 <= y < gt.shape[0]):
        return None
    disparity = float(gt[y, x]) * sample.disparity_scale
    if not np.isfinite(disparity) or disparity <= 0:
        return None
    projected = q @ np.asarray([float(result.left_x), float(result.left_y), disparity, 1.0])
    if projected[3] == 0 or not np.isfinite(projected).all():
        return None
    return tuple(float(value) for value in (projected[:3] / projected[3]))


def _phase(case: object, frame_id: str) -> str:
    if frame_id in case.warmup_frame_ids:
        return "warmup"
    if frame_id in case.injected.active_frame_ids:
        return "exposure"
    if frame_id in case.recovery_frame_ids:
        return "recovery"
    return "stability"


def _confirmed_anomaly(result: object, *, hard_failure: bool) -> bool:
    fault = result.fault_class.strip().upper()
    return bool(
        not hard_failure
        and fault not in {"", "NORMAL"}
        and result.c_phy_valid is True
        and result.c_phy is not None
        and result.c_phy < 0.55
    )


def _run_trace_arm(
    case: object,
    *,
    arm: str,
    frames: dict[str, DatasetSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run one M2 shadow arm and expose only offline evaluation records."""

    injector = None if arm == "clean" else ImageAnomalyInjector(case.injected)
    pipeline = TemporalStereoPipeline(
        "M3", _controlled_q(), "m", experiment_authority=ExperimentAuthority.shadow()
    )
    q = _controlled_q()
    disparity_cache: dict[Path, np.ndarray] = {}
    candidates: list[dict[str, object]] = []
    recovery_trace: list[dict[str, object]] = []
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
        frame_phase = _phase(case, frame_id)
        for result in results:
            baseline = I1BaselineView.from_result(result)
            hard_failure = not baseline.valid
            prediction_m = _prediction_xyz_m(result)
            raw_m = baseline.xyz_m
            confirmed = _confirmed_anomaly(result, hard_failure=hard_failure)
            recovery_trace.append({
                "case_id": case.injected.case_id,
                "point_id": result.point_id,
                "frame_index": local_index,
                "frame_phase": frame_phase,
                "i1_valid": baseline.valid,
                "i2_state": result.i2_state or "OFF",
                "hard_failure": hard_failure,
                "confirmed_anomaly": confirmed,
                "raw_xyz_mm": _json_xyz_mm(raw_m),
                "prediction_xyz_mm": _json_xyz_mm(prediction_m),
            })
            if (
                raw_m is None
                or prediction_m is None
                or not result.i2_correction_applied
                or not result.candidate_safe
            ):
                continue
            gt_m = _gt_xyz_m(result, sample, q, disparity_cache)
            if gt_m is None:
                continue
            raw_mm = tuple(value * 1000.0 for value in raw_m)
            prediction_mm = tuple(value * 1000.0 for value in prediction_m)
            gt_mm = tuple(value * 1000.0 for value in gt_m)
            c1_mm, c1_magnitude = bounded_candidate_xyz_mm(
                raw_xyz_mm=raw_mm,
                prediction_xyz_mm=prediction_mm,
            )
            raw_error = _distance(raw_mm, gt_mm)
            c0_error = _distance(prediction_mm, gt_mm)
            c1_error = _distance(c1_mm, gt_mm)
            candidates.append({
                "case_id": case.injected.case_id,
                "pair_id": case.injected.pair_id,
                "subset": phase4_6_subset(case.injected.case_id),
                "arm": arm,
                "anomaly_type": case.injected.anomaly_type,
                "frame_index": local_index,
                "frame_phase": frame_phase,
                "point_id": result.point_id,
                "raw_i1_xyz_mm": json.dumps(raw_mm),
                "i2_prediction_xyz_mm": json.dumps(prediction_mm),
                "gt_xyz_mm": json.dumps(gt_mm),
                "raw_to_prediction_mm": _distance(raw_mm, prediction_mm),
                "c0_prediction_error_mm": c0_error,
                "c1_bounded_candidate_xyz_mm": json.dumps(c1_mm),
                "c1_correction_magnitude_mm": c1_magnitude,
                "c1_bounded_error_mm": c1_error,
                "raw_baseline_error_mm": raw_error,
                "c0_improvement_mm": raw_error - c0_error,
                "c1_improvement_mm": raw_error - c1_error,
                "c0_outcome": classify_candidate_outcome(
                    raw_error_mm=raw_error, candidate_error_mm=c0_error
                ),
                "c1_outcome": classify_candidate_outcome(
                    raw_error_mm=raw_error, candidate_error_mm=c1_error
                ),
            })
    return candidates, recovery_trace


def run_phase4_7a_offline_analysis(
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Generate Phase 4.7A traces without modifying production I2 behavior."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = build_phase4_5_case_windows(manifest_path)
    source = _load_verified_phase3_source(manifest_path)
    frames = {frame.frame_id: frame for frame in source.frames}
    candidate_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    for case in cases:
        subset = phase4_6_subset(case.injected.case_id)
        if subset == "development":
            for arm in ("clean", "injected"):
                candidates, recovery = _run_trace_arm(case, arm=arm, frames=frames)
                candidate_rows.extend(candidates)
                if arm == "injected":
                    recovery_rows.extend(recovery)
        else:
            # Holdout is not used for candidate-formula selection.  Its M2
            # trace is collected only because stale-predictor diagnosis has no
            # adjustable threshold and must cover the known 18 failures.
            _, recovery = _run_trace_arm(case, arm="injected", frames=frames)
            recovery_rows.extend(recovery)
    candidate_trace = pd.DataFrame(candidate_rows)
    candidate_trace.to_csv(output / "development_candidate_trace.csv", index=False, encoding="utf-8-sig")
    recovery_trace = pd.DataFrame(recovery_rows)
    recovery_class = classify_i2_recovery(recovery_trace)
    failures = recovery_class.loc[recovery_class["recovery_class"] == "I2_RECOVERY_FAILURE"]
    stale_input = recovery_trace.merge(
        failures[["case_id", "point_id", "exposure_end_frame_index"]],
        on=["case_id", "point_id"], how="inner", validate="many_to_one",
    )
    stale_input = stale_input.loc[stale_input["frame_index"] > stale_input["exposure_end_frame_index"]]
    stale_input = stale_input.loc[stale_input["frame_phase"].isin(["recovery", "stability"])]
    stale_input = stale_input.loc[
        stale_input["raw_xyz_mm"].notna() & stale_input["prediction_xyz_mm"].notna()
    ]
    stale = classify_stale_prediction_recovery(stale_input)
    stale.to_csv(output / "stale_prediction_recovery_analysis.csv", index=False, encoding="utf-8-sig")
    recovery_class.to_csv(output / "recovery_episode_classification.csv", index=False, encoding="utf-8-sig")
    return {
        "development_candidate_rows": len(candidate_trace),
        "i2_recovery_failures": int(len(failures)),
        "stale_prediction_recovery_cases": int((stale["recovery_pattern"] == "STALE_PREDICTION").sum()),
        "genuine_unstable_recovery_cases": int((stale["recovery_pattern"] != "STALE_PREDICTION").sum()),
    }
