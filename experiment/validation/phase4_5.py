"""Frozen Phase 4.5 image-level anomaly case definitions and validation runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stereo_research.final_arbitration import ExperimentAuthority
from stereo_research.models import PointSpec
from stereo_research.pipeline import TemporalStereoPipeline

from ..datasets.io import read_image, read_pfm
from ..datasets.models import DatasetSample
from ..sequence.ablation import _controlled_q
from ..simulation.image_anomaly_injector import ImageAnomalyCase, ImageAnomalyInjector
from .phase4_full import load_phase3_fixed_sequence


PHASE4_5_CASE_COUNT = 32
PHASE3_500_SOURCE_IMAGES_SHA256 = "17CF7321816A87BE45D3A4A61F04F419B53D7CEC13F9DF861EFEC93865C1C816"
_POINTS = (
    PointSpec("P1", (160.0, 60.0)),
    PointSpec("P2", (200.0, 90.0)),
    PointSpec("P3", (240.0, 120.0)),
)
_TYPE_PRIMITIVE = {
    "local_occlusion": "occlusion",
    "local_blur": "blur",
    "unilateral_roi_shift": "roi_shift",
}


@dataclass(frozen=True)
class Phase45CaseWindow:
    """One selected frozen window and its image-only injected counterpart."""

    injected: ImageAnomalyCase
    source_manifest_sha256: str
    source_images_sha256: str
    window_frame_ids: tuple[str, ...]
    warmup_frame_ids: tuple[str, ...]
    recovery_frame_ids: tuple[str, ...]

    @property
    def clean_pair_id(self) -> str:
        return self.injected.pair_id

    @property
    def clean_source_frame_ids(self) -> tuple[str, ...]:
        return self.window_frame_ids

    @property
    def injected_source_frame_ids(self) -> tuple[str, ...]:
        return self.window_frame_ids


def _load_verified_phase3_source(manifest_path: str | Path):
    sequence = load_phase3_fixed_sequence(manifest_path)
    digest = hashlib.sha256()
    for frame in sequence.frames:
        digest.update(frame.left_path.read_bytes())
        digest.update(frame.right_path.read_bytes())
    if digest.hexdigest().upper() != PHASE3_500_SOURCE_IMAGES_SHA256:
        raise ValueError("Phase 3 frozen source image hash does not match")
    return sequence


def build_phase4_5_case_windows(manifest_path: str | Path) -> tuple[Phase45CaseWindow, ...]:
    """Select the fixed 32 windows from, and only from, the Phase 3 source."""

    path = Path(manifest_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    sequence = _load_verified_phase3_source(path)
    frames = sequence.frames
    if len(frames) != 500:
        raise ValueError("Phase 4.5 requires the frozen 500-frame source")
    cases: list[Phase45CaseWindow] = []
    for index in range(PHASE4_5_CASE_COUNT):
        group, within_group = divmod(index, 8)
        anomaly_type = (
            "local_occlusion", "local_blur", "unilateral_roi_shift", "continuous_anomaly"
        )[group]
        primitive = (
            ("occlusion", "blur", "roi_shift")[within_group % 3]
            if anomaly_type == "continuous_anomaly"
            else _TYPE_PRIMITIVE[anomaly_type]
        )
        duration = 2 + within_group % 3 if anomaly_type == "continuous_anomaly" else 1
        start = 20 + index * 14
        window = frames[start:start + 2 + duration + 3]
        if len(window) != 2 + duration + 3:
            raise ValueError("frozen source is too short for the declared Phase 4.5 windows")
        warmup, active = window[:2], window[2:2 + duration]
        recovery = window[2 + duration:2 + duration + 2]
        point = _POINTS[index % len(_POINTS)]
        roi = (int(point.xy[0]) - 16, int(point.xy[1]) - 12, 32, 24)
        case = ImageAnomalyCase(
            case_id=f"P45-{index + 1:02d}",
            pair_id=f"P45-PAIR-{index + 1:02d}",
            anomaly_type=anomaly_type,
            primitive=primitive,
            active_frame_ids=tuple(frame.frame_id for frame in active),
            point_id=point.point_id,
            side="right" if index % 2 == 0 else "left",
            roi=roi,
            seed=20260828 + index,
            shift_px=1 + within_group % 3 if primitive == "roi_shift" else 0,
            blur_kernel=5 + 2 * (within_group % 3) if primitive == "blur" else 5,
        )
        cases.append(Phase45CaseWindow(
            injected=case,
            source_manifest_sha256=digest,
            source_images_sha256=PHASE3_500_SOURCE_IMAGES_SHA256,
            window_frame_ids=tuple(frame.frame_id for frame in window),
            warmup_frame_ids=tuple(frame.frame_id for frame in warmup),
            recovery_frame_ids=tuple(frame.frame_id for frame in recovery),
        ))
    return tuple(cases)


def count_false_final_corrections(audit: pd.DataFrame) -> int:
    """Count only actual clean-arm corrected writes, never REJECT/keep-baseline."""

    selected = audit.loc[
        (audit["arm"] == "clean")
        & (audit["mode"] == "M3")
        & (audit["committed_decision"] == "USE_CORRECTED")
        & audit["write_committed"].astype(bool)
    ]
    return int(len(selected))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _xyz_error_mm(
    result_xyz_m: tuple[float, float, float] | None,
    result: Any,
    sample: DatasetSample,
    q: np.ndarray,
    disparity_cache: dict[Path, np.ndarray],
) -> float | None:
    if result_xyz_m is None or result.left_x is None or result.left_y is None or sample.disparity_gt_path is None:
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
    gt_xyz = projected[:3] / projected[3]
    return float(np.linalg.norm(np.asarray(result_xyz_m) - gt_xyz) * 1000.0)


def _legacy_hash(result: Any) -> str:
    """Compare the stable values of the legacy 199-column production row."""

    row = dict(result.as_csv_row())
    for runtime_field in ("flow_ms", "matching_ms", "total_ms"):
        row.pop(runtime_field, None)
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _i2_candidate_xyz_m(result: Any) -> tuple[float, float, float] | None:
    """Return only the actual I2 candidate, never the legacy Shadow candidate.

    ``FramePointResult.candidate_xyz_m`` is the older Shadow trajectory field.
    The FULL_ENHANCED commit path instead receives the I2 trusted prediction.
    This helper deliberately exposes that latter value only when I2 actually
    proposed a complete correction; a prediction by itself is not a candidate.
    """

    if not result.i2_correction_applied:
        return None
    values = (
        result.i2_prediction_x_m,
        result.i2_prediction_y_m,
        result.i2_prediction_z_m,
    )
    if any(value is None for value in values):
        return None
    xyz = tuple(float(value) for value in values)
    return xyz if np.isfinite(np.asarray(xyz, dtype=float)).all() else None


def _case_rows(
    case: Phase45CaseWindow,
    frames: dict[str, DatasetSample],
) -> list[dict[str, object]]:
    source_hashes = {
        frame_id: {
            "left": _sha256_file(frames[frame_id].left_path),
            "right": _sha256_file(frames[frame_id].right_path),
        }
        for frame_id in case.window_frame_ids
    }
    common = {
        "case_id": case.injected.case_id,
        "pair_id": case.injected.pair_id,
        "anomaly_type": case.injected.anomaly_type,
        "primitive": case.injected.primitive,
        "point_id": case.injected.point_id,
        "side": case.injected.side,
        "roi": json.dumps(case.injected.roi),
        "active_frame_ids": json.dumps(case.injected.active_frame_ids),
        "source_frame_ids": json.dumps(case.window_frame_ids),
        "source_timestamps": json.dumps([frames[item].timestamp for item in case.window_frame_ids]),
        "warmup_frame_ids": json.dumps(case.warmup_frame_ids),
        "recovery_frame_ids": json.dumps(case.recovery_frame_ids),
        "seed": case.injected.seed,
        "phase3_manifest_sha256": case.source_manifest_sha256,
        "phase3_source_images_sha256": case.source_images_sha256,
        "source_image_hashes": json.dumps(source_hashes, sort_keys=True),
        "injector_version": "phase4_5_image_copy_v1",
    }
    return [{**common, "arm": arm} for arm in ("clean", "injected")]


def _run_one_arm(
    case: Phase45CaseWindow,
    *,
    arm: str,
    mode: str,
    frames: dict[str, DatasetSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]], bool]:
    method, authority = {
        "M0": ("M0", None),
        "M1": ("M3", None),
        "M2": ("M3", ExperimentAuthority.shadow()),
        "M3": ("M3", ExperimentAuthority.full_experiment(write_enabled=True)),
        "SHADOW_I2": ("M3", ExperimentAuthority.shadow()),
        "FULL_SHADOW": ("M3", ExperimentAuthority.full_experiment(write_enabled=False)),
    }[mode]
    injector = None if arm == "clean" else ImageAnomalyInjector(case.injected)
    pipeline = TemporalStereoPipeline(method, _controlled_q(), "m", experiment_authority=authority)
    q = _controlled_q()
    disparity_cache: dict[Path, np.ndarray] = {}
    previous_final: dict[str, tuple[float, float, float] | None] = {}
    frame_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    deterministic = True
    for local_index, frame_id in enumerate(case.window_frame_ids):
        sample = frames[frame_id]
        source_left, source_right = read_image(sample.left_path), read_image(sample.right_path)
        left, right = source_left.copy(), source_right.copy()
        if injector is not None:
            left = injector.transform(source_left, frame_id=frame_id, side="left")
            right = injector.transform(source_right, frame_id=frame_id, side="right")
            deterministic = deterministic and np.array_equal(
                left, injector.transform(source_left, frame_id=frame_id, side="left")
            ) and np.array_equal(right, injector.transform(source_right, frame_id=frame_id, side="right"))
        frame_number = int(frame_id)
        results = (
            pipeline.initialize(left, right, _POINTS, frame_number)
            if local_index == 0 else pipeline.step(left, right, frame_number)
        )
        phase = (
            "warmup" if frame_id in case.warmup_frame_ids
            else "exposure" if frame_id in case.injected.active_frame_ids
            else "recovery" if frame_id in case.recovery_frame_ids else "stability"
        )
        for result in results:
            baseline_xyz = result.pipeline_baseline_xyz_m
            i1_status = result.i1_status or result.status
            i1_valid = i1_status == "valid" and baseline_xyz is not None and np.isfinite(np.asarray(baseline_xyz)).all()
            i1_distance = None if baseline_xyz is None else float(np.linalg.norm(np.asarray(baseline_xyz)))
            i3 = TemporalStereoPipeline._structured_i3_recommendation(result, hard_failure=not i1_valid)
            final_xyz = result.final_xyz_m
            i2_candidate_xyz = _i2_candidate_xyz_m(result)
            final_error = _xyz_error_mm(final_xyz, result, sample, q, disparity_cache)
            baseline_error = _xyz_error_mm(baseline_xyz, result, sample, q, disparity_cache)
            i2_candidate_error = _xyz_error_mm(i2_candidate_xyz, result, sample, q, disparity_cache)
            prior = previous_final.get(result.point_id)
            jump_mm = (
                None if prior is None or final_xyz is None
                else float(np.linalg.norm(np.asarray(final_xyz) - np.asarray(prior)) * 1000.0)
            )
            previous_final[result.point_id] = final_xyz
            common = {
                "case_id": case.injected.case_id, "pair_id": case.injected.pair_id,
                "arm": arm, "mode": mode, "frame_index": local_index,
                "source_frame_id": frame_id, "timestamp": sample.timestamp,
                "point_id": result.point_id, "frame_phase": phase,
                "anomaly_type": case.injected.anomaly_type,
                "anomaly_active": frame_id in case.injected.active_frame_ids and arm == "injected",
                "source_left_hash": _sha256_file(sample.left_path),
                "source_right_hash": _sha256_file(sample.right_path),
                "input_left_hash": hashlib.sha256(left.tobytes()).hexdigest(),
                "input_right_hash": hashlib.sha256(right.tobytes()).hexdigest(),
                "i1_status": i1_status, "i1_valid": i1_valid,
                "i1_xyz_m": json.dumps(baseline_xyz), "i1_distance_m": i1_distance,
                "i2_state": result.i2_state or "OFF", "i2_episode_id": result.i2_episode_id,
                "i2_prediction_xyz_m": json.dumps((result.i2_prediction_x_m, result.i2_prediction_y_m, result.i2_prediction_z_m)),
                "i2_trusted_committed": result.i2_trusted_committed,
                "i2_correction_applied": result.i2_correction_applied,
                "i2_reason": result.i2_reason,
                # These two fields must remain separate: the first is the
                # legacy Shadow diagnostic candidate and the second is the
                # only candidate the FULL_ENHANCED arbitrator can commit.
                "shadow_candidate_xyz_m": json.dumps(result.candidate_xyz_m),
                "i2_candidate_xyz_m": json.dumps(i2_candidate_xyz),
                "i2_candidate_error_mm": i2_candidate_error,
                "candidate_safe": result.candidate_safe,
                "candidate_safety_reasons": result.candidate_safety_reasons,
                "i3_fault": result.fault_class or "NORMAL", "i3_risk": i3.risk.value,
                "i3_confidence": result.fault_confidence, "i3_recommendation": i3.action.value,
                "proposed_decision": result.proposed_decision or "BASELINE",
                "committed_decision": result.committed_decision or "BASELINE",
                "write_committed": bool(result.write_committed), "result_source": result.result_source or "I1_BASELINE",
                "final_owner": result.final_owner, "fallback": "enhanced_exception" in result.final_decision_reason,
                "final_reason": result.final_decision_reason, "final_status": result.status,
                "final_valid": result.final_valid if result.final_valid is not None else i1_valid,
                "final_xyz_m": json.dumps(final_xyz), "baseline_error_mm": baseline_error,
                "final_error_mm": final_error, "jump_mm": jump_mm,
                # These are existing runtime measurements recorded for the
                # Phase 4.6 offline candidate-quality analysis only.  None
                # participates in the I1/I2/I3 runtime decision here.
                "runtime_i1_confidence": result.confidence,
                "runtime_measurement_quality_score": result.measurement_quality_score,
                "runtime_texture_std": result.texture_std,
                "runtime_uniqueness_margin": result.uniqueness_margin_value,
                "runtime_lr_error_px": result.lr_error_px,
                "runtime_flow_fb_error_px": result.flow_fb_error_px,
                "runtime_prediction_residual_px": result.prediction_residual_px,
                "runtime_c_phy": result.c_phy,
                "legacy_199_hash": _legacy_hash(result),
            }
            frame_rows.append(common)
            audit_rows.append({
                **common,
                "candidate_finite": "candidate_not_finite" not in result.candidate_safety_reasons,
                "candidate_geometry_valid": "geometry_invalid" not in result.candidate_safety_reasons,
                "candidate_authorized": "correction_unauthorized" not in result.candidate_safety_reasons,
                "candidate_evidence_sufficient": "evidence_insufficient" not in result.candidate_safety_reasons,
                "candidate_post_safe": "post_correction_unsafe" not in result.candidate_safety_reasons,
            })
    return frame_rows, audit_rows, deterministic


def _add_cross_mode_metrics(frame: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the M2 I1 baseline error without exposing it to runtime decisions."""

    keys = ["case_id", "arm", "frame_index", "point_id"]
    m2 = frame.loc[frame["mode"] == "M2", keys + ["final_error_mm"]].rename(
        columns={"final_error_mm": "m2_baseline_error_mm"}
    )
    frame = frame.merge(m2, on=keys, how="left", validate="many_to_one")
    frame["error_improvement_mm"] = np.where(
        frame["mode"] == "M3",
        frame["m2_baseline_error_mm"] - frame["final_error_mm"],
        np.nan,
    )
    frame["candidate_error_improvement_mm"] = np.where(
        frame["mode"] == "M3",
        frame["m2_baseline_error_mm"] - frame["i2_candidate_error_mm"],
        np.nan,
    )
    audit = audit.merge(
        frame[keys + [
            "mode", "m2_baseline_error_mm", "error_improvement_mm",
            "candidate_error_improvement_mm",
        ]],
        on=keys + ["mode"], how="left", validate="one_to_one",
    )
    return frame, audit


def _precommit_parity_ok(audit: pd.DataFrame) -> bool:
    keys = ["case_id", "arm", "frame_index", "point_id"]
    fields = ["i1_status", "i1_valid", "i2_state", "i2_episode_id", "candidate_safe", "candidate_safety_reasons", "i3_fault", "i3_risk", "i3_confidence", "i3_recommendation", "proposed_decision"]
    m2 = audit.loc[audit["mode"] == "M2", keys + fields].sort_values(keys).reset_index(drop=True)
    m3 = audit.loc[audit["mode"] == "M3", keys + fields].sort_values(keys).reset_index(drop=True)
    return m2.equals(m3)


def _gate_rows(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    deterministic: bool,
    tests_passed: bool,
) -> tuple[list[dict[str, object]], bool]:
    corrected = audit.loc[
        (audit["mode"] == "M3")
        & audit["write_committed"].astype(bool)
        & (audit["committed_decision"] == "USE_CORRECTED")
    ].copy()
    false_corrections = count_false_final_corrections(audit)
    false_rejects = int(len(audit.loc[
        (audit["arm"] == "clean") & (audit["mode"] == "M3")
        & audit["write_committed"].astype(bool) & (audit["committed_decision"] == "REJECT")
    ]))
    writes_outside_m3 = int(len(audit.loc[
        (audit["mode"].isin(["M1", "M2"])) & audit["write_committed"].astype(bool)
    ]))
    authority_bad = int(len(audit.loc[
        audit["write_committed"].astype(bool) & (audit["final_owner"] != "TemporalStereoPipeline")
    ]))
    harmful = int(np.sum(
        corrected["final_error_mm"].notna() & corrected["m2_baseline_error_mm"].notna()
        & (corrected["final_error_mm"] > corrected["m2_baseline_error_mm"] + 1e-6)
    ))
    improvements = corrected.loc[
        corrected["error_improvement_mm"].notna(), "error_improvement_mm"
    ].to_numpy(float)
    median_improvement = float(np.median(improvements)) if len(improvements) else float("nan")
    clean = frame.loc[frame["arm"] == "clean"]
    clean_m1 = clean.loc[clean["mode"] == "M1", ["case_id", "frame_index", "point_id", "legacy_199_hash"]]
    clean_m2 = clean.loc[clean["mode"] == "M2", ["case_id", "frame_index", "point_id", "legacy_199_hash"]]
    clean_equivalence = clean_m1.sort_values(list(clean_m1.columns)).reset_index(drop=True).equals(
        clean_m2.sort_values(list(clean_m2.columns)).reset_index(drop=True)
    )
    episode_rows = audit.loc[(audit["arm"] == "injected") & (audit["mode"] == "M2")]
    recovery_failures = 0
    for _, group in episode_rows.groupby(["case_id", "point_id"]):
        exposure = group.loc[group["frame_phase"] == "exposure"]
        if exposure.empty or not exposure["i2_state"].isin(["QUARANTINED", "RECOVERY"]).any():
            continue
        end_index = int(exposure["frame_index"].max())
        recovered = group.loc[(group["frame_index"] > end_index) & (group["i2_state"] == "NORMAL")]
        if recovered.empty or int(recovered["frame_index"].min()) - end_index > 2:
            recovery_failures += 1
    entries = [
        ("tests", "all existing and new tests pass", tests_passed, "tests not yet supplied as passing"),
        ("frozen_source", "Phase 3 manifest and full source-image hash match", True, "source verification failed before run"),
        ("deterministic_injector", "same case input has identical transformed image", deterministic, "image copy transformation changed on repeat"),
        ("clean_equivalence", "clean M1/M2 stable legacy 199 rows equal", clean_equivalence, "clean M1/M2 legacy output differs"),
        ("shadow_non_interference", "M2 has zero final writes", writes_outside_m3 == 0, f"M1/M2 writes={writes_outside_m3}"),
        ("precommit_parity", "M2/M3 share all pre-commit enhanced evidence", _precommit_parity_ok(audit), "M2/M3 pre-commit evidence differs"),
        ("authority", "all writes owned by TemporalStereoPipeline", authority_bad == 0, f"authority violations={authority_bad}"),
        ("commit_count", "enhanced corrected writes >= 5", len(corrected) >= 5, f"corrected writes={len(corrected)}"),
        ("anomaly_coverage", "corrected writes cover >= 2 anomaly types", corrected["anomaly_type"].nunique() >= 2, f"types={corrected['anomaly_type'].nunique()}"),
        ("harmful_commit", "harmful corrected writes = 0", harmful == 0, f"harmful={harmful}"),
        ("clean_false_final_correction", "clean corrected writes = 0", false_corrections == 0, f"count={false_corrections}"),
        ("clean_false_reject", "clean final rejects = 0", false_rejects == 0, f"count={false_rejects}"),
        ("median_improvement", "median committed improvement > 0", bool(np.isfinite(median_improvement) and median_improvement > 0), f"median_mm={median_improvement}"),
        ("recovery", "observed I2 episodes recover within two frames", recovery_failures == 0, f"failures={recovery_failures}"),
    ]
    rows = [{"gate_id": gate, "required": required, "observed": "PASS" if passed else reason, "pass": passed, "failure_reason": "" if passed else reason, "evidence_path": "frame_level_results.csv;arbitration_audit.csv"} for gate, required, passed, reason in entries]
    return rows, all(bool(row["pass"]) for row in rows)


def run_phase4_5_validation(
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
    case_windows: tuple[Phase45CaseWindow, ...] | None = None,
    tests_passed: bool = False,
) -> dict[str, object]:
    """Run bounded frozen image windows; this never runs the 500-frame sequence."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = case_windows or build_phase4_5_case_windows(manifest_path)
    if not cases:
        raise ValueError("at least one Phase 4.5 case is required")
    source = _load_verified_phase3_source(manifest_path)
    frames = {frame.frame_id: frame for frame in source.frames}
    manifest_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    deterministic = True
    for case in cases:
        manifest_rows.extend(_case_rows(case, frames))
        for arm in ("clean", "injected"):
            for mode in ("M1", "M2", "M3"):
                rows, audits, arm_deterministic = _run_one_arm(case, arm=arm, mode=mode, frames=frames)
                frame_rows.extend(rows)
                audit_rows.extend(audits)
                deterministic = deterministic and arm_deterministic
    manifest = pd.DataFrame(manifest_rows)
    frame, audit = _add_cross_mode_metrics(pd.DataFrame(frame_rows), pd.DataFrame(audit_rows))
    gates, allow_2000 = _gate_rows(frame, audit, deterministic=deterministic, tests_passed=tests_passed)
    summary = frame.groupby(["case_id", "pair_id", "arm", "mode", "anomaly_type"], dropna=False).agg(
        frame_count=("frame_index", "count"), valid_count=("final_valid", "sum"),
        final_commits=("write_committed", "sum"), mean_final_error_mm=("final_error_mm", "mean"),
        median_error_improvement_mm=("error_improvement_mm", "median"), max_jump_mm=("jump_mm", "max"),
        i2_episode_count=("i2_episode_id", "nunique"),
    ).reset_index()
    manifest.to_csv(output / "case_manifest.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(output / "frame_level_results.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "arbitration_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gates).to_csv(output / "gate_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "case_count": len(cases), "source": "PHASE3_FIXED_500", "deterministic": deterministic,
        "allow_2000": "YES" if allow_2000 else "NO",
    }
