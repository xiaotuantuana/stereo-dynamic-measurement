"""Phase 4 controlled validation without feeding evaluation labels into runtime."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from stereo_dynamic_measurement.innovation2.physics_validation import run_runtime_physics_analysis
from stereo_research.enhanced_processor import EnhancedPointProcessor, I2Evidence, TimedObservation

from ..simulation.temporal_scenarios import TemporalScenario, build_temporal_scenarios


CORE_SCENARIOS = {
    "stable": "stable",
    "single_jump": "single_jump",
    "continuous_outlier": "continuous_outlier",
    "slow_drift": "slow_drift",
    "fast_legitimate_motion": "fast_legitimate_motion",
    "noise": "gaussian_noise",
}


def _runtime_input(scenario: TemporalScenario) -> pd.DataFrame:
    return scenario.raw_measurements[[
        "frame", "timestamp_s", "point_id", "raw_X_mm", "raw_Y_mm", "raw_Z_mm",
        "flow_dx_mm", "flow_dy_mm", "flow_dz_mm",
    ]].copy()


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _runtime_evidence(row: object) -> I2Evidence:
    c_phy = getattr(row, "C_phy")
    c_phy_valid = bool(getattr(row, "C_phy_valid")) and _finite(c_phy)
    allow_correction = bool(getattr(row, "allow_candidate_correction"))
    transient = str(getattr(row, "transient_classification"))
    confirmed = c_phy_valid and float(c_phy) < 0.65 and allow_correction
    return I2Evidence(
        confirmed_anomaly=confirmed,
        legitimate_motion=transient == "possible_real_transient",
        hard_failure=False,
        geometry_valid=True,
        evidence_sufficient=c_phy_valid if confirmed else True,
        post_correction_safe=True,
        suspicious=confirmed,
    )


def _evaluate_scenario(scenario: TemporalScenario) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    runtime = run_runtime_physics_analysis(_runtime_input(scenario), fs_hz=30.0, threshold=0.65)
    confidence = runtime.confidence
    runtime_rows = scenario.raw_measurements.merge(
        confidence[[
            "frame", "point_id", "C_phy", "C_phy_valid", "transient_classification",
            "allow_candidate_correction",
        ]],
        on=["frame", "point_id"],
        how="left",
        validate="one_to_one",
    ).sort_values(["point_id", "frame"])
    processors: dict[str, EnhancedPointProcessor] = {}
    trace_rows: list[dict[str, object]] = []
    for row in runtime_rows.itertuples(index=False):
        point_id = str(row.point_id)
        processor = processors.setdefault(point_id, EnhancedPointProcessor())
        outcome = processor.process(
            TimedObservation(
                frame=int(row.frame),
                timestamp_s=float(row.timestamp_s),
                xyz_mm=(float(row.raw_X_mm), float(row.raw_Y_mm), float(row.raw_Z_mm)),
                evidence=_runtime_evidence(row),
            )
        )
        corrected = outcome.corrected_xyz_mm or outcome.raw_xyz_mm
        trusted = processor.trusted_history[-1].xyz_mm if processor.trusted_history else None
        trace_rows.append({
            "scenario": scenario.name,
            "frame": int(row.frame),
            "timestamp_s": float(row.timestamp_s),
            "point_id": point_id,
            "raw_x_mm": float(row.raw_X_mm),
            "raw_y_mm": float(row.raw_Y_mm),
            "raw_z_mm": float(row.raw_Z_mm),
            "prediction_x_mm": None if outcome.prediction_xyz_mm is None else outcome.prediction_xyz_mm[0],
            "prediction_y_mm": None if outcome.prediction_xyz_mm is None else outcome.prediction_xyz_mm[1],
            "prediction_z_mm": None if outcome.prediction_xyz_mm is None else outcome.prediction_xyz_mm[2],
            "corrected_x_mm": corrected[0],
            "corrected_y_mm": corrected[1],
            "corrected_z_mm": corrected[2],
            "trusted_state_x_mm": None if trusted is None else trusted[0],
            "trusted_state_y_mm": None if trusted is None else trusted[1],
            "trusted_state_z_mm": None if trusted is None else trusted[2],
            "decision": "USE_CORRECTED" if outcome.correction_applied else (
                "ABSTAIN" if outcome.state.value in {"QUARANTINED", "RECOVERY"} else "ACCEPT_BASELINE"
            ),
            "C_phy": getattr(row, "C_phy"),
            "correction_trigger": outcome.correction_applied,
            "state": outcome.state.value,
            "episode_id": outcome.episode_id,
            "candidate_safe": outcome.candidate_safety.safe,
            "candidate_safety_reasons": ";".join(outcome.candidate_safety.failed_reasons),
        })
    trace = pd.DataFrame(trace_rows).merge(
        scenario.ground_truth,
        on=["frame", "point_id"],
        how="left",
        validate="one_to_one",
    )
    trace["ground_truth_y_mm"] = trace["Y_gt_mm"]
    raw = trace[["raw_x_mm", "raw_y_mm", "raw_z_mm"]].to_numpy(float)
    corrected = trace[["corrected_x_mm", "corrected_y_mm", "corrected_z_mm"]].to_numpy(float)
    truth = trace[["X_gt_mm", "Y_gt_mm", "Z_gt_mm"]].to_numpy(float)
    trace["raw_error_mm"] = np.linalg.norm(raw - truth, axis=1)
    trace["corrected_error_mm"] = np.linalg.norm(corrected - truth, axis=1)
    trace["is_injected_fault"] = [
        (int(row.frame), str(row.point_id)) in scenario.fault_keys
        for row in trace.itertuples(index=False)
    ]
    raw_jumps = int((trace["raw_error_mm"] > 10.0).sum())
    corrected_jumps = int((trace["corrected_error_mm"] > 10.0).sum())
    recovery_values: list[int] = []
    for point_id in sorted({point for _, point in scenario.fault_keys}):
        final_fault = max(frame for frame, point in scenario.fault_keys if point == point_id)
        later = trace[(trace.point_id == point_id) & (trace.frame > final_fault) & (trace.state == "NORMAL")]
        if not later.empty:
            recovery_values.append(int(later.iloc[0].frame) - final_fault)
    nonfault = ~trace["is_injected_fault"]
    false_correction = float(
        trace.loc[nonfault, "correction_trigger"].astype(bool).sum() / max(int(nonfault.sum()), 1)
    )
    metric = {
        "scenario": next(name for name, source in CORE_SCENARIOS.items() if source == scenario.name),
        "raw_rmse": float(np.sqrt(np.mean(trace["raw_error_mm"] ** 2))),
        "corrected_rmse": float(np.sqrt(np.mean(trace["corrected_error_mm"] ** 2))),
        "raw_jump_count": raw_jumps,
        "corrected_jump_count": corrected_jumps,
        "jump_suppression": (None if raw_jumps == 0 else (raw_jumps - corrected_jumps) / raw_jumps),
        "false_correction": false_correction,
        "recovery_frames": (None if not recovery_values else float(np.mean(recovery_values))),
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
    }
    return metric, trace


def run_phase4_innovation2_validation(
    output_dir: str | Path,
    *,
    frame_count: int = 96,
    seed: int = 20260827,
) -> dict[str, object]:
    """Run the six approved controlled scenarios with state-isolated I2."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = build_temporal_scenarios(frame_count=frame_count, seed=seed)
    metrics: list[dict[str, object]] = []
    traces: list[pd.DataFrame] = []
    for source_name in CORE_SCENARIOS.values():
        metric, trace = _evaluate_scenario(scenarios[source_name])
        metrics.append(metric)
        traces.append(trace)
    core = pd.DataFrame(metrics)
    trace_table = pd.concat(traces, ignore_index=True)
    core_columns = [
        "scenario", "raw_rmse", "corrected_rmse", "raw_jump_count",
        "corrected_jump_count", "jump_suppression", "false_correction", "recovery_frames",
    ]
    core[core_columns].to_csv(output / "innovation2_core_validation.csv", index=False, encoding="utf-8-sig")
    trace_table.to_csv(output / "innovation2_frame_trace.csv", index=False, encoding="utf-8-sig")
    return {
        "provenance": "CONTROLLED",
        "scenario_count": len(core),
        "raw_rmse": float(core["raw_rmse"].mean()),
        "corrected_rmse": float(core["corrected_rmse"].mean()),
        "jump_not_worse": bool((core["corrected_jump_count"] <= core["raw_jump_count"]).all()),
        "ground_truth_online_access": False,
    }
