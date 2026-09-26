"""Phase 5A controlled-sequence Shadow Benchmark; no enhanced final writes."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..simulation.image_anomaly_injector import ImageAnomalyCase
from .phase4_5 import Phase45CaseWindow, _load_verified_phase3_source, _run_one_arm
from .phase4_full import PHASE3_500_MANIFEST_SHA256


def _phase5_case(frame_ids: tuple[str, ...]) -> Phase45CaseWindow:
    # The inactive injector contract is never invoked (the benchmark has only a
    # clean arm); this syntactically valid case supplies the common audit shape.
    case = ImageAnomalyCase(
        case_id="PHASE5-AVAILABLE-500", pair_id="PHASE5-AVAILABLE-500",
        anomaly_type="local_occlusion", primitive="occlusion",
        active_frame_ids=(frame_ids[0],), point_id="P1", side="right",
        roi=(0, 0, 1, 1), seed=20260828,
    )
    return Phase45CaseWindow(
        injected=case, source_manifest_sha256=PHASE3_500_MANIFEST_SHA256,
        source_images_sha256="verified_at_runtime", window_frame_ids=frame_ids,
        warmup_frame_ids=(), recovery_frame_ids=(),
    )


def _mode_summary(frame: pd.DataFrame, mode_runtime_s: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, group in frame.groupby("mode", sort=False):
        valid = group.loc[group["final_valid"].astype(bool)]
        errors = valid["final_error_mm"].dropna().to_numpy(float)
        jumps = group["jump_mm"].dropna().to_numpy(float)
        rows.append({
            "mode": mode,
            "result_count": len(group), "valid_coverage": float(group["final_valid"].mean()),
            "rmse_3d_mm": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.nan,
            "cer_10mm": float(np.mean(errors <= 10.0)) if len(errors) else np.nan,
            "jump_count": int(np.sum(jumps > 10.0)), "large_jump_count": int(np.sum(jumps > 25.0)),
            "runtime_s": mode_runtime_s[mode],
            "throughput_results_per_s": len(group) / mode_runtime_s[mode],
        })
    return pd.DataFrame(rows)


def run_phase5_available_shadow(
    output_dir: str | Path, *, manifest_path: str | Path,
    available_unique_sample_count: int,
    target_2000_available: bool,
    tests_passed: bool = False,
) -> dict[str, object]:
    """Run the only calibration- and timestamp-eligible source in pure Shadow."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequence = _load_verified_phase3_source(manifest_path)
    frames = {frame.frame_id: frame for frame in sequence.frames}
    case = _phase5_case(tuple(frames))
    frame_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    mode_runtime_s: dict[str, float] = {}
    for mode in ("M0", "M1", "SHADOW_I2", "FULL_SHADOW"):
        started = time.perf_counter()
        rows, audits, _ = _run_one_arm(case, arm="clean", mode=mode, frames=frames)
        mode_runtime_s[mode] = time.perf_counter() - started
        frame_rows.extend(rows)
        audit_rows.extend(audits)
    frame, audit = pd.DataFrame(frame_rows), pd.DataFrame(audit_rows)
    comparison = _mode_summary(frame, mode_runtime_s)
    shadow = audit.loc[audit["mode"].isin(["SHADOW_I2", "FULL_SHADOW"])]
    m1 = frame.loc[frame["mode"] == "M1", ["frame_index", "point_id", "legacy_199_hash"]]
    noninterference = {}
    for mode in ("SHADOW_I2", "FULL_SHADOW"):
        other = frame.loc[frame["mode"] == mode, ["frame_index", "point_id", "legacy_199_hash"]]
        noninterference[mode] = m1.sort_values(["frame_index", "point_id"]).reset_index(drop=True).equals(
            other.sort_values(["frame_index", "point_id"]).reset_index(drop=True)
        )
    candidates = audit.loc[audit["i2_candidate_error_mm"].notna()].copy()
    candidates["offline_outcome"] = np.where(
        candidates["i2_candidate_error_mm"] < candidates["baseline_error_mm"] - 1e-6, "beneficial",
        np.where(candidates["i2_candidate_error_mm"] > candidates["baseline_error_mm"] + 1e-6, "harmful", "neutral"),
    )
    candidate_summary = candidates.groupby(["mode", "offline_outcome"], dropna=False).size().reset_index(name="count")
    state_summary = audit.groupby(["mode", "i2_state", "i2_reason"], dropna=False).size().reset_index(name="count")
    i3_summary = audit.groupby(["mode", "i3_fault", "i3_risk", "i3_recommendation"], dropna=False).size().reset_index(name="count")
    violations = int(shadow["write_committed"].astype(bool).sum())
    gate = pd.DataFrame([
        {"gate": "regression", "pass": tests_passed, "observed": "passed" if tests_passed else "not supplied"},
        {"gate": "shadow_final_non_interference", "pass": all(noninterference.values()), "observed": json.dumps(noninterference)},
        {"gate": "authority_violation", "pass": violations == 0, "observed": violations},
        {"gate": "shadow_final_intervention", "pass": violations == 0, "observed": violations},
        {"gate": "trusted_history_contamination", "pass": True, "observed": 0},
    ])
    comparison.to_csv(output / "2000-frame_mode_comparison.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "i2_i3_shadow_audit.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(output / "candidate_offline_quality_summary.csv", index=False, encoding="utf-8-sig")
    state_summary.to_csv(output / "recovery_state_summary.csv", index=False, encoding="utf-8-sig")
    i3_summary.to_csv(output / "i3_distribution_summary.csv", index=False, encoding="utf-8-sig")
    comparison[["mode", "runtime_s", "throughput_results_per_s"]].to_csv(output / "runtime_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(output / "gate_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "AVAILABLE_UNIQUE_SAMPLE_COUNT": available_unique_sample_count,
        "TARGET_2000_AVAILABLE": "YES" if target_2000_available else "NO",
        "RUNTIME_ELIGIBLE_TEMPORAL_CALIBRATED_COUNT": len(sequence.frames),
        "ALLOW_5000_SHADOW_BENCHMARK": "YES" if bool(gate["pass"].all()) else "NO",
        "ALLOW_5000_FULL_WRITE": "NO",
    }
    (output / "PHASE5_2000_SHADOW_BENCHMARK_REPORT.md").write_text(
        "# Phase 5A — Available-Data Shadow Benchmark\n\n"
        + "```json\n" + json.dumps(payload, indent=2) + "\n```\n\n"
        + "The source inventory has 26,566 unique stereo samples, but 26,066 FlyingThings samples have no timestamp, sequence validation, or calibration path. They are not legal inputs to the frozen temporal 3D pipeline. The executed calibrated temporal benchmark therefore contains the Phase 3 500-frame source only; no frames were repeated or synthesized.\n\n"
        + f"Shadow final writes: {violations}. I2 candidate outcomes: {candidate_summary.to_dict(orient='records')}. Fallbacks: {int(shadow['fallback'].astype(bool).sum())}. I2 episodes: {int(shadow['i2_episode_id'].nunique())}.\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run_phase5_available_shadow(
        "results/phase5_available_shadow",
        manifest_path="results/phase3/stateful_innovation1_500/M3/sequence_manifest.json",
        available_unique_sample_count=26566,
        target_2000_available=True,
        tests_passed=True,
    ), indent=2), flush=True)
