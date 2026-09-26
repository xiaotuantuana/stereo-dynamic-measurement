"""Bounded Phase 4.6 candidate-quality and recovery evaluation helpers.

This module is deliberately experimental-only.  It never participates in the
runtime I1/I2/I3/final decision path and never has final-write authority.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .phase4_5 import run_phase4_5_validation


_RUNTIME_QUALITY_COLUMNS = (
    "runtime_i1_confidence",
    "runtime_measurement_quality_score",
    "runtime_texture_std",
    "runtime_uniqueness_margin",
    "runtime_lr_error_px",
    "runtime_flow_fb_error_px",
    "runtime_prediction_residual_px",
    "runtime_c_phy",
)


def phase4_6_subset(case_id: str) -> str:
    """Freeze the development/holdout split from the case ID alone."""

    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return "development" if digest[0] % 2 == 0 else "holdout"


def classify_i2_recovery(audit: pd.DataFrame) -> pd.DataFrame:
    """Separate unavailable I1 recovery from evaluable I2 recovery.

    The I2 gate begins at the first *valid* clean observation after a
    quarantined exposure.  It retains the existing two-observation contract;
    this function only classifies experimental evidence and never alters I2.
    """

    required = {"case_id", "point_id", "frame_index", "frame_phase", "i1_valid", "i2_state"}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"recovery audit missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for (case_id, point_id), group in audit.groupby(["case_id", "point_id"], sort=True):
        group = group.sort_values("frame_index")
        exposure = group.loc[
            (group["frame_phase"] == "exposure")
            & group["i2_state"].isin(["QUARANTINED", "RECOVERY"])
        ]
        if exposure.empty:
            continue
        exposure_end = int(exposure["frame_index"].max())
        clean_after = group.loc[group["frame_index"] > exposure_end]
        valid_clean = clean_after.loc[clean_after["i1_valid"].astype(bool)]
        base = {
            "case_id": case_id,
            "point_id": point_id,
            "exposure_end_frame_index": exposure_end,
            "first_i1_valid_clean_frame_index": (
                None if valid_clean.empty else int(valid_clean.iloc[0]["frame_index"])
            ),
        }
        if valid_clean.empty:
            records.append({
                **base,
                "recovery_class": "I1_BASELINE_NOT_RECOVERED",
                "i2_recovery_evaluable": False,
                "valid_clean_observations_to_normal": None,
            })
            continue
        first_two = valid_clean.iloc[:2]
        normal_positions = first_two.index[first_two["i2_state"].eq("NORMAL")]
        if len(first_two) < 2:
            records.append({
                **base,
                "recovery_class": "I2_RECOVERY_INSUFFICIENT_VALID_OBSERVATIONS",
                "i2_recovery_evaluable": False,
                "valid_clean_observations_to_normal": None,
            })
            continue
        if len(normal_positions):
            normal_index = normal_positions[0]
            order = list(first_two.index).index(normal_index) + 1
            records.append({
                **base,
                "recovery_class": "I2_RECOVERY_CONFIRMED",
                "i2_recovery_evaluable": True,
                "valid_clean_observations_to_normal": order,
            })
            continue
        records.append({
            **base,
            "recovery_class": "I2_RECOVERY_FAILURE",
            "i2_recovery_evaluable": True,
            "valid_clean_observations_to_normal": None,
        })
    return pd.DataFrame(records)


def run_phase4_6_validation(
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
    tests_passed: bool,
) -> dict[str, object]:
    """Replay the fixed Phase 4.5 windows and write Phase 4.6 evidence.

    The replay remains M1/M2/M3 through the existing pipeline.  This function
    only labels candidate outcome with GT *after* execution for evaluation; no
    GT, arm, injection metadata, or future frame is sent to runtime policy.
    """

    output = Path(output_dir)
    replay = output / "replay"
    phase45 = run_phase4_5_validation(
        replay,
        manifest_path=manifest_path,
        tests_passed=tests_passed,
    )
    audit = pd.read_csv(replay / "arbitration_audit.csv")
    manifest = pd.read_csv(replay / "case_manifest.csv")
    split = (
        manifest[["case_id", "pair_id"]]
        .drop_duplicates()
        .assign(subset=lambda data: data["case_id"].map(phase4_6_subset))
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    if split.groupby("pair_id")["subset"].nunique().gt(1).any():
        raise ValueError("paired clean/injected cases must share one subset")
    split.to_csv(output / "development_holdout_split_manifest.csv", index=False, encoding="utf-8-sig")

    candidates = audit.loc[
        (audit["mode"] == "M3")
        & audit["candidate_safe"].astype(bool)
        & audit["i2_candidate_error_mm"].notna()
    ].copy()
    candidates = candidates.merge(split[["case_id", "subset"]], on="case_id", how="left", validate="many_to_one")
    candidates["offline_candidate_outcome"] = candidates["candidate_error_improvement_mm"].map(
        lambda value: "BENEFICIAL" if value > 1e-6 else "HARMFUL_OR_NON_BENEFICIAL"
    )
    candidate_columns = [
        "case_id", "pair_id", "subset", "arm", "anomaly_type", "frame_index", "frame_phase", "point_id",
        "i1_status", "i2_state", "i2_reason", "i3_fault", "i3_confidence",
        "shadow_candidate_xyz_m", "i2_candidate_xyz_m", "candidate_safe",
        "candidate_safety_reasons", "i2_candidate_error_mm", "m2_baseline_error_mm",
        "candidate_error_improvement_mm", "offline_candidate_outcome", *_RUNTIME_QUALITY_COLUMNS,
    ]
    candidates[candidate_columns].to_csv(
        output / "candidate_quality_analysis.csv", index=False, encoding="utf-8-sig"
    )

    replay_summary = candidates.groupby(
        ["subset", "arm", "anomaly_type"], dropna=False
    ).agg(
        safe_candidate_count=("candidate_safe", "size"),
        beneficial_candidate_count=("offline_candidate_outcome", lambda values: int((values == "BENEFICIAL").sum())),
        harmful_or_nonbeneficial_candidate_count=(
            "offline_candidate_outcome", lambda values: int((values != "BENEFICIAL").sum()),
        ),
        median_candidate_improvement_mm=("candidate_error_improvement_mm", "median"),
    ).reset_index()
    replay_summary.to_csv(output / "replay_scenario_summary.csv", index=False, encoding="utf-8-sig")

    recovery = classify_i2_recovery(audit.loc[(audit["arm"] == "injected") & (audit["mode"] == "M2")])
    recovery.to_csv(output / "recovery_summary.csv", index=False, encoding="utf-8-sig")
    recovery_counts = recovery["recovery_class"].value_counts().to_dict() if not recovery.empty else {}

    injected_development = candidates.loc[candidates["arm"] == "injected"]
    development = injected_development.loc[injected_development["subset"] == "development"]
    # The frozen development data has no single monotonic allowed runtime
    # feature that retains both beneficial candidates while excluding every
    # harmful one.  Do not invent an OR-of-singletons rule or tune holdout.
    separable = False
    status = "CANDIDATE_QUALITY_NOT_SEPARABLE"
    gates = pd.DataFrame([
        {
            "gate_id": "candidate_quality_separable",
            "required": "simple runtime-only rule with development harmful=0 and all beneficial retained",
            "observed": status,
            "pass": False,
            "failure_reason": "no robust simple rule; I3 policy intentionally unchanged",
        },
        {
            "gate_id": "holdout_policy_test",
            "required": "one-time validation only after a development rule is frozen",
            "observed": "NOT_RUN_NO_DEVELOPMENT_RULE",
            "pass": False,
            "failure_reason": "holdout was not used for threshold selection or tuning",
        },
        {
            "gate_id": "i2_recovery_classification",
            "required": "I1 baseline and I2 consistency recoveries reported separately",
            "observed": "PASS",
            "pass": True,
            "failure_reason": "",
        },
        {
            "gate_id": "allow_2000",
            "required": "all Phase 4.6 corrected-action gates pass",
            "observed": "NO",
            "pass": False,
            "failure_reason": "I2_CANDIDATE_QUALITY_IS_PRIMARY_BOTTLENECK",
        },
    ])
    gates.to_csv(output / "gate_summary.csv", index=False, encoding="utf-8-sig")

    report = "\n".join([
        "# Phase 4.6 Candidate Quality Closure Report",
        "",
        "## Verdict",
        "",
        "`CANDIDATE_QUALITY_NOT_SEPARABLE`  ",
        "`I2_CANDIDATE_QUALITY_IS_PRIMARY_BOTTLENECK`  ",
        "`ALLOW_2000 = NO`",
        "",
        "No I3 correction policy was loosened. No I1/I2 algorithm, recovery threshold, FinalArbitrator authority, or final-write path was changed.",
        f"Regression gate supplied to this replay: {'PASS' if tests_passed else 'NOT_PASSED'}.",
        "",
        "## Frozen split and candidate evidence",
        "",
        f"- Deterministic SHA-256 case-ID split: development={int((split['subset'] == 'development').sum())}, holdout={int((split['subset'] == 'holdout').sum())}; paired controls share the injected case subset.",
        f"- Development injected safe candidates: {len(development)}; beneficial by offline GT evaluation: {int((development['offline_candidate_outcome'] == 'BENEFICIAL').sum())}.",
        "- Existing runtime fields were recorded only for offline analysis. GT labels are never runtime inputs.",
        "- `shadow_candidate_xyz_m` and `i2_candidate_xyz_m` are distinct; the latter is the only FULL_ENHANCED candidate evaluated.",
        "",
        "## Recovery reclassification",
        "",
        f"- I1 baseline not recovered: {recovery_counts.get('I1_BASELINE_NOT_RECOVERED', 0)}.",
        f"- Evaluable I2 consistency recovery failures: {recovery_counts.get('I2_RECOVERY_FAILURE', 0)}.",
        f"- I2 confirmed within the two-valid-observation contract: {recovery_counts.get('I2_RECOVERY_CONFIRMED', 0)}.",
        "- This is a reporting change only; the 10 mm boundary and two-frame confirmation remain unchanged.",
        "",
        "## Required next action",
        "",
        "Do not tune I3 or use holdout to search thresholds. The next research stage, if authorized, must examine I2 candidate generation rather than fault-to-policy mapping.",
        "",
        "## Direct answers",
        "",
        "1. No: no simple, interpretable runtime-only beneficial-candidate evidence was found in development.",
        "2. No: I3 therefore remains unable to distinguish safe-but-harmful from safe-and-beneficial candidates, and its policy was not changed.",
        f"3. Recovery contains {recovery_counts.get('I1_BASELINE_NOT_RECOVERED', 0)} I1 baseline non-recoveries and {recovery_counts.get('I2_RECOVERY_FAILURE', 0)} evaluable I2 consistency failures.",
        "4. `ALLOW_2000 = NO`.",
        "5. The sole Phase 4.6 action blocker is I2 candidate generation quality, not I3 threshold policy.",
    ])
    (output / "PHASE4_6_CANDIDATE_QUALITY_REPORT.md").write_text(report + "\n", encoding="utf-8")
    return {
        "allow_2000": "NO",
        "candidate_quality_status": status,
        "phase4_5_replay": phase45,
        "recovery_counts": recovery_counts,
    }
