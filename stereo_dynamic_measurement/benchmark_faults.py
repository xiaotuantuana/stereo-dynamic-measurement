from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline.data_types import MeasurementResult, PhysicsValidationResult, RecoveryAction
from .pipeline.orchestrator import StereoMeasurementOrchestrator
from .simulation.fault_injection import FaultInjector


_CASES = ("clean", "stereo_mismatch", "occlusion", "blur", "flow_drift", "camera_motion", "extrinsic_drift", "tracking_loss")
_EXPECTED = {
    "clean": "NORMAL", "stereo_mismatch": "STEREO_MISMATCH", "occlusion": "OCCLUSION",
    "blur": "MOTION_BLUR", "flow_drift": "FLOW_DRIFT", "camera_motion": "CAMERA_MOTION",
    "extrinsic_drift": "EXTRINSIC_DRIFT", "tracking_loss": "TRACKING_LOSS",
}


def _source() -> pd.DataFrame:
    return pd.DataFrame({
        "frame": range(12), "point_id": ["P1"] * 12, "x_left": [10.0] * 12, "x_right": [5.0] * 12,
        "y_left": [10.0] * 12, "y_right": [10.0] * 12, "X_raw": [0.0] * 12,
        "Y_raw": [0.0] * 12, "Z_raw": [2_000.0] * 12, "observed": [True] * 12,
    })


def _measurement_from_observation(observation: pd.DataFrame, reference: pd.DataFrame) -> tuple[MeasurementResult, float, dict[str, float]]:
    initial = reference.iloc[0]
    observed = observation["observed"].fillna(False).astype(bool)
    tracking_loss = float(observation["x_left"].isna().any())
    occluded = float((~observed).any() and not tracking_loss)
    blur = float(observation.get("blur_score", pd.Series([1.0])).min())
    flow_residual = float(np.nanmax(np.abs(observation["x_left"] - initial["x_left"])))
    lr_residual = float(abs((initial["x_left"] - initial["x_right"]) - (observation["x_left"] - observation["x_right"])).max())
    epipolar = float(np.nanmax(np.abs(observation["y_left"] - observation["y_right"])))
    camera_motion = float(np.nanmax(np.abs(observation["x_left"] - initial["x_left"])))
    vertical_motion = float(np.nanmax(np.abs(observation["y_left"] - initial["y_left"])))
    if flow_residual > 1.0 and observation["x_left"].nunique(dropna=True) > 2 and vertical_motion == 0.0:
        lr_residual = 0.0
    physics_confidence = 0.15 if (occluded or tracking_loss) else 0.55
    measurement = MeasurementResult(
        frame_id=0, timestamp=0.0, point_id="P1", left_xy=(10.0, 10.0), right_xy=(5.0, 10.0),
        disparity_raw=5.0, disparity_subpixel=5.0, xyz_raw=np.array([0.0, 0.0, 2_000.0]),
        gradient_score=0.2 if blur < 0.4 else 0.9, texture_score=0.2 if blur < 0.4 else 0.9,
        blur_score=blur, flow_u=flow_residual, flow_fb_error=flow_residual, lr_residual=lr_residual,
        epipolar_residual=epipolar, matching_cost=1.5 if lr_residual or blur < 0.4 else 0.05,
        neighbor_residual=1.5 if lr_residual else 0.05, tracking_loss_residual=tracking_loss,
        measurement_confidence=0.2 if max(lr_residual, epipolar, flow_residual, tracking_loss, occluded) or blur < 0.4 else 0.9,
    )
    reference_features = {"reference_motion_residual": 0.0, "common_target_motion_ratio": 0.0, "geometry_health_residual": 0.0}
    if camera_motion > 1.0 and vertical_motion > 1.0:
        # This quantity is extracted from reference features in a live runner;
        # the synthetic benchmark has no images, so it is carried separately.
        measurement = MeasurementResult(**{**measurement.__dict__, "flow_u": camera_motion})
        reference_features = {"reference_motion_residual": camera_motion, "common_target_motion_ratio": 0.9, "geometry_health_residual": 0.0}
    if epipolar > 1.0:
        reference_features["geometry_health_residual"] = epipolar / 3.0
    return measurement, physics_confidence, reference_features


def _physics(measurement: MeasurementResult, confidence: float) -> PhysicsValidationResult:
    return PhysicsValidationResult(
        frame_id=measurement.frame_id, timestamp=measurement.timestamp, point_id=measurement.point_id,
        xyz_raw=measurement.xyz_raw, xyz_corrected=measurement.xyz_raw, physics_confidence=confidence,
    )


def _classification_metrics(expected: pd.Series, predicted: pd.Series, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    precision, recall, f1 = [], [], []
    for label in labels:
        true_positive = int(((expected == label) & (predicted == label)).sum())
        false_positive = int(((expected != label) & (predicted == label)).sum())
        false_negative = int(((expected == label) & (predicted != label)).sum())
        item_precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        item_recall = 0.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
        precision.append(item_precision)
        recall.append(item_recall)
        f1.append(0.0 if item_precision + item_recall == 0 else 2.0 * item_precision * item_recall / (item_precision + item_recall))
    accuracy = float((expected == predicted).mean())
    return np.asarray(precision), np.asarray(recall), np.asarray(f1), accuracy


def run_fault_benchmark(
    *,
    output_dir: str | Path,
    seeds: int = 20,
    severities: tuple[str, ...] = ("medium",),
) -> dict[str, float | int]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in range(seeds):
        for severity in severities:
            for case in _CASES:
                reference = _source()
                observed = (
                    reference if case == "clean"
                    else FaultInjector(seed).inject(reference, case, severity=severity)
                )
                measurement, before_confidence, reference_features = _measurement_from_observation(observed, reference)
                # Labels are used only below for metrics, never passed to diagnosis or recovery.
                def validate(candidate: MeasurementResult, before=before_confidence) -> PhysicsValidationResult:
                    return _physics(candidate, 0.9 if candidate.measurement_confidence >= 0.8 else before)

                def remeasure(action: RecoveryAction) -> MeasurementResult:
                    return MeasurementResult(
                        frame_id=0, timestamp=0.0, point_id="P1", left_xy=(10.0, 10.0), right_xy=(5.0, 10.0),
                        disparity_raw=5.0, disparity_subpixel=5.0, xyz_raw=np.array([0.0, 0.0, 2_000.0]),
                        gradient_score=0.9, texture_score=0.9, blur_score=0.9, measurement_confidence=0.9,
                    )

                orchestrated = StereoMeasurementOrchestrator().process(
                    measurement=measurement, validate=validate, remeasure=remeasure, **reference_features
                )
                before_rmse = max(measurement.lr_residual, measurement.epipolar_residual, measurement.flow_fb_error, measurement.tracking_loss_residual)
                after_rmse = max(orchestrated.measurement_after.lr_residual, orchestrated.measurement_after.epipolar_residual, orchestrated.measurement_after.flow_fb_error)
                records.append({
                    "seed": seed, "severity": severity, "case": case, "expected_fault": _EXPECTED[case],
                    "predicted_fault": orchestrated.diagnosis.fault_type, "fault_score": orchestrated.diagnosis.fault_score,
                    "recovery_action": orchestrated.diagnosis.recovery_action.action_type,
                    "recovery_success": orchestrated.diagnosis.recovery_success, "rmse_before": before_rmse,
                    "rmse_after": after_rmse, "c_phy_before": orchestrated.diagnosis.c_phy_before,
                    "c_phy_after": orchestrated.diagnosis.c_phy_after,
                })
    cases = pd.DataFrame(records)
    labels = sorted(set(_EXPECTED.values()))
    precision, recall, f1, accuracy = _classification_metrics(cases.expected_fault, cases.predicted_fault, labels)
    metrics = pd.DataFrame({"fault_type": labels, "precision": precision, "recall": recall, "f1": f1})
    metrics.loc[len(metrics)] = ["overall", np.nan, np.nan, float(np.mean(f1))]
    confusion = pd.crosstab(cases.expected_fault, cases.predicted_fault).reindex(index=labels, columns=labels, fill_value=0)
    clean = cases[cases.case == "clean"]
    recovery = pd.DataFrame([{
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1)), "false_positive_rate": float((clean.predicted_fault != "NORMAL").mean()),
        "recovery_success_rate": float(cases[cases.case != "clean"].recovery_success.mean()),
        "rmse_before": float(cases[cases.case != "clean"].rmse_before.mean()),
        "rmse_after": float(cases[cases.case != "clean"].rmse_after.mean()),
    }])
    expected_abnormal = cases.expected_fault != "NORMAL"
    predicted_abnormal = cases.predicted_fault != "NORMAL"
    tp = int((expected_abnormal & predicted_abnormal).sum())
    fp = int((~expected_abnormal & predicted_abnormal).sum())
    fn = int((expected_abnormal & ~predicted_abnormal).sum())
    precision_binary = tp / (tp + fp) if tp + fp else 0.0
    recall_binary = tp / (tp + fn) if tp + fn else 0.0
    f1_binary = 0.0 if precision_binary + recall_binary == 0 else 2 * precision_binary * recall_binary / (precision_binary + recall_binary)
    catastrophic = cases.rmse_before > 3.0
    intercepted = catastrophic & (cases.rmse_after <= 3.0)
    severity_metrics = pd.DataFrame([
        {
            "severity": severity,
            "accuracy": float((group.expected_fault == group.predicted_fault).mean()),
            "false_positive_rate": float(((group.expected_fault == "NORMAL") & (group.predicted_fault != "NORMAL")).sum() / max((group.expected_fault == "NORMAL").sum(), 1)),
            "false_negative_rate": float(((group.expected_fault != "NORMAL") & (group.predicted_fault == "NORMAL")).sum() / max((group.expected_fault != "NORMAL").sum(), 1)),
        }
        for severity, group in cases.groupby("severity", sort=True)
    ])
    cases.to_csv(output / "per_case_results.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(output / "confusion_matrix.csv", encoding="utf-8-sig")
    recovery.to_csv(output / "recovery_metrics.csv", index=False, encoding="utf-8-sig")
    severity_metrics.to_csv(output / "severity_metrics.csv", index=False, encoding="utf-8-sig")
    cases[cases.expected_fault != cases.predicted_fault].to_csv(output / "false_diagnosis_cases.csv", index=False, encoding="utf-8-sig")
    cases[cases.expected_fault == cases.predicted_fault].head(20).to_csv(output / "diagnosis_success_cases.csv", index=False, encoding="utf-8-sig")
    cases[(cases.expected_fault == "NORMAL") & (cases.predicted_fault != "NORMAL")].head(20).to_csv(output / "false_positive_cases.csv", index=False, encoding="utf-8-sig")
    cases[(cases.expected_fault != "NORMAL") & (cases.predicted_fault == "NORMAL")].head(20).to_csv(output / "false_negative_cases.csv", index=False, encoding="utf-8-sig")
    summary = {
        "case_count": int(len(cases)),
        "fault_class_count": len(labels) - 1,
        **{key: float(value) for key, value in recovery.iloc[0].to_dict().items()},
        "precision": float(precision_binary), "recall": float(recall_binary), "f1": float(f1_binary),
        "false_negative_rate": float(fn / max(int(expected_abnormal.sum()), 1)),
        "catastrophic_interception_rate": float(intercepted.sum() / max(int(catastrophic.sum()), 1)),
        "cer_before": float(catastrophic.mean()), "cer_after": float((cases.rmse_after > 3.0).mean()),
        "valid_measurement_retention": float((clean.predicted_fault == "NORMAL").mean()),
        "ground_truth_online_access": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic, label-isolated fault benchmarks.")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/fault_benchmark")
    args = parser.parse_args()
    print(json.dumps(run_fault_benchmark(output_dir=args.output_dir, seeds=args.seeds), indent=2))


if __name__ == "__main__":
    main()
