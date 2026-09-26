"""Validation harness for Innovation 2 without ground-truth leakage.

Runtime physics analysis only observes measurements available at or before the
current frame.  Ground truth is accepted exclusively by the separate
evaluation function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .physics_confidence import (
    EvidenceTerm,
    PhysicsConfidenceConfig,
    compute_physics_confidence,
)
from .spectral_analysis import SpectrumResult, analyze_spectrum
from .trajectory_corrector import CorrectionConfig, correct_trajectory_point
from .transient_gate import TransientGateConfig, decide_transient


@dataclass(frozen=True)
class PhysicsValidationConfig:
    fs_hz: float = 100.0
    duration_s: float = 5.0
    frequency_hz: float = 5.0
    jump_frame: int = 250
    jump_mm: float = 80.0
    threshold: float = 0.65


@dataclass(frozen=True)
class RuntimePhysicsAnalysis:
    trajectories: pd.DataFrame
    confidence: pd.DataFrame
    spectra: dict[str, SpectrumResult]


@dataclass(frozen=True)
class GroundTruthEvaluation:
    xyz_mm: pd.DataFrame
    phase_rad: dict[str, float]
    frequency_hz: float
    injected_fault: str


@dataclass(frozen=True)
class PhysicsValidationResult:
    raw_vs_corrected_csv: Path
    physics_confidence_csv: Path
    trajectory_comparison_png: Path
    psd_comparison_png: Path
    summary: dict[str, float | str]

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "raw_vs_corrected_csv": self.raw_vs_corrected_csv,
            "physics_confidence_csv": self.physics_confidence_csv,
            "trajectory_comparison_png": self.trajectory_comparison_png,
            "psd_comparison_png": self.psd_comparison_png,
        }


_REQUIRED_COLUMNS = {
    "frame",
    "timestamp_s",
    "point_id",
    "raw_X_mm",
    "raw_Y_mm",
    "raw_Z_mm",
}


def _unavailable(reason: str) -> EvidenceTerm:
    return EvidenceTerm.unavailable(reason)


def _residual_term(residual_mm: float | None, scale_mm: float = 15.0) -> EvidenceTerm:
    if residual_mm is None or not np.isfinite(residual_mm):
        return _unavailable("insufficient measured history")
    return EvidenceTerm(
        value=float(np.exp(-max(float(residual_mm), 0.0) / scale_mm)),
        valid=True,
        reliability=1.0,
    )


def _dominant_frequency_term(
    point_history: list[np.ndarray],
    *,
    fs_hz: float,
) -> tuple[EvidenceTerm, float | None, SpectrumResult | None]:
    if len(point_history) < 16:
        return _unavailable("spectral window too short"), None, None
    y = np.asarray([xyz[1] for xyz in point_history[-64:]], dtype=float)
    spectrum = analyze_spectrum(y, fs_hz=fs_hz, nperseg=min(64, len(y)))
    return (
        EvidenceTerm(value=1.0, valid=True, reliability=min(1.0, len(y) / 32.0)),
        float(spectrum.dominant_frequency_hz),
        spectrum,
    )


def run_runtime_physics_analysis(
    raw_measurements: pd.DataFrame,
    *,
    fs_hz: float,
    threshold: float,
) -> RuntimePhysicsAnalysis:
    """Run causal, measurement-only analysis.

    The function deliberately has no ground-truth, injected-frequency, phase,
    or fault arguments.  Histories are updated only after a frame is analyzed.
    """

    missing = _REQUIRED_COLUMNS.difference(raw_measurements.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")

    data = raw_measurements.sort_values(["frame", "point_id"]).reset_index(drop=True)
    histories: dict[str, list[np.ndarray]] = {}
    previous: dict[str, np.ndarray] = {}
    trajectory_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    latest_spectra: dict[str, SpectrumResult] = {}

    confidence_config = PhysicsConfidenceConfig(threshold=threshold)
    correction_config = CorrectionConfig(max_correction_mm=25.0)
    gate_config = TransientGateConfig()

    for frame, frame_data in data.groupby("frame", sort=True):
        current = {
            str(row.point_id): np.asarray(
                [row.raw_X_mm, row.raw_Y_mm, row.raw_Z_mm], dtype=float
            )
            for row in frame_data.itertuples(index=False)
        }
        displacements = {
            point_id: xyz - previous[point_id]
            for point_id, xyz in current.items()
            if point_id in previous
        }
        median_displacement = (
            np.median(np.stack(list(displacements.values())), axis=0)
            if displacements
            else None
        )

        measured_frequencies: dict[str, float] = {}
        provisional_spectra: dict[str, SpectrumResult] = {}
        for point_id, xyz in current.items():
            causal_history = [*histories.get(point_id, []), xyz.copy()]
            _, dominant_frequency, spectrum = _dominant_frequency_term(
                causal_history, fs_hz=fs_hz
            )
            if dominant_frequency is not None and spectrum is not None:
                measured_frequencies[point_id] = dominant_frequency
                provisional_spectra[point_id] = spectrum
        frequency_consensus = (
            float(np.median(list(measured_frequencies.values())))
            if measured_frequencies
            else None
        )

        pending_history_updates: dict[str, np.ndarray] = {}
        for row in frame_data.itertuples(index=False):
            point_id = str(row.point_id)
            xyz = current[point_id]
            history = histories.get(point_id, [])

            temporal_residual: float | None = None
            predicted_xyz: np.ndarray | None = None
            if len(history) >= 2:
                predicted_xyz = history[-1] + (history[-1] - history[-2])
                temporal_residual = float(np.linalg.norm(xyz - predicted_xyz))

            spatial_residual: float | None = None
            sync_ratio: float | None = None
            if point_id in displacements and median_displacement is not None:
                spatial_residual = float(
                    np.linalg.norm(displacements[point_id] - median_displacement)
                )
                sync_ratio = 1.0 if spatial_residual <= 5.0 else 0.0

            flow_3d_residual: float | None = None
            flow_columns = ("flow_dx_mm", "flow_dy_mm", "flow_dz_mm")
            if point_id in displacements and all(hasattr(row, name) for name in flow_columns):
                observed_motion = np.asarray(
                    [getattr(row, name) for name in flow_columns], dtype=float
                )
                if np.isfinite(observed_motion).all():
                    flow_3d_residual = float(
                        np.linalg.norm(displacements[point_id] - observed_motion)
                    )

            frequency_term: EvidenceTerm
            dominant_frequency = measured_frequencies.get(point_id)
            if dominant_frequency is None or frequency_consensus is None:
                frequency_term = _unavailable("spectral window too short")
            else:
                resolution = max(fs_hz / max(len(history) + 1, 1), 0.1)
                frequency_term = EvidenceTerm(
                    value=float(
                        np.exp(
                            -abs(dominant_frequency - frequency_consensus) / resolution
                        )
                    ),
                    valid=True,
                    reliability=min(1.0, (len(history) + 1) / 32.0),
                )

            components = {
                "spectral": frequency_term,
                "phase": _unavailable(
                    "no independent causal phase baseline in validation input"
                ),
                "coherence": _unavailable(
                    "no independent causal coherence baseline in validation input"
                ),
                "spatial": _residual_term(spatial_residual),
                "temporal": _residual_term(temporal_residual),
                "flow_3d": _residual_term(flow_3d_residual),
            }
            physics = compute_physics_confidence(
                evidence=components,
                config=confidence_config,
            )

            transient = decide_transient(
                temporal_residual_mm=(
                    float("nan") if temporal_residual is None else temporal_residual
                ),
                flow_3d=components["flow_3d"],
                spatial=components["spatial"],
                visual_consistency=_unavailable("2D visual evidence not supplied"),
                synchronous_motion_ratio=sync_ratio,
                config=gate_config,
            )
            neighbor_xyz = np.asarray(
                [other for other_id, other in current.items() if other_id != point_id],
                dtype=float,
            )
            correction = correct_trajectory_point(
                raw_xyz_mm=xyz,
                predicted_xyz_mm=predicted_xyz,
                neighbor_xyz_mm=neighbor_xyz,
                c_phy=1.0 if physics.c_phy is None else physics.c_phy,
                config=correction_config,
                transient_decision=transient,
            )

            trajectory_rows.append(
                {
                    **row._asdict(),
                    "corrected_X_mm": float(correction.corrected_xyz_mm[0]),
                    "corrected_Y_mm": float(correction.corrected_xyz_mm[1]),
                    "corrected_Z_mm": float(correction.corrected_xyz_mm[2]),
                    "correction_applied": bool(correction.correction_applied),
                    "correction_method": correction.correction_method,
                    "correction_magnitude_mm": float(correction.correction_magnitude_mm),
                    "correction_reason": correction.correction_reason,
                    "transient_protected": bool(correction.transient_protected),
                }
            )

            confidence_row: dict[str, object] = {
                "frame": int(frame),
                "timestamp_s": float(row.timestamp_s),
                "point_id": point_id,
                "C_phy": physics.c_phy,
                "C_phy_valid": physics.valid,
                "R_phy": physics.r_phy,
                "num_valid_evidence": physics.num_valid_evidence,
                "available_evidence": "|".join(physics.available_evidence_names),
                "missing_evidence": "|".join(physics.missing_evidence_names),
                "temporal_residual_mm": temporal_residual,
                "spatial_residual_mm": spatial_residual,
                "flow_3d_residual_mm": flow_3d_residual,
                "synchronous_motion_ratio": sync_ratio,
                "dominant_frequency_hz": dominant_frequency,
                "transient_classification": (
                    "possible_real_transient"
                    if transient.is_possible_real_transient
                    else "possible_measurement_error"
                    if transient.possible_measurement_error
                    else "undetermined"
                ),
                "allow_candidate_correction": transient.allow_correction,
            }
            for name, term in physics.components.items():
                confidence_row[f"{name}_value"] = term.value
                confidence_row[f"{name}_valid"] = term.valid
                confidence_row[f"{name}_reliability"] = term.reliability
                confidence_row[f"{name}_effective_weight"] = physics.effective_weights.get(
                    name, 0.0
                )
            confidence_rows.append(confidence_row)
            pending_history_updates[point_id] = xyz.copy()

        for point_id, xyz in pending_history_updates.items():
            histories.setdefault(point_id, []).append(xyz)
            previous[point_id] = xyz.copy()
        latest_spectra.update(provisional_spectra)

    return RuntimePhysicsAnalysis(
        trajectories=pd.DataFrame(trajectory_rows),
        confidence=pd.DataFrame(confidence_rows),
        spectra=latest_spectra,
    )


def evaluate_runtime_against_ground_truth(
    runtime: RuntimePhysicsAnalysis,
    ground_truth: GroundTruthEvaluation,
) -> dict[str, float | str]:
    """Evaluate immutable runtime output; never feed labels back into analysis."""

    truth_columns = {
        "frame",
        "point_id",
        "X_gt_mm",
        "Y_gt_mm",
        "Z_gt_mm",
    }
    missing = truth_columns.difference(ground_truth.xyz_mm.columns)
    if missing:
        raise ValueError(f"missing ground-truth columns: {sorted(missing)}")
    merged = runtime.trajectories.merge(
        ground_truth.xyz_mm[list(truth_columns)],
        on=["frame", "point_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("runtime output and ground truth have no matching samples")

    gt = merged[["X_gt_mm", "Y_gt_mm", "Z_gt_mm"]].to_numpy(dtype=float)
    raw = merged[["raw_X_mm", "raw_Y_mm", "raw_Z_mm"]].to_numpy(dtype=float)
    candidate = merged[
        ["corrected_X_mm", "corrected_Y_mm", "corrected_Z_mm"]
    ].to_numpy(dtype=float)
    raw_rmse = float(np.sqrt(np.mean(np.sum((raw - gt) ** 2, axis=1))))
    candidate_rmse = float(
        np.sqrt(np.mean(np.sum((candidate - gt) ** 2, axis=1)))
    )
    return {
        "raw_3d_rmse_mm": raw_rmse,
        "candidate_3d_rmse_mm": candidate_rmse,
        "candidate_improvement_mm": raw_rmse - candidate_rmse,
        "evaluation_frequency_hz": float(ground_truth.frequency_hz),
        "evaluation_phase_count": float(len(ground_truth.phase_rad)),
        "injected_fault": ground_truth.injected_fault,
    }


def _generate_validation_data(config: PhysicsValidationConfig) -> pd.DataFrame:
    frames = int(round(config.duration_s * config.fs_hz)) + 1
    if not 2 <= config.jump_frame < frames:
        raise ValueError("jump_frame must lie in the generated sequence")
    timestamps = np.arange(frames, dtype=float) / config.fs_hz
    phases = {"P1": 0.0, "P2": 0.35, "P3": 0.70, "P4": 1.05}
    amplitudes = {"P1": 8.0, "P2": 11.0, "P3": 14.0, "P4": 17.0}
    x_positions = {"P1": -150.0, "P2": -50.0, "P3": 50.0, "P4": 150.0}
    rows: list[dict[str, object]] = []
    for frame, timestamp in enumerate(timestamps):
        for point_id, phase in phases.items():
            gt = np.array([x_positions[point_id], 0.0, 2000.0], dtype=float)
            gt[1] = amplitudes[point_id] * np.sin(
                2.0 * np.pi * config.frequency_hz * timestamp + phase
            )
            raw = gt.copy()
            if point_id == "P3" and frame == config.jump_frame:
                raw[1] += config.jump_mm
            rows.append(
                {
                    "frame": frame,
                    "timestamp_s": timestamp,
                    "point_id": point_id,
                    "raw_X_mm": raw[0],
                    "raw_Y_mm": raw[1],
                    "raw_Z_mm": raw[2],
                    "X_gt_mm": gt[0],
                    "Y_gt_mm": gt[1],
                    "Z_gt_mm": gt[2],
                }
            )
    return pd.DataFrame(rows)


def _plot_trajectory(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for point_id, group in data.groupby("point_id"):
        ax.plot(group["timestamp_s"], group["raw_Y_mm"], alpha=0.45, label=f"{point_id} raw")
        ax.plot(
            group["timestamp_s"],
            group["corrected_Y_mm"],
            linewidth=1.3,
            label=f"{point_id} candidate",
        )
        if "Y_gt_mm" in group:
            ax.plot(
                group["timestamp_s"],
                group["Y_gt_mm"],
                linestyle="--",
                linewidth=1.0,
                label=f"{point_id} GT",
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y (mm)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_psd(
    spectra: dict[str, SpectrumResult],
    output_path: Path,
    evaluation_frequency_hz: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for point_id, spectrum in spectra.items():
        ax.semilogy(
            spectrum.frequencies_hz,
            np.maximum(spectrum.psd, np.finfo(float).tiny),
            label=point_id,
        )
    ax.axvline(
        evaluation_frequency_hz,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="evaluation frequency",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_physics_validation(
    config: PhysicsValidationConfig,
    output_dir: str | Path,
) -> PhysicsValidationResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = _generate_validation_data(config)
    raw_columns = sorted(_REQUIRED_COLUMNS)
    runtime = run_runtime_physics_analysis(
        generated[raw_columns],
        fs_hz=config.fs_hz,
        threshold=config.threshold,
    )
    phases = {"P1": 0.0, "P2": 0.35, "P3": 0.70, "P4": 1.05}
    evaluation = GroundTruthEvaluation(
        xyz_mm=generated[
            ["frame", "point_id", "X_gt_mm", "Y_gt_mm", "Z_gt_mm"]
        ].copy(),
        phase_rad=phases,
        frequency_hz=config.frequency_hz,
        injected_fault=f"P3 single-frame +{config.jump_mm:g} mm Y jump",
    )
    summary = evaluate_runtime_against_ground_truth(runtime, evaluation)
    trajectories = runtime.trajectories.merge(
        evaluation.xyz_mm,
        on=["frame", "point_id"],
        how="left",
        validate="one_to_one",
    )

    raw_csv = output / "raw_vs_corrected.csv"
    confidence_csv = output / "physics_confidence.csv"
    trajectory_plot = output / "trajectory_comparison.png"
    psd_plot = output / "psd_comparison.png"
    trajectories.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    runtime.confidence.to_csv(confidence_csv, index=False, encoding="utf-8-sig")
    _plot_trajectory(trajectories, trajectory_plot)
    _plot_psd(runtime.spectra, psd_plot, config.frequency_hz)
    return PhysicsValidationResult(
        raw_vs_corrected_csv=raw_csv,
        physics_confidence_csv=confidence_csv,
        trajectory_comparison_png=trajectory_plot,
        psd_comparison_png=psd_plot,
        summary=summary,
    )
