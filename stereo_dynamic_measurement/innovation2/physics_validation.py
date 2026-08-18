from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .physics_confidence import PhysicsConfidenceConfig, compute_physics_confidence
from .spectral_analysis import analyze_spectrum, coherence_and_phase
from .temporal_consistency import temporal_prediction_residual
from .trajectory_corrector import CorrectionConfig, correct_trajectory_point


@dataclass(frozen=True)
class PhysicsValidationConfig:
    fs_hz: float = 100.0
    duration_s: float = 5.0
    frequency_hz: float = 5.0
    jump_frame: int = 250
    jump_mm: float = 80.0
    threshold: float = 0.65


@dataclass(frozen=True)
class PhysicsValidationResult:
    raw_vs_corrected_csv: Path
    physics_confidence_csv: Path
    trajectory_comparison_png: Path
    psd_comparison_png: Path

    @property
    def paths(self) -> dict[str, Path]:
        return {"raw_vs_corrected_csv": self.raw_vs_corrected_csv, "physics_confidence_csv": self.physics_confidence_csv, "trajectory_comparison_png": self.trajectory_comparison_png, "psd_comparison_png": self.psd_comparison_png}


def run_physics_validation(config: PhysicsValidationConfig, output_dir: str | Path) -> PhysicsValidationResult:
    """End-to-end four-point 5 Hz validation with a deliberately corrupted P3 sample."""
    frames = int(round(config.duration_s * config.fs_hz)) + 1
    if not 2 <= config.jump_frame < frames:
        raise ValueError("jump_frame must lie in the generated sequence")
    time = np.arange(frames, dtype=float) / config.fs_hz
    point_ids = ("P1", "P2", "P3", "P4")
    phases = np.deg2rad([0.0, 20.0, 40.0, 60.0])
    amplitudes = np.array([8.0, 11.0, 14.0, 17.0])
    x_positions = np.array([-150.0, -50.0, 50.0, 150.0])
    records: list[dict[str, float | int | str | bool]] = []
    for index, point_id in enumerate(point_ids):
        truth_y = amplitudes[index] * np.sin(2.0 * np.pi * config.frequency_hz * time + phases[index])
        raw_y = truth_y.copy()
        if point_id == "P3": raw_y[config.jump_frame] += config.jump_mm
        for frame, timestamp in enumerate(time):
            records.append({"frame": frame, "timestamp_s": timestamp, "point_id": point_id, "X_gt_mm": x_positions[index], "Y_gt_mm": truth_y[frame], "Z_gt_mm": 2000.0, "raw_X_mm": x_positions[index], "raw_Y_mm": raw_y[frame], "raw_Z_mm": 2000.0, "measurement_confidence": 0.95})
    data = pd.DataFrame(records)
    spectra = {point: analyze_spectrum(data[data.point_id == point].raw_Y_mm.to_numpy(), fs_hz=config.fs_hz, nperseg=min(256, frames)) for point in point_ids}
    relations = {point: coherence_and_phase(data[data.point_id == "P1"].raw_Y_mm.to_numpy(), data[data.point_id == point].raw_Y_mm.to_numpy(), fs_hz=config.fs_hz, target_frequency_hz=config.frequency_hz, nperseg=min(256, frames)) for point in point_ids if point != "P1"}
    corrected_rows: list[dict[str, float | int | str | bool]] = []
    conf_rows: list[dict[str, float | int | str | bool]] = []
    confidence_config = PhysicsConfidenceConfig(threshold=config.threshold)
    correction_config = CorrectionConfig(threshold=config.threshold)
    for frame in range(frames):
        current = data[data.frame == frame].copy().sort_values("point_id")
        coeff = np.polyfit(current.raw_X_mm, current.raw_Y_mm, 1)
        spatial_fit = np.polyval(coeff, current.raw_X_mm)
        for _, row in current.iterrows():
            point_id = str(row.point_id)
            point_series = data[data.point_id == point_id].sort_values("frame")
            if frame >= 2:
                history = point_series.iloc[frame - 2:frame][["raw_X_mm", "raw_Y_mm", "raw_Z_mm"]].to_numpy()
                raw = row[["raw_X_mm", "raw_Y_mm", "raw_Z_mm"]].to_numpy(dtype=float)
                temporal = temporal_prediction_residual(history, raw, transient_acceleration_mm=25.0, residual_scale_mm=15.0)
                prediction = np.asarray(temporal.predicted_xyz_mm)
                temporal_evidence = 1.0 - temporal.confidence_penalty
            else:
                raw = row[["raw_X_mm", "raw_Y_mm", "raw_Z_mm"]].to_numpy(dtype=float)
                prediction, temporal_evidence = raw.copy(), 1.0
            spatial_residual = abs(float(row.raw_Y_mm) - float(spatial_fit[current.index.get_loc(row.name)]))
            # A true impact is protected only when it is structurally coherent.
            # An isolated high-acceleration sample (the injected P3 jump) must
            # retain its temporal penalty rather than being treated as a real transient.
            if spatial_residual > 20.0:
                temporal_evidence = min(temporal_evidence, float(np.exp(-float(np.linalg.norm(raw - prediction)) / 15.0)))
            relation = relations.get(point_id)
            coherence = 1.0 if relation is None else relation.coherence
            phase_evidence = 1.0 if relation is None else float(np.exp(-abs(abs(relation.phase_difference_rad) - abs(phases[point_ids.index(point_id)] - phases[0]))))
            spectral_evidence = float(np.exp(-abs(spectra[point_id].dominant_frequency_hz - config.frequency_hz)))
            evidence = {"flow_3d": 1.0, "temporal": temporal_evidence, "spatial": float(np.exp(-spatial_residual / 15.0)), "spectral": spectral_evidence, "phase": phase_evidence, "coherence": coherence}
            physics = compute_physics_confidence(evidence, confidence_config)
            neighbor_xyz = current[current.point_id != point_id][["raw_X_mm", "raw_Y_mm", "raw_Z_mm"]].to_numpy(dtype=float)
            correction = correct_trajectory_point(raw, prediction, neighbor_xyz, c_phy=physics.c_phy, config=correction_config)
            corrected_rows.append({**row.to_dict(), "corrected_X_mm": correction.corrected_xyz_mm[0], "corrected_Y_mm": correction.corrected_xyz_mm[1], "corrected_Z_mm": correction.corrected_xyz_mm[2], "correction_applied": correction.correction_applied, "correction_method": correction.correction_method, "correction_magnitude_mm": correction.correction_magnitude_mm})
            conf_rows.append({"frame": frame, "timestamp_s": row.timestamp_s, "point_id": point_id, "C_phy": physics.c_phy, "r_phy": physics.r_phy, "r_temporal_mm": float(np.linalg.norm(raw - prediction)), "r_spatial_mm": spatial_residual, "dominant_frequency_hz": spectra[point_id].dominant_frequency_hz, "coherence": coherence, "phase_difference_rad": 0.0 if relation is None else relation.phase_difference_rad, **{f"evidence_{key}": value for key, value in physics.components.items()}})
    corrected = pd.DataFrame(corrected_rows)
    confidence = pd.DataFrame(conf_rows)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    raw_path, conf_path = output / "raw_vs_corrected.csv", output / "physics_confidence.csv"
    trajectory_path, psd_path = output / "trajectory_comparison.png", output / "psd_comparison.png"
    corrected.to_csv(raw_path, index=False, encoding="utf-8-sig"); confidence.to_csv(conf_path, index=False, encoding="utf-8-sig")
    _plot_trajectory(corrected, trajectory_path); _plot_psd(spectra, psd_path)
    return PhysicsValidationResult(raw_path, conf_path, trajectory_path, psd_path)


def _plot_trajectory(data: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    p3 = data[data.point_id == "P3"]
    axis.plot(p3.timestamp_s, p3.Y_gt_mm, label="P3 ground truth", linewidth=2)
    axis.plot(p3.timestamp_s, p3.raw_Y_mm, label="P3 raw", alpha=0.75)
    axis.plot(p3.timestamp_s, p3.corrected_Y_mm, label="P3 corrected", linestyle="--")
    axis.set(xlabel="Time (s)", ylabel="Y displacement (mm)", title="P3 jump correction while preserving vibration")
    axis.legend(); axis.grid(alpha=0.25); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_psd(spectra: dict[str, object], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for point_id, spectrum in spectra.items():
        axis.semilogy(spectrum.frequencies_hz, spectrum.psd, label=point_id)
    axis.set(xlim=(0, 15), xlabel="Frequency (Hz)", ylabel="PSD", title="PSD comparison: dominant 5 Hz retained")
    axis.legend(); axis.grid(alpha=0.25); fig.savefig(path, dpi=150); plt.close(fig)
