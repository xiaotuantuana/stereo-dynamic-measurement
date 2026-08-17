from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


TrajectoryKind = Literal["sine", "multi_frequency", "impact_decay"]


@dataclass(frozen=True)
class TrajectorySpec:
    """A scalar displacement specification; all amplitudes are mm."""

    kind: TrajectoryKind = "sine"
    amplitude_mm: float = 10.0
    frequency_hz: float = 2.0
    phase_rad: float = 0.0
    secondary_amplitude_mm: float = 4.0
    secondary_frequency_hz: float = 4.5
    secondary_phase_rad: float = 0.4
    damping_per_s: float = 1.5

    def __post_init__(self) -> None:
        if self.kind not in {"sine", "multi_frequency", "impact_decay"}:
            raise ValueError(f"Unsupported trajectory kind: {self.kind}")
        if self.frequency_hz <= 0 or self.secondary_frequency_hz <= 0:
            raise ValueError("Frequencies must be positive")
        if self.damping_per_s < 0:
            raise ValueError("damping_per_s must be non-negative")


@dataclass(frozen=True)
class MultiPointStructure:
    """Four target points sharing a primary frequency with known rigid offsets."""

    frequency_hz: float = 2.0
    base_positions_mm: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "P1": (-225.0, 0.0, 2000.0), "P2": (-75.0, 0.0, 2000.0),
            "P3": (75.0, 0.0, 2000.0), "P4": (225.0, 0.0, 2000.0),
        }
    )
    amplitudes_mm: dict[str, float] = field(
        default_factory=lambda: {"P1": 8.0, "P2": 12.0, "P3": 16.0, "P4": 10.0}
    )
    phases_rad: dict[str, float] = field(
        default_factory=lambda: {"P1": 0.0, "P2": 0.15, "P3": 0.30, "P4": 0.45}
    )
    motion_axis: int = 1
    trajectory_kind: TrajectoryKind = "sine"

    @classmethod
    def default(cls) -> "MultiPointStructure":
        return cls()

    def __post_init__(self) -> None:
        required = {"P1", "P2", "P3", "P4"}
        if set(self.base_positions_mm) != required or set(self.amplitudes_mm) != required or set(self.phases_rad) != required:
            raise ValueError("MultiPointStructure requires P1 through P4 in every mapping")
        if self.frequency_hz <= 0 or self.motion_axis not in {0, 1, 2}:
            raise ValueError("frequency_hz must be positive and motion_axis must be 0, 1, or 2")


def generate_trajectory(time_s: np.ndarray, spec: TrajectorySpec) -> np.ndarray:
    """Generate displacement in mm without filtering impact or decay components."""
    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or np.any(time < 0) or not np.isfinite(time).all():
        raise ValueError("time_s must be a finite non-negative one-dimensional array")
    omega = 2.0 * np.pi * spec.frequency_hz
    primary = spec.amplitude_mm * np.sin(omega * time + spec.phase_rad)
    if spec.kind == "sine":
        return primary
    if spec.kind == "multi_frequency":
        return primary + spec.secondary_amplitude_mm * np.sin(
            2.0 * np.pi * spec.secondary_frequency_hz * time + spec.secondary_phase_rad
        )
    return spec.amplitude_mm * np.exp(-spec.damping_per_s * time) * np.sin(omega * time + spec.phase_rad)


def generate_multipt_trajectory(time_s: np.ndarray, structure: MultiPointStructure) -> pd.DataFrame:
    """Return ground-truth 3D trajectories for P1–P4 in long-form CSV-ready layout."""
    time = np.asarray(time_s, dtype=np.float64)
    rows: list[dict[str, float | str]] = []
    for point_id in ("P1", "P2", "P3", "P4"):
        spec = TrajectorySpec(
            kind=structure.trajectory_kind, amplitude_mm=structure.amplitudes_mm[point_id],
            frequency_hz=structure.frequency_hz, phase_rad=structure.phases_rad[point_id],
        )
        displacement = generate_trajectory(time, spec)
        base = np.asarray(structure.base_positions_mm[point_id], dtype=np.float64)
        coordinates = np.repeat(base[None, :], len(time), axis=0)
        coordinates[:, structure.motion_axis] += displacement
        rows.extend(
            {
                "time_s": float(t), "X_gt_mm": float(x), "Y_gt_mm": float(y), "Z_gt_mm": float(z),
                "point_id": point_id, "displacement_mm": float(d), "trajectory_kind": structure.trajectory_kind,
            }
            for t, (x, y, z), d in zip(time, coordinates, displacement, strict=True)
        )
    return pd.DataFrame(rows, columns=["time_s", "X_gt_mm", "Y_gt_mm", "Z_gt_mm", "point_id", "displacement_mm", "trajectory_kind"])
