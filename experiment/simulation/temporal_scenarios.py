from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalScenario:
    name: str
    raw_measurements: pd.DataFrame
    ground_truth: pd.DataFrame
    fault_keys: frozenset[tuple[int, str]]
    expected_rows: int
    legitimate_motion: bool = False
    provenance: str = "CONTROLLED"


def build_temporal_scenarios(*, frame_count: int = 96, seed: int = 20260827) -> dict[str, TemporalScenario]:
    if frame_count < 40:
        raise ValueError("temporal validation requires at least 40 frames")
    rng = np.random.default_rng(seed)
    point_ids = ("P1", "P2", "P3")
    x_positions = {"P1": -100.0, "P2": 0.0, "P3": 100.0}

    def make(
        name: str,
        truth_y: np.ndarray,
        noise_std: float,
        injections: dict[tuple[int, str], float] | None = None,
        *,
        missing: frozenset[tuple[int, str]] = frozenset(),
        legitimate_motion: bool = False,
    ) -> TemporalScenario:
        truth_rows: list[dict[str, object]] = []
        raw_rows: list[dict[str, object]] = []
        injected = injections or {}
        for frame in range(frame_count):
            for point_id in point_ids:
                y = float(truth_y[frame])
                truth_rows.append({
                    "frame": frame, "point_id": point_id,
                    "X_gt_mm": x_positions[point_id], "Y_gt_mm": y, "Z_gt_mm": 2000.0,
                })
                key = (frame, point_id)
                if key in missing:
                    continue
                raw_rows.append({
                    "frame": frame, "timestamp_s": frame / 30.0, "point_id": point_id,
                    "raw_X_mm": x_positions[point_id],
                    "raw_Y_mm": y + float(rng.normal(0.0, noise_std)) + injected.get(key, 0.0),
                    "raw_Z_mm": 2000.0,
                    "flow_dx_mm": 0.0,
                    "flow_dy_mm": 0.0 if frame == 0 else float(truth_y[frame] - truth_y[frame - 1]),
                    "flow_dz_mm": 0.0,
                })
        return TemporalScenario(
            name=name,
            raw_measurements=pd.DataFrame(raw_rows),
            ground_truth=pd.DataFrame(truth_rows),
            fault_keys=frozenset(set(injected) | set(missing)),
            expected_rows=frame_count * len(point_ids),
            legitimate_motion=legitimate_motion,
        )

    middle = frame_count // 2
    continuous = {(frame, "P2"): 60.0 for frame in range(middle, middle + 4)}
    outlier_frames = sorted(rng.choice(np.arange(8, frame_count - 4), size=4, replace=False))
    random_outliers = {(int(frame), "P2"): float(rng.choice((-55.0, 55.0))) for frame in outlier_frames}
    zero = np.zeros(frame_count, dtype=float)
    return {
        "stable": make("stable", zero, 0.2),
        "gaussian_noise": make("gaussian_noise", zero, 3.0),
        "single_jump": make("single_jump", zero, 0.3, {(middle, "P2"): 80.0}),
        "continuous_outlier": make("continuous_outlier", zero, 0.3, continuous),
        "slow_drift": make("slow_drift", np.arange(frame_count) * 0.5, 0.3, legitimate_motion=True),
        "fast_legitimate_motion": make(
            "fast_legitimate_motion", 1.5 * np.arange(frame_count) + 0.03 * np.arange(frame_count) ** 2,
            0.3, legitimate_motion=True,
        ),
        "frame_loss": make(
            "frame_loss", zero, 0.3, missing=frozenset({(middle, "P2")}),
        ),
        "measurement_outlier": make("measurement_outlier", zero, 0.3, random_outliers),
    }
