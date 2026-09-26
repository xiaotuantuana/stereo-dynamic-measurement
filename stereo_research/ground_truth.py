"""External physical ground-truth CSV support for thesis experiments."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExternalGroundTruthLoader:
    timestamps_s: np.ndarray
    xyz_m: np.ndarray
    source: str

    @classmethod
    def from_csv(cls, path: str | Path, position_axis: str = "Z") -> "ExternalGroundTruthLoader":
        csv_path = Path(path)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "timestamp_s" not in rows[0]:
            raise ValueError("External ground truth requires a timestamp_s column")
        timestamps = np.asarray([float(row["timestamp_s"]) for row in rows], dtype=np.float64)
        if timestamps.ndim != 1 or timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("External ground-truth timestamps must be strictly increasing")
        xyz = np.full((len(rows), 3), np.nan, dtype=np.float64)
        if any(key in rows[0] for key in ("X_gt", "Y_gt", "Z_gt")):
            for index, row in enumerate(rows):
                for axis, key in enumerate(("X_gt", "Y_gt", "Z_gt")):
                    if row.get(key) not in (None, ""):
                        xyz[index, axis] = float(row[key])
        elif "position_mm" in rows[0]:
            axis = {"X": 0, "Y": 1, "Z": 2}.get(position_axis.upper())
            if axis is None:
                raise ValueError("position_axis must be X, Y, or Z")
            xyz[:, axis] = np.asarray([float(row["position_mm"]) for row in rows]) / 1000.0
        else:
            raise ValueError("External GT needs X_gt/Y_gt/Z_gt, Z_gt, or position_mm")
        return cls(timestamps, xyz, str(csv_path.resolve()))

    def interpolate(self, timestamp_s: float) -> dict[str, float | bool | str | None]:
        if timestamp_s < self.timestamps_s[0] or timestamp_s > self.timestamps_s[-1]:
            return {"timestamp_gt_s": None, "X_gt": None, "Y_gt": None, "Z_gt": None, "gt_valid": False, "gt_source": self.source}
        xyz = np.asarray([
            np.interp(timestamp_s, self.timestamps_s[~np.isnan(self.xyz_m[:, axis])], self.xyz_m[~np.isnan(self.xyz_m[:, axis]), axis])
            if np.any(~np.isnan(self.xyz_m[:, axis])) else np.nan
            for axis in range(3)
        ])
        return {"timestamp_gt_s": timestamp_s, "X_gt": None if np.isnan(xyz[0]) else float(xyz[0]), "Y_gt": None if np.isnan(xyz[1]) else float(xyz[1]), "Z_gt": None if np.isnan(xyz[2]) else float(xyz[2]), "gt_valid": bool(np.any(~np.isnan(xyz))), "gt_source": self.source}
