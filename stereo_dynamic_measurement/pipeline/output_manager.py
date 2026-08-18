from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data_types import DiagnosisResult, MeasurementResult, PhysicsValidationResult


class ExperimentOutputManager:
    """Persist raw, validated and recovered results without overwriting evidence."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)

    @classmethod
    def create(cls, root: str | Path, config: dict[str, Any]) -> "ExperimentOutputManager":
        manager = cls(Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        (manager.output_dir / "figures").mkdir()
        (manager.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return manager

    def write_frame(
        self, measurement: MeasurementResult, physics: PhysicsValidationResult, diagnosis: DiagnosisResult, *, runtime_ms: float
    ) -> dict[str, Path]:
        record = {
            "timestamp": measurement.timestamp, "frame_id": measurement.frame_id, "point_id": measurement.point_id,
            "x_left": measurement.left_xy[0], "y_left": measurement.left_xy[1], "x_right": measurement.right_xy[0], "y_right": measurement.right_xy[1],
            "disparity_raw": measurement.disparity_raw, "disparity_subpixel": measurement.disparity_subpixel,
            "X_raw": measurement.xyz_raw[0], "Y_raw": measurement.xyz_raw[1], "Z_raw": measurement.xyz_raw[2],
            "X_validated": physics.xyz_corrected[0], "Y_validated": physics.xyz_corrected[1], "Z_validated": physics.xyz_corrected[2],
            "X_final": physics.xyz_corrected[0], "Y_final": physics.xyz_corrected[1], "Z_final": physics.xyz_corrected[2],
            "gradient_score": measurement.gradient_score, "texture_score": measurement.texture_score, "blur_score": measurement.blur_score,
            "flow_u": measurement.flow_u, "flow_v": measurement.flow_v, "flow_fb_error": measurement.flow_fb_error,
            "lr_residual": measurement.lr_residual, "epipolar_residual": measurement.epipolar_residual,
            "matching_cost": measurement.matching_cost, "neighbor_residual": measurement.neighbor_residual,
            "temporal_residual": measurement.temporal_residual, "measurement_confidence": measurement.measurement_confidence,
            "physics_confidence": physics.physics_confidence, "fault_type": diagnosis.fault_type,
            "fault_score": diagnosis.fault_score, "recovery_action": diagnosis.recovery_action.action_type,
            "recovery_success": diagnosis.recovery_success,
        }
        paths = {
            "raw_measurements": self.output_dir / "raw_measurements.csv",
            "validated_measurements": self.output_dir / "validated_measurements.csv",
            "final_measurements": self.output_dir / "final_measurements.csv",
            "measurement_confidence": self.output_dir / "measurement_confidence.csv",
            "physics_confidence": self.output_dir / "physics_confidence.csv",
            "fault_log": self.output_dir / "fault_log.csv",
            "runtime": self.output_dir / "runtime.csv",
        }
        self._append(paths["raw_measurements"], {key: value for key, value in record.items() if key.startswith(("X_raw", "Y_raw", "Z_raw", "timestamp", "frame_id", "point_id"))})
        self._append(paths["validated_measurements"], record)
        self._append(paths["final_measurements"], record)
        self._append(paths["measurement_confidence"], {"frame_id": measurement.frame_id, "point_id": measurement.point_id, "measurement_confidence": measurement.measurement_confidence})
        self._append(paths["physics_confidence"], {"frame_id": physics.frame_id, "point_id": physics.point_id, "physics_confidence": physics.physics_confidence})
        self._append(paths["fault_log"], {"frame_id": diagnosis.frame_id, "point_id": diagnosis.point_id, "fault_type": diagnosis.fault_type, "fault_score": diagnosis.fault_score, "recovery_action": diagnosis.recovery_action.action_type, "recovery_success": diagnosis.recovery_success})
        self._append(paths["runtime"], {"frame_id": measurement.frame_id, "point_id": measurement.point_id, "runtime_ms": runtime_ms})
        return paths

    @staticmethod
    def _append(path: Path, row: dict[str, Any]) -> None:
        pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8-sig")
