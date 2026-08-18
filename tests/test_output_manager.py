from __future__ import annotations

import numpy as np

from stereo_dynamic_measurement.pipeline.data_types import DiagnosisResult, MeasurementResult, PhysicsValidationResult, RecoveryAction
from stereo_dynamic_measurement.pipeline.output_manager import ExperimentOutputManager


def test_output_manager_writes_auditable_separate_raw_validated_and_final_outputs(tmp_path) -> None:
    manager = ExperimentOutputManager.create(tmp_path, {"simulation": {"random_seed": 1}})
    measurement = MeasurementResult(0, 0.0, "P1", (10.0, 20.0), (5.0, 20.0), 5.0, 5.1, np.array([1.0, 2.0, 3.0]), measurement_confidence=0.8)
    physics = PhysicsValidationResult(0, 0.0, "P1", measurement.xyz_raw, np.array([1.0, 2.1, 3.0]), physics_confidence=0.7, correction_applied=True, correction_method="history", correction_magnitude_mm=0.1)
    diagnosis = DiagnosisResult(0, 0.0, "P1", "NORMAL", 0.0, RecoveryAction("NONE"), True, 0.7, 0.7)

    paths = manager.write_frame(measurement, physics, diagnosis, runtime_ms=2.5)

    assert (manager.output_dir / "config.yaml").exists()
    assert paths["final_measurements"].exists()
    assert "X_raw" in paths["final_measurements"].read_text(encoding="utf-8-sig")
