from __future__ import annotations

import numpy as np
import pytest

from stereo_dynamic_measurement.pipeline.data_types import MeasurementResult, PhysicsValidationResult


def test_pipeline_results_reject_nonfinite_coordinates_and_keep_raw_separate() -> None:
    measurement = MeasurementResult(
        frame_id=1, timestamp=0.1, point_id="P1", left_xy=(100.0, 50.0), right_xy=(90.0, 50.0),
        disparity_raw=10.0, disparity_subpixel=10.2, xyz_raw=np.array([1.0, 2.0, 3.0]),
    )
    physics = PhysicsValidationResult(
        frame_id=1, timestamp=0.1, point_id="P1", xyz_raw=measurement.xyz_raw,
        xyz_corrected=np.array([1.0, 2.1, 3.0]), physics_confidence=0.6,
    )

    assert measurement.xyz_raw.tolist() == [1.0, 2.0, 3.0]
    assert physics.xyz_corrected.tolist() == [1.0, 2.1, 3.0]
    with pytest.raises(ValueError, match="finite"):
        MeasurementResult(0, 0.0, "P1", (0.0, 0.0), (0.0, 0.0), 1.0, 1.0, np.array([np.nan, 0.0, 0.0]))
