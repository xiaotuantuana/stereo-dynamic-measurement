from __future__ import annotations

import numpy as np
import pytest

from stereo_dynamic_measurement.simulation.trajectory_generator import (
    MultiPointStructure,
    TrajectorySpec,
    generate_multipt_trajectory,
    generate_trajectory,
)


@pytest.mark.parametrize("kind", ["sine", "multi_frequency", "impact_decay"])
def test_all_supported_trajectories_return_mm_displacement(kind: str) -> None:
    spec = TrajectorySpec(kind=kind, amplitude_mm=12.0, frequency_hz=2.0, phase_rad=0.2)
    time_s = np.linspace(0.0, 2.0, 121)

    displacement = generate_trajectory(time_s, spec)

    assert displacement.shape == time_s.shape
    assert np.isfinite(displacement).all()
    assert np.max(np.abs(displacement)) > 0.0


def test_impact_decay_preserves_nonzero_damped_tail() -> None:
    spec = TrajectorySpec(kind="impact_decay", amplitude_mm=20.0, frequency_hz=5.0, damping_per_s=1.5)
    time_s = np.array([0.05, 0.45, 1.05])

    displacement = generate_trajectory(time_s, spec)

    assert abs(displacement[1]) > 0.0
    assert abs(displacement[2]) > 0.0
    assert abs(displacement[2]) < abs(displacement[0])


def test_four_points_share_frequency_and_keep_known_spatial_offsets() -> None:
    time_s = np.linspace(0.0, 1.0, 31)
    structure = MultiPointStructure.default()
    samples = generate_multipt_trajectory(time_s, structure)

    assert set(samples["point_id"]) == {"P1", "P2", "P3", "P4"}
    assert samples.shape == (len(time_s) * 4, 7)
    p1 = samples[samples["point_id"] == "P1"]
    p2 = samples[samples["point_id"] == "P2"]
    assert np.allclose(p2["X_gt_mm"].to_numpy() - p1["X_gt_mm"].to_numpy(), 150.0)
    assert np.allclose(p2["Z_gt_mm"].to_numpy() - p1["Z_gt_mm"].to_numpy(), 0.0)
    expected_p1 = structure.amplitudes_mm["P1"] * np.sin(2.0 * np.pi * structure.frequency_hz * time_s)
    expected_p2 = structure.amplitudes_mm["P2"] * np.sin(
        2.0 * np.pi * structure.frequency_hz * time_s + structure.phases_rad["P2"]
    )
    assert np.allclose(p1["Y_gt_mm"].to_numpy(), expected_p1)
    assert np.allclose(p2["Y_gt_mm"].to_numpy(), expected_p2)
    assert structure.frequency_hz == 2.0
