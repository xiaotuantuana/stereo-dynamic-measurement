from __future__ import annotations

import numpy as np
import pytest

from stereo_dynamic_measurement.innovation2.physics_confidence import PhysicsConfidenceConfig, compute_physics_confidence
from stereo_dynamic_measurement.innovation2.spectral_analysis import analyze_spectrum, coherence_and_phase


def test_welch_psd_coherence_and_phase_recover_known_five_hz_relation() -> None:
    fs = 100.0
    time = np.arange(0.0, 8.0, 1.0 / fs)
    first = np.sin(2.0 * np.pi * 5.0 * time)
    second = 0.8 * np.sin(2.0 * np.pi * 5.0 * time + np.deg2rad(20.0))

    spectrum = analyze_spectrum(first, fs_hz=fs, nperseg=256)
    relation = coherence_and_phase(first, second, fs_hz=fs, target_frequency_hz=5.0, nperseg=256)

    assert spectrum.dominant_frequency_hz == pytest.approx(5.0, abs=0.4)
    assert spectrum.band_energy > 0.0
    assert relation.coherence > 0.99
    assert abs(abs(relation.phase_difference_rad) - np.deg2rad(20.0)) < 0.05


def test_weighted_physics_confidence_is_normalized_and_reports_residual() -> None:
    config = PhysicsConfidenceConfig(weights={"flow_3d": 0.2, "temporal": 0.2, "spatial": 0.2, "spectral": 0.15, "phase": 0.1, "coherence": 0.15})
    result = compute_physics_confidence({"flow_3d": 0.9, "temporal": 0.8, "spatial": 0.1, "spectral": 0.9, "phase": 0.8, "coherence": 0.95}, config)

    assert 0.0 <= result.c_phy <= 1.0
    assert result.r_phy == pytest.approx(1.0 - result.c_phy)
    assert result.c_phy < 0.8
