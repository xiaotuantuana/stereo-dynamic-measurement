from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SpectrumResult:
    frequencies_hz: np.ndarray
    psd: np.ndarray
    dominant_frequency_hz: float
    dominant_amplitude: float
    band_energy: float


@dataclass(frozen=True)
class CoherencePhaseResult:
    frequency_hz: float
    coherence: float
    phase_difference_rad: float
    amplitude_ratio: float


def _validate_series(values: np.ndarray, fs_hz: float, nperseg: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 8 or not np.isfinite(series).all():
        raise ValueError("Spectral analysis needs a finite one-dimensional series with at least 8 samples")
    if fs_hz <= 0 or nperseg < 8:
        raise ValueError("fs_hz and nperseg must be valid")
    return series


def analyze_spectrum(values: np.ndarray, *, fs_hz: float, nperseg: int, band_half_width_hz: float = 0.5) -> SpectrumResult:
    series = _validate_series(values, fs_hz, nperseg)
    frequencies, psd = signal.welch(series, fs=fs_hz, nperseg=min(nperseg, series.size), detrend="linear", scaling="density")
    usable = np.flatnonzero(frequencies > 0)
    if usable.size == 0:
        raise ValueError("PSD contains no positive frequencies")
    index = int(usable[np.argmax(psd[usable])])
    dominant = float(frequencies[index])
    band = np.abs(frequencies - dominant) <= band_half_width_hz
    return SpectrumResult(frequencies, psd, dominant, float(np.sqrt(max(psd[index], 0.0))), float(np.trapezoid(psd[band], frequencies[band])))


def coherence_and_phase(first: np.ndarray, second: np.ndarray, *, fs_hz: float, target_frequency_hz: float, nperseg: int) -> CoherencePhaseResult:
    x, y = _validate_series(first, fs_hz, nperseg), _validate_series(second, fs_hz, nperseg)
    if y.shape != x.shape or target_frequency_hz <= 0:
        raise ValueError("Series shapes must match and target frequency must be positive")
    segment = min(nperseg, x.size)
    frequencies, coherence = signal.coherence(x, y, fs=fs_hz, nperseg=segment, detrend="linear")
    _, cross_spectrum = signal.csd(x, y, fs=fs_hz, nperseg=segment, detrend="linear")
    _, psd_x = signal.welch(x, fs=fs_hz, nperseg=segment, detrend="linear")
    _, psd_y = signal.welch(y, fs=fs_hz, nperseg=segment, detrend="linear")
    index = int(np.argmin(abs(frequencies - target_frequency_hz)))
    return CoherencePhaseResult(float(frequencies[index]), float(np.clip(coherence[index], 0.0, 1.0)), float(np.angle(cross_spectrum[index])), float(np.sqrt(max(psd_y[index], 0.0) / max(psd_x[index], 1e-15))))
