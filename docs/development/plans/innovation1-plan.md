# Implementation Plan: Innovation 1 Adaptive Stereo

1. [x] Quality, FB-LK, state predictor and precision planning with unit tests.
2. [x] Adaptive radius, local ZNCC-gradient matching, subpixel and neighbor checks with unit tests.
3. [x] Confidence controller and end-to-end adaptive measurement pipeline with fallback tests.
4. [x] Baseline-distance Monte Carlo experiment, config, CSV/plot and integration test.

Each phase is committed only after its focused tests pass. New code remains under `stereo_dynamic_measurement/innovation1`.
