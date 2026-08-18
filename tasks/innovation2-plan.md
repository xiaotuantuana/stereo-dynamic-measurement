# Implementation Plan: Innovation 2 Physics Confidence

1. [x] Implement input schema, 2D–3D reprojection and temporal/spatial residuals with unit tests.
2. [x] Implement Welch PSD, coherence, phase extraction and weighted `C_phy` with unit tests.
3. [x] Implement auditable trajectory corrector and multi-point evaluator integration test.
4. [x] Implement P3-jump validation generator, CSV/plots, configuration and command-line runner.

All coordinates remain mm; all residual fields encode their unit in the name. No innovation-3 fault-label logic enters this package.
