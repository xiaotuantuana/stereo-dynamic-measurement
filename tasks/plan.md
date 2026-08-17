# Implementation Plan: Stereo Dynamic Measurement Task 1

## Overview

Add a mm-native camera model and deterministic stereo simulation package beside the existing research package.

## Architecture Decisions

- New package avoids changing existing uncommitted work.
- OpenCV FileStorage handles XML; YAML/JSON/NPZ loaders normalize to one `StereoCameraModel`.
- The simulator operates only on point geometry; it does not duplicate later matching algorithms.

## Task List

### Phase 1: Foundation

- [ ] Task 1: Add package layout, typed calibration model, loaders, and DLT triangulation.
- [ ] Task 2: Add unit tests for all calibration formats and zero-noise reconstruction.

### Checkpoint: Foundation

- [ ] Focused calibration tests pass.

### Phase 2: Dynamic synthetic data

- [ ] Task 3: Add three trajectory classes and four-point structural configuration.
- [ ] Task 4: Add stereo projection, noise injection, dataset export, and error metrics.
- [ ] Task 5: Add tests for trajectories, point constraints, noise reporting, and an end-to-end dataset.

### Checkpoint: Simulation

- [ ] Simulation tests pass and output schema is inspected.

### Phase 3: Usability

- [ ] Task 6: Add YAML configuration, one-command runner, visualizations, requirements, and usage documentation.
- [ ] Task 7: Run the complete test suite and a manual simulation smoke test.

### Checkpoint: Complete

- [ ] All task-1 acceptance criteria pass without modifying existing `stereo_research` files.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Existing project mixes metres and millimetres | High | New APIs and output field names explicitly carry `_mm`; adapters are deferred. |
| XML calibration schemas vary | Medium | Accept common OpenCV node names and provide clear missing-field errors. |
| Noisy points become non-visible | Low | Validate positive depth and flag invalid projection rows. |
