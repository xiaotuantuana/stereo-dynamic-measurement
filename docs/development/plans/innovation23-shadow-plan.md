# Implementation Plan: Innovation 2/3 Shadow Mode First Round

## Overview

Add an auditable, GT-free Innovation 2 runtime evidence path and an Innovation 3 recommendation-only path after the existing stereo pipeline's camera-compensation stage. Preserve every legacy result and state transition, append shadow diagnostics to existing CSV files, and prove complete-sequence Shadow OFF/ON equivalence.

## Architecture Decisions

- Add a thin stateful `ShadowAnalyzer` after `_apply_camera_compensation()`; do not restructure the matching pipeline.
- Treat `FramePointResult` and pipeline state as read-only inputs; shadow history owns copies.
- Use compensated → estimated → measured fallback to select `shadow_input_xyz`, and export the selected stage.
- Keep GT evaluation outside every runtime API.
- Make optional evidence explicit and reliability-weighted; missing evidence is excluded.
- Hard-reject active correction/recovery flags during this round.
- Append shadow CSV fields after all legacy fields.

## Dependency Order

```text
auditable evidence contract
  → transient gate and bounded candidate correction
    → optional fault fingerprint/rules
      → FramePointResult shadow fields and CSV contract
        → ShadowAnalyzer adapter
          → pipeline integration
            → full-sequence isolation and experiment regression
```

## Phase 1: Scientific Runtime Contracts

### Task 1: Auditable missing-aware C_phy fusion

**Description:** Add `EvidenceTerm` and expand the confidence result with validity, available/missing names, and effective weights while retaining compatibility with current numeric/None inputs.

**Acceptance criteria:**
- Missing or invalid evidence is excluded and never becomes 1.0.
- No valid evidence yields an invalid confidence result.
- Reliability changes effective weights and every component is auditable.

**Verification:** `python -m pytest -q tests/test_innovation2_spectral_confidence.py tests/test_shadow_scientific_safety.py`

**Dependencies:** None  
**Files likely touched:** `physics_confidence.py`, new safety test file  
**Estimated scope:** S

### Task 2: Transient gate and bounded candidate correction

**Description:** Add a public transient decision and make correction candidate-only, transient-aware, and bounded by a configurable safety limit.

**Acceptance criteria:**
- Synchronous supported transient returns the unchanged raw candidate.
- Isolated multi-evidence outlier allows correction toward the structural relation.
- Candidate delta cannot exceed `max_correction_mm`.

**Verification:** `python -m pytest -q tests/test_innovation2_correction.py tests/test_innovation2_residuals.py tests/test_shadow_scientific_safety.py`

**Dependencies:** Task 1  
**Files likely touched:** `transient_gate.py`, `trajectory_corrector.py`, safety tests  
**Estimated scope:** M

### Checkpoint: Scientific Safety

- Focused Innovation 2 tests pass.
- Search confirms runtime confidence/correction APIs accept no GT fields.

## Phase 2: Shadow Fault Contracts

### Task 3: Optional fault evidence and recommendation-only diagnosis

**Description:** Make fingerprint evidence optional and teach the existing rule engine to ignore unavailable fields without adding fault classes.

**Acceptance criteria:**
- Missing evidence triggers neither healthy nor faulty rules.
- Existing complete-fingerprint behavior remains compatible.
- Recovery output is a plan only and has no execution callback.

**Verification:** `python -m pytest -q tests/test_innovation3_diagnostics.py tests/test_innovation3_recovery.py tests/test_shadow_fault_adapter.py`

**Dependencies:** Task 1  
**Files likely touched:** `fault_fingerprint.py`, `fault_classifier.py`, fault adapter tests  
**Estimated scope:** S

## Phase 3: Real Pipeline Adapter

### Task 4: Append immutable shadow result and CSV fields

**Description:** Minimally extend `FramePointResult` with optional shadow diagnostics and append corresponding names to the end of `CSV_FIELDS`.

**Acceptance criteria:**
- Every pre-existing CSV field and its order is unchanged.
- Missing shadow evidence serializes as an empty cell.
- XYZ stage fields identify shadow input, candidate, and final without changing legacy fields.

**Verification:** `python -m pytest -q tests/test_backward_compatibility.py tests/test_pipeline_data_types.py tests/test_shadow_pipeline.py`

**Dependencies:** Tasks 1–3  
**Files likely touched:** `models.py`, `runner.py`, shadow pipeline tests  
**Estimated scope:** M

### Task 5: Implement read-only ShadowAnalyzer

**Description:** Build actual runtime physics evidence and fault fingerprint from same-frame `FramePointResult` values plus copied past measurements.

**Acceptance criteria:**
- Input XYZ follows compensated → estimated → measured semantics.
- Fingerprint fields come from actual result fields or remain unavailable.
- Candidate/fault recommendation never changes legacy/final fields or mutable pipeline state.

**Verification:** `python -m pytest -q tests/test_shadow_pipeline.py tests/test_shadow_fault_adapter.py`

**Dependencies:** Task 4  
**Files likely touched:** new `shadow_analysis.py`, adapter tests  
**Estimated scope:** M

### Task 6: Attach adapter after camera compensation with hard safety flags

**Description:** Add four configuration fields, reject active flags, and call the adapter after the existing compensation result is complete.

**Acceptance criteria:**
- Shadow is enabled by default but final remains the pre-shadow formal output.
- Either active flag set true raises `NotImplementedError` during configuration validation.
- Invalid/failed shadow analysis cannot invalidate a stereo result.

**Verification:** `python -m pytest -q tests/test_global_pipeline.py tests/test_camera_compensation.py tests/test_shadow_pipeline.py`

**Dependencies:** Task 5  
**Files likely touched:** `models.py`, `pipeline.py`, shadow pipeline tests  
**Estimated scope:** M

### Checkpoint: Pipeline Integration

- Focused pipeline and backward-compatibility tests pass.
- A manual CSV schema inspection confirms shadow fields are appended only.

## Phase 4: Leakage Removal and Full Regression

### Task 7: Remove validation-time runtime leakage

**Description:** Refactor the synthetic Innovation 2 validation so configured truth phase/frequency and unavailable 2D–3D evidence never enter runtime fusion; retain GT only for post-runtime metrics and plots.

**Acceptance criteria:**
- Changing GT phase/frequency/XYZ cannot change runtime C_phy, candidate correction, or diagnosis.
- Frequency/phase/coherence are measurement-derived or unavailable.
- The validation experiment still produces its expected artifacts.

**Verification:** `python -m pytest -q tests/test_innovation2_validation.py tests/test_shadow_scientific_safety.py`

**Dependencies:** Tasks 1–2  
**Files likely touched:** `physics_validation.py`, scientific safety tests  
**Estimated scope:** M

### Task 8: Complete-sequence Shadow OFF/ON isolation

**Description:** Run an identical deterministic sequence with shadow disabled and enabled and compare all legacy fields and relevant pipeline states.

**Acceptance criteria:**
- Every original CSV key has identical values within explicit floating tolerance.
- Measured/estimated/compensated XYZ, disparity, prediction, radius, tracking/confidence/Kalman/reacquisition/compensation outputs match.
- Only appended shadow fields differ.

**Verification:** `python -m pytest -q tests/test_shadow_state_isolation.py`

**Dependencies:** Task 6  
**Files likely touched:** one sequence-isolation test file  
**Estimated scope:** M

### Task 9: Full suite and experiment regression

**Description:** Run all original and new tests plus every required experiment; compare synthetic raw metrics to the audited baseline.

**Acceptance criteria:**
- Original 205 tests and all new tests pass.
- Existing simulation and experiment commands complete.
- Synthetic raw X/Y/Z/3D RMSE has no meaningful regression from 0.612/0.202/6.435/6.467 mm.

**Verification:** Full commands from the approved specification.

**Dependencies:** Tasks 1–8  
**Files likely touched:** none, except generated ignored outputs  
**Estimated scope:** S

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Existing dirty changes overlap `models.py`, `pipeline.py`, `runner.py` | High | Apply narrow patches against current content; inspect diffs by path after every slice; never reset or commit. |
| Shadow observer accidentally mutates state | High | Copy arrays/history; frozen result replacement only; complete-sequence equivalence test. |
| Missing evidence inflates confidence | High | Explicit validity and reliability; exclude unavailable terms; invalid result with zero valid terms. |
| Candidate smoothing destroys a real transient | High | Transient gate defaults to no correction when evidence is insufficient; bounded candidate only. |
| Existing fault benchmark expects numeric defaults | Medium | Preserve complete-input behavior while making missing fields optional; update only scientifically invalid assumptions. |
| Spectral windows are too short | Medium | Mark spectral/phase/coherence terms unavailable and expose reasons. |

## Open Questions

None. The design and test seams are approved.
