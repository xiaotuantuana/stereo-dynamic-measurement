# Innovation 2/3 Shadow Mode — First-Round Design

Date: 2026-08-20

## Scope

This first modification round has two goals:

1. Remove scientific-risk paths from Innovation 2: runtime ground-truth leakage, use of configured true phase/frequency as confidence evidence, missing evidence treated as healthy, and unsafe smoothing-like correction.
2. Attach Innovation 2 and Innovation 3 to the real `stereo_research` video pipeline in shadow mode without changing stereo geometry, matching, raw XYZ, or executing recovery.

The existing path through rectification, LK/FB, disparity prediction, adaptive matching, LR/epipolar/neighborhood checks, subpixel/IC-GN, Q reconstruction, Kalman estimation, and reference-camera compensation remains unchanged.

## Safety Invariants

- Innovation 2/3 never mutates the raw visual measurement.
- `final_xyz` is unconditionally equal to the existing formal output for this first round.
- Candidate correction is diagnostic output only.
- Recovery is recommendation only; it never calls a matcher, changes calibration, updates extrinsics, changes baseline, or re-rectifies images.
- Ground truth, injected-fault labels, clean signals, configured true phase, and configured true frequency are excluded from runtime evidence APIs.
- Missing evidence is unknown, not healthy and not faulty.
- Existing CSV fields keep their current order; shadow fields are appended.

## XYZ Stage Semantics

The following names describe distinct pipeline stages. Existing fields retain their current meanings.

```text
measured_xyz
  Direct floating-point stereo/Q reconstruction before Kalman/state estimation.

estimated_xyz
  Existing TemporalStereoPipeline Kalman/state-estimation result, when that
  method enables estimation. This is not renamed to raw XYZ.

compensated_xyz
  Existing result after reference-camera common-motion compensation, when
  compensation is available. If compensation is unavailable, this stage falls
  back to the existing estimated or measured result exactly as it does today.

shadow_input_xyz
  Read-only XYZ consumed by Innovation 2. It is the existing compensated XYZ
  when compensation is valid; otherwise it is the existing estimated XYZ when
  available; otherwise it is measured XYZ. The selected source is exported as
  `shadow_input_stage`.

candidate_corrected_xyz
  Innovation 2 diagnostic candidate derived from shadow_input_xyz. It never
  overwrites measured, estimated, or compensated XYZ.

final_xyz
  Current formal external output. During this first round it is the exact
  pre-shadow formal pipeline output and cannot be replaced by candidate
  correction or a recovery result.
```

The adapter therefore observes the most downstream already-existing XYZ stage without changing the meaning of any legacy field. In particular, `shadow_input_xyz` must not be labelled `raw_xyz`, because it may contain an existing Kalman estimate or existing camera compensation.

## Integration Point

Use a thin stateful adapter in `stereo_research/shadow_analysis.py`.

`TemporalStereoPipeline.initialize()` and `step()` already collect a complete frame of point results and then apply reference-camera compensation. The adapter runs after that compensation step, so it can inspect:

- actual per-point results;
- same-frame multi-point relations;
- prior raw XYZ history;
- actual stereo, tracking, and geometry diagnostics.

It returns copies of `FramePointResult` with appended shadow diagnostics. It does not change existing raw, measured, estimated, or compensated fields.

Runner-only integration is rejected because GUI and direct pipeline consumers would bypass it. Per-point `_valid_result()` integration is rejected because it lacks the complete same-frame multi-point context.

## Runtime Evidence Model

Add an auditable evidence term:

```text
EvidenceTerm
  value: float | None
  valid: bool
  reliability: float
  reason: str
```

`compute_physics_confidence()` accepts runtime evidence terms and returns:

```text
c_phy
r_phy
valid
num_valid_evidence
available_evidence_names
missing_evidence_names
components
effective_weights
```

Only finite terms with `valid=true` and positive reliability enter the normalized weighted score. Effective weight is the configured weight multiplied by reliability, then normalized across the available set. If no term is valid, C_phy is invalid and no numerical confidence claim is made.

The current fixed weights remain centralized in `PhysicsConfidenceConfig`. This round makes them explicit and auditable but does not claim they are scientifically calibrated.

## Ground-Truth Boundary

Runtime code receives only measured data and past measured history. Evaluation ground truth is represented separately and is consumed only by metric and plotting functions.

The validation experiment is split conceptually:

```text
synthetic generator
  ├─ raw observations → runtime evidence → C_phy → candidate correction
  └─ ground truth     → post-runtime metrics and plots
```

Configured `frequency_hz` and generated phases may generate the evaluation sequence, but cannot be passed to runtime evidence construction.

Runtime frequency evidence is based on agreement among dominant frequencies estimated from measured point histories. Runtime coherence comes from measured cross-spectra. Runtime phase evidence, when available, is based on measured cross-phase stability between adjacent history windows. It does not compare against a configured or ground-truth phase.

When window length, frequency resolution, or coherence is insufficient, frequency/phase/coherence evidence is marked unavailable rather than assigned `1.0`.

The existing 2D–3D term is also unavailable unless it can be computed from independent observed 2D motion and a 3D-based prediction. It is never filled with a healthy constant.

## Transient Preservation Gate

Add a public `decide_transient()` seam returning:

```text
TransientDecision
  is_possible_real_transient
  possible_measurement_error
  allow_correction
  reason
```

Minimum behavior:

- A high temporal residual supported by normal valid 2D–3D/visual evidence and coherent same-frame multi-point motion is treated as a possible real transient. Candidate correction is blocked.
- A high temporal residual accompanied by abnormal valid 2D–3D evidence, abnormal spatial evidence, and an isolated point response is treated as a possible measurement error. Candidate correction is allowed.
- Insufficient evidence does not authorize a large correction.

The real pipeline may use valid stereo/tracking consistency as explicitly labelled supporting visual evidence when independent 2D–3D evidence is unavailable. This does not turn the unavailable 2D–3D term into a healthy value.

## Candidate Correction

`correct_trajectory_point()` remains a separate public seam and produces:

```text
raw_xyz_mm
candidate_corrected_xyz_mm
correction_applied_candidate
correction_method
correction_reason
correction_delta_mm
transient_protected
```

If the transient gate blocks correction, candidate XYZ equals raw XYZ.

A configurable `max_correction_mm` caps the magnitude of a candidate correction. It is a conservative safety guard, not a claimed physical threshold. The default and its rationale are visible in configuration and output metadata.

## Shadow Physics Flow

```text
Raw XYZ
  → measured runtime evidence
  → auditable C_phy fusion
  → transient preservation gate
  → bounded candidate correction
```

History contains current and past measurements only. No future frame is accessed.

## Shadow Fault Flow

`FaultFingerprint` fields become optional. Rule evaluation checks availability before applying a rule.

```text
actual image/stereo/tracking/geometry/physical residuals
  → FaultFingerprint
  → RuleDiagnosticEngine
  → FaultClass
  → RecoveryManager.plan
  → recommended action only
```

Available sources include actual LR error, epipolar/vertical residual, match cost or uniqueness margin, neighbor disparity MAD, FB error, temporal residual, reference-camera diagnostic, and valid `1-C_phy`.

Unavailable fields stay `None`. They are not replaced by zero or one.

No new fault class is added. No recovery plan is executed.

## Configuration

Append conservative options to `MatcherConfig`:

```text
enable_physics_shadow = true
enable_fault_shadow = true
allow_physics_correction_to_final = false
allow_fault_recovery_to_final = false
```

The two `allow_*_to_final` flags are first-round guardrails, not functional feature flags. They cannot be enabled. Configuration validation raises `NotImplementedError` if either is `true`. This prevents the API from implying that active correction or recovery is implemented.

This first round therefore enforces, without exception:

```text
final_xyz = the exact pre-shadow formal pipeline output
```

Shadow analysis cannot modify final output through configuration, direct calls, or recovery recommendations.

## Result and CSV Contract

Do not replace or substantially restructure `FramePointResult`. Add optional shadow fields and append their CSV names after the existing field list.

Required appended data:

```text
shadow_input_x_m, shadow_input_y_m, shadow_input_z_m
shadow_input_stage
candidate_corrected_x_m, candidate_corrected_y_m, candidate_corrected_z_m
c_phy, c_phy_valid, c_phy_valid_terms, c_phy_missing_terms
transient_protected
fault_class, fault_confidence, recommended_recovery
r_2d3d, r_temporal, r_spatial
r_frequency, r_phase, r_coherence
r_lr, r_epi, r_fb, r_ref, r_calib
```

Missing evidence serializes as an empty CSV value. Units are included in field names where the quantity is dimensional. The formal output continues to use the existing `final_X_m`, `final_Y_m`, and `final_Z_m` columns; appending case-only duplicates would make the CSV ambiguous in Excel and PowerShell.

## Error Handling

- Shadow analysis failure must not invalidate a valid stereo measurement.
- Per-term invalid or non-finite values become unavailable evidence with a reason.
- A whole shadow-analysis exception produces unavailable shadow diagnostics and preserves every legacy XYZ stage and the pre-shadow formal final XYZ.
- Invalid point results remain invalid; shadow mode does not fabricate XYZ.
- Runtime history stores only accepted, finite measurements and is reset with each pipeline instance.
- Shadow history owns copies of every array/value it stores. It never retains a mutable reference to `PointState`, pipeline disparity histories, Kalman state, reference images, or a mutable `FramePointResult` payload.
- Shadow analysis uses immutable/replaced result values and cannot mutate input result objects in place.

## TDD Seams

The approved public seams are:

1. `compute_physics_confidence()` for missing evidence, validity, reliability, effective weights, and GT-independent behavior.
2. `decide_transient()` and `correct_trajectory_point()` for synchronous real-transient protection, isolated-outlier correction, and correction limits.
3. `ShadowAnalyzer.process_frame()` for actual `FramePointResult` adaptation, runtime evidence, fingerprint construction, and recommendation-only recovery.
4. `TemporalStereoPipeline.initialize()/step()` plus `FramePointResult.as_csv_row()` for complete legacy-stage regression, final=the pre-shadow formal output, and CSV append behavior.
5. `RuleDiagnosticEngine.diagnose()` for optional evidence and fault-specific recommendations.

Tests use ground truth only after runtime output to assess error; runtime public seams do not accept ground truth arguments.

### Full-Sequence Shadow State Isolation

The regression seam runs the same deterministic complete sequence twice using identical inputs and configuration except for:

```text
enable_physics_shadow = false / true
enable_fault_shadow = false / true
```

The test compares every frame and point and requires all legacy outputs and state evolution to match within justified floating-point tolerance, including:

- measured XYZ;
- estimated XYZ and Kalman output;
- compensated XYZ and camera-compensation diagnostics;
- disparity and predicted disparity;
- selected search radius;
- tracking/lost/recovering state;
- confidence and confidence state;
- reacquisition/recovery stage;
- every pre-existing CSV field.

The comparison excludes only newly appended shadow fields. It snapshots legacy CSV keys before Shadow Mode is introduced so the test cannot pass by silently dropping an old field.

This proves `ShadowAnalyzer` is a read-only observer over the complete sequence, rather than only proving that one current-frame XYZ value is unchanged.

## Verification

Required verification:

- Original 205 tests remain passing.
- New tests cover GT isolation, missing evidence, real transient protection, isolated outlier correction, shadow-mode raw regression, actual fingerprint construction, and recommendation-only recovery.
- A full deterministic Shadow OFF versus Shadow ON sequence test proves identical legacy outputs and state evolution, including all original CSV columns.
- Synthetic raw X/Y/Z/3D RMSE remains approximately 0.612/0.202/6.435/6.467 mm.
- Existing simulation, Innovation 1 experiment, Innovation 2 validation, fault benchmark, and the reproducible `car.avi` M0–M3 run complete successfully.

## Deferred

- Active correction of final XYZ.
- Executing recovery against matchers.
- Runtime variable baseline.
- Extrinsic recalibration or re-rectification.
- Scientifically calibrated C_phy weights.
- A full structural dynamics spatial model.
- New fault classes or deep-learning components.
