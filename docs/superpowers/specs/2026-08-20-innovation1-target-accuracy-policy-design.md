# Innovation 1 Target-Accuracy Policy — Second-Round Design

## Scope and Safety Boundary

This round connects the existing Innovation 1 precision mathematics and the existing `stereo_research` visual-state controller into an opt-in causal measurement policy. `enable_target_accuracy_policy` defaults to `False`. When disabled, the complete first-round behavior, all 169 CSV columns, Shadow Mode, pipeline state, and raw/final XYZ remain unchanged.

The work does not change stereo geometry, disparity sign, Q units, LK/FB, SGBM, IC-GN mathematics, Kalman mathematics, camera compensation, Innovation 2/3 evidence, active baseline, rectification, or extrinsics.

## Existing Precision Planner Audit

`stereo_dynamic_measurement/innovation1/precision_planner.py` currently implements the first-order depth relation

```text
Z = f B / d
sigma_Z ≈ Z² / (f B) · sigma_d
sigma_d_required = sigma_Z_target · f B / Z²
```

Although its public arguments say `target_depth_error_mm`, the formula is a standard-deviation/standard-uncertainty formula, not RMSE, MAE, maximum error, or a confidence interval. It currently:

- controls only Z uncertainty;
- computes required disparity precision;
- scans candidate baselines and reports an offline recommendation;
- reports baseline feasibility under an assumed sigma disparity;
- does not consume the runtime disparity uncertainty already exported by `stereo_research`;
- does not produce X/Y/Z estimated uncertainty, a runtime precision status, or a bounded retry decision;
- is used by the isolated `AdaptiveStereoMeasurementPipeline`, but is not called by `TemporalStereoPipeline`.

This round preserves the formula and clarifies its sigma semantics. Baseline recommendation remains offline only.

## Considered Approaches

### A. Z-sigma closed loop with a thin policy adapter — selected

Reuse the planner formula and the existing runtime `disparity_variance_px2`. Make Z the controlling axis; export X/Y targets but mark their runtime propagation `PARTIAL_NOT_CONTROLLING`. This is the smallest scientifically defensible complete loop.

### B. Full Q-Jacobian XYZ covariance in this round

Propagate left-image and disparity covariance through the full reprojection Jacobian. This could control all axes, but requires careful covariance assumptions and calibration validation that exceed this round. It is deferred rather than presented as calibrated science.

### C. Extend only the isolated Innovation 1 prototype

This minimizes main-pipeline risk but fails the requirement that target accuracy actually control `TemporalStereoPipeline`. It is rejected.

## Public Contracts

### TargetAccuracySpec

```text
metric = "sigma"                 # the only supported metric
target_sigma_x_mm = optional     # reported, not controlling this round
target_sigma_y_mm = optional     # reported, not controlling this round
target_sigma_z_mm = optional     # controlling target
```

At least one target must be supplied when the feature is enabled. Any metric other than `sigma` is rejected instead of silently reinterpreted.

### PrecisionPlan

```text
target_spec
current_depth_m
required_sigma_disparity_px
estimated_sigma_disparity_px
estimated_sigma_x_mm             # unavailable/PARTIAL this round
estimated_sigma_y_mm             # unavailable/PARTIAL this round
estimated_sigma_z_mm
precision_ratio                  # estimated sigma_d / required sigma_d
feasible
feasibility_reason
limiting_axis                    # "z" or explicit partial/unavailable reason
```

`precision_ratio <= 1` means the estimated standard uncertainty satisfies the target. A larger ratio means the stereo measurement can remain geometrically valid but the requested precision is unmet.

### MeasurementPolicyDecision

The policy is a pure, GT-free decision over the precision plan and pre-match vision/motion state. It exports enabled/warmup, required and estimated uncertainties, status, base/final radius, refinement level, retry budget/count, acceptance reason, and policy reason.

## Causal Two-Stage Flow

```text
TargetAccuracySpec + previous valid depth/disparity + Vision State
                              ↓
                    PRE-MATCH POLICY
       motion-driven base radius + precision refinement effort
                              ↓
             existing Stereo / Subpixel / IC-GN
                              ↓
 current match variance + LR/epipolar/confidence/quality evidence
                              ↓
                 POST-MATCH PRECISION PLAN
                              ↓
             ACCEPT / RETRY_STRONGER / PRECISION_UNMET
```

The first valid initialization result is warmup: existing initialization is preserved, then its depth/variance establishes the causal state for subsequent frames. No future frame or GT is accepted by planner or policy APIs.

## Search and Refinement Separation

The existing motion/vision controller remains authoritative for the search center and base radius. Target accuracy does not impose a smaller radius. The policy may retain or enlarge the base radius when poor vision requires more coverage, but never overrides recovery radius with a precision-only rule.

Target strictness controls refinement effort and bounded retry:

- level 0: existing matching/refinement effort;
- level 1: use the existing quality/recovery matcher path once if post-match sigma is unmet and visual evidence remains usable;
- level 2: same bounded stronger path with the configured maximum IC-GN iteration/tolerance controls already exposed by the matcher configuration.

No IC-GN equation is changed. Per-call effort controls are passed to existing refinement inputs. `max_precision_retry` bounds retries; the implementation uses no unbounded loop.

## Feasibility and Status

Pre-match feasibility is estimated using the previous causal depth and the best physically available disparity sigma floor from configuration. Post-match feasibility uses the actual estimated `sqrt(disparity_variance_px2)`.

Statuses are separate from stereo validity:

- `WARMUP`: no causal previous depth exists;
- `MET`: valid stereo and precision ratio at most one;
- `VALID_BUT_PRECISION_UNMET`: valid stereo, actual estimated precision exceeds the target, and retry is exhausted or not justified;
- `INFEASIBLE`: even the configured uncertainty floor cannot meet the target at the current geometry;
- `UNAVAILABLE`: no valid depth/uncertainty exists.

An unmet or infeasible target does not discard a valid stereo measurement or prevent the legacy state update. It only adds an auditable precision status.

## Configuration

Minimal new `MatcherConfig` fields:

```text
enable_target_accuracy_policy = False
target_metric = "sigma"
target_sigma_x_mm = None
target_sigma_y_mm = None
target_sigma_z_mm = None
max_precision_retry = 1
max_precision_refinement_level = 2
```

M0/M1/M2/M3 remain opt-out by default. Research experiments explicitly enable the policy.

## Pipeline Integration

A small `TargetAccuracyController` is owned by `TemporalStereoPipeline` only when enabled. It reads copied scalar state and returns policy decisions. The integration points are:

1. before local matching, after the existing base motion/confidence radius is chosen;
2. after a valid match has exposed its actual uncertainty;
3. before `FramePointResult` is finalized, to attach status fields without changing measured XYZ semantics;
4. before Shadow analysis, so Shadow sees the same legacy stages plus additional read-only policy metadata.

Feature-off branches return before any policy computation or retry and do not instantiate controller history.

## CSV Contract

All existing 169 fields stay byte-for-byte ordered at the front. These fields are appended only:

```text
accuracy_policy_enabled
precision_policy_warmup
target_metric
target_x_mm, target_y_mm, target_z_mm
required_sigma_d_px, estimated_sigma_d_px
estimated_sigma_x_mm, estimated_sigma_y_mm, estimated_sigma_z_mm
precision_ratio, precision_feasible, precision_status, limiting_axis
policy_base_search_radius_px, policy_final_search_radius_px
policy_refinement_level, policy_retry_budget, policy_precision_retry_count
policy_acceptance_reason, policy_reason
```

Unavailable X/Y estimates serialize as empty cells, never zero or one. Existing Shadow fields are not renamed or reordered.

## Experiment Design

Extend the existing Innovation 1 experiment entry point rather than creating a parallel framework. The second-round output adds a policy experiment table over:

- A0 Fixed legacy;
- A1 Vision-state adaptive only;
- A2 Target-accuracy only;
- A3 Target accuracy plus vision state;

and Loose, Medium, Strict, plus deliberately Infeasible Z-sigma targets. Runtime policy never reads experiment GT. Where synthetic GT exists, disparity and XYZ errors are joined only in evaluation output.

## Test Seams

The approved public seams are:

1. `precision_planner`: metric validation, worked numeric example, target/distance/focal-baseline monotonicity, feasibility, and GT-free signature;
2. pure measurement-policy decision: target and vision-state joint monotonicity and bounded retry;
3. `TemporalStereoPipeline` results: feature-off full-sequence equivalence, warmup, real policy fields, valid-but-unmet behavior, best measurement retention;
4. CSV: existing 169-column prefix unchanged and new fields appended;
5. existing Shadow public tests: GT isolation, missing evidence, hard active flags, and recommendation-only safety;
6. Innovation 1 experiment entry point: A0–A3 × target levels and evaluation-only GT metrics.

## Error Handling

- Unsupported metric, nonpositive target, and invalid retry/refinement bounds raise `ValueError` at configuration/spec construction.
- Missing causal depth produces `WARMUP`/`UNAVAILABLE`, not fabricated precision.
- Missing post-match variance leaves estimated fields empty and never claims MET.
- An infeasible target stops precision retry immediately.
- Policy exceptions do not fall back silently; invalid configuration fails before processing.

## Deferred

- calibrated full XYZ covariance/Jacobian;
- uncertainty calibration against real GT;
- runtime variable baseline, Q/P/T updates, re-rectification;
- extrinsic recalibration;
- active Innovation 2 correction and active Innovation 3 recovery;
- scientific C_phy calibration and full structural dynamics.

