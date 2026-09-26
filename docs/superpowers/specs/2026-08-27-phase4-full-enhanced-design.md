# Phase 4 Design: FULL_ENHANCED End-to-End Integration

## Status and scope

This document is the approved Phase 4 implementation baseline for the stereo
measurement project.  It solves only the two blocking concerns: Innovation 2
state contamination/jump multiplication and a controlled end-to-end
FULL_ENHANCED result path.  It does not modify the GUI, manual ranging,
Dataset Benchmark, Innovation 1 parameter tuning, the default configuration,
or the legacy 199-column production CSV contract.

No 2000-frame, 5000-frame, or full-dataset run is permitted in this phase.
The only scale validation after implementation is the existing fixed 500-frame
manifest, and only after the controlled gates pass.

## Goals

1. Prevent abnormal raw observations and corrected candidates from contaminating
   Innovation 2 prediction history.
2. Prevent a single observation error from becoming multiple corrected jumps.
3. Integrate Innovation 1, Innovation 2, Innovation 3, and final arbitration
   into the existing `TemporalStereoPipeline`, which remains the sole final
   result owner.
4. Permit final-result writes only for a deliberately enabled experiment.
5. Preserve default, GUI, manual-ranging, Shadow, and legacy CSV behavior.

## Non-goals

- Do not create another pipeline or make the legacy dynamic-measurement
  orchestrator a final-result owner.
- Do not expand fault classes, run parameter sweeps, or optimize isolated
  occlusion samples.
- Do not feed ground truth, injected labels, or scenario labels into runtime
  decisions.
- Do not execute I3 remeasurement in the production pipeline in this phase.

## Architecture

```text
TemporalStereoPipeline (sole final owner)
    |
    +-- Innovation 1 --> immutable I1BaselineView
    |
    +-- EnhancedProcessor
    |       +-- Innovation 2: stateful temporal/physics assessment
    |       +-- Innovation 3: structured diagnosis and recommendation
    |       `-- EnhancedOutcome (candidate plus audit evidence)
    |
    +-- FinalArbitrator (pure decision)
    |
    `-- _apply_final_decision() (the only experimental final write)
```

`I1BaselineView` is a true immutable value snapshot.  It contains copied scalar
baseline values and quality evidence only; it must not hold references to I1
`PointState`, Kalman state, tracking state, or mutable arrays.

`EnhancedProcessor` and `FinalArbitrator` never modify `final_x_m`, `final_y_m`,
or `final_z_m`.  They return immutable outcomes and decisions.  Only
`TemporalStereoPipeline._apply_final_decision()` may produce a replacement
result with changed final fields.

## Authority policy

The experimental authority is a structured, versioned contract:

```text
requested_mode
effective_mode
write_enabled
scope = experiment
policy_version = phase4_controlled_v1
```

Final writes require all of the following:

1. requested mode is `FULL_ENHANCED`;
2. `write_enabled` is true;
3. scope is exactly `experiment`;
4. the candidate passes the complete safety gate.

All other combinations are Shadow-only.  In particular,
`FULL_ENHANCED + write_enabled=False` has `effective_mode=ENHANCED_SHADOW` and
must preserve final values, status, and the 199-column CSV exactly as Shadow.
`write_enabled=True` outside `FULL_ENHANCED` has no final-write authority.
GUI and normal production callers do not receive this authority contract.

Shadow computes a `proposed_decision`, including `REJECT` or `USE_CORRECTED`,
but never submits it.  Audit output distinguishes:

```text
proposed_decision
committed_decision
write_committed
```

This makes a candidate safety recommendation observable without changing
formal output.

## Innovation 2 state isolation

Each point maintains four isolated records:

| Record | Purpose | Prediction authority |
|---|---|---|
| `raw_measurement_state` | Audit record of every baseline observation, including faults | Never |
| `corrected_output_state` | Audit record of I2 candidates | Never |
| `trusted_state` | Confirmed, timestamped real observations | Only source |
| `prediction_state` | Read-only current prediction derived before the frame decision | Read-only |

The state additionally stores `last_raw`, `last_corrected`, `last_trusted`,
timestamped `trusted_history`, recovery observations, `episode_id`, and the
current I2 state.  Every prediction uses only trusted history and timestamps.
Large frame gaps trigger a lower-order predictor or reset; they are not treated
as adjacent samples.

Recovery may append confirmed observations to trusted history, preserving each
observation's original timestamp.  Recovery never rewrites an already emitted
historical final result.

### State transitions

| State | Condition | Trusted update | Next state |
|---|---|---|---|
| `NORMAL` | Normal evidence or supported legitimate motion | Commit raw | `NORMAL` |
| `NORMAL` | Weak anomaly evidence | No commit | `SUSPECT` |
| `NORMAL` | Confirmed anomaly or hard failure | No commit | `QUARANTINED` |
| `SUSPECT` | Later clean observation agrees with prediction | Commit raw | `NORMAL` |
| `SUSPECT` | Trend supports legitimate motion | Commit raw | `NORMAL` |
| `SUSPECT` | Confirmed anomaly | No commit | `QUARANTINED` |
| `QUARANTINED` | Abnormal episode persists | No commit | `QUARANTINED` |
| `QUARANTINED` | First clean recovery observation | No commit | `RECOVERY` |
| `RECOVERY` | Second timestamped clean observation | Commit recovery observations | `NORMAL` |
| `RECOVERY` | New anomaly | No commit | `QUARANTINED` |

Hard failures—NaN/Inf, invalid geometry, or complete matching failure—may enter
`QUARANTINED` and abstain immediately.  They do not need temporal plus a second
independent evidence source to be safely rejected.

For non-hard failures, correction requires fusion of temporal, I1 quality,
physics, and trend evidence.  Missing evidence is represented as unavailable,
never as zero.  Supported slow or fast legitimate motion accepts the baseline
raw observation and updates trusted history rather than applying a low-pass
correction.

### Complete correction or abstention

The former 25 mm partial-output clipping policy is removed.  A proposed
correction is either complete and safe, or the processor abstains.  If a
25 mm quantity remains, it is only an authorization bound; it may never create
a partial candidate that remains clearly catastrophic.  A correction that
exceeds its authorized bound, lacks evidence, fails geometry, or cannot enter
the post-correction safety band becomes an abstention.  The arbitrator then
uses the baseline with warning or rejects it; it does not submit an artificial
intermediate result.

This addresses the verified Phase 3 failure chain: an outlier was written into
raw history, poisoned the next second-order prediction, and then caused a
return-to-normal frame to be corrected by the 25 mm cap.  The new design treats
both frames as one `episode_id` and keeps both raw and candidate outputs out of
the predictor.

## Candidate safety contract

`candidate_safe` is not an assignable loose boolean.  It is a derived,
auditable result created by one gate function.  The gate records the outcome
and reason for each required condition:

1. candidate coordinates are finite;
2. candidate geometry is valid;
3. correction is within authorization bounds;
4. evidence sufficiency is met, unless the path is a hard-failure abstention;
5. post-correction residual is inside the safety band;
6. current I2 state permits a candidate;
7. candidate does not violate timestamp/trusted-history invariants.

The final arbitrator consumes this structured gate result, not a string and not
an arbitrary module-provided flag.

## Innovation 3 boundary

I3 receives frozen I1/I2 evidence and returns a structured `I3Recommendation`
contract with explicit risk and action enums.  It may return normal, warning,
or blocking/reject recommendations.  The arbitrator uses only this contract;
it does not parse scattered labels or ambiguous strings.

I3 may diagnose, score, and recommend recovery.  It may not mutate a result,
write a final coordinate, update I1/I2 history, consume labels/GT, or invoke the
existing orchestrator's unsafe after-result path.  This phase's recovery is the
I2 trusted-state recovery described above.  A future active remeasurement must
be transactional: snapshot, candidate remeasurement, absolute and relative
verification, commit on success, and discard/rollback on failure.

## Final arbitration and atomic commit

`FinalArbitrator` returns one of:

| Decision | Preconditions | Committed source |
|---|---|---|
| `ACCEPT` | Valid I1 baseline; normal I2/I3 | I1 baseline |
| `ACCEPT_WITH_WARNING` | Valid I1 baseline with non-blocking warning | I1 baseline |
| `USE_CORRECTED` | Complete safe candidate, non-blocking I3 policy, full experiment authority | I2 candidate |
| `REJECT` | Hard failure, blocking I3 policy without safe candidate, or unsafe/unavailable baseline | Rejected |

`ACCEPT` and `ACCEPT_WITH_WARNING` always use the I1 baseline.  Only
`USE_CORRECTED` may use an I2 candidate.  `REJECT` clears final coordinates and
marks the experimental final result invalid.

`_apply_final_decision()` first builds a complete `FinalCommit`, validates
coordinates, distance, final validity, source, and decision consistency, then
performs one replacement operation.  It never incrementally writes final
fields.

If enhanced processing, safety-gate construction, arbitration, or commit
preparation raises an exception, the implementation discards the enhanced
attempt and restores the original I1 baseline final result and original
baseline validity.  It may use `ACCEPT_WITH_WARNING` only when the baseline was
already valid; an invalid baseline remains invalid/rejected and must never be
converted to valid by fallback.

New fields (`proposed_decision`, `committed_decision`, `write_committed`,
`result_source`, `final_decision_reason`, and experimental final validity) are
internal/experiment diagnostics.  They are intentionally excluded from the
legacy 199-column CSV.

## Test matrix

### I2 state and safety

- `test_no_correction_feedback_loop`
- `test_raw_history_is_audit_only`
- `test_corrected_output_never_feeds_prediction`
- `test_single_outlier_recovers_after_two_clean_frames`
- `test_continuous_outlier_stays_in_one_episode_until_recovery`
- `test_recovery_preserves_original_timestamps`
- `test_recovery_never_rewrites_prior_final_results`
- `test_slow_motion_is_accepted_as_legitimate`
- `test_fast_motion_is_accepted_as_legitimate`
- `test_hard_failure_quarantines_without_secondary_evidence`
- `test_large_correction_is_complete_or_abstains_never_partial`

### Arbitration and authority

- `test_accept_preserves_i1_baseline`
- `test_warning_preserves_i1_baseline`
- `test_use_corrected_commits_exact_candidate`
- `test_reject_has_no_valid_final_measurement`
- `test_final_commit_is_internally_consistent`
- `test_full_enhanced_without_write_flag_is_shadow_equivalent`
- `test_write_flag_outside_full_enhanced_has_no_final_authority`
- `test_experiment_scope_is_required_for_final_write`
- `test_enhanced_exception_restores_original_baseline_validity_and_final`

### I3 and lifecycle

- `test_i3_diagnosis_cannot_mutate_measurement_result`
- `test_structured_blocking_i3_policy_rejects_without_safe_candidate`
- `test_full_enhanced_smoke_has_i1_i2_i3_and_final_per_frame`
- `test_default_pipeline_and_gui_path_remain_unchanged`
- `test_legacy_199_column_csv_contract_remains_unchanged`

### Experimental outputs

- `innovation2_core_validation.csv` contains the six retained scenarios and
  required core fields.
- Per-frame traces include raw, prediction, corrected, trusted state, decision,
  I2 state, and `C_phy`.
- `full_enhanced_comparison.csv` contains M0/M1/M2/M3 without relabeling a
  matcher profile as a system mode.
- `final_arbitration_audit.csv` contains I1/I2/I3 status, proposed/committed
  decisions, source, write status, and final validity.
- The Phase 3 500-frame manifest is loaded and hash-checked rather than
  regenerated.

## Implementation order

1. Contracts and red tests.
2. I2 state isolation.
3. Controlled pipeline integration and atomic final commit.
4. I3 structured end-to-end integration and smoke.
5. Controlled Phase 4 validation, then the fixed 500-frame manifest only if
   all prior gates pass.

At every stage, default pipeline, GUI, manual ranging, and legacy CSV
regressions must remain green.  A decision about `ALLOW_2000` is made only after
Phase 4 controlled validation and the fixed 500-frame comparison; it does not
start a 2000-frame run.
