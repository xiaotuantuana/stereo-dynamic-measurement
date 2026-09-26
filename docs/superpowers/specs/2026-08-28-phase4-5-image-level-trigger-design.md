# Phase 4.5 Design: Image-Level Anomaly Trigger Validation

## Status, scope, and frozen baseline

This is the approved evidence-completion phase following Phase 4.  It does
not optimize Innovation 1, 2, or 3; change their parameters; alter the GUI,
manual-ranging path, default configuration, legacy 199-column CSV, or the
Phase 4 authority contract.  It also does not run a 2000-frame, 5000-frame,
or full-dataset benchmark.

The sole raw input is the existing Phase 3 frozen 500-frame manifest with its
approved SHA-256.  Each Phase 4.5 case is a frozen short window selected from
that manifest.  A paired clean control uses exactly the same source frames,
timestamps, disparity GT, calibration, points, method, and matcher config.
Only the injected arm receives a deterministic image-array copy after image
read.  The Phase 3 source files are never rewritten or regenerated.

`ALLOW_2000` remains `NO` until every Phase 4.5 gate below passes.

## Data and injector contract

The new injector is an independent experimental module.  It accepts only
image arrays, immutable case specifications, source frame identifiers and a
fixed seed.  It returns new image arrays.  Its API deliberately does not
accept xyz, distance, disparity, ground-truth values, `FramePointResult`, or
pipeline state, so numeric/result-level fault injection is impossible through
this component.

The runtime reader is copy-on-read:

```text
frozen image file -> read image -> copied image array -> deterministic injector
                                                    -> existing pipeline
```

The clean arm also receives a copied array but no transform.  Source image
hashes are verified before each experiment and recorded in the manifest.  An
invalid source manifest, missing frame, mismatched hash, invalid ROI, invalid
side, or illegal duration aborts the run explicitly; no data are regenerated
or silently substituted.

There are 32 injected cases and 32 paired clean controls:

| Type | Cases | Image-only transform |
|---|---:|---|
| `local_occlusion` | 8 | Cover one fixed point ROI with deterministic local texture/occluder data. |
| `local_blur` | 8 | Apply deterministic local Gaussian blur to one fixed point ROI. |
| `unilateral_roi_shift` | 8 | Translate only the left or right target ROI by 1, 2, or 3 pixels. |
| `continuous_anomaly` | 8 | Apply one listed primitive for 2, 3, or 4 consecutive frames. |

Each window contains two clean warm-up frames, one or more exposure frames,
two clean recovery observations, and one optional clean stability frame.  ROI
coordinates are fixed from the frozen `PointSpec`; they are never inferred
from an intermediate measurement result.

## M1/M2/M3 execution matrix

Every clean and injected window runs these existing labels only:

| Label | Authority | Runtime path | Formal final behavior |
|---|---|---|---|
| M1 baseline | no `ExperimentAuthority` | image -> stereo -> I1 | original I1 baseline final |
| M2-equivalent | `ENHANCED_SHADOW` | image -> stereo -> I1 -> I2 -> I3 -> arbitrator | proposed only; no final write |
| M3 | `FULL_ENHANCED`, `write_enabled=True`, `scope=experiment` | same as M2 | final write only through pipeline atomic commit |

M2 and M3 must share exactly the same pre-commit enhanced-evaluation path:
same image, initial state, I1 snapshot, I2 observation/outcome, I3
recommendation, candidate safety object, and `FinalArbitrator` proposed
decision.  The only permitted difference is final-write authorization in the
arbitrator/atomic-commit boundary.  M1 and M2 must never commit a final write.

`committed_decision` and `write_committed` have different meanings:

- `committed_decision` is the semantic decision represented in the returned
  result/audit.  In Shadow, it can be `ACCEPT_WITH_WARNING` or `REJECT` even
  though no new formal final value was written.
- `write_committed` is a factual flag that a final result was atomically
  written under valid M3 experimental authorization.

Consequently, neither `REJECT` nor `KEEP_BASELINE` counts as a false final
correction.  `clean_false_final_correction` counts only an actual clean-arm
M3 write of `USE_CORRECTED`; clean rejections are reported separately as
`clean_false_reject_count` and remain a safety failure even though they are
not corrections.

## Output and audit contract

All Phase 4.5 outputs are isolated under `results/phase4_5/...`:

1. `case_manifest.csv`: pair/case IDs, arm, frozen source frame IDs and
   timestamps, anomaly specification, ROI/side, fixed seed, source hashes,
   injector version, and Phase 3 manifest hash.
2. `frame_level_results.csv`: case/arm/mode/frame/point identity; injection
   activity; I1 baseline; I2 raw/prediction/candidate/state/episode/trusted
   update/safety reasons; I3 fault/risk/confidence/recommendation; decision,
   final status/source/owner/write/fallback; jump, GT error, and recovery.
3. `arbitration_audit.csv`: immutable I1 baseline snapshot, each complete
   candidate-safety predicate, structured I3 policy, proposal, semantic
   committed decision, authorization, write fact, final owner, fallback, and
   exception evidence.
4. `scenario_summary.csv`: case/pair/type/mode aggregation of coverage,
   actions, candidate count, errors, jumps, recovery, clean/injected deltas,
   harmful commits, false corrections, and false rejects.
5. `gate_summary.csv`: `gate_id`, required condition, observed value, pass,
   precise failure reason, and evidence path.

Error is the Euclidean final-XYZ error in millimetres against the unmodified
source frame GT, derived only after processing.  `error_improvement_mm` is
the injected-frame M2 I1-baseline error minus the matching M3 final error.
The injected label is audit/evaluation metadata and never enters I1/I2/I3 or
arbitration.

## Gates

| Gate | Required condition |
|---|---|
| Tests | Existing plus Phase 4.5 tests all pass. |
| Frozen source | Manifest and every participating source image hash match. |
| Determinism | Repeated same manifest/seed gives identical transformed-image hashes and result keys. |
| Clean equivalence | Clean M1/M2 final/status/199-column rows match; clean M3 has zero corrected writes. |
| Shadow | M2 has zero writes and preserves I1 final/status/199-column output. |
| Authority | M1/M2 zero writes; every M3 write has full experiment authorization and pipeline atomic final owner. |
| Trigger evidence | `enhanced_final_commit_count >= 5`; actual corrected commits span at least two image-anomaly types. |
| Harm | `harmful_commit_count = 0`, where committed M3 correction has final error greater than matched M2 baseline error beyond 1e-6 mm. |
| Clean safety | `clean_false_final_correction = 0`; `clean_false_reject_count = 0`. |
| Benefit | Eligible corrected commits have median `error_improvement_mm > 0`. |
| Recovery | Every observed I2 episode after exposure ends returns to `NORMAL` within two clean frames; no raw/candidate history contamination. |
| Fallback | Enhanced exceptions restore the original I1 final and validity. |

All gates must pass for `ALLOW_2000=YES`; otherwise it stays `NO` with
case-level failure reasons.  A lack of qualifying commits is a gate failure,
not a reason to tune algorithms or thresholds in Phase 4.5.

## Test and implementation plan

1. Add manifest/injector contracts and red tests: deterministic copy-only
   transformations, validation failure paths, frozen paired provenance.
2. Add a controlled window runner that executes M1/M2/M3 through the existing
   temporal pipeline and records pre-commit plus post-commit evidence.
3. Add output, audit, gate, clean-equivalence, authority, fallback, trusted
   history, and recovery tests.
4. Run focused tests after each slice; then run the full suite.  Only then run
   the 32-window experiment and write the report
   `PHASE4_5_IMAGE_LEVEL_TRIGGER_REPORT.md`.

No code in the GUI, manual measurement, normal production call path, I1 core,
I2 core algorithm, I3 core algorithm, or their thresholds is in scope.

## Design self-review

- The source rule is strict: the Phase 3 fixed 500-frame manifest is the only
  raw source; all data changes occur in copied image arrays.
- M2/M3 parity is explicit and testable before the authorization boundary.
- `committed_decision` is not conflated with `write_committed`.
- False correction is limited to an actual `USE_CORRECTED` final write;
  REJECT and keeping the baseline are audited separately.
- The authority, fallback, legacy CSV, and no-2000 constraints remain those of
  Phase 4, not a new production mode.
