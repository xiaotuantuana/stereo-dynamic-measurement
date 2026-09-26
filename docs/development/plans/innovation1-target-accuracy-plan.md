# Implementation Plan: Innovation 1 Target-Accuracy Policy

## Overview

Build an opt-in Z-standard-uncertainty accuracy loop over the existing precision formula, runtime match uncertainty, vision-state search controller, and existing refinement/recovery seams. Preserve the first-round pipeline exactly when disabled.

## Dependency Graph

```text
sigma contract + accuracy budget
  → pure measurement policy
    → auditable result/CSV fields
      → causal pre/post-match pipeline integration
        → bounded precision retry
          → Innovation 1 ablation experiment
            → full regression and report
```

## Task 1: Clarify sigma planner and add PrecisionPlan

**Description:** Extend the existing planner without replacing its formula. Add explicit sigma target semantics, runtime estimated sigma input, Z uncertainty, ratio, and feasibility.

**Acceptance criteria:**
- Only metric `sigma` is accepted.
- Worked example and target/distance/focal-baseline monotonicity pass.
- Planner API contains no GT/fault-label input; X/Y are explicit unavailable/PARTIAL.

**Verification:** `python -m pytest -q tests/test_target_accuracy_planner.py tests/test_innovation1_foundation.py tests/test_stereo_precision_sanity.py`

**Dependencies:** None  
**Files:** `precision_planner.py`, new planner test  
**Scope:** S

## Task 2: Add pure joint MeasurementPolicy

**Description:** Convert plan plus vision/motion state into refinement and bounded retry decisions while preserving motion authority over search radius.

**Acceptance criteria:**
- Strict targets never request less refinement/retry effort than loose targets.
- Poor vision independently changes search/refinement policy.
- Infeasible target stops retry and reports honestly.

**Verification:** `python -m pytest -q tests/test_target_accuracy_policy.py`

**Dependencies:** Task 1  
**Files:** new `accuracy_policy.py`, new policy test  
**Scope:** S

## Checkpoint: Mathematical and Policy Safety

- Focused tests pass.
- No runtime API contains GT fields.
- No matcher or geometry code has changed.

## Task 3: Append result/config/CSV contract

**Description:** Add minimal opt-in config and optional result fields; append new CSV names after the current 169 fields.

**Acceptance criteria:**
- Default feature is False.
- Existing 169-field prefix is identical.
- Unavailable values serialize empty; Shadow field order/meaning is unchanged.

**Verification:** `python -m pytest -q tests/test_target_accuracy_csv.py tests/test_shadow_pipeline.py`

**Dependencies:** Tasks 1–2  
**Files:** `models.py`, `runner.py`, new CSV test  
**Scope:** M

## Task 4: Integrate causal pre/post-match policy

**Description:** Instantiate a controller only when enabled, use previous valid depth for pre-match policy, and attach post-match precision status without changing stereo validity/state updates.

**Acceptance criteria:**
- Initialization is warmup and subsequent valid frames use causal state.
- Different targets produce different real refinement/acceptance policy.
- Valid stereo remains valid when precision is unmet.

**Verification:** `python -m pytest -q tests/test_target_accuracy_pipeline.py tests/test_global_pipeline.py`

**Dependencies:** Task 3  
**Files:** `pipeline.py`, policy module, new pipeline test  
**Scope:** M

## Task 5: Add one bounded precision retry

**Description:** If a valid match is precision-unmet, feasible, and visually usable, invoke the existing stronger matcher/refinement path up to `max_precision_retry`; keep the best valid measurement.

**Acceptance criteria:**
- Unmet first match triggers at most the configured retry count.
- Retry stops at the bound and never loops indefinitely.
- Infeasible target does not waste retries; best valid stereo output remains available.

**Verification:** `python -m pytest -q tests/test_target_accuracy_pipeline.py`

**Dependencies:** Task 4  
**Files:** `pipeline.py`, `local_matching.py` only if a per-call existing refinement input is required, pipeline test  
**Scope:** M

## Checkpoint: End-to-End Safety

- Feature-off complete sequence equals first-round baseline across legacy, Shadow, and pipeline state.
- Feature-on tests demonstrate actual policy control and bounded retry.
- Innovation 2/3 safety suite passes.

## Task 6: Extend Innovation 1 experiment

**Description:** Add A0–A3 and Loose/Medium/Strict/Infeasible policy rows to the existing experiment output, with runtime and GT evaluation clearly separated.

**Acceptance criteria:**
- Output reports plan, effort, retries, status, runtime, valid rate, and target attainment.
- GT error fields appear only in evaluation columns.
- Baseline remains an offline recommendation only.

**Verification:** Run focused experiment test and `python run_innovation1_experiment.py --config configs/innovation1_experiment.yaml`.

**Dependencies:** Tasks 1–5  
**Files:** existing Innovation 1 experiment module/entry point, config if required, experiment test  
**Scope:** M

## Task 7: Full regression and report

**Description:** Run old/new tests, simulation, Shadow safety, experiments, and car smoke; document actual controlled parameters and limitations.

**Acceptance criteria:**
- All original 219 and all new tests pass.
- Feature-off full-sequence equality is proven.
- Synthetic baseline metrics have no meaningful unintended regression when feature is off.

**Verification:** Full commands from the approved specification.

**Dependencies:** Tasks 1–6  
**Files:** final report only  
**Scope:** S

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Existing dirty first-round work overlaps core files | High | Narrow additive patches; never reset, commit, or rewrite unrelated content. |
| Uncalibrated variance is mistaken for guaranteed accuracy | High | Call it estimated sigma; expose feasibility/status; document calibration as PARTIAL. |
| Target precision overrides motion search needs | High | Preserve existing base radius; precision affects refinement/retry, not search-center physics. |
| Retry changes a valid measurement for the worse | High | Bound retry and retain the best valid uncertainty result. |
| Full XYZ propagation becomes speculative | High | Z controls; X/Y stay explicit unavailable/PARTIAL this round. |
| Shadow behavior changes | High | Feature-off and Shadow safety regression tests after every integration slice. |

## Open Questions

None. The supplied specification authorizes direct implementation if no major conflict is found.

