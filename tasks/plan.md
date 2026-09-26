# Implementation Plan: Phase 3 Stateful Innovation Validation

## Overview

Phase 3 reuses `TemporalStereoPipeline` as the sole measurement owner. Work is delivered in gated vertical slices: first prove sequence lifecycle, then validate Innovation1 on stateful image sequences, then reuse and extend the existing controlled Innovation2 and Innovation3 harnesses, and finally aggregate only scientifically comparable results. Every experimental feature remains outside the GUI path or behind existing non-production modes.

## Architecture Decisions

- Reuse `stereo_research.runner` and `SequenceManifest`; add experiment-layer adapters/metrics instead of another pipeline.
- Create a fresh `TemporalStereoPipeline` per sequence. Object disposal is the current reset-equivalent because the pipeline has no public `reset()` method.
- Mark generated sequences and injected faults as `CONTROLLED`; never merge them with FlyingThings3D metrics.
- Keep FlyingThings3D static: its flattened files contain no reliable scene/sequence identity.
- Preserve `MatcherConfig()` defaults, GUI behavior, and the 199-column legacy CSV.
- Keep Innovation2/3 writes to the final result disabled until controlled acceptance gates pass.

## Dependency Graph

```text
Lifecycle contract + sequence model
                ↓
Controlled stateful image smoke
                ↓
Innovation1 stateful ablation
        ┌───────┴────────┐
        ↓                ↓
Innovation2 scenarios  Innovation3 perturbations
        └───────┬────────┘
                ↓
Unified dashboard + scale decision + audit report
```

## Task List

### Phase A: Stateful foundation

- [ ] Task 1: Add sequence grouping/model contract without changing `DatasetSample`.
  - Acceptance: deterministic frame order; rejects duplicate/non-monotonic frames; controlled provenance recorded.
  - Verification: focused sequence-model tests.
- [ ] Task 2: Add experiment sequence runner around the existing pipeline.
  - Acceptance: one initialize, N-1 steps per sequence; new pipeline per sequence; lifecycle counters and frame rows saved.
  - Verification: failing-then-passing lifecycle/reset/state-leak tests.
- [ ] Task 3: Add sequence metrics and result bundle.
  - Acceptance: coverage, accepted errors, CER, jitter, tracking loss, runtime; N/A represented as null/empty, never zero.
  - Verification: pure metric tests plus controlled smoke.

### Checkpoint A

- [ ] Stateful smoke executes `initialize → step → step`.
- [ ] Two sequences do not share state.
- [ ] Existing full test suite passes.

### Phase B: Innovation1 stateful validation

- [ ] Task 4: Build a deterministic controlled stereo sequence using existing simulation/image primitives.
- [ ] Task 5: Run only real method profiles M0/M1/M2/M3 and export a stateful ablation table.
- [ ] Task 6: Add occlusion/discontinuity diagnostic evidence as experiment-only outputs before considering any gate.

### Checkpoint B

- [ ] M0/M3 execute different code paths and produce explainable differences.
- [ ] Coverage and CER are reported together.
- [ ] No GT reaches runtime decisions.

### Phase C: Innovation2 controlled validation

- [ ] Task 7: Extend the existing causal physics harness to stable, noise, single jump, continuous outlier, drift, fast motion, and frame loss.
- [ ] Task 8: Compute raw/corrected MAE/RMSE, jitter, jump suppression, false correction, recovery frames, lag, and runtime.
- [ ] Task 9: Export evidence participation for C_phy, PSD, phase, and coherence; unavailable evidence remains N/A.

### Checkpoint C

- [ ] Corrected differs from raw in intended fault cases.
- [ ] Slow/fast legitimate motion false correction is explicitly measured.
- [ ] Production final result remains unchanged.

### Phase D: Innovation3 controlled perturbation validation

- [ ] Task 10: Extend existing deterministic perturbations with severity and at least five defensible classes.
- [ ] Task 11: Run clean/injected samples through existing diagnosis/orchestration and compute confusion/safety metrics.
- [ ] Task 12: Export false positives, false negatives, catastrophic interception, retention, and case studies.

### Checkpoint D

- [ ] Labels are isolated from runtime diagnosis.
- [ ] Precision/recall/F1 and false-positive rate are reported.
- [ ] Diagnosis safety is not achieved by rejecting everything.

### Phase E: Unified evidence and scale gate

- [ ] Task 13: Aggregate BASELINE, Innovation1, I1+I2, and FULL candidate rows into `experiment_summary.csv` with N/A fields.
- [ ] Task 14: Run protocol smoke, then 500-level validation only if checkpoints pass.
- [ ] Task 15: Produce the Phase 3 report, maturity ratings, file manifest, provenance, and 2000/5000/full recommendation.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| FlyingThings3D frame numbers are mistaken for time | Invalid temporal claims | Keep it static; use CONTROLLED sequences or verified videos |
| Existing shadow fields are mistaken for final correction | False Innovation2/3 claim | Report candidate vs final separately; retain write guards |
| Synthetic feature tables bypass the real pipeline | Weak functional evidence | Distinguish mechanism tests from image-pipeline perturbation tests |
| Coverage is reduced to improve error | Misleading result | Always pair accuracy/CER with coverage and retention |
| Dirty worktree obscures provenance | Poor reproducibility | Save config, manifest, hashes, Git commit, and dirty status per run |

## Scale Gates

- 500 only after lifecycle and smoke checkpoints pass.
- 2000 only if M0/M3 differ, Innovation2 corrected output is measurable, Innovation3 has non-trivial F1, and tests pass.
- 5000/full remain blocked if coverage collapses, GT leaks, outputs are shadow-only, or results are not reproducible.
