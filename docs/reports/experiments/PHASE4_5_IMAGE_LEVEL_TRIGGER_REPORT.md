# Phase 4.5 — Image-Level Anomaly Trigger Validation Report

## Verdict

`ALLOW_2000 = NO`.

Phase 4.5 completed its bounded image-level evidence run without changing the
Innovation 1/2/3 algorithms or thresholds, GUI, default configuration, manual
measurement path, authority contract, or legacy 199-column CSV.  The frozen
source, deterministic injection, clean equivalence, Shadow non-interference,
M2/M3 pre-commit parity, and final-owner authority gates passed.  The required
end-to-end corrected-final-action gate did not pass, so a 2000/5000/full run
remains prohibited.

## Frozen protocol

- Raw source: the existing Phase 3 fixed 500-frame manifest only.
- Manifest SHA-256:
  `CA3329880484CA046D54DFB44FF7EC5873BA9CB3F6CD1EA42F582D0A7A6F978C`.
- Aggregate left/right source-image SHA-256:
  `17CF7321816A87BE45D3A4A61F04F419B53D7CEC13F9DF861EFEC93865C1C816`.
- Design: 32 frozen short windows, each with a same-source clean control.
- Injection: deterministic copy-on-read image arrays only; no source image,
  GT, timestamp, calibration, xyz, distance, disparity, or result object was
  changed by the injector.
- Treatments: eight each of local occlusion, local blur, unilateral 1–3 px
  ROI shift, and continuous 2–4 frame anomaly.
- Execution labels: M1 baseline, M2 `ENHANCED_SHADOW`, and M3
  `FULL_ENHANCED + write_enabled=True + scope=experiment`.
- Scope: 207 selected source-frame observations per arm/mode, 3,726
  frame-point audit rows in total.  This is not a 500-frame, 2000-frame,
  5000-frame, or full-dataset run.

## Regression and contract evidence

The complete regression suite passed before the bounded run:

```text
350 passed in 32.47s
```

New tests cover image-copy-only deterministic transforms, all four declared
anomaly forms, invalid injector specifications, fixed 32-case construction,
source-manifest rejection, M1/M2/M3 audit generation, M2/M3 pre-commit parity,
and the distinction between semantic `committed_decision` and factual
`write_committed`.

An explicit Phase 4 bug was fixed under this contract: an authorized M3
`REJECT` atomically cleared the experimental final but reported
`write_committed=False`.  It now reports `True`; this is an audit-fact repair,
not an algorithm or threshold change.  `REJECT` and keeping the baseline are
not counted as final corrections.  Only an actual M3 `USE_CORRECTED` write is
counted by `clean_false_final_correction`.

## Results

The run produced the required artifacts under
`results/phase4_5/image_level_trigger_validation/`:

- `case_manifest.csv`
- `frame_level_results.csv`
- `arbitration_audit.csv`
- `scenario_summary.csv`
- `gate_summary.csv`

The image-level anomalies exercised the enhanced path.  On injected M3 rows,
I2 states were `NORMAL=479`, `QUARANTINED=138`, and `RECOVERY=4`; I3 produced
`BLOCKING/BLOCK_FINAL=45`, `WARNING/WARN=39`, and
`NORMAL/ALLOW_CORRECTION=537`.  The 45 blocking decisions resulted in
authorized atomic M3 `REJECT` writes.  There were no enhanced exceptions or
fallbacks.

However, there were no `USE_CORRECTED` proposals or final writes.  I2 produced
30 safe candidates, but all 30 had typed I3 `WARNING/WARN`, whose frozen Phase
4 policy preserves the I1 baseline rather than granting correction permission.
This shows the I3 boundary and authority contract operated as designed; it
does not supply the required corrected-final-action evidence.

## Gate result

| Gate | Result | Evidence |
|---|---|---|
| Existing + new tests | Pass | 350 passed |
| Frozen manifest/source images | Pass | Both fixed hashes matched |
| Deterministic injector | Pass | Repeated transforms had identical hashes |
| Clean equivalence | Pass | Clean M1/M2 stable legacy rows equal |
| M2 Shadow non-interference | Pass | Zero M2 writes |
| M2/M3 pre-commit parity | Pass | All compared enhanced evidence equal |
| Final authority owner | Pass | All writes owned by `TemporalStereoPipeline` |
| Corrected final commits >= 5 | **Fail** | 0 |
| Corrected commits cover >= 2 anomaly types | **Fail** | 0 types |
| Harmful corrected commits = 0 | Pass | 0 |
| Clean false final corrections = 0 | Pass | 0 |
| Clean false rejects = 0 | Pass | 0 |
| Median committed improvement > 0 | **Fail** | No eligible corrected commits |
| I2 recovery <= 2 clean frames | **Fail** | 27 observed episodes did not return to `NORMAL` in the available two-frame recovery window |

## Interpretation and next action

This phase was evidence completion, not algorithm development.  It therefore
does not modify I3 policy, relax candidate safety, change thresholds, add a
special correction mode, bypass dual authorization, or reinterpret a REJECT
as a correction simply to pass the Gate.

The exact blockers are: zero authorized corrected writes, zero anomaly-class
coverage by such writes, no resulting positive median improvement, and 27
recovery failures under the strict two-clean-frame criterion.  Until a new
separately approved design addresses those blockers, the official decision is
unchanged: `ALLOW_2000 = NO`.
