# Phase 4 — FULL_ENHANCED Integration Report

## Executive verdict

- **Innovation1: Preliminary Validated / Stable.** The fixed 500-frame run
  preserves the Phase 3 M0 and M1 results exactly to reported precision.
- **Innovation2: Preliminary Validated on the approved six-scenario CONTROLLED
  temporal benchmark.** The new trusted-state processor removes the verified
  feedback loop: corrected jump counts never exceed raw counts, and the
  single-jump case improves from 1 to 0.
- **Innovation3: Pipeline Integrated.** It now supplies a typed risk/action
  recommendation to final arbitration.  It has no direct final-write path.
- **FULL_ENHANCED: Experimental Integrated, not yet Preliminary Validated as a
  performance improvement.** The end-to-end authority and arbitration path is
  tested and smoke-validated.  The fixed 500-frame sequence contains no final
  correction/rejection event, so it cannot establish an M3 advantage over M1.

`ALLOW_2000 = NO`.

The negative scale decision is deliberate: the mandatory safety, regression,
and no-propagation gates pass, but the fixed 500-frame image sequence does not
exercise an approved end-to-end anomalous final-action case.  Running 2000
frames now would enlarge evidence without resolving that missing comparison.

## Core implementation result

The Phase 3 propagation chain was audited and replaced:

```text
old: raw outlier -> raw history -> poisoned prediction -> capped aftershock
new: raw audit only -> trusted prediction -> quarantine episode -> two-frame recovery
```

Each point now has separate raw, corrected, trusted, and prediction records.
Candidates never feed prediction.  A return to the trusted prediction enters
recovery even if legacy raw-history evidence is still anomalous.  A continuing
outlier remains quarantined if that legacy evidence temporarily weakens.

The former 25 mm partial-output cap is no longer an output strategy.  A
candidate is complete and passes the derived safety gate, or the processor
abstains.  The safety audit records finiteness, geometry, authorization,
evidence sufficiency, post-correction safety, and I2-state eligibility.

## Authority and arbitration

`TemporalStereoPipeline` remains the sole final owner.  `EnhancedProcessor`
and `FinalArbitrator` are non-mutating.  A final write requires all of:

1. `FULL_ENHANCED` requested mode;
2. `write_enabled=True`;
3. `scope="experiment"`;
4. a complete safe candidate; and
5. a typed I3 action that explicitly allows correction.

`FULL_ENHANCED + write_enabled=False` is Shadow-effective.  It computes a
proposed decision but leaves final values, status, and the 199-column CSV
unchanged.  I3 `WARNING/WARN` retains the I1 baseline; only explicit
`ALLOW_CORRECTION` may authorize a safe I2 candidate.  This prevented an
observed clean-smoke false correction from an `OCCLUSION/WARN` diagnosis.

Enhancement exceptions restore the original I1 final value and original I1
validity.  An invalid I1 baseline remains invalid; fallback never upgrades it
to `ACCEPT_WITH_WARNING`.

## Innovation2 core validation

Output: `results/phase4/innovation2_controlled/innovation2_core_validation.csv`

| Scenario | Raw RMSE mm | Corrected RMSE mm | Raw → corrected jumps | Jump suppression | False correction | Recovery frames |
|---|---:|---:|---:|---:|---:|---:|
| Stable | 0.1850 | 0.1850 | 0 → 0 | N/A | 0.0% | N/A |
| Single jump | 4.7159 | 0.3065 | 1 → 0 | 100% | 0.0% | 2 |
| Continuous outlier | 7.0528 | 6.1238 | 4 → 3 | 25% | 0.0% | 2 |
| Slow drift | 0.2915 | 0.2915 | 0 → 0 | N/A | 0.0% | N/A |
| Fast legitimate motion | 0.2963 | 0.2963 | 0 → 0 | N/A | 0.0% | N/A |
| Noise | 3.0258 | 3.0258 | 1 → 1 | 0% | 0.0% | N/A |

Mean raw RMSE is 2.5945 mm and mean corrected RMSE is 1.7048 mm.  The trace
table records ground truth (evaluation-only), raw, prediction, corrected,
trusted state, decision, `C_phy`, trigger, state, episode, and safety reasons.

## End-to-end smoke and fixed 500-frame comparison

The four-mode smoke produced the required comparison and arbitration tables:

- `results/phase4/full_smoke/full_enhanced_comparison.csv`
- `results/phase4/full_smoke/final_arbitration_audit.csv`

The fixed 500-frame input was loaded directly from
`results/phase3/stateful_innovation1_500/M3/sequence_manifest.json`, whose
SHA-256 is `CA3329880484CA046D54DFB44FF7EC5873BA9CB3F6CD1EA42F582D0A7A6F978C`.
No frames were regenerated.

| Mode | Coverage | MAE px | RMSE px | CER@10 | Jitter px | False correction |
|---|---:|---:|---:|---:|---:|---:|
| M0 baseline | 100% | 0.1399 | 0.1724 | 0% | 0.1355 | 0% |
| M1 Innovation1 | 100% | 0.0448 | 0.0708 | 0% | 0.1384 | 0% |
| M2 I1 + I2 Shadow | 100% | 0.0448 | 0.0708 | 0% | 0.1384 | 0% |
| M3 FULL_ENHANCED | 100% | 0.0448 | 0.0708 | 0% | 0.1384 | 0% |

The M3 audit contains 1,500 frame-point rows, with 0 candidate writes, 0
rejects, and 0 invalid final outputs.  This is expected for this smooth
sequence and confirms that FULL does not force corrections.  It is not
evidence that M3 improves over M1 under real image-level anomalies.

## Regression and compatibility

- Full regression: **339 passed**.
- Default pipeline and GUI path receive no experiment authority.
- Existing Shadow tests continue to preserve formal final fields and legacy
  state.
- New experimental diagnostics are omitted from `FramePointResult.as_csv_row`;
  the legacy 199-column CSV remains unchanged.
- No 2000-frame, 5000-frame, or full-dataset run was started.

## Required next scientific gate

Before allowing 2000 frames, add one deterministic image-level controlled
fault sequence that actually triggers an I2 candidate and a typed I3 policy
through `TemporalStereoPipeline`, then compare M2 and M3 on CER, coverage,
false reject, and final-action audit.  This is a targeted end-to-end gate, not
a Dataset Benchmark redesign or parameter sweep.
