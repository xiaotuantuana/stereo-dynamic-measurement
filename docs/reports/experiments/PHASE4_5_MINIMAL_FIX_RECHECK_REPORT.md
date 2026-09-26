# Phase 4.5 Minimal-Fix Recheck

## Scope

This recheck used exactly the same frozen Phase 3 source, 32 injected cases,
32 clean controls, image copy-on-read injector, timestamps, GT, points,
calibration, and configuration as Phase 4.5.  It did not run 2000/5000/full
or add data/cases.  The only temporary code change was a typed I3 policy trial:
map `OCCLUSION` from `WARNING/WARN` to `WARNING/ALLOW_CORRECTION` while leaving
CandidateSafety, FinalArbitrator, dual authorization, and atomic commit
unchanged.

## Root-cause evidence

1. All 30 Phase 4.5 I2 safe candidates had I3
   `OCCLUSION + WARNING/WARN`.  Because FinalArbitrator accepts a candidate
   only with explicit `ALLOW_CORRECTION`, this exactly explains
   `USE_CORRECTED=0` in the baseline run.
2. I2 trusted history is not contaminated.  Its isolation tests remain green;
   raw/corrected candidates do not enter the predictor.
3. Of the 27 recovery failures, some clean post-exposure frames remained I1
   `ambiguous/lost` (hard failures), which I2 cannot safely turn valid.  The
   valid-baseline unresolved frames had raw-to-trusted-prediction residuals of
   10.13–94.30 mm, above the frozen 10 mm recovery gate.  Current runtime
   evidence cannot safely distinguish a continuing anomaly from a clean frame
   with such residual, so widening this gate would risk committing abnormal raw
   observations to trusted history.

## Temporary policy result

The policy trial removed the authorization bottleneck but failed the safety
evidence gate:

| Measurement | Result |
|---|---:|
| Corrected M3 commits | 45 |
| Anomaly types covered | 4 |
| Harmful corrected commits | 39 |
| Clean false corrections | 15 |
| Median committed improvement | -18.806 mm |
| Recovery failures | 27 |
| Regression | 351 passed |

This shows that the existing `candidate_safe` evidence is sufficient to allow
an auditable candidate but not sufficient to make a broad OCCLUSION correction
policy scientifically safe.  The trial was therefore reverted.  The final
source code retains the approved conservative `OCCLUSION -> WARNING/WARN`
mapping, and no recovery threshold or trusted-history rule was changed.

## Decision

`ALLOW_2000 = NO` remains mandatory.  The requested minimal correction was
tested on the frozen protocol and was rejected because it worsened final
results and created clean false corrections.  Making it pass would require a
new, separately approved safety-evidence design; that is outside the current
minimal-correction scope.

Evidence for the rejected trial remains preserved in
`results/phase4_5/image_level_trigger_validation_minfix/`.
