# Optimization Log

## Phase 1.5 Baseline Freeze

- Change: No algorithm change; copied the accepted Phase 1 smoke artifacts.
- Reason: Establish immutable before/after evidence.
- Before: valid rate 0.4000, disparity MAE 3.65625 px, RMSE 14.08762 px, bad-3 0.08333.
- After: Not applicable yet.
- Dataset manifest: Phase 1 deterministic smoke selection, 10 pairs / 90 grid points.
- Config: method M3, default `MatcherConfig`, disparity-only benchmark Q.
- Git: `872b8dba6294eade34ca4031237960441227571d` with a dirty working tree recorded separately by Git status.

## Phase 1.5 Failure Analysis and Opt-in Initial Confidence

- Change: Added runtime-only initialization diagnostics and an opt-in ambiguity gate. Default GUI/pipeline behavior remains unchanged because `enable_initial_confidence_calibration` defaults to `False`.
- Reason: The frozen smoke baseline contained three errors above 10 px while reporting confidence 1.0 for every initialized match.
- Root causes: all three catastrophic smoke points were ambiguous matches; two were near disparity discontinuities/occlusions and two had weak texture (categories overlap).
- Smoke before: coverage 40.00%, MAE 3.6563 px, RMSE 14.0876 px, CER@10 8.33%.
- Smoke after (margin 0.03, confidence 0.45): coverage 27.78%, MAE 0.2263 px, RMSE 0.3175 px, CER@10 0%; all three known catastrophes rejected.
- Cost: average initialization time increased from 77.32 ms to 85.35 ms per stereo pair (+10.39%).
- Safety: GT is loaded only after inference to compute metrics and offline labels; it is not passed to the matching or rejection decision.

## Phase 2 Preliminary Sensitivity and Ablation

- Fixed manifests: 500 train samples for development and 500 val samples for validation; seed 20260827. No independent test split or scene identity is available in the flattened subset.
- Near-permissive development evidence: coverage 43.16%, MAE 1.9202 px, RMSE 9.5837 px, CER@10 3.862%.
- Near-permissive validation evidence: coverage 43.38%, MAE 1.5820 px, RMSE 7.6704 px, CER@10 3.535%.
- Gate search result: no tested confidence/margin pair achieved development CER@10=0. The minimum was 0.832% at margin 0.10/confidence 0.60 with only 16.02% development coverage; validation CER@10 was 1.909% at 15.13% coverage.
- Decision: do not promote a tuned threshold to the default configuration. Confidence/margin gating is useful for smoke failure containment but is insufficient as a robust standalone catastrophe detector.
- M0 vs M3 static ablation: predictions and statuses are identical across all 90 points. This is expected because the runner calls only `initialize()`; Innovation 1 temporal/local components are not identifiable.
- Subpixel note: fractional SGBM output slightly improves smoke MAE (3.7188 to 3.6563 px) but does not change CER@10; this is not evidence for LocalMatcher temporal subpixel refinement.

## Phase 3 Stateful Validation

- Change: Added an experiment-only stateful sequence model/runner, CONTROLLED image generators, lifecycle provenance, sequence metrics, Innovation2 scenario matrix, severity-aware Innovation3 benchmark, dashboard, figures, and case-study exports.
- Production impact: none. `MatcherConfig()` defaults, GUI path, `TemporalStereoPipeline` final ownership, write guards, and the 199-column CSV remain unchanged.
- Innovation1 500-frame CONTROLLED result: M0 coverage/RMSE 100%/0.1724 px; M3 100%/0.0708 px. Median runtime increased from 3.77 to 10.56 ms.
- Occlusion/discontinuity result: M3 coverage 87.5%, CER@10 3.81%; repeated 15.9 px errors passed with confidence about 0.716 and LR error about 0.1 px.
- Innovation2: eight CONTROLLED scenarios reduced mean candidate RMSE from 2.7993 to 2.3692 mm and mean jitter by 0.6200 mm, with 0.2636% false correction. However jump counts worsened (single 1→2, continuous 4→5, random outliers 4→8), so final-result authority stays disabled.
- Innovation3: 480 controlled cases, seven fault classes, three severities; binary precision 1.0, recall 0.9524, F1 0.9756, FPR 0, FNR 0.0476. These are feature/orchestrator functional results, not end-to-end image-pipeline generalization evidence.
- Scale decision: completed 500-frame stateful validation; do not run 2000/5000/full until occlusion CER and Innovation2 jump-count regression are addressed.
