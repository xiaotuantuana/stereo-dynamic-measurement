# Phase 1.5 + Phase 2 Preliminary Report

## 1. Catastrophic mismatches

| Sample / point | Error (px) | Runtime evidence | Root cause |
|---|---:|---|---|
| train/0011584 grid_0_1 | 75.656 | competing minima, predicted/local disagreement, disparity edge | occlusion/discontinuity + repetitive ambiguity |
| train/0017376 grid_1_2 | 35.125 | weak gradient, close candidate costs, local disagreement | occlusion/discontinuity + weak texture + ambiguity |
| train/0017376 grid_2_1 | 13.563 | weak texture, low uniqueness, LR error, 16.31 px disagreement | weak/repetitive texture ambiguity |

All three were accepted because initialization trusted the SGBM valid status and assigned confidence 1.0 without using the available ambiguity signals.

## 2. Confidence audit

The initialization confidence problem is confirmed: valid initial points received a hard-coded `1.0`. Phase 1.5 adds an opt-in runtime-only confidence based on the conservative minimum of texture, photometric cost, uniqueness, LR consistency, and global/local disparity agreement. GT is not passed into this computation. `valid`, `confidence`, and `failure_reason` remain separate concepts.

## 3–4. Code changes and rationale

- Added fixed manifest creation/loading and benchmark manifest CLI support for reproducible splits.
- Added cost-curve diagnostics, failure classification, rejection analysis, confidence threshold curves, and visualizations.
- Added CER@3/5/10, coverage, Valid-MAE, and Valid-RMSE.
- Added opt-in initialization confidence diagnostics and `initial_ambiguous` rejection. The flag defaults to off, preserving formal GUI/pipeline behavior and the legacy 199-column CSV contract.
- Added offline parameter sensitivity, disparity/texture stratification, and preliminary static ablation scripts.
- Added integrity-audit reports, hashes, tests, and an optimization log.

## 5–7. Baseline vs improved smoke

| Metric | Frozen baseline | Opt-in improved | Change |
|---|---:|---:|---:|
| Valid / total | 36 / 90 | 25 / 90 | -11 accepted |
| Coverage | 40.00% | 27.78% | -12.22 pp |
| MAE (accepted) | 3.6563 px | 0.2263 px | -93.81% |
| RMSE (accepted) | 14.0876 px | 0.3175 px | -97.75% |
| bad-1 | 11.11% | 4.00% | -7.11 pp |
| bad-2 | 8.33% | 0% | -8.33 pp |
| bad-3 | 8.33% | 0% | -8.33 pp |
| CER@10 | 8.33% | 0% | -8.33 pp |
| Average runtime | 77.32 ms | 85.35 ms | +10.39% |

The three known catastrophic points are all reasonably rejected. Coverage did not improve or remain constant, but it stayed well above the explicit 10% failure boundary. This is a useful containment result, not yet a production operating point.

## 8–9. Fixed manifests

- Development: 500 samples from the official `train` split, seed 20260827.
- Validation: 500 samples from the official `val` split, seed 20260827 + 1.
- Each sample contributes nine grid points (4500 rows per split).
- No independent test exists. The flattened subset has no scene identity, so scene-disjoint sampling cannot be proven. Validation is not called a final test.

## 10–11. Sensitivity and recommended parameters

The near-permissive evidence runs produced:

| Split | Coverage | MAE | RMSE | CER@10 |
|---|---:|---:|---:|---:|
| Development | 43.16% | 1.9202 px | 9.5837 px | 3.862% |
| Validation | 43.38% | 1.5820 px | 7.6704 px | 3.535% |

No tested confidence/uniqueness pair achieved development CER@10=0. The minimum development CER@10 was 0.832% at uniqueness margin 0.10 and confidence 0.60, but coverage fell to 16.02%; validation gave 1.909% CER@10 at 15.13% coverage. Therefore there is **no recommended production threshold** from this sweep. Keep the new feature disabled by default. The smoke setting (margin 0.03, confidence 0.45) is diagnostic only.

## 12. Preliminary Innovation 1 ablation

M0 and M3 produced identical statuses and disparities for all 90 static smoke points. Both have 40% coverage, 3.6563 px MAE, 14.0876 px RMSE, and 8.33% CER@10. This is expected: the runner recreates the pipeline and calls only `initialize()`, so temporal/local M3 components never execute. Innovation 1 is not identifiable under this protocol.

Rounded-integer versus SGBM fractional output changed MAE from 3.7188 to 3.6563 px but left CER@10 at 8.33%. This measures OpenCV SGBM fractional output, not LocalMatcher temporal subpixel refinement.

## 13. Remaining failure modes

- High-confidence false correspondences at occlusions/disparity discontinuities can have low photometric cost, strong uniqueness, and pass LR checks.
- Confidence/margin gating alone cannot reliably detect these cases.
- The baseline rejects many potentially measurable points: 30 potential over-rejects, 7 good rejects, and 17 unassessable among 54 rejected smoke points.
- The current static data protocol cannot test LK-FB temporal tracking, adaptive search over time, recovery, or Innovation 1 subpixel refinement.

## 14. Scale-up decision

Do not run all 26,066 pairs or claim a final configuration yet. The next worthwhile experiment is a persistent sequential benchmark that calls `step()` on genuinely continuous stereo sequences, plus an occlusion/discontinuity-aware runtime cue. After that protocol is verified, run coarse-to-fine sensitivity on the fixed manifests or a scene-aware dataset and validate only the top configurations.

## Verification

- Default M3 smoke rerun is row-for-row identical to the frozen baseline for status and disparity.
- Full test suite: 299 passed.
- Integrity audit verdict: internal artifacts pass consistency checks; Innovation 1 claim-level audit fails because the current benchmark is initialization-only.
