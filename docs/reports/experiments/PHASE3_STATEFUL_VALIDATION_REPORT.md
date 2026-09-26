# Phase 3 — Stateful Innovation Validation Report

## Executive verdict

- **Innovation1: Preliminary Validated** on a 500-frame CONTROLLED stateful image sequence. It is genuinely different from M0 and materially improves disparity accuracy, but occlusion/discontinuity robustness remains unresolved.
- **Innovation2: Functional.** Causal candidate correction works and improves error magnitude in several outlier scenarios, but it creates secondary threshold crossings, so it does not yet satisfy jump-suppression acceptance.
- **Innovation3: Functional.** Seven controlled feature-level fault classes are diagnosable with strong aggregate metrics, but the benchmark is not yet an end-to-end image perturbation through final `TemporalStereoPipeline` authority.
- **FULL_ENHANCED: BLOCKED.** Existing production write guards remain enabled. No 2000/5000/full run is recommended.

All generated/synthetic evidence is labeled `CONTROLLED`. FlyingThings3D remains static because its flattened subset has no reliable scene or temporal identity.

## A. Stateful Benchmark

### Construction and lifecycle

`DatasetSequence` groups ordered `DatasetSample` frames without changing the existing dataset model. REAL_DATASET sequences require adapter-verified `is_sequence=True`; generated sequences are explicitly `CONTROLLED`.

The experiment runner creates one fresh `TemporalStereoPipeline` per sequence:

```text
frame 0   → initialize(points)
frame 1+  → step()
end       → discard pipeline; next sequence receives a fresh instance
```

For each method in the 500-frame run:

- initialize calls: 1
- step calls: 499
- reset-equivalent calls: 1 (`new_pipeline_per_sequence`)

A two-sequence test proves separate pipeline instances and no cross-sequence state reuse. Each run stores lifecycle, frame results, sequence manifest, config, environment, Git commit, timestamp, and dirty-worktree status.

### Sequence source decision

FlyingThings3D_subset cannot reliably reconstruct scene/sequence order: its adapter identifies only flattened train/val frame identifiers and explicitly marks `is_sequence=False`. It was not relabeled as temporal data. Stateful functional tests use generated stereo images with offline disparity GT.

## B. Innovation1

### Components actually activated

| Method | Active path |
|---|---|
| M0 | dense SGBM each frame |
| M1 | LK flow |
| M2 | LK flow + temporal prediction |
| M3 | flow + prediction + epipolar + neighborhood + subpixel + LR + adaptive search |

### 500-frame CONTROLLED ablation

| Method | Coverage | Valid MAE px | Valid RMSE px | CER@10 | Tracking loss | Median runtime ms |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 100.00% | 0.1399 | 0.1724 | 0% | 0% | 3.77 |
| M1 | 39.07% | 0.2066 | 0.2596 | 0% | 59.80% | 1.98 |
| M2 | 33.87% | 0.2341 | 0.3202 | 0% | 65.53% | 1.96 |
| M3 | 100.00% | 0.0448 | 0.0708 | 0% | 0% | 10.56 |

M0 and M3 are now observably different because `step()` executes. M3 reduces RMSE by 58.96% relative to M0 without coverage loss on the smooth controlled sequence, at approximately 2.8× median runtime.

### Occlusion/discontinuity evidence

The 30-frame discontinuity sequence produces:

- coverage: 87.5%
- CER@3: 3.81%
- CER@10: 3.81%
- repeated worst error: 15.9 px
- wrong-match confidence: approximately 0.716
- LR error: approximately 0.1 px

Thus confidence and LR consistency still cannot identify all foreground/background swaps. This is the current dominant Innovation1 failure mode and the reason not to scale further.

## C. Innovation2

### Controlled scenarios

Stable, Gaussian noise, single jump, continuous outlier, slow drift, fast legitimate motion, frame loss, and random measurement outlier were evaluated. Runtime analysis receives only measured coordinates and an independent causal 3D motion observation; GT and scenario labels are joined only afterward.

Across eight scenarios:

- mean raw RMSE: 2.7993 mm
- mean corrected-candidate RMSE: 2.3692 mm
- mean jitter improvement: 0.6200 mm
- aggregate false correction rate: 0.2636%
- lag on slow/fast motion: 0 frames
- false correction on slow drift: 0%
- false correction on fast legitimate motion: 0%

Key fault cases:

| Scenario | Raw RMSE | Corrected RMSE | Magnitude suppression | Raw→corrected jump count | Recovery |
|---|---:|---:|---:|---:|---:|
| Single jump | 4.7159 | 3.5677 | 31.30% | 1→2 | 2 frames |
| Continuous outlier | 7.0528 | 6.6155 | 10.45% | 4→5 | 2 frames |
| Random outlier | 6.5065 | 4.6515 | 45.33% | 4→8 | 2 frames |

Although error magnitude and jitter improve, bounded correction creates additional frames above the 10 mm jump threshold. Therefore Innovation2 does **not** yet pass the requested jump-suppression criterion.

### Evidence participation

Of 2303 runtime rows:

- C_phy valid: 2279
- flow-3D evidence valid: 2279
- spectral/PSD evidence valid: 1943 after history warm-up
- phase evidence valid: 0
- coherence evidence valid: 0

C_phy, temporal, spatial, flow-3D, and spectral evidence participate in the candidate decision. Phase and coherence are correctly marked unavailable rather than zero. Candidate corrected output differs from raw, but final Pipeline write authority remains disabled.

## D. Innovation3

### Faults and severity

Runtime injection covers stereo mismatch, occlusion, blur, flow drift, camera motion, extrinsic drift, and tracking loss at mild/medium/severe levels. Twenty seeds yield 480 cases including clean controls. Injected labels are never passed to diagnosis or recovery.

### Results

- multiclass accuracy: 91.67%
- multiclass macro F1: 91.43%
- binary abnormal precision: 100%
- binary abnormal recall: 95.24%
- binary abnormal F1: 97.56%
- false-positive rate: 0%
- false-negative rate: 4.76%
- mild false-negative rate: 14.29%
- medium/severe false-negative rate: 0%
- clean measurement retention: 100%
- controlled residual CER before/after recovery: 25% → 0%
- catastrophic residual interception: 100%

The CER here is a controlled diagnostic-residual threshold, not disparity CER and not real-world depth accuracy. The current benchmark validates rule/orchestrator functionality; it does not authorize FULL_ENHANCED final output.

## E. Unified comparison and maturity

The unified dashboard intentionally leaves scientifically incomparable cells blank:

| Mode | Status | Evidence |
|---|---|---|
| BASELINE | Functional | M0 stateful controlled image sequence |
| INNOVATION1 | Preliminary Validated | M3 stateful controlled image sequence |
| INNOVATION1 + INNOVATION2 | Functional Component Only | controlled temporal candidate correction; not end-to-end |
| FULL_ENHANCED | Blocked | diagnosis component evidence only; no final write |

The existing `FULL_ENHANCED` authorization guard is preserved. Synthetic temporal, image, and perturbation results are not mixed with FlyingThings3D dataset metrics.

## Scale recommendation

- 500: completed.
- 2000: **not recommended yet** because occlusion CER@10 remains 3.81% and Innovation2 increases jump counts.
- 5000: blocked.
- Full 26,066: blocked.

Next work should target an explicit occlusion/disparity-edge evidence model and a correction recovery policy that avoids one-error-to-two-error propagation. Only after both pass controlled regression should 2000 be run.

## Production regression and scientific safety

- Default M3 FlyingThings3D smoke remains row-for-row identical to the frozen baseline: coverage 40%, MAE 3.65625 px, RMSE 14.08762 px.
- GUI/default `MatcherConfig`, 199-column CSV, final-result ownership, and correction/recovery write guards are unchanged.
- GT is evaluator-only.
- Git commit recorded: `872b8dba6294eade34ca4031237960441227571d`; worktree is dirty and artifact hashes are provided.

## Code modification list

### Added

- `experiment/sequence/{models,runner,metrics,ablation,occlusion_diagnostics}.py`
- `experiment/simulation/{controlled_stereo,temporal_scenarios}.py`
- `experiment/validation/{innovation2,innovation3}.py`
- `experiment/reporting/{phase3_dashboard,phase3_figures}.py`
- `run_stateful_ablation.py`, `run_innovation2_controlled.py`, `run_innovation3_controlled.py`
- `run_phase3_dashboard.py`, `run_phase3_figures.py`
- nine Phase 3 test files/test additions for lifecycle, metrics, controlled sequences, ablation, temporal validation, diagnosis, dashboard, figures, and occlusion evidence
- `tasks/plan.md`, `tasks/todo.md`

### Modified

- `stereo_dynamic_measurement/innovation2/physics_validation.py`: optional causal flow-3D evidence; no GT input and no default production change.
- `stereo_dynamic_measurement/simulation/fault_injection.py`: backward-compatible severity parameter.
- `stereo_dynamic_measurement/benchmark_faults.py`: severity, safety metrics, confusion/case-study outputs.
- `experiment/sequence/runner.py`: experimental state fields and reproducibility metadata.
- `docs/development/optimization_log.md`: before/after audit trail.

### Deleted

- None.

## Result locations

- Stateful 500 ablation: `results/phase3/stateful_innovation1_500/`
- Occlusion diagnostics: `results/phase3/occlusion_diagnostics/`
- Innovation2: `results/phase3/innovation2_controlled/`
- Innovation3: `results/phase3/innovation3_controlled/`
- Unified table: `results/phase3/experiment_summary.csv`
- Figures: `results/phase3/figures/`
- Hashes: `results/phase3/provenance_hashes.sha256`
