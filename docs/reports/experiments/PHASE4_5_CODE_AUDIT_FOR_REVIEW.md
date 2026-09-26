# Phase 4.5 代码审计（只读，供外部评审）

## 审计范围与结论

本报告只读检查当前工作树中的实现、测试和既有 Phase 4/4.5 输出；未修改生产代码、测试、参数或 Gate，未运行 2000/5000/full。代码行号均对应本次审计时的工作树。

**Current verdict: `ALLOW_2000 = NO`**。

| 项目 | 当前实现状态 | 主要证据 | 当前阻塞点 | 建议是否修改 |
|---|---|---|---|---|
| I1 | Mostly complete | `pipeline.py:_valid_result` 形成已滤波/补偿的 baseline；351 tests | 图像异常可令 I1 `ambiguous/lost`，I2 不能把 invalid baseline 变 valid | 是，后续仅针对恢复接口证据 |
| I2 | Mostly complete | 独立 raw/corrected/trusted history 与四状态机 | `candidate_safe` 是可提交安全门，不是 beneficial 证明；recovery 的 10 mm 门严格 | 是，先补质量证据与恢复语义测试 |
| I3 | Partial | Shadow fault classifier + typed risk/action mapper | 实际更像 fault classifier/risk policy，不评估 I2 candidate quality | 是，先补 candidate-quality discrimination 设计 |
| FinalArbitrator | Mostly complete | 纯决策、实验双重授权、原子 commit | 正确执行现有契约，但输入没有 beneficial evidence | 否，先不要扩展仲裁器 |
| FULL_ENHANCED | Partial | M2/M3 pre-commit parity、权限/CSV Gate 均通过 | 无安全的 corrected-final evidence；recovery Gate 未过 | 是，但仅在新证据契约获批后 |

### 实验事实（非设计推断）

- 原 Phase 4.5：3,726 审计行；M3 atomic commits 45，均为 REJECT；`USE_CORRECTED=0`。
- 失败的最小 I3 policy trial：M3 atomic commits 90，其中 REJECT 45、`USE_CORRECTED` 45；但 harmful corrections 39、clean false corrections 15、中位 improvement -18.806 mm，故已回退。
- 现有完整回归：**351 passed in 31.60s**（本审计不再改代码后复跑）。

## 1. 当前正式调用链（真实位置）

```text
image arrays
  -> TemporalStereoPipeline.initialize()/step()
  -> _valid_result() + _apply_camera_compensation()          [I1 baseline]
  -> ShadowAnalyzer.process_frame()                          [I3 feature/fault inputs; legacy shadow candidate]
  -> _process_enhanced_results()
       -> I1BaselineView.from_result()
       -> EnhancedPointProcessor.process()                   [I2]
       -> _structured_i3_recommendation()
       -> FinalArbitrator.decide()
       -> _apply_final_decision()                            [only enhanced final replacement]
  -> FramePointResult
```

| 步骤 | 实际文件 / 类 / 函数 | 输入 / 输出 | 状态 | final write authority |
|---|---|---|---|---|
| 正式入口 | `stereo_research/pipeline.py`, `TemporalStereoPipeline.initialize` (107), `step` (254) | 左右图、点、frame → `list[FramePointResult]` | `PointState`、flow、filter、matcher | 有 I1 baseline 写入，但非 enhanced 覆盖 |
| I1 测量 | `TemporalStereoPipeline._valid_result` (1513) | match/disparity/state → valid/invalid result | 修改 `PointState` 与其两帧 disparity history | 形成 initial baseline final |
| I1 补偿 | `_apply_camera_compensation`（结果写入见 1497–1499） | valid results → compensated baseline | camera compensation state | 形成 I1 `final_x/y/z_m` |
| Shadow 诊断输入 | `stereo_research/shadow_analysis.py:ShadowAnalyzer.process_frame` (69) | completed I1 results → enriched copied results | `_history_mm`，每点最长 256 | 无 |
| I2 | `enhanced_processor.py:EnhancedPointProcessor.process` (89) | `TimedObservation(xyz_mm,evidence)` → immutable `EnhancedStateOutcome` | raw/corrected/trusted/recovery histories | 无 |
| I3 policy | `pipeline.py:_structured_i3_recommendation` (916) | I1/Shadow fields + hard failure → `I3Recommendation` | 无 | 无 |
| 仲裁 | `final_arbitration.py:FinalArbitrator.decide` (183) | I1 view + candidate safety + I3 + authority → `ArbitrationDecision` | 无 | 无 |
| 最终提交 | `pipeline.py:_apply_final_decision` (957) | result + authorized decision → replacement result | 无 | **唯一 enhanced final replacement** |

在 `stereo_research`、`stereo_dynamic_measurement` 和 `experiment` 中搜索 `final_x_m/final_y_m/final_z_m` 写入，发现 I1 baseline 写入位于 `pipeline.py:1497–1499,1889–1891`，增强 final replacement 仅在 `pipeline.py:984–987,1001–1004`。未发现 I2、I3 或 `FinalArbitrator` 直接写 final 的旁路。

### 关键摘录 1：增强入口与 I2/I3/仲裁衔接

`stereo_research/pipeline.py:_process_enhanced_results`，843–883：

```python
for result in results:
    baseline = I1BaselineView.from_result(result)
    xyz_mm = tuple(float(value) * 1000.0 for value in baseline.xyz_m) \
        if baseline.xyz_m is not None else (float("nan"),) * 3
    hard_failure = not baseline.valid
    fault_class = result.fault_class.strip().upper()
    confirmed_anomaly = (not hard_failure and fault_class not in {"", "NORMAL"}
        and result.c_phy_valid is True and result.c_phy is not None and result.c_phy < 0.55)
    evidence = I2Evidence(
        confirmed_anomaly=confirmed_anomaly,
        legitimate_motion=bool(result.transient_protected),
        hard_failure=hard_failure, geometry_valid=baseline.valid,
        evidence_sufficient=(result.c_phy_valid is True) if confirmed_anomaly else True,
        post_correction_safe=True, suspicious=(fault_class not in {"", "NORMAL"}),
    )
    processor = self.enhanced_processors.setdefault(result.point_id, EnhancedPointProcessor())
    outcome = processor.process(TimedObservation(frame=result.frame,
        timestamp_s=float(result.frame), xyz_mm=xyz_mm, evidence=evidence))
    diagnosis = self._structured_i3_recommendation(result, hard_failure=hard_failure)
    decision = self.final_arbitrator.decide(baseline=baseline,
        candidate_safety=outcome.candidate_safety, diagnosis=diagnosis,
        authority=self.experiment_authority)
```

注意：这里传给 I2 的 timestamp 是 `float(result.frame)`，不是数据集的 `DatasetSample.timestamp`；Phase 4.5 runner 仅将后者记录到 CSV。

## 2. Innovation 1 审计

### 2.1 向后续层提供的字段

I1 最终直接被 `I1BaselineView.from_result` 读取的是 `FramePointResult.status`、`final_xyz_m`、`distance_m`、`confidence`（`final_arbitration.py:65–74`）。完整 result 同时向 Shadow/I2/I3 提供：

- 测量和估计：`raw_disparity`、`measured_disparity`、`estimated_disparity`、measured/estimated xyz、`pipeline_baseline_xyz_m`；
- matching/quality：`match_cost`、`texture_std`、`uniqueness_margin_value`、`lr_error_px`、`flow_fb_error_px`、cycle/ICGN/curvature、uncertainty、Kalman NIS；
- geometry/temporal：left/right xy、neighbor MAD、prediction residual、compensation residual；
- status：`valid`、`ambiguous`、`lost`、`depth_out_of_range`、`motion_rejected`、`confidence_rejected` 等。

`pipeline_baseline_xyz_m` 的优先级是 compensated xyz，其次 estimated xyz（`models.py:1025–1041`）。I1 result 自身是 frozen dataclass；但其数值形成前，`PointState`、Kalman/filter、matcher 和 camera compensation state 是可变的。`I1BaselineView` 则复制 tuple/scalar，不持有 `PointState` 引用，因而是不可变快照。

### 2.2 valid/lost/ambiguous 与历史

`_valid_result` 重投影失败会 `record_failure` 并输出 `ambiguous` 或在失败计数达到 `MatcherConfig.max_failures` 后 `lost`；深度或 motion 检查同样如此。`PointState.record_failure`（`models.py:786–788`）只累积失败次数并切换 `recovering/lost`。有效测量才会调用 `record_stereo_measurement`，它更新 I1 two-frame disparity/estimated-disparity history（`models.py:706–714`）。I1 也有局部/全局重初始化、Kalman predict-only 和 camera compensation fallback；这些是 baseline tracking/recovery，不是 I2 final correction。

### 关键摘录 2：I1 有效结果形成及 state mutation

`stereo_research/pipeline.py:_valid_result`，1760–1829：

```python
measured_xyz = tuple(float(value) for value in measured_xyz_array)
estimated_xyz = tuple(float(value) for value in estimated_xyz_array)
if measurement_accepted_for_state:
    state.record_stereo_measurement(right_xy, measured, confidence,
        estimated_disparity=estimated, xyz=measured_xyz, estimated_xyz=estimated_xyz)
else:
    state.record_estimated_only(... estimated_xyz_m=estimated_xyz, frame=frame, ...)
state.confirm_valid(score)
...
return FramePointResult(
    method=self.method, frame=frame, point_id=state.point_id, status="valid",
    disparity=estimated, x_m=estimated_xyz[0], y_m=estimated_xyz[1], z_m=estimated_xyz[2],
    distance_m=float(np.linalg.norm(estimated_xyz_array)),
    raw_disparity=measured, measured_disparity=measured,
    estimated_disparity=estimated, measured_x_m=measured_xyz[0],
    estimated_x_m=estimated_xyz[0], confidence=confidence, ...)
```

## 3. Innovation 2 审计

### 3.1 实际输入

I2 **不直接读取图像、disparity 或 I1 mutable state**。`_process_enhanced_results` 将 I1 final xyz 转成 mm 作为 `TimedObservation.xyz_mm`，传入 frame-index timestamp 和 `I2Evidence`。Evidence 的来源是 I1 baseline validity、`fault_class`、`c_phy_valid`、`c_phy < 0.55` 与 `transient_protected`。因此 I2 读取的是 I1 final/baseline xyz、I1 status、Shadow fault/physics 信号和自己的 trusted history；不读取 GT，也不读取 Phase 4.5 injected label。

### 3.2 trusted history：数据结构与写入规则

`EnhancedPointProcessor` 持有四个独立容器（`enhanced_processor.py:81–85`）：

| 容器 | 内容 | 预测权 |
|---|---|---|
| `raw_history` | 每个 observation，包括异常 | 无 |
| `corrected_history` | candidate 或 `None` | 无 |
| `trusted_history` | 已确认正常/恢复 observation | **唯一预测源** |
| `_recovery_observations` | 两帧待确认恢复 observation | 暂无 |

正常/合法运动走 `self._commit_trusted(observation)`；恢复确认后才批量写两个 `_recovery_observations`。hard failure、confirmed anomaly、SUSPECT、QUARANTINED、无 prediction、unsafe candidate 都不会写 trusted。候选本身从未传入 `_commit_trusted`；final committed result 也不反馈给 processor。REJECT/fallback 都发生在 I2 之后，故不会进入 I2 trusted history。Fallback 仅替换 result audit/final fields，processor 已有的 history 不回滚。

### 关键摘录 3：trusted history 的唯一更新点

`stereo_research/enhanced_processor.py:_commit_trusted/_predict`，304–323：

```python
def _commit_trusted(self, observation):
    if self.trusted_history and observation.timestamp_s <= self.trusted_history[-1].timestamp_s:
        raise ValueError("trusted observations require increasing timestamps")
    self.trusted_history.append(observation)

def _predict(self, timestamp_s):
    if not self.trusted_history:
        return None
    latest = self.trusted_history[-1]
    dt = float(timestamp_s) - latest.timestamp_s
    if len(self.trusted_history) < 2 or dt > self.max_prediction_gap_s:
        return latest.xyz_mm
    previous = self.trusted_history[-2]
    velocity = tuple((latest.xyz_mm[i] - previous.xyz_mm[i]) / previous_dt for i in range(3))
    return tuple(latest.xyz_mm[i] + velocity[i] * dt for i in range(3))
```

### 3.3 状态机（按代码）

| 当前状态 | 条件（按 `process` 顺序） | 下一状态 | candidate | trusted update |
|---|---|---|---|---|
| 任意 | hard failure 或 raw 非有限 | QUARANTINED | 否 | 否 |
| QUARANTINED/RECOVERY | prediction 存在且 Euclidean `||raw-prediction|| <= 10 mm` | RECOVERY / NORMAL（第二帧） | 否 | 第二确认帧才写两项 |
| 任意 | `confirmed_anomaly` | QUARANTINED | prediction candidate 可被构造 | 否 |
| QUARANTINED | 未满足前两项 | QUARANTINED | 否 | 否 |
| RECOVERY | 未满足前两项 | QUARANTINED | 否 | 否 |
| NORMAL/SUSPECT | suspicious 且非 legitimate motion | SUSPECT | 否 | 否 |
| 正常路径 | 其余 | NORMAL | 否 | 是 |

实际不存在单独的“QUARANTINE”枚举，代码名称是 `QUARANTINED`；另有 `SUSPECT`。

### 为什么 27 个 episode 没在 exposure 后两 clean frame 回到 NORMAL

代码条件是 `enhanced_processor.py:97–132`。复核 CSV 显示：

1. 一部分 clean 后 I1 已是 `ambiguous/lost`，从而 `hard_failure=True`，先于 recovery 判断直接走 97–104 的 `_quarantine(..."hard_failure")`；I2 正确地不能把无效 I1 变为 valid。
2. 有效 I1 的 `quarantine_unresolved` 行中，raw-to-I2-trusted-prediction Euclidean residual 范围为 **10.13–94.30 mm**，因此严格不满足 106–111 的 10 mm gate，随后仍走 113 的 confirmed anomaly 或 116–123 的 unresolved quarantine。
3. 即使第一恢复帧满足 10 mm，代码还要求第二个同样满足的 observation；任一不满足会在 125–132 从 RECOVERY 回 QUARANTINED。

这里的 10 mm 是当前 raw I1 XYZ 与 **I2 trusted-history prediction** 的三维 Euclidean 距离，不是 disparity px、GT error、candidate residual 或 I3 physics residual。超过它不能 trusted commit，是为了避免将仍可能异常的 raw 写进 predictor；这与 I2 无污染设计一致。

### 关键摘录 4：recovery gate 与 candidate 生成

`stereo_research/enhanced_processor.py:97–114,232–260`：

```python
if evidence.hard_failure or not self._finite(raw):
    return self._quarantine(... CandidateSafety.unavailable("hard_failure"), "hard_failure")
if (self.state in {I2State.QUARANTINED, I2State.RECOVERY}
        and prediction is not None
        and self._distance(raw, prediction) <= self.recovery_consistency_mm):
    return self._recover_clean_observation(observation, raw, prediction)
if evidence.confirmed_anomaly:
    return self._confirmed_anomaly(observation, raw, prediction)
...
correction_magnitude = self._distance(raw, prediction)
safety = CandidateSafety.evaluate(
    candidate_xyz_m=tuple(value / 1000.0 for value in prediction),
    geometry_valid=observation.evidence.geometry_valid,
    correction_authorized=correction_magnitude <= self.max_authorized_correction_mm,
    evidence_sufficient=observation.evidence.evidence_sufficient,
    post_correction_safe=observation.evidence.post_correction_safe,
    i2_state=self.state.value)
...
return self._record(observation, raw, prediction, prediction, True, False,
                    safety, "complete_safe_correction")
```

### 3.4 candidate 的真实含义与 CandidateSafety

FULL_ENHANCED 最终使用的 I2 candidate 是 **I2 trusted predictor 的 `prediction` 本身**，不是 `raw + partial correction`，也不是 ShadowAnalyzer 的 legacy `candidate_corrected_xyz_m`。它没有输出 25 mm clip；只以 `max_authorized_correction_mm=100.0` 拒绝 raw-to-prediction 距离更大的候选。

然而同一 result 还保留一个不同的 Shadow candidate：`ShadowAnalyzer.correct_trajectory_point` 可将 raw 向 history/neighbor candidate 推进并以 `CorrectionConfig.max_correction_mm=25.0` 截断（`trajectory_corrector.py:88–93`）。该 25 mm candidate 被写入 `FramePointResult.candidate_corrected_*`，但 `FinalArbitrator` 不读取它。这是当前两个 candidate 语义并存的事实。

`CandidateSafety.safe` 仅要求：候选有限、geometry valid、raw-prediction correction <=100 mm、evidence_sufficient、post_correction_safe、I2 state 为 QUARANTINED/RECOVERY。代码不比较 candidate 对 GT（正确，GT 不可进 runtime），也不比较 candidate 与 I1 baseline 的真实误差；更关键的是 pipeline 当前将 `post_correction_safe=True` 硬编码传入（`pipeline.py:859–866`），并无独立 post-correction residual 的运行时计算。

**结论：`candidate_safe != candidate_beneficial`。**

它代表“坐标有限、几何/授权/可用证据/状态满足当前安全门”，不代表“candidate 比 I1 更准确”。这由失败 policy trial 的 39 harmful、15 clean false corrections直接反证。

### 关键摘录 5：CandidateSafety 的实际门

`stereo_research/final_arbitration.py:98–131`：

```python
finite = candidate_xyz_m is not None and all(math.isfinite(value) for value in candidate_xyz_m)
if not finite: reasons.append("candidate_not_finite")
if not geometry_valid: reasons.append("geometry_invalid")
if not correction_authorized: reasons.append("correction_unauthorized")
if not evidence_sufficient: reasons.append("evidence_insufficient")
if not post_correction_safe: reasons.append("post_correction_unsafe")
if i2_state not in {"QUARANTINED", "RECOVERY"}:
    reasons.append("i2_state_disallows_candidate")
return cls(..., tuple(reasons))
```

### 3.5 I2/Shadow 直接相关参数清单

| 参数 | 当前值 | 文件 | 作用 | 测试 |
|---|---:|---|---|---|
| `max_authorized_correction_mm` | 100 mm | `enhanced_processor.py:64` | FULL I2 raw→prediction candidate 授权上限 | large correction / state isolation |
| `recovery_confirmation_frames` | 2 | 同上:65 | 两个 clean observation 才 trust | recovery tests |
| `max_prediction_gap_s` | 1.0 s | 同上:66 | 大 gap 时退化为 latest trusted | 间接 |
| `recovery_consistency_mm` | 10 mm | 同上:67 | raw→trusted prediction recovery gate | recovery 行为测试；**无 10 mm 边界测试** |
| trusted history | 无固定上限 | 同上:84 | I2 predictor history | state-isolation tests |
| I2 anomaly physics threshold | `c_phy < 0.55` | `pipeline.py:852–857` | confirmed anomaly | 间接 |
| `PhysicsConfidenceConfig.threshold` | 0.55 | `physics_confidence.py:32` | legacy Shadow correction threshold | shadow/I2 tests |
| Shadow `prediction_weight/neighbor_weight` | .65/.35 | `trajectory_corrector.py:13–14` | legacy candidate blend | correction tests |
| Shadow `max_correction_mm` | 25 mm | 同上:15 | legacy candidate clip | correction tests |
| transient temporal threshold | 10 mm | `transient_gate.py:12` | legacy transient classification | indirect |
| evidence normal/abnormal | .7/.3 | 同上:13–14 | transient support | indirect |
| synchronous ratio | .6 | 同上:15 | transient support | indirect |

## 4. Innovation 3 审计

### 4.1 输入、fault 与 policy

Shadow I3 的 `FaultFingerprint` 实际使用 texture、LR/epipolar/matching/neighbor/flow/FB/temporal/reference/geometry/physics/synchronous-motion 字段（`shadow_analysis.py:155–168`）。但部分指纹输入固定不可用或固定值：`flow_3d` unavailable；spectral/phase/coherence unavailable；`tracking_loss_residual=0.0`；`blur_score` 未在该 fingerprint constructor 传入，因此 motion-blur 的双条件规则不能完整利用 blur score。

`RuleDiagnosticEngine` 映射如下：

| 条件（先匹配者） | fault_class | score |
|---|---|---:|
| tracking loss residual > .8 | TRACKING_LOSS | .95 |
| ref motion >1 且 common motion >.7 | CAMERA_MOTION | .95 |
| epipolar >1 或 geometry health >.6 | EXTRINSIC_DRIFT | .9 |
| LR>1 且 neighbor>1 且 ref<1 | STEREO_MISMATCH | .9 |
| gradient<.25 且 blur<.4 | MOTION_BLUR | .8 |
| gradient<.25 | WEAK_TEXTURE | .75 |
| flow/FB >1 | FLOW_DRIFT | .75 |
| matching>1 且 temporal<1 | SEARCH_RANGE_FAILURE | .7 |
| physics residual >.7 | OCCLUSION | .65 |
| 其余 | NORMAL | .05 |

随后 pipeline 的 typed policy mapping 为：

| I1/I3 输入 | risk | recommendation |
|---|---|---|
| I1 hard failure | BLOCKING | BLOCK_FINAL |
| empty/NORMAL fault | NORMAL | ALLOW_CORRECTION |
| `fault_confidence >= .8` | BLOCKING | BLOCK_FINAL |
| STEREO_MISMATCH < .8 | WARNING | ALLOW_CORRECTION |
| 其他 non-normal（含 OCCLUSION .65） | WARNING | WARN |

### 关键摘录 6：I3 policy mapping

`stereo_research/pipeline.py:_structured_i3_recommendation`，921–930：

```python
if hard_failure:
    return I3Recommendation(I3Risk.BLOCKING, "i1_hard_failure", I3Action.BLOCK_FINAL)
fault_class = result.fault_class.strip().upper()
if fault_class in {"", "NORMAL"}:
    return I3Recommendation.normal()
if (result.fault_confidence or 0.0) >= 0.8:
    return I3Recommendation(I3Risk.BLOCKING, f"i3:{fault_class}", I3Action.BLOCK_FINAL)
if fault_class == "STEREO_MISMATCH":
    return I3Recommendation(I3Risk.WARNING, f"i3:{fault_class}", I3Action.ALLOW_CORRECTION)
return I3Recommendation(I3Risk.WARNING, f"i3:{fault_class}", I3Action.WARN)
```

Phase 4.5 的 30 safe candidates 全部是 `OCCLUSION + WARNING/WARN`，是**数据与规则的共同结果**：Shadow fault rule 因 `physics_residual > .7` 输出 OCCLUSION/.65；policy 对所有不属于 STEREO_MISMATCH 的 warning 固定输出 WARN。它不是 I3 对 candidate 的直接质量判定。

### 4.2 I3 是否判断 candidate quality

当前 I3 是 **A. fault classifier + C. risk policy mapper**，不是 B. candidate-quality evaluator。它不读取 I2 `candidate_safety`、candidate xyz、candidate-vs-I1 residual、candidate temporal consistency、candidate geometry 或 candidate expected benefit。I3 的 typed action只由 I1 hard failure、`fault_class`、`fault_confidence`决定。I1 quality只以指纹中的部分原始字段间接参与 fault classification。

因此，冻结策略下 30 candidates 被 WARN 拦下；最小 trial 将 OCCLUSION 改为 ALLOW 后，FinalArbitrator 自然提交 45 corrections，但系统没有额外 candidate-quality evidence 来阻止 harmful/clean cases。这是代码与数据一致的结果，不是仲裁器绕权。

## 5. FinalArbitrator 审计

### 完整决策表

| I1 valid | I3 risk/action | candidate safe | FULL + write + experiment | proposed | committed / source / write |
|---:|---|---:|---:|---|---|
| 否 | 任意 | 任意 | 任意 | REJECT | REJECT / REJECTED / false（无授权或 baseline invalid） |
| 是 | BLOCKING | 任意 | 是 | REJECT | REJECT / REJECTED / true |
| 是 | BLOCKING | 任意 | 否 | REJECT | ACCEPT_WITH_WARNING / I1_BASELINE / false |
| 是 | 非 BLOCKING + ALLOW_CORRECTION | 是 | 是 | USE_CORRECTED | USE_CORRECTED / I2_CORRECTED / true |
| 是 | 非 BLOCKING + ALLOW_CORRECTION | 是 | 否 | USE_CORRECTED | ACCEPT_WITH_WARNING / I1_BASELINE / false |
| 是 | WARNING/WARN 或 unsafe candidate | 否/任意 | 任意 | ACCEPT_WITH_WARNING | ACCEPT_WITH_WARNING / I1_BASELINE / false |
| 是 | NORMAL、无 safe candidate | 否 | 任意 | ACCEPT | ACCEPT / I1_BASELINE / false |

`ExperimentAuthority.may_write_final` 必须同时为 `requested_mode == FULL_ENHANCED`、`write_enabled` 和 `scope == "experiment"`。`FULL_ENHANCED + write_enabled=False` 的 effective mode 是 Shadow。

### 关键摘录 7：authority 与仲裁必要条件

`stereo_research/final_arbitration.py:141–153,191–250`：

```python
def may_write_final(self):
    return (self.requested_mode is SystemMode.FULL_ENHANCED
            and self.write_enabled and self.scope == "experiment")
...
proposed = self._proposed(baseline, candidate_safety, diagnosis)
if proposed is FinalDecision.USE_CORRECTED and authority.may_write_final:
    return ArbitrationDecision(... I2_CORRECTED, True, candidate_safety.candidate_xyz_m, ..., True, authority)
if proposed is FinalDecision.REJECT and authority.may_write_final:
    return ArbitrationDecision(... REJECTED, False, None, ..., True, authority)
...
if not baseline.valid: return FinalDecision.REJECT
if diagnosis.risk is I3Risk.BLOCKING: return FinalDecision.REJECT
if candidate_safety.safe and diagnosis.action is I3Action.ALLOW_CORRECTION:
    return FinalDecision.USE_CORRECTED
if diagnosis.risk is I3Risk.WARNING: return FinalDecision.ACCEPT_WITH_WARNING
return FinalDecision.ACCEPT
```

### fallback

Enhanced exceptions are caught only around `_process_enhanced_results` in `_finalize_frame_results` (`pipeline.py:825–832`). `_enhanced_fallback` preserves an already-valid I1 final (semantic committed decision `ACCEPT_WITH_WARNING`, no write) or preserves invalidity (`REJECT`, no write). It does not re-run I1 and does not mutate I2 history.

### 5.1 `write_committed` 的真实语义

`write_committed` 是**原子 enhanced final decision 是否被写入 result**，不是“corrected value 成为 final”的同义词。

- `USE_CORRECTED`：将 candidate xyz/distance 写到 final fields，`write_committed=True`。
- `REJECT`：将 `status="rejected"`、final xyz/distance=`None`、`final_valid=False` 写入，`write_committed=True`。
- KEEP_BASELINE（ACCEPT/ACCEPT_WITH_WARNING）: final 保持 I1，`write_committed=False`。
- fallback：`write_committed=False`。

没有独立名为 “actual corrected-final replacement” 的 bool；应使用组合条件：`write_committed=True AND committed_decision==USE_CORRECTED AND result_source==I2_CORRECTED`。计数必须区分：

| 计数 | 原 Phase 4.5 | 失败 trial |
|---|---:|---:|
| atomic decision commits | 45 | 90 |
| reject commits | 45 | 45 |
| use_corrected commits | 0 | 45 |
| actual corrected-final replacements | 0 | 45 |

### 关键摘录 8：唯一增强 final write

`stereo_research/pipeline.py:_apply_final_decision`，968–1011：

```python
if not decision.write_committed:
    return result
if not decision.authority.may_write_final:
    raise PermissionError(...)
if decision.committed_decision is FinalDecision.USE_CORRECTED:
    ... validate finite candidate ...
    return replace(result, final_x_m=float(xyz[0]), final_y_m=float(xyz[1]),
        final_z_m=float(xyz[2]), final_distance_m=final_distance, final_valid=True,
        ..., write_committed=True, result_source=decision.result_source.value)
if decision.committed_decision is FinalDecision.REJECT:
    ...
    return replace(result, status="rejected", final_x_m=None, final_y_m=None,
        final_z_m=None, final_distance_m=None, final_valid=False,
        ..., write_committed=True, result_source=decision.result_source.value)
```

## 6. Root Cause Ranking（代码 + Phase 4.5 数据）

### P0 — CandidateSafety 未包含 beneficial evidence

证据：`post_correction_safe=True` 被 pipeline 硬编码；CandidateSafety 未比较 candidate/I1 relative quality；I2 candidate 是 trusted prediction；trial 产生 45 corrections，其中 39 harmful、15 clean false。故 safe 仅表示当前 gate 可用，不表示更好。这是有效增强闭环的第一阻塞点。

### P0 — I3 不区分 candidate quality，只按 fault→policy 映射

证据：`_structured_i3_recommendation` 不接收 candidate；30 safe candidates 全为 OCCLUSION/WARN 而没有 correction；把该 fault 一刀切放开后即出现 harmful/clean failures。问题既不是“仅 I3 太保守”，也不是“仅 I2 坏”，而是 I2 safe 与 I3 policy之间没有质量判别接口。

### P1 — Recovery Gate 混合了两种失败语义

证据：I1 hard failure 在 I2 前即被拒绝；其余 valid I1 必须通过 10 mm raw-prediction gate再连续两帧确认。Phase 4.5 的 27 既含 baseline tracking recovery failure，又含 I2 consistency failure。单一“episode ≤2”统计把二者合并；它是严格安全 Gate，不等价于“算法一定已经恢复不了”。

### P1 — Shadow 与 FULL 有两个 candidate contract

Shadow candidate 是 trajectory corrector 的 25 mm capped blend；FULL candidate 是 I2 prediction 的 100 mm-authorized candidate。两者写入同一 `FramePointResult` 的不同字段，但最终仲裁只消费后者。这不是 final-write 旁路，却会让评审/实验分析若未明确字段来源而误解 candidate quality。

### P1 — I2 标称 timestamp-aware，但 pipeline 实际用 frame index

证据：`timestamp_s=float(result.frame)`。固定 30 fps controlled source 中比例关系固定，影响被掩盖；对不规则 timestamps，`max_prediction_gap_s=1.0` 并不按真实 sample time 评估。

### P2 — Shadow diagnostic history 与 I2 trusted history不同

ShadowAnalyzer 对每个 valid baseline append raw `_history_mm`（`shadow_analysis.py:203–207`），包括后来 I2 会 quarantine 的 baseline。它不污染 I2 trusted predictor，却会影响后续 `c_phy`、physics residual 和 fault class；应避免在讨论“history contamination”时混淆两者。

## 7. 创新点实现完整性

| 模块 | Architecture | Implementation | Validation | 主要缺口 |
|---|---|---|---|---|
| Innovation1 | Mostly complete | Mostly complete | Mostly complete | 本审计不复验性能；图像异常下的 baseline tracking recovery 与 I2 gate 要分层统计 |
| Innovation2 | Mostly complete | Mostly complete | Mostly complete（controlled） | candidate safety 缺 beneficial proof；真实 timestamps 未贯通；recovery边界未直接测试 |
| Innovation3 | Partial | Mostly complete as classifier | Partial | 没有 candidate-quality evaluation；部分 fingerprint evidence 永远 unavailable/未接入 |
| FULL_ENHANCED | Mostly complete authority architecture | Mostly complete | Partial | M2/M3 parity、安全写入已验证；尚无零 harmful 且有正收益的 corrected-final evidence |

## 8. 过度工程化 / 契约漂移检查

真正存在的项：

1. **双 candidate 表达**：Shadow 25 mm candidate 与 FULL prediction candidate并存，最终只消费后者。不是重复 pipeline，但会增加模型认知负担。
2. **双 history 语义**：I2 trusted history安全隔离，Shadow history保留 raw baseline用于诊断；合理但必须在报告/字段中分名，否则会误读为 I2 contamination。
3. **诊断字段多于 final 决策使用字段**：`FramePointResult` 中大量 Shadow/physics/fault fields只用于 audit，I3 policy最终只用 fault class/confidence。它们当前有实验用途，不能称为死代码。

未发现重复的正式 Pipeline 或 I2/I3 final write owner；也未发现 GUI/生产路径获实验 authority。下一步不应再加模块，应先简化/明确现有 candidate-quality contract并用冻结协议验证。

## 9. 测试覆盖审计

相关测试文件直接定义的数量：I1 foundation/matching/pipeline 10；I2 correction/residual/spectral/state/validation 16；I3 diagnostics/recovery 4；FinalArbitrator 7；FULL enhanced 11；Phase4/4.5 + injector 10；Shadow/occlusion safety 13；另有 target-accuracy相关 18。总数与其余全局/GUI/dataset tests共同构成 351。

| 分支 | 直接覆盖 | 结论 |
|---|---|---|
| I1 basic matching / result baseline | I1 foundation/matching/pipeline tests | 有 |
| I2 state/history isolation | `test_innovation2_state_isolation.py` 8 tests | 有：raw/candidate不反馈、two-frame recovery、hard failure |
| I2 candidate authorization | I2 correction + final pipeline tests | 有，但非 beneficial quality |
| I3 diagnosis rule | innovation3 diagnostics + Shadow tests | 有分类/不变性；candidate-quality无 |
| FinalArbitrator authority | `test_final_arbitration.py` 7 + full enhanced | 有 |
| atomic corrected/reject commit | full enhanced pipeline tests | 有 |
| fallback | full enhanced pipeline test | 有 |
| image-level injector/frozen pairing | injector + phase4_5 manifest/validation tests | 有 |
| M2/M3 pre-commit parity | phase4_5 validation | 有 |
| beneficial candidate → USE_CORRECTED | 仅构造 safe candidate 的单元测试 | **没有使用真实图像、证明 beneficial 的直接测试** |
| harmful candidate → reject | blocking I3 reject有 | **没有“candidate harmful”运行时判别测试** |
| clean safe candidate → no false correction | Gate / failed trial | **失败 trial 已反证，现无通过策略测试** |
| lost/ambiguous → recovery | I2 hard failure与 pipeline recovery间接覆盖 | **没有 image-level exact case direct test** |
| 10 mm gate边界 | 无 | **缺失** |
| 精确两帧 recovery边界 | normal synthetic case覆盖 | 缺少 image-level delayed/invalid boundary |

## 10. 不改代码的下一步建议

### A. 必须先解决（最多三项）

1. 写出并审查一个**candidate beneficial evidence contract**：明确可观测的候选相对 I1 证据，而不是把 `candidate_safe` 解释为准确性承诺。
2. 将 recovery Gate 报告拆成 I1 invalid/hard-failure 与 I2 valid-but-inconsistent 两类，先决定各自的科学语义；不得仅放宽 10 mm。
3. 为上述两项补直接红测：clean safe candidate、不利 candidate、10 mm 边界、两帧确认和真实 timestamp 行为。

### B. 可以后解决（最多三项）

1. 明确或统一 Shadow candidate 与 FULL candidate 的字段命名/使用说明。
2. 将 `DatasetSample.timestamp` 贯通到 I2 `TimedObservation`，在不规则时间间隔下验证。
3. 审查 Shadow fingerprint 中当前不可用/未接入的 evidence，删除或明确标注其研究边界。

### C. 不建议再做

- 不建议继续以 fault class 批量放宽 `ALLOW_CORRECTION`；failed trial 已表明该做法不安全。
- 不建议为了 Gate 成功而放宽 10 mm recovery gate或把 hard failure 视为已恢复。
- 不建议添加新的 correction mode、重复 Pipeline、神经网络/新传感器或新的创新点。

## 审计限制与工作树说明

未发现未预期的 enhanced final-write 旁路。审计期间没有改动生产代码或测试；仅新增本报告。开始审计时工作树已包含大量历史 staged/unstaged/untracked 文件（其中含既有 Phase4/4.5 代码与无关改动），因此不能将整个仓库当前 `git diff` 解释为“本次审计造成的差异”。本次只读操作未新增任何生产代码差异。
