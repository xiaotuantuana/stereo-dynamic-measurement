# 第二轮 Innovation 1 Target-Accuracy Policy 实施报告

日期：2026-08-20  
范围：仅处理 Innovation 1；未自动 commit；未清理或覆盖工作区中与本轮无关的用户改动。

## 1. Existing Precision Planner Audit

修改前的 `precision_planner.py` 使用：

```text
Z = fB / d
sigma_Z ≈ Z² / (fB) · sigma_d
sigma_d_required = sigma_Z_target · fB / Z²
```

其数学含义实际是标准差/标准不确定度 `sigma`，不是 RMSE、MAE、最大误差或 95% 区间，但旧参数名 `target_depth_error_mm` 没有明确表达这一点。

修改前已有能力：

- 计算 Z 目标对应的 required disparity precision；
- 对候选 baseline 做离线扫描；
- 在给定 assumed disparity noise 时判断推荐 baseline 是否可行。

修改前缺失：

- 显式 target metric；
- runtime estimated sigma disparity；
- PrecisionPlan、precision ratio、limiting axis；
- runtime feasibility/unmet 状态；
- 与 `TemporalStereoPipeline` 的真实连接。

旧函数保留以兼容现有调用；新增合同只允许 `metric="sigma"`。Baseline recommendation 仍然只是 offline planning，不修改 T/P/Q 或 rectification。

## 2. Files Changed

本轮实现文件：

- `stereo_dynamic_measurement/innovation1/precision_planner.py`
  - 新增 `TargetAccuracySpec`、`PrecisionPlan` 和 `build_precision_plan()`。
- `stereo_research/accuracy_policy.py`
  - 新增纯 `MeasurementPolicyDecision`，融合 precision plan、vision state、search radius 和 retry bound。
- `stereo_research/models.py`
  - 最小增加默认关闭的配置和可审计结果字段。
- `stereo_research/local_matching.py`
  - 新增 per-call `refinement_level`；仅改变现有 IC-GN 的迭代预算与终止容差输入，不修改 IC-GN 方程。
- `stereo_research/pipeline.py`
  - 增加 causal pre-match policy、post-match precision check 和有限 retry。
- `stereo_research/runner.py`
  - 在第一轮 169 列之后追加 22 个 accuracy-policy 字段。
- `stereo_dynamic_measurement/innovation1/baseline_experiment.py`
  - 在现有实验框架内增加 A0–A3 × 四目标等级的 policy ablation CSV。
- `stereo_dynamic_measurement/run_innovation1_experiment.py`
  - 输出新增的 policy CSV 路径。

新增/更新测试：

- `tests/test_target_accuracy_planner.py`
- `tests/test_target_accuracy_policy.py`
- `tests/test_target_accuracy_csv.py`
- `tests/test_target_accuracy_pipeline.py`
- `tests/test_innovation1_baseline_experiment.py`
- `tests/test_shadow_pipeline.py`
- `tests/test_shadow_state_isolation.py`

设计与计划：

- `docs/superpowers/specs/2026-08-20-innovation1-target-accuracy-policy-design.md`
- `tasks/innovation1-target-accuracy-plan.md`
- `tasks/innovation1-target-accuracy-todo.md`

## 3. Accuracy Budget

本轮支持的目标语义：

```text
metric = sigma
target_sigma_x_mm = optional, PARTIAL_NOT_CONTROLLING
target_sigma_y_mm = optional, PARTIAL_NOT_CONTROLLING
target_sigma_z_mm = optional, controlling
```

运行期使用：

```text
required_sigma_d_px = target_sigma_z_mm · f_px · B_mm / Z_mm²
estimated_sigma_d_px = sqrt(disparity_variance_px2)
estimated_sigma_z_mm = Z_mm² / (f_px · B_mm) · estimated_sigma_d_px
precision_ratio = estimated_sigma_d_px / required_sigma_d_px
```

- `precision_ratio <= 1`：估计标准不确定度满足目标；
- `precision_ratio > 1`：Stereo 可以有效，但目标精度未满足；
- X/Y target 被保留在合同和 CSV 中，estimated sigma X/Y 为空，不编造传播结果。

运行期 focal/baseline 只从现有 Q 的标准几何项读取；不修改 Q。单位按现有 `calibration_unit` 转为 mm 后进入 planner。

## 4. Measurement Policy Flow

```text
Target sigma XYZ + previous valid depth/disparity + Vision State
                              ↓
                       Accuracy Budget
                              ↓
                 causal PRE-MATCH POLICY
     existing motion radius + target-driven refinement level
                              ↓
          existing Stereo / Subpixel / IC-GN
                              ↓
 actual match variance + LR/FB/cycle/texture/confidence
                              ↓
                POST-MATCH PRECISION CHECK
                              ↓
          ACCEPT / RETRY_STRONGER / PRECISION_UNMET
```

初始化帧继续使用 legacy 策略，并标记 `WARMUP`。后续帧只使用过去状态和当前图像/匹配结果，不读取 future frame 或 GT。

## 5. Runtime Parameters Actually Controlled

Target accuracy 真实控制了以下现有输入：

- IC-GN 最大迭代预算：
  - level 0：`icgn_max_iterations`
  - level 1：`2 × icgn_max_iterations`
  - level 2：`3 × icgn_max_iterations`
- IC-GN 终止容差：
  - `icgn_epsilon / (1 + refinement_level)`
- post-match bounded retry：
  - 当 Stereo 有效、precision unmet、目标可达且 vision state 为 HIGH/MEDIUM 时，追加更强 refinement；
  - `for range(max_precision_retry)` 明确限制次数；
  - retry 结果只有在有效且 estimated sigma d 不更差时才替换当前候选。
- acceptance status：
  - `MET`、`VALID_BUT_PRECISION_UNMET`、`INFEASIBLE`、`UNAVAILABLE`、`WARMUP`。

Target accuracy 不直接缩小 search radius。现有 motion/prediction/confidence/recovery controller 继续决定 base radius；policy 只读取并原样保留该 radius。Vision State 还决定 precision retry 是否安全可用。

## 6. Feasibility Logic

配置中的 `sqrt(uncertainty_min_disparity_variance_px2)` 作为当前模型的最优 uncertainty floor：

```text
required_sigma_d >= minimum_achievable_sigma_d → feasible
required_sigma_d < minimum_achievable_sigma_d  → INFEASIBLE
```

`INFEASIBLE` 时不 retry，不缩小搜索范围，也不丢弃有效 Stereo 测量。该 floor 是工程估计，不是已用真实 GT 标定的科学保证。

## 7. Tests

- 原有：219
- 新增：19
- Total：238
- Passed：238
- Failed：0

新增测试覆盖：

- sigma metric 与 worked numeric example；
- target/distance/focal/baseline 单调性；
- GT-free planner API；
- feasible/infeasible；
- target 与 vision state 联合控制；
- 1 次和 2 次 bounded retry；
- valid Stereo 与 precision unmet 分离；
- warmup；
- 169 列前缀与 22 列 append-only；
- Feature OFF 完整序列；
- Innovation 2/3 Shadow safety；
- A0–A3 experiment output。

## 8. OFF-mode Regression

`enable_target_accuracy_policy=False` 时，以同一三帧 `research_full` 确定性序列比较：

- 所有 legacy 字段；
- 所有第一轮 Shadow 字段；
- 所有新增 policy 字段的 disabled/empty 语义；
- PointState；
- Kalman state 与 covariance。

除墙钟耗时外全部完全一致。Shadow OFF/ON 第一轮完整序列测试继续通过。两个 active Shadow 写 final 配置仍会抛出 `NotImplementedError`。

CSV 当前为 191 列，大小写不敏感条件下 191 列全部唯一；前 169 列名称、顺序和意义不变。

## 9. Innovation 1 Experiment

现有命令：

```text
python run_innovation1_experiment.py --config configs/innovation1_experiment.yaml
```

继续生成原 baseline-distance 文件，并新增：

```text
outputs/innovation1/target_accuracy_policy_ablation.csv
```

实验组：

- A0_FIXED：legacy 固定策略；
- A1_VISION_ONLY：仅视觉状态自适应；
- A2_ACCURACY_ONLY：固定 motion radius + accuracy budget；
- A3_FULL：vision radius + accuracy refinement/retry。

目标：Loose 10 mm、Medium 3 mm、Strict 2.5 mm、Infeasible 0.1 mm，均表示 1-sigma Z target。

## 10. Policy Behavior

A3_FULL 的代表性分析结果：

| Target | required σd (px) | estimated σd (px) | estimated σZ (mm) | Radius | Level | Retry | Status | Estimated attainment |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Loose 10 mm | 0.08887 | 0.05 | 5.626 | 12 | 0 | 0 | MET | 1.0 |
| Medium 3 mm | 0.02666 | 0.05 | 5.626 | 12 | 1 | 1 | VALID_BUT_PRECISION_UNMET | 0.0 |
| Strict 2.5 mm | 0.02222 | 0.05 | 5.626 | 12 | 2 | 1 | VALID_BUT_PRECISION_UNMET | 0.0 |
| Infeasible 0.1 mm | 0.000889 | 0.05 | 5.626 | 12 | 2 | 0 | INFEASIBLE | 0.0 |

A2 的 radius 为 8；A3 的 vision-state radius 为 12。Target 等级改变 refinement/retry，但没有粗暴替代 motion radius。

真实 `car.avi`、`research_full`、10 mm sigma target 冒烟：

- 303 rows；有效率 OFF/ON 均为 99.67%；
- OFF 平均 frame time：74.97 ms；
- ON 平均 frame time：97.75 ms；
- ON 平均 retry：0.328；
- ON 平均 radius：4.132 px；
- ON 平均 refinement level：1.669；
- 状态：3 WARMUP、99 VALID_BUT_PRECISION_UNMET、200 INFEASIBLE。

该视频无 GT，因此这些值只证明真实控制闭环与计算投入，不能作为真实精度结论。

## 11. Accuracy Results

Policy analytic experiment 的 GT 仅在 decision 完成后用于 evaluation：

- Depth RMSE：5.708 mm；
- Depth MAE：4.541 mm；
- GT target attainment：Loose 92.2%、Medium 41.6%、Strict 35.8%、Infeasible 1.0%。

Feature OFF 的原合成回归保持：

| 指标 | 第一轮 | 第二轮 OFF |
|---|---:|---:|
| X RMSE | 0.612320 mm | 0.612320 mm |
| Y RMSE | 0.201672 mm | 0.201672 mm |
| Z RMSE | 6.435146 mm | 6.435146 mm |
| 3D RMSE | 6.467358 mm | 6.467358 mm |

## 12. Remaining Risks

- **PARTIAL — XYZ budget**：Z 已闭环；X/Y target 可配置和记录，但尚未作为控制轴。
- **PARTIAL — uncertainty calibration**：`disparity_variance_px2` 是基于视觉残差的估计，尚未用真实实验 GT 做 calibration/coverage 验证。
- **PARTIAL — target attainment**：当前完整 target-level 对比主要来自合成/解析实验；`car.avi` 没有真实 XYZ GT。
- **PARTIAL — policy runtime experiment**：ablation CSV 中 `runtime_ms` 是 planner/policy 决策耗时；实际端到端 frame time 由 car smoke 单独报告。
- **PARTIAL — refinement effectiveness**：更高预算不保证 RMSE 一定降低；系统只保证真实投入增加、重试有界并诚实报告 unmet。
- **MISSING — real GT**：仍需要带同步真实位移/三维坐标真值的数据，验证 estimated sigma coverage 与实际误差的一致性。

## 13. Deferred

- runtime variable baseline；
- T/P/Q 动态更新与 re-rectification；
- extrinsic recalibration；
- calibrated full Q-Jacobian XYZ covariance；
- active Innovation 2 correction；
- active Innovation 3 recovery；
- scientific C_phy calibration；
- full structural dynamics model。

## Acceptance Summary

第二轮验收项已满足：feature 默认关闭且完整回归；metric 明确为 sigma；GT 不进入 planner/policy/matching/retry；target accuracy 真实控制 IC-GN effort、acceptance 与 bounded retry；search radius 继续由 motion/vision 主导；不可达目标诚实标记；有效 Stereo 不因 precision unmet 被丢弃；第一轮 Innovation 2/3 和硬禁用规则无回归；Geometry、disparity、Q 单位与核心算法语义未修改。
