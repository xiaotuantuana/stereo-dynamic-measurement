# 第三轮设计：不确定度校准、Refinement 验证与 E0/E1 实验接口

## 目标与范围

本轮建立严格的 **运行时测量** 与 **离线评价** 边界，验证 disparity/Z 不确定度、refinement 和 retry 的真实有效性；不改变双目几何、Q、视差符号、IC-GN/LK/SGBM 数学，也不启用 Innovation 2/3 主动写回。

运行时数据流：

```text
video -> TemporalStereoPipeline -> estimated sigma (raw[/optional calibrated]) -> measurement.csv
```

评价数据流：

```text
measurement.csv + independent gt.csv -> EvaluationRecord -> evaluation CSV / tables
```

GT、实际误差、coverage、gain 均只存在于第二条链，绝不成为 matcher、policy、retry、confidence、diagnosis 或 recovery 的输入。既有主测量 CSV 字段保持兼容（旧 GT 列不再由 runner 写入）；仅追加 8 个 retry runtime telemetry 字段，因此为 199 列。所有新的 GT/actual-error/coverage/gain 细节均写入独立评价 CSV。

## Phase 1 审计结论

`disparity_variance_px2` 由 `stereo_research.uncertainty.estimate_match_uncertainty()` 生成，并在 `TemporalStereoPipeline._process_point()` 的 Kalman 状态变更前调用。

公式为：

```text
weighted_risk = sum(w_i * risk_i) / sum(w_i), 仅对可用项
variance_d = clip(base_variance_d * (1 + weighted_risk), min_variance_d, max_variance_d)
sigma_d = sqrt(variance_d)
```

默认 `base_variance_d=0.01 px²`、`min_variance_d=0.0025 px²`、`max_variance_d=4 px²`，故 hard floor 是 `0.05 px`，而正常 base 的 sigma 是 `0.1 px`。风险项来自 texture、photo cost、uniqueness margin、左右/时间 flow FB、LR、cycle、IC-GN residual、IC-GN Hessian density、ZNCC curvature 和邻域 disparity MAD。故：与 matching cost、margin、LR/FB、IC-GN residual/Hessian/curvature 均有关，但不是由 Hessian 协方差公式直接推导。

每次 `LocalMatcher.match()`（包括 retry）都产生新的 LocalMatchResult，pipeline 随后重新调用估计器；没有覆盖回旧 variance 的路径。refinement level 仅改变 IC-GN iteration/epsilon；若风险项不变、不可用、或 clip 到 floor，variance 不会随 level 改变。第二轮 `~0.05 px` 的直接数学来源只能是该 floor；第三轮 U0/U1 会输出 raw variance 和组件，验证具体样本是否为 floor 截断。

`sigma_Z` 保持第二轮一阶传播：`sigma_Z_mm = Z_mm² * sigma_d / (f_px * baseline_mm)`。X/Y sigma 继续明确为 `PARTIAL`，不伪造完整 XYZ covariance。

## 设计

### Evaluation-only 模块

新增 `stereo_research.uncertainty_evaluation`，公开 seam 为：

- `UncertaintyEvaluationRecord.from_measurement_and_gt(...)`：只将已输出的 measurement row 和独立 GT record 结合，计算 disparity/Z/XYZ 实际误差。
- `evaluate_records(...)`：输出 RMSE、mean sigma、1/2/3 sigma coverage、calibration ratio、Pearson/Spearman、偏差、MAE、std、drift、max error、outlier rate。
- `fit_scale_calibration(...)` / `UncertaintyCalibrationModel`：只以独立 calibration sequence 拟合单一正比例系数；运行时默认 `UNCALIBRATED`，未载入模型时 calibrated sigma 为 unavailable，绝不静默采用 alpha=1。
- `align_ground_truth(...)`：`frame` 或 `timestamp` 对齐；timestamp 最近样本超过 `max_time_offset_ms` 就排除。

任何 runtime uncertainty 函数的签名不得接受 gt/error 参数；评价只读 CSV/records，不修改 pipeline state 或 measurement 行。

### U0--U3

新增 evaluation experiment runner，在 synthetic outputs 上生成：

- `uncertainty_evaluation.csv`、`uncertainty_calibration_table.csv`；
- `refinement_ablation.csv`（R0/R1/R2，独立运行）；
- `retry_effectiveness.csv`（No Retry / Bounded Retry，before/after、accepted、improved/worsened/no-change）；
- `vision_quality_stratification.csv`（good/medium/poor）。

GT 仅在每个运行完成后 join。Refinement gain 和 retry actual gain 仅为评价列。运行结果不要求预设“变好”：若收益不足，报告 `LOW_EFFECTIVENESS`，不改算法掩盖结论。

### E0/E1

新增分离的真实 GT schema：`frame` 或 `timestamp_s`、`point_id`、`gt_x_mm`、`gt_y_mm`、`gt_z_mm`，可选各轴 GT sigma。E0 runner 在 GT 缺失时返回 `DATA_REQUIRED`，不影响普通测量。

E1 manifest 的每个 condition 包含 `distance_m`、`baseline_mm`、`target_sigma_z_mm`、`method`、`repeat`、`calibration_id` 和 calibration 路径。E1 runner 在运行前比较 `baseline_mm` 与校准 `abs(T_x)`（按单位转为 mm），超出容差即 fail-fast；不运行时修改 baseline 或复用不匹配 Q。

## 验证 seam

1. runtime uncertainty API 不含 GT；GT 改变不改变 runtime sigma；
2. record、coverage、ratio、ranking 的独立数值合同；
3. frame/timestamp 对齐及超时拒绝；
4. E0 评价只读取独立 GT，GT 缺失时 measurement pipeline 正常；
5. E1 identity guard；
6. U1/U2 输出独立等级和 before/after；
7. Feature OFF 与 Shadow 安全完整回归。

## 风险与非目标

当前 sigma 是启发式风险估计，不保证已校准；合成数据和真实数据的统计结论必须分开。第一版 scale factor 仅提供可审计的离线接口且默认关闭。真实 E0/E1 数据不在工作区时，接口完成而实验状态为 `DATA_REQUIRED`。本轮不做 full Q-Jacobian XYZ covariance。
