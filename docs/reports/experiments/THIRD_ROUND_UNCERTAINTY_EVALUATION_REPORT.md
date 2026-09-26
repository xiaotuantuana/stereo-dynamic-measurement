# 第三轮实施报告：不确定度校准、Refinement 验证、E0/E1 接口

## 1. 不确定度来源审计

`stereo_research/uncertainty.py::estimate_match_uncertainty()` 在
`stereo_research/pipeline.py::_valid_result()` 中、Kalman 状态更新之前计算
`disparity_variance_px2`。它不是 IC-GN Hessian 的解析协方差，而是风险加权的启发式方差：

```text
variance_d = clip(0.01 * (1 + weighted_risk), 0.0025, 4.0) px²
sigma_d = sqrt(variance_d)
```

风险使用 texture、photo matching cost、margin、LR、左右/时间 FB、cycle、IC-GN residual、IC-GN Hessian density、curvature 和邻域 MAD；缺失项不参与权重平均。故它与 cost、LR/FB、Hessian/curvature 有关，但不等同于 Hessian-derived covariance。

`uncertainty_min_disparity_variance_px2=0.0025` 给出 `sigma_d=0.05 px` 的硬 floor；默认 base 的 sigma 是 `0.1 px`。因此第二轮中接近 0.05 px 的样本确实是 floor 截断，而非“所有 refinement 都真正估出 0.05 px”。retry 每次都会重新 match 并重新调用该估计器；若观测风险不变/缺失或结果裁剪到 floor，level/retry 的输出就不会变化。

Z 仍以 `sigma_Z_mm=Z_mm²*sigma_d/(f_px*baseline_mm)` 传播。X/Y covariance 仍是 `PARTIAL`，没有伪造完整 XYZ covariance。

## 2. 运行时与评价边界

```text
Runtime: video -> pipeline -> estimated raw sigma -> measurement.csv
Offline: measurement.csv + independent gt.csv -> evaluation records -> tables
```

主 runner 不再读取、插值或写入 external GT；旧 CSV GT 列为兼容性保留但为空。评价模块是只读消费者。GT 不会进入 matcher、precision policy、retry、confidence、Innovation 2/3 diagnosis 或 recovery。

## 3. 已实现文件

- `stereo_research/uncertainty_evaluation.py`：GT schema、frame/timestamp 对齐、error、coverage、ratio、Pearson/Spearman、显式单比例 calibration model。
- `stereo_research/evaluation_runner.py`：E0 静态评价、E1 manifest/identity guard、U0--U3 评价表接口。
- `stereo_research/runner.py`：解除 runtime GT loader；仅保留 runtime CSV。
- `stereo_research/models.py`、`stereo_research/pipeline.py`、`stereo_research/runner.py`：最小 retry telemetry（trigger/accept、前后 sigma/cost/LR）。
- `tests/test_uncertainty_evaluation.py`、`tests/test_e0_e1_experiment_interfaces.py`：评价隔离与 E0/E1 contracts。

## 4. U0 合成 sanity（实际运行）

`outputs/uncertainty_round3/U0/uncertainty_calibration_table.csv` 基于 604 个已有合成样本，以已声明 `localization_sigma_px=0.05` 的 `sqrt(2)` disparity noise 作为 **simulator noise reference**。这不是 runtime estimator calibration。

| Metric | disparity | Z |
|---|---:|---:|
| RMSE | 0.15442 px | 6.43515 mm |
| Mean sigma | 0.07071 px | 2.94688 mm |
| Coverage 1/2/3 sigma | 37.25% / 65.89% / 81.46% | 37.25% / 65.73% / 81.46% |
| Calibration ratio | 2.18380 | 2.18367 |

该参考在此合成条件下明显低估实际误差；固定 sigma 没有 ranking 信息，disparity Spearman 不可用。第二轮 Feature-OFF 合成误差保持：X/Y/Z/3D RMSE = 0.612320 / 0.201672 / 6.435146 / 6.467358 mm。

## 5. Refinement、retry 与真实实验状态

U1/U2/U3 的独立输出接口已经生成 `refinement_ablation.csv`、`retry_effectiveness.csv`、`vision_quality_stratification.csv`。它们要求各 level/retry variant 的 measurement CSV 加独立 GT，且 retry 真值增益只在 output join 后计算。当前工作区没有对应的真实 GT 序列，故尚不能声称 refinement/retry 带来真实误差改善；状态应为 `DATA_REQUIRED`。

E0 支持 `frame`/`timestamp_s`、`point_id`、`gt_x_mm/gt_y_mm/gt_z_mm`（及可选 GT sigma）；timestamp 只有在最近样本不超过 `max_time_offset_ms` 时对齐。无 GT 返回 `DATA_REQUIRED`，测量不受影响。

E1 condition 包含 distance、baseline、target、method、repeat、calibration identity、calibration 与 measurement 路径；runner 将 baseline 与标定 `abs(T_x)` 比较，不一致 fail-fast。模板位于 `outputs/uncertainty_round3/E1_manifest_template.json`，在尚未提供真实 GT 时返回 `DATA_REQUIRED`。

## 6. 风险与第四轮建议

- runtime heuristic sigma 尚未在独立真实数据上校准；calibration factor 目前只提供显式离线拟合模型，默认不会应用或暗中使用 alpha=1。
- refinement/retry 的真实收益尚未证明；不得将计算量增加包装成精度提升。
- E0/E1 缺独立物理 GT；X/Y uncertainty 仍为 PARTIAL。

下一轮应先采集按 sequence 划分的独立 calibration/evaluation GT 数据：若 runtime estimator 无效，优先修 estimator；若 refinement/retry 无收益，降低其论文权重并重新审视亚像素策略；只有 Z uncertainty 有可信验证后，再扩展 full XYZ covariance 或升级 Innovation 2 confidence。

## 7. 测试与回归

新增 9 个第三轮测试；完整 `python -m pytest -q` 结果为 **247 passed in 41.64s**。Feature-OFF 合成基线未更改；Innovation 2/3 Shadow 隔离与安全测试继续通过。未执行 commit。
