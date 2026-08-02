# 硕士论文版算法修改说明

## 修改文件

- `stereo_research/models.py`：扩展方法、配置、状态和逐点输出模型。
- `stereo_research/pipeline.py`：分离候选验证与状态提交，增加时间估计、位移参考和有限恢复。
- `stereo_research/local_matching.py`：动态归一化代价、三级金字塔、双尺度独立验证和连续亚像素搜索。
- `stereo_research/tracking.py`：继续使用金字塔 LK 和前后向检查，恢复时使用最后有效参考帧。
- `stereo_research/runner.py`、`run.py`：扩展 CSV、方法别名、配置开关和消融套件。
- `stereo_research/annotate_gt.py`：人工左右像素真值自动通过 Q 重投影为三维真值。
- `stereo_research/metrics.py`：增加像素、毫米级三维/位移、稳定性、跳变和恢复指标。
- `stereo_research/report.py`：增加论文轨迹、误差、覆盖率和恢复图，并输出 PNG/PDF。
- `tests/`：增加状态污染、测量/估计顺序、独立验证、金字塔、亚像素、恢复、真值和指标测试。

## 算法改进原理

1. 候选左右点和原始视差先完成 LR、三维重投影、深度范围与运动检查；只有全部通过才提交 `right_xy`、原始/估计视差历史、XYZ 和置信度。失败仅更新失败计数与跟踪状态。
2. `measured_disparity` 是当前左右图像直接得到的亚像素观测，`estimated_disparity` 是观测通过验证后的时间估计。LR、唯一性和图像代价不使用滤波值。
3. 小窗口的大创新由大上下文独立复核；大窗口恢复的大创新由三级金字塔粗到细结果复核，不允许自验证。
4. 外观、预测、极线和邻域项只在启用且有效时进入加权和，并除以当前有效权重之和，保持不同消融配置的代价尺度一致。
5. 三级金字塔按 1/4、1/2、原分辨率传递视差，粗层扩大等效搜索范围，原层完成精细匹配。
6. 默认在整数最优值的 ±1 px 内按 0.1 px 连续采样，通过 `getRectSubPix` 和同一外观代价选择亚像素结果；可切回抛物线快速模式。
7. `tracking → recovering → lost` 管理连续失败。`full_quality` 在 lost 后按间隔使用最后有效左图、扩大窗口、金字塔匹配；无可用视差历史时使用 SGBM 重新初始化。恢复成功后清除失败计数并重置速度。

## 新增主要配置

- 融合权重：`photo_weight`、`prediction_weight`、`epipolar_weight`、`neighborhood_weight`。
- 金字塔：`enable_pyramid`、`pyramid_levels`、`pyramid_search_radius`、`pyramid_refine_radius`。
- 亚像素：`subpixel_method`、`subpixel_step`。
- 恢复：`enable_recovery`、`recovery_interval_frames`、`recovery_search_radius`。
- LK：`lk_window_size`、`lk_max_level`、`recovery_lk_window_size`、`recovery_lk_max_level`。
- 时间估计：`enable_temporal_estimation`、`quality_disparity_alpha`、创新与尺度一致性门限。
- 物理检查：`min_depth_m`、`max_depth_m`、可选 `max_motion_m_per_frame`。

## 新增输出字段

保留旧字段 `disparity`、`raw_disparity`、`X_m/Y_m/Z_m`，并增加：

- `integer_disparity`、`measured_disparity`、`estimated_disparity`、`subpixel_offset`；
- `measured_right_x/y`、`estimated_right_x/y`；
- `measured_X/Y/Z_m`、`estimated_X/Y/Z_m`；
- `delta_X/Y/Z_mm`；
- `neighbor_disparity`、`used_search_radius`、`quality_stage`；
- `recovery_stage`、恢复尝试/成功次数、`mean_recovery_frames`。

## 新增评价指标与图表

汇总增加左右像素 MAE、视差 MAE/RMSE、XYZ 与距离毫米误差、三方向相对位移误差、静态标准差/峰峰值/漂移、1/2/5/10 mm 跳变率、三维创新 RMSE/异常率、lost 与恢复统计。报告生成位移三轴轨迹、原始/滤波对比、视差误差、覆盖率、错误匹配率、恢复率和运行时间图；有真值时再生成预测—真值与位移误差图。

人工像素真值重投影使用与算法相同的标定 Q，因此适合评价跟踪和匹配误差，但不能独立评价相机标定系统误差。

## 运行完整方法

```powershell
python -m stereo_research.run --manifest experiments/car/manifest.json --methods full_quality --repeats 5 --warmup-frames 30
python -m stereo_research.evaluate --experiment experiments/car --output experiments/car/report
```

## 运行主方法与消融

```powershell
python -m stereo_research.run --manifest experiments/car/manifest.json --methods sgbm_fixed sgbm_flow local_fixed local_flow full full_quality --repeats 5 --warmup-frames 30
python -m stereo_research.run --manifest experiments/car/manifest.json --ablation-suite --repeats 5 --warmup-frames 30
python -m stereo_research.evaluate --experiment experiments/car/results/ablations --output experiments/car/report_ablations
```

消融文件写入 `results/ablations/`，不会覆盖六组主实验结果。该目录中的 `full` 是最终完整质量方法，`full_no_*` 与它复用同一清单、测点、帧范围、深度范围、动态归一化代价、有效性定义和评价脚本。
