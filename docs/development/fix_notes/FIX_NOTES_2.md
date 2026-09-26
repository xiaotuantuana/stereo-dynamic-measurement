# 时序双目测量代码第二轮修正说明

## 1. 修改范围

本轮只修改 `stereo_research/` 研究代码及其测试，没有修改原始 GUI、DMT、BM、点云程序，也没有引入深度学习。

主要修改文件：

- `stereo_research/models.py`
- `stereo_research/pipeline.py`
- `stereo_research/icgn.py`
- `stereo_research/local_matching.py`
- `stereo_research/uncertainty.py`
- `stereo_research/metrics.py`
- `stereo_research/report.py`
- `stereo_research/runner.py`
- `tests/test_adaptive_kalman.py`
- `tests/test_cycle_consistency.py`
- `tests/test_icgn.py`
- `tests/test_metrics.py`
- `tests/test_backward_compatibility.py`

## 2. 卡尔曼拒绝测量后的状态更新

`PointState` 现在明确区分三类量：原始测量、滤波估计、真正写入下一帧跟踪状态的量。

- `record_stereo_measurement()` 只记录已接受的原始测量，并维护 `last_accepted_measured_*`。
- `record_estimated_only()` 用于 `predict_only_outlier`，只推进估计位置、估计右点、估计视差和估计三维坐标。
- 被拒绝的原始视差不会进入 `disparity_history`，原始右点不会进入 `state.right_xy`，原始三维坐标不会覆盖最近有效原始状态。
- `research_full` 的下一帧视差预测明确使用 `estimated_disparity_history`。
- 原始异常值仍完整保留在 `FramePointResult` 和 CSV 的 `measured_*` 字段中。
- 连续预测达到 `kalman_max_predict_only_frames` 后，点进入 `recovering`，不允许无限依靠预测保持表面有效。

新增状态字段包括 `latest_disparity`、`last_accepted_measured_right_xy`、`last_accepted_measured_disparity`、`last_estimated_left_xy`、`last_estimated_right_xy` 和 `last_estimated_disparity`。

## 3. 右图参考图像、坐标与帧号同步

右图时间参考现在由以下三项共同定义：

- `right_flow_reference_gray[point_id]`
- `PointState.right_flow_reference_xy`
- `PointState.right_flow_reference_frame`

所有更新只允许通过 `_update_right_flow_reference()` 完成。该函数复制当前右图并在同一次调用中写入同帧坐标和帧号。`record_stereo_measurement()` 不再隐式修改右图参考坐标。

右图 LK 调用前通过 `_right_flow_reference()` 检查三项是否齐全，并强制要求 `reference_frame < current_frame`。同帧或未来帧参考返回 `invalid_right_reference`，随后退化为历史视差预测。

参考更新策略：

- 正常接受测量：使用 `measured_right_xy`，来源为 `measured`；
- 卡尔曼拒绝原始测量：使用有限且在图像内的 `estimated_right_xy`，来源为 `estimated`；
- 立体匹配失败：保留旧参考，不更新；
- 闭环恢复成功：使用恢复后的最终立体测量点。

## 4. IC-GN 真收敛与回退

`refine_disparity_icgn()` 现在只执行 IC-GN 迭代，不再执行原来的 97 点密集扫描，也不再把回退结果伪装为收敛。

只有同时满足以下条件才返回 `icgn_converged`：

1. 最终增量 `abs(delta) < icgn_epsilon`；
2. 残差不超过 `icgn_max_residual`；
3. 总偏移不超过 `icgn_max_offset_px`；
4. 数值有限且未触发发散、低梯度或越界。

IC-GN 迭代状态包括 `icgn_converged`、`icgn_max_iterations`、`icgn_diverged`、`icgn_out_of_bounds`、`icgn_low_gradient` 和 `icgn_high_residual`。

回退由 `local_matching.py` 显式执行。默认 `line_search` 在迭代最佳点附近 `±0.6 px` 采样 13 点，并在内部最优点处进行三点抛物线细化。另保留 `parabolic`、`continuous` 和 `integer` 兼容选项。回退成功只改变最终亚像素结果，不改变 `icgn_converged=False`。

为避免 OpenCV `remap` 插值表的坐标量化造成小幅极限环，IC-GN patch 采样改用连续双线性 `getRectSubPix`；这属于数值采样修正，不改变整体技术路线。

## 5. 真正的 ZNCC 代价曲率

IC-GN Hessian 与匹配代价曲率已经分离：

- `icgn_hessian`：IC-GN 模板梯度 Hessian；
- `icgn_hessian_density`：最终迭代 Jacobian Hessian 除以 patch 像素数；
- `zncc_cost_curvature`：最终视差附近真实 ZNCC 光度代价二阶差分。

曲率在最终视差 `d*` 附近按默认 `delta=0.10 px` 计算：

```text
(C(d* - delta) - 2 C(d*) + C(d* + delta)) / delta^2
```

非有限值输出为空；非正值截到 `curvature_min_value`；正值截到 `[1e-6, 1e4]`。兼容字段 `cost_curvature` 和 `icgn_cost_curvature` 现在均等于真实 `zncc_cost_curvature`，不再保存 Hessian 密度。

不确定性模块将 IC-GN 残差、IC-GN Hessian 密度和 ZNCC 曲率拆成三个风险项。保留旧 `uncertainty_weight_icgn`，并新增 `uncertainty_weight_icgn_hessian`、`uncertainty_weight_curvature` 和 `uncertainty_curvature_reference`。

## 6. 评价指标新定义

- 闭环失败率：分母为具有真实 `cycle_status` 的尝试行数，每行只计一次。
- 相机补偿：未启用或未尝试时 `camera_compensation_available=False`、尝试数为 0、成功率为 `None`，不再显示为 0%。
- 视差时间标准差：先按 `(repeat, point_id)` 分组，每组至少两个有效样本，再汇总各点标准差的均值、中位数和 P95。
- 静态参考点波动：按参考点分别对原始/补偿后三维坐标求相对时间质心的三维 RMS 波动，单位 mm。
- 静态漂移：按参考点分别计算 `norm(xyz_last - xyz_first)`，再汇总均值、中位数和 P95。
- 兼容字段 `measured_disparity_std_px`、`estimated_disparity_std_px`、`raw_static_point_std_mm`、`compensated_static_point_std_mm`、`raw_drift_mm` 和 `compensated_drift_mm` 现在对应各点结果的均值。
- 报告绘图不再把 `None` 画成 0；未启用闭环或补偿的方法不会进入对应比例图。

## 7. 新增 CSV 字段

原 CSV 字段未删除、未改名。新增：

```text
measurement_accepted_for_state
state_update_source
kalman_predict_only_frames
right_flow_reference_frame
right_reference_source
right_reference_age_frames
icgn_termination_reason
icgn_fallback_used
icgn_fallback_method
icgn_iterative_disparity
icgn_final_increment_px
subpixel_final_status
icgn_hessian_density
zncc_cost_curvature
curvature_sample_step_px
```

## 8. 新增测试

新增或加强了以下测试：

- 卡尔曼拒绝的 25 px 异常视差保留在结果中，但不进入状态和有效测量历史；
- 右图参考图像、坐标、帧号原子同步；
- 右图 LK 失败但立体匹配成功时，三项参考共同刷新；
- 卡尔曼拒绝的原始右点不能成为下一帧右图参考；
- 同帧右图参考被识别为 `invalid_right_reference`；
- IC-GN 正常收敛、单次迭代未收敛、低纹理、错误初值和回退语义；
- 丰富纹理与模糊纹理的真实曲率对比、非有限/负/过大曲率处理；
- 低曲率产生更大的视差测量方差；
- 闭环失败分母不重复计数；
- 未启用补偿返回 `None`；
- 多测点视差标准差和静态三维波动按点分组；
- 新 CSV 字段存在且旧字段顺序前缀保持不变。

最终完整测试：`135 passed`。

## 9. car.avi 回归结果

数据段：`car.avi` 第 250–350 帧，3 个测点，共 303 个点帧；单线程、预热 30 帧、重复 1 次。该视频没有真值，以下只用于流程、稳定性和运行回归。

| 方法 | 有效点帧 | 覆盖率 | 跳点率 | 中位运行时间 |
|---|---:|---:|---:|---:|
| SGBM | 303/303 | 100.00% | 5.72% | 约 89 ms |
| local | 41/303 | 13.53% | 0.00% | 约 6 ms |
| local_flow | 40/303 | 13.20% | 0.00% | 约 5 ms |
| full | 267/303 | 88.12% | 10.98% | 约 28 ms |
| full_quality | 298/303 | 98.35% | 10.21% | 约 30 ms |
| research_full | 302/303 | 99.67% | 0.34% | 约 74 ms |

旧五种方法除耗时字段外的旧 CSV 内容 SHA-256 与修改前逐项完全一致，说明本轮没有改变旧方法算法结果。`research_full` 连续两次运行的非耗时字段 SHA-256 均为：

```text
a6d57f5b0266f642b62e5ec01ed58ce32decc5e93f9a11b7588cfab8d9e85e8d
```

本段 `research_full` 的 299 次 IC-GN 尝试均为 `icgn_high_residual`，随后明确使用 `line_search` 回退；真实 IC-GN 收敛数为 0，且没有任何“回退成功却标成 IC-GN 收敛”的记录。299 个有效局部结果都输出了有限 ZNCC 曲率，`cost_curvature == zncc_cost_curvature`。所有有效参考输出的参考帧号都与最终更新帧一致。

## 10. 尚未解决及实验限制

- `car.avi` 没有左右像素、XYZ 或距离真值，不能用它证明三维精度、错误匹配率或论文显著性结论。
- 该片段没有触发卡尔曼异常拒绝；拒绝状态安全由可控合成测试验证，仍应在后续人为异常或遮挡实验中复核。
- 当前 3 个点均不足以形成可靠的静态参考点相机补偿实验，因此补偿尝试为不可成功状态；需要至少 3 个非共线静态参考点并增加冗余点。
- 本视频上的 IC-GN 光度残差均超过默认阈值，最终质量来自显式轻量回退。论文中必须分别报告 IC-GN 真收敛率和回退率，不能把二者合并为 IC-GN 成功率。
- 最终精度结论仍需使用后续静态已知距离/XYZ 数据和动态人工左右对应点标注，按既定 bootstrap 协议评价。
