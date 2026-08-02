# 时空预测驱动的局部双目匹配研究代码

论文升级版完整方法为 `research_full`。新增四视图闭环、IC-GN、自适应卡尔曼和静态参考点
相机补偿的说明与运行命令见 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)。

该目录对应论文的核心创新实验，和现有 Tkinter 深度相机程序相互独立。程序在校正后的多个人工测点上比较 `sgbm_fixed`、`sgbm_flow`、`local_fixed`、`local_flow`、`full`、`full_quality` 和 `research_full`。旧名称 `sgbm`、`local` 分别作为前两种固定左点方法的兼容别名。

## 环境

```powershell
python -m pip install -r requirements_research.txt
python -m pytest -q
```

运行时固定 OpenCV 单线程。CSV 中 `frame_total_ms` 是去除视频读取、显示和写文件后的整帧算法耗时；`flow_ms`、`matching_ms` 和 `total_ms` 是点级诊断耗时。

## 1. 选择第一帧测点

```powershell
python -m stereo_research.annotate --video car.avi --start-frame 250 --calibration configs/calibration_640x480.json --output experiments/car/points.json
```

在校正后的左图单击多个纹理清晰点，Backspace 撤销，Enter 保存。也可无界面生成：

```powershell
python -m stereo_research.annotate --video car.avi --start-frame 250 --output experiments/car/points.json --points "140,320;239,350;348,353"
```

## 2. 运行四组主实验

```powershell
python -m stereo_research.run --manifest experiments/car/manifest.json --methods sgbm_fixed local_fixed local_flow full full_quality
```

附录公平控制：

```powershell
python -m stereo_research.run --manifest experiments/car/manifest.json --methods sgbm_flow
```

完整机制删除实验：

```powershell
python -m stereo_research.run --manifest experiments/car/manifest.json --ablation-suite
python -m stereo_research.evaluate --experiment experiments/car/results/ablations --output experiments/car/report_ablations
```

消融 CSV 单独写入 `results/ablations/`，不会覆盖主实验的 `results/full.csv`。消融目录中的 `full` 表示包含质量控制、多尺度与恢复机制的最终完整方法。

质量优先版本：

```powershell
python -m stereo_research.run --manifest experiments/car/manifest.json --methods full full_quality --repeats 5 --warmup-frames 30
```

`full_quality` 以 11×11 单尺度匹配为主，主匹配失败时依次使用 15×15 上下文和三级金字塔恢复。上下文恢复产生大视差创新时，必须由独立金字塔粗层结果验证；不会再用恢复结果验证自身。左右一致性、图像代价和唯一性始终只检查 `measured_disparity`。原始观测通过三维范围和运动检查后，才以默认 0.2 权重生成 `estimated_disparity`；大创新保留原始运动量，避免削弱真实振动。CSV 同时保留整数、原始亚像素和时间估计视差及其三维坐标。

消融套件自动运行：`full`、`full_no_flow`、`full_no_prediction`、`full_no_epipolar`、`full_no_neighborhood`、`full_no_subpixel`、`full_no_lr`、`full_no_pyramid`、`full_no_recovery` 和 `full_no_temporal_estimation`。所有组合使用动态归一化融合代价和相同唯一性阈值。

正式计时使用 `--repeats 5 --warmup-frames 30`。前 30 帧仍参与初始化和状态更新，但不进入耗时统计；每次重复独立重新初始化，评价程序按 `repeat/frame` 去重统计整帧耗时，并导出相对 SGBM 加速比。

## 3. 标注真值与评价

```powershell
python -m stereo_research.annotate_gt --manifest experiments/car/manifest.json --interval 10 --output experiments/car/ground_truth.csv
python -m stereo_research.evaluate --experiment experiments/car --output experiments/car/report
```

评价结果包括像素跟踪/匹配误差、视差 MAE/RMSE、毫米级 XYZ 与相对位移误差、静态标准差/峰峰值/漂移、1/2/5/10 mm 跳变率、创新量、恢复成功率、覆盖率和运行时间。人工左右像素真值会用同一 Q 矩阵自动重投影；它用于评价跟踪与匹配误差，不能独立包含相机标定系统误差。

多个正式序列完成后，使用序列级配对 bootstrap 汇总置信区间：

```powershell
python -m stereo_research.aggregate --summaries experiments/seq1/report/summary.json experiments/seq2/report/summary.json experiments/seq3/report/summary.json --baseline local_flow --improved full --output experiments/aggregate.json
```

## 状态码

- `valid`：光流、局部唯一性、左右一致性和三维重建均有效。
- `flow_failed`：LK 前后向误差超限或跟踪失败。
- `low_texture`：测点邻域纹理不足。
- `out_of_bounds`：匹配块或预测窗口越界。
- `ambiguous`：最优与次优代价间隔不足。
- `lr_failed`：右到左反向匹配误差超过 1 px。
- `depth_out_of_range`：重建深度不在配置的有效范围内。
- `saturated`：初始化视差达到搜索上界。
- `motion_rejected`：候选三维运动超过配置的物理门限。
- `lost`：连续失败达到上限；`full_quality` 每隔配置帧数尝试有限重新捕获，其余方法不伪造输出。

`car.avi` 的第 0 帧明显过曝，因此示例从第 250 帧开始，只用于流程与稳定性冒烟，不作为论文准确度真值。
