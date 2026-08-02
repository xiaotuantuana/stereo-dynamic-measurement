# `research_full` 测试报告

测试目录：`D:\桌面\大论文\daima`

## 1. 自动化验证

执行命令：

```powershell
python -m pytest -q
python -m compileall -q stereo_research tests
python -m stereo_research.run --help
git diff --check
```

最终结果：`120 passed in 23.34s`；Python 全模块编译通过；命令行帮助正常；
`git diff --check` 无空白错误（仅有 Windows CRLF 转换提示）。

覆盖内容包括：旧方法与旧 CSV 前缀回归、标定/Q 重投影、SGBM 初始化、局部匹配、
左右一致性、LK 前后向检查、恢复状态、IC-GN 小数视差、闭环三级判定、不确定度估计、
静止/匀速/正弦/异常值卡尔曼、加权 Kabsch/RANSAC、相机补偿退化回退、指标和报告生成。

## 2. `car.avi` 冒烟对比

清单：`experiments/car/manifest.json`；帧范围 250–350；3 个测点；1 次运行；前 30 帧
不计运行时间。所有方法复用同一清单和测点。

| 方法 | 有效点帧/总点帧 | 覆盖率 | 轨迹跳点率 | 中位耗时/帧 |
|---|---:|---:|---:|---:|
| SGBM | 303/303 | 100.00% | 5.72% | 89.78 ms |
| 仅局部搜索 | 41/303 | 13.53% | 0.00%* | 6.27 ms |
| 局部搜索+光流 | 40/303 | 13.20% | 0.00%* | 5.70 ms |
| 完整旧方法 `full` | 267/303 | 88.12% | 10.98% | 27.61 ms |
| 质量基线 `full_quality` | 298/303 | 98.35% | 10.21% | 29.58 ms |
| 新方法 `research_full` | 299/303 | 98.68% | 3.86% | 686.34 ms |

`*` 两个简单局部方法只保留约 13% 输出，0% 跳点率不能脱离覆盖率单独解释。

`research_full` 的附加诊断：

- 四视图闭环样本 296 个，平均误差 0.136 px，p95 误差 0.552 px，闭环硬失败 0；
- 有效点帧 299 个；失败包括 3 个 `ambiguous` 和 1 个 `lr_failed`；
- Kalman：3 次初始化、289 次正常更新、7 次 NIS 方差放大；
- 原始三维恒速创新 RMSE 464.94 mm，滤波估计为 233.80 mm；
- 296 次 IC-GN 数值收敛均因归一化残差超过默认 0.35 阈值而回退，故“实际采用成功数”为 0；
- 当前测点文件没有 `reference` 点，303 个点帧均明确记录 `insufficient_references`，
  相机补偿没有被伪造启用。

耗时明显增加来自质量优先的 IC-GN 迭代、局部线搜索、右图 LK 和恢复验证，符合本轮
“先追求处理质量、不优先优化耗时”的要求。

## 3. 可得与不可得结论

本视频可验证：完整流程可运行、状态与新增字段可追溯、闭环误差较小、有效跟踪率高、
滤波后的时序创新下降、旧方法仍可运行。

本视频不能验证：三维坐标真实误差、错误匹配率、IC-GN 真值精度和相机补偿实际效果。
原因是清单没有人工左右点/XYZ 真值，也没有静态参考点。上述论文结论必须在后续带真值实验中给出。

## 4. 输出位置

- 每方法 CSV：`experiments/car/results/`
- 汇总 JSON/CSV：[summary.json](experiments/car/report_research/summary.json)、
  [summary.csv](experiments/car/report_research/summary.csv)
- 闭环、Kalman、补偿、轨迹、耗时和消融图：`experiments/car/report_research/`
