# 双目时序测量论文版测试报告

测试日期：2026-08-01  
测试环境：Windows、Python 3.14、OpenCV CPU 单线程

## 1. 自动化验证

执行：

```powershell
python -m pytest -q
python -m compileall -q stereo_research tests
git diff --check
```

结果：`76 passed`；Python 编译检查通过；Git 差异检查无空白错误。测试覆盖：

- 无效深度/运动候选不污染原始和估计视差历史、右点、XYZ 与置信度；
- LR 只验证原始图像测量，时间估计在三维与物理检查后执行；
- 动态归一化代价在约束关闭后保持同一尺度；
- 上下文大运动由独立金字塔证据验证，`full_no_pyramid` 不调用金字塔；
- 三级金字塔恢复超出单尺度 ±16 px 的视差；
- 连续亚像素输出整数视差、原始亚像素视差与偏移量；
- recovering/lost 周期恢复、SGBM 重新初始化和恢复统计；
- 人工像素真值 Q 重投影、毫米级指标和多测点稳定性隔离；
- 六组主方法、十组消融、报告与命令行集成。

## 2. `car.avi` 流程验证

范围：250–350 帧，3 个测点，101 帧，共 303 个点帧；1 次重复，前 30 帧不计时。

| 方法 | 有效点帧 | 覆盖率 | 中位耗时/帧 | 恢复尝试/成功 |
|---|---:|---:|---:|---:|
| `sgbm_fixed` | 303 | 100.00% | 87.13 ms | 0 / 0 |
| `sgbm_flow` | 303 | 100.00% | 94.68 ms | 0 / 0 |
| `local_fixed` | 41 | 13.53% | 5.22 ms | 0 / 0 |
| `local_flow` | 40 | 13.20% | 4.99 ms | 0 / 0 |
| `full` | 267 | 88.12% | 25.56 ms | 0 / 0 |
| `full_quality` | 298 | 98.35% | 30.72 ms | 5 / 4 |

该结果证明主流程、多尺度恢复和统计输出可运行；`full_quality` 在本片段的覆盖率高于基础 `full`。它不证明毫米级精度：本视频没有人工左右对应点或独立位移真值，报告中的 XYZ、距离、错误匹配率和位移误差真值指标均明确为不可用。

1 mm 跳变率在该动态车载片段上较高，且不同方法覆盖帧不相同，不能单独解释为精度优劣。必须使用静态标靶和已知三维位移实验评价亚像素精度与振动保持能力。

## 3. 消融流程验证

以下十组已在同一清单上运行并输出同构 CSV：

```text
full
full_no_flow
full_no_prediction
full_no_epipolar
full_no_neighborhood
full_no_subpixel
full_no_lr
full_no_pyramid
full_no_recovery
full_no_temporal_estimation
```

消融结果位于 `experiments/car/results/ablations/`，不会覆盖六组主实验；对应报告位于 `experiments/car/report_ablations/`。

## 4. 报告产物检查

主报告和消融报告均生成 `summary.json`、`summary.csv`，以及 PNG/PDF 两种格式的：

- 视差误差；
- ΔX、ΔY、ΔZ 轨迹；
- 原始与滤波位移；
- 运行时间箱线图；
- 覆盖率、错误匹配率和恢复率对比。

有三维真值时还会生成预测—真值与位移误差轨迹；无真值时不伪造误差数值。

## 5. 论文实验前仍需完成

1. 用 `annotate_gt` 每隔固定帧标注左右对应点，获得匹配算法三维真值；
2. 采集具有独立毫米级位移真值的静态和动态序列，用于包含标定误差的系统级验证；
3. 对连续亚像素与 `full_no_subpixel` 比较精度—稳定性权衡，不能仅根据 `car.avi` 平滑度选择；
4. 正式实验使用至少 3 段/场景、5 次计时重复和序列级配对 bootstrap。
