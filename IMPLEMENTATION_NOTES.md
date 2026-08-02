# `research_full` 实现说明

## 修改范围

本次升级只修改 `D:\桌面\大论文\daima`，没有修改原项目 `Stereo-Detection-main`。
整体路线仍为 SGBM 初始化、LK 跟踪、传统局部匹配和 Q 矩阵重投影；没有深度学习依赖。
旧方法 `sgbm`、`local`、`local_flow`、`full`、`full_quality` 保留，新增完整方法
`research_full`。新功能仅由新方法启用，旧 CSV 字段和顺序保持不变，新字段只追加。

## 新增算法模块

- `stereo_research/icgn.py`：一维水平 IC-GN。模板零均值标准差归一化，Sobel 梯度进行
  Hessian/纹理检查，右图三次插值采样，迭代限制在已验证整数视差附近。失败按配置回退。
- `stereo_research/cycle_consistency.py`：右图时间 LK 与当前立体匹配的闭环误差，采用
  “软接受—扩大搜索恢复—硬拒绝”三级策略。
- `stereo_research/uncertainty.py`：由纹理、外观代价、唯一性、左右一致性、双侧光流、
  闭环误差和 IC-GN 质量估计左点/视差测量方差。
- `stereo_research/filtering.py`：每点六状态 `[u,v,d,du,dv,dd]` 自适应卡尔曼滤波，
  支持 NIS 正常更新、测量方差放大和异常帧仅预测。
- `stereo_research/camera_compensation.py`：加权 Kabsch/SVD 和确定性 RANSAC，统一使用
  `P_reference = R @ P_current + t`。

## `research_full` 流程

1. 左图 LK 前后向跟踪；
2. 上一有效右点到当前右图的 LK 前后向跟踪；
3. 融合历史视差预测和右图时间预测；
4. Census+ZNCC 局部搜索，加入预测、极线、邻域和闭环软代价；
5. IC-GN 精化及抛物线/连续/整数回退；
6. 左右一致性和闭环三级判定；
7. 匹配不确定度估计与自适应卡尔曼；
8. 分别重投影 `measured_*` 和 `estimated_*`；
9. 整帧收集静态参考点并估计相机刚体运动；
10. 同时保存 raw、estimated、compensated 和 final 结果。

右图参考帧只在最终结果有效时更新。参考点不足、共线或 RANSAC 失败时，不使整帧失效，
而是记录失败原因、设置 `compensation_applied=False`，并令 `final_*` 回退为 `estimated_*`。

## 测点角色

旧 JSON 不含 `role` 时默认是测量点：

```json
{"id": "P1", "x": 100, "y": 200}
```

相机补偿实验应在静态背景上添加至少 4 个分散、非共线参考点：

```json
{"id": "R1", "x": 300, "y": 180, "role": "reference"}
```

被测结构点使用 `"role": "measurement"`。

## 配置与输出

新增配置全部集中在 `stereo_research/models.py` 的 `MatcherConfig`：闭环权重/三级阈值、
IC-GN 窗口/迭代/质量阈值、不确定度权重和方差边界、Kalman 过程噪声/NIS 阈值、
相机补偿内点阈值/比例/RANSAC 次数等，构造时统一校验范围。

新 CSV 字段追加在 `frame_total_ms` 后，包括右图时间点和闭环质量、实际唯一性与纹理、
IC-GN 过程量、不确定度与 Kalman 过程量、点角色与相机位姿、`compensated_*`、`final_*`、
`raw_delta_*` 和 `compensated_delta_*`。在 `research_full` 中旧 `delta_X/Y/Z_mm` 表示最终
补偿位移，原始位移可由 `raw_delta_*` 追溯。

## 运行

```powershell
cd "D:\桌面\大论文\daima"

# 完整方法
python -m stereo_research.run --manifest experiments/car/manifest.json --methods research_full --subpixel-method icgn --repeats 3

# 同清单对比
python -m stereo_research.run --manifest experiments/car/manifest.json --methods sgbm local local_flow full full_quality research_full --repeats 1
python -m stereo_research.evaluate --experiment experiments/car --output experiments/car/report_research

# 完整消融
python -m stereo_research.run --manifest experiments/car/manifest.json --ablation-suite --repeats 3

# 测试
python -m pytest -q
```

`car.avi` 仅用于流程、状态、稳定性和耗时冒烟测试。当前清单没有像素/三维真值，也没有
静态参考点，因此三维精度、错误匹配率和补偿效果不可作为论文定量结论，必须由后续带真值
和静态参考点的数据生成。
