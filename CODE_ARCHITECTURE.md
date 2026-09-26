# 项目真实架构与数据流

## 审计范围与证据边界

审计主体是本目录的 Git 仓库；`archive/` 是历史备份，不作为当前实现。另一个用户给出的 E 盘目录是独立的单目模板匹配脚本，未被本仓库导入，故不并入本项目的双目能力结论。

当前工作树已有大量未提交的用户改动。本报告只读审计，未改动源码。直接 `pytest -q` 的首次收集因本地包不在导入路径而失败；以临时 `PYTHONPATH=.` 启动的套件已显示至 94% 以上且未显示断言失败，但本次终端会话没有取得 pytest 的最终汇总行。因此，测试状态记为 **INCONCLUSIVE**，不能表述为“全量测试已通过”。

## 这个项目实际上是什么

这是一个 Python/OpenCV 的**研究型双目动态三维测量原型**，不是已经完成真实工业部署闭环的产品。它以并排双目视频和人工给定的初始左图点为输入；对图像校正、初始化视差、逐帧跟踪、局部对应、三维重投影、置信度/不确定度估计、可选相机公共运动补偿、CSV/图表输出进行编排。实际主链位于 `stereo_research/`；`stereo_dynamic_measurement/innovation1-3/` 提供可复用的研究模块或影子分析，部分并不获准写入最终结果。

| 问题 | 源码支持的回答 |
|---|---|
| 最终输入 | 并排双目视频、标定 JSON（或内置 640×480 标定）、初始帧左图人工点及可选 measurement/reference 角色。证据：`stereo_research/run.py`、`runner.py:split_side_by_side`、`annotate.py:annotate_video`。 |
| 最终输出 | 每帧每点的 CSV 行，含左右坐标、视差、XYZ(m)、置信度、失败/恢复状态、计时和诊断字段；另有汇总与图表。证据：`models.py:FramePointResult.as_csv_row`、`report.py`。 |
| 工程问题 | 由双目图像恢复目标点的相对三维位置/位移时程，并拒绝或标注低质量量测；不等价于已证明的现场结构监测精度。 |
| 单目部分 | 左图 LK 光流仅用于时间传播与预测左点，不是独立完成毫米尺度单目测量的链路。证据：`tracking.py:track_point_lk`、`pipeline.py:step`。 |
| 双目部分 | SGBM 用于初始化/某些模式的稠密视差；逐点局部匹配、LR/循环检查后用 Q 重投影 XYZ。 |
| 静态/动态 | 静态样本可独立初始化；动态链路维持 `PointState`、LK、预测、Kalman 与恢复。Phase 5B 明确声明静态源没有校准/深度 GT，不能宣称 3D 或时程指标。 |

## 当前目录结构

```text
stereo_research/                 当前主运行时：标定、匹配、状态、评价、GUI
  calibration.py                 读取 K/D/R/T，rectify/remap，生成 Q
  pipeline.py                    TemporalStereoPipeline 主编排
  global_matching.py             StereoSGBM 初始化/可选逐帧稠密匹配
  tracking.py                    Pyramid LK 与 forward-backward 检查
  local_matching.py              ZNCC 局部搜索、LR、连续/IC-GN 亚像素
  geometry.py                    Q 重投影（米）
  filtering.py                   自适应 Kalman
  camera_compensation.py         多参考点刚体/RANSAC 补偿
  enhanced_processor.py          I2 影子状态处理
stereo_dynamic_measurement/      研究模块、仿真和 I1/I2/I3
experiment/                      数据清单、受控场景、评价和报告
configs/                         标定与受控仿真配置
tests/                           单元/集成/研究门禁测试
experiments/real_gt/             真实 E0/E1 的模板与只读预检；当前无真实数据
results/, outputs/               已产生的候选结果/审计物，不自动等于正式结论
```

## 真实主数据流

```text
并排双目视频 + 标定JSON + 初始左图人工点
  -> split_side_by_side
  -> stereoRectify + initUndistortRectifyMap + remap
  -> 灰度化
  -> 初始：StereoSGBM视差 + 左右一致性 + 初始局部质量诊断
  -> 每帧：左图 Pyramid LK(FB) 传播点 / （可选）右图 LK
  -> 局部水平邻域 ZNCC 搜索（预测、极线、邻域约束）
  -> 连续线搜索或 IC-GN 亚像素 refinement
  -> LR + right-flow/stereo cycle + 置信度/不确定度/深度范围门控
  -> Q 对 (uL,vL,d) 重投影为左校正相机坐标系 XYZ(m)
  -> （可选）多参考点 RANSAC 刚体公共运动补偿
  -> Kalman 状态更新、失败计数和重捕获
  -> 原始/候选/最终来源字段的 CSV、汇总、图表
```

代码未实现的步骤不得硬说已实现：没有从棋盘格图像自动执行 `calibrateCamera/stereoCalibrate` 的主程序；没有在运行时自动检测 ArUco/结构目标；没有通用的全局 Homography 透视补偿；没有真实 E0/E1 的独立仪器 GT 数据。

## 坐标、单位与位移

`calibration.py:StereoCalibration._rectification` 由 `stereoRectify` 产生 Q，并将 `unit=mm` 的基线换成 m。 `geometry.py:reproject_point_m` 对 `(u_L,v_L,d)` 调用 `cv2.perspectiveTransform`，输出**左侧校正相机坐标系**的米制 XYZ；未发现到结构世界坐标系的外部坐标变换。

`FramePointResult` 保存的是绝对 XYZ。相对位移须在评价/绘图时以同一点参考帧 `XYZ(t)-XYZ(0)` 构造；主运行时不是“逐帧差分再累积”。若启用参考点角色，`camera_compensation.py:estimate_camera_compensation` 以参考点当前/基准 XYZ 拟合刚体变换并补偿公共相机运动，但它要求至少 3 个不共线参考点，且只能称为可选实现，不能称为所有运行都启用的事实。

## 模块成熟度

| 模块 | 状态 | 说明 |
|---|---|---|
| 标定参数读取、立体校正与 remap | 已实现 | 有 K/D/R/T JSON，`rectify_pair` 真正调用 remap。 |
| 棋盘格采集、质量筛片、重投影误差与标定求解 | 未发现实现 | 参数已存在不等于本仓库完成了标定实验。 |
| 逐点双目动态跟踪 | 已实现（研究原型） | LK + 局部匹配 + 状态/恢复均有代码。 |
| 亚像素匹配 | 已实现 | 连续采样/抛物线/IC-GN 路径均存在；具体激活取决于 method/config。 |
| 物理置信度、轨迹校正、故障恢复 | 部分实现/影子模式 | 默认禁止 I2/I3 写最终值：`MatcherConfig.__post_init__` 限制相关开关。 |
| 真实精度验证 | 未完成 | `experiments/real_gt/README.md` 明确 `REAL_DATA_REQUIRED`。 |
| GUI/工业相机接入 | 部分实现 | 有 PySide GUI 和视频输入；未发现可证明工业相机 SDK、同步触发、在线部署监控的实现。 |

## 关键参数（默认值，不等于经实验最优）

`models.py:MatcherConfig`：LK `winSize=21,maxLevel=3`（恢复 41/4）、FB=1 px、局部 patch=11、水平搜索半径 8（恢复 32）、垂直半径 1、LR=1 px、SGBM 64 disparities/3×3 block/uniqueness=15、IC-GN 15×15/20 迭代/残差阈值 0.35、cycle 0.5/1.5/3 px、相机补偿内点阈值 3 mm。这些在代码中有合法性检查，但未看到对真实工况逐项寻优的证据。

