# 主程序、GUI 与创新点集成调用链审计

审计日期：2026-08-25  
范围：静态只读代码审计。未修改算法、GUI、import、配置、参数、测试、目录或输出。

## 1. Executive Summary

三个创新点均有代码、测试和实验入口，但没有形成“创新点 1 → 创新点 2 → 创新点 3 接管最终结果”的完整在线链路。

| 创新点 | 主程序中的真实状态 | 直接结论 |
| --- | --- | --- |
| 创新点一 | **部分接入** | 主链路复用 `precision_planner` 的精度规划；完整 `innovation1.measurement_pipeline` 未被 GUI 或主测量流程实例化。 |
| 创新点二 | **部分接入（shadow）** | GUI 主处理链调用物理置信度、transient gate 和候选轨迹校正，但候选值不写入 `final_*`。 |
| 创新点三 | **部分接入（shadow）** | GUI 主处理链调用指纹、规则诊断和恢复建议，但不执行恢复、不反馈最终结果。 |

因此：三个创新点不是仅有离线模块；创新点二、三已作为运行时只读观测器进入 GUI 批处理主链路。但它们尚未成为正式测量输出的决策者或执行器。

## 2. 两个包的真实职责

### `stereo_research/`

这是当前实际软件和视频双目测量主系统：

- GUI：`gui/app.py`、`gui/window.py`、`gui/processing.py`；
- 视频输入、左右拆分与标定校正：`gui/processing.py`、`runner.py::split_side_by_side`、`calibration.py::StereoCalibration.rectify_pair`；
- 主测量：`pipeline.py::TemporalStereoPipeline`，内部使用 `GlobalStereoMatcher`、`LocalMatcher`/`QualityLocalMatcher`、LK 跟踪、预测、左右一致性、亚像素、三角重投影与滤波；
- 3D 结果与相机运动补偿：`pipeline.py::_apply_camera_compensation`；
- CSV 保存、离线评价、真实实验预检与报告：`runner.py`、`evaluation_runner.py`、`real_experiment.py`、`report.py`。

### `stereo_dynamic_measurement/`

这是创新点、仿真、验证与诊断实验库：

- `calibration/`：独立的相机模型和三角测量；
- `simulation/`：合成场景、轨迹和故障注入；
- `innovation1/`：自适应局部测量管线、光流、预测、亚像素、置信度与精度规划；
- `innovation2/`：物理证据、谱/相位/相干分析、瞬态保护与候选轨迹校正；
- `innovation3/`：故障指纹、规则诊断、参考监测、恢复建议和验证；
- `pipeline/orchestrator.py`：面向独立 `MeasurementResult`/验证回调的诊断恢复编排器；
- `run_*` 与 `benchmark_faults.py`：独立实验、仿真、validation 和 benchmark 入口。

## 3. 程序入口分类

| 类别 | 入口 | 实际作用 |
| --- | --- | --- |
| GUI/软件 | `启动双目测量软件.bat` → `python -m stereo_research.gui` → `gui/__main__.py` → `gui/app.py::main` | 创建 `StereoMainWindow`，处理左右并排视频。 |
| GUI 批处理 | `StereoMainWindow.start_processing` / `start_thesis_experiment` | 创建 `ProcessingWorker`/`BatchProcessingWorker`。 |
| 主离线 CLI | `stereo_research/run.py`、`stereo_research/evaluate.py` | 视频测量与评估。 |
| 仿真 | `run_simulation.py` → `stereo_dynamic_measurement.run_simulation::main` | 合成双目数据和图表。 |
| 创新点一实验 | `run_innovation1_experiment.py` → `run_baseline_distance_experiment` | M0–M3/精度策略实验，不是 GUI 主处理器。 |
| 创新点二验证 | `run_innovation2_validation.py` → `run_physics_validation` | 合成轨迹物理/频域验证。 |
| 创新点三 benchmark | `benchmark_faults.py` → `run_fault_benchmark` | 合成故障分类与恢复评估。 |
| 真实实验接口 | `validate_real_experiment.py`、`evaluate_e0.py`、`evaluate_e1.py`、`run_synthetic_dry_run.py` | 清单校验和离线 E0/E1 评价。 |
| 自动测试 | `tests/test_*.py` | 单元、集成、GUI 非交互和 shadow 安全性验证；不代表 GUI 已接管创新输出。 |

## 4. GUI 真实调用链

```text
启动双目测量软件.bat
  → python -m stereo_research.gui
  → stereo_research/gui/__main__.py
  → gui/app.py::main()
  → StereoMainWindow()
  → StereoMainWindow.start_processing()
  → StereoMainWindow.build_processing_request()
  → ProcessingWorker.run()
  → StereoVideoProcessor.run()
  → TemporalStereoPipeline.initialize(...) / step(...)
  → FramePreview
  → StereoMainWindow._on_processing_frame(...)
```

`StereoMainWindow` 通过 `cv2.VideoCapture` 读取用户选择的左右并排视频；它不是相机实时采集循环。预览和处理均先调用 `split_side_by_side`，再调用 `StereoCalibration.rectify_pair`。

## 5. 实际双目测量调用链

```text
左右并排视频
  → cv2.VideoCapture（gui/processing.py::StereoVideoProcessor.run）
  → runner.py::split_side_by_side
  → StereoCalibration.rectify_pair
  → TemporalStereoPipeline.initialize / step
      → 全局 SGBM 初始化或恢复（GlobalStereoMatcher）
      → LK 跟踪、历史预测、局部匹配/左右检查/亚像素（按 MethodProfile）
      → geometry.py::reproject_point_m（三角重投影为 3D）
      → 置信度、Kalman/闭环状态及可选相机运动补偿
      → _finalize_frame_results
  → CSV：FramePointResult.as_csv_row
  → GUI：左右图、视差图、测量渲染图，及帧号/有效点数/平均深度/耗时/方法
```

主流程未发现独立“实际相机设备采集”模块；当前 GUI 的实际输入是视频文件。GUI 没有控件显示 `c_phy`、`fault_class`、`recommended_recovery` 或候选校正 XYZ；这些被写入 CSV 字段。

## 6. Innovation 1 调用链

### 实现与实验

`innovation1.measurement_pipeline::AdaptiveStereoMeasurementPipeline` 包含其独立相机模型、运动预测、LK-FB、局部匹配、亚像素和置信度路径。它由创新点一测试覆盖，但全工程没有 GUI/主流程对该类的实例化。

独立入口链为：

```text
run_innovation1_experiment.py
  → stereo_dynamic_measurement/run_innovation1_experiment.py::main
  → innovation1/baseline_experiment.py::run_baseline_distance_experiment
```

### 主链路接点

`stereo_research/pipeline.py` 和 `stereo_research/accuracy_policy.py` 导入：

```text
innovation1.precision_planner::{TargetAccuracySpec, build_precision_plan}
```

当且仅当 `MatcherConfig.enable_target_accuracy_policy=True` 时，`TemporalStereoPipeline` 的 `_pre_match_accuracy_policy`、`_match_accuracy_policy` 与 `_annotate_accuracy_policy` 会调用该规划器和 `decide_measurement_policy`，影响搜索半径/重试决策及结果标注。该开关默认值为 `False`。

**状态：B，部分接入。** 接入的是精度规划子模块；`AdaptiveStereoMeasurementPipeline`、其 `MotionPredictor`、`track_forward_backward`、`LocalStereoMatcher` 等 `stereo_dynamic_measurement.innovation1` 的完整实现未进入 GUI 主调用链。GUI 的 M3 名称对应 `stereo_research.models::MethodProfile` 内部实现，而不是对该独立类的调用。

## 7. Innovation 2 调用链

`TemporalStereoPipeline.__init__` 在 `enable_physics_shadow` 或 `enable_fault_shadow` 为真时创建 `ShadowAnalyzer`；两项默认值均为 `True`。每帧主测量完成后：

```text
TemporalStereoPipeline._finalize_frame_results
  → _apply_camera_compensation
  → ShadowAnalyzer.process_frame
      → innovation2.physics_confidence::compute_physics_confidence
      → innovation2.transient_gate::decide_transient
      → innovation2.trajectory_corrector::correct_trajectory_point
```

`ShadowAnalyzer` 仅将输出写为 `candidate_corrected_*`、`c_phy`、残差与瞬态保护字段。`MatcherConfig.__post_init__` 明确拒绝 `allow_physics_correction_to_final=True`，因此创新点二的候选校正不能修改 `final_x_m/final_y_m/final_z_m`。

独立 `run_physics_validation` 会运行包含 PSD、phase、coherence 的合成验证；而 GUI shadow 中将 spectral/phase/coherence 标为 runtime history 不足，未提供这些正式频域证据。

**状态：B，部分接入（shadow mode）。** 它被 GUI 主链调用，但未接管正式轨迹或最终 3D 输出。

## 8. Innovation 3 调用链

同一个 `ShadowAnalyzer.process_frame` 在得到创新点二的物理置信度后：

```text
FramePointResult + C_phy/残差
  → innovation3.fault_fingerprint::FaultFingerprint
  → innovation3.fault_classifier::RuleDiagnosticEngine.diagnose
  → innovation3.recovery_manager::RecoveryManager.plan
  → fault_class / fault_confidence / recommended_recovery（CSV 字段）
```

没有 GUI 信号、控制器回调或 `TemporalStereoPipeline` 分支执行 `recommended_recovery`。`allow_fault_recovery_to_final=True` 同样会被 `MatcherConfig` 明确拒绝。独立 `benchmark_faults.py` 则经 `StereoMeasurementOrchestrator.process` 在合成数据上评估诊断/恢复。

**状态：B，部分接入（shadow mode）。** 它不是当前在线测量结果的恢复执行器，而是每帧运行的诊断建议器。

## 9. 三个创新点之间的连接关系

| 关系 | 真实状态 | 证据 |
| --- | --- | --- |
| 创新点一 → 创新点二 | 间接且不构成专用数据接口 | 主管线先产生 `FramePointResult`；创新点一精度策略若启用，仅标注/调整匹配决策；随后 shadow 从完成结果读取 XYZ。没有 `innovation1` 对 `innovation2` 的直接 import/call。 |
| 创新点二 → 创新点三 | 直接，但仅 shadow | `ShadowAnalyzer.process_frame` 先计算 `physics`，再用 `physics.r_phy` 等构造 `FaultFingerprint` 并调用诊断和恢复计划。 |
| 创新点三 → 主输出 | 无执行反馈 | 只产生 `recommended_recovery`；未调用重新测量、未修改最终 XYZ。 |

## 10. GUI 是否使用三个创新点

| 创新点 | 代码存在 | 自动测试 | 独立实验/验证 | GUI 调用 | 主测量链路 |
| --- | ---: | ---: | ---: | ---: | --- |
| 创新点一 | √ | √ | √ | 部分 | 部分：仅精度规划，且默认关闭 |
| 创新点二 | √ | √ | √ | 部分 | Shadow：候选校正不写入最终输出 |
| 创新点三 | √ | √ | √ | 部分 | Shadow：诊断建议不执行 |

## 11. Shadow mode 情况

Shadow mode 在这里不是推断，而是代码契约：

- `MatcherConfig` 默认启用 physics/fault shadow；
- `ShadowAnalyzer` 复制完成的 `FramePointResult` 并维护独立历史；
- `allow_physics_correction_to_final` 与 `allow_fault_recovery_to_final` 为真会抛出 `NotImplementedError`；
- `tests/test_shadow_state_isolation.py` 验证开关 shadow 后旧有最终结果和核心状态保持不变。

因此，shadow 计算、CSV 记录和 GUI 主处理同时发生，但 **Shadow mode 不等于正式接管主链路**。

## 12. 当前真实架构图

```text
GUI（StereoMainWindow）
  → ProcessingWorker / StereoVideoProcessor
  → stereo_research::TemporalStereoPipeline
      → 基础/论文方法：SGBM、LK、预测、局部匹配、亚像素、重投影、滤波
      → 可选：创新点一 precision_planner（目标精度策略）
      → 可选：research 内部相机运动补偿
      → 最终 XYZ / CSV / GUI 平均深度
      → ShadowAnalyzer（默认运行，但不改最终结果）
          → 创新点二：C_phy、transient、candidate correction
          → 创新点三：fingerprint、diagnosis、recovery recommendation

独立实验系统
  → dynamic::innovation1 baseline experiment
  → dynamic::innovation2 physics validation（PSD/phase/coherence）
  → dynamic::innovation3 fault benchmark / orchestrator
```

## 13. 论文目标架构图（建议，不是当前代码事实）

```text
视频/相机 → 标定校正 → 创新点一正式匹配与不确定度输出
  → 三角测量/动态轨迹 → 创新点二正式质量门控或校正
  → 创新点三在线诊断与受控恢复 → 可追溯最终输出 → GUI
```

## 14. 当前与目标的差距清单

| 优先级 | 差距 |
| --- | --- |
| P0 | 未定义创新点二候选校正、创新点三恢复建议何时可安全写入正式最终结果的验证门槛与责任边界。 |
| P0 | `stereo_dynamic_measurement.innovation1.measurement_pipeline` 与 GUI 主 `TemporalStereoPipeline` 是两条不同实现路径；尚未定义统一接口或明确论文中哪条是正式实现。 |
| P1 | GUI 未展示 C_phy、故障类别、恢复建议及其“shadow/非最终”状态，用户无法区分候选与正式输出。 |
| P1 | GUI shadow 未接入创新点二验证中的 PSD/phase/coherence 完整证据链。 |
| P2 | `stereo_dynamic_measurement` 的独立 camera model/triangulation 与 `stereo_research` 的标定/重投影并存，需要后续论文层面明确职责和一致性。 |

## 15. 后续集成建议（不实施）

1. 先确定论文正式主测量实现：继续以 `TemporalStereoPipeline` 为基线，或为创新点一设计明确适配接口；不要仅以模块名称判断。
2. 为创新点二建立从候选结果到正式输出的离线验收标准，再决定是否解除 shadow 写保护。
3. 为创新点三限定可自动执行的恢复动作、回退策略和审计记录，再连接到主流程。
4. 在上述数据契约和验证完成前，保持当前 shadow 模式，避免候选校正影响科研结果可追溯性。

## 附：包依赖判断

`stereo_research → stereo_dynamic_measurement` 是当前运行时的主要方向：它导入创新点一精度规划器以及创新点二、三的 shadow 组件。反向依赖仅见于 `innovation1/baseline_experiment.py` 对 `stereo_research.accuracy_policy` 的实验调用。因此整体为 **双向依赖（C）**，但主 GUI 的生产调用方向是 `stereo_research → stereo_dynamic_measurement`。
