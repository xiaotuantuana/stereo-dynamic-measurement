# 双目视觉硕士论文代码工程审计报告

审计日期：2026-08-20  
审计范围：`<project-root>` 当前工作树  
审计模式：只评估；未修改任何源代码、测试代码、配置文件或实验脚本。

## 1. Overall Assessment

| 项目 | 评估 |
|---|---:|
| 项目整体完成度 | 35% |
| 基础 Stereo 可信度 | 65% |
| 创新点一完成度 | 45% |
| 创新点二完成度 | 20% |
| 创新点三完成度 | 25% |
| 实验体系完成度 | 25% |

当前是否适合进入硕士论文正式实验：**暂时不可以**。

主视频流程具备可运行的双目测量、LK/FB、局部匹配、亚像素、预测、置信度、重捕获与相机补偿能力；但论文定义的三创新闭环没有进入同一主流程，且 `car` 实验没有 GT，无法证明真实 XYZ 精度或论文结论。

## 2. Current Real Pipeline

```text
car.avi
  → split_side_by_side
  → StereoCalibration.rectify_pair
  → TemporalStereoPipeline.initialize / step
  → LK + FB（可选）
  → 历史视差预测
  → 自适应局部搜索 / SGBM 重捕获
  → LR、极线、邻域、IC-GN / 亚像素（按方法启用）
  → Q 重投影为米
  → Kalman 时间估计（research_full）
  → 多参考点刚体相机补偿（仅有足够 reference 点时）
  → CSV / 评估
```

| 模块 | 状态 | 说明 |
|---|---|---|
| `stereo_research` | INTEGRATED | 实际视频主流程。 |
| `stereo_dynamic_measurement.innovation1` | ISOLATED | 可独立运行和测试，但未被主流程调用。 |
| Innovation 2 的 `C_phy → corrected XYZ` | ISOLATED | 未进入视频主流程。 |
| Innovation 3 的 `Fingerprint → Diagnose → Recover → Verify` | ISOLATED | 未进入视频主流程。 |

主流程中没有 `PhysicsValidation`、`C_phy`、`FaultFingerprint`、`RuleDiagnosticEngine` 或 `RecoveryManager` 的引用。

关键入口：`stereo_research/run.py`、`stereo_research/runner.py`、`stereo_research/pipeline.py`。

## 3. Stereo Geometry Audit

| Item | Status | Evidence / Risk |
|---|---|---|
| K_L / K_R / D_L / D_R / R / T | PASS | `calibration.py` 以 mm 保存 T，并校验矩阵形状。 |
| Rectification / P1 / P2 / Q | PASS | `cv2.stereoRectify(..., CALIB_ZERO_DISPARITY)` 生成 Q 和 remap。 |
| disparity sign | PASS | 主流程要求 `disparity > 0`，符合该标定 `T_x < 0` 和 `d=xL-xR>0` 的 OpenCV 约定。 |
| Q 单位 | PASS | mm 标定的 Q 重投影后显式 `/1000`，输出 m。 |
| 亚像素进入 XYZ | PASS | 未发现取整后重建；最终重建使用浮点视差。 |
| absolute XYZ / displacement | PARTIAL | 输出区分 measured、estimated、displacement，但真实序列无 GT。 |
| 实物标定健康度 | PARTIAL | 有极线、竖直视差和参考点补偿模块；`car` 没有 reference 点。 |
| 可变 baseline | FAIL | 只有离线蒙特卡洛扫描；没有运行时同步更新 T、P1/P2、rectification、Q 的接口。 |

未发现“仅修改 baseline 数值却继续使用旧 Q”的路径；准确结论是：当前主流程没有可变基线运行机制。

## 4. Innovation 1 Audit

| Feature | Status | Evidence |
|---|---|---|
| Vision state（纹理、LK/FB、LR、代价等） | PASS | 主流程质量、匹配和不确定性指标参与匹配及状态更新。 |
| 历史视差预测 | PASS | 预测结果传入 matcher。 |
| Adaptive ROI / disparity range | PASS | 搜索半径由运动、预测残差和恢复状态决定。 |
| Subpixel | PASS | `M3` / `research_full` 使用连续亚像素或 IC-GN。 |
| Confidence feedback | PARTIAL | `THESIS_FULL` 有状态机覆盖搜索半径；`research_full` 未启用 `use_confidence_feedback`。 |
| Target XYZ accuracy → policy | FAIL | 主流程不存在 target accuracy、error budget、required disparity precision。 |
| 精度规划原型 | IMPLEMENTED_BUT_NOT_INTEGRATED | `precision_planner.py` 的公式正确，但主视频流程不调用。 |
| Variable baseline coordination | FAIL | 只有离线扫描。 |

Innovation 1 Completion：**45%**。

当前实现的是“视觉状态和历史预测驱动的自适应匹配”；尚不是“目标 XYZ 精度 + Visual State → Measurement Policy”。

## 5. Innovation 2 Audit

| Feature | Status | Evidence |
|---|---|---|
| 2D–3D 重投影残差 | IMPLEMENTED_BUT_NOT_INTEGRATED | 几何含义正确，但验证脚本实际将该项固定为 `1.0`。 |
| Temporal residual | PARTIAL | 有二阶预测与瞬态阈值保护；仅在原型验证中使用。 |
| Spatial residual | PARTIAL | 只比较相对初始形状的位移幅值，不能表达结构邻域合理变形关系。 |
| Welch PSD / coherence / phase | PARTIAL | 调用正确；“幅值”并非严格时域振幅，且仅针对合成 5 Hz 信号。 |
| C_phy | IMPLEMENTED_BUT_NOT_INTEGRATED | 固定权重，没有真实数据标定或消融依据。 |
| Corrected XYZ | IMPLEMENTED_BUT_NOT_INTEGRATED | 低 C_phy 时以历史预测和邻点中值加权替换。 |
| Transient preservation | PARTIAL | 仅降低时间残差惩罚，校正仍可能用历史/邻点替代真实冲击。 |

Innovation 2 Completion：**20%**。

**P0：存在 `CORRECTION_DEGENERATES_TO_GENERIC_SMOOTHING` 风险。** 现有校正没有可验证的结构动力学模型；真实局部冲击、模态差异或快速变形可能被“历史 + 邻点中值”抹平。

**P0：物理验证含合成真值泄漏。** `physics_validation.py` 使用预设真实相位构造 `phase_evidence`，不能作为未知真实结构数据上的 C_phy 证据。

## 6. Innovation 3 Audit

| Feature | Status | Evidence |
|---|---|---|
| Image / stereo / tracking / geometry residual | PARTIAL | `FaultFingerprint` 定义了字段，但主要依赖外部手工传值。 |
| Unified fingerprint | IMPLEMENTED_BUT_NOT_INTEGRATED | 有统一结构，未接入视频主流程。 |
| Rule diagnosis | PARTIAL | 规则能区分若干类型，但阈值均为 magic numbers。 |
| Differential recovery | IMPLEMENTED_BUT_NOT_INTEGRATED | 只返回动作名和参数字典，不直接驱动主 matcher。 |
| Recovery verification / G_rec | PARTIAL | 仅要求 C_phy 与 primary residual 同时改善。 |
| Fixed reference compensation | PARTIAL | 主流程有真实 RANSAC/Kabsch 刚体补偿；`car` 无 reference 点，运行中不可用。 |
| Extrinsic recovery | MISSING | 只有动作标签，无真正外参更新、重校正或 Q 更新。 |

Innovation 3 Completion：**25%**。

**P0：故障基准的 100% 结果不可作为论文证据。** 基准重测时直接返回参考/干净观测，而不是重新执行图像、双目、跟踪和几何链路；`accuracy=1.0`、`macro-F1=1.0`、恢复后 RMSE=0 是设计保证，不是恢复性能。

## 7. Actual Test Results

| Command | Result | Runtime |
|---|---|---:|
| `python -m pytest --collect-only -q` | 205 tests collected | 2.36 s |
| `python -m pytest -q` | 205 passed；`lastfailed` 为空 | 约 1 分钟 |
| `python run_simulation.py --config configs/simulation.yaml` | PASS | 2.3 s |
| `python run_innovation1_experiment.py ...` | PASS | 合并运行 10.7 s |
| `python run_innovation2_validation.py ...` | PASS | 合并运行 10.7 s |
| `python benchmark_faults.py --seeds 20 ...` | PASS | 合并运行 10.7 s |
| `python -m stereo_research.run ... --methods M0 M1 M2 M3` | PASS | 16.4 s |
| `python -m stereo_research.evaluate ...` | PASS | 16.8 s |

- Total tests: 205
- Passed: 205
- Failed: 0
- Skipped: 0
- Warnings: 未观察到 pytest warning summary

## 8. Current Accuracy

真实 `car.avi`：**NOT_MEASURABLE_WITH_CURRENT_REPO**。

其 manifest 的 `ground_truth` 为 `null`，当前视频输出的 XYZ RMSE、MAE、Max Error 均为空，不能编造。

可运行合成仿真结果：

| Metric | Value |
|---|---:|
| X RMSE | 0.612 mm |
| Y RMSE | 0.202 mm |
| Z RMSE | 6.435 mm |
| 3D RMSE | 6.467 mm |
| displacement 3D RMSE | 7.606 mm |
| Peak displacement error | 13.335 mm |

这只证明指定噪声模型下的投影—三角化闭环；不证明真实相机、真实结构运动或论文精度。

## 9. Test Trustworthiness

### Strong

- 主几何单元测试检查 Q 输出单位、正视差和深度公式。
- 局部匹配、LR、IC-GN、Kalman、重捕获、相机刚体补偿有行为级测试。
- 测试明确区分 GT 评价与运行时输入。

### Weak

- 大量合成测试和实现使用同一投影/三角化模型，存在 circular validation 风险。
- 物理验证使用预设频率、相位和注入跳变，场景理想化。
- fault benchmark 的恢复是“返回干净参考”，不是端到端恢复。

### Missing

- 实物标定误差、不同距离、不同 baseline、遮挡/模糊/相机扰动下的独立 GT。
- 在线条件下真实冲击和多点空间关系的保护实验。
- 外参漂移→重新标定→Q/rectification 更新→恢复收益实验。

## 10. E0–E11 Readiness

| Experiment | Status |
|---|---|
| E0 静态 XYZ 精度 | MISSING |
| E1 不同距离 / baseline / error budget | PARTIAL（仅合成 baseline 扫描） |
| E2 创新点一消融 | PARTIAL |
| E3 多点周期 / 扫频 | PARTIAL（合成固定 5 Hz） |
| E4 单点人工异常 | PARTIAL |
| E5 真实冲击 / 快速运动 | MISSING |
| E6 多点空间异常 | MISSING |
| E7 blur / occlusion | PARTIAL（合成故障） |
| E8 stereo mismatch | PARTIAL（合成故障） |
| E9 camera/support perturbation | PARTIAL（补偿代码存在，未有真实验证） |
| E10 extrinsic perturbation | MISSING |
| E11 完整闭环 | MISSING |

## 11. Problems

### P0

- 创新点 2、3 未接入真实视频主流程，不能作为正式实验结论。
- Innovation 2 合成相位真值参与 C_phy 计算；fault benchmark 以 oracle 重测产生 100% 结果。
- 真实视频没有 GT，基础 XYZ 精度不可测。
- 轨迹校正可能平滑真实冲击。

### P1

- 目标精度没有控制主流程 Measurement Policy。
- 故障诊断、差异恢复、恢复验证没有运行时可达路径。
- 无运行时 baseline / extrinsic 更新机制。

### P2

- E0、E5、E6、E10、E11 缺失。

### P3

- 大文件集中度高，如 `pipeline.py` 1304 行、`local_matching.py` 1001 行；本轮不建议重构。

## 12. Existing Useful Modules

- 主流程的 LK/FB、历史视差预测、自适应搜索、左右一致性、邻域约束、SGBM 重捕获。
- 连续亚像素、IC-GN、匹配曲率和不确定性估计。
- Kalman 的 measurement / estimate 分离。
- 多静态参考点 RANSAC/Kabsch 刚体补偿。
- Innovation 1 的精度规划与 baseline 扫描原型。
- Innovation 2 的 Welch PSD、CPSD/coherence、phase、2D–3D 残差原型。
- Innovation 3 的统一残差指纹、规则诊断和恢复验证数据结构。

## 13. Dead / Redundant / Isolated Modules

- `stereo_dynamic_measurement.pipeline.StereoMeasurementOrchestrator`：ISOLATED，仅由测试和合成故障基准调用。
- Innovation 1 `AdaptiveStereoMeasurementPipeline`：ISOLATED，未被 `stereo_research` 调用。
- Innovation 2 `run_physics_validation`：ISOLATED，仅合成 CSV/绘图验证。
- Innovation 3 `RecoveryManager` / `RuleDiagnosticEngine`：ISOLATED，动作未映射为真实主流程配置和重测。
- 运行时可变 baseline：MISSING，并非已集成但未调用。

## 14. Top Modification Tasks

### Task 1：接通统一论文主闭环

- Priority: P0
- Innovation: 1→2→3
- Current Problem: 两套工程分离。
- Files: `stereo_research/pipeline.py`、`stereo_dynamic_measurement/pipeline/orchestrator.py`
- Suggested Modification: 以主流程实际 `FramePointResult` 构造统一 MeasurementResult，顺序执行验证、诊断、差异恢复、重测和验证。
- Tests Needed: 端到端 E11；断言 raw、corrected、final 分别可审计。
- Acceptance Criteria: 每个模块真实影响最终输出，且不使用 GT 或未来帧。

### Task 2：建立真实 GT 精度实验

- Priority: P0
- Innovation: 基础科学正确性
- Current Problem: `car` 无 GT，XYZ RMSE 不可测。
- Files: manifest、GT 采集与 evaluate 入口。
- Suggested Modification: 建立独立于算法的静态/位移 GT 与硬件时间戳同步流程。
- Tests Needed: E0、E1。
- Acceptance Criteria: 报告 X/Y/Z/3D RMSE、MAE、Max Error、静态 std/drift，且 GT 不进入运行时决策。

### Task 3：用目标精度驱动 Measurement Policy

- Priority: P1
- Innovation: 1
- Current Problem: 目标误差公式仅在隔离原型中。
- Files: `precision_planner.py`、主 pipeline 配置与搜索策略。
- Suggested Modification: 将 `epsilon_Z / Sigma_XYZ → required sigma_d → search/matcher/subpixel/retry policy` 接入主流程。
- Acceptance Criteria: 消融显示 target policy 改变实际搜索、接受/重测行为，并改善目标误差或效率。

### Task 4：重建 C_phy 与保真校正证据链

- Priority: P0
- Innovation: 2
- Current Problem: 合成真值泄漏，校正可能变成泛化平滑。
- Files: `physics_validation.py`、`trajectory_corrector.py`
- Suggested Modification: 禁止预设相位/频率/故障标签参与运行时 C_phy；增加真实瞬态保留和错误跳点抑制的独立 GT 对照。
- Acceptance Criteria: 对冲击幅值、频率、相位和误差均给出 before/after，且不以低通平滑替代校正。

### Task 5：端到端故障恢复与外参健康实验

- Priority: P1
- Innovation: 3
- Current Problem: 当前恢复基准使用 oracle 参考值；外参更新只是动作名。
- Files: `recovery_manager.py`、主 pipeline、calibration。
- Suggested Modification: 将每个 fault 映射到实际 matcher/flow/reinitialize/compensation/extrinsic-update 行为，恢复后重算 R、C_phy 与不确定度。
- Acceptance Criteria: E7–E10 中报告非零且真实的 recovery gain，特别是外参变化后 P/Q/rectification 同步更新。

## 15. Not Recommended Yet

- 不要大规模重构或美化大文件。
- 不要为了“创新”删除 LK、FB、Kalman、亚像素、预测、参考点补偿等有效模块。
- 不要加入深度学习或大量新 fault class。
- 不要将当前合成 100% fault benchmark 或合成 C_phy 结果写成真实系统性能。
- 不要在没有 E0/E1 GT 的情况下宣称毫米级真实测量精度。

## 最终五问

1. **当前基础双目三维测量到底准不准？**
   - 代码层面的基础几何链条基本正确；真实系统精度尚不可证明。

2. **创新点一是否真正实现 Target XYZ Accuracy + Vision State → Measurement Policy？**
   - 部分实现视觉状态→自适应匹配；未实现目标 XYZ 精度→Measurement Policy 主闭环。

3. **创新点二的 r_phy → C_phy → Corrected XYZ 是否是真闭环，并有无错误平滑风险？**
   - 不是实际主流程闭环；存在错误平滑真实快速运动的明确风险。

4. **创新点三是否真正实现 Detect → Diagnose → Recover → Verify？**
   - 未真正实现；当前为隔离原型与合成基准。

5. **距离正式开展硕士论文实验，最关键还差什么？**
   - 真实 GT；三创新统一接入；目标精度策略；无泄漏 C_phy/保真校正；端到端故障恢复与外参实验。
