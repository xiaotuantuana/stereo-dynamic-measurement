# 第一轮 Innovation 2/3 Shadow Mode 实施报告

日期：2026-08-20  
范围：仅执行已批准的 Task A 与 Task B；未提交 Git commit；未主动改动与本轮无关的既有修改。

## 1. Files Changed

本轮新增或修改的实现文件：

- `stereo_dynamic_measurement/innovation2/physics_confidence.py`
  - 新增 `EvidenceTerm(value, valid, reliability, reason)`。
  - 缺失证据从融合有效集合排除；输出有效项、缺失项和 effective weights。
- `stereo_dynamic_measurement/innovation2/transient_gate.py`
  - 新增保守的真实瞬态保护与孤立测量异常判定。
- `stereo_dynamic_measurement/innovation2/trajectory_corrector.py`
  - 改为 candidate-only correction；增加理由、瞬态保护标记和 25 mm 可配置安全上限。
- `stereo_dynamic_measurement/innovation2/physics_validation.py`
  - 拆分 causal runtime analysis 与 Ground Truth evaluation；运行期不接收 GT、预设频率、预设相位或故障标签。
- `stereo_dynamic_measurement/innovation3/fault_fingerprint.py`
  - 证据字段允许 `None`，缺失不再伪造成正常常数。
- `stereo_dynamic_measurement/innovation3/fault_classifier.py`
  - 规则仅在对应证据可用时触发；未新增 fault class。
- `stereo_research/shadow_analysis.py`
  - 新增只读 Shadow adapter、独立拷贝 history、candidate correction、真实 fingerprint 与 recovery recommendation。
- `stereo_research/models.py`
  - 最小追加 Shadow 配置与结果字段；第一轮主动写 final 的配置会抛出 `NotImplementedError`。
- `stereo_research/pipeline.py`
  - 只在现有相机补偿完成后追加 Shadow observer 调用。
- `stereo_research/runner.py`
  - Shadow 字段直接追加到各方法原 CSV 尾部，不生成 `*_shadow.csv`。

本轮新增/更新的测试与文档：

- `tests/test_shadow_scientific_safety.py`
- `tests/test_shadow_fault_adapter.py`
- `tests/test_shadow_pipeline.py`
- `tests/test_shadow_state_isolation.py`
- `tests/test_innovation2_validation.py`
- `tests/test_innovation2_correction.py`
- `docs/superpowers/specs/2026-08-20-innovation2-3-shadow-mode-design.md`
- `tasks/innovation23-shadow-plan.md`
- `tasks/innovation23-shadow-todo.md`

工作区在本轮前已有其他修改；本报告不把这些既有修改宣称为本轮成果。

## 2. GT Leakage Removal

### Before

- `physics_validation.py` 使用配置中的真实频率构造 spectral evidence。
- 使用预设测点真实相位差构造 phase evidence。
- 无实际来源的 2D–3D/参考项曾以 `1.0` 进入融合。
- 同一验证流程同时持有运行时测量和 GT，软件边界不清晰。

### After

- `run_runtime_physics_analysis(raw_measurements, fs_hz, threshold)` 的参数中不存在 GT、truth、真实频率、真实相位或故障标签。
- 运行期只读取当前帧与过去测量；frame history 在当前帧分析完成后才更新，不读取 future frame。
- 频率证据来自当前因果测量窗口内各测点的 dominant-frequency 共识。
- 当前无法建立可靠独立基线的 phase/coherence 与 2D–3D 证据明确标为 unavailable，不填 `1.0`。
- `GroundTruthEvaluation` 仅传给 `evaluate_runtime_against_ground_truth()`，只改变 RMSE/绘图等评价结果。
- 自动测试证明改变 GT XYZ、GT phase、GT frequency 和 injected fault 不改变 runtime trajectory 或 C_phy。

搜索 Innovation 2 后，`ground_truth`/`injected_fault` 只存在于独立 evaluation dataclass、评价函数和合成评价构造位置。

## 3. Innovation 2 Runtime Flow

```text
measured / estimated / compensated XYZ
                ↓
      select shadow_input_xyz
  compensated → estimated → measured
                ↓
 causal runtime evidence (no GT/future)
                ↓
   missing-aware, auditable C_phy
                ↓
        Transient Preservation Gate
                ↓
 bounded candidate_corrected_xyz only
```

证据不足时采取保守策略：不授权 correction，而不是假定证据健康。

## 4. Innovation 3 Shadow Flow

```text
actual FramePointResult residuals
                ↓
 optional FaultFingerprint fields
                ↓
 existing RuleDiagnosticEngine
                ↓
 existing FaultClass
                ↓
 recommended recovery action only
```

RecoveryManager 不连接 matcher、calibration、baseline、rectification 或 extrinsic update，不存在执行回调。

## 5. Raw / Corrected / Final Relationship

- `measured_xyz`：Stereo/Q 直接重建，含义不变。
- `estimated_xyz`：现有 Kalman/状态估计结果，含义不变。
- `compensated_xyz`：现有公共相机运动补偿结果，含义不变。
- `shadow_input_xyz`：优先读取 compensated，其次 estimated，最后 measured，并输出 `shadow_input_stage`。
- `candidate_corrected_xyz`：Innovation 2 的诊断候选；可因证据和瞬态门控而变化，但不回写任何 legacy 阶段。
- `final_xyz`：本轮无条件保持 Shadow 接入前主流程的正式输出。

两个 active 配置并非“仅默认 False”：任一设置为 True 都会立即抛出 `NotImplementedError`，因此第一轮功能上不可启用。

CSV 继续使用 legacy `final_X_m/Y_m/Z_m`。未追加仅大小写不同的 `final_x_m/y_m/z_m`，避免 Excel/PowerShell 将其识别成重复字段。实际 `M3.csv` 为 169 个大小写不敏感的唯一列，Shadow 字段位于尾部。

## 6. Tests

- Total: 219
- Passed: 219
- Failed: 0
- 原测试基线：205 项全部保留并通过
- 新增：14 项科学安全、Shadow 接入、GT 隔离、missing evidence、瞬态保护、孤立异常、hard flag、真实 fingerprint、recommendation-only 和完整序列隔离测试

完整序列隔离测试使用 `research_full`，比较 Shadow OFF 与 ON 的：

- 全部 legacy `FramePointResult` CSV 字段；
- measured / estimated / compensated / final XYZ；
- disparity、predicted disparity、search radius、status、confidence、recovery stage；
- PointState 的全部状态；
- Kalman state、covariance、predict-only 计数；
- 左右光流参考图像。

墙钟耗时字段非确定性，只验证其有效域；Shadow 调用点位于 legacy 计时完成之后。

## 7. Regression

| 指标 | 审计基线 | 修改后 | 结论 |
|---|---:|---:|---|
| X RMSE | ≈ 0.612 mm | 0.612320 mm | 无有意义退化 |
| Y RMSE | ≈ 0.202 mm | 0.201672 mm | 无有意义退化 |
| Z RMSE | ≈ 6.435 mm | 6.435146 mm | 无有意义退化 |
| 3D RMSE | ≈ 6.467 mm | 6.467358 mm | 无有意义退化 |

已成功运行：

- `python run_simulation.py --config configs/simulation.yaml`
- `python run_innovation1_experiment.py --config configs/innovation1_experiment.yaml`
- `python run_innovation2_validation.py --config configs/innovation2_validation.yaml`
- `python benchmark_faults.py`
- `python -m stereo_research.run --manifest experiments/car/manifest.json --methods M0 M1 M2 M3`

`car.avi` M0–M3 冒烟均成功；manifest 没有 GT，因此不能报告真实 XYZ 精度。

## 8. Remaining Risks

- **PARTIAL — C_phy 权重**：仍为集中配置中的临时固定权重；虽然 effective weights 可审计，但尚未经过正式标定。
- **PARTIAL — spatial residual**：当前为同帧多测点位移相对中位数的简单残差，尚非完整结构动力学模型。
- **PARTIAL — phase/coherence**：主 Shadow adapter 当前没有足够可靠的独立因果参考窗口，明确输出 unavailable；没有用真实相位或常数补齐。
- **PARTIAL — 2D–3D evidence**：当前没有独立 3D flow prediction 时为 unavailable。visual residual 只用于瞬态门控和 fingerprint，不伪装成 2D–3D term。
- **PARTIAL — transient gate**：最低可用规则门控，对阈值、同步比率和复杂非刚体瞬态仍需下一轮系统标定。
- **PARTIAL — candidate correction**：仍基于历史预测和邻点中位关系，但只有多源证据授权时才运行，并受 25 mm configurable safety guard 限制；该默认值是工程保护，不是科学结论。
- **MISSING — active correction/recovery**：按第一轮要求故意未实现，配置为 True 会硬失败。

## 9. Acceptance Summary

第一轮验收项均已满足：运行时不使用 GT；缺失证据不等于健康；同步瞬态受保护；孤立多源异常可生成受限候选；Innovation 2/3 已进入真实主流程 Shadow Mode；Recovery 只推荐；legacy XYZ、状态演化和 raw 仿真性能无回归；未修改 Stereo geometry、disparity sign、Q 单位逻辑、LK/FB/SGBM/IC-GN 核心。
