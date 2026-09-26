# CODE_AUDIT_REPORT — 点式时序双目测量论文代码系统审计报告

- 审计日期：2026-08-18
- 审计范围：`<project-root>`（`stereo_research/` 论文核心包 + `stereo_dynamic_measurement/` 研究包 + `gui/` + `tests/` + 实验与文档）
- 审计性质：**第一阶段只读审计，未修改任何代码**
- 基线环境：Linux / Python 3.10.12 / OpenCV 5.0.0 / NumPy 2.2.6 / scipy 1.15.3 / pytest（项目原始环境为 Windows / Python 3.14 / OpenCV 4.x）

---

## 1. 执行摘要

| 级别 | 数量 | 一句话描述 |
|---|---|---|
| **P0** | **0** | 未发现确定导致算法或实验结论错误的硬伤 |
| **P1** | **2** | 两个高概率 bug，均位于 `stereo_dynamic_measurement` 研究包（创新点 1/2 辅助模块） |
| **P2** | **8** | 数值/鲁棒性/复现性问题，含 OpenCV 5.0 兼容性（阻塞新环境复现） |
| **P3** | **6** | 性能/元数据/字段语义问题 |
| **P4** | **6** | 代码质量/约定问题 |

**最重要的结论**：`stereo_research/` 论文核心包的**双目几何、时序跟踪、局部匹配、亚像素（含 IC-GN）、卡尔曼滤波、相机补偿、循环一致性、消融控制与指标体系在数学与数值上全部验证正确**，未发现数据泄漏，未发现"预测值冒充测量值"的自强化反馈。两个 P1 问题位于配套研究包，不直接污染论文核心 `stereo_research` 结果，但若论文引用创新点 1/2 的辅助实验则必须修复。

**当前环境 20 个测试失败全部源于同一根因**：OpenCV 5.0 的 `stereoRectify` 要求平移向量 `T` 为 `(3,1)` 列向量，而代码传入 `(3,)`。这是 OpenCV 4.x→5.0 的 API 破坏性变化，不是算法 bug，修复极小。

---

## 2. 基线

### 2.1 测试基线（本环境）

```
python -m pytest -q --ignore=tests/test_gui_app.py
169 passed, 20 failed
```

- 20 个失败全部为 `cv2.stereoRectify` 的 `gemm` 断言错误（OpenCV 5.0 兼容性，见 P2-1）。
- `tests/test_gui_app.py` 因缺 PySide6/pytest-qt 无法收集（环境限制，非代码问题）。
- 项目历史基线：`TEST_REPORT.md` 报告 76 passed（2026-08-01）；`FIX_NOTES_2.md` 报告 135 passed（其后测试继续增加，现总计约 190 项）。

### 2.2 已通过数学真值验证的现有测试（证据）

以下测试是真正的数值真值验证，不是"只测能运行"：

| 测试 | 验证内容 | 容差 |
|---|---|---|
| `test_calibration_geometry.py::test_q_reprojection_returns_metre_coordinates` | Q 重投影 Z=fB/d（主点 2.0m） | atol 1e-5 |
| `test_stereo_precision_sanity.py` | d=fB/Z 与一阶精度公式 σ_d=E·f·B/Z² | 1e-12 |
| `test_icgn.py::test_icgn_recovers_noise_free_fractional_disparity` | IC-GN 恢复 0.25/0.5/0.75/1.2px 分数视差 | <0.02 px |
| `test_camera_compensation.py::test_weighted_kabsch_recovers_known_transform` | 已知旋转/平移刚体恢复 | atol 1e-10 |
| `test_simulation_calibration.py::test_dlt_triangulation_recovers_mm_coordinates_without_noise` | 已知 3D 点投影→三角化 | <1e-6 mm |

### 2.3 审计期间独立数值验证（本报告作者执行）

见第 3 节，全部通过。

---

## 3. 已通过验证的模块（数学证据）

### 3.1 双目几何（最高优先级，通过）✅

对内置标定 `builtin_640x480()` 实测 Q 矩阵：

- **Q 为标准 OpenCV 形式**：`Q[0,3]=-cx`、`Q[1,3]=-cy`、`Q[2,3]=f`、`Q[3,2]=-1/Tx`、`Q[3,0]≈0`（主点已对齐）。
- **基线正确**：`baseline_m = abs(1/Q[3,2]) = 0.12036 m`，与 `|T| = 120.36 mm` 完全一致。
- **Z = fB/d 完全成立**：d=5→12.393 m、d=10→6.196 m、d=20→3.098 m、d=50→1.239 m，重投影 Z 与 `f·B/d` 逐位一致，且深度为正、近大视差→小 Z。
- **X 右手系正确**：x 像素右移 → X 增大（x=300→-0.060m, 320→+0.061m, 340→+0.181m）。
- **单位统一**：标定 mm → 重投影 `reproject_point_m` 除以 1000 输出 **m**；CSV 中 `*_m` 与 `*_mm` 字段定义一致。
- **disparity 定义全局统一为 d = x_left − x_right**：SGBM（`sample_initial` 中 `xr = xl − d`）、局部匹配（`right_xy = (xl − disparity, y)`）、亚像素（disparity 增大 → 右点左移）、IC-GN、恢复路径、Q 重投影全部一致。右图 LK 用 `cycle_predicted = xl − xr_temporal`，同样一致。
- **ROI**：`valid_roi_left=(5,6,634,473)`、`valid_roi_right=(0,0,634,476)`，合理。
- **左右畸变顺序**：`(k1,k2,p1,p2,k3)` 5 参数，与 OpenCV 一致。

### 3.2 SBS 拆分 ✅

- `split_side_by_side`：`frame[:, :w/2]` 与 `frame[:, w/2:]`，右图裁剪为局部坐标。
- `annotate_gt` 鼠标点击：左图局部坐标直接存；右图 `x≥width` 时减 `width` 转局部坐标。**无 x±width/2 混淆**。

### 3.3 SGBM 基线 ✅

- `minDisparity=0`（左）、`minDisparity=-num+1`（右），`numDisparities` 向上取 16 倍数。
- `P1=8·cn·bs²`、`P2=32·cn·bs²`，`disp12MaxDiff`、`uniquenessRatio`、`speckleWindowSize/Range` 合理。
- 输出统一 `/16.0`；无效/负视差→`ambiguous`；达到上界→`saturated`。

### 3.4 LK 光流 ✅

- **FB error 正确实现**：`fb_error = ||backward_point − previous_point||`（即 `||p_t − p_back||`，**不是** `||p_forward − p_back||`）。
- 数值验证：已知平移 (dx=2, dy=1)，LK 返回 (66.0, 65.0)，fb_err=0.0006 px。
- 边界/低纹理/恢复窗口切换均有处理。

### 3.5 局部匹配 / 多代价融合 / 亚像素 / IC-GN ✅

- 搜索区间 `[center−r, center+r]` 以预测视差为中心，无 off-by-one；`x_right_pred = x_left − d_pred` 方向正确。
- **动态归一化代价** `Σ(w_i·C_i)/Σ(w_valid)`：关闭任意约束时仅对激活项归一化，代价尺度稳定（符合设计意图）。
- 抛物线亚像素：`0.5·(c₋−c₊)/(c₋−2c₀+c₊)`，标准公式，符号方向正确。
- **IC-GN 数值验证**：在测试风格图像上，0.1/0.25/0.5/0.75 px 已知分数视差恢复误差分别为 0.0054/0.0010/0.0002/0.0023 px（<0.01 px），方向与 d=xl−xr 一致。
- 连续采样 subpixel 以整数最优为中心 ±1 px，正确。

### 3.6 卡尔曼滤波 / 置信度 / 状态机 ✅

- **数值验证**：测量 45 px（真实 20 px，方差 0.01）→ `predict_only_outlier`，状态保持 20.0 不被污染；正常测量 20.5 → 更新至 20.4975，NIS=0.12；连续 5 帧野值 → `predict_only_frames=5` ≥ `kalman_max_predict_only_frames` → 触发 recovery。
- **无自强化反馈**：被拒绝的原始测量不进入 `disparity_history`；`research_full` 下一帧预测使用 `estimated_disparity_history`（滤波估计），`measurement_accepted_for_state` 区分 `measured`/`estimated`/`predicted` 三源，CSV 逐项记录。
- 状态机 `tracking→recovering→lost→recovery→tracking`：失败计数、恢复间隔、`confirm_recovery` 重置 `previous_left_xy` 逻辑正确。
- 置信度指标方向统一（数值越高质量分越高），`decide_confidence` 的 `lr_only/single_margin/multi_reject/closed_loop` 各模式行为正确。

### 3.7 循环一致性 ✅

- 三级判定（软接受→扩大搜索恢复→硬拒绝）阈值数学正确。实测：err0.2→valid(scale=1.0)、err1.0→soft(0.75)、err3.0→recovery_required、恢复候选不合格→failed。
- 立体匹配右点与右图时间跟踪右点均在**右图局部坐标**比较，坐标系一致。

### 3.8 不确定度 ✅

- `estimate_match_uncertainty` 风险项加权合成 `quality_score=1/(1+Σw·risk)`，方向正确（风险越大分数越低、方差越大）。与 σ_Z = fB/d²·σ_d 的"越远越不稳"趋势一致（disparity 方差随风险放大，深度越远视差越小）。

### 3.9 相机补偿 / RANSAC ✅（数值验证通过）

- 刚体方向统一为 `P_reference = R @ P_current + t`（加权 Kabsch/SVD），补偿时对测量点施加同一变换。**数值验证**：相机平移 (10,−5,0)mm + 旋转 0.02rad，目标点真实移动 5mm 且相机同移，raw 观测位移 [15,−45,−0.4]mm，补偿后 **[5,0,0]mm，精确恢复**。
- RANSAC：5 参考点含 1 outlier → inliers=4/valid；共线 → `degenerate_references`；仅 1 点 → `insufficient_references`。失败时 `final_*` 回退 `estimated_*` 且 `compensation_applied=False`，**不会返回看似正常的错误刚体矩阵**。

### 3.10 Evaluation / 泄漏 ✅

- pipeline 完全不使用任何 GT；GT 仅在 `evaluate`/`report` 后处理阶段。**无 GT 泄漏**（未用 GT 选参、选阈值、过滤帧或决定恢复）。
- 指标同时报告 Accuracy（RMSE/MAE）与 Coverage/effective_tracking_rate，不只看成功帧 RMSE（见 P3-4 的分母细节）。

---

## 4. 详细问题清单

### P0 — 确定导致算法或实验结论错误

**无。**

### P1 — 高概率 Bug

#### P1-1 像素速度单位混用：mm/s 被当作 px/s

- **文件**：`stereo_dynamic_measurement/innovation1/motion_predictor.py`
- **位置**：`MotionPredictor.predict()` L47-48：
  ```python
  velocity_xy = state.velocity_xy_px_s or state.velocity_xyz_mm_s[:2]
  xy = tuple(float(state.xy_px[i] + velocity_xy[i] * dt) for i in range(2))
  ```
- **问题**：`velocity_xy_px_s` 在 `PointMeasurementState` 中默认 `None`，且全包**从未**为其赋值（`measurement_pipeline.py` 构造状态时只传 `velocity_xyz_mm_s`）。因此 `or` 恒命中右支，把 **mm/s 的三维速度前两分量直接加到像素坐标**上。
- **为什么是问题**：像素速度 ≠ 物理速度。正确换算为 `px/s = mm/s · f / Z`。在 Z=2000mm、f≈800px 时误差系数达 Z/f≈2.5 倍。
- **影响**：LK 初始猜测、`temporal_residual`、自适应搜索半径、置信度被系统性放大；`test_innovation1_foundation.py::test_motion_predictor_tracks_xyz_disparity_velocity_and_confidence` 固化了错误物理（断言 10mm/s 直接当 10px/s 的结果）。
- **复现**：构造 `PointMeasurementState(xy_px=(100,50), velocity_xyz_mm_s=(10,0,0), ...)`，`predict()` 返回 `xy_px=(101,50)`（应为 `≈100+10·f/Z`，Z=2000、f=800 时约 104）。
- **推荐修改**：要么由上层传入正确的 `velocity_xy_px_s`（用 `f·V/Z` 换算），要么在 `predict` 内用 `disparity_px` 反推 Z 换算。需配套修正测试断言。
- **是否影响论文算法定义**：不影响 `stereo_research` 论文核心；影响创新点 1（自适应搜索）的辅助研究包。若论文引用该模块结果则必须修复。

#### P1-2 瞬态保护判据与残差同源，无法区分野值与真实瞬态

- **文件**：`stereo_dynamic_measurement/innovation2/temporal_consistency.py`
- **位置**：`temporal_prediction_residual()` L22-29：
  ```python
  predicted = 2.0 * history[1] - history[0]
  innovation = current - predicted
  residual = float(np.linalg.norm(innovation))
  acceleration = float(np.linalg.norm(current - 2.0 * history[1] + history[0]))
  protected = acceleration >= transient_acceleration_mm
  ```
- **问题**：`current − 2·h1 + h0` 与 `current − predicted`（其中 `predicted = 2·h1 − h0`）**数学上完全相同**，故 `acceleration ≡ residual`。`protected = residual ≥ transient_acceleration_mm` 等价于"残差大即受保护"，随后 `penalty *= 0.25`。
- **影响**：任何大跳变——包括**测量野值**——自动获得"瞬态保护"，时序惩罚被削弱 4 倍，降低野值检出能力；且残差在阈值处产生不连续（14.9mm 时 penalty≈1，20mm 时≈0.25）。`run_physics_validation.py` L84-85 被迫用空间拟合残差手动补回惩罚，正说明该缺陷。
- **复现**：`temporal_prediction_residual(np.array([[0,0,0],[1,0,0]]), np.array([30,0,0]), transient_acceleration_mm=20)` → `residual=29, protected=True, penalty≈0.725`，野值被当作受保护瞬态。
- **推荐修改**：瞬态判据应独立于残差（例如基于物理加速度上限、或与测量噪声 σ 统计独立判定），使野值不自动受保护。
- **是否影响论文算法定义**：影响创新点 2（物理可信度/轨迹校正）辅助模块；论文若引用该模块需修复。

### P2 — 潜在数值问题 / 鲁棒性问题 / 复现性

#### P2-1 OpenCV 5.0 兼容性：`stereoRectify` 要求 T 为列向量

- **文件**：`stereo_research/calibration.py` L51；`stereo_dynamic_measurement/calibration/camera_model.py` L84
- **位置**：`cv2.stereoRectify(..., self.translation, ...)` / `cv2.stereoRectify(..., T, ...)`
- **问题**：OpenCV 5.0 中 `T` 必须为 `(3,1)` 列向量；代码传入 `(3,)`，触发 `gemm: a_size.width == len` 断言。OpenCV 4.x 正常。
- **影响**：**当前环境所有双目校正与管线运行失败（20 个测试失败的唯一根因）**；在新环境（OpenCV 5）无法复现任何 `stereo_research` 实验。
- **复现**：任意调用 `builtin_640x480().rectification()`。
- **推荐修改**：`stereoRectify` 调用处将 `T` 做 `np.asarray(T).reshape(3,1)`（两处）。修复后 20 个失败应全部恢复。
- **是否影响论文算法定义**：否。纯 API 兼容性，不改变任何数值结果（已验证 reshape 后 Q 与 4.x 一致）。

#### P2-2 消融公平性：M0/M1/M2 无恢复机制，M3+ 有

- **文件**：`stereo_research/pipeline.py` step() L162-196；`stereo_research/models.py` `MethodProfile`
- **位置**：`can_attempt` 条件要求 `adaptive_search or use_confidence_feedback or method=="full_quality"`。
- **问题**：恢复（recovery）只在 M3/THESIS_FULL/research_full/full_quality 启用；M0/M1/M2 完全无恢复。M2→M3 的覆盖率提升除论文声明的局部约束/亚像素/LR 外，还包含恢复机制，但 `MethodProfile` 中恢复不是显式字段。
- **影响**：M0–M3 逐级对比的"增量归因"不够干净；若论文声称 M3 相对 M2 的增益仅来自局部约束，需补充说明恢复是 M3 行为的一部分。
- **推荐修改**：论文消融设计部分明确写出"M3 同时引入自适应搜索与配套的 lost 恢复机制"；或（若需更严格归因）增加 `enable_recovery` 显式剖面控制。
- **是否影响论文算法定义**：不改变代码行为，但影响消融实验的解释与公平性论证。

#### P2-3 消融公平性：`full` 与 `full_quality` 使用不同匹配引擎

- **文件**：`stereo_research/pipeline.py` L47-51
- **位置**：`self.local_matcher = QualityLocalMatcher(...) if method in {"full_quality","research_full"} else LocalMatcher(...)`
- **问题**：`full` 与 `full_quality` 的 `MethodProfile` 相同，但 pipeline 仅对 `full_quality`/`research_full` 使用 `QualityLocalMatcher`（含上下文大窗口复核、金字塔恢复、时间验证）。模型中的 "full" 用 `LocalMatcher`。
- **影响**：论文若比较 `full` vs `full_quality`，差异不仅是名字；消融套件中 `full` 实际用 `full_quality` 引擎（`run_ablation_suite` variants["full"]=("full_quality", base)），与 `run_manifest` 直跑的 `full`（LocalMatcher）行为不同。可能造成跨命令结果不可比。
- **推荐修改**：论文明确区分"模型 full（LocalMatcher）"与"消融 full（QualityLocalMatcher 全功能）"；或在 `run_manifest` 与 `run_ablation_suite` 之间统一语义。
- **是否影响论文算法定义**：不改变论文定义本身，但影响实验报告的一致性与可复现性。

#### P2-4 消融语义：`full_no_recovery` 并非完全无恢复

- **文件**：`stereo_research/runner.py` run_ablation_suite variants；`stereo_research/local_matching.py` QualityLocalMatcher
- **位置**：`"full_no_recovery": ("full_quality", replace(base, enable_recovery=False))`
- **问题**：`enable_recovery=False` 只关闭 pipeline 级 lost 恢复；QualityLocalMatcher 内部的上下文复核（15×15）与三级金字塔恢复仍启用（它们不受 `enable_recovery` 控制）。
- **影响**：`full_no_recovery` 的消融结果仍包含匹配器级的多尺度恢复能力，不能严格归因"无恢复"。
- **推荐修改**：若论文将"恢复"作为一个整体创新点，消融需同时关闭 pipeline 级与匹配器级恢复，或重新命名该消融项为 `full_no_track_recovery`。
- **是否影响论文算法定义**：影响消融解释，不改变算法定义。

#### P2-5 消融公平性：M2 固定半径 vs M3 自适应半径

- **文件**：`stereo_research/models.py` MatcherConfig（`search_radius=8`，`adaptive_search_small/normal/large/recovery=4/8/16/32`）
- **问题**：M2 用固定 `search_radius=8`；M3 用自适应 4/8/16/32。M3 在稳定帧用更小半径（4），可能系统性降低误匹配率（小窗口搜索更不易误匹配）。
- **影响**：M2→M3 的误匹配率/覆盖率差异部分来自搜索范围策略，而非仅局部约束本身。
- **推荐修改**：论文中说明自适应搜索是 M3 定义的一部分；可另加 M3 用固定半径 8 的对照（若需归因）。
- **是否影响论文算法定义**：不改变定义，但影响消融归因讨论。

#### P2-6 GUI 参数透传：`search_radius` 对默认方法无效

- **文件**：`stereo_research/gui/settings.py` L36；`stereo_research/pipeline.py` `_determine_search_radius()`
- **问题**：GUI 暴露 `search_radius`/`expanded_search_radius`，但 M3/THESIS_FULL/research_full（GUI 常用方法）走自适应搜索，返回 `adaptive_search_*` 半径，GUI 未暴露这些参数 → 用户在面板调整搜索半径无任何效果。
- **影响**：GUI 实验的参数面板宣称与实际行为不符，影响论文 GUI 实验的可复现性与可信度。
- **推荐修改**：GUI 暴露自适应半径参数，或在非自适应方法上才显示搜索半径控件。
- **是否影响论文算法定义**：否（仅 GUI 透传）。

#### P2-7 指标分母不一致：false_match vs accurate_tracking

- **文件**：`stereo_research/metrics.py` `evaluate_rows`
- **位置**：`false_match_rate_pct = _percent(false_matches, accepted_with_pixel_gt)`（分母为有效且有真值的行）；`accurate_tracking_rate_pct = _percent(correct_tracks, pixel_gt_count)`（分母为所有有真值的行）。
- **问题**：两个率分母不同，当存在非有效但有真值的行时，两率之和 ≠ 100%，可能误导读者。`test_metrics.py` 固化了该行为（33.3% vs 50%）。
- **推荐修改**：统一分母语义（都基于有效行或有真值行），或文档明确两个指标含义不同。
- **是否影响论文算法定义**：不影响算法，影响指标解释。

#### P2-8 频谱幅值计算错误（辅助模块）

- **文件**：`stereo_dynamic_measurement/innovation2/spectral_analysis.py` L44
- **位置**：`dominant_amplitude = sqrt(psd[index])`
- **问题**：Welch PSD 密度的幅值不是信号振幅（量纲 mm/√Hz），`sqrt(PSD)` ≠ 正弦振幅。
- **影响**：`amplitude_ratio` 等上报字段数学错误；当前决策路径未使用（仅上报），影响有限。
- **推荐修改**：对正弦分量用 `amplitude = 2·|X[k]|/N`（DFT 谱线）或在 Welch 下用能量等效换算。
- **是否影响论文算法定义**：影响创新点 2 的频谱证据字段。

### P3 — 性能 / 元数据 / 字段语义

- **P3-1** `stereo_research/pipeline.py` `_with_confidence_feedback()` 为 dead code（定义后从未调用）。THESIS_FULL 闭环置信度实际由 `_valid_result` 内部 `decide_confidence` 处理。建议删除或明确其用途，避免论文描述与代码不一致。
- **P3-2** `FramePointResult.confidence`（旧字段，用 `match.confidence` 启发式）与 `confidence_total`（基于不确定度重算的 score）可能不同。CSV 需在论文附录说明两字段语义。
- **P3-3** GUI 输出缺配置元数据：`gui/processing.py` 只写裸 CSV，无 CLI `run_manifest` 那样的 `run_metadata.json` sidecar。GUI 跑的正式实验不可复现。建议 GUI 落盘时同时输出完整 `MatcherConfig`。
- **P3-4** `stereo_dynamic_measurement/innovation1/motion_predictor.py`：`motion_model="kalman"` 名不副实——`_covariances` 只加 process_noise，无测量更新、无增益，输出与 constant_velocity 相同。属冗余计算，无自反馈污染。
- **P3-5** `stereo_dynamic_measurement/innovation2/physics_validation.py` L88：`abs(abs(phase_diff) − abs(φᵢ−φ₀))` 对近 ±π 的相位差会错误放大相位证据，弱化证据强度。
- **P3-6** `stereo_research/report.py`：`speedup_vs_sgbm` 与 `speedup_vs_M0` 被设为同一值（当 M0 存在时）。字段语义重复，建议明确 baseline 选择规则。

### P4 — 代码质量 / 约定

- **P4-1** 内置标定 `builtin_640x480()` 的 `translation` 的 `Tx=-120.36mm` 为负（右相机位于左相机 X 负方向）。Q 矩阵自洽所以数学正确，但标定坐标系约定建议在论文附录说明。
- **P4-2** `stereo_dynamic_measurement/calibration/camera_model.py`：Q（校正域）与 P1/P2（物理域）域不一致；Q 在全包未被使用，属潜伏不一致。
- **P4-3** `stereo_dynamic_measurement/simulation/stereo_scene.py`：定位噪声对左右独立注入，物理上应共模（视差噪声被 √2 放大，略微偏悲观）。
- **P4-4** `stereo_dynamic_measurement/benchmark_faults.py` L42：特设规则（垂直运动为 0 时清零 lr_residual）反映 `fault_classifier` 对 flow_drift 与 stereo_mismatch 不可区分，属合成基准特设。
- **P4-5** `tests/test_calibration_geometry.py::test_builtin_calibration_produces_expected_rectified_geometry` 硬编码 OpenCV 输出值（focal 514.8282456、B 0.1203579594），属回归钉，OpenCV 版本升级可能假红（本环境已因此类差异显示，但当前失败实为 P2-1）。
- **P4-6** `tests/test_local_matching.py::test_no_pyramid_ablation_uses_non_pyramid_independent_evidence` 断言过弱（接受 `ambiguous`），实际只验证 `_pyramid_disparity` 未被调用，未证明无金字塔路径正确恢复。

---

## 5. 论文定义一致性评价

| 论文关注点 | 结论 |
|---|---|
| 1. 双目几何数学正确？ | ✅ 是（Z=fB/d，单位 m，d=xl−xr 统一） |
| 2. 时序跟踪正确？ | ✅ 是（LK FB 往返、边界/恢复处理正确） |
| 3. 局部视差搜索正确？ | ✅ 是（搜索方向、动态归一化、off-by-one 无） |
| 4. 亚像素是否真提高精度？ | ✅ 是（IC-GN 恢复 <0.01px；现有测试 <0.02px） |
| 5. 置信度机制有效？ | ✅ 是（方向统一，closed_loop 决策正确） |
| 6. Kalman 是否有 prediction feedback？ | ✅ 否（野值 predict-only，状态不被污染，预测用滤波估计而非测量回灌） |
| 7. 恢复机制真实有效？ | ✅ 是（但见 P2-2/P2-4 消融语义） |
| 8. 相机补偿数学正确？ | ✅ 是（数值验证精确恢复 5mm 目标运动） |
| 9. M0–M3/FULL 消融公平？ | ⚠️ 基本公平，但有 P2-2/P2-3/P2-4/P2-5 的归因边界需在论文说明 |
| 10. 统计指标正确？ | ⚠️ 基本正确，P2-7 分母语义需明确 |
| 11. 有 GT leakage？ | ✅ 无 |
| 12. Synthetic GT 验证？ | ✅ 核心模块已有真值测试；仍需端到端完整管线测试（见 §7） |
| 13. 完整 pipeline 在已知位移下误差？ | ❓ 待端到端合成序列实验给出（见 §7 Test F） |
| 14. 阻碍"毫米级精度"声明的事项 | 见 §6 |

---

## 6. 关于"毫米级精度"声明的现状与阻碍

当前 `car.avi` 实验（TEST_REPORT/README）已明确：**无真值，只能做流程/稳定性冒烟，不能作为精度结论**。要支撑"毫米级精度"声明，尚需：

1. **带已知毫米位移真值的实验数据**（静态标靶 + 已知平移/旋转，或动态滑轨），这是当前最大缺口。
2. **人工左右像素对应真值**（`annotate_gt`）覆盖动态序列，用于评价匹配误差。
3. **静态参考点**（≥3 个非共线）验证相机补偿（`car.avi` 当前 3 个点均为测量点，补偿不可成功）。
4. **端到端合成序列精度测试**（§7 Test F），量化完整管线的位移恢复误差。
5. 正式实验协议：≥3 段/场景、5 次计时重复、序列级配对 bootstrap（代码已支持）。

## 7. 第二阶段建议的修复顺序与 Synthetic Test 计划

### 修复优先级

1. **P2-1（OpenCV 5.0 兼容性）**：两处 `stereoRectify` 的 T reshape。修复后当前环境 20 个失败全部恢复 → 建立干净基线。
2. **P1-1 / P1-2（辅助研究包 bug）**：若论文引用创新点 1/2 辅助实验，必须修复（含修正被固化的测试）。
3. **P2-2/3/4/5（消融语义）**：不修改算法，在论文方法与消融设计章节明确归因边界；`full_no_recovery` 建议改名或同时关闭匹配器级恢复。
4. **P2-6（GUI 参数）**：GUI 暴露自适应半径或隐藏无效控件。
5. **P2-7（指标分母）**：统一或文档化。
6. **P2-8 / P3 / P4**：按论文引用情况逐步清理。

### 建议新增 Synthetic Ground-Truth 测试（第二阶段实施）

| 测试 | 内容 | 目的 |
|---|---|---|
| **A** | 已知 f、B、d，验证 Z=fB/d 与 Q 重投影一致（含不同 d） | 几何闭环 |
| **B** | d ∈ {5,10,20,50}px 的 3D 重建 | 深度标度 |
| **C** | 已知亚像素视差 10.25/10.50/10.75，测 subpixel estimator | 亚像素精度（IC-GN 已覆盖，补充连续/抛物线） |
| **D** | 人为生成 LK 平移 (dx,dy)=(2,1)，测光流 | LK 闭环（本审计已执行，建议固化为测试） |
| **E** | 人为加入相机平移，验证补偿 | 补偿闭环（本审计已执行，建议固化为测试） |
| **F** | **端到端**：合成双目序列（已知内参/基线/3D 点/真实位移/相机运动/噪声），跑完整 `TemporalStereoPipeline`，量化位移恢复误差 | 论文精度核心证据 |

> 注意：现有 `stereo_dynamic_measurement/simulation/` 已具备合成双目场景生成能力（`stereo_scene.py`、`trajectory_generator.py`），可作为 Test F 的构造基础，但需确认其合成图像纹理对 `stereo_research` 的 LK/SGBM 足够友好。

---

## 8. 审计方法说明

- 代码逐模块通读：`stereo_research` 全部 27 个 `.py` + `gui/` 9 个 `.py`；`stereo_dynamic_measurement` 43 个 `.py`（经子代理逐文件核对）。
- 独立数值验证脚本（作者执行，位于临时目录，未写入项目）：几何/Q/深度、IC-GN、Kalman、相机补偿、RANSAC、LK、循环一致性，全部通过。
- 两个并行子代理交叉审计 `stereo_dynamic_measurement` 与 `gui/`+`tests/`，发现均已人工复核确认。
- 未修改任何项目文件；`backup_before_refactor/`、实验数据、GT 标注均未触碰。

*— 审计结束，等待第二阶段修复授权 —*
