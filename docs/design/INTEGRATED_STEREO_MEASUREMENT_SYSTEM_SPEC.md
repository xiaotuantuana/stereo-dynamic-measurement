# 一体化双目动态测量系统设计规格

## 1. 决策与范围

采用“渐进式单主链整合”。`stereo_research.pipeline.TemporalStereoPipeline` 是唯一正式在线测量处理器和唯一 `FINAL` 结果拥有者；禁止创建第二条生产 Pipeline，也不将 `AdaptiveStereoMeasurementPipeline` 嵌套进其中。

本规格定义后续集成边界，不实现算法、GUI 重构或真实相机采集。科研验证入口（仿真、baseline、validation、benchmark、E0/E1）保留为验证层。

## 2. 当前事实

- GUI 主链为 `StereoMainWindow → ProcessingWorker → StereoVideoProcessor → TemporalStereoPipeline`。
- 创新点一：主链仅复用 `innovation1.precision_planner`；独立 `AdaptiveStereoMeasurementPipeline` 未被 GUI 实例化。
- 创新点二、三：`ShadowAnalyzer` 在帧结果完成后运行；候选校正与恢复建议不能改写 `final_*`。
- `MatcherConfig` 已拒绝 `allow_physics_correction_to_final` 和 `allow_fault_recovery_to_final`，这是必须保持到正式验收通过前的安全边界。

## 3. 统一数据契约

复用并最小扩展既有 `MatcherConfig`、`MethodProfile`、`FramePointResult` 与 `PointState`；不创建 `*V2`、`UnifiedPipeline` 或重复测量结果类。

### 3.1 坐标生命周期与结果语义

| 阶段结果 | 现有/目标字段 | 语义 | 写权限 |
| --- | --- | --- | --- |
| `TRIANGULATED` | `measured_*`（必要时新增明确来源标记） | 由已接受匹配直接重投影得到的三角测量结果；不等同滤波或补偿结果 | `TemporalStereoPipeline` |
| `PIPELINE_BASELINE` | `estimated_*`、`compensated_*` 及其状态/来源标记 | 当前 Pipeline 经既有滤波、置信度和可选相机补偿后的基准结果；是创新点二科学比较的准确输入点 | `TemporalStereoPipeline` |
| `CANDIDATE` | `candidate_corrected_*` | Innovation 2 基于 `PIPELINE_BASELINE` 产生的候选校正 | Innovation 2，仅记录 |
| `FINAL` | `final_x_m/final_y_m/final_z_m` | 唯一正式结果，供 GUI、正式 CSV 和论文结果使用 | **仅 `TemporalStereoPipeline`** |

禁止以语义模糊的 `RAW` 统称以上阶段，也禁止将 `TRIANGULATED`、`PIPELINE_BASELINE` 和 `FINAL` 混作科学比较对象。Shadow/Diagnose 模式不得写 FINAL。每个未来 Controlled 决策必须在 `FramePointResult` 记录：`correction_applied`、`correction_reason`、`gate_policy_id`、`gate_policy_version`、`threshold_source`。每个恢复决策必须记录：`system_state`、`recovery_executed`、`recovery_result`、`rollback_performed`。

### 3.2 内部阶段输入输出

| 阶段 | 输入 | 输出 | 是否可改 FINAL | 状态所有者 |
| --- | --- | --- | ---: | --- |
| Innovation 1 | 矫正帧、`PointState`、`MatcherConfig` | 匹配/置信度/精度策略结果 | 是，作为正式测量前端 | `TemporalStereoPipeline` |
| Triangulation | 已接受的匹配 | `TRIANGULATED` XYZ | 是 | `TemporalStereoPipeline` |
| Innovation 2 Shadow | `PIPELINE_BASELINE` 与只读历史 | CANDIDATE、C_phy、transient、残差 | 否 | `ShadowAnalyzer` |
| Innovation 2 Controlled | `PIPELINE_BASELINE`+CANDIDATE+已标定门控策略 | Trusted XYZ 或 `PIPELINE_BASELINE` fallback | 有条件 | `TemporalStereoPipeline` |
| Innovation 3 Diagnose | 结果、诊断历史 | 指纹、故障类别、建议 | 否 | 诊断组件 |
| Innovation 3 Recovery | diagnosis+主链可恢复状态 | `RecoveryRequest`、执行/验证/回滚记录 | 否，永不直接写 XYZ | `TemporalStereoPipeline` |

## 4. 运行模式与 Feature Flag

引入一个面向现有 `MatcherConfig` 的显式模式解析层，而非平行配置体系。解析后必须生成不可歧义的功能开关。

| 模式 | Phase 3 新增/统一 I1 | Innovation 2 | Innovation 3 | FINAL 行为 |
| --- | --- | --- | --- | --- |
| `BASELINE` | OFF（仅新增/统一能力） | SHADOW（冻结既有行为） | DIAGNOSE（冻结既有行为） | **严格复现** Phase 0 GUI `research_full` + `MatcherConfig()`；prediction/LK/local matching/subpixel/confidence 等既有主链能力保持原状；不等同论文 M0 |
| `INNOVATION1` | ON | SHADOW（保留冻结既有行为） | DIAGNOSE（保留冻结既有行为） | SYSTEM BASELINE + Phase 3 正式统一后的 Innovation 1 能力；I2/I3 不获得新的 FINAL 权限 |
| `ENHANCED_SHADOW` | ON | SHADOW | DIAGNOSE | FINAL 仍来自 `PIPELINE_BASELINE`/既有主链最终选择 |
| `FULL_ENHANCED` | ON | CONTROLLED* | CONTROLLED_RECOVERY* | 仅经授权的门控/恢复可影响 FINAL |

`FULL_ENHANCED` 不是静默降级名称。若 Controlled 所需的已验证策略、策略版本或授权不存在，配置验证必须失败并给出原因；调用者须显式选择 `ENHANCED_SHADOW`。`*` 代表后续验收完成后才能开启。

`BASELINE` resolver 的 “I1 = OFF” 只能关闭 Phase 3 后新增/统一的 I1 能力，不能关闭现有 `TemporalStereoPipeline` 的 prediction、LK、local matching、subpixel、confidence 或其他冻结生产能力；同时必须保留当前 I2 Shadow 与 I3 Diagnose，且二者继续不得写 FINAL。论文 M0–M3 是研究/消融方法标签，必须保留为独立的 `MethodProfile`/实验维度；不得用系统模式 `BASELINE` 代替或重命名为 M0。

## 5. Innovation 1 统一策略

| 功能 | `TemporalStereoPipeline` | 独立 innovation1 | 决策 |
| --- | --- | --- | --- |
| 运动预测 | `predict_disparity`、`PointState` 历史 | `MotionPredictor` | 先作数值等价/性能映射；保持主链版本，非等价能力以小接口引入 |
| LK / LK-FB | `track_point_lk`、`track_xy_lk`、FB 阈值 | `track_forward_backward` | 对照失败语义和阈值，避免两条跟踪状态并存 |
| 局部匹配 | `LocalMatcher`/`QualityLocalMatcher` | `LocalStereoMatcher` | 以结果契约比较，不嵌套独立 pipeline |
| 左右一致性 | 主链 LR/cycle 路径 | 独立 matcher 内路径 | 保留主链控制权 |
| 亚像素 | 主链 subpixel/IC-GN 路径 | `refine_parabolic` | 对照数学与回退，再决定适配函数 |
| confidence | `confidence.py`、`uncertainty.py` | 邻域/纹理等分量 | 统一为 `FramePointResult` 字段 |
| 搜索范围 | 主链 adaptive radius | `AdaptiveSearchConfig` | 以消融与 shadow 对照决定 |
| precision planner | 已导入并可选使用 | `precision_planner` | 继续作为唯一共享功能；保留可追溯输出 |
| fallback | SGBM 全局重初始化 | 独立失败处理 | 保留主链回退；不新增并行恢复路径 |
| triangulation | `reproject_point_m` | dynamic camera model | 主链保持唯一正式坐标生成器 |

Phase 2 的输出是逐项数学、状态和输出差异报告；未经该报告与回归证据，任何独立实现不得替换主链能力。

## 6. Innovation 2 状态机与受控校正

`OFF → SHADOW → CONTROLLED`。

- OFF：不创建物理分析器。
- SHADOW：保留当前 `C_phy`、transient、候选校正和 CSV 审计；FINAL 不变。
- CONTROLLED：仅由 `TemporalStereoPipeline` 应用经 `CorrectionGatePolicy` 批准的 CANDIDATE，否则保留 `PIPELINE_BASELINE`。

门控策略必须来自版本化离线标定产物，而非代码常量。其输入至少为 C_phy、measurement confidence、transient、历史长度、校正幅度、候选改善证据和几何有效性；策略元数据必须包含数据集/实验 ID、指标、阈值来源、版本与适用范围。最终验收数据运行前，必须先预注册并冻结通过/失败指标、数据划分、排除规则、阈值及统计规则；标定与最终验收数据不得混用。

PSD/phase/coherence 在正式定位前为离线完整验证证据。在线只可使用已验证的滑动窗适配器；不能伪造“实时频域证据”。

## 7. Innovation 3 状态机与受控恢复

`OFF → DIAGNOSE → CONTROLLED_RECOVERY`。

- DIAGNOSE：指纹、类别、置信度和 `RecoveryRequest`，仅记录。
- CONTROLLED_RECOVERY：`Diagnosis → RecoveryRequest → SafetyGate → Execute → Verify → Continue | Rollback → SAFE`。

Innovation 3 永远不能直接写 XYZ，也不能把候选坐标提升为 FINAL；它只能产生诊断和 `RecoveryRequest`。只有 `TemporalStereoPipeline` 可在授权后执行恢复，并通过重新产生或重新选择测量结果来更新 FINAL。恢复分级：Level 0 仅告警；Level 1 为经验证、幂等且有次数上限的低风险动作（例如使用现有主链重初始化接口）；Level 2 涉及参数/算法/系统重置，默认拒绝自动执行。恢复状态、前后快照、验证结论和回滚都必须审计；同一 fault 的连续尝试需预算和冷却期，禁止无限循环。Controlled Recovery 的成功/失败、回滚和 SAFE 进入标准同样必须在最终验收数据运行前预注册并冻结。

## 8. 系统状态与数据源边界

目标系统状态：`INITIALIZING`、`TRACKING`、`MEASURING`、`WARNING`、`DEGRADED`、`RECOVERING`、`SAFE`、`FAILED`。状态转换由主 Pipeline 拥有；创新组件仅提出证据或建议。

未来定义 `StereoDataSource.open/read_pair/close` 与帧时间戳契约。`VideoSource` 必须完全保持当前视频行为；`StereoCameraSource` 仅定义生命周期、异常、帧同步和标定兼容边界，本阶段不实现硬件采集。

## 9. 非目标

- 不新建第二条生产 Pipeline；
- 不立即解除 shadow 写保护；
- 不实现真实相机或 GUI 信息页；
- 不删除科研实验入口；
- 不以测试通过替代 Controlled 的离线科学验收。

## 10. 验收总则

每个实现阶段需保留 `python -m pytest` 全绿，并新增：baseline 等价、feature flag 可重复、shadow FINAL 隔离、controlled gate fallback、恢复隔离/回滚和输入→1→2→3→FINAL 系统测试。任何不满足的 Controlled 功能必须回到 Shadow/Diagnose。
