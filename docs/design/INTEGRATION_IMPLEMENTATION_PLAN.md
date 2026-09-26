# 一体化系统分阶段实施计划

本计划只定义后续工作；当前不实施生产代码。每个 Phase 完成后都运行 `python -m pytest`，记录结果并以小型、可回滚提交保存。

## 依赖图

```text
Phase 0 → 1 → 2 → 3 → 4 → 4.5 → 5 → 6 → 7 → 8 → 10
                                      └────────→ 9 → 10
```

## Phase 0：当前基线冻结

- 目标：冻结当前 `TemporalStereoPipeline` 的生产行为；它是工程 `BASELINE`，不是论文 M0 消融方法。
- 修改范围：仅冻结清单、哈希/元数据、测试报告。
- 不修改：生产算法、GUI、输出含义。
- 实现步骤：记录 HEAD、`git status`、diff/stat、modified/untracked 清单；记录核心文件和配置 hash、Python/依赖环境、pytest 结果、代表性输入与输出的元数据/hash。
- Git 安全：绝不对用户已有未提交修改执行自动 `stash`、`reset`、`checkout` 或 `commit`；冻结报告只记录工作区事实，后续比较必须引用该快照身份。
- 验收：全量 pytest；固定输入下工程 BASELINE CSV 对比策略已定义；论文 M0–M3 消融标签单独记录且不混用。
- 风险/回滚：dirty 工作区导致不可重现；报告不一致即停止后续改动，不尝试清理用户工作区。
- 输出：`BASELINE_FREEZE_REPORT.md`。

## Phase 1：统一数据契约和模式解析

- 目标：在现有 `MatcherConfig`/`FramePointResult` 中表达模式、`TRIANGULATED`/`PIPELINE_BASELINE`/CANDIDATE/FINAL、审计字段和配置验证；`BASELINE` resolver 必须严格复现 Phase 0 的 GUI `research_full` + `MatcherConfig()`，包括既有 prediction/flow、I2 Shadow 与 I3 Diagnose。
- 修改范围：模型、配置解析、最小序列化与 focused tests（3–5 文件/任务切分）。
- 不修改：匹配数学、GUI 布局、shadow 行为。
- 验收：`BASELINE` 与冻结输出和冻结 flag 状态逐项等价；其 I1=OFF 仅禁用 Phase 3 新增/统一能力，不得关闭既有 prediction、LK、local matching、subpixel、confidence；非法 `FULL_ENHANCED` Controlled 配置被明确拒绝。
- 风险/回滚：CSV 兼容性；新增字段只能追加，失败则回退该契约提交。
- 输出：模式与数据契约测试报告。

## Phase 2：Innovation 1 功能映射

- 目标：逐项比较两套预测、LK-FB、匹配、亚像素、置信度、搜索、fallback 和三角化实现。
- 修改范围：只读分析、可重复数值对照测试与映射报告。
- 不修改：任何生产选择逻辑。
- 验收：每一项有输入、输出、状态语义、数学/参数差异和采用决定。
- 风险/回滚：误把同名功能视为等价；无通过证据则维持主链实现。
- 输出：`INNOVATION1_CAPABILITY_MAPPING.md`。

## Phase 3：Innovation 1 正式统一

- 目标：仅把 Phase 2 证明缺失且可适配的能力以小接口引入 `TemporalStereoPipeline`。
- 修改范围：主 Pipeline、现有辅助函数、feature flag、系统级测试。
- 不修改：独立实验 pipeline、GUI 架构、Innovation 2/3 FINAL 权限。
- 验收：INNOVATION1 路径真实可达；BASELINE 等价；M0–M3 消融可运行。
- 风险/回滚：状态双写和数值漂移；按功能单独提交、失败即关闭 flag。
- 输出：集成/消融对比报告。

## Phase 4：Innovation 2 Controlled 接口（不转正）

- 目标：定义版本化 `CorrectionGatePolicy`、拒绝未授权 Controlled 配置、记录决策原因。
- 修改范围：策略契约、主链 gate 接口、测试。
- 不修改：阈值常量、FINAL 写入、PSD 实时实现。
- 验收：SHADOW FINAL 隔离；Controlled 没有经批准策略时明确失败。
- 风险/回滚：静默降级；配置错误必须失败而非伪装 FULL_ENHANCED。
- 输出：Controlled 接口安全报告。

## Phase 4.5：预注册验收协议

- 目标：在查看最终验收数据前，冻结 Innovation 2 Controlled 与 Innovation 3 Controlled Recovery 的通过/失败定义。
- 修改范围：版本化预注册协议、数据划分清单、指标定义、阈值来源与统计规则。
- 不修改：生产 gate 授权、任何 FINAL 写入或自动恢复。
- 验收：协议明确训练/标定、开发、最终验收数据边界；定义 Innovation 2 的改善与灾难性误校正标准，以及 Innovation 3 的成功、失败、回滚和 SAFE 标准。
- 风险/回滚：事后调整阈值；任何修改必须形成新协议版本，旧版本保留。
- 输出：`CONTROLLED_ACCEPTANCE_PREREGISTRATION.md`。

## Phase 5：Innovation 2 离线验收与策略标定

- 目标：按已预注册协议，以版本化数据集评估 `PIPELINE_BASELINE` 与 CANDIDATE 的 RMSE、MAE、P95、最大误差、跳变率、连续性、有效率、错误/灾难性校正率。
- 修改范围：验证数据清单、离线评价、策略产物与报告。
- 不修改：生产 gate 的授权状态，除非人工批准标定结果。
- 验收：覆盖正常、速度变化、遮挡、低纹理、模糊、噪声、视差突变、跟踪失败；结果达到 Phase 4.5 已冻结的科学标准。
- 风险/回滚：数据泄漏或只挑有利样本；失败则保持 SHADOW。
- 输出：`INNOVATION2_CONTROLLED_ACCEPTANCE_REPORT.md` 与可追溯策略版本。

## Phase 6：Innovation 3 Controlled Recovery

- 目标：实现 Level 0/1/2 策略、`RecoveryRequest`、恢复预算、验证、回滚和审计日志。
- 修改范围：主 Pipeline 的明确恢复接口、策略、安全状态与测试。
- 不修改：Level 2 自动执行、未经验证的动作。
- 验收：diagnose→RecoveryRequest→execute→verify；失败回滚不污染有效状态；无无限恢复循环；成功/失败标准遵循 Phase 4.5 预注册协议。
- 风险/回滚：错误恢复扩大故障；默认 DIAGNOSE，Level 1 逐项授权。
- 输出：恢复安全与回滚报告。

## Phase 7：Innovation 1 → 2 → 3 系统链

- 目标：在授权 Controlled 功能存在时形成单主链系统级流程。
- 修改范围：模式解析、阶段连接和端到端测试。
- 不修改：新建 pipeline 或绕开 FINAL 所有者。
- 验收：完整输入→I1→TRIANGULATED→PIPELINE_BASELINE→I2 CANDIDATE→I3 RecoveryRequest→FINAL 审计链；每种模式结果/权限符合规格。
- 风险/回滚：阶段耦合；将每阶段 flag 独立关闭。
- 输出：系统集成验证报告。

## Phase 8：GUI 一体化展示

- 目标：在现有 GUI 内展示同一主链的 `TRIANGULATED`/`PIPELINE_BASELINE`/CANDIDATE/FINAL、状态和审计信息。
- 修改范围：现有窗口、worker preview、非交互 GUI tests。
- 不修改：GUI 主框架、算法控制权、实验入口。
- 验收：不伪造未运行的 PSD/phase/coherence；明确标记 Shadow/Controlled 状态。
- 风险/回滚：将候选误呈现为最终值；最终值标识作为必测项。
- 输出：GUI 验证报告。

## Phase 9：真实双目相机数据源

- 目标：实现已设计的 `StereoDataSource` 相机后端，同时不改变 VideoSource 行为。
- 修改范围：数据源、同步、生命周期、硬件 smoke tests。
- 不修改：测量算法、视频语义、无硬件环境的 CI。
- 验收：视频回归等价；相机帧对时间戳、同步、标定尺寸和关闭异常可验证。
- 风险/回滚：硬件差异；后端独立关闭并回退视频源。
- 输出：硬件兼容性报告。

## Phase 10：整机系统验收

- 目标：确认各模式、GUI、CSV、审计日志、恢复安全和论文可复现实验一致。
- 修改范围：只修复验收发现的已批准缺陷。
- 不修改：科学阈值或结果定义的临时放宽。
- 验收：全量 pytest、代表性视频、受控故障、离线指标、人工 GUI 核验全部通过。
- 风险/回滚：范围膨胀；任何未验收 Controlled 功能退回 Shadow/Diagnose。
- 输出：最终系统验收报告与可复现实验清单。

## 实施任务粒度与检查点

每个 Phase 拆为 1–5 个小任务：先写失败测试，再实现最小行为；每两项任务全量回归；每个 Phase 都需人工审阅输出报告后才能进入下一 Phase。Phase 3、5、6、7、8、9、10 是显式人工批准关卡。

## 当前下一步

实施前先执行 Phase 0；随后 Phase 1。不得跳过 Phase 2 而直接把创新点一独立 pipeline 接入，也不得跳过 Phase 5/6 而授权 FULL_ENHANCED 的 Controlled 行为。
