# Phase 1.5 Plan: Failure Analysis and Confidence Calibration

## Scope

保持 Dataset Benchmark、GUI、人工测距和 `TemporalStereoPipeline` 默认行为不变。
所有算法性尝试必须通过默认关闭的实验配置启用；Ground Truth 只进入实验评价，绝不进入正式预测或有效性判断。

## Ordered Tasks

1. [x] 冻结 `results/phase1_smoke` 到独立快照，并记录 Git/hash。
2. [ ] 用运行时可得证据重放 90 个 smoke 点，输出完整诊断、Top-20 图和拒绝质量分析。
3. [ ] 增加 CER、accepted accuracy、coverage 和 confidence threshold curve 测试及实现。
4. [ ] 用失败测试证明初始化 confidence 恒为 1.0，再实现默认关闭的初始化置信度校准。
5. [ ] 重跑同一 10 对 smoke，逐点核验三个 catastrophic mismatch，并输出 before/after。
6. [ ] 固定生成 Development=500、Validation=500 manifests；禁止把 val 称为 Test。
7. [ ] 只在 Development 子集做 coarse sensitivity，在 Validation 比较少量候选。
8. [ ] 审计当前静态初始化路径可识别的 Innovation1 消融项；不可识别项明确标注，不伪造结果。
9. [ ] 写 optimization log、独立实验完整性审计并运行完整 pytest。

## Checkpoints

- Failure analysis checkpoint：诊断 CSV 与图片能够解释三个严重误匹配。
- Confidence checkpoint：CER_10 降低且 smoke coverage 不出现灾难性下降。
- Phase 2 gate：manifest 固定、指标齐全、Validation 不参与规则开发。

## Risks

- 当前扁平 FlyingThings3D 子集没有 scene 身份；manifest 只能固定分层等距抽样，不能声称 scene-disjoint。
- 每个样本只调用 `initialize()`；LK、FB、temporal 和多数 subpixel 路径不会执行，因此完整 Innovation1 消融不可由该基准识别。
- 工作树已有大量用户改动；本轮不提交、不覆盖无关文件。

