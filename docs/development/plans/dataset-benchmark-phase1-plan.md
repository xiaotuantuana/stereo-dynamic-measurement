# Implementation Plan: Dataset Benchmark Phase 1

## Overview

在不修改 GUI 和 `TemporalStereoPipeline` 正式测量主链的前提下，新增轻量实验层：缓存数据集索引、输出真实数据审计、通过 Adapter 统一样本、以小规模 smoke 模式调用主 Pipeline、计算基础视差指标并支持逐样本断点续跑。

## Architecture Decisions

- 新代码放在 `experiment/`，只调用现有 `stereo_research.pipeline.TemporalStereoPipeline`，不复制匹配算法。
- 索引只保存路径和轻量元数据；图像与 PFM Ground Truth 在样本运行时按需读取。
- 自动识别通用左右目录；对当前 FlyingThings3D 分离图像/视差根目录提供专用 Adapter。
- 当前数据无标定，因此审计明确标注“仅可做 disparity”；深度指标仅在存在可信标定/深度 GT 时启用。
- 默认 `smoke`，禁止默认全量；每个完成样本立即追加结果并写 checkpoint。

## Completed Tasks

- [x] 为索引、配对和审计行为编写失败测试。
- [x] 实现 DatasetSample、Adapter 注册和增量索引缓存。
- [x] 实现 dataset audit CSV/Markdown 与 CLI。
- [x] 对 `E:/数据集` 生成真实索引和审计结果。
- [x] 为 metrics、筛选和 resume 编写失败测试。
- [x] 实现按需加载、PFM 读取、基础 metrics 和结果写入。
- [x] 实现调用 `TemporalStereoPipeline` 的 smoke runner 与 Windows CLI。
- [x] 用少量真实样本运行 smoke benchmark，并验证 resume 不重复执行。
- [x] 完整 pytest（含 GUI 回归）通过。

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 工作树已有大量用户修改 | 高 | 仅新增实验层；不覆盖、不提交用户改动 |
| FlyingThings3D 子集缺少标定 | 高 | 只报告视差指标；深度能力标记不可用 |
| 目录扁平化导致序列身份未知 | 中 | 标记为非连续样本，不用于时序指标 |
| 全量扫描耗时 | 中 | 首次只读元数据生成 JSONL，后续复用并支持 `--rebuild-index` |

## Phase 2 Gate

Phase 2 前需要确认对应相机参数来源，并先冻结 Development/Validation/Test 的无泄漏划分策略。

