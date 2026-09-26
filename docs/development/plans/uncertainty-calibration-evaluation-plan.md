# 第三轮实施计划

## Phase 1：评价数据合同

- [ ] 定义独立 GT、对齐、EvaluationRecord 和统计 API。
- [ ] 先写数值和 runtime/GT 隔离失败测试。

## Phase 2：测量与 GT 分层

- [ ] 从主 runner 移除 GT 读取/写入；measurement CSV 保持运行时字段。
- [ ] 增加 E0 measurement+GT evaluation runner 和 `DATA_REQUIRED` 路径。

## Phase 3：校准和不确定度输出

- [ ] 实现 raw/calibrated sigma 的离线模型与 sequence-level fit/evaluate 接口。
- [ ] 不改变默认 runtime 行为，未加载模型即 `UNCALIBRATED`。

## Phase 4：U0--U3 与 E1

- [ ] 增加 refinement/retry/quality ablation 的 evaluation-only 输出。
- [ ] 增加 E1 manifest、校准身份 guard 和汇总。

## Phase 5：回归与报告

- [ ] 执行 focused tests、全量 pytest、OFF mode 对比和实验。
- [ ] 写入第三轮报告，如实记录 calibration/refinement/retry 结论和遗留风险。
