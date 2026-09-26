# 第四轮实施计划

## Phase 1：合同与校验器

- [ ] 写 manifest/GT/identity/split/alignment 的失败测试。
- [ ] 实现只读 validator 与结构化 PASS/WARNING/FAIL。
- [ ] 验证：focused validator tests。

## Phase 2：一键评价与表格

- [ ] 写 E0/E1 REAL_DATA_REQUIRED 和聚合输出失败测试。
- [ ] 复用第三轮 evaluator 实现 E0/E1 workflow。
- [ ] 验证：focused workflow tests，主 CSV 199 列。

## Phase 3：模板、环境快照和 synthetic dry-run

- [ ] 建立真实实验目录、README、checklist、CSV/JSON 模板。
- [ ] 建立显式 SYNTHETIC_DRY_RUN demo 和一键入口。
- [ ] 执行完整 dry-run并检查所有输出标签。

## Phase 4：回归与报告

- [ ] 全量 pytest。
- [ ] 确认核心算法文件无第四轮差异、未 commit。
- [ ] 输出第四轮报告。
