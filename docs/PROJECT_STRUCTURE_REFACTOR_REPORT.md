# 工程目录整理报告

整理日期：2026-08-25  
范围：仅目录、文档入口和忽略规则；未改变科研算法、参数、GUI、Python 导入或命令行参数。

## 1. 整理前问题

- 根目录混有代码入口、审计报告、测试报告、实验实施记录和开发记录。
- `backup_before_refactor/` 与当前代码并列，容易被误用。
- 开发计划位于根目录 `tasks/`，与可运行工程内容混杂。
- Python 缓存目录存在于源码、测试和备份中。
- `.gitignore` 原先整体忽略 `outputs/`，不利于有意纳入可复现实验结果。

## 2. 实际移动文件清单

| 原位置 | 新位置 | 类型 | 移动原因 |
| --- | --- | --- | --- |
| `CHANGES.md` | `docs/development/changes/CHANGES.md` | 开发记录 | 移出根目录 |
| `FIX_NOTES_2.md` | `docs/development/fix_notes/FIX_NOTES_2.md` | 修复记录 | 按内容归档 |
| `IMPLEMENTATION_NOTES.md` | `docs/development/implementation_notes/IMPLEMENTATION_NOTES.md` | 实现记录 | 按内容归档 |
| `GUI_TEST_REPORT.md` | `docs/reports/testing/GUI_TEST_REPORT.md` | 测试报告 | 统一测试报告 |
| `RESEARCH_FULL_TEST_REPORT.md` | `docs/reports/testing/RESEARCH_FULL_TEST_REPORT.md` | 测试报告 | 统一测试报告 |
| `TEST_REPORT.md` | `docs/reports/testing/TEST_REPORT.md` | 测试报告 | 统一测试报告 |
| `CODE_AUDIT_REPORT*.md` | `docs/reports/code_audit/` | 代码审计 | 统一审计报告 |
| `FIRST_ROUND_SHADOW_MODE_REPORT.md` | `docs/reports/experiments/` | 实验实施报告 | 统一实验报告 |
| `SECOND_ROUND_INNOVATION1_REPORT.md` | `docs/reports/experiments/` | 实验实施报告 | 统一实验报告 |
| `THIRD_ROUND_UNCERTAINTY_EVALUATION_REPORT.md` | `docs/reports/experiments/` | 实验实施报告 | 统一实验报告 |
| `FOURTH_ROUND_REAL_EXPERIMENT_PREFLIGHT_REPORT.md` | `docs/reports/experiments/` | 实验实施报告 | 统一实验报告 |
| `README_RESEARCH.md` | `docs/README_RESEARCH.md` | 科研说明 | 由根 README 统一索引 |
| `tasks/*.md` | `docs/development/plans/` | 开发计划 | 保留记录但移出根目录 |
| `backup_before_refactor/` | `archive/backup_before_refactor/` | 历史备份 | 与当前工程隔离 |

原本已追踪的文档均使用 `git mv` 移动；当时尚未追踪的报告和备份使用文件系统移动，没有复制或内容改写。

## 3. 没有移动的内容

- `stereo_research/`：核心科研算法和 GUI；保留以防破坏既有 import 与 GUI 资源定位。
- `stereo_dynamic_measurement/`：动态测量、仿真和三个创新点模块；保留其独立职责边界。
- 根目录薄 CLI 入口（`run_simulation.py`、`run_innovation1_experiment.py`、`run_innovation2_validation.py`、`benchmark_faults.py`、`validate_real_experiment.py`、`evaluate_e0.py`、`evaluate_e1.py`、`run_synthetic_dry_run.py`）：已有 README、报告或实验说明直接调用，保留以维持命令入口。
- `configs/outputs/simulation/`：未发现代码文本引用，但它与 `outputs/simulation/` 存在同名结果；为避免误判为冗余或覆盖数据，本轮不移动。
- `experiments/`、`outputs/`、`car.avi`：均可能承载实验可复现性或大体积资产，未改动。
- `docs/superpowers/specs/`：保留既有设计规格的路径，避免破坏历史开发引用。

## 4. 删除内容

未删除任何文件。执行环境拒绝递归删除缓存的命令，因此已通过 `.gitignore` 覆盖 `__pycache__/`、`*.py[cod]` 和 `.pytest_cache/`；可在本机后续手动清理这些可再生缓存。

## 5. 修改的 import 与路径

- Python import：无修改。
- 程序运行路径、GUI 路径、配置路径、输出数据路径：无修改。
- `.gitignore`：增加 IDE/操作系统缓存规则，并仅忽略 `outputs/` 下明确的 `tmp`、`debug` 子目录；保留正式研究输出可被有意纳入 Git 的能力。
- 新增根目录 `README.md`：作为统一索引，不替代现有 CLI。

## 6. 测试结果

| 阶段 | 命令 | 结果 |
| --- | --- | --- |
| 整理前 | `python -m pytest` | 271 passed，39.71 s |
| 整理后 | `python -m pytest` | 271 passed，40.22 s |

## 7. 风险与后续建议

- `tasks/` 中的 Markdown 已移动，目录本身为空；执行环境不允许删除该空目录，可在本机确认无内容后删除。
- `configs/outputs/simulation/` 与 `outputs/simulation/` 的关系需要人工确认，暂列为疑似重复结果而非删除候选。
- 未发现名称含 `_old`、`_new`、`_v2`、`_final`、`_copy`、`_backup` 或 `_temp` 的 Python 重复版本。
- 当前 Git 工作区在整理前已存在多项算法、实验与测试改动；本次未覆盖、还原或合并这些改动。

## 8. 完整性检查

- 创新点一模块仍在 `stereo_dynamic_measurement/innovation1/`，包含预测、LK-FB、局部匹配、亚像素、置信度和基线实验相关实现。
- 创新点二模块仍在 `stereo_dynamic_measurement/innovation2/`，包含物理置信度、PSD/相位分析、轨迹校正与 transient gate。
- 创新点三模块仍在 `stereo_dynamic_measurement/innovation3/`，包含 fault fingerprint、规则诊断、reference monitor 与 recovery 验证。
