# Phase 0：SYSTEM BASELINE 冻结报告

冻结日期：2026-08-25  
状态：**PASS**  
范围：记录当前工作区中 `TemporalStereoPipeline` 的实际生产行为；未修改生产代码、GUI、创新点行为、配置含义、CLI 或 CSV 语义。

## 1. Snapshot Identity

| 项目 | 值 |
| --- | --- |
| HEAD | `872b8dba6294eade34ca4031237960441227571d` |
| Branch | `research-full-upgrade` |
| WORKTREE | DIRTY |
| WORKTREE_SNAPSHOT_ID | `b9d949cbcf62e0b431c249b1607b8a733ae2a4a5e816c83168d8c1ac1197f6e6` |
| BASELINE_CONTENT_SNAPSHOT_ID | `f9fda4fb65b027d23f7b4e882b57f99067917b3920e445250666113f35e3b998` |

`WORKTREE_SNAPSHOT_ID` 是冻结前 `HEAD + branch + git status --porcelain=v1` 的 UTF-8 SHA256，表示 **Git 状态身份**。本文档及 `outputs/baseline_freeze/` 是冻结动作的产物，不属于该冻结前状态。

`BASELINE_CONTENT_SNAPSHOT_ID` 是 `BASELINE_FILE_MANIFEST.tsv` 中规范化条目（UTF-8、LF、末尾 LF；每行 `relative_path<TAB>file_size<TAB>sha256`）的 SHA256，表示 **实际文件内容身份**；它不覆盖也不替代 `WORKTREE_SNAPSHOT_ID`。

## 2. Dirty Working Tree

冻结时存在 12 项已暂存 rename、26 项未暂存修改和 46 项未跟踪路径。未执行 `stash`、`reset`、`checkout`、`restore`、`clean` 或 `commit`。

- 已暂存：12 个文档 rename（README、开发记录、测试报告、既有计划），无内容行修改。
- 未暂存：`.gitignore`，三个创新点若干模块，`stereo_research` 的主链/GUI/匹配/报告模块及相关测试；`git diff --stat` 为 26 files changed、2296 insertions、258 deletions。
- 未跟踪：整理归档、设计/审计文档、真实实验资料、输出和新科研模块等 46 个路径。

本冻结只记录事实，不尝试将用户原有改动变为干净状态。后续 Phase 必须引用本 Snapshot ID，并在每次比较前报告工作区差异。

## 3. Core File Hashes

| 文件 | SHA256 |
| --- | --- |
| `stereo_research/pipeline.py` | `71F5BB6AB63FD7DCEE27E86239E712327FFB7476B8762B8345F49D3F423DAD99` |
| `stereo_research/models.py` | `20FDA2BBF1BDBCA8DB7C4AD73EF0CD9E5D9EB4DE3E81EFBEF6E3DABFD8ED79B2` |
| `stereo_research/runner.py` | `0810E755C5F565CEBA5AF659A39B49B72F7173E6C44B88D53C22689423D6D1FC` |
| `stereo_research/calibration.py` | `BD50203129CB3F6406E64E3878E998E7C8D2FC167CE7748E73D84FFF14F43DE5` |
| `stereo_research/accuracy_policy.py` | `9DD1D56B1B4FB416A32479393FB33812E97C1FBEEE7C12EB18CD0EA51DD8F943` |
| `stereo_research/local_matching.py` | `AFD7EB5FB44F636EC75B4C852CFF29B43530C3F759D1E13613B7D8F5BEADCCB5` |
| `stereo_research/tracking.py` | `CB2DA9FCE5D03BFDB110BB9E2072FB74E0B04BEBCA8B1D691D5E1EC462963CF9` |
| `stereo_research/confidence.py` | `669BF3C301798A07672A90C1037800522859BA34D1BDDFAFE66B7BCB379CB5B3` |
| `stereo_research/uncertainty.py` | `904C89D29B19ACFD6C8B3602DF64450403A6ECF943220591CA623CEF7642F754` |
| `stereo_research/geometry.py` | `7BAB9209C260B5ABDE37360E5BCA0246DB6A9422770C3744F2195F7C42FEC2A2` |
| `stereo_research/shadow_analysis.py` | `074D4E405C007EC9F6161BEA3AD1866929AB867D0AE22956D45C2961D716606E` |

### 3.1 Baseline File Manifest

- 路径：`docs/reports/baseline/BASELINE_FILE_MANIFEST.tsv`
- 文件数：**146**
- 覆盖：`stereo_research/`、`stereo_dynamic_measurement/`、`configs/`、`tests/`，以及根目录正式 CLI/批处理入口、依赖清单和 pytest 配置。
- 每项：`relative_path`、`file_size`、`sha256`。
- 排除：`__pycache__`、`.pytest_cache`、`*.pyc`/`*.pyo`、临时/再生输出目录、结果/报告目录、墙钟性能日志及其他可再生缓存。

后续实施必须重新计算 manifest，并以其逐文件差异和 `BASELINE_CONTENT_SNAPSHOT_ID` 报告文件内容变化。

## 4. Config Hashes

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| `configs/calibration_640x480.json` | 640×480 双目标定 | `5D4EE7419E752865FEAFDD1455D42C9BA894202CA1B7426670F2A2ACFCBB4C11` |
| `configs/innovation1_experiment.yaml` | 创新点一实验 | `E96F15E819FB482787B8E393AC3D8DAAF2A1070A4365D17A2956C3359451550B` |
| `configs/innovation2_validation.yaml` | 创新点二验证 | `9DDCA01E264DB7322E7CF756B8D2F3B17D3D547049A9444A479B0A8662B25295` |
| `configs/simulation.yaml` | 合成仿真 | `989E4D11464891D3EE4B31E719242A438173698EA7B007B697B883113F9A4CE0` |

## 5. Environment

| 项目 | 冻结值 |
| --- | --- |
| OS | Windows 11 Home Chinese, 64-bit, build 26200 |
| Python | 3.14.5 (`Python 3.14.5 (local installation)`) |
| OpenCV | 4.13.0 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |
| PySide6 | 6.11.1 |
| pytest | 9.1.1 |

`python -m pip freeze` 已在冻结会话执行；关键科研依赖版本为 numpy 2.4.6、opencv-python 4.13.0.92、opencv-contrib-python 5.0.0.93、scipy 1.18.0、pandas 3.0.5、matplotlib 3.11.1、PySide6_Essentials 6.11.1、pytest-qt 4.5.0、PyYAML 6.0.3。

## 6. Pytest Baseline

命令：`python -m pytest`  
结果：**271 passed in 38.58s**；失败 0，跳过 0。

## 7. Representative Inputs

| 输入 | 属性 |
| --- | --- |
| `car.avi` | 67,861,614 bytes；SHA256 `B81F3377F858E6794BE11CB797CDF0C6FB20C28699E3A3D91716E42E4B28C441`；2167 帧；1280×480（左右各 640×480）；20 FPS |
| `configs/calibration_640x480.json` | 上表标定 hash；与视频单目尺寸匹配 |
| `experiments/car/points.json` | 帧 250 的 P1/P2/P3 三测点 |
| `configs/simulation.yaml` | 固定合成/仿真输入配置；hash 见上表 |

## 8. Representative Outputs

通过与 GUI 相同的 `StereoVideoProcessor` 路径运行 `research_full`，`MatcherConfig()` 默认值，帧 250–260，三测点，输出独立于既有实验结果：

| Run | 输出 | SHA256 | 行数/有效行 |
| --- | --- | --- | --- |
| A | `outputs/baseline_freeze/run_a_research_full.csv` | `3B68619B0DC98254C532ACEAA5E17A0FACF07B28880E1FFEDE143FA14DF2E68B` | 33 / 33 |
| B | `outputs/baseline_freeze/run_b_research_full.csv` | `A058948CCAA42869EA0C9289877D91A11E8E7FD62BAB4CF675A525790C8266F0` | 33 / 33 |

每次运行处理 11 帧。A 用时 1.7833672 s，B 用时 1.9545723 s。

## 9. Baseline Repeatability

结论：**PARTIAL（测量语义确定，性能遥测不确定）**。

两次输出均有 199 个字段和 33 行。逐单元比较显示，195 个字段完全一致；仅以下四个墙钟遥测字段不同：`flow_ms`、`matching_ms`、`total_ms`、`frame_total_ms`。因此 CSV 文件 hash 不同，但结果语义字段在该冻结样本上完全一致。

## 10. CSV 等价比较规则

- 精确一致：行身份、`method`、`frame`、`point_id`、`status`、离散状态/布尔/枚举字段，以及当前冻结样本上除性能遥测外的全部数值和文本字段。
- 性能遥测排除：`flow_ms`、`matching_ms`、`total_ms`、`frame_total_ms`。它们只作性能趋势分析，不作基线等价判断。
- 数值容差：本冻结的 A/B 非性能字段实测差为 0；在尚未完成跨平台重复性实验前，不人为设定浮点容差。未来若需跨平台比较，必须先单独测量自然漂移并版本化比较策略。

## 11. SYSTEM BASELINE 定义

SYSTEM BASELINE **严格等价于** Phase 0 冻结的 GUI `research_full` + `MatcherConfig()` 默认生产行为，不是“把 I1/I2/I3 都关闭”的抽象模式，也不是论文 M0。

- prediction / flow：ON / ON；
- Innovation 1 target accuracy policy：OFF；
- Innovation 2 physics shadow：ON；
- Innovation 3 fault shadow / diagnose：ON；
- physics correction to FINAL：OFF；fault recovery to FINAL：OFF；
- 其余冻结默认项：`confidence_mode=closed_loop`、`camera_compensation_mode=multi_reference_rigid`。

因此，Phase 1 的 `BASELINE` mode resolver 必须逐项复现以上状态和原有 `TemporalStereoPipeline` 行为。这里的 “Innovation 1 = OFF” 仅表示**不启用 Phase 3 后新增或统一的 Innovation 1 能力**；绝不能关闭已冻结的 prediction、LK、local matching、subpixel、confidence 或其他既有主链能力。Shadow/Diagnose 仍须运行，但不得写入 FINAL。

## 12. M0–M3 定义

| 标签 | 当前真实含义 | 与 SYSTEM BASELINE 的关系 |
| --- | --- | --- |
| SYSTEM BASELINE | 当前冻结 GUI/`research_full` 生产行为 | 工程回归基线 |
| M0 | 每帧稠密 SGBM | 科研消融 |
| M1 | 光流 | 科研消融 |
| M2 | 光流 + 历史预测 | 科研消融 |
| M3 | 光流 + 预测 + 极线/邻域/亚像素/LR + 自适应搜索 | 科研消融 |

SYSTEM BASELINE 是工程回归对象，锁定 GUI 实际生产组合；M0–M3 是用于论文方法比较的实验 profile。两套命名、开关与验收指标必须分离，任何 `BASELINE` 等价测试均不得被 M0 结果替代。

## 13. Current Innovation Feature Flags

`MatcherConfig()` 的冻结默认状态：

| 项目 | 值 |
| --- | --- |
| prediction / flow | ON / ON |
| Innovation 1 target accuracy policy | OFF |
| Innovation 2 physics shadow | ON |
| Innovation 3 fault shadow / diagnose | ON |
| physics correction to FINAL | OFF（明确拒绝） |
| fault recovery to FINAL | OFF（明确拒绝） |
| confidence mode | `closed_loop` |
| camera compensation mode | `multi_reference_rigid` |

## 14. Known Nondeterminism

仅发现四个墙钟耗时字段具有运行间差异。未发现 frame/result/status/XYZ/disparity/confidence/候选校正/诊断字段的差异。本阶段未为消除这些耗时差异而改变算法。

## 15. Phase 1 Readiness

Phase 0 验收项均满足：工作区、核心代码、配置、环境、测试、代表性输入/输出和等价规则均已冻结；SYSTEM BASELINE 与 M0–M3 已分离。

**Phase 1 可以开始，但本阶段在此停止，等待人工审核。**
