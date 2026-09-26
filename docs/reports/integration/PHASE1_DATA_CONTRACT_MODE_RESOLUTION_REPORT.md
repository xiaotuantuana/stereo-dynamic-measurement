# Phase 1：数据契约与模式解析报告

日期：2026-08-25  
状态：**PASS**

## 冻结身份

- `WORKTREE_SNAPSHOT_ID`：`b9d949cbcf62e0b431c249b1607b8a733ae2a4a5e816c83168d8c1ac1197f6e6`
- `BASELINE_CONTENT_SNAPSHOT_ID`：`f9fda4fb65b027d23f7b4e882b57f99067917b3920e445250666113f35e3b998`

工作区在 Phase 0 时已是 dirty。本阶段未执行 `stash`、`reset`、`restore`、`checkout`、`clean` 或 `commit`；也没有将已有用户改动归入本 Phase。

## 实现范围

允许且实际修改的生产接口仅为：

- `stereo_research/models.py`：增加 `SystemMode`、`Innovation2Mode`、`Innovation3Mode`、确定性 `resolve_system_mode`、`ResolvedSystemMode`，以及 `FramePointResult` 的只读坐标生命周期视图。
- `stereo_research/runner.py`：作为 CSV 兼容性核对对象；本 Phase 最终没有对其留下行为或 schema 改动。
- `tests/test_phase1_data_contract.py`：新增 9 个 Phase 1 契约测试。

未修改：`pipeline.py`、GUI、LK、预测、local matching、subpixel、confidence、triangulation、Kalman、相机补偿、Innovation 2 数学或 Innovation 3 数学。

## 模式解析契约

`SystemMode` 与论文 `MethodProfile`/M0–M3 分离。`resolve_system_mode(SystemMode.BASELINE)` 严格返回 `MatcherConfig()` 冻结值：prediction/flow ON，target-accuracy policy OFF，physics shadow ON，fault shadow/diagnose ON，两个写 FINAL 权限 OFF，且保留 `closed_loop` 与 `multi_reference_rigid`。

`BASELINE` 不接受任何 legacy `MatcherConfig` 覆盖；冲突会明确抛出 `ValueError`，不会静默关闭 Shadow。`INNOVATION1` 与 `ENHANCED_SHADOW` 保留冻结的 I2 Shadow/I3 Diagnose；二者只表达 Phase 3 统一 I1 能力的授权意图，不给 I2/I3 新的 FINAL 权限。`FULL_ENHANCED` 当前一律抛出 `PermissionError`，不会降级运行。

## 坐标生命周期与 FINAL

`FramePointResult` 现在通过只读视图明确提供：

- `triangulated_xyz_m`：现有 `measured_*` 的直接三角化结果；
- `pipeline_baseline_xyz_m`：优先既有 `compensated_*`，否则 `estimated_*`，并以 `pipeline_baseline_source` 说明来源；
- `candidate_xyz_m`：既有 `candidate_corrected_*`；
- `final_xyz_m`：既有 `final_*`。

没有新增或重命名模糊的 `RAW XYZ`。`FINAL_RESULT_OWNER` 固定为 `TemporalStereoPipeline`；结果对象拒绝其他 owner 名称。Innovation 2 的 candidate 与 FINAL 独立，Innovation 3 仍只能产生诊断/建议，不能写 XYZ。

## CSV 兼容性决策

第一轮实现曾尝试追加生命周期 CSV 列；既有兼容性测试证明这会破坏冻结的列尾顺序和 Shadow 隔离比较。因此该尝试已在同一 Phase 内撤回。正式 CSV 列、顺序与既有语义均未改变；新生命周期契约仅作为内存中的只读接口存在。此决策避免对 GUI、已有研究 CSV 和论文脚本造成隐性兼容性破坏。

## 测试与回归

| 检查 | 结果 |
| --- | --- |
| TDD RED | 新测试在接口不存在时以 `ImportError` 失败 |
| Phase 1 新测试 | 9 passed |
| Phase 1 + Shadow/CSV/后向兼容 focused tests | 33 passed |
| 全量 pytest | **280 passed in 38.79s**（原 271 + 新 9） |
| Python 语法检查 | `models.py`、`runner.py` 通过 `py_compile` |
| 短窗口 GUI 回归（250–260） | 33/33 行有效；直接 `MatcherConfig()` 与 BASELINE resolver 的非性能字段差异 0 |
| 长窗口 GUI 回归（250–350） | 303 行、302 有效；两配置非性能字段差异 0 |

两项 GUI 回归均通过 `StereoVideoProcessor`、`research_full`、`car.avi`、`configs/calibration_640x480.json` 和 P1/P2/P3 运行。比较排除冻结报告已定义的墙钟字段：`flow_ms`、`matching_ms`、`total_ms`、`frame_total_ms`。

## 内容 Hash 差异

下列文件相对 Phase 0.1 content manifest 已变化；该差异包含本 Phase 允许接口工作，也可能包含冻结前就已存在的用户未提交改动，因此不以 Git diff 行数归因。

| 文件 | Phase 0.1 SHA256 | 当前 SHA256 |
| --- | --- | --- |
| `stereo_research/models.py` | `20fda2bbf1bdbca8db7c4ad73ef0cd9e5d9eb4de3e81efbef6e3dabfd8ed79b2` | `a91778e70f8717131f0dbf00e987e88c63a9988ae64cf1ac36901034a75b7c22` |
| `stereo_research/runner.py` | `0810e755c5f565ceba5af659a39b49b72f7173e6c44b88d53c22689423d6d1fc` | `f724ac8bc3caaf3955e064f6fdb5c152d7c8ba78cd3cdfe388ec080d9fc1b283` |

## 结论

Phase 1 仅建立数据契约和模式解析；没有改变生产测量算法或 GUI。I2 Controlled 和 I3 Controlled Recovery 仍未授权，`FULL_ENHANCED` 被明确拒绝。Phase 在此停止，等待人工审核；下一阶段仅可为 Phase 2（Innovation 1 Capability Mapping）。
