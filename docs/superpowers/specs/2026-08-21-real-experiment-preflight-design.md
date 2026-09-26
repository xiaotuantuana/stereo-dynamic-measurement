# 第四轮设计：实验前软件收尾与 Synthetic Dry-Run

## 范围

本轮只新增离线实验软件与模板。`TemporalStereoPipeline`、Stereo Geometry、Q、匹配、IC-GN、Kalman、camera compensation、uncertainty estimator、Innovation 2/3 均不修改。主 measurement CSV 保持 199 列。

## 方案选择

采用“模板目录 + 单一实验服务模块 + 薄 CLI”。备选方案是为 E0/E1 各复制一套 evaluator，或把实验能力塞入 measurement runner；前者容易分叉，后者破坏 runtime/GT 边界。选定方案复用第三轮 `evaluation_runner` 与 `uncertainty_evaluation`，新模块只负责 manifest、校验、聚合、表格和环境快照。

## 目录

```text
experiments/real_gt/
  README.md
  REAL_EXPERIMENT_CHECKLIST.md
  E0_static/{calibration,evaluation,templates,results}/
  E1_distance_baseline/{calibration,evaluation,templates,results}/
  synthetic_demo/{measurement.csv,gt.csv,e0_manifest.json,e1_manifest.json,calibration.json,results}/
```

真实模板不引用 synthetic demo；所有 synthetic 输入与输出带 `data_source=SYNTHETIC_DRY_RUN`。无真实 GT 时统一返回 `REAL_DATA_REQUIRED`，不生成 0 RMSE 或虚假准确率。

## 数据合同

E0 manifest 顶层包含 `experiment_id`、`gt_unit=mm`、`data_source`、`sequences`。每个 sequence 包含 `sequence_id`、`split`（仅 calibration/evaluation）、measurement/GT 路径、alignment、calibration identity、distance、baseline、method、repeat、notes。同一 `sequence_id` 跨 split 重复时 fail-fast。

E1 沿用第三轮 `conditions`，补齐 `sequence_id`、`split`、`gt_unit` 和 data source。baseline 必须与 calibration `abs(T_x)` 在容差内一致。

GT template 使用 `frame,timestamp_s,point_id,gt_x_mm,gt_y_mm,gt_z_mm,gt_sigma_x_mm,gt_sigma_y_mm,gt_sigma_z_mm,data_source`。frame/timestamp 可择一，但每行必须满足当前 alignment mode。XYZ 必须有限；frame 非负整数；timestamp 对同一点严格单调；frame+point 和 timestamp+point 均不得重复。

## 公开 seam

- `load_real_experiment_manifest(path, experiment_type)`
- `validate_real_experiment(path)`：只读，返回 PASS/WARNING/FAIL issues 和 alignment rate。
- `evaluate_e0_manifest(path, output)` / `evaluate_e1_manifest(path, output)`：一键离线评价和聚合表格。
- `write_environment_snapshot(...)`：配置与只读 Git revision/status。
- `run_synthetic_dry_run(root)`：validate → align → evaluate → aggregate → tables。

CLI `validate_real_experiment.py`、`evaluate_e0.py`、`evaluate_e1.py` 和 `run_synthetic_dry_run.py` 只是这些 seam 的命令行适配器。

## 输出

E0：`e0_per_sequence.csv`、`e0_summary.csv`、`uncertainty_calibration.csv` 及四个论文表模板。E1：`e1_per_condition.csv`、`e1_summary.csv`。输出中的 synthetic 行必须保留 `data_source=SYNTHETIC_DRY_RUN`；真实表在没有真实 GT 时只记录状态 `REAL_DATA_REQUIRED`，accuracy 单元格为空。

## 测试与安全

TDD 只覆盖实验软件公共接口：合法 manifest、split guard、GT duplicate/finite、frame/timestamp 对齐、identity guard、REAL_DATA_REQUIRED、完整 dry-run、synthetic 标签、runtime 不读取 GT。最终运行原 247 项和新增测试，并断言 `CSV_FIELDS` 仍为 199。
