# 第四轮实施报告：实验前软件收尾与 Synthetic Dry-Run

## 1. Files Added / Changed

新增：

- `stereo_research/real_experiment.py`：manifest 合同、只读 validator、E0/E1 聚合、论文表格、环境快照与 synthetic dry-run。
- `validate_real_experiment.py`、`evaluate_e0.py`、`evaluate_e1.py`、`run_synthetic_dry_run.py`：薄命令行入口。
- `experiments/real_gt/`：E0/E1 目录、README、GT/manifest/表格模板、环境模板、synthetic demo 和结果。
- `tests/test_real_experiment_preflight.py`：12 个实验软件层测试。
- 第四轮设计、plan、todo 和本报告。

调整：

- `stereo_research/evaluation_runner.py`：离线缺 GT 状态从 `DATA_REQUIRED` 统一为 `REAL_DATA_REQUIRED`。
- `tests/test_e0_e1_experiment_interfaces.py`：同步状态合同。

没有修改第四轮禁止的核心算法文件；主 measurement CSV 仍为 199 个唯一字段。

## 2. Experiment Folder Structure

```text
experiments/real_gt/
  README.md
  REAL_EXPERIMENT_CHECKLIST.md
  experiment_environment_template.json
  E0_static/
    calibration/ evaluation/ templates/ results/
  E1_distance_baseline/
    calibration/ evaluation/ templates/ results/
  synthetic_demo/
    measurement.csv gt.csv calibration.json
    e0_manifest.json e1_manifest.json
    experiment_environment.json results/
```

## 3. E0 Template

`gt_e0_template.csv` 使用 mm，frame 或 timestamp 可按 alignment mode 选择；必须填写 point ID、XYZ 和可选 GT sigma。`e0_manifest_template.json` 需要填写 sequence/split、measurement/GT、alignment、calibration identity、distance、baseline、method 和 repeat。同一 sequence ID 不得跨 calibration/evaluation split。

E0 自动输出 `e0_per_sequence.csv`、`e0_summary.csv`、`uncertainty_calibration.csv` 以及论文表格模板。没有独立真实 GT 时 accuracy 为空，状态为 `REAL_DATA_REQUIRED`。

## 4. E1 Template

`e1_manifest_template.json` 沿用第三轮 conditions，并增加 sequence/split/data-source 合同。每个 condition 绑定 distance、physical baseline、target sigma Z、method、calibration、measurement 和 GT。无论 GT 是否存在，baseline 都会与 calibration `abs(T_x)` 比较；不一致 fail-fast，绝不改 baseline 或 Q。

## 5. Validator

只读检查：manifest/文件/必要字段、split、frame、timestamp 单调性、point ID、finite XYZ、frame+point/timestamp+point duplicate、measurement/GT 对齐率、baseline/calibration identity 和显式 `gt_unit=mm`。输出只有 PASS/WARNING/FAIL；缺真实 GT 是 WARNING + `REAL_DATA_REQUIRED`，数据错误是 FAIL。

## 6. One-Command Workflow

测量仍使用现有 runtime runner：

```powershell
python -m stereo_research.run --manifest <measurement_manifest.json> --methods research_full
```

E0：

```powershell
python validate_real_experiment.py --manifest <e0_manifest.json>
python evaluate_e0.py --manifest <e0_manifest.json> --output <E0_results>
```

E1：

```powershell
python validate_real_experiment.py --manifest <e1_manifest.json>
python evaluate_e1.py --manifest <e1_manifest.json> --output <E1_results>
```

evaluator 会再次运行 validator，故不能绕过 FAIL。表格由 evaluator 同步生成，无需手工复制指标。

## 7. Synthetic Dry-Run

执行命令：

```powershell
python run_synthetic_dry_run.py --root experiments/real_gt/synthetic_demo
```

完整流程 `validate → align → evaluate → aggregate → tables` 已跑通；E0/E1 validator 均为 PASS，alignment success rate 均为 1.0。所有 synthetic 输入和递归输出 CSV/JSON 均明确带 `SYNTHETIC_DRY_RUN`。

示例 E0 结果为 Z RMSE 2.9155 mm、mean estimated sigma Z 4.65 mm；这些数值仅验证软件计算和表格连通性。

**SYNTHETIC ONLY — NOT A REAL EXPERIMENT RESULT。不得用于正式论文实验结论或 runtime calibration factor。**

## 8. Tests

- 第三轮基线：247
- 第四轮新增：12
- Total：259 passed in 41.77s
- Failed：0

## 9. Regression

第四轮没有改变 pipeline、Measurement Policy、Innovation 2/3 Shadow 或 uncertainty estimator；0.05 px floor 仍为 `sqrt(0.0025)`，未应用 synthetic ratio，未拟合 runtime calibration，未执行 active correction/recovery。主 CSV 检查为 `199 / 199 unique`。

## 10. Remaining Work

- `REAL E0 DATA REQUIRED`
- `REAL E1 DATA REQUIRED`
- `REAL GT REQUIRED`

获得真实 video、GT、baseline、distance 和对应 calibration 后，复制模板、填写 manifest 并运行上述命令即可；无需临时修改 pipeline 或评价代码。
