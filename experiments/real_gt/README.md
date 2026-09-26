# E0/E1 真实实验目录

本目录只保存真实实验模板、独立 GT、对应标定和离线评价结果。当前没有真实数据，正式状态为 `REAL_DATA_REQUIRED`。

E0 用于固定距离、固定标定条件下的静态 XYZ 精度与不确定度评价。E1 用于距离、固定物理 baseline、目标 Z sigma 和方法之间的对比。E1 每个 baseline 必须使用为该 baseline 采集的 calibration；软件会比较 manifest baseline 与 calibration `abs(T_x)`，不一致即拒绝。

measurement 和 GT 必须分离：measurement CSV 由测量 pipeline 产生，GT CSV 由独立仪器或人工记录产生，pipeline 不读取 GT。`split=calibration` 的 sequence 只能用于离线拟合；`split=evaluation` 只能用于独立报告，同一个 sequence ID 不能跨 split 复用。

数据放置：

- E0 标定放 `E0_static/calibration/`，GT/manifest 放 `E0_static/evaluation/`，模板在 `E0_static/templates/`，输出在 `E0_static/results/`。
- E1 对应目录同理。
- `synthetic_demo/` 只验证软件流程，所有文件标记 `SYNTHETIC_DRY_RUN`，不得并入真实论文结果。

运行命令：

```powershell
python validate_real_experiment.py --manifest experiments/real_gt/E0_static/evaluation/e0_manifest.json
python evaluate_e0.py --manifest experiments/real_gt/E0_static/evaluation/e0_manifest.json --output experiments/real_gt/E0_static/results

python validate_real_experiment.py --manifest experiments/real_gt/E1_distance_baseline/evaluation/e1_manifest.json
python evaluate_e1.py --manifest experiments/real_gt/E1_distance_baseline/evaluation/e1_manifest.json --output experiments/real_gt/E1_distance_baseline/results
```
