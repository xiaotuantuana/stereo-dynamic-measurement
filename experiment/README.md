# Dataset Benchmark Phase 1

本目录是现有双目工程的附加实验层。正式测量仍由
`stereo_research.pipeline.TemporalStereoPipeline` 负责；这里不实现第二套双目匹配算法，
也不向原始数据目录写入文件。

## 数据审计

```powershell
python tools/audit_dataset.py
python tools/audit_dataset.py --rebuild-index
```

首次扫描生成 `cache/dataset_index.jsonl`，以后默认复用。审计输出位于
`results/dataset_audit.csv` 和 `results/dataset_summary.md`。

## Benchmark

```powershell
python run_benchmark.py --mode smoke
python run_benchmark.py --mode development --max-frames 100
python run_benchmark.py --mode full --dataset FlyingThings3D_subset
python run_benchmark.py --resume --run-dir results/phase1_smoke
```

默认模式始终是 `smoke`。可以组合使用 `--dataset`、`--sequence`、
`--max-frames`、`--start-frame`、`--end-frame` 和 `--sample-rate`。

当前 FlyingThings3D 子集只有视差 GT，没有项目内可验证的对应标定，因此仅报告视差指标。
它的扁平帧编号不会被假定为连续序列，也不用于时序、深度或三维结论。
