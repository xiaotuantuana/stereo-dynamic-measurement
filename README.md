# 双目视觉动态测量硕士论文工程

本工程用于双目视觉动态测量研究，保留科研算法、动态测量应用、实验数据和论文归档材料的清晰边界。

## 工程结构

- `stereo_research/`：核心科研算法、评估流程与 GUI 实现。
- `stereo_dynamic_measurement/`：动态测量应用层、相机标定、仿真和三个创新点模块。
- `configs/`：标定和实验配置。
- `tests/`：自动化回归、创新点和 GUI 非交互测试。
- `experiments/`：实验清单、原始/标注数据模板、结果与图表。
- `outputs/`：程序生成的实验、审计和验证输出；重要结果应按需纳入版本控制。
- `docs/`：科研说明、审计报告、测试报告和开发记录。
- `archive/`：历史备份，仅供追溯，不作为当前运行代码。

## 三个创新点

- 创新点一：`stereo_dynamic_measurement/innovation1/`，包含运动预测、LK-FB、局部匹配、亚像素与置信度相关实现。
- 创新点二：`stereo_dynamic_measurement/innovation2/`，包含物理一致性、PSD/相位分析、置信度与轨迹校正实现。
- 创新点三：`stereo_dynamic_measurement/innovation3/`，包含故障指纹、规则诊断、参考监测与恢复验证实现。

核心研究模块和动态测量模块保持分离，以避免目录整理改变现有 Python 导入和运行行为。

## 环境安装

```bash
pip install -r requirements.txt
```

科研与 GUI 环境也可使用：

```bash
pip install -r requirements_research.txt
```

## 常用入口

```bash
pytest
python run_simulation.py --config configs/simulation.yaml
python run_innovation1_experiment.py --config configs/innovation1_experiment.yaml
python run_innovation2_validation.py --config configs/innovation2_validation.yaml
python benchmark_faults.py
```

真实实验预检与离线评估入口见 `experiments/real_gt/README.md`；GUI 可运行 `启动双目测量软件.bat`，或执行 `python -m stereo_research.gui`。

历史研究说明见 `docs/README_RESEARCH.md`，结构整理记录见 `docs/PROJECT_STRUCTURE_REFACTOR_REPORT.md`。
