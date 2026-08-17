# 双目动态位移测量：任务 1 设计

## 范围

本阶段在既有 `stereo_research` 工程旁新增 `stereo_dynamic_measurement` 公共基础层与仿真平台。既有时空匹配、GUI 和未提交修改均不改动。后续三个创新点将以本阶段产生的标准化仿真数据为共同输入。

## 坐标、单位与约定

- 世界坐标、相机平移、三维轨迹和重建结果：mm。
- 时间：s；相位和内收角：rad；像点：pixel。
- 左相机光心为世界原点，X 向右、Y 向下、Z 向前。
- 外参 `(R, T)` 将左相机坐标转换至右相机坐标；`T` 一律为 mm。

## 模块边界

- `calibration.camera_model`：存储、验证和读取双目标定参数；支持 YAML、JSON、OpenCV XML 和 NPZ。
- `calibration.triangulation`：以相机投影矩阵对对应像点进行 DLT 三角化，并输出 mm 单位坐标。
- `simulation.trajectory_generator`：只负责生成四测点的已知动力学真值，不进行任何视觉计算。
- `simulation.stereo_scene`：只负责由真值和相机模型投影左右点并注入可控噪声。
- `simulation.synthetic_dataset`：编排仿真、重建、误差表、二进制数据集和摘要。
- `visualization`：从输出数据绘制轨迹与误差，不改变测量或仿真结果。

## 数据流

`TrajectoryGenerator` 生成 `(frame, time_s, point_id, X/Y/Z_gt_mm)`，由 `StereoScene` 生成理想与含噪左右像点；`triangulate_points` 从含噪点重建 `XYZ_estimated_mm`；数据集模块输出逐轴误差与 `rmse_3d_mm`。零噪声场景用于几何正确性验证，噪声场景用于后续鲁棒性实验。

## 默认仿真

默认 640×480、焦距 800 px、基线 120 mm、零内收角、30 fps、5 s、基准深度 2000 mm。四测点 P1–P4 具有同一主频、不同振幅和相位，且基准空间偏移固定。轨迹类型支持正弦、多频、冲击加指数衰减振动。

## 验收标准

1. 标定模型能从 YAML/JSON/XML/NPZ 加载所需字段，拒绝缺失或非 mm 单位的输入。
2. 零噪声投影/三角化的最大三维误差小于 `1e-6 mm`。
3. 所有三类轨迹与四点空间/频率约束可由单元测试验证。
4. 一键命令生成 CSV、NPZ、JSON 摘要和 PNG 图；摘要包含 `error_x_mm`、`error_y_mm`、`error_z_mm`、`rmse_3d_mm`。
