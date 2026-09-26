# 真实实验操作清单

1. 固定并测量真实 baseline。
2. 加载该 baseline 对应的 calibration，记录 calibration ID 和文件。
3. 记录 experiment/sequence ID、日期、方法和目标精度。
4. 采集视频，现场记录 fps、分辨率和时间源。
5. 使用独立设备记录 GT，记录设备名称和标称精度。
6. 按 mm 填写独立 `gt.csv`。
7. 填写 E0/E1 manifest，并设置 calibration/evaluation split。
8. 运行 validator，处理所有 FAIL 和 alignment WARNING。
9. 运行 measurement pipeline，保存 measurement CSV 和环境快照。
10. 运行 offline E0/E1 evaluation。
11. 检查 alignment success rate 和被排除样本。
12. 检查并归档自动生成的论文表格 CSV。

实验现场必须记录、事后通常无法可靠补救：真实 baseline、calibration ID、distance、fps、resolution、GT 来源与精度、timestamp/frame 对应关系、相机/GT 是否在同一时间基准、目标点 ID 和任何实验中断。
