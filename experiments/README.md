# 实验数据约定

每个序列使用一个独立目录，至少包含 `manifest.json` 和 `points.json`。视频路径、点文件、真值文件和输出目录均相对于清单文件解析。

`points.json` 只定义第一处理帧的测点。动态错误匹配率需要额外的稀疏左右对应真值，可运行：

```powershell
python -m stereo_research.annotate_gt --manifest experiments/car/manifest.json --interval 10 --output experiments/car/ground_truth.csv
```

真值 CSV 可以只填写左右像素、只填写 `distance_m`，或填写完整的 `X_m/Y_m/Z_m`。固定静态距离/坐标可将 `frame` 留空，此时该真值应用于对应测点的所有帧。缺失的真值保持空白；评价程序不会用距离真值冒充 XYZ 真值。

建议正式数据至少包括横向运动、接近、远离、短时遮挡和低纹理/光照变化五类，每类三段。静态距离或 XYZ 实验每个位置至少录制 100 帧并重复三次。
