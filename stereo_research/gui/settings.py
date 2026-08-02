from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..models import MatcherConfig


ControlKind = Literal["int", "float", "bool", "choice"]


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    kind: ControlKind
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    tooltip: str = ""


@dataclass(frozen=True)
class ParameterGroup:
    title: str
    items: tuple[ParameterSpec, ...]


def parameter_groups() -> tuple[ParameterGroup, ...]:
    return (
        ParameterGroup(
            "局部匹配",
            (
                _int("patch_size", "匹配块尺寸", 3, 51, 2, "必须为奇数"),
                _int("search_radius", "水平搜索半径", 1, 64),
                _int("expanded_search_radius", "扩展搜索半径", 1, 128),
                _int("vertical_radius", "垂直搜索半径", 0, 8),
                _float("min_texture_std", "最小纹理标准差", 0.1, 100.0, 0.5),
                _float("max_photo_cost", "最大光度代价", 0.01, 1.0, 0.01),
                _float("uniqueness_margin", "唯一性间隔", 0.0, 1.0, 0.005),
                _int("num_disparities", "SGBM 视差范围", 16, 256, 16),
                _int("expanded_num_disparities", "扩展视差范围", 16, 512, 16),
                _int("block_size", "SGBM 块尺寸", 3, 21, 2, "必须为奇数"),
                _int("uniqueness_ratio", "SGBM 唯一性", 0, 50),
                _int("speckle_window_size", "散斑窗口", 0, 500),
                _int("speckle_range", "散斑范围", 0, 16),
            ),
        ),
        ParameterGroup(
            "光流与运动预测",
            (
                _bool("enable_flow", "启用左图光流"),
                _int("lk_window_size", "LK 窗口尺寸", 3, 61, 2, "必须为奇数"),
                _int("lk_max_level", "LK 金字塔层级", 0, 8),
                _float("flow_fb_threshold", "左图前后向阈值 / px", 0.05, 10.0, 0.05),
                _float("right_flow_fb_threshold", "右图前后向阈值 / px", 0.05, 10.0, 0.05),
                _bool("enable_prediction", "启用历史预测"),
                _float("max_disparity_velocity", "最大视差速度 / px", 0.1, 20.0, 0.1),
                _float("cycle_prediction_blend", "右图预测融合权重", 0.0, 1.0, 0.05),
            ),
        ),
        ParameterGroup(
            "约束与亚像素",
            (
                _bool("enable_epipolar", "启用极线约束"),
                _bool("enable_neighborhood", "启用邻域一致性"),
                _bool("enable_subpixel", "启用亚像素精化"),
                _bool("enable_lr_check", "启用左右一致性"),
                _float("photo_weight", "光度代价权重", 0.0, 2.0, 0.05),
                _float("prediction_weight", "历史预测权重", 0.0, 2.0, 0.05),
                _float("epipolar_weight", "极线约束权重", 0.0, 2.0, 0.05),
                _float("neighborhood_weight", "邻域一致性权重", 0.0, 2.0, 0.05),
                _float("lr_threshold", "左右一致性阈值 / px", 0.05, 10.0, 0.05),
                _choice("subpixel_method", "常规亚像素方法", ("continuous", "parabolic", "icgn")),
                _float("subpixel_step", "连续搜索步长 / px", 0.01, 1.0, 0.01),
            ),
        ),
        ParameterGroup(
            "IC-GN 与曲率",
            (
                _bool("enable_icgn", "启用 IC-GN"),
                _choice("research_subpixel_method", "研究方法亚像素", ("icgn", "continuous", "parabolic")),
                _int("icgn_patch_size", "IC-GN 块尺寸", 3, 61, 2, "必须为奇数"),
                _int("icgn_max_iterations", "最大迭代次数", 1, 100),
                _float("icgn_epsilon", "收敛阈值", 0.000001, 0.1, 0.0001, decimals=6),
                _float("icgn_max_offset_px", "最大偏移 / px", 0.1, 10.0, 0.1),
                _float("icgn_max_residual", "最大归一化残差", 0.001, 2.0, 0.01),
                _choice("icgn_fallback_method", "失败回退方式", ("line_search", "parabolic", "continuous", "integer")),
                _float("icgn_fallback_search_radius_px", "回退半径 / px", 0.1, 2.0, 0.1),
                _int("icgn_fallback_sample_count", "回退采样数", 5, 51, 2, "必须为奇数"),
                _float("curvature_sample_step_px", "曲率采样步长 / px", 0.01, 0.5, 0.01),
            ),
        ),
        ParameterGroup(
            "闭环与自适应滤波",
            (
                _bool("enable_cycle_consistency", "启用四视图闭环"),
                _float("cycle_soft_threshold_px", "闭环软阈值 / px", 0.05, 5.0, 0.05),
                _float("cycle_hard_threshold_px", "闭环硬阈值 / px", 0.1, 10.0, 0.1),
                _float("cycle_recovery_threshold_px", "闭环恢复阈值 / px", 0.2, 20.0, 0.1),
                _bool("enable_adaptive_filter", "启用自适应卡尔曼"),
                _float("kalman_nis_soft_threshold", "NIS 软阈值", 0.1, 100.0, 0.1),
                _float("kalman_nis_hard_threshold", "NIS 硬阈值", 0.2, 200.0, 0.1),
                _float("kalman_nis_variance_scale", "异常方差放大倍数", 1.01, 20.0, 0.1),
                _int("kalman_max_predict_only_frames", "连续预测上限", 1, 30),
            ),
        ),
        ParameterGroup(
            "恢复、深度与补偿",
            (
                _bool("enable_recovery", "启用失效恢复"),
                _int("max_failures", "连续失败上限", 1, 30),
                _int("recovery_interval_frames", "恢复尝试间隔", 1, 100),
                _int("recovery_search_radius", "恢复搜索半径", 4, 256),
                _bool("enable_pyramid", "启用金字塔恢复"),
                _int("pyramid_levels", "金字塔层数", 1, 6),
                _float("min_depth_m", "最小深度 / m", 0.01, 100.0, 0.1),
                _float("max_depth_m", "最大深度 / m", 0.1, 1000.0, 1.0),
                _bool("enable_camera_compensation", "启用相机运动补偿"),
                _int("camera_compensation_min_points", "最少参考点", 3, 30),
                _float("camera_compensation_inlier_threshold_mm", "补偿内点阈值 / mm", 0.1, 100.0, 0.1),
                _float("camera_compensation_max_rmse_mm", "补偿最大 RMSE / mm", 0.1, 100.0, 0.1),
            ),
        ),
    )


def build_matcher_config(values: dict[str, Any]) -> MatcherConfig:
    payload = asdict(MatcherConfig())
    unknown = set(values) - set(payload)
    if unknown:
        raise ValueError(f"未知参数: {', '.join(sorted(unknown))}")
    payload.update(values)
    return MatcherConfig(**payload)


def _int(key: str, label: str, minimum: int, maximum: int, step: int = 1, tooltip: str = "") -> ParameterSpec:
    return ParameterSpec(key, label, "int", minimum, maximum, step, tooltip=tooltip)


def _float(
    key: str,
    label: str,
    minimum: float,
    maximum: float,
    step: float,
    tooltip: str = "",
    decimals: int = 3,
) -> ParameterSpec:
    del decimals
    return ParameterSpec(key, label, "float", minimum, maximum, step, tooltip=tooltip)


def _bool(key: str, label: str) -> ParameterSpec:
    return ParameterSpec(key, label, "bool")


def _choice(key: str, label: str, choices: tuple[str, ...]) -> ParameterSpec:
    return ParameterSpec(key, label, "choice", choices=choices)
