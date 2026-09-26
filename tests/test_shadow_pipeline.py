from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo_research.models import FramePointResult, MatcherConfig, PointSpec
from stereo_research.pipeline import TemporalStereoPipeline
from stereo_research.runner import ACCURACY_POLICY_CSV_FIELDS, CSV_FIELDS, SHADOW_CSV_FIELDS
from stereo_research.shadow_analysis import ShadowAnalyzer


EXPECTED_SHADOW_CSV_FIELDS = [
    "shadow_input_x_m", "shadow_input_y_m", "shadow_input_z_m", "shadow_input_stage",
    "candidate_corrected_x_m", "candidate_corrected_y_m", "candidate_corrected_z_m",
    "c_phy", "c_phy_valid", "c_phy_valid_terms", "c_phy_missing_terms",
    "transient_protected", "fault_class", "fault_confidence", "recommended_recovery",
    "r_2d3d", "r_temporal", "r_spatial", "r_frequency", "r_phase", "r_coherence",
    "r_lr", "r_epi", "r_fb", "r_ref", "r_calib",
]


def _textured_stereo(disparity: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    left = cv2.GaussianBlur(rng.integers(0, 256, (140, 240), dtype=np.uint8), (3, 3), 0.5)
    transform = np.array([[1.0, 0.0, -disparity], [0.0, 1.0, 0.0]], dtype=np.float32)
    right = cv2.warpAffine(left, transform, (240, 140), borderMode=cv2.BORDER_REFLECT101)
    return left, right


def _q() -> np.ndarray:
    return np.array([[1, 0, 0, -120], [0, 1, 0, -70], [0, 0, 0, 100], [0, 0, 10, 0]], dtype=float)


def test_shadow_csv_contract_is_appended_after_every_legacy_field() -> None:
    assert SHADOW_CSV_FIELDS == EXPECTED_SHADOW_CSV_FIELDS
    shadow_end = len(CSV_FIELDS) - len(ACCURACY_POLICY_CSV_FIELDS)
    assert CSV_FIELDS[shadow_end - len(EXPECTED_SHADOW_CSV_FIELDS):shadow_end] == EXPECTED_SHADOW_CSV_FIELDS


def test_shadow_result_serializes_missing_evidence_as_empty_cells() -> None:
    result = FramePointResult(
        method="M3",
        frame=4,
        point_id="P1",
        status="valid",
        final_x_m=0.1,
        final_y_m=0.2,
        final_z_m=2.0,
        shadow_input_x_m=0.1,
        shadow_input_y_m=0.2,
        shadow_input_z_m=2.0,
        shadow_input_stage="estimated",
        c_phy_valid=False,
    )

    row = result.as_csv_row()

    assert row["shadow_input_stage"] == "estimated"
    assert row["c_phy"] == ""
    assert row["r_2d3d"] == ""
    assert row["final_X_m"] == 0.1


def _runtime_result(frame: int, point_id: str, y_m: float, *, faulty: bool = False) -> FramePointResult:
    x_m = {"P1": 0.0, "P2": 0.1, "P3": 0.2}[point_id]
    return FramePointResult(
        method="M3",
        frame=frame,
        point_id=point_id,
        status="valid",
        left_x=100.0 + x_m * 100.0,
        left_y=200.0 + y_m * 100.0,
        right_x=60.0 + x_m * 100.0,
        right_y=200.0 + y_m * 100.0,
        disparity=40.0,
        measured_x_m=x_m,
        measured_y_m=y_m,
        measured_z_m=2.0,
        estimated_x_m=x_m,
        estimated_y_m=y_m,
        estimated_z_m=2.0,
        compensated_x_m=x_m,
        compensated_y_m=y_m,
        compensated_z_m=2.0,
        final_x_m=x_m,
        final_y_m=y_m,
        final_z_m=2.0,
        confidence=0.2 if faulty else 0.9,
        texture_std=12.0,
        match_cost=0.9 if faulty else 0.1,
        lr_error_px=3.0 if faulty else 0.1,
        neighbor_disparity_mad=2.0 if faulty else 0.1,
        flow_fb_error_px=2.0 if faulty else 0.1,
        compensation_rmse_mm=0.1,
    )


def test_shadow_analyzer_builds_real_fingerprint_and_only_recommends_recovery() -> None:
    analyzer = ShadowAnalyzer()
    analyzer.process_frame([_runtime_result(0, point, 0.0) for point in ("P1", "P2", "P3")])
    analyzer.process_frame([_runtime_result(1, point, 0.001) for point in ("P1", "P2", "P3")])
    current = [
        _runtime_result(2, "P1", 0.080, faulty=True),
        _runtime_result(2, "P2", 0.002),
        _runtime_result(2, "P3", 0.002),
    ]
    legacy_before = [result.as_csv_row() for result in current]

    shadow = analyzer.process_frame(current)
    p1 = next(result for result in shadow if result.point_id == "P1")

    assert p1.shadow_input_stage == "compensated"
    assert p1.shadow_input_y_m == 0.080
    assert p1.r_lr == 3.0
    assert p1.r_fb == 2.0
    assert p1.fault_class == "STEREO_MISMATCH"
    assert p1.recommended_recovery == "LOCAL_REMATCH"
    assert p1.candidate_corrected_y_m < p1.shadow_input_y_m
    assert p1.final_y_m == 0.080
    for before, result in zip(legacy_before, current, strict=True):
        assert result.as_csv_row() == before


@pytest.mark.parametrize("field", ["allow_physics_correction_to_final", "allow_fault_recovery_to_final"])
def test_first_round_rejects_enabling_shadow_writes_to_final(field: str) -> None:
    with pytest.raises(NotImplementedError):
        MatcherConfig(**{field: True})


def test_real_pipeline_appends_shadow_without_changing_formal_final_xyz() -> None:
    left, right = _textured_stereo()
    pipeline = TemporalStereoPipeline("M3", _q(), "m", MatcherConfig(uniqueness_margin=0.005))

    result = pipeline.initialize(left, right, (PointSpec("P1", (150.0, 70.0)),), frame=0)[0]

    assert result.status == "valid"
    assert result.shadow_input_stage == "estimated"
    assert result.shadow_input_x_m == result.estimated_x_m
    assert result.candidate_corrected_x_m == result.shadow_input_x_m
    assert (result.final_x_m, result.final_y_m, result.final_z_m) == (
        result.estimated_x_m,
        result.estimated_y_m,
        result.estimated_z_m,
    )
