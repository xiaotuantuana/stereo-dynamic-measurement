from __future__ import annotations

import numpy as np
import pytest

from experiment.simulation.image_anomaly_injector import ImageAnomalyCase, ImageAnomalyInjector


def _image() -> np.ndarray:
    return np.arange(80 * 120, dtype=np.uint16).reshape(80, 120).astype(np.uint8)


def _case(**overrides: object) -> ImageAnomalyCase:
    values: dict[str, object] = {
        "case_id": "occ-01",
        "pair_id": "pair-01",
        "anomaly_type": "local_occlusion",
        "primitive": "occlusion",
        "active_frame_ids": ("000010",),
        "point_id": "P2",
        "side": "right",
        "roi": (40, 20, 24, 18),
        "seed": 20260828,
    }
    values.update(overrides)
    return ImageAnomalyCase(**values)


def test_injector_is_deterministic_copy_only_and_does_not_modify_source() -> None:
    source = _image()
    before = source.copy()
    injector = ImageAnomalyInjector(_case())

    first = injector.transform(source, frame_id="000010", side="right")
    second = injector.transform(source, frame_id="000010", side="right")

    assert np.array_equal(source, before)
    assert first is not source
    assert np.array_equal(first, second)
    assert not np.array_equal(first, source)
    assert np.array_equal(injector.transform(source, frame_id="000010", side="left"), source)
    assert np.array_equal(injector.transform(source, frame_id="000011", side="right"), source)


@pytest.mark.parametrize(
    ("anomaly_type", "primitive", "extra"),
    [
        ("local_occlusion", "occlusion", {}),
        ("local_blur", "blur", {"blur_kernel": 7}),
        ("unilateral_roi_shift", "roi_shift", {"shift_px": 3}),
        ("continuous_anomaly", "blur", {"blur_kernel": 5, "active_frame_ids": ("000010", "000011")}),
    ],
)
def test_each_supported_image_anomaly_changes_only_an_image_copy(
    anomaly_type: str, primitive: str, extra: dict[str, object],
) -> None:
    source = _image()
    injector = ImageAnomalyInjector(_case(anomaly_type=anomaly_type, primitive=primitive, **extra))

    transformed = injector.transform(source, frame_id="000010", side="right")

    assert transformed.shape == source.shape
    assert transformed.dtype == source.dtype
    assert not np.shares_memory(transformed, source)
    assert np.array_equal(source, _image())


def test_roi_shift_rejects_any_pixel_offset_outside_one_to_three() -> None:
    with pytest.raises(ValueError, match="shift_px"):
        ImageAnomalyCase(**{**_case().__dict__, "anomaly_type": "unilateral_roi_shift", "primitive": "roi_shift", "shift_px": 4})


def test_continuous_case_requires_two_to_four_consecutive_frames() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        ImageAnomalyCase(**{**_case().__dict__, "anomaly_type": "continuous_anomaly", "primitive": "blur", "active_frame_ids": ("000010",)})
