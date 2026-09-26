from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics_confidence import EvidenceTerm


@dataclass(frozen=True)
class TransientGateConfig:
    temporal_residual_threshold_mm: float = 10.0
    normal_evidence_threshold: float = 0.7
    abnormal_evidence_threshold: float = 0.3
    synchronous_motion_ratio_threshold: float = 0.6

    def __post_init__(self) -> None:
        values = (
            self.normal_evidence_threshold,
            self.abnormal_evidence_threshold,
            self.synchronous_motion_ratio_threshold,
        )
        if self.temporal_residual_threshold_mm <= 0 or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
        ):
            raise ValueError("invalid transient gate configuration")


@dataclass(frozen=True)
class TransientDecision:
    is_possible_real_transient: bool
    possible_measurement_error: bool
    allow_correction: bool
    reason: str


def decide_transient(
    *,
    temporal_residual_mm: float,
    flow_3d: EvidenceTerm,
    spatial: EvidenceTerm,
    visual_consistency: EvidenceTerm,
    synchronous_motion_ratio: float | None,
    config: TransientGateConfig = TransientGateConfig(),
) -> TransientDecision:
    """Conservatively separate supported synchronous motion from isolated error."""
    if not np.isfinite(temporal_residual_mm) or temporal_residual_mm < 0:
        return TransientDecision(False, False, False, "temporal_residual_unavailable")
    if temporal_residual_mm < config.temporal_residual_threshold_mm:
        return TransientDecision(False, False, False, "not_a_fast_transient")

    support = flow_3d if flow_3d.valid else visual_consistency
    sync_valid = synchronous_motion_ratio is not None and np.isfinite(synchronous_motion_ratio)
    spatial_normal = spatial.valid and spatial.value is not None and spatial.value >= config.normal_evidence_threshold
    support_normal = support.valid and support.value is not None and support.value >= config.normal_evidence_threshold
    synchronous = sync_valid and synchronous_motion_ratio >= config.synchronous_motion_ratio_threshold
    if support_normal and spatial_normal and synchronous:
        return TransientDecision(True, False, False, "supported_synchronous_motion")

    spatial_abnormal = spatial.valid and spatial.value is not None and spatial.value <= config.abnormal_evidence_threshold
    support_abnormal = support.valid and support.value is not None and support.value <= config.abnormal_evidence_threshold
    isolated = sync_valid and synchronous_motion_ratio < config.synchronous_motion_ratio_threshold
    if support_abnormal and spatial_abnormal and isolated:
        return TransientDecision(False, True, True, "isolated_multisource_measurement_error")

    return TransientDecision(False, False, False, "insufficient_evidence_for_safe_correction")
