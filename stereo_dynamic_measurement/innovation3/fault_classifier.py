from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from .fault_fingerprint import FaultFingerprint

class FaultType(str, Enum):
    NORMAL="NORMAL"; WEAK_TEXTURE="WEAK_TEXTURE"; MOTION_BLUR="MOTION_BLUR"; OCCLUSION="OCCLUSION"; STEREO_MISMATCH="STEREO_MISMATCH"; FLOW_DRIFT="FLOW_DRIFT"; SEARCH_RANGE_FAILURE="SEARCH_RANGE_FAILURE"; CAMERA_MOTION="CAMERA_MOTION"; EXTRINSIC_DRIFT="EXTRINSIC_DRIFT"; TRACKING_LOSS="TRACKING_LOSS"
@dataclass(frozen=True)
class Diagnosis: fault_type: FaultType; score: float; evidence: tuple[str,...]
class RuleDiagnosticEngine:
    def diagnose(self, f: FaultFingerprint) -> Diagnosis:
        if f.reference_motion_residual > 1 and f.common_target_motion_ratio > .7: return Diagnosis(FaultType.CAMERA_MOTION,.95,("reference_motion","common_target_motion"))
        if f.epipolar_residual > 1 or f.geometry_health_residual > .6: return Diagnosis(FaultType.EXTRINSIC_DRIFT,.9,("epipolar","geometry_health"))
        if f.lr_residual > 1 and f.neighbor_residual > 1 and f.reference_motion_residual < 1: return Diagnosis(FaultType.STEREO_MISMATCH,.9,("lr","neighbor"))
        if f.gradient_quality < .25 and f.blur_score < .4: return Diagnosis(FaultType.MOTION_BLUR,.8,("texture","blur"))
        if f.gradient_quality < .25: return Diagnosis(FaultType.WEAK_TEXTURE,.75,("texture",))
        if f.flow_residual > 1 or f.fb_error > 1: return Diagnosis(FaultType.FLOW_DRIFT,.75,("flow",))
        if f.matching_cost_residual > 1 and f.temporal_residual < 1: return Diagnosis(FaultType.SEARCH_RANGE_FAILURE,.7,("matching",))
        if f.physics_residual > .7: return Diagnosis(FaultType.OCCLUSION,.65,("physics",))
        return Diagnosis(FaultType.NORMAL,.05,())
