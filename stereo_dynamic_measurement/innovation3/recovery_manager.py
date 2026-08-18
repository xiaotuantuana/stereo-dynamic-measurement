from __future__ import annotations
from dataclasses import dataclass
from .fault_classifier import FaultType
@dataclass(frozen=True)
class RecoveryPlan: action:str; parameter_updates:dict[str,float]; requires_extrinsic_update:bool=False
class RecoveryManager:
    _plans={FaultType.WEAK_TEXTURE:("HISTORY_WEIGHTED_CONTEXT",{"history_weight":.8,"context_scale":1.5}),FaultType.MOTION_BLUR:("TEMPORAL_HOLD",{"current_frame_weight":.2}),FaultType.OCCLUSION:("KALMAN_PREDICT",{}),FaultType.STEREO_MISMATCH:("LOCAL_REMATCH",{}),FaultType.SEARCH_RANGE_FAILURE:("EXPAND_SEARCH",{"search_scale":2.0}),FaultType.FLOW_DRIFT:("REDETECT_FEATURES",{}),FaultType.CAMERA_MOTION:("REFERENCE_MOTION_COMPENSATION",{}),FaultType.EXTRINSIC_DRIFT:("QUICK_EXTRINSIC_UPDATE",{}),FaultType.TRACKING_LOSS:("GLOBAL_REINITIALIZE",{}),FaultType.NORMAL:("NONE",{})}
    def plan(self, fault:FaultType)->RecoveryPlan:
        action, updates=self._plans[fault]; return RecoveryPlan(action,updates,fault is FaultType.EXTRINSIC_DRIFT)
