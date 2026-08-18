from __future__ import annotations
import cv2, numpy as np
from stereo_dynamic_measurement.innovation3.reference_monitor import ReferenceMonitor
def test_reference_monitor_estimates_common_motion():
 rng=np.random.default_rng(2); a=cv2.GaussianBlur(rng.integers(0,255,(120,160),dtype=np.uint8),(3,3),.5); b=cv2.warpAffine(a,np.array([[1,0,3],[0,1,-2]],np.float32),(160,120))
 r=ReferenceMonitor().estimate(a,b,(10,10,140,100)); assert r.valid and np.allclose(r.common_motion_px,(3,-2),atol=.4)
