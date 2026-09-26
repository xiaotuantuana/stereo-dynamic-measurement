from __future__ import annotations
import pandas as pd
from stereo_dynamic_measurement.simulation.fault_injection import FaultInjector

def test_fault_injector_is_seeded_and_covers_requested_faults() -> None:
    source=pd.DataFrame({"frame":[0,1,2,3],"point_id":["P1"]*4,"x_left":[10.]*4,"x_right":[5.]*4,"y_left":[10.]*4,"y_right":[10.]*4,"X_raw":[0.]*4,"Y_raw":[0.]*4,"Z_raw":[2000.]*4,"observed":[True]*4})
    injector=FaultInjector(seed=4)
    for kind in ("stereo_mismatch","occlusion","blur","flow_drift","camera_motion","extrinsic_drift","tracking_loss"):
        result=injector.inject(source,kind)
        assert "injected_fault" in result and set(result.injected_fault)=={kind}
