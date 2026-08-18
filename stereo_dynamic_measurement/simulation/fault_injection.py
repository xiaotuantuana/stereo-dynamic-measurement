from __future__ import annotations
import numpy as np
import pandas as pd
class FaultInjector:
    def __init__(self,seed:int): self.rng=np.random.default_rng(seed)
    def inject(self,data:pd.DataFrame,kind:str)->pd.DataFrame:
        out=data.copy(); n=len(out)
        if n==0: raise ValueError("Cannot inject into empty data")
        if kind=="stereo_mismatch": out["x_right"]+=5
        elif kind=="occlusion": out.loc[out.index[n//3:2*n//3],"observed"]=False
        elif kind=="blur": out["blur_score"]=.1
        elif kind=="flow_drift": out["x_left"]+=np.linspace(0,4,n)
        elif kind=="camera_motion":
            out["x_left"]+=3; out["x_right"]+=3; out["y_left"]+=2; out["y_right"]+=2
        elif kind=="extrinsic_drift": out["y_right"]+=2; out["Z_raw"]+=10
        elif kind=="tracking_loss": out.loc[out.index[n//3:2*n//3],"observed"]=False; out.loc[out.index[n//3:2*n//3],"x_left"]=np.nan
        else: raise ValueError(f"Unsupported fault type: {kind}")
        out["injected_fault"]=kind; return out
