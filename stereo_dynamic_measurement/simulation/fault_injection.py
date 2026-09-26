from __future__ import annotations

import numpy as np
import pandas as pd


class FaultInjector:
    _SCALES = {"mild": 0.5, "medium": 1.0, "severe": 2.0}

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        data: pd.DataFrame,
        kind: str,
        *,
        severity: str = "medium",
    ) -> pd.DataFrame:
        if severity not in self._SCALES:
            raise ValueError(f"Unsupported severity: {severity}")
        out = data.copy()
        count = len(out)
        if count == 0:
            raise ValueError("Cannot inject into empty data")
        scale = self._SCALES[severity]
        span = max(1, min(count, int(round(count * min(0.75, scale / 3.0)))))
        start = max(0, (count - span) // 2)
        region = out.index[start:start + span]
        if kind == "stereo_mismatch":
            out["x_right"] += 5.0 * scale
        elif kind == "occlusion":
            out.loc[region, "observed"] = False
        elif kind == "blur":
            out["blur_score"] = {"mild": 0.35, "medium": 0.1, "severe": 0.02}[severity]
        elif kind == "flow_drift":
            out["x_left"] += np.linspace(0.0, 4.0 * scale, count)
        elif kind == "camera_motion":
            out["x_left"] += 3.0 * scale
            out["x_right"] += 3.0 * scale
            out["y_left"] += 2.0 * scale
            out["y_right"] += 2.0 * scale
        elif kind == "extrinsic_drift":
            out["y_right"] += 2.0 * scale
            out["Z_raw"] += 10.0 * scale
        elif kind == "tracking_loss":
            out.loc[region, "observed"] = False
            out.loc[region, "x_left"] = np.nan
        else:
            raise ValueError(f"Unsupported fault type: {kind}")
        out["injected_fault"] = kind
        out["injected_severity"] = severity
        return out
