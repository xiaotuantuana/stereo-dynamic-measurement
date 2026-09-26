from __future__ import annotations

import numpy as np
import pandas as pd


def spatial_consistency_residuals(reference_shape: pd.DataFrame, current_shape: pd.DataFrame) -> pd.DataFrame:
    """Evaluate per-point deformation from the known reference shape (mm)."""
    columns = ["point_id", "X_mm", "Y_mm", "Z_mm"]
    if not set(columns) <= set(reference_shape.columns) or not set(columns) <= set(current_shape.columns):
        raise ValueError("Reference and current shape require point_id and X/Y/Z_mm")
    reference = reference_shape[columns].rename(columns={"X_mm": "X_ref_mm", "Y_mm": "Y_ref_mm", "Z_mm": "Z_ref_mm"})
    merged = current_shape[columns].merge(reference, on="point_id", validate="one_to_one")
    displacement = merged[["X_mm", "Y_mm", "Z_mm"]].to_numpy() - merged[["X_ref_mm", "Y_ref_mm", "Z_ref_mm"]].to_numpy()
    merged["spatial_residual_mm"] = np.linalg.norm(displacement, axis=1)
    return merged[["point_id", "spatial_residual_mm"]]
