from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class RecoveryVerification: recovery_success:bool; c_phy_improvement:float; residual_improvement:float
def verify_recovery(*,c_phy_before:float,c_phy_after:float,primary_residual_before:float,primary_residual_after:float)->RecoveryVerification:
    c=float(c_phy_after-c_phy_before); r=float(primary_residual_before-primary_residual_after)
    return RecoveryVerification(c>0 and r>0,c,r)
