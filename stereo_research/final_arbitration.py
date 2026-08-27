"""Pure Phase 4 final-arbitration contracts.

This module deliberately has no pipeline state and never mutates a measurement
result.  ``TemporalStereoPipeline`` remains responsible for applying a returned
decision when an experiment has explicit write authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from .models import FramePointResult, SystemMode


class FinalDecision(str, Enum):
    ACCEPT = "ACCEPT"
    ACCEPT_WITH_WARNING = "ACCEPT_WITH_WARNING"
    USE_CORRECTED = "USE_CORRECTED"
    REJECT = "REJECT"


class ResultSource(str, Enum):
    I1_BASELINE = "I1_BASELINE"
    I2_CORRECTED = "I2_CORRECTED"
    REJECTED = "REJECTED"


class I3Risk(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    BLOCKING = "BLOCKING"


class I3Action(str, Enum):
    NONE = "NONE"
    WARN = "WARN"
    BLOCK_FINAL = "BLOCK_FINAL"


@dataclass(frozen=True)
class I3Recommendation:
    risk: I3Risk
    reason: str = ""
    action: I3Action = I3Action.NONE

    @classmethod
    def normal(cls) -> "I3Recommendation":
        return cls(I3Risk.NORMAL, "normal", I3Action.NONE)


@dataclass(frozen=True)
class I1BaselineView:
    """Copied scalar baseline values; no mutable I1 state is retained."""

    status: str
    xyz_m: tuple[float, float, float] | None
    distance_m: float | None
    valid: bool
    confidence: float

    @classmethod
    def from_result(cls, result: FramePointResult) -> "I1BaselineView":
        xyz = result.final_xyz_m
        finite_xyz = xyz is not None and all(math.isfinite(value) for value in xyz)
        return cls(
            status=str(result.status),
            xyz_m=None if xyz is None else tuple(float(value) for value in xyz),
            distance_m=(None if result.distance_m is None else float(result.distance_m)),
            valid=bool(result.status == "valid" and finite_xyz),
            confidence=float(result.confidence),
        )


@dataclass(frozen=True)
class CandidateSafety:
    """Complete, derived candidate safety gate with auditable failures."""

    candidate_xyz_m: tuple[float, float, float] | None
    geometry_valid: bool
    correction_authorized: bool
    evidence_sufficient: bool
    post_correction_safe: bool
    i2_state: str
    failed_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def safe(self) -> bool:
        return not self.failed_reasons

    @classmethod
    def unavailable(cls, reason: str) -> "CandidateSafety":
        return cls(None, False, False, False, False, "UNAVAILABLE", (reason,))

    @classmethod
    def evaluate(
        cls,
        *,
        candidate_xyz_m: tuple[float, float, float] | None,
        geometry_valid: bool,
        correction_authorized: bool,
        evidence_sufficient: bool,
        post_correction_safe: bool,
        i2_state: str,
    ) -> "CandidateSafety":
        reasons: list[str] = []
        finite = candidate_xyz_m is not None and all(math.isfinite(value) for value in candidate_xyz_m)
        if not finite:
            reasons.append("candidate_not_finite")
        if not geometry_valid:
            reasons.append("geometry_invalid")
        if not correction_authorized:
            reasons.append("correction_unauthorized")
        if not evidence_sufficient:
            reasons.append("evidence_insufficient")
        if not post_correction_safe:
            reasons.append("post_correction_unsafe")
        if i2_state not in {"QUARANTINED", "RECOVERY"}:
            reasons.append("i2_state_disallows_candidate")
        xyz = None if candidate_xyz_m is None else tuple(float(value) for value in candidate_xyz_m)
        return cls(
            xyz,
            bool(geometry_valid),
            bool(correction_authorized),
            bool(evidence_sufficient),
            bool(post_correction_safe),
            str(i2_state),
            tuple(reasons),
        )


@dataclass(frozen=True)
class ExperimentAuthority:
    requested_mode: SystemMode
    write_enabled: bool = False
    scope: str = ""
    policy_version: str = "phase4_controlled_v1"

    @property
    def effective_mode(self) -> SystemMode:
        if self.requested_mode is SystemMode.FULL_ENHANCED and not self.write_enabled:
            return SystemMode.ENHANCED_SHADOW
        return self.requested_mode

    @property
    def may_write_final(self) -> bool:
        return (
            self.requested_mode is SystemMode.FULL_ENHANCED
            and self.write_enabled
            and self.scope == "experiment"
        )

    @classmethod
    def shadow(cls, mode: SystemMode = SystemMode.ENHANCED_SHADOW) -> "ExperimentAuthority":
        return cls(requested_mode=mode, write_enabled=False, scope="")

    @classmethod
    def full_experiment(cls, *, write_enabled: bool) -> "ExperimentAuthority":
        return cls(
            requested_mode=SystemMode.FULL_ENHANCED,
            write_enabled=write_enabled,
            scope="experiment",
        )


@dataclass(frozen=True)
class ArbitrationDecision:
    proposed_decision: FinalDecision
    committed_decision: FinalDecision
    result_source: ResultSource
    final_valid: bool
    final_xyz_m: tuple[float, float, float] | None
    reason: str
    write_committed: bool
    authority: ExperimentAuthority


class FinalArbitrator:
    """Return an auditable decision without mutating pipeline results."""

    def decide(
        self,
        *,
        baseline: I1BaselineView,
        candidate_safety: CandidateSafety,
        diagnosis: I3Recommendation,
        authority: ExperimentAuthority,
    ) -> ArbitrationDecision:
        proposed = self._proposed(baseline, candidate_safety, diagnosis)
        if proposed is FinalDecision.USE_CORRECTED and authority.may_write_final:
            return ArbitrationDecision(
                proposed,
                FinalDecision.USE_CORRECTED,
                ResultSource.I2_CORRECTED,
                True,
                candidate_safety.candidate_xyz_m,
                "complete_safe_candidate",
                True,
                authority,
            )
        if proposed is FinalDecision.REJECT and authority.may_write_final:
            return ArbitrationDecision(
                proposed,
                FinalDecision.REJECT,
                ResultSource.REJECTED,
                False,
                None,
                diagnosis.reason or "blocking_or_invalid_baseline",
                False,
                authority,
            )
        if baseline.valid:
            committed = proposed if proposed is FinalDecision.ACCEPT else FinalDecision.ACCEPT_WITH_WARNING
            return ArbitrationDecision(
                proposed,
                committed,
                ResultSource.I1_BASELINE,
                True,
                baseline.xyz_m,
                "shadow_or_no_write_authority" if proposed is not committed else "baseline_accepted",
                False,
                authority,
            )
        return ArbitrationDecision(
            proposed,
            FinalDecision.REJECT,
            ResultSource.REJECTED,
            False,
            None,
            "invalid_baseline",
            False,
            authority,
        )

    @staticmethod
    def _proposed(
        baseline: I1BaselineView,
        candidate_safety: CandidateSafety,
        diagnosis: I3Recommendation,
    ) -> FinalDecision:
        if not baseline.valid:
            return FinalDecision.REJECT
        if diagnosis.risk is I3Risk.BLOCKING:
            return FinalDecision.REJECT
        if candidate_safety.safe:
            return FinalDecision.USE_CORRECTED
        if diagnosis.risk is I3Risk.WARNING:
            return FinalDecision.ACCEPT_WITH_WARNING
        return FinalDecision.ACCEPT
