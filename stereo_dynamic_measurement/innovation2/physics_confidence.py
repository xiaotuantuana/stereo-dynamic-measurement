from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvidenceTerm:
    value: float | None
    valid: bool
    reliability: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.reliability) or not 0.0 <= self.reliability <= 1.0:
            raise ValueError("evidence reliability must be finite and in [0, 1]")
        if self.valid:
            if self.value is None or not np.isfinite(self.value):
                raise ValueError("valid evidence must have a finite value")
        elif self.value is not None:
            raise ValueError("invalid evidence must not carry a value")

    @classmethod
    def unavailable(cls, reason: str) -> "EvidenceTerm":
        return cls(value=None, valid=False, reliability=0.0, reason=reason)


@dataclass(frozen=True)
class PhysicsConfidenceConfig:
    weights: dict[str, float] = field(default_factory=lambda: {"flow_3d": 0.2, "temporal": 0.2, "spatial": 0.2, "spectral": 0.15, "phase": 0.1, "coherence": 0.15})
    threshold: float = 0.55

    def __post_init__(self) -> None:
        if not self.weights or any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("Physics confidence weights must include positive total weight")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0,1]")


@dataclass(frozen=True)
class PhysicsConfidenceResult:
    c_phy: float | None
    r_phy: float | None
    valid: bool
    num_valid_evidence: int
    available_evidence_names: tuple[str, ...]
    missing_evidence_names: tuple[str, ...]
    components: dict[str, EvidenceTerm]
    effective_weights: dict[str, float]


def compute_physics_confidence(
    evidence: dict[str, EvidenceTerm | float | None],
    config: PhysicsConfidenceConfig,
) -> PhysicsConfidenceResult:
    """Fuse measured evidence without interpreting missing terms as healthy."""
    components: dict[str, EvidenceTerm] = {}
    for key in config.weights:
        supplied = evidence.get(key)
        if isinstance(supplied, EvidenceTerm):
            term = supplied
        elif supplied is None or not np.isfinite(supplied):
            term = EvidenceTerm.unavailable("not supplied")
        else:
            term = EvidenceTerm(float(np.clip(supplied, 0.0, 1.0)), True)
        components[key] = term

    available = {
        key: config.weights[key] * term.reliability
        for key, term in components.items()
        if term.valid and term.value is not None and term.reliability > 0.0
    }
    missing = tuple(key for key, term in components.items() if not term.valid)
    if not available:
        return PhysicsConfidenceResult(
            c_phy=None,
            r_phy=None,
            valid=False,
            num_valid_evidence=0,
            available_evidence_names=(),
            missing_evidence_names=missing,
            components=components,
            effective_weights={},
        )
    denominator = sum(available.values())
    effective_weights = {key: weight / denominator for key, weight in available.items()}
    score = sum(effective_weights[key] * float(components[key].value) for key in effective_weights)
    score = float(np.clip(score, 0.0, 1.0))
    names = tuple(effective_weights)
    return PhysicsConfidenceResult(
        c_phy=score,
        r_phy=1.0 - score,
        valid=True,
        num_valid_evidence=len(names),
        available_evidence_names=names,
        missing_evidence_names=missing,
        components=components,
        effective_weights=effective_weights,
    )
