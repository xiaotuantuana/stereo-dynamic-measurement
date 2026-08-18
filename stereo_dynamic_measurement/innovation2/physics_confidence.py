from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    c_phy: float
    r_phy: float
    components: dict[str, float]


def compute_physics_confidence(evidence: dict[str, float | None], config: PhysicsConfidenceConfig) -> PhysicsConfidenceResult:
    """Weighted confidence with renormalization over available finite evidence."""
    available = [(key, config.weights[key], float(evidence[key])) for key in config.weights if key in evidence and evidence[key] is not None and np.isfinite(evidence[key])]
    if not available:
        return PhysicsConfidenceResult(0.0, 1.0, {})
    denominator = sum(weight for _, weight, _ in available)
    components = {key: float(np.clip(value, 0.0, 1.0)) for key, _, value in available}
    score = sum(config.weights[key] * components[key] for key in components) / denominator
    score = float(np.clip(score, 0.0, 1.0))
    return PhysicsConfidenceResult(score, 1.0 - score, components)
