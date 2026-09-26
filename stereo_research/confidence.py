"""Confidence-state decisions used between stereo matching and state update."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceDecision:
    score: float
    state: str
    accept_measurement: bool
    predict_only: bool
    trigger_recovery: bool
    trigger_reinitialization: bool


def decide_confidence(
    score: float,
    very_low_frames: int,
    *,
    high: float,
    medium: float,
    low: float,
    lost_after: int,
    mode: str,
    uniqueness_margin: float | None = None,
    single_margin_reference: float = 0.10,
) -> tuple[ConfidenceDecision, int]:
    """Classify a measurement before any Kalman or PointState mutation."""
    if mode == "none":
        return ConfidenceDecision(score, "HIGH", True, False, False, False), 0
    if mode in {"lr_only", "single_margin"}:
        accepted = uniqueness_margin is not None and uniqueness_margin >= single_margin_reference
        state = "HIGH" if accepted else "LOW"
        return ConfidenceDecision(score, state, accepted, not accepted, False, False), 0
    if score >= high:
        state, very_low_frames = "HIGH", 0
    elif score >= medium:
        state, very_low_frames = "MEDIUM", 0
    elif score >= low:
        state, very_low_frames = "LOW", 0
    else:
        very_low_frames += 1
        state = "LOST" if very_low_frames >= lost_after else "LOW"
    closed_loop = mode == "closed_loop"
    accepted = state in {"HIGH", "MEDIUM"}
    return (
        ConfidenceDecision(
            score, state, accepted, not accepted,
            closed_loop and state in {"LOW", "LOST"},
            closed_loop and state == "LOST",
        ),
        very_low_frames,
    )
