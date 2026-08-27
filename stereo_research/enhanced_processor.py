"""State-isolated Innovation 2 processing for controlled enhanced runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from .final_arbitration import CandidateSafety


XYZ = tuple[float, float, float]


class I2State(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    QUARANTINED = "QUARANTINED"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class I2Evidence:
    """Result of causal evidence fusion supplied by the runtime adapter."""

    confirmed_anomaly: bool = False
    legitimate_motion: bool = False
    hard_failure: bool = False
    geometry_valid: bool = True
    evidence_sufficient: bool = True
    post_correction_safe: bool = True
    suspicious: bool = False


@dataclass(frozen=True)
class TimedObservation:
    frame: int
    timestamp_s: float
    xyz_mm: XYZ
    evidence: I2Evidence = field(default_factory=I2Evidence)


@dataclass(frozen=True)
class EnhancedStateOutcome:
    frame: int
    timestamp_s: float
    raw_xyz_mm: XYZ
    prediction_xyz_mm: XYZ | None
    corrected_xyz_mm: XYZ | None
    state: I2State
    correction_applied: bool
    trusted_committed: bool
    candidate_safety: CandidateSafety
    episode_id: int | None
    reason: str


class EnhancedPointProcessor:
    """Maintain I2 audit and trusted histories without feedback contamination."""

    def __init__(
        self,
        *,
        max_authorized_correction_mm: float = 100.0,
        recovery_confirmation_frames: int = 2,
        max_prediction_gap_s: float = 1.0,
        recovery_consistency_mm: float = 10.0,
    ) -> None:
        if max_authorized_correction_mm <= 0:
            raise ValueError("max_authorized_correction_mm must be positive")
        if recovery_confirmation_frames < 2:
            raise ValueError("recovery_confirmation_frames must be at least two")
        if max_prediction_gap_s <= 0:
            raise ValueError("max_prediction_gap_s must be positive")
        if recovery_consistency_mm <= 0:
            raise ValueError("recovery_consistency_mm must be positive")
        self.max_authorized_correction_mm = float(max_authorized_correction_mm)
        self.recovery_confirmation_frames = int(recovery_confirmation_frames)
        self.max_prediction_gap_s = float(max_prediction_gap_s)
        self.recovery_consistency_mm = float(recovery_consistency_mm)
        self.state = I2State.NORMAL
        self.raw_history: list[TimedObservation] = []
        self.corrected_history: list[XYZ | None] = []
        self.trusted_history: list[TimedObservation] = []
        self._recovery_observations: list[TimedObservation] = []
        self._next_episode_id = 1
        self._episode_id: int | None = None

    def process(self, observation: TimedObservation) -> EnhancedStateOutcome:
        """Process one observation; only validated normal observations become trusted."""

        raw = tuple(float(value) for value in observation.xyz_mm)
        prediction = self._predict(observation.timestamp_s)
        self.raw_history.append(observation)
        evidence = observation.evidence

        if evidence.hard_failure or not self._finite(raw):
            return self._quarantine(
                observation,
                raw,
                prediction,
                CandidateSafety.unavailable("hard_failure"),
                "hard_failure",
            )

        if (
            self.state in {I2State.QUARANTINED, I2State.RECOVERY}
            and prediction is not None
            and self._distance(raw, prediction) <= self.recovery_consistency_mm
        ):
            return self._recover_clean_observation(observation, raw, prediction)

        if evidence.confirmed_anomaly:
            return self._confirmed_anomaly(observation, raw, prediction)

        if self.state is I2State.QUARANTINED:
            return self._quarantine(
                observation,
                raw,
                prediction,
                CandidateSafety.unavailable("quarantine_unresolved"),
                "quarantine_unresolved",
            )

        if self.state is I2State.RECOVERY:
            return self._quarantine(
                observation,
                raw,
                prediction,
                CandidateSafety.unavailable("recovery_invalidated"),
                "recovery_invalidated",
            )

        if evidence.suspicious and not evidence.legitimate_motion:
            self.state = I2State.SUSPECT
            return self._record(
                observation,
                raw,
                prediction,
                None,
                False,
                False,
                CandidateSafety.unavailable("suspicious_observation"),
                "suspect",
            )

        self._commit_trusted(observation)
        self.state = I2State.NORMAL
        return self._record(
            observation,
            raw,
            prediction,
            None,
            False,
            True,
            CandidateSafety.unavailable("no correction proposed"),
            "legitimate_motion" if evidence.legitimate_motion else "normal",
        )

    def _recover_clean_observation(
        self,
        observation: TimedObservation,
        raw: XYZ,
        prediction: XYZ | None,
    ) -> EnhancedStateOutcome:
        if self.state is I2State.QUARANTINED:
            self.state = I2State.RECOVERY
            self._recovery_observations = [observation]
            return self._record(
                observation,
                raw,
                prediction,
                None,
                False,
                False,
                CandidateSafety.unavailable("recovery_confirmation_pending"),
                "recovery_first_clean_observation",
            )

        if self.state is I2State.RECOVERY:
            self._recovery_observations.append(observation)
            if len(self._recovery_observations) < self.recovery_confirmation_frames:
                return self._record(
                    observation,
                    raw,
                    prediction,
                    None,
                    False,
                    False,
                    CandidateSafety.unavailable("recovery_confirmation_pending"),
                    "recovery_confirmation_pending",
                )
            for recovered in self._recovery_observations:
                self._commit_trusted(recovered)
            self._recovery_observations = []
            self.state = I2State.NORMAL
            self._episode_id = None
            return self._record(
                observation,
                raw,
                prediction,
                None,
                False,
                True,
                CandidateSafety.unavailable("no correction proposed"),
                "recovery_confirmed",
            )
        raise AssertionError("recovery helper requires a recovery state")

    def _confirmed_anomaly(
        self,
        observation: TimedObservation,
        raw: XYZ,
        prediction: XYZ | None,
    ) -> EnhancedStateOutcome:
        if self._episode_id is None:
            self._episode_id = self._next_episode_id
            self._next_episode_id += 1
        self.state = I2State.QUARANTINED
        self._recovery_observations = []
        if prediction is None:
            return self._record(
                observation,
                raw,
                prediction,
                None,
                False,
                False,
                CandidateSafety.unavailable("prediction_unavailable"),
                "anomaly_abstained_without_prediction",
            )
        correction_magnitude = self._distance(raw, prediction)
        safety = CandidateSafety.evaluate(
            candidate_xyz_m=tuple(value / 1000.0 for value in prediction),
            geometry_valid=observation.evidence.geometry_valid,
            correction_authorized=correction_magnitude <= self.max_authorized_correction_mm,
            evidence_sufficient=observation.evidence.evidence_sufficient,
            post_correction_safe=observation.evidence.post_correction_safe,
            i2_state=self.state.value,
        )
        if not safety.safe:
            return self._record(
                observation,
                raw,
                prediction,
                None,
                False,
                False,
                safety,
                "anomaly_abstained_unsafe_candidate",
            )
        return self._record(
            observation,
            raw,
            prediction,
            prediction,
            True,
            False,
            safety,
            "complete_safe_correction",
        )

    def _quarantine(
        self,
        observation: TimedObservation,
        raw: XYZ,
        prediction: XYZ | None,
        safety: CandidateSafety,
        reason: str,
    ) -> EnhancedStateOutcome:
        if self._episode_id is None:
            self._episode_id = self._next_episode_id
            self._next_episode_id += 1
        self.state = I2State.QUARANTINED
        self._recovery_observations = []
        return self._record(observation, raw, prediction, None, False, False, safety, reason)

    def _record(
        self,
        observation: TimedObservation,
        raw: XYZ,
        prediction: XYZ | None,
        corrected: XYZ | None,
        correction_applied: bool,
        trusted_committed: bool,
        safety: CandidateSafety,
        reason: str,
    ) -> EnhancedStateOutcome:
        self.corrected_history.append(corrected)
        return EnhancedStateOutcome(
            frame=observation.frame,
            timestamp_s=observation.timestamp_s,
            raw_xyz_mm=raw,
            prediction_xyz_mm=prediction,
            corrected_xyz_mm=corrected,
            state=self.state,
            correction_applied=correction_applied,
            trusted_committed=trusted_committed,
            candidate_safety=safety,
            episode_id=self._episode_id,
            reason=reason,
        )

    def _commit_trusted(self, observation: TimedObservation) -> None:
        if self.trusted_history and observation.timestamp_s <= self.trusted_history[-1].timestamp_s:
            raise ValueError("trusted observations require increasing timestamps")
        self.trusted_history.append(observation)

    def _predict(self, timestamp_s: float) -> XYZ | None:
        if not self.trusted_history:
            return None
        latest = self.trusted_history[-1]
        dt = float(timestamp_s) - latest.timestamp_s
        if dt < 0:
            raise ValueError("observation timestamps must be monotonic")
        if len(self.trusted_history) < 2 or dt > self.max_prediction_gap_s:
            return latest.xyz_mm
        previous = self.trusted_history[-2]
        previous_dt = latest.timestamp_s - previous.timestamp_s
        if previous_dt <= 0:
            return latest.xyz_mm
        velocity = tuple((latest.xyz_mm[index] - previous.xyz_mm[index]) / previous_dt for index in range(3))
        return tuple(latest.xyz_mm[index] + velocity[index] * dt for index in range(3))

    @staticmethod
    def _finite(values: XYZ) -> bool:
        return all(math.isfinite(value) for value in values)

    @staticmethod
    def _distance(left: XYZ, right: XYZ) -> float:
        return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))
