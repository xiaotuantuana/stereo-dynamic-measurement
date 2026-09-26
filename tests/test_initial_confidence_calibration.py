from __future__ import annotations

import numpy as np

from stereo_research.global_matching import GlobalSample
from stereo_research.local_matching import InitialMatchDiagnostics
from stereo_research.models import MatcherConfig, PointSpec
from stereo_research.pipeline import TemporalStereoPipeline


def _q() -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0],
    ])


def test_default_initialization_behavior_remains_opt_out() -> None:
    assert MatcherConfig().enable_initial_confidence_calibration is False


def test_opt_in_initialization_rejects_ambiguous_runtime_match(monkeypatch) -> None:
    config = MatcherConfig(
        enable_initial_confidence_calibration=True,
        uniqueness_margin=0.03,
        min_depth_m=1e-6,
    )
    pipeline = TemporalStereoPipeline("M3", _q(), "m", config)
    image = np.random.default_rng(4).integers(0, 256, (80, 160), dtype=np.uint8)
    fake_global = type("Global", (), {"elapsed_ms": 1.0})()
    monkeypatch.setattr(pipeline.global_matcher, "compute", lambda *_args, **_kwargs: fake_global)
    monkeypatch.setattr(
        pipeline.global_matcher,
        "sample_initial",
        lambda _result, xy: GlobalSample("valid", 8.0, (xy[0] - 8.0, xy[1]), 0.1),
    )
    monkeypatch.setattr(
        pipeline.local_matcher,
        "diagnose_initial",
        lambda *_args, **_kwargs: InitialMatchDiagnostics(
            status="valid", predicted_disparity=8.0, predicted_cost=0.2,
            best_disparity=8.0, best_cost=0.2, second_best_disparity=12.0,
            second_best_cost=0.25, uniqueness_margin=0.05, texture_std=20.0,
            candidate_count=100, predicted_best_disagreement=0.0, confidence=0.4,
        ),
    )

    result = pipeline.initialize(image, image, [PointSpec("p", (80.0, 40.0))], 0)[0]

    assert result.status == "initial_ambiguous"
    assert result.confidence == 0.4
    assert result.measured_disparity == 8.0
    assert result.candidate_count == 100
    assert result.confidence_source == "initial_runtime_evidence"
