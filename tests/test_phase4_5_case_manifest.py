from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from experiment.validation.phase4_5 import build_phase4_5_case_windows


_MANIFEST = Path("results/phase3/stateful_innovation1_500/M3/sequence_manifest.json")


def test_phase4_5_builds_32_frozen_cases_with_paired_clean_controls() -> None:
    cases = build_phase4_5_case_windows(_MANIFEST)

    assert len(cases) == 32
    assert Counter(case.injected.anomaly_type for case in cases) == {
        "local_occlusion": 8,
        "local_blur": 8,
        "unilateral_roi_shift": 8,
        "continuous_anomaly": 8,
    }
    for case in cases:
        assert case.source_manifest_sha256
        assert set(case.injected.active_frame_ids) <= set(case.window_frame_ids)
        assert len(case.warmup_frame_ids) == 2
        assert len(case.recovery_frame_ids) == 2
        assert case.clean_pair_id == case.injected.pair_id
        assert case.clean_source_frame_ids == case.window_frame_ids
        assert case.injected_source_frame_ids == case.window_frame_ids


def test_phase4_5_case_manifest_rejects_non_frozen_source(tmp_path: Path) -> None:
    changed = tmp_path / "sequence_manifest.json"
    changed.write_bytes(_MANIFEST.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="frozen"):
        build_phase4_5_case_windows(changed)
