from __future__ import annotations

from pathlib import Path

from experiment.validation.phase5_static_manifest import select_frozen_static_records


def test_select_frozen_static_records_uses_sha256_rank_and_excludes_temporal_rows() -> None:
    records = [
        {"sample_id": "z", "is_sequence": False},
        {"sample_id": "a", "is_sequence": False},
        {"sample_id": "temporal", "is_sequence": True},
    ]

    selected = select_frozen_static_records(records, target_count=2)

    assert len(selected) == 2
    assert all(not row["is_sequence"] for row in selected)
    assert [row["sample_id"] for row in selected] == ["z", "a"]
