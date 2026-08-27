"""Small end-to-end Phase 4 smoke and its three-table audit surface."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from stereo_research.final_arbitration import ExperimentAuthority
from stereo_research.models import PointSpec, SystemMode
from stereo_research.pipeline import TemporalStereoPipeline

from ..datasets.models import DatasetSample
from ..datasets.io import read_image
from ..sequence.models import DatasetSequence
from ..sequence.ablation import _controlled_q
from ..sequence.runner import run_sequences
from ..simulation.controlled_stereo import create_controlled_stereo_sequence


PHASE3_500_MANIFEST_SHA256 = "CA3329880484CA046D54DFB44FF7EC5873BA9CB3F6CD1EA42F582D0A7A6F978C"


def _modes() -> tuple[tuple[str, str, ExperimentAuthority | None], ...]:
    return (
        ("M0", "M0", None),
        ("M1", "M3", None),
        ("M2", "M3", ExperimentAuthority.shadow(SystemMode.ENHANCED_SHADOW)),
        ("M3", "M3", ExperimentAuthority.full_experiment(write_enabled=True)),
    )


def _factory(authority: ExperimentAuthority | None):
    def build(method, q, calibration_unit, config):
        return TemporalStereoPipeline(method, q, calibration_unit, config, experiment_authority=authority)
    return build


def _audit_sequence(
    sequence,
    *,
    method: str,
    authority: ExperimentAuthority | None,
    q: np.ndarray,
    points: tuple[PointSpec, ...],
    mode: str,
) -> list[dict[str, object]]:
    pipeline = TemporalStereoPipeline(method, q, "m", experiment_authority=authority)
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(sequence.frames):
        left, right = read_image(sample.left_path), read_image(sample.right_path)
        results = pipeline.initialize(left, right, points, index) if index == 0 else pipeline.step(left, right, index)
        for result in results:
            baseline_valid = result.status == "valid" and result.final_xyz_m is not None
            rows.append({
                "mode": mode,
                "frame": index,
                "point_id": result.point_id,
                "I1_status": result.i1_status or result.status,
                "I2_status": result.i2_state or "OFF",
                "I3_status": result.fault_class or "OFF",
                "decision": result.committed_decision or "BASELINE",
                "proposed_decision": result.proposed_decision or "BASELINE",
                "result_source": result.result_source or "I1_BASELINE",
                "write_committed": bool(result.write_committed),
                "final_valid": baseline_valid if result.final_valid is None else bool(result.final_valid),
            })
    return rows


def _jump_count(rows: pd.DataFrame) -> int:
    count = 0
    for _, group in rows.groupby("point_id"):
        values = group.sort_values("sequence_frame_index")["final_disparity"].dropna().to_numpy(float)
        count += int(np.sum(np.abs(np.diff(values)) > 10.0))
    return count


def load_phase3_fixed_sequence(manifest_path: str | Path) -> DatasetSequence:
    """Load the Phase 3 500-frame sequence without regenerating any sample."""

    path = Path(manifest_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != PHASE3_500_MANIFEST_SHA256:
        raise ValueError("Phase 3 manifest hash does not match the frozen 500-frame protocol")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences", [])
    if len(sequences) != 1:
        raise ValueError("expected exactly one fixed controlled sequence")
    item = sequences[0]
    if item.get("provenance") != "CONTROLLED" or item.get("metadata", {}).get("frame_count") != 500:
        raise ValueError("manifest is not the approved 500-frame controlled sequence")
    root = Path.cwd()
    frames = [
        DatasetSample(
            dataset_name="CONTROLLED",
            sequence_name="stateful_stereo",
            frame_id=str(frame["frame_id"]),
            left_path=(root / Path(frame["left_path"])).resolve(),
            right_path=(root / Path(frame["right_path"])).resolve(),
            disparity_gt_path=(root / Path(frame["disparity_gt_path"])).resolve(),
            timestamp=float(frame["timestamp"]),
            official_split="controlled",
            is_sequence=True,
        )
        for frame in item["frames"]
    ]
    if len(frames) != 500 or not all(frame.left_path.exists() and frame.right_path.exists() for frame in frames):
        raise FileNotFoundError("fixed Phase 3 image sequence is incomplete")
    return DatasetSequence.from_frames(frames, provenance="CONTROLLED", metadata=dict(item["metadata"]))


def run_phase4_full_smoke(
    output_dir: str | Path,
    *,
    frame_count: int = 12,
    sequence: DatasetSequence | None = None,
) -> dict[str, object]:
    """Run the four approved system labels on a small CONTROLLED sequence."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if sequence is None:
        sequence = create_controlled_stereo_sequence(output / "controlled_sequence", frame_count=frame_count)
    frame_count = len(sequence.frames)
    q = _controlled_q()
    points = (
        PointSpec("P1", (160.0, 60.0)),
        PointSpec("P2", (200.0, 90.0)),
        PointSpec("P3", (240.0, 120.0)),
    )
    comparison_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    lifecycle_ok = True
    for mode, method, authority in _modes():
        started = time.perf_counter()
        summary = run_sequences(
            [sequence], output / mode, method=method, q=q, calibration_unit="m", points=points,
            pipeline_factory=_factory(authority),
        )
        runtime_ms = (time.perf_counter() - started) * 1000.0
        frame_rows = pd.read_csv(output / mode / "frame_results.csv")
        audit = _audit_sequence(sequence, method=method, authority=authority, q=q, points=points, mode=mode)
        audit_rows.extend(audit)
        false_correction = sum(
            row["write_committed"] and row["result_source"] == "I2_CORRECTED" for row in audit
        ) / max(len(audit), 1)
        comparison_rows.append({
            "mode": mode,
            "coverage": summary.get("coverage"),
            "mae": summary.get("valid_mae"),
            "rmse": summary.get("valid_rmse"),
            "cer10": summary.get("cer_10"),
            "jitter": summary.get("frame_to_frame_jitter_px"),
            "jump_count": _jump_count(frame_rows),
            "false_correction": false_correction,
            "runtime": runtime_ms,
        })
        lifecycle = summary["lifecycle"]
        lifecycle_ok = lifecycle_ok and lifecycle["initialize_calls"] == 1 and lifecycle["step_calls"] == frame_count - 1
    comparison_fields = [
        "mode", "coverage", "mae", "rmse", "cer10", "jitter", "jump_count", "false_correction", "runtime",
    ]
    audit_fields = [
        "mode", "frame", "point_id", "I1_status", "I2_status", "I3_status", "decision",
        "proposed_decision", "result_source", "write_committed", "final_valid",
    ]
    with (output / "full_enhanced_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    with (output / "final_arbitration_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    return {"provenance": "CONTROLLED", "stateful_protocol": lifecycle_ok, "modes": [row["mode"] for row in comparison_rows]}


def run_phase4_full_manifest_validation(
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Run the Phase 4 four-mode matrix on the frozen Phase 3 500-frame input."""

    return run_phase4_full_smoke(output_dir, sequence=load_phase3_fixed_sequence(manifest_path))
