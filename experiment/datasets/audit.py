from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .models import DatasetSample


AUDIT_FIELDS = [
    "Dataset", "Frames", "Stereo", "GT", "Sequence", "Calibration", "Resolution",
    "Suitable Task", "Official Splits", "Timestamps", "Indexed GiB", "Notes",
]


def _existing_size(paths: Iterable[Path | None]) -> int:
    total = 0
    seen: set[Path] = set()
    for path in paths:
        if path is not None and path not in seen and path.is_file():
            seen.add(path)
            total += path.stat().st_size
    return total


def audit_rows(samples: Iterable[DatasetSample]) -> list[dict[str, object]]:
    groups: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.dataset_name].append(sample)
    rows: list[dict[str, object]] = []
    for dataset_name, items in sorted(groups.items()):
        disparity_count = sum(item.disparity_gt_path is not None for item in items)
        depth_count = sum(item.depth_gt_path is not None for item in items)
        calibration_count = sum(item.calibration_path is not None for item in items)
        timestamp_count = sum(item.timestamp is not None for item in items)
        sequential = any(item.is_sequence for item in items)
        resolutions = sorted({f"{item.width}x{item.height}" for item in items if item.width and item.height})
        if disparity_count and depth_count:
            gt = "disparity+depth"
        elif disparity_count:
            gt = "disparity"
        elif depth_count:
            gt = "depth"
        else:
            gt = "No"
        tasks = ["Innovation1"] if gt != "No" else []
        if sequential:
            tasks.extend(["Innovation2", "Innovation3"])
        notes: list[str] = []
        if not sequential:
            notes.append("flat/static samples; temporal continuity unavailable")
        if calibration_count == 0:
            notes.append("no calibration; metric depth/3D evaluation disabled")
        paths = (
            path
            for item in items
            for path in (item.left_path, item.right_path, item.disparity_gt_path, item.depth_gt_path)
        )
        rows.append({
            "Dataset": dataset_name,
            "Frames": len(items),
            "Stereo": "Yes" if all(item.left_path and item.right_path for item in items) else "Partial",
            "GT": gt,
            "Sequence": "Yes" if sequential else "No",
            "Calibration": "Yes" if calibration_count == len(items) else ("Partial" if calibration_count else "No"),
            "Resolution": ";".join(resolutions) or "Unknown",
            "Suitable Task": "+".join(tasks) or "Pipeline smoke only",
            "Official Splits": ";".join(sorted({item.official_split for item in items if item.official_split})) or "No",
            "Timestamps": "Yes" if timestamp_count == len(items) else ("Partial" if timestamp_count else "No"),
            "Indexed GiB": round(_existing_size(paths) / (1024 ** 3), 3),
            "Notes": "; ".join(notes),
        })
    return rows


def write_audit(samples: Iterable[DatasetSample], results_dir: str | Path) -> tuple[Path, Path]:
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = audit_rows(samples)
    csv_path = output / "dataset_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output / "dataset_summary.md"
    lines = [
        "# Dataset Audit Summary", "", "本报告来自缓存索引，仅统计文件元数据，不加载完整数据集。", "",
    ]
    if not rows:
        lines.append("未发现可配对的双目样本。")
    for row in rows:
        lines.extend([
            f"## {row['Dataset']}", "",
            f"- 双目样本：{row['Frames']} 对；分辨率：{row['Resolution']}。",
            f"- Ground Truth：{row['GT']}；标定：{row['Calibration']}；连续序列：{row['Sequence']}。",
            f"- 推荐用途：{'静态视差精度与创新点一验证' if row['GT'] == 'disparity' else row['Suitable Task']}。",
            f"- 限制：{row['Notes'] or '未发现明显限制。'}", "",
        ])
    lines.extend([
        "## 使用建议", "",
        "- 有视差 GT 的静态样本用于匹配精度、bad-pixel 与创新点一消融。",
        "- 只有具备可信标定或深度 GT 时才报告米制深度和三维误差。",
        "- 只有明确连续且有时间戳/帧序的序列才用于创新点二、三的时序结论。",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path

