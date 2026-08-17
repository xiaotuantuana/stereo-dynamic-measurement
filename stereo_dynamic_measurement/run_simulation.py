from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .calibration.camera_model import StereoCameraModel
from .simulation.synthetic_dataset import SimulationConfig, generate_synthetic_dataset, write_synthetic_dataset
from .simulation.trajectory_generator import MultiPointStructure
from .visualization.plots import plot_error_components, plot_trajectories


def _structure_from_mapping(values: dict[str, Any]) -> MultiPointStructure:
    defaults = asdict(MultiPointStructure.default())
    defaults.update(values)
    for field_name in ("base_positions_mm", "amplitudes_mm", "phases_rad"):
        defaults[field_name] = {key: tuple(value) if field_name == "base_positions_mm" else value for key, value in defaults[field_name].items()}
    return MultiPointStructure(**defaults)


def run_from_config(config_path: str | Path) -> dict[str, Path]:
    source = Path(config_path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "camera" not in payload:
        raise ValueError("Simulation YAML requires a camera mapping")
    simulation_values = dict(payload.get("simulation", {}))
    camera = StereoCameraModel.from_mapping(payload["camera"])
    structure = _structure_from_mapping(dict(payload.get("structure", {})))
    dataset = generate_synthetic_dataset(SimulationConfig(camera=camera, structure=structure, **simulation_values))
    output_dir = Path(payload.get("output_dir", "outputs/simulation"))
    if not output_dir.is_absolute():
        output_dir = source.parent / output_dir
    paths = write_synthetic_dataset(dataset, output_dir)
    paths["trajectory_plot"] = plot_trajectories(dataset.samples, output_dir / "trajectories.png")
    paths["error_plot"] = plot_error_components(dataset.samples, output_dir / "reconstruction_errors.png")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and reconstruct a dynamic stereo synthetic dataset.")
    parser.add_argument("--config", default="configs/simulation.yaml", help="Path to simulation YAML")
    args = parser.parse_args()
    paths = run_from_config(args.config)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
