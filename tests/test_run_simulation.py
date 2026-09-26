from __future__ import annotations

from stereo_dynamic_measurement.run_simulation import run_from_config


def test_yaml_runner_writes_dataset_summary_and_visualizations(tmp_path) -> None:
    config = tmp_path / "simulation.yaml"
    config.write_text(
        """
camera:
  name: smoke-rig
  image_size: [640, 480]
  K_left: [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
  D_left: [0, 0, 0, 0, 0]
  K_right: [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
  D_right: [0, 0, 0, 0, 0]
  R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  T: [-120, 0, 0]
  unit: mm
simulation:
  duration_s: 0.4
  fps: 10
  image_noise_std_px: 0.1
  localization_noise_std_px: 0.05
  random_seed: 4
structure:
  trajectory_kind: impact_decay
  frequency_hz: 3
output_dir: output
""".strip(),
        encoding="utf-8",
    )

    paths = run_from_config(config)

    assert all(path.exists() for path in paths.values())
    assert {"samples_csv", "dataset_npz", "summary_json", "trajectory_plot", "error_plot"} == set(paths)
