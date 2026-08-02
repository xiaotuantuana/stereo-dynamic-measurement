from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"))

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QSize, Qt

from stereo_research.gui.settings import build_matcher_config, parameter_groups
from stereo_research.gui.processing import ProcessingRequest, StereoVideoProcessor
from stereo_research.gui.video import (
    inspect_stereo_video,
    normalize_disparity_image,
    validate_stereo_frame_shape,
)
from stereo_research.gui.widgets import AspectImageView
from stereo_research.gui.window import StereoMainWindow
from stereo_research.gui.app import configure_application
from stereo_research.models import PointSpec
from stereo_research.runner import CSV_FIELDS, load_calibration


def test_configure_application_sets_product_identity(qapp) -> None:
    configure_application(qapp)

    assert qapp.applicationName() == "Stereo Measurement Studio"
    assert qapp.organizationName() == "Temporal Stereo Research"
    assert qapp.font().pointSize() >= 9


def test_windows_launcher_uses_cmd_compatible_crlf_and_ascii() -> None:
    launcher = Path(__file__).resolve().parents[1] / "启动双目测量软件.bat"
    raw = launcher.read_bytes()

    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")
    raw.decode("ascii")


def test_clicking_visible_image_selects_the_matching_image_coordinate(qtbot) -> None:
    view = AspectImageView("左相机 / LEFT")
    qtbot.addWidget(view)
    view.resize(320, 272)
    view.selectable = True
    view.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    view.show()
    qtbot.waitExposed(view)

    with qtbot.waitSignal(view.pointClicked, timeout=1_000) as blocker:
        qtbot.mouseClick(
            view.image_label,
            Qt.MouseButton.LeftButton,
            pos=QPoint(view.image_label.width() // 2, view.image_label.height() // 2),
        )

    x, y = blocker.args
    assert x == pytest.approx(320.0, abs=1.0)
    assert y == pytest.approx(240.0, abs=1.0)


def test_closing_window_requests_worker_stop_and_waits_for_thread(qtbot) -> None:
    class FakeWorker:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeThread:
        waited = False

        def isRunning(self) -> bool:
            return True

        def quit(self) -> None:
            pass

        def wait(self, timeout: int) -> bool:
            self.waited = timeout >= 1_000
            return True

    window = StereoMainWindow()
    qtbot.addWidget(window)
    worker = FakeWorker()
    thread = FakeThread()
    window._worker = worker
    window._thread = thread
    window._processing = True

    window.close()

    assert worker.stopped is True
    assert thread.waited is True


def test_sidebar_table_fits_and_advanced_group_has_usable_viewport(qtbot) -> None:
    window = StereoMainWindow()
    qtbot.addWidget(window)
    window.resize(1560, 920)
    window.add_point((140.0, 320.0), role="measurement")
    window.add_point((239.0, 350.0), role="reference")
    window.show()
    window.advanced_toggle.setChecked(True)
    qtbot.wait(50)

    assert window.point_table.horizontalScrollBar().maximum() == 0
    assert window.advanced_group_combo.count() == len(parameter_groups())
    assert window.advanced_stack.currentWidget().height() >= 100


def _write_stereo_video(path: Path, frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (1280, 480),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(91)
    for _ in range(frames):
        left = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        right = np.roll(left, -4, axis=1)
        writer.write(np.hstack((left, right)))
    writer.release()


def test_video_inspection_reports_side_by_side_metadata(tmp_path: Path) -> None:
    video = tmp_path / "tiny.avi"
    _write_stereo_video(video)

    metadata = inspect_stereo_video(video)

    assert metadata.frame_size == (1280, 480)
    assert metadata.view_size == (640, 480)
    assert metadata.frame_count == 3
    assert metadata.fps == pytest.approx(10.0, rel=0.1)


def test_odd_width_frame_is_rejected_with_recovery_hint() -> None:
    with pytest.raises(ValueError, match="左右并排"):
        validate_stereo_frame_shape((48, 127, 3))


def test_parameter_schema_builds_valid_backend_config() -> None:
    groups = parameter_groups()
    keys = {item.key for group in groups for item in group.items}

    assert {"patch_size", "search_radius", "icgn_max_iterations", "kalman_nis_hard_threshold"} <= keys
    config = build_matcher_config(
        {
            "patch_size": 15,
            "search_radius": 12,
            "enable_cycle_consistency": False,
            "icgn_fallback_method": "line_search",
        }
    )
    assert config.patch_size == 15
    assert config.search_radius == 12
    assert config.enable_cycle_consistency is False


def test_disparity_visualization_handles_invalid_pixels() -> None:
    disparity = np.array([[np.nan, -1.0, 0.0], [4.0, 8.0, 16.0]], dtype=np.float32)

    image = normalize_disparity_image(disparity)

    assert image.shape == (2, 3, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image[0, 0], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(image[0, 1], np.zeros(3, dtype=np.uint8))


def test_aspect_view_maps_letterboxed_click_to_image_coordinates(qtbot) -> None:
    view = AspectImageView("左相机")
    qtbot.addWidget(view)
    view.resize(QSize(320, 320))
    view.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    center = view.widget_to_image(QPointF(160.0, 176.0))
    outside = view.widget_to_image(QPointF(160.0, 20.0))

    assert center == pytest.approx((320.0, 240.0), abs=1.0)
    assert outside is None


def test_main_window_has_four_views_and_progressive_parameters(qtbot) -> None:
    window = StereoMainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Stereo Measurement Studio"
    assert [view.title for view in window.video_views] == [
        "左相机 / LEFT",
        "右相机 / RIGHT",
        "视差质量 / DISPARITY",
        "测量结果 / MEASUREMENT",
    ]
    assert window.advanced_toggle.isChecked() is False
    assert window.advanced_container.isHidden() is True
    assert window.run_button.isEnabled() is False
    assert window.method_combo.count() >= 6


def test_window_requires_a_point_before_processing(qtbot, tmp_path: Path) -> None:
    video = tmp_path / "tiny.avi"
    _write_stereo_video(video)
    window = StereoMainWindow()
    qtbot.addWidget(window)

    window.load_video(video)
    assert window.run_button.isEnabled() is False
    window.add_point((32.0, 24.0), role="measurement")

    assert window.run_button.isEnabled() is True
    assert window.point_table.rowCount() == 1


def test_processor_runs_real_backend_and_exports_csv(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "gui_result.csv"
    request = ProcessingRequest(
        video_path=project / "car.avi",
        output_path=output,
        method="sgbm",
        start_frame=250,
        end_frame=251,
        points=(
            PointSpec("P1", (140.0, 320.0)),
            PointSpec("P2", (239.0, 350.0)),
            PointSpec("P3", (348.0, 353.0)),
        ),
        calibration=load_calibration(str(project / "configs" / "calibration_640x480.json")),
        config=build_matcher_config({}),
    )
    previews = []

    summary = StereoVideoProcessor(request).run(on_frame=previews.append)

    assert summary.frames_processed == 2
    assert summary.rows_written == 6
    assert summary.cancelled is False
    assert output.exists()
    assert len(previews) == 2
    assert previews[-1].left.shape[:2] == (480, 640)
    assert previews[-1].disparity_bgr.shape == (480, 640, 3)
    header = output.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert header == CSV_FIELDS


def test_processing_request_rejects_reversed_frame_range(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="结束帧"):
        ProcessingRequest(
            video_path=project / "car.avi",
            output_path=tmp_path / "invalid.csv",
            method="sgbm",
            start_frame=20,
            end_frame=10,
            points=(PointSpec("P1", (140.0, 320.0)),),
            calibration=load_calibration(str(project / "configs" / "calibration_640x480.json")),
            config=build_matcher_config({}),
        )


def test_window_builds_request_and_completes_background_processing(qtbot, tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "window_result.csv"
    window = StereoMainWindow()
    qtbot.addWidget(window)
    window.load_video(project / "car.avi")
    window.start_frame_spin.setValue(250)
    window.end_frame_spin.setValue(251)
    window.add_point((140.0, 320.0), role="measurement")
    window.add_point((239.0, 350.0), role="reference")

    request = window.build_processing_request(output)
    assert request.start_frame == 250
    assert request.end_frame == 251
    assert request.method == "research_full"
    assert request.points[1].role == "reference"

    with qtbot.waitSignal(window.processingFinished, timeout=15_000) as blocker:
        window.start_processing(output)

    summary = blocker.args[0]
    assert summary.frames_processed == 2
    assert output.exists()
    assert window.progress_bar.value() == 100
    assert window.run_button.isEnabled() is True
