from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..calibration import StereoCalibration
from ..models import MatcherConfig, MethodName, PointSpec
from ..runner import load_calibration, split_side_by_side
from .settings import ParameterSpec, build_matcher_config, parameter_groups
from .processing import BatchProcessingSummary, BatchProcessingWorker, FramePreview, ProcessingRequest, ProcessingSummary, ProcessingWorker
from .theme import APP_STYLE
from .video import VideoMetadata, inspect_stereo_video
from .widgets import AspectImageView


METHOD_OPTIONS: tuple[tuple[str, MethodName], ...] = (
    ("完整研究方法 · 时空预测 + 闭环 + IC-GN + 滤波", "research_full"),
    ("论文最终方法 · 创新1 + 闭环 + 多参考刚体补偿", "THESIS_FULL"),
    ("M3 · 创新点1自适应局部匹配", "M3"),
    ("M2 · 光流 + 历史预测", "M2"),
    ("M1 · 光流 + 固定局部窗口", "M1"),
    ("M0 · 全局 SGBM", "M0"),
    ("质量优先局部方法 · 多尺度恢复", "full_quality"),
    ("完整局部方法 · 预测 + 约束 + 亚像素", "full"),
    ("局部搜索 + 左图光流", "local_flow"),
    ("仅局部搜索", "local"),
    ("全局 SGBM", "sgbm"),
    ("SGBM + 左图光流（附录控制）", "sgbm_flow"),
)


class StereoMainWindow(QMainWindow):
    processingFinished = Signal(object)
    processingFailed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stereo Measurement Studio")
        self.setObjectName("mainWindow")
        self.resize(1560, 920)
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(APP_STYLE)
        self.video_path: Path | None = None
        self.video_metadata: VideoMetadata | None = None
        self.calibration_path = self._default_calibration_path()
        self.calibration: StereoCalibration = load_calibration(str(self.calibration_path))
        self.current_left = None
        self.current_right = None
        self.points: list[PointSpec] = []
        self.parameter_controls: dict[str, QWidget] = {}
        self._processing = False
        self._thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self.last_output_path: Path | None = None
        self._build_ui()
        self._install_shortcuts()
        self._set_status("等待输入", "请选择左右并排的双目视频")

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_workspace())
        splitter.setSizes([360, 1160])
        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(430)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        brand = QLabel("Stereo Measurement Studio")
        brand.setObjectName("brand")
        sub = QLabel("时空预测驱动 · 双目三维测量工作站")
        sub.setObjectName("brandSub")
        layout.addWidget(brand)
        layout.addWidget(sub)

        layout.addWidget(self._section_label("01  输入视频"))
        path_row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("选择左右并排视频…")
        self.browse_button = QPushButton("选择视频")
        self.browse_button.setAccessibleName("选择输入视频")
        self.browse_button.clicked.connect(self.choose_video)
        path_row.addWidget(self.video_path_edit, 1)
        path_row.addWidget(self.browse_button)
        layout.addLayout(path_row)

        self.video_info_label = QLabel("分辨率 —  |  帧数 —  |  FPS —")
        self.video_info_label.setObjectName("brandSub")
        self.video_info_label.setWordWrap(True)
        layout.addWidget(self.video_info_label)

        frame_row = QGridLayout()
        frame_row.addWidget(QLabel("起始帧"), 0, 0)
        frame_row.addWidget(QLabel("结束帧"), 0, 1)
        self.start_frame_spin = QSpinBox()
        self.end_frame_spin = QSpinBox()
        for control in (self.start_frame_spin, self.end_frame_spin):
            control.setRange(0, 0)
        self.start_frame_spin.valueChanged.connect(self._on_preview_frame_changed)
        frame_row.addWidget(self.start_frame_spin, 1, 0)
        frame_row.addWidget(self.end_frame_spin, 1, 1)
        layout.addLayout(frame_row)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.start_frame_spin.setValue)
        layout.addWidget(self.frame_slider)

        layout.addWidget(self._section_label("02  方法与测点"))
        self.method_combo = QComboBox()
        for label, method in METHOD_OPTIONS:
            self.method_combo.addItem(label, method)
        self.method_combo.setAccessibleName("选择双目处理方法")
        layout.addWidget(self.method_combo)

        point_actions = QHBoxLayout()
        self.point_role_combo = QComboBox()
        self.point_role_combo.addItem("测量点", "measurement")
        self.point_role_combo.addItem("静态参考点", "reference")
        self.undo_point_button = QPushButton("撤销")
        self.clear_points_button = QPushButton("清空")
        self.undo_point_button.clicked.connect(self.undo_point)
        self.clear_points_button.clicked.connect(self.clear_points)
        point_actions.addWidget(self.point_role_combo, 1)
        point_actions.addWidget(self.undo_point_button)
        point_actions.addWidget(self.clear_points_button)
        layout.addLayout(point_actions)
        hint = QLabel("在左相机画面单击添加测点；黄色为测量点，绿色为静态参考点。")
        hint.setObjectName("brandSub")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.point_table = QTableWidget(0, 4)
        self.point_table.setHorizontalHeaderLabels(("ID", "类型", "X", "Y"))
        header = self.point_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setMinimumSectionSize(54)
        self.point_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.point_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.point_table.setMaximumHeight(140)
        layout.addWidget(self.point_table)

        self.advanced_toggle = QPushButton("高级参数  ▸")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setAccessibleName("展开或折叠高级参数")
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)
        self.advanced_container = self._build_advanced_panel()
        self.advanced_container.hide()
        layout.addWidget(self.advanced_container, 1)

        self.run_button = QPushButton("开始处理")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.start_processing)
        controls = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.setCheckable(True)
        self.stop_button = QPushButton("停止")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        self.pause_button.toggled.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.run_button)
        layout.addWidget(self._section_label("03  论文正式实验"))
        self.experiment_combo = QComboBox()
        self.experiment_combo.addItem("创新点1：M0 / M1 / M2 / M3", "innovation1")
        self.experiment_combo.addItem("创新点2：M3_C0 / C1 / C2 / C3", "confidence")
        self.experiment_combo.addItem("创新点3：THESIS_FULL_R0 / R1 / R2", "compensation")
        self.experiment_combo.addItem("最终方法：THESIS_FULL_R2", "final")
        self.experiment_combo.setAccessibleName("选择论文正式实验")
        layout.addWidget(self.experiment_combo)
        self.experiment_button = QPushButton("运行所选正式实验")
        self.experiment_button.setEnabled(False)
        self.experiment_button.clicked.connect(self.start_thesis_experiment)
        layout.addWidget(self.experiment_button)
        layout.addLayout(controls)
        return sidebar

    def _build_advanced_panel(self) -> QWidget:
        container = QFrame()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.advanced_group_combo = QComboBox()
        self.advanced_group_combo.setAccessibleName("高级参数分组")
        reset = QPushButton("恢复默认参数")
        reset.clicked.connect(self.reset_parameters)
        toolbar.addWidget(self.advanced_group_combo, 1)
        toolbar.addWidget(reset)
        layout.addLayout(toolbar)
        self.advanced_stack = QStackedWidget()
        self.advanced_stack.setMinimumHeight(120)
        self.advanced_group_combo.currentIndexChanged.connect(self.advanced_stack.setCurrentIndex)
        defaults = asdict(MatcherConfig())
        for group in parameter_groups():
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(10, 10, 10, 10)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(7)
            for spec in group.items:
                control = self._parameter_control(spec, defaults[spec.key])
                self.parameter_controls[spec.key] = control
                form.addRow(spec.label, control)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.advanced_group_combo.addItem(group.title)
            self.advanced_stack.addWidget(scroll)
        layout.addWidget(self.advanced_stack, 1)
        return container

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        self.status_pill = QLabel("等待输入")
        self.status_pill.setObjectName("statusPill")
        self.status_detail = QLabel("请选择视频")
        self.status_detail.setObjectName("brandSub")
        top_layout.addWidget(self.status_pill)
        top_layout.addWidget(self.status_detail, 1)
        self.calibration_button = QPushButton("标定：640×480")
        self.calibration_button.clicked.connect(self.choose_calibration)
        top_layout.addWidget(self.calibration_button)
        layout.addWidget(top_bar)

        grid = QGridLayout()
        grid.setSpacing(8)
        titles = (
            "左相机 / LEFT",
            "右相机 / RIGHT",
            "视差质量 / DISPARITY",
            "测量结果 / MEASUREMENT",
        )
        self.video_views = [AspectImageView(title) for title in titles]
        self.video_views[0].selectable = True
        self.video_views[0].pointClicked.connect(self._on_point_clicked)
        for index, view in enumerate(self.video_views):
            grid.addWidget(view, index // 2, index % 2)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)

        metric_strip = QFrame()
        metric_strip.setObjectName("metricStrip")
        metric_layout = QHBoxLayout(metric_strip)
        self.metric_values: dict[str, QLabel] = {}
        for key, label, value in (
            ("frame", "当前帧", "—"),
            ("valid", "有效测点", "—"),
            ("depth", "平均深度", "—"),
            ("runtime", "处理耗时", "—"),
            ("method", "当前方法", "—"),
        ):
            card = QWidget()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 4, 12, 4)
            value_label = QLabel(value)
            value_label.setObjectName("metricValue")
            name_label = QLabel(label)
            name_label.setObjectName("metricLabel")
            card_layout.addWidget(value_label)
            card_layout.addWidget(name_label)
            metric_layout.addWidget(card, 1)
            self.metric_values[key] = value_label
        layout.addWidget(metric_strip)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("等待处理")
        layout.addWidget(self.progress_bar)
        return workspace

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择双目视频",
            str(self.video_path.parent if self.video_path else Path.cwd()),
            "视频文件 (*.avi *.mp4 *.mov *.mkv *.wmv);;所有文件 (*.*)",
        )
        if path:
            self.load_video(path)

    def load_video(self, path: str | Path) -> None:
        try:
            metadata = inspect_stereo_video(path)
            if metadata.view_size != self.calibration.image_size:
                expected = self.calibration.image_size
                raise ValueError(
                    f"当前标定要求单目 {expected[0]}×{expected[1]}，"
                    f"视频单目为 {metadata.view_size[0]}×{metadata.view_size[1]}。"
                    "请在右上角选择匹配的标定文件。"
                )
            self.video_path = metadata.path
            self.video_metadata = metadata
            self.video_path_edit.setText(str(metadata.path))
            maximum = max(0, metadata.frame_count - 1)
            for control in (self.start_frame_spin, self.end_frame_spin):
                control.blockSignals(True)
                control.setRange(0, maximum)
                control.blockSignals(False)
            self.start_frame_spin.setValue(0)
            self.end_frame_spin.setValue(maximum)
            self.frame_slider.setRange(0, maximum)
            self.frame_slider.setValue(0)
            self.video_info_label.setText(
                f"{metadata.frame_size[0]}×{metadata.frame_size[1]}  |  "
                f"{metadata.frame_count} 帧  |  {metadata.fps:.2f} FPS  |  "
                f"{metadata.duration_s:.1f} 秒"
            )
            self.clear_points()
            self._load_preview_frame(0)
            self._set_status("视频已载入", "请在左图选择至少一个纹理清晰的测点")
        except Exception as exc:
            self._show_error("视频载入失败", str(exc))
            self.video_path = None
            self.video_metadata = None
            self._update_run_enabled()

    def choose_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标定 JSON",
            str(self.calibration_path.parent),
            "JSON 标定文件 (*.json)",
        )
        if not path:
            return
        try:
            calibration = load_calibration(path)
            self.calibration = calibration
            self.calibration_path = Path(path).resolve()
            self.calibration_button.setText(
                f"标定：{calibration.image_size[0]}×{calibration.image_size[1]}"
            )
            if self.video_path is not None:
                self.load_video(self.video_path)
        except Exception as exc:
            self._show_error("标定载入失败", str(exc))

    def add_point(self, xy: tuple[float, float], role: str = "measurement") -> None:
        if self.current_left is None:
            return
        if role not in {"measurement", "reference"}:
            raise ValueError("测点类型必须是 measurement 或 reference")
        height, width = self.current_left.shape[:2]
        x, y = float(xy[0]), float(xy[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("测点超出左图范围")
        point_id = f"P{len(self.points) + 1}"
        self.points.append(PointSpec(point_id, (x, y), role=role))
        self._refresh_point_table()
        self._refresh_preview_points()
        self._update_run_enabled()

    def undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self._renumber_points()

    def clear_points(self) -> None:
        self.points.clear()
        if hasattr(self, "point_table"):
            self._refresh_point_table()
            self._refresh_preview_points()
            self._update_run_enabled()

    def current_config(self) -> MatcherConfig:
        values: dict[str, Any] = {}
        for key, control in self.parameter_controls.items():
            if isinstance(control, QCheckBox):
                values[key] = control.isChecked()
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                values[key] = control.value()
            elif isinstance(control, QComboBox):
                values[key] = control.currentData() or control.currentText()
        return build_matcher_config(values)

    def reset_parameters(self) -> None:
        defaults = asdict(MatcherConfig())
        for key, control in self.parameter_controls.items():
            value = defaults[key]
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setValue(value)
            elif isinstance(control, QComboBox):
                index = control.findData(value)
                control.setCurrentIndex(max(index, 0))
        self.statusBar().showMessage("高级参数已恢复默认值", 3000)

    def build_processing_request(self, output_path: str | Path) -> ProcessingRequest:
        if self.video_path is None:
            raise ValueError("请先选择视频")
        if not self.points:
            raise ValueError("请至少添加一个测点")
        return ProcessingRequest(
            video_path=self.video_path,
            output_path=Path(output_path).resolve(),
            method=self.method_combo.currentData(),
            start_frame=self.start_frame_spin.value(),
            end_frame=self.end_frame_spin.value(),
            points=tuple(self.points),
            calibration=self.calibration,
            config=self.current_config(),
        )

    def _thesis_requests(self) -> tuple[ProcessingRequest, ...]:
        if self.video_path is None or not self.points:
            raise ValueError("请先选择视频并添加测点")
        group = str(self.experiment_combo.currentData())
        base = self.current_config()
        root = self.video_path.parent / "stereo_outputs" / "thesis_experiments"
        if group == "innovation1":
            variants = ((name, name, base) for name in ("M0", "M1", "M2", "M3"))
        elif group == "confidence":
            variants = (
                ("M3_C0", "THESIS_FULL", replace(base, confidence_mode="none", enable_camera_compensation=False)),
                ("M3_C1", "THESIS_FULL", replace(base, confidence_mode="single_margin", enable_camera_compensation=False)),
                ("M3_C2", "THESIS_FULL", replace(base, confidence_mode="multi_reject", enable_camera_compensation=False)),
                ("M3_C3", "THESIS_FULL", replace(base, confidence_mode="closed_loop", enable_camera_compensation=False)),
            )
        elif group == "compensation":
            variants = tuple(
                (label, "THESIS_FULL", replace(base, confidence_mode="closed_loop", camera_compensation_mode=mode, enable_camera_compensation=mode != "none"))
                for label, mode in (("THESIS_FULL_R0", "none"), ("THESIS_FULL_R1", "single_reference"), ("THESIS_FULL_R2", "multi_reference_rigid"))
            )
        else:
            variants = (("THESIS_FULL_R2", "THESIS_FULL", replace(base, confidence_mode="closed_loop", camera_compensation_mode="multi_reference_rigid", enable_camera_compensation=True)),)
        return tuple(
            ProcessingRequest(
                video_path=self.video_path,
                output_path=root / f"{label}.csv",
                method=method,
                start_frame=self.start_frame_spin.value(),
                end_frame=self.end_frame_spin.value(),
                points=tuple(self.points), calibration=self.calibration,
                config=config, result_method_label=label,
            )
            for label, method, config in variants
        )

    def start_thesis_experiment(self) -> None:
        try:
            requests = self._thesis_requests()
        except ValueError as exc:
            self._show_error("无法启动正式实验", str(exc))
            return
        self.last_output_path = requests[0].output_path
        self._start_worker(BatchProcessingWorker(requests), f"正式实验将保存到 {requests[0].output_path.parent}")

    def start_processing(self, output_path: str | Path | None = None) -> None:
        if not self.video_path or not self.points:
            return
        try:
            if output_path is None:
                method = str(self.method_combo.currentData())
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = (
                    self.video_path.parent
                    / "stereo_outputs"
                    / f"{self.video_path.stem}_{method}_{stamp}.csv"
                )
            request = self.build_processing_request(output_path)
        except ValueError as exc:
            self._show_error("参数无效", str(exc))
            return
        self.last_output_path = request.output_path
        self.metric_values["method"].setText(str(request.method))
        self._start_worker(ProcessingWorker(request), f"结果将保存到 {request.output_path}")

    def _start_worker(self, worker: ProcessingWorker | BatchProcessingWorker, detail: str) -> None:
        self._processing = True
        self._set_processing_controls(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备处理…")
        self._set_status("处理中", detail)

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.frameReady.connect(self._on_processing_frame)
        worker.statusChanged.connect(lambda text: self.status_pill.setText(text))
        worker.completed.connect(thread.quit)
        worker.completed.connect(self._on_processing_complete)
        worker.failed.connect(thread.quit)
        worker.failed.connect(self._on_processing_failed)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def stop_processing(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.stop_button.setEnabled(False)
            self._set_status("正在停止", "当前帧完成后安全停止并保留已写入 CSV")

    def _toggle_pause(self, paused: bool) -> None:
        if self._worker is None:
            self.pause_button.blockSignals(True)
            self.pause_button.setChecked(False)
            self.pause_button.blockSignals(False)
            return
        self._worker.set_paused(paused)
        self.pause_button.setText("继续" if paused else "暂停")

    def _on_processing_frame(self, preview: FramePreview) -> None:
        role_by_id = {point.point_id: point.role for point in self.points}
        left_markers = [
            (result.left_x, result.left_y, result.point_id, role_by_id.get(result.point_id, "measurement"))
            for result in preview.results
            if result.left_x is not None and result.left_y is not None
        ]
        right_markers = [
            (result.right_x, result.right_y, result.point_id, role_by_id.get(result.point_id, "measurement"))
            for result in preview.results
            if result.right_x is not None and result.right_y is not None
        ]
        self.current_left = preview.left
        self.current_right = preview.right
        self.video_views[0].set_frame(preview.left, left_markers)
        self.video_views[1].set_frame(preview.right, right_markers)
        self.video_views[2].set_frame(preview.disparity_bgr)
        self.video_views[3].set_frame(preview.measurement_bgr)
        valid = [result for result in preview.results if result.status == "valid"]
        depths = [result.z_m for result in valid if result.z_m is not None]
        self.metric_values["frame"].setText(str(preview.frame))
        self.metric_values["valid"].setText(f"{len(valid)} / {len(preview.results)}")
        self.metric_values["depth"].setText(
            f"{sum(depths) / len(depths):.3f} m" if depths else "无有效值"
        )
        self.metric_values["runtime"].setText(f"{preview.frame_total_ms:.1f} ms")
        self.progress_bar.setValue(preview.progress_pct)
        self.progress_bar.setFormat(f"第 {preview.frame} 帧  ·  {preview.progress_pct}%")

    def _on_processing_complete(self, summary: ProcessingSummary | BatchProcessingSummary) -> None:
        self._processing = False
        self._set_processing_controls(False)
        if isinstance(summary, BatchProcessingSummary):
            paths = "\n".join(str(item.output_path) for item in summary.summaries)
            self.progress_bar.setValue(100 if not summary.cancelled else self.progress_bar.value())
            self.progress_bar.setFormat("正式实验完成" if not summary.cancelled else "正式实验已停止")
            self._set_status("正式实验完成" if not summary.cancelled else "正式实验已停止", f"已输出 {len(summary.summaries)} 组 CSV：\n{paths}")
            return
        if summary.cancelled:
            self._set_status(
                "已停止",
                f"已保留 {summary.frames_processed} 帧结果：{summary.output_path}",
            )
        else:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("处理完成  ·  100%")
            self._set_status(
                "处理完成",
                f"{summary.frames_processed} 帧，{summary.valid_rows}/{summary.rows_written} 个有效点帧；已保存 CSV",
            )
        self.processingFinished.emit(summary)

    def _on_processing_failed(self, message: str) -> None:
        self._processing = False
        self._set_processing_controls(False)
        self.progress_bar.setFormat("处理失败")
        self.processingFailed.emit(message)
        self._show_error("视频处理失败", f"{message}\n\n请检查视频、标定尺寸和高级参数。")

    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None

    def _set_processing_controls(self, processing: bool) -> None:
        self.browse_button.setEnabled(not processing)
        self.method_combo.setEnabled(not processing)
        self.start_frame_spin.setEnabled(not processing)
        self.end_frame_spin.setEnabled(not processing)
        self.frame_slider.setEnabled(not processing)
        self.advanced_toggle.setEnabled(not processing)
        self.experiment_combo.setEnabled(not processing)
        self.experiment_button.setEnabled(False if processing else bool(self.video_path and self.points))
        self.run_button.setEnabled(False if processing else bool(self.video_path and self.points))
        self.pause_button.setEnabled(processing)
        self.stop_button.setEnabled(processing)
        if not processing:
            self.pause_button.blockSignals(True)
            self.pause_button.setChecked(False)
            self.pause_button.setText("暂停")
            self.pause_button.blockSignals(False)

    def _on_point_clicked(self, x: float, y: float) -> None:
        self.add_point((x, y), str(self.point_role_combo.currentData()))

    def _on_preview_frame_changed(self, frame: int) -> None:
        if self.video_path is None or self._processing:
            return
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame)
        self.frame_slider.blockSignals(False)
        if self.end_frame_spin.value() < frame:
            self.end_frame_spin.setValue(frame)
        self.clear_points()
        self._load_preview_frame(frame)

    def _load_preview_frame(self, frame_index: int) -> None:
        assert self.video_path is not None
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"无法读取第 {frame_index} 帧")
            left_raw, right_raw = split_side_by_side(frame)
            self.current_left, self.current_right = self.calibration.rectify_pair(left_raw, right_raw)
            self.video_views[0].set_frame(self.current_left)
            self.video_views[1].set_frame(self.current_right)
            self.video_views[2].set_frame(None)
            self.video_views[3].set_frame(self.current_left)
            self.metric_values["frame"].setText(str(frame_index))
        finally:
            capture.release()

    def _refresh_point_table(self) -> None:
        self.point_table.setRowCount(len(self.points))
        for row, point in enumerate(self.points):
            role = "测量" if point.role == "measurement" else "参考"
            for column, value in enumerate(
                (point.point_id, role, f"{point.xy[0]:.1f}", f"{point.xy[1]:.1f}")
            ):
                self.point_table.setItem(row, column, QTableWidgetItem(value))

    def _refresh_preview_points(self) -> None:
        if self.current_left is None:
            return
        markers = [
            (point.xy[0], point.xy[1], point.point_id, point.role)
            for point in self.points
        ]
        self.video_views[0].set_frame(self.current_left, markers)
        self.video_views[3].set_frame(self.current_left, markers)

    def _renumber_points(self) -> None:
        self.points = [
            PointSpec(f"P{index}", point.xy, point.role)
            for index, point in enumerate(self.points, 1)
        ]
        self._refresh_point_table()
        self._refresh_preview_points()
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        ready = self.video_path is not None and bool(self.points) and not self._processing
        self.run_button.setEnabled(ready)
        self.experiment_button.setEnabled(ready)

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_container.setVisible(checked)
        self.advanced_toggle.setText("高级参数  ▾" if checked else "高级参数  ▸")

    def _parameter_control(self, spec: ParameterSpec, value: Any) -> QWidget:
        if spec.kind == "bool":
            control = QCheckBox()
            control.setChecked(bool(value))
        elif spec.kind == "int":
            control = QSpinBox()
            control.setRange(int(spec.minimum or 0), int(spec.maximum or 100))
            control.setSingleStep(int(spec.step or 1))
            control.setValue(int(value))
        elif spec.kind == "float":
            control = QDoubleSpinBox()
            control.setDecimals(6 if (spec.step or 1) < 0.001 else 3)
            control.setRange(float(spec.minimum or 0), float(spec.maximum or 100))
            control.setSingleStep(float(spec.step or 0.1))
            control.setValue(float(value))
        else:
            control = QComboBox()
            for choice in spec.choices:
                control.addItem(choice, choice)
            index = control.findData(value)
            control.setCurrentIndex(max(index, 0))
        control.setToolTip(spec.tooltip)
        control.setAccessibleName(spec.label)
        return control

    def _install_shortcuts(self) -> None:
        open_action = QAction(self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.choose_video)
        self.addAction(open_action)
        undo_action = QAction(self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo_point)
        self.addAction(undo_action)

    def _set_status(self, label: str, detail: str) -> None:
        self.status_pill.setText(label)
        self.status_detail.setText(detail)
        self.statusBar().showMessage(detail)

    def _show_error(self, title: str, message: str) -> None:
        self._set_status("需要处理", message)
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the processing loop before Qt destroys its worker thread."""

        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(5_000):
                self._set_status("正在安全停止", "当前帧尚未处理完，请稍后再关闭。")
                event.ignore()
                return
        event.accept()

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _default_calibration_path() -> Path:
        return Path(__file__).resolve().parents[2] / "configs" / "calibration_640x480.json"
