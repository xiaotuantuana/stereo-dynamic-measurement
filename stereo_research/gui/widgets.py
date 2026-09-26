from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class AspectImageView(QFrame):
    pointClicked = Signal(float, float)

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self._frame: np.ndarray | None = None
        self._points: list[tuple[float, float, str, str]] = []
        self.selectable = False
        self.setObjectName("videoCard")
        self.setMinimumSize(260, 190)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("viewTitle")
        self.title_label.setFixedHeight(32)
        self.title_label.setContentsMargins(12, 0, 8, 0)
        self.image_label = QLabel("等待视频")
        self.image_label.setObjectName("imageViewport")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def set_frame(
        self,
        frame: np.ndarray | None,
        points: list[tuple[float, float, str, str]] | None = None,
    ) -> None:
        self._frame = None if frame is None else np.asarray(frame).copy()
        self._points = list(points or [])
        self._refresh_pixmap()

    def widget_to_image(self, position: QPointF) -> tuple[float, float] | None:
        if self._frame is None:
            return None
        content = QRectF(0.0, 32.0, float(self.width()), max(1.0, float(self.height() - 32)))
        image_h, image_w = self._frame.shape[:2]
        scale = min(content.width() / image_w, content.height() / image_h)
        shown_w, shown_h = image_w * scale, image_h * scale
        left = content.left() + (content.width() - shown_w) * 0.5
        top = content.top() + (content.height() - shown_h) * 0.5
        shown = QRectF(left, top, shown_w, shown_h)
        if not shown.contains(position):
            return None
        return (
            float((position.x() - left) / scale),
            float((position.y() - top) / scale),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.selectable and event.button() == Qt.MouseButton.LeftButton:
            mapped = self.widget_to_image(event.position())
            if mapped is not None:
                self.pointClicked.emit(*mapped)
                event.accept()
                return
        super().mousePressEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._frame is None or self.image_label.width() <= 1 or self.image_label.height() <= 1:
            return
        frame = self._frame.copy()
        for x, y, label, role in self._points:
            color = (55, 200, 255) if role == "measurement" else (120, 220, 120)
            center = (int(round(x)), int(round(y)))
            cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (center[0] + 8, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)
