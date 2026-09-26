from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .window import StereoMainWindow


def configure_application(application: QApplication) -> None:
    """Apply stable product identity and a Chinese-capable system font."""

    application.setApplicationName("Stereo Measurement Studio")
    application.setApplicationDisplayName("双目时序三维测量平台")
    application.setOrganizationName("Temporal Stereo Research")
    application.setFont(QFont("Microsoft YaHei UI", 10))


def main(argv: Sequence[str] | None = None) -> int:
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(list(argv) if argv is not None else sys.argv)
    configure_application(application)

    window = StereoMainWindow()
    window.show()
    if not owns_application:
        return 0
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
