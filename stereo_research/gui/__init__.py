"""PySide6 desktop front end for the temporal stereo research backend."""

import os
from pathlib import Path

# PySide6-Essentials does not bundle fallback fonts.  Point Qt at the Windows
# font directory before importing any Qt widgets so Chinese labels render on
# clean research workstations as well as on the development machine.
os.environ.setdefault(
    "QT_QPA_FONTDIR",
    str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"),
)

from .window import StereoMainWindow

__all__ = ["StereoMainWindow"]
