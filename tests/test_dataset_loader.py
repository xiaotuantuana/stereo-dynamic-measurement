from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from experiment.datasets.io import read_pfm


def test_read_pfm_applies_bottom_to_top_storage_order(tmp_path: Path) -> None:
    path = tmp_path / "disparity.pfm"
    displayed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    stored = np.flipud(displayed)
    path.write_bytes(b"Pf\n2 2\n-1.0\n" + struct.pack("<4f", *stored.ravel()))

    actual = read_pfm(path)

    np.testing.assert_allclose(actual, displayed)

