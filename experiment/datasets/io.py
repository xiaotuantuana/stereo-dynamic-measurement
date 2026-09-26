from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image through bytes so non-ASCII Windows paths remain reliable."""

    image_path = Path(path)
    payload = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(payload, flags)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    return image


def image_size(path: str | Path) -> tuple[int, int]:
    image = read_image(path, cv2.IMREAD_UNCHANGED)
    return int(image.shape[1]), int(image.shape[0])


def read_pfm(path: str | Path) -> np.ndarray:
    """Read grayscale or RGB PFM and return display-oriented float32 pixels."""

    pfm_path = Path(path)
    with pfm_path.open("rb") as handle:
        color_line = handle.readline().decode("ascii").strip()
        if color_line not in {"PF", "Pf"}:
            raise ValueError(f"Unsupported PFM header in {pfm_path}: {color_line!r}")
        dimensions = handle.readline().decode("ascii").strip().split()
        if len(dimensions) != 2:
            raise ValueError(f"Invalid PFM dimensions in {pfm_path}")
        width, height = (int(value) for value in dimensions)
        scale = float(handle.readline().decode("ascii").strip())
        dtype = np.dtype("<f4" if scale < 0 else ">f4")
        channels = 3 if color_line == "PF" else 1
        data = np.fromfile(handle, dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM payload has {data.size} values; expected {expected}: {pfm_path}")
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32, copy=False)

