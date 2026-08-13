"""Small-data facial HOG representation trained entirely from supplied images."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize((48, 48)), dtype=np.float32) / 255.0


def hog_features(array: np.ndarray, cell_size: int = 8, bins: int = 9) -> np.ndarray:
    """Compute 2x2-block L2-Hys HOG without external/pretrained feature models."""

    if array.shape != (48, 48):
        raise ValueError("HOG input must be 48x48 grayscale")
    gradient_y, gradient_x = np.gradient(array.astype(np.float32))
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0
    cells_per_axis = 48 // cell_size
    histograms = np.zeros((cells_per_axis, cells_per_axis, bins), dtype=np.float32)
    bin_width = 180.0 / bins
    for row in range(cells_per_axis):
        for column in range(cells_per_axis):
            row_slice = slice(row * cell_size, (row + 1) * cell_size)
            column_slice = slice(column * cell_size, (column + 1) * cell_size)
            cell_magnitude = magnitude[row_slice, column_slice].ravel()
            positions = orientation[row_slice, column_slice].ravel() / bin_width
            lower = np.floor(positions).astype(int) % bins
            upper = (lower + 1) % bins
            fraction = positions - np.floor(positions)
            np.add.at(histograms[row, column], lower, cell_magnitude * (1.0 - fraction))
            np.add.at(histograms[row, column], upper, cell_magnitude * fraction)
    blocks: list[np.ndarray] = []
    for row in range(cells_per_axis - 1):
        for column in range(cells_per_axis - 1):
            block = histograms[row : row + 2, column : column + 2].ravel()
            block /= np.sqrt(float(np.dot(block, block)) + 1e-6)
            block = np.clip(block, 0, 0.2)
            block /= np.sqrt(float(np.dot(block, block)) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def extract_hog_from_image(image: Image.Image) -> np.ndarray:
    return hog_features(image_array(image))


def extract_hog_from_paths(paths: Iterable[Path]) -> np.ndarray:
    features: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            features.append(extract_hog_from_image(image))
    return np.stack(features)

