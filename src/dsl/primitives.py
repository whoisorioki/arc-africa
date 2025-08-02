"""DSL primitives for ARC task transformations."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def rotate90(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)

def horizontal_mirror(grid: np.ndarray) -> np.ndarray:
    """Mirror grid horizontally."""
    return np.fliplr(grid)

def vertical_mirror(grid: np.ndarray) -> np.ndarray:
    """Mirror grid vertically."""
    return np.flipud(grid)

def fill(grid: np.ndarray, color: int = 1) -> np.ndarray:
    """Fill entire grid with a color."""
    result = grid.copy()
    result.fill(color)
    return result

def replace_color(grid: np.ndarray, old_color: int = 1, new_color: int = 2) -> np.ndarray:
    """Replace all instances of old_color with new_color."""
    result = grid.copy()
    result[result == old_color] = new_color
    return result

def move(grid: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Move grid by dx, dy."""
    result = np.zeros_like(grid)
    h, w = grid.shape
    y1, y2 = max(0, dy), min(h, h + dy)
    x1, x2 = max(0, dx), min(w, w + dx)
    result[y1:y2, x1:x2] = grid[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return result

def scale(grid: np.ndarray, factor: int = 2) -> np.ndarray:
    """Scale grid by factor."""
    h, w = grid.shape
    new_h, new_w = h * factor, w * factor
    result = np.zeros((new_h, new_w), dtype=grid.dtype)
    for i in range(h):
        for j in range(w):
            result[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = grid[i, j]
    return result

def crop(grid: np.ndarray, x1: int = 0, y1: int = 0, x2: int = None, y2: int = None) -> np.ndarray:
    """Crop grid to specified region."""
    if x2 is None:
        x2 = grid.shape[1]
    if y2 is None:
        y2 = grid.shape[0]
    return grid[y1:y2, x1:x2]

def pad(grid: np.ndarray, padding: int = 1, value: int = 0) -> np.ndarray:
    """Pad grid with value."""
    return np.pad(grid, padding, mode='constant', constant_values=value)

def colorfilter(grid: np.ndarray, color: int = 1) -> np.ndarray:
    """Keep only specified color, set others to 0."""
    result = grid.copy()
    result[result != color] = 0
    return result

def compose(grid: np.ndarray, func1=None, func2=None) -> np.ndarray:
    """Compose two functions."""
    if func1 is None:
        func1 = lambda x: x
    if func2 is None:
        func2 = lambda x: x
    return func2(func1(grid))

def chain(grid: np.ndarray, *functions) -> np.ndarray:
    """Chain multiple functions."""
    result = grid
    for func in functions:
        if func is not None:
            result = func(result)
    return result

def identity(grid: np.ndarray) -> np.ndarray:
    """Identity function - return grid unchanged."""
    return grid.copy()

def negate(grid: np.ndarray) -> np.ndarray:
    """Negate all non-zero values."""
    result = grid.copy()
    result[result != 0] = 1 - result[result != 0]
    return result

def threshold(grid: np.ndarray, thresh: int = 1) -> np.ndarray:
    """Apply threshold to grid."""
    result = grid.copy()
    result[result >= thresh] = 1
    result[result < thresh] = 0
    return result

def blur(grid: np.ndarray) -> np.ndarray:
    """Simple blur operation."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(grid.astype(float), sigma=0.5).astype(grid.dtype)
    except ImportError:
        return grid

def edge_detect(grid: np.ndarray) -> np.ndarray:
    """Simple edge detection."""
    try:
        from scipy.ndimage import sobel
        edges = np.sqrt(sobel(grid.astype(float), axis=0)**2 + sobel(grid.astype(float), axis=1)**2)
        return (edges > 0.1).astype(grid.dtype)
    except ImportError:
        return grid

def median_filter(grid: np.ndarray) -> np.ndarray:
    """Apply median filter."""
    try:
        from scipy.ndimage import median_filter as scipy_median_filter
        return scipy_median_filter(grid, size=3)
    except ImportError:
        return grid

# List of available primitives (17 total to match model output)
PRIMITIVE_FUNCTIONS = {
    'rotate90': rotate90,
    'horizontal_mirror': horizontal_mirror,
    'vertical_mirror': vertical_mirror,
    'fill': fill,
    'replace_color': replace_color,
    'move': move,
    'scale': scale,
    'crop': crop,
    'pad': pad,
    'colorfilter': colorfilter,
    'compose': compose,
    'chain': chain,
    'identity': identity,
    'negate': negate,
    'threshold': threshold,
    'blur': blur,
    'edge_detect': edge_detect,
}

PRIMITIVE_NAMES = list(PRIMITIVE_FUNCTIONS.keys())
