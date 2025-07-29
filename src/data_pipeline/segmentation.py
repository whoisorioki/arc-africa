import numpy as np
from typing import List, Dict, Tuple
from scipy.ndimage import label


def segment_grid(grid: np.ndarray) -> List[Dict]:
    """Segments a 2D ARC grid into distinct objects using connected-component labeling.

    This function implements a connected-component labeling algorithm to identify and segment contiguous groups
    of same-colored pixels into distinct objects, as required by FR1 of the ARC Challenge. Each object is
    represented as a dictionary containing its color, size (pixel count), bounding box, and a local pixel mask.
    This object-centric representation is foundational for downstream reasoning and manipulation tasks.

    Args:
        grid (np.ndarray): A 2D NumPy array representing the ARC grid. Each integer value corresponds to a color.

    Returns:
        List[Dict]: A list of dictionaries, each representing a segmented object with the following keys:
            - 'color': The integer color value of the object.
            - 'size': The number of pixels in the object.
            - 'bounding_box': ((min_row, min_col), (max_row, max_col)) coordinates of the object's bounding box.
            - 'pixel_mask': A 2D NumPy array (local grid) of the object's shape, with 1s for object pixels and 0s elsewhere.

    Raises:
        ValueError: If the input grid is not a 2D NumPy array.
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Input grid must be a 2D NumPy array.")

    objects = []
    for color in np.unique(grid):
        if color == 0:
            continue  # Often 0 is background in ARC
        mask = grid == color
        labeled, num_features = label(mask)
        for obj_idx in range(1, num_features + 1):
            obj_mask = labeled == obj_idx
            coords = np.argwhere(obj_mask)
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)
            bounding_box = ((min_row, min_col), (max_row, max_col))
            local_mask = obj_mask[min_row : max_row + 1, min_col : max_col + 1].astype(
                np.uint8
            )
            size = obj_mask.sum()
            objects.append(
                {
                    "color": int(color),
                    "size": int(size),
                    "bounding_box": bounding_box,
                    "pixel_mask": local_mask,
                }
            )
    return objects
