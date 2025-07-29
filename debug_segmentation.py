#!/usr/bin/env python3
"""
Debug script to understand segmentation and pixel_mask issues.
"""

import sys
import os
import numpy as np
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_pipeline.segmentation import segment_grid


def debug_segmentation():
    """Debug the segmentation function with a simple test case."""

    # Create a simple test grid
    test_grid = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 2]])

    print("Test grid:")
    print(test_grid)
    print()

    # Segment the grid
    objects = segment_grid(test_grid)

    print(f"Found {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"Object {i}:")
        print(f"  Color: {obj['color']}")
        print(f"  Size: {obj['size']}")
        print(f"  Bounding box: {obj['bounding_box']}")
        print(f"  Pixel mask shape: {obj['pixel_mask'].shape}")
        print(f"  Pixel mask:")
        print(obj["pixel_mask"])
        print()

        # Test the mask application
        (min_row, min_col), (max_row, max_col) = obj["bounding_box"]
        region = test_grid[min_row : max_row + 1, min_col : max_col + 1]
        print(f"  Region shape: {region.shape}")
        print(f"  Region:")
        print(region)
        print(
            f"  Mask and region shapes match: {obj['pixel_mask'].shape == region.shape}"
        )
        print()


if __name__ == "__main__":
    debug_segmentation()
