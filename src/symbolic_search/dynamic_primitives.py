"""
Dynamic DSL Parameterization for ARC Challenge Solver.

This module implements FR10 from the Product Requirements Document, generating task-specific
parameterized versions of DSL primitives based on the colors and objects present in each task.
This transforms a small set of static primitives into a large, task-specific action space.
"""

import numpy as np
from typing import List, Dict, Callable, Tuple, Any
from functools import partial
from src.data_pipeline.segmentation import segment_grid
from src.dsl.primitives import colorfilter, fill, move, replace_color, remove


def extract_task_colors(demo_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]:
    """Extract all unique colors present in a task's demonstration pairs.

    This function analyzes all input and output grids in the demonstration pairs
    to identify the complete color palette used in the task.

    Args:
        demo_pairs (List[Tuple[np.ndarray, np.ndarray]]): List of (input, output) grid pairs.

    Returns:
        List[int]: Sorted list of unique color values (excluding background 0).
    """
    colors = set()
    for inp, out in demo_pairs:
        colors.update(np.unique(inp))
        colors.update(np.unique(out))

    # Remove background color (0) and sort
    colors.discard(0)
    return sorted(list(colors))


def extract_task_objects(demo_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
    """Extract all unique objects present in a task's demonstration pairs.

    This function segments all input and output grids to identify all objects
    that could be manipulated by object-based primitives.

    Args:
        demo_pairs (List[Tuple[np.ndarray, np.ndarray]]): List of (input, output) grid pairs.

    Returns:
        List[Dict]: List of unique object dictionaries from all grids.
    """
    all_objects = []
    seen_objects = set()

    for inp, out in demo_pairs:
        # Get objects from input and output grids
        for grid in [inp, out]:
            objects = segment_grid(grid)
            for obj in objects:
                # Create a hashable representation to avoid duplicates
                obj_key = (obj["color"], obj["size"], tuple(obj["bounding_box"]))
                if obj_key not in seen_objects:
                    seen_objects.add(obj_key)
                    all_objects.append(obj)

    return all_objects


def generate_color_primitives(
    base_primitive: Callable, colors: List[int]
) -> List[Callable]:
    """Generate color-specific variants of a primitive function.

    This function creates parameterized versions of color-based primitives
    (like colorfilter and fill) for each color present in the task.

    Args:
        base_primitive (Callable): The base primitive function to parameterize.
        colors (List[int]): List of colors to generate variants for.

    Returns:
        List[Callable]: List of parameterized primitive functions.
    """
    variants = []

    if base_primitive.__name__ == "colorfilter":
        # Generate colorfilter variants for each color
        for color in colors:

            def make_colorfilter(c):
                def colorfilter_variant(grid):
                    return colorfilter(segment_grid(grid), c)

                colorfilter_variant.__name__ = f"colorfilter_{c}"
                return colorfilter_variant

            variants.append(make_colorfilter(color))

    elif base_primitive.__name__ == "fill":
        # Generate fill variants for each color
        for color in colors:

            def make_fill(c):
                def fill_variant(grid):
                    objects = segment_grid(grid)
                    return fill(grid, objects, c)

                fill_variant.__name__ = f"fill_{c}"
                return fill_variant

            variants.append(make_fill(color))

    elif base_primitive.__name__ == "replace_color":
        # Generate replace_color variants for meaningful color combinations
        for src_color in colors:
            for dst_color in colors:
                if src_color != dst_color:

                    def make_replace(src, dst):
                        def replace_variant(grid):
                            return replace_color(grid, src, dst)

                        replace_variant.__name__ = f"replace_{src}_to_{dst}"
                        return replace_variant

                    variants.append(make_replace(src_color, dst_color))

    return variants


def generate_object_primitives(
    base_primitive: Callable, objects: List[Dict]
) -> List[Callable]:
    """Generate object-specific variants of a primitive function.

    This function creates parameterized versions of object-based primitives
    (like move and remove) for each object detected in the task.

    Args:
        base_primitive (Callable): The base primitive function to parameterize.
        objects (List[Dict]): List of objects to generate variants for.

    Returns:
        List[Callable]: List of parameterized primitive functions.
    """
    variants = []

    if base_primitive.__name__ == "move":
        # Generate move variants for each object with common movement patterns
        movement_patterns = [
            (0, 1),  # Right
            (0, -1),  # Left
            (1, 0),  # Down
            (-1, 0),  # Up
            (1, 1),  # Diagonal down-right
            (1, -1),  # Diagonal down-left
            (-1, 1),  # Diagonal up-right
            (-1, -1),  # Diagonal up-left
        ]

        for obj in objects:
            for delta in movement_patterns:

                def make_move(target_obj, movement):
                    def move_variant(grid):
                        return move(grid, target_obj, movement)

                    move_variant.__name__ = f"move_obj_{target_obj['color']}_{target_obj['size']}_{movement}"
                    return move_variant

                variants.append(make_move(obj, delta))

    elif base_primitive.__name__ == "remove":
        # Generate remove variants for each object
        for obj in objects:

            def make_remove(target_obj):
                def remove_variant(grid):
                    return remove(grid, [target_obj])

                remove_variant.__name__ = (
                    f"remove_obj_{target_obj['color']}_{target_obj['size']}"
                )
                return remove_variant

            variants.append(make_remove(obj))

    return variants


def generate_dynamic_primitives(
    demo_pairs: List[Tuple[np.ndarray, np.ndarray]], base_primitives: List[Callable]
) -> List[Callable]:
    """Generate dynamic, task-specific primitive variants.

    This is the main function implementing FR10. It analyzes the task's demonstration
    pairs to extract colors and objects, then generates parameterized variants of
    the base primitives that are specifically relevant to the current task.

    Args:
        demo_pairs (List[Tuple[np.ndarray, np.ndarray]]): Task demonstration pairs.
        base_primitives (List[Callable]): List of base primitive functions.

    Returns:
        List[Callable]: List of all primitive functions (base + dynamic variants).
    """
    # Extract task-specific information
    colors = extract_task_colors(demo_pairs)
    objects = extract_task_objects(demo_pairs)

    # Color-based primitives that can be parameterized
    color_primitives = [colorfilter, fill, replace_color]

    # Object-based primitives that can be parameterized
    object_primitives = [move, remove]

    all_primitives = base_primitives.copy()

    # Generate color-based variants
    for primitive in color_primitives:
        if primitive in base_primitives:
            variants = generate_color_primitives(primitive, colors)
            all_primitives.extend(variants)

    # Generate object-based variants
    for primitive in object_primitives:
        if primitive in base_primitives:
            variants = generate_object_primitives(primitive, objects)
            all_primitives.extend(variants)

    return all_primitives


def count_dynamic_variants(
    demo_pairs: List[Tuple[np.ndarray, np.ndarray]], base_primitives: List[Callable]
) -> Dict[str, int]:
    """Count the number of dynamic variants generated for a task.

    This function provides metrics on the expanded search space size,
    which is useful for tracking the effectiveness of dynamic parameterization.

    Args:
        demo_pairs (List[Tuple[np.ndarray, np.ndarray]]): Task demonstration pairs.
        base_primitives (List[Callable]): List of base primitive functions.

    Returns:
        Dict[str, int]: Statistics about the generated variants.
    """
    colors = extract_task_colors(demo_pairs)
    objects = extract_task_objects(demo_pairs)

    stats = {
        "base_primitives": len(base_primitives),
        "unique_colors": len(colors),
        "unique_objects": len(objects),
        "color_variants": 0,
        "object_variants": 0,
        "total_variants": 0,
    }

    # Count color variants
    color_primitives = [colorfilter, fill, replace_color]
    for primitive in color_primitives:
        if primitive in base_primitives:
            if primitive.__name__ == "replace_color":
                stats["color_variants"] += len(colors) * (
                    len(colors) - 1
                )  # All color pairs
            else:
                stats["color_variants"] += len(colors)

    # Count object variants
    object_primitives = [move, remove]
    for primitive in object_primitives:
        if primitive in base_primitives:
            if primitive.__name__ == "move":
                stats["object_variants"] += len(objects) * 8  # 8 movement patterns
            else:
                stats["object_variants"] += len(objects)

    stats["total_variants"] = stats["color_variants"] + stats["object_variants"]

    return stats
