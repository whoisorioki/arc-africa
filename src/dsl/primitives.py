import numpy as np
from typing import List, Dict, Optional, Tuple, Callable


def rotate90(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotates the input grid by 90 degrees counterclockwise k times.

    This function performs a geometric transformation on the input grid, rotating it by 90 degrees
    counterclockwise for each increment of k. Used for data augmentation and as a DSL primitive
    for ARC task solutions.

    Args:
        grid (np.ndarray): A 2D NumPy array representing the ARC grid.
        k (int, optional): Number of times to rotate by 90 degrees counterclockwise. Defaults to 1.

    Returns:
        np.ndarray: The rotated grid as a new NumPy array.

    Raises:
        ValueError: If the input grid is not a 2D NumPy array.
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Input grid must be a 2D NumPy array.")
    return np.rot90(grid, k)


def horizontal_mirror(grid: np.ndarray) -> np.ndarray:
    """Mirrors the input grid horizontally (left-right flip).

    This function flips the grid along its vertical axis, producing a left-right mirror image.
    Useful for geometric transformations in ARC tasks.

    Args:
        grid (np.ndarray): A 2D NumPy array representing the ARC grid.

    Returns:
        np.ndarray: The horizontally mirrored grid.

    Raises:
        ValueError: If the input grid is not a 2D NumPy array.
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Input grid must be a 2D NumPy array.")
    return np.fliplr(grid)


def vertical_mirror(grid: np.ndarray) -> np.ndarray:
    """Mirrors the input grid vertically (up-down flip).

    This function flips the grid along its horizontal axis, producing an up-down mirror image.
    Useful for geometric transformations in ARC tasks.

    Args:
        grid (np.ndarray): A 2D NumPy array representing the ARC grid.

    Returns:
        np.ndarray: The vertically mirrored grid.

    Raises:
        ValueError: If the input grid is not a 2D NumPy array.
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Input grid must be a 2D NumPy array.")
    return np.flipud(grid)


def colorfilter(objects: List[Dict], color: int) -> List[Dict]:
    """Filters the list of objects to only those of a specified color.

    This function selects and returns only the objects whose 'color' property matches the specified color.
    Useful for object-centric reasoning and manipulation in ARC tasks.

    Args:
        objects (List[Dict]): List of object dictionaries as produced by segment_grid.
        color (int): The color value to filter for.

    Returns:
        List[Dict]: A list of objects with the specified color.
    """
    return [obj for obj in objects if obj["color"] == color]


def fill(grid: np.ndarray, objects: List[Dict], color: int) -> np.ndarray:
    """Fills the pixels of the given objects in the grid with a specified color.

    This function modifies a copy of the input grid, setting all pixels belonging to the provided objects
    to the specified color. Useful for object manipulation and compositional logic in ARC tasks.

    Args:
        grid (np.ndarray): The original 2D ARC grid.
        objects (List[Dict]): List of object dictionaries as produced by segment_grid.
        color (int): The color value to fill the objects with.

    Returns:
        np.ndarray: A new grid with the specified objects filled with the given color.
    """
    new_grid = grid.copy()
    for obj in objects:
        try:
            (min_row, min_col), (max_row, max_col) = obj["bounding_box"]
            mask = obj["pixel_mask"]

            # Validate mask
            if not isinstance(mask, np.ndarray):
                continue
            if mask.dtype != bool:
                mask = mask.astype(bool)

            # Get the region from the grid
            region = new_grid[min_row : max_row + 1, min_col : max_col + 1]

            # Verify shapes match
            if mask.shape != region.shape:
                continue

            # Apply the mask to the region
            region[mask] = color

        except (KeyError, ValueError, IndexError) as e:
            # Skip objects with invalid data
            continue

    return new_grid


def move(grid: np.ndarray, obj: Dict, delta: Tuple[int, int]) -> np.ndarray:
    """Moves an object within the grid by a specified (row, col) offset.

    This function creates a new grid with the specified object moved by the given delta. The object's
    pixels are set to its color at the new location, and the original location is set to background (0).
    If the move would place any part of the object outside the grid, it is clipped.

    Args:
        grid (np.ndarray): The original 2D ARC grid.
        obj (Dict): The object dictionary to move (from segment_grid).
        delta (Tuple[int, int]): (row_offset, col_offset) specifying the move.

    Returns:
        np.ndarray: A new grid with the object moved.
    """
    new_grid = grid.copy()
    (min_row, min_col), (max_row, max_col) = obj["bounding_box"]
    mask = obj["pixel_mask"].astype(bool)
    color = obj["color"]
    # Remove object from original location
    new_grid[min_row : max_row + 1, min_col : max_col + 1][mask] = 0
    # Compute new location
    nrows, ncols = mask.shape
    new_min_row = min_row + delta[0]
    new_min_col = min_col + delta[1]
    new_max_row = new_min_row + nrows - 1
    new_max_col = new_min_col + ncols - 1
    # Clip to grid boundaries
    grid_rows, grid_cols = grid.shape
    if (
        new_min_row < 0
        or new_min_col < 0
        or new_max_row >= grid_rows
        or new_max_col >= grid_cols
    ):
        # If out of bounds, do not move
        return new_grid
    # Place object at new location
    new_grid[new_min_row : new_max_row + 1, new_min_col : new_max_col + 1][mask] = color
    return new_grid


def compose(f: Callable, g: Callable) -> Callable:
    """Composes two functions f and g into a new function h(x) = f(g(x)).

    This higher-order function allows for the sequential application of two DSL primitives, enabling
    more complex program construction in the symbolic search. Used to build composite transformations
    for ARC tasks.

    Args:
        f (Callable): The outer function to apply.
        g (Callable): The inner function to apply.

    Returns:
        Callable: A new function that applies g, then f, to its input.

    Example:
        >>> h = compose(rotate90, horizontal_mirror)
        >>> out = h(grid)
    """
    return lambda x: f(g(x))


def chain(functions: List[Callable]) -> Callable:
    """Chains a list of functions into a single function applied in sequence.

    This higher-order function enables the application of multiple DSL primitives in a specified order,
    supporting the construction of complex ARC programs.

    Args:
        functions (List[Callable]): List of functions to apply in order.

    Returns:
        Callable: A function that applies all functions in sequence to its input.

    Example:
        >>> f = chain([rotate90, horizontal_mirror, vertical_mirror])
        >>> out = f(grid)
    """

    def chained(x):
        for fn in functions:
            x = fn(x)
        return x

    return chained


def replace_color(grid: np.ndarray, src_color: int, dst_color: int) -> np.ndarray:
    """Replaces all pixels of src_color in the grid with dst_color.

    This primitive is useful for color manipulation tasks in ARC, such as recoloring objects or backgrounds.

    Args:
        grid (np.ndarray): The input 2D ARC grid.
        src_color (int): The color to replace.
        dst_color (int): The color to use as replacement.

    Returns:
        np.ndarray: A new grid with the color replaced.

    Example:
        >>> new_grid = replace_color(grid, 2, 5)
    """
    new_grid = grid.copy()
    new_grid[new_grid == src_color] = dst_color
    return new_grid


def remove(grid: np.ndarray, objects: List[Dict]) -> np.ndarray:
    """Removes the given objects from the grid by setting their pixels to background (0).

    This primitive is useful for object deletion or masking in ARC tasks.

    Args:
        grid (np.ndarray): The input 2D ARC grid.
        objects (List[Dict]): List of object dictionaries to remove.

    Returns:
        np.ndarray: A new grid with the specified objects removed (set to 0).
    """
    new_grid = grid.copy()
    for obj in objects:
        try:
            (min_row, min_col), (max_row, max_col) = obj["bounding_box"]
            mask = obj["pixel_mask"]

            # Validate mask
            if not isinstance(mask, np.ndarray):
                continue
            if mask.dtype != bool:
                mask = mask.astype(bool)

            # Get the region from the grid
            region = new_grid[min_row : max_row + 1, min_col : max_col + 1]

            # Verify shapes match
            if mask.shape != region.shape:
                continue

            # Apply the mask to the region
            region[mask] = 0

        except (KeyError, ValueError, IndexError) as e:
            # Skip objects with invalid data
            continue

    return new_grid


def crop(
    grid: np.ndarray, bounding_box: Tuple[Tuple[int, int], Tuple[int, int]]
) -> np.ndarray:
    """Crops the grid to the specified bounding box.

    This primitive is useful for extracting subgrids or focusing on specific regions/objects in ARC tasks.

    Args:
        grid (np.ndarray): The input 2D ARC grid.
        bounding_box (Tuple[Tuple[int, int], Tuple[int, int]]): ((min_row, min_col), (max_row, max_col))
            specifying the region to crop.

    Returns:
        np.ndarray: The cropped subgrid.

    Raises:
        ValueError: If the bounding box is out of grid bounds.

    Example:
        >>> subgrid = crop(grid, ((2, 2), (5, 5)))
    """
    (min_row, min_col), (max_row, max_col) = bounding_box
    if (
        min_row < 0
        or min_col < 0
        or max_row >= grid.shape[0]
        or max_col >= grid.shape[1]
        or min_row > max_row
        or min_col > max_col
    ):
        raise ValueError("Bounding box is out of grid bounds.")
    return grid[min_row : max_row + 1, min_col : max_col + 1]


# === Enhanced Primitives for Complex ARC Tasks ===


def find_objects(grid: np.ndarray, min_size: int = 1) -> List[Dict]:
    """Find all objects in the grid using connected component analysis.

    This is a grid-based version that returns a list of object dictionaries
    with properties like color, size, bounding_box, and pixel_mask.

    Args:
        grid (np.ndarray): The input 2D ARC grid.
        min_size (int): Minimum object size to consider.

    Returns:
        List[Dict]: List of object dictionaries.
    """
    from src.data_pipeline.segmentation import segment_grid

    objects = segment_grid(grid)
    return [obj for obj in objects if obj["size"] >= min_size]


def select_largest_object(grid: np.ndarray) -> np.ndarray:
    """Select the largest object in the grid and return a grid with only that object.

    Args:
        grid (np.ndarray): The input 2D ARC grid.

    Returns:
        np.ndarray: Grid containing only the largest object.
    """
    objects = find_objects(grid)
    if not objects:
        return grid.copy()

    # Find largest object
    largest_obj = max(objects, key=lambda obj: obj["size"])

    # Create new grid with only the largest object
    result = np.zeros_like(grid)
    (min_row, min_col), (max_row, max_col) = largest_obj["bounding_box"]
    mask = largest_obj["pixel_mask"]
    color = largest_obj["color"]

    result[min_row : max_row + 1, min_col : max_col + 1][mask] = color
    return result


def select_smallest_object(grid: np.ndarray) -> np.ndarray:
    """Select the smallest object in the grid and return a grid with only that object.

    Args:
        grid (np.ndarray): The input 2D ARC grid.

    Returns:
        np.ndarray: Grid containing only the smallest object.
    """
    objects = find_objects(grid)
    if not objects:
        return grid.copy()

    # Find smallest object
    smallest_obj = min(objects, key=lambda obj: obj["size"])

    # Create new grid with only the smallest object
    result = np.zeros_like(grid)
    (min_row, min_col), (max_row, max_col) = smallest_obj["bounding_box"]
    mask = smallest_obj["pixel_mask"]
    color = smallest_obj["color"]

    result[min_row : max_row + 1, min_col : max_col + 1][mask] = color
    return result


def count_objects(grid: np.ndarray) -> np.ndarray:
    """Count the number of objects in the grid and return a grid with the count.

    Args:
        grid (np.ndarray): The input 2D ARC grid.

    Returns:
        np.ndarray: Grid with the object count in the center.
    """
    objects = find_objects(grid)
    count = len(objects)

    # Create a grid with the count in the center
    result = np.zeros_like(grid)
    center_row = grid.shape[0] // 2
    center_col = grid.shape[1] // 2

    if center_row < grid.shape[0] and center_col < grid.shape[1]:
        result[center_row, center_col] = count

    return result


def find_symmetry_axis(grid: np.ndarray) -> np.ndarray:
    """Find the axis of symmetry in the grid and return a grid highlighting it.

    Args:
        grid (np.ndarray): The input 2D ARC grid.

    Returns:
        np.ndarray: Grid with symmetry axis highlighted.
    """
    h, w = grid.shape

    # Check horizontal symmetry
    for row in range(h // 2):
        if not np.array_equal(grid[row, :], grid[h - 1 - row, :]):
            break
    else:
        # Horizontal symmetry found
        result = np.zeros_like(grid)
        result[h // 2, :] = 1  # Highlight the symmetry line
        return result

    # Check vertical symmetry
    for col in range(w // 2):
        if not np.array_equal(grid[:, col], grid[:, w - 1 - col]):
            break
    else:
        # Vertical symmetry found
        result = np.zeros_like(grid)
        result[:, w // 2] = 1  # Highlight the symmetry line
        return result

    # No symmetry found
    return grid.copy()


def complete_symmetry(grid: np.ndarray) -> np.ndarray:
    """Complete the symmetry pattern in the grid.

    Args:
        grid (np.ndarray): The input 2D ARC grid.

    Returns:
        np.ndarray: Grid with completed symmetry.
    """
    h, w = grid.shape
    result = grid.copy()

    # Try to complete horizontal symmetry
    for row in range(h // 2):
        if np.array_equal(grid[row, :], grid[h - 1 - row, :]):
            continue
        # Fill missing parts
        for col in range(w):
            if grid[row, col] != 0 and grid[h - 1 - row, col] == 0:
                result[h - 1 - row, col] = grid[row, col]
            elif grid[h - 1 - row, col] != 0 and grid[row, col] == 0:
                result[row, col] = grid[h - 1 - row, col]

    # Try to complete vertical symmetry
    for col in range(w // 2):
        if np.array_equal(grid[:, col], grid[:, w - 1 - col]):
            continue
        # Fill missing parts
        for row in range(h):
            if grid[row, col] != 0 and grid[row, w - 1 - col] == 0:
                result[row, w - 1 - col] = grid[row, col]
            elif grid[row, w - 1 - col] != 0 and grid[row, col] == 0:
                result[row, col] = grid[row, w - 1 - col]

    return result


def find_pattern_repetition(grid: np.ndarray) -> np.ndarray:
    """Find repeating patterns in the grid and return a grid highlighting them.

    Args:
        grid (np.ndarray): The input 2D ARC grid.

    Returns:
        np.ndarray: Grid with repeating patterns highlighted.
    """
    h, w = grid.shape
    result = np.zeros_like(grid)

    # Look for horizontal repetition
    for pattern_size in range(1, w // 2 + 1):
        if w % pattern_size == 0:
            pattern = grid[:, :pattern_size]
            repetitions = w // pattern_size
            reconstructed = np.tile(pattern, (1, repetitions))
            if np.array_equal(grid, reconstructed):
                result[:, :pattern_size] = 1  # Highlight the pattern
                return result

    # Look for vertical repetition
    for pattern_size in range(1, h // 2 + 1):
        if h % pattern_size == 0:
            pattern = grid[:pattern_size, :]
            repetitions = h // pattern_size
            reconstructed = np.tile(pattern, (repetitions, 1))
            if np.array_equal(grid, reconstructed):
                result[:pattern_size, :] = 1  # Highlight the pattern
                return result

    return grid.copy()


def align_objects(grid: np.ndarray, alignment: str = "center") -> np.ndarray:
    """Align objects in the grid according to the specified alignment.

    Args:
        grid (np.ndarray): The input 2D ARC grid.
        alignment (str): Alignment type ("center", "left", "right", "top", "bottom").

    Returns:
        np.ndarray: Grid with aligned objects.
    """
    objects = find_objects(grid)
    if not objects:
        return grid.copy()

    result = np.zeros_like(grid)
    h, w = grid.shape

    for obj in objects:
        (min_row, min_col), (max_row, max_col) = obj["bounding_box"]
        mask = obj["pixel_mask"]
        color = obj["color"]
        obj_h, obj_w = mask.shape

        # Calculate new position based on alignment
        if alignment == "center":
            new_row = (h - obj_h) // 2
            new_col = (w - obj_w) // 2
        elif alignment == "left":
            new_row = (h - obj_h) // 2
            new_col = 0
        elif alignment == "right":
            new_row = (h - obj_h) // 2
            new_col = w - obj_w
        elif alignment == "top":
            new_row = 0
            new_col = (w - obj_w) // 2
        elif alignment == "bottom":
            new_row = h - obj_h
            new_col = (w - obj_w) // 2
        else:
            new_row, new_col = min_row, min_col

        # Ensure bounds
        new_row = max(0, min(new_row, h - obj_h))
        new_col = max(0, min(new_col, w - obj_w))

        # Place object at new position
        result[new_row : new_row + obj_h, new_col : new_col + obj_w][mask] = color

    return result


def conditional_transform(
    grid: np.ndarray, condition: str = "has_objects"
) -> np.ndarray:
    """Apply a transformation based on a condition.

    Args:
        grid (np.ndarray): The input 2D ARC grid.
        condition (str): Condition to check ("has_objects", "is_symmetric", "has_pattern").

    Returns:
        np.ndarray: Transformed grid based on condition.
    """
    if condition == "has_objects":
        objects = find_objects(grid)
        if objects:
            return select_largest_object(grid)
        else:
            return grid.copy()

    elif condition == "is_symmetric":
        # Check if grid is symmetric
        h, w = grid.shape
        if h > 1 and np.array_equal(grid, horizontal_mirror(grid)):
            return complete_symmetry(grid)
        elif w > 1 and np.array_equal(grid, vertical_mirror(grid)):
            return complete_symmetry(grid)
        else:
            return grid.copy()

    elif condition == "has_pattern":
        # Check if grid has repeating patterns
        pattern_grid = find_pattern_repetition(grid)
        if np.any(pattern_grid != 0):
            return align_objects(grid, "center")
        else:
            return grid.copy()

    else:
        return grid.copy()
