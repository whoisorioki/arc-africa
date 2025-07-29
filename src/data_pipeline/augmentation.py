import numpy as np
import json
import random
import inspect
import re
from typing import List, Tuple, Dict, Callable
from src.dsl.primitives import rotate90, horizontal_mirror, vertical_mirror
import itertools
import os
from .task_mappings import GENERATOR_MAP, SOLVER_MAP

# List of all DSL primitive names for extraction
PRIMITIVE_NAMES = [
    # Basic geometric transformations
    "rotate90",
    "rotate180",
    "rotate270",
    "horizontal_mirror",
    "vertical_mirror",
    # Color transformations
    "replace_color",
    # Object selection and manipulation
    "select_largest_object",
    "select_smallest_object",
    "count_objects",
    "find_objects",
    # Pattern recognition and completion
    "find_symmetry_axis",
    "complete_symmetry",
    "find_pattern_repetition",
    # Spatial organization
    "align_objects",
    # Conditional transformations
    "conditional_transform",
    # Utility operations
    "crop",
    "remove",
    # Legacy primitives (for backward compatibility)
    "colorfilter",
    "fill",
    "move",
]


def extract_primitives_from_program(program_fn: Callable) -> List[str]:
    """Extracts the names of DSL primitives used in a solver function's source code.

    Args:
        program_fn (Callable): The solver function to analyze.

    Returns:
        List[str]: List of primitive names used in the function.
    """
    try:
        source_code = inspect.getsource(program_fn)
        used_primitives = []
        for name in PRIMITIVE_NAMES:
            if re.search(r"\b" + name + r"\b", source_code):
                used_primitives.append(name)
        return list(set(used_primitives))  # Remove duplicates
    except (OSError, TypeError, AttributeError, ValueError) as e:
        # Handle cases where source code cannot be extracted
        # This can happen with compiled functions, built-ins, or functions without source
        print(
            f"Warning: Could not extract source code for function {program_fn.__name__}: {e}"
        )
        return []


def augment_demonstrations(
    demo_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates augmented demonstration pairs using geometric and color transformations.

    This function takes a list of (input_grid, output_grid) demonstration pairs and applies a set of
    logic-preserving transformations to each pair. The augmentations include 90/180/270 degree rotations,
    horizontal and vertical flips, and all possible color palette permutations. This is used for test-time
    augmentation (TTA) to increase the effective number of examples for few-shot ARC tasks.

    Args:
        demo_pairs (List[Tuple[np.ndarray, np.ndarray]]): List of (input_grid, output_grid) pairs.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: Augmented list of (input_grid, output_grid) pairs, including originals.
    """
    augmented = []
    for inp, out in demo_pairs:
        # All 4 rotations
        for k in range(4):
            inp_rot = rotate90(inp, k)
            out_rot = rotate90(out, k)
            augmented.append((inp_rot, out_rot))
            # Horizontal flip
            inp_h = horizontal_mirror(inp_rot)
            out_h = horizontal_mirror(out_rot)
            augmented.append((inp_h, out_h))
            # Vertical flip
            inp_v = vertical_mirror(inp_rot)
            out_v = vertical_mirror(out_rot)
            augmented.append((inp_v, out_v))
        # Color permutations
        colors = np.unique(inp)
        if 0 in colors:
            colors = colors[colors != 0]  # Exclude background
        for perm in itertools.permutations(colors):
            color_map = {c: p for c, p in zip(colors, perm)}

            def permute(grid):
                grid_new = grid.copy()
                for c, p in color_map.items():
                    grid_new[grid == c] = p
                return grid_new

            inp_perm = permute(inp)
            out_perm = permute(out)
            augmented.append((inp_perm, out_perm))
    # Remove duplicates
    unique_aug = []
    seen = set()
    for inp, out in augmented:
        key = (inp.tobytes(), out.tobytes())
        if key not in seen:
            unique_aug.append((inp, out))
            seen.add(key)
    return unique_aug


def example_generator() -> np.ndarray:
    """Example generator: creates a random 5x5 grid with a single red square (color 2)."""
    grid = np.zeros((5, 5), dtype=int)
    x, y = random.randint(0, 4), random.randint(0, 4)
    grid[x, y] = 2
    return grid


def example_program(grid: np.ndarray) -> np.ndarray:
    """Example program: rotates the grid 90 degrees."""
    return rotate90(grid)


def try_solver(solver_fn, new_input):
    """Try several common calling conventions for ARC solvers."""
    for inp in [new_input, {"input": new_input}, [{"input": new_input}]]:
        try:
            result = solver_fn(inp)
            return result
        except Exception:
            continue
    raise RuntimeError("Solver could not process input in any known format.")


def generate_synthetic_dataset(
    num_samples: int, output_path: str, whitelist_path: str = None
):
    """Generates a large-scale synthetic dataset of (input, output, program) pairs for neural guide training.

    Args:
        num_samples (int): Number of synthetic samples to generate (e.g., 100,000).
        output_path (str): Path to save the generated dataset (JSON).
        whitelist_path (str, optional): Path to a JSON file containing a list of compatible task_ids. If provided, only these tasks will be used.

    Returns:
        None
    """
    # Load whitelist if provided
    if whitelist_path is not None:
        with open(whitelist_path, "r") as f:
            whitelist = set(json.load(f))
        task_ids = [tid for tid in GENERATOR_MAP.keys() if tid in whitelist]
        print(f"Using whitelist: {len(task_ids)} compatible tasks.")
    else:
        task_ids = list(GENERATOR_MAP.keys())
        print(f"Using all available tasks: {len(task_ids)}.")

    # Filter to only include tasks that have both generators and solvers
    complete_task_ids = [
        tid for tid in task_ids if tid in GENERATOR_MAP and tid in SOLVER_MAP
    ]
    print(f"Tasks with both generator and solver: {len(complete_task_ids)}")

    if len(complete_task_ids) < len(task_ids):
        missing_solvers = [
            tid for tid in task_ids if tid in GENERATOR_MAP and tid not in SOLVER_MAP
        ]
        print(f"Tasks missing solvers: {len(missing_solvers)}")
        if len(missing_solvers) <= 10:
            print(f"Missing solver tasks: {missing_solvers}")

    task_ids = complete_task_ids

    if not task_ids:
        raise ValueError(
            "No task generators found. Ensure generators.py and solvers.py are properly loaded."
        )

    print(
        f"Generating {num_samples} synthetic samples from {len(task_ids)} available tasks..."
    )

    dataset = []
    for i in range(num_samples):
        # Randomly select one of the available tasks
        task_id = random.choice(task_ids)

        # Check if both generator and solver exist for this task
        if task_id not in GENERATOR_MAP:
            print(f"Warning: Task {task_id} not found in generators, skipping...")
            continue

        if task_id not in SOLVER_MAP:
            print(f"Warning: Task {task_id} not found in solvers, skipping...")
            continue

        generator_fn = GENERATOR_MAP[task_id]
        solver_fn = SOLVER_MAP[task_id]

        try:
            new_input = generator_fn(0.0, 1.0)["input"]
            new_input = np.array(new_input)
            new_output = try_solver(solver_fn, new_input)
            new_output = np.array(new_output)
            # Extract DSL primitives used in the solver function
            program_primitives = extract_primitives_from_program(solver_fn)
            dataset.append(
                {
                    "input": new_input.tolist(),
                    "output": new_output.tolist(),
                    "program": program_primitives,
                    "task_id": task_id,
                }
            )
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1} samples...")
        except Exception as e:
            print(f"Warning: Skipping sample for task {task_id} due to error: {e}")
            print(
                f"  Generator: {generator_fn.__name__ if hasattr(generator_fn, '__name__') else 'Unknown'}"
            )
            print(
                f"  Solver: {solver_fn.__name__ if hasattr(solver_fn, '__name__') else 'Unknown'}"
            )
            continue
    print(f"Generated {len(dataset)} valid samples. Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print("Dataset generation complete.")
