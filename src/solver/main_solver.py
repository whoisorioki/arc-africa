"""
Main Neuro-Symbolic Solver for ARC Challenge.

This module implements the integrated solver that combines the neural guide with the symbolic
search engine to efficiently solve ARC tasks. The neural guide predicts promising DSL primitives,
which are then used to guide the symbolic search process.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path
import copy
from src.data_pipeline.augmentation import augment_demonstrations
import torch.optim as optim
import torch.nn as nn
from functools import partial
from collections import defaultdict

from src.neural_guide.architecture import create_neural_guide
from src.symbolic_search.search import best_first_search, beam_search
from src.symbolic_search.verifier import verify_program
from src.dsl.primitives import (
    colorfilter,
    fill,
    move,
    rotate90,
    horizontal_mirror,
    vertical_mirror,
    compose,
    chain,
    replace_color,
    remove,
    crop,
    # Enhanced primitives
    find_objects,
    select_largest_object,
    select_smallest_object,
    count_objects,
    find_symmetry_axis,
    complete_symmetry,
    find_pattern_repetition,
    align_objects,
    conditional_transform,
)
from src.data_pipeline.segmentation import segment_grid


def pad_to_shape(grid, shape=(48, 48), pad_value=0):
    """Pad a 2D numpy array to the target shape with pad_value.

    Pads the input grid to the specified shape (height, width) using the given pad_value. Padding is applied evenly on all sides; if the difference is odd, extra padding is added to the bottom/right. This ensures compatibility with models expecting fixed-size input.

    Args:
        grid (np.ndarray): 2D input grid to pad.
        shape (tuple, optional): Target (height, width). Defaults to (48, 48).
        pad_value (int, optional): Value to use for padding. Defaults to 0.

    Returns:
        np.ndarray: The padded grid of shape (shape[0], shape[1]).

    Example:
        >>> pad_to_shape(np.ones((3,3)), (5,5), 0)
        array([[0,0,0,0,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,0,0,0,0]])
    """
    h, w = grid.shape
    target_h, target_w = shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(
        grid, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=pad_value
    )


class NeuroSymbolicSolver:
    """Main neuro-symbolic solver that integrates neural guide with symbolic search."""

    def __init__(
        self,
        model_path: str = "models/neural_guide_best.pth",
        top_k_primitives: int = 3,
        max_search_depth: int = 5,
        max_search_iterations: int = 1000,
        beam_width: int = 5,
        device: str = "auto",
    ):
        """
        Args:
            model_path: Path to the trained neural guide model weights.
            top_k_primitives: Number of top primitives to use from neural guide predictions.
            max_search_depth: Maximum depth for symbolic search.
            max_search_iterations: Maximum iterations for symbolic search.
            beam_width: Beam width for beam search (number of candidates to keep at each step).
            device: Device to run the neural guide on ('auto', 'cuda', or 'cpu').
        """
        self.top_k_primitives = top_k_primitives
        self.max_search_depth = max_search_depth
        self.max_search_iterations = max_search_iterations
        self.beam_width = beam_width

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load neural guide
        self.neural_guide = self._load_neural_guide(model_path)

        # Define available primitives (including enhanced ones)
        self.primitives = {
            # Basic geometric transformations
            "rotate90": rotate90,
            "rotate180": partial(rotate90, k=2),
            "rotate270": partial(rotate90, k=3),
            "horizontal_mirror": horizontal_mirror,
            "vertical_mirror": vertical_mirror,
            # Color transformations
            "replace_color_1_2": partial(replace_color, src_color=1, dst_color=2),
            "replace_color_2_1": partial(replace_color, src_color=2, dst_color=1),
            "replace_color_1_3": partial(replace_color, src_color=1, dst_color=3),
            "replace_color_3_1": partial(replace_color, src_color=3, dst_color=1),
            # Object selection and manipulation
            "select_largest_object": select_largest_object,
            "select_smallest_object": select_smallest_object,
            "count_objects": count_objects,
            # Pattern recognition and completion
            "find_symmetry_axis": find_symmetry_axis,
            "complete_symmetry": complete_symmetry,
            "find_pattern_repetition": find_pattern_repetition,
            # Spatial organization
            "align_objects_center": partial(align_objects, alignment="center"),
            "align_objects_left": partial(align_objects, alignment="left"),
            "align_objects_right": partial(align_objects, alignment="right"),
            "align_objects_top": partial(align_objects, alignment="top"),
            "align_objects_bottom": partial(align_objects, alignment="bottom"),
            # Conditional transformations
            "conditional_has_objects": partial(
                conditional_transform, condition="has_objects"
            ),
            "conditional_is_symmetric": partial(
                conditional_transform, condition="is_symmetric"
            ),
            "conditional_has_pattern": partial(
                conditional_transform, condition="has_pattern"
            ),
            # Utility operations
            "crop_center": lambda grid: (
                crop(grid, ((1, 1), (grid.shape[0] - 2, grid.shape[1] - 2)))
                if grid.shape[0] > 2 and grid.shape[1] > 2
                else grid
            ),
            # Composed primitives
            "rotate90_then_horizontal_mirror": compose(horizontal_mirror, rotate90),
            "horizontal_then_vertical_mirror": compose(
                vertical_mirror, horizontal_mirror
            ),
            "chain_rotate90_mirror": chain([rotate90, horizontal_mirror]),
            "symmetry_then_align": compose(
                partial(align_objects, alignment="center"), complete_symmetry
            ),
        }
        self.primitive_names = list(self.primitives.keys())

        print(f"✓ Neuro-Symbolic Solver initialized")
        print(f"✓ Device: {self.device}")
        print(f"✓ Top-k primitives: {top_k_primitives}")
        print(f"✓ Available primitives: {len(self.primitives)}")

    def _load_neural_guide(self, model_path: str) -> torch.nn.Module:
        """Load the trained neural guide model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please train the model first."
            )

        # Create model
        model = create_neural_guide()

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"✓ Loaded neural guide from {model_path}")
        return model

    def predict_primitives(
        self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[str]:
        """
        Use the neural guide to predict which primitives are most likely to solve the task.

        Args:
            demo_pairs: List of (input_grid, output_grid) demonstration pairs.

        Returns:
            List of primitive names, ranked by predicted likelihood.
        """
        if not demo_pairs:
            return self.primitive_names

        # Use the first demonstration pair for prediction
        input_grid, output_grid = demo_pairs[0]

        # Pad grids to expected size
        input_grid_padded = pad_to_shape(input_grid, (48, 48))
        output_grid_padded = pad_to_shape(output_grid, (48, 48))

        # Convert to tensors (add batch dimension)
        input_tensor = (
            torch.from_numpy(input_grid_padded).unsqueeze(0).long().to(self.device)
        )
        output_tensor = (
            torch.from_numpy(output_grid_padded).unsqueeze(0).long().to(self.device)
        )

        # Get predictions
        with torch.no_grad():
            logits = self.neural_guide(input_tensor, output_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Rank primitives by probability
        primitive_scores = list(zip(self.primitive_names, probabilities))
        primitive_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k primitives
        top_primitives = [
            name for name, score in primitive_scores[: self.top_k_primitives]
        ]

        # Debug prints removed for performance

        return top_primitives

    def _extract_colors(self, demo_pairs):
        """Extract all unique colors present in the demonstration grids."""
        colors = set()
        for inp, out in demo_pairs:
            colors.update(np.unique(inp))
            colors.update(np.unique(out))
        colors.discard(0)  # Optionally ignore background
        return list(colors)

    def _extract_objects(self, demo_pairs):
        """Extract all unique objects from all demonstration input grids using the segmentation module."""
        from src.data_pipeline.segmentation import segment_grid

        objects = []
        for inp, _ in demo_pairs:
            objs = segment_grid(inp)
            objects.extend(objs)
        return objects

    def _generate_dynamic_primitives(self, demo_pairs):
        """Dynamically generate parameterized DSL primitives for the current task.

        This function creates variants of color-based and object-based primitives for all relevant
        colors and objects present in the demonstration grids, as required by FR10.
        Enhanced to include sophisticated pattern recognition and spatial reasoning primitives.

        Args:
            demo_pairs (List[Tuple[np.ndarray, np.ndarray]]): Demonstration pairs for the current task.

        Returns:
            dict: Mapping from primitive name to callable.
        """
        colors = self._extract_colors(demo_pairs)
        objects = self._extract_objects(demo_pairs)
        dynamic_primitives = {}

        # Color-based primitives (replace_color variants)
        for c1 in colors:
            for c2 in colors:
                if c1 != c2:
                    pname = f"replace_color_{c1}_{c2}"
                    dynamic_primitives[pname] = partial(
                        replace_color, src_color=c1, dst_color=c2
                    )

        # Object-based primitives (size-based selection)
        if objects:
            # Create size-based object selection primitives
            sizes = [obj["size"] for obj in objects]
            if sizes:
                min_size = min(sizes)
                max_size = max(sizes)
                median_size = sorted(sizes)[len(sizes) // 2]

                dynamic_primitives["select_objects_min_size"] = partial(
                    find_objects, min_size=min_size
                )
                dynamic_primitives["select_objects_max_size"] = partial(
                    find_objects, min_size=max_size
                )
                dynamic_primitives["select_objects_median_size"] = partial(
                    find_objects, min_size=median_size
                )

        # Pattern-based primitives (symmetry and repetition)
        # These are automatically generated based on task characteristics
        dynamic_primitives["complete_horizontal_symmetry"] = (
            lambda grid: complete_symmetry(horizontal_mirror(grid))
        )
        dynamic_primitives["complete_vertical_symmetry"] = (
            lambda grid: complete_symmetry(vertical_mirror(grid))
        )

        # Spatial organization primitives
        for alignment in ["center", "left", "right", "top", "bottom"]:
            pname = f"align_objects_{alignment}"
            dynamic_primitives[pname] = partial(align_objects, alignment=alignment)

        # Conditional primitives based on task characteristics
        dynamic_primitives["conditional_symmetry"] = lambda grid: (
            complete_symmetry(grid)
            if np.array_equal(grid, horizontal_mirror(grid))
            or np.array_equal(grid, vertical_mirror(grid))
            else grid.copy()
        )

        dynamic_primitives["conditional_pattern"] = lambda grid: (
            align_objects(grid, "center")
            if np.any(find_pattern_repetition(grid) != 0)
            else grid.copy()
        )

        # Add static and composed primitives
        dynamic_primitives.update(self.primitives)
        return dynamic_primitives

    def solve(self, demo_pairs, max_search_depth=None, beam_width=None):
        """
        Given demonstration pairs, return a function that takes an input grid and returns a predicted output grid.
        This implementation uses the neural guide to select the most likely primitive and applies it.
        All demo grids are padded to (48, 48) for model compatibility.
        Dynamically generates task-specific primitives as required by FR10.

        Args:
            demo_pairs: List of (input_grid, output_grid) demonstration pairs.
            max_search_depth: Override for maximum search depth (optional).
            beam_width: Override for beam width (optional).
        """
        primitive_funcs = self._generate_dynamic_primitives(demo_pairs)
        # Use original grids for verification, but create padded versions for primitives
        input_grids = [np.array(inp) for inp, _ in demo_pairs]
        output_grids = [np.array(out) for _, out in demo_pairs]
        # Use beam search as default
        search_depth = (
            max_search_depth if max_search_depth is not None else self.max_search_depth
        )
        search_beam_width = beam_width if beam_width is not None else self.beam_width
        print(
            f"[Solver] Using beam_search: max_depth={search_depth}, beam_width={search_beam_width}"
        )

        # For now, use primitives directly without padding
        # TODO: Fix padding issue later
        padded_primitives = list(primitive_funcs.values())
        print(f"[Solver] Generated {len(padded_primitives)} primitives for search")
        from src.symbolic_search.verifier import verify_program

        program, success = beam_search(
            input_grids=input_grids,
            output_grids=output_grids,
            primitives_list=padded_primitives,
            max_depth=search_depth,
            beam_width=search_beam_width,
            verifier=verify_program,
        )
        if success:

            def solution_fn(grid):
                result = grid.copy()
                for primitive in program:
                    result = primitive(result)
                return result

            return solution_fn
        else:
            print(
                "❌ No solution found with beam search. Trying best_first_search as fallback..."
            )
            program, success = best_first_search(
                input_grids=input_grids,
                output_grids=output_grids,
                primitives_list=padded_primitives,
                max_depth=search_depth,
                verifier=verify_program,
            )
            if success:
                print("✓ Fallback best_first_search solution found!")

                def solution_fn(grid):
                    result = grid.copy()
                    for primitive in program:
                        result = primitive(result)
                    return result

                return solution_fn
            else:
                print("❌ No solution found even with fallback.")
                return None

    # --- Primitive implementations (very basic) ---
    def _rotate90(self, grid):
        return np.rot90(grid, k=-1)  # Counterclockwise

    def _horizontal_mirror(self, grid):
        return np.flipud(grid)

    def _vertical_mirror(self, grid):
        return np.fliplr(grid)

    def _colorfilter(self, grid):
        # As a placeholder, just return the grid unchanged
        return grid

    def _fill(self, grid):
        # As a placeholder, fill the grid with the most common color
        vals, counts = np.unique(grid, return_counts=True)
        fill_color = vals[np.argmax(counts)]
        return np.full_like(grid, fill_color)

    def _move(self, grid):
        # As a placeholder, shift the grid right by 1 (wrap around)
        return np.roll(grid, shift=1, axis=1)

    def solve_task(
        self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Callable]:
        """Simplified solve_task using enhanced beam search without TTT for memory efficiency.

        This function focuses on the enhanced symbolic search (FR11) while avoiding
        the memory-intensive TTT process that was causing OOM errors.

        Args:
            demo_pairs (List[Tuple[np.ndarray, np.ndarray]]): Demonstration pairs for the current task.

        Returns:
            Optional[Callable]: Solution function if found, None otherwise.
        """
        print("[Solver] Using enhanced beam search with dynamic primitives")

        # Generate dynamic primitives (FR10)
        dynamic_primitives = self._generate_dynamic_primitives(demo_pairs)

        # Get base model predictions for primitive filtering
        base_predictions = self.predict_primitives(demo_pairs)

        # Filter primitives based on base model predictions
        filtered_primitives = []
        for prim_name, prim_func in dynamic_primitives.items():
            if any(pred in prim_name for pred in base_predictions):
                filtered_primitives.append(prim_func)

        # If filtering is too restrictive, include some base primitives
        if len(filtered_primitives) < 5:
            base_primitives = list(self.primitives.values())[
                :10
            ]  # Top 10 base primitives
            filtered_primitives.extend(base_primitives)

        print(
            f"[Solver] Using {len(filtered_primitives)} primitives (filtered from {len(dynamic_primitives)})"
        )

        # Use enhanced beam search (FR11)
        input_grids = [np.array(inp) for inp, _ in demo_pairs]
        output_grids = [np.array(out) for _, out in demo_pairs]

        try:
            from src.symbolic_search.search import (
                enhanced_beam_search,
                composite_heuristic,
            )
            from src.symbolic_search.verifier import verify_program

            solution_program, success = enhanced_beam_search(
                input_grids=input_grids,
                output_grids=output_grids,
                primitives_list=filtered_primitives,
                max_depth=self.max_search_depth,
                beam_width=self.beam_width,
                verifier=verify_program,
                heuristic=composite_heuristic,
                pruning_threshold=50.0,  # Aggressive pruning
                early_termination_threshold=5.0,  # Early termination for near-perfect solutions
                adaptive_pruning=True,
                max_candidates_per_depth=500,  # Reduced for speed
            )

            if solution_program:

                def solution_fn(grid):
                    result = grid.copy()
                    for fn in solution_program:
                        result = fn(result)
                    return result

                print(
                    f"[Solver] Solution found with {len(solution_program)} primitives"
                )
                return solution_fn
            else:
                print(
                    "[Solver] No solution found with enhanced beam search, trying fallback"
                )
                return self._solve_with_base_model(demo_pairs)

        except Exception as e:
            print(f"[Solver] Enhanced beam search failed: {e}, trying fallback")
            return self._solve_with_base_model(demo_pairs)

    def _solve_with_base_model(self, demo_pairs):
        """Fallback to base model when TTT fails."""
        print("[Fallback] Using base model without TTT")
        return self.solve(demo_pairs)

    def solve_single_grid(
        self, solution_func: Callable, input_grid: np.ndarray
    ) -> np.ndarray:
        """
        Apply a solution function to a single input grid.

        Args:
            solution_func: The solution function returned by solve_task.
            input_grid: Input grid to solve.

        Returns:
            Output grid.
        """
        return solution_func(input_grid)


def load_arc_task(task_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load an ARC task from a JSON file.

    Args:
        task_path: Path to the task JSON file.

    Returns:
        List of (input_grid, output_grid) demonstration pairs.
    """
    with open(task_path, "r") as f:
        task_data = json.load(f)

    demo_pairs = []
    for train in task_data.get("train", []):
        input_grid = np.array(train["input"])
        output_grid = np.array(train["output"])
        demo_pairs.append((input_grid, output_grid))

    return demo_pairs


def solve_arc_task_file(
    task_path: str, solver: NeuroSymbolicSolver
) -> Optional[Callable]:
    """
    Solve an ARC task from a file.

    Args:
        task_path: Path to the task JSON file.
        solver: Neuro-symbolic solver instance.

    Returns:
        Solution function or None.
    """
    demo_pairs = load_arc_task(task_path)
    return solver.solve_task(demo_pairs)


def main():
    """Main function for testing the solver."""
    import argparse

    parser = argparse.ArgumentParser(description="Neuro-Symbolic ARC Solver")
    parser.add_argument(
        "--task_path", type=str, required=True, help="Path to ARC task JSON file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/neural_guide_best.pth",
        help="Path to trained neural guide model",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top primitives to use from neural guide",
    )

    args = parser.parse_args()

    # Create solver
    solver = NeuroSymbolicSolver(
        model_path=args.model_path, top_k_primitives=args.top_k
    )

    # Solve task
    solution = solve_arc_task_file(args.task_path, solver)

    if solution is not None:
        print("\n🎉 Task solved successfully!")

        # Test on demonstration pairs
        demo_pairs = load_arc_task(args.task_path)
        print("\nTesting solution on demonstration pairs:")

        for i, (input_grid, expected_output) in enumerate(demo_pairs):
            actual_output = solution(input_grid)
            is_correct = np.array_equal(actual_output, expected_output)
            print(f"Demo {i+1}: {'✓' if is_correct else '❌'}")

            if not is_correct:
                print(f"  Expected: {expected_output}")
                print(f"  Got: {actual_output}")
    else:
        print("\n❌ Failed to solve task")


if __name__ == "__main__":
    main()
