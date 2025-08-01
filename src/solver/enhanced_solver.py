"""
Enhanced Neuro-Symbolic Solver for Complex ARC Tasks

This module implements an enhanced solver that combines the improved neural guide
with the enhanced beam search for better performance on complex ARC tasks.
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
from src.symbolic_search.enhanced_search import enhanced_beam_search
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
    crop,
    scale,
    pad,
    identity,
    negate,
    threshold,
    blur,
    edge_detect,
    median_filter,
)
from src.data_pipeline.segmentation import segment_grid


def pad_to_shape(grid, shape=(48, 48), pad_value=0):
    """Pad a 2D numpy array to the target shape with pad_value."""
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


class EnhancedNeuroSymbolicSolver:
    """Enhanced neuro-symbolic solver with improved components."""

    def __init__(
        self,
        model_path: str = "models/neural_guide_enhanced.pth",
        top_k_primitives: int = 5,
        max_search_depth: int = 8,
        beam_width: int = 20,
        device: str = "auto",
        use_enhanced_search: bool = True,
        use_ttt: bool = True,
        ttt_steps: int = 10,
    ):
        """
        Args:
            model_path: Path to the enhanced neural guide model weights.
            top_k_primitives: Number of top primitives to use from neural guide predictions.
            max_search_depth: Maximum depth for symbolic search.
            beam_width: Beam width for beam search.
            device: Device to run the neural guide on.
            use_enhanced_search: Whether to use enhanced beam search.
            use_ttt: Whether to use test-time training.
            ttt_steps: Number of TTT steps.
        """
        self.top_k_primitives = top_k_primitives
        self.max_search_depth = max_search_depth
        self.beam_width = beam_width
        self.use_enhanced_search = use_enhanced_search
        self.use_ttt = use_ttt
        self.ttt_steps = ttt_steps

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Define available primitives first - use the original 17 primitives that the persistent model was trained on
        self.primitives = {
            # Basic geometric transformations (5)
            "rotate90": rotate90,
            "horizontal_mirror": horizontal_mirror,
            "vertical_mirror": vertical_mirror,
            "rotate180": lambda grid: rotate90(rotate90(grid)),
            "rotate270": lambda grid: rotate90(rotate90(rotate90(grid))),
            
            # Color transformations (6)
            "replace_color_1_2": lambda grid: replace_color(grid, old_color=1, new_color=2),
            "replace_color_2_1": lambda grid: replace_color(grid, old_color=2, new_color=1),
            "replace_color_1_3": lambda grid: replace_color(grid, old_color=1, new_color=3),
            "replace_color_3_1": lambda grid: replace_color(grid, old_color=3, new_color=1),
            "replace_color_2_3": lambda grid: replace_color(grid, old_color=2, new_color=3),
            "replace_color_3_2": lambda grid: replace_color(grid, old_color=3, new_color=2),
            
            # Basic operations (6)
            "fill_1": lambda grid: fill(grid, 1),
            "fill_2": lambda grid: fill(grid, 2),
            "fill_3": lambda grid: fill(grid, 3),
            "colorfilter_1": lambda grid: colorfilter(grid, 1),
            "colorfilter_2": lambda grid: colorfilter(grid, 2),
            "colorfilter_3": lambda grid: colorfilter(grid, 3),
        }
        self.primitive_names = list(self.primitives.keys())

        # Load neural guide after primitives are defined
        self.neural_guide = self._load_neural_guide(model_path)

        print(f"✓ Enhanced Neuro-Symbolic Solver initialized")
        print(f"✓ Device: {self.device}")
        print(f"✓ Top-k primitives: {top_k_primitives}")
        print(f"✓ Available primitives: {len(self.primitives)}")
        print(f"✓ Enhanced search: {use_enhanced_search}")
        print(f"✓ TTT enabled: {use_ttt}")

    def _load_neural_guide(self, model_path: str) -> torch.nn.Module:
        """Load the enhanced neural guide model."""
        if model_path is None:
            print("⚠️  No model path provided, creating untrained model")
            # Create untrained model
            model = create_neural_guide(
                num_primitives=len(self.primitive_names),
                grid_size=48,
                max_colors=64,
                embed_dim=128,
                num_layers=4,
                num_heads=8,
                dropout=0.1,
            )
            model.to(self.device)
            model.eval()
            print("✓ Created untrained neural guide model")
            return model
            
        if not os.path.exists(model_path):
            print(f"⚠️  Enhanced model not found at {model_path}, using basic model")
            model_path = "models/neural_guide_best.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model found at {model_path}")

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model with saved configuration
        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
            model = create_neural_guide(
                num_primitives=config["num_primitives"],
                grid_size=config["grid_size"],
                max_colors=config.get("max_colors", 64),
                embed_dim=config.get("embed_dim", 128),
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                dropout=config["dropout"],
            )
        else:
            # Infer configuration from the saved weights
            # Check the output projection layer to determine number of primitives
            if "output_proj.weight" in checkpoint["model_state_dict"]:
                num_primitives = checkpoint["model_state_dict"]["output_proj.weight"].shape[0]
            else:
                num_primitives = 17  # Default for persistent model
            
            # Check input embedding to determine max_colors
            if "input_embedding.weight" in checkpoint["model_state_dict"]:
                max_colors = checkpoint["model_state_dict"]["input_embedding.weight"].shape[0]
            else:
                max_colors = 21  # Default for persistent model
            
            # The architecture adds 1 to max_colors, so we need to subtract 1
            # to match the saved weights
            max_colors = max_colors - 1
            
            # Count transformer layers
            num_layers = 0
            for key in checkpoint["model_state_dict"].keys():
                if "transformer.layers." in key:
                    layer_num = int(key.split(".")[2])
                    num_layers = max(num_layers, layer_num + 1)
            
            if num_layers == 0:
                num_layers = 2  # Default for persistent model
            
            print(f"📋 Inferred model config: {num_primitives} primitives, {max_colors} colors, {num_layers} layers")
            
            model = create_neural_guide(
                num_primitives=num_primitives,
                grid_size=48,
                max_colors=max_colors,
                embed_dim=128,
                num_layers=num_layers,
                num_heads=8,
                dropout=0.1,
            )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"✓ Loaded enhanced neural guide from {model_path}")
        return model

    def predict_primitives(
        self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[str]:
        """Use the enhanced neural guide to predict primitives."""
        if not demo_pairs:
            return self.primitive_names[: self.top_k_primitives]

        # Use the first demonstration pair for prediction
        input_grid, output_grid = demo_pairs[0]

        # Pad grids to expected size
        input_grid_padded = pad_to_shape(input_grid, (48, 48))
        output_grid_padded = pad_to_shape(output_grid, (48, 48))

        # Convert to tensors
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

        return top_primitives

    def _extract_colors(self, demo_pairs):
        """Extract all unique colors present in the demonstration grids."""
        colors = set()
        for inp, out in demo_pairs:
            colors.update(np.unique(inp))
            colors.update(np.unique(out))
        colors.discard(0)  # Ignore background
        return list(colors)

    def _extract_objects(self, demo_pairs):
        """Extract all unique objects from demonstration grids."""
        objects = []
        for inp, _ in demo_pairs:
            objs = segment_grid(inp)
            objects.extend(objs)
        return objects

    def _generate_dynamic_primitives(self, demo_pairs):
        """Generate dynamic primitives for the current task."""
        colors = self._extract_colors(demo_pairs)
        objects = self._extract_objects(demo_pairs)
        dynamic_primitives = {}

        # Color-based primitives
        for c1 in colors:
            for c2 in colors:
                if c1 != c2:
                    pname = f"replace_color_{c1}_{c2}"
                    dynamic_primitives[pname] = partial(
                        replace_color, old_color=c1, new_color=c2
                    )

        # Object-based primitives
        if objects:
            sizes = [obj["size"] for obj in objects]
            if sizes:
                min_size = min(sizes)
                max_size = max(sizes)

                # Size-based selection primitives (removed - functions don't exist)
                pass

        # Add base primitives
        for name, func in self.primitives.items():
            dynamic_primitives[name] = func

        return dynamic_primitives

    def _test_time_training(self, demo_pairs):
        """Perform test-time training on the current task."""
        if not self.use_ttt or len(demo_pairs) < 2:
            return

        print(f"🔄 Performing test-time training ({self.ttt_steps} steps)")

        # Prepare training data
        train_inputs = []
        train_outputs = []
        train_targets = []

        for inp, out in demo_pairs:
            inp_padded = pad_to_shape(inp, (48, 48))
            out_padded = pad_to_shape(out, (48, 48))

            train_inputs.append(
                torch.from_numpy(inp_padded).unsqueeze(0).long().to(self.device)
            )
            train_outputs.append(
                torch.from_numpy(out_padded).unsqueeze(0).long().to(self.device)
            )

            # Create target labels (multi-hot encoding)
            target = torch.zeros(len(self.primitive_names)).to(self.device)
            # For TTT, we'll use a simple heuristic based on grid differences
            if not np.array_equal(inp, out):
                # If grids are different, predict some common primitives
                common_prims = ["rotate90", "horizontal_mirror", "vertical_mirror"]
                for prim in common_prims:
                    if prim in self.primitive_names:
                        target[self.primitive_names.index(prim)] = 1.0
            train_targets.append(target)

        # Set up optimizer for TTT
        optimizer = optim.Adam(self.neural_guide.parameters(), lr=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        # TTT loop
        self.neural_guide.train()
        for step in range(self.ttt_steps):
            total_loss = 0.0

            for inp, out, target in zip(train_inputs, train_outputs, train_targets):
                optimizer.zero_grad()

                logits = self.neural_guide(inp, out)
                loss = criterion(logits.squeeze(0), target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if step % 5 == 0:
                print(
                    f"  TTT step {step+1}/{self.ttt_steps}, loss: {total_loss/len(train_inputs):.4f}"
                )

        self.neural_guide.eval()
        print(f"✅ Test-time training completed")

    def solve(self, demo_pairs, max_search_depth=None, beam_width=None):
        """Solve the task using enhanced components."""
        print(f"🎯 Solving task with {len(demo_pairs)} demonstration pairs")

        # Test-time training
        if self.use_ttt:
            self._test_time_training(demo_pairs)

        # Generate dynamic primitives
        dynamic_primitives = self._generate_dynamic_primitives(demo_pairs)

        # Get neural guide predictions
        predicted_primitives = self.predict_primitives(demo_pairs)
        print(f"🧠 Neural guide predictions: {predicted_primitives}")

        # Filter primitives based on predictions
        filtered_primitives = []
        for prim_name, prim_func in dynamic_primitives.items():
            if any(pred in prim_name for pred in predicted_primitives):
                filtered_primitives.append(prim_func)

        # If filtering is too restrictive, include more primitives
        if len(filtered_primitives) < 10:
            base_primitives = list(dynamic_primitives.values())[:20]
            filtered_primitives.extend(base_primitives)

        print(f"🔧 Using {len(filtered_primitives)} primitives for search")

        # Prepare grids
        input_grids = [np.array(inp) for inp, _ in demo_pairs]
        output_grids = [np.array(out) for _, out in demo_pairs]

        # Use enhanced beam search
        if self.use_enhanced_search:
            search_depth = (
                max_search_depth
                if max_search_depth is not None
                else self.max_search_depth
            )
            search_beam_width = (
                beam_width if beam_width is not None else self.beam_width
            )

            print(
                f"🔍 Using enhanced beam search: depth={search_depth}, beam_width={search_beam_width}"
            )

            program, success = enhanced_beam_search(
                input_grids=input_grids,
                output_grids=output_grids,
                primitives_list=filtered_primitives,
                max_depth=search_depth,
                beam_width=search_beam_width,
                verifier=verify_program,
                max_candidates_per_depth=2000,
                early_termination_threshold=0.1,
                adaptive_pruning=True,
                use_advanced_primitives=True,
            )
        else:
            # Fallback to basic beam search
            from src.symbolic_search.search import beam_search

            program, success = beam_search(
                input_grids=input_grids,
                output_grids=output_grids,
                primitives_list=filtered_primitives,
                max_depth=self.max_search_depth,
                beam_width=self.beam_width,
                verifier=verify_program,
            )

        if success and program:
            print(f"✅ Solution found with {len(program)} primitives")

            def solution_fn(grid):
                result = grid.copy()
                for primitive in program:
                    result = primitive(result)
                return result

            return solution_fn
        else:
            print(f"❌ No solution found")
            return None

    def solve_task(
        self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Callable]:
        """Main solving interface."""
        return self.solve(demo_pairs)


def load_arc_task(task_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load an ARC task from a JSON file."""
    with open(task_path, "r") as f:
        task_data = json.load(f)

    demo_pairs = []
    for train in task_data.get("train", []):
        input_grid = np.array(train["input"])
        output_grid = np.array(train["output"])
        demo_pairs.append((input_grid, output_grid))

    return demo_pairs


def solve_arc_task_file(
    task_path: str, solver: EnhancedNeuroSymbolicSolver
) -> Optional[Callable]:
    """Solve an ARC task from a file."""
    demo_pairs = load_arc_task(task_path)
    return solver.solve_task(demo_pairs)


def main():
    """Main function for testing the enhanced solver."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Neuro-Symbolic ARC Solver")
    parser.add_argument(
        "--task_path", type=str, required=True, help="Path to ARC task JSON file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/neural_guide_enhanced.pth",
        help="Path to enhanced neural guide model",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top primitives to use from neural guide",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
        help="Maximum search depth",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=20,
        help="Beam width for search",
    )

    args = parser.parse_args()

    # Create enhanced solver
    solver = EnhancedNeuroSymbolicSolver(
        model_path=args.model_path,
        top_k_primitives=args.top_k,
        max_search_depth=args.max_depth,
        beam_width=args.beam_width,
        use_enhanced_search=True,
        use_ttt=True,
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
