#!/usr/bin/env python3
"""
Enhanced Neuro-Symbolic Solver with 39+ primitives.

This solver combines neural guide predictions with symbolic search to solve ARC tasks.
It uses the enhanced model trained on 39+ primitives including cropping, resizing, and other operations.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
import random
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.neural_guide.enhanced_architecture import EnhancedNeuralGuide, create_enhanced_model, ENHANCED_TRAINING_CONFIG
from src.dsl.primitives import (
    colorfilter,
    fill,
    move,
    rotate90,
    horizontal_mirror,
    vertical_mirror,
    replace_color,
)


def create_crop_primitive(target_shape: Tuple[int, int]):
    """Create a cropping primitive that extracts a region of target_shape."""
    def crop_grid(grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        target_h, target_w = target_shape
        
        # Center the crop
        start_h = max(0, (h - target_h) // 2)
        start_w = max(0, (w - target_w) // 2)
        
        # Extract the region
        cropped = grid[start_h:start_h + target_h, start_w:start_w + target_w]
        
        # Pad if necessary
        if cropped.shape != target_shape:
            padded = np.zeros(target_shape, dtype=grid.dtype)
            actual_h, actual_w = cropped.shape
            padded[:actual_h, :actual_w] = cropped
            return padded
        
        return cropped
    
    return crop_grid


def create_resize_primitive(target_shape: Tuple[int, int]):
    """Create a resize primitive."""
    def resize_grid(grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        target_h, target_w = target_shape
        
        # Simple nearest neighbor resize
        result = np.zeros(target_shape, dtype=grid.dtype)
        
        for i in range(target_h):
            for j in range(target_w):
                # Map target coordinates to source coordinates
                src_i = int(i * h / target_h)
                src_j = int(j * w / target_w)
                
                # Ensure bounds
                src_i = min(src_i, h - 1)
                src_j = min(src_j, w - 1)
                
                result[i, j] = grid[src_i, src_j]
        
        return result
    
    return resize_grid


def create_remove_color_primitive(color: int):
    """Create a primitive that removes a specific color."""
    def remove_color(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        result[result == color] = 0  # Replace with background
        return result
    
    return remove_color


def pad_to_shape(grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Pad a grid to the target shape with zeros."""
    h, w = grid.shape
    target_h, target_w = target_shape
    
    padded = np.zeros(target_shape, dtype=grid.dtype)
    padded[:h, :w] = grid
    
    return padded


class EnhancedNeuroSymbolicSolver:
    """Enhanced neuro-symbolic solver with 39+ primitives."""
    
    def __init__(
        self,
        model_path: str = "models/enhanced_neural_guide_best.pth",
        top_k_primitives: int = 10,
        max_search_depth: int = 8,
        beam_width: int = 25,
        use_ttt: bool = True,
        use_enhanced_search: bool = True
    ):
        """Initialize the enhanced neuro-symbolic solver."""
        self.model_path = model_path
        self.top_k_primitives = top_k_primitives
        self.max_search_depth = max_search_depth
        self.beam_width = beam_width
        self.use_ttt = use_ttt
        self.use_enhanced_search = use_enhanced_search
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create primitives
        self.primitives = self._create_enhanced_primitives()
        
        # Load neural guide
        self.neural_guide = self._load_neural_guide()
        
        print(f"✅ Enhanced Neuro-Symbolic Solver initialized")
        print(f"✅ Device: {self.device}")
        print(f"✅ Top-k primitives: {self.top_k_primitives}")
        print(f"✅ Available primitives: {len(self.primitives)}")
        print(f"✅ Enhanced search: {self.use_enhanced_search}")
        print(f"✅ TTT enabled: {self.use_ttt}")
    
    def _create_enhanced_primitives(self):
        """Create the enhanced primitive set with 39+ primitives."""
        enhanced_primitives = {}
        
        # Original 17 primitives
        original_primitives = {
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
        
        enhanced_primitives.update(original_primitives)
        
        # Add cropping primitives for common output sizes
        crop_sizes = [(9, 9), (10, 9), (12, 10), (8, 8), (10, 10), (12, 12), (15, 15), (20, 20)]
        for h, w in crop_sizes:
            enhanced_primitives[f"crop_{h}x{w}"] = create_crop_primitive((h, w))
        
        # Add resize primitives
        resize_sizes = [(9, 9), (10, 9), (12, 10), (8, 8), (10, 10), (12, 12), (15, 15), (20, 20)]
        for h, w in resize_sizes:
            enhanced_primitives[f"resize_{h}x{w}"] = create_resize_primitive((h, w))
        
        # Add remove color primitives
        for color in range(1, 10):  # Colors 1-9
            enhanced_primitives[f"remove_color_{color}"] = create_remove_color_primitive(color)
        
        # Add copy primitive
        enhanced_primitives["copy"] = lambda grid: grid.copy()
        
        # Add identity primitive
        enhanced_primitives["identity"] = lambda grid: grid
        
        print(f"📊 Enhanced primitive set created: {len(enhanced_primitives)} primitives")
        return enhanced_primitives
    
    def _load_neural_guide(self):
        """Load the enhanced neural guide model."""
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model not found at {self.model_path}")
            print(f"🔧 Creating new enhanced model with {len(self.primitives)} primitives")
            
            # Create new model with current primitive count
            config = {
                'max_colors': 21,
                'embed_dim': 256,
                'num_primitives': len(self.primitives),
                'num_layers': 4,
                'max_grid_size': 30
            }
            model = create_enhanced_model(config)
            model.to(self.device)
            
            print(f"✅ Created new enhanced model with {len(self.primitives)} primitives")
            return model
        
        # Load existing model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with saved configuration
        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
            model = create_enhanced_model(config)
        else:
            # Infer configuration from checkpoint
            state_dict = checkpoint["model_state_dict"]
            
            # Infer number of primitives from output layer
            for key, value in state_dict.items():
                if "output_layer" in key or "classifier" in key:
                    num_primitives = value.shape[0]
                    break
            else:
                # Default to current primitive count
                num_primitives = len(self.primitives)
            
            config = {
                'max_colors': 21,
                'embed_dim': 256,
                'num_primitives': num_primitives,
                'num_layers': 4,
                'max_grid_size': 30
            }
            model = create_enhanced_model(config)
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Loaded enhanced neural guide from {self.model_path}")
        print(f"📋 Model config: {config['num_primitives']} primitives")
        
        return model
    
    def predict_primitives(self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        """Predict primitives using the neural guide."""
        if not demo_pairs:
            return list(self.primitives.keys())[:self.top_k_primitives]
        
        # Use the first demonstration pair for prediction
        input_grid, output_grid = demo_pairs[0]
        
        # Pad input to expected size (30x30)
        input_grid_padded = pad_to_shape(input_grid, (30, 30))
        
        # Convert to tensor
        input_tensor = torch.from_numpy(input_grid_padded).unsqueeze(0).long().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            top_indices, top_probs = self.neural_guide.predict_primitives(input_tensor, top_k=self.top_k_primitives)
            
            # Convert indices to primitive names
            predicted_primitives = []
            primitive_names = list(self.primitives.keys())
            for idx in top_indices[0]:
                if idx < len(primitive_names):
                    predicted_primitives.append(primitive_names[idx])
        
        return predicted_primitives
    
    def solve(self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
        """Solve the task using enhanced neuro-symbolic approach."""
        print(f"🎯 Solving task with {len(demo_pairs)} demonstration pairs")
        
        # Get neural guide predictions
        predicted_primitives = self.predict_primitives(demo_pairs)
        print(f"🧠 Neural guide predictions: {predicted_primitives}")
        
        # Test-time training if enabled
        if self.use_ttt:
            print(f"🔄 Running test-time training...")
            self._test_time_training(demo_pairs)
        
        # Perform beam search
        if self.use_enhanced_search:
            solution = self._enhanced_beam_search(demo_pairs, predicted_primitives)
        else:
            solution = self._basic_beam_search(demo_pairs, predicted_primitives)
        
        return solution
    
    def _test_time_training(self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """Perform test-time training on the demonstration pairs."""
        # Prepare training data
        train_inputs = []
        train_targets = []
        
        for input_grid, output_grid in demo_pairs:
            # Pad input to 30x30
            inp_padded = pad_to_shape(input_grid, (30, 30))
            input_tensor = torch.from_numpy(inp_padded).unsqueeze(0).long().to(self.device)
            
            # Create target (one-hot encoding)
            target = torch.zeros(len(self.primitives)).to(self.device)
            
            # For now, use uniform distribution over predicted primitives
            predicted_primitives = self.predict_primitives(demo_pairs)
            for primitive in predicted_primitives:
                if primitive in self.primitives:
                    idx = list(self.primitives.keys()).index(primitive)
                    target[idx] = 1.0 / len(predicted_primitives)
            
            train_inputs.append(input_tensor)
            train_targets.append(target)
        
        # Quick fine-tuning
        optimizer = torch.optim.Adam(self.neural_guide.parameters(), lr=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.neural_guide.train()
        for _ in range(5):  # 5 steps of fine-tuning
            for inp, target in zip(train_inputs, train_targets):
                optimizer.zero_grad()
                logits = self.neural_guide(inp)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
        
        self.neural_guide.eval()
    
    def _enhanced_beam_search(self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]], predicted_primitives: List[str]) -> Optional[Callable]:
        """Enhanced beam search with better candidate scoring."""
        print(f"🔧 Using {len(predicted_primitives)} primitives for search")
        print(f"🔍 Using enhanced beam search: depth={self.max_search_depth}, beam_width={self.beam_width}")
        
        # Initialize beam with identity function
        beam = [(lambda grid: grid, 0.0, [])]  # (function, score, history)
        
        for depth in range(self.max_search_depth):
            print(f"  Depth {depth+1}/{self.max_search_depth}: {len(beam)} candidates")
            
            new_candidates = []
            
            for current_func, current_score, history in beam:
                # Try each predicted primitive
                for primitive_name in predicted_primitives:
                    if primitive_name not in self.primitives:
                        continue
                    
                    primitive_func = self.primitives[primitive_name]
                    
                    # Compose function
                    def compose_functions(f1, f2):
                        return lambda grid: f2(f1(grid))
                    
                    new_func = compose_functions(current_func, primitive_func)
                    
                    # Evaluate on demonstration pairs
                    score = self._evaluate_candidate(new_func, demo_pairs)
                    
                    # Add to candidates
                    new_history = history + [primitive_name]
                    new_candidates.append((new_func, score, new_history))
                    
                    # Early termination if perfect score
                    if score >= 1.0:
                        print(f"  ✅ Perfect solution found at depth {depth+1}: {new_history}")
                        return new_func
            
            # Select top candidates for next iteration
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = new_candidates[:self.beam_width]
            
            if not beam:
                print(f"  ❌ No candidates generated at depth {depth+1}")
                break
        
        # Return best solution
        if beam:
            best_func, best_score, best_history = beam[0]
            print(f"  🏆 Best solution found: score={best_score:.3f}, history={best_history}")
            
            if best_score >= 0.8:  # High confidence threshold
                return best_func
        
        print(f"  ❌ No verified solution found. Best score: {beam[0][1]:.3f}" if beam else "  ❌ No candidates found")
        return None
    
    def _basic_beam_search(self, demo_pairs: List[Tuple[np.ndarray, np.ndarray]], predicted_primitives: List[str]) -> Optional[Callable]:
        """Basic beam search implementation."""
        # Similar to enhanced but with simpler scoring
        return self._enhanced_beam_search(demo_pairs, predicted_primitives)
    
    def _evaluate_candidate(self, candidate_func: Callable, demo_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate a candidate function on demonstration pairs."""
        correct_count = 0
        total_count = len(demo_pairs)
        
        for input_grid, expected_output in demo_pairs:
            try:
                actual_output = candidate_func(input_grid)
                if np.array_equal(actual_output, expected_output):
                    correct_count += 1
            except Exception:
                # Function failed, count as incorrect
                pass
        
        return correct_count / total_count if total_count > 0 else 0.0


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
    return solver.solve(demo_pairs)


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
        default="models/enhanced_neural_guide_best.pth",
        help="Path to enhanced neural guide model",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
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
        default=25,
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
