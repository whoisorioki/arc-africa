"""
Enhanced Synthetic Data Generation for Neural Guide Training.

This module implements an advanced synthetic data generation pipeline that addresses
the limitations of the current system and produces higher quality, more diverse
training data for the neural guide model.

Key improvements:
1. Dynamic task generation with program composition
2. Advanced augmentation strategies
3. Quality filtering and validation
4. Balanced primitive distribution
5. Task complexity stratification
6. Multi-stage generation pipeline
"""

import numpy as np
import json
import random
import inspect
import re
import time
import os
from typing import List, Tuple, Dict, Callable, Optional, Set
from collections import defaultdict, Counter
from pathlib import Path
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from src.dsl.primitives import (
    rotate90,
    horizontal_mirror,
    vertical_mirror,
    replace_color,
    select_largest_object,
    select_smallest_object,
    count_objects,
    find_objects,
    find_symmetry_axis,
    complete_symmetry,
    find_pattern_repetition,
    align_objects,
    conditional_transform,
    crop,
    remove,
    colorfilter,
    fill,
    move,
)
from src.data_pipeline.segmentation import segment_grid
from src.symbolic_search.dynamic_primitives import generate_dynamic_primitives
from .task_mappings import GENERATOR_MAP, SOLVER_MAP
from .semantic_quality_filter import ARCLikeClassifier, is_arc_like
import torch


class EnhancedSyntheticGenerator:
    """Enhanced synthetic data generator with advanced features."""

    def __init__(
        self,
        target_samples: int = 500000,
        min_task_complexity: int = 1,
        max_task_complexity: int = 5,
        quality_threshold: float = 0.4,
        balance_primitives: bool = True,
        use_dynamic_primitives: bool = True,
    ):
        """
        Initialize the enhanced synthetic generator.

        Args:
            target_samples: Total number of samples to generate
            min_task_complexity: Minimum program length for generated tasks
            max_task_complexity: Maximum program length for generated tasks
            quality_threshold: Minimum quality score for samples (0-1)
            balance_primitives: Whether to balance primitive distribution
            use_dynamic_primitives: Whether to use dynamic primitive generation
        """
        self.target_samples = target_samples
        self.min_task_complexity = min_task_complexity
        self.max_task_complexity = max_task_complexity
        self.quality_threshold = quality_threshold
        self.balance_primitives = balance_primitives
        self.use_dynamic_primitives = use_dynamic_primitives

        # Load the semantic quality classifier
        self.quality_classifier = ARCLikeClassifier()
        classifier_path = "models/arc_like_classifier.pth"
        if os.path.exists(classifier_path):
            self.quality_classifier.load_state_dict(torch.load(classifier_path))
            self.quality_classifier.eval()
            print("✅ Semantic quality classifier loaded.")
        else:
            self.quality_classifier = None
            print(
                "⚠️  Semantic quality classifier not found. Falling back to heuristic quality."
            )

        # Helper functions for rotations
        def rotate180(grid):
            return rotate90(grid, 2)

        def rotate270(grid):
            return rotate90(grid, 3)

        # Available primitives for composition (expanded set)
        self.base_primitives = {
            "rotate90": rotate90,
            "horizontal_mirror": horizontal_mirror,
            "vertical_mirror": vertical_mirror,
            "replace_color": replace_color,
            "select_largest_object": select_largest_object,
            "select_smallest_object": select_smallest_object,
            "count_objects": count_objects,
            "find_symmetry_axis": find_symmetry_axis,
            "complete_symmetry": complete_symmetry,
            "align_objects": align_objects,
            "crop": crop,
            "colorfilter": colorfilter,
            "fill": fill,
            "move": move,
            # Add more complex/useful primitives below
            "find_pattern_repetition": find_pattern_repetition,
            "conditional_transform": conditional_transform,
            "remove": remove,
        }

        # Primitive categories for balanced generation
        self.primitive_categories = {
            "geometric": [
                "rotate90",
                "rotate180",
                "rotate270",
                "horizontal_mirror",
                "vertical_mirror",
            ],
            "color": ["replace_color", "colorfilter", "fill"],
            "object": [
                "select_largest_object",
                "select_smallest_object",
                "count_objects",
                "find_objects",
                "move",
                "remove",
            ],
            "pattern": [
                "find_symmetry_axis",
                "complete_symmetry",
                "find_pattern_repetition",
            ],
            "spatial": ["align_objects", "crop"],
            "conditional": ["conditional_transform"],
        }

        # Statistics tracking
        self.generation_stats = {
            "total_generated": 0,
            "quality_filtered": 0,
            "primitive_usage": Counter(),
            "complexity_distribution": Counter(),
            "category_balance": defaultdict(Counter),
        }

    def generate_composite_program(
        self, complexity: int
    ) -> Tuple[List[Callable], List[str]]:
        """
        Generate a composite program of specified complexity.

        Args:
            complexity: Number of primitives to compose

        Returns:
            Tuple of (program_functions, primitive_names)
        """
        if complexity == 1:
            # Single primitive
            prim_name = random.choice(list(self.base_primitives.keys()))
            return [self.base_primitives[prim_name]], [prim_name]

        # Multi-primitive composition
        program = []
        primitive_names = []

        # Ensure diversity in primitive categories
        categories_used = set()

        for i in range(complexity):
            if self.balance_primitives and len(categories_used) < len(
                self.primitive_categories
            ):
                # Prefer unused categories
                available_categories = [
                    cat
                    for cat in self.primitive_categories.keys()
                    if cat not in categories_used
                ]
                if available_categories:
                    category = random.choice(available_categories)
                    categories_used.add(category)
                    prim_name = random.choice(self.primitive_categories[category])
                else:
                    prim_name = random.choice(list(self.base_primitives.keys()))
            else:
                prim_name = random.choice(list(self.base_primitives.keys()))

            program.append(self.base_primitives[prim_name])
            primitive_names.append(prim_name)

        return program, primitive_names

    def execute_program(
        self, program: List[Callable], input_grid: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Execute a program on an input grid with error handling.

        Args:
            program: List of primitive functions
            input_grid: Input grid to transform

        Returns:
            Transformed grid or None if execution fails
        """
        try:
            result = input_grid.copy()
            for primitive in program:
                result = primitive(result)
            return result
        except Exception as e:
            return None

    def generate_input_grid(self, size_range: Tuple[int, int] = (3, 15)) -> np.ndarray:
        """
        Generate a random input grid with controlled properties.

        Args:
            size_range: Range for grid dimensions

        Returns:
            Random input grid
        """
        height = random.randint(*size_range)
        width = random.randint(*size_range)

        # Generate grid with controlled properties
        grid = np.zeros((height, width), dtype=int)

        # Add random objects with controlled density
        num_objects = random.randint(1, min(height * width // 4, 10))

        for _ in range(num_objects):
            # Random object properties
            color = random.randint(1, 9)  # Constrained color palette
            obj_height = random.randint(1, min(5, height))
            obj_width = random.randint(1, min(5, width))

            # Random position
            y = random.randint(0, height - obj_height)
            x = random.randint(0, width - obj_width)

            # Place object
            grid[y : y + obj_height, x : x + obj_width] = color

        return grid

    def generate_better_input_grid(
        self, size_range: Tuple[int, int] = (3, 15)
    ) -> np.ndarray:
        """
        Generate a better input grid with more structured patterns.

        Args:
            size_range: Range for grid dimensions

        Returns:
            Better structured input grid
        """
        height = random.randint(*size_range)
        width = random.randint(*size_range)

        # Start with a simple pattern
        grid = np.zeros((height, width), dtype=int)

        # Strategy 1: Simple geometric shapes
        if random.random() < 0.4:
            # Create rectangles
            num_shapes = random.randint(1, 3)
            for _ in range(num_shapes):
                color = random.randint(1, 9)  # Constrained color palette
                h = random.randint(1, min(4, height))
                w = random.randint(1, min(4, width))
                y = random.randint(0, height - h)
                x = random.randint(0, width - w)
                grid[y : y + h, x : x + w] = color

        # Strategy 2: Border pattern
        elif random.random() < 0.3:
            color = random.randint(1, 9)  # Constrained color palette
            grid[0, :] = color
            grid[-1, :] = color
            grid[:, 0] = color
            grid[:, -1] = color

        # Strategy 3: Diagonal pattern
        elif random.random() < 0.3:
            color = random.randint(1, 9)  # Constrained color palette
            for i in range(min(height, width)):
                grid[i, i] = color

        # Strategy 4: Random dots
        else:
            num_dots = random.randint(2, min(height * width // 3, 8))
            for _ in range(num_dots):
                color = random.randint(1, 9)  # Constrained color palette
                y = random.randint(0, height - 1)
                x = random.randint(0, width - 1)
                grid[y, x] = color

        return grid

    def calculate_sample_quality(
        self, input_grid: np.ndarray, output_grid: np.ndarray, program: List[Callable]
    ) -> float:
        """
        Calculate a quality score using the semantic classifier if available.
        """
        if self.quality_classifier:
            # Use the trained CNN to get a semantic quality score
            task_pair = (input_grid, output_grid)
            return (
                1.0
                if is_arc_like(task_pair, self.quality_classifier, threshold=0.7)
                else 0.0
            )
        else:
            # Fallback to the old heuristic method if the classifier is not available
            return self.heuristic_quality_assessment(input_grid, output_grid, program)

    def heuristic_quality_assessment(
        self, input_grid: np.ndarray, output_grid: np.ndarray, program: List[Callable]
    ) -> float:
        """
        Original heuristic-based quality assessment.
        """
        try:
            if np.array_equal(input_grid, output_grid):
                return 0.0
            input_objects = len(segment_grid(input_grid))
            output_objects = len(segment_grid(output_grid))
            if input_objects == 0 or output_objects == 0:
                return 0.1
            complexity_score = min(len(program) / 5.0, 1.0)
            input_colors = len(np.unique(input_grid))
            output_colors = len(np.unique(output_grid))
            color_score = min(max(input_colors, output_colors) / 5.0, 1.0)
            spatial_score = 0.0
            if input_grid.shape != output_grid.shape:
                spatial_score = 1.0
            elif np.sum(np.abs(input_grid - output_grid)) > 0:
                spatial_score = 0.8
            quality = complexity_score * 0.2 + color_score * 0.2 + spatial_score * 0.6
            if len(program) <= 2 and spatial_score > 0:
                quality = min(quality + 0.2, 1.0)
            return quality
        except Exception:
            return 0.0

    def generate_high_quality_sample(self) -> Optional[Dict]:
        """
        Generate a single high-quality sample with improved strategies.

        Returns:
            Sample dictionary or None if generation fails
        """
        # Try multiple strategies
        strategies = [
            self._try_simple_program,
            self._try_medium_program,
            self._try_complex_program,
        ]

        for strategy in strategies:
            sample = strategy()
            if sample is not None:
                return sample

        return None

    def _try_simple_program(self) -> Optional[Dict]:
        """Try generating a simple program (1-2 primitives)."""
        # Generate simple program
        complexity = random.randint(1, 2)
        program, primitive_names = self.generate_composite_program(complexity)

        # Try different input generation strategies
        input_strategies = [
            lambda: self.generate_better_input_grid((3, 8)),
            lambda: self.generate_input_grid((3, 8)),
        ]

        for input_strategy in input_strategies:
            try:
                input_grid = input_strategy()
                output_grid = self.execute_program(program, input_grid)

                if output_grid is not None:
                    quality = self.calculate_sample_quality(
                        input_grid, output_grid, program
                    )

                    # Lower threshold for simple programs
                    if quality >= 0.3:  # Relaxed threshold
                        return self._create_sample(
                            input_grid,
                            output_grid,
                            program,
                            primitive_names,
                            complexity,
                            quality,
                        )
            except Exception:
                continue

        return None

    def _try_medium_program(self) -> Optional[Dict]:
        """Try generating a medium complexity program (2-3 primitives)."""
        complexity = random.randint(2, 3)
        program, primitive_names = self.generate_composite_program(complexity)

        # Try with better input grids
        input_grid = self.generate_better_input_grid((5, 12))
        output_grid = self.execute_program(program, input_grid)

        if output_grid is not None:
            quality = self.calculate_sample_quality(input_grid, output_grid, program)

            if quality >= 0.4:  # Medium threshold
                return self._create_sample(
                    input_grid,
                    output_grid,
                    program,
                    primitive_names,
                    complexity,
                    quality,
                )

        return None

    def _try_complex_program(self) -> Optional[Dict]:
        """Try generating a complex program (3-5 primitives)."""
        complexity = random.randint(3, 5)
        program, primitive_names = self.generate_composite_program(complexity)

        # Try with larger, more structured grids
        input_grid = self.generate_better_input_grid((8, 15))
        output_grid = self.execute_program(program, input_grid)

        if output_grid is not None:
            quality = self.calculate_sample_quality(input_grid, output_grid, program)

            if quality >= 0.5:  # Higher threshold for complex programs
                return self._create_sample(
                    input_grid,
                    output_grid,
                    program,
                    primitive_names,
                    complexity,
                    quality,
                )

        return None

    def _create_sample(
        self, input_grid, output_grid, program, primitive_names, complexity, quality
    ):
        """Create a sample dictionary and update statistics."""
        # Update statistics
        self.generation_stats["total_generated"] += 1
        self.generation_stats["primitive_usage"].update(primitive_names)
        self.generation_stats["complexity_distribution"][complexity] += 1

        for prim_name in primitive_names:
            for category, prims in self.primitive_categories.items():
                if prim_name in prims:
                    self.generation_stats["category_balance"][category][prim_name] += 1
                    break

        return {
            "input": input_grid.tolist(),
            "output": output_grid.tolist(),
            "program": primitive_names,
            "quality": quality,
            "complexity": complexity,
            "task_id": f'synthetic_{self.generation_stats["total_generated"]:06d}',
        }

    def generate_balanced_dataset(self, target_samples: int = None) -> List[Dict]:
        """
        Generate a balanced dataset with controlled primitive distribution.

        Args:
            target_samples: Number of samples to generate (overrides self.target_samples)

        Returns:
            List of sample dictionaries
        """
        if target_samples is None:
            target_samples = self.target_samples

        dataset = []
        attempts = 0
        max_attempts = target_samples * 50  # Increased from 10 to 50

        print(f"Generating {target_samples} high-quality synthetic samples...")
        start_time = time.time()

        while len(dataset) < target_samples and attempts < max_attempts:
            sample = self.generate_high_quality_sample()
            attempts += 1

            if sample is not None:
                dataset.append(sample)

                if len(dataset) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = len(dataset) / elapsed
                    success_rate = len(dataset) / attempts
                    print(
                        f"Generated {len(dataset)}/{target_samples} samples "
                        f"({rate:.1f} samples/sec, {attempts} attempts, {success_rate:.2%} success rate)"
                    )

        print(
            f"Generation complete: {len(dataset)} samples in {time.time() - start_time:.1f}s"
        )
        print(f"Success rate: {len(dataset)/attempts:.2%}")

        return dataset

    def generate_from_existing_tasks(
        self, whitelist_path: str = None, samples_per_task: int = 100
    ) -> List[Dict]:
        """
        Generate samples from existing reliable tasks with enhanced augmentation.

        Args:
            whitelist_path: Path to task whitelist
            samples_per_task: Number of samples per task

        Returns:
            List of sample dictionaries
        """
        # Load whitelist
        if whitelist_path and os.path.exists(whitelist_path):
            with open(whitelist_path, "r") as f:
                whitelist = set(json.load(f))
            task_ids = [tid for tid in GENERATOR_MAP.keys() if tid in whitelist]
        else:
            task_ids = list(GENERATOR_MAP.keys())

        # Filter to complete tasks
        complete_task_ids = [
            tid for tid in task_ids if tid in GENERATOR_MAP and tid in SOLVER_MAP
        ]

        print(f"Generating from {len(complete_task_ids)} existing tasks...")

        dataset = []
        for task_id in complete_task_ids:
            generator_fn = GENERATOR_MAP[task_id]
            solver_fn = SOLVER_MAP[task_id]

            # Extract primitives from solver
            program_primitives = self.extract_primitives_from_program(solver_fn)

            for i in range(samples_per_task):
                try:
                    # Generate input
                    new_input = generator_fn(0.0, 1.0)["input"]
                    new_input = np.array(new_input)

                    # Generate output
                    new_output = self.try_solver(solver_fn, new_input)
                    new_output = np.array(new_output)

                    # Calculate quality
                    quality = self.calculate_sample_quality(
                        new_input, new_output, [solver_fn]
                    )

                    if quality >= self.quality_threshold:
                        dataset.append(
                            {
                                "input": new_input.tolist(),
                                "output": new_output.tolist(),
                                "program": program_primitives,
                                "quality": quality,
                                "complexity": len(program_primitives),
                                "task_id": task_id,
                            }
                        )

                        # Update statistics
                        self.generation_stats["total_generated"] += 1
                        self.generation_stats["primitive_usage"].update(
                            program_primitives
                        )
                        self.generation_stats["complexity_distribution"][
                            len(program_primitives)
                        ] += 1

                except Exception as e:
                    continue

        print(f"Generated {len(dataset)} samples from existing tasks")
        return dataset

    def extract_primitives_from_program(self, program_fn: Callable) -> List[str]:
        """Extract primitive names from program function source code."""
        try:
            source_code = inspect.getsource(program_fn)
            used_primitives = []
            for name in self.base_primitives.keys():
                if re.search(r"\b" + name + r"\b", source_code):
                    used_primitives.append(name)
            return list(set(used_primitives))
        except:
            return []

    def try_solver(self, solver_fn, new_input):
        """Try to execute a solver function with error handling."""
        try:
            return solver_fn(new_input)
        except:
            return new_input

    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset with metadata."""
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save main dataset
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

        # Save metadata
        metadata_path = output_path.replace(".json", "_metadata.json")
        metadata = {
            "total_samples": len(dataset),
            "generation_stats": dict(self.generation_stats),
            "primitive_distribution": dict(self.generation_stats["primitive_usage"]),
            "complexity_distribution": dict(
                self.generation_stats["complexity_distribution"]
            ),
            "category_balance": {
                k: dict(v) for k, v in self.generation_stats["category_balance"].items()
            },
            "generation_parameters": {
                "target_samples": self.target_samples,
                "min_task_complexity": self.min_task_complexity,
                "max_task_complexity": self.max_task_complexity,
                "quality_threshold": self.quality_threshold,
                "balance_primitives": self.balance_primitives,
                "use_dynamic_primitives": self.use_dynamic_primitives,
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {output_path}")
        print(f"Metadata saved to {metadata_path}")


def generate_enhanced_synthetic_dataset(
    target_samples: int = 500000,
    output_path: str = "data/synthetic/enhanced_synthetic_dataset.json",
    whitelist_path: str = "data/synthetic/reliable_task_whitelist.json",
    min_complexity: int = 1,
    max_complexity: int = 5,
    quality_threshold: float = 0.4,
    balance_primitives: bool = True,
    use_existing_tasks: bool = True,
    use_composite_tasks: bool = True,
):
    """
    Generate an enhanced synthetic dataset with advanced features.

    Args:
        target_samples: Total number of samples to generate
        output_path: Path to save the dataset
        whitelist_path: Path to task whitelist
        min_complexity: Minimum program complexity
        max_complexity: Maximum program complexity
        quality_threshold: Minimum quality score
        balance_primitives: Whether to balance primitive distribution
        use_existing_tasks: Whether to use existing reliable tasks
        use_composite_tasks: Whether to generate composite programs
    """
    generator = EnhancedSyntheticGenerator(
        target_samples=target_samples,
        min_task_complexity=min_complexity,
        max_task_complexity=max_complexity,
        quality_threshold=quality_threshold,
        balance_primitives=balance_primitives,
    )

    dataset = []

    # Generate from existing tasks
    if use_existing_tasks:
        print("Phase 1: Generating from existing reliable tasks...")
        existing_samples = generator.generate_from_existing_tasks(
            whitelist_path=whitelist_path, samples_per_task=100
        )
        dataset.extend(existing_samples)
        print(f"Generated {len(existing_samples)} samples from existing tasks")

    # Generate composite tasks
    if use_composite_tasks:
        remaining_samples = target_samples - len(dataset)
        if remaining_samples > 0:
            print(f"Phase 2: Generating {remaining_samples} composite task samples...")
            composite_samples = generator.generate_balanced_dataset(remaining_samples)
            dataset.extend(composite_samples)
            print(f"Generated {len(composite_samples)} composite task samples")

    # Save dataset
    generator.save_dataset(dataset, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("ENHANCED SYNTHETIC DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print(f"Unique primitives: {len(generator.generation_stats['primitive_usage'])}")
    print(f"Complexity range: {min_complexity}-{max_complexity}")
    print(f"Quality threshold: {quality_threshold}")
    print(f"Primitive balance: {balance_primitives}")

    # Print top primitives
    print("\nTop 10 most used primitives:")
    for prim, count in generator.generation_stats["primitive_usage"].most_common(10):
        print(f"  {prim}: {count}")

    # Print complexity distribution
    print("\nComplexity distribution:")
    for comp, count in sorted(
        generator.generation_stats["complexity_distribution"].items()
    ):
        print(f"  Complexity {comp}: {count} samples")

    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    generate_enhanced_synthetic_dataset(
        target_samples=500000,
        min_complexity=1,
        max_complexity=5,
        quality_threshold=0.8,
        balance_primitives=True,
        use_existing_tasks=True,
        use_composite_tasks=True,
    )
