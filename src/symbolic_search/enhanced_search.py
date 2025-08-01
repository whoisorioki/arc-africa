"""
Enhanced Beam Search for Complex ARC Tasks

This module implements an advanced beam search algorithm specifically designed
to handle complex ARC tasks that require sophisticated transformations.
"""

import numpy as np
from typing import List, Tuple, Callable, Any, Dict, Optional
import heapq
import itertools
from collections import defaultdict
import time
from functools import lru_cache

from src.dsl.primitives import (
    rotate90,
    horizontal_mirror,
    vertical_mirror,
    replace_color,
    compose,
    chain,
    crop,
    fill,
    colorfilter,
    move,
    scale,
    pad,
    identity,
    negate,
    threshold,
    blur,
    edge_detect,
    median_filter,
)


class EnhancedBeamSearch:
    """Enhanced beam search with sophisticated primitives and heuristics."""

    def __init__(
        self,
        max_depth: int = 8,
        beam_width: int = 20,
        max_candidates_per_depth: int = 2000,
        early_termination_threshold: float = 0.1,
        adaptive_pruning: bool = True,
        use_advanced_primitives: bool = True,
    ):
        """
        Args:
            max_depth: Maximum program depth
            beam_width: Number of candidates to keep at each depth
            max_candidates_per_depth: Maximum candidates to generate per depth
            early_termination_threshold: Threshold for early termination
            adaptive_pruning: Whether to use adaptive pruning
            use_advanced_primitives: Whether to use advanced primitives
        """
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_candidates_per_depth = max_candidates_per_depth
        self.early_termination_threshold = early_termination_threshold
        self.adaptive_pruning = adaptive_pruning
        self.use_advanced_primitives = use_advanced_primitives

        # Initialize primitive library
        self._init_primitive_library()

    def _init_primitive_library(self):
        """Initialize the enhanced primitive library."""
        self.basic_primitives = [
            rotate90,
            horizontal_mirror,
            vertical_mirror,
            lambda grid: rotate90(rotate90(grid)),  # rotate180
            lambda grid: rotate90(rotate90(rotate90(grid))),  # rotate270
        ]

        # Color-based primitives
        self.color_primitives = []
        for src_color in range(1, 10):
            for dst_color in range(1, 10):
                if src_color != dst_color:
                    self.color_primitives.append(
                        lambda grid, src=src_color, dst=dst_color: replace_color(
                            grid, src, dst
                        )
                    )

        # Advanced primitives for complex patterns
        self.advanced_primitives = [
            # Basic transformations
            lambda grid: rotate90(rotate90(grid)),  # rotate180
            lambda grid: rotate90(rotate90(rotate90(grid))),  # rotate270
            # Utility operations
            lambda grid: crop(grid, 1, 1, grid.shape[1]-1, grid.shape[0]-1) if grid.shape[0] > 2 and grid.shape[1] > 2 else grid,
            lambda grid: pad(grid, 1),
            lambda grid: scale(grid, 2),
            # Color operations
            lambda grid: colorfilter(grid, 1),
            lambda grid: colorfilter(grid, 2),
            lambda grid: colorfilter(grid, 3),
            # Threshold operations
            lambda grid: threshold(grid, 1),
            lambda grid: threshold(grid, 2),
            # Edge detection and blur
            edge_detect,
            blur,
            median_filter,
        ]

        # Composed primitives for complex transformations
        self.composed_primitives = [
            lambda grid: horizontal_mirror(rotate90(grid)),
            lambda grid: vertical_mirror(horizontal_mirror(grid)),
            lambda grid: rotate90(horizontal_mirror(grid)),
            lambda grid: rotate90(vertical_mirror(grid)),
            lambda grid: horizontal_mirror(rotate90(rotate90(grid))),
            lambda grid: vertical_mirror(rotate90(rotate90(rotate90(grid)))),
        ]

        # Combine all primitives
        self.all_primitives = self.basic_primitives.copy()
        self.all_primitives.extend(self.color_primitives[:20])  # Limit color primitives

        if self.use_advanced_primitives:
            self.all_primitives.extend(self.advanced_primitives)
            self.all_primitives.extend(self.composed_primitives)

        print(
            f"Enhanced beam search initialized with {len(self.all_primitives)} primitives"
        )

    def _calculate_grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculate similarity between two grids using multiple metrics."""
        if grid1.shape != grid2.shape:
            return 0.0

        # Pixel-wise similarity
        pixel_similarity = 1.0 - np.mean(np.abs(grid1 - grid2)) / 10.0

        # Structure similarity (object count, size distribution)
        from src.data_pipeline.segmentation import segment_grid

        try:
            objects1 = segment_grid(grid1)
            objects2 = segment_grid(grid2)

            # Object count similarity
            count_similarity = 1.0 - abs(len(objects1) - len(objects2)) / max(
                len(objects1) + len(objects2), 1
            )

            # Size distribution similarity
            sizes1 = [obj["size"] for obj in objects1]
            sizes2 = [obj["size"] for obj in objects2]

            if sizes1 and sizes2:
                size_similarity = 1.0 - abs(np.mean(sizes1) - np.mean(sizes2)) / max(
                    np.mean(sizes1) + np.mean(sizes2), 1
                )
            else:
                size_similarity = 1.0 if not sizes1 and not sizes2 else 0.0

            # Color distribution similarity
            colors1 = np.bincount(grid1.flatten(), minlength=10)
            colors2 = np.bincount(grid2.flatten(), minlength=10)
            color_similarity = 1.0 - np.mean(np.abs(colors1 - colors2)) / max(
                np.sum(colors1) + np.sum(colors2), 1
            )

            # Weighted combination
            similarity = float(
                0.4 * pixel_similarity
                + 0.2 * count_similarity
                + 0.2 * size_similarity
                + 0.2 * color_similarity
            )

            return max(0.0, min(1.0, similarity))

        except Exception:
            # Fallback to pixel similarity only
            return float(pixel_similarity)

    def _evaluate_candidate(
        self,
        candidate: Tuple[List[Callable], np.ndarray],
        target_grids: List[np.ndarray],
        input_grids: List[np.ndarray],
    ) -> float:
        """Evaluate a candidate program using multiple metrics."""
        program, current_output = candidate

        # Calculate similarity to target
        similarities = []
        for i, target_grid in enumerate(target_grids):
            # Apply program to input
            try:
                result = input_grids[i].copy()
                for fn in program:
                    result = fn(result)
                similarity = self._calculate_grid_similarity(result, target_grid)
                similarities.append(similarity)
            except Exception:
                similarities.append(0.0)

        avg_similarity = np.mean(similarities)

        # Penalize long programs
        length_penalty = len(program) * 0.05

        # Bonus for exact matches
        exact_match_bonus = 0.1 if avg_similarity > 0.95 else 0.0

        # Bonus for consistent performance across all examples
        consistency_bonus = 0.05 if np.std(similarities) < 0.1 else 0.0

        score = avg_similarity - length_penalty + exact_match_bonus + consistency_bonus

        return float(score)

    def _generate_candidates(
        self,
        current_candidates: List[Tuple[List[Callable], np.ndarray]],
        primitives: List[Callable],
        input_grids: List[np.ndarray],
    ) -> List[Tuple[List[Callable], np.ndarray]]:
        """Generate new candidates by extending current programs."""
        new_candidates = []

        for program, _ in current_candidates:
            for primitive in primitives:
                try:
                    # Test the extended program
                    extended_program = program + [primitive]

                    # Apply to first input to get current output
                    result = input_grids[0].copy()
                    for fn in extended_program:
                        result = fn(result)

                    new_candidates.append((extended_program, result))

                    # Limit candidates per depth
                    if len(new_candidates) >= self.max_candidates_per_depth:
                        break

                except Exception:
                    # Skip failed primitives
                    continue

            # Limit candidates per depth
            if len(new_candidates) >= self.max_candidates_per_depth:
                break

        return new_candidates

    def search(
        self,
        input_grids: List[np.ndarray],
        output_grids: List[np.ndarray],
        verifier: Optional[Callable] = None,
    ) -> Tuple[List[Callable], bool]:
        """
        Perform enhanced beam search.

        Args:
            input_grids: List of input grids
            output_grids: List of target output grids
            verifier: Optional verification function

        Returns:
            Tuple of (best_program, success)
        """
        print(
            f"🔍 Enhanced beam search: {len(input_grids)} examples, max_depth={self.max_depth}"
        )

        # Initialize with empty program
        current_candidates = [([], input_grids[0])]
        best_score = 0.0
        best_program = []

        start_time = time.time()

        for depth in range(1, self.max_depth + 1):
            print(
                f"  Depth {depth}/{self.max_depth}: {len(current_candidates)} candidates"
            )

            # Generate new candidates
            new_candidates = self._generate_candidates(
                current_candidates, self.all_primitives, input_grids
            )

            if not new_candidates:
                print(f"  No new candidates generated at depth {depth}")
                break

            # Evaluate all candidates
            candidate_scores = []
            for candidate in new_candidates:
                score = self._evaluate_candidate(candidate, output_grids, input_grids)
                candidate_scores.append((score, candidate))

            # Sort by score and keep top candidates
            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            current_candidates = [
                candidate for _, candidate in candidate_scores[: self.beam_width]
            ]

            # Check for early termination
            best_candidate_score = candidate_scores[0][0] if candidate_scores else 0.0
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_program = candidate_scores[0][1][0]

            # Early termination if we have a very good solution
            if best_score > 0.95:
                print(f"  🎉 Early termination: score {best_score:.3f}")
                break

            # Adaptive pruning
            if self.adaptive_pruning and depth > 3:
                # Remove candidates that are clearly worse
                threshold = best_score - 0.2
                current_candidates = [
                    candidate
                    for candidate in current_candidates
                    if self._evaluate_candidate(candidate, output_grids, input_grids)
                    > threshold
                ]

            # Time limit check
            if time.time() - start_time > 300:  # 5 minutes
                print(f"  ⏰ Time limit reached at depth {depth}")
                break

        # Final verification
        if verifier and best_program:
            try:
                def solution_fn(grid):
                    result = grid.copy()
                    for fn in best_program:
                        result = fn(result)
                    return result

                if verifier(solution_fn, input_grids, output_grids):
                    print(f"  ✅ Verified solution found: {len(best_program)} primitives")
                    return best_program, True
            except Exception as e:
                print(f"  ❌ Verification failed: {e}")

        print(f"  ❌ No verified solution found. Best score: {best_score:.3f}")
        return [], False


def enhanced_beam_search(
    input_grids: List[np.ndarray],
    output_grids: List[np.ndarray],
    primitives_list: List[Callable],
    max_depth: int = 8,
    beam_width: int = 20,
    verifier: Optional[Callable] = None,
    heuristic: Optional[Callable] = None,
    **kwargs,
) -> Tuple[List[Callable], bool]:
    """
    Enhanced beam search function for complex ARC tasks.

    Args:
        input_grids: List of input grids
        output_grids: List of target output grids
        primitives_list: List of available primitives
        max_depth: Maximum program depth
        beam_width: Beam width
        verifier: Verification function
        heuristic: Heuristic function (not used in enhanced version)
        **kwargs: Additional arguments

    Returns:
        Tuple of (best_program, success)
    """
    searcher = EnhancedBeamSearch(max_depth=max_depth, beam_width=beam_width, **kwargs)

    return searcher.search(input_grids, output_grids, verifier)
