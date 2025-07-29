import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from src.dsl import primitives
import itertools
import heapq


def best_first_search(
    input_grids: List[np.ndarray],
    output_grids: List[np.ndarray],
    primitives_list: List[Callable],
    max_depth: int = 3,
    verifier: Callable = None,
) -> Tuple[List[Callable], bool]:
    """Performs a best-first search over DSL primitive compositions to solve an ARC task.

    This function explores the program space defined by the provided DSL primitives, searching for a sequence
    of operations (a program) that transforms each input grid into its corresponding output grid. The search
    is guided by a simple cost function (program length) and can be extended to use neural guidance in the future.

    Args:
        input_grids (List[np.ndarray]): List of input grids for the task demonstrations.
        output_grids (List[np.ndarray]): List of target output grids for the demonstrations.
        primitives_list (List[Callable]): List of DSL primitive functions to compose.
        max_depth (int, optional): Maximum program length (number of composed primitives). Defaults to 3.
        verifier (Callable, optional): Function to verify candidate programs. Must accept (program, input_grids, output_grids).

    Returns:
        Tuple[List[Callable], bool]: The first successful program (as a list of primitives) and a boolean indicating success.

    Raises:
        ValueError: If input and output grid lists are not the same length.
    """
    if len(input_grids) != len(output_grids):
        raise ValueError("Input and output grid lists must be the same length.")

    # Each program is a tuple of primitive functions
    for depth in range(1, max_depth + 1):
        for program in itertools.product(primitives_list, repeat=depth):

            def candidate_fn(grid):
                result = grid.copy()
                for fn in program:
                    result = fn(result)
                return result

            if verifier is not None and verifier(
                candidate_fn, input_grids, output_grids
            ):
                return list(program), True
    return [], False


def pixel_difference(pred: np.ndarray, target: np.ndarray) -> int:
    """Compute the sum of absolute pixel differences between two grids."""
    return np.sum(np.abs(pred.astype(int) - target.astype(int)))


def object_count_difference(pred: np.ndarray, target: np.ndarray) -> int:
    """Compute the difference in number of objects between two grids.

    This heuristic measures how well the candidate program preserves or transforms
    the object structure of the grid, which is often important in ARC tasks.

    Args:
        pred (np.ndarray): Predicted grid.
        target (np.ndarray): Target grid.

    Returns:
        int: Absolute difference in object count.
    """
    from src.data_pipeline.segmentation import segment_grid

    pred_objects = len(segment_grid(pred))
    target_objects = len(segment_grid(target))
    return abs(pred_objects - target_objects)


def grid_size_difference(pred: np.ndarray, target: np.ndarray) -> int:
    """Compute the difference in grid dimensions between two grids.

    This heuristic penalizes candidates that change grid dimensions when
    the target maintains the same dimensions.

    Args:
        pred (np.ndarray): Predicted grid.
        target (np.ndarray): Target grid.

    Returns:
        int: Sum of differences in height and width.
    """
    pred_shape = pred.shape
    target_shape = target.shape
    return abs(pred_shape[0] - target_shape[0]) + abs(pred_shape[1] - target_shape[1])


def composite_heuristic(
    preds: List[np.ndarray], targets: List[np.ndarray], weights: Dict[str, float] = None
) -> float:
    """Composite heuristic combining multiple metrics for better pruning.

    This function combines pixel difference, object count difference, and grid size
    difference to provide a more comprehensive evaluation of candidate programs.

    Args:
        preds (List[np.ndarray]): List of predicted grids.
        targets (List[np.ndarray]): List of target grids.
        weights (Dict[str, float], optional): Weights for different metrics.
            Defaults to {'pixel': 1.0, 'object': 10.0, 'size': 100.0}.

    Returns:
        float: Combined heuristic score (lower is better).
    """
    if weights is None:
        weights = {"pixel": 1.0, "object": 10.0, "size": 100.0}

    total_score = 0.0
    for pred, target in zip(preds, targets):
        pixel_score = pixel_difference(pred, target)
        object_score = object_count_difference(pred, target)
        size_score = grid_size_difference(pred, target)

        total_score += (
            weights["pixel"] * pixel_score
            + weights["object"] * object_score
            + weights["size"] * size_score
        )

    return total_score


def beam_search(
    input_grids: List[np.ndarray],
    output_grids: List[np.ndarray],
    primitives_list: List[Callable],
    max_depth: int = 3,
    beam_width: int = 5,
    verifier: Callable = None,
    heuristic: Callable = None,
    pruning_threshold: float = None,
    max_candidates_per_depth: int = 1000,
) -> Tuple[List[Callable], bool]:
    """Performs a beam search over DSL primitive compositions to solve an ARC task.

    This function explores the program space defined by the provided DSL primitives, keeping the top-k
    (beam_width) most promising program sequences at each step, as measured by a heuristic. The search
    incorporates heuristic pruning to discard unpromising candidates early, improving efficiency.

    Args:
        input_grids (List[np.ndarray]): List of input grids for the task demonstrations.
        output_grids (List[np.ndarray]): List of target output grids for the demonstrations.
        primitives_list (List[Callable]): List of DSL primitive functions to compose.
        max_depth (int, optional): Maximum program length (number of composed primitives). Defaults to 3.
        beam_width (int, optional): Number of top candidates to keep at each step. Defaults to 5.
        verifier (Callable, optional): Function to verify candidate programs. Must accept (program, input_grids, output_grids).
        heuristic (Callable, optional): Function to score candidate outputs. Defaults to composite_heuristic.
        pruning_threshold (float, optional): Threshold for pruning candidates. If None, no threshold pruning.
        max_candidates_per_depth (int, optional): Maximum candidates to evaluate per depth level. Defaults to 1000.

    Returns:
        Tuple[List[Callable], bool]: The first successful program (as a list of primitives) and a boolean indicating success.
    """
    if len(input_grids) != len(output_grids):
        raise ValueError("Input and output grid lists must be the same length.")

    if heuristic is None:
        heuristic = composite_heuristic

    # Each candidate is (score, [primitive functions])
    beam = [(0, [])]  # Start with the identity program

    for depth in range(1, max_depth + 1):
        candidates = []
        candidates_evaluated = 0

        for score, program in beam:
            for primitive in primitives_list:
                # Early termination if we've evaluated too many candidates
                if candidates_evaluated >= max_candidates_per_depth:
                    break

                new_program = program + [primitive]

                def candidate_fn(grid):
                    result = grid.copy()
                    for fn in new_program:
                        result = fn(result)
                    return result

                # Apply heuristic to evaluate candidate
                try:
                    preds = [candidate_fn(inp) for inp in input_grids]
                    h_score = heuristic(preds, output_grids)
                    candidates_evaluated += 1

                    # Pruning: discard candidates that are clearly diverging
                    if pruning_threshold is not None and h_score > pruning_threshold:
                        continue

                    candidates.append((h_score, new_program))

                    # Check for exact match
                    if verifier is not None and verifier(
                        candidate_fn, input_grids, output_grids
                    ):
                        return new_program, True

                except Exception as e:
                    # Skip candidates that cause errors
                    continue

            if candidates_evaluated >= max_candidates_per_depth:
                break

        # Keep top beam_width candidates
        if candidates:
            beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])
        else:
            # If no candidates survived pruning, return failure
            break

    return [], False


def enhanced_beam_search(
    input_grids: List[np.ndarray],
    output_grids: List[np.ndarray],
    primitives_list: List[Callable],
    max_depth: int = 3,
    beam_width: int = 5,
    verifier: Callable = None,
    heuristic: Callable = None,
    pruning_threshold: float = None,
    max_candidates_per_depth: int = 1000,
    early_termination_threshold: float = 0.1,
    adaptive_pruning: bool = True,
) -> Tuple[List[Callable], bool]:
    """Enhanced beam search with aggressive heuristic pruning and early termination.

    This function implements FR11.2 with improved pruning strategies:
    - Adaptive pruning threshold based on best candidate score
    - Early termination when a near-perfect solution is found
    - Multi-stage pruning with different criteria at different depths
    - Memory-efficient candidate management

    Args:
        input_grids (List[np.ndarray]): List of input grids for the task demonstrations.
        output_grids (List[np.ndarray]): List of target output grids for the demonstrations.
        primitives_list (List[Callable]): List of DSL primitive functions to compose.
        max_depth (int, optional): Maximum program length. Defaults to 3.
        beam_width (int, optional): Number of top candidates to keep. Defaults to 5.
        verifier (Callable, optional): Function to verify candidate programs.
        heuristic (Callable, optional): Function to score candidate outputs.
        pruning_threshold (float, optional): Base threshold for pruning.
        max_candidates_per_depth (int, optional): Maximum candidates per depth.
        early_termination_threshold (float, optional): Score threshold for early termination.
        adaptive_pruning (bool, optional): Use adaptive pruning based on best score.

    Returns:
        Tuple[List[Callable], bool]: The best program found and success indicator.
    """
    if len(input_grids) != len(output_grids):
        raise ValueError("Input and output grid lists must be the same length.")

    if heuristic is None:
        heuristic = composite_heuristic

    # Initialize beam with identity program
    beam = [(0, [])]
    best_score = float("inf")
    best_program = []

    for depth in range(1, max_depth + 1):
        candidates = []
        candidates_evaluated = 0

        # Calculate adaptive pruning threshold
        if adaptive_pruning and beam:
            current_best = min(score for score, _ in beam)
            adaptive_threshold = current_best * 2.0  # Allow 2x worse than current best
            effective_threshold = min(
                pruning_threshold or float("inf"), adaptive_threshold
            )
        else:
            effective_threshold = pruning_threshold

        for score, program in beam:
            for primitive in primitives_list:
                if candidates_evaluated >= max_candidates_per_depth:
                    break

                new_program = program + [primitive]

                def candidate_fn(grid):
                    result = grid.copy()
                    for fn in new_program:
                        result = fn(result)
                    return result

                try:
                    # Evaluate candidate
                    preds = [candidate_fn(inp) for inp in input_grids]
                    h_score = heuristic(preds, output_grids)
                    candidates_evaluated += 1

                    # Multi-stage pruning
                    # Stage 1: Basic threshold pruning
                    if (
                        effective_threshold is not None
                        and h_score > effective_threshold
                    ):
                        continue

                    # Stage 2: Early termination for near-perfect solutions
                    if h_score <= early_termination_threshold:
                        if verifier is not None and verifier(
                            candidate_fn, input_grids, output_grids
                        ):
                            return new_program, True
                        # Even if not perfect, keep very good candidates
                        candidates.append((h_score, new_program))
                        continue

                    # Stage 3: Depth-specific pruning
                    if depth == 1:
                        # At depth 1, be more lenient
                        candidates.append((h_score, new_program))
                    elif depth == 2:
                        # At depth 2, moderate pruning
                        if h_score < effective_threshold * 0.8:
                            candidates.append((h_score, new_program))
                    else:
                        # At depth 3+, aggressive pruning
                        if h_score < effective_threshold * 0.6:
                            candidates.append((h_score, new_program))

                    # Track best candidate
                    if h_score < best_score:
                        best_score = h_score
                        best_program = new_program

                    # Check for exact match
                    if verifier is not None and verifier(
                        candidate_fn, input_grids, output_grids
                    ):
                        return new_program, True

                except Exception as e:
                    # Skip candidates that cause errors
                    continue

            if candidates_evaluated >= max_candidates_per_depth:
                break

        # Keep top beam_width candidates
        if candidates:
            beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])
        else:
            # If no candidates survived pruning, try with relaxed threshold
            if adaptive_pruning and effective_threshold < float("inf"):
                relaxed_candidates = []
                for score, program in beam:
                    for primitive in primitives_list:
                        new_program = program + [primitive]
                        try:

                            def candidate_fn(grid):
                                result = grid.copy()
                                for fn in new_program:
                                    result = fn(result)
                                return result

                            preds = [candidate_fn(inp) for inp in input_grids]
                            h_score = heuristic(preds, output_grids)

                            if h_score < effective_threshold * 1.5:  # Relaxed threshold
                                relaxed_candidates.append((h_score, new_program))

                                if verifier is not None and verifier(
                                    candidate_fn, input_grids, output_grids
                                ):
                                    return new_program, True
                        except:
                            continue

                if relaxed_candidates:
                    beam = heapq.nsmallest(
                        beam_width, relaxed_candidates, key=lambda x: x[0]
                    )
                else:
                    break
            else:
                break

    # Return best program found (even if not perfect)
    if best_program:
        return best_program, False
    return [], False
