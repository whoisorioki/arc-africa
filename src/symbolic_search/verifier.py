import numpy as np
from typing import List, Callable


def verify_program(program: Callable, input_grids: List[np.ndarray], output_grids: List[np.ndarray]) -> bool:
    """Verifies if a candidate program solves an ARC task by matching all demonstration outputs.

    This function executes the candidate program on each input grid and checks if the resulting outputs
    exactly match the provided output grids. Used as the core verifier in the symbolic search engine.

    Args:
        program (Callable): A function that takes a grid and returns a transformed grid.
        input_grids (List[np.ndarray]): List of input grids for the task demonstrations.
        output_grids (List[np.ndarray]): List of target output grids for the demonstrations.

    Returns:
        bool: True if the program's outputs match all demonstration outputs exactly, False otherwise.
    """
    for inp, out in zip(input_grids, output_grids):
        try:
            pred = program(inp)
        except Exception:
            return False
        if not np.array_equal(pred, out):
            return False
    return True
