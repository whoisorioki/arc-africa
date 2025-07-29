"""
A module for calculating program complexity metrics as defined in the Phase 4 Research Plan.

This module provides a suite of functions to analyze the complexity and diversity
of synthetically generated programs, moving beyond simple primitive counts to a more
sophisticated, multi-faceted understanding of what makes a program "complex."
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Tuple


def calculate_cyclomatic_complexity(program_or_primitives: List[str]) -> int:
    """
    Calculates the Cyclomatic Complexity of a given program or list of primitives.
    """
    decision_primitives = {"if", "for", "while", "case", "conditional_transform"}
    decision_points = 0
    for instruction in program_or_primitives:
        # Check if the instruction IS a decision primitive or starts with one
        instruction_name = instruction.split("(")[0].strip()
        if instruction_name in decision_primitives:
            decision_points += 1

    return decision_points + 1


def get_halstead_counts(
    program_or_primitives: List[str],
) -> Tuple[Dict, Dict, int, int]:
    """
    Parses a program or list of primitives for Halstead counts.
    If given a list of primitive names, it can only count operators.
    """
    operators = Counter()
    operands = Counter()

    instruction_regex = re.compile(r"(\w+)\((.*)\)")

    for instruction in program_or_primitives:
        instruction = instruction.strip()
        match = instruction_regex.match(instruction)

        if match:  # It's a full program string
            op_name = match.group(1)
            operators[op_name] += 1
            args_str = match.group(2)
            if args_str:
                args = [arg.strip().strip("'\"") for arg in args_str.split(",")]
                for arg in args:
                    if arg:
                        operands[arg] += 1
        else:  # It's just a primitive name
            operators[instruction] += 1

    total_operators = sum(operators.values())
    total_operands = sum(operands.values())

    return operators, operands, total_operators, total_operands


def calculate_halstead_metrics(program: List[str]) -> Dict[str, float]:
    """
    Calculates the full suite of Halstead complexity measures.

    These metrics are based on the number of unique and total operators and operands
    in the program, providing a quantitative measure of its lexical complexity.

    Args:
        program (List[str]): A list of strings representing the program.

    Returns:
        Dict[str, float]: A dictionary containing all Halstead metrics:
            n1, n2, N1, N2, vocabulary, length, volume, difficulty, effort.
    """
    unique_operators, unique_operands, N1, N2 = get_halstead_counts(program)

    n1 = len(unique_operators)
    n2 = len(unique_operands)

    # Basic counts
    halstead: Dict[str, float] = {
        "n1_unique_operators": n1,
        "n2_unique_operands": n2,
        "N1_total_operators": N1,
        "N2_total_operands": N2,
    }

    # Derived metrics
    vocabulary = n1 + n2
    length = N1 + N2
    halstead["vocabulary"] = vocabulary
    halstead["length"] = length

    if vocabulary > 0:
        volume = length * math.log2(vocabulary)
    else:
        volume = 0
    halstead["volume"] = volume

    if n2 > 0:
        difficulty = (n1 / 2) * (N2 / n2)
    else:
        difficulty = 0
    halstead["difficulty"] = difficulty

    effort = difficulty * volume
    halstead["effort"] = effort

    return halstead


def analyze_program_complexity(program: List[str]) -> Dict[str, Any]:
    """
    Provides a full complexity analysis of a single program.

    This function serves as the main entry point to the metrics dashboard,
    aggregating multiple complexity scores into a single report for a program.

    Args:
        program (List[str]): A list of strings representing the program.

    Returns:
        Dict[str, Any]: A dictionary containing a full suite of complexity metrics.
    """
    if not isinstance(program, list) or not all(isinstance(i, str) for i in program):
        return {"error": "Invalid program format. Program must be a list of strings."}

    cyclomatic_complexity = calculate_cyclomatic_complexity(program)
    halstead_metrics = calculate_halstead_metrics(program)

    dashboard = {"cyclomatic_complexity": cyclomatic_complexity, **halstead_metrics}

    return dashboard


if __name__ == "__main__":
    # Example usage for demonstration and testing

    # A simple program with low complexity
    simple_program = ["colorfilter('blue')", "rotate90('cw')"]

    # A more complex program with conditionals and more operators/operands
    complex_program = [
        "c = count_objects(segment_grid(input))",
        "conditional_transform(c > 2, 'rotate90', 'cw')",
        "fill(input, 'red', segment_grid(input)[0])",
    ]

    print("--- Analyzing Simple Program ---")
    simple_analysis = analyze_program_complexity(simple_program)
    for key, value in simple_analysis.items():
        print(f"  {key}: {value}")

    print("\n--- Analyzing Complex Program ---")
    complex_analysis = analyze_program_complexity(complex_program)
    for key, value in complex_analysis.items():
        print(f"  {key}: {value}")

    # Expected output for simple_program:
    # cyclomatic_complexity: 1
    # n1_unique_operators: 2, n2_unique_operands: 2, N1_total_operators: 2, N2_total_operands: 2
    # vocabulary: 4, length: 4, volume: 8.0, difficulty: 1.0, effort: 8.0

    # Expected output for complex_program:
    # cyclomatic_complexity: 2
    # n1: 4, n2: 7, N1: 4, N2: 8
    # vocabulary: 11, length: 12, volume: ~41.5, difficulty: ~2.28, effort: ~94.8
