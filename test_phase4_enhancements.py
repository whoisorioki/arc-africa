#!/usr/bin/env python3
"""
Test script for Phase 4 enhancements of the Neuro-Symbolic Solver.

This script validates the implementation of:
- FR11.2: Enhanced beam search with aggressive heuristic pruning
- FR13.1: More aggressive Test-Time Training (TTT)
- FR12: Optimized local validation pipeline
- NFR5: Iteration speed requirements

Usage:
    python test_phase4_enhancements.py
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.symbolic_search.search import enhanced_beam_search, composite_heuristic
from src.symbolic_search.verifier import verify_program
from src.solver.main_solver import NeuroSymbolicSolver
from src.data_pipeline.segmentation import segment_grid


def create_simple_test_task():
    """Create a simple test task for validation."""
    # Simple task: replace color 1 with color 2
    input_grid = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    output_grid = np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]])

    return [(input_grid, output_grid)]


def test_enhanced_beam_search():
    """Test the enhanced beam search algorithm."""
    print("🧪 Testing Enhanced Beam Search (FR11.2)...")

    # Create test data
    demo_pairs = create_simple_test_task()
    input_grids = [pair[0] for pair in demo_pairs]
    output_grids = [pair[1] for pair in demo_pairs]

    # Simple primitives for testing
    def replace_1_to_2(grid):
        return np.where(grid == 1, 2, grid)

    def identity(grid):
        return grid.copy()

    primitives = [replace_1_to_2, identity]

    # Test enhanced beam search
    start_time = time.time()
    solution_program, success = enhanced_beam_search(
        input_grids=input_grids,
        output_grids=output_grids,
        primitives_list=primitives,
        max_depth=2,
        beam_width=3,
        verifier=verify_program,
        heuristic=composite_heuristic,
        pruning_threshold=10.0,
        early_termination_threshold=1.0,
        adaptive_pruning=True,
        max_candidates_per_depth=100,
    )
    search_time = time.time() - start_time

    print(f"  Search time: {search_time:.3f}s")
    print(f"  Success: {success}")
    print(f"  Solution length: {len(solution_program)}")

    if solution_program:
        # Test the solution
        test_input = input_grids[0]
        result = test_input.copy()
        for fn in solution_program:
            result = fn(result)

        is_correct = np.array_equal(result, output_grids[0])
        print(f"  Solution correct: {is_correct}")

        if is_correct:
            print("  ✅ Enhanced beam search test PASSED")
            return True
        else:
            print("  ❌ Enhanced beam search test FAILED")
            return False
    else:
        print("  ❌ Enhanced beam search test FAILED - No solution found")
        return False


def test_aggressive_ttt():
    """Test the aggressive TTT implementation."""
    print("\n🧪 Testing Aggressive TTT (FR13.1)...")

    try:
        # Initialize solver
        solver = NeuroSymbolicSolver(
            model_path="models/neural_guide_best.pth",
            top_k_primitives=3,
            max_search_depth=3,
            beam_width=3,
            device="cpu",  # Use CPU for testing
        )

        # Create test task
        demo_pairs = create_simple_test_task()

        # Test TTT-enhanced solving
        start_time = time.time()
        solution_fn = solver.solve_task(demo_pairs)
        solve_time = time.time() - start_time

        print(f"  TTT solve time: {solve_time:.3f}s")

        if solution_fn is not None:
            # Test the solution
            test_input = demo_pairs[0][0]
            result = solution_fn(test_input)
            expected = demo_pairs[0][1]

            is_correct = np.array_equal(result, expected)
            print(f"  Solution correct: {is_correct}")

            if is_correct:
                print("  ✅ Aggressive TTT test PASSED")
                return True
            else:
                print("  ❌ Aggressive TTT test FAILED")
                return False
        else:
            print("  ❌ Aggressive TTT test FAILED - No solution found")
            return False

    except Exception as e:
        print(f"  ❌ Aggressive TTT test FAILED - Error: {e}")
        return False


def test_validation_pipeline():
    """Test the optimized validation pipeline."""
    print("\n🧪 Testing Validation Pipeline (FR12)...")

    try:
        # Create a small test dataset
        test_tasks = {}
        for i in range(3):
            task_id = f"test_{i:04d}"
            demo_pairs = create_simple_test_task()

            test_tasks[task_id] = {
                "train": [
                    {
                        "input": demo_pairs[0][0].tolist(),
                        "output": demo_pairs[0][1].tolist(),
                    }
                ],
                "test": [
                    {
                        "input": demo_pairs[0][0].tolist(),
                        "output": demo_pairs[0][1].tolist(),
                    }
                ],
            }

        # Save test tasks
        test_data_dir = "test_validation_data"
        os.makedirs(test_data_dir, exist_ok=True)

        for task_id, task in test_tasks.items():
            task_file = os.path.join(test_data_dir, f"{task_id}.json")
            with open(task_file, "w") as f:
                json.dump(task, f, indent=2)

        # Run validation
        start_time = time.time()

        # Import and run validation
        from scripts.local_validate import main as run_validation

        # Temporarily modify sys.argv to simulate command line arguments
        original_argv = sys.argv
        sys.argv = [
            "test_validation",
            "--data_path",
            test_data_dir,
            "--model_path",
            "models/neural_guide_best.pth",
            "--beam_width",
            "3",
            "--max_search_depth",
            "3",
            "--output_dir",
            "test_validation_results",
            "--num_workers",
            "1",
            "--batch_size",
            "2",
        ]

        try:
            run_validation()
            validation_time = time.time() - start_time

            print(f"  Validation time: {validation_time:.3f}s")

            # Check results
            results_file = "test_validation_results/validation_summary.json"
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    summary = json.load(f)

                solve_rate = summary.get("solve_rate", 0)
                total_time = summary.get("total_time", 0)

                print(f"  Solve rate: {solve_rate:.2%}")
                print(f"  Total time: {total_time:.2f}s")

                # Check NFR5 compliance (<3 hours for full validation)
                if total_time < 10800:  # 3 hours
                    print("  ✅ Validation pipeline meets NFR5 (speed requirement)")
                    pipeline_success = True
                else:
                    print("  ❌ Validation pipeline violates NFR5 (too slow)")
                    pipeline_success = False

                # Check FR12.3 compliance (failures.log)
                failures_file = "test_validation_results/failures.log"
                if os.path.exists(failures_file):
                    print("  ✅ Failures log generated (FR12.3)")
                    failures_success = True
                else:
                    print("  ❌ Failures log missing (FR12.3)")
                    failures_success = False

                return pipeline_success and failures_success
            else:
                print("  ❌ Validation results not found")
                return False

        finally:
            # Restore original argv
            sys.argv = original_argv

    except Exception as e:
        print(f"  ❌ Validation pipeline test FAILED - Error: {e}")
        return False


def test_dynamic_primitives():
    """Test dynamic primitive generation."""
    print("\n🧪 Testing Dynamic Primitives (FR10)...")

    try:
        from src.symbolic_search.dynamic_primitives import (
            generate_dynamic_primitives,
            count_dynamic_variants,
        )

        # Create test task with multiple colors
        input_grid = np.array([[0, 1, 2], [1, 2, 1], [2, 1, 0]])

        output_grid = np.array([[0, 2, 1], [2, 1, 2], [1, 2, 0]])

        demo_pairs = [(input_grid, output_grid)]

        # Test dynamic primitive generation
        from src.dsl.primitives import colorfilter, fill

        base_primitives = [colorfilter, fill]
        dynamic_primitives = generate_dynamic_primitives(demo_pairs, base_primitives)

        variant_stats = count_dynamic_variants(demo_pairs, base_primitives)

        print(f"  Base primitives: {variant_stats['base_primitives']}")
        print(f"  Total variants: {variant_stats['total_variants']}")
        print(f"  Unique colors: {variant_stats['unique_colors']}")
        print(f"  Unique objects: {variant_stats['unique_objects']}")

        if variant_stats["total_variants"] > variant_stats["base_primitives"]:
            print("  ✅ Dynamic primitives test PASSED")
            return True
        else:
            print("  ❌ Dynamic primitives test FAILED")
            return False

    except Exception as e:
        print(f"  ❌ Dynamic primitives test FAILED - Error: {e}")
        return False


def main():
    """Run all Phase 4 enhancement tests."""
    print("🚀 Phase 4 Enhancement Tests")
    print("=" * 50)

    test_results = {}

    # Test enhanced beam search
    test_results["enhanced_beam_search"] = test_enhanced_beam_search()

    # Test aggressive TTT
    test_results["aggressive_ttt"] = test_aggressive_ttt()

    # Test validation pipeline
    test_results["validation_pipeline"] = test_validation_pipeline()

    # Test dynamic primitives
    test_results["dynamic_primitives"] = test_dynamic_primitives()

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Phase 4 enhancements are working correctly!")
        return True
    else:
        print("⚠️  Some enhancements need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
