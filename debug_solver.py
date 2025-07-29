#!/usr/bin/env python3
"""
Debug script to investigate why the solver produces all-zero grids.
This script tests each component of the solver systematically.
"""

import sys
import os
import numpy as np
import json
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.solver.main_solver import NeuroSymbolicSolver, load_arc_task


def test_neural_guide_loading():
    """Test if the neural guide model loads correctly."""
    print("=== Testing Neural Guide Loading ===")

    model_path = "models/neural_guide_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False

    try:
        solver = NeuroSymbolicSolver(model_path=model_path)
        print("✓ Neural guide loaded successfully")
        print(f"✓ Device: {solver.device}")
        print(f"✓ Available primitives: {len(solver.primitives)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load neural guide: {e}")
        return False


def test_primitive_prediction():
    """Test if the neural guide can predict primitives."""
    print("\n=== Testing Primitive Prediction ===")

    try:
        solver = NeuroSymbolicSolver(model_path="models/neural_guide_best.pth")

        # Create a simple test case
        test_grid = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]])
        demo_pairs = [(test_grid, test_grid)]  # Identity transformation for testing

        predictions = solver.predict_primitives(demo_pairs)
        print(f"✓ Neural guide predictions: {predictions}")
        print(f"✓ Number of predictions: {len(predictions)}")

        return True
    except Exception as e:
        print(f"❌ Failed to predict primitives: {e}")
        return False


def test_dynamic_primitive_generation():
    """Test if dynamic primitives are generated correctly."""
    print("\n=== Testing Dynamic Primitive Generation ===")

    try:
        solver = NeuroSymbolicSolver(model_path="models/neural_guide_best.pth")

        # Create a test case with multiple colors
        test_grid = np.array([[1, 1, 2], [1, 0, 2], [0, 0, 3]])
        demo_pairs = [(test_grid, test_grid)]

        dynamic_prims = solver._generate_dynamic_primitives(demo_pairs)
        print(f"✓ Generated {len(dynamic_prims)} dynamic primitives")

        # Test a few primitives
        for i, (name, func) in enumerate(list(dynamic_prims.items())[:3]):
            try:
                result = func(test_grid.copy())
                print(
                    f"✓ {name}: Input shape {test_grid.shape} -> Output shape {result.shape}"
                )
            except Exception as e:
                print(f"❌ {name} failed: {e}")

        return True
    except Exception as e:
        print(f"❌ Failed to generate dynamic primitives: {e}")
        return False


def test_beam_search():
    """Test if beam search works correctly."""
    print("\n=== Testing Beam Search ===")

    try:
        from src.symbolic_search.search import beam_search
        from src.symbolic_search.verifier import verify_program

        # Create a simple test case
        input_grid = np.array([[1, 1], [0, 0]])
        output_grid = np.array([[0, 0], [1, 1]])  # Vertical flip

        # Simple primitives for testing
        def flip_vertical(grid):
            return np.flipud(grid)

        def flip_horizontal(grid):
            return np.fliplr(grid)

        primitives = [flip_vertical, flip_horizontal]

        program, success = beam_search(
            input_grids=[input_grid],
            output_grids=[output_grid],
            primitives_list=primitives,
            max_depth=2,
            beam_width=3,
            verifier=verify_program,
        )

        if success:
            print("✓ Beam search found solution")
            print(f"✓ Program length: {len(program)}")

            # Test the solution
            result = input_grid.copy()
            for fn in program:
                result = fn(result)

            is_correct = np.array_equal(result, output_grid)
            print(f"✓ Solution correct: {is_correct}")
            return True
        else:
            print("❌ Beam search failed to find solution")
            return False

    except Exception as e:
        print(f"❌ Beam search test failed: {e}")
        return False


def test_solver_on_simple_task():
    """Test the solver on a simple task."""
    print("\n=== Testing Solver on Simple Task ===")

    try:
        solver = NeuroSymbolicSolver(model_path="models/neural_guide_best.pth")

        # Create a simple task: vertical flip
        input_grid = np.array([[1, 1, 0], [0, 0, 2]])
        output_grid = np.array([[0, 0, 2], [1, 1, 0]])  # Vertical flip

        demo_pairs = [(input_grid, output_grid)]

        print("Input grid:")
        print(input_grid)
        print("Expected output:")
        print(output_grid)

        # Try to solve
        solution_fn = solver.solve(demo_pairs)

        if solution_fn is not None:
            print("✓ Solver found a solution")

            # Test the solution
            result = solution_fn(input_grid)
            print("Actual output:")
            print(result)

            is_correct = np.array_equal(result, output_grid)
            print(f"✓ Solution correct: {is_correct}")

            if not is_correct:
                print("❌ Solution is incorrect")
                return False
            return True
        else:
            print("❌ Solver failed to find solution")
            return False

    except Exception as e:
        print(f"❌ Solver test failed: {e}")
        return False


def test_solver_on_real_task():
    """Test the solver on a real ARC task."""
    print("\n=== Testing Solver on Real Task ===")

    try:
        # Load a real task
        task_path = "data/training/train_0000.json"
        if not os.path.exists(task_path):
            print(f"❌ Task file not found: {task_path}")
            return False

        demo_pairs = load_arc_task(task_path)
        print(f"✓ Loaded task with {len(demo_pairs)} demonstration pairs")

        # Show first demo pair
        input_grid, output_grid = demo_pairs[0]
        print("First demo input:")
        print(input_grid)
        print("First demo output:")
        print(output_grid)

        solver = NeuroSymbolicSolver(model_path="models/neural_guide_best.pth")

        # Try to solve
        solution_fn = solver.solve(demo_pairs)

        if solution_fn is not None:
            print("✓ Solver found a solution")

            # Test on the first demo
            result = solution_fn(input_grid)
            print("Solver output:")
            print(result)

            is_correct = np.array_equal(result, output_grid)
            print(f"✓ Solution correct: {is_correct}")

            if not is_correct:
                print("❌ Solution is incorrect")
                return False
            return True
        else:
            print("❌ Solver failed to find solution")
            return False

    except Exception as e:
        print(f"❌ Real task test failed: {e}")
        return False


def main():
    """Run all debug tests."""
    print("🔍 ARC Solver Debug Audit")
    print("=" * 50)

    tests = [
        ("Neural Guide Loading", test_neural_guide_loading),
        ("Primitive Prediction", test_primitive_prediction),
        ("Dynamic Primitive Generation", test_dynamic_primitive_generation),
        ("Beam Search", test_beam_search),
        ("Simple Task Solver", test_solver_on_simple_task),
        ("Real Task Solver", test_solver_on_real_task),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("🔍 DEBUG AUDIT SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The solver should be working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for issues.")


if __name__ == "__main__":
    main()
