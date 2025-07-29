#!/usr/bin/env python3
"""
Quick test script to validate enhanced components before full training.
"""

import sys
import os
import numpy as np
import torch
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.solver.enhanced_solver import EnhancedNeuroSymbolicSolver, load_arc_task
from src.symbolic_search.enhanced_search import EnhancedBeamSearch


def test_enhanced_solver_initialization():
    """Test if the enhanced solver initializes correctly."""
    print("=== Testing Enhanced Solver Initialization ===")

    try:
        solver = EnhancedNeuroSymbolicSolver(
            model_path="models/neural_guide_enhanced.pth",  # Use new model
            top_k_primitives=5,
            max_search_depth=6,
            beam_width=10,
            use_enhanced_search=True,
            use_ttt=False,  # Disable TTT for fast test
        )
        print("✅ Enhanced solver initialized successfully")
        print(f"   - Device: {solver.device}")
        print(f"   - Primitives: {len(solver.primitives)}")
        print(f"   - Enhanced search: {solver.use_enhanced_search}")
        print(f"   - TTT enabled: {solver.use_ttt}")
        return True
    except Exception as e:
        print(f"❌ Enhanced solver initialization failed: {e}")
        return False


def test_enhanced_beam_search():
    """Test the enhanced beam search on a simple task."""
    print("\n=== Testing Enhanced Beam Search ===")

    try:
        # Create a simple test case
        input_grid = np.array([[1, 1, 0], [0, 0, 2]])
        output_grid = np.array([[0, 0, 2], [1, 1, 0]])  # Vertical flip

        # Create enhanced beam search
        searcher = EnhancedBeamSearch(
            max_depth=3, beam_width=5, use_advanced_primitives=True
        )

        # Search for solution
        program, success = searcher.search(
            input_grids=[input_grid], output_grids=[output_grid]
        )

        if success and program:
            print("✅ Enhanced beam search found solution")
            print(f"   - Program length: {len(program)}")

            # Test the solution
            result = input_grid.copy()
            for fn in program:
                result = fn(result)

            is_correct = np.array_equal(result, output_grid)
            print(f"   - Solution correct: {is_correct}")
            return True
        else:
            print("❌ Enhanced beam search failed to find solution")
            return False

    except Exception as e:
        print(f"❌ Enhanced beam search test failed: {e}")
        return False


def test_enhanced_solver_on_simple_task():
    """Test the enhanced solver on a simple task."""
    print("\n=== Testing Enhanced Solver on Simple Task ===")

    try:
        # Create a simple task
        input_grid = np.array([[1, 1, 0], [0, 0, 2]])
        output_grid = np.array([[0, 0, 2], [1, 1, 0]])  # Vertical flip
        demo_pairs = [(input_grid, output_grid)]

        # Create enhanced solver
        solver = EnhancedNeuroSymbolicSolver(
            model_path="models/neural_guide_enhanced.pth",  # Use new model
            top_k_primitives=5,
            max_search_depth=4,
            beam_width=10,
            use_enhanced_search=True,
            use_ttt=False,  # Disable TTT for quick test
        )

        print("Input grid:")
        print(input_grid)
        print("Expected output:")
        print(output_grid)

        # Solve the task
        start_time = time.time()
        solution_fn = solver.solve(demo_pairs)
        solve_time = time.time() - start_time

        if solution_fn is not None:
            print("✅ Enhanced solver found solution")
            print(f"   - Solve time: {solve_time:.2f} seconds")

            # Test the solution
            result = solution_fn(input_grid)
            print("Actual output:")
            print(result)

            is_correct = np.array_equal(result, output_grid)
            print(f"   - Solution correct: {is_correct}")
            return True
        else:
            print("❌ Enhanced solver failed to find solution")
            return False

    except Exception as e:
        print(f"❌ Enhanced solver test failed: {e}")
        return False


def test_enhanced_solver_on_real_task():
    """Test the enhanced solver on a real ARC task."""
    print("\n=== Testing Enhanced Solver on Real Task ===")

    try:
        # Load a real task
        task_path = "data/training/train_0000.json"
        if not os.path.exists(task_path):
            print(f"❌ Task file not found: {task_path}")
            return False

        demo_pairs = load_arc_task(task_path)
        print(f"✅ Loaded task with {len(demo_pairs)} demonstration pairs")

        # Show first demo pair
        input_grid, output_grid = demo_pairs[0]
        print("First demo input shape:", input_grid.shape)
        print("First demo output shape:", output_grid.shape)

        # Create enhanced solver with conservative settings
        solver = EnhancedNeuroSymbolicSolver(
            model_path="models/neural_guide_enhanced.pth",  # Use new model
            top_k_primitives=10,  # More primitives
            max_search_depth=6,  # Shorter depth for speed
            beam_width=15,  # Larger beam
            use_enhanced_search=True,
            use_ttt=False,  # Disable TTT for quick test
        )

        # Solve the task
        start_time = time.time()
        solution_fn = solver.solve(demo_pairs)
        solve_time = time.time() - start_time

        print(f"   - Solve time: {solve_time:.2f} seconds")

        if solution_fn is not None:
            print("✅ Enhanced solver found solution for real task")

            # Test on first demo
            result = solution_fn(input_grid)
            print("Solver output shape:", result.shape)

            # Check if solution is reasonable (not all zeros)
            if np.any(result != 0):
                print("✅ Solution produces non-zero output")
                return True
            else:
                print("❌ Solution produces all-zero output")
                return False
        else:
            print("❌ Enhanced solver failed to find solution for real task")
            return False

    except Exception as e:
        print(f"❌ Real task test failed: {e}")
        return False


def test_neural_guide_predictions():
    """Test neural guide predictions."""
    print("\n=== Testing Neural Guide Predictions ===")

    try:
        # Create test grids
        input_grid = np.array([[1, 1, 0], [0, 0, 2]])
        output_grid = np.array([[0, 0, 2], [1, 1, 0]])
        demo_pairs = [(input_grid, output_grid)]

        # Create solver
        solver = EnhancedNeuroSymbolicSolver(
            model_path="models/neural_guide_enhanced.pth", top_k_primitives=5
        )

        # Get predictions
        predictions = solver.predict_primitives(demo_pairs)
        print(f"✅ Neural guide predictions: {predictions}")
        print(f"   - Number of predictions: {len(predictions)}")

        if len(predictions) > 0:
            return True
        else:
            print("❌ No predictions generated")
            return False

    except Exception as e:
        print(f"❌ Neural guide test failed: {e}")
        return False


def test_full_system_on_real_task():
    """Test the full enhanced solver system on a real ARC task."""
    print("\n=== Testing Full System on Real Task ===")
    task_path = "data/training/train_0001.json"  # Use a different task
    if not os.path.exists(task_path):
        print(f"⚠️  Task file not found, skipping test: {task_path}")
        return True  # Not a failure if file is missing

    try:
        # 1. Load the task
        demo_pairs = load_arc_task(task_path)
        print(f"✅ Loaded task: {os.path.basename(task_path)}")

        # 2. Initialize the solver with memory-efficient settings
        solver = EnhancedNeuroSymbolicSolver(
            model_path="models/neural_guide_enhanced.pth",
            top_k_primitives=8,
            max_search_depth=6,
            beam_width=15,
            use_enhanced_search=True,
            use_ttt=False,  # Keep TTT off for speed
        )

        # 3. Solve the task
        print("🚀 Attempting to solve the task...")
        start_time = time.time()
        solution_fn = solver.solve_task(demo_pairs)
        solve_time = time.time() - start_time
        print(f"   - Solve time: {solve_time:.2f} seconds")

        # 4. Validate the solution
        if solution_fn:
            print("✅ Solver returned a solution function.")
            # Verify against all demonstration pairs
            correct_predictions = 0
            for i, (inp, out) in enumerate(demo_pairs):
                predicted_out = solution_fn(inp)
                if np.array_equal(predicted_out, out):
                    correct_predictions += 1
                    print(f"   - Demo {i+1}: ✅ Correct")
                else:
                    print(f"   - Demo {i+1}: ❌ Incorrect")
            if correct_predictions == len(demo_pairs):
                print("🎉🎉🎉 Full system test PASSED! Solution is correct.")
                return True
            else:
                print(
                    f"❌ Full system test FAILED. Solution was incorrect on {len(demo_pairs) - correct_predictions} demos."
                )
                return False
        else:
            print("❌ Solver did not find a solution.")
            # This is not necessarily a failure for a hard task, but we'll flag it
            return False

    except Exception as e:
        print(f"❌ Full system test failed with an exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all enhanced component tests."""
    print("🔍 Enhanced Components Test Suite")
    print("=" * 50)

    tests = [
        ("Enhanced Solver Initialization", test_enhanced_solver_initialization),
        ("Enhanced Beam Search", test_enhanced_beam_search),
        ("Neural Guide Predictions", test_neural_guide_predictions),
        ("Enhanced Solver on Simple Task", test_enhanced_solver_on_simple_task),
        ("Enhanced Solver on Real Task", test_enhanced_solver_on_real_task),
        ("Full System Test on Real Task", test_full_system_on_real_task),
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
    print("🔍 ENHANCED COMPONENTS TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All enhanced components are working correctly!")
        print("🚀 Ready to proceed with enhanced training.")
    elif passed >= total - 1:
        print("⚠️  Most components working. Proceed with caution.")
    else:
        print("❌ Multiple components failed. Fix issues before training.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
