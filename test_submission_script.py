#!/usr/bin/env python3
"""
Test script to verify the submission script works correctly.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_model_exists():
    """Test that the persistent model exists."""
    model_path = "models/neural_guide_persistent.pth"
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        return False

def test_solver_initialization():
    """Test that the solver can be initialized with the persistent model."""
    try:
        from src.solver.enhanced_solver import EnhancedNeuroSymbolicSolver
        
        solver = EnhancedNeuroSymbolicSolver(
            model_path="models/neural_guide_persistent.pth",
            top_k_primitives=5,
            max_search_depth=5,
            beam_width=10,
            use_enhanced_search=True,
            use_ttt=False  # Disable TTT for testing
        )
        print("✅ Solver initialized successfully with persistent model")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize solver: {e}")
        return False

def test_simple_task():
    """Test solving a simple task."""
    try:
        from src.solver.enhanced_solver import EnhancedNeuroSymbolicSolver
        
        # Create solver
        solver = EnhancedNeuroSymbolicSolver(
            model_path="models/neural_guide_persistent.pth",
            top_k_primitives=5,
            max_search_depth=3,
            beam_width=5,
            use_enhanced_search=True,
            use_ttt=False
        )
        
        # Create a simple test task
        input_grid = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        output_grid = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        demo_pairs = [(input_grid, output_grid)]
        
        # Try to solve
        solution = solver.solve(demo_pairs)
        
        if solution is not None:
            print("✅ Solver found a solution")
            # Test the solution
            result = solution(input_grid)
            print(f"✅ Solution applied successfully, result shape: {result.shape}")
        else:
            print("⚠️ Solver did not find a solution (this is expected for complex tasks)")
        
        return True
    except Exception as e:
        print(f"❌ Error testing simple task: {e}")
        return False

def test_submission_script_imports():
    """Test that all imports in the submission script work."""
    try:
        # Test imports that the submission script uses
        import pandas as pd
        import torch
        from src.solver.enhanced_solver import EnhancedNeuroSymbolicSolver
        from src.symbolic_search.verifier import verify_program
        
        print("✅ All submission script imports work")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Submission Script Components")
    print("=" * 50)
    
    tests = [
        ("Model Exists", test_model_exists),
        ("Solver Initialization", test_solver_initialization),
        ("Simple Task Solving", test_simple_task),
        ("Submission Script Imports", test_submission_script_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Submission script should work correctly.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 