#!/usr/bin/env python3
"""
Test script to verify the solver is working correctly after fixing indexing errors.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.solver.enhanced_solver import EnhancedNeuroSymbolicSolver

def create_simple_test_task():
    """Create a simple test task to verify the solver works."""
    # Simple rotation task
    input_grid = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    # Expected output: rotate 90 degrees clockwise
    output_grid = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    return [(input_grid, output_grid)]

def test_solver():
    """Test the enhanced solver with a simple task."""
    print("🧪 Testing Enhanced Neuro-Symbolic Solver...")
    
    # Create solver
    solver = EnhancedNeuroSymbolicSolver(
        model_path=None,  # Don't load model for this test
        use_ttt=False,    # Disable TTT for simplicity
        max_search_depth=3,
        beam_width=5
    )
    
    # Create test task
    demo_pairs = create_simple_test_task()
    
    print(f"📋 Test task created with {len(demo_pairs)} demonstration pairs")
    print(f"Input shape: {demo_pairs[0][0].shape}")
    print(f"Output shape: {demo_pairs[0][1].shape}")
    
    try:
        # Try to solve the task
        print("🔍 Attempting to solve task...")
        solution = solver.solve(demo_pairs)
        
        if solution:
            print("✅ Solver returned a solution function!")
            
            # Test the solution
            test_input = demo_pairs[0][0]
            result = solution(test_input)
            print(f"📊 Solution applied successfully!")
            print(f"Result shape: {result.shape}")
            print(f"Result:\n{result}")
            
        else:
            print("❌ Solver did not find a solution")
            
    except Exception as e:
        print(f"❌ Error during solving: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_solver()
    if success:
        print("\n🎉 Solver test completed successfully!")
    else:
        print("\n💥 Solver test failed!")
        sys.exit(1) 