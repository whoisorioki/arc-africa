#!/usr/bin/env python3
"""
Debug script to understand why solutions aren't producing exact matches.
"""

import numpy as np
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.solver.enhanced_solver import EnhancedNeuroSymbolicSolver

def load_arc_task(task_path: str):
    """Load an ARC task from JSON file."""
    with open(task_path, 'r') as f:
        data = json.load(f)
    
    demo_pairs = []
    for train in data.get('train', []):
        input_grid = np.array(train['input'])
        output_grid = np.array(train['output'])
        demo_pairs.append((input_grid, output_grid))
    
    return demo_pairs

def debug_task(task_file):
    """Debug a specific task to understand the solution."""
    print(f"🔍 Debugging task: {task_file}")
    
    # Load task
    demo_pairs = load_arc_task(task_file)
    print(f"Demo pairs: {len(demo_pairs)}")
    
    # Show first demo pair
    input_grid, expected_output = demo_pairs[0]
    print(f"Input shape: {input_grid.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Input:\n{input_grid}")
    print(f"Expected output:\n{expected_output}")
    
    # Create solver
    solver = EnhancedNeuroSymbolicSolver(
        model_path=None,
        use_ttt=False,
        max_search_depth=3,
        beam_width=5
    )
    
    # Solve task
    print("\n🔍 Solving...")
    solution = solver.solve(demo_pairs)
    
    if solution:
        print("✅ Solution found!")
        
        # Test solution on first demo pair
        result = solution(input_grid)
        print(f"Result shape: {result.shape}")
        print(f"Result:\n{result}")
        
        # Check if exact match
        is_exact = np.array_equal(result, expected_output)
        print(f"Exact match: {is_exact}")
        
        # Show differences
        if not is_exact:
            print(f"Differences:")
            print(f"Expected:\n{expected_output}")
            print(f"Got:\n{result}")
            
            # Calculate similarity
            if result.shape == expected_output.shape:
                similarity = 1.0 - np.mean(np.abs(result - expected_output)) / 10.0
                print(f"Similarity score: {similarity:.3f}")
            else:
                print(f"Shape mismatch!")
    else:
        print("❌ No solution found")

if __name__ == "__main__":
    task_file = "data/training/train_0003.json"  # One that found a solution
    debug_task(task_file) 