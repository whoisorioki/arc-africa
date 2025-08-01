#!/usr/bin/env python3
"""Complete TTT test with synthetic ARC tasks."""

import sys
import os
sys.path.append('src')

import numpy as np
import time
from solver.main_solver import NeuroSymbolicSolver

def create_arc_tasks():
    """Create realistic ARC-style tasks."""
    
    tasks = [
        {
            "name": "Simple Rotation",
            "description": "Rotate pattern 90 degrees",
            "train": [
                (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), 
                 np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])),
                (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), 
                 np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
            ],
            "test": [
                np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            ]
        },
        {
            "name": "Color Replacement",
            "description": "Replace color 1 with color 2",
            "train": [
                (np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]), 
                 np.array([[0, 2, 0], [2, 1, 2], [0, 2, 0]]))
            ],
            "test": [
                np.array([[0, 2, 0], [2, 1, 2], [0, 2, 0]])
            ]
        },
        {
            "name": "Pattern Mirror",
            "description": "Mirror the pattern horizontally",
            "train": [
                (np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]]), 
                 np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1]]))
            ],
            "test": [
                np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]])
            ]
        }
    ]
    
    return tasks

def test_ttt_with_tasks():
    """Test TTT with synthetic ARC tasks."""
    
    print("Testing TTT with synthetic ARC tasks")
    print("=" * 50)
    
    solver = NeuroSymbolicSolver()
    tasks = create_arc_tasks()
    
    results = []
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}: {task['name']}")
        print(f"Description: {task['description']}")
        
        # Extract training data
        train_inputs = [pair[0] for pair in task['train']]
        train_outputs = [pair[1] for pair in task['train']]
        
        print(f"Training pairs: {len(train_inputs)}")
        print(f"Input shapes: {[grid.shape for grid in train_inputs]}")
        print(f"Output shapes: {[grid.shape for grid in train_outputs]}")
        
        # Solve task
        start_time = time.time()
        solution = solver.solve_task(train_inputs, train_outputs)
        solve_time = time.time() - start_time
        
        print(f"Solution: {solution}")
        print(f"Solve time: {solve_time:.3f}s")
        
        # Test each predicted primitive
        print("Testing predicted primitives:")
        for j, primitive_name in enumerate(solution[:3]):  # Test first 3
            try:
                from dsl.primitives import PRIMITIVE_FUNCTIONS
                func = PRIMITIVE_FUNCTIONS[primitive_name]
                result = func(train_inputs[0])
                print(f"  {j+1}. {primitive_name}: {result.shape}")
            except Exception as e:
                print(f"  {j+1}. {primitive_name}: ERROR - {e}")
        
        results.append({
            "task_name": task['name'],
            "solution": solution,
            "solve_time": solve_time,
            "num_primitives": len(solution)
        })
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    
    for result in results:
        print(f"{result['task_name']}: {result['num_primitives']} primitives, {result['solve_time']:.3f}s")
    
    avg_time = np.mean([r['solve_time'] for r in results])
    print(f"\nAverage solve time: {avg_time:.3f}s")
    
    return results

def main():
    """Main function."""
    results = test_ttt_with_tasks()
    print("\nComplete TTT testing finished!")

if __name__ == "__main__":
    main()
