#!/usr/bin/env python3
"""
Comprehensive test script to validate the solver on actual ARC training tasks.
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
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

def evaluate_solution(solution_func, demo_pairs):
    """Evaluate a solution function on demonstration pairs."""
    if solution_func is None:
        return 0.0
    
    correct = 0
    total = len(demo_pairs)
    
    for input_grid, expected_output in demo_pairs:
        try:
            result = solution_func(input_grid)
            if np.array_equal(result, expected_output):
                correct += 1
        except Exception as e:
            print(f"Error applying solution: {e}")
            continue
    
    return correct / total if total > 0 else 0.0

def test_solver_on_training_tasks():
    """Test the solver on a few training tasks."""
    print("🧪 Testing Enhanced Neuro-Symbolic Solver on ARC Training Tasks...")
    
    # Create solver
    solver = EnhancedNeuroSymbolicSolver(
        model_path="models/neural_guide_persistent.pth",  # Use the trained persistent model
        use_ttt=False,    # Disable TTT for faster testing
        max_search_depth=5,
        beam_width=10
    )
    
    # Test on a few training tasks
    training_dir = Path("data/training")
    if not training_dir.exists():
        print("❌ Training data directory not found!")
        return False
    
    task_files = list(training_dir.glob("*.json"))[:5]  # Test first 5 tasks
    
    results = []
    total_accuracy = 0.0
    
    for i, task_file in enumerate(task_files):
        print(f"\n📋 Testing task {i+1}/{len(task_files)}: {task_file.name}")
        
        try:
            # Load task
            demo_pairs = load_arc_task(task_file)
            print(f"  Demo pairs: {len(demo_pairs)}")
            
            # Solve task
            print("  🔍 Solving...")
            solution = solver.solve(demo_pairs)
            
            # Evaluate solution
            accuracy = evaluate_solution(solution, demo_pairs)
            results.append({
                'task_id': task_file.stem,
                'accuracy': accuracy,
                'demo_pairs': len(demo_pairs),
                'solution_found': solution is not None
            })
            
            total_accuracy += accuracy
            
            print(f"  📊 Accuracy: {accuracy:.1%}")
            print(f"  ✅ Solution found: {solution is not None}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'task_id': task_file.stem,
                'accuracy': 0.0,
                'demo_pairs': 0,
                'solution_found': False,
                'error': str(e)
            })
    
    # Summary
    avg_accuracy = total_accuracy / len(task_files) if task_files else 0.0
    solutions_found = sum(1 for r in results if r['solution_found'])
    
    print(f"\n📊 SUMMARY:")
    print(f"  Tasks tested: {len(task_files)}")
    print(f"  Solutions found: {solutions_found}/{len(task_files)}")
    print(f"  Average accuracy: {avg_accuracy:.1%}")
    
    # Save results
    results_summary = {
        'total_tasks': len(task_files),
        'solutions_found': solutions_found,
        'average_accuracy': avg_accuracy,
        'results': results
    }
    
    with open('solver_test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"  📄 Results saved to solver_test_results.json")
    
    return avg_accuracy > 0.0

if __name__ == "__main__":
    success = test_solver_on_training_tasks()
    if success:
        print("\n🎉 Solver test completed successfully with non-zero accuracy!")
    else:
        print("\n💥 Solver test failed or achieved zero accuracy!")
        sys.exit(1) 