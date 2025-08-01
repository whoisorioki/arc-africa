#!/usr/bin/env python3
"""
Test TTT Integration with Neural Guide
Tests how well our trained neural guide works with the TTT system.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neural_guide.architecture import create_neural_guide
from solver.main_solver import NeuroSymbolicSolver

def test_ttt_integration(
    model_path: str,
    test_tasks_dir: str,
    output_path: str,
    num_test_tasks: int = 5
):
    """Test TTT integration with neural guide."""
    
    print(f"Testing TTT Integration")
    print(f"Model: {model_path}")
    print(f"Test tasks: {test_tasks_dir}")
    
    # Load the trained neural guide
    print("Loading neural guide model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with same architecture
    model = create_neural_guide(
        grid_size=48,
        embed_dim=128,
        max_colors=20,
        num_primitives=len(checkpoint['primitive_names'])
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with {len(checkpoint['primitive_names'])} primitives")
    print(f"Primitives: {checkpoint['primitive_names']}")
    
    # Test on a few ARC tasks
    test_results = []
    
    # Get test task files
    test_files = list(Path(test_tasks_dir).glob("*.json"))[:num_test_tasks]
    
    for task_file in test_files:
        print(f"\nTesting task: {task_file.name}")
        
        try:
            # Load task
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            # Extract demonstration pairs
            train_pairs = task_data.get('train', [])
            
            if not train_pairs:
                print(f"No training pairs found in {task_file.name}")
                continue
            
            # Test neural guide predictions
            predictions = []
            for pair in train_pairs[:2]:  # Test first 2 pairs
                input_grid = np.array(pair['input'])
                output_grid = np.array(pair['output'])
                
                # Preprocess grids
                input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)
                output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0)
                
                # Get predictions
                with torch.no_grad():
                    logits = model(input_tensor, output_tensor)
                    probs = torch.sigmoid(logits)
                    predicted_primitives = [
                        checkpoint['primitive_names'][i] 
                        for i, p in enumerate(probs[0]) 
                        if p > 0.5
                    ]
                
                predictions.append(predicted_primitives)
            
            # Store results
            test_results.append({
                'task_id': task_file.stem,
                'predictions': predictions,
                'num_pairs': len(train_pairs)
            })
            
            print(f"Predictions: {predictions}")
            
        except Exception as e:
            print(f"Error testing {task_file.name}: {e}")
            test_results.append({
                'task_id': task_file.stem,
                'error': str(e)
            })
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {output_path}")
    return test_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TTT Integration")
    parser.add_argument("--model", required=True, help="Path to neural guide model")
    parser.add_argument("--test_tasks", required=True, help="Directory with test tasks")
    parser.add_argument("--output", required=True, help="Output results file")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of test tasks")
    
    args = parser.parse_args()
    
    test_ttt_integration(
        model_path=args.model,
        test_tasks_dir=args.test_tasks,
        output_path=args.output,
        num_test_tasks=args.num_tasks
    )

if __name__ == "__main__":
    main()
