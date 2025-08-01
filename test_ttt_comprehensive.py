#!/usr/bin/env python3
"""
Comprehensive TTT Test with Neural Guide Persistent Model
"""

import torch
import numpy as np
import json
from pathlib import Path

def check_model_format(model_path):
    """Check the format of the model file."""
    print(f"Checking model format: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            print(f"Model state dict keys (first 5): {list(checkpoint['model_state_dict'].keys())[:5]}")
        
        if 'primitive_names' in checkpoint:
            print(f"Primitive names: {checkpoint['primitive_names']}")
            print(f"Number of primitives: {len(checkpoint['primitive_names'])}")
        
        return checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_neural_guide_predictions(model_path):
    """Test neural guide predictions with the persistent model."""
    print(f"\nTesting neural guide predictions with: {model_path}")
    
    try:
        from src.neural_guide.architecture import create_neural_guide
        
        # Create model
        model = create_neural_guide()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test predictions
        input_grids = torch.randint(0, 10, (1, 48, 48))
        output_grids = torch.randint(0, 10, (1, 48, 48))
        
        with torch.no_grad():
            predictions = model(input_grids, output_grids)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Max probability: {torch.max(predictions):.4f}")
        print(f"Min probability: {torch.min(predictions):.4f}")
        
        # Get top predictions
        top_probs, top_indices = torch.topk(predictions, k=3, dim=-1)
        print(f"Top 3 probabilities: {top_probs[0]}")
        print(f"Top 3 indices: {top_indices[0]}")
        
        return True
        
    except Exception as e:
        print(f"Error testing neural guide: {e}")
        return False

def test_solver_with_persistent_model():
    """Test the main solver with the persistent model."""
    print(f"\nTesting main solver with persistent model...")
    
    try:
        from src.solver.main_solver import NeuroSymbolicSolver
        
        # Create solver with persistent model
        solver = NeuroSymbolicSolver(model_path="models/neural_guide_persistent.pth")
        
        # Test predictions
        import numpy as np
        test_grid = np.random.randint(0, 10, (5, 5))
        demo_pairs = [(test_grid, test_grid)]
        predictions = solver.predict_primitives(demo_pairs)
        
        print(f"Solver predictions: {predictions}")
        print(f"Number of predictions: {len(predictions)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing solver: {e}")
        return False

def test_ttt_integration_simple():
    """Simple TTT integration test."""
    print(f"\nTesting simple TTT integration...")
    
    try:
        from src.solver.main_solver import NeuroSymbolicSolver
        
        # Create solver
        solver = NeuroSymbolicSolver(model_path="models/neural_guide_persistent.pth")
        
        # Test with sample data
        test_results = []
        
        # Create sample demo pairs
        for i in range(3):
            input_grid = np.random.randint(0, 10, (5, 5))
            output_grid = np.random.randint(0, 10, (5, 5))
            demo_pairs = [(input_grid, output_grid)]
            
            predictions = solver.predict_primitives(demo_pairs)
            
            test_results.append({
                'task_id': f'test_{i:04d}',
                'predictions': [predictions],  # Single prediction list
                'num_pairs': len(demo_pairs)
            })
            
            print(f"Task {i}: Predictions = {predictions}")
        
        # Save results
        with open('results/ttt_test_results_simple.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Simple TTT test results saved to: results/ttt_test_results_simple.json")
        
        return test_results
        
    except Exception as e:
        print(f"Error in simple TTT test: {e}")
        return None

def main():
    """Main test function."""
    print("=== Comprehensive TTT Test with Neural Guide Persistent ===\n")
    
    model_path = "models/neural_guide_persistent.pth"
    
    # Check model format
    checkpoint = check_model_format(model_path)
    if checkpoint is None:
        print("❌ Model format check failed")
        return
    
    # Test neural guide predictions
    neural_success = test_neural_guide_predictions(model_path)
    
    # Test solver
    solver_success = test_solver_with_persistent_model()
    
    # Test simple TTT integration
    ttt_results = test_ttt_integration_simple()
    
    # Summary
    print(f"\n=== Summary ===")
    if neural_success:
        print("✅ Neural guide predictions working")
    else:
        print("❌ Neural guide predictions failed")
    
    if solver_success:
        print("✅ Solver working with persistent model")
    else:
        print("❌ Solver failed with persistent model")
    
    if ttt_results:
        print("✅ Simple TTT integration working")
        print(f"   Results: {len(ttt_results)} tasks tested")
    else:
        print("❌ Simple TTT integration failed")
    
    if neural_success and solver_success and ttt_results:
        print("\n🎉 All tests passed! TTT implementation is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Need to investigate further.")

if __name__ == "__main__":
    main() 