#!/usr/bin/env python3
"""
Test script that correctly calls the neural guide model with proper tensor shapes.
"""

import torch
import numpy as np

def test_corrected_inference():
    """Test model inference with the correct tensor shapes as expected by the model."""
    print("Testing corrected model inference...")
    
    try:
        from src.neural_guide.architecture import create_neural_guide
        model = create_neural_guide()
        model.eval()
        
        # Create inputs with CORRECT shapes as expected by the model
        batch_size = 1
        grid_size = 48
        max_colors = 10
        
        # The model expects:
        # input_grids: (batch_size, grid_size, grid_size)
        # output_grids: (batch_size, grid_size, grid_size)
        input_grids = torch.randint(0, max_colors, (batch_size, grid_size, grid_size))
        output_grids = torch.randint(0, max_colors, (batch_size, grid_size, grid_size))
        
        print(f"  Input grids shape: {input_grids.shape}")
        print(f"  Output grids shape: {output_grids.shape}")
        print(f"  Expected: (1, 48, 48)")
        
        with torch.no_grad():
            predictions = model(input_grids, output_grids)
            
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Predictions: {predictions}")
        print(f"  Max probability: {torch.max(predictions):.4f}")
        print(f"  Min probability: {torch.min(predictions):.4f}")
        print(f"  Mean probability: {torch.mean(predictions):.4f}")
        
        # Get top predictions
        top_probs, top_indices = torch.topk(predictions, k=3, dim=-1)
        print(f"  Top 3 probabilities: {top_probs[0]}")
        print(f"  Top 3 indices: {top_indices[0]}")
        
        if torch.max(predictions) > 0:
            print("✓ Model produces non-zero predictions")
            return True
        else:
            print("✗ Model produces only zero predictions")
            return False
            
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        return False

def test_loaded_model_inference():
    """Test loaded model inference with correct shapes."""
    print("\nTesting loaded model inference...")
    
    try:
        from src.neural_guide.architecture import create_neural_guide
        model = create_neural_guide()
        
        # Load the model
        checkpoint = torch.load('models/neural_guide_persistent.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create inputs with correct shapes
        batch_size = 1
        grid_size = 48
        max_colors = 10
        
        input_grids = torch.randint(0, max_colors, (batch_size, grid_size, grid_size))
        output_grids = torch.randint(0, max_colors, (batch_size, grid_size, grid_size))
        
        with torch.no_grad():
            predictions = model(input_grids, output_grids)
            
        print(f"  Loaded model predictions shape: {predictions.shape}")
        print(f"  Max probability: {torch.max(predictions):.4f}")
        print(f"  Min probability: {torch.min(predictions):.4f}")
        
        # Get top predictions
        top_probs, top_indices = torch.topk(predictions, k=3, dim=-1)
        print(f"  Top 3 probabilities: {top_probs[0]}")
        print(f"  Top 3 indices: {top_indices[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loaded model inference failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== Corrected Neural Guide Inference Test ===\n")
    
    # Test basic model inference
    inference_success = test_corrected_inference()
    
    # Test loaded model inference
    loaded_success = test_loaded_model_inference()
    
    # Summary
    print("\n=== Summary ===")
    if inference_success:
        print("✓ Basic model inference works with correct shapes")
    else:
        print("✗ Basic model inference has issues")
    
    if loaded_success:
        print("✓ Loaded model inference works")
    else:
        print("✗ Loaded model inference has issues")
    
    if inference_success and loaded_success:
        print("\n🎉 All tests passed! The tensor shape issue is resolved.")
        print("The problem was that the model expects separate input_grids and output_grids")
        print("tensors of shape (batch_size, grid_size, grid_size), not combined tensors.")
    else:
        print("\n❌ Some tests failed. Need to investigate further.")

if __name__ == "__main__":
    main() 