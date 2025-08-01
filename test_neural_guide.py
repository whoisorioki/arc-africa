#!/usr/bin/env python3
"""
Simple test script to debug neural guide loading and inference.
"""

import torch
import numpy as np
import json
from pathlib import Path

def test_basic_imports():
    """Test basic imports without scipy dependency."""
    print("Testing basic imports...")
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  Version: {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_neural_guide_creation():
    """Test creating neural guide model."""
    print("\nTesting neural guide creation...")
    try:
        from src.neural_guide.architecture import create_neural_guide
        model = create_neural_guide()
        print("✓ Neural guide created successfully")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        return model
    except Exception as e:
        print(f"✗ Neural guide creation failed: {e}")
        return None

def test_model_inference(model):
    """Test basic model inference."""
    print("\nTesting model inference...")
    try:
        model.eval()
        
        # Create dummy input
        batch_size = 1
        num_pairs = 2
        grid_size = 48
        max_colors = 10
        
        input_grid = torch.randint(0, max_colors, (batch_size, num_pairs, grid_size, grid_size))
        output_grid = torch.randint(0, max_colors, (batch_size, num_pairs, grid_size, grid_size))
        
        print(f"  Input shape: {input_grid.shape}")
        print(f"  Output shape: {output_grid.shape}")
        
        with torch.no_grad():
            predictions = model(input_grid, output_grid)
            
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Predictions: {predictions}")
        print(f"  Max probability: {torch.max(predictions):.4f}")
        print(f"  Min probability: {torch.min(predictions):.4f}")
        print(f"  Mean probability: {torch.mean(predictions):.4f}")
        
        # Check if predictions are reasonable
        if torch.max(predictions) > 0:
            print("✓ Model produces non-zero predictions")
            return True
        else:
            print("✗ Model produces only zero predictions")
            return False
            
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        return False

def test_model_loading():
    """Test loading pre-trained model."""
    print("\nTesting model loading...")
    model_paths = [
        "models/neural_guide_best.pth",
        "models/neural_guide_persistent.pth",
        "models/neural_guide_test_100.pth"
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"  Found model: {model_path}")
            try:
                model = create_neural_guide()
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Successfully loaded {model_path}")
                return model
            except Exception as e:
                print(f"✗ Failed to load {model_path}: {e}")
    
    print("✗ No models could be loaded")
    return None

def analyze_ttt_results():
    """Analyze the TTT test results."""
    print("\nAnalyzing TTT results...")
    try:
        with open("results/ttt_test_results.json", 'r') as f:
            results = json.load(f)
        
        total_tasks = len(results)
        tasks_with_predictions = sum(1 for r in results if any(r['predictions']))
        
        print(f"  Total tasks: {total_tasks}")
        print(f"  Tasks with predictions: {tasks_with_predictions}")
        print(f"  Success rate: {tasks_with_predictions/total_tasks*100:.1f}%")
        
        # Extract all predictions
        all_predictions = []
        for result in results:
            for pred_list in result['predictions']:
                all_predictions.extend(pred_list)
        
        if all_predictions:
            unique_predictions = set(all_predictions)
            print(f"  Unique primitives: {unique_predictions}")
            print(f"  Total predictions: {len(all_predictions)}")
        else:
            print("  No predictions made at all!")
            
    except Exception as e:
        print(f"✗ Failed to analyze TTT results: {e}")

def main():
    """Main test function."""
    print("=== Neural Guide Debug Test ===\n")
    
    # Test basic imports
    if not test_basic_imports():
        print("Basic imports failed. Exiting.")
        return
    
    # Test model creation
    model = test_neural_guide_creation()
    if model is None:
        print("Model creation failed. Exiting.")
        return
    
    # Test model inference
    inference_success = test_model_inference(model)
    
    # Test model loading
    loaded_model = test_model_loading()
    
    # Analyze TTT results
    analyze_ttt_results()
    
    # Summary
    print("\n=== Summary ===")
    if inference_success:
        print("✓ Basic model inference works")
    else:
        print("✗ Model inference has issues")
    
    if loaded_model is not None:
        print("✓ Model loading works")
    else:
        print("✗ Model loading has issues")
    
    print("\nNext steps:")
    if not inference_success:
        print("1. Debug model architecture")
        print("2. Check input preprocessing")
    if loaded_model is None:
        print("3. Fix model loading issues")
    print("4. Debug TTT adaptation process")

if __name__ == "__main__":
    main() 