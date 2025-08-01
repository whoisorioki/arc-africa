#!/usr/bin/env python3
"""Debug script to check color values and embedding configuration."""

import json
import numpy as np
import torch
from src.neural_guide.architecture import create_neural_guide

def pad_to_shape(grid, shape=(48, 48), pad_value=0):
    """Pad a grid to the target shape."""
    h, w = grid.shape
    target_h, target_w = shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return torch.nn.functional.pad(
        grid, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value
    )

def debug_colors():
    """Debug color values in the dataset and model configuration."""
    
    # Load dataset and check colors
    print("🔍 Loading dataset...")
    with open("data/synthetic/enhanced_synthetic_dataset_v2.json", "r") as f:
        data = json.load(f)
    
    print(f"📊 Total samples: {len(data)}")
    
    # Check first few samples for color ranges
    all_colors = set()
    for i, sample in enumerate(data[:100]):  # Check first 100 samples
        input_grid = np.array(sample["input"])
        output_grid = np.array(sample["output"])
        
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        
        all_colors.update(input_colors)
        all_colors.update(output_colors)
        
        if i < 5:  # Show first 5 samples
            print(f"Sample {i}: Input colors: {sorted(input_colors)}, Output colors: {sorted(output_colors)}")
            print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
    
    print(f"🎨 All unique colors found: {sorted(all_colors)}")
    print(f"🎨 Min color: {min(all_colors)}, Max color: {max(all_colors)}")
    print(f"🎨 Color range: {max(all_colors) - min(all_colors) + 1} colors")
    
    # Check model configuration
    print("\n🔧 Checking model configuration...")
    model = create_neural_guide(
        grid_size=48,
        embed_dim=256,
        max_colors=9,  # Current setting
        num_primitives=15,
    )
    
    print(f"Model max_colors setting: 9")
    print(f"Embedding layer size: {model.grid_embedding.color_embedding.num_embeddings}")
    print(f"Expected embedding size: 10 (max_colors + 1)")
    
    # Test with actual data (with padding)
    print("\n🧪 Testing with actual data (padded to 48x48)...")
    sample = data[0]
    input_grid = torch.tensor(sample["input"], dtype=torch.long)
    output_grid = torch.tensor(sample["output"], dtype=torch.long)
    
    print(f"Original input grid shape: {input_grid.shape}")
    print(f"Original output grid shape: {output_grid.shape}")
    print(f"Input grid unique values: {torch.unique(input_grid).tolist()}")
    print(f"Output grid unique values: {torch.unique(output_grid).tolist()}")
    
    # Pad grids to 48x48
    input_grid_padded = pad_to_shape(input_grid, (48, 48))
    output_grid_padded = pad_to_shape(output_grid, (48, 48))
    
    print(f"Padded input grid shape: {input_grid_padded.shape}")
    print(f"Padded output grid shape: {output_grid_padded.shape}")
    
    # Check if any values exceed max_colors
    max_input = torch.max(input_grid_padded).item()
    max_output = torch.max(output_grid_padded).item()
    
    print(f"Max input value: {max_input}")
    print(f"Max output value: {max_output}")
    print(f"Max allowed by model: 9")
    
    if max_input > 9 or max_output > 9:
        print("❌ ERROR: Found values > 9!")
        print(f"   Input max: {max_input} > 9: {'YES' if max_input > 9 else 'NO'}")
        print(f"   Output max: {max_output} > 9: {'YES' if max_output > 9 else 'NO'}")
    else:
        print("✅ All values within range!")
    
    # Try to run the model
    print("\n🚀 Testing model forward pass...")
    try:
        with torch.no_grad():
            predictions = model(input_grid_padded.unsqueeze(0), output_grid_padded.unsqueeze(0))
        print("✅ Model forward pass successful!")
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    debug_colors() 