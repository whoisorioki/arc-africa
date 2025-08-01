#!/usr/bin/env python3
"""
Test TTT Integration with Neural Guide - Fixed Architecture
Uses the correct model architecture that matches our trained model.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class CompatibleNeuralGuide(nn.Module):
    """Neural guide with architecture matching our trained model."""
    
    def __init__(self, grid_size=48, embed_dim=128, max_colors=20, num_primitives=8):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.max_colors = max_colors
        self.num_primitives = num_primitives
        
        # Embeddings (matching trained model)
        self.input_embedding = nn.Embedding(max_colors + 1, embed_dim)
        self.output_embedding = nn.Embedding(max_colors + 1, embed_dim)
        
        # Transformer (matching trained model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection (matching trained model)
        self.output_proj = nn.Linear(embed_dim, num_primitives)
        
    def forward(self, input_grids, output_grids):
        batch_size = input_grids.shape[0]
        
        # Clamp color values to valid range
        input_grids = torch.clamp(input_grids, 0, self.max_colors)
        output_grids = torch.clamp(output_grids, 0, self.max_colors)
        
        # Flatten grids and embed
        input_flat = input_grids.view(batch_size, -1)
        output_flat = output_grids.view(batch_size, -1)
        
        # Embed
        input_emb = self.input_embedding(input_flat)
        output_emb = self.output_embedding(output_flat)
        
        # Concatenate and transform
        combined = torch.cat([input_emb, output_emb], dim=1)
        transformed = self.transformer(combined)
        pooled = torch.mean(transformed, dim=1)
        logits = self.output_proj(pooled)
        
        return logits

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
    
    # Create model with matching architecture
    model = CompatibleNeuralGuide(
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
    
    parser = argparse.ArgumentParser(description="Test TTT Integration - Fixed")
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