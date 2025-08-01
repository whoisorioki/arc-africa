#!/usr/bin/env python3
"""
Test TTT Integration with Neural Guide - Debug Version
Shows detailed prediction information and uses lower thresholds.
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

def test_ttt_integration_debug(
    model_path: str,
    test_tasks_dir: str,
    output_path: str,
    num_test_tasks: int = 5
):
    """Test TTT integration with detailed debugging."""
    
    print(f"🔍 Testing TTT Integration - Debug Mode")
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
    
    print(f"✅ Model loaded with {len(checkpoint['primitive_names'])} primitives")
    print(f"📋 Primitives: {checkpoint['primitive_names']}")
    
    # Test on a few ARC tasks
    test_results = []
    
    # Get test task files
    test_files = list(Path(test_tasks_dir).glob("*.json"))[:num_test_tasks]
    print(f"📁 Found {len(test_files)} test files")
    
    for task_file in test_files:
        print(f"\n{'='*60}")
        print(f"🧪 Testing task: {task_file.name}")
        print(f"{'='*60}")
        
        try:
            # Load task
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            # Extract demonstration pairs
            train_pairs = task_data.get('train', [])
            
            if not train_pairs:
                print(f"❌ No training pairs found in {task_file.name}")
                continue
            
            print(f"📊 Found {len(train_pairs)} training pairs")
            
            # Test neural guide predictions
            predictions = []
            all_probs = []
            
            for pair_idx, pair in enumerate(train_pairs[:3]):  # Test first 3 pairs
                print(f"\n--- Pair {pair_idx + 1} ---")
                
                input_grid = np.array(pair['input'])
                output_grid = np.array(pair['output'])
                
                print(f"Input shape: {input_grid.shape}, colors: {np.unique(input_grid)}")
                print(f"Output shape: {output_grid.shape}, colors: {np.unique(output_grid)}")
                
                # Preprocess grids
                input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)
                output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0)
                
                # Get predictions
                with torch.no_grad():
                    logits = model(input_tensor, output_tensor)
                    probs = torch.sigmoid(logits)
                    
                    # Show all probabilities
                    print(f"Raw logits: {logits[0].tolist()}")
                    print(f"Probabilities: {probs[0].tolist()}")
                    
                    # Try different thresholds
                    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
                    for threshold in thresholds:
                        predicted_primitives = [
                            checkpoint['primitive_names'][i] 
                            for i, p in enumerate(probs[0]) 
                            if p > threshold
                        ]
                        print(f"Threshold {threshold}: {predicted_primitives}")
                    
                    # Use lower threshold for final prediction
                    predicted_primitives = [
                        checkpoint['primitive_names'][i] 
                        for i, p in enumerate(probs[0]) 
                        if p > 0.2  # Lower threshold
                    ]
                
                predictions.append(predicted_primitives)
                all_probs.append(probs[0].tolist())
            
            # Store results
            test_results.append({
                'task_id': task_file.stem,
                'predictions': predictions,
                'all_probabilities': all_probs,
                'num_pairs': len(train_pairs),
                'primitive_names': checkpoint['primitive_names']
            })
            
            print(f"\n🎯 Final predictions: {predictions}")
            
        except Exception as e:
            print(f"❌ Error testing {task_file.name}: {e}")
            import traceback
            traceback.print_exc()
            test_results.append({
                'task_id': task_file.stem,
                'error': str(e)
            })
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {output_path}")
    
    # Summary
    print(f"\n📈 Summary:")
    successful_tasks = [r for r in test_results if 'predictions' in r]
    print(f"Successful tasks: {len(successful_tasks)}/{len(test_results)}")
    
    for result in successful_tasks:
        non_empty_predictions = [p for p in result['predictions'] if p]
        print(f"  {result['task_id']}: {len(non_empty_predictions)}/{len(result['predictions'])} pairs with predictions")
    
    return test_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TTT Integration - Debug")
    parser.add_argument("--model", required=True, help="Path to neural guide model")
    parser.add_argument("--test_tasks", required=True, help="Directory with test tasks")
    parser.add_argument("--output", required=True, help="Output results file")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of test tasks")
    
    args = parser.parse_args()
    
    test_ttt_integration_debug(
        model_path=args.model,
        test_tasks_dir=args.test_tasks,
        output_path=args.output,
        num_test_tasks=args.num_tasks
    )

if __name__ == "__main__":
    main() 