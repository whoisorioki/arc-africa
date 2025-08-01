#!/usr/bin/env python3
"""
Analyze Neural Guide Performance
Comprehensive analysis of the trained neural guide model.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

def analyze_model_checkpoint(model_path: str):
    """Analyze the model checkpoint structure and content."""
    print(f"🔍 Analyzing model checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\n📋 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Analyze model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n🏗️ Model layers:")
        for key, tensor in state_dict.items():
            print(f"  {key}: {tensor.shape} - {tensor.dtype}")
    
    # Analyze primitive names
    if 'primitive_names' in checkpoint:
        primitives = checkpoint['primitive_names']
        print(f"\n🎯 Primitives ({len(primitives)}): {primitives}")
    
    # Analyze training history
    if 'training_history' in checkpoint:
        history = checkpoint['training_history']
        print(f"\n📈 Training history keys: {list(history.keys())}")
        if 'losses' in history:
            losses = history['losses']
            print(f"  Losses: {len(losses)} epochs")
            print(f"  Final loss: {losses[-1] if losses else 'N/A'}")
            print(f"  Loss trend: {losses[:5]}...{losses[-5:] if len(losses) > 5 else ''}")
    
    return checkpoint

def test_model_on_synthetic_data(model_path: str, synthetic_data_path: str):
    """Test the model on synthetic training data to verify it works."""
    print(f"\n🧪 Testing model on synthetic data: {synthetic_data_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = CompatibleNeuralGuide(
        grid_size=48,
        embed_dim=128,
        max_colors=20,
        num_primitives=len(checkpoint['primitive_names'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load synthetic data
    with open(synthetic_data_path, 'r') as f:
        synthetic_data = json.load(f)
    
    print(f"📊 Synthetic dataset size: {len(synthetic_data)}")
    
    # Test on a few samples
    test_samples = synthetic_data[:10]
    predictions = []
    
    for i, sample in enumerate(test_samples):
        input_grid = np.array(sample['input'])
        output_grid = np.array(sample['output'])
        
        # Preprocess
        input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)
        output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            logits = model(input_tensor, output_tensor)
            probs = torch.sigmoid(logits)
            
            # Get top predictions
            top_indices = torch.topk(probs[0], k=3).indices
            top_primitives = [checkpoint['primitive_names'][idx] for idx in top_indices]
            top_probs = [probs[0][idx].item() for idx in top_indices]
            
            predictions.append({
                'sample_id': i,
                'true_primitives': sample.get('primitives', []),
                'predicted_primitives': top_primitives,
                'predicted_probs': top_probs,
                'max_prob': probs[0].max().item()
            })
    
    # Analyze predictions
    print(f"\n📊 Prediction Analysis:")
    max_probs = [p['max_prob'] for p in predictions]
    print(f"  Max probability range: {min(max_probs):.3f} - {max(max_probs):.3f}")
    print(f"  Average max probability: {np.mean(max_probs):.3f}")
    
    # Check if predictions match true primitives
    matches = 0
    for pred in predictions:
        true_set = set(pred['true_primitives'])
        pred_set = set(pred['predicted_primitives'])
        if true_set & pred_set:  # Intersection
            matches += 1
    
    print(f"  Prediction accuracy: {matches}/{len(predictions)} ({matches/len(predictions)*100:.1f}%)")
    
    return predictions

def analyze_arc_task_predictions(model_path: str, test_tasks_dir: str):
    """Analyze predictions on actual ARC tasks."""
    print(f"\n🎯 Analyzing ARC task predictions")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = CompatibleNeuralGuide(
        grid_size=48,
        embed_dim=128,
        max_colors=20,
        num_primitives=len(checkpoint['primitive_names'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get test files
    test_files = list(Path(test_tasks_dir).glob("*.json"))[:10]
    
    all_predictions = []
    task_analysis = []
    
    for task_file in test_files:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        train_pairs = task_data.get('train', [])
        if not train_pairs:
            continue
        
        task_predictions = []
        for pair in train_pairs[:2]:  # Test first 2 pairs
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # Preprocess
            input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)
            output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                logits = model(input_tensor, output_tensor)
                probs = torch.sigmoid(logits)
                
                # Get all predictions above threshold
                threshold = 0.1
                predicted_primitives = [
                    checkpoint['primitive_names'][i] 
                    for i, p in enumerate(probs[0]) 
                    if p > threshold
                ]
                
                task_predictions.append({
                    'input_shape': input_grid.shape,
                    'output_shape': output_grid.shape,
                    'predicted_primitives': predicted_primitives,
                    'max_prob': probs[0].max().item(),
                    'probabilities': probs[0].tolist()
                })
        
        task_analysis.append({
            'task_id': task_file.stem,
            'num_pairs': len(train_pairs),
            'predictions': task_predictions
        })
        
        all_predictions.extend([p['predicted_primitives'] for p in task_predictions])
    
    # Analyze results
    print(f"\n📊 ARC Task Analysis:")
    print(f"  Tasks analyzed: {len(task_analysis)}")
    
    # Count predictions
    flat_predictions = [p for preds in all_predictions for p in preds]
    if flat_predictions:
        prediction_counts = Counter(flat_predictions)
        print(f"  Most common predictions:")
        for primitive, count in prediction_counts.most_common(5):
            print(f"    {primitive}: {count}")
    
    # Analyze probability distributions
    all_max_probs = []
    for task in task_analysis:
        for pred in task['predictions']:
            all_max_probs.append(pred['max_prob'])
    
    if all_max_probs:
        print(f"  Max probability stats:")
        print(f"    Min: {min(all_max_probs):.3f}")
        print(f"    Max: {max(all_max_probs):.3f}")
        print(f"    Mean: {np.mean(all_max_probs):.3f}")
        print(f"    Median: {np.median(all_max_probs):.3f}")
    
    return task_analysis

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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Neural Guide Performance")
    parser.add_argument("--model", required=True, help="Path to neural guide model")
    parser.add_argument("--synthetic_data", help="Path to synthetic dataset")
    parser.add_argument("--test_tasks", help="Directory with test tasks")
    parser.add_argument("--output", required=True, help="Output analysis file")
    
    args = parser.parse_args()
    
    analysis_results = {}
    
    # Analyze model checkpoint
    checkpoint = analyze_model_checkpoint(args.model)
    analysis_results['checkpoint_analysis'] = {
        'num_primitives': len(checkpoint.get('primitive_names', [])),
        'primitive_names': checkpoint.get('primitive_names', []),
        'has_training_history': 'training_history' in checkpoint
    }
    
    # Test on synthetic data if available
    if args.synthetic_data and os.path.exists(args.synthetic_data):
        synthetic_predictions = test_model_on_synthetic_data(args.model, args.synthetic_data)
        analysis_results['synthetic_analysis'] = {
            'num_samples_tested': len(synthetic_predictions),
            'max_prob_range': [min(p['max_prob'] for p in synthetic_predictions), 
                             max(p['max_prob'] for p in synthetic_predictions)],
            'avg_max_prob': np.mean([p['max_prob'] for p in synthetic_predictions])
        }
    
    # Analyze ARC task predictions if available
    if args.test_tasks and os.path.exists(args.test_tasks):
        arc_analysis = analyze_arc_task_predictions(args.model, args.test_tasks)
        analysis_results['arc_analysis'] = {
            'num_tasks_analyzed': len(arc_analysis),
            'total_predictions': sum(len(task['predictions']) for task in arc_analysis)
        }
    
    # Save analysis
    with open(args.output, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {args.output}")

if __name__ == "__main__":
    main() 