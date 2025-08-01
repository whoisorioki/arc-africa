#!/usr/bin/env python3
"""
Enhanced Neuro-Symbolic Solver with Neural Guide Integration
Uses our trained neural guide to improve symbolic search.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

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

class EnhancedNeuroSymbolicSolver:
    """Enhanced solver with neural guide integration."""
    
    def __init__(self, neural_guide_path: str):
        # Load neural guide
        self.neural_guide = self._load_neural_guide(neural_guide_path)
        
        # Initialize primitive functions (simplified for now)
        self.primitive_functions = self._get_primitive_functions()
        
    def _load_neural_guide(self, model_path: str):
        """Load the trained neural guide."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = CompatibleNeuralGuide(
            grid_size=48,
            embed_dim=128,
            max_colors=20,
            num_primitives=len(checkpoint['primitive_names'])
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.primitive_names = checkpoint['primitive_names']
        return model
    
    def _get_primitive_functions(self):
        """Get available primitive functions."""
        # This would integrate with your existing DSL primitives
        return {
            'rotate90': lambda x: np.rot90(x, k=1),
            'rotate180': lambda x: np.rot90(x, k=2),
            'rotate270': lambda x: np.rot90(x, k=3),
            'horizontal_mirror': lambda x: np.fliplr(x),
            'vertical_mirror': lambda x: np.flipud(x),
            'fill': lambda x, color=1: np.full_like(x, color),
            'colorfilter': lambda x, color=1: np.where(x == color, x, 0),
            'remove': lambda x, color=1: np.where(x == color, 0, x),
            'crop': lambda x: x[1:-1, 1:-1] if x.shape[0] > 2 and x.shape[1] > 2 else x,
        }
    
    def predict_primitives(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[str]:
        """Predict relevant primitives using neural guide."""
        # Preprocess grids
        input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)
        output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            logits = self.neural_guide(input_tensor, output_tensor)
            probs = torch.sigmoid(logits)
            
            # Get top primitives
            predicted_primitives = [
                self.primitive_names[i] 
                for i, p in enumerate(probs[0]) 
                if p > 0.3  # Lower threshold for more candidates
            ]
        
        return predicted_primitives
    
    def solve_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve an ARC task using neural guide + symbolic search."""
        
        train_pairs = task_data.get('train', [])
        test_pairs = task_data.get('test', [])
        
        if not train_pairs:
            return {'error': 'No training pairs found'}
        
        # Get neural guide predictions for all training pairs
        all_predicted_primitives = []
        for pair in train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            predicted = self.predict_primitives(input_grid, output_grid)
            all_predicted_primitives.extend(predicted)
        
        # Get most common predictions
        primitive_counts = Counter(all_predicted_primitives)
        top_primitives = [p for p, c in primitive_counts.most_common(8)]
        
        print(f"Neural guide predictions: {top_primitives}")
        
        # Simple symbolic search with guided primitives
        solutions = self._symbolic_search(train_pairs, top_primitives)
        
        return {
            'predicted_primitives': top_primitives,
            'primitive_counts': dict(primitive_counts),
            'solutions': solutions,
            'num_train_pairs': len(train_pairs),
            'num_test_pairs': len(test_pairs)
        }
    
    def _symbolic_search(self, train_pairs: List[Dict], primitives: List[str]) -> List[Dict]:
        """Simple symbolic search using neural guide predictions."""
        solutions = []
        
        for pair in train_pairs:
            input_grid = np.array(pair['input'])
            expected_output = np.array(pair['output'])
            
            # Try each predicted primitive
            for primitive in primitives:
                if primitive in self.primitive_functions:
                    try:
                        func = self.primitive_functions[primitive]
                        result = func(input_grid)
                        
                        # Check if this primitive works
                        if np.array_equal(result, expected_output):
                            solutions.append({
                                'primitive': primitive,
                                'input_shape': input_grid.shape,
                                'output_shape': result.shape,
                                'success': True
                            })
                            break
                    except Exception as e:
                        continue
        
        return solutions

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Neuro-Symbolic Solver")
    parser.add_argument("--model", required=True, help="Path to neural guide model")
    parser.add_argument("--task", required=True, help="Path to task file")
    parser.add_argument("--output", required=True, help="Output results file")
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = EnhancedNeuroSymbolicSolver(args.model)
    
    # Load task
    with open(args.task, 'r') as f:
        task_data = json.load(f)
    
    # Solve task
    result = solver.solve_task(task_data)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to: {args.output}")
    print(f"Predicted primitives: {result.get('predicted_primitives', [])}")
    print(f"Solutions found: {len(result.get('solutions', []))}")

if __name__ == "__main__":
    main() 