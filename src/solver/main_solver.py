"""Main neuro-symbolic solver for ARC tasks."""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_guide.architecture import create_neural_guide
from dsl.primitives import PRIMITIVE_FUNCTIONS, PRIMITIVE_NAMES
from data_pipeline.augmentation import augment_demonstrations

class NeuroSymbolicSolver:
    """Neuro-symbolic solver that combines neural guide with symbolic search."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the solver.
        
        Args:
            model_path: Path to pre-trained neural guide model
        """
        self.neural_guide = create_neural_guide()
        self.primitive_names = PRIMITIVE_NAMES
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load pre-trained neural guide model.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.neural_guide.load_state_dict(checkpoint['model_state_dict'])
            self.neural_guide.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model from {model_path}: {e}")
            print("Using untrained model")
    
    def predict_primitives(
        self, 
        input_grids: List[np.ndarray], 
        output_grids: List[np.ndarray],
        top_k: int = 5
    ) -> List[str]:
        """Predict most likely primitives using neural guide.
        
        Args:
            input_grids: List of input grids
            output_grids: List of output grids
            top_k: Number of top predictions to return
            
        Returns:
            List of predicted primitive names
        """
        # Ensure all grids have the same shape by padding
        max_h = max(grid.shape[0] for grid in input_grids + output_grids)
        max_w = max(grid.shape[1] for grid in input_grids + output_grids)
        
        # Pad all grids to the same size
        padded_inputs = []
        padded_outputs = []
        
        for grid in input_grids:
            padded = np.zeros((max_h, max_w), dtype=grid.dtype)
            padded[:grid.shape[0], :grid.shape[1]] = grid
            padded_inputs.append(padded)
        
        for grid in output_grids:
            padded = np.zeros((max_h, max_w), dtype=grid.dtype)
            padded[:grid.shape[0], :grid.shape[1]] = grid
            padded_outputs.append(padded)
        
        # Convert to tensors
        input_tensor = torch.tensor(np.array(padded_inputs), dtype=torch.long)
        output_tensor = torch.tensor(np.array(padded_outputs), dtype=torch.long)
        
        # Get predictions
        with torch.no_grad():
            logits = self.neural_guide(input_tensor, output_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # Average across batch
            avg_probs = torch.mean(probs, dim=0)
            
            # Get top-k predictions (ensure we don't exceed available primitives)
            top_k = min(top_k, len(self.primitive_names))
            top_indices = torch.topk(avg_probs, top_k).indices
            
            # Filter indices to valid range
            valid_indices = [int(i.item()) for i in top_indices if i.item() < len(self.primitive_names)]
            predicted_primitives = [self.primitive_names[i] for i in valid_indices]
        
        return predicted_primitives
    
    def solve_task(
        self, 
        input_grids: List[np.ndarray], 
        output_grids: List[np.ndarray]
    ) -> List[str]:
        """Solve an ARC task using TTT approach.
        
        Args:
            input_grids: List of input grids
            output_grids: List of output grids
            
        Returns:
            List of predicted primitive names
        """
        # Step 1: Augment demonstration pairs
        demo_pairs = list(zip(input_grids, output_grids))
        aug_pairs = augment_demonstrations(demo_pairs)
        aug_inputs = [pair[0] for pair in aug_pairs]
        aug_outputs = [pair[1] for pair in aug_pairs]
        
        # Step 2: Predict primitives using neural guide
        predicted_primitives = self.predict_primitives(
            aug_inputs, aug_outputs, top_k=5
        )
        
        return predicted_primitives

def create_test_task() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create a test task for validation.
    
    Returns:
        Tuple of (input_grids, output_grids)
    """
    # Create a simple test task
    input_grid = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    # Import here to avoid circular imports
    from dsl.primitives import rotate90
    output_grid = rotate90(input_grid)
    
    return [input_grid], [output_grid]
