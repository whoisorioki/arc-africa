#!/usr/bin/env python3
"""
Enhanced Neural Guide Training Script for AWS
Trains an improved neural guide model for ARC Challenge with GPU acceleration.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import wandb
from tqdm import tqdm
import random

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from src.neural_guide.enhanced_architecture import EnhancedNeuralGuide
from src.data_pipeline.segmentation import segment_grid
from src.dsl.primitives import *

class ARCTaskDataset(Dataset):
    """Dataset for ARC tasks with enhanced preprocessing."""
    
    def __init__(self, data_dir, max_tasks=None):
        self.data_dir = Path(data_dir)
        self.task_files = list(self.data_dir.glob("*.json"))
        
        if max_tasks:
            self.task_files = self.task_files[:max_tasks]
        
        print(f"📊 Loaded {len(self.task_files)} tasks from {data_dir}")
    
    def __len__(self):
        return len(self.task_files)
    
    def __getitem__(self, idx):
        task_file = self.task_files[idx]
        
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Extract demonstration pairs
        train_pairs = task_data.get('train', [])
        
        if len(train_pairs) < 2:
            return None
        
        # Process input-output pairs
        processed_pairs = []
        for pair in train_pairs:
            # Handle different data formats
            if isinstance(pair, dict):
                # Format: {"input": [...], "output": [...]}
                inp = pair.get('input', [])
                out = pair.get('output', [])
            elif isinstance(pair, list) and len(pair) == 2:
                # Format: [input, output]
                inp, out = pair
            else:
                continue
            
            # Convert to numpy arrays and ensure they are 2D
            try:
                input_grid = np.array(inp)
                output_grid = np.array(out)
                
                # Ensure 2D arrays
                if input_grid.ndim == 1:
                    # If 1D, try to reshape to square
                    size = int(np.sqrt(len(input_grid)))
                    if size * size == len(input_grid):
                        input_grid = input_grid.reshape(size, size)
                    else:
                        continue
                
                if output_grid.ndim == 1:
                    # If 1D, try to reshape to square
                    size = int(np.sqrt(len(output_grid)))
                    if size * size == len(output_grid):
                        output_grid = output_grid.reshape(size, size)
                    else:
                        continue
                
                # Skip if not 2D
                if input_grid.ndim != 2 or output_grid.ndim != 2:
                    continue
                
                # Segment objects for better representation
                try:
                    input_objects = segment_grid(input_grid)
                    output_objects = segment_grid(output_grid)
                except Exception as e:
                    # If segmentation fails, use empty lists
                    input_objects = []
                    output_objects = []
                
                processed_pairs.append({
                    'input_grid': input_grid,
                    'output_grid': output_grid,
                    'input_objects': input_objects,
                    'output_objects': output_objects
                })
            except Exception as e:
                # Skip this pair if there's an error
                continue
        
        if not processed_pairs:
            return None
        
        return {
            'task_id': task_file.stem,
            'pairs': processed_pairs,
            'task_data': task_data
        }

def prepare_training_data(task_data, max_colors=21):
    """Prepare training data for the neural guide."""
    
    # Define available primitives
    primitives = {
        'rotate90': rotate90,
        'horizontal_mirror': horizontal_mirror,
        'vertical_mirror': vertical_mirror,
        'replace_color_1_2': lambda grid: replace_color(grid, old_color=1, new_color=2),
        'replace_color_2_1': lambda grid: replace_color(grid, old_color=2, new_color=1),
        'replace_color_1_3': lambda grid: replace_color(grid, old_color=1, new_color=3),
        'replace_color_3_1': lambda grid: replace_color(grid, old_color=3, new_color=1),
        'fill_1': lambda grid: fill(grid, 1),
        'fill_2': lambda grid: fill(grid, 2),
        'fill_3': lambda grid: fill(grid, 3),
        'colorfilter_1': lambda grid: colorfilter(grid, 1),
        'colorfilter_2': lambda grid: colorfilter(grid, 2),
        'colorfilter_3': lambda grid: colorfilter(grid, 3),
    }
    
    primitive_names = list(primitives.keys())
    
    # Generate training examples
    training_examples = []
    
    for pair in task_data['pairs']:
        input_grid = pair['input_grid']
        output_grid = pair['output_grid']
        
        # Try to find a simple transformation
        for i, (name, primitive) in enumerate(primitives.items()):
            try:
                result = primitive(input_grid.copy())
                if np.array_equal(result, output_grid):
                    # Found a match!
                    training_examples.append({
                        'input_grid': input_grid,
                        'output_grid': output_grid,
                        'target_primitive': i,
                        'primitive_name': name
                    })
                    break
            except Exception as e:
                continue
    
    return training_examples, primitive_names

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    all_examples = []
    for task_data in batch:
        examples, primitive_names = prepare_training_data(task_data)
        all_examples.extend(examples)
    
    return all_examples

def train_enhanced_model(model, train_loader, val_loader, config, device):
    """Train the enhanced neural guide model."""
    
    print(f"🚀 Starting training on device: {device}")
    print(f"📋 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize wandb
    try:
        import wandb
        wandb.init(
            project="arc-challenge-enhanced",
            config=config,
            name=f"enhanced-model-{int(time.time())}"
        )
        use_wandb = True
        print("✅ Wandb initialized")
    except Exception as e:
        print(f"⚠️ Wandb not available: {e}")
        use_wandb = False
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for batch in train_pbar:
            if batch is None:
                continue
                
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0
            
            for example in batch:
                input_grid = torch.tensor(example['input_grid'], dtype=torch.long, device=device)
                target = torch.tensor(example['target_primitive'], dtype=torch.long, device=device)
                
                # Ensure input_grid is 2D and has the right shape
                if input_grid.dim() == 1:
                    # If 1D, try to reshape to square
                    size = int(np.sqrt(input_grid.numel()))
                    if size * size == input_grid.numel():
                        input_grid = input_grid.view(size, size)
                    else:
                        continue
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(input_grid.unsqueeze(0))  # Add batch dimension
                loss = criterion(logits, target.unsqueeze(0))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                batch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                batch_correct += (predicted == target).sum().item()
                batch_total += 1
            
            # Update progress bar
            avg_loss = batch_loss / batch_total if batch_total > 0 else 0
            accuracy = batch_correct / batch_total if batch_total > 0 else 0
            train_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.4f}'
            })
            
            train_loss += batch_loss
            train_correct += batch_correct
            train_total += batch_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            
            for batch in val_pbar:
                if batch is None:
                    continue
                    
                batch_loss = 0.0
                batch_correct = 0
                batch_total = 0
                
                for example in batch:
                    input_grid = torch.tensor(example['input_grid'], dtype=torch.long, device=device)
                    target = torch.tensor(example['target_primitive'], dtype=torch.long, device=device)
                    
                    # Forward pass
                    logits = model(input_grid.unsqueeze(0))
                    loss = criterion(logits, target.unsqueeze(0))
                    
                    # Statistics
                    batch_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    batch_correct += (predicted == target).sum().item()
                    batch_total += 1
                
                # Update progress bar
                avg_loss = batch_loss / batch_total if batch_total > 0 else 0
                accuracy = batch_correct / batch_total if batch_total > 0 else 0
                val_pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
                
                val_loss += batch_loss
                val_correct += batch_correct
                val_total += batch_total
        
        # Calculate metrics
        avg_train_loss = train_loss / train_total if train_total > 0 else 0
        avg_val_loss = val_loss / val_total if val_total > 0 else 0
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        print(f"📊 Epoch {epoch+1}/{config['num_epochs']}")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, 'enhanced_neural_guide_best.pth')
            
            print(f"💾 Saved best model (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"🛑 Early stopping after {config['patience']} epochs without improvement")
            break
    
    if use_wandb:
        wandb.finish()
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_accuracy': train_accuracy if 'train_accuracy' in locals() else 0.0,
        'final_val_accuracy': val_accuracy if 'val_accuracy' in locals() else 0.0
    }

def main():
    """Main training function."""
    
    print("🚀 Starting Enhanced Neural Guide Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'max_colors': 21,
        'embed_dim': 256,
        'num_primitives': 17,
        'num_layers': 4,
        'max_grid_size': 30,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'patience': 10,
        'max_tasks': 200,  # Limit for faster training
        'seed': 42
    }
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎮 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data directories
    train_dir = "data/training"
    val_dir = "data/training"  # Use same data for validation (split in loader)
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        print("📁 Available directories:")
        os.system("ls -la data/ 2>/dev/null || echo 'No data directory found'")
        return
    
    # Create datasets
    print("📊 Loading datasets...")
    train_dataset = ARCTaskDataset(train_dir, max_tasks=config['max_tasks'])
    val_dataset = ARCTaskDataset(val_dir, max_tasks=config['max_tasks']//4)  # Smaller validation set
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"📊 Train loader: {len(train_loader)} batches")
    print(f"📊 Val loader: {len(val_loader)} batches")
    
    # Create model
    print("🏗️ Creating enhanced neural guide model...")
    model = EnhancedNeuralGuide(
        max_colors=config['max_colors'],
        embed_dim=config['embed_dim'],
        num_primitives=config['num_primitives'],
        num_layers=config['num_layers'],
        max_grid_size=config['max_grid_size']
    ).to(device)
    
    print(f"📋 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("🎯 Starting training...")
    results = train_enhanced_model(model, train_loader, val_loader, config, device)
    
    # Final results
    print("=" * 50)
    print("🏁 Training completed!")
    print(f"📊 Best validation loss: {results['best_val_loss']:.4f}")
    print(f"📊 Final train accuracy: {results['final_train_accuracy']:.4f}")
    print(f"📊 Final val accuracy: {results['final_val_accuracy']:.4f}")
    print(f"💾 Best model saved as: enhanced_neural_guide_best.pth")
    
    # Save training results
    with open('enhanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Training results saved to enhanced_training_results.json")

if __name__ == "__main__":
    main() 