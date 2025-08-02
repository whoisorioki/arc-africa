#!/usr/bin/env python3
"""
Enhanced Neural Guide Architecture for ARC Challenge

This module implements an advanced neural architecture for predicting DSL primitives
with multi-scale attention, spatial relation encoding, and program composition decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for pattern recognition at various scales."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scales = scales
        
        # Create attention layers for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in scales
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Enhanced tensor with multi-scale attention
        """
        batch_size, seq_len, embed_dim = x.shape
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale > 1:
                # Group tokens by scale
                grouped_len = seq_len // scale
                if grouped_len > 0:
                    # Reshape to (batch_size, grouped_len, scale, embed_dim)
                    grouped_x = x[:, :grouped_len * scale, :].view(batch_size, grouped_len, scale, embed_dim)
                    # Average across scale dimension to get (batch_size, grouped_len, embed_dim)
                    grouped_x = grouped_x.mean(dim=2)
                    # Apply attention within groups
                    attn_out, _ = self.scale_attentions[i](grouped_x, grouped_x, grouped_x)
                    # Reshape back to original sequence length
                    attn_out = attn_out.repeat_interleave(scale, dim=1)
                    # Pad if necessary
                    if attn_out.size(1) < seq_len:
                        padding = torch.zeros(batch_size, seq_len - attn_out.size(1), embed_dim, device=x.device)
                        attn_out = torch.cat([attn_out, padding], dim=1)
                else:
                    attn_out = x
            else:
                # Standard attention for scale 1
                attn_out, _ = self.scale_attentions[i](x, x, x)
            
            scale_outputs.append(attn_out)
        
        # Average the scale outputs instead of concatenating to maintain embed_dim
        if len(scale_outputs) > 1:
            fused = torch.stack(scale_outputs, dim=0).mean(dim=0)
        else:
            fused = scale_outputs[0]
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + fused)
        
        return output


class SpatialRelationEncoder(nn.Module):
    """Encodes spatial relationships between objects in the grid."""
    
    def __init__(self, embed_dim: int, max_grid_size: int = 30):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_grid_size = max_grid_size
        
        # Position embeddings
        self.row_embedding = nn.Embedding(max_grid_size, embed_dim // 2)
        self.col_embedding = nn.Embedding(max_grid_size, embed_dim // 2)
        
        # Spatial relation MLP
        self.spatial_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial relationships.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, embed_dim)
            
        Returns:
            Enhanced tensor with spatial encoding
        """
        batch_size, height, width, embed_dim = x.shape
        
        # Create position indices
        row_indices = torch.arange(height, device=x.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, height, width)
        col_indices = torch.arange(width, device=x.device).unsqueeze(0).unsqueeze(0).expand(batch_size, height, width)
        
        # Get position embeddings
        row_emb = self.row_embedding(row_indices)
        col_emb = self.col_embedding(col_indices)
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        
        # Combine with input features
        combined = torch.cat([x, pos_emb], dim=-1)
        
        # Apply spatial relation encoding
        spatial_encoded = self.spatial_mlp(combined)
        
        return spatial_encoded


class ProgramCompositionDecoder(nn.Module):
    """Decodes program composition patterns from grid representations."""
    
    def __init__(self, embed_dim: int, num_primitives: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_primitives = num_primitives
        
        # Program composition layers
        self.composition_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2)
        )
        
        # Primitive prediction head
        self.primitive_head = nn.Linear(embed_dim // 2, num_primitives)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode program composition.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Primitive logits of shape (batch_size, num_primitives)
        """
        # Global average pooling
        pooled = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        # Apply composition layers
        composed = self.composition_layers(pooled)
        
        # Predict primitives
        primitive_logits = self.primitive_head(composed)
        
        return primitive_logits


class EnhancedNeuralGuide(nn.Module):
    """
    Enhanced Neural Guide for ARC Challenge.
    
    This model uses multi-scale attention, spatial relation encoding, and
    program composition decoding to predict promising DSL primitives.
    """
    
    def __init__(self, 
                 max_colors: int = 21,
                 embed_dim: int = 256,
                 num_primitives: int = 17,
                 num_layers: int = 4,
                 max_grid_size: int = 30):
        super().__init__()
        
        self.max_colors = max_colors
        self.embed_dim = embed_dim
        self.num_primitives = num_primitives
        self.num_layers = num_layers
        self.max_grid_size = max_grid_size
        
        # Input embedding (add 1 for padding token)
        self.input_embedding = nn.Embedding(max_colors + 1, embed_dim)
        
        # Multi-scale attention layers
        self.multi_scale_attention = nn.ModuleList([
            MultiScaleAttention(embed_dim, num_heads=8, scales=[1, 2, 4])
            for _ in range(num_layers)
        ])
        
        # Spatial relation encoder
        self.spatial_encoder = SpatialRelationEncoder(embed_dim, max_grid_size)
        
        # Program composition decoder
        self.program_decoder = ProgramCompositionDecoder(embed_dim, num_primitives)
        
        # Task aggregator (Transformer encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.task_aggregator = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, num_primitives)
        
        # Confidence predictor
        self.confidence_predictor = nn.Linear(embed_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced neural guide.
        
        Args:
            grid: Input grid tensor of shape (batch_size, height, width)
                 with values in range [0, max_colors]
        
        Returns:
            Primitive logits of shape (batch_size, num_primitives)
        """
        batch_size, height, width = grid.shape
        
        # Embed input grid
        embedded = self.input_embedding(grid)  # (batch_size, height, width, embed_dim)
        
        # Apply spatial relation encoding
        spatial_encoded = self.spatial_encoder(embedded)
        
        # Reshape for transformer processing
        # Flatten spatial dimensions
        seq_len = height * width
        spatial_encoded = spatial_encoded.view(batch_size, seq_len, self.embed_dim)
        
        # Apply multi-scale attention layers
        attention_output = spatial_encoded
        for attention_layer in self.multi_scale_attention:
            attention_output = attention_layer(attention_output)
            attention_output = self.dropout(attention_output)
        
        # Apply task aggregator
        aggregated = self.task_aggregator(attention_output)
        
        # Global average pooling
        pooled = torch.mean(aggregated, dim=1)  # (batch_size, embed_dim)
        
        # Apply program composition decoder
        composed = self.program_decoder(aggregated)
        
        # Final output projection
        output = self.output_proj(pooled)
        
        return output
    
    def predict_primitives(self, grid: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k most likely primitives.
        
        Args:
            grid: Input grid tensor
            top_k: Number of top primitives to return
            
        Returns:
            Tuple of (primitive_indices, probabilities)
        """
        with torch.no_grad():
            logits = self.forward(grid)
            probabilities = F.softmax(logits, dim=-1)
            
            # Get top-k primitives
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_primitives), dim=-1)
            
            return top_indices, top_probs
    
    def get_confidence(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Get prediction confidence.
        
        Args:
            grid: Input grid tensor
            
        Returns:
            Confidence scores
        """
        with torch.no_grad():
            # Get intermediate representation
            embedded = self.input_embedding(grid)
            batch_size, height, width, embed_dim = embedded.shape
            
            # Apply spatial encoding
            spatial_encoded = self.spatial_encoder(embedded)
            spatial_encoded = spatial_encoded.view(batch_size, height * width, embed_dim)
            
            # Apply attention layers
            attention_output = spatial_encoded
            for attention_layer in self.multi_scale_attention:
                attention_output = attention_layer(attention_output)
            
            # Pool and get confidence
            pooled = torch.mean(attention_output, dim=1)
            confidence = torch.sigmoid(self.confidence_predictor(pooled))
            
            return confidence


def create_enhanced_model(config: Dict) -> EnhancedNeuralGuide:
    """
    Create an enhanced neural guide model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        EnhancedNeuralGuide model
    """
    return EnhancedNeuralGuide(
        max_colors=config.get('max_colors', 21),
        embed_dim=config.get('embed_dim', 256),
        num_primitives=config.get('num_primitives', 17),
        num_layers=config.get('num_layers', 4),
        max_grid_size=config.get('max_grid_size', 30)
    )


# Default training configuration
ENHANCED_TRAINING_CONFIG = {
    'max_colors': 21,
    'embed_dim': 256,
    'num_primitives': 17,
    'num_layers': 4,
    'max_grid_size': 30,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'num_epochs': 50,
    'patience': 10,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'early_stopping_patience': 10
} 