"""
Neural Guide Architecture for ARC Challenge Solver.

This module implements a Transformer-based neural network that predicts which DSL primitives
are most likely to solve a given ARC task. The model takes input/output grid pairs as input
and outputs a probability distribution over DSL primitives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


class GridEmbedding(nn.Module):
    """Embeds 2D grids into a sequence of tokens for the Transformer.

    This module expects all input grids to be padded to (grid_size, grid_size).
    The color and positional embeddings are created for grid_size*grid_size positions.
    This ensures that all batches are consistent and compatible with the model's expectations.

    Args:
        grid_size (int): Maximum grid size (both dimensions). All grids must be padded to this size.
        embed_dim (int): Embedding dimension for the Transformer.
        max_colors (int): Maximum number of colors (excluding background).

    Example:
        >>> emb = GridEmbedding(grid_size=50)
        >>> x = torch.randint(0, 10, (8, 2, 50, 50))
        >>> out = emb(x)
        >>> print(out.shape)
        torch.Size([8, 2*50*50, 256])
    """

    def __init__(self, grid_size: int = 30, embed_dim: int = 256, max_colors: int = 10):
        """
        Args:
            grid_size: Maximum grid size (both dimensions).
            embed_dim: Embedding dimension for the Transformer.
            max_colors: Maximum number of colors (excluding background).
        """
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.max_colors = max_colors

        # Color embedding: maps color values to embeddings
        self.color_embedding = nn.Embedding(max_colors + 1, embed_dim // 2)

        # Positional embedding: encodes grid positions
        self.pos_embedding = nn.Parameter(
            torch.randn(grid_size * grid_size, embed_dim // 2)
        )

        # Projection to final embedding dimension
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grids: Tensor of shape (batch_size, num_grids, grid_size, grid_size)
                  where num_grids is typically 2 (input, output). All grids must be padded to grid_size.

        Returns:
            Tensor of shape (batch_size, num_grids * grid_size * grid_size, embed_dim)
        """
        batch_size, num_grids, height, width = grids.shape
        assert (
            height == self.grid_size and width == self.grid_size
        ), f"All grids must be padded to ({self.grid_size}, {self.grid_size}), got ({height}, {width})"
        # Reshape to (batch_size * num_grids, grid_size * grid_size)
        grids_flat = grids.view(batch_size * num_grids, self.grid_size * self.grid_size)

        # Get color embeddings
        color_embeds = self.color_embedding(grids_flat)

        # Get positional embeddings for each position
        pos_embeds = self.pos_embedding.unsqueeze(0).expand(
            batch_size * num_grids, -1, -1
        )

        # Combine color and positional embeddings
        combined = torch.cat([color_embeds, pos_embeds], dim=-1)

        # Project to final embedding dimension
        embeddings = self.projection(combined)

        # Reshape back to (batch_size, num_grids * grid_size * grid_size, embed_dim)
        embeddings = embeddings.view(
            batch_size, num_grids * self.grid_size * self.grid_size, self.embed_dim
        )

        return embeddings


class NeuralGuide(nn.Module):
    """Neural Guide model for predicting DSL primitives.

    This model expects all input and output grids to be padded to (grid_size, grid_size).
    The embedding and transformer layers are constructed for this fixed size.
    """

    def __init__(
        self,
        grid_size: int = 48,
        embed_dim: int = 128,  # Further reduced for memory
        num_heads: int = 4,  # Changed to match persistent model
        num_layers: int = 2,  # Changed to match persistent model
        max_colors: int = 20,  # Changed to match persistent model
        num_primitives: int = 17,  # Changed to match persistent model (17 primitives)
        dropout: float = 0.1,
    ):
        """
        Args:
            grid_size: Maximum grid size (default: 48).
            embed_dim: Embedding dimension (default: 128).
            num_heads: Number of attention heads (default: 4).
            num_layers: Number of Transformer layers (default: 2).
            max_colors: Maximum number of colors (default: 20).
            num_primitives: Number of DSL primitives to predict (default: 8).
            dropout: Dropout rate.
        """
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.max_colors = max_colors
        self.num_primitives = num_primitives

        # Embeddings (matching persistent model)
        self.input_embedding = nn.Embedding(max_colors + 1, embed_dim)
        self.output_embedding = nn.Embedding(max_colors + 1, embed_dim)
        
        # Transformer (matching persistent model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,  # Changed to match persistent model
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (matching persistent model)
        self.output_proj = nn.Linear(embed_dim, num_primitives)

    def forward(
        self, input_grids: torch.Tensor, output_grids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_grids: Tensor of shape (batch_size, grid_size, grid_size)
            output_grids: Tensor of shape (batch_size, grid_size, grid_size)

        Returns:
            Tensor of shape (batch_size, num_primitives) with primitive probabilities
        """
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


def create_neural_guide(
    grid_size: int = 48,
    embed_dim: int = 128,  # Further reduced for memory
    num_heads: int = 4,  # Changed to match persistent model
    num_layers: int = 2,  # Changed to match persistent model
    max_colors: int = 20,  # Changed to match persistent model
    num_primitives: int = 17,  # Changed to match persistent model (17 primitives)
    dropout: float = 0.1,
) -> NeuralGuide:
    """Factory function to create a NeuralGuide model with defaults matching persistent model."""
    return NeuralGuide(
        grid_size=grid_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_colors=max_colors,
        num_primitives=num_primitives,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = create_neural_guide()
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    grid_size = 10
    input_grids = torch.randint(0, 6, (batch_size, grid_size, grid_size))
    output_grids = torch.randint(0, 6, (batch_size, grid_size, grid_size))

    with torch.no_grad():
        output = model(input_grids, output_grids)
        print(f"Input shape: {input_grids.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {F.softmax(output[0], dim=0)}")
