"""
Training script for the Neural Guide model.

This script loads the synthetic dataset, trains the Neural Guide model to predict
DSL primitives, and saves the trained model weights for use in the neuro-symbolic solver.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
from pathlib import Path

from .architecture import create_neural_guide, count_parameters

MAX_ALLOWED_GRID_SIZE = 48  # Lowered for faster debug runs


def get_max_grid_size(data_path: str) -> int:
    """Detect the largest grid size in the dataset, capped at MAX_ALLOWED_GRID_SIZE.

    Scans the dataset to find the maximum height and width among all input and output grids.
    Returns the minimum of the detected size and MAX_ALLOWED_GRID_SIZE.

    Args:
        data_path (str): Path to the JSON file containing synthetic data.

    Returns:
        int: The largest grid size found, capped at MAX_ALLOWED_GRID_SIZE.
    """
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        max_h = max(max(len(sample["input"]), len(sample["output"])) for sample in data)
        max_w = max(
            max(len(sample["input"][0]), len(sample["output"][0])) for sample in data
        )
        detected = max(max_h, max_w)
        capped = min(detected, MAX_ALLOWED_GRID_SIZE)
        if detected > MAX_ALLOWED_GRID_SIZE:
            print(
                f"[WARN] Detected grid size {detected} exceeds cap {MAX_ALLOWED_GRID_SIZE}. Will filter out large samples."
            )
        return capped
    except Exception as e:
        print(
            f"[WARN] Could not detect max grid size, using default {MAX_ALLOWED_GRID_SIZE}. Error: {e}"
        )
        return MAX_ALLOWED_GRID_SIZE


class ARCDataset(Dataset):
    """Dataset for ARC task data.

    This dataset loads ARC grid samples and pads them to max_grid_size x max_grid_size.
    Any samples with input or output grids larger than max_grid_size are filtered out.
    This ensures compatibility with the neural model and prevents memory errors.

    Args:
        data_path (str): Path to the JSON file containing synthetic data.
        max_grid_size (int): Maximum grid size for padding. Samples larger than this are skipped.
    """

    def __init__(self, data_path: str, max_grid_size: int = 48):
        self.max_grid_size = max_grid_size

        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)
        # Identify and log samples with grids larger than max_grid_size
        oversized_samples = []
        for idx, sample in enumerate(data):
            h_in, w_in = len(sample["input"]), len(sample["input"][0])
            h_out, w_out = len(sample["output"]), len(sample["output"][0])
            if (
                max(h_in, h_out) > self.max_grid_size
                or max(w_in, w_out) > self.max_grid_size
            ):
                # Try to get a sample ID if present, else use index
                sample_id = sample.get("id", None)
                oversized_samples.append(
                    {
                        "index": idx,
                        "id": sample_id,
                        "input_shape": (h_in, w_in),
                        "output_shape": (h_out, w_out),
                    }
                )
        if oversized_samples:
            print(
                f"[INFO] Skipping {len(oversized_samples)} samples with grid size > {self.max_grid_size}:"
            )
            for s in oversized_samples[:10]:  # Print up to 10 for brevity
                print(
                    f"  Index: {s['index']}, ID: {s['id']}, input: {s['input_shape']}, output: {s['output_shape']}"
                )
            if len(oversized_samples) > 10:
                print(f"  ...and {len(oversized_samples) - 10} more.")
        # Filter out samples with grids larger than max_grid_size
        self.data = [
            sample
            for idx, sample in enumerate(data)
            if not (
                max(len(sample["input"]), len(sample["output"])) > self.max_grid_size
                or max(len(sample["input"][0]), len(sample["output"][0]))
                > self.max_grid_size
            )
        ]
        print(
            f"[INFO] Filtered dataset to {len(self.data)} samples with grid size <= {self.max_grid_size}"
        )
        # Subsample for fast debug runs
        self.data = self.data[:10000]
        print(f"[INFO] Subsampled dataset to {len(self.data)} samples for quick debug.")

        # Define primitive mapping
        self.primitives = [
            "rotate90",
            "horizontal_mirror",
            "vertical_mirror",
            "colorfilter",
            "fill",
            "move",
        ]
        self.primitive_to_idx = {prim: idx for idx, prim in enumerate(self.primitives)}

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def pad_grid(self, grid: List[List[int]]) -> np.ndarray:
        """Pad grid to max_grid_size x max_grid_size."""
        grid = np.array(grid)
        height, width = grid.shape

        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int64)
        padded[:height, :width] = grid

        return padded

    def create_target_vector(self, program: List[str]) -> np.ndarray:
        """Create a binary target vector for the primitives used in the program."""
        target = np.zeros(len(self.primitives), dtype=np.float32)
        for prim in program:
            if prim in self.primitive_to_idx:
                target[self.primitive_to_idx[prim]] = 1.0
        return target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.data[idx]

        # Process input and output grids
        input_grid = self.pad_grid(sample["input"])
        output_grid = self.pad_grid(sample["output"])

        # Clip grid values to valid range (0-9 for typical ARC colors)
        # This prevents values like 48 from causing embedding layer errors
        input_grid = np.clip(input_grid, 0, 9)
        output_grid = np.clip(output_grid, 0, 9)

        # Create target vector
        target = self.create_target_vector(sample["program"])

        return (
            torch.from_numpy(input_grid).long(),
            torch.from_numpy(output_grid).long(),
            torch.from_numpy(target).float(),
        )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, (input_grids, output_grids, targets) in enumerate(progress_bar):
        input_grids = input_grids.to(device)
        output_grids = output_grids.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_grids, output_grids)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct = (predictions == targets).float().mean()

        total_loss += loss.item()
        total_correct += correct.item() * input_grids.size(0)
        total_samples += input_grids.size(0)

        # Update progress bar
        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{correct.item():.4f}"}
        )

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_grids, output_grids, targets in tqdm(dataloader, desc="Validation"):
            input_grids = input_grids.to(device)
            output_grids = output_grids.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(input_grids, output_grids)
            loss = criterion(logits, targets)

            # Calculate accuracy
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct = (predictions == targets).float().mean()

            total_loss += loss.item()
            total_correct += correct.item() * input_grids.size(0)
            total_samples += input_grids.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Neural Guide model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/synthetic/synthetic_dataset.json",
        help="Path to synthetic dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (debug)")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs (debug)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    max_grid_size = get_max_grid_size(args.data_path)
    print(f"[INFO] Using max_grid_size={max_grid_size}")
    dataset = ARCDataset(args.data_path, max_grid_size=max_grid_size)

    # Split into train/val
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model (Colab-matching defaults)
    model = create_neural_guide()
    model = model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
                os.path.join(args.output_dir, "neural_guide_best.pth"),
            )
            print("Saved best model!")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
                os.path.join(args.output_dir, f"neural_guide_epoch_{epoch+1}.pth"),
            )

    # Save final model
    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        },
        os.path.join(args.output_dir, "neural_guide_final.pth"),
    )

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
