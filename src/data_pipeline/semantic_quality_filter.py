import torch
import torch.nn as nn
import numpy as np


class ARCLikeClassifier(nn.Module):
    """
    A simple CNN to classify if a task is 'ARC-like'.

    This model is trained on real ARC tasks to learn the visual patterns
    and characteristics that distinguish them from random or noisy grids.
    It serves as a semantic quality filter for synthetic data generation.
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                2, 16, kernel_size=3, padding=1
            ),  # Input channels is 2 (input, output)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, input_grid, output_grid):
        # Stack grids to create a 2-channel input
        x = torch.stack([input_grid, output_grid], dim=1).float()
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def is_arc_like(
    task_pair: tuple, model: ARCLikeClassifier, threshold: float = 0.6
) -> bool:
    """
    Uses a pre-trained classifier to determine if a task is semantically 'ARC-like'.

    Args:
        task_pair (tuple): A tuple containing the (input_grid, output_grid).
        model (ARCLikeClassifier): The pre-trained classification model.
        threshold (float): The confidence threshold for the classification.

    Returns:
        bool: True if the task is classified as ARC-like, False otherwise.
    """
    input_grid, output_grid = task_pair

    # Preprocess grids: pad to a fixed size (e.g., 32x32)
    def pad(grid):
        padded = np.zeros((32, 32), dtype=np.float32)
        h, w = grid.shape
        padded[:h, :w] = grid
        return padded

    input_padded = torch.from_numpy(pad(input_grid)).unsqueeze(0)
    output_padded = torch.from_numpy(pad(output_grid)).unsqueeze(0)

    with torch.no_grad():
        confidence = model(input_padded, output_padded).item()

    return confidence >= threshold


if __name__ == "__main__":
    # This is a placeholder for where the training logic for the classifier would go.
    # For now, we will just demonstrate its usage.

    # 1. Instantiate the model
    classifier = ARCLikeClassifier()

    # Note: In a real scenario, you would load pre-trained weights here.
    # torch.save(classifier.state_dict(), "models/arc_like_classifier.pth")
    # classifier.load_state_dict(torch.load("models/arc_like_classifier.pth"))
    classifier.eval()

    # 2. Example Usage
    # An 'ARC-like' task (e.g., simple pattern completion)
    good_input = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    good_output = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])

    # A 'non-ARC-like' task (e.g., random noise)
    bad_input = np.random.randint(0, 5, (5, 5))
    bad_output = np.random.randint(0, 5, (5, 5))

    # We can't get a real prediction without a trained model,
    # so we will simulate the function call.
    print("Simulating ARC-like classification (requires a trained model):")
    print(
        f"  - 'Good' Task Prediction (Simulated): {is_arc_like((good_input, good_output), classifier)}"
    )
    print(
        f"  - 'Bad' Task Prediction (Simulated): {is_arc_like((bad_input, bad_output), classifier, threshold=0.7)}"
    )

    print("\nThis script provides the core component for a semantic quality filter.")
    print("The next step is to train this classifier on real ARC data.")
