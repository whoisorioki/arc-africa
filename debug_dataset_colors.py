import json
from tqdm import tqdm
import numpy as np
import os


def find_max_color_value(data_path):
    """
    Scans a synthetic dataset to find the maximum color value present in any grid.

    This helps debug embedding layer errors by ensuring the model's color
    vocabulary is large enough to handle all data.

    Args:
        data_path (str): The path to the JSON dataset file.

    Returns:
        int: The maximum color value found in the dataset.
    """
    print(f"🔍 Scanning dataset for max color value: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        return -1

    with open(data_path, "r") as f:
        data = json.load(f)

    max_color = 0
    for item in tqdm(data, desc="Scanning grids"):
        input_grid = np.array(item["input"])
        output_grid = np.array(item["output"])

        current_max = max(input_grid.max(), output_grid.max())
        if current_max > max_color:
            max_color = current_max

    return int(max_color)


if __name__ == "__main__":
    dataset_path = "data/synthetic/enhanced_synthetic_dataset_v2.json"
    max_val = find_max_color_value(dataset_path)

    if max_val != -1:
        print(f"\n✅ Scan complete.")
        print(f"🎨 Maximum color value found in the dataset: {max_val}")

        if max_val > 10:
            print("   - This confirms the source of the 'Assertion failed' error.")
            print(
                "   - The model's embedding layer needs to be configured for a larger color vocabulary."
            )
        else:
            print(
                "   - The maximum color is within the expected range. The error may have another cause."
            )
