import torch


def check_gpu():
    """Checks for GPU availability and prints the status."""
    print("=== Checking for GPU (Accelerator) ===")
    if torch.cuda.is_available():
        print(f"✅ GPU Found: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Version: {torch.version.cuda}")
        print("   - Training will use the GPU for acceleration.")
    else:
        print("❌ No GPU (accelerator) found.")
        print("   - Training will fall back to the CPU.")
        print("   - The 'pin_memory' warning is expected and can be safely ignored.")
        print("   - Training will be significantly slower without a GPU.")


if __name__ == "__main__":
    check_gpu()
