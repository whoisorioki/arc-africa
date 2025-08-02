# ARC Challenge Africa - Google Colab Training Setup

This guide provides step-by-step instructions for training the neural guide model using Google Colab.

## 🚀 Quick Start

### 1. Prepare Your Project for Colab

1. **Zip your project folder**:
   ```bash
   # From your local project directory
   zip -r arc-africa.zip . -x "*.git*" "*.venv*" "__pycache__/*" "*.pyc"
   ```

2. **Upload to Google Drive**:
   - Go to [Google Drive](https://drive.google.com)
   - Create a folder called `arc-africa`
   - Upload the `arc-africa.zip` file
   - Extract the zip file in Google Drive

### 2. Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Set runtime type to **GPU** (Runtime → Change runtime type → GPU)

### 3. Run the Training Setup

Copy and paste this code into your Colab notebook:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set project path (update this to match your Google Drive path)
PROJECT_PATH = '/content/drive/MyDrive/arc-africa'

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate tqdm numpy pandas scipy matplotlib

# Add project to Python path
import sys
import os
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)
    sys.path.insert(0, os.path.join(PROJECT_PATH, 'src'))

# Verify setup
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### 4. Run Training

```python
# Change to project directory
import os
os.chdir(PROJECT_PATH)

# Run training
!python -m src.neural_guide.train \
    --data_path data/synthetic/synthetic_dataset_cleaned.json \
    --output_dir models \
    --batch_size 16 \
    --epochs 30 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --val_split 0.1 \
    --seed 42
```

## 📋 Detailed Instructions

### Project Structure Requirements

Your Google Drive should contain:
```
arc-africa/
├── data/
│   └── synthetic/
│       └── synthetic_dataset_cleaned.json (553MB)
├── src/
│   ├── neural_guide/
│   │   ├── architecture.py
│   │   └── train.py
│   └── __init__.py
├── models/ (will be created)
└── requirements.txt
```

### Training Configuration

The training script supports the following parameters:

| Parameter        | Default                                 | Description                                  |
| ---------------- | --------------------------------------- | -------------------------------------------- |
| `--data_path`    | `data/synthetic/synthetic_dataset.json` | Path to synthetic dataset                    |
| `--output_dir`   | `models`                                | Directory to save model checkpoints          |
| `--batch_size`   | `8`                                     | Training batch size (increase for Colab GPU) |
| `--epochs`       | `50`                                    | Number of training epochs                    |
| `--lr`           | `1e-4`                                  | Learning rate                                |
| `--weight_decay` | `1e-5`                                  | Weight decay for regularization              |
| `--val_split`    | `0.1`                                   | Validation split ratio                       |
| `--seed`         | `42`                                    | Random seed for reproducibility              |

### Recommended Colab Settings

- **Runtime Type**: GPU (T4 or V100)
- **Runtime Shape**: High-RAM (if available)
- **Hardware Accelerator**: GPU

### Monitoring Training

The training script will output:
- Training and validation loss/accuracy for each epoch
- Learning rate adjustments
- Model checkpoint saves (every 10 epochs)
- Best model save (when validation loss improves)

### Downloading Results

After training completes, download the model files:

```python
from google.colab import files
import os

# Download all model files
models_dir = '/content/drive/MyDrive/arc-africa/models'
for file in os.listdir(models_dir):
    if file.endswith('.pth'):
        files.download(os.path.join(models_dir, file))
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch_size 8`
   - Use gradient accumulation
   - Restart runtime and try again

2. **Import Errors**:
   - Verify project path is correct
   - Check that all source files are uploaded
   - Restart runtime after installing dependencies

3. **Data Loading Issues**:
   - Verify synthetic dataset exists and is not corrupted
   - Check file permissions in Google Drive
   - Ensure dataset format matches expected structure

4. **Training Stuck**:
   - Check GPU memory usage
   - Monitor learning rate (should decrease over time)
   - Verify data is loading correctly

### Performance Tips

- **Use Colab Pro** for longer training sessions and better GPUs
- **Save checkpoints frequently** to avoid losing progress
- **Monitor GPU memory** during training
- **Use mixed precision** for faster training (if implemented)

### Memory Optimization

For large datasets, consider:
- Reducing batch size
- Using gradient accumulation
- Implementing data streaming
- Using model checkpointing

## 📊 Expected Results

With the current configuration, you should expect:
- Training time: 2-4 hours (depending on GPU)
- Final validation accuracy: 60-80%
- Model size: ~50-100MB
- Checkpoints saved every 10 epochs

## 🎯 Next Steps

After successful training:

1. **Test the model** on validation data
2. **Integrate with neuro-symbolic solver**
3. **Run complete pipeline** on test data
4. **Generate submission file** for competition

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your project structure matches requirements
3. Ensure all dependencies are properly installed
4. Check GPU memory usage and adjust batch size accordingly

---

**Note**: This setup is optimized for Google Colab's GPU environment. For local training, you may need to adjust batch sizes and other parameters based on your hardware. 