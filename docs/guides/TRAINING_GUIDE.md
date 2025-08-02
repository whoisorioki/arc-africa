# 🚀 Neural Guide Training Guide

## Overview

This guide provides step-by-step instructions for training the enhanced neural guide with the new sophisticated primitives. The training can be done locally (optimized for Intel Iris Xe Graphics) or in the cloud (Google Colab with GPU).

## 📋 Prerequisites

### Local Training (Intel Iris Xe Graphics)
- **GPU**: Intel Iris Xe Graphics (integrated)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space
- **Python**: 3.8+ with virtual environment

### Cloud Training (Google Colab)
- **GPU**: Tesla T4/V100 (free) or A100 (Pro)
- **Memory**: 12GB+ RAM
- **Storage**: 15GB+ free space
- **Internet**: Stable connection

## 🎯 Quick Start (Recommended)

### Step 1: Regenerate Enhanced Dataset
```bash
# Activate virtual environment
.venv\Scripts\activate

# Regenerate dataset with enhanced primitives
python scripts/regenerate_enhanced_dataset.py
```

### Step 2: Test Your Setup
```bash
# Run quick training test (100 samples, 5 epochs)
python scripts/quick_train_local.py
```

This will:
- ✅ Verify your environment
- ✅ Test with minimal data
- ✅ Validate the trained model
- ✅ Confirm everything works

**Expected time**: 5-10 minutes

## 🏠 Local Training (Intel Iris Xe)

### Step 1: Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Verify GPU detection
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 2: Small-Scale Training (Recommended First)
```bash
# Train with 1,000 samples (good for testing)
python scripts/train_local_gpu.py \
    --max_samples 1000 \
    --batch_size 8 \
    --num_epochs 10 \
    --save_path models/neural_guide_small.pth
```

**Expected time**: 30-60 minutes

### Step 3: Medium-Scale Training
```bash
# Train with 10,000 samples (good balance)
python scripts/train_local_gpu.py \
    --max_samples 10000 \
    --batch_size 8 \
    --num_epochs 20 \
    --save_path models/neural_guide_medium.pth
```

**Expected time**: 4-8 hours

### Step 4: Full-Scale Training (If you have time)
```bash
# Train with full dataset (best results)
python scripts/train_local_gpu.py \
    --max_samples None \
    --batch_size 8 \
    --num_epochs 50 \
    --save_path models/neural_guide_full.pth
```

**Expected time**: 12-24 hours

## ☁️ Cloud Training (Google Colab)

### Step 1: Prepare Your Data
1. **Upload to Google Drive**:
   - Upload `data/synthetic/synthetic_dataset_cleaned.json` to your Google Drive
   - Create folder: `MyDrive/arc-africa/data/synthetic/`

2. **Upload Code**:
   - Upload the entire project to Google Drive
   - Or use GitHub integration

### Step 2: Setup Colab Notebook
```python
# In Google Colab
!git clone https://github.com/your-repo/arc-africa.git
%cd arc-africa

# Install dependencies
!pip install -r requirements.txt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset from Drive
!cp "/content/drive/MyDrive/arc-africa/data/synthetic/synthetic_dataset_cleaned.json" "data/synthetic/"
```

### Step 3: Run Cloud Training
```bash
# Full-scale training with GPU acceleration
python scripts/train_cloud_colab.py \
    --max_samples None \
    --batch_size 32 \
    --num_epochs 50 \
    --save_path models/neural_guide_cloud.pth \
    --setup_colab
```

**Expected time**: 2-4 hours (much faster with cloud GPU)

## 📊 Training Parameters Guide

### Local Training (Intel Iris Xe Optimized)
| Parameter     | Small    | Medium | Full     |
| ------------- | -------- | ------ | -------- |
| `max_samples` | 1,000    | 10,000 | None     |
| `batch_size`  | 8        | 8      | 8        |
| `num_epochs`  | 10       | 20     | 50       |
| `hidden_dim`  | 128      | 128    | 128      |
| Expected Time | 30-60min | 4-8hrs | 12-24hrs |

### Cloud Training (GPU Optimized)
| Parameter     | Small   | Medium   | Full   |
| ------------- | ------- | -------- | ------ |
| `max_samples` | 1,000   | 10,000   | None   |
| `batch_size`  | 32      | 32       | 32     |
| `num_epochs`  | 10      | 20       | 50     |
| `hidden_dim`  | 256     | 256      | 256    |
| Expected Time | 5-10min | 30-60min | 2-4hrs |

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 4

# Reduce hidden dimension
--hidden_dim 64
```

#### 2. Slow Training
```bash
# For local training, reduce workers
--num_workers 0

# For cloud training, increase workers
--num_workers 4
```

#### 3. CUDA Not Available
```bash
# Check PyTorch installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Intel GPU, try Intel Extension for PyTorch
pip install intel-extension-for-pytorch
```

#### 4. Dataset Not Found
```bash
# Generate synthetic dataset first
python scripts/generate_synthetic_data.py
```

## 📈 Monitoring Training

### Local Training
- Watch console output for loss/accuracy
- Check GPU usage: `nvidia-smi` (if available)
- Monitor memory usage in Task Manager

### Cloud Training
- Use Colab's built-in monitoring
- Check GPU usage in Colab's hardware info
- Monitor training curves in real-time

## 🎯 Expected Results

### Training Metrics
- **Loss**: Should decrease from ~0.7 to ~0.3
- **Accuracy**: Should increase from ~0.3 to ~0.7
- **Validation**: Should track training closely

### Solver Performance
After training, test the solver:
```bash
python scripts/local_validate.py \
    --data_path data/train.json \
    --output_dir validation_results \
    --beam_width 5 \
    --max_search_depth 3
```

**Expected improvement**: 2-5x better accuracy compared to untrained model

## 🚀 Next Steps

After successful training:

1. **Test the Model**:
   ```bash
   python scripts/local_validate.py --data_path data/train.json
   ```

2. **Generate Submission**:
   ```bash
   python scripts/generate_submission.py
   ```

3. **Implement TTT** (Phase 4 Step 3):
   - Add aggressive test-time training
   - Fine-tune on each task

## 💡 Tips for Best Results

### Local Training
- **Use SSD storage** for faster data loading
- **Close other applications** to free up memory
- **Monitor temperature** - integrated GPUs can throttle
- **Use smaller batches** if you encounter OOM errors

### Cloud Training
- **Use Colab Pro** for better GPUs (A100)
- **Save checkpoints** to Google Drive
- **Monitor runtime** - Colab has time limits
- **Use mixed precision** for faster training

### General
- **Start small** - test with 100-1000 samples first
- **Monitor validation** - stop if overfitting
- **Save best model** - use validation loss as metric
- **Experiment** - try different hyperparameters

## 🆘 Need Help?

If you encounter issues:

1. **Check logs** - look for error messages
2. **Reduce complexity** - try smaller dataset/batch size
3. **Verify environment** - ensure all dependencies installed
4. **Check hardware** - ensure sufficient memory/GPU

The enhanced solver with sophisticated primitives should significantly improve performance once properly trained! 