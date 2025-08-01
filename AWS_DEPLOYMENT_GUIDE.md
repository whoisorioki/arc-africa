# AWS Deployment Guide for ARC Challenge Africa

## 🚀 Quick Start

### 1. Prerequisites

- AWS account with EC2 access
- AWS CLI configured
- SSH key pair for EC2 access

### 2. Setup AWS Environment

```bash
# Install AWS CLI (if not already installed)
pip install boto3

# Configure AWS credentials
aws configure

# Run AWS setup script
python aws_setup_arc.py
```

### 3. Connect to AWS Instance

```bash
# Get instance IP (replace with your instance ID)
aws ec2 describe-instances --instance-ids i-1234567890abcdef0 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# Connect via SSH
ssh -i arc-challenge-key.pem ec2-user@<instance-ip>
```

## 🖥️ Instance Configuration

### Recommended Instance Types

| Instance Type | vCPUs | RAM  | GPU    | Cost/Hour | Use Case              |
| ------------- | ----- | ---- | ------ | --------- | --------------------- |
| g4dn.xlarge   | 4     | 16GB | 1xT4   | ~$0.50    | Training & Submission |
| g4dn.2xlarge  | 8     | 32GB | 1xT4   | ~$0.90    | Faster Training       |
| g5.xlarge     | 4     | 16GB | 1xA10G | ~$1.20    | Best Performance      |

### Deep Learning AMI Features

- ✅ PyTorch with CUDA support
- ✅ Pre-installed ML libraries
- ✅ GPU drivers configured
- ✅ Jupyter notebook support

## 📁 Project Setup on AWS

### 1. Upload Project Files

```bash
# From your local machine
aws s3 sync . s3://your-bucket-name/arc-challenge/

# On AWS instance
aws s3 sync s3://your-bucket-name/arc-challenge/ /home/ec2-user/arc-africa/
```

### 2. Install Dependencies

```bash
# Activate virtual environment
source arc-env/bin/activate

# Install project requirements
cd arc-africa
pip install -r requirements.txt

# Install additional packages for enhanced training
pip install wandb tqdm
```

### 3. Verify GPU Setup

```bash
# Check GPU availability
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎯 Training Enhanced Model

### 1. Start Training

```bash
# Activate environment
source arc-env/bin/activate
cd arc-africa

# Start enhanced training
python aws_train_enhanced.py
```

### 2. Monitor Training

```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f training.log

# Check wandb dashboard
# Visit https://wandb.ai/your-username/arc-challenge-enhanced
```

### 3. Training Configuration

The enhanced model uses:

- **Architecture**: Multi-scale attention + spatial relations
- **Training**: 100 epochs with early stopping
- **Batch Size**: 32 (adjustable based on GPU memory)
- **Learning Rate**: 1e-4 with cosine annealing
- **Regularization**: Dropout + weight decay

## 📊 Generating Competition Submission

### 1. Generate Submission with Enhanced Model

```bash
# Use enhanced model for submission
python scripts/generate_submission_enhanced.py
```

### 2. Monitor Submission Progress

```bash
# Check submission progress
tail -f submission.log

# Monitor system resources
htop
```

### 3. Download Results

```bash
# Download submission file
aws s3 cp submission.csv s3://your-bucket-name/results/

# Download from local machine
aws s3 cp s3://your-bucket-name/results/submission.csv ./
```

## 💰 Cost Optimization

### 1. Spot Instances

```bash
# Use spot instances for cost savings (up to 90% discount)
# Modify aws_setup_arc.py to use spot instances
```

### 2. Auto-shutdown

```bash
# Set up auto-shutdown after training
echo "sudo shutdown -h +60" | at now + 2 hours
```

### 3. Cost Monitoring

```bash
# Monitor costs
aws ce get-cost-and-usage --time-period Start=2025-01-01,End=2025-01-02 --granularity DAILY --metrics BlendedCost
```

## 🔧 Performance Optimization

### 1. GPU Optimization

```bash
# Set GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Optimize PyTorch performance
export CUDA_LAUNCH_BLOCKING=1
```

### 2. Data Loading Optimization

```bash
# Use multiple workers for data loading
# Set num_workers=4 in DataLoader

# Use pinned memory for faster GPU transfer
# Set pin_memory=True in DataLoader
```

### 3. Model Optimization

```bash
# Use mixed precision training
# Enable torch.cuda.amp.autocast()

# Use gradient accumulation for larger effective batch sizes
```

## 📈 Expected Performance Improvements

### Enhanced Model Features

1. **Multi-scale Attention**: Better pattern recognition at different scales
2. **Spatial Relations**: Understanding of spatial relationships in grids
3. **Program Composition**: Better understanding of primitive combinations
4. **Hierarchical Features**: Multi-level feature extraction

### Expected Results

- **Training Time**: 2-4 hours on g4dn.xlarge
- **Model Size**: ~50MB (enhanced architecture)
- **Memory Usage**: ~8GB GPU memory
- **Expected Accuracy**: 10-25% improvement over baseline

## 🚨 Troubleshooting

### Common Issues

1. **GPU Out of Memory**

   ```bash
   # Reduce batch size
   # Use gradient accumulation
   # Enable mixed precision
   ```

2. **Slow Data Loading**

   ```bash
   # Increase num_workers
   # Use SSD storage
   # Pre-load data to memory
   ```

3. **Connection Issues**
   ```bash
   # Check security group settings
   # Verify SSH key permissions
   # Use AWS Systems Manager for access
   ```

### Monitoring Commands

```bash
# Check system resources
htop
nvidia-smi
df -h

# Check training progress
tail -f training.log
ps aux | grep python

# Check network connectivity
ping google.com
curl -I https://wandb.ai
```

## 📋 Checklist

### Before Training

- [ ] AWS instance running with GPU
- [ ] Project files uploaded
- [ ] Dependencies installed
- [ ] GPU drivers working
- [ ] Wandb configured

### During Training

- [ ] Monitor GPU usage
- [ ] Check training logs
- [ ] Verify model saves
- [ ] Monitor costs

### After Training

- [ ] Download trained model
- [ ] Generate submission
- [ ] Validate submission format
- [ ] Terminate instance
- [ ] Backup results

## 🎉 Success Metrics

### Training Success

- ✅ Model converges (loss decreases)
- ✅ Validation accuracy improves
- ✅ No GPU memory errors
- ✅ Training completes without interruption

### Submission Success

- ✅ Submission file generated
- ✅ Correct format (ID, row columns)
- ✅ All tasks processed
- ✅ File size reasonable (~100KB-1MB)

---

**Estimated Total Cost**: $2-5 for complete training and submission generation
**Estimated Time**: 4-8 hours for full pipeline
**Expected Improvement**: 10-25% better exact matches
