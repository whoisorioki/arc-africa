# EC2 Training Report - Neural Guide Model

## 📋 **Executive Summary**

This report documents the setup and optimization attempts for training the Neural Guide model on AWS EC2 instance `i-0a84d110ef585eedb`. The training aims to create a neural guide for the ARC Challenge neuro-symbolic solver using enhanced synthetic data.

## 🖥️ **EC2 Instance Specifications**

### **Hardware Details:**

- **Instance ID**: `i-0a84d110ef585eedb`
- **Instance Type**: `c5.4xlarge` (Compute Optimized)
- **Public IP**: `18.207.92.156`
- **Region**: `us-east-1`

### **Hardware Specifications:**

- **CPU**: Intel Xeon Platinum 8124M @ 3.00GHz
- **vCPUs**: 16 (8 cores × 2 threads)
- **Memory**: 30GB RAM (29GB available)
- **Storage**: 200GB EBS volume (160GB available)
- **GPU**: None (CPU-only instance)

### **Performance Characteristics:**

- **CPU Architecture**: x86_64
- **Cache**: 256KB L1d cache per core
- **Virtualization**: KVM (full virtualization)
- **Current Load**: Very low (99.6% idle)

## 📊 **Dataset Information**

### **Enhanced Synthetic Dataset:**

- **File**: `data/synthetic/enhanced_synthetic_dataset_v2.json`
- **Total Samples**: 100,000
- **Quality Threshold**: 0.7-0.85 (configurable)
- **Size**: 160.8 MiB
- **Primitives**: 17 unique DSL primitives

### **Dataset Structure:**

```json
{
  "input": [[2D grid]],
  "output": [[2D grid]],
  "primitives": ["list", "of", "primitives"],
  "quality": 0.85,
  "complexity": 0.3,
  "task_id": "synthetic_0001"
}
```

## 🚀 **Training Configuration Attempts**

### **Attempt 1: Initial Setup**

```bash
python scripts/train_with_enhanced_data.py \
    --dataset data/synthetic/enhanced_synthetic_dataset_v2.json \
    --output models/neural_guide_ec2_trained.pth \
    --batch_size 4 \
    --epochs 20 \
    --max_samples 50000 \
    --quality_threshold 0.7 \
    --device cpu
```

**Status**: Started successfully, loss: 0.6824
**Issue**: Slow training (estimated 10-20 hours)

### **Attempt 2: Speed Optimization**

```bash
python scripts/train_with_enhanced_data.py \
    --dataset data/synthetic/enhanced_synthetic_dataset_v2.json \
    --output models/neural_guide_ec2_optimized.pth \
    --batch_size 32 \
    --epochs 8 \
    --max_samples 20000 \
    --quality_threshold 0.85 \
    --device cpu
```

**Status**: Killed (Out of Memory)
**Issue**: Batch size too large for available memory

### **Attempt 3: Memory Optimization**

```bash
python scripts/train_with_enhanced_data.py \
    --dataset data/synthetic/enhanced_synthetic_dataset_v2.json \
    --output models/neural_guide_ec2_optimized.pth \
    --batch_size 16 \
    --epochs 10 \
    --max_samples 15000 \
    --quality_threshold 0.8 \
    --device cpu
```

**Status**: Killed (Out of Memory)
**Issue**: Still too much memory usage

### **Attempt 4: Conservative Settings**

```bash
python scripts/train_with_enhanced_data.py \
    --dataset data/synthetic/enhanced_synthetic_dataset_v2.json \
    --output models/neural_guide_ec2_optimized.pth \
    --batch_size 8 \
    --epochs 12 \
    --max_samples 12000 \
    --quality_threshold 0.8 \
    --device cpu
```

**Status**: Killed (Out of Memory)
**Issue**: CUDA-enabled PyTorch causing memory overhead

## 🔧 **Technical Issues Identified**

### **1. PyTorch Installation Problem**

- **Current**: PyTorch 2.3.1+cu121 (CUDA-enabled)
- **Issue**: Designed for GPU, causing memory overhead on CPU
- **Solution**: Install CPU-only PyTorch

### **2. Memory Management**

- **Available Memory**: 29GB
- **PyTorch Overhead**: ~8-10GB (CUDA libraries)
- **Dataset Loading**: ~2-4GB
- **Model Memory**: ~4-6GB
- **Batch Processing**: Variable based on batch size

### **3. Environment Optimization**

- **CPU Threads**: Set to 8 (optimal for 16 vCPUs)
- **OMP Threads**: 16 (interop threads)
- **Memory Allocation**: Limited to 128MB chunks

## 📈 **Performance Analysis**

### **Memory Usage Patterns:**

- **Dataset Loading**: 2-4GB
- **Model Creation**: 4-6GB
- **Batch Processing**: 2-8GB (depending on batch size)
- **Total Peak**: 12-18GB (theoretical)

### **Training Speed Estimates:**

- **Batch Size 4**: ~10-20 hours (original)
- **Batch Size 8**: ~5-10 hours (optimized)
- **Batch Size 16**: ~3-5 hours (target)
- **Batch Size 32**: ~2-3 hours (ideal, but OOM)

## 🚨 **EMERGENCY 4-HOUR ACTION PLAN**

### **Strategic Assessment:**

The CPU-only approach is fundamentally flawed for meeting the 4-hour deadline. Even with perfect optimization, 3-5 hours leaves no margin for error, validation, or submission.

### **Step 1: Immediate GPU Quota Check**

```bash
# Check current GPU spot quotas
aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-85EED4F8 \
    --region us-east-1
```

### **Step 2: Launch GPU Instance (g4dn.xlarge)**

```bash
# Launch g4dn.xlarge spot instance with Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type g4dn.xlarge \
    --key-name arc-us-key \
    --security-group-ids sg-0907f46ba2874801b \
    --subnet-id subnet-068b6499b017ae63c \
    --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=0.20}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"DeleteOnTermination":false}}]'
```

### **Step 3: GPU-Optimized Training**

```bash
# Use the new GPU-optimized script
python scripts/train_gpu_optimized.py \
    --dataset data/synthetic/enhanced_synthetic_dataset_v2.json \
    --output models/neural_guide_gpu_optimized.pth \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --epochs 6 \
    --max_samples 15000 \
    --quality_threshold 0.9 \
    --device cuda
```

### **GPU Optimizations Implemented:**

- **Mixed Precision Training**: `torch.cuda.amp.autocast()` and `GradScaler()`
- **cuDNN Benchmark**: `torch.backends.cudnn.benchmark = True`
- **PyTorch 2.0 Compile**: `torch.compile(model)`
- **Optimized DataLoader**: `pin_memory=True`, `num_workers=4`
- **OneCycleLR Scheduler**: Faster convergence
- **Non-blocking Transfers**: `non_blocking=True`

### **Expected Results:**

- **Training Time**: < 1 hour (vs 3-5 hours on CPU)
- **Cost**: ~$0.15/hour (Spot Instance vs $0.68/hour On-Demand)
- **Model Quality**: Higher (0.9 threshold)
- **Risk**: Low (meets deadline with margin)

## 📋 **Next Steps**

1. **Check GPU spot quotas** immediately (most time-critical)
2. **Launch g4dn.xlarge spot instance** with Deep Learning AMI
3. **Sync project files** to new GPU instance
4. **Run GPU-optimized training** with `train_gpu_optimized.py`
5. **Monitor training progress** (should complete in < 1 hour)
6. **Save trained model** to S3 for local use
7. **Evaluate model performance** on validation set
8. **Submit to competition** with trained model

## 🔍 **Monitoring Commands**

### **Memory Monitoring:**

```bash
# Real-time memory usage
watch -n 5 'free -h'

# Process memory usage
ps aux --sort=-%mem | head -10
```

### **Training Progress:**

```bash
# Check for checkpoints
ls -la models/*.pth

# Monitor logs
tail -f training.log
```

## 📊 **Cost Analysis**

### **EC2 Instance Costs:**

- **Instance Type**: c5.4xlarge
- **Hourly Rate**: ~$0.68/hour
- **Estimated Training Time**: 3-5 hours
- **Total Cost**: ~$2.04 - $3.40

### **S3 Storage Costs:**

- **Dataset Size**: 160.8 MiB
- **Monthly Cost**: ~$0.004
- **Model Storage**: ~$0.001/month

### **Total Project Cost**: ~$3-4 for complete training

## ✅ **Success Criteria**

- [ ] GPU spot quotas approved (immediate priority)
- [ ] g4dn.xlarge instance launched successfully
- [ ] Training completes without errors
- [ ] Model converges within 6 epochs
- [ ] Loss decreases consistently
- [ ] Model saves successfully
- [ ] Training time < 1 hour
- [ ] Total project time < 4 hours (deadline met)

---

**Report Generated**: July 31, 2025
**Status**: In Progress - Memory optimization required
**Next Update**: After CPU-only PyTorch installation
