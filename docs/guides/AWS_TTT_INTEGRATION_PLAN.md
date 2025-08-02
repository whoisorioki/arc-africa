# AWS TTT Integration Plan

## Overview

The local TTT implementation is causing high CPU and memory usage, making it impractical for development. This plan outlines the integration of AWS cloud resources for efficient Test-Time Training.

## Problem Analysis

### Current Issues

- **High CPU Usage**: Transformer model training on CPU causes system unresponsiveness
- **Memory Pressure**: 768K parameter model consumes significant RAM (15.8GB system)
- **Performance Impact**: Local system becomes unusable during training
- **Technical Bugs**: Index errors and tensor size mismatches in current implementation

### Resource Constraints

- **Local GPU**: NVIDIA GeForce MX450 (2GB VRAM) - insufficient for training
- **System RAM**: 15.8GB - adequate for inference, not training
- **CPU**: Limited cores for transformer operations

## AWS Solution Architecture

### 1. Cloud Infrastructure

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local System  │    │   AWS S3        │    │   AWS EC2       │
│                 │    │                 │    │                 │
│ • Data Prep     │◄──►│ • Model Storage │◄──►│ • GPU Training  │
│ • Inference     │    │ • Results       │    │ • TTT Pipeline  │
│ • Validation    │    │ • Checkpoints   │    │ • Fast Training │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. Instance Specifications

- **Instance Type**: `g4dn.xlarge`
- **GPU**: NVIDIA T4 (16GB VRAM)
- **CPU**: 4 vCPUs
- **Memory**: 16GB RAM
- **Storage**: 125GB NVMe SSD
- **Cost**: ~$0.526/hour

### 3. Training Pipeline

```
Local → S3 Upload → EC2 Training → S3 Results → Local Download
```

## Implementation Steps

### Phase 1: AWS Setup (1-2 hours)

1. **Prerequisites**

   - Install AWS CLI
   - Configure AWS credentials
   - Create key pair in AWS console

2. **Infrastructure Creation**

   ```bash
   # Check AWS setup
   python scripts/aws_setup.py --action check

   # Create infrastructure
   python scripts/aws_setup.py --action setup --key-name your-key-name
   ```

3. **File Upload**
   - Upload project code to S3
   - Upload pre-trained model
   - Upload ARC training data

### Phase 2: Cloud Training (2-4 hours)

1. **SSH to EC2 Instance**

   ```bash
   ssh -i your-key.pem ubuntu@<instance-ip>
   ```

2. **Run TTT Training**

   ```bash
   # Single task training
   python3 scripts/aws_ttt_training.py --task_id train_0000 --epochs 10

   # Batch training
   python3 scripts/aws_ttt_training.py --task_id train_0001 --epochs 10
   ```

3. **Monitor Training**
   - CloudWatch metrics
   - Training logs
   - GPU utilization

### Phase 3: Results Integration (30 minutes)

1. **Download Results**

   ```bash
   aws s3 sync s3://bucket-name/results ./aws_results
   ```

2. **Local Integration**
   - Use adapted models for local inference
   - Validate performance improvements
   - Update solver with cloud-trained models

## Cost Analysis

### Estimated Costs

- **EC2 Instance**: $0.526/hour × 4 hours = $2.10
- **S3 Storage**: ~$0.02 (minimal usage)
- **Data Transfer**: ~$0.01 (minimal)
- **Total per session**: ~$2.13

### Cost Optimization

- **Spot Instances**: 60-90% cost reduction
- **Reserved Instances**: For frequent usage
- **Auto-termination**: Stop instances when idle

## Benefits

### Performance Improvements

- **Training Speed**: 10-50x faster with GPU
- **System Impact**: Zero local performance impact
- **Scalability**: Can train multiple models simultaneously
- **Reliability**: No local system crashes

### Development Benefits

- **Parallel Development**: Train while developing locally
- **Resource Isolation**: Training doesn't affect other work
- **Reproducibility**: Consistent training environment
- **Backup**: All models and results in S3

## Risk Mitigation

### Technical Risks

- **Network Issues**: Implement retry logic and local fallbacks
- **Instance Failures**: Use spot instances with auto-restart
- **Data Loss**: S3 versioning and multiple backups

### Cost Risks

- **Unexpected Usage**: Set up billing alerts
- **Instance Left Running**: Auto-termination scripts
- **Data Transfer**: Minimize uploads/downloads

## Next Steps

### Immediate Actions

1. **Fix Local TTT Bugs** (1 hour)

   - Resolve index errors
   - Implement proper loss functions
   - Add memory management

2. **AWS Setup** (2 hours)

   - Install and configure AWS CLI
   - Create key pair and security groups
   - Run setup script

3. **First Cloud Training** (1 hour)
   - Test with single task
   - Validate results
   - Optimize parameters

### Medium Term

1. **Batch Training Pipeline**

   - Train on multiple tasks
   - Automated result collection
   - Performance analysis

2. **Local Integration**
   - Use cloud-trained models locally
   - Implement hybrid approach
   - Validate performance gains

### Long Term

1. **Production Pipeline**
   - Automated training schedules
   - Cost optimization
   - Performance monitoring

## Commands to Run

### 1. Check AWS Setup

```bash
python scripts/aws_setup.py --action check
```

### 2. Create AWS Infrastructure

```bash
python scripts/aws_setup.py --action setup --key-name your-key-name --instance-type g4dn.xlarge
```

### 3. Run Cloud Training

```bash
# SSH to instance first
ssh -i your-key.pem ubuntu@<instance-ip>

# Run training
python3 scripts/aws_ttt_training.py --task_id train_0000 --epochs 10 --batch_size 4
```

### 4. Download Results

```bash
aws s3 sync s3://bucket-name/results ./aws_results
```

### 5. Cleanup (when done)

```bash
python scripts/aws_setup.py --action cleanup
```

## Success Metrics

### Technical Metrics

- **Training Time**: < 5 minutes per task (vs 30+ minutes locally)
- **System Impact**: 0% local CPU/memory usage during training
- **Success Rate**: > 5% on validation tasks
- **Cost Efficiency**: < $5 per training session

### Development Metrics

- **Iteration Speed**: Multiple training sessions per day
- **Reliability**: No local system crashes
- **Scalability**: Can train on multiple tasks simultaneously

## Conclusion

AWS integration provides a cost-effective solution to the TTT performance issues while enabling faster development iterations. The estimated $2-5 cost per training session is justified by the significant performance improvements and development efficiency gains.

The hybrid approach (local inference + cloud training) maximizes the benefits while minimizing costs and complexity.
