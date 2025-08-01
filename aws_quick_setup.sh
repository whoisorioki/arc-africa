#!/bin/bash
# Quick Setup Script for ARC Challenge on AWS
# Run this on your AWS instance

echo "🚀 Setting up ARC Challenge environment on AWS..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
sudo apt install -y git htop tmux python3-pip python3-venv nvidia-cuda-toolkit

# Check GPU
echo "🎮 Checking GPU availability..."
nvidia-smi

# Create project directory
echo "📁 Setting up project directory..."
cd /home/ubuntu
mkdir -p arc-africa
cd arc-africa

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv arc-env
source arc-env/bin/activate

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "📚 Installing additional packages..."
pip install numpy pandas scikit-learn matplotlib seaborn tqdm wandb

# Test CUDA
echo "🧪 Testing CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo "✅ Setup complete! Your environment is ready for ARC Challenge training."
echo "💡 Next steps:"
echo "   1. Upload your project files"
echo "   2. Run: source arc-env/bin/activate"
echo "   3. Start training with: python aws_train_enhanced.py" 