#!/usr/bin/env python3
"""
Test Local Setup for ARC Challenge Africa

This script tests the local development environment to ensure
everything is working correctly.
"""

import sys
import platform

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('boto3', 'Boto3'),
        ('sklearn', 'Scikit-learn')
    ]
    
    failed_imports = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_pytorch():
    """Test PyTorch functionality."""
    print("\n🎮 Testing PyTorch...")
    
    try:
        import torch
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations work: {z.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_aws_sdk():
    """Test AWS SDK functionality."""
    print("\n☁️ Testing AWS SDK...")
    
    try:
        import boto3
        
        # Test basic boto3 functionality
        session = boto3.Session()
        print(f"✅ Boto3 version: {boto3.__version__}")
        print(f"✅ AWS session created")
        
        # Test if credentials are configured
        try:
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            print(f"✅ AWS credentials configured")
            print(f"✅ Account: {identity['Account']}")
        except Exception as e:
            print(f"⚠️ AWS credentials not configured: {e}")
            print("   This is normal for local development")
        
        return True
    except Exception as e:
        print(f"❌ AWS SDK test failed: {e}")
        return False

def test_cost_estimation():
    """Test cost estimation functionality."""
    print("\n💰 Testing cost estimation...")
    
    try:
        # Import and test the cost estimation
        sys.path.append('scripts/aws')
        from quick_cost_estimate import calculate_training_costs
        
        costs = calculate_training_costs(168)  # 1 week
        print(f"✅ Cost estimation works")
        print(f"✅ 1 week training cost: ${costs['spot']['total_cost']:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Cost estimation test failed: {e}")
        return False

def test_system_info():
    """Display system information."""
    print("\n💻 System Information:")
    print(f"📍 Platform: {platform.system()}")
    print(f"🐍 Python: {sys.version}")
    print(f"🏗️  Architecture: {platform.architecture()[0]}")
    print(f"⚙️  Processor: {platform.processor()}")

def main():
    """Main test function."""
    print("🧪 ARC Challenge Africa - Local Setup Test")
    print("=" * 50)
    
    # Display system info
    test_system_info()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test PyTorch
    pytorch_ok = test_pytorch()
    
    # Test AWS SDK
    aws_ok = test_aws_sdk()
    
    # Test cost estimation
    cost_ok = test_cost_estimation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Package imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"✅ PyTorch: {'PASS' if pytorch_ok else 'FAIL'}")
    print(f"✅ AWS SDK: {'PASS' if aws_ok else 'FAIL'}")
    print(f"✅ Cost estimation: {'PASS' if cost_ok else 'FAIL'}")
    
    all_passed = imports_ok and pytorch_ok and aws_ok and cost_ok
    
    if all_passed:
        print("\n🎉 All tests passed! Your local setup is ready.")
        print("\n📋 You can now:")
        print("1. Run cost estimation: python scripts/aws/quick_cost_estimate.py")
        print("2. Deploy AWS infrastructure: cd scripts/aws/cdk && ./deploy.sh deploy")
        print("3. Start development on your ARC solver")
    else:
        print("\n❌ Some tests failed. Please check the setup.")
        print("💡 Try running: python scripts/setup_local.py")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 