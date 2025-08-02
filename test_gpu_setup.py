#!/usr/bin/env python3
"""
GPU Setup Test for ARC Challenge Africa
Tests GPU functionality for NVIDIA GeForce MX450
"""

import sys
import platform
import json

def test_gpu_detection():
    """Test GPU detection."""
    print("🔍 Testing GPU detection...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {device_count} device(s)")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   Device {i}: {props.name}")
                print(f"   Memory: {props.total_memory / 1024**3:.1f}GB")
                print(f"   Compute: {props.major}.{props.minor}")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_gpu_memory():
    """Test GPU memory operations."""
    print("\n💾 Testing GPU memory...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1024**3
        
        print(f"✅ Total GPU memory: {total_memory:.1f}GB")
        
        # Test memory allocation
        try:
            # Allocate a reasonable amount of memory (1GB)
            test_size = min(1024**3, props.total_memory // 2)
            x = torch.randn(test_size // 4, dtype=torch.float32, device='cuda')
            print(f"✅ Allocated {test_size / 1024**3:.1f}GB tensor")
            
            # Test computation
            y = torch.mm(x, x.t())
            print(f"✅ GPU computation successful: {y.shape}")
            
            # Clean up
            del x, y
            torch.cuda.empty_cache()
            print("✅ Memory cleanup successful")
            
            return True
        except RuntimeError as e:
            print(f"❌ Memory test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False

def test_mixed_precision():
    """Test mixed precision training."""
    print("\n⚡ Testing mixed precision...")
    
    try:
        import torch
        from torch.cuda.amp import autocast, GradScaler
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        # Test autocast
        with autocast():
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.mm(x, y)
            print(f"✅ Mixed precision computation: {z.dtype}")
        
        # Test GradScaler
        scaler = GradScaler()
        print("✅ GradScaler created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        return False

def test_gpu_config():
    """Test GPU configuration file."""
    print("\n⚙️ Testing GPU configuration...")
    
    try:
        with open("gpu_config.json", "r") as f:
            config = json.load(f)
        
        print("✅ GPU configuration loaded:")
        print(f"   Device: {config['gpu']['device']}")
        print(f"   Memory limit: {config['gpu']['memory_limit_gb']}GB")
        print(f"   Mixed precision: {config['gpu']['mixed_precision']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        
        return True
    except FileNotFoundError:
        print("❌ GPU configuration file not found")
        return False
    except Exception as e:
        print(f"❌ GPU configuration test failed: {e}")
        return False

def test_training_simulation():
    """Simulate a small training run."""
    print("\n🎯 Testing training simulation...")
    
    try:
        import torch
        import torch.nn as nn
        from torch.cuda.amp import autocast, GradScaler
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).cuda()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()
        
        # Simulate training
        for epoch in range(3):
            # Forward pass with mixed precision
            with autocast():
                x = torch.randn(32, 100, device='cuda')
                y = torch.randn(32, 10, device='cuda')
                output = model(x)
                loss = nn.functional.mse_loss(output, y)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        print("✅ Training simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization techniques."""
    print("\n🔧 Testing memory optimization...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        # Test gradient checkpointing
        from torch.utils.checkpoint import checkpoint
        
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(100, 100) for _ in range(5)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = checkpoint(layer, x)
                return x
        
        model = TestModel().cuda()
        x = torch.randn(16, 100, device='cuda')
        
        # Test forward pass with checkpointing
        output = model(x)
        print(f"✅ Gradient checkpointing: {output.shape}")
        
        # Test memory cleanup
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"✅ Memory cleanup: {allocated:.2f}GB allocated")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_system_info():
    """Display system information."""
    print("\n💻 System Information:")
    print(f"📍 Platform: {platform.system()}")
    print(f"🐍 Python: {sys.version}")
    print(f"🏗️  Architecture: {platform.architecture()[0]}")

def main():
    """Main GPU test function."""
    print("🧪 ARC Challenge Africa - GPU Setup Test (MX450)")
    print("=" * 60)
    
    # Display system info
    test_system_info()
    
    # Run tests
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("GPU Memory", test_gpu_memory),
        ("Mixed Precision", test_mixed_precision),
        ("GPU Configuration", test_gpu_config),
        ("Training Simulation", test_training_simulation),
        ("Memory Optimization", test_memory_optimization)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 GPU Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"✅ {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All GPU tests passed! Your MX450 is ready for training.")
        print("\n📋 You can now:")
        print("1. Start training your ARC models")
        print("2. Monitor GPU usage: python scripts/monitor_gpu.py")
        print("3. Deploy to AWS for larger training: cd scripts/aws/cdk && ./deploy.sh deploy")
        
        print("\n💡 MX450 Optimization Tips:")
        print("   - Use batch size 32 or smaller")
        print("   - Enable mixed precision (already configured)")
        print("   - Use gradient checkpointing for large models")
        print("   - Monitor memory usage during training")
        
    else:
        print("\n❌ Some GPU tests failed.")
        print("💡 Try running: python scripts/setup_gpu.py")
        print("💡 Or use CPU setup: python scripts/setup_local.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 