#!/usr/bin/env python3
"""
GPU setup validation script for VKTRS.

Checks if torch was installed with CUDA support after pip install.
Run this after installing requirements to verify GPU setup.

Usage:
    python validate_gpu_setup.py
"""

import sys


def validate_gpu_setup():
    """Validate that torch is properly installed with CUDA support."""
    print("=" * 70)
    print("🔍 VKTRS GPU Setup Validation")
    print("=" * 70)
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/<cuda_version>")
        print("  See INSTALL.md for detailed instructions")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("✓ CUDA available")
    else:
        print("✗ CUDA not available")

    if not cuda_available:
        print()
        print("⚠️  WARNING: CUDA not available!")
        print()
        print("This usually means:")
        print("  1. CUDA Toolkit not installed (visit https://developer.nvidia.com/cuda-downloads)")
        print("  2. torch installed without CUDA support (wrong pip index URL used)")
        print("  3. GPU drivers not installed or not compatible")
        print()
        print("To fix:")
        print("  1. Install CUDA Toolkit (check nvcc --version)")
        print("  2. Reinstall torch with CUDA:")
        print("     pip uninstall torch")
        print("     pip install torch --index-url https://download.pytorch.org/whl/<cuda_version>")
        print("     (Replace <cuda_version> with cu118, cu121, cu124, etc.)")
        print()
        print("See INSTALL.md for complete instructions")
        return False
    
    # Get GPU info and test each device
    try:
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU devices: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  [{i}] {gpu_name} ({gpu_memory:.1f} GB)")

            # Test tensor creation on each GPU device
            try:
                device = torch.device(f'cuda:{i}')
                test_tensor = torch.randn(10, 10, device=device)
                print(f"      ✓ Test tensor created on cuda:{i}")
                # Clean up
                del test_tensor
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            except Exception as e:
                print(f"      ✗ Error on cuda:{i}: {e}")
                return False
    except Exception as e:
        print(f"✗ Error accessing GPU devices: {e}")
        return False
    
    # Check for xformers
    try:
        import xformers
        print(f"✓ xformers installed: {xformers.__version__}")
    except ImportError:
        print("ℹ️  xformers not installed (optional, for 20-30% memory savings)")
        print("   Install with: pip install -r requirements-gpu.txt")
    
    print()
    print("=" * 70)
    print("✓ GPU setup validated successfully!")
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = validate_gpu_setup()
    sys.exit(0 if success else 1)
