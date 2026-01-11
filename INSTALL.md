# Installation Guide for VKTRS

## Quick Start

### CPU Installation (Recommended for first-time setup)

```bash
pip install -r requirements.txt
```

### GPU Installation (Requires CUDA Toolkit)

GPU support requires CUDA to be installed on your system **before** installing PyTorch.

#### Step 1: Install CUDA Toolkit

1. Download CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Install CUDA Toolkit (version 11.8, 12.1, or 12.4 are recommended)
3. Verify installation: `nvcc --version`

#### Step 2: Install PyTorch with CUDA Support

Replace `<cuda_version>` with your CUDA version (e.g., cu118, cu121, cu124):

```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Example for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Example for CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

#### Step 3: Install VKTRS with GPU Optimizations

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

#### Step 4: Verify GPU Setup

Quick check with multi-line script:
```bash
python -c "
import torch
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
device_name = torch.cuda.get_device_name(0) if device_count > 0 else 'CPU'
print(f'CUDA available: {cuda_available}')
print(f'Device: {device_name}')
"
```

Or use the included validation script:
```bash
python validate_gpu_setup.py
```

Or use VKTRS's built-in check:

```bash
python -c "from vktrs.api import print_and_get_system_info; print_and_get_system_info()"
```

### Troubleshooting

#### "ModuleNotFoundError: No module named 'torch'" or CUDA unavailable

**Problem**: PyTorch installed without CUDA support or wrong CUDA version.

**Solution**:
1. Identify your CUDA version: `nvcc --version`
2. Check current torch install: `python -c "import torch; print(torch.__version__)"`
3. Reinstall torch with correct CUDA:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu<YOUR_VERSION>
   ```

#### xformers Installation Fails

**Problem**: xformers requires matching torch+CUDA versions.

**Solution**:
1. Ensure torch is installed first with correct CUDA
2. Try explicit xformers version:
   ```bash
   pip install xformers==0.0.20 --no-deps
   ```
3. (Optional) To skip GPU optimizations entirely, install only the base requirements:
   ```bash
   pip install -r requirements.txt
   ```
   (This skips `requirements-gpu.txt` and avoids installing `xformers`.)

#### "CUDA out of memory"

**Workaround**: Reduce model size or use CPU for initial testing:
```bash
CUDA_VISIBLE_DEVICES="" python your_script.py  # Force CPU mode
```

## Development Setup

Install development dependencies for testing and type checking:

```bash
pip install -r requirements-dev.txt
```

Run tests:
```bash
pytest tests/
```

## Python Version

VKTRS requires Python 3.8 or later. Check your version:

```bash
python --version
```

Upgrade if needed:
```bash
python3.11 -m pip install -r requirements.txt  # Explicit version
```

## Uninstall

```bash
pip uninstall vktrs torch transformers diffusers xformers
```
