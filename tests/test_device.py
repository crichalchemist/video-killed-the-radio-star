"""
Tests for CPU/GPU auto-detection functionality.

Validates that Phase 1 device detection works on both CPU and GPU systems.
"""

import pytest
import torch


def test_torch_device_available():
    """Verify torch can detect device availability and move tensors."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a tensor and move it to the device
    tensor = torch.randn(10, 10).to(device)
    # Verify tensor is on the correct device
    assert tensor.device.type == device.type


def test_device_selection_logic():
    """Test the device selection logic used throughout vktrs."""
    from unittest.mock import patch
    
    # Test when CUDA is available
    with patch('torch.cuda.is_available', return_value=True):
        from vktrs import asr
        # Force module reload to pick up patched value
        import importlib
        importlib.reload(asr)
        # The DEVICE constant should be cuda
        assert 'cuda' in str(asr.DEVICE)
    
    # Test when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        importlib.reload(asr)
        assert 'cpu' in str(asr.DEVICE)


def test_device_none_default():
    """Test that None defaults to auto-detection (used in HfHelper)."""
    from vktrs.hf import HfHelper
    
    # Create HfHelper with device=None
    helper = HfHelper(device=None)
    
    # Should auto-detect to cuda or cpu based on availability
    expected = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert helper.device == expected


class TestDeviceCompatibility:
    """Test device compatibility across different scenarios."""

    def test_cpu_mode_always_available(self):
        """CPU mode should work on all systems."""
        device = torch.device('cpu')
        assert device.type == 'cpu'

        # Test tensor creation works
        tensor = torch.randn(10, 10).to(device)
        assert tensor.device.type == 'cpu'

    @pytest.mark.skipif(torch.cuda.is_available() is False, reason="CUDA not available")
    def test_cuda_conditional(self):
        """CUDA mode only if available."""
        device = torch.device('cuda')
        assert device.type == 'cuda'

        # Test tensor creation works
        tensor = torch.randn(10, 10).to(device)
        assert tensor.device.type == 'cuda'
