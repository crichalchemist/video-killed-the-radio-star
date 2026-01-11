"""
Tests for API client singleton pattern.

Validates Phase 1 optimization: singleton API client for batch operations.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Optional imports with type-checker friendly fallbacks
torch: Any
torch_available = False
try:  # pragma: no cover - import resolution only
    import torch  # type: ignore[import]  # noqa: F401

    torch_available = True
except ImportError:  # pragma: no cover - handled by skip markers
    torch = None

get_api_client: Any
validate_environment: Any
print_system_info: Any
vktrs_available = False
try:  # pragma: no cover - import resolution only
    from vktrs.api import get_api_client as _get_api_client, print_system_info, validate_environment  # type: ignore

    get_api_client = _get_api_client  # type: ignore[assignment]
    vktrs_available = True
except ImportError:  # pragma: no cover - handled by skip markers
    get_api_client = None
    validate_environment = None
    print_system_info = None


@pytest.mark.skipif(not vktrs_available, reason="vktrs package not installed")
class TestSingletonPattern:
    """Test that API client uses singleton pattern."""

    @patch.dict(os.environ, {'STABILITY_KEY': 'test_key_12345'})
    @patch('vktrs.api.client.StabilityInference')
    def test_singleton_returns_same_instance(self, mock_stability: MagicMock):
        """Verify get_api_client returns the same instance on multiple calls."""
        assert get_api_client is not None

        # Mock the StabilityInference constructor
        mock_instance = MagicMock()
        mock_stability.return_value = mock_instance

        # Clear the lru_cache to start fresh
        get_api_client.cache_clear()

        # First call
        client1 = get_api_client()

        # Second call
        client2 = get_api_client()

        # Should be the same instance
        assert client1 is client2

        # StabilityInference should only be called once
        assert mock_stability.call_count == 1

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_error(self):
        """Verify that missing API key raises ValueError."""
        # Clear cache
        assert get_api_client is not None

        get_api_client.cache_clear()

        with pytest.raises(ValueError, match="STABILITY_KEY not set"):
            get_api_client()


@pytest.mark.skipif(not vktrs_available, reason="vktrs package not installed")
class TestEnvironmentValidation:
    """Test environment validation functions."""

    def test_validate_environment_returns_dict(self):
        """Verify validate_environment returns capability dict."""
        assert validate_environment is not None

        caps = validate_environment()

        # Should return a dict with expected keys
        assert isinstance(caps, dict)
        assert 'cuda_available' in caps
        assert 'stability_api' in caps

    @pytest.mark.skipif(not torch_available, reason="torch not installed")
    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        assert validate_environment is not None

        caps = validate_environment()

        # Should match torch.cuda.is_available()
        assert torch is not None
        assert caps['cuda_available'] == torch.cuda.is_available()

    @patch.dict(os.environ, {'STABILITY_KEY': 'test_key'})
    def test_api_key_detection(self):
        """Test API key detection."""
        assert validate_environment is not None

        caps = validate_environment()

        assert caps['stability_api'] is True

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_detection(self):
        """Test missing API key detection."""
        assert validate_environment is not None

        caps = validate_environment()

        assert caps['stability_api'] is False

    def test_print_system_info_runs(self, capsys: pytest.CaptureFixture[str]):
        """Test that print_system_info executes and prints expected content."""
        # Should not raise any exceptions
        assert print_system_info is not None

        print_system_info()
        
        # Capture and verify output
        captured = capsys.readouterr()
        output = captured.out.lower()
        
        # Verify key system info is printed
        assert 'cuda' in output  # Should mention CUDA availability
        assert 'stability api' in output  # Should mention API/keys explicitly


@pytest.mark.skipif(not vktrs_available, reason="vktrs package not installed")
class TestPerformanceImprovement:
    """Test that singleton pattern improves performance."""

    @patch.dict(os.environ, {'STABILITY_KEY': 'test_key_12345'})
    @patch('vktrs.api.client.StabilityInference')
    def test_batch_operation_efficiency(self, mock_stability: MagicMock):
        """Verify singleton pattern reduces overhead in batch operations."""
        mock_instance = MagicMock()
        mock_stability.return_value = mock_instance

        assert get_api_client is not None

        # Clear cache
        get_api_client.cache_clear()

        # Simulate batch operation (100 calls)
        for _ in range(100):
            get_api_client()

        # StabilityInference constructor should only be called once
        # (not 100 times, which would add 15-30 seconds overhead)
        assert mock_stability.call_count == 1

