"""
Tests for TSP (Traveling Salesman Problem) frame reordering.

Validates Phase 2 optimizations: perceptual hashing and OR-Tools solver.
"""

import pytest
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine, pdist, squareform

try:
    from vktrs.tsp import (
        compute_perceptual_hash,
        compute_frame_hashes,
        tsp_sort,
        tsp_permute_frames
    )
    VKTRS_AVAILABLE = True
except ImportError:
    VKTRS_AVAILABLE = False


@pytest.mark.skipif(not VKTRS_AVAILABLE, reason="vktrs package not installed")
class TestPerceptualHashing:
    """Test perceptual hashing functionality."""

    def test_hash_generation(self):
        """Test that perceptual hashes are generated correctly."""
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='red')

        # Compute hash
        hash_vec = compute_perceptual_hash(img, hash_size=8)

        # Should be 64-dimensional (8x8 hash)
        assert hash_vec.shape == (64,)
        assert hash_vec.dtype == np.float32

    def test_similar_images_similar_hashes(self):
        """Similar images should have similar perceptual hashes."""
        img1 = Image.new('RGB', (512, 512), color='red')
        img2 = Image.new('RGB', (512, 512), color=(255, 10, 10))  # Slightly different red

        hash1 = compute_perceptual_hash(img1)
        hash2 = compute_perceptual_hash(img2)

        # Compute similarity (cosine distance)
        distance = cosine(hash1, hash2)

        # Should be very similar (low distance)
        assert distance < 0.5

    def test_different_images_different_hashes(self):
        """Very different images should have different hashes."""
        img1 = Image.new('RGB', (512, 512), color='red')
        img2 = Image.new('RGB', (512, 512), color='blue')

        hash1 = compute_perceptual_hash(img1)
        hash2 = compute_perceptual_hash(img2)

        # Should be different
        assert not np.array_equal(hash1, hash2)
        
        # Compute cosine distance to verify perceptual dissimilarity
        PERCEPTUAL_DISTANCE_THRESHOLD = 0.35  # Tunable threshold for test stability
        distance = cosine(hash1, hash2)
        # Very different colors should have substantial distance
        assert distance > PERCEPTUAL_DISTANCE_THRESHOLD

    def test_batch_hashing(self):
        """Test hashing multiple frames at once."""
        frames = [
            Image.new('RGB', (512, 512), color='red'),
            Image.new('RGB', (512, 512), color='green'),
            Image.new('RGB', (512, 512), color='blue')
        ]

        hashes = compute_frame_hashes(frames, hash_size=8)

        # Should have 3 hashes, each 64-dimensional
        assert hashes.shape == (3, 64)


@pytest.mark.skipif(not VKTRS_AVAILABLE, reason="vktrs package not installed")
class TestTSPSolver:
    """Test TSP solver functionality."""

    def test_small_sequence(self):
        """Test TSP on a small sequence."""
        # Create 5 simple frames
        frames = [
            np.full((64, 64, 3), i * 50, dtype=np.uint8)
            for i in range(5)
        ]

        # Solve TSP
        permutation, distance = tsp_sort(frames, method='perceptual')

        # Should return valid permutation
        assert len(permutation) == 5
        assert set(permutation) == set(range(5))
        assert distance >= 0

    @pytest.mark.timeout(30)
    def test_medium_sequence(self):
        """Test TSP on a medium sequence (would have failed before optimization)."""
        # Create 25 frames (old implementation would hang/crash)
        frames = [
            np.full((64, 64, 3), i * 10, dtype=np.uint8)
            for i in range(25)
        ]

        # This should complete in reasonable time with OR-Tools
        permutation, distance = tsp_sort(frames, method='perceptual')

        assert len(permutation) == 25
        assert set(permutation) == set(range(25))

        # Verify solution quality: compute baseline (original order) distance
        hashes = compute_frame_hashes(frames, hash_size=8)
        dmat = squareform(pdist(hashes, metric='cosine'))
        
        # Compute original ordering cycle distance (includes return to start)
        original_distance = sum(dmat[i, (i+1) % len(frames)] for i in range(len(frames)))
        
        # TSP solution should be better than or equal to original order
        assert distance <= original_distance

    def test_backward_compatibility(self):
        """Test that old API still works."""
        frames = [
            np.full((64, 64, 3), i * 50, dtype=np.uint8)
            for i in range(5)
        ]

        # Old function should still work
        reordered = tsp_permute_frames(frames)

        assert len(reordered) == 5
        assert all(isinstance(f, np.ndarray) for f in reordered)


@pytest.mark.skipif(not VKTRS_AVAILABLE, reason="vktrs package not installed")
class TestMemoryEfficiency:
    """Test that perceptual hashing reduces memory usage."""

    def test_hash_vs_pixel_memory(self):
        """Verify perceptual hashing is more memory-efficient than pixel comparison."""
        # 512x512 RGB image
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img)

        # Pixel representation: 512 * 512 * 3 = 786,432 dimensions
        pixel_size = img.size

        # Perceptual hash: 8 * 8 = 64 dimensions
        hash_vec = compute_perceptual_hash(pil_img, hash_size=8)
        hash_size = hash_vec.size

        # Perceptual hash should be ~10,000× smaller
        reduction_factor = pixel_size / hash_size
        assert reduction_factor > 10000
