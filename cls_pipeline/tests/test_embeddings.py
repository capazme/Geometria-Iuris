"""
test_embeddings.py — Unit tests for embedding client and cache.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We'll test the cache directly without needing the full client


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    @pytest.fixture
    def cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cache_key_deterministic(self, cache_dir):
        """Cache keys should be deterministic for same inputs."""
        from src.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir)

        texts = ["hello", "world"]
        model = "test-model"

        key1 = cache._cache_key(texts, model)
        key2 = cache._cache_key(texts, model)

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex

    def test_cache_key_different_for_different_inputs(self, cache_dir):
        """Different inputs should produce different keys."""
        from src.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir)

        key1 = cache._cache_key(["hello"], "model")
        key2 = cache._cache_key(["world"], "model")
        key3 = cache._cache_key(["hello"], "other-model")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_miss_returns_none(self, cache_dir):
        """Cache miss should return None."""
        from src.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir)

        result = cache.get(["test"], "model")
        assert result is None

    def test_cache_set_and_get(self, cache_dir):
        """Should be able to set and retrieve cached embeddings."""
        from src.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir)

        texts = ["hello", "world"]
        model = "test-model"
        embeddings = np.random.randn(2, 768).astype(np.float64)

        cache.set(texts, model, embeddings)
        result = cache.get(texts, model)

        assert result is not None
        np.testing.assert_array_almost_equal(result, embeddings)

    def test_cache_clear(self, cache_dir):
        """Cache clear should remove all cached files."""
        from src.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir)

        # Add some cached items
        for i in range(5):
            cache.set([f"text{i}"], "model", np.random.randn(1, 768))

        assert cache.stats()["count"] == 5

        count = cache.clear()
        assert count == 5
        assert cache.stats()["count"] == 0

    def test_cache_stats(self, cache_dir):
        """Cache stats should return correct information."""
        from src.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir)

        stats = cache.stats()
        assert stats["count"] == 0
        assert stats["total_size_bytes"] == 0

        # Add item
        embeddings = np.random.randn(10, 1024).astype(np.float64)
        cache.set(["test"], "model", embeddings)

        stats = cache.stats()
        assert stats["count"] == 1
        assert stats["total_size_bytes"] > 0


class TestL2Normalize:
    """Tests for L2 normalization."""

    def test_normalize_unit_vectors(self):
        """Normalized vectors should have unit norm."""
        from src.embeddings.client import _l2_normalize

        vectors = np.random.randn(10, 768)
        normalized = _l2_normalize(vectors)

        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10))

    def test_normalize_preserves_direction(self):
        """Normalization should preserve vector direction."""
        from src.embeddings.client import _l2_normalize

        vectors = np.array([[3.0, 4.0]])  # 3-4-5 triangle
        normalized = _l2_normalize(vectors)

        expected = np.array([[0.6, 0.8]])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_zero_vector(self):
        """Zero vectors should remain zero (or handled gracefully)."""
        from src.embeddings.client import _l2_normalize

        vectors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        normalized = _l2_normalize(vectors)

        # Zero vector stays zero
        assert normalized[0, 0] == 0.0
        # Non-zero vector is normalized
        assert np.isclose(np.linalg.norm(normalized[1]), 1.0)
