"""
cache.py — Embedding cache management.

Provides disk-based caching for embeddings using .npy files to avoid
redundant model inference.
"""

import logging
from pathlib import Path

import numpy as np

from ..core.hashing import compute_hash

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Disk-based cache for embedding vectors.

    Stores embeddings as .npy files with deterministic keys based on
    model name and input texts.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize EmbeddingCache.

        Parameters
        ----------
        cache_dir : Path
            Directory for cache files.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, texts: list[str], model_name: str) -> str:
        """
        Generate deterministic cache key.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        model_name : str
            Model identifier.

        Returns
        -------
        str
            SHA256 hash string.
        """
        payload = f"{model_name}::{':'.join(texts)}"
        return compute_hash(payload)

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.npy"

    def get(self, texts: list[str], model_name: str) -> np.ndarray | None:
        """
        Retrieve cached embeddings if available.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        model_name : str
            Model identifier.

        Returns
        -------
        np.ndarray | None
            Cached embeddings or None if not found.
        """
        key = self._cache_key(texts, model_name)
        path = self._cache_path(key)

        if path.exists():
            try:
                embeddings = np.load(path)
                logger.debug(
                    "Cache hit for %s (%d texts): %s",
                    model_name, len(texts), key[:12]
                )
                return embeddings
            except Exception as e:
                logger.warning("Failed to load cache %s: %s", key[:12], e)
                return None

        return None

    def set(
        self,
        texts: list[str],
        model_name: str,
        embeddings: np.ndarray,
    ) -> None:
        """
        Store embeddings in cache.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        model_name : str
            Model identifier.
        embeddings : np.ndarray
            Embedding vectors to cache.
        """
        key = self._cache_key(texts, model_name)
        path = self._cache_path(key)

        try:
            np.save(path, embeddings)
            logger.debug(
                "Cached embeddings for %s (%d texts): %s",
                model_name, len(texts), key[:12]
            )
        except Exception as e:
            logger.warning("Failed to save cache %s: %s", key[:12], e)

    def clear(self) -> int:
        """
        Clear all cached embeddings.

        Returns
        -------
        int
            Number of files deleted.
        """
        count = 0
        for path in self.cache_dir.glob("*.npy"):
            try:
                path.unlink()
                count += 1
            except Exception as e:
                logger.warning("Failed to delete %s: %s", path.name, e)

        logger.info("Cleared %d cached embeddings", count)
        return count

    def stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns
        -------
        dict[str, int]
            Statistics including count and total size.
        """
        files = list(self.cache_dir.glob("*.npy"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
