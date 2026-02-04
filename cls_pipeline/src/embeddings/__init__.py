"""Embedding extraction and caching for CLS Pipeline."""

from .client import EmbeddingClient, ModelType
from .cache import EmbeddingCache

__all__ = ["EmbeddingClient", "ModelType", "EmbeddingCache"]
