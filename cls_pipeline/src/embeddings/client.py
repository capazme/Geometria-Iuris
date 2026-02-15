"""
client.py — HuggingFace-based embedding client.

Replaces the API-based embedding extraction with local models using
sentence-transformers. Optimized for Apple Silicon (MPS) and 16GB RAM.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..core.config_loader import Config, ModelConfig
from ..core.device import DeviceManager, clear_device_cache
from .cache import EmbeddingCache

logger = logging.getLogger(__name__)

ModelType = Literal["weird", "sinic"]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize each row to unit L2 norm. Zero-norm rows remain unchanged."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class EmbeddingClient:
    """
    HuggingFace-based embedding client with caching and device management.

    Features:
    - Local model inference with sentence-transformers
    - Sequential model loading for memory efficiency (16GB RAM)
    - MPS/CUDA/CPU device support
    - Disk-based caching (.npy files)
    """

    def __init__(
        self,
        config: Config,
        device_manager: DeviceManager | None = None,
    ):
        """
        Initialize EmbeddingClient.

        Parameters
        ----------
        config : Config
            Pipeline configuration.
        device_manager : DeviceManager | None
            Device manager for PyTorch. Created if not provided.
        """
        self.config = config
        self.device_manager = device_manager or DeviceManager(
            config.device.preferred
        )
        self.device = self.device_manager.device
        self.batch_size = config.device.batch_size

        # Initialize cache
        cache_dir = config.get_absolute_path("cache")
        self.cache = EmbeddingCache(cache_dir)

        # Model storage (loaded on demand)
        self._models: dict[str, SentenceTransformer] = {}
        self._current_model: str | None = None

        # HuggingFace cache directory
        self.models_dir = config.get_absolute_path("models")

        logger.info(
            "EmbeddingClient initialized (device=%s, batch_size=%d)",
            self.device_manager.device_type,
            self.batch_size,
        )

    def _get_model_config(self, model_type: ModelType) -> tuple[str, int, str]:
        """
        Get model name, dimension, and input prefix for a model type.

        Parameters
        ----------
        model_type : ModelType
            Either "weird" or "sinic".

        Returns
        -------
        tuple[str, int, str]
            Model name, embedding dimension, and input prefix.
        """
        model_config = self.config.models[model_type]
        return model_config.name, model_config.dimension, model_config.prefix

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a model, unloading others first for memory efficiency.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.

        Returns
        -------
        SentenceTransformer
            Loaded model.
        """
        # If model already loaded, return it
        if model_name in self._models:
            logger.info("Modello già in memoria: %s", model_name)
            return self._models[model_name]

        # Unload current model to free memory (sequential loading strategy)
        if self._current_model and self._current_model != model_name:
            logger.info("Scaricamento modello precedente dalla memoria: %s", self._current_model)
            del self._models[self._current_model]
            self._models.pop(self._current_model, None)
            clear_device_cache()
            self._current_model = None
            logger.info("Memoria liberata, cache device svuotata")

        # Load new model
        logger.info("Caricamento modello: %s (device=%s) ...", model_name, self.device)

        # Set cache directory
        model = SentenceTransformer(
            model_name,
            cache_folder=str(self.models_dir),
            device=str(self.device),
            trust_remote_code=True,
        )

        self._models[model_name] = model
        self._current_model = model_name

        dim = model.get_sentence_embedding_dimension()
        logger.info(
            "Modello caricato: %s (dimensione=%d, device=%s)",
            model_name, dim, self.device,
        )
        return model

    def get_embeddings(
        self,
        texts: list[str],
        model_type: ModelType,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Get embeddings for texts using the specified model.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        model_type : ModelType
            Model to use: "weird" or "sinic".
        normalize : bool
            Whether to L2-normalize the output vectors.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_texts, dimension).
        """
        model_name, expected_dim, prefix = self._get_model_config(model_type)

        # Check cache first (cache key uses original texts, not prefixed)
        logger.info(
            "Richiesta embedding %s: %d testi (modello: %s)",
            model_type, len(texts), model_name,
        )
        cached = self.cache.get(texts, model_name)
        if cached is not None:
            logger.info(
                "Cache HIT per %s — %d testi, shape=%s (caricamento istantaneo)",
                model_type, len(texts), cached.shape,
            )
            return cached

        logger.info("Cache MISS per %s — generazione embedding necessaria", model_type)

        # Load model and generate embeddings
        model = self._load_model(model_name)

        # Apply model-specific prefix (e.g. E5 requires "query: " prefix)
        if prefix:
            encode_texts = [f"{prefix}{t}" for t in texts]
            logger.info(
                "Prefisso '%s' applicato a %d testi per %s",
                prefix, len(texts), model_type,
            )
        else:
            encode_texts = texts

        logger.info(
            "Encoding %d testi con %s (batch_size=%d) ...",
            len(texts), model_name, self.batch_size,
        )

        # Encode with batching
        embeddings = model.encode(
            encode_texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize separately
        )

        # Ensure float64 for numerical precision
        embeddings = embeddings.astype(np.float64)

        # Normalize if requested
        if normalize:
            embeddings = _l2_normalize(embeddings)
            logger.info("Vettori normalizzati L2 (norma unitaria)")

        # Verify dimension
        actual_dim = embeddings.shape[1]
        if actual_dim != expected_dim:
            logger.warning(
                "ATTENZIONE dimensione: attesa %d, ottenuta %d per %s",
                expected_dim, actual_dim, model_name,
            )

        # Cache results
        self.cache.set(texts, model_name, embeddings)

        logger.info(
            "Embedding generati e salvati in cache: shape=%s, dtype=%s",
            embeddings.shape, embeddings.dtype,
        )
        return embeddings

    def get_embeddings_for_model(
        self,
        texts: list[str],
        model_config: ModelConfig,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Get embeddings using an arbitrary ModelConfig.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        model_config : ModelConfig
            Model configuration with name, dimension, prefix.
        normalize : bool
            Whether to L2-normalize the output vectors.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_texts, dimension).
        """
        model_name = model_config.name
        prefix = model_config.prefix
        expected_dim = model_config.dimension

        logger.info(
            "Richiesta embedding per modello %s: %d testi",
            model_config.label, len(texts),
        )
        cached = self.cache.get(texts, model_name)
        if cached is not None:
            logger.info(
                "Cache HIT per %s — %d testi, shape=%s",
                model_config.label, len(texts), cached.shape,
            )
            return cached

        logger.info("Cache MISS per %s — generazione embedding necessaria", model_config.label)

        model = self._load_model(model_name)

        if prefix:
            encode_texts = [f"{prefix}{t}" for t in texts]
        else:
            encode_texts = texts

        embeddings = model.encode(
            encode_texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        embeddings = embeddings.astype(np.float64)

        if normalize:
            embeddings = _l2_normalize(embeddings)

        actual_dim = embeddings.shape[1]
        if actual_dim != expected_dim:
            logger.warning(
                "ATTENZIONE dimensione: attesa %d, ottenuta %d per %s",
                expected_dim, actual_dim, model_name,
            )

        self.cache.set(texts, model_name, embeddings)

        logger.info(
            "Embedding generati e salvati in cache: shape=%s (%s)",
            embeddings.shape, model_config.label,
        )
        return embeddings

    def get_embeddings_batch(
        self,
        text_batches: list[list[str]],
        model_type: ModelType,
        normalize: bool = True,
    ) -> list[np.ndarray]:
        """
        Get embeddings for multiple text batches.

        Parameters
        ----------
        text_batches : list[list[str]]
            List of text lists.
        model_type : ModelType
            Model to use.
        normalize : bool
            Whether to L2-normalize.

        Returns
        -------
        list[np.ndarray]
            List of embedding matrices.
        """
        return [
            self.get_embeddings(texts, model_type, normalize)
            for texts in text_batches
        ]

    def clear_cache(self) -> int:
        """Clear embedding cache."""
        return self.cache.clear()

    def unload_models(self) -> None:
        """Unload all models from memory."""
        for name in list(self._models.keys()):
            logger.info("Unloading model: %s", name)
            del self._models[name]
        self._models.clear()
        self._current_model = None
        clear_device_cache()
        logger.info("All models unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_models()
        except Exception:
            pass
