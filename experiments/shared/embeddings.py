"""
Unified embedding client for Geometria Iuris experiments.

Wraps sentence-transformers to provide a consistent interface for all six
models defined in models/config.yaml. Handles model loading (lazy, in-memory),
optional instruction prepending, L2 normalization, and SHA-256-keyed disk cache.

All output arrays have shape (N, D), dtype float32, and are L2-normalized.
After normalization, cosine(u, v) = u · v — a property exploited by all five
Lens analyses.

Reference
---------
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
siamese BERT-networks. EMNLP 2019. https://arxiv.org/abs/1908.10084
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precomputed embedding loader (populated by shared/precompute.py)
# ---------------------------------------------------------------------------

def load_precomputed(
    model_label: str,
    embeddings_dir: str | Path,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load precomputed embeddings and the shared term index.

    Parameters
    ----------
    model_label : str
        Short model label as defined in config.yaml (e.g. "BGE-EN-large").
    embeddings_dir : str or Path
        Directory produced by ``shared/precompute.py``.
        Typically ``experiments/data/processed/embeddings/``.

    Returns
    -------
    vectors : np.ndarray
        Float32 array of shape ``(N, dim)``, L2-normalized.
        ``vectors[i]`` is the embedding for ``index[i]``.
    index : list[dict]
        Ordered list of term records, each with keys
        ``en``, ``zh_canonical``, ``domain``, ``tier``.

    Raises
    ------
    FileNotFoundError
        If the model directory or index file does not exist.
    """
    embeddings_dir = Path(embeddings_dir)
    index_path = embeddings_dir / "index.json"
    vec_path = embeddings_dir / model_label / "vectors.npy"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            "Run `python shared/precompute.py` first."
        )
    if not vec_path.exists():
        raise FileNotFoundError(
            f"Vectors not found: {vec_path}\n"
            f"Run `python shared/precompute.py --models {model_label}` first."
        )

    index: list[dict] = json.load(index_path.open(encoding="utf-8"))
    vectors: np.ndarray = np.load(vec_path)
    return vectors, index


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single embedding model, as defined in config.yaml."""

    id: str            # HuggingFace model identifier
    label: str         # Short human-readable label (used in plots and tables)
    lang: str          # "en" (WEIRD tradition) or "zh" (Sinic tradition)
    dim: int           # Output embedding dimension
    instruction: str   # Prefix prepended to each text before encoding (empty = none)
    note: str          # Free-text annotation (not used programmatically)


class EmbeddingClient:
    """
    Unified embedding client for all models in the Geometria Iuris pipeline.

    Loads model specifications from config.yaml and exposes a single ``embed``
    method that returns L2-normalized embedding arrays. Models are loaded lazily
    and kept in memory; computed embeddings are stored on disk as .npy files
    keyed by a SHA-256 digest of the (model_id, texts) pair.

    Parameters
    ----------
    config_path : str or Path
        Path to models/config.yaml.
    cache_dir : str or Path or None
        Directory for on-disk embedding cache. Defaults to the path in config.yaml
        (resolved relative to config_path). Pass None to disable caching.
    device : str
        PyTorch device string ("cpu" or "cuda").
    batch_size : int or None
        Override the batch_size from config.yaml.

    Examples
    --------
    >>> client = EmbeddingClient("experiments/models/config.yaml")
    >>> vecs = client.embed(["mens rea", "habeas corpus"], "BAAI/bge-large-en-v1.5")
    >>> vecs.shape
    (2, 1024)
    >>> import numpy as np; np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    True
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        config_path = Path(config_path).resolve()
        with config_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Parse model specifications from all tradition groups
        self._specs: dict[str, ModelSpec] = {}
        self._groups: dict[str, list[str]] = {}
        for group in ("weird", "sinic", "bilingual"):
            ids: list[str] = []
            for entry in raw.get(group, []):
                spec = ModelSpec(
                    id=entry["id"],
                    label=entry["label"],
                    lang=entry["lang"],
                    dim=entry["dim"],
                    instruction=entry.get("instruction", ""),
                    note=entry.get("note", ""),
                )
                self._specs[spec.id] = spec
                ids.append(spec.id)
            self._groups[group] = ids

        # Embedding settings
        emb_cfg = raw.get("embedding", {})
        self._normalize: bool = emb_cfg.get("normalize", True)
        # Device priority: constructor arg > config.yaml > auto-detect
        self._device: str = device or emb_cfg.get("device") or self._detect_device()
        self._batch_size: int = batch_size or emb_cfg.get("batch_size", 32)

        # Disk cache setup
        cache_cfg = raw.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True)
        if cache_dir is not None:
            resolved_cache: Path | None = Path(cache_dir)
        elif cache_enabled:
            raw_dir = cache_cfg.get("dir", "")
            if raw_dir:
                # Path in config is relative to the project root
                resolved_cache = config_path.parent.parent / raw_dir
            else:
                resolved_cache = (
                    config_path.parent.parent
                    / "data"
                    / "processed"
                    / "embeddings_cache"
                )
        else:
            resolved_cache = None

        self._cache_dir: Path | None = resolved_cache
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory model registry (populated on first use)
        self._loaded: dict[str, SentenceTransformer] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        model_id: str,
        *,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Return L2-normalized embeddings for ``texts`` using ``model_id``.

        Parameters
        ----------
        texts : list[str]
            Input texts. Ordering is preserved in the output array.
        model_id : str
            HuggingFace model identifier as listed in config.yaml.
        use_cache : bool
            Read from and write to the on-disk cache.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(texts), dim)``, L2-normalized.
            cosine(u, v) = u · v holds after normalization.
        """
        if model_id not in self._specs:
            raise ValueError(
                f"Unknown model '{model_id}'.\n"
                f"Available: {list(self._specs)}"
            )

        if use_cache and self._cache_dir is not None:
            cached = self._load_cache(model_id, texts)
            if cached is not None:
                logger.debug("Cache hit: %s (%d texts)", model_id, len(texts))
                return cached

        model = self._get_model(model_id)
        spec = self._specs[model_id]

        # Prepend instruction if the model requires it
        inputs = (
            [spec.instruction + t for t in texts]
            if spec.instruction
            else texts
        )

        logger.info("Encoding %d texts with %s ...", len(texts), spec.label)
        vecs: np.ndarray = model.encode(
            inputs,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(texts) > 100,
        )
        vecs = vecs.astype(np.float32)

        if use_cache and self._cache_dir is not None:
            self._save_cache(model_id, texts, vecs)

        return vecs

    def embed_all(
        self,
        texts: list[str],
        *,
        group: Literal["weird", "sinic", "all"] = "all",
        use_cache: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Embed ``texts`` with every model in ``group``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from model_id to embedding array of shape ``(N, dim)``.
        """
        model_ids = (
            list(self._specs) if group == "all" else self._groups[group]
        )
        return {
            mid: self.embed(texts, mid, use_cache=use_cache)
            for mid in model_ids
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def all_specs(self) -> list[ModelSpec]:
        """All model specs (WEIRD first, then Sinic)."""
        return list(self._specs.values())

    @property
    def weird_specs(self) -> list[ModelSpec]:
        """WEIRD model specs in config order."""
        return [self._specs[mid] for mid in self._groups["weird"]]

    @property
    def sinic_specs(self) -> list[ModelSpec]:
        """Sinic model specs in config order."""
        return [self._specs[mid] for mid in self._groups["sinic"]]

    @property
    def bilingual_specs(self) -> list[ModelSpec]:
        """Bilingual control model specs in config order."""
        return [self._specs[mid] for mid in self._groups.get("bilingual", [])]

    def group_ids(self, group: Literal["weird", "sinic", "bilingual"]) -> list[str]:
        """Model IDs for a given tradition group."""
        return list(self._groups.get(group, []))

    def unload_model(self, model_id: str) -> None:
        """Remove a loaded model from memory (useful on RAM-constrained machines)."""
        if model_id in self._loaded:
            del self._loaded[model_id]
            logger.info("Unloaded model '%s'", model_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_model(self, model_id: str) -> SentenceTransformer:
        if model_id not in self._loaded:
            logger.info("Loading model '%s' ...", model_id)
            self._loaded[model_id] = SentenceTransformer(
                model_id, device=self._device, trust_remote_code=True,
            )
        return self._loaded[model_id]

    def _cache_key(self, model_id: str, texts: list[str]) -> str:
        """
        Deterministic cache filename for a (model, texts) pair.

        Key = SHA-256 digest (truncated to 20 hex chars) of the JSON-serialized
        payload. The model label is prepended for human readability.
        Order of texts is significant: embed(["a","b"]) and embed(["b","a"])
        produce different cache files.
        """
        payload = json.dumps(
            {"model": model_id, "texts": texts},
            ensure_ascii=False,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        label = self._specs[model_id].label.replace("/", "_")
        return f"{label}_{digest}.npy"

    def _load_cache(self, model_id: str, texts: list[str]) -> np.ndarray | None:
        path = self._cache_dir / self._cache_key(model_id, texts)  # type: ignore[operator]
        if path.exists():
            return np.load(path)
        return None

    def _save_cache(
        self, model_id: str, texts: list[str], vecs: np.ndarray
    ) -> None:
        path = self._cache_dir / self._cache_key(model_id, texts)  # type: ignore[operator]
        np.save(path, vecs)
        logger.debug("Cached: %s", path.name)
