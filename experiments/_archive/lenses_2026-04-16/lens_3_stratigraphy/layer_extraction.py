"""
Per-layer hidden state extraction for Lens III — Layer Stratigraphy (§3.1.3).

Extracts hidden states from all transformer layers for each model, replicating
the native pooling strategy (CLS or Mean) so that the final layer matches the
precomputed embeddings from Lens I within float32 tolerance.

Design decisions: lens_3_stratigraphy/trace.md — D1 (extraction method)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

EMB_DIR = ROOT / "data" / "processed" / "embeddings"
CONFIG_PATH = ROOT / "models" / "config.yaml"
CACHE_DIR = Path(__file__).parent / "results" / "layer_vectors"


def _load_model_config(label: str) -> dict:
    """Look up a model entry in config.yaml by its label."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for group in ("weird", "sinic"):
        for entry in raw.get(group, []):
            if entry["label"] == label:
                return entry
    raise ValueError(f"Model '{label}' not found in {CONFIG_PATH}")


def _detect_pooling(model: SentenceTransformer) -> str:
    """Detect pooling mode from the SentenceTransformer Pooling module."""
    pooling_layer = model[1]
    if getattr(pooling_layer, "pooling_mode_cls_token", False):
        return "cls"
    return "mean"


def _pool_hidden_state(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> np.ndarray:
    """
    Pool a single hidden state tensor to (batch, dim).

    Parameters
    ----------
    hidden : (batch, seq_len, dim)
    attention_mask : (batch, seq_len)
    mode : "cls" or "mean"

    Returns
    -------
    (batch, dim) float32, L2-normalized
    """
    if mode == "cls":
        pooled = hidden[:, 0, :]
    else:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

    norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pooled = pooled / norms
    return pooled.cpu().numpy().astype(np.float32)


def extract_per_layer(
    model_label: str,
    terms: list[str],
    device: str = "cpu",
    batch_size: int = 32,
    cache: bool = True,
    force: bool = False,
    cache_label: str | None = None,
) -> np.ndarray:
    """
    Extract per-layer embeddings for all terms.

    Returns (N_terms, N_layers+1, dim) float32, L2-normalized per layer.
    Layer 0 = embedding layer output, layer L = final hidden state.

    Parameters
    ----------
    model_label : str
        Model label from config.yaml (e.g. "BGE-EN-large").
    terms : list[str]
        Input texts to encode. Must match model language (EN texts for WEIRD
        models, ZH texts for Sinic models).
    device : str
        PyTorch device string.
    batch_size : int
        Batch size for forward passes.
    cache : bool
        If True, load from / save to disk cache.
    force : bool
        If True, ignore cache and re-extract.
    cache_label : str | None
        Override cache filename (default: model_label). Use to store
        different term sets (e.g. core+control pool) separately.
    """
    effective_label = cache_label or model_label
    cache_path = CACHE_DIR / f"{effective_label}.npz"
    if cache and not force and cache_path.exists():
        data = np.load(cache_path)
        layers = data["layers"]
        if layers.shape[0] == len(terms):
            print(f"  [cache hit] {model_label} — {cache_path.name}")
            return layers

    config = _load_model_config(model_label)
    model_id = config["id"]
    instruction = config.get("instruction", "")

    model = SentenceTransformer(model_id, device=device)
    pooling_mode = _detect_pooling(model)
    print(f"  {model_label}: pooling={pooling_mode}, "
          f"layers={model[0].auto_model.config.num_hidden_layers}")

    auto_model = model[0].auto_model
    auto_model.config.output_hidden_states = True
    tokenizer = model[0].tokenizer

    if instruction:
        inputs = [instruction + t for t in terms]
    else:
        inputs = terms

    all_layers: list[np.ndarray] = []  # will become list of (N, dim) per layer

    n_layers_plus_1 = None

    for start in range(0, len(inputs), batch_size):
        batch_texts = inputs[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = auto_model(**encoded)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, dim)
        attention_mask = encoded["attention_mask"]

        if n_layers_plus_1 is None:
            n_layers_plus_1 = len(hidden_states)
            all_layers = [[] for _ in range(n_layers_plus_1)]

        for layer_idx, hs in enumerate(hidden_states):
            pooled = _pool_hidden_state(hs, attention_mask, pooling_mode)
            all_layers[layer_idx].append(pooled)

    # Stack batches per layer, then combine into (N, L+1, dim)
    stacked = [np.concatenate(layer_batches, axis=0) for layer_batches in all_layers]
    result = np.stack(stacked, axis=1)  # (N, L+1, dim)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, layers=result)
        print(f"  [cached] {cache_path.name}  shape={result.shape}")

    return result


def verify_final_layer(
    model_label: str,
    layer_vecs: np.ndarray,
    core_idx: list[int],
    atol: float = 1e-4,
) -> bool:
    """
    Sanity check: final layer of layer_vecs must match precomputed vectors.

    For models with a Normalize module (BGE-ZH, Dmeta-ZH), minor differences
    are expected due to tokenization path differences between direct
    auto_model inference and SentenceTransformer.encode(). A cosine
    similarity check (> 0.999) is used as a secondary pass.

    Parameters
    ----------
    model_label : str
    layer_vecs : (N_core, L+1, dim)
    core_idx : list[int]
        Indices into the full precomputed vectors array.
    atol : float
        Absolute tolerance for allclose check.

    Returns
    -------
    bool — True if match, False otherwise (with diagnostic printed).
    """
    from shared.embeddings import load_precomputed

    vecs_full, _ = load_precomputed(model_label, EMB_DIR)
    vecs_core = vecs_full[core_idx]
    final = layer_vecs[:, -1, :]

    if np.allclose(final, vecs_core, atol=atol):
        print(f"  [sanity] {model_label} — final layer matches precomputed (atol={atol})")
        return True

    # Secondary check: cosine similarity (tolerant to minor tokenization diffs)
    cos_sims = np.sum(final * vecs_core, axis=1)
    min_cos = float(cos_sims.min())
    mean_cos = float(cos_sims.mean())

    if min_cos > 0.999:
        print(f"  [sanity] {model_label} — final layer ~matches precomputed "
              f"(min_cos={min_cos:.4f}, mean_cos={mean_cos:.4f})")
        return True

    max_diff = float(np.abs(final - vecs_core).max())
    mean_diff = float(np.abs(final - vecs_core).mean())
    print(f"  [WARNING] {model_label} — final layer mismatch! "
          f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
          f"min_cos={min_cos:.4f}")
    return False
