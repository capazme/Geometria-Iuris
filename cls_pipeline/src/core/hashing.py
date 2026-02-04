"""
hashing.py — SHA256 hashing utilities for reproducibility tracking.

Provides deterministic hashing for configuration, input data, and embeddings
to ensure reproducibility across pipeline runs.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_hash(data: str | bytes) -> str:
    """
    Compute SHA256 hash of data.

    Parameters
    ----------
    data : str | bytes
        Data to hash.

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(path: Path | str) -> str:
    """
    Compute SHA256 hash of a file's contents.

    Parameters
    ----------
    path : Path | str
        Path to the file.

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    path = Path(path)
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_json_hash(data: dict | list) -> str:
    """
    Compute SHA256 hash of JSON-serializable data.

    Uses deterministic serialization (sorted keys, no whitespace).

    Parameters
    ----------
    data : dict | list
        JSON-serializable data.

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return compute_hash(serialized)


def compute_config_hash(config: Any) -> str:
    """
    Compute SHA256 hash of configuration.

    Extracts relevant fields from config dataclass for hashing.

    Parameters
    ----------
    config : Any
        Configuration object (dataclass with relevant fields).

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    # Extract hashable config components
    config_dict = {
        "version": getattr(config, "version", "unknown"),
        "random_seed": getattr(config, "random_seed", 42),
        "models": {},
        "experiments": {},
    }

    # Models
    if hasattr(config, "models"):
        for name, model in config.models.items():
            config_dict["models"][name] = {
                "name": model.name,
                "dimension": model.dimension,
            }

    # Experiments
    if hasattr(config, "experiments"):
        exp = config.experiments
        if hasattr(exp, "gromov_wasserstein"):
            gw = exp.gromov_wasserstein
            config_dict["experiments"]["gw"] = {
                "entropic_reg": gw.entropic_reg,
                "use_sinkhorn": gw.use_sinkhorn,
            }
        if hasattr(exp, "umap"):
            umap = exp.umap
            config_dict["experiments"]["umap"] = {
                "n_neighbors": umap.n_neighbors,
                "min_dist": umap.min_dist,
                "metric": umap.metric,
            }
        if hasattr(exp, "rsa"):
            config_dict["experiments"]["rsa"] = {
                "n_permutations": exp.rsa.n_permutations,
            }
        if hasattr(exp, "axes"):
            config_dict["experiments"]["axes"] = {
                "n_bootstrap": exp.axes.n_bootstrap,
            }
        if hasattr(exp, "clustering"):
            clust = exp.clustering
            config_dict["experiments"]["clustering"] = {
                "method": clust.method,
                "k_values": clust.k_values,
                "n_permutations": clust.n_permutations,
            }
        if hasattr(exp, "nda"):
            config_dict["experiments"]["nda"] = {
                "k": exp.nda.k,
                "n_permutations": exp.nda.n_permutations,
            }

    return compute_json_hash(config_dict)


def compute_array_hash(array: np.ndarray) -> str:
    """
    Compute SHA256 hash of a numpy array.

    Parameters
    ----------
    array : np.ndarray
        Array to hash.

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    # Use tobytes for deterministic serialization
    return compute_hash(array.tobytes())


def compute_texts_hash(texts: list[str]) -> str:
    """
    Compute SHA256 hash of a list of texts.

    Parameters
    ----------
    texts : list[str]
        List of strings.

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    combined = "\x00".join(texts)  # Null separator
    return compute_hash(combined)
