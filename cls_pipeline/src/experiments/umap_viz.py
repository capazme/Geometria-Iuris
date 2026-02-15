"""
umap_viz.py — UMAP Visualization (Manifold Learning).

Reduces high-dimensional embeddings to 2D for visualization using UMAP,
allowing visual comparison of latent space structures.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap

logger = logging.getLogger(__name__)


@dataclass
class UMAPResult:
    """Result of UMAP dimensionality reduction."""
    coords_2d: np.ndarray
    model_labels: np.ndarray  # 0 = WEIRD, 1 = Sinic
    term_labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        weird_mask = self.model_labels == 0
        sinic_mask = self.model_labels == 1

        weird_coords = [
            {
                "label": self.term_labels[i],
                "x": float(self.coords_2d[i, 0]),
                "y": float(self.coords_2d[i, 1]),
            }
            for i in np.where(weird_mask)[0]
        ]

        sinic_coords = [
            {
                "label": self.term_labels[i],
                "x": float(self.coords_2d[i, 0]),
                "y": float(self.coords_2d[i, 1]),
            }
            for i in np.where(sinic_mask)[0]
        ]

        return {
            "coordinates": {
                "weird": weird_coords,
                "sinic": sinic_coords,
            },
            "total_points": len(self.term_labels),
        }


def umap_reduce(
    vectors_weird: np.ndarray,
    vectors_sinic: np.ndarray,
    labels_weird: list[str],
    labels_sinic: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> UMAPResult:
    """
    Reduce concatenated embeddings to 2D using UMAP.

    Parameters
    ----------
    vectors_weird : np.ndarray
        WEIRD embeddings, shape (n, d1).
    vectors_sinic : np.ndarray
        Sinic embeddings, shape (m, d2).
    labels_weird : list[str]
        Labels for WEIRD vectors.
    labels_sinic : list[str]
        Labels for Sinic vectors.
    n_neighbors : int
        UMAP n_neighbors parameter.
    min_dist : float
        UMAP min_dist parameter.
    metric : str
        Distance metric for UMAP.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    UMAPResult
        2D coordinates, model labels, and term labels.

    Notes
    -----
    If embedding dimensions differ, zero-padding is applied to align them.
    """
    # Align dimensions via zero-padding if needed
    d_max = max(vectors_weird.shape[1], vectors_sinic.shape[1])
    if vectors_weird.shape[1] < d_max:
        pad = np.zeros((vectors_weird.shape[0], d_max - vectors_weird.shape[1]))
        vectors_weird = np.hstack([vectors_weird, pad])
    if vectors_sinic.shape[1] < d_max:
        pad = np.zeros((vectors_sinic.shape[0], d_max - vectors_sinic.shape[1]))
        vectors_sinic = np.hstack([vectors_sinic, pad])

    combined = np.vstack([vectors_weird, vectors_sinic])
    model_labels = np.array(
        [0] * vectors_weird.shape[0] + [1] * vectors_sinic.shape[0]
    )
    term_labels = labels_weird + labels_sinic

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
        init="spectral",
    )
    coords_2d = reducer.fit_transform(combined)

    logger.info("UMAP reduction completed: %d points -> 2D", combined.shape[0])

    return UMAPResult(
        coords_2d=coords_2d,
        model_labels=model_labels,
        term_labels=term_labels,
    )


def plot_umap(
    result: UMAPResult,
    title: str = "UMAP - Latent Space Comparison",
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 300,
    weird_label: str = "WEIRD Model (BGE-EN)",
    sinic_label: str = "Sinic Model (BGE-ZH)",
) -> Path:
    """
    Generate UMAP scatter plot with model-colored points.

    Parameters
    ----------
    result : UMAPResult
        UMAP reduction result.
    title : str
        Plot title.
    output_dir : Path | None
        Output directory. Uses current directory if None.
    figsize : tuple[int, int]
        Figure size in inches.
    dpi : int
        Plot resolution.
    weird_label : str
        Label for WEIRD model in legend.
    sinic_label : str
        Label for Sinic model in legend.

    Returns
    -------
    Path
        Path to saved PNG file.
    """
    out = output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=figsize)
    palette = {0: "#4C72B0", 1: "#DD8452"}
    label_map = {0: weird_label, 1: sinic_label}

    for model_id in [0, 1]:
        mask = result.model_labels == model_id
        ax.scatter(
            result.coords_2d[mask, 0],
            result.coords_2d[mask, 1],
            c=palette[model_id],
            label=label_map[model_id],
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5,
        )
        # Text labels
        for idx in np.where(mask)[0]:
            ax.annotate(
                result.term_labels[idx],
                (result.coords_2d[idx, 0], result.coords_2d[idx, 1]),
                fontsize=7,
                alpha=0.8,
                textcoords="offset points",
                xytext=(5, 5),
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best")

    fig.tight_layout()

    path = out / "umap_comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    logger.info("UMAP plot saved: %s", path)
    return path
