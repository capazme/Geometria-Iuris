"""
clustering_plots.py — Visualizzazioni PNG per Esperimento 4 (Clustering).

Genera:
1. Truncated dendrogram (solo primi N livelli)
2. FM index bar chart con linea soglia
"""
# ─── Dendrogrammi troncati per leggibilità ──────────────────────────
# Con 394 foglie, un dendrogramma completo è illeggibile. Il troncamento
# mostra solo i primi 30 cluster, indicando il numero di foglie
# in ciascuno. Questo rivela la struttura gerarchica senza il rumore.
# ─────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from ..config import (
    COLORS,
    FIGURE_SIZES,
    FONT_SIZES,
    DENDROGRAM_MAX_LEAVES,
    significance_label,
    significance_marker,
)
from .common import (
    create_figure,
    save_figure,
    setup_style,
    truncate_labels,
)

logger = logging.getLogger(__name__)


def plot_truncated_dendrogram(
    linkage_weird: np.ndarray,
    linkage_sinic: np.ndarray,
    labels: list[str],
    fm_results: list[dict],
    output_dir: Path,
    max_leaves: int = DENDROGRAM_MAX_LEAVES,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera dendrogrammi troncati side-by-side.

    I dendrogrammi sono troncati per mostrare solo i primi `max_leaves`
    cluster, rendendo la struttura gerarchica leggibile.

    Parameters
    ----------
    linkage_weird, linkage_sinic : np.ndarray
        Matrici di linkage (output di scipy.cluster.hierarchy.linkage).
    labels : list[str]
        Etichette dei termini.
    fm_results : list[dict]
        Risultati FM per annotazione.
    output_dir : Path
        Directory di output.
    max_leaves : int
        Numero massimo di foglie da mostrare.
    dpi : int
        Risoluzione.
    formats : list[str]
        Formati di output.

    Returns
    -------
    list[Path]
        Percorsi dei file salvati.
    """
    if formats is None:
        formats = ["png"]

    setup_style()

    n_terms = len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES.side_by_side)

    # WEIRD dendrogram
    dendrogram(
        linkage_weird,
        ax=ax1,
        truncate_mode="lastp",
        p=max_leaves,
        leaf_rotation=90,
        leaf_font_size=FONT_SIZES["small"],
        show_contracted=True,
        above_threshold_color=COLORS["weird"],
        color_threshold=0,
    )
    ax1.set_title("WEIRD", fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    ax1.set_xlabel("Cluster (n = number of terms)", fontsize=FONT_SIZES["tick"])
    ax1.set_ylabel("Distance", fontsize=FONT_SIZES["axis"])

    # Sinic dendrogram
    dendrogram(
        linkage_sinic,
        ax=ax2,
        truncate_mode="lastp",
        p=max_leaves,
        leaf_rotation=90,
        leaf_font_size=FONT_SIZES["small"],
        show_contracted=True,
        above_threshold_color=COLORS["sinic"],
        color_threshold=0,
    )
    ax2.set_title("Sinic", fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    ax2.set_xlabel("Cluster (n = number of terms)", fontsize=FONT_SIZES["tick"])
    ax2.set_ylabel("Distance", fontsize=FONT_SIZES["axis"])

    # Titolo con FM
    fm_str = " | ".join(
        f"k={r['k']}: FM={r['fm_index']:.3f}{significance_marker(r['p_value'])}"
        for r in fm_results
    )

    fig.suptitle(
        f"Hierarchical Clustering Dendrograms (N={n_terms}, truncated to {max_leaves} clusters)\n"
        f"{fm_str}",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return save_figure(fig, output_dir, "clustering_dendrograms", formats, dpi)


def plot_fm_chart(
    fm_results: list[dict],
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera bar chart dell'indice FM per diversi valori di k.

    L'indice FM (Fowlkes-Mallows) misura la similarità tra le partizioni
    ottenute dai due dendrogrammi. FM > 0.5 indica similarità strutturale.

    Parameters
    ----------
    fm_results : list[dict]
        Lista con keys: k, fm_index, p_value.
    output_dir : Path
        Directory di output.
    dpi : int
        Risoluzione.
    formats : list[str]
        Formati di output.

    Returns
    -------
    list[Path]
        Percorsi dei file salvati.
    """
    if formats is None:
        formats = ["png"]

    setup_style()

    ks = [r["k"] for r in fm_results]
    fms = [r["fm_index"] for r in fm_results]
    ps = [r["p_value"] for r in fm_results]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES.double_column)

    # Colori in base a significatività
    colors = [
        COLORS["significant"] if p < 0.05 else COLORS["not_significant"]
        for p in ps
    ]

    bars = ax.bar(range(len(ks)), fms, color=colors, alpha=0.8, edgecolor="white", linewidth=1)

    # Annotazioni
    for i, (bar, fm, p) in enumerate(zip(bars, fms, ps)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{fm:.3f}{significance_marker(p)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["tick"],
            fontweight="bold" if p < 0.05 else "normal",
        )

    # Linea soglia 0.5
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="FM = 0.5")

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=FONT_SIZES["tick"])
    ax.set_ylabel("Fowlkes-Mallows Index", fontsize=FONT_SIZES["axis"])
    ax.set_xlabel("Number of clusters", fontsize=FONT_SIZES["axis"])
    ax.set_ylim(0, 1.1)

    ax.set_title(
        "Fowlkes-Mallows Index by Number of Clusters",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["significant"], label="Significant (p < 0.05)"),
        Patch(facecolor=COLORS["not_significant"], label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=FONT_SIZES["legend"])

    fig.tight_layout()

    return save_figure(fig, output_dir, "clustering_fm_chart", formats, dpi)
