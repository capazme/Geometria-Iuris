"""
umap_plots.py — Visualizzazioni PNG per UMAP (Supplementary).

Genera:
1. UMAP scatter con smart label placement usando adjustText
2. Pannelli separati WEIRD | Sinic per confronto
"""
# ─── Smart label placement ──────────────────────────────────────────
# Con 394 termini, le etichette si sovrappongono completamente.
# La libreria adjustText sposta iterativamente le etichette per
# minimizzare le sovrapposizioni. Per ulteriore chiarezza, si possono
# etichettare solo i punti "isolati" (distanti dai vicini).
# ─────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from ..config import (
    COLORS,
    DOMAIN_COLORS,
    FIGURE_SIZES,
    FONT_SIZES,
    UMAP_LABEL_ISOLATION_THRESHOLD,
)
from .common import (
    create_figure,
    save_figure,
    setup_style,
    truncate_labels,
)

logger = logging.getLogger(__name__)


def plot_umap_smart_labels(
    coords_weird: np.ndarray,
    coords_sinic: np.ndarray,
    labels: list[str],
    output_dir: Path,
    domains: Optional[list[str]] = None,
    label_threshold: float = UMAP_LABEL_ISOLATION_THRESHOLD,
    max_labels: int = 50,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera UMAP scatter con smart label placement.

    Usa adjustText per evitare sovrapposizioni. Etichetta solo i punti
    più isolati per mantenere la leggibilità.

    Parameters
    ----------
    coords_weird, coords_sinic : np.ndarray
        Coordinate UMAP 2D (N×2).
    labels : list[str]
        Etichette dei termini.
    output_dir : Path
        Directory di output.
    domains : list[str], optional
        Domini per colorazione.
    label_threshold : float
        Distanza minima dal vicino più prossimo per mostrare etichetta.
    max_labels : int
        Numero massimo di etichette da mostrare.
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

    n = len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES.side_by_side)

    # Colori
    if domains is not None:
        colors = [DOMAIN_COLORS.get(d, "#CCCCCC") for d in domains]
    else:
        colors = [COLORS["weird"]] * n

    # ─── WEIRD panel ────────────────────────────────────────────────
    ax1.scatter(
        coords_weird[:, 0],
        coords_weird[:, 1],
        c=colors,
        alpha=0.6,
        s=30,
        edgecolors="white",
        linewidth=0.3,
    )
    ax1.set_title("WEIRD embedding space", fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    ax1.set_xlabel("UMAP 1", fontsize=FONT_SIZES["axis"])
    ax1.set_ylabel("UMAP 2", fontsize=FONT_SIZES["axis"])

    # Smart labels per WEIRD
    _add_smart_labels(ax1, coords_weird, labels, label_threshold, max_labels)

    # ─── Sinic panel ────────────────────────────────────────────────
    ax2.scatter(
        coords_sinic[:, 0],
        coords_sinic[:, 1],
        c=colors,
        alpha=0.6,
        s=30,
        edgecolors="white",
        linewidth=0.3,
    )
    ax2.set_title("Sinic embedding space", fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    ax2.set_xlabel("UMAP 1", fontsize=FONT_SIZES["axis"])
    ax2.set_ylabel("UMAP 2", fontsize=FONT_SIZES["axis"])

    # Smart labels per Sinic
    _add_smart_labels(ax2, coords_sinic, labels, label_threshold, max_labels)

    # Legenda domini se disponibili
    if domains is not None:
        _add_domain_legend(fig, domains)

    fig.suptitle(
        f"UMAP Projection (N={n} terms)",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return save_figure(fig, output_dir, "umap_smart_labels", formats, dpi)


def _add_smart_labels(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: list[str],
    threshold: float,
    max_labels: int,
):
    """
    Aggiunge etichette smart usando adjustText.

    Etichetta solo i punti più isolati (distanti dai vicini).
    """
    n = len(labels)

    # Calcola distanze tra tutti i punti
    dists = cdist(coords, coords)
    np.fill_diagonal(dists, np.inf)

    # Distanza dal vicino più prossimo
    min_dists = dists.min(axis=1)

    # Normalizza distanze (relativo al range)
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    scale = (x_range + y_range) / 2
    normalized_dists = min_dists / scale

    # Seleziona punti da etichettare (più isolati)
    # Prendi i top max_labels per distanza dal vicino
    label_indices = np.argsort(normalized_dists)[::-1][:max_labels]

    # Filtra per soglia
    label_indices = [i for i in label_indices if normalized_dists[i] > threshold * 0.5]

    if not label_indices:
        # Se nessun punto supera la soglia, prendi comunque alcuni
        label_indices = np.argsort(normalized_dists)[::-1][:min(10, max_labels)]

    short_labels = truncate_labels(labels, max_len=12)

    try:
        from adjustText import adjust_text

        texts = []
        for i in label_indices:
            t = ax.text(
                coords[i, 0],
                coords[i, 1],
                short_labels[i],
                fontsize=FONT_SIZES["small"],
                alpha=0.9,
            )
            texts.append(t)

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
            expand_points=(1.5, 1.5),
            force_points=(0.5, 0.5),
        )

    except ImportError:
        # Fallback senza adjustText
        logger.warning("adjustText non disponibile, uso annotazioni semplici")
        for i in label_indices:
            ax.annotate(
                short_labels[i],
                (coords[i, 0], coords[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=FONT_SIZES["small"],
                alpha=0.8,
            )


def _add_domain_legend(fig: plt.Figure, domains: list[str]):
    """Aggiunge legenda dei domini."""
    from matplotlib.patches import Patch

    unique_domains = sorted(set(domains))
    legend_elements = [
        Patch(
            facecolor=DOMAIN_COLORS.get(d, "#CCCCCC"),
            label=d.replace("_", " ").title(),
        )
        for d in unique_domains
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(5, len(unique_domains)),
        fontsize=FONT_SIZES["small"],
        bbox_to_anchor=(0.5, 0.01),
    )


def plot_umap_combined(
    coords_weird: np.ndarray,
    coords_sinic: np.ndarray,
    labels: list[str],
    output_dir: Path,
    domains: Optional[list[str]] = None,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera UMAP combinato (entrambi gli spazi sovrapposti).

    Utile per vedere la "migrazione" dei termini tra gli spazi.
    Ogni termine ha un punto WEIRD e uno Sinic, connessi da una linea.

    Parameters
    ----------
    coords_weird, coords_sinic : np.ndarray
        Coordinate UMAP 2D (N×2).
    labels : list[str]
        Etichette dei termini.
    output_dir : Path
        Directory di output.
    domains : list[str], optional
        Domini per colorazione.
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

    # Per overlay, serve fare UMAP su dati concatenati
    # Qui assumiamo che coords_weird e coords_sinic siano già in scala comparabile

    n = len(labels)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES.square)

    # WEIRD points
    ax.scatter(
        coords_weird[:, 0],
        coords_weird[:, 1],
        c=COLORS["weird"],
        alpha=0.5,
        s=30,
        marker="o",
        label="WEIRD",
    )

    # Sinic points
    ax.scatter(
        coords_sinic[:, 0],
        coords_sinic[:, 1],
        c=COLORS["sinic"],
        alpha=0.5,
        s=30,
        marker="s",
        label="Sinic",
    )

    # Connetti stesso termine con linea
    for i in range(n):
        ax.plot(
            [coords_weird[i, 0], coords_sinic[i, 0]],
            [coords_weird[i, 1], coords_sinic[i, 1]],
            color="grey",
            alpha=0.2,
            linewidth=0.5,
        )

    ax.set_xlabel("UMAP 1", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel("UMAP 2", fontsize=FONT_SIZES["axis"])
    ax.legend(loc="upper right", fontsize=FONT_SIZES["legend"])

    ax.set_title(
        f"UMAP Combined Projection (N={n} terms)\n"
        "Lines connect same term across spaces",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "umap_combined", formats, dpi)
