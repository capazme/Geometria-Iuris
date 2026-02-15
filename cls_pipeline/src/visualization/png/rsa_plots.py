"""
rsa_plots.py — Visualizzazioni PNG per Esperimento 1 (RSA).

Genera:
1. Clustered heatmap con seaborn.clustermap (riordina per similarità)
2. Inter-domain matrix K×K (distanze medie tra domini)
3. RDM correlation scatter con density colormap
"""
# ─── Heatmap clusterizzate per leggibilità ──────────────────────────
# Con 394 termini, una heatmap lineare è illeggibile. La clustermap
# riordina righe e colonne in base alla similarità (clustering
# gerarchico), rivelando la struttura a blocchi della matrice.
# La matrice inter-dominio riduce a K×K, mostrando le relazioni
# tra categorie giuridiche.
# ─────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage

from ..config import (
    COLORS,
    COLORMAPS,
    DOMAIN_COLORS,
    FIGURE_SIZES,
    FONT_SIZES,
    significance_label,
)
from .common import (
    create_figure,
    save_figure,
    setup_style,
    truncate_labels,
    add_colorbar,
    compute_inter_domain_matrix,
    get_domain_order,
)

logger = logging.getLogger(__name__)


def plot_clustered_heatmap(
    rdm_weird: np.ndarray,
    rdm_sinic: np.ndarray,
    labels: list[str],
    spearman_r: float,
    p_value: float,
    output_dir: Path,
    domains: Optional[list[str]] = None,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera clustered heatmap side-by-side per le due RDM.

    La clustermap riordina i termini per similarità, rivelando
    la struttura a blocchi. I domini sono indicati da colori laterali.

    Parameters
    ----------
    rdm_weird, rdm_sinic : np.ndarray
        Matrici di dissimilarità N×N.
    labels : list[str]
        Etichette dei termini.
    spearman_r : float
        Correlazione Spearman tra le RDM.
    p_value : float
        P-value del test di Mantel.
    output_dir : Path
        Directory di output.
    domains : list[str], optional
        Domini dei termini per colorazione laterale.
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
    short_labels = truncate_labels(labels, max_len=12)

    # Calcola linkage comune per ordinamento coerente
    # Usa la media delle due RDM per un ordinamento "neutro"
    combined = (rdm_weird + rdm_sinic) / 2
    # Condensed distance matrix per linkage
    condensed = combined[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="ward")

    # Crea figura con due heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Range indipendenti per ogni matrice
    triu_w = rdm_weird[np.triu_indices(n, k=1)]
    triu_s = rdm_sinic[np.triu_indices(n, k=1)]

    # Prepara row colors se abbiamo i domini
    row_colors = None
    if domains is not None:
        row_colors = [DOMAIN_COLORS.get(d, "#CCCCCC") for d in domains]

    # WEIRD heatmap
    sns.heatmap(
        rdm_weird,
        ax=ax1,
        cmap=COLORMAPS["distance"],
        vmin=triu_w.min(),
        vmax=triu_w.max(),
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6, "label": "Cosine distance"},
    )
    ax1.set_title(
        f"WEIRD RDM\n(range {triu_w.min():.3f}–{triu_w.max():.3f})",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    # Sinic heatmap
    sns.heatmap(
        rdm_sinic,
        ax=ax2,
        cmap=COLORMAPS["distance"],
        vmin=triu_s.min(),
        vmax=triu_s.max(),
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6, "label": "Cosine distance"},
    )
    ax2.set_title(
        f"Sinic RDM\n(range {triu_s.min():.3f}–{triu_s.max():.3f})",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    fig.suptitle(
        f"Representational Dissimilarity Matrices (N={n})\n"
        f"Spearman ρ = {spearman_r:.4f}, {significance_label(p_value)}",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return save_figure(fig, output_dir, "rsa_heatmaps", formats, dpi)


def plot_inter_domain_matrix(
    rdm_weird: np.ndarray,
    rdm_sinic: np.ndarray,
    domains: list[str],
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera matrice di distanza media inter-dominio K×K.

    Riduce la complessità N×N → K×K aggregando per dominio giuridico.
    Mostra quali categorie sono semanticamente più vicine/lontane.

    Parameters
    ----------
    rdm_weird, rdm_sinic : np.ndarray
        Matrici di dissimilarità N×N.
    domains : list[str]
        Domini dei termini.
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

    # Calcola matrici inter-dominio
    inter_w, domain_labels = compute_inter_domain_matrix(rdm_weird, domains)
    inter_s, _ = compute_inter_domain_matrix(rdm_sinic, domains)

    k = len(domain_labels)

    # Formatta nomi domini per display
    display_labels = [d.replace("_", " ").title() for d in domain_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # WEIRD inter-domain
    im1 = ax1.imshow(inter_w, cmap=COLORMAPS["distance"], aspect="auto")
    ax1.set_xticks(range(k))
    ax1.set_yticks(range(k))
    ax1.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=FONT_SIZES["tick"])
    ax1.set_yticklabels(display_labels, fontsize=FONT_SIZES["tick"])
    ax1.set_title("WEIRD", fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Mean cosine distance")

    # Annotazioni numeriche
    for i in range(k):
        for j in range(k):
            ax1.text(
                j, i, f"{inter_w[i, j]:.2f}",
                ha="center", va="center",
                fontsize=FONT_SIZES["small"],
                color="white" if inter_w[i, j] > inter_w.mean() else "black",
            )

    # Sinic inter-domain
    im2 = ax2.imshow(inter_s, cmap=COLORMAPS["distance"], aspect="auto")
    ax2.set_xticks(range(k))
    ax2.set_yticks(range(k))
    ax2.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=FONT_SIZES["tick"])
    ax2.set_yticklabels(display_labels, fontsize=FONT_SIZES["tick"])
    ax2.set_title("Sinic", fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Mean cosine distance")

    # Annotazioni numeriche
    for i in range(k):
        for j in range(k):
            ax2.text(
                j, i, f"{inter_s[i, j]:.2f}",
                ha="center", va="center",
                fontsize=FONT_SIZES["small"],
                color="white" if inter_s[i, j] > inter_s.mean() else "black",
            )

    fig.suptitle(
        f"Inter-Domain Distance Matrix ({k} domains)",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return save_figure(fig, output_dir, "rsa_inter_domain", formats, dpi)


def plot_rdm_correlation(
    rdm_weird: np.ndarray,
    rdm_sinic: np.ndarray,
    spearman_r: float,
    p_value: float,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera scatter plot della correlazione tra RDM con density colormap.

    Ogni punto è una coppia di termini. L'asse X mostra la distanza
    nello spazio WEIRD, l'asse Y nello spazio Sinic.
    Usa hexbin per gestire l'overplotting con 77k+ coppie.

    Parameters
    ----------
    rdm_weird, rdm_sinic : np.ndarray
        Matrici di dissimilarità N×N.
    spearman_r : float
        Correlazione Spearman.
    p_value : float
        P-value del test di Mantel.
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

    n = rdm_weird.shape[0]
    triu = np.triu_indices(n, k=1)
    vec_w = rdm_weird[triu]
    vec_s = rdm_sinic[triu]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES.square)

    # Hexbin per density (gestisce overplotting)
    hb = ax.hexbin(
        vec_w, vec_s,
        gridsize=50,
        cmap="Blues",
        mincnt=1,
        edgecolors="none",
    )

    # Colorbar
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("Count", fontsize=FONT_SIZES["tick"])

    # Linea diagonale (identità)
    lims = [0, max(vec_w.max(), vec_s.max()) * 1.05]
    ax.plot(lims, lims, "--", color="grey", linewidth=1, alpha=0.7, label="Identity")

    # Fit lineare per riferimento
    z = np.polyfit(vec_w, vec_s, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(vec_w.min(), vec_w.max(), 100)
    ax.plot(x_fit, p(x_fit), "-", color=COLORS["sinic"], linewidth=1.5, alpha=0.8, label="Linear fit")

    ax.set_xlabel("Cosine distance (WEIRD)", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel("Cosine distance (Sinic)", fontsize=FONT_SIZES["axis"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    ax.legend(loc="lower right", fontsize=FONT_SIZES["legend"])

    ax.set_title(
        f"RDM Correlation (N pairs = {len(vec_w):,})\n"
        f"Spearman ρ = {spearman_r:.4f}, {significance_label(p_value)}",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "rsa_correlation", formats, dpi)


def plot_null_distribution(
    null_dist: np.ndarray,
    observed: float,
    p_value: float,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera istogramma della distribuzione nulla del Mantel test.

    Mostra la distribuzione delle correlazioni Spearman sotto H₀
    (nessuna corrispondenza tra le RDM), con il valore osservato
    evidenziato come linea verticale.

    Parameters
    ----------
    null_dist : np.ndarray
        Distribuzione nulla (array 1D di r permutati).
    observed : float
        Valore osservato di Spearman r.
    p_value : float
        P-value del test di Mantel.
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

    fig, ax = plt.subplots(figsize=FIGURE_SIZES.double_column)

    # Istogramma distribuzione nulla
    ax.hist(
        null_dist,
        bins=60,
        density=True,
        color=COLORS["grid"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        label="Null distribution",
    )

    # Linea verticale per il valore osservato
    ax.axvline(
        observed,
        color=COLORS["weird"],
        linewidth=2.5,
        linestyle="--",
        label=f"Observed r = {observed:.4f}",
    )

    # Annotazione p-value
    ax.annotate(
        f"p = {p_value:.4f}" if p_value >= 0.001 else "p < 0.001",
        xy=(observed, ax.get_ylim()[1] * 0.9),
        xytext=(observed + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05,
                ax.get_ylim()[1] * 0.9),
        fontsize=FONT_SIZES["annotation"] + 1,
        fontweight="bold",
        color=COLORS["weird"],
        arrowprops=dict(arrowstyle="->", color=COLORS["weird"], lw=1.2),
    )

    ax.set_xlabel("Spearman r (permuted)", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel("Density", fontsize=FONT_SIZES["axis"])
    ax.legend(loc="upper left", fontsize=FONT_SIZES["legend"])

    ax.set_title(
        f"Mantel Test Null Distribution ({len(null_dist):,} permutations)\n"
        f"Observed r = {observed:.4f}, {significance_label(p_value)}",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "rsa_null_distribution", formats, dpi)
