"""
gw_plots.py — Visualizzazioni PNG per Esperimento 2 (Gromov-Wasserstein).

Genera:
1. Histogram distribuzione masse nel transport plan
2. Top-K alignments bar chart (coppie con peso massimo)
3. Transport plan thresholded (solo valori > percentile)
"""
# ─── Visualizzazione del transport plan ─────────────────────────────
# Il transport plan GW 394×394 è per lo più "rumore" (masse piccole
# distribuite uniformemente). Le informazioni utili sono:
# (a) la distribuzione delle masse (quanto è concentrato l'allineamento?)
# (b) le coppie con massa massima (quali termini sono allineati?)
# ─────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import (
    COLORS,
    COLORMAPS,
    FIGURE_SIZES,
    FONT_SIZES,
    GW_TOP_K_ALIGNMENTS,
    GW_TRANSPORT_PERCENTILE,
    significance_label,
)
from .common import (
    create_figure,
    save_figure,
    setup_style,
    truncate_labels,
)

logger = logging.getLogger(__name__)


def plot_transport_histogram(
    transport_plan: np.ndarray,
    gw_distance: float,
    p_value: float,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera istogramma della distribuzione delle masse nel transport plan.

    Un piano uniforme (tutte le masse uguali) indica che l'allineamento
    è indeterminato. Una distribuzione con picco indica allineamenti forti.

    Parameters
    ----------
    transport_plan : np.ndarray
        Piano di trasporto N×N.
    gw_distance : float
        Distanza GW.
    p_value : float
        P-value del test di permutazione.
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

    # Flatten e rimuovi diagonale (self-alignment)
    n = transport_plan.shape[0]
    masses = transport_plan.flatten()

    # Calcola statistiche
    uniform_mass = 1.0 / (n * n)
    mean_mass = masses.mean()
    max_mass = masses.max()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES.double_column)

    # Istogramma (log scale per vedere la coda)
    counts, bins, patches = ax.hist(
        masses,
        bins=100,
        color=COLORS["weird"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    # Linea verticale per massa uniforme
    ax.axvline(
        uniform_mass,
        color=COLORS["sinic"],
        linestyle="--",
        linewidth=2,
        label=f"Uniform mass = {uniform_mass:.2e}",
    )

    # Linea verticale per massimo
    ax.axvline(
        max_mass,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Max mass = {max_mass:.2e}",
    )

    ax.set_xlabel("Transport mass", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel("Frequency", fontsize=FONT_SIZES["axis"])
    ax.set_yscale("log")

    ax.legend(loc="upper right", fontsize=FONT_SIZES["legend"])

    ax.set_title(
        f"Transport Plan Mass Distribution (N={n})\n"
        f"GW distance = {gw_distance:.6f}, {significance_label(p_value)}",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    # Annotazione concentrazione
    gini = _gini_coefficient(masses)
    ax.text(
        0.95, 0.85,
        f"Gini coefficient = {gini:.3f}\n"
        f"(0 = uniform, 1 = concentrated)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZES["annotation"],
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "gw_mass_histogram", formats, dpi)


def plot_top_alignments(
    transport_plan: np.ndarray,
    labels: list[str],
    output_dir: Path,
    top_k: int = GW_TOP_K_ALIGNMENTS,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera bar chart dei top-K alignments (coppie con massa massima).

    Parameters
    ----------
    transport_plan : np.ndarray
        Piano di trasporto N×N.
    labels : list[str]
        Etichette dei termini.
    output_dir : Path
        Directory di output.
    top_k : int
        Numero di alignments da mostrare.
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

    n = transport_plan.shape[0]

    # Trova top-K coppie per massa
    flat_idx = np.argsort(transport_plan.flatten())[::-1][:top_k]
    top_pairs = []
    for idx in flat_idx:
        i, j = divmod(idx, n)
        mass = transport_plan[i, j]
        top_pairs.append((labels[i], labels[j], mass))

    # Crea labels per il grafico
    pair_labels = [f"{truncate_labels([p[0]], 12)[0]} ↔ {truncate_labels([p[1]], 12)[0]}"
                   for p in top_pairs]
    masses = [p[2] for p in top_pairs]

    # Altezza dinamica in base a top_k
    fig_height = max(6, top_k * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(pair_labels))
    bars = ax.barh(y_pos, masses, color=COLORS["weird"], alpha=0.8, edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=FONT_SIZES["tick"])
    ax.invert_yaxis()  # Top alignment in cima

    ax.set_xlabel("Transport mass", fontsize=FONT_SIZES["axis"])
    ax.set_title(
        f"Top {top_k} Term Alignments by Transport Mass",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    # Aggiungi valori numerici
    for i, (bar, mass) in enumerate(zip(bars, masses)):
        ax.text(
            bar.get_width() + masses[0] * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{mass:.2e}",
            va="center",
            fontsize=FONT_SIZES["small"],
        )

    fig.tight_layout()

    return save_figure(fig, output_dir, "gw_top_alignments", formats, dpi)


def plot_transport_thresholded(
    transport_plan: np.ndarray,
    labels: list[str],
    output_dir: Path,
    percentile: float = GW_TRANSPORT_PERCENTILE,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera heatmap del transport plan con soglia (solo valori significativi).

    Maschera i valori sotto il percentile specificato per evidenziare
    solo gli allineamenti più forti.

    Parameters
    ----------
    transport_plan : np.ndarray
        Piano di trasporto N×N.
    labels : list[str]
        Etichette dei termini.
    output_dir : Path
        Directory di output.
    percentile : float
        Percentile soglia (default: 99).
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

    n = transport_plan.shape[0]
    threshold = np.percentile(transport_plan, percentile)

    # Crea matrice mascherata
    masked_plan = np.where(transport_plan >= threshold, transport_plan, np.nan)

    # Trova righe/colonne con almeno un valore sopra soglia
    row_mask = np.any(~np.isnan(masked_plan), axis=1)
    col_mask = np.any(~np.isnan(masked_plan), axis=0)

    # Subset della matrice
    subset_plan = transport_plan[np.ix_(row_mask, col_mask)]
    subset_labels_row = [labels[i] for i in np.where(row_mask)[0]]
    subset_labels_col = [labels[i] for i in np.where(col_mask)[0]]

    # Limita dimensione per leggibilità
    max_size = 50
    if subset_plan.shape[0] > max_size or subset_plan.shape[1] > max_size:
        # Prendi solo le righe/colonne con massimi più alti
        row_maxes = subset_plan.max(axis=1)
        col_maxes = subset_plan.max(axis=0)
        top_rows = np.argsort(row_maxes)[::-1][:max_size]
        top_cols = np.argsort(col_maxes)[::-1][:max_size]

        subset_plan = subset_plan[np.ix_(top_rows, top_cols)]
        subset_labels_row = [subset_labels_row[i] for i in top_rows]
        subset_labels_col = [subset_labels_col[i] for i in top_cols]

    short_row = truncate_labels(subset_labels_row, max_len=15)
    short_col = truncate_labels(subset_labels_col, max_len=15)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(
        subset_plan,
        cmap=COLORMAPS["transport"],
        aspect="auto",
    )

    ax.set_xticks(range(len(short_col)))
    ax.set_yticks(range(len(short_row)))
    ax.set_xticklabels(short_col, rotation=45, ha="right", fontsize=FONT_SIZES["small"])
    ax.set_yticklabels(short_row, fontsize=FONT_SIZES["small"])

    ax.set_xlabel("Sinic terms", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel("WEIRD terms", fontsize=FONT_SIZES["axis"])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Transport mass", fontsize=FONT_SIZES["tick"])

    ax.set_title(
        f"Transport Plan (top {100-percentile:.0f}% masses, threshold = {threshold:.2e})",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "gw_transport_thresholded", formats, dpi)


def _gini_coefficient(values: np.ndarray) -> float:
    """Calcola coefficiente di Gini per misurare concentrazione."""
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
