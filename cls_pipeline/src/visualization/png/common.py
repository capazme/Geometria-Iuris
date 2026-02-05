"""
common.py — Utility condivise per la generazione di plot PNG.

Fornisce funzioni helper per setup stile, salvataggio figure,
e operazioni comuni tra tutti i moduli di visualizzazione.
"""
# ─── Funzioni utility per matplotlib ────────────────────────────────
# Centralizza la configurazione dello stile e il salvataggio per
# garantire coerenza tra tutti i plot generati dalla pipeline.
# ─────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

import matplotlib
# Backend Agg solo se non c'e' gia' un backend interattivo (es. notebook)
if matplotlib.get_backend() == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import (
    apply_matplotlib_style,
    COLORS,
    DEFAULT_DPI,
    FIGURE_SIZES,
    SUPPORTED_FORMATS,
)

logger = logging.getLogger(__name__)


def setup_style():
    """
    Applica lo stile matplotlib publication-ready.

    Deve essere chiamato all'inizio di ogni funzione di plotting
    per garantire coerenza anche quando matplotlib viene resettato.
    """
    apply_matplotlib_style()


def save_figure(
    fig: plt.Figure,
    output_dir: Optional[Path],
    filename: str,
    formats: Optional[list[str]] = None,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> list[Path]:
    """
    Salva una figura in uno o più formati.

    Parameters
    ----------
    fig : plt.Figure
        Figura matplotlib da salvare.
    output_dir : Path or None
        Directory di output. Se None, la figura non viene salvata
        e non viene chiusa (utile per display inline in notebook).
    filename : str
        Nome file senza estensione.
    formats : list[str], optional
        Lista di formati (default: ["png"]).
    dpi : int
        Risoluzione per formati raster.
    close : bool
        Se True, chiude la figura dopo il salvataggio.

    Returns
    -------
    list[Path]
        Lista dei percorsi dei file salvati.
    """
    if output_dir is None:
        return []

    if formats is None:
        formats = ["png"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            logger.warning("Formato non supportato: %s (skip)", fmt)
            continue

        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi if fmt == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        saved_paths.append(path)
        logger.debug("Salvato: %s", path)

    if close:
        plt.close(fig)

    return saved_paths


def create_figure(
    figsize: Optional[tuple[float, float]] = None,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """
    Crea una figura con lo stile appropriato.

    Parameters
    ----------
    figsize : tuple, optional
        Dimensioni figura. Default: FIGURE_SIZES.double_column
    nrows, ncols : int
        Numero righe/colonne subplots.
    **kwargs
        Argomenti aggiuntivi per plt.subplots().

    Returns
    -------
    fig, ax
        Figura e assi.
    """
    setup_style()

    if figsize is None:
        if nrows == 1 and ncols == 1:
            figsize = FIGURE_SIZES.double_column
        elif ncols == 2:
            figsize = FIGURE_SIZES.side_by_side
        else:
            figsize = FIGURE_SIZES.wide

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    return fig, ax


def truncate_labels(labels: list[str], max_len: int = 15) -> list[str]:
    """Tronca etichette lunghe aggiungendo ellipsis."""
    return [
        label[:max_len-1] + "…" if len(label) > max_len else label
        for label in labels
    ]


def add_significance_annotation(
    ax: plt.Axes,
    p_value: float,
    x: float,
    y: float,
    fontsize: int = 9,
):
    """Aggiunge annotazione di significatività statistica."""
    from ..config import significance_marker

    marker = significance_marker(p_value)
    if marker != "n.s.":
        ax.annotate(
            marker,
            xy=(x, y),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )


def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    label: str = "",
    shrink: float = 0.8,
):
    """Aggiunge colorbar con label."""
    cbar = fig.colorbar(mappable, ax=ax, shrink=shrink)
    if label:
        cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return cbar


def get_domain_order(domains: list[str]) -> list[int]:
    """
    Restituisce indici per ordinare i termini per dominio.

    Utile per raggruppare termini dello stesso dominio in heatmap/dendrogrammi.
    """
    # Ordine preferito dei domini
    domain_priority = {
        "constitutional": 0,
        "rights": 1,
        "civil": 2,
        "criminal": 3,
        "governance": 4,
        "jurisprudence": 5,
        "international": 6,
        "labor_social": 7,
        "environmental_tech": 8,
        "control": 9,
        "unknown": 10,
    }

    return sorted(
        range(len(domains)),
        key=lambda i: (domain_priority.get(domains[i], 99), i)
    )


def compute_inter_domain_matrix(
    rdm: np.ndarray,
    domains: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Calcola matrice di distanza media inter-dominio.

    Parameters
    ----------
    rdm : np.ndarray
        Matrice di dissimilarità N×N.
    domains : list[str]
        Lista di N domini.

    Returns
    -------
    inter_matrix : np.ndarray
        Matrice K×K di distanze medie tra domini.
    domain_labels : list[str]
        Lista di K nomi dominio unici.
    """
    unique_domains = sorted(set(domains))
    k = len(unique_domains)
    domain_to_idx = {d: i for i, d in enumerate(unique_domains)}

    inter_matrix = np.zeros((k, k))
    counts = np.zeros((k, k))

    for i in range(len(domains)):
        for j in range(len(domains)):
            if i != j:
                di = domain_to_idx[domains[i]]
                dj = domain_to_idx[domains[j]]
                inter_matrix[di, dj] += rdm[i, j]
                counts[di, dj] += 1

    # Media (evita divisione per zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        inter_matrix = np.divide(inter_matrix, counts)
        inter_matrix = np.nan_to_num(inter_matrix)

    return inter_matrix, unique_domains
