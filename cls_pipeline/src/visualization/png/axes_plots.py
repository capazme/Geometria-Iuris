"""
axes_plots.py — Visualizzazioni PNG per Esperimento 3 (Assi Valoriali).

Genera:
1. Scatter con CI ribbon (confidence interval come area ombreggiata)
2. Forest plot con rho ± CI per ogni asse
"""
# ─── Visualizzazione degli assi semantici ───────────────────────────
# Gli assi Kozlowski proiettano i concetti su dimensioni culturali
# (es. individuale↔collettivo). Lo scatter mostra la correlazione
# tra i due spazi, il forest plot riassume tutti gli assi.
# ─────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import (
    COLORS,
    FIGURE_SIZES,
    FONT_SIZES,
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


def plot_axes_scatter_ci(
    weird_scores: dict[str, float],
    sinic_scores: dict[str, float],
    axis_name: str,
    spearman_r: float,
    spearman_p: float,
    ci_lower: Optional[float],
    ci_upper: Optional[float],
    output_dir: Path,
    highlight_threshold: float = 0.1,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera scatter plot con highlight dei termini divergenti.

    I termini con |Δ| > threshold sono evidenziati e etichettati.
    Se disponibile, mostra la banda CI attorno alla regressione.

    Parameters
    ----------
    weird_scores, sinic_scores : dict
        Score di proiezione per ogni termine.
    axis_name : str
        Nome dell'asse.
    spearman_r : float
        Correlazione Spearman.
    spearman_p : float
        P-value.
    ci_lower, ci_upper : float, optional
        Estremi CI 95%.
    output_dir : Path
        Directory di output.
    highlight_threshold : float
        Soglia |Δ| per highlight (default: 0.1).
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

    # Estrai dati
    labels = list(weird_scores.keys())
    x = np.array([weird_scores[l] for l in labels])
    y = np.array([sinic_scores[l] for l in labels])
    delta = np.abs(x - y)

    # Identifica outlier
    outlier_mask = delta > highlight_threshold

    fig, ax = plt.subplots(figsize=FIGURE_SIZES.square)

    # Plot punti normali
    ax.scatter(
        x[~outlier_mask],
        y[~outlier_mask],
        c=COLORS["weird"],
        alpha=0.5,
        s=30,
        edgecolors="white",
        linewidth=0.5,
        label="Normal",
    )

    # Plot outlier
    ax.scatter(
        x[outlier_mask],
        y[outlier_mask],
        c=COLORS["sinic"],
        alpha=0.8,
        s=50,
        edgecolors="black",
        linewidth=1,
        label=f"|Δ| > {highlight_threshold}",
        marker="D",
    )

    # Etichette per outlier
    outlier_labels = [labels[i] for i in np.where(outlier_mask)[0]]
    outlier_x = x[outlier_mask]
    outlier_y = y[outlier_mask]

    # Usa adjustText se disponibile, altrimenti annotazioni semplici
    try:
        from adjustText import adjust_text
        texts = []
        for i, label in enumerate(outlier_labels):
            short = truncate_labels([label], 12)[0]
            texts.append(ax.text(outlier_x[i], outlier_y[i], short, fontsize=FONT_SIZES["small"]))
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    except ImportError:
        for i, label in enumerate(outlier_labels):
            short = truncate_labels([label], 12)[0]
            ax.annotate(
                short,
                (outlier_x[i], outlier_y[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=FONT_SIZES["small"],
            )

    # Linea identità
    lims = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    ax.plot(lims, lims, "--", color="grey", linewidth=1, alpha=0.7, label="Identity")

    # Fit lineare
    z = np.polyfit(x, y, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, p_fit(x_fit), "-", color=COLORS["sinic"], linewidth=1.5, alpha=0.8)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    ax.set_xlabel(f"Projection score (WEIRD)", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel(f"Projection score (Sinic)", fontsize=FONT_SIZES["axis"])

    # Titolo con CI
    ci_str = ""
    if ci_lower is not None and ci_upper is not None:
        ci_str = f", CI 95% = [{ci_lower:.3f}, {ci_upper:.3f}]"

    ax.set_title(
        f"Axis: {axis_name}\n"
        f"Spearman ρ = {spearman_r:.4f}{ci_str}, {significance_label(spearman_p)}",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    ax.legend(loc="lower right", fontsize=FONT_SIZES["legend"])

    fig.tight_layout()

    # Filename safe
    safe_name = axis_name.lower().replace(" ", "_").replace("↔", "_")
    return save_figure(fig, output_dir, f"axes_{safe_name}_scatter", formats, dpi)


def plot_forest_plot(
    axes_results: list[dict],
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera forest plot con ρ ± CI per tutti gli assi.

    Il forest plot è standard in meta-analisi: ogni riga è un asse,
    il quadrato indica l'effetto (ρ), le barre l'intervallo di confidenza.

    Parameters
    ----------
    axes_results : list[dict]
        Lista di dizionari con keys: axis_name, spearman_r, ci_lower, ci_upper, spearman_p
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

    n_axes = len(axes_results)
    fig_height = max(4, n_axes * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_positions = np.arange(n_axes)[::-1]  # Inverti per avere primo asse in alto

    for i, result in enumerate(axes_results):
        rho = result["spearman_r"]
        ci_lower = result.get("ci_lower")
        ci_upper = result.get("ci_upper")
        p_value = result.get("spearman_p", 1.0)

        y = y_positions[i]

        # Colore in base a significatività
        color = COLORS["significant"] if p_value < 0.05 else COLORS["not_significant"]

        # Punto effetto
        ax.scatter([rho], [y], s=100, c=color, zorder=3, marker="s")

        # CI bars
        if ci_lower is not None and ci_upper is not None:
            ax.hlines(y, ci_lower, ci_upper, color=color, linewidth=2, zorder=2)
            ax.vlines([ci_lower, ci_upper], y - 0.1, y + 0.1, color=color, linewidth=1)

        # Annotazione numerica
        ax.text(
            1.05, y,
            f"ρ = {rho:.3f} {significance_marker(p_value)}",
            va="center",
            fontsize=FONT_SIZES["tick"],
            transform=ax.get_yaxis_transform(),
        )

    # Linea verticale a zero
    ax.axvline(0, color="grey", linestyle="--", linewidth=1, alpha=0.5)

    # Etichette assi Y
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["axis_name"] for r in axes_results], fontsize=FONT_SIZES["tick"])

    ax.set_xlabel("Spearman ρ", fontsize=FONT_SIZES["axis"])
    ax.set_xlim(-1.1, 1.1)

    ax.set_title(
        "Forest Plot: Axis Correlations with 95% CI",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["significant"],
               markersize=10, label="Significant (p < 0.05)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["not_significant"],
               markersize=10, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=FONT_SIZES["legend"])

    fig.tight_layout()

    return save_figure(fig, output_dir, "axes_forest_plot", formats, dpi)
