"""
multi_model_plots.py — Visualizzazioni PNG per analisi multi-modello.

Genera:
1. Heatmap di consistenza WEIRD×Sinic (matrice r per coppia)
2. Forest plot con CI per ogni coppia + media aggregata
3. Overlay delle distribuzioni nulle di tutte le coppie
"""
# ─── Perché questi grafici ───────────────────────────────────────────
# Il forest plot è lo standard nella meta-analisi per mostrare
# la consistenza di un effetto tra studi indipendenti. L'heatmap
# di consistenza rivela immediatamente se l'effetto dipende da una
# specifica combinazione di modelli. L'overlay delle null distributions
# mostra che TUTTI i valori osservati cadono lontano dal caso.
# ─────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..config import (
    COLORS,
    COLORMAPS,
    FIGURE_SIZES,
    FONT_SIZES,
    OKABE_ITO,
    significance_label,
)
from .common import save_figure, setup_style

logger = logging.getLogger(__name__)


def plot_multi_model_heatmap(
    pair_results: list[dict],
    weird_labels: list[str],
    sinic_labels: list[str],
    output_dir: Path,
    stat_key: str = "spearman_r",
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera heatmap di consistenza WEIRD × Sinic.

    Righe: modelli WEIRD, Colonne: modelli Sinic.
    Ogni cella mostra il valore della statistica per quella coppia.

    Parameters
    ----------
    pair_results : list[dict]
        Risultati per coppia (da MultiModelResult.pair_results).
    weird_labels : list[str]
        Etichette modelli WEIRD (ordinate).
    sinic_labels : list[str]
        Etichette modelli Sinic (ordinate).
    output_dir : Path
        Directory di output.
    stat_key : str
        Chiave della statistica da visualizzare.
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

    n_w = len(weird_labels)
    n_s = len(sinic_labels)
    matrix = np.zeros((n_w, n_s))

    # Costruisci la matrice: lookup per (model_weird, model_sinic) -> stat
    for pr in pair_results:
        w_label = pr.get("model_weird", "")
        s_label = pr.get("model_sinic", "")
        val = pr.get(stat_key, 0)
        if w_label in weird_labels and s_label in sinic_labels:
            i = weird_labels.index(w_label)
            j = sinic_labels.index(s_label)
            matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, n_s * 1.8 + 2), max(4, n_w * 1.2 + 2)))

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto",
                   vmin=matrix.min() - 0.02, vmax=matrix.max() + 0.02)

    # Annotazioni nelle celle
    for i in range(n_w):
        for j in range(n_s):
            val = matrix[i, j]
            # Colore testo: bianco su sfondo scuro, nero su chiaro
            threshold = (matrix.max() + matrix.min()) / 2
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=FONT_SIZES["axis"], fontweight="bold", color=color)

    ax.set_xticks(range(n_s))
    ax.set_yticks(range(n_w))
    ax.set_xticklabels(sinic_labels, fontsize=FONT_SIZES["tick"], rotation=25, ha="right")
    ax.set_yticklabels(weird_labels, fontsize=FONT_SIZES["tick"])
    ax.set_xlabel("Modelli Sinic", fontsize=FONT_SIZES["axis"], fontweight="bold")
    ax.set_ylabel("Modelli WEIRD", fontsize=FONT_SIZES["axis"], fontweight="bold")

    # Etichette appropriate per la statistica
    stat_display = {
        "spearman_r": "Spearman ρ",
        "distance": "GW Distance",
    }.get(stat_key, stat_key)

    fig.colorbar(im, ax=ax, shrink=0.8, label=stat_display)

    mean_val = matrix.mean()
    std_val = matrix.std()
    ax.set_title(
        f"Cross-Model Consistency: {stat_display}\n"
        f"Mean = {mean_val:.4f} ± {std_val:.4f} ({n_w}×{n_s} = {n_w*n_s} pairs)",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )

    fig.tight_layout()

    suffix = f"_{stat_key}" if stat_key != "spearman_r" else ""
    return save_figure(fig, output_dir, f"multi_model_heatmap{suffix}", formats, dpi)


def plot_multi_model_forest(
    pair_results: list[dict],
    aggregate: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera forest plot stile meta-analisi per le coppie di modelli.

    Ogni coppia è un punto con barra di errore (bootstrap CI).
    La media aggregata è mostrata come diamante in fondo.

    Parameters
    ----------
    pair_results : list[dict]
        Risultati per coppia (da MultiModelResult.pair_results).
    aggregate : dict
        Statistiche aggregate (mean, std, etc.).
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

    n = len(pair_results)

    # Estrai dati per il forest plot
    labels = []
    r_values = []
    ci_lowers = []
    ci_uppers = []
    p_values = []

    for pr in pair_results:
        w = pr.get("model_weird", "?")
        s = pr.get("model_sinic", "?")
        labels.append(f"{w} × {s}")
        r_values.append(pr.get("spearman_r", 0))
        p_values.append(pr.get("p_value", 1))
        ci = pr.get("bootstrap_ci", {})
        ci_lowers.append(ci.get("ci_lower", r_values[-1]))
        ci_uppers.append(ci.get("ci_upper", r_values[-1]))

    r_values = np.array(r_values)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)

    # Ordina per r decrescente
    order = np.argsort(r_values)[::-1]
    labels = [labels[i] for i in order]
    r_values = r_values[order]
    ci_lowers = ci_lowers[order]
    ci_uppers = ci_uppers[order]
    p_values = [p_values[i] for i in order]

    fig_height = max(5, n * 0.55 + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n)

    # Colori per coppia: usa palette Okabe-Ito ciclica
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]

    # Barre di errore + punti
    for i in range(n):
        ax.errorbar(
            r_values[i], y_positions[i],
            xerr=[[r_values[i] - ci_lowers[i]], [ci_uppers[i] - r_values[i]]],
            fmt="o", color=colors[i], markersize=8,
            capsize=4, capthick=1.5, linewidth=1.5,
        )
        # Annotazione p-value
        sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else "n.s."
        ax.text(
            ci_uppers[i] + 0.005, y_positions[i],
            f"  {sig}", va="center", fontsize=FONT_SIZES["annotation"],
            color="#666",
        )

    # Linea della media aggregata
    mean_r = aggregate.get("mean", np.mean(r_values))
    ax.axvline(mean_r, color=COLORS["accent"], linewidth=2, linestyle="--",
               alpha=0.7, zorder=0)

    # Diamante per la media aggregata
    y_agg = n + 0.8
    ax.scatter([mean_r], [y_agg], marker="D", s=120, color=COLORS["accent"],
               zorder=5, edgecolors="black", linewidth=0.8)
    std_r = aggregate.get("std", 0)
    ax.errorbar(
        mean_r, y_agg,
        xerr=std_r,
        fmt="none", color=COLORS["accent"], capsize=5, capthick=2, linewidth=2,
    )

    # Etichette
    all_labels = labels + [f"Media ± SD"]
    all_y = list(y_positions) + [y_agg]
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=FONT_SIZES["tick"])

    # Linea di separazione prima della media
    ax.axhline(n + 0.2, color="#ccc", linewidth=0.8, linestyle="-")

    ax.set_xlabel("Spearman ρ", fontsize=FONT_SIZES["axis"])
    ax.set_xlim(
        min(ci_lowers.min(), mean_r - std_r) - 0.03,
        max(ci_uppers.max(), mean_r + std_r) + 0.04,
    )
    ax.set_ylim(-0.5, y_agg + 0.8)
    ax.invert_yaxis()

    ax.set_title(
        f"Multi-Model RSA: Forest Plot ({n} pairs)\n"
        f"Mean ρ = {mean_r:.4f} ± {std_r:.4f}",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return save_figure(fig, output_dir, "multi_model_forest", formats, dpi)


def plot_multi_model_null_distributions(
    pair_results: list[dict],
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Overlay delle distribuzioni nulle del Mantel test per tutte le coppie.

    Mostra che TUTTI i valori osservati cadono lontano dalla
    distribuzione nulla, indipendentemente dalla coppia di modelli.

    Parameters
    ----------
    pair_results : list[dict]
        Risultati per coppia, ciascuno con null_distribution e spearman_r.
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

    # Filtra coppie con distribuzione nulla disponibile
    valid_pairs = [
        pr for pr in pair_results if pr.get("null_distribution")
    ]
    if not valid_pairs:
        logger.warning("Nessuna distribuzione nulla disponibile per il plot")
        return []

    n = len(valid_pairs)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colori per coppia
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]

    for i, pr in enumerate(valid_pairs):
        null_dist = np.array(pr["null_distribution"])
        observed = pr.get("spearman_r", 0)
        w = pr.get("model_weird", "?")
        s = pr.get("model_sinic", "?")
        label = f"{w} × {s}"

        # Istogramma con trasparenza (density normalizzata)
        ax.hist(
            null_dist, bins=40, density=True, alpha=0.25,
            color=colors[i], edgecolor="none",
        )

        # Linea verticale per il valore osservato
        ax.axvline(
            observed, color=colors[i], linewidth=2, linestyle="--",
            label=f"{label}: r={observed:.4f}",
        )

    ax.set_xlabel("Spearman ρ (permuted)", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel("Density", fontsize=FONT_SIZES["axis"])
    ax.legend(fontsize=FONT_SIZES["annotation"], loc="upper left",
              bbox_to_anchor=(0, 1), framealpha=0.9)

    ax.set_title(
        f"Null Distributions vs Observed ({n} model pairs)\n"
        f"All observed values fall far from the null",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "multi_model_null_dists", formats, dpi)
