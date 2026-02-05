"""
nda_plots.py — Visualizzazioni PNG per Esperimento 5 (NDA).

Genera:
1. Jaccard histogram con linea media
2. Network graph dei top "false friends" (termini con basso Jaccard)
3. Decomposition comparison cards
"""
# ─── Visualizzazione dei vicinati semantici ─────────────────────────
# L'NDA confronta i k-NN di ogni termine nei due spazi. Un basso
# Jaccard indica che i "vicini semantici" differiscono tra le due
# culture linguistiche. Il network graph evidenzia i "false friends".
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
    NDA_NEIGHBORS_DISPLAY,
    jaccard_color,
    significance_label,
)
from .common import (
    create_figure,
    save_figure,
    setup_style,
    truncate_labels,
)

logger = logging.getLogger(__name__)


def plot_jaccard_histogram(
    per_term_results: list[dict],
    mean_jaccard: float,
    p_value: float,
    k: int,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera istogramma della distribuzione Jaccard per termine.

    Parameters
    ----------
    per_term_results : list[dict]
        Lista con keys: term, jaccard.
    mean_jaccard : float
        Media Jaccard.
    p_value : float
        P-value del test di permutazione.
    k : int
        Numero di vicini usato.
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

    jaccards = [r["jaccard"] for r in per_term_results]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES.double_column)

    # Istogramma con bins colorati per range Jaccard
    n, bins, patches = ax.hist(
        jaccards,
        bins=20,
        edgecolor="white",
        linewidth=0.5,
    )

    # Colora ogni bin
    for patch, left_edge in zip(patches, bins[:-1]):
        bin_center = left_edge + (bins[1] - bins[0]) / 2
        patch.set_facecolor(jaccard_color(bin_center))

    # Linea media
    ax.axvline(
        mean_jaccard,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_jaccard:.3f}",
    )

    # Annotazione mediana
    median_j = np.median(jaccards)
    ax.axvline(
        median_j,
        color="grey",
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {median_j:.3f}",
    )

    ax.set_xlabel(f"Jaccard index (k={k})", fontsize=FONT_SIZES["axis"])
    ax.set_ylabel("Number of terms", fontsize=FONT_SIZES["axis"])
    ax.set_xlim(0, 1)

    ax.legend(loc="upper right", fontsize=FONT_SIZES["legend"])

    ax.set_title(
        f"Distribution of Jaccard Indices (N={len(jaccards)} core terms)\n"
        f"Mean = {mean_jaccard:.4f}, {significance_label(p_value)}",
        fontsize=FONT_SIZES["subtitle"],
        fontweight="bold",
    )

    # Annotazione interpretazione
    low_count = sum(1 for j in jaccards if j < 0.2)
    high_count = sum(1 for j in jaccards if j >= 0.6)
    ax.text(
        0.02, 0.95,
        f"Low overlap (J < 0.2): {low_count} terms ({100*low_count/len(jaccards):.1f}%)\n"
        f"High overlap (J ≥ 0.6): {high_count} terms ({100*high_count/len(jaccards):.1f}%)",
        transform=ax.transAxes,
        fontsize=FONT_SIZES["annotation"],
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    return save_figure(fig, output_dir, "nda_jaccard_histogram", formats, dpi)


def plot_false_friends_network(
    per_term_results: list[dict],
    output_dir: Path,
    top_n: int = 20,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera network graph dei "false friends" (termini con basso Jaccard).

    Mostra i termini con i vicinati più divergenti, con nodi colorati
    per Jaccard e edges che collegano a vicini condivisi vs esclusivi.

    Parameters
    ----------
    per_term_results : list[dict]
        Lista con keys: term, jaccard, weird_neighbors, sinic_neighbors, shared_neighbors.
    output_dir : Path
        Directory di output.
    top_n : int
        Numero di false friends da mostrare.
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

    # Ordina per Jaccard crescente (false friends = basso Jaccard)
    sorted_results = sorted(per_term_results, key=lambda r: r["jaccard"])[:top_n]

    try:
        import networkx as nx

        G = nx.Graph()

        # Aggiungi nodi centrali (termini core)
        for r in sorted_results:
            G.add_node(
                r["term"],
                node_type="core",
                jaccard=r["jaccard"],
            )

        # Aggiungi vicini e edges
        for r in sorted_results:
            term = r["term"]
            weird_n = set(r.get("weird_neighbors", [])[:5])
            sinic_n = set(r.get("sinic_neighbors", [])[:5])
            shared_n = set(r.get("shared_neighbors", []))

            # Vicini esclusivi WEIRD
            for n in weird_n - shared_n:
                if n not in G:
                    G.add_node(n, node_type="weird_only")
                G.add_edge(term, n, edge_type="weird")

            # Vicini esclusivi Sinic
            for n in sinic_n - shared_n:
                if n not in G:
                    G.add_node(n, node_type="sinic_only")
                G.add_edge(term, n, edge_type="sinic")

            # Vicini condivisi
            for n in shared_n & (weird_n | sinic_n):
                if n not in G:
                    G.add_node(n, node_type="shared")
                G.add_edge(term, n, edge_type="shared")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Disegna edges per tipo
        edge_colors = {
            "weird": COLORS["weird"],
            "sinic": COLORS["sinic"],
            "shared": "#2ecc71",
        }
        for edge_type, color in edge_colors.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == edge_type]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, alpha=0.5, ax=ax)

        # Disegna nodi per tipo
        core_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "core"]
        core_colors = [jaccard_color(G.nodes[n]["jaccard"]) for n in core_nodes]
        nx.draw_networkx_nodes(
            G, pos, nodelist=core_nodes,
            node_color=core_colors, node_size=300,
            edgecolors="black", linewidths=1.5, ax=ax,
        )

        weird_only = [n for n, d in G.nodes(data=True) if d.get("node_type") == "weird_only"]
        nx.draw_networkx_nodes(
            G, pos, nodelist=weird_only,
            node_color=COLORS["weird"], node_size=100, alpha=0.6, ax=ax,
        )

        sinic_only = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sinic_only"]
        nx.draw_networkx_nodes(
            G, pos, nodelist=sinic_only,
            node_color=COLORS["sinic"], node_size=100, alpha=0.6, ax=ax,
        )

        shared_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "shared"]
        nx.draw_networkx_nodes(
            G, pos, nodelist=shared_nodes,
            node_color="#2ecc71", node_size=100, alpha=0.6, ax=ax,
        )

        # Etichette solo per nodi core
        core_labels = {n: truncate_labels([n], 12)[0] for n in core_nodes}
        nx.draw_networkx_labels(G, pos, core_labels, font_size=FONT_SIZES["small"], ax=ax)

        ax.set_title(
            f"Top {top_n} 'False Friends' (lowest Jaccard)\n"
            "Node color = Jaccard (red=low, green=high)",
            fontsize=FONT_SIZES["subtitle"],
            fontweight="bold",
        )
        ax.axis("off")

        # Legenda manuale
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["weird"],
                   markersize=8, label="WEIRD-only neighbor"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["sinic"],
                   markersize=8, label="Sinic-only neighbor"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
                   markersize=8, label="Shared neighbor"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=FONT_SIZES["legend"])

        fig.tight_layout()

        return save_figure(fig, output_dir, "nda_false_friends_network", formats, dpi)

    except ImportError:
        logger.warning("networkx non disponibile, skip network graph")
        return []


def plot_decomposition_comparison(
    decompositions: list[dict],
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] = None,
) -> list[Path]:
    """
    Genera figure comparative per le decomposizioni normative.

    Ogni decomposizione è visualizzata come bar chart dei top vicini
    del residuo nei due spazi.

    Parameters
    ----------
    decompositions : list[dict]
        Lista con keys: en_formula, zh_formula, jaccard, weird_neighbors, sinic_neighbors.
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

    n_decomp = len(decompositions)
    if n_decomp == 0:
        return []

    # Una figura con subplots per ogni decomposizione
    fig, axes = plt.subplots(n_decomp, 2, figsize=(14, 3 * n_decomp))

    if n_decomp == 1:
        axes = axes.reshape(1, -1)

    for i, d in enumerate(decompositions):
        ax_w, ax_s = axes[i]

        # Estrai dati
        en_formula = d.get("en_formula", "")
        zh_formula = d.get("zh_formula", "")
        jaccard = d.get("jaccard", 0)
        weird_neighbors = d.get("weird_neighbors", [])
        sinic_neighbors = d.get("sinic_neighbors", [])

        # Top-5 vicini
        top_k = min(5, len(weird_neighbors), len(sinic_neighbors))

        if top_k > 0:
            # WEIRD neighbors
            w_labels = [truncate_labels([n["label"]], 15)[0] for n in weird_neighbors[:top_k]]
            w_dists = [n.get("cosine_distance", 0) for n in weird_neighbors[:top_k]]

            ax_w.barh(range(top_k), w_dists, color=COLORS["weird"], alpha=0.8)
            ax_w.set_yticks(range(top_k))
            ax_w.set_yticklabels(w_labels, fontsize=FONT_SIZES["tick"])
            ax_w.invert_yaxis()
            ax_w.set_xlabel("Cosine distance", fontsize=FONT_SIZES["tick"])
            ax_w.set_title("WEIRD neighbors", fontsize=FONT_SIZES["tick"])

            # Sinic neighbors
            s_labels = [truncate_labels([n["label"]], 15)[0] for n in sinic_neighbors[:top_k]]
            s_dists = [n.get("cosine_distance", 0) for n in sinic_neighbors[:top_k]]

            ax_s.barh(range(top_k), s_dists, color=COLORS["sinic"], alpha=0.8)
            ax_s.set_yticks(range(top_k))
            ax_s.set_yticklabels(s_labels, fontsize=FONT_SIZES["tick"])
            ax_s.invert_yaxis()
            ax_s.set_xlabel("Cosine distance", fontsize=FONT_SIZES["tick"])
            ax_s.set_title("Sinic neighbors", fontsize=FONT_SIZES["tick"])

        # Titolo riga
        j_color = jaccard_color(jaccard)
        ax_w.set_ylabel(
            f"{en_formula}\n{zh_formula}\nJ={jaccard:.3f}",
            fontsize=FONT_SIZES["tick"],
            color=j_color,
            fontweight="bold",
        )

    fig.suptitle(
        "Normative Decompositions: Residual Vector Neighbors",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return save_figure(fig, output_dir, "nda_decompositions", formats, dpi)
