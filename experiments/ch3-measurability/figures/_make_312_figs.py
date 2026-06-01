#!/usr/bin/env python3
"""
Generate paper-style B/W figures for §3.1.2 (maps of inter-domain distance).

Fig.4 — 7×7 inter-domain topology heatmap for BGE-EN-large attested,
        greyscale, with cell values annotated.
Fig.5 — 10 small-multiples of the 7×7 topology, one per encoder, in the
        canonical display order (EN-side row 1, ZH-side row 2; bilingual
        controls italicised in the panel titles), shared greyscale.

Outputs PNG @ 300 dpi in this directory.

Data source: experiment_1_structure/results_attested/experiment_1_results.json
             → section_312.per_model.<encoder>.{domains, matrix}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ----------------------------------------------------------------------------
# Shared design system (Nature / Distill / IEEE Transactions house style).
# Applied at module import so every figure in this file inherits the canon.
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

REPO = Path(__file__).resolve().parents[4]
RES_ATT = REPO / "experiments" / "ch3-measurability" / "experiment_1_structure" / "results_attested" / "experiment_1_results.json"
OUT = Path(__file__).parent

# Canonical display order (mirrors _make_311_figs.py and dashboard loader_31)
ALL_MODELS_ORDERED = [
    "BGE-EN-large", "E5-large", "FreeLaw-EN",
    "BGE-M3-EN", "Qwen3-0.6B-EN",
    "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH",
    "BGE-M3-ZH", "Qwen3-0.6B-ZH",
]
SIGNAL_PANEL = {"BGE-EN-large", "E5-large", "FreeLaw-EN",
                "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"}

DISP = {
    "BGE-EN-large": "BGE-EN-large",
    "E5-large": "E5-large",
    "FreeLaw-EN": "FreeLaw-EN",
    "BGE-ZH-large": "BGE-ZH-large",
    "Text2vec-large-ZH": "Text2vec-ZH",
    "Dmeta-ZH": "Dmeta-ZH",
    "BGE-M3-EN": "BGE-M3 (EN)",
    "BGE-M3-ZH": "BGE-M3 (ZH)",
    "Qwen3-0.6B-EN": "Qwen3-0.6B (EN)",
    "Qwen3-0.6B-ZH": "Qwen3-0.6B (ZH)",
}

# Compact domain labels for axis ticks
DOMAIN_LABEL = {
    "administrative": "admin",
    "civil": "civil",
    "constitutional": "const",
    "criminal": "crim",
    "international": "intl",
    "labor_social": "labor",
    "procedure": "proc",
}


def load_topology() -> dict:
    with RES_ATT.open() as fh:
        d = json.load(fh)
    return d["section_312"]["per_model"]


# ============================================================================
# Fig.4 — Heatmap for BGE-EN-large attested (paper-style, viridis)
# ============================================================================
def make_fig4(pm: dict) -> None:
    model = "BGE-EN-large"
    M = np.array(pm[model]["matrix"])
    domains = pm[model]["domains"]
    labels = [DOMAIN_LABEL[d] for d in domains]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    # Perceptually uniform sequential colormap. vmin/vmax tight to the data
    # range so the topology fills the dynamic range of the palette.
    cmap = plt.get_cmap("viridis")
    vmin = float(M.min())
    vmax = float(M.max())
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")

    # Luminance-aware text colour. Map each cell's normalised value through
    # the colormap, compute Rec. 709 relative luminance, and pick black on
    # bright backgrounds, white on dark ones. Threshold 0.55 chosen
    # empirically against viridis to keep mid-greens readable in black.
    norm_vals = (M - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(M)
    rgba = cmap(norm_vals)
    lum = 0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2]

    for i in range(7):
        for j in range(7):
            colour = "black" if lum[i, j] > 0.55 else "white"
            weight = "semibold" if i == j else "normal"
            ax.text(j, i, f"{M[i, j]:.3f}",
                    ha="center", va="center", fontsize=9,
                    color=colour, fontweight=weight)

    # Highlight the diagonal: thin black border around each within-domain cell.
    for k in range(7):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1,
                               fill=False, edgecolor="black",
                               linewidth=1.0, zorder=3))

    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="both", which="major", length=0, pad=4)

    # Thin separator between cells for visual structure (white hairlines).
    ax.set_xticks(np.arange(8) - 0.5, minor=True)
    ax.set_yticks(np.arange(8) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Suppress spines around the heatmap for a cleaner panel.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("BGE-EN-large attested  ·  7 × 7 inter-domain distance topology",
                 fontsize=12, fontweight="bold", pad=12)

    # Colorbar matched to axis height for a tidy right-edge legend.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine distance  (1 − cos)", fontsize=10)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("0.4")

    fig.tight_layout()
    out = OUT / "fig4_topology_bge_en.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ wrote {out.name}")


# ============================================================================
# Fig.5 — Small multiples for all 10 encoders, shared viridis scale
# ============================================================================
def make_fig5(pm: dict) -> None:
    models = [m for m in ALL_MODELS_ORDERED if m in pm]
    assert len(models) == 10, f"expected 10 encoders in section_312, got {len(models)}"

    # 5 cols × 2 rows: EN-side top, ZH-side bottom.
    # Single shared colour scale across all ten panels: enables direct
    # cross-encoder comparison of both pattern and absolute magnitude.
    row_models = [models[:5], models[5:]]
    all_mats = [np.array(pm[m]["matrix"]) for m in models]
    vmin = min(float(M.min()) for M in all_mats)
    vmax = max(float(M.max()) for M in all_mats)

    domains = pm[models[0]]["domains"]
    labels = [DOMAIN_LABEL[d] for d in domains]

    # Landscape 14 × 6 in, generous whitespace between panels
    fig, axes = plt.subplots(
        2, 5,
        figsize=(14.0, 6.0),
        gridspec_kw={"hspace": 0.32, "wspace": 0.22,
                     "left": 0.055, "right": 0.91,
                     "top": 0.86, "bottom": 0.13},
    )

    last_im = None
    for r, row in enumerate(row_models):
        for c, m in enumerate(row):
            ax = axes[r, c]
            M = np.array(pm[m]["matrix"])
            im = ax.imshow(M, cmap="viridis",
                           vmin=vmin, vmax=vmax,
                           interpolation="nearest", aspect="equal")
            last_im = im

            ax.set_xticks(np.arange(7))
            ax.set_yticks(np.arange(7))

            # Tufte-style edge labels: x-ticks only on bottom row,
            # y-ticks only on leftmost column. Inner panels remain unlabeled
            # so the eye focuses on the matrix pattern, not on label noise.
            if r == 1:
                ax.set_xticklabels(labels, fontsize=8, rotation=45,
                                   ha="right", rotation_mode="anchor")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_yticklabels(labels, fontsize=8)
            else:
                ax.set_yticklabels([])

            # Subtle cell separator (white minor grid)
            ax.set_xticks(np.arange(7) - 0.5, minor=True)
            ax.set_yticks(np.arange(7) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(which="major", length=2, width=0.6, pad=2)

            # Thin frame around each heatmap (override default spine config)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("0.15")

            # Title: italics for bilingual controls (§2.3 reporting rule)
            disp = DISP[m]
            is_signal = m in SIGNAL_PANEL
            style = "normal" if is_signal else "italic"
            colour = "0.10" if is_signal else "0.35"
            ax.set_title(disp, fontsize=9.5, pad=5,
                         style=style, fontweight="bold", color=colour)

    # ---- Side annotations: EN-side / ZH-side ----
    # Row centres sit at y≈0.66 (top) and y≈0.27 (bot) in figure coords
    fig.text(0.018, 0.66, "EN-side", fontsize=10, rotation=90,
             ha="center", va="center", style="italic",
             fontweight="bold", color="0.25")
    fig.text(0.018, 0.27, "ZH-side", fontsize=10, rotation=90,
             ha="center", va="center", style="italic",
             fontweight="bold", color="0.25")

    # ---- Visual separator between the two rows ----
    sep_y = 0.475
    fig.add_artist(plt.Line2D([0.055, 0.91], [sep_y, sep_y],
                              transform=fig.transFigure,
                              color="0.80", linewidth=0.5, zorder=0))

    # ---- Single shared colorbar on the right, spanning both rows ----
    cax = fig.add_axes([0.925, 0.16, 0.012, 0.68])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("cosine distance  (1 − cos)", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("0.15")

    # ---- Suptitle ----
    fig.suptitle(
        "Inter-domain topology, attested  ·  six monolingual signal-panel encoders + four bilingual-control readings",
        fontsize=11, y=0.955, fontweight="bold", color="0.10",
    )

    out = OUT / "fig5_topology_small_multiples.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    pm = load_topology()
    print(f"Loaded section_312.per_model: {len(pm)} encoders")
    print("Generating Fig.4 (BGE-EN-large attested topology heatmap)...")
    make_fig4(pm)
    print("\nGenerating Fig.5 (10-encoder small multiples)...")
    make_fig5(pm)
    print("\nDone.")
