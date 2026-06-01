#!/usr/bin/env python3
"""
Generate paper-style figure for §3.2.2 (axes independence).

Fig.10 — Heatmap 6×6 of inter-axis cosine similarity for BGE-EN-large attested.
         Diagonal masked (=1 by construction). Cells annotated with signed
         cosine value to two decimals. Divergent colormap (RdBu_r) centred at 0:
         red = positive alignment, blue = negative alignment, white = orthogonal.
         The four pair sign-coherent across the entire 10/10 cohort are
         highlighted with a black cell border.

Source: experiment_2_axes/results_attested/experiment_2_results.json
        § section_322[BGE-EN-large].cosine_matrix
        (re-derived deterministically from the frozen axis vectors
         experiment_2_axes/results_attested/axes/BGE-EN-large_<axis>.npy).

Outputs PNG @ 300 dpi in this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------------
# Shared design system (Nature / Distill.pub / IEEE Transactions style)
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
RES_ATT = REPO / "experiments" / "ch3-measurability" / "experiment_2_axes" / \
          "results_attested" / "experiment_2_results.json"
AXES_DIR_ATT = REPO / "experiments" / "ch3-measurability" / "experiment_2_axes" / \
               "results_attested" / "axes"
OUT = Path(__file__).parent

# Axis order as stored in experiment_2_results.json (meta.axes and per-encoder
# section_322[*].axes). Preserved verbatim for direct cross-reference.
AXES = [
    "individual_collective",
    "rights_duties",
    "public_private",
    "state_market",
    "natural_positive",
    "status_contract",
]

# Compact display labels (line break for readability on a 6×6 grid).
# "/" separator renders reliably across fonts; the polarity convention
# (first lemma = positive pole) is set in §3.2.1 and is the same used
# throughout §3.2.
AXES_DISPLAY = [
    "individual\n/ collective",
    "rights\n/ duties",
    "public\n/ private",
    "state\n/ market",
    "natural\n/ positive",
    "status\n/ contract",
]

REPRESENTATIVE = "BGE-EN-large"

# Four axis-pairs that are sign-coherent across the entire 10/10 cohort
# (signal panel + bilingual controls), as identified in §3.2.2 body text.
# Stored as (axis_a, axis_b) tuples using AXES indices.
SIGN_COHERENT_PAIRS = [
    ("individual_collective", "public_private"),
    ("state_market", "status_contract"),
    ("individual_collective", "rights_duties"),
    ("public_private", "state_market"),
]


def load_cosine_from_results(label: str) -> np.ndarray:
    """Load the frozen 6×6 cosine matrix from experiment_2_results.json."""
    with RES_ATT.open() as fh:
        d = json.load(fh)
    M = np.array(d["section_322"][label]["cosine_matrix"], dtype=np.float64)
    axes_in_file = d["section_322"][label]["axes"]
    assert axes_in_file == AXES, (
        f"axis order mismatch: file={axes_in_file} expected={AXES}"
    )
    return M


def verify_against_axis_vectors(label: str, M_frozen: np.ndarray) -> None:
    """Sanity check: reload axis vectors and recompute cosine matrix."""
    V = np.stack([
        np.load(AXES_DIR_ATT / f"{label}_{ax}.npy").astype(np.float64)
        for ax in AXES
    ], axis=0)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    V = V / norms
    M_recomp = V @ V.T
    diff = np.max(np.abs(M_recomp - M_frozen))
    assert diff < 1e-4, f"frozen-vs-recomputed disagreement: max|Δ|={diff:.2e}"
    print(f"  axis-vector verification OK: max|Δ| = {diff:.2e}")


def _luminance(rgba) -> float:
    """Relative luminance per WCAG (used for text-on-cell contrast switch)."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ============================================================================
# Fig.10 — Heatmap 6×6 inter-axis cosine, BGE-EN-large attested
# ============================================================================
def make_fig10() -> None:
    label = REPRESENTATIVE
    M = load_cosine_from_results(label)
    verify_against_axis_vectors(label, M)

    # Mask diagonal (=1 by construction) for display only
    mask = np.eye(6, dtype=bool)
    M_disp_masked = np.ma.array(M, mask=mask)

    # Symmetric divergent range centred at 0.
    # vmax driven by the largest |cos| on the off-diagonal, rounded up to 0.05.
    off_max = float(np.max(np.abs(M[~mask])))
    vmax = float(np.ceil(off_max * 20.0) / 20.0)  # round up to 0.05
    vmin = -vmax

    # Divergent colormap: red = positive, blue = negative, white = 0.
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")  # neutral light grey for masked diagonal

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(
        M_disp_masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    # Precompute the indices of sign-coherent pairs (both (i,j) and (j,i))
    name_to_idx = {ax_name: k for k, ax_name in enumerate(AXES)}
    highlight_cells = set()
    for a, b in SIGN_COHERENT_PAIRS:
        i, j = name_to_idx[a], name_to_idx[b]
        highlight_cells.add((i, j))
        highlight_cells.add((j, i))

    # Cell annotations: signed cosine to two decimals, auto-switched text colour
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(6):
        for j in range(6):
            if i == j:
                ax.text(j, i, "·", ha="center", va="center",
                        color="#9a9a9a", fontsize=14, fontweight="bold")
                continue
            v = M[i, j]
            rgba = cmap(norm(v))
            text_color = "white" if _luminance(rgba) < 0.55 else "black"
            sign = "+" if v > 0 else ("−" if v < 0 else " ")
            ax.text(j, i, f"{sign}{abs(v):.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="semibold")

    # Highlight the four sign-coherent (10/10) cell pairs with a thin black border
    for (i, j) in highlight_cells:
        rect = Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.6, zorder=5,
        )
        ax.add_patch(rect)

    # Ticks and labels
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(AXES_DISPLAY, fontsize=8.5, rotation=0, ha="center")
    ax.set_yticklabels(AXES_DISPLAY, fontsize=8.5, rotation=0, ha="right")
    ax.tick_params(axis="both", length=0, pad=4)

    # Subtle minor grid between cells (thin, neutral)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    # Remove spines fully (clean editorial look)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title (two lines: editorial header + numeric subtitle)
    abs_mean = float(np.mean(np.abs(M[~mask])))
    ax.set_title(
        "Inter-axis cosine alignment — BGE-EN-large, attested",
        fontsize=12, fontweight="bold", pad=16, loc="left",
    )
    # Subtitle (lighter weight, smaller)
    ax.text(
        0.0, 1.015,
        f"off-diagonal: |cos|$_{{\\mathrm{{mean}}}}$ = {abs_mean:.3f}   ·   "
        f"|cos|$_{{\\mathrm{{max}}}}$ = {off_max:.3f}   ·   "
        f"bordered cells: sign-coherent across 10/10 cohort",
        transform=ax.transAxes,
        fontsize=8.5, color="#555555", ha="left", va="bottom",
    )

    # Colourbar: centred at 0, symmetric ticks
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, shrink=0.85)
    cbar.set_label("cosine(axis$_i$, axis$_j$)", fontsize=9.5, labelpad=8)
    # Symmetric ticks at -0.30, -0.15, 0, +0.15, +0.30 (capped by vmax)
    candidate_ticks = np.array([-0.30, -0.15, 0.0, 0.15, 0.30])
    ticks = candidate_ticks[np.abs(candidate_ticks) <= vmax + 1e-9]
    if 0.0 not in ticks:
        ticks = np.sort(np.append(ticks, 0.0))
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=8.5, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.5)
    # Zero-line marker on the colorbar
    cbar.ax.axhline(0, color="#222222", linewidth=0.6, alpha=0.7)

    fig.tight_layout()
    out = OUT / "fig10_axes_independence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.10 (heatmap 6×6 inter-axis cosine, BGE-EN-large attested)...")
    make_fig10()
    print("\nDone.")
