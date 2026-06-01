#!/usr/bin/env python3
"""
Generate paper-style figures for §3.2.1 (Building an axis from pairs
of opposites).

Fig.9 — Schematic visualisation of the Kozlowski-style axis construction.

        Panel (a) — pedagogical 2D schematic of the construction
        recipe: K = 5 antonym-pair difference vectors (illustrative
        positions, fixed seed for reproducibility), their arithmetic mean,
        and the L2-normalised axis direction.

        Panel (b) — real data. The production public_private axis
        vector (BGE-EN-large, attested) projects a selection of legal
        terms from the 364-core pool onto a one-dimensional axis with
        annotated polarity.

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

# ----------------------------------------------------------------------------
# Shared rcParams — Nature / Distill / IEEE Transactions look-and-feel
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
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

# Palette — Tableau-10 (qualitative, perceptually balanced)
PAIR_PALETTE = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
]
COL_MEAN = "#5A5A5A"
COL_AXIS = "#000000"
COL_ANCHOR = "#1F4E79"

REPO = Path(__file__).resolve().parents[4]
EMB = REPO / "experiments" / "ch3-measurability" / "embeddings"
INPUTS = REPO / "experiments" / "ch3-measurability" / "inputs"
RES_ATT = REPO / "experiments" / "ch3-measurability" / "experiment_2_axes" / "results_attested"
OUT = Path(__file__).parent


def load_index() -> list[dict]:
    with (EMB / "index.json").open() as fh:
        return json.load(fh)


def load_vecs(label: str, variant: str) -> np.ndarray:
    fname = "vecs_bare.npy" if variant == "bare" else "vecs_attested.npy"
    return np.load(EMB / label / fname).astype(np.float64)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


# ============================================================================
# Fig.9 — Schematic of axis construction
# ============================================================================
def panel_a_schematic(ax) -> None:
    """Pedagogical 2D diagram of the Kozlowski recipe.

    Five antonym-pair difference vectors are drawn at controlled angles
    around a target direction. Each pair receives a distinct colour from
    the qualitative palette. Their arithmetic mean (dashed grey) and the
    L2 normalisation (bold black) produce the axis vector."""

    pair_labels = [
        "public / private",
        "state / individual",
        "government / citizen",
        "regulation / autonomy",
        "statutory / customary",
    ]

    target_angle = np.deg2rad(35.0)
    K = len(pair_labels)
    offsets_deg = np.array([-32, -16, 0, 16, 32], dtype=float)
    angles = target_angle + np.deg2rad(offsets_deg)
    mags = np.array([1.05, 0.92, 1.18, 0.88, 1.00])
    diffs = np.column_stack([mags * np.cos(angles), mags * np.sin(angles)])

    mean_diff = diffs.mean(axis=0)
    axis_vec = mean_diff / np.linalg.norm(mean_diff)
    axis_disp = axis_vec * float(np.mean(mags)) * 1.18

    extent = max(np.abs(diffs).max(), np.abs(axis_disp).max()) * 1.95

    # Background: very subtle grid hinting at the 2D vector space
    ax.set_axisbelow(True)
    grid_step = extent * 0.20
    grid_lines = np.arange(-extent, extent + grid_step, grid_step)
    for g in grid_lines:
        ax.axhline(g, color="0.92", linewidth=0.4, zorder=0)
        ax.axvline(g, color="0.92", linewidth=0.4, zorder=0)
    # Slightly stronger zero lines
    ax.axhline(0, color="0.78", linewidth=0.6, zorder=0)
    ax.axvline(0, color="0.78", linewidth=0.6, zorder=0)

    # Draw the K pair-difference vectors (coloured arrows)
    for i, (x, y) in enumerate(diffs):
        col = PAIR_PALETTE[i % len(PAIR_PALETTE)]
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=col,
                            linewidth=1.6, alpha=0.95,
                            mutation_scale=14),
            zorder=2,
        )

    # Legend-style pair labels stacked on the upper-right corner.
    label_x = extent * 0.78
    label_y_top = extent * 0.92
    label_y_step = extent * 0.11
    for i in range(K):
        ly = label_y_top - i * label_y_step
        col = PAIR_PALETTE[i % len(PAIR_PALETTE)]
        # colour swatch
        ax.plot([label_x - extent * 0.045], [ly], marker="s",
                markersize=6.5, color=col, markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)
        ax.text(label_x, ly, pair_labels[i], fontsize=7.5, style="italic",
                color="0.20", ha="left", va="center")

    # Draw the mean direction (dashed, mid grey, intermediate step)
    ax.annotate(
        "", xy=(mean_diff[0], mean_diff[1]), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=COL_MEAN,
                        linewidth=1.6, linestyle=(0, (5, 2)),
                        alpha=0.95, mutation_scale=14),
        zorder=3,
    )
    # Mean label, slightly above the mean arrow tip
    ax.text(mean_diff[0] * 0.55, mean_diff[1] + 0.06,
            "mean", fontsize=8.5, color=COL_MEAN,
            ha="right", va="bottom", style="italic", fontweight="bold")

    # Draw the L2-normalised axis as bold black arrow (most prominent)
    ax.annotate(
        "", xy=(axis_disp[0], axis_disp[1]), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=COL_AXIS,
                        linewidth=3.0, mutation_scale=22),
        zorder=5,
    )
    # Axis label placed at the tip of the arrow (offset down-right) so it
    # does not collide with the origin tick lines.
    ax.text(axis_disp[0] + 0.06, axis_disp[1] - 0.08,
            "axis", fontsize=10, fontweight="bold", color=COL_AXIS,
            ha="left", va="top", zorder=6)
    ax.text(axis_disp[0] + 0.06, axis_disp[1] - 0.22,
            "(L2-normalised)", fontsize=7.5, style="italic",
            color="0.30", ha="left", va="top", zorder=6)

    # Origin marker — small filled black dot, clearly visible
    ax.plot(0, 0, marker="o", color="black", markersize=5.5,
            markeredgecolor="white", markeredgewidth=1.0, zorder=7)

    # Cosmetics
    ax.set_xlim(-extent * 0.30, extent * 1.50)
    ax.set_ylim(-extent * 0.40, extent * 1.10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "(a)  Axis construction  ·  K = 5 antonym-pair differences",
        loc="left", fontsize=11, fontweight="bold", pad=10,
    )
    # Sub-caption under the title
    ax.text(0.0, 1.005,
            "Schematic; the production axis averages 10 pairs.",
            transform=ax.transAxes, fontsize=8.5, style="italic",
            color="0.35", ha="left", va="bottom")
    ax.set_aspect("equal")


def panel_b_projection(ax) -> None:
    """1D projection of selected legal terms on the production
    public_private axis (BGE-EN-large, attested)."""

    label = "BGE-EN-large"
    axis_name = "public_private"

    idx = load_index()
    en_lookup = {t["en"].lower(): i for i, t in enumerate(idx)}
    vecs = l2_normalize(load_vecs(label, "attested"))
    axis_vec = np.load(RES_ATT / "axes" / f"{label}_{axis_name}.npy").astype(np.float64)

    anchors = [
        "gift", "liberty", "trust", "property", "lease", "contract",
        "company", "land", "tax", "agreement", "court", "licence",
        "constitution", "regulation",
    ]

    scores = []
    for term in anchors:
        i = en_lookup.get(term.lower())
        if i is None:
            continue
        s = float(vecs[i] @ axis_vec)
        scores.append((term, s))
    scores.sort(key=lambda x: x[1])

    all_scores = vecs @ axis_vec
    smin, smax = float(all_scores.min()), float(all_scores.max())
    span = smax - smin

    y0 = 0.0
    # ----- main axis line (black, slightly thicker) -------------------------
    ax.plot([smin, smax], [y0, y0], color=COL_AXIS, linewidth=1.6,
            zorder=2, solid_capstyle="round")

    # ----- polarity arrows: point OUTWARD to each pole ----------------------
    arrow_pad = span * 0.06
    arrow_len = span * 0.10
    # Private pole (left): arrow points LEFT, away from the axis centre
    ax.annotate(
        "", xy=(smin - arrow_pad - arrow_len, y0),
        xytext=(smin - arrow_pad, y0),
        arrowprops=dict(arrowstyle="-|>", color=COL_AXIS,
                        linewidth=1.6, mutation_scale=14),
        zorder=2,
    )
    ax.text(smin - arrow_pad - arrow_len - span * 0.015, y0, "private",
            fontsize=10, fontweight="bold", ha="right", va="center",
            color=COL_AXIS)

    # Public pole (right): arrow points RIGHT, away from the axis centre
    ax.annotate(
        "", xy=(smax + arrow_pad + arrow_len, y0),
        xytext=(smax + arrow_pad, y0),
        arrowprops=dict(arrowstyle="-|>", color=COL_AXIS,
                        linewidth=1.6, mutation_scale=14),
        zorder=2,
    )
    ax.text(smax + arrow_pad + arrow_len + span * 0.015, y0, "public",
            fontsize=10, fontweight="bold", ha="left", va="center",
            color=COL_AXIS)

    # ----- zero tick --------------------------------------------------------
    ax.plot([0, 0], [y0 - 0.045, y0 + 0.045], color=COL_AXIS, linewidth=1.0)
    ax.text(0, y0 - 0.085, "0", ha="center", va="top", fontsize=8,
            color="0.30")

    # ----- anchor stems + labels --------------------------------------------
    # Strategy: sweep anchors left-to-right; alternate sides; on each side,
    # assign the SMALLEST tier whose vertical row is empty within a generous
    # horizontal exclusion window. The exclusion window is sized to a
    # plausible label width so neighbouring labels never touch.
    n = len(scores)
    row_h = 0.18
    # Estimated half-width of a typical anchor label, in axis (score) units.
    # Chosen empirically against the longest anchor here ("constitution").
    min_x_sep = span * 0.16

    sides = ["above" if i % 2 == 0 else "below" for i in range(n)]
    placed = {"above": [], "below": []}
    tier_options = [1, 2, 3, 4, 5]

    for i, (term, s) in enumerate(scores):
        side = sides[i]
        chosen_tier = tier_options[-1]
        for t in tier_options:
            ok = True
            for (xs, tt) in placed[side]:
                if tt == t and abs(s - xs) < min_x_sep:
                    ok = False
                    break
            if ok:
                chosen_tier = t
                break
        placed[side].append((s, chosen_tier))

        if side == "above":
            y_label = y0 + row_h * chosen_tier
            va = "bottom"
            connect_y0 = y0 + 0.040
            text_offset = +0.015
        else:
            y_label = y0 - row_h * chosen_tier
            va = "top"
            connect_y0 = y0 - 0.040
            text_offset = -0.015

        # Stem (vertical thin line, deep blue, subtle)
        ax.plot([s, s], [connect_y0, y_label], color=COL_ANCHOR,
                linewidth=0.7, linestyle="-", alpha=0.45, zorder=3)
        # Marker on the axis
        ax.plot([s], [y0], marker="o", markersize=5.5,
                color=COL_ANCHOR, markerfacecolor=COL_ANCHOR,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        # Label
        ax.text(s, y_label + text_offset, term, ha="center", va=va,
                fontsize=8.5, color=COL_ANCHOR, fontweight="bold")

    # ----- cosmetics --------------------------------------------------------
    ax.set_xlim(smin - span * 0.32, smax + span * 0.32)
    ax.set_ylim(-1.10, 1.10)
    ax.set_yticks([])
    ax.set_xlabel(r"projection score   $s(t) = \langle v(t),\;$axis$\rangle$",
                  fontsize=10, labelpad=6)
    ax.tick_params(axis="x", labelsize=8)
    for s_name in ("top", "right", "left"):
        ax.spines[s_name].set_visible(False)
    # Keep bottom spine, thin
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("0.55")

    ax.set_title(
        "(b)  Projection of selected legal terms  ·  BGE-EN-large, attested",
        loc="left", fontsize=11, fontweight="bold", pad=10,
    )
    ax.text(0.0, 1.005,
            f"pool score range  [{smin:+.3f}, {smax:+.3f}]   ·   "
            f"mean  {float(all_scores.mean()):+.3f}",
            transform=ax.transAxes, fontsize=8.5, style="italic",
            color="0.35", ha="left", va="bottom")


def make_fig9() -> None:
    fig = plt.figure(figsize=(11.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.18)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    panel_a_schematic(axL)
    panel_b_projection(axR)
    out = OUT / "fig9_axis_construction.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.9 (schematic of axis construction, public_private)...")
    make_fig9()
    print("\nDone.")
