#!/usr/bin/env python3
"""
Generate paper-style figure for §3.2.4 (Cross-linguistic agreement per axis).

Fig.12 — Forest plot of cross-tradition mean Spearman ρ̄ per axis (run #4
         post-BLP, N=364), in the attested (●), bare (□) and bilingual
         β-control (△) regimes. Axes ordered most-divergent first in the
         attested column. The signal-panel-only 9-pair spread is shown as
         a horizontal whisker behind the attested marker. A within-tradition
         baseline band (§3.2.3) is shaded on the right as reference.

Design: high-end scientific paper style (Nature / Distill.pub / IEEE T.).
Outputs PNG @ 300 dpi in this directory.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# -- Shared design system -----------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Palette
COL_ATTESTED = "#1F4E79"   # deep blue, full saturation (signal panel reference)
COL_BARE = "#5B82A8"       # lighter hue of the attested blue
COL_BETA = "#7A7A7A"       # medium grey for bilingual β control
COL_WHISKER = "#5B82A8"    # whisker hue matched to bare marker
COL_BASELINE = "#E8E8E8"   # within-tradition baseline band
COL_ZERO = "#9A9A9A"       # vertical zero line

REPO = Path(__file__).resolve().parents[4]
RES_ATT = (
    REPO
    / "experiments"
    / "ch3-measurability"
    / "experiment_2_axes"
    / "results_attested"
    / "experiment_2_results.json"
)
RES_BAR = (
    REPO
    / "experiments"
    / "ch3-measurability"
    / "experiment_2_axes"
    / "results_bare"
    / "experiment_2_results.json"
)
OUT = Path(__file__).parent

# Signal-panel encoders (3 EN x 3 ZH). The 9 cross-tradition pairs of the
# signal panel are computed from this product, in either direction.
SIGNAL_EN = {"BGE-EN-large", "E5-large", "FreeLaw-EN"}
SIGNAL_ZH = {"BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"}

# Bilingual β-control pairs: same encoder identity across the two language
# sides (BGE-M3 and Qwen3-Embedding-0.6B). Verified against
# section_323.per_pair model labels.
BILINGUAL_PAIRS = {
    frozenset({"BGE-M3-EN", "BGE-M3-ZH"}),
    frozenset({"Qwen3-0.6B-EN", "Qwen3-0.6B-ZH"}),
}

# Axis display labels (compact paper form). Underscores expanded to slashes.
AXIS_LABEL = {
    "natural_positive": "natural / positive",
    "state_market": "state / market",
    "individual_collective": "individual / collective",
    "public_private": "public / private",
    "status_contract": "status / contract",
    "rights_duties": "rights / duties",
}


def signal_panel_cross_pair(item: dict) -> bool:
    """True iff the per-pair entry is a cross-tradition pair restricted to
    the 9 signal-panel monolingual EN × monolingual ZH pairs (either order)."""
    if item.get("group") != "cross":
        return False
    a, b = item.get("model_a"), item.get("model_b")
    return (a in SIGNAL_EN and b in SIGNAL_ZH) or (a in SIGNAL_ZH and b in SIGNAL_EN)


def aggregate_cross_per_axis(per_pair: list[dict], signal_only: bool) -> dict[str, list[float]]:
    """Return {axis: [rho per cross pair]}."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for item in per_pair:
        if item.get("group") != "cross":
            continue
        if signal_only and not signal_panel_cross_pair(item):
            continue
        bucket[item["axis"]].append(float(item["rho"]))
    return bucket


def bilingual_beta_per_axis(per_pair: list[dict]) -> dict[str, float]:
    """Return {axis: mean ρ over the bilingual β-control pairs}."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for item in per_pair:
        pair_set = frozenset({item.get("model_a"), item.get("model_b")})
        if pair_set in BILINGUAL_PAIRS:
            bucket[item["axis"]].append(float(item["rho"]))
    return {ax: float(sum(vs) / len(vs)) for ax, vs in bucket.items() if vs}


def make_fig12() -> None:
    res_att = json.load(RES_ATT.open())
    res_bar = json.load(RES_BAR.open())

    # Published 25-pair aggregation (matches section_324)
    published_att = res_att["section_324"]["cross_rho_mean_per_axis"]
    published_bar = res_bar["section_324"]["cross_rho_mean_per_axis"]

    # 9-pair signal-panel-only spread, attested only (for whisker)
    spread_att = aggregate_cross_per_axis(
        res_att["section_323"]["per_pair"], signal_only=True
    )

    # Bilingual β-control mean per axis (attested), used as third marker
    beta_att = bilingual_beta_per_axis(res_att["section_323"]["per_pair"])

    # Order axes by published attested ρ̄ cross, most-divergent first
    order = sorted(published_att.keys(), key=lambda a: published_att[a])

    # Y-axis layout: top = most divergent (first in `order`)
    n = len(order)
    y_positions = np.arange(n)[::-1].astype(float)
    offset = 0.22  # vertical offset between attested (center), bare (above), β (below)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # Within-tradition baseline reference band (range from §3.2.3) — drawn
    # first so it sits behind every data element.
    ax.axvspan(0.55, 0.75, color=COL_BASELINE, alpha=0.6, zorder=0, lw=0)
    ax.text(
        0.65, n - 0.45,
        "within-tradition\nbaseline (§3.2.3)",
        fontsize=9, ha="center", va="top", style="italic",
        color="#5C5C5C",
    )

    # Vertical zero reference line
    ax.axvline(
        0.0, color=COL_ZERO, linewidth=0.7, linestyle=(0, (1, 2)), zorder=1,
    )

    # Faint horizontal guideline per axis (paper-style row separator)
    for y in y_positions:
        ax.hlines(
            y=y, xmin=-0.03, xmax=0.78,
            colors="#F1F1F1", linewidth=0.6, zorder=0,
        )

    for i, axis in enumerate(order):
        y = y_positions[i]
        # Signal-panel-only spread as horizontal whisker behind the markers
        spread_vals = np.array(spread_att.get(axis, []), dtype=float)
        if len(spread_vals) >= 2:
            lo = float(np.min(spread_vals))
            hi = float(np.max(spread_vals))
            ax.hlines(
                y=y, xmin=lo, xmax=hi,
                colors=COL_WHISKER, linewidth=0.8, alpha=0.6,
                linestyles="-", zorder=2,
            )
            # Whisker terminator caps
            for x_cap in (lo, hi):
                ax.vlines(
                    x=x_cap, ymin=y - 0.06, ymax=y + 0.06,
                    colors=COL_WHISKER, linewidth=0.8, alpha=0.6, zorder=2,
                )
            # Individual signal-pair tick marks
            for v in spread_vals:
                ax.vlines(
                    x=v, ymin=y - 0.04, ymax=y + 0.04,
                    colors=COL_WHISKER, linewidth=0.7, alpha=0.45, zorder=2,
                )

        # Bare marker (□ open, lighter blue) — above center
        ax.plot(
            [published_bar[axis]], [y + offset],
            marker="s", markerfacecolor="white",
            markeredgecolor=COL_BARE, markeredgewidth=1.2,
            markersize=7.0, linestyle="None", zorder=4,
        )
        # Attested marker (● filled deep blue) — center
        ax.plot(
            [published_att[axis]], [y],
            marker="o", markerfacecolor=COL_ATTESTED,
            markeredgecolor=COL_ATTESTED, markeredgewidth=0.8,
            markersize=8.0, linestyle="None", zorder=5,
        )
        # Bilingual β-control marker (△ open grey) — below center
        if axis in beta_att:
            ax.plot(
                [beta_att[axis]], [y - offset],
                marker="^", markerfacecolor="white",
                markeredgecolor=COL_BETA, markeredgewidth=1.2,
                markersize=7.5, linestyle="None", zorder=4,
            )

    # Y-axis labels (rotation 0, regular weight)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [AXIS_LABEL[a] for a in order],
        fontsize=11, rotation=0,
    )
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, pad=4)

    # X-axis: clean major ticks at 0.0 .. 0.7
    ax.set_xticks(np.arange(0.0, 0.71, 0.1))
    ax.set_xlim(-0.04, 0.78)
    ax.set_ylim(-0.95, n - 0.30)

    ax.set_xlabel(
        r"cross-tradition mean Spearman $\bar\rho$  (per axis)",
        fontsize=11, labelpad=8,
    )

    # Title (left-aligned, paper-style)
    ax.set_title(
        "Cross-tradition agreement per axis",
        fontsize=12, fontweight="bold", loc="left", pad=30,
    )

    # Legend rendered as a proper matplotlib legend using proxy artists.
    # This avoids mathtext / unicode-glyph font-fallback issues and yields
    # a typographically uniform legend strip just under the title.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COL_ATTESTED, markeredgecolor=COL_ATTESTED,
               markersize=8.0, label="attested"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="white", markeredgecolor=COL_BARE,
               markeredgewidth=1.2, markersize=7.0, label="bare"),
        Line2D([0], [0], marker="^", color="none",
               markerfacecolor="white", markeredgecolor=COL_BETA,
               markeredgewidth=1.2, markersize=7.5,
               label=r"bilingual $\beta$-control"),
        Line2D([0], [0], color=COL_WHISKER, linewidth=0.8, alpha=0.6,
               label="spread of the 9 signal-panel pairs (attested)"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="lower left", bbox_to_anchor=(0.0, 1.02),
        ncol=4, frameon=False,
        handletextpad=0.5, columnspacing=1.6,
        fontsize=9, labelcolor="#3A3A3A",
    )
    for text in leg.get_texts():
        text.set_color("#3A3A3A")

    # Spines polish
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8C8C8")
    ax.spines["bottom"].set_color("#3A3A3A")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "fig12_cross_tradition_per_axis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.12 (per-axis cross-tradition ρ̄ forest plot)...")
    make_fig12()
    print("\nDone.")
