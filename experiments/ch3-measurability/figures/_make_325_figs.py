#!/usr/bin/env python3
"""
Generate high-end scientific paper figure for §3.2.5 (between-group differences).

Fig.13 — Top-5 cross-tradition divergent terms per axis (signal panel only,
         attested). 3×2 grid, one panel per axis. Per panel: paired horizontal
         bars per term (WEIRD vs Sinic mean projection), with |Δ| annotation.

Design: Nature / Distill.pub / IEEE Transactions inspired.
  - palette: WEIRD = deep blue #1F4E79, Sinic = deep red/burgundy #A4262C
  - typography: Inter → Helvetica Neue fallback chain with CJK fallback
  - rcParams: hairline spines, no top/right, 300 dpi, white background

Source: recompute on-the-fly signal-only (3+3 monolingual) from the frozen
        per-encoder axis-score .npy files. The numbers in `section_325`
        of `experiment_2_results.json` aggregate 5+5 (incl. bilingual
        controls); §2.3 cohort design requires signal-only for the
        cross-tradition claims, so the figure uses the recomputed 3+3.

Output: fig13_top_divergent_terms.png @ 300 dpi.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------------
# Design system (shared with other Ch.3 figures)
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    # Inter (preferred) → Helvetica Neue → Helvetica → Arial → DejaVu Sans
    # with CJK fallback (Arial Unicode MS / Hiragino Sans GB / Heiti TC) so
    # that simplified and traditional Chinese glyphs resolve without boxes.
    "font.sans-serif": [
        "Inter", "Helvetica Neue", "Helvetica", "Arial",
        "DejaVu Sans", "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC",
    ],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 0.0,         # no y-ticks (labels are bar names)
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# Palette
COLOR_WEIRD = "#1F4E79"   # deep blue
COLOR_SINIC = "#A4262C"   # deep red / burgundy
COLOR_ZERO = "#7A7A7A"    # dotted neutral
COLOR_DELTA = "#4D4D4D"   # |Δ| annotation grey ("0.30")
COLOR_TITLE = "#1A1A1A"
COLOR_SUBTITLE = "#4D4D4D"

# Dedicated FontProperties for term labels that mix Latin + CJK glyphs.
# Arial Unicode MS contains both Latin and CJK glyphs natively, so a single
# resolved font handles "estate / 產業" without falling back glyph-by-glyph
# (which silently emits tofu boxes for some matplotlib configurations).
_CJK_CANDIDATES = ["Arial Unicode MS", "Hiragino Sans GB", "Hiragino Sans",
                   "Heiti TC", "Songti SC", "STHeiti"]
CJK_FONT = FontProperties(family=_CJK_CANDIDATES, size=9)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / "experiments" / "ch3-measurability"
SCORES = ROOT / "experiment_2_axes" / "results_attested" / "scores"
EMB = ROOT / "embeddings"
OUT = Path(__file__).parent

# Cohort definition (signal panel, §2.3)
EN_MONO = ["BGE-EN-large", "E5-large", "FreeLaw-EN"]
ZH_MONO = ["BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"]

# Axis order: most-divergent first (cf. §3.2.4 ranking)
AXES_ORDERED = [
    "natural_positive",
    "state_market",
    "individual_collective",
    "public_private",
    "status_contract",
    "rights_duties",
]

# Display titles for axes (en-dash separator)
AXIS_TITLE = {
    "natural_positive": "natural–positive",
    "state_market": "state–market",
    "individual_collective": "individual–collective",
    "public_private": "public–private",
    "status_contract": "status–contract",
    "rights_duties": "rights–duties",
}

# Pool-sensitivity tag (HANDOFF framing C)
AXIS_POOL_STABLE = {"natural_positive", "individual_collective", "public_private"}


def load_index() -> list[dict]:
    with (EMB / "index.json").open() as fh:
        return json.load(fh)


def load_score(model: str, axis: str) -> np.ndarray:
    return np.load(SCORES / f"{model}_{axis}.npy")


def compute_top_K_signal(axis: str, K: int = 5) -> list[dict]:
    """Recompute signal-only (3+3) top-K divergent terms for an axis."""
    idx = load_index()
    W = np.mean([load_score(m, axis) for m in EN_MONO], axis=0)
    S = np.mean([load_score(m, axis) for m in ZH_MONO], axis=0)
    delta = W - S
    abs_delta = np.abs(delta)
    top_idx = np.argsort(-abs_delta)[:K]
    rows = []
    for i in top_idx:
        t = idx[int(i)]
        rows.append({
            "en": t["en"],
            "zh": t["zh"],
            "domain": t["domain"],
            "w": float(W[i]),
            "s": float(S[i]),
            "abs_delta": float(abs_delta[i]),
        })
    return rows


def compute_axis_summary(axis: str) -> tuple[float, float]:
    """delta_mean_abs and delta_max_abs computed signal-only."""
    W = np.mean([load_score(m, axis) for m in EN_MONO], axis=0)
    S = np.mean([load_score(m, axis) for m in ZH_MONO], axis=0)
    abs_delta = np.abs(W - S)
    return float(abs_delta.mean()), float(abs_delta.max())


# ============================================================================
# Fig.13 — 3×2 grid of top-5 divergent terms per axis (high-end scientific)
# ============================================================================
def make_fig13(K: int = 5) -> None:
    # 3 rows × 2 cols, ~12×9 inch — paper-style aspect.
    # hspace / wspace set later via subplots_adjust (together with top reserve).
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0))
    axes = axes.flatten()

    # Determine global x-range across all panels so bars are visually comparable
    all_vals = []
    panels = []
    for axis in AXES_ORDERED:
        rows = compute_top_K_signal(axis, K=K)
        mean_abs, max_abs = compute_axis_summary(axis)
        panels.append((axis, rows, mean_abs, max_abs))
        for r in rows:
            all_vals.append(r["w"])
            all_vals.append(r["s"])
    # Generous right-side padding so the |Δ| annotation never overlaps a long bar
    xlim = max(0.45, max(abs(v) for v in all_vals) + 0.13)

    for k, (axis, rows, mean_abs, max_abs) in enumerate(panels):
        ax = axes[k]
        n = len(rows)
        y = np.arange(n)[::-1]  # rank 1 at top
        bar_h = 0.35
        offset = bar_h / 2 + 0.015

        # Labels: "EN / ZH"
        labels = [f'{r["en"]}  /  {r["zh"]}' for r in rows]

        # WEIRD bars — deep blue (top of pair)
        ax.barh(
            y + offset, [r["w"] for r in rows],
            height=bar_h, color=COLOR_WEIRD, edgecolor="none",
            linewidth=0.0, label="WEIRD", zorder=3,
        )
        # Sinic bars — deep burgundy (bottom of pair)
        ax.barh(
            y - offset, [r["s"] for r in rows],
            height=bar_h, color=COLOR_SINIC, edgecolor="none",
            linewidth=0.0, label="Sinic", zorder=3,
        )

        # Zero line — dotted neutral
        ax.axvline(
            0.0, color=COLOR_ZERO, linewidth=0.8, linestyle=(0, (1.4, 1.4)),
            zorder=2,
        )

        # |Δ| annotation, right edge of panel, italic grey
        for i, r in enumerate(rows):
            yy = y[i]
            ax.text(
                xlim - 0.012, yy,
                f"|Δ| = {r['abs_delta']:.3f}",
                fontsize=8.5, ha="right", va="center",
                style="italic", color=COLOR_DELTA, zorder=4,
            )

        # Cosmetics — use CJK-capable FontProperties for term labels (EN / ZH)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontproperties=CJK_FONT)
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-0.7, n - 0.3)
        ax.tick_params(axis="x", labelsize=8, pad=2)
        ax.tick_params(axis="y", length=0, pad=4)

        # Subtle grid: light vertical gridlines at major x ticks
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.4, color="#E5E5E5", zorder=1)
        ax.yaxis.grid(False)

        # Sensible major locator: 0.2 step (–0.4, –0.2, 0, +0.2, +0.4)
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([-0.4, -0.2, 0.0, 0.2, 0.4]))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

        # Panel title (bold axis name) + meta line (mean |Δ|, max |Δ|, pool tag)
        pool_tag = "pool-stable" if axis in AXIS_POOL_STABLE else "pool-sensitive"
        ax.set_title(
            AXIS_TITLE[axis],
            fontsize=11, fontweight="bold", color=COLOR_TITLE,
            loc="left", pad=18,
        )
        # Meta line under the title (drawn just above the axes top edge)
        meta_text = (
            f"top-5 divergent   ·   mean |Δ| = {mean_abs:.3f}   ·   "
            f"max |Δ| = {max_abs:.3f}   ·   {pool_tag}"
        )
        ax.text(
            0.0, 1.02, meta_text,
            transform=ax.transAxes,
            fontsize=8.5, color=COLOR_SUBTITLE,
            ha="left", va="bottom",
        )

        # X-label only on bottom row
        if k >= 4:
            ax.set_xlabel(
                "axis projection score",
                fontsize=9, color=COLOR_TITLE, labelpad=6,
            )

    # Layout: subplots use 0..0.82 of vertical space; top 0.82..1.0 is reserved
    # for suptitle (top), subtitle, legend, in that vertical order. Use
    # subplots_adjust rather than tight_layout to preserve absolute positions.
    fig.subplots_adjust(
        left=0.085, right=0.985,
        bottom=0.06, top=0.80,
        hspace=0.65, wspace=0.42,
    )

    # Suptitle — bold, top
    fig.suptitle(
        "Top-5 cross-tradition divergent terms per axis",
        fontsize=14, fontweight="bold", color=COLOR_TITLE,
        x=0.5, y=0.975,
    )
    # Subtitle line (one notch below suptitle)
    fig.text(
        0.5, 0.935,
        "signal panel (3+3 monolingual encoders), attested contexts   ·   "
        "axes ordered by §3.2.4 cross-tradition divergence",
        fontsize=10, color=COLOR_SUBTITLE,
        ha="center", va="center",
    )

    # Global legend — between subtitle and first panel row
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLOR_WEIRD, edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_SINIC, edgecolor="none"),
    ]
    fig.legend(
        legend_handles,
        ["WEIRD  (mean of 3 monolingual encoders)",
         "Sinic  (mean of 3 monolingual encoders)"],
        loc="upper center", ncol=2, fontsize=10, frameon=False,
        handlelength=1.6, handleheight=1.1, columnspacing=2.8,
        bbox_to_anchor=(0.5, 0.895),
    )

    out = OUT / "fig13_top_divergent_terms.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    # Print the numerical record to stdout for the audit trail
    print("=" * 90)
    print("§3.2.5 — Top-5 cross-tradition divergent terms per axis (signal panel 3+3, attested)")
    print("Source: experiment_2_axes/results_attested/scores/<Encoder>_<axis>.npy")
    print("=" * 90)
    for axis in AXES_ORDERED:
        mean_abs, max_abs = compute_axis_summary(axis)
        rows = compute_top_K_signal(axis, K=5)
        pool = "pool-stable" if axis in AXIS_POOL_STABLE else "pool-sensitive"
        print(f"\n--- {axis}   mean|Δ|={mean_abs:.4f}  max|Δ|={max_abs:.4f}  [{pool}] ---")
        for i, r in enumerate(rows, 1):
            print(f"  {i}. {r['en']:25s} / {r['zh']:10s} [{r['domain']:15s}]  "
                  f"W={r['w']:+.4f}  S={r['s']:+.4f}  |Δ|={r['abs_delta']:.4f}")

    print("\nGenerating Fig.13 (3×2 grid, high-end scientific paper style)...")
    make_fig13(K=5)
    print("\nDone.")
