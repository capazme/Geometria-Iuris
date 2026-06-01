#!/usr/bin/env python3
"""
Generate paper-style figure for §3.1.1 Movement III (legal vs control).

Fig.3 — Forest plot of Δmedian = median(legal-control distances) −
        median(legal-legal distances) for the ten readings of the cohort
        in canonical display order (EN-side top, ZH-side bottom). Filled
        diamond = signal-panel monolingual encoder (deep blue), open
        square = bilingual control reading (muted grey). 95% CI from
        non-parametric bootstrap (B=1000, seed=42) resampling
        independently the legal-legal and legal-control populations.
        The Δ<0 half-plane is shaded in light pink to mark the failure
        region; a solid vertical at Δ=0 acts as failure threshold.
        Readings to the left of zero correspond to encoders for which
        the ordinary vocabulary sits, on average, closer to legal
        vocabulary than legal vocabulary sits to itself.

Outputs PNG @ 300 dpi in this directory. File name is kept identical to
the previous boxplot version for caption/path compatibility.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parents[4]
EMB = REPO / "experiments" / "ch3-measurability" / "embeddings"
OUT = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Shared high-end design system (Nature / Distill / IEEE)                     #
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Palette
COLOR_SIGNAL = "#1F3A93"        # deep saturated blue (signal panel)
COLOR_CONTROL = "#8A8F99"       # muted grey (bilingual control)
COLOR_FAILURE_FILL = "#D32F2F"  # warm red (failure half-plane shading)
COLOR_ZERO = "#222222"          # near-black (failure threshold axis)
COLOR_SEPARATOR = "#B8BEC7"     # light grey (EN/ZH divider)
COLOR_TEXT_MUTED = "#5A6068"    # muted text for control labels and side tags


# Canonical display order (mirrors loader_31.py ALL_MODELS_ORDERED): EN-side
# encoders first, then ZH-side; within each side, monolingual signal-panel
# first, bilingual control next.
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


def _load_pair(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (legal-legal distances, legal-control distances) bare."""
    vL = np.load(EMB / model / "vecs_bare.npy").astype(np.float64)
    vC = np.load(EMB / "control_bare" / model / "vecs.npy").astype(np.float64)
    vL /= np.linalg.norm(vL, axis=1, keepdims=True).clip(1e-12)
    vC /= np.linalg.norm(vC, axis=1, keepdims=True).clip(1e-12)
    sim = vL @ vL.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dist_LL = 1.0 - sim
    iu, ju = np.triu_indices(len(vL), k=1)
    LL = dist_LL[iu, ju]
    LC = cdist(vL, vC, metric="cosine").ravel()
    return LL, LC


def _bootstrap_delta_ci(
    LL: np.ndarray,
    LC: np.ndarray,
    B: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for Δmedian = median(LC) − median(LL).

    Resamples LL and LC independently with replacement, recomputes the
    Δmedian per replicate, returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    nLL, nLC = len(LL), len(LC)
    deltas = np.empty(B)
    for b in range(B):
        ll_b = rng.choice(LL, size=nLL, replace=True)
        lc_b = rng.choice(LC, size=nLC, replace=True)
        deltas[b] = np.median(lc_b) - np.median(ll_b)
    point = float(np.median(LC) - np.median(LL))
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return point, lo, hi


def make_fig3_forest() -> None:
    """Forest plot of Δmedian (legal-control − legal-legal) per encoder."""
    labels = ALL_MODELS_ORDERED
    data: dict[str, tuple[float, float, float]] = {}
    for m in labels:
        LL, LC = _load_pair(m)
        point, lo, hi = _bootstrap_delta_ci(LL, LC, B=1000, seed=42)
        data[m] = (point, lo, hi)
        print(f"  {m:22s}  Δmed = {point:+.4f}  CI95 = [{lo:+.4f}, {hi:+.4f}]")

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # y-axis: top = first label, bottom = last
    y_positions = np.arange(len(labels))[::-1]

    # ---- Failure half-plane shading (drawn first, behind everything) ---- #
    # Compute x-extent of the data to set sensible xlim then anchor shading
    all_lo = min(d[1] for d in data.values())
    all_hi = max(d[2] for d in data.values())
    span = all_hi - all_lo
    pad = max(0.004, 0.12 * span)
    xlim_lo = all_lo - pad
    xlim_hi = all_hi + pad
    ax.set_xlim(xlim_lo, xlim_hi)

    # Light pink/red wash on Δ<0 half-plane (failure region)
    ax.axvspan(xlim_lo, 0.0, alpha=0.12, color=COLOR_FAILURE_FILL,
               zorder=0, linewidth=0)

    # ---- Per-encoder forest entries ---- #
    for i, lab in enumerate(labels):
        y = y_positions[i]
        point, lo, hi = data[lab]
        is_signal = lab in SIGNAL_PANEL
        if is_signal:
            marker = "D"            # filled diamond
            mfc = COLOR_SIGNAL
            mec = COLOR_SIGNAL
            ms = 7.5
            ew = 1.3
            ecolor = COLOR_SIGNAL
        else:
            marker = "s"            # open square
            mfc = "white"
            mec = COLOR_CONTROL
            ms = 6.5
            ew = 1.0
            ecolor = COLOR_CONTROL
        ax.errorbar(
            point, y,
            xerr=[[point - lo], [hi - point]],
            fmt=marker,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markeredgewidth=1.2,
            markersize=ms,
            ecolor=ecolor,
            elinewidth=ew,
            capsize=3.2,
            capthick=ew,
            zorder=3,
        )

    # ---- Failure threshold at Δ=0 (solid, near-black, medium-thick) ---- #
    ax.axvline(0.0, color=COLOR_ZERO, linewidth=1.1, linestyle="-",
               zorder=2)

    # ---- Horizontal separator between EN-side and ZH-side ---- #
    sep_y = y_positions[5] + 0.5
    ax.axhline(sep_y, color=COLOR_SEPARATOR, linewidth=0.8,
               linestyle="-", zorder=1)

    # ---- Y tick labels: italics + muted for bilingual controls ---- #
    ax.set_yticks(y_positions)
    yticklabels = [DISP[lab] for lab in labels]
    ax.set_yticklabels(yticklabels, fontsize=9.5)
    for tick, lab in zip(ax.get_yticklabels(), labels):
        if lab not in SIGNAL_PANEL:
            tick.set_style("italic")
            tick.set_color(COLOR_TEXT_MUTED)
        else:
            tick.set_color("#1A1A1A")

    # ---- Side annotations: EN-side / ZH-side (elegant, vertical) ---- #
    en_y = float(np.mean(y_positions[:5]))
    zh_y = float(np.mean(y_positions[5:]))
    ax.text(1.015, en_y, "EN-side", transform=ax.get_yaxis_transform(),
            fontsize=9.5, ha="left", va="center", style="italic",
            color=COLOR_TEXT_MUTED, rotation=270, weight="semibold")
    ax.text(1.015, zh_y, "ZH-side", transform=ax.get_yaxis_transform(),
            fontsize=9.5, ha="left", va="center", style="italic",
            color=COLOR_TEXT_MUTED, rotation=270, weight="semibold")

    # ---- Axis labels and title (premium typography) ---- #
    ax.set_xlabel(
        r"$\Delta_{\mathrm{median}}\ =\ \mathrm{median}(\mathrm{legal\!-\!control})\ -\ \mathrm{median}(\mathrm{legal\!-\!legal})$",
        fontsize=10.5, labelpad=8,
    )

    # Title + subtitle (two-line, hierarchical)
    fig.suptitle(
        "Legal-vs-control median shift across the cohort",
        x=0.5, y=0.985,
        fontsize=12.5, fontweight="bold", ha="center",
        color="#0E1116",
    )
    ax.set_title(
        r"filled diamond = signal panel   $\cdot$   "
        r"open square = bilingual control   $\cdot$   "
        r"$\Delta<0$ marks the failure modes",
        fontsize=9.5, pad=8, fontweight="normal",
        color=COLOR_TEXT_MUTED, loc="center",
    )

    # ---- Cosmetic axis / tick tweaks ---- #
    ax.tick_params(axis="x", length=4, width=0.8, color="#444")
    ax.tick_params(axis="y", length=0)  # no y-ticks (labels carry identity)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    # Optional very-faint vertical gridlines for read-off (paper-style)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.grid(axis="x", which="major", color="#E5E7EB",
            linewidth=0.6, zorder=0)

    # Provenance footer (very small, low-contrast)
    fig.text(
        0.99, 0.005,
        "bare regime  ·  B=1000 bootstrap  ·  seed=42  ·  N=10 readings",
        ha="right", va="bottom",
        fontsize=7.5, color="#9AA0A6", style="italic",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = OUT / "fig3_boxplot_legal_vs_control.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\n→ wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.3 (forest plot of Δmedian legal-control vs legal-legal, "
          "B=1000 bootstrap, seed=42)...")
    make_fig3_forest()
    print("Done.")
