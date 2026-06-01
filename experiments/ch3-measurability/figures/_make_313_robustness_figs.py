#!/usr/bin/env python3
"""
Generate paper-style figures for the two §3.1.3 robustness curves
(currently dashboard screenshots in the draft):

Fig.8 — Background injection (Extension D). Δρ̄_sym (attested) recomputed
        as out-of-curation legal vocabulary is mixed into the 364-term
        pool, at five mix levels. The gap does not decay.
Fig.9 — Control injection (Extension X, the dual of D). Δρ̄_sym (bare)
        recomputed as everyday-language control vocabulary is mixed into
        the bare pool, at seven mix levels. The gap erodes monotonically.

Inputs (frozen JSON, no recomputation):
    ext/D_robustness/robustness_curve.json
    ext/X_control_robustness/control_robustness_curve.json

Outputs PNG @ 300 dpi in this directory:
    fig_D_background_injection.png
    fig_X_control_injection.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Shared design system (matches _make_313_figs.py: Nature / Distill register)
# ---------------------------------------------------------------------------
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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

REPO = Path(__file__).resolve().parents[4]
EXT = REPO / "experiments" / "ch3-measurability" / "ext"
OUT = Path(__file__).parent

# House palette (shared with the other §3.1.3 figures)
SIGNAL = "#1F4E79"   # deep blue  -> attested / legal-signal register
CONTROL = "#7A7A7A"  # muted grey -> bare / control register
INK = "#1A1A1A"
SUBTLE = "#444444"


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        return json.load(fh)["results"]


def _draw_curve(ax, x, y, lo, hi, color, band_alpha):
    """Shared grammar: CI band, connecting line, markers."""
    ax.fill_between(x, lo, hi, color=color, alpha=band_alpha,
                    linewidth=0, zorder=1)
    ax.plot(x, y, color=color, linewidth=2.0, zorder=3,
            solid_capstyle="round")
    ax.plot(x, y, "o", color=color, markersize=6.5,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=1.0, zorder=4)


def _style_axes(ax):
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.30,
                  color=SUBTLE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="both", length=3, width=0.8)
    ax.tick_params(axis="x", which="both", length=4, width=0.8)


# ============================================================================
# Fig.8 — Background injection (Extension D), attested
# ============================================================================
def make_fig_D() -> None:
    rows = load(EXT / "D_robustness" / "robustness_curve.json")
    x = np.array([r["p_bg"] * 100 for r in rows])
    y = np.array([r["mean_delta_sym"] for r in rows])
    lo = np.array([r["ci_low_delta_sym"] for r in rows])
    hi = np.array([r["ci_high_delta_sym"] for r in rows])
    baseline = y[0]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    _draw_curve(ax, x, y, lo, hi, SIGNAL, 0.14)
    _style_axes(ax)

    # baseline reference
    ax.axhline(baseline, color=CONTROL, linewidth=1.0,
               linestyle=(0, (4, 3)), alpha=0.8, zorder=2)
    ax.text(x[-1], baseline - 0.006, f"baseline {baseline:.3f}",
            fontsize=8.5, ha="right", va="top", color=CONTROL)

    # endpoint annotation
    ax.annotate(f"{y[-1]:.3f} at 75%",
                xy=(x[-1], y[-1]), xytext=(x[-1] - 11, y[-1] + 0.018),
                fontsize=9.5, color=SIGNAL, fontweight="bold",
                ha="center", va="bottom")

    ax.set_xlim(-3, 80)
    ax.set_xticks([0, 10, 25, 50, 75])
    ax.set_xticklabels(["0%", "10%", "25%", "50%", "75%"])
    ax.set_ylim(0.46, 0.66)
    ax.set_yticks(np.arange(0.46, 0.661, 0.04))
    ax.set_xlabel("% background terms in pool")
    ax.set_ylabel(r"$\Delta\bar{\rho}_{\mathrm{sym}}$   (attested,  within $-$ cross)")

    ax.set_title("The cross-tradition gap holds under background injection",
                 loc="left", pad=24, color=INK)
    ax.text(0.0, 1.045,
            "Out-of-curation legal vocabulary mixed into the 364-term pool; "
            r"$\Delta\bar{\rho}_{\mathrm{sym}}$ does not decay, if anything it rises.",
            transform=ax.transAxes, fontsize=9.5, color=SUBTLE,
            ha="left", va="bottom", style="italic")
    ax.text(0.0, -0.235,
            "Five mix levels, ten pool re-draws per level (two EN-side and two ZH-side "
            "encoders, attested).\nShaded band: 5–95% across re-draws.",
            transform=ax.transAxes, fontsize=8, color="0.45",
            ha="left", va="top", style="italic", linespacing=1.3)

    fig.tight_layout()
    out = OUT / "fig_D_background_injection.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"→ wrote {out.name}")


# ============================================================================
# Fig.9 — Control injection (Extension X, dual of D), bare
# ============================================================================
def make_fig_X() -> None:
    rows = load(EXT / "X_control_robustness" / "control_robustness_curve.json")
    x = np.array([r["p_control"] * 100 for r in rows])
    y = np.array([r["mean_delta_sym"] for r in rows])
    lo = np.array([r["ci_low_delta_sym"] for r in rows])
    hi = np.array([r["ci_high_delta_sym"] for r in rows])
    baseline = y[0]
    drift = y[-1] - y[0]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    _draw_curve(ax, x, y, lo, hi, CONTROL, 0.20)
    _style_axes(ax)

    # baseline reference
    ax.axhline(baseline, color=SIGNAL, linewidth=1.0,
               linestyle=(0, (4, 3)), alpha=0.55, zorder=2)
    ax.text(0.0, baseline + 0.0035, f"baseline {baseline:.3f}",
            fontsize=8.5, ha="left", va="bottom", color=SIGNAL, alpha=0.9)

    # endpoint + drift annotation
    ax.annotate(f"{y[-1]:.3f} at 27%",
                xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] - 0.006),
                fontsize=9.5, color=INK, fontweight="bold",
                ha="right", va="top")
    ax.text(0.5, 0.06, f"total drift {drift:+.3f}",
            transform=ax.transAxes, fontsize=9, color=CONTROL,
            ha="center", va="bottom", style="italic")

    ax.set_xlim(-1.5, 29)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%", "25%", "27%"])
    ax.set_ylim(0.193, 0.256)
    ax.set_yticks(np.arange(0.20, 0.251, 0.01))
    ax.set_xlabel("% control terms in pool")
    ax.set_ylabel(r"$\Delta\bar{\rho}_{\mathrm{sym}}$   (bare,  within $-$ cross)")

    ax.set_title("Control injection erodes the bare gap, as the construction predicts",
                 loc="left", pad=24, color=INK)
    ax.text(0.0, 1.045,
            "Everyday-language vocabulary mixed into the bare pool: the dual of the "
            "background-injection check.",
            transform=ax.transAxes, fontsize=9.5, color=SUBTLE,
            ha="left", va="bottom", style="italic")
    ax.text(0.0, -0.235,
            "Seven mix levels, fifteen pool re-draws per level. The controls carry no "
            "attested reading, so the curve runs on bare embeddings.\nShaded band: "
            "5–95% across re-draws.",
            transform=ax.transAxes, fontsize=8, color="0.45",
            ha="left", va="top", style="italic", linespacing=1.3)

    fig.tight_layout()
    out = OUT / "fig_X_control_injection.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.8 (background injection, Extension D)...")
    make_fig_D()
    print("Generating Fig.9 (control injection, Extension X)...")
    make_fig_X()
    print("Done.")
