#!/usr/bin/env python3
"""
Generate paper-style figure for §3.1.4 (categorical probe).

Fig.8 — Five-panel high-end scientific layout (Nature / Distill / IEEE
        Transactions register). One panel per pre-registered test. Each panel
        shows the PC1 projection of the 11 templated categories for a
        representative encoder per test, averaged across the 5 paraphrase
        templates. A dashed amber vertical line marks the pre-registered legal
        threshold; a dotted teal vertical line with open square markers on the
        adjacent categories marks the modal max-gap break for the represented
        encoder. When the two coincide (exact hits T1, T3, T5) a small
        EXACT-HIT badge in the corner records the coincidence and the dashed
        line is offset by half-step to keep both visible.

Output PNG @ 300 dpi in this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Design system (shared across §3.1.4 figures).
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    # Inter is not installed on the build host; the cascade falls through to
    # Helvetica Neue / Helvetica / Arial. Arial Unicode MS + Hiragino Sans GB
    # provide CJK fallback for the Chinese category labels in T1 and T4.
    "font.sans-serif": [
        "Inter", "Helvetica Neue", "Helvetica", "Arial",
        "DejaVu Sans", "Arial Unicode MS", "Hiragino Sans GB",
    ],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})

# Palette ------------------------------------------------------------------
COL_CURVE        = "#1F4E79"   # deep blue, full saturation
COL_LEGAL        = "#D67A1A"   # amber/orange — pre-registered legal threshold
COL_MODAL        = "#2E7570"   # deep teal — encoder modal max-gap break
COL_GRID         = "#E5E5E5"
COL_AXIS_TEXT    = "#2A2A2A"
COL_BADGE_BG     = "#F2EAD8"   # soft amber tint for exact-hit badge
COL_BADGE_EDGE   = "#D67A1A"
COL_BADGE_NEUTRAL_BG   = "#EFEFEF"
COL_BADGE_NEUTRAL_EDGE = "#9A9A9A"

# Explicit CJK FontProperties for tick labels on Chinese-side panels.
# Matplotlib's per-glyph fallback isn't reliable across renderer paths, so we
# pin a known CJK face directly. Arial Unicode covers Simplified + Traditional
# and ships with macOS; on other hosts the cascade below substitutes a sibling.
_CJK_CANDIDATES = [
    "Arial Unicode MS", "Hiragino Sans GB", "Hiragino Sans TC",
    "Heiti TC", "Noto Sans CJK SC", "Noto Sans CJK TC",
]
def _resolve_cjk_font() -> FontProperties:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in _CJK_CANDIDATES:
        if cand in available:
            return FontProperties(family=cand)
    return FontProperties()  # last-resort default
CJK_FP = _resolve_cjk_font()

REPO = Path(__file__).resolve().parents[4]
PROBE = REPO / "experiments" / "ch3-measurability" / "experiment_1_structure" / "results_attested" / "categorical_probe.json"
OUT = Path(__file__).parent

SIGNAL_PANEL = {"BGE-EN-large", "E5-large", "FreeLaw-EN",
                "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"}

# Representative encoder per test (one panel each). Chosen for visual clarity:
# the encoder whose PC1 projection most cleanly displays the dominant
# breakpoint reported in the JSON summary.
REPRESENTATIVE = {
    "test_1_age_imputability":           ("BGE-ZH-large",      "ZH"),
    "test_2_magnitude_negative_control": ("BGE-EN-large",      "EN"),
    "test_3_age_contractual_capacity":   ("FreeLaw-EN",        "EN"),
    "test_4_offence_severity":           ("Dmeta-ZH",          "ZH"),
    "test_5_disposal_severity":          ("BGE-EN-large",      "EN"),
}

PANEL_TITLES = {
    "test_1_age_imputability":           "T1 — age / imputability",
    "test_2_magnitude_negative_control": "T2 — contract value",
    "test_3_age_contractual_capacity":   "T3 — age / capacity",
    "test_4_offence_severity":           "T4 — offence severity",
    "test_5_disposal_severity":          "T5 — disposal severity",
}

# Tag (small caps style) appended below the panel title.
PANEL_TAG = {
    "test_1_age_imputability":           "borderline (Δ midpoint = 1)",
    "test_2_magnitude_negative_control": "negative control",
    "test_3_age_contractual_capacity":   "positive test",
    "test_4_offence_severity":           "positive test",
    "test_5_disposal_severity":          "positive test",
}

# Short labels for axis ticks (to fit horizontally).
LABEL_OVERRIDES_EN = {
    "test_2_magnitude_negative_control": {},
    "test_4_offence_severity": {
        "a bylaw contravention":      "bylaw",
        "a regulatory breach":         "regulatory",
        "a minor infraction":          "minor infr.",
        "a petty offence":             "petty off.",
        "a minor offence":             "minor off.",
        "a simple offence":            "simple off.",
        "a summary offence":           "summary",
        "an indictable offence":       "indictable",
        "a serious indictable offence":"serious ind.",
        "a grave offence":             "grave",
        "the most serious offence":    "most serious",
    },
    "test_5_disposal_severity": {
        "a caution":                       "caution",
        "a fine":                          "fine",
        "a community service order":       "comm. svc",
        "a probation order":               "probation",
        "a suspended sentence":            "suspended",
        "a short custodial sentence":      "short cust.",
        "a determinate prison sentence":   "determinate",
        "a long prison sentence":          "long prison",
        "a very long prison sentence":     "very long",
        "an indeterminate detention":      "indetermin.",
        "life imprisonment":               "life impr.",
    },
}


def short_label(test_id: str, lang: str, raw: str) -> str:
    if lang == "EN":
        ov = LABEL_OVERRIDES_EN.get(test_id, {})
        return ov.get(raw, raw)
    return raw


def ensemble_projection(per_template: list[dict]) -> np.ndarray:
    """Mean PC1 projection across the 5 paraphrase templates.

    Each template was sign-fixed at probe-time so that ρ(idx, proj) >= 0;
    averaging across templates therefore yields a monotone-increasing
    ensemble curve in the ordinal index.
    """
    arr = np.array([t["pc1_projection"] for t in per_template], dtype=np.float64)
    return arr.mean(axis=0)


def add_badge(ax, text, *, color_bg, color_edge, color_text, loc="upper right"):
    """Place a small tag-style badge inside the axes."""
    x = 0.04 if loc.endswith("left") else 0.96
    y = 0.965 if loc.startswith("upper") else 0.05
    ha = "left" if loc.endswith("left") else "right"
    va = "top"  if loc.startswith("upper") else "bottom"
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=6.8,
        fontweight="semibold",
        color=color_text,
        bbox=dict(
            boxstyle="round,pad=0.28,rounding_size=0.25",
            facecolor=color_bg,
            edgecolor=color_edge,
            linewidth=0.7,
        ),
        zorder=10,
    )


def make_fig8() -> None:
    data = json.loads(PROBE.read_text(encoding="utf-8"))
    tests = data["tests"]

    test_ids = [
        "test_1_age_imputability",
        "test_2_magnitude_negative_control",
        "test_3_age_contractual_capacity",
        "test_4_offence_severity",
        "test_5_disposal_severity",
    ]

    fig, axes = plt.subplots(
        1, 5,
        figsize=(16.0, 5.0),
        gridspec_kw={"wspace": 0.36},
    )
    # Reserve generous top room for the three-line title block above each axes
    # and bottom room for rotated CJK/EN tick labels + shared legend.
    fig.subplots_adjust(top=0.78, bottom=0.30, left=0.045, right=0.995)

    for ax, tid in zip(axes, test_ids):
        t = tests[tid]
        encoder, lang = REPRESENTATIVE[tid]
        per_model = t["per_model"][encoder]
        ens_proj = ensemble_projection(per_model["per_template"])

        cats_raw = t["categories_en"] if lang == "EN" else t["categories_zh"]
        cats = [short_label(tid, lang, c) for c in cats_raw]
        n = len(cats)
        x = np.arange(n)

        # Light horizontal grid for read-off support; behind everything.
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=COL_GRID, linewidth=0.6, zorder=0)
        ax.xaxis.grid(False)

        # --- Reference lines ------------------------------------------------
        exp_idx = t["expected_gap_index"]
        ens = per_model["ensemble"]
        modal_pos = ens["modal_max_gap_position"]
        modal_freq = ens["modal_max_gap_freq"]

        coincides = (
            exp_idx is not None and exp_idx >= 0
            and 0 <= modal_pos < n - 1
            and modal_pos == exp_idx
        )

        # Legal threshold (orange dashed). Offset by a small horizontal nudge
        # when it coincides with the modal break, so both lines remain visible.
        if exp_idx is not None and exp_idx >= 0:
            legal_x = exp_idx + 0.5 + (-0.06 if coincides else 0.0)
            ax.axvline(
                legal_x,
                color=COL_LEGAL, linestyle=(0, (5, 3)),
                linewidth=1.4, alpha=0.95, zorder=2,
            )

        # Modal max-gap (teal dotted) + open-square markers on adjacent points.
        if 0 <= modal_pos < n - 1:
            modal_x = modal_pos + 0.5 + (0.06 if coincides else 0.0)
            ax.axvline(
                modal_x,
                color=COL_MODAL, linestyle=(0, (1, 2.2)),
                linewidth=1.7, alpha=0.95, zorder=2,
            )
            ax.scatter(
                [modal_pos, modal_pos + 1],
                [ens_proj[modal_pos], ens_proj[modal_pos + 1]],
                s=72, marker="s",
                facecolors="white",
                edgecolors=COL_MODAL,
                linewidths=1.5, zorder=6,
            )

        # --- Main projection curve (drawn on top of refs for legibility) ----
        ax.plot(
            x, ens_proj,
            color=COL_CURVE, linewidth=1.6,
            marker="o", markersize=4.2,
            markerfacecolor=COL_CURVE,
            markeredgecolor=COL_CURVE,
            markeredgewidth=0.0,
            zorder=4,
            solid_capstyle="round",
        )

        # --- Cosmetics ------------------------------------------------------
        ax.set_xticks(x)
        # Pin a CJK font on the Chinese-side panels so glyphs render at 300 dpi
        # without falling back to a tofu rectangle.
        tick_kwargs = dict(
            rotation=55, ha="right",
            fontsize=7.0, color=COL_AXIS_TEXT,
        )
        if lang == "ZH":
            tick_kwargs["fontproperties"] = CJK_FP
        ax.set_xticklabels(cats, **tick_kwargs)
        ax.tick_params(axis="y", labelsize=8, colors=COL_AXIS_TEXT)
        ax.tick_params(axis="x", colors=COL_AXIS_TEXT, length=2)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#444444")
        ax.set_xlim(-0.55, n - 0.45)

        # Padding on Y so badges + labels don't clip the curve.
        ymin, ymax = float(np.min(ens_proj)), float(np.max(ens_proj))
        span = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - 0.14 * span, ymax + 0.22 * span)

        # --- Title block ----------------------------------------------------
        summary = t["summary"]
        rho_bar = summary["mean_ensemble_rho"]
        n_exact_sp = sum(
            1 for lbl, m in t["per_model"].items()
            if lbl in SIGNAL_PANEL and m["ensemble"]["modal_is_exact"]
        )
        n_near_sp = sum(
            1 for lbl, m in t["per_model"].items()
            if lbl in SIGNAL_PANEL and m["ensemble"]["modal_is_near"]
        )
        # Compact metrics line — kept short enough to fit a 1×5 layout panel.
        if t["polarity"] == "negative":
            metrics_line = (
                f"{encoder} · "
                f"$\\bar{{\\rho}}$ = {rho_bar:.2f} · "
                f"modal {modal_freq}/5"
            )
        else:
            metrics_line = (
                f"{encoder} · "
                f"$\\bar{{\\rho}}$ = {rho_bar:.2f} · "
                f"SP {n_exact_sp}/6 exact · {n_near_sp}/6 near · "
                f"modal {modal_freq}/5"
            )

        # Stack title / tag / metrics above the axes, top-down.
        # Axes-coord Y values (above the spine):
        #   y = 1.28 : bold panel title
        #   y = 1.17 : italic tag
        #   y = 1.06 : metrics line
        ax.text(
            0.0, 1.30, PANEL_TITLES[tid],
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=11.0, fontweight="bold",
            color="#1A1A1A",
        )
        ax.text(
            0.0, 1.18, PANEL_TAG[tid],
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=7.8, fontstyle="italic",
            color="#5A5A5A",
        )
        ax.text(
            0.0, 1.07, metrics_line,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=7.8, color="#2A2A2A",
        )

        # --- Corner badges --------------------------------------------------
        if coincides and t["polarity"] != "negative":
            add_badge(
                ax, "EXACT HIT",
                color_bg=COL_BADGE_BG,
                color_edge=COL_BADGE_EDGE,
                color_text="#7A4310",
                loc="upper left",
            )
        elif t["polarity"] == "negative":
            add_badge(
                ax, "NEGATIVE CONTROL",
                color_bg=COL_BADGE_NEUTRAL_BG,
                color_edge=COL_BADGE_NEUTRAL_EDGE,
                color_text="#3A3A3A",
                loc="upper left",
            )
        elif t.get("borderline"):
            add_badge(
                ax, "BORDERLINE",
                color_bg=COL_BADGE_NEUTRAL_BG,
                color_edge=COL_BADGE_NEUTRAL_EDGE,
                color_text="#3A3A3A",
                loc="upper left",
            )

        # Y-label only on the leftmost panel.
        if tid == test_ids[0]:
            ax.set_ylabel(
                "PC1 projection  (ensemble mean over 5 paraphrases)",
                fontsize=8.5, color=COL_AXIS_TEXT,
            )

    # --- Shared legend (bottom) --------------------------------------------
    legend_handles = [
        Line2D([0], [0], color=COL_CURVE, linewidth=1.8, marker="o",
               markersize=4.5, markerfacecolor=COL_CURVE,
               label="PC1 projection · ensemble mean"),
        Line2D([0], [0], color=COL_LEGAL, linewidth=1.6,
               linestyle=(0, (5, 3)),
               label="Pre-registered legal threshold"),
        Line2D([0], [0], color=COL_MODAL, linewidth=1.8,
               linestyle=(0, (1, 2.2)),
               marker="s", markersize=7, markerfacecolor="white",
               markeredgecolor=COL_MODAL, markeredgewidth=1.4,
               label="Modal max-gap break (representative encoder)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.0,
        bbox_to_anchor=(0.5, 0.02),
        handlelength=2.6,
        columnspacing=2.6,
    )

    out = OUT / "fig8_categorical_probe.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.8 (categorical probe, 5-panel scientific style)...")
    make_fig8()
    print("Done.")
