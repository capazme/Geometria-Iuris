#!/usr/bin/env python3
"""
Generate paper-style figure for §3.2.3 (per-axis Spearman agreement).

Fig.11 — Six-panel forest plot (3×2 grid), one panel per value axis.
         Within each panel the 17 cohort-coherent encoder pairs are
         stacked vertically in four groups (within-WEIRD, within-Sinic,
         cross-tradition signal panel, within-bilingual β-control), with
         the Spearman ρ and its 95% block-bootstrap CI read directly from
         the frozen JSON.  Reference line at ρ̄_cross^a.

Inputs (frozen JSON, no recomputation):
    experiment_2_axes/results_attested/experiment_2_results.json
        → section_323.per_pair

Cohort restriction (per §2.3 reporting rule, mirroring §3.1.3):
    signal panel = 6 monolingual encoders (WEIRD: BGE-EN-large, E5-large,
        FreeLaw-EN; Sinic: BGE-ZH-large, Text2vec-large-ZH, Dmeta-ZH)
        → 3 within-WEIRD pairs, 3 within-Sinic pairs, 9 cross-tradition pairs
    β-control = 2 within-bilingual pairs (BGE-M3 EN×ZH, Qwen3-0.6B EN×ZH)
    Total: 17 pairs per axis, 6 axes → 102 ρ on the figure.

Design system: high-end scientific paper (Nature / Distill / IEEE).
Output PNG @ 300 dpi:
    fig11_spearman_axes_forest.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- rcParams
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

REPO = Path(__file__).resolve().parents[4]
RES_ATT = REPO / "experiments" / "ch3-measurability" / "experiment_2_axes" / "results_attested" / "experiment_2_results.json"
OUT = Path(__file__).parent

MONO_WEIRD = ["BGE-EN-large", "E5-large", "FreeLaw-EN"]
MONO_SINIC = ["BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"]
BILI_PAIRS = [("BGE-M3-EN", "BGE-M3-ZH"),
              ("Qwen3-0.6B-EN", "Qwen3-0.6B-ZH")]

AXES_ORDER = [
    "public_private",
    "status_contract",
    "rights_duties",
    "individual_collective",
    "state_market",
    "natural_positive",
]

AXIS_LABEL = {
    "public_private":        "public / private",
    "status_contract":       "status / contract",
    "rights_duties":         "rights / duties",
    "individual_collective": "individual / collective",
    "state_market":          "state / market",
    "natural_positive":      "natural / positive",
}

GROUP_ORDER = ["within_WEIRD", "within_Sinic", "cross_tradition", "within_bilingual"]
GROUP_LABEL = {
    "within_WEIRD":     "within-WEIRD",
    "within_Sinic":     "within-Sinic",
    "cross_tradition":  "cross-tradition",
    "within_bilingual": "within-bilingual (β)",
}

# Shared four-colour palette (mirrors Fig.6 / Fig.7 family)
COLORS = {
    "within_WEIRD":     "#1F4E79",   # deep blue
    "within_Sinic":     "#A4262C",   # deep burgundy
    "cross_tradition":  "#2E7570",   # medium teal
    "within_bilingual": "#7A7A7A",   # medium grey (β-control)
}

# Outlier emphasis on individual_collective: any pair involving Dmeta-ZH
DMETA = "Dmeta-ZH"
OUTLIER_AXIS = "individual_collective"


def cohort_group(a: str, b: str) -> str | None:
    """Classify a pair under the §2.3 cohort scheme; return None if outside the 17-pair subset."""
    sa = "W" if a in MONO_WEIRD else ("S" if a in MONO_SINIC else "B")
    sb = "W" if b in MONO_WEIRD else ("S" if b in MONO_SINIC else "B")
    if sa == "W" and sb == "W":
        return "within_WEIRD"
    if sa == "S" and sb == "S":
        return "within_Sinic"
    if (sa == "W" and sb == "S") or (sa == "S" and sb == "W"):
        return "cross_tradition"
    if (a, b) in BILI_PAIRS or (b, a) in BILI_PAIRS:
        return "within_bilingual"
    return None  # mixed (monolingual × bilingual): outside the 17-pair cohort


def load_per_pair() -> list[dict]:
    with RES_ATT.open() as fh:
        d = json.load(fh)
    return d["section_323"]["per_pair"]


def build_axis_table(per_pair: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Return {axis: {group: [pair_dicts]}} for the 17 cohort-coherent pairs."""
    out: dict[str, dict[str, list[dict]]] = {ax: {g: [] for g in GROUP_ORDER} for ax in AXES_ORDER}
    for e in per_pair:
        g = cohort_group(e["model_a"], e["model_b"])
        if g is None:
            continue
        ax = e["axis"]
        out[ax][g].append(e)
    # Sort within each cluster by rho descending for predictable layout
    for ax in AXES_ORDER:
        for g in GROUP_ORDER:
            out[ax][g].sort(key=lambda p: -float(p["rho"]))
    return out


def compute_layout(cluster_sizes: list[int], cluster_gaps: float):
    """Return {group: [y positions top-down]}, total y span, cluster centres."""
    positions: dict[str, list[float]] = {}
    cursor = 0.0
    for grp, sz in zip(GROUP_ORDER, cluster_sizes):
        ys = [cursor + i for i in range(sz)]
        positions[grp] = ys
        cursor += sz + cluster_gaps
    total_height = cursor - cluster_gaps
    layout = {g: [total_height - y for y in ys] for g, ys in positions.items()}
    centres = {g: float(np.mean(layout[g])) for g in GROUP_ORDER}
    return layout, total_height, centres


def make_fig11() -> None:
    per_pair = load_per_pair()
    table = build_axis_table(per_pair)

    # Sanity check: 3 + 3 + 9 + 2 = 17 per axis
    for ax in AXES_ORDER:
        counts = {g: len(table[ax][g]) for g in GROUP_ORDER}
        assert counts == {"within_WEIRD": 3, "within_Sinic": 3,
                          "cross_tradition": 9, "within_bilingual": 2}, \
            f"axis {ax}: unexpected counts {counts}"

    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10.0, 8.0),
        sharex=True, sharey=True,
    )
    axes_flat = axes.flatten()

    cluster_sizes = [3, 3, 9, 2]
    cluster_gaps = 1.2
    layout, total_h, centres = compute_layout(cluster_sizes, cluster_gaps)

    x_min, x_max = -0.25, 0.95

    for panel_i, ax_name in enumerate(AXES_ORDER):
        ax = axes_flat[panel_i]
        row, col = divmod(panel_i, n_cols)
        is_bottom_row = (row == n_rows - 1)
        is_left_col = (col == 0)
        cluster_data = table[ax_name]

        # Compute cluster means & Δρ_a
        cross_rhos = [float(p["rho"]) for p in cluster_data["cross_tradition"]]
        within_W_rhos = [float(p["rho"]) for p in cluster_data["within_WEIRD"]]
        within_S_rhos = [float(p["rho"]) for p in cluster_data["within_Sinic"]]
        rho_cross_bar = float(np.mean(cross_rhos))
        rho_W_bar = float(np.mean(within_W_rhos))
        rho_S_bar = float(np.mean(within_S_rhos))
        delta_a = 0.5 * (rho_W_bar + rho_S_bar) - rho_cross_bar

        # ------------ Background cluster bands (very faint) ------------
        for grp in GROUP_ORDER:
            ys = layout[grp]
            if not ys:
                continue
            y_top = max(ys) + 0.5
            y_bot = min(ys) - 0.5
            ax.axhspan(
                y_bot, y_top,
                facecolor=COLORS[grp],
                alpha=0.045,
                zorder=0,
            )

        # ------------ Zero reference (thin neutral) ------------
        ax.axvline(0.0, color="#666666", linewidth=0.5,
                   linestyle=(0, (1, 2)), zorder=1)

        # ------------ Cross-tradition mean reference ------------
        ax.axvline(
            rho_cross_bar,
            color=COLORS["cross_tradition"],
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            zorder=2,
            alpha=0.85,
        )
        # Tiny label for ρ̄_cross, top of panel
        ax.text(
            rho_cross_bar, total_h + 0.6,
            f"ρ̄$_{{cross}}$ = {rho_cross_bar:+.3f}",
            fontsize=6.5,
            color=COLORS["cross_tradition"],
            ha="center", va="bottom",
            zorder=5,
        )

        # ------------ Plot each cluster ------------
        for grp in GROUP_ORDER:
            pairs = cluster_data[grp]
            ys = layout[grp]
            color = COLORS[grp]
            is_signal = grp != "within_bilingual"
            for y, p in zip(ys, pairs):
                rho = float(p["rho"])
                lo = float(p["ci_low"])
                hi = float(p["ci_high"])

                # Outlier flag: any Dmeta-ZH pair on individual_collective
                is_outlier = (
                    ax_name == OUTLIER_AXIS
                    and (p["model_a"] == DMETA or p["model_b"] == DMETA)
                )

                if is_signal:
                    marker = "o"
                    face = color
                    edge = color
                    base_ms = 5.5
                    base_ew = 0.9
                else:
                    marker = "s"          # open square for β-control
                    face = "white"
                    edge = color
                    base_ms = 5.0
                    base_ew = 0.9

                # Outlier emphasis: gold halo + larger marker
                if is_outlier:
                    # Halo: a slightly larger marker plotted behind
                    ax.plot(
                        rho, y,
                        marker=marker,
                        markerfacecolor="none",
                        markeredgecolor="#D4A017",   # warm gold
                        markersize=base_ms + 5.5,
                        markeredgewidth=1.4,
                        linestyle="none",
                        zorder=3,
                    )
                    ms = base_ms + 1.0
                    ew = base_ew + 0.2
                else:
                    ms = base_ms
                    ew = base_ew

                ax.errorbar(
                    rho, y,
                    xerr=[[rho - lo], [hi - rho]],
                    fmt=marker,
                    markerfacecolor=face,
                    markeredgecolor=edge,
                    markersize=ms,
                    markeredgewidth=0.9,
                    ecolor=color,
                    elinewidth=ew,
                    capsize=2.2,
                    capthick=ew,
                    alpha=0.95,
                    zorder=4,
                )

        # ------------ Axes cosmetics ------------
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.2, total_h + 1.6)

        # Y ticks: one per cluster centre (Tufte-style, only on left column)
        if is_left_col:
            yticks = [centres[g] for g in GROUP_ORDER]
            yticklabels = [GROUP_LABEL[g] for g in GROUP_ORDER]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=8)
            for tick, g in zip(ax.get_yticklabels(), GROUP_ORDER):
                tick.set_color(COLORS[g])
                if g == "within_bilingual":
                    tick.set_style("italic")
        else:
            ax.set_yticks([])
            ax.tick_params(axis="y", which="both", left=False)
            ax.spines["left"].set_visible(False)

        # X ticks: only bottom row gets labels
        if is_bottom_row:
            ax.tick_params(axis="x", labelbottom=True, labelsize=8)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        ax.spines["bottom"].set_color("#333333")
        if is_left_col:
            ax.spines["left"].set_color("#333333")

        # Panel title: axis name + Δρ_a (Δρ_a in coloured emphasis)
        title_main = AXIS_LABEL[ax_name]
        ax.set_title(
            f"{title_main}",
            fontsize=11, fontweight="bold",
            loc="left", pad=10,
            color="#1A1A1A",
        )
        # Subtitle line: Δρ_a, right-aligned in top-right of axes area
        ax.text(
            0.99, 1.015,
            f"Δρ$_a$ = {delta_a:+.3f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#444444",
            ha="right", va="bottom",
            fontstyle="italic",
        )

    # ------------ X label on bottom row ------------
    for ax in axes[-1, :]:
        ax.set_xlabel(
            "Spearman ρ  (rank agreement on 364 projections, attested)",
            fontsize=9, labelpad=8, color="#222222",
        )

    # ------------ Suptitle ------------
    fig.suptitle(
        "Per-axis rank agreement across 17 cohort-coherent encoder pairs",
        fontsize=12.5, fontweight="bold", y=0.995, color="#0F0F0F",
    )
    # Subtitle line (below suptitle)
    fig.text(
        0.5, 0.962,
        "Spearman ρ with 95% block-bootstrap CI (B = 10,000); reference line at ρ̄$_{cross}$ per axis.",
        ha="center", va="top",
        fontsize=8.5, color="#555555", fontstyle="italic",
    )

    # ------------ Legend (custom, anchored under suptitle, single row) ------------
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color=COLORS["within_WEIRD"],
               markerfacecolor=COLORS["within_WEIRD"],
               markeredgecolor=COLORS["within_WEIRD"],
               markersize=6, linestyle="none",
               label="within-WEIRD (3)"),
        Line2D([0], [0], marker="o", color=COLORS["within_Sinic"],
               markerfacecolor=COLORS["within_Sinic"],
               markeredgecolor=COLORS["within_Sinic"],
               markersize=6, linestyle="none",
               label="within-Sinic (3)"),
        Line2D([0], [0], marker="o", color=COLORS["cross_tradition"],
               markerfacecolor=COLORS["cross_tradition"],
               markeredgecolor=COLORS["cross_tradition"],
               markersize=6, linestyle="none",
               label="cross-tradition (9)"),
        Line2D([0], [0], marker="s", color=COLORS["within_bilingual"],
               markerfacecolor="white",
               markeredgecolor=COLORS["within_bilingual"],
               markersize=6, linestyle="none",
               label="within-bilingual β (2)"),
        Line2D([0], [0], marker="o", color="#D4A017",
               markerfacecolor="none",
               markeredgecolor="#D4A017",
               markersize=9, markeredgewidth=1.4,
               linestyle="none",
               label="Dmeta-ZH outlier (ind./coll.)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.subplots_adjust(
        left=0.13, right=0.97,
        top=0.92, bottom=0.085,
        hspace=0.45, wspace=0.10,
    )

    out = OUT / "fig11_spearman_axes_forest.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.11 (per-axis Spearman forest, 6 panels × 17 pairs)...")
    make_fig11()
    print("\nDone.")
