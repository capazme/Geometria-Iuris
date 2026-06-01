#!/usr/bin/env python3
"""
Generate paper-style B/W figures for §3.1.3 (RSA cross-tradition).

Fig.6 — Forest plot of the 17 RSA Spearman ρ across the 4 groups
        (within-WEIRD, within-Sinic, cross-tradition, within-bilingual
        β-control), with 95% block-bootstrap CI from the frozen JSON
        and reference lines on the three group means.
Fig.7 — Per-pair slope chart: bare → attested ρ for each of the 17
        pairs, with within-tradition pairs drawn heavy/opaque and
        cross-tradition pairs thin/semi-transparent.

Inputs (frozen JSON, no recomputation):
    experiment_1_structure/results_attested/experiment_1_results.json → section_313
    experiment_1_structure/results_bare/experiment_1_results.json     → section_313

Outputs PNG @ 300 dpi in this directory:
    fig6_rsa_forest.png
    fig7_bare_attested_slope.png
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
# Shared design system (Nature / Distill.pub / IEEE register)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
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
RES_ATT = REPO / "experiments" / "ch3-measurability" / "experiment_1_structure" / "results_attested" / "experiment_1_results.json"
RES_BAR = REPO / "experiments" / "ch3-measurability" / "experiment_1_structure" / "results_bare" / "experiment_1_results.json"
OUT = Path(__file__).parent

# Group order top → bottom in the forest plot
GROUP_ORDER = ["within_weird", "within_sinic", "cross_tradition", "within_bilingual"]
GROUP_LABEL = {
    "within_weird":     "within-WEIRD",
    "within_sinic":     "within-Sinic",
    "cross_tradition":  "cross-tradition",
    "within_bilingual": "within-bilingual (β-control)",
}

# Saturated four-colour palette for the four groups
# (Nature/Distill-style: deep, distinguishable, print-safe)
COLORS = {
    "within_weird":     "#1F4E79",   # deep blue
    "within_sinic":     "#A4262C",   # deep burgundy
    "cross_tradition":  "#2E7570",   # medium teal
    "within_bilingual": "#7A7A7A",   # muted grey (β-control)
}

# Legacy alias kept for Fig.7, which still uses the greyscale register.
GREY = {
    "within_weird":     "0.45",
    "within_sinic":     "0.00",
    "cross_tradition":  "0.60",
    "within_bilingual": "0.25",
}

DISP_MODEL = {
    "BGE-EN-large":           "BGE-EN-large",
    "E5-large":               "E5-large",
    "FreeLaw-EN":             "FreeLaw-EN",
    "BGE-ZH-large":           "BGE-ZH-large",
    "Text2vec-large-ZH":      "Text2vec-ZH",
    "Dmeta-ZH":               "Dmeta-ZH",
    "BGE-M3-EN":              "BGE-M3 (EN)",
    "BGE-M3-ZH":              "BGE-M3 (ZH)",
    "Qwen3-0.6B-EN":          "Qwen3-0.6B (EN)",
    "Qwen3-0.6B-ZH":          "Qwen3-0.6B (ZH)",
}


def load_section_313(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)["section_313"]


def pair_label(p: dict) -> str:
    return f"{DISP_MODEL.get(p['model_a'], p['model_a'])} × {DISP_MODEL.get(p['model_b'], p['model_b'])}"


def all_pairs(s313: dict) -> list[dict]:
    """Return the 17 pair dicts as a flat list, in canonical group order."""
    rows = []
    for g in GROUP_ORDER:
        rows.extend(s313[g])
    return rows


# ============================================================================
# Fig.6 — Forest plot of the 17 attested ρ with 95% CI, grouped
# ============================================================================
def make_fig6() -> None:
    s313 = load_section_313(RES_ATT)
    summary = s313["summary"]
    pairs = all_pairs(s313)
    n = len(pairs)
    assert n == 17, f"expected 17 pairs, got {n}"

    # Bare baseline (Δρ_sym on bare embeddings of the 364-term core)
    # for the Caveat Y framing in the title: legal signal = attested − bare.
    delta_bare = float(load_section_313(RES_BAR)["summary"]["delta_rho_symmetric"])
    delta_att = float(summary["delta_rho_symmetric"])

    # y positions top → bottom (matplotlib y increases upward)
    y_positions = list(range(n, 0, -1))

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # ------------------------------------------------------------------
    # Background banding for the four groups (alternating very-light
    # grey stripes for visual separation; drawn first → zorder=0).
    # ------------------------------------------------------------------
    band_specs = [
        (0,  3,  0.06),   # within-WEIRD
        (3,  6,  0.0),    # within-Sinic
        (6,  15, 0.06),   # cross-tradition
        (15, 17, 0.0),    # β-control
    ]
    for i0, i1, alpha in band_specs:
        if alpha <= 0:
            continue
        y_top = y_positions[i0] + 0.5
        y_bot = y_positions[i1 - 1] - 0.5
        ax.axhspan(y_bot, y_top, facecolor="0.20", alpha=alpha,
                   edgecolor="none", zorder=0)

    # ------------------------------------------------------------------
    # Group reference lines (vertical dashed, colour-matched to groups)
    # — drawn BEFORE markers so points sit visually on top.
    # ------------------------------------------------------------------
    w_mean = float(summary["mean_rho_within_weird"])
    s_mean = float(summary["mean_rho_within_sinic"])
    cross_mean = float(summary["mean_rho_cross_tradition"])
    bili_mean = float(summary["mean_rho_within_bilingual"])
    ref_lines = [
        (w_mean,     COLORS["within_weird"],     (0, (5, 2.5))),
        (s_mean,     COLORS["within_sinic"],     (0, (5, 2.5))),
        (cross_mean, COLORS["cross_tradition"],  (0, (4, 2))),
        (bili_mean,  COLORS["within_bilingual"], (0, (2, 2))),
    ]
    for x, color, dash in ref_lines:
        ax.axvline(x, color=color, linewidth=1.0, linestyle=dash,
                   alpha=0.85, zorder=1)

    # ------------------------------------------------------------------
    # Soft separators between the four groups (subtle hairlines)
    # ------------------------------------------------------------------
    boundaries = [3, 6, 15]  # number of pairs above the separator
    for n_above in boundaries:
        y_sep = y_positions[n_above - 1] - 0.5
        ax.axhline(y_sep, color="0.70", linewidth=0.7, linestyle="-",
                   zorder=1)

    # ------------------------------------------------------------------
    # Error bars + per-pair ρ labels (colour-matched to group)
    # ------------------------------------------------------------------
    for i, p in enumerate(pairs):
        y = y_positions[i]
        g = p["group"]
        rho = float(p["rho"])
        lo = float(p["ci_low"])
        hi = float(p["ci_high"])
        color = COLORS[g]
        is_signal = g != "within_bilingual"
        # Signal panel: saturated filled circles. β-control: filled but
        # in the muted grey to convey "control".
        ms = 7.5 if is_signal else 7.0
        ew = 1.4 if is_signal else 1.1
        ax.errorbar(
            rho, y,
            xerr=[[rho - lo], [hi - rho]],
            fmt="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=ms,
            ecolor=color,
            elinewidth=ew,
            capsize=3.0,
            capthick=ew,
            alpha=0.95 if is_signal else 0.85,
            zorder=3,
        )
        # Per-pair ρ value, right-aligned in a fixed column at x = 1.005
        ax.text(1.005, y, f"{rho:.3f}",
                fontsize=8, va="center", ha="left",
                color="0.15", fontweight="medium")

    # ------------------------------------------------------------------
    # y-axis labels (pair names)
    # ------------------------------------------------------------------
    ax.set_yticks(y_positions)
    ax.set_yticklabels([pair_label(p) for p in pairs], fontsize=8.5)
    for tick, p in zip(ax.get_yticklabels(), pairs):
        if p["group"] == "within_bilingual":
            tick.set_style("italic")
            tick.set_color("0.40")
        else:
            tick.set_color("0.10")
    ax.tick_params(axis="y", length=0, pad=4)

    # ------------------------------------------------------------------
    # x axis
    # ------------------------------------------------------------------
    ax.set_xlim(0.05, 1.10)
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.set_xlabel("Spearman ρ   (RSA on 364-term post-BLP pool, attested)",
                  fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9)

    # ------------------------------------------------------------------
    # Group means — labels stacked above the plot area, colour-matched,
    # bold numerals.
    # Vertical staggering avoids any collision regardless of x.
    # Order (left→right by x): cross (0.246) · bili (0.316) ·
    #                          W (0.712) · S (0.868)
    # ------------------------------------------------------------------
    y_top_row = n + 1.7
    y_bot_row = n + 0.55
    mean_labels = [
        (cross_mean, r"$\bar{\rho}_{\mathrm{cross}}$", f"{cross_mean:.3f}",
         COLORS["cross_tradition"], y_bot_row),
        (bili_mean,  r"$\bar{\rho}_{\mathrm{bili}}$",  f"{bili_mean:.3f}",
         COLORS["within_bilingual"], y_top_row),
        (w_mean,     r"$\bar{\rho}_{W}$",              f"{w_mean:.3f}",
         COLORS["within_weird"], y_bot_row),
        (s_mean,     r"$\bar{\rho}_{S}$",              f"{s_mean:.3f}",
         COLORS["within_sinic"], y_top_row),
    ]
    for x, sym, val, color, y in mean_labels:
        ax.text(x, y, f"{sym} = ", ha="right", va="center",
                fontsize=9, color=color)
        ax.text(x, y, f" {val}", ha="left", va="center",
                fontsize=9.5, color=color, fontweight="bold")

    # ------------------------------------------------------------------
    # Right-side group bracket labels — drawn as vertical brackets
    # outside the data area to convey "this band is group X".
    # Placed in axis fraction coordinates (just outside the right spine).
    # ------------------------------------------------------------------
    bracket_x = 1.13   # in y-axis-transform x coords (data y, axes x)
    label_x = 1.155
    band_rows = {
        "within_weird":     (0, 3),
        "within_sinic":     (3, 6),
        "cross_tradition":  (6, 15),
        "within_bilingual": (15, 17),
    }
    for g, (i0, i1) in band_rows.items():
        y_top = y_positions[i0] + 0.40
        y_bot = y_positions[i1 - 1] - 0.40
        y_mid = (y_top + y_bot) / 2
        color = COLORS[g]
        # vertical bracket as three short line segments
        ax.plot([bracket_x, bracket_x], [y_bot, y_top],
                transform=ax.get_yaxis_transform(),
                color=color, linewidth=1.4, clip_on=False, zorder=4)
        for y_end in (y_top, y_bot):
            ax.plot([bracket_x, bracket_x + 0.012], [y_end, y_end],
                    transform=ax.get_yaxis_transform(),
                    color=color, linewidth=1.4, clip_on=False, zorder=4)
        is_bili = g == "within_bilingual"
        ax.text(label_x, y_mid, GROUP_LABEL[g],
                transform=ax.get_yaxis_transform(),
                fontsize=9.5, ha="left", va="center",
                color=color, clip_on=False,
                fontweight="bold" if not is_bili else "normal",
                style="italic" if is_bili else "normal")

    # ------------------------------------------------------------------
    # Title — VERY prominent headline. Two lines:
    # (1) editorial title; (2) the two numbers that carry the chapter.
    # ------------------------------------------------------------------
    fig.text(
        0.5, 0.995,
        "Within-tradition agreement separates cleanly from cross-tradition disagreement",
        ha="center", va="bottom",
        fontsize=11, color="0.25", style="italic",
    )
    fig.text(
        0.5, 0.955,
        (
            r"$\Delta\bar{\rho}_{\mathrm{sym}}^{\,\mathrm{attested}}\,=\,$"
            f"{delta_att:.3f}"
            r"$\quad\cdot\quad$"
            r"legal contribution (attested$\,-\,$bare)$\,=\,$"
            f"{delta_att - delta_bare:.3f}"
        ),
        ha="center", va="bottom",
        fontsize=14.5, color="0.05", fontweight="bold",
    )

    ax.set_ylim(0.3, n + 2.6)

    # ------------------------------------------------------------------
    # Holm note below the x-axis (small italic, muted)
    # ------------------------------------------------------------------
    ax.text(
        0.5, -0.14,
        r"All 17 Mantel $p$ at permutation floor ($B = 10^{4}$);  "
        r"Holm-adjusted $p_{\mathrm{max}} = 0.0017$.  "
        r"95% intervals: block bootstrap on the 364-term axis ($B = 10^{4}$).",
        transform=ax.transAxes, ha="center",
        fontsize=8.5, style="italic", color="0.40",
    )

    # Generous padding — give the headline + brackets room to breathe.
    fig.subplots_adjust(left=0.22, right=0.78, top=0.90, bottom=0.10)

    out = OUT / "fig6_rsa_forest.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"→ wrote {out.name}")


# ============================================================================
# Fig.7 — Bare → attested slope for each of the 17 pairs
# ============================================================================
def make_fig7() -> None:
    att = load_section_313(RES_ATT)
    bar = load_section_313(RES_BAR)

    pairs_att = all_pairs(att)
    pairs_bar = all_pairs(bar)
    # Build dict keyed by (model_a, model_b) to align reliably
    def key(p): return (p["model_a"], p["model_b"])
    bar_by_key = {key(p): p for p in pairs_bar}
    aligned = []
    for p in pairs_att:
        k = key(p)
        bp = bar_by_key.get(k)
        if bp is None:
            raise SystemExit(f"missing bare pair for {k}")
        aligned.append((p["group"], p, bp))
    assert len(aligned) == 17

    # ------------------------------------------------------------------
    # Local design system (Nature / Distill register), scoped via
    # rc_context so it does not leak into other figures.
    # ------------------------------------------------------------------
    import matplotlib as mpl
    rc = {
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica",
                              "Arial", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    }

    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8.0, 6.0))

        # X positions for the two endpoints
        x_bare = 0.0
        x_att = 1.0

        # ---- per-pair slope segments ---------------------------------
        # Cross-tradition drawn first so within-tradition overlays it.
        # Bilingual β-control uses a dashed line + lower alpha to
        # mark its causal-control status visually.
        plot_order = [
            # group_key,        lw,  alpha, marker_size, linestyle
            ("cross_tradition",  1.5, 0.85, 5.5, "-"),
            ("within_bilingual", 1.0, 0.70, 5.0, (0, (3, 2))),
            ("within_weird",     1.5, 0.85, 5.5, "-"),
            ("within_sinic",     1.5, 0.85, 5.5, "-"),
        ]
        for plot_g, lw, alpha, ms, ls in plot_order:
            color = COLORS[plot_g]
            for g, ap, bp in aligned:
                if g != plot_g:
                    continue
                ax.plot(
                    [x_bare, x_att],
                    [float(bp["rho"]), float(ap["rho"])],
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
                    alpha=alpha,
                    marker="o",
                    markersize=ms,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.0,
                    solid_capstyle="round",
                    zorder=2,
                )

        # ---- group-mean overlays (thick, opaque) ---------------------
        s313_att = att["summary"]
        s313_bar = bar["summary"]
        mean_fields = {
            "within_weird":     "mean_rho_within_weird",
            "within_sinic":     "mean_rho_within_sinic",
            "cross_tradition":  "mean_rho_cross_tradition",
            "within_bilingual": "mean_rho_within_bilingual",
        }
        group_means_bare = {g: float(s313_bar[mean_fields[g]])
                            for g in mean_fields}
        group_means_att = {g: float(s313_att[mean_fields[g]])
                           for g in mean_fields}

        for g, field in mean_fields.items():
            mb = group_means_bare[g]
            ma = group_means_att[g]
            color = COLORS[g]
            ls = "-" if g != "within_bilingual" else (0, (3, 2))
            ax.plot(
                [x_bare, x_att], [mb, ma],
                color=color,
                linewidth=3.6,
                linestyle=ls,
                alpha=1.0,
                marker="o",
                markersize=9.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1.3,
                solid_capstyle="round",
                zorder=4,
            )

        # ---- right-side group labels with leader lines ---------------
        # Labels sit just to the right of attested column, leadered
        # back to the attested-mean endpoint for unambiguous mapping.
        leader_x0 = x_att + 0.025
        leader_x1 = x_att + 0.16
        label_x = x_att + 0.18
        for g in ["within_sinic", "within_weird",
                  "within_bilingual", "cross_tradition"]:
            y = group_means_att[g]
            color = COLORS[g]
            pretty = {
                "within_sinic":     "within-Sinic",
                "within_weird":     "within-WEIRD",
                "within_bilingual": "within-bilingual β-control",
                "cross_tradition":  "cross-tradition",
            }[g]
            label = (
                f"{pretty}\n"
                f"$\\bar{{\\rho}}$ = {group_means_att[g]:.3f}"
            )
            ax.plot([leader_x0, leader_x1], [y, y],
                    color=color, linewidth=0.8, alpha=0.55, zorder=1)
            ax.text(label_x, y, label,
                    fontsize=9.5, va="center", ha="left",
                    color=color,
                    fontweight="semibold",
                    linespacing=1.3)

        # ---- bare-side baseline labels (Caveat Y at-a-glance) --------
        for g in ["within_sinic", "within_weird",
                  "within_bilingual", "cross_tradition"]:
            mb = group_means_bare[g]
            ax.text(x_bare - 0.025, mb, f"{mb:.3f}",
                    fontsize=8.0, va="center", ha="right",
                    color=COLORS[g], alpha=0.85)

        # ---- Caveat Y bracket on the far right ------------------------
        # Outer span: Δρ_sym^att = 0.543, from cross mean up to
        #             arithmetic mean of (WEIRD-att, Sinic-att).
        # Inner split (lower, cross teal): bare baseline = 0.165, the
        #             encoder-tradition share already present on the
        #             everyday control lexicon. Upper portion = the
        #             legal contribution proper, 0.378.
        y_within_mean_att = 0.5 * (group_means_att["within_weird"]
                                   + group_means_att["within_sinic"])
        y_cross_att = group_means_att["cross_tradition"]
        y_baseline_top = y_cross_att + 0.165

        x_brk = x_att + 0.82  # bracket x (well to the right of labels)
        tick_w = 0.04

        # outer bracket stem
        ax.plot([x_brk, x_brk], [y_cross_att, y_within_mean_att],
                color="#1A1A1A", lw=1.1, zorder=5)
        # outer terminal ticks
        for y_tick in (y_within_mean_att, y_cross_att):
            ax.plot([x_brk - tick_w, x_brk + tick_w], [y_tick, y_tick],
                    color="#1A1A1A", lw=1.1, zorder=5)
        # inner stem highlighting the bare-baseline portion
        ax.plot([x_brk, x_brk], [y_cross_att, y_baseline_top],
                color=COLORS["cross_tradition"], lw=3.2, alpha=0.55,
                solid_capstyle="butt", zorder=4)
        # split tick where legal signal starts
        ax.plot([x_brk - tick_w, x_brk + tick_w],
                [y_baseline_top, y_baseline_top],
                color=COLORS["cross_tradition"], lw=1.0,
                alpha=0.75, zorder=5)

        # outer label: legal contribution
        ax.text(
            x_brk + 0.10, (y_within_mean_att + y_baseline_top) / 2,
            "legal signal\n$\\mathbf{= 0.378}$",
            fontsize=9.5, va="center", ha="left",
            color="#1A1A1A", fontweight="semibold",
            linespacing=1.3,
        )
        # inner label: encoder-tradition baseline
        ax.text(
            x_brk + 0.10, (y_baseline_top + y_cross_att) / 2,
            "bare baseline\n$= 0.165$",
            fontsize=8.5, va="center", ha="left",
            color=COLORS["cross_tradition"], alpha=0.95,
            linespacing=1.3,
        )
        # top-of-bracket header: total attested gap
        ax.text(
            x_brk, y_within_mean_att + 0.028,
            "$\\Delta\\bar{\\rho}^{\\,\\mathrm{att}}_{\\mathrm{sym}}$ = 0.543",
            fontsize=10, va="bottom", ha="center",
            color="#1A1A1A", fontweight="bold",
        )

        # ---- axes setup ----------------------------------------------
        ax.set_xticks([x_bare, x_att])
        ax.set_xticklabels(["bare", "attested"], fontsize=11,
                           fontweight="semibold")
        ax.set_xlim(-0.18, x_att + 1.35)
        ax.set_ylim(0.10, 0.95)
        ax.set_ylabel("Spearman $\\rho$", fontsize=11)
        ax.tick_params(axis="y", which="both", length=3, width=0.8)
        ax.tick_params(axis="x", which="both", length=4, width=0.8)

        # subtle horizontal gridlines every 0.1
        ax.set_yticks(np.arange(0.1, 1.0, 0.1))
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6,
                      alpha=0.30, color="#444444")
        ax.set_axisbelow(True)

        # very faint column guides
        for xv in (x_bare, x_att):
            ax.axvline(xv, color="#444444", linewidth=0.5,
                       alpha=0.18, zorder=0)

        # ---- title + subtitle ----------------------------------------
        ax.set_title(
            "Cross-tradition agreement, bare vs attested",
            fontsize=12, pad=26, loc="left", fontweight="bold",
            color="#1A1A1A",
        )
        ax.text(
            0.0, 1.020,
            "Within-tradition pairs amplify under legal attestation; "
            "cross-tradition pairs stagnate; β-control descends slightly.",
            transform=ax.transAxes,
            fontsize=10, color="#444444", ha="left", va="bottom",
            style="italic",
        )

        fig.tight_layout()
        out = OUT / "fig7_bare_attested_slope.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"→ wrote {out.name}")


if __name__ == "__main__":
    print("Generating Fig.6 (RSA forest plot, 17 pairs)...")
    make_fig6()
    print("\nGenerating Fig.7 (bare → attested slope, 17 pairs)...")
    make_fig7()
    print("\nDone.")
