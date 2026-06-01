#!/usr/bin/env python3
"""
Generate paper-style B/W figures for §3.1.1 (within/between domain).

Fig.1 — Histograms of within-domain vs between-domain cosine distances for
        BGE-EN-large attested, with medians marked.
Fig.2 — Forest plot of rank-biserial r for the 10 readings (6 signal panel
        + 4 bilingual control) × 2 regimes (attested, bare), with 95% CI
        from non-parametric bootstrap (B=1000 over distance pairs).

Outputs PNG @ 300 dpi in this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patheffects  # registers mpl.patheffects.withStroke
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde, mannwhitneyu

# ---------------------------------------------------------------------------
# Shared design system (Geometria Iuris Ch.3 figures, 2026-05-23 redesign).
# Target aesthetic: Nature / Distill.pub static / IEEE Transactions.
# ---------------------------------------------------------------------------
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
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Qualitative categorical palette (colorblind-safe, Tableau-10 / Set2 hybrid).
PALETTE = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C"]
# Semantic assignments for §3.1.1 within/between.
COLOR_WITHIN = "#2F5C8F"   # deep blue: signal (within-domain)
COLOR_BETWEEN = "#DD8452"  # warm orange: contrast (between-domain)

# Fig.2 forest plot — attested vs bare regimes.
# Signal panel uses full-saturation colours; bilingual controls use muted /
# desaturated counterparts so the secondary readings recede visually but
# remain readable on the same chart. Marker shape carries the regime
# (attested vs bare); colour carries panel membership.
COLOR_ATTESTED_SIGNAL = "#1F4E79"   # deep blue (attested, signal panel)
COLOR_BARE_SIGNAL = "#D67A1A"       # warm orange (bare, signal panel)
COLOR_ATTESTED_CONTROL = "#7C99B8"  # muted blue (attested, bilingual control)
COLOR_BARE_CONTROL = "#D9B98A"      # muted orange (bare, bilingual control)
COLOR_AXIS_GUIDE = "#8C8C8C"        # grey for zero line and EN/ZH separator

REPO = Path(__file__).resolve().parents[4]
EMB = REPO / "experiments" / "ch3-measurability" / "embeddings"
RES_ATT = REPO / "experiments" / "ch3-measurability" / "experiment_1_structure" / "results_attested" / "experiment_1_results.json"
RES_BAR = REPO / "experiments" / "ch3-measurability" / "experiment_1_structure" / "results_bare" / "experiment_1_results.json"
OUT = Path(__file__).parent

MONO = ["BGE-EN-large", "E5-large", "FreeLaw-EN",
        "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"]
BILI = ["BGE-M3-EN", "BGE-M3-ZH", "Qwen3-0.6B-EN", "Qwen3-0.6B-ZH"]

# Canonical display order (mirrors experiments/dashboard_final/data/loader_31.py
# ALL_MODELS_ORDERED): top half = EN-side encoders, bottom half = ZH-side
# encoders; within each side, monolingual signal-panel first, bilingual control
# next. The English/Chinese split is the primary visual axis.
ALL_MODELS_ORDERED = [
    "BGE-EN-large", "E5-large", "FreeLaw-EN",
    "BGE-M3-EN", "Qwen3-0.6B-EN",
    "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH",
    "BGE-M3-ZH", "Qwen3-0.6B-ZH",
]
SIGNAL_PANEL = set(MONO)

# Display labels (compact)
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


def load_index() -> list[str]:
    with (EMB / "index.json").open() as fh:
        idx = json.load(fh)
    return [t["domain"] for t in idx]


def load_vecs(label: str, variant: str) -> np.ndarray:
    fname = "vecs_bare.npy" if variant == "bare" else "vecs_attested.npy"
    return np.load(EMB / label / fname).astype(np.float64)


def compute_within_between(label: str, variant: str, domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    v = load_vecs(label, variant)
    # L2-normalize defensively
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    v = v / n
    sim = v @ v.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dist = 1.0 - sim
    N = len(v)
    iu, ju = np.triu_indices(N, k=1)
    dom = np.array(domains)
    same = dom[iu] == dom[ju]
    tri = dist[iu, ju]
    return tri[same], tri[~same]


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    U = mannwhitneyu(x, y, alternative="less").statistic
    return 1.0 - 2.0 * U / (len(x) * len(y))


def bootstrap_r_ci(x: np.ndarray, y: np.ndarray, B: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rs = np.empty(B)
    nx, ny = len(x), len(y)
    for b in range(B):
        xb = rng.choice(x, size=nx, replace=True)
        yb = rng.choice(y, size=ny, replace=True)
        rs[b] = rank_biserial(xb, yb)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


# ============================================================================
# Fig.1 — Histograms + KDE overlay, within vs between for BGE-EN-large attested
# ============================================================================
def make_fig1(domains: list[str]) -> None:
    label = "BGE-EN-large"
    within, between = compute_within_between(label, "attested", domains)
    med_w = float(np.median(within))
    med_b = float(np.median(between))
    delta = med_b - med_w  # ~0.026
    r_eff = rank_biserial(within, between)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    # Shared bins for clean visual alignment.
    lo = min(within.min(), between.min())
    hi = max(within.max(), between.max())
    bins = np.linspace(lo, hi, 55)

    # Stepfilled histograms (alpha 0.55) + crisp full-saturation edges.
    ax.hist(
        within, bins=bins, density=True, histtype="stepfilled",
        facecolor=COLOR_WITHIN, edgecolor=COLOR_WITHIN, alpha=0.55,
        linewidth=0.9,
        label=f"within-domain   (n = {len(within):,})",
        zorder=2,
    )
    ax.hist(
        between, bins=bins, density=True, histtype="stepfilled",
        facecolor=COLOR_BETWEEN, edgecolor=COLOR_BETWEEN, alpha=0.45,
        linewidth=0.9,
        label=f"between-domain   (n = {len(between):,})",
        zorder=1,
    )

    # KDE smoothed overlays on a dense grid.
    grid = np.linspace(lo, hi, 512)
    kde_w = gaussian_kde(within, bw_method="scott")(grid)
    kde_b = gaussian_kde(between, bw_method="scott")(grid)
    ax.plot(grid, kde_w, color=COLOR_WITHIN, linewidth=1.8, zorder=4)
    ax.plot(grid, kde_b, color=COLOR_BETWEEN, linewidth=1.8, zorder=3)

    # Compute a clean upper y-bound now (before adding annotations).
    # Headroom hosts the Δ arrow + label without crowding KDE peaks.
    ymax_data = float(max(kde_w.max(), kde_b.max(),
                          ax.get_ylim()[1])) * 1.30
    ax.set_ylim(0, ymax_data)

    # Median verticals (sit on top of all distributional layers).
    ax.axvline(med_w, color=COLOR_WITHIN, linestyle="--",
               linewidth=1.1, zorder=5)
    ax.axvline(med_b, color=COLOR_BETWEEN, linestyle="--",
               linewidth=1.1, zorder=5)

    # Median labels: low on the curve (around mid-flank), with a thick
    # white halo so they remain crisp over the filled KDE area.
    halo = [mpl.patheffects.withStroke(linewidth=3.2, foreground="white")]
    ax.text(
        med_w - 0.006, ymax_data * 0.32,
        f"median = {med_w:.3f}",
        color=COLOR_WITHIN, fontsize=9, fontweight="bold",
        ha="right", va="center",
        path_effects=halo, zorder=7,
    )
    ax.text(
        med_b + 0.006, ymax_data * 0.32,
        f"median = {med_b:.3f}",
        color=COLOR_BETWEEN, fontsize=9, fontweight="bold",
        ha="left", va="center",
        path_effects=halo, zorder=7,
    )

    # Directional Δ arrow between the two medians, placed up high above
    # the KDE peaks so it never collides with the median labels.
    arrow_y = ymax_data * 0.92
    ax.annotate(
        "", xy=(med_b, arrow_y), xytext=(med_w, arrow_y),
        arrowprops=dict(arrowstyle="-|>", color="#444444",
                        lw=1.0, mutation_scale=11),
        zorder=6,
    )
    ax.text(
        (med_w + med_b) / 2.0, arrow_y + ymax_data * 0.015,
        rf"$\Delta = {delta:.3f}$",
        color="#333333", fontsize=9, fontweight="bold",
        ha="center", va="bottom",
        path_effects=halo,
    )

    # Axis labels and titles (title + subtitle pattern).
    ax.set_xlabel("Cosine distance  (1 − cos)")
    ax.set_ylabel("Density")
    fig.suptitle(
        "Within-domain vs between-domain cosine distances  —  "
        "BGE-EN-large (attested)",
        fontsize=12, fontweight="bold", x=0.5, y=1.00,
        ha="center",
    )
    ax.set_title(
        f"rank-biserial $r = +{r_eff:.3f}$   ·   "
        rf"$p \ll 10^{{-300}}$   ·   "
        f"$n_{{within}}={len(within):,}$,  $n_{{between}}={len(between):,}$",
        fontsize=9.5, fontweight="normal", color="#444444", pad=8,
    )

    # Legend: upper-right (KDE peaks sit center-left, so this is clean).
    leg = ax.legend(loc="upper right", frameon=False, handlelength=1.6,
                    handletextpad=0.6, borderaxespad=0.4)
    for txt in leg.get_texts():
        txt.set_color("#222222")

    # Cosmetic axis polish.
    ax.tick_params(colors="#333333")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
    ax.set_xlim(lo, min(hi, 0.55))  # trim long right tail above 0.55

    fig.tight_layout(pad=1.5)
    out = OUT / "fig1_within_between_bge_en.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ wrote {out.name}")


# ============================================================================
# Fig.2 — Forest plot of r for 10 readings × 2 variants with 95% CI
# ============================================================================
def make_fig2(domains: list[str]) -> None:
    labels = ALL_MODELS_ORDERED  # EN-side first, then ZH-side; mirrors dashboard
    data = {}  # label -> {variant -> (r, ci_low, ci_high)}
    for lab in labels:
        data[lab] = {}
        for variant in ("attested", "bare"):
            w, b = compute_within_between(lab, variant, domains)
            r = rank_biserial(w, b)
            ci_lo, ci_hi = bootstrap_r_ci(w, b, B=500)
            data[lab][variant] = (r, ci_lo, ci_hi)
            print(f"  {lab:22s} {variant:8s}  r = {r:+.4f}  CI95 = [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # y-axis: top = first label, bottom = last.
    y_positions = np.arange(len(labels))[::-1]
    offset = 0.20  # vertical spacing between attested (above) and bare (below)

    # Zero reference line — drawn first so all markers sit on top.
    ax.axvline(
        0.0, color=COLOR_AXIS_GUIDE, linewidth=0.7, linestyle=(0, (1, 2)),
        zorder=1,
    )

    # Faint horizontal guides at each encoder row, to anchor the eye.
    for y in y_positions:
        ax.axhline(y, color="#ECECEC", linewidth=0.6, zorder=0)

    for i, lab in enumerate(labels):
        y = y_positions[i]
        is_signal = lab in SIGNAL_PANEL

        # Colour selection: full-saturation for signal panel, muted for
        # bilingual controls. Marker shape carries the regime (attested vs
        # bare); colour carries panel membership.
        att_color = COLOR_ATTESTED_SIGNAL if is_signal else COLOR_ATTESTED_CONTROL
        bare_color = COLOR_BARE_SIGNAL if is_signal else COLOR_BARE_CONTROL

        ms = 6.5 if is_signal else 5.5
        mew = 1.2 if is_signal else 1.0
        elw = 1.1 if is_signal else 0.9
        cap = 2.5
        marker_zorder = 4 if is_signal else 3

        # Attested: filled circle (above row centre).
        r_a, lo_a, hi_a = data[lab]["attested"]
        ax.errorbar(
            r_a, y + offset,
            xerr=[[r_a - lo_a], [hi_a - r_a]],
            fmt="o",
            markerfacecolor=att_color,
            markeredgecolor=att_color,
            markeredgewidth=0.5,
            markersize=ms,
            ecolor=att_color,
            elinewidth=elw,
            capsize=cap, capthick=elw,
            alpha=1.0 if is_signal else 0.92,
            zorder=marker_zorder,
        )

        # Bare: open square (below row centre), coloured edge matching the
        # bare-regime palette (full-sat for signal, muted for control).
        r_b, lo_b, hi_b = data[lab]["bare"]
        ax.errorbar(
            r_b, y - offset,
            xerr=[[r_b - lo_b], [hi_b - r_b]],
            fmt="s",
            markerfacecolor="white",
            markeredgecolor=bare_color,
            markeredgewidth=mew,
            markersize=ms,
            ecolor=bare_color,
            elinewidth=elw,
            capsize=cap, capthick=elw,
            alpha=1.0 if is_signal else 0.92,
            zorder=marker_zorder,
        )

    # EN-side / ZH-side separator (between row index 4 and 5 in y_positions).
    sep_y = (y_positions[4] + y_positions[5]) / 2.0
    ax.axhline(
        sep_y, color=COLOR_AXIS_GUIDE, linewidth=0.8, linestyle="-",
        alpha=0.55, zorder=2,
    )

    # Y tick labels: roman + dark for signal, italic + muted for controls.
    ax.set_yticks(y_positions)
    yticklabels = [DISP[lab] for lab in labels]
    ax.set_yticklabels(yticklabels)
    for tick, lab in zip(ax.get_yticklabels(), labels):
        if lab in SIGNAL_PANEL:
            tick.set_color("#222222")
        else:
            tick.set_style("italic")
            tick.set_color("#7A7A7A")

    # Side annotations: EN-side / ZH-side (vertical, sober).
    en_y = float(np.mean(y_positions[:5]))
    zh_y = float(np.mean(y_positions[5:]))
    ax.text(
        1.025, en_y, "EN-side",
        transform=ax.get_yaxis_transform(),
        fontsize=9.5, ha="left", va="center",
        color="#555555", fontweight="bold", rotation=270,
    )
    ax.text(
        1.025, zh_y, "ZH-side",
        transform=ax.get_yaxis_transform(),
        fontsize=9.5, ha="left", va="center",
        color="#555555", fontweight="bold", rotation=270,
    )

    # X axis range — adapted to data with a small margin.
    all_lo = min(min(data[l]["attested"][1], data[l]["bare"][1]) for l in labels)
    all_hi = max(max(data[l]["attested"][2], data[l]["bare"][2]) for l in labels)
    x_pad = 0.020
    ax.set_xlim(min(all_lo, 0.0) - x_pad, all_hi + x_pad)

    # Tighten y limits so the topmost / bottom-most error bars are not
    # clipped by the axes box.
    ax.set_ylim(y_positions[-1] - 0.55, y_positions[0] + 0.55)

    # Axis labels, title, subtitle.
    ax.set_xlabel("Rank-biserial $r$   (within-domain $<$ between-domain)")
    ax.set_ylabel("")  # encoder names speak for themselves
    fig.suptitle(
        "Effect sizes across the cohort",
        fontsize=12, fontweight="bold", x=0.5, y=1.00, ha="center",
    )
    ax.set_title(
        "rank-biserial $r$ with 95% bootstrap CI   ·   "
        "italicised labels = bilingual control (§2.3)",
        fontsize=9.5, fontweight="normal", color="#444444", pad=8,
    )

    # Custom legend: 4 proxy artists (regime × panel) in two columns at the
    # lower-right corner of the plot, so the chart is self-documenting.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=COLOR_ATTESTED_SIGNAL,
               markeredgecolor=COLOR_ATTESTED_SIGNAL,
               markersize=6.5, label="attested · signal"),
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor="white",
               markeredgecolor=COLOR_BARE_SIGNAL,
               markeredgewidth=1.2,
               markersize=6.5, label="bare · signal"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=COLOR_ATTESTED_CONTROL,
               markeredgecolor=COLOR_ATTESTED_CONTROL,
               markersize=5.5, label="attested · control"),
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor="white",
               markeredgecolor=COLOR_BARE_CONTROL,
               markeredgewidth=1.0,
               markersize=5.5, label="bare · control"),
    ]
    # Legend sits below the x-axis so it never overlaps the data, and the
    # four-cell layout reads naturally as a 2x2 (regime × panel) matrix.
    leg = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        ncol=4, handlelength=1.0, handletextpad=0.5,
        columnspacing=1.8, borderpad=0.4,
        fontsize=8.8,
    )
    for txt in leg.get_texts():
        txt.set_color("#333333")

    # Cosmetic axis polish.
    ax.tick_params(colors="#333333")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")

    fig.tight_layout(pad=1.5)
    out = OUT / "fig2_forest_r_10encoders.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ wrote {out.name}")


if __name__ == "__main__":
    import sys
    domains = load_index()
    assert len(domains) == 364

    # CLI: pass --fig1 / --fig2 to regenerate only one; default = both.
    args = set(sys.argv[1:])
    run_all = not (args & {"--fig1", "--fig2"})

    if run_all or "--fig1" in args:
        print("Generating Fig.1 (histograms + KDE, BGE-EN-large attested)...")
        make_fig1(domains)

    if run_all or "--fig2" in args:
        print("\nGenerating Fig.2 (forest plot 10 readings × 2 variants, "
              "B=500 bootstrap)...")
        make_fig2(domains)

    print("\nDone.")
