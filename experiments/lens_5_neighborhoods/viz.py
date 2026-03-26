"""
Visualization for Lens V — Semantic Neighborhoods.

PNG figures (thesis-quality, 300 DPI):
  fig_jaccard_violins       — §3.2.1 cross vs within-WEIRD vs within-Sinic (violins)
  fig_pair_forest           — §3.2.1 forest plot: 15 pairs, mean ± std Jaccard
  fig_term_histogram        — §3.2.1 histogram of 397 per-term mean cross-tradition Jaccard
  fig_false_friends_bars    — §3.2.2 horizontal bars: top-20 by divergence (1−J̄)
  fig_domain_boxplot        — §3.2.3 domain box plots (real distributions) + significance
  fig_domain_pair_heatmap   — §3.2.3 heatmap: 7 domains × 9 cross-tradition pairs

Interactive HTML (Plotly CDN, self-contained, 8 tabs):
  build_html                — single lens5_interactive.html

Orchestrator:
  run_viz(results_dir, results) — called from lens5.main()
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.html_style import (
    page_head, tabs_bar, plots_script,
    C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_VERMIL, C_PURPLE, C_BLACK,
)

DOMAIN_COLORS = {
    "administrative":  "#0072B2",
    "civil":           "#E69F00",
    "constitutional":  "#009E73",
    "criminal":        "#D55E00",
    "international":   "#56B4E9",
    "labor_social":    "#CC79A7",
    "procedure":       "#999999",
}

DPI = 300

_RC = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def _apply_style() -> None:
    plt.rcParams.update(_RC)


def _short(label: str) -> str:
    return {
        "BGE-EN-large": "BGE-EN",
        "E5-large": "E5",
        "FreeLaw-EN": "FreeLaw",
        "BGE-ZH-large": "BGE-ZH",
        "Text2vec-large-ZH": "Text2vec",
        "Dmeta-ZH": "Dmeta",
    }.get(label, label)


# ---------------------------------------------------------------------------
# PNG: §3.2.1 — Jaccard violins
# ---------------------------------------------------------------------------

def fig_jaccard_violins(results: dict, save_dir: Path) -> Path:
    _apply_style()
    s321 = results["section_321"]

    cross_vals = [r["mean_jaccard"] for r in s321["cross_tradition"]]
    weird_vals = [r["mean_jaccard"] for r in s321["within_weird"]]
    sinic_vals = [r["mean_jaccard"] for r in s321["within_sinic"]]

    data = [cross_vals, weird_vals, sinic_vals]
    labels = ["Cross-tradition\n(9 pairs)", "Within-WEIRD\n(3 pairs)", "Within-Sinic\n(3 pairs)"]
    colors = [C_GREEN, C_BLUE, C_VERMIL]

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color(C_BLACK)
    parts["cmedians"].set_linewidth(1.5)

    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
        ax.scatter([i + j for j in jitter], vals, color=color, s=40,
                   zorder=3, edgecolors="white", linewidth=0.5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Jaccard similarity (J̄)")
    ax.set_title("§3.2.1 — Neighborhood overlap by pair group", fontsize=11)
    plt.tight_layout()
    out = save_dir / "321_jaccard_violins.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG: §3.2.1 — Pair forest plot
# ---------------------------------------------------------------------------

def fig_pair_forest(results: dict, save_dir: Path) -> Path:
    _apply_style()
    s321 = results["section_321"]

    pairs: list[dict] = []
    for group, color in [
        ("within_weird", C_BLUE),
        ("within_sinic", C_VERMIL),
        ("cross_tradition", C_GREEN),
    ]:
        for r in s321[group]:
            pairs.append({
                "label": f"{_short(r['model_a'])} × {_short(r['model_b'])}",
                "mean": r["mean_jaccard"],
                "std": r["std_jaccard"],
                "color": color,
            })

    n = len(pairs)
    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.45 + 1.5)))
    for i, p in enumerate(pairs):
        y = n - 1 - i
        ax.plot(p["mean"], y, "o", color=p["color"], ms=7, zorder=3)
        ax.plot([p["mean"] - p["std"], p["mean"] + p["std"]], [y, y],
                "-", color=p["color"], lw=2.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([p["label"] for p in reversed(pairs)], fontsize=8)
    ax.set_xlabel("Mean Jaccard ± std", fontsize=10)
    ax.set_title("§3.2.1 — Pair-level Jaccard similarity", fontsize=11)
    ax.legend(
        handles=[
            mpatches.Patch(color=C_BLUE, label="Within-WEIRD"),
            mpatches.Patch(color=C_VERMIL, label="Within-Sinic"),
            mpatches.Patch(color=C_GREEN, label="Cross-tradition"),
        ],
        loc="lower right", frameon=False, fontsize=8,
    )
    plt.tight_layout()
    out = save_dir / "321_pair_forest.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG: §3.2.1 — Per-term histogram (NEW)
# ---------------------------------------------------------------------------

def fig_term_histogram(results: dict, save_dir: Path) -> Path:
    """Histogram of 397 per-term mean cross-tradition Jaccard values."""
    _apply_style()
    all_terms = results["section_322"]["all_terms"]
    jaccards = [t["mean_jaccard"] for t in all_terms]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(jaccards, bins=50, color=C_BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
    mean_j = np.mean(jaccards)
    ax.axvline(mean_j, color=C_VERMIL, linewidth=1.5, linestyle="--",
               label=f"Mean J̄ = {mean_j:.3f}")
    ax.set_xlabel("Mean cross-tradition Jaccard (per term)")
    ax.set_ylabel("Number of terms")
    ax.set_title("§3.2.1 — Distribution of per-term neighborhood overlap (n=397)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out = save_dir / "321_term_histogram.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG: §3.2.2 — False friends bars
# ---------------------------------------------------------------------------

def fig_false_friends_bars(results: dict, save_dir: Path) -> Path:
    _apply_style()
    s322 = results["section_322"]
    top_20 = s322["top_20"]

    labels = [e["en"] for e in top_20]
    divs = [e["divergence"] for e in top_20]
    colors = [DOMAIN_COLORS.get(e["domain"], "#999999") for e in top_20]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = range(len(top_20) - 1, -1, -1)
    ax.barh(list(y_pos), divs, color=colors, height=0.7, alpha=0.85)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Divergence score (1 − J̄)", fontsize=10)
    ax.set_title("§3.2.2 — Top-20 juridical false friends", fontsize=11)

    used_domains = sorted(set(e["domain"] for e in top_20))
    handles = [mpatches.Patch(color=DOMAIN_COLORS.get(d, "#999"), label=d)
               for d in used_domains]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    plt.tight_layout()
    out = save_dir / "322_false_friends.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG: §3.2.3 — Domain box plot (real distributions)
# ---------------------------------------------------------------------------

def fig_domain_boxplot(results: dict, save_dir: Path) -> Path:
    _apply_style()
    s323 = results["section_323"]
    domain_values = s323["domain_values"]
    pairwise = s323.get("pairwise", [])
    domains = sorted(domain_values.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    bp_data = [domain_values[d] for d in domains]
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.6,
                    medianprops=dict(color=C_BLACK, linewidth=1.5))
    for patch, d in zip(bp["boxes"], domains):
        patch.set_facecolor(DOMAIN_COLORS.get(d, "#999"))
        patch.set_alpha(0.7)

    # Jittered scatter overlay
    rng = np.random.default_rng(42)
    for i, (d, vals) in enumerate(zip(domains, bp_data)):
        jx = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter([i + 1 + j for j in jx], vals,
                   color=DOMAIN_COLORS.get(d, "#999"), s=8, alpha=0.5, zorder=3)

    ns = [len(domain_values[d]) for d in domains]
    ax.set_xticks(range(1, len(domains) + 1))
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(domains, ns)],
                       fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Mean cross-tradition Jaccard (per term)")
    ax.set_title("§3.2.3 — Domain-level neighborhood divergence", fontsize=11)

    # Significance brackets
    sig_pairs = sorted([p for p in pairwise if p["significant"]],
                       key=lambda p: p["p_bonferroni"])
    if sig_pairs:
        y_max = max(max(v) for v in bp_data if v)
        y_step = y_max * 0.07
        for i, p in enumerate(sig_pairs[:5]):
            d1_idx = domains.index(p["domain_a"]) + 1
            d2_idx = domains.index(p["domain_b"]) + 1
            y = y_max + y_step * (i + 1)
            ax.plot([d1_idx, d1_idx, d2_idx, d2_idx],
                    [y - y_step*0.2, y, y, y - y_step*0.2], color="#333", lw=0.8)
            stars = "***" if p["p_bonferroni"] < 0.001 else "**" if p["p_bonferroni"] < 0.01 else "*"
            ax.text((d1_idx + d2_idx) / 2, y, stars, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = save_dir / "323_domain_boxplot.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG: §3.2.3 — Domain × pair heatmap (NEW)
# ---------------------------------------------------------------------------

def fig_domain_pair_heatmap(results: dict, save_dir: Path) -> Path:
    """Heatmap: 7 domains × 9 cross-tradition pairs — mean Jaccard per cell."""
    _apply_style()
    s323 = results["section_323"]
    heatmap_data = s323["pair_domain_heatmap"]
    domains = sorted(s323["domain_stats"].keys())

    pair_labels = [_short(r["pair"].split(" × ")[0]) + " × " +
                   _short(r["pair"].split(" × ")[1]) for r in heatmap_data]
    matrix = np.array([[r[d] for d in domains] for r in heatmap_data])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlBu", aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_yticks(range(len(pair_labels)))
    ax.set_xticklabels([d[:6] for d in domains], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pair_labels, fontsize=8)

    for i in range(len(pair_labels)):
        for j in range(len(domains)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Mean Jaccard")
    ax.set_title("§3.2.3 — Domain × model pair Jaccard heatmap", fontsize=11)
    plt.tight_layout()
    out = save_dir / "323_domain_pair_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Interactive HTML — Plotly
# ===========================================================================

def _pj_jaccard_violins(results: dict) -> str:
    s321 = results["section_321"]
    fig = go.Figure()
    for group, name, color in [
        ("cross_tradition", "Cross-tradition", C_GREEN),
        ("within_weird", "Within-WEIRD", C_BLUE),
        ("within_sinic", "Within-Sinic", C_VERMIL),
    ]:
        vals = [r["mean_jaccard"] for r in s321[group]]
        fig.add_trace(go.Box(
            y=vals, name=name, marker_color=color,
            boxpoints="all", jitter=0.3, pointpos=-1.5,
        ))
    fig.update_layout(
        title="§3.2.1 — Neighborhood overlap by pair group",
        yaxis_title="Mean Jaccard similarity",
        template="simple_white", height=450, showlegend=False,
    )
    return fig.to_json()


def _pj_pair_forest(results: dict) -> str:
    s321 = results["section_321"]
    traces = []
    for group, color, group_label in [
        ("within_weird", C_BLUE, "Within-WEIRD"),
        ("within_sinic", C_VERMIL, "Within-Sinic"),
        ("cross_tradition", C_GREEN, "Cross-tradition"),
    ]:
        ylabels, means, stds = [], [], []
        for r in s321[group]:
            ylabels.append(f"{_short(r['model_a'])} × {_short(r['model_b'])}")
            means.append(r["mean_jaccard"])
            stds.append(r["std_jaccard"])
        traces.append(go.Scatter(
            x=means, y=ylabels, mode="markers",
            marker=dict(size=9, color=color),
            error_x=dict(type="data", array=stds, symmetric=True),
            name=group_label,
            hovertemplate="%{y}<br>J̄=%{x:.4f} ± %{error_x.array:.4f}<extra></extra>",
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        title="§3.2.1 — Pair-level Jaccard forest plot",
        xaxis_title="Mean Jaccard ± std",
        template="simple_white", height=550,
    )
    return fig.to_json()


def _pj_term_histogram(results: dict) -> str:
    """Histogram of 397 per-term mean cross-tradition Jaccard."""
    all_terms = results["section_322"]["all_terms"]
    jaccards = [t["mean_jaccard"] for t in all_terms]
    domains = [t["domain"] for t in all_terms]

    fig = go.Figure()
    # One trace per domain for colored stacked histogram
    domain_set = sorted(set(domains))
    for d in domain_set:
        vals = [j for j, dom in zip(jaccards, domains) if dom == d]
        fig.add_trace(go.Histogram(
            x=vals, name=d, marker_color=DOMAIN_COLORS.get(d, "#999"),
            nbinsx=40, opacity=0.75,
        ))
    mean_j = np.mean(jaccards)
    fig.add_vline(x=mean_j, line_dash="dash", line_color=C_VERMIL,
                  annotation_text=f"Mean = {mean_j:.3f}")
    fig.update_layout(
        title="§3.2.1 — Per-term cross-tradition Jaccard distribution (n=397)",
        xaxis_title="Mean cross-tradition Jaccard",
        yaxis_title="Number of terms",
        barmode="stack",
        template="simple_white", height=500,
    )
    return fig.to_json()


def _pj_term_scatter(results: dict, results_dir: Path) -> str:
    """Scatter: each of 397 terms, one point per cross-tradition pair, with hover."""
    all_terms = results["section_322"]["all_terms"]
    s321 = results["section_321"]
    cross_pairs = s321["cross_tradition"]

    npz_dir = results_dir / "jaccard_per_term"
    fig = go.Figure()

    for r in cross_pairs:
        la, lb = r["model_a"], r["model_b"]
        npy_path = npz_dir / f"{la}_x_{lb}.npy"
        if not npy_path.exists():
            continue
        jacc = np.load(npy_path)
        pair_label = f"{_short(la)} × {_short(lb)}"
        hovers = [
            f"<b>{t['en']}</b> ({t['zh']})<br>"
            f"Domain: {t['domain']}<br>"
            f"J = {j:.4f}"
            for t, j in zip(all_terms, jacc)
        ]
        fig.add_trace(go.Scatter(
            x=list(range(len(jacc))),
            y=jacc.tolist(),
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            name=pair_label,
            hovertext=hovers,
            hoverinfo="text",
            visible="legendonly" if r != cross_pairs[0] else True,
        ))

    fig.update_layout(
        title="§3.2.1 — Per-term Jaccard for each cross-tradition pair",
        xaxis_title="Term index (sorted by pool position)",
        yaxis_title="Jaccard similarity",
        template="simple_white", height=550,
        legend=dict(title="Model pair (click to toggle)"),
    )
    return fig.to_json()


def _pj_false_friends_table(results: dict) -> str:
    """Detailed table of top-20 false friends with WEIRD vs Sinic neighbor lists."""
    s322 = results["section_322"]
    top_20 = s322["top_20"]

    # Build a rich go.Table
    header_vals = ["#", "Term (EN)", "Term (ZH)", "Domain", "Div.",
                   "WEIRD neighbors (majority-vote)", "Sinic neighbors (majority-vote)"]
    cells = [[] for _ in range(7)]
    for e in top_20:
        cells[0].append(e["rank"])
        cells[1].append(e["en"])
        cells[2].append(e["zh"])
        cells[3].append(e["domain"])
        cells[4].append(f"{e['divergence']:.3f}")
        w_nb = "<br>".join(
            f"{n['en']} [{n['domain']}] ({n['votes']}v)"
            for n in e["weird_neighbors"]
        )
        s_nb = "<br>".join(
            f"{n['en']} [{n['domain']}] ({n['votes']}v)"
            for n in e["sinic_neighbors"]
        )
        cells[5].append(w_nb)
        cells[6].append(s_nb)

    fig = go.Figure(go.Table(
        header=dict(
            values=header_vals,
            fill_color=C_BLUE,
            font=dict(color="white", size=11),
            align="left",
        ),
        cells=dict(
            values=cells,
            fill_color=[["#f9f9f9", "#ffffff"] * 10] * 7,
            align="left",
            font=dict(size=10),
            height=60,
        ),
    ))
    fig.update_layout(
        title="§3.2.2 — False friends: WEIRD vs Sinic neighborhood comparison",
        height=900,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig.to_json()


def _pj_false_friends_bars(results: dict) -> str:
    """Bar chart version with full hover."""
    s322 = results["section_322"]
    top_20 = s322["top_20"]

    labels = [e["en"] for e in top_20]
    divs = [e["divergence"] for e in top_20]
    colors = [DOMAIN_COLORS.get(e["domain"], "#999") for e in top_20]
    hovers = []
    for e in top_20:
        weird_nb = " | ".join(f"{n['en']} [{n['domain']}]" for n in e["weird_neighbors"])
        sinic_nb = " | ".join(f"{n['en']} [{n['domain']}]" for n in e["sinic_neighbors"])
        hovers.append(
            f"<b>{e['en']}</b> ({e['zh']})<br>"
            f"Domain: {e['domain']}<br>"
            f"Divergence: {e['divergence']:.4f}<br><br>"
            f"<b>WEIRD top-5:</b><br>{weird_nb}<br><br>"
            f"<b>Sinic top-5:</b><br>{sinic_nb}"
        )

    fig = go.Figure(go.Bar(
        y=labels[::-1], x=divs[::-1], orientation="h",
        marker_color=colors[::-1],
        hovertext=hovers[::-1], hoverinfo="text",
    ))
    fig.update_layout(
        title="§3.2.2 — Top-20 by divergence (hover for neighbor detail)",
        xaxis_title="Divergence score (1 − J̄)",
        template="simple_white", height=600,
    )
    return fig.to_json()


def _pj_domain_boxplot(results: dict) -> str:
    """Real box plots with individual data points per domain."""
    s323 = results["section_323"]
    domain_values = s323["domain_values"]
    domains = sorted(domain_values.keys())

    fig = go.Figure()
    for d in domains:
        fig.add_trace(go.Box(
            y=domain_values[d], name=f"{d} (n={len(domain_values[d])})",
            marker_color=DOMAIN_COLORS.get(d, "#999"),
            boxpoints="all", jitter=0.3, pointpos=-1.5,
            hovertemplate="J̄ = %{y:.4f}<extra></extra>",
        ))
    kw = s323["kruskal_wallis"]
    fig.update_layout(
        title=f"§3.2.3 — Domain divergence (Kruskal-Wallis H={kw['H']:.1f}, p={kw['p_value']:.2e})",
        yaxis_title="Mean cross-tradition Jaccard (per term)",
        template="simple_white", height=500, showlegend=False,
    )
    return fig.to_json()


def _pj_domain_pair_heatmap(results: dict) -> str:
    """Heatmap: 7 domains × 9 cross-tradition pairs."""
    s323 = results["section_323"]
    heatmap_data = s323["pair_domain_heatmap"]
    domains = sorted(s323["domain_stats"].keys())

    pair_labels = [_short(r["pair"].split(" × ")[0]) + " × " +
                   _short(r["pair"].split(" × ")[1]) for r in heatmap_data]
    matrix = [[r[d] for d in domains] for r in heatmap_data]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[d[:6] for d in domains],
        y=pair_labels,
        colorscale="RdYlBu",
        text=[[f"{v:.3f}" for v in row] for row in matrix],
        hovertemplate="%{y} × %{x}: J̄ = %{z:.4f}<extra></extra>",
        colorbar=dict(title="Mean J"),
    ))
    fig.update_layout(
        title="§3.2.3 — Domain × model pair Jaccard heatmap",
        template="simple_white", height=500,
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(results_dir: Path, results: dict, save_path: Path) -> Path:
    print("    Building Plotly figures...")
    plots = {
        "violins":        _pj_jaccard_violins(results),
        "forest":         _pj_pair_forest(results),
        "histogram":      _pj_term_histogram(results),
        "scatter":        _pj_term_scatter(results, results_dir),
        "ff_bars":        _pj_false_friends_bars(results),
        "ff_table":       _pj_false_friends_table(results),
        "domain_box":     _pj_domain_boxplot(results),
        "domain_heatmap": _pj_domain_pair_heatmap(results),
    }
    save_path.write_text(_html_template(plots), encoding="utf-8")
    return save_path


def _html_template(plots: dict[str, str]) -> str:
    tabs = tabs_bar([
        ("p1", "Overview"),
        ("p2", "Per-term histogram"),
        ("p3", "Per-term scatter"),
        ("p4", "Pair forest"),
        ("p5", "False friends (bars)"),
        ("p6", "False friends (table)"),
        ("p7", "Domain distributions"),
        ("p8", "Domain × pair heatmap"),
    ])

    plt_plots = {f"plt_{k}": v for k, v in plots.items()}

    return f"""<!DOCTYPE html>
<html lang="en">
{page_head("Lens V — Semantic Neighborhoods")}
<body>
<h1>Lens V — Semantic Neighborhoods</h1>
<p class="subtitle">§3.2 — For each of 397 legal terms, the 15 most semantically similar concepts ($k$-nearest neighbors) are identified in each of 6 embedding models. The Jaccard index measures the degree of overlap between the neighbor sets produced by different models, across all 15 model pairs.</p>
{tabs}

<div id="p1" class="panel active">
  <div class="question">
    <b>What are "nearest neighbors" in an embedding space, and what does the Jaccard index measure?</b>
    In an embedding model's vector space, every term has a set of "nearest neighbors" — the other terms whose vectors are closest to it. These are the terms the model treats as most semantically associated.
    For example, the nearest neighbors of "negligence" might include "liability", "damages", "tort", and "duty" in one model, but "recklessness", "carelessness", "fault", and "breach" in another.
    The Jaccard index measures how much two such neighbor sets overlap.
    Given two sets $A$ and $B$, $J = |A \\cap B| \\;/\\; |A \\cup B|$: the number of shared neighbors divided by the total number of distinct neighbors across both sets.
    $J = 1$ means the two models assign exactly the same 15 neighbors to a term; $J = 0$ means completely different neighbors — no overlap at all.
  </div>
  <div class="card">
    <h2>Computation</h2>
    <p>The neighbor pool contains 9,472 terms: 397 core legal terms (the focus of this experiment), 8,975 background legal terms (also drawn from the Hong Kong Department of Justice Bilingual Legal Glossary, covering the full breadth of legal vocabulary), and 100 Swadesh-100 control words (basic everyday terms like "water", "fire", "hand", used as a baseline reference).
    For each core term, the 15 nearest neighbors are identified by cosine similarity in each model.
    The Jaccard index is then computed for each of the 15 model pairs: 3 within-WEIRD (comparing two English-language models), 3 within-Sinic (comparing two Chinese-language models), and 9 cross-tradition (comparing one English model with one Chinese model).
    Each violin in the plot below shows the distribution of Jaccard values across the 397 core terms for one model pair. Each dot is one model pair's mean Jaccard.</p>
  </div>
  <div id="plt_violins"></div>
</div>

<div id="p2" class="panel">
  <div class="question">
    <b>How are per-term Jaccard values distributed across the 397 legal terms?</b>
    This histogram shows the distribution of the mean cross-tradition Jaccard value ($\\bar{{J}}$) for each of the 397 terms, where the average is taken over the 9 cross-tradition model pairs.
    Each bar represents a bin of $\\bar{{J}}$ values; the height of the bar shows how many terms fall into that bin.
    The colours indicate which branch of law each term belongs to (e.g., constitutional, criminal, civil), allowing visual inspection of whether certain branches cluster at particular overlap levels.
  </div>
  <div class="card">
    <h2>Reading the histogram</h2>
    <p>The horizontal axis (X) shows the mean cross-tradition Jaccard $\\bar{{J}}$ for each term.
    The vertical axis (Y) shows the count of terms in each bin.
    A term with $\\bar{{J}} = 0.05$ has, on average, almost no neighbors in common across WEIRD and Sinic models — the models associate it with entirely different concepts.
    A term with $\\bar{{J}} = 0.40$ shares roughly 40% of its neighborhood.
    The dashed vertical line marks the overall mean across all 397 terms.
    Colours show which branch of law each term belongs to: stacked segments within each bar indicate the composition by legal domain.</p>
  </div>
  <div id="plt_histogram"></div>
</div>

<div id="p3" class="panel">
  <div class="question">
    <b>What does each dot in this scatter plot represent?</b>
    Each dot represents a single legal term's Jaccard similarity ($J$) for one specific cross-tradition model pair.
    Because there are 9 cross-tradition pairs, each term can appear up to 9 times (once per pair), each shown in a different colour.
    The interactive legend on the right allows you to click on a model pair name to show or hide its dots, making it possible to compare specific pairs or isolate a single pair for inspection. Hovering over any dot reveals the English term, its Chinese translation, and its branch of law.
  </div>
  <div class="card">
    <h2>Reading the scatter</h2>
    <p>The horizontal axis (X) represents the term index, sorted by pool position (approximately alphabetical).
    The vertical axis (Y) shows the Jaccard similarity $J$ for that term and model pair: higher values mean more overlap in the two models' neighbor sets.
    Each colour corresponds to one of the 9 cross-tradition model pairs.
    By default, only one pair is visible; click the legend entries to toggle additional pairs on or off.
    This view is useful for identifying specific terms that behave differently across different model pair combinations.</p>
  </div>
  <div id="plt_scatter"></div>
</div>

<div id="p4" class="panel">
  <div class="question">
    <b>What is a forest plot, and how should it be read?</b>
    A forest plot displays point estimates (the dot) with uncertainty measures (the error bars) for multiple items on the same scale, allowing visual comparison at a glance.
    Here, each item is a model pair: the dot shows the mean Jaccard $\\bar{{J}}$ across all 397 terms, and the horizontal bars extend $\\pm 1$ standard deviation from that mean.
    The three colours distinguish the pair groups: blue for within-WEIRD (two English-language models compared), red for within-Sinic (two Chinese-language models compared), and green for cross-tradition (one English model compared with one Chinese model).
  </div>
  <div class="card">
    <h2>Reading the forest plot</h2>
    <p>Model pairs are listed on the vertical axis (Y). The horizontal axis (X) shows the mean Jaccard value.
    The dot marks the average neighborhood overlap for that pair across all 397 terms.
    The error bars span one standard deviation in each direction: wider bars mean there is more variability from term to term within that pair — some terms have high overlap while others have low overlap.
    Narrow bars mean the pair's overlap level is relatively consistent across terms.
    Comparing dots across the three colour groups reveals whether within-tradition pairs (blue and red) tend to cluster at different positions than cross-tradition pairs (green).</p>
  </div>
  <div id="plt_forest"></div>
</div>

<div id="p5" class="panel">
  <div class="question">
    <b>What are "false friends" in this context, and what does the divergence score measure?</b>
    In comparative linguistics, "false friends" (faux amis) are words that look or sound similar across two languages but carry different meanings — for example, the Italian "attuale" means "current", not "actual".
    Here the concept is extended to legal terminology: a legal "false friend" is a term whose surface translation is equivalent across English and Chinese, but whose semantic associations — as captured by the embedding model — differ substantially.
    A term like "sovereignty" might be embedded near "autonomy", "independence", and "self-determination" in an English model, but near "state", "authority", and "territorial integrity" in a Chinese model.
    The divergence score quantifies this: $\\text{{divergence}} = 1 - \\bar{{J}}$, where $\\bar{{J}}$ is the mean Jaccard across the 9 cross-tradition pairs. A divergence of 1.0 means no overlap at all; a divergence of 0.0 means perfect overlap.
    This chart ranks the 20 terms with the highest divergence — those whose semantic neighborhoods differ most dramatically between WEIRD and Sinic embedding models.
  </div>
  <div class="card">
    <h2>Definition of divergence score and hover details</h2>
    <p>For each term $t$: $\\text{{div}}(t) = 1 - \\bar{{J}}_{{\\text{{cross}}}}(t)$,
    where $\\bar{{J}}_{{\\text{{cross}}}}$ is the mean Jaccard over the 9 cross-tradition
    model pairs. The term "false friends" is borrowed from comparative linguistics, where it describes cross-linguistic deceptive cognates.
    Hovering over any bar reveals the top-5 neighbors for that term in the WEIRD tradition and the top-5 in the Sinic tradition, allowing direct inspection of which semantic associations differ.
    Colours indicate the branch of law to which each term belongs.</p>
  </div>
  <div id="plt_ff_bars"></div>
</div>

<div id="p6" class="panel">
  <div class="question">
    <b>How are the "consensus neighbors" for each tradition determined?</b>
    For each of the top-20 high-divergence terms, this table shows the top-5 nearest neighbors as agreed upon by the models within each tradition, using a majority-vote procedure.
    Each tradition has 3 models. A term is included as a "consensus neighbor" if at least 2 of the 3 models in that tradition list it among their 15 nearest neighbors.
    This ensures that the reported neighbors are not artefacts of a single model but reflect a pattern shared across multiple models trained on similar data.
  </div>
  <div class="card">
    <h2>Reading the table</h2>
    <p>The left column ("WEIRD neighbors") lists the top-5 consensus neighbors from the English-language models, with a vote count next to each (e.g., "3/3" or "2/3") indicating how many of the 3 WEIRD models agree that this term is a close neighbor.
    A vote of 3/3 means all three models in the tradition agree that this is a close neighbor; 2/3 means two out of three agree.
    The right column ("Sinic neighbors") follows the same format for the Chinese-language models.
    Comparing the two columns for any given term reveals which semantic associations are shared and which differ across the two traditions.
    The domain tag in brackets (e.g., [criminal], [constitutional]) shows the branch of law that each neighbor belongs to.</p>
  </div>
  <div id="plt_ff_table"></div>
</div>

<div id="p7" class="panel">
  <div class="question">
    <b>Does neighborhood overlap vary across branches of law?</b>
    This panel presents one box plot per branch of law (e.g., constitutional, criminal, civil, international), showing the distribution of per-term mean cross-tradition Jaccard values ($\\bar{{J}}_{{\\text{{cross}}}}$) within each branch.
    Each dot overlaid on the box plot represents one term's mean cross-tradition Jaccard.
    The title includes the result of a Kruskal-Wallis $H$ test, which is a non-parametric statistical test (an analogue of one-way ANOVA that does not assume normally distributed data) that assesses whether the distributions of Jaccard values differ across branches of law.
  </div>
  <div class="card">
    <h2>Reading the box plots</h2>
    <p>The box spans from the 25th to the 75th percentile (the interquartile range, IQR): the middle 50% of terms in that branch fall within this range.
    The horizontal line inside the box marks the median — the value below which half the terms fall.
    Whiskers extend to the most extreme data points within 1.5 times the IQR.
    Each dot is one term's mean cross-tradition Jaccard.
    The Kruskal-Wallis $H$ test (a non-parametric analogue of one-way ANOVA) tests whether the distributions of Jaccard values differ across branches of law.
    A low $p$-value (conventionally $p < 0.05$) would be consistent with non-uniform distributions across branches, but the test alone does not identify which specific branches differ.</p>
  </div>
  <div id="plt_domain_box"></div>
</div>

<div id="p8" class="panel">
  <div class="question">
    <b>Is the pattern of neighborhood overlap consistent across all 9 cross-tradition model pairs, or does it depend on which specific models are compared?</b>
    This heatmap serves as a robustness check. If a branch of law shows low overlap, it is important to verify that this pattern holds across all 9 cross-tradition pairs — not just one or two.
    A branch that is consistently "cold" (low Jaccard) across all 9 rows represents a pattern that is robust to model choice: the low overlap is not an artefact of a single model pair but a structural feature observed across multiple independent comparisons.
  </div>
  <div class="card">
    <h2>Reading the heatmap</h2>
    <p>Rows correspond to the 9 cross-tradition model pairs (each identified by the names of its two models, one English and one Chinese).
    Columns correspond to the 7 branches of law.
    Each cell shows the mean Jaccard across all terms in that branch for that model pair.
    The colour scale runs from cold (blue, low Jaccard = low overlap between neighbor sets) to warm (red, high Jaccard = high overlap).
    A column that is consistently cold across all 9 rows represents a branch where low overlap is not an artefact of a single model pair.
    Conversely, a column that is consistently warm across all rows represents a branch where models tend to agree on the semantic neighborhoods regardless of which specific English and Chinese model is used.</p>
  </div>
  <div id="plt_domain_heatmap"></div>
</div>

{plots_script(plt_plots)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_viz(results_dir: Path, results: dict) -> None:
    """Generate all Lens V figures (PNG + HTML). Called from lens5.main()."""
    png_dir = results_dir / "figures" / "png"
    html_dir = results_dir / "figures" / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    print("\n[viz] Generating PNG figures...")

    if "section_321" in results:
        p = fig_jaccard_violins(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_pair_forest(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_322" in results:
        p = fig_term_histogram(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_false_friends_bars(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_323" in results:
        p = fig_domain_boxplot(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_domain_pair_heatmap(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    print("[viz] Generating interactive HTML...")
    html_path = build_html(results_dir, results, html_dir / "lens5_interactive.html")
    print(f"  {html_path.name}")

    print(f"[viz] Done — {len(generated)} PNG + 1 HTML")
