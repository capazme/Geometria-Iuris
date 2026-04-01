"""
Visualization for Lens II — Emergent Taxonomy.

PNG figures (thesis-quality, 300 DPI):
  fig_441_fm_bar         — §4.4.1 FM at k=7 per model + null baseline
  fig_442_fm_curves      — §4.4.2 FM(k) curves: vs human + cross-tradition
  fig_443_cross_forest   — §4.4.3 Forest plot: 15 pairs, FM at k=7

Interactive HTML (Plotly CDN, self-contained, 3 tabs):
  build_html             — single lens2_interactive.html

Orchestrator:
  run_viz(results_dir, results) — called from lens2.main()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.html_style import (
    page_head, tabs_bar, plots_script, format_p,
    C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_VERMIL, C_PURPLE, C_BLACK,
)

DPI = 300
_RC = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# Tradition colors (consistent with Lens I)
WEIRD_COLOR = C_BLUE
SINIC_COLOR = C_VERMIL
CROSS_COLOR = C_GREEN

# Shade variants for individual models
WEIRD_SHADES = ["#0072B2", "#4A9FD9", "#7FC4F0"]
SINIC_SHADES = ["#D55E00", "#E88B4D", "#F0B899"]


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
# PNG: §4.4.1 — FM bar chart
# ---------------------------------------------------------------------------

def fig_441_fm_bar(results: dict, save_dir: Path) -> Path:
    _apply_style()
    s441 = results["section_441"]
    pm = s441["per_model"]

    labels = list(pm.keys())
    fms = [pm[l]["fm"] for l in labels]
    colors = [WEIRD_COLOR if pm[l]["tradition"] == "WEIRD" else SINIC_COLOR for l in labels]
    null_means = [pm[l]["null_mean"] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = list(range(len(labels) - 1, -1, -1))
    ax.barh(y_pos, fms, color=colors, height=0.6, alpha=0.85, zorder=2)

    mean_null = np.mean(null_means)
    ax.axvline(mean_null, color="#888", linewidth=1.2, linestyle="--", zorder=1,
               label=f"Null mean = {mean_null:.3f}")

    for i, (y, fm, lab) in enumerate(zip(y_pos, fms, labels)):
        p = pm[lab]["p_value"]
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(fm + 0.003, y, f"{fm:.3f} ({p_str})", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_short(l) for l in labels], fontsize=9)
    ax.set_xlabel("Fowlkes-Mallows index (FM at k=7)")
    ax.set_title("§4.4.1 — Taxonomic recovery: model vs human domain labels", fontsize=11)
    ax.legend(
        handles=[
            mpatches.Patch(color=WEIRD_COLOR, label="WEIRD"),
            mpatches.Patch(color=SINIC_COLOR, label="Sinic"),
            plt.Line2D([0], [0], color="#888", ls="--", label=f"Null mean ({mean_null:.3f})"),
        ],
        loc="lower right", frameon=False, fontsize=8,
    )
    plt.tight_layout()
    out = save_dir / "441_fm_bar.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG: §4.4.2 — FM curves (2-panel)
# ---------------------------------------------------------------------------

def fig_442_fm_curves(results: dict, save_dir: Path) -> Path:
    _apply_style()
    s442 = results["section_442"]
    k_range = s442["k_range"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: FM vs human
    weird_models = results["meta"]["weird_models"]
    sinic_models = results["meta"]["sinic_models"]

    for i, label in enumerate(weird_models):
        curve = s442["human_curves"][label]
        ax1.plot(k_range, curve, color=WEIRD_SHADES[i], marker="o", ms=3,
                 label=_short(label), linewidth=1.5)
    for i, label in enumerate(sinic_models):
        curve = s442["human_curves"][label]
        ax1.plot(k_range, curve, color=SINIC_SHADES[i], marker="s", ms=3,
                 label=_short(label), linewidth=1.5)

    ax1.axvline(7, color="#888", linewidth=1, linestyle="--", alpha=0.6)
    ax1.text(7.2, ax1.get_ylim()[0] + 0.001, "k=7", fontsize=8, color="#888")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("FM (model vs human labels)")
    ax1.set_title("FM(k) vs human taxonomy", fontsize=10)
    ax1.legend(fontsize=7, frameon=False, ncol=2)

    # Right: cross-tradition FM curves
    cross_curves = s442["cross_tradition_curves"]
    all_cross = np.array(list(cross_curves.values()))
    mean_cross = all_cross.mean(axis=0)

    for curve in all_cross:
        ax2.plot(k_range, curve, color=CROSS_COLOR, alpha=0.15, linewidth=0.8)
    ax2.plot(k_range, mean_cross, color=CROSS_COLOR, linewidth=2.5,
             label=f"Cross-tradition mean (n=9)")

    # Within-tradition means
    within_w = np.array(list(s442["within_weird_curves"].values()))
    within_s = np.array(list(s442["within_sinic_curves"].values()))
    ax2.plot(k_range, within_w.mean(axis=0), color=WEIRD_COLOR, linewidth=2,
             linestyle="--", label="Within-WEIRD mean")
    ax2.plot(k_range, within_s.mean(axis=0), color=SINIC_COLOR, linewidth=2,
             linestyle="--", label="Within-Sinic mean")

    ax2.axvline(7, color="#888", linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("FM (model A vs model B)")
    ax2.set_title("Cross-tradition taxonomic agreement", fontsize=10)
    ax2.legend(fontsize=7, frameon=False)

    plt.suptitle("§4.4.2 — Taxonomic horizons", fontsize=12, y=1.02)
    plt.tight_layout()
    out = save_dir / "442_fm_curves.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG: §4.4.3 — Cross-tradition forest plot
# ---------------------------------------------------------------------------

def fig_443_cross_forest(results: dict, save_dir: Path) -> Path:
    _apply_style()
    s443 = results["section_443"]

    pairs: list[dict] = []
    for group, color in [
        ("within_weird", WEIRD_COLOR),
        ("within_sinic", SINIC_COLOR),
        ("cross_tradition", CROSS_COLOR),
    ]:
        for r in s443[group]:
            pairs.append({
                "label": f"{_short(r['model_a'])} × {_short(r['model_b'])}",
                "fm": r["fm"],
                "color": color,
            })

    n = len(pairs)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.4 + 1)))
    for i, p in enumerate(pairs):
        y = n - 1 - i
        ax.plot(p["fm"], y, "o", color=p["color"], ms=8, zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels([p["label"] for p in reversed(pairs)], fontsize=8)
    ax.set_xlabel("Fowlkes-Mallows index (k=7)", fontsize=10)
    ax.set_title("§4.4.3 — Cross-tradition taxonomic agreement", fontsize=11)
    ax.legend(
        handles=[
            mpatches.Patch(color=WEIRD_COLOR, label="Within-WEIRD"),
            mpatches.Patch(color=SINIC_COLOR, label="Within-Sinic"),
            mpatches.Patch(color=CROSS_COLOR, label="Cross-tradition"),
        ],
        loc="lower left", frameon=False, fontsize=8,
    )
    plt.tight_layout()
    out = save_dir / "443_cross_fm_forest.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Interactive HTML — Plotly
# ===========================================================================

def _pj_fm_bar(results: dict) -> str:
    s441 = results["section_441"]
    pm = s441["per_model"]
    labels = list(pm.keys())

    colors = [WEIRD_COLOR if pm[l]["tradition"] == "WEIRD" else SINIC_COLOR for l in labels]
    fms = [pm[l]["fm"] for l in labels]
    hovers = [
        f"<b>{_short(l)}</b> ({pm[l]['tradition']})<br>"
        f"FM = {pm[l]['fm']:.4f}<br>"
        f"p = {pm[l]['p_value']:.4f}<br>"
        f"Null: {pm[l]['null_mean']:.4f} ± {pm[l]['null_std']:.4f}<br>"
        f"Complete linkage: {pm[l]['fm_complete_linkage']:.4f}"
        for l in labels
    ]

    fig = go.Figure(go.Bar(
        y=[_short(l) for l in labels][::-1],
        x=fms[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertext=hovers[::-1],
        hoverinfo="text",
    ))
    null_mean = np.mean([pm[l]["null_mean"] for l in labels])
    fig.add_vline(x=null_mean, line_dash="dash", line_color="#888",
                  annotation_text=f"Null mean = {null_mean:.3f}")
    fig.update_layout(
        title="§4.4.1 — FM at k=7: model partition vs human domain labels",
        xaxis_title="Fowlkes-Mallows index",
        template="simple_white", height=400,
    )
    return fig.to_json()


def _pj_fm_curves(results: dict) -> str:
    s442 = results["section_442"]
    k_range = s442["k_range"]
    weird_models = results["meta"]["weird_models"]
    sinic_models = results["meta"]["sinic_models"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["FM(k) vs human taxonomy",
                                        "Cross-tradition FM(k)"])

    # Left: vs human
    for i, label in enumerate(weird_models):
        fig.add_trace(go.Scatter(
            x=k_range, y=s442["human_curves"][label],
            mode="lines+markers", marker=dict(size=4),
            line=dict(color=WEIRD_SHADES[i]), name=_short(label),
            hovertemplate=f"{_short(label)}<br>k=%{{x}}<br>FM=%{{y:.4f}}<extra></extra>",
        ), row=1, col=1)
    for i, label in enumerate(sinic_models):
        fig.add_trace(go.Scatter(
            x=k_range, y=s442["human_curves"][label],
            mode="lines+markers", marker=dict(size=4, symbol="square"),
            line=dict(color=SINIC_SHADES[i]), name=_short(label),
            hovertemplate=f"{_short(label)}<br>k=%{{x}}<br>FM=%{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

    # Right: cross-tradition
    cross_curves = s442["cross_tradition_curves"]
    all_cross = np.array(list(cross_curves.values()))
    mean_cross = all_cross.mean(axis=0)

    for key, curve in cross_curves.items():
        parts = key.split("_x_")
        pair_label = f"{_short(parts[0])} × {_short(parts[1])}"
        fig.add_trace(go.Scatter(
            x=k_range, y=curve,
            mode="lines", line=dict(color=CROSS_COLOR, width=0.8),
            opacity=0.3, showlegend=False,
            hovertemplate=f"{pair_label}<br>k=%{{x}}<br>FM=%{{y:.4f}}<extra></extra>",
        ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=k_range, y=mean_cross.tolist(),
        mode="lines+markers", marker=dict(size=4),
        line=dict(color=CROSS_COLOR, width=3), name="Cross mean",
    ), row=1, col=2)

    within_w = np.array(list(s442["within_weird_curves"].values()))
    within_s = np.array(list(s442["within_sinic_curves"].values()))
    fig.add_trace(go.Scatter(
        x=k_range, y=within_w.mean(axis=0).tolist(),
        mode="lines", line=dict(color=WEIRD_COLOR, width=2, dash="dash"),
        name="Within-WEIRD mean",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=k_range, y=within_s.mean(axis=0).tolist(),
        mode="lines", line=dict(color=SINIC_COLOR, width=2, dash="dash"),
        name="Within-Sinic mean",
    ), row=1, col=2)

    fig.update_layout(
        title="§4.4.2 — Taxonomic horizons",
        template="simple_white", height=500,
    )
    fig.update_xaxes(title_text="Number of clusters (k)")
    fig.update_yaxes(title_text="Fowlkes-Mallows index")
    return fig.to_json()


def _pj_cross_forest(results: dict) -> str:
    s443 = results["section_443"]
    traces = []
    for group, color, group_label in [
        ("within_weird", WEIRD_COLOR, "Within-WEIRD"),
        ("within_sinic", SINIC_COLOR, "Within-Sinic"),
        ("cross_tradition", CROSS_COLOR, "Cross-tradition"),
    ]:
        ylabels = [f"{_short(r['model_a'])} × {_short(r['model_b'])}" for r in s443[group]]
        fms = [r["fm"] for r in s443[group]]
        traces.append(go.Scatter(
            x=fms, y=ylabels, mode="markers",
            marker=dict(size=10, color=color),
            name=group_label,
            hovertemplate="%{y}<br>FM = %{x:.4f}<extra></extra>",
        ))
    fig = go.Figure(traces)
    mw = s443["perm_cross_vs_within"]
    fig.update_layout(
        title=f"§4.4.3 — Taxonomic agreement at k=7 (MW p={mw['p_value']:.4f}, r={mw['effect_r']:+.3f})",
        xaxis_title="Fowlkes-Mallows index (k=7)",
        template="simple_white", height=500,
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(results_dir: Path, results: dict, save_path: Path) -> Path:
    print("    Building Plotly figures...")
    plots = {
        "fm_bar":    _pj_fm_bar(results),
        "fm_curves": _pj_fm_curves(results),
        "forest":    _pj_cross_forest(results),
    }
    save_path.write_text(_html_template(results, plots), encoding="utf-8")
    return save_path


def _html_template(results: dict, plots: dict[str, str]) -> str:
    s441 = results.get("section_441", {})
    s443 = results.get("section_443", {})
    pm = s441.get("per_model", {})
    mw = s443.get("perm_cross_vs_within", {})

    tabs = tabs_bar([
        ("p1", "§4.4.1 Recovery"),
        ("p2", "§4.4.2 Horizons"),
        ("p3", "§4.4.3 Cross-tradition"),
    ])

    # Metrics row for §4.4.1
    metric_cards = ""
    for label, info in pm.items():
        tradition = info["tradition"]
        color = WEIRD_COLOR if tradition == "WEIRD" else SINIC_COLOR
        metric_cards += (
            f'<div class="metric" style="border-left: 3px solid {color}">'
            f'<div class="metric-value">{info["fm"]:.3f}</div>'
            f'<div class="metric-label">{_short(label)}</div>'
            f'<div class="metric-detail">p={info["p_value"]:.3f}</div>'
            f'</div>\n'
        )

    plt_plots = {f"plt_{k}": v for k, v in plots.items()}

    return f"""<!DOCTYPE html>
<html lang="en">
{page_head("Lens II — Emergent Taxonomy")}
<body>
<h1>Lens II — Emergent Taxonomy</h1>
<p class="subtitle">§4.4 — Horizons. The 397 core legal terms carry human domain labels (7 domains). Agglomerative hierarchical clustering (average linkage) is applied to each model's cosine-distance matrix. The Fowlkes-Mallows index measures how well the geometric partition recovers the human taxonomy.</p>
{tabs}

<div id="p1" class="panel active">
  <div class="question">
    <b>Does the geometric structure of the embedding space recover the human classification of legal vocabulary?</b>
    Each of the 6 embedding models produces a 397 x 397 distance matrix. Agglomerative hierarchical clustering partitions the 397 terms into k=7 clusters (matching the 7 human domain labels). The Fowlkes-Mallows index (FM) measures agreement between the model's partition and the human labels: FM=1 means perfect agreement, FM=0 means no pair is co-clustered consistently.
  </div>
  <div class="card">
    <h2>Results</h2>
    <div class="metrics">{metric_cards}</div>
    <p>Statistical significance is assessed by permutation test: human labels are randomly shuffled 1,000 times, generating an empirical null distribution. The dashed line in the chart marks the mean null FM.</p>
  </div>
  <div id="plt_fm_bar"></div>
</div>

<div id="p2" class="panel">
  <div class="question">
    <b>At which taxonomic granularity do the embeddings best align with human categories? And where do WEIRD and Sinic traditions agree or diverge?</b>
    The left panel shows the FM(k) curve for each model: FM is computed between the model's k-cluster partition and the human 7-domain labels, for k ranging from 2 to 20. A peak at k=7 would confirm that the geometry's natural resolution matches the human taxonomy.
    The right panel shows cross-tradition agreement: for each of the 9 WEIRD x Sinic model pairs, FM is computed between their respective k-cluster partitions. The thick green line is the mean across all 9 pairs; dashed lines show within-tradition means.
  </div>
  <div class="card">
    <h2>Reading the curves</h2>
    <p>Left panel: if all curves peak at k=7, the models' geometric taxonomy has the same resolution as the human classification. If peaks occur at different k, the geometry encodes a different organizational granularity.
    Right panel: high FM values mean the two traditions partition legal vocabulary similarly at that k. The gap between cross-tradition (green) and within-tradition (blue/red dashed) shows how much taxonomy diverges across cultural boundaries.</p>
  </div>
  <div id="plt_fm_curves"></div>
</div>

<div id="p3" class="panel">
  <div class="question">
    <b>Do WEIRD and Sinic models agree more with each other than either agrees with human taxonomy?</b>
    At k=7, the Fowlkes-Mallows index is computed between every pair of models' partitions. Within-WEIRD pairs (blue) compare two English-language models; within-Sinic pairs (red) compare two Chinese-language models; cross-tradition pairs (green) compare one of each.
  </div>
  <div class="card">
    <h2>Key finding</h2>
    <p>Cross-tradition FM ({s443.get('summary', {}).get('mean_fm_cross', 0):.3f} mean) is far higher than model-vs-human FM (~0.44). The models agree with each other about how to partition legal vocabulary far more than they agree with the human domain classification. This suggests the geometric taxonomy encodes structural properties of legal language that differ from expert categorical knowledge.
    Mann-Whitney test: p={mw.get('p_value', 1):.4f}, effect r={mw.get('effect_r', 0):+.3f}.</p>
  </div>
  <div id="plt_forest"></div>
</div>

{plots_script(plt_plots)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_viz(results_dir: Path, results: dict) -> None:
    png_dir = results_dir / "figures" / "png"
    html_dir = results_dir / "figures" / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    print("\n[viz] Generating PNG figures...")

    if "section_441" in results:
        p = fig_441_fm_bar(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_442" in results:
        p = fig_442_fm_curves(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_443" in results:
        p = fig_443_cross_forest(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    print("[viz] Generating interactive HTML...")
    html_path = build_html(results_dir, results, html_dir / "lens2_interactive.html")
    print(f"  {html_path.name}")

    print(f"[viz] Done — {len(generated)} PNG + 1 HTML")
