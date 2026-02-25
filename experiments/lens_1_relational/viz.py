"""
Visualization for Lens I — Relational Distance Structure.

PNG figures (thesis-quality, 300 DPI):
  fig_311_domains        — §3.1.1 domain distribution bar chart
  fig_311_confidence     — §3.1.1 k-NN confidence histogram
  fig_311_intra_inter    — §3.1.1 intra vs inter violin (3-panel, WEIRD)
  fig_311_legal_control  — §3.1.1 legal vs control violin (3-panel, WEIRD)
  fig_312_topology       — §3.1.2 K×K domain topology heatmaps (3-panel)
  fig_rsa_forest         — §3.1.4 RSA forest plot (15 pairs, colored by group)
  fig_rsa_null           — §3.1.4 RSA null distributions (15-panel grid)

Interactive HTML (Plotly CDN, self-contained):
  build_html             — single lens1_interactive.html with 5 tabs

Orchestrator:
  run_viz(results_dir, results) — called from lens1.main()
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Style — Okabe-Ito colorblind-safe palette
# ---------------------------------------------------------------------------

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_SKY    = "#56B4E9"
C_VERMIL = "#D55E00"
C_PURPLE = "#CC79A7"
C_BLACK  = "#000000"

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
# §3.1.1 — Domain distribution
# ---------------------------------------------------------------------------

def fig_311_domains(results: dict, save_dir: Path) -> Path:
    """Bar chart of background term domain distribution (§3.1.1)."""
    _apply_style()
    domain_counts = results["section_311"]["domain_counts"]
    domains = sorted(domain_counts, key=domain_counts.__getitem__, reverse=True)
    counts = [domain_counts[d] for d in domains]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(domains)), counts, color=C_BLUE, alpha=0.85, width=0.6)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Term count")
    ax.set_title("§3.1.1 — Background term domain distribution", fontsize=11)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count), ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    out = save_dir / "311_domain_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_311_confidence(results_dir: Path, results: dict, save_dir: Path) -> Path:
    """Histogram of k-NN confidence scores with threshold line (§3.1.1)."""
    _apply_style()
    csv_path = results_dir / "background_review.csv"
    confidences = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            confidences.append(float(row["confidence"]))
    conf = np.array(confidences)
    threshold = results["section_311"]["low_confidence_threshold"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf, bins=40, color=C_BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(
        threshold, color=C_VERMIL, linewidth=1.5, linestyle="--",
        label=f"Threshold {threshold:.2f}  (4/k)",
    )
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title("§3.1.1 — k-NN confidence distribution", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out = save_dir / "311_confidence.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# §3.1.1 — Intra vs inter-domain distances
# ---------------------------------------------------------------------------

def fig_311_intra_inter(
    results_dir: Path, weird_labels: list[str], save_dir: Path
) -> Path:
    """3-panel violin: intra vs inter-domain distances per WEIRD model (§3.1.1)."""
    _apply_style()
    dist_dir = results_dir / "distances"

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    for ax, label in zip(axes, weird_labels):
        npz = np.load(dist_dir / f"{label}.npz")
        parts = ax.violinplot(
            [npz["intra"], npz["inter"]], positions=[0, 1],
            showmedians=True, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], [C_SKY, C_ORANGE]):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color(C_BLACK)
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Intra-domain", "Inter-domain"])
        ax.set_title(_short(label), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Cosine distance")

    fig.suptitle("§3.1.1 — Intra vs inter-domain distances (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "311_intra_inter.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# §3.1.1 — Legal vs control distances
# ---------------------------------------------------------------------------

def fig_311_legal_control(
    results_dir: Path, weird_labels: list[str], save_dir: Path
) -> Path:
    """3-panel violin: legal vs control distances per WEIRD model (§3.1.1)."""
    _apply_style()
    dist_dir = results_dir / "distances"

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    for ax, label in zip(axes, weird_labels):
        npz = np.load(dist_dir / f"{label}.npz")
        parts = ax.violinplot(
            [npz["legal"], npz["control"]], positions=[0, 1],
            showmedians=True, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], [C_BLUE, C_VERMIL]):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color(C_BLACK)
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Legal", "Control"])
        ax.set_title(_short(label), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Cosine distance")

    fig.suptitle("§3.1.1 — Legal vs control distances (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "311_legal_control.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# §3.1.2 — Domain topology
# ---------------------------------------------------------------------------

def fig_312_topology(results: dict, save_dir: Path) -> Path:
    """3-panel annotated heatmaps: K×K domain topology per WEIRD model (§3.1.2)."""
    _apply_style()
    per_model = results["section_31"]["per_model"]
    model_labels = list(per_model.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, mlabel in zip(axes, model_labels):
        topo = per_model[mlabel]["domain_topology"]
        domains = topo["domains"]
        matrix = np.array(topo["matrix"])
        k = len(domains)
        abbrev = [d[:5] for d in domains]

        im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(abbrev, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(abbrev, fontsize=7)
        ax.set_title(_short(mlabel), fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=5, color="black")

    fig.suptitle("§3.1.2 — Domain topology K×K (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "312_topology.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# RSA — Forest plot
# ---------------------------------------------------------------------------

def fig_rsa_forest(results: dict, save_dir: Path) -> Path:
    """Forest plot: Spearman ρ ± 95% CI for all 15 RSA pairs (§3.1.4)."""
    _apply_style()
    rsa_data = results["section_314"]

    pairs: list[dict] = []
    for group, color in [
        ("within_weird", C_BLUE),
        ("within_sinic", C_VERMIL),
        ("cross_tradition", C_GREEN),
    ]:
        for r in rsa_data[group]:
            pairs.append({
                "label": f"{_short(r['model_a'])} × {_short(r['model_b'])}",
                "rho": r["rho"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "color": color,
            })

    n = len(pairs)
    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.5 + 1.5)))

    for i, p in enumerate(pairs):
        y = n - 1 - i
        ax.plot(p["rho"], y, "o", color=p["color"], ms=7, zorder=3)
        ax.plot([p["ci_low"], p["ci_high"]], [y, y], "-", color=p["color"], lw=2.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([p["label"] for p in reversed(pairs)], fontsize=8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_xlim(-0.15, 0.85)
    ax.set_title("RSA forest plot — §3.1.4", fontsize=11)
    ax.legend(
        handles=[
            mpatches.Patch(color=C_BLUE, label="Within-WEIRD"),
            mpatches.Patch(color=C_VERMIL, label="Within-Sinic"),
            mpatches.Patch(color=C_GREEN, label="Cross-tradition"),
        ],
        loc="lower right", frameon=False, fontsize=8,
    )
    plt.tight_layout()
    out = save_dir / "rsa_forest.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# RSA — Null distributions
# ---------------------------------------------------------------------------

def fig_rsa_null(results_dir: Path, results: dict, save_dir: Path) -> Path:
    """5×3 grid of null distributions with observed ρ line (§3.1.4)."""
    _apply_style()
    dist_dir = results_dir / "distributions"
    rsa_data = results["section_314"]

    all_pairs: list[tuple[dict, str, str]] = []
    for group, color in [
        ("within_weird", C_BLUE),
        ("within_sinic", C_VERMIL),
        ("cross_tradition", C_GREEN),
    ]:
        for r in rsa_data[group]:
            all_pairs.append((r, color, group))

    n = len(all_pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))
    axes_flat = axes.flatten()

    for i, (r, color, _) in enumerate(all_pairs):
        ax = axes_flat[i]
        la, lb = r["model_a"], r["model_b"]
        fname = dist_dir / f"{la}_x_{lb}.npz"

        if not fname.exists():
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(f"{_short(la)} × {_short(lb)}", fontsize=8)
            continue

        null = np.load(fname)["null"]
        ax.hist(null, bins=50, color=color, alpha=0.6, density=True, edgecolor="none")
        ax.axvline(r["rho"], color=C_BLACK, linewidth=1.5,
                   label=f"ρ={r['rho']:.3f}")
        ax.set_title(f"{_short(la)} × {_short(lb)}", fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_xlabel("ρ (null)", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("RSA null distributions — §3.1.4", fontsize=11)
    plt.tight_layout()
    out = save_dir / "rsa_null_distributions.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Interactive HTML — Plotly helpers
# ---------------------------------------------------------------------------

def _pj_domains_bar(results: dict) -> str:
    domain_counts = results["section_311"]["domain_counts"]
    domains = sorted(domain_counts, key=domain_counts.__getitem__, reverse=True)
    fig = go.Figure(go.Bar(
        x=domains, y=[domain_counts[d] for d in domains],
        marker_color=C_BLUE, opacity=0.85,
    ))
    fig.update_layout(
        title="§3.1.1 — Background domain distribution",
        xaxis_title="Domain", yaxis_title="Term count",
        template="simple_white", height=400,
    )
    return fig.to_json()


def _pj_confidence_hist(results_dir: Path, results: dict) -> str:
    confidences = []
    with open(results_dir / "background_review.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            confidences.append(float(row["confidence"]))
    threshold = results["section_311"]["low_confidence_threshold"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=confidences, nbinsx=40, marker_color=C_BLUE,
                               opacity=0.8, name="Confidence"))
    fig.add_vline(x=threshold, line_dash="dash", line_color=C_VERMIL,
                  annotation_text=f"Threshold {threshold:.2f}")
    fig.update_layout(
        title="§3.1.1 — k-NN confidence distribution",
        xaxis_title="Confidence", yaxis_title="Count",
        template="simple_white", height=400, showlegend=False,
    )
    return fig.to_json()


def _downsample(arr: np.ndarray, max_n: int = 5000) -> np.ndarray:
    """Downsample array for Plotly violin (KDE is stable at ~5k points)."""
    if len(arr) <= max_n:
        return arr
    rng = np.random.default_rng(42)
    return rng.choice(arr, size=max_n, replace=False)


def _pj_violin_intra_inter(results_dir: Path, weird_labels: list[str]) -> str:
    dist_dir = results_dir / "distances"
    fig = make_subplots(cols=3, rows=1,
                        subplot_titles=[_short(l) for l in weird_labels])
    for i, label in enumerate(weird_labels, 1):
        path = dist_dir / f"{label}.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for arr, name, color in [
            (npz["intra"], "Intra", C_SKY),
            (npz["inter"], "Inter", C_ORANGE),
        ]:
            fig.add_trace(go.Violin(
                y=_downsample(arr), name=name, box_visible=True,
                meanline_visible=True, fillcolor=color, line_color=C_BLACK,
                opacity=0.75, showlegend=(i == 1),
            ), row=1, col=i)
    fig.update_layout(title="§3.1.1 — Intra vs inter-domain distances",
                      template="simple_white", height=500, violinmode="group")
    return fig.to_json()


def _pj_violin_legal_control(results_dir: Path, weird_labels: list[str]) -> str:
    dist_dir = results_dir / "distances"
    fig = make_subplots(cols=3, rows=1,
                        subplot_titles=[_short(l) for l in weird_labels])
    for i, label in enumerate(weird_labels, 1):
        path = dist_dir / f"{label}.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for arr, name, color in [
            (npz["legal"], "Legal", C_BLUE),
            (npz["control"], "Control", C_VERMIL),
        ]:
            fig.add_trace(go.Violin(
                y=_downsample(arr), name=name, box_visible=True,
                meanline_visible=True, fillcolor=color, line_color=C_BLACK,
                opacity=0.75, showlegend=(i == 1),
            ), row=1, col=i)
    fig.update_layout(title="§3.1.1 — Legal vs control distances",
                      template="simple_white", height=500, violinmode="group")
    return fig.to_json()


def _pj_topology(results: dict) -> str:
    per_model = results["section_31"]["per_model"]
    models = list(per_model.keys())
    fig = make_subplots(cols=3, rows=1,
                        subplot_titles=[_short(m) for m in models])
    for i, mlabel in enumerate(models, 1):
        topo = per_model[mlabel]["domain_topology"]
        domains = [d[:6] for d in topo["domains"]]
        matrix = np.array(topo["matrix"])
        fig.add_trace(go.Heatmap(
            z=matrix, x=domains, y=domains, colorscale="RdYlBu_r",
            showscale=(i == 3),
            text=[[f"{v:.3f}" for v in row] for row in matrix],
            hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>",
        ), row=1, col=i)
    fig.update_layout(title="§3.1.2 — Domain topology",
                      template="simple_white", height=450)
    return fig.to_json()


def _pj_rsa_forest(results: dict) -> str:
    rsa_data = results["section_314"]
    traces = []
    for group, color, group_label in [
        ("within_weird", C_BLUE, "Within-WEIRD"),
        ("within_sinic", C_VERMIL, "Within-Sinic"),
        ("cross_tradition", C_GREEN, "Cross-tradition"),
    ]:
        ylabels, rhos, lo, hi = [], [], [], []
        for r in rsa_data[group]:
            ylabels.append(f"{_short(r['model_a'])} × {_short(r['model_b'])}")
            rhos.append(r["rho"])
            lo.append(r["ci_low"])
            hi.append(r["ci_high"])
        traces.append(go.Scatter(
            x=rhos, y=ylabels, mode="markers",
            marker=dict(size=9, color=color),
            error_x=dict(
                type="data", symmetric=False,
                array=[h - r for h, r in zip(hi, rhos)],
                arrayminus=[r - l for r, l in zip(rhos, lo)],
            ),
            name=group_label,
        ))
    fig = go.Figure(traces)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="RSA forest plot — §3.1.4",
        xaxis_title="Spearman ρ",
        template="simple_white", height=550,
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(results_dir: Path, results: dict, save_path: Path) -> Path:
    """Build a single self-contained Plotly HTML with 5-tab navigation."""
    weird_labels = results["meta"]["weird_models"]
    print("    Building Plotly figures...")
    plots = {
        "domains":       _pj_domains_bar(results),
        "confidence":    _pj_confidence_hist(results_dir, results),
        "intra_inter":   _pj_violin_intra_inter(results_dir, weird_labels),
        "legal_control": _pj_violin_legal_control(results_dir, weird_labels),
        "topology":      _pj_topology(results),
        "forest":        _pj_rsa_forest(results),
    }
    save_path.write_text(_html_template(plots), encoding="utf-8")
    return save_path


def _html_template(plots: dict[str, str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lens I — Relational Distance Structure</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 16px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 12px; }}
  .tabs {{ display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }}
  .tab-btn {{
    padding: 7px 16px; border: 1px solid #ccc; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 0.85rem; color: #444;
  }}
  .tab-btn.active {{ background: {C_BLUE}; color: #fff; border-color: {C_BLUE}; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .row2 {{ display: flex; gap: 12px; }}
  .row2 > div {{ flex: 1; min-width: 0; }}
  .note {{
    background: #f5f5f5; border-left: 3px solid {C_BLUE}; padding: 10px 14px;
    margin-bottom: 14px; font-size: 0.85rem; line-height: 1.5;
  }}
  .note h3 {{ margin: 0 0 6px 0; font-size: 0.9rem; }}
  .note p {{ margin: 4px 0; }}
</style>
</head>
<body>
<h1>Lens I — Relational Distance Structure</h1>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('p311', this)">§3.1.1 Background</button>
  <button class="tab-btn" onclick="showTab('p311b', this)">§3.1.1 Intra/Inter</button>
  <button class="tab-btn" onclick="showTab('p311c', this)">§3.1.1 Legal/Control</button>
  <button class="tab-btn" onclick="showTab('p312', this)">§3.1.2 Topology</button>
  <button class="tab-btn" onclick="showTab('pRSA', this)">§3.1.4 RSA</button>
</div>

<div id="p311" class="panel active">
  <div class="note">
    <h3>§3.1.1 — Background term domain assignment</h3>
    <p>Each background term $b$ is assigned to a domain via $k$-NN majority vote among the
    $k=7$ nearest core terms in cosine similarity space.</p>
    <p><b>Left:</b> distribution of assigned domains across all background terms.</p>
    <p><b>Right:</b> confidence = fraction of $k$ neighbors agreeing on the majority domain.
    $$\\text{{confidence}}(b) = \\frac{{\\#\\{{\\text{{neighbors with majority domain}}\\}}}}{{k}}$$
    The dashed line marks the low-confidence threshold $4/k$.</p>
  </div>
  <div class="row2">
    <div id="plt_domains"></div>
    <div id="plt_confidence"></div>
  </div>
</div>

<div id="p311b" class="panel">
  <div class="note">
    <h3>§3.1.1 — Intra-domain vs inter-domain distances</h3>
    <p>For each WEIRD model, the Relational Dissimilarity Matrix (RDM) over core terms is computed:</p>
    <p>$$\\text{{RDM}}[i,j] = 1 - \\cos(\\mathbf{{v}}_i, \\mathbf{{v}}_j) = 1 - \\frac{{\\mathbf{{v}}_i \\cdot \\mathbf{{v}}_j}}{{\\|\\mathbf{{v}}_i\\| \\, \\|\\mathbf{{v}}_j\\|}}$$</p>
    <p>The upper triangle of the RDM is split into <b>intra-domain</b> pairs (both terms share the
    same domain) and <b>inter-domain</b> pairs. The distributions shown are these two sets.
    If the embedding encodes domain structure, intra-domain distances should be systematically
    smaller than inter-domain distances.</p>
    <p>Statistical test: Mann-Whitney $U$ with rank-biserial $r = 1 - 2U/(n_x n_y)$, one-sided (intra &lt; inter).</p>
  </div>
  <div id="plt_intra_inter"></div>
</div>

<div id="p311c" class="panel">
  <div class="note">
    <h3>§3.1.1 — Legal vs control distances</h3>
    <p>Same RDM construction. Distances are split into:</p>
    <ul style="margin:4px 0; padding-left:20px;">
      <li><b>Legal–legal:</b> upper triangle of the core-term sub-matrix (only legal terms).</li>
      <li><b>Legal–control:</b> all pairwise distances between core terms and control terms
      (non-legal concrete/everyday words).</li>
    </ul>
    <p>If the embedding clusters legal terms more tightly than random words, legal–legal
    distances should be smaller than legal–control distances.</p>
    <p>Statistical test: Mann-Whitney $U$, one-sided (legal &lt; control), rank-biserial $r$.</p>
  </div>
  <div id="plt_legal_control"></div>
</div>

<div id="p312" class="panel">
  <div class="note">
    <h3>§3.1.2 — Domain topology</h3>
    <p>A $K \\times K$ matrix where $K$ is the number of legal domains. Each cell is the mean
    cosine distance between all pairs of terms belonging to domains $d_i$ and $d_j$:</p>
    <p>$$T[d_i, d_j] = \\frac{{1}}{{|d_i| \\cdot |d_j|}} \\sum_{{a \\in d_i}} \\sum_{{b \\in d_j}} \\text{{RDM}}[a, b]$$</p>
    <p>Diagonal entries use only the upper triangle (intra-domain mean distance).
    Lower values indicate domains that are closer in the embedding space.</p>
  </div>
  <div id="plt_topology"></div>
</div>

<div id="pRSA" class="panel">
  <div class="note">
    <h3>§3.1.4 — Representational Similarity Analysis (RSA)</h3>
    <p>For each pair of models $(A, B)$, an RDM is computed over the same 397 core terms.
    RSA measures the correlation between the two RDMs:</p>
    <p>$$\\rho = \\text{{Spearman}}\\big(\\text{{upper\\_tri}}(\\text{{RDM}}_A),\\; \\text{{upper\\_tri}}(\\text{{RDM}}_B)\\big)$$</p>
    <p><b>Significance:</b> Mantel permutation test ($B=10\\,000$). Rows and columns of $\\text{{RDM}}_B$
    are jointly permuted to generate a null distribution of $\\rho$ values.
    $p = \\max\\big(\\#\\{{\\rho_\\pi \\geq \\rho_{{\\text{{obs}}}}\\}} / B,\\; 1/B\\big)$ (Phipson &amp; Smyth 2010).</p>
    <p><b>Confidence interval:</b> Block bootstrap ($B=1\\,000$). Term indices (not pairs) are
    resampled with replacement to respect the dependency structure (Nili et al. 2014).</p>
    <p>Each point is one model pair. Error bars = 95% bootstrap CI. Colors distinguish
    within-WEIRD, within-Sinic, and cross-tradition pairs.</p>
  </div>
  <div id="plt_forest"></div>
</div>

<script>
function showTab(id, btn) {{
  document.querySelectorAll(".panel").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  btn.classList.add("active");
  setTimeout(function() {{
    var panel = document.getElementById(id);
    var plots = panel.querySelectorAll("[id^='plt_']");
    plots.forEach(function(el) {{ Plotly.Plots.resize(el); }});
  }}, 50);
}}

const figs = {{
  plt_domains:       {plots["domains"]},
  plt_confidence:    {plots["confidence"]},
  plt_intra_inter:   {plots["intra_inter"]},
  plt_legal_control: {plots["legal_control"]},
  plt_topology:      {plots["topology"]},
  plt_forest:        {plots["forest"]}
}};

for (const [id, spec] of Object.entries(figs)) {{
  Plotly.newPlot(id, spec.data, spec.layout, {{responsive: true}});
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_viz(results_dir: Path, results: dict) -> None:
    """Generate all Lens I figures (PNG + HTML). Called from lens1.main()."""
    png_dir = results_dir / "figures" / "png"
    html_dir = results_dir / "figures" / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    weird_labels: list[str] = results["meta"]["weird_models"]
    dist_dir = results_dir / "distances"
    generated: list[Path] = []

    print("\n[viz] Generating PNG figures...")

    if "section_311" in results:
        p = fig_311_domains(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_311_confidence(results_dir, results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_31" in results:
        if any((dist_dir / f"{m}.npz").exists() for m in weird_labels):
            p = fig_311_intra_inter(results_dir, weird_labels, png_dir)
            generated.append(p)
            print(f"  {p.name}")

            p = fig_311_legal_control(results_dir, weird_labels, png_dir)
            generated.append(p)
            print(f"  {p.name}")

        p = fig_312_topology(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_314" in results:
        p = fig_rsa_forest(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        ddist = results_dir / "distributions"
        if ddist.exists() and any(ddist.iterdir()):
            p = fig_rsa_null(results_dir, results, png_dir)
            generated.append(p)
            print(f"  {p.name}")

    print("[viz] Generating interactive HTML...")
    html_path = build_html(results_dir, results, html_dir / "lens1_interactive.html")
    print(f"  {html_path.name}")

    print(f"[viz] Done — {len(generated)} PNG + 1 HTML")
