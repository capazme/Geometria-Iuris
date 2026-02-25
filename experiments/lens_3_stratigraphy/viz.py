"""
Visualization for Lens III — Layer Stratigraphy (§3.1.3).

PNG figures (thesis-quality, 300 DPI):
  fig_domain_signal_curve  — r vs layer (6 models, Okabe-Ito)
  fig_rsa_convergence      — ρ vs layer (6 models)
  fig_drift_by_domain      — 3-panel (WEIRD): mean drift per domain per layer
  fig_jaccard_by_domain    — 3-panel (WEIRD): mean Jaccard per domain per layer
  fig_drift_heatmap        — Terms × layers heatmap (BGE-EN-large, sorted by total drift)

Interactive HTML (Plotly CDN, self-contained):
  build_html               — single lens3_interactive.html with 4 tabs

Orchestrator:
  run_viz(results_dir, results) — called from lens3.main()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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

# One color per model (6 models)
MODEL_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_VERMIL, C_PURPLE]

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
# PNG — §3.1.3b domain signal curve
# ---------------------------------------------------------------------------

def fig_domain_signal_curve(results: dict, save_dir: Path) -> Path:
    """r (domain signal) vs layer for all 6 models."""
    _apply_style()
    per_model = results["section_313b"]["per_model"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, data) in enumerate(per_model.items()):
        r_vals = data["domain_signal_r"]
        layers = list(range(len(r_vals)))
        ax.plot(layers, r_vals, "o-", color=MODEL_COLORS[i % len(MODEL_COLORS)],
                label=_short(label), markersize=3, linewidth=1.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Domain signal r (rank-biserial)")
    ax.set_title("§3.1.3b — Domain signal emergence across layers", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    out = save_dir / "313b_domain_signal.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG — §3.1.3b RSA convergence
# ---------------------------------------------------------------------------

def fig_rsa_convergence(results: dict, save_dir: Path) -> Path:
    """ρ(layer vs final) for all 6 models."""
    _apply_style()
    per_model = results["section_313b"]["per_model"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, data) in enumerate(per_model.items()):
        rho_vals = data["rsa_vs_final_rho"]
        layers = list(range(len(rho_vals)))
        ax.plot(layers, rho_vals, "o-", color=MODEL_COLORS[i % len(MODEL_COLORS)],
                label=_short(label), markersize=3, linewidth=1.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ (layer vs final)")
    ax.set_title("§3.1.3b — RSA convergence to final layer", fontsize=11)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(0.9, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    out = save_dir / "313b_rsa_convergence.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG — §3.1.3a drift by domain (3-panel, WEIRD)
# ---------------------------------------------------------------------------

def fig_drift_by_domain(results: dict, save_dir: Path) -> Path:
    """3-panel: mean drift per domain per layer for each WEIRD model."""
    _apply_style()
    weird_labels = results["meta"]["weird_models"]
    per_model = results["section_313a"]["per_model"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, label in zip(axes, weird_labels):
        data = per_model[label]
        drift_by_dom = data["drift_by_domain"]
        domains = sorted(drift_by_dom.keys())
        for j, dom in enumerate(domains):
            curve = drift_by_dom[dom]
            ax.plot(range(len(curve)), curve, linewidth=1.0, alpha=0.7,
                    label=dom[:8])
        ax.set_xlabel("Transition (l → l+1)")
        ax.set_title(_short(label), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Mean cosine drift")

    axes[-1].legend(frameon=False, fontsize=6, ncol=2, loc="upper right",
                    bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("§3.1.3a — Drift by domain (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "313a_drift_by_domain.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG — §3.1.3a Jaccard by domain (3-panel, WEIRD)
# ---------------------------------------------------------------------------

def fig_jaccard_by_domain(results: dict, save_dir: Path) -> Path:
    """3-panel: mean Jaccard per domain per layer for each WEIRD model."""
    _apply_style()
    weird_labels = results["meta"]["weird_models"]
    per_model = results["section_313a"]["per_model"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, label in zip(axes, weird_labels):
        data = per_model[label]
        jaccard_by_dom = data["jaccard_by_domain"]
        domains = sorted(jaccard_by_dom.keys())
        for j, dom in enumerate(domains):
            curve = jaccard_by_dom[dom]
            ax.plot(range(len(curve)), curve, linewidth=1.0, alpha=0.7,
                    label=dom[:8])
        ax.set_xlabel("Transition (l → l+1)")
        ax.set_title(_short(label), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Mean Jaccard distance")

    axes[-1].legend(frameon=False, fontsize=6, ncol=2, loc="upper right",
                    bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("§3.1.3a — Jaccard instability by domain (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "313a_jaccard_by_domain.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PNG — §3.1.3a drift heatmap (BGE-EN-large)
# ---------------------------------------------------------------------------

def fig_drift_heatmap(results: dict, save_dir: Path) -> Path:
    """Terms × layers heatmap (BGE-EN-large, sorted by total drift)."""
    _apply_style()
    per_model = results["section_313a"]["per_model"]

    # Use first WEIRD model (BGE-EN-large)
    primary = results["meta"]["weird_models"][0]
    data = per_model[primary]

    # Reconstruct drift matrix from top_drift_terms? No — we need full matrix.
    # Load from cache.
    layer_cache = Path(__file__).parent / "results" / "layer_vectors" / f"{primary}.npz"
    if not layer_cache.exists():
        print(f"  [skip] Drift heatmap — no cached layer vectors for {primary}")
        return save_dir / "313a_drift_heatmap.png"

    layers_data = np.load(layer_cache)["layers"]  # (N, L+1, dim)
    from scipy.spatial.distance import cosine as cos_dist
    n_terms, n_states, _ = layers_data.shape
    drift = np.zeros((n_terms, n_states - 1), dtype=np.float32)
    for t in range(n_terms):
        for l in range(n_states - 1):
            drift[t, l] = cos_dist(layers_data[t, l], layers_data[t, l + 1])

    # Sort by total drift descending
    total = drift.sum(axis=1)
    order = np.argsort(total)[::-1]
    drift_sorted = drift[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(drift_sorted, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Transition (l → l+1)")
    ax.set_ylabel("Terms (sorted by total drift)")
    ax.set_title(f"§3.1.3a — Drift heatmap ({_short(primary)})", fontsize=11)
    plt.colorbar(im, ax=ax, label="Cosine drift", fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = save_dir / "313a_drift_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Interactive HTML — Plotly
# ---------------------------------------------------------------------------

def _pj_domain_signal_rsa(results: dict) -> str:
    """Domain signal + RSA convergence with dropdown per model."""
    per_model = results["section_313b"]["per_model"]
    labels = list(per_model.keys())

    fig = go.Figure()
    for i, label in enumerate(labels):
        data = per_model[label]
        layers = list(range(len(data["domain_signal_r"])))
        visible = (i == 0)

        fig.add_trace(go.Scatter(
            x=layers, y=data["domain_signal_r"],
            mode="lines+markers", name=f"{_short(label)} — r",
            marker=dict(size=5), line=dict(color=C_BLUE),
            visible=visible,
        ))
        fig.add_trace(go.Scatter(
            x=layers, y=data["rsa_vs_final_rho"],
            mode="lines+markers", name=f"{_short(label)} — ρ",
            marker=dict(size=5), line=dict(color=C_VERMIL, dash="dash"),
            visible=visible, yaxis="y2",
        ))

    # Dropdown buttons
    buttons = []
    for i, label in enumerate(labels):
        vis = [False] * (len(labels) * 2)
        vis[i * 2] = True
        vis[i * 2 + 1] = True
        buttons.append(dict(label=_short(label), method="update",
                            args=[{"visible": vis}]))

    fig.update_layout(
        title="§3.1.3b — Domain signal & RSA convergence",
        xaxis_title="Layer",
        yaxis=dict(title="Domain signal r", side="left"),
        yaxis2=dict(title="RSA ρ (vs final)", side="right", overlaying="y"),
        template="simple_white", height=500,
        updatemenus=[dict(buttons=buttons, direction="down",
                         x=0.98, xanchor="right", y=1.15, yanchor="top")],
    )
    return fig.to_json()


def _pj_drift_curves(results: dict) -> str:
    """Drift curves per domain with dropdown per model."""
    per_model = results["section_313a"]["per_model"]
    labels = list(per_model.keys())
    all_domains = sorted(set().union(
        *(d["drift_by_domain"].keys() for d in per_model.values())
    ))

    # Consistent domain colors
    domain_colors = {}
    palette = [C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_VERMIL, C_PURPLE,
               C_BLACK, "#999999", "#666666"]
    for j, dom in enumerate(all_domains):
        domain_colors[dom] = palette[j % len(palette)]

    fig = go.Figure()
    traces_per_model = len(all_domains)

    for i, label in enumerate(labels):
        data = per_model[label]
        visible = (i == 0)
        for dom in all_domains:
            curve = data["drift_by_domain"].get(dom, [])
            fig.add_trace(go.Scatter(
                x=list(range(len(curve))), y=curve,
                mode="lines+markers", name=dom,
                marker=dict(size=4), line=dict(color=domain_colors[dom]),
                visible=visible,
                showlegend=visible,
            ))

    buttons = []
    for i, label in enumerate(labels):
        vis = [False] * (len(labels) * traces_per_model)
        for j in range(traces_per_model):
            vis[i * traces_per_model + j] = True
        buttons.append(dict(label=_short(label), method="update",
                            args=[{"visible": vis}]))

    fig.update_layout(
        title="§3.1.3a — Drift by domain",
        xaxis_title="Transition (l → l+1)",
        yaxis_title="Mean cosine drift",
        template="simple_white", height=500,
        updatemenus=[dict(buttons=buttons, direction="down",
                         x=0.98, xanchor="right", y=1.15, yanchor="top")],
    )
    return fig.to_json()


def _pj_jaccard_curves(results: dict) -> str:
    """Jaccard curves per domain with dropdown per model."""
    per_model = results["section_313a"]["per_model"]
    labels = list(per_model.keys())
    all_domains = sorted(set().union(
        *(d["jaccard_by_domain"].keys() for d in per_model.values())
    ))

    domain_colors = {}
    palette = [C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_VERMIL, C_PURPLE,
               C_BLACK, "#999999", "#666666"]
    for j, dom in enumerate(all_domains):
        domain_colors[dom] = palette[j % len(palette)]

    fig = go.Figure()
    traces_per_model = len(all_domains)

    for i, label in enumerate(labels):
        data = per_model[label]
        visible = (i == 0)
        for dom in all_domains:
            curve = data["jaccard_by_domain"].get(dom, [])
            fig.add_trace(go.Scatter(
                x=list(range(len(curve))), y=curve,
                mode="lines+markers", name=dom,
                marker=dict(size=4), line=dict(color=domain_colors[dom]),
                visible=visible,
                showlegend=visible,
            ))

    buttons = []
    for i, label in enumerate(labels):
        vis = [False] * (len(labels) * traces_per_model)
        for j in range(traces_per_model):
            vis[i * traces_per_model + j] = True
        buttons.append(dict(label=_short(label), method="update",
                            args=[{"visible": vis}]))

    fig.update_layout(
        title="§3.1.3a — Jaccard instability by domain",
        xaxis_title="Transition (l → l+1)",
        yaxis_title="Mean Jaccard distance",
        template="simple_white", height=500,
        updatemenus=[dict(buttons=buttons, direction="down",
                         x=0.98, xanchor="right", y=1.15, yanchor="top")],
    )
    return fig.to_json()


def _pj_drift_heatmap(results: dict) -> str:
    """Drift heatmap with hover and dropdown for all 6 models."""
    from scipy.spatial.distance import cosine as cos_dist
    import json as _json

    cache_dir = Path(__file__).parent / "results" / "layer_vectors"
    all_labels = results["meta"]["weird_models"] + results["meta"]["sinic_models"]

    # Load term names for hover
    _ROOT = Path(__file__).parent.parent
    index_path = _ROOT / "data" / "processed" / "embeddings" / "index.json"
    with open(index_path, encoding="utf-8") as f:
        index = _json.load(f)
    core_terms = [t for t in index if t["tier"] == "core" and t.get("domain")]
    term_names = [f"{t['en']} [{t['domain']}]" for t in core_terms]

    fig = go.Figure()

    for i, label in enumerate(all_labels):
        layer_cache = cache_dir / f"{label}.npz"
        if not layer_cache.exists():
            # Empty placeholder trace
            fig.add_trace(go.Heatmap(
                z=[[0]], x=["N/A"], y=["No data"],
                visible=(i == 0), showscale=False,
            ))
            continue

        layers_data = np.load(layer_cache)["layers"]
        n_terms, n_states, _ = layers_data.shape

        drift = np.zeros((n_terms, n_states - 1), dtype=np.float32)
        for t in range(n_terms):
            for ll in range(n_states - 1):
                drift[t, ll] = cos_dist(layers_data[t, ll], layers_data[t, ll + 1])

        total = drift.sum(axis=1)
        order = np.argsort(total)[::-1]
        drift_sorted = drift[order]
        names_sorted = [term_names[j] for j in order]

        fig.add_trace(go.Heatmap(
            z=drift_sorted.tolist(),
            x=[f"L{ll}\u2192{ll+1}" for ll in range(n_states - 1)],
            y=names_sorted,
            colorscale="YlOrRd",
            hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
            visible=(i == 0),
        ))

    buttons = []
    for i, label in enumerate(all_labels):
        vis = [False] * len(all_labels)
        vis[i] = True
        buttons.append(dict(label=_short(label), method="update",
                            args=[{"visible": vis}]))

    fig.update_layout(
        title="§3.1.3a — Drift heatmap",
        xaxis_title="Transition",
        yaxis_title="Terms (sorted by total drift)",
        template="simple_white",
        height=max(600, len(term_names) * 3),
        updatemenus=[dict(buttons=buttons, direction="down",
                         x=0.98, xanchor="right", y=1.15, yanchor="top")],
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(results: dict, save_path: Path) -> Path:
    """Build a single self-contained Plotly HTML with 4-tab navigation."""
    print("    Building Plotly figures...")
    plots = {}

    has_b = "section_313b" in results
    has_a = "section_313a" in results

    if has_b:
        plots["signal_rsa"] = _pj_domain_signal_rsa(results)
    if has_a:
        plots["drift"] = _pj_drift_curves(results)
        plots["jaccard"] = _pj_jaccard_curves(results)
        plots["heatmap"] = _pj_drift_heatmap(results)

    save_path.write_text(_html_template(plots), encoding="utf-8")
    return save_path


def _html_template(plots: dict[str, str]) -> str:
    tabs_html = ""
    panels_html = ""
    plot_entries = []

    # Note blocks per tab (formula + description, no interpretation)
    notes = {
        "signal_rsa": """
    <h3>§3.1.3b — Domain signal emergence &amp; RSA convergence</h3>
    <p>Two curves per model, one per metric. Use the dropdown to switch model.</p>
    <p><b>Domain signal $r$ (solid, left axis):</b> at each layer $\\ell$, the RDM over
    core terms is computed, then split into intra-domain and inter-domain distances.
    The Mann-Whitney rank-biserial correlation measures separation:</p>
    <p>$$r_\\ell = 1 - \\frac{2\\,U_\\ell}{n_{\\text{intra}} \\cdot n_{\\text{inter}}}$$</p>
    <p>$r > 0$ means intra-domain distances are systematically smaller than inter-domain
    distances at layer $\\ell$. $r = 0$ means no separation.</p>
    <p><b>RSA convergence $\\rho$ (dashed, right axis):</b> Spearman correlation between
    the RDM at layer $\\ell$ and the RDM at the final layer $L$:</p>
    <p>$$\\rho_\\ell = \\text{Spearman}\\big(\\text{upper\\_tri}(\\text{RDM}_\\ell),\\;
    \\text{upper\\_tri}(\\text{RDM}_L)\\big)$$</p>
    <p>$\\rho_\\ell = 1$ at the final layer by definition. Lower layers with
    $\\rho \\approx 1$ indicate that the relational structure is already formed early.</p>
""",
        "drift": """
    <h3>§3.1.3a — Cosine drift by domain</h3>
    <p>For each term $t$ and each layer transition $\\ell \\to \\ell+1$, the cosine drift
    measures how much the representation changes:</p>
    <p>$$\\text{drift}(t, \\ell) = 1 - \\cos\\big(\\mathbf{h}_t^{(\\ell)},\\; \\mathbf{h}_t^{(\\ell+1)}\\big)$$</p>
    <p>where $\\mathbf{h}_t^{(\\ell)}$ is the pooled (CLS or Mean) hidden state of term $t$
    at layer $\\ell$, L2-normalized.</p>
    <p>Each line is the <b>mean drift across all terms in a domain</b> at each transition.
    High drift at a given transition means that layer is actively transforming the
    representations of terms in that domain.</p>
""",
        "jaccard": """
    <h3>§3.1.3a — Jaccard neighborhood instability by domain</h3>
    <p>For each term $t$ and transition $\\ell \\to \\ell+1$, the $k$-NN sets at the two
    layers are compared using Jaccard distance:</p>
    <p>$$J(t, \\ell) = 1 - \\frac{|\\text{kNN}(t, \\ell) \\cap \\text{kNN}(t, \\ell+1)|}
    {|\\text{kNN}(t, \\ell) \\cup \\text{kNN}(t, \\ell+1)|}$$</p>
    <p>with $k=7$. $\\text{kNN}(t, \\ell)$ is the set of $k$ nearest neighbors of term $t$
    in cosine similarity at layer $\\ell$ (excluding self).</p>
    <p>$J = 0$: identical neighborhoods. $J = 1$: completely different neighborhoods.
    Each line is the mean $J$ across all terms in a domain.</p>
""",
        "heatmap": """
    <h3>§3.1.3a — Drift heatmap (primary model)</h3>
    <p>Each row is a core term, each column is a layer transition $\\ell \\to \\ell+1$.
    Cell color encodes $\\text{drift}(t, \\ell)$ (cosine distance between consecutive
    hidden states). Terms are sorted top-to-bottom by decreasing total drift
    $\\sum_\\ell \\text{drift}(t, \\ell)$.</p>
    <p>Hover to see term name, domain, and exact drift value.</p>
""",
    }

    tab_defs = [
        ("signal_rsa", "§3.1.3b Signal + RSA", "pSignal"),
        ("drift", "§3.1.3a Drift", "pDrift"),
        ("jaccard", "§3.1.3a Jaccard", "pJaccard"),
        ("heatmap", "§3.1.3a Heatmap", "pHeat"),
    ]

    first = True
    for key, title, div_id in tab_defs:
        if key not in plots:
            continue
        active = " active" if first else ""
        tabs_html += (
            f'  <button class="tab-btn{active}" '
            f'onclick="showTab(\'{div_id}\', this)">{title}</button>\n'
        )
        note_html = f'<div class="note">{notes.get(key, "")}</div>' if key in notes else ""
        panels_html += (
            f'<div id="{div_id}" class="panel{active}">'
            f'{note_html}'
            f'<div id="plt_{key}"></div></div>\n'
        )
        plot_entries.append(f"  plt_{key}: {plots[key]}")
        first = False

    plots_js = ",\n".join(plot_entries)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lens III — Layer Stratigraphy</title>
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
  .tab-btn.active {{ background: #0072B2; color: #fff; border-color: #0072B2; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .note {{
    background: #f5f5f5; border-left: 3px solid #0072B2; padding: 10px 14px;
    margin-bottom: 14px; font-size: 0.85rem; line-height: 1.5;
  }}
  .note h3 {{ margin: 0 0 6px 0; font-size: 0.9rem; }}
  .note p {{ margin: 4px 0; }}
</style>
</head>
<body>
<h1>Lens III — Layer Stratigraphy</h1>
<div class="tabs">
{tabs_html}</div>

{panels_html}
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
{plots_js}
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
    """Generate all Lens III figures (PNG + HTML). Called from lens3.main()."""
    png_dir = results_dir / "figures" / "png"
    html_dir = results_dir / "figures" / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    print("\n[viz] Generating PNG figures...")

    if "section_313b" in results:
        p = fig_domain_signal_curve(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_rsa_convergence(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_313a" in results:
        p = fig_drift_by_domain(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_jaccard_by_domain(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_drift_heatmap(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    print("[viz] Generating interactive HTML...")
    html_path = build_html(results, html_dir / "lens3_interactive.html")
    print(f"  {html_path.name}")

    print(f"[viz] Done — {len(generated)} PNG + 1 HTML")
