"""
Lens VI interactive dashboard — SAE feature decomposition.

Produces a self-contained HTML file with:
  Tab 1: Overview — reconstruction quality, training curve, headline metrics
  Tab 2: Domain enrichment — heatmap + bar chart of significant features
  Tab 3: Feature explorer — top features with terms and activations

Usage
-----
    python lens_6_sae/viz.py [--expansion 4] [--k 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.html_style import (  # noqa: E402
    CSS, HEAD_LINKS, LAZY_JS, C_BLUE, C_ORANGE, C_GREEN, C_VERMIL,
    C_PURPLE, C_SKY,
    page_head, tabs_bar, plots_script, format_p,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = RESULTS_DIR / "figures" / "html"

# Domain colors (Okabe-Ito + extensions)
DOMAIN_COLORS = {
    "constitutional": C_BLUE,
    "criminal": C_VERMIL,
    "civil": C_ORANGE,
    "international": C_GREEN,
    "labor_social": C_PURPLE,
    "administrative": C_SKY,
    "procedure": "#999999",
}


def load_results(expansion: int, k: int) -> tuple[dict, dict, list[dict]]:
    """Load training metrics, feature summary, and enrichment results."""
    suffix = f"_x{expansion}_k{k}"
    with open(RESULTS_DIR / f"training_metrics{suffix}.json") as f:
        training = json.load(f)
    with open(RESULTS_DIR / f"feature_summary{suffix}.json") as f:
        summary = json.load(f)
    with open(RESULTS_DIR / f"domain_enrichment{suffix}.json") as f:
        enrichment = json.load(f)
    return training, summary, enrichment


def build_training_plot(training: dict) -> str:
    """Plotly JSON for training loss curve."""
    history = training["history"]
    epochs = [h["epoch"] for h in history]
    recon = [h["recon_loss"] for h in history]
    total = [h["total_loss"] for h in history]
    active = [h["n_active"] for h in history]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Reconstruction loss", "Active features"),
    )
    fig.add_trace(go.Scatter(
        x=epochs, y=recon, name="Recon MSE",
        line=dict(color=C_BLUE, width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=total, name="Total loss",
        line=dict(color=C_ORANGE, width=2, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=active, name="Active features",
        line=dict(color=C_GREEN, width=2), fill="tozeroy",
        fillcolor="rgba(0,158,115,0.1)",
    ), row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_layout(
        height=450, template="plotly_white", showlegend=True,
        legend=dict(x=0.7, y=1.0), margin=dict(t=40, b=40),
    )
    return fig.to_json()


def build_domain_bar(summary: dict) -> str:
    """Plotly JSON for domain feature count bar chart."""
    import plotly.graph_objects as go

    counts = summary["domain_feature_counts"]
    domains = sorted(counts.keys(), key=lambda d: -counts[d])
    values = [counts[d] for d in domains]
    colors = [DOMAIN_COLORS.get(d, "#999") for d in domains]

    fig = go.Figure(go.Bar(
        x=domains, y=values, marker_color=colors,
        text=values, textposition="outside",
    ))
    fig.update_layout(
        title="Significant features per domain (Holm p < 0.05)",
        yaxis_title="Number of features",
        height=350, template="plotly_white",
        margin=dict(t=50, b=40),
    )
    return fig.to_json()


def build_dsi_histogram(enrichment: list[dict]) -> str:
    """Plotly JSON for DSI distribution histogram."""
    import plotly.graph_objects as go

    # Only features with >= 3 labeled terms
    dsi_vals = [r["dsi"] for r in enrichment if r["n_labeled_in_top"] >= 3]

    fig = go.Figure(go.Histogram(
        x=dsi_vals, nbinsx=30, marker_color=C_BLUE,
        marker_line=dict(color="white", width=0.5),
    ))
    fig.update_layout(
        title="Domain Selectivity Index (features with >= 3 labeled terms in top-50)",
        xaxis_title="DSI (1 = maximally selective)",
        yaxis_title="Count",
        height=350, template="plotly_white",
        margin=dict(t=50, b=40),
    )
    return fig.to_json()


def build_feature_heatmap(enrichment: list[dict]) -> str:
    """Plotly heatmap: significant features x domains (activation count)."""
    import plotly.graph_objects as go

    # Select features with at least one significant enrichment
    sig_features = [
        r for r in enrichment
        if any(e.get("significant") for e in r["enrichments"])
    ]
    sig_features.sort(
        key=lambda r: max(
            (e["count_in_top"] for e in r["enrichments"] if e.get("significant")),
            default=0,
        ),
        reverse=True,
    )

    if not sig_features:
        return "{}"

    domains = sorted(set(
        e["domain"] for r in sig_features for e in r["enrichments"]
    ))
    labels = [f"F{r['feature_idx']}" for r in sig_features]

    z = []
    hover = []
    for r in sig_features:
        row = []
        hover_row = []
        enrichment_map = {e["domain"]: e for e in r["enrichments"]}
        for d in domains:
            e = enrichment_map.get(d, {})
            count = e.get("count_in_top", 0)
            row.append(count)
            terms = ", ".join(t["en"] for t in r["top_terms"][:5])
            sig_str = "YES" if e.get("significant") else "no"
            hover_row.append(
                f"Feature {r['feature_idx']}<br>"
                f"Domain: {d}<br>"
                f"Count in top-50: {count}<br>"
                f"Significant: {sig_str}<br>"
                f"Top terms: {terms}"
            )
        z.append(row)
        hover.append(hover_row)

    fig = go.Figure(go.Heatmap(
        z=z, x=domains, y=labels, text=hover, hoverinfo="text",
        colorscale="YlOrRd", colorbar=dict(title="Count"),
    ))
    fig.update_layout(
        title=f"Domain enrichment: {len(sig_features)} significant features",
        height=max(400, 25 * len(sig_features)),
        template="plotly_white",
        margin=dict(l=80, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig.to_json()


def build_top_features_table(enrichment: list[dict], n: int = 30) -> str:
    """HTML table for the top-N features by DSI (with >= 3 labeled terms)."""
    good = [r for r in enrichment if r["n_labeled_in_top"] >= 3]
    good.sort(key=lambda r: r["dsi"], reverse=True)

    rows = []
    for r in good[:n]:
        sig_domains = [e["domain"] for e in r["enrichments"]
                       if e.get("significant")]
        terms = ", ".join(t["en"] for t in r["top_terms"][:8])
        domain_color = DOMAIN_COLORS.get(r["best_domain"], "#999")

        sig_str = ", ".join(sig_domains) if sig_domains else "—"
        rows.append(
            f"<tr>"
            f"<td>{r['feature_idx']}</td>"
            f"<td>{r['dsi']:.3f}</td>"
            f'<td style="color:{domain_color};font-weight:600">{r["best_domain"]}</td>'
            f"<td>{r['best_domain_count']}/{r['n_labeled_in_top']}</td>"
            f"<td>{sig_str}</td>"
            f"<td style='font-size:0.8rem'>{terms}</td>"
            f"</tr>"
        )

    return f"""
    <table class="data">
    <thead><tr>
      <th>Feature</th><th>DSI</th><th>Best domain</th>
      <th>Count</th><th>Significant</th><th>Top terms</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
    </table>"""


def build_html(expansion: int, k: int) -> str:
    """Build complete self-contained HTML dashboard."""
    training, summary, enrichment = load_results(expansion, k)
    q = training["quality"]

    plots = {}
    plots["plt_training"] = build_training_plot(training)
    plots["plt_domain_bar"] = build_domain_bar(summary)
    plots["plt_dsi_hist"] = build_dsi_histogram(enrichment)
    plots["plt_heatmap"] = build_feature_heatmap(enrichment)

    n_sig = summary["n_features_with_significant_enrichment"]
    dict_size = training["config"]["dict_size"]

    tabs = [
        ("tab_overview", "Overview"),
        ("tab_enrichment", "Domain Enrichment"),
        ("tab_features", "Feature Explorer"),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
{page_head("Lens VI — SAE Decomposition")}
<body>
<h1>Lens VI — Sparse Autoencoder Decomposition</h1>
<p class="subtitle">
  TopK SAE on BGE-EN-v1.5 bare embeddings | {dict_size} features, k={k} |
  {training['config']['n_samples']} terms
</p>

{tabs_bar(tabs)}

<!-- Tab 1: Overview -->
<div id="tab_overview" class="panel active">

  <div class="metrics">
    <div class="metric blue">
      <div class="label">Explained Variance</div>
      <div class="value">{q['explained_variance_ratio']:.1%}</div>
    </div>
    <div class="metric green">
      <div class="label">Cosine Similarity</div>
      <div class="value">{q['cosine_sim_mean']:.3f}</div>
    </div>
    <div class="metric orange">
      <div class="label">Features Active</div>
      <div class="value">{q['n_active_features']}</div>
    </div>
    <div class="metric vermil">
      <div class="label">Domain-Selective</div>
      <div class="value">{n_sig}</div>
    </div>
  </div>

  <div class="card">
    <h2>Reconstruction quality</h2>
    <p>The SAE decomposes each 1024-dim BGE-EN-v1.5 embedding into a sparse
    combination of {k} features (out of {dict_size}), capturing
    <b>{q['explained_variance_ratio']:.1%}</b> of the variance with a mean
    cosine similarity of <b>{q['cosine_sim_mean']:.3f}</b>.</p>
    <table class="data">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>MSE</td><td>{q['mse']:.6f}</td></tr>
    <tr><td>Explained Variance Ratio</td><td>{q['explained_variance_ratio']:.4f}</td></tr>
    <tr><td>Cosine similarity</td><td>{q['cosine_sim_mean']:.4f} &pm; {q['cosine_sim_std']:.4f}</td></tr>
    <tr><td>L0 (features/sample)</td><td>{q['l0_mean']:.1f} &pm; {q['l0_std']:.1f}</td></tr>
    <tr><td>Active features</td><td>{q['n_active_features']} / {dict_size}</td></tr>
    <tr><td>Dead features</td><td>{q['n_dead_features']}</td></tr>
    <tr><td>Training time</td><td>{training['training_time_s']:.0f}s ({training['training_time_s']/60:.1f} min)</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Training curve</h2>
    <div id="plt_training" style="height:450px"></div>
  </div>
</div>

<!-- Tab 2: Domain Enrichment -->
<div id="tab_enrichment" class="panel">

  <div class="finding">
    <b>Finding.</b> {n_sig} features show significant domain enrichment
    (Fisher's exact test, p &lt; 0.05, Holm-Bonferroni correction across
    {len(enrichment) * 7:,} tests).
  </div>

  <div class="method">
    For each feature, the top-50 activating terms are collected. Domain
    enrichment is tested with Fisher's exact test (one-sided) for each
    of the 7 legal domains, with Holm-Bonferroni correction for multiple
    comparisons.
  </div>

  <div class="two-col">
    <div class="card">
      <h2>Significant features by domain</h2>
      <div id="plt_domain_bar" style="height:350px"></div>
    </div>
    <div class="card">
      <h2>DSI distribution</h2>
      <div id="plt_dsi_hist" style="height:350px"></div>
    </div>
  </div>

  <div class="card">
    <h2>Enrichment heatmap (significant features only)</h2>
    <div id="plt_heatmap" style="min-height:400px"></div>
  </div>
</div>

<!-- Tab 3: Feature Explorer -->
<div id="tab_features" class="panel">

  <div class="card">
    <h2>Top features by Domain Selectivity Index</h2>
    <p class="note-sm">
      DSI = 1 means all labeled terms in the top-50 come from one domain.
      Only features with &ge; 3 labeled terms shown. "Significant" = Holm p &lt; 0.05.
    </p>
    {build_top_features_table(enrichment, n=30)}
  </div>

  <div class="warning">
    <b>Caveat.</b> Only 430 of 9,472 terms carry domain labels. Features
    activating predominantly for unlabeled terms appear domain-agnostic
    even if they encode meaningful semantic structure.
  </div>
</div>

{plots_script(plots)}
</body>
</html>"""

    return html


def main(args: argparse.Namespace) -> int:
    print("[Lens VI viz] Building dashboard ...")
    html = build_html(args.expansion, args.k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "lens6_interactive.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[Lens VI viz] Saved: {out_path} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--k", type=int, default=32)
    sys.exit(main(parser.parse_args()))
