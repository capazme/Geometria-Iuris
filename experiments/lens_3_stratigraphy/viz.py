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

from shared.html_style import (
    page_head, tabs_bar, plots_script,
    C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_VERMIL, C_PURPLE, C_BLACK,
)

# ---------------------------------------------------------------------------
# Style — Okabe-Ito colorblind-safe palette (imported from shared.html_style)
# ---------------------------------------------------------------------------

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
# Drilldown — HTML table (no Plotly, pure HTML/CSS)
# ---------------------------------------------------------------------------

# Domain color mapping for the NTA table
_DOMAIN_COLORS = {
    "administrative": "#4e79a7",
    "civil": "#f28e2b",
    "constitutional": "#e15759",
    "criminal": "#76b7b2",
    "international": "#59a14f",
    "labor_social": "#edc948",
    "procedure": "#b07aa1",
    "control": "#999999",
}


def _nta_html_block(nta: dict) -> str:
    """Generate HTML block for NTA with a model dropdown selector."""
    models = list(nta.keys())
    if not models:
        return "<p>No NTA data.</p>"

    blocks: list[str] = []

    # Dropdown
    options = "\n".join(
        f'<option value="nta_{i}"{" selected" if i == 0 else ""}>'
        f'{_short(m)}</option>'
        for i, m in enumerate(models)
    )
    blocks.append(
        f'<select id="ntaModelSelect" class="nta-select" '
        f'onchange="ntaSwitchModel(this.value)">\n{options}\n</select>'
    )

    # One div per model, only first visible
    for i, (model_label, dd_data) in enumerate(nta.items()):
        display = "block" if i == 0 else "none"
        k = dd_data["k"]
        blocks.append(
            f'<div id="nta_{i}" class="nta-model" style="display:{display}">'
        )

        for term_name, term_data in dd_data["terms"].items():
            blocks.append(_nta_term_table(term_name, term_data, k))

        blocks.append('</div>')

    return "\n".join(blocks)


def _nta_term_table(term_name: str, term_data: dict, k: int) -> str:
    """Render a single term's NTA trajectory as an HTML table."""
    domain = term_data["domain"]
    zh = term_data.get("zh", "")
    color = _DOMAIN_COLORS.get(domain, "#888")
    parts: list[str] = []

    parts.append(
        f'<h4 style="margin-top:20px">'
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:3px;font-size:0.8em">{domain}</span> '
        f'{term_name}'
        f'<span style="color:#888;font-size:0.8em;margin-left:8px">'
        f'{zh}</span></h4>'
    )

    parts.append(
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:0.82em;margin-bottom:16px">'
    )
    parts.append(
        '<tr style="background:#eee;font-weight:bold">'
        '<td style="padding:4px 8px;border:1px solid #ddd;width:60px">'
        'Layer</td>'
    )
    for r in range(1, k + 1):
        parts.append(
            f'<td style="padding:4px 8px;border:1px solid #ddd;'
            f'text-align:center">#{r}</td>'
        )
    parts.append('</tr>')

    for layer_entry in term_data["layers"]:
        layer = layer_entry["layer"]
        parts.append(
            f'<tr><td style="padding:4px 8px;border:1px solid #ddd;'
            f'font-weight:bold;background:#f9f9f9">L{layer}</td>'
        )
        for nb in layer_entry["neighbors"]:
            nb_color = _DOMAIN_COLORS.get(nb["domain"], "#888")
            is_control = nb.get("tier") == "control"
            entered = nb.get("status") == "entered"
            if entered:
                bg = "#e8f5e9"
            elif is_control:
                bg = "#f0f0f0"
            else:
                bg = "#fff"
            border_style = (
                "2px solid #4caf50" if entered
                else "1px solid #ddd"
            )
            sim_str = f'{nb["sim"]:.3f}'
            name_style = "font-style:italic;color:#666" if is_control else ""
            label_tag = (
                '<span style="background:#ddd;color:#666;padding:0 4px;'
                'border-radius:2px;font-size:0.75em;margin-left:4px">'
                'ctrl</span>'
                if is_control else ""
            )
            parts.append(
                f'<td style="padding:4px 6px;border:{border_style};'
                f'background:{bg};position:relative">'
                f'<span style="color:{nb_color};font-weight:600">'
                f'●</span> <span style="{name_style}">{nb["en"]}</span>'
                f'{label_tag}'
                f'<br><span style="color:#999;font-size:0.85em">'
                f'{nb["domain"]} · {sim_str}</span>'
                f'</td>'
            )
        parts.append('</tr>')

        # Show exited terms
        exited = layer_entry.get("exited", [])
        if exited:
            ex_parts = []
            for e in exited:
                ec = _DOMAIN_COLORS.get(e["domain"], "#888")
                is_ctrl = e.get("tier") == "control"
                style = 'font-style:italic;color:#666' if is_ctrl else ''
                ctrl_tag = (
                    ' <span style="background:#ddd;color:#666;'
                    'padding:0 3px;border-radius:2px;font-size:0.75em">'
                    'ctrl</span>'
                    if is_ctrl else ""
                )
                ex_parts.append(
                    f'<span style="color:{ec}">●</span> '
                    f'<span style="{style}">{e["en"]}</span>{ctrl_tag}'
                )
            ex_names = ", ".join(ex_parts)
            parts.append(
                f'<tr><td style="border:1px solid #ddd"></td>'
                f'<td colspan="{k}" style="padding:2px 8px;'
                f'border:1px solid #ddd;background:#fff3e0;'
                f'font-size:0.85em;color:#bf360c">'
                f'↗ exited: {ex_names}</td></tr>'
            )

    parts.append('</table>')

    # Domain composition summary
    evol = term_data.get("domain_evolution", [])
    if evol:
        parts.append(
            '<div style="margin-bottom:20px;font-size:0.82em">'
            '<b>Domain composition across layers:</b><br>'
        )
        for e in evol:
            layer = e["layer"]
            n_legal = e.get("n_legal", 0)
            n_control = e.get("n_control", 0)
            dom_str = " · ".join(
                f'<span style="color:{_DOMAIN_COLORS.get(d, "#888")}">'
                f'{d}={c}</span>'
                for d, c in sorted(
                    e["domains"].items(), key=lambda x: -x[1]
                )
            )
            tier_str = (
                f' <span style="color:#666">['
                f'legal={n_legal}, ctrl={n_control}]</span>'
            )
            parts.append(f'L{layer}: {dom_str}{tier_str}<br>')
        parts.append('</div>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(results: dict, save_path: Path) -> Path:
    """Build a single self-contained Plotly HTML with tabbed navigation."""
    print("    Building Plotly figures...")
    plots = {}
    extra_html = {}

    has_b = "section_313b" in results
    has_a = "section_313a" in results
    has_nta = "nta" in results

    if has_b:
        plots["signal_rsa"] = _pj_domain_signal_rsa(results)
    if has_a:
        plots["drift"] = _pj_drift_curves(results)
        plots["jaccard"] = _pj_jaccard_curves(results)
        plots["heatmap"] = _pj_drift_heatmap(results)
    if has_nta:
        extra_html["nta"] = _nta_html_block(results["nta"])

    save_path.write_text(
        _html_template(plots, extra_html), encoding="utf-8"
    )
    return save_path


def _html_template(
    plots: dict[str, str],
    extra_html: dict[str, str] | None = None,
) -> str:
    if extra_html is None:
        extra_html = {}

    # Build tab definitions and panels for Plotly tabs
    tab_defs = [
        ("signal_rsa", "§3.1.3b Signal + RSA", "pSignal"),
        ("drift",      "§3.1.3a Drift",         "pDrift"),
        ("jaccard",    "§3.1.3a Jaccard",        "pJaccard"),
        ("heatmap",    "§3.1.3a Heatmap",        "pHeat"),
    ]

    # Full div IDs keyed by tab key (plots_script expects "plt_signal_rsa", etc.)
    plotly_plots: dict[str, str] = {}
    panels_html = ""
    tabs_list: list[tuple[str, str]] = []  # (panel_id, label)
    first = True

    for key, title, div_id in tab_defs:
        if key not in plots:
            continue
        tabs_list.append((div_id, title))
        active = " active" if first else ""

        question_content = _panel_questions.get(key, "")
        question_html = (
            f'<div class="question">{question_content}</div>'
            if question_content else ""
        )
        method_content = _panel_methods.get(key, "")
        method_html = (
            f'<div class="card">{method_content}</div>'
            if method_content else ""
        )

        full_div_id = f"plt_{key}"
        plotly_plots[full_div_id] = plots[key]

        panels_html += (
            f'<div id="{div_id}" class="panel{active}">'
            f'{question_html}'
            f'{method_html}'
            f'<div id="{full_div_id}"></div>'
            f'</div>\n'
        )
        first = False

    # NTA tab (pure HTML tables + custom JS)
    nta_js = ""
    if "nta" in extra_html:
        active = " active" if not plots else ""
        tabs_list.append(("pNTA", "§3.1.3c NTA"))
        nta_question = (
            '<b>\u00a73.1.3c \u2014 Neighborhood Trajectory Analysis (NTA).</b> '
            'While the previous charts show aggregate statistics (averages across many terms), '
            'this table zooms in on a small number of individually selected terms \u2014 chosen '
            'because they are polysemous (they carry different meanings in different branches '
            'of law), because they have particular cross-tradition relevance, or because they '
            'exhibited unusual behaviour in the aggregate metrics. '
            'For each selected term, the table lists the exact identity of its $k = 7$ nearest '
            'neighbors at several sampled layers (not every layer is shown, only a representative '
            'subset). The <b>neighbor pool</b> from which these 7 closest terms are drawn consists '
            'of 397 core legal terms (covering 7 branches of law) and 100 non-legal control words '
            'from the Swadesh-100 list (basic human-experience terms like \"water\", \"fire\", '
            '\"mother\", used as a baseline). As the model processes through its layers, the '
            'composition of each term\u2019s neighborhood may change: some neighbors enter the top-7 '
            'set, others exit. By reading the table from top to bottom (earlier layers to later '
            'layers), one can observe whether the term\u2019s closest associates shift from one branch '
            'of law to another, or from non-legal control terms to legal terms, or vice versa.'
        )
        nta_legend = """
<div class="card">
<h2>Legend \u2014 How to read the NTA tables</h2>
<p>Each table shows one selected term. The rows represent sampled layers (from earlier layers at the top
to later layers at the bottom). The columns show the 7 nearest neighbors at that layer, ranked from
closest (left, #1) to seventh-closest (right, #7). Each neighbor cell displays the neighbor\u2019s name,
its branch of law (domain), and the cosine similarity score (a number between 0 and 1 measuring how
close the two vectors are; higher means more similar).</p>

<p><span style="background:#e8f5e9;padding:1px 6px;border:2px solid #4caf50;border-radius:3px">Green cell with green border</span>
= a neighbor that <b>entered</b> the top-7 set at this layer. This term was <em>not</em> among the 7
nearest neighbors at the previous sampled layer but has now become close enough to appear. Its arrival
may reflect the model reorganising its representation of the focal term at this depth.</p>

<p><span style="background:#f0f0f0;padding:1px 6px;border-radius:3px;font-style:italic;color:#666">Gray italic text</span>
<span style="background:#ddd;color:#666;padding:0 4px;border-radius:2px;font-size:0.85em">ctrl</span>
= a <b>control term</b> (a non-legal word from the Swadesh-100 basic vocabulary list, such as
\"water\", \"fire\", or \"night\"). Control terms serve as a baseline: if a legal term\u2019s nearest
neighbors include many control words, it means the model has not yet developed a specifically
legal representation of that term at that layer.</p>

<p><span style="background:#fff3e0;padding:1px 6px;border-radius:3px;color:#bf360c">&#8599; exited</span>
= a neighbor that <b>left</b> the top-7 set between sampled layers. This term was among the 7 nearest
neighbors at the previous sampled layer but has now moved farther away. Exited terms appear in an
orange row beneath the main row, listing all terms that dropped out.</p>

<p>The <b>colored dot</b> &#9679; next to each neighbor\u2019s name marks the <b>branch of law</b> (domain)
that neighbor belongs to. Each domain has a consistent colour throughout all tables (e.g., criminal
law is always the same colour). This makes it easy to scan visually: a row dominated by dots of one
colour means most neighbors at that layer come from the same branch of law.</p>

<p><b>Domain composition</b> (shown below each table): for each sampled layer, this summary counts
how many of the 7 nearest neighbors belong to each branch of law, and how many are legal terms versus
control terms. The notation <code>[legal=N, ctrl=M]</code> indicates that N of the 7 neighbors are
core legal terms and M are Swadesh-100 control words (N + M = 7). This allows tracking whether a term\u2019s
neighborhood shifts from mixed legal/non-legal at early layers to predominantly legal at later layers,
or whether certain branches of law gain or lose representation across layers.</p>
</div>"""
        panels_html += (
            f'<div id="pNTA" class="panel{active}">'
            f'<div class="question">{nta_question}</div>'
            f'{nta_legend}'
            f'{extra_html["nta"]}'
            f'</div>\n'
        )
        # Extract the <script> block from nta_html_block (it is appended at the end)
        nta_js = (
            "function ntaSwitchModel(id) {\n"
            "  document.querySelectorAll('.nta-model').forEach(\n"
            "    el => el.style.display = 'none');\n"
            "  document.getElementById(id).style.display = 'block';\n"
            "}\n"
        )

    tabs_bar_html = tabs_bar(tabs_list)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        + page_head("Lens III \u2014 Layer Stratigraphy")
        + "\n<body>\n"
        "<h1>Lens III \u2014 Layer Stratigraphy</h1>\n"
        '<p class="subtitle">'
        '\u00a7\u00a73.1.3a\u2013c \u2014 '
        'Layer-by-layer analysis of 6 embedding models (12\u201324 layers each). '
        'Modern text-embedding models transform each input term through a series of successive computational stages called layers; '
        'at each layer the model refines its internal numerical representation of the term, and only the final layer\u2019s output is normally used as the term\u2019s embedding. '
        'This lens opens the black box: instead of examining only the final output, it extracts the representation at every intermediate layer and measures how it changes. '
        'Metrics: cosine drift between consecutive layers (how much a term\u2019s vector moves at each step), '
        'k-nearest-neighbor Jaccard instability (whether a term\u2019s closest associates change identity from one layer to the next), '
        'domain signal (rank-biserial r: do terms from the same branch of law cluster together at each depth?), '
        'RSA convergence (Spearman \u03c1 vs. final layer: how early does the overall distance pattern stabilise?), '
        'and Neighborhood Trajectory Analysis (NTA) for selected terms (tracking the exact identity of each term\u2019s nearest neighbors across layers).'
        '</p>\n'
        + tabs_bar_html + "\n\n"
        + panels_html
        + "<script>\n"
        + nta_js
        + "</script>\n"
        + plots_script(plotly_plots)
        + "\n</body>\n</html>"
    )


# Per-tab question boxes (what does this visualization answer?)
_panel_questions: dict[str, str] = {
    "signal_rsa": (
        "<b>Domain signal and RSA convergence per layer.</b> "
        "Embedding models process text through a series of successive computational stages called "
        "\"layers\" \u2014 typically between 12 and 24, depending on the model. Each layer refines the "
        "model\u2019s internal numerical representation of a term, and only the very last layer\u2019s output "
        "is normally used as the term\u2019s final embedding. This chart tracks two distinct quantities "
        "across all layers of each model (use the dropdown in the upper-right corner to switch between models). "
        "<br><br>"
        "The <b>solid line</b> (left axis) shows the <b>domain signal</b> at each layer: it answers the question "
        "\"at this depth in the model, do terms belonging to the same branch of law (e.g., two criminal-law terms) "
        "tend to have more similar representations than terms from different branches?\" "
        "The metric used is the rank-biserial $r$ from a Mann-Whitney $U$ test. A value of $r > 0$ means that "
        "same-domain pairs are, on average, closer together than cross-domain pairs; $r = 0$ means no detectable "
        "separation; $r < 0$ would mean same-domain pairs are actually farther apart. "
        "<br><br>"
        "The <b>dashed line</b> (right axis) shows the <b>RSA convergence</b>: it answers the question \"how similar "
        "is the full pattern of pairwise distances at layer $\\ell$ to the pattern at the final layer?\" "
        "This is measured by the Spearman rank correlation $\\rho$ between the two distance matrices. "
        "A value of $\\rho = 1$ means the rank ordering of all 78,606 pairwise distances is identical to the final "
        "layer\u2019s ordering; lower values mean the distance pattern at that layer is still different from the final output."
    ),
    "drift": (
        "<b>Cosine drift per domain and layer transition.</b> "
        "Between any two consecutive layers in an embedding model, each term\u2019s internal numerical representation "
        "(called a \"hidden state\" or \"vector\") changes. Cosine drift quantifies how large that change is: "
        "it measures the angular difference between the vector at layer $\\ell$ and the vector at layer $\\ell+1$. "
        "A drift value of 0 means the representation did not change at all; values approaching 1 mean a large change "
        "(the two vectors point in very different directions); values above 1 are theoretically possible but rare in practice. "
        "<br><br>"
        "Each curve in this chart shows the <b>average drift</b> across all terms belonging to one branch of law "
        "(e.g., all criminal-law terms, all constitutional-law terms). This allows comparison of whether certain "
        "legal domains undergo more representational change at particular layers. "
        "The x-axis spans all layer transitions: for a 24-layer model, these are "
        "layer 0\u21921, 1\u21922, 2\u21923, \u2026, 23\u219224. "
        "The y-axis shows the mean drift value. Use the dropdown to switch between models."
    ),
    "jaccard": (
        "<b>Neighborhood instability per domain and layer transition.</b> "
        "For a given term at a given layer, the <b>$k$-nearest neighbors</b> (here $k = 7$) are the 7 other terms "
        "whose numerical representations (vectors) are most similar to it at that layer. For example, at a certain "
        "layer the 7 nearest neighbors of \"negligence\" might be {manslaughter, liability, damages, recklessness, tort, duty, fault}. "
        "The <b>Jaccard distance</b> compares two such neighbor sets across consecutive layers: it counts how many "
        "neighbors are shared and divides by the total number of distinct neighbors appearing in either set. "
        "Concretely: if 5 of the 7 neighbors remain the same between layer $\\ell$ and layer $\\ell+1$, the intersection "
        "has 5 terms and the union has 9 terms (7 + 7 \u2212 5), so the overlap is 5/9, and the Jaccard distance is "
        "$J = 1 - 5/9 \\approx 0.44$. A Jaccard distance of $J = 0$ means the neighbor set is completely identical "
        "across the transition; $J = 1$ means the two sets are entirely disjoint (no shared neighbors at all). "
        "<br><br>"
        "Each curve shows the <b>mean Jaccard distance</b> across all terms in one branch of law. "
        "A high value at a particular transition means that, on average, the identity of each term\u2019s closest "
        "associates is changing rapidly at that layer \u2014 the \"neighborhood\" is being reorganised."
    ),
    "heatmap": (
        "<b>Per-term drift heatmap.</b> "
        "This heatmap shows the cosine drift for every individual term at every layer transition, laid out as a "
        "two-dimensional colour-coded grid. Each <b>row</b> is one of the 397 core legal terms; each <b>column</b> "
        "is one layer transition (e.g., layer 0\u21921, 1\u21922, etc.). The colour intensity of each cell encodes "
        "the magnitude of the drift at that transition: darker or warmer colours mean larger drift (more change), "
        "lighter colours mean smaller drift (less change). "
        "<br><br>"
        "The terms are sorted top-to-bottom by decreasing <b>total drift</b> $\\sum_\\ell \\text{{drift}}(t, \\ell)$: "
        "terms at the top of the heatmap undergo the most total representational change across all layers, while "
        "terms at the bottom are the most stable. "
        "Hover over any cell to see the exact term name and drift value. Use the dropdown to compare across models."
    ),
}

# Per-tab method cards (formula + description)
_panel_methods: dict[str, str] = {
    "signal_rsa": (
        "<h2>Definitions</h2>"
        "<p><b>Domain signal $r_\\ell$</b> (solid line, left axis): at each layer $\\ell$, "
        "the model\u2019s internal representations of all 397 core legal terms are extracted. "
        "From these representations, a 397\u00d7397 distance matrix is computed: each cell "
        "records how different two terms\u2019 vectors are at that layer. This matrix is then split "
        "into two groups of pairs: <em>intra-domain</em> pairs (two terms from the same branch of "
        "law, e.g., two criminal-law terms) and <em>inter-domain</em> pairs (two terms from different "
        "branches). The Mann-Whitney rank-biserial $r$ measures the degree of separation between these "
        "two groups:</p>"
        "<p>$$r_\\ell = 1 - \\frac{2\\,U_\\ell}{n_{\\text{intra}} \\cdot n_{\\text{inter}}}$$</p>"
        "<p>In plain language: $r > 0$ means that intra-domain distances tend to be <em>smaller</em> "
        "than inter-domain distances \u2014 terms from the same branch of law are, on average, closer "
        "together in the model\u2019s representational space. $r = 0$ means there is no detectable "
        "difference between same-domain and cross-domain distances. $r < 0$ would mean that same-domain "
        "terms are actually <em>farther apart</em> than cross-domain terms.</p>"
        "<h3>RSA convergence $\\rho_\\ell$</h3>"
        "<p>(dashed line, right axis): the Spearman rank correlation between the distance matrix at "
        "layer $\\ell$ and the distance matrix at the final layer $L$:</p>"
        "<p>$$\\rho_\\ell = \\text{Spearman}\\!\\left(\\text{upper\\_tri}(\\text{RDM}_\\ell),\\; "
        "\\text{upper\\_tri}(\\text{RDM}_L)\\right)$$</p>"
        "<p>In plain language: Spearman\u2019s $\\rho$ compares the <em>rank ordering</em> of all 78,606 "
        "pairwise distances at layer $\\ell$ with the rank ordering at the final layer. A value of "
        "$\\rho = 1$ means the two rank orderings are identical \u2014 every pair of terms is in the exact "
        "same relative position as in the final output. Lower values mean the distance pattern at that "
        "layer has not yet converged to its final form. At the final layer itself, $\\rho_L = 1$ by "
        "definition (any matrix is perfectly correlated with itself).</p>"
    ),
    "drift": (
        "<h2>Definition</h2>"
        "<p>For each term $t$ and layer transition $\\ell \\to \\ell+1$:</p>"
        "<p>$$\\text{drift}(t, \\ell) = 1 - \\cos\\!\\left(\\mathbf{h}_t^{(\\ell)},\\; "
        "\\mathbf{h}_t^{(\\ell+1)}\\right)$$</p>"
        "<p>Here, $\\mathbf{h}_t^{(\\ell)}$ is the model\u2019s internal representation (a high-dimensional "
        "numerical vector) of term $t$ at layer $\\ell$. The cosine of the angle between two vectors "
        "measures their <em>directional similarity</em>: a cosine of 1 means the two vectors point in "
        "exactly the same direction (identical representation), a cosine of 0 means they are perpendicular "
        "(unrelated directions), and a cosine of \u22121 means they point in opposite directions. "
        "Drift is defined as 1 minus this cosine, so it ranges from 0 (no change at all between layers) "
        "to 2 (maximally different directions), though values above 1 are rare in practice. "
        "In short: drift quantifies <em>how much a term\u2019s numerical representation moves</em> when "
        "the model transitions from one computational stage to the next.</p>"
        "<p>Each curve is the mean drift across all terms belonging to one branch of law. "
        "The colour legend identifies the branch.</p>"
    ),
    "jaccard": (
        "<h2>Definition</h2>"
        "<p>For each term $t$ and layer transition $\\ell \\to \\ell+1$, the $k$-nearest-neighbor "
        "sets ($k=7$) are compared:</p>"
        "<p>$$J(t, \\ell) = 1 - \\frac{|\\text{kNN}(t, \\ell) \\cap \\text{kNN}(t, \\ell+1)|}"
        "{|\\text{kNN}(t, \\ell) \\cup \\text{kNN}(t, \\ell+1)|}$$</p>"
        "<p>A concrete example: suppose that at layer 5, the 7 nearest neighbors of "
        "\"negligence\" are {manslaughter, liability, damages, recklessness, tort, duty, fault}. "
        "At layer 6, they become {liability, damages, tort, duty, contributory, breach, carelessness}. "
        "The <em>intersection</em> (terms present in both sets) has 4 terms: liability, damages, tort, duty. "
        "The <em>union</em> (all distinct terms appearing in either set) has 10 terms. Therefore the "
        "Jaccard similarity is 4/10 = 0.4, and the Jaccard <em>distance</em> is $J = 1 - 4/10 = 0.6$.</p>"
        "<p>Each curve is the mean $J$ across all terms in one branch of law. Higher values at a "
        "particular transition mean that, on average, the identity of each term\u2019s closest associates "
        "is changing more rapidly at that layer.</p>"
    ),
    "heatmap": (
        "<h2>Reading the heatmap</h2>"
        "<p>Each row is one of 397 core legal terms. Each column is one layer transition "
        "$\\ell \\to \\ell+1$ (e.g., layer 0\u21921, 1\u21922, etc.). The colour of each cell encodes "
        "the cosine drift $\\text{drift}(t, \\ell)$ at that transition for that term: warmer, more "
        "intense colours correspond to larger drift values (more representational change), while "
        "lighter colours correspond to smaller drift values (less change).</p>"
        "<p>Terms are sorted top-to-bottom by decreasing <b>total drift</b> "
        "$\\sum_\\ell \\text{drift}(t, \\ell)$, which is the sum of all per-transition drift values "
        "across the entire depth of the model. Terms at the top of the heatmap are those whose "
        "representations undergo the most total change as the model processes them through all layers; "
        "terms at the bottom are the most stable. "
        "Use the model dropdown to compare across models. Hover over any cell to see the exact "
        "term name and drift value.</p>"
    ),
}


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
