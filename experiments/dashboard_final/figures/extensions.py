"""Plotly figures for the 9 extensions A-Z.

Each `fig_*(...)` function returns a `{"data": [...], "layout": {...}}` dict.

Y caveat is rendered as an HTML table by `shared_ui.data_table` and a
`number_callout`, not as a Plotly figure — see `pages/robustness_caveats.py`.

Figures provided:
    fig_D_robustness_curve     Δρ_sym attested vs %bg (with 95% CI ribbon)
    fig_X_control_curve        Δρ_sym bare vs %control (dual of D)
    fig_H_k_saturation         ρ_cross attested as a function of K
    fig_F_confidence_bars      Δρ_sym per confidence stratum
    fig_G_false_friends_scatter cross-encoder cosine vs bilingual cosine
    fig_Z_tier_medians         per-model core-core / core-bg / core-control
    fig_A_bg_domain_distribution domain assignment of the 9 045 bg terms
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from shared_ui import (  # noqa: E402
    PLOT_COLORS, PLOTLY_LAYOUT_DEFAULTS, PLOTLY_AXIS_DEFAULTS,
)


def _layout(**overrides) -> dict:
    import copy
    L = copy.deepcopy(PLOTLY_LAYOUT_DEFAULTS)
    L.setdefault("xaxis", {}).update(PLOTLY_AXIS_DEFAULTS)
    L.setdefault("yaxis", {}).update(PLOTLY_AXIS_DEFAULTS)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(L.get(k), dict):
            L[k] = {**L[k], **v}
        else:
            L[k] = v
    return L


# --------------------------------------------------------------------------
# D — Δρ_sym attested vs %background injection (0 → 75%)

def fig_D_robustness_curve(D_table: list[dict]) -> dict:
    """Curve with 95% CI ribbon: stability under background-term injection.

    D_table from `loader_extensions.d_robustness_table()`.
    """
    xs = [r["p_bg"] * 100 for r in D_table]
    ys = [r["mean_delta_sym"] for r in D_table]
    lo = [r["ci_low"] for r in D_table]
    hi = [r["ci_high"] for r in D_table]

    color = PLOT_COLORS["accent_dark"]
    fillcolor = "rgba(138,109,59,0.18)"
    return {
        "data": [
            {"type": "scatter", "mode": "lines",
             "x": xs + xs[::-1],
             "y": hi + lo[::-1],
             "fill": "toself", "fillcolor": fillcolor,
             "line": {"color": "rgba(0,0,0,0)"},
             "name": "95% CI", "hoverinfo": "skip", "showlegend": True},
            {"type": "scatter", "mode": "lines+markers",
             "x": xs, "y": ys,
             "line": {"color": color, "width": 2.2},
             "marker": {"size": 9, "color": color,
                        "line": {"color": "#222", "width": 0.6}},
             "name": "Δρ_sym attested",
             "hovertemplate": "%{x:.0f}% bg<br>Δρ_sym = %{y:.3f}<extra></extra>"},
        ],
        "layout": _layout(
            title="D — Δρ_sym attested vs background injection",
            xaxis={"title": "% background terms in pool", "ticksuffix": "%",
                    "range": [-3, 80]},
            yaxis={"title": "Δρ_sym (within − cross)", "range": [0.45, 0.65]},
            shapes=[{
                "type": "line", "xref": "x", "yref": "y",
                "x0": -3, "x1": 80, "y0": 0.4, "y1": 0.4,
                "line": {"color": "#999", "width": 0.7, "dash": "dot"},
            }],
            annotations=[
                {"x": 0, "y": ys[0] + 0.012, "xref": "x", "yref": "y",
                 "text": f"baseline {ys[0]:.3f}", "showarrow": False,
                 "font": {"size": 10, "color": color}},
                {"x": xs[-1], "y": ys[-1] + 0.012, "xref": "x", "yref": "y",
                 "text": f"{ys[-1]:.3f} at 75%", "showarrow": False,
                 "font": {"size": 10, "color": color}},
            ],
            height=420,
            legend={"orientation": "h", "yanchor": "bottom",
                    "y": -0.22, "x": 0.5, "xanchor": "center",
                    "font": {"size": 11}},
        ),
    }


# --------------------------------------------------------------------------
# X — Dual of D: Δρ_sym bare vs %control injection

def fig_X_control_curve(X_table: list[dict]) -> dict:
    xs = [r["p_control"] * 100 for r in X_table]
    ys = [r["mean_delta_sym"] for r in X_table]
    lo = [r["ci_low"] for r in X_table]
    hi = [r["ci_high"] for r in X_table]
    color = PLOT_COLORS["control"]
    fillcolor = "rgba(127,127,127,0.18)"
    return {
        "data": [
            {"type": "scatter", "mode": "lines",
             "x": xs + xs[::-1],
             "y": hi + lo[::-1],
             "fill": "toself", "fillcolor": fillcolor,
             "line": {"color": "rgba(0,0,0,0)"},
             "name": "95% CI", "hoverinfo": "skip", "showlegend": True},
            {"type": "scatter", "mode": "lines+markers",
             "x": xs, "y": ys,
             "line": {"color": color, "width": 2.2},
             "marker": {"size": 8, "color": color,
                        "line": {"color": "#222", "width": 0.6}},
             "name": "Δρ_sym bare",
             "hovertemplate": "%{x:.0f}% control<br>Δρ_sym = %{y:.3f}<extra></extra>"},
        ],
        "layout": _layout(
            title="X — Δρ_sym bare vs control injection (dual of D)",
            xaxis={"title": "% control terms in pool", "ticksuffix": "%"},
            yaxis={"title": "Δρ_sym (within − cross, bare)",
                    "range": [0.18, 0.28]},
            height=400,
            legend={"orientation": "h", "yanchor": "bottom",
                    "y": -0.22, "x": 0.5, "xanchor": "center",
                    "font": {"size": 11}},
        ),
    }


# --------------------------------------------------------------------------
# H — ρ_cross attested as a function of minimum K

def fig_H_k_saturation(H_table: list[dict]) -> dict:
    xs = [r["K"] for r in H_table]
    ys = [r["mean_rho_cross"] for r in H_table]
    color = PLOT_COLORS["accent_dark"]
    bar_colors = [
        PLOT_COLORS["sinic"] if y < 0
        else (PLOT_COLORS["warn"] if y < 0.15 else PLOT_COLORS["bilingual"])
        for y in ys
    ]
    text = [f"{y:+.3f}" for y in ys]
    return {
        "data": [{
            "type": "bar",
            "x": xs, "y": ys,
            "marker": {"color": bar_colors,
                        "line": {"color": "#222", "width": 0.4}},
            "text": text, "textposition": "outside",
            "hovertemplate": "K=%{x}<br>ρ̄_cross = %{y:.3f}<extra></extra>",
            "name": "ρ̄_cross attested",
        }],
        "layout": _layout(
            title="H — Minimum-K saturation of ρ̄_cross attested",
            xaxis={"title": "minimum K (attestations per term)",
                    "type": "category"},
            yaxis={"title": "ρ̄_cross attested",
                    "range": [-0.2, 0.32],
                    "zeroline": True, "zerolinecolor": "#999",
                    "zerolinewidth": 1},
            shapes=[{
                "type": "rect", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1, "y0": 0.13, "y1": 0.17,
                "fillcolor": "rgba(176,141,87,0.10)", "line_width": 0,
                "layer": "below",
            }],
            annotations=[
                {"x": 1, "y": 0.15, "xref": "paper", "yref": "y",
                 "xanchor": "right", "yanchor": "middle",
                 "text": "K≥4 saturation band", "showarrow": False,
                 "font": {"size": 10, "color": color}},
            ],
            height=420,
            showlegend=False,
        ),
    }


# --------------------------------------------------------------------------
# F — Confidence-stratified Δρ_sym

def fig_F_confidence_bars(F_table: list[dict]) -> dict:
    labels = [r["stratum"] for r in F_table]
    ys = [r["mean_delta_sym"] for r in F_table]
    err_arrays = []
    for r in F_table:
        if r["ci_low"] is None:
            err_arrays.append((0.0, 0.0))
        else:
            err_arrays.append((r["ci_high"] - r["mean_delta_sym"],
                                r["mean_delta_sym"] - r["ci_low"]))
    hi_err = [e[0] for e in err_arrays]
    lo_err = [e[1] for e in err_arrays]

    colors = []
    for r in F_table:
        if r["stratum"].startswith("baseline"):
            colors.append(PLOT_COLORS["accent_dark"])
        elif "high" in r["stratum"]:
            colors.append(PLOT_COLORS["weird"])
        elif "low" in r["stratum"]:
            colors.append(PLOT_COLORS["sinic"])
        else:
            colors.append(PLOT_COLORS["control"])

    return {
        "data": [{
            "type": "bar",
            "x": labels, "y": ys,
            "marker": {"color": colors,
                        "line": {"color": "#222", "width": 0.4}},
            "error_y": {"type": "data", "symmetric": False,
                         "array": hi_err, "arrayminus": lo_err,
                         "color": "#444", "thickness": 1.2, "width": 4},
            "text": [f"{y:.3f}" for y in ys], "textposition": "outside",
            "hoverinfo": "y+text",
            "name": "Δρ_sym",
        }],
        "layout": _layout(
            title="F — Δρ_sym by injection stratum (n=20 replicates)",
            xaxis={"title": "", "tickangle": -15},
            yaxis={"title": "Δρ_sym (within − cross)", "range": [0.49, 0.61]},
            height=380,
            showlegend=False,
        ),
    }


# --------------------------------------------------------------------------
# G — Same-lemma divergence: cross-encoder cosine vs bilingual cosine

def fig_G_false_friends_scatter(G_rows: list[dict]) -> dict:
    """Scatter of (cos_cross, cos_bilingual). Highlight the top-N most
    divergent (largest bilingual − cross gap) with labels.
    """
    if not G_rows:
        return {"data": [], "layout": _layout(title="G — (no rows)")}

    xs = [r["cos_cross"] for r in G_rows]
    ys = [r["cos_bilingual"] for r in G_rows]
    text = [f"{r['en']} / {r['zh']}<br>cross = {r['cos_cross']:+.3f}"
             f"<br>bilingual = {r['cos_bilingual']:+.3f}" for r in G_rows]

    return {
        "data": [
            # Diagonal reference y=x.
            {"type": "scatter", "mode": "lines",
             "x": [-0.4, 1.0], "y": [-0.4, 1.0],
             "line": {"color": "#999", "width": 0.8, "dash": "dot"},
             "name": "y = x", "hoverinfo": "skip", "showlegend": False},
            {"type": "scatter", "mode": "markers+text",
             "x": xs, "y": ys,
             "text": [r["en"] for r in G_rows],
             "textposition": "top center", "textfont": {"size": 9},
             "marker": {"size": 11, "color": PLOT_COLORS["sinic"],
                         "line": {"color": "#222", "width": 0.6},
                         "opacity": 0.85},
             "hovertext": text, "hoverinfo": "text",
             "name": "same-lemma divergence"},
        ],
        "layout": _layout(
            title="G — Same-lemma terms: cross-encoder vs bilingual cosine",
            xaxis={"title": "cosine · BGE-EN-large × BGE-ZH-large",
                    "range": [-0.4, 1.0],
                    "zeroline": True, "zerolinecolor": "#999"},
            yaxis={"title": "cosine · BGE-M3-EN × BGE-M3-ZH",
                    "range": [-0.4, 1.0],
                    "zeroline": True, "zerolinecolor": "#999"},
            height=460,
            showlegend=False,
        ),
    }


# --------------------------------------------------------------------------
# Z — Tier hierarchy: per-model medians on three pair populations

def fig_Z_tier_medians(Z_table: list[dict]) -> dict:
    """Grouped bars: per-model median(core-core / core-bg / core-control).

    Monotonic-hierarchy models highlighted with a check mark.
    """
    models = [r["model"] for r in Z_table]
    cc = [r["median_core_core"] for r in Z_table]
    cb = [r["median_core_bg"] for r in Z_table]
    cl = [r["median_core_ctrl"] for r in Z_table]
    mono_text = ["✓" if r["monotonic"] else "" for r in Z_table]
    return {
        "data": [
            {"type": "bar", "name": "core × core",
             "x": models, "y": cc,
             "marker": {"color": PLOT_COLORS["accent_dark"]}},
            {"type": "bar", "name": "core × bg",
             "x": models, "y": cb,
             "marker": {"color": PLOT_COLORS["cross"]}},
            {"type": "bar", "name": "core × control",
             "x": models, "y": cl,
             "marker": {"color": PLOT_COLORS["control"]}},
            {"type": "scatter", "mode": "text",
             "x": models, "y": [max(cc[i], cb[i], cl[i]) + 0.04
                                 for i in range(len(models))],
             "text": mono_text,
             "textfont": {"size": 16, "color": PLOT_COLORS["good"]},
             "hoverinfo": "skip", "showlegend": False, "name": "monotonic"},
        ],
        "layout": _layout(
            title="Z — Tier hierarchy: median cosine distance per population",
            xaxis={"title": "", "tickangle": -25, "automargin": True},
            yaxis={"title": "median cosine distance", "range": [0, 1.0]},
            barmode="group",
            height=460,
            margin={"l": 60, "r": 25, "t": 50, "b": 130},
            legend={"orientation": "h", "yanchor": "bottom",
                    "y": -0.35, "x": 0, "font": {"size": 11}},
        ),
    }


# --------------------------------------------------------------------------
# A — Background k-NN domain assignment

def fig_A_bg_domain_distribution(A_raw: dict) -> dict:
    """Bar chart of domain distribution for the 9 045 bg terms."""
    dist = A_raw["meta"].get("domain_distribution", {})
    rows = sorted(dist.items(), key=lambda r: -r[1])
    domains = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    return {
        "data": [{
            "type": "bar",
            "x": domains, "y": counts,
            "marker": {"color": PLOT_COLORS["accent"]},
            "text": [f"{c:,}" for c in counts], "textposition": "outside",
            "hovertemplate": "%{x}<br>%{y:,} bg terms<extra></extra>",
        }],
        "layout": _layout(
            title=f"A — Background k-NN domain assignment (k=7, n={A_raw['meta']['n_bg']:,})",
            xaxis={"title": "", "tickangle": -20, "automargin": True},
            yaxis={"title": "number of background terms"},
            height=380,
            showlegend=False,
        ),
    }


# --------------------------------------------------------------------------
# Robustness page — bilingual control forest (C1)

def fig_bilingual_control_forest(s313: dict) -> dict:
    """Four group means (within-W, within-S, cross-tradition, bilingual)
    as a horizontal forest plot. Shows where the bilingual control sits
    relative to the two within-tradition floors and the cross-tradition
    mean.
    """
    summary = s313["attested"]["summary"]
    groups = [
        ("Within Western-trained (3 pairs)",
          float(summary.get("mean_rho_within_weird", 0)),
          PLOT_COLORS["weird"]),
        ("Within Chinese-trained (3 pairs)",
          float(summary.get("mean_rho_within_sinic", 0)),
          PLOT_COLORS["sinic"]),
        ("Cross-tradition (9 pairs)",
          float(summary.get("mean_rho_cross_tradition", 0)),
          PLOT_COLORS["cross"]),
        ("Bilingual control (2 pairs)",
          float(summary.get("mean_rho_within_bilingual", 0)),
          PLOT_COLORS["bilingual"]),
    ]
    labels = [g[0] for g in groups]
    rhos = [g[1] for g in groups]
    colors = [g[2] for g in groups]
    return {
        "data": [{
            "type": "bar", "orientation": "h",
            "x": rhos, "y": labels,
            "marker": {"color": colors,
                        "line": {"color": "#333", "width": 0.5}},
            "text": [f"{r:.3f}" for r in rhos],
            "textposition": "outside",
            "textfont": {"size": 12, "color": "#222"},
            "hovertemplate": "%{y}<br>mean ρ = %{x:.3f}<extra></extra>",
        }],
        "layout": _layout(
            title="The bilingual control next to the within-tradition "
                  "floors and the cross-tradition band (attested)",
            xaxis={"title": "mean Spearman ρ", "range": [0, 1.0],
                   "gridcolor": "#f0f0f0", "showgrid": True,
                   "zeroline": True, "zerolinecolor": "#b08d57"},
            yaxis={"title": "", "automargin": True,
                   "tickfont": {"size": 11}},
            margin={"l": 280, "r": 50, "t": 70, "b": 50},
            height=320,
        ),
    }


# --------------------------------------------------------------------------
# Robustness page — FreeLaw-EN failure visual (C2)

def fig_freelaw_failure_bars(s311_legal_vs_control: dict) -> dict:
    """Bar chart of legal-vs-control rank-biserial r per model, with
    FreeLaw-EN and Qwen3-0.6B-EN highlighted in muted grey.
    """
    from data.loader_31 import ALL_MODELS_ORDERED, model_group
    rows = [(m, float(s311_legal_vs_control[m]["effect_r"]),
              float(s311_legal_vs_control[m]["p_value"]))
            for m in ALL_MODELS_ORDERED
            if m in s311_legal_vs_control]
    failures = {"FreeLaw-EN", "Qwen3-0.6B-EN"}
    models = [r[0] for r in rows]
    rs = [r[1] for r in rows]
    pvals = [r[2] for r in rows]
    group_color = {
        "weird":     PLOT_COLORS["weird"],
        "sinic":     PLOT_COLORS["sinic"],
        "bilingual": PLOT_COLORS["bilingual"],
    }
    colors = [
        PLOT_COLORS["control"] if (m in failures or r < 0)
        else group_color.get(model_group(m), "#999")
        for m, r in zip(models, rs)
    ]
    hover_text = [
        (f"<b>{m}</b><br>r = {r:+.3f} · p = {p:.1e}"
         + ("<br><i>diagnostic failure (see §4.2)</i>" if m in failures or r < 0
            else ""))
        for m, r, p in zip(models, rs, pvals)
    ]
    return {
        "data": [{
            "type": "bar",
            "x": models, "y": rs,
            "marker": {"color": colors,
                        "line": {"color": "#333", "width": 0.5}},
            "hovertext": hover_text, "hoverinfo": "text",
            "text": [f"{r:+.3f}" for r in rs],
            "textposition": "outside",
            "textfont": {"size": 10},
        }],
        "layout": _layout(
            title="§3.1.1 legal-vs-control · FreeLaw-EN and Qwen3-0.6B-EN "
                  "as the two non-conforming readings",
            xaxis={"title": "", "tickangle": -30, "automargin": True},
            yaxis={"title": "rank-biserial r",
                   "zeroline": True, "zerolinecolor": "#b08d57",
                   "zerolinewidth": 1.5,
                   "range": [-0.18, 0.34]},
            margin={"l": 60, "r": 30, "t": 70, "b": 110},
            height=400,
            shapes=[{
                "type": "line", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                "line": {"color": "#b08d57", "width": 1.5},
            }],
        ),
    }
