"""Plotly figure factories for Experiment 1 (§3.1).

Each `fig_*(...)` function returns a `{"data": [...], "layout": {...}}` dict
ready to be passed to `shared_ui.plot_block()`. No Plotly dependency at
build time — the dict is serialised to JSON and rendered client-side by
the vendored `assets/plotly.min.js`.

Figures provided:
    fig_legal_control_bar       §3.1.1 — Mann-Whitney effect r, 10 models
    fig_intra_inter_bar         §3.1.1 — intra vs inter effect r, 3 WEIRD
    fig_topology_heatmap        §3.1.2 — 7×7 inter-domain RDM, one model
    fig_topology_smallmultiples §3.1.2 — 7×7 RDM, all 10 models
    fig_rsa_forest              §3.1.3 — 17 model-pair ρ with 95% CI
    fig_rsa_bare_attested_slope §3.1.3 — bare→attested gain per pair
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
from data.loader_31 import (  # noqa: E402
    ALL_MODELS_ORDERED, GROUP_ORDER, GROUP_LABEL, model_group,
)


_GROUP_COLOR = {
    "within_weird":     PLOT_COLORS["weird"],
    "within_sinic":     PLOT_COLORS["sinic"],
    "within_bilingual": PLOT_COLORS["bilingual"],
    "cross":            PLOT_COLORS["cross"],
    "weird":            PLOT_COLORS["weird"],
    "sinic":            PLOT_COLORS["sinic"],
    "bilingual":        PLOT_COLORS["bilingual"],
}


def _layout(**overrides) -> dict:
    """Base layout dict, deep-copy of PLOTLY_LAYOUT_DEFAULTS + overrides."""
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
# §3.1.1 — Mann-Whitney effect r bars

def fig_legal_control_bar(s311: dict, variant: str = "bare") -> dict:
    """Effect-size bar chart: legal vs control distances, per model.

    Negative r flagged in muted grey (FreeLaw-EN, Qwen3-0.6B-EN under run #4).
    """
    lvc = s311[variant]["legal_vs_control"]
    if not lvc:
        return {"data": [], "layout": _layout(title="(no legal-vs-control data)")}
    rows = [(m, float(lvc[m]["effect_r"]), float(lvc[m]["p_value"]))
            for m in ALL_MODELS_ORDERED if m in lvc]
    models = [r[0] for r in rows]
    rs = [r[1] for r in rows]
    pvals = [r[2] for r in rows]
    colors = [
        PLOT_COLORS["control"] if r < 0
        else _GROUP_COLOR[model_group(m)]
        for m, r in zip(models, rs)
    ]
    text = [f"r={r:.3f}<br>p={p:.1e}" for r, p in zip(rs, pvals)]
    return {
        "data": [{
            "type": "bar",
            "x": models, "y": rs,
            "marker": {"color": colors,
                       "line": {"color": "#333", "width": 0.4}},
            "hovertext": text, "hoverinfo": "text",
            "name": "effect r",
        }],
        "layout": _layout(
            title=f"§3.1.1 — Legal vs control distances (rank-biserial r, {variant})",
            xaxis={"title": "", "tickangle": -35},
            yaxis={"title": "rank-biserial r", "zeroline": True,
                   "zerolinecolor": "#999", "zerolinewidth": 1},
            shapes=[{
                "type": "line", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                "line": {"color": "#999", "width": 0.8, "dash": "dot"},
            }],
            annotations=[{
                "x": 0.02, "y": 0.04, "xref": "paper", "yref": "paper",
                "text": "<i>r &lt; 0: encoder specialisation effect</i>",
                "showarrow": False,
                "font": {"size": 10, "color": PLOT_COLORS["control"]},
                "xanchor": "left",
            }],
            height=440,
        ),
    }


def fig_intra_inter_bar(s311: dict, variant: str = "bare") -> dict:
    """Effect-size bar chart: intra-domain vs inter-domain (WEIRD only)."""
    ii = s311[variant]["intra_inter"]
    rows = [(m, float(ii[m]["effect_r"]), float(ii[m]["p_value"]))
            for m in ii]
    models = [r[0] for r in rows]
    rs = [r[1] for r in rows]
    pvals = [r[2] for r in rows]
    text = [f"r={r:.3f}<br>p={p:.1e}" for r, p in zip(rs, pvals)]
    return {
        "data": [{
            "type": "bar",
            "x": models, "y": rs,
            "marker": {"color": PLOT_COLORS["weird"]},
            "hovertext": text, "hoverinfo": "text",
        }],
        "layout": _layout(
            title=f"§3.1.1 — Intra- vs inter-domain (WEIRD models, {variant})",
            xaxis={"title": "", "tickangle": -25},
            yaxis={"title": "rank-biserial r"},
            height=360,
        ),
    }


# --------------------------------------------------------------------------
# §3.1.2 — 7×7 inter-domain topology RDM

_RDM_COLORSCALE = [
    [0.0, "#faf7ee"], [0.25, "#e9d6a0"], [0.5, "#b08d57"],
    [0.75, "#7c5c2e"], [1.0, "#3a2614"],
]


def fig_topology_heatmap(s312: dict, model: str = "BGE-EN-large",
                          variant: str = "bare") -> dict:
    """7×7 cosine-distance RDM for one model."""
    pack = s312[variant]
    all_m = pack["all_models"]
    if model not in all_m:
        model = pack["primary"]
    matrix = all_m[model]["matrix"]
    domains = all_m[model]["domains"]
    return {
        "data": [{
            "type": "heatmap",
            "z": matrix,
            "x": domains, "y": domains,
            "colorscale": _RDM_COLORSCALE,
            "showscale": True,
            "colorbar": {"title": "cosine d", "thickness": 12, "len": 0.7},
            "hovertemplate": "%{y} × %{x}: %{z:.3f}<extra></extra>",
        }],
        "layout": _layout(
            title=f"§3.1.2 — Inter-domain topology · {model} ({variant})",
            xaxis={"title": "", "side": "bottom", "tickangle": -20},
            yaxis={"title": "", "autorange": "reversed"},
            height=480,
        ),
    }


def fig_topology_smallmultiples(s312: dict, variant: str = "bare") -> dict:
    """7×7 RDM with a dropdown to switch between models.

    One heatmap rendered at full size; the dropdown toggles which model's
    matrix is visible. Replaces the older grid of small multiples — same
    information, far more legible at one model at a time.
    """
    pack = s312[variant]
    models = [m for m in ALL_MODELS_ORDERED if m in pack["all_models"]]
    n = len(models)
    default = pack.get("primary") if pack.get("primary") in models else models[0]

    data = []
    for m in models:
        blob = pack["all_models"][m]
        matrix = blob["matrix"]
        domains = blob["domains"]
        data.append({
            "type": "heatmap",
            "z": matrix,
            "x": domains, "y": domains,
            "colorscale": _RDM_COLORSCALE,
            "showscale": True,
            "colorbar": {"title": "cosine d", "thickness": 12, "len": 0.7},
            "hovertemplate": f"<b>{m}</b><br>%{{y}} × %{{x}}: %{{z:.3f}}<extra></extra>",
            "visible": (m == default),
            "name": m,
        })

    buttons = []
    for idx, m in enumerate(models):
        visible = [i == idx for i in range(n)]
        buttons.append({
            "label": m,
            "method": "update",
            "args": [
                {"visible": visible},
                {"title": {"text": f"§3.1.2 — Inter-domain topology · {m} ({variant})"}},
            ],
        })

    layout = _layout(
        title=f"§3.1.2 — Inter-domain topology · {default} ({variant})",
        xaxis={"title": "", "side": "bottom", "tickangle": -20},
        yaxis={"title": "", "autorange": "reversed"},
        height=520,
    )
    layout["updatemenus"] = [{
        "type": "dropdown",
        "buttons": buttons,
        "direction": "down",
        "x": 1.0, "y": 1.14,
        "xanchor": "right", "yanchor": "top",
        "pad": {"l": 6, "r": 6, "t": 4, "b": 4},
        "bgcolor": "#faf7ee",
        "bordercolor": "#b08d57",
        "borderwidth": 1,
        "font": {"size": 11},
        "showactive": True,
    }]
    layout["annotations"] = [{
        "x": 1.0, "y": 1.18,
        "xref": "paper", "yref": "paper",
        "text": "<i>language model:</i>",
        "showarrow": False,
        "xanchor": "right", "yanchor": "bottom",
        "font": {"size": 11, "color": "#7c5c2e"},
    }]
    return {"data": data, "layout": layout}


# --------------------------------------------------------------------------
# §3.1.3 — RSA forest plot, 17 model pairs

def _pair_label(p: dict) -> str:
    return f"{p['model_a']} × {p['model_b']}"


def _group_sort_key(p: dict) -> tuple:
    g_rank = {"within_weird": 0, "within_sinic": 1,
              "within_bilingual": 2, "cross": 3}
    return (g_rank.get(p["group"], 9), p["model_a"], p["model_b"])


def fig_rsa_forest(s313: dict, variant: str = "attested") -> dict:
    """Forest plot of 17 model pairs: ρ ± 95% CI, coloured by group.

    Vertical band marks Δρ_sym; the within ρ̄_W and ρ̄_S means as dashed
    references on the right.
    """
    pairs = sorted(s313[variant]["pairs"], key=_group_sort_key)
    summary = s313[variant]["summary"]
    labels = [_pair_label(p) for p in pairs]
    rhos = [float(p["rho"]) for p in pairs]
    ci_low = [float(p["ci_low"]) for p in pairs]
    ci_high = [float(p["ci_high"]) for p in pairs]
    groups = [p["group"] for p in pairs]

    # One trace per group so the legend works cleanly.
    traces = []
    y_positions = list(range(len(pairs), 0, -1))
    pos_by_idx = {i: y for i, y in enumerate(y_positions)}
    for g in GROUP_ORDER:
        xs, ys, lo, hi, txt = [], [], [], [], []
        for i, p in enumerate(pairs):
            if p["group"] != g:
                continue
            xs.append(rhos[i])
            ys.append(pos_by_idx[i])
            lo.append(rhos[i] - ci_low[i])
            hi.append(ci_high[i] - rhos[i])
            txt.append(f"{labels[i]}<br>ρ={rhos[i]:.3f} · 95% CI [{ci_low[i]:.3f}, {ci_high[i]:.3f}]")
        if not xs:
            continue
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": xs, "y": ys,
            "error_x": {"type": "data", "symmetric": False,
                        "array": hi, "arrayminus": lo,
                        "color": _GROUP_COLOR[g], "thickness": 1.4, "width": 4},
            "marker": {"size": 9, "color": _GROUP_COLOR[g],
                       "line": {"color": "#222", "width": 0.7}},
            "name": GROUP_LABEL[g],
            "hovertext": txt, "hoverinfo": "text",
        })

    # Mean reference lines.
    shapes = []
    w_mean = float(summary.get("mean_rho_within_weird", 0))
    s_mean = float(summary.get("mean_rho_within_sinic", 0))
    cross_mean = float(summary.get("mean_rho_cross_tradition", 0))
    for x, color, dash in [(w_mean, PLOT_COLORS["weird"], "dash"),
                            (s_mean, PLOT_COLORS["sinic"], "dash"),
                            (cross_mean, PLOT_COLORS["cross"], "dot")]:
        shapes.append({
            "type": "line", "xref": "x", "yref": "paper",
            "x0": x, "x1": x, "y0": 0, "y1": 1,
            "line": {"color": color, "width": 1.2, "dash": dash},
        })

    annotations = [
        {"x": w_mean, "y": 1.03, "xref": "x", "yref": "paper",
         "text": f"ρ̄_W = {w_mean:.3f}", "showarrow": False,
         "font": {"size": 10, "color": PLOT_COLORS["weird"]}},
        {"x": s_mean, "y": 1.07, "xref": "x", "yref": "paper",
         "text": f"ρ̄_S = {s_mean:.3f}", "showarrow": False,
         "font": {"size": 10, "color": PLOT_COLORS["sinic"]}},
        {"x": cross_mean, "y": 1.03, "xref": "x", "yref": "paper",
         "text": f"ρ̄_cross = {cross_mean:.3f}", "showarrow": False,
         "font": {"size": 10, "color": PLOT_COLORS["cross"]}},
    ]

    title = (
        f"§3.1.3 — RSA · 17 model pairs · {variant} · "
        f"Δρ_sym = {summary.get('delta_rho_symmetric', 0):.3f}"
    )
    return {
        "data": traces,
        "layout": _layout(
            title=title,
            xaxis={"title": "Spearman ρ", "range": [0.15, 0.95],
                   "gridcolor": "#f0f0f0", "showgrid": True},
            yaxis={"title": "", "tickmode": "array",
                   "tickvals": y_positions, "ticktext": labels,
                   "tickfont": {"size": 9},
                   "showgrid": False, "zeroline": False},
            shapes=shapes,
            annotations=annotations,
            margin={"l": 280, "r": 130, "t": 80, "b": 50},
            height=560,
            legend={"orientation": "v", "yanchor": "top", "y": 0.99,
                    "x": 1.02, "font": {"size": 10},
                    "bgcolor": "rgba(255,255,255,0.8)"},
        ),
    }


def fig_rsa_forest_toggle(s313: dict) -> dict:
    """RSA forest with a bare/attested toggle.

    The 17 pairs are sorted into four groups (within Western-trained,
    within Chinese-trained, within bilingual control, cross-tradition).
    For each variant the same group ordering is preserved, so toggling
    swaps both the point estimates and the group-mean reference lines.
    """
    variants = ["attested", "bare"]
    default = "attested"

    # Pre-compute per-variant data.
    per_variant: dict[str, dict] = {}
    for variant in variants:
        pairs = sorted(s313[variant]["pairs"], key=_group_sort_key)
        summary = s313[variant]["summary"]
        labels = [_pair_label(p) for p in pairs]
        rhos = [float(p["rho"]) for p in pairs]
        ci_low = [float(p["ci_low"]) for p in pairs]
        ci_high = [float(p["ci_high"]) for p in pairs]
        y_positions = list(range(len(pairs), 0, -1))
        group_traces: dict[str, dict] = {}
        for g in GROUP_ORDER:
            xs, ys, lo, hi, txt = [], [], [], [], []
            for i, p in enumerate(pairs):
                if p["group"] != g:
                    continue
                xs.append(rhos[i])
                ys.append(y_positions[i])
                lo.append(rhos[i] - ci_low[i])
                hi.append(ci_high[i] - rhos[i])
                txt.append(
                    f"{labels[i]}<br>"
                    f"ρ={rhos[i]:.3f} · 95% CI [{ci_low[i]:.3f}, {ci_high[i]:.3f}]"
                )
            group_traces[g] = {
                "x": xs, "y": ys, "lo": lo, "hi": hi, "txt": txt,
            }
        per_variant[variant] = {
            "labels":   labels,
            "y_pos":    y_positions,
            "groups":   group_traces,
            "summary":  summary,
        }

    # Build the trace list: per (variant, group) -> one trace.
    data = []
    trace_index_by_variant: dict[str, list] = {v: [] for v in variants}
    for variant in variants:
        pv = per_variant[variant]
        for g in GROUP_ORDER:
            gt = pv["groups"][g]
            if not gt["x"]:
                trace_index_by_variant[variant].append(None)
                continue
            data.append({
                "type": "scatter",
                "mode": "markers",
                "x": gt["x"], "y": gt["y"],
                "error_x": {"type": "data", "symmetric": False,
                             "array": gt["hi"], "arrayminus": gt["lo"],
                             "color": _GROUP_COLOR[g],
                             "thickness": 1.4, "width": 4},
                "marker": {"size": 9, "color": _GROUP_COLOR[g],
                            "line": {"color": "#222", "width": 0.7}},
                "name": GROUP_LABEL[g],
                "legendgroup": g,
                "showlegend": (variant == default),
                "hovertext": gt["txt"], "hoverinfo": "text",
                "visible": (variant == default),
            })
            trace_index_by_variant[variant].append(len(data) - 1)

    n_traces = len(data)

    def _layout_update(variant: str) -> dict:
        pv = per_variant[variant]
        summary = pv["summary"]
        labels = pv["labels"]
        y_pos = pv["y_pos"]
        w_mean = float(summary.get("mean_rho_within_weird", 0))
        s_mean = float(summary.get("mean_rho_within_sinic", 0))
        cross_mean = float(summary.get("mean_rho_cross_tradition", 0))
        d_sym = float(summary.get("delta_rho_symmetric", 0))
        shapes = []
        for x, color, dash in [(w_mean, PLOT_COLORS["weird"], "dash"),
                                (s_mean, PLOT_COLORS["sinic"], "dash"),
                                (cross_mean, PLOT_COLORS["cross"], "dot")]:
            shapes.append({
                "type": "line", "xref": "x", "yref": "paper",
                "x0": x, "x1": x, "y0": 0, "y1": 1,
                "line": {"color": color, "width": 1.2, "dash": dash},
            })
        annotations = [
            {"x": w_mean, "y": 1.03, "xref": "x", "yref": "paper",
             "text": f"ρ̄_W = {w_mean:.3f}", "showarrow": False,
             "font": {"size": 10, "color": PLOT_COLORS["weird"]}},
            {"x": s_mean, "y": 1.07, "xref": "x", "yref": "paper",
             "text": f"ρ̄_S = {s_mean:.3f}", "showarrow": False,
             "font": {"size": 10, "color": PLOT_COLORS["sinic"]}},
            {"x": cross_mean, "y": 1.03, "xref": "x", "yref": "paper",
             "text": f"ρ̄_cross = {cross_mean:.3f}", "showarrow": False,
             "font": {"size": 10, "color": PLOT_COLORS["cross"]}},
            {"x": 1.0, "y": 1.22, "xref": "paper", "yref": "paper",
             "text": "<i>encoding:</i>", "showarrow": False,
             "xanchor": "right", "yanchor": "bottom",
             "font": {"size": 11, "color": "#7c5c2e"}},
        ]
        return {
            "title": {"text":
                f"§3.1.3 — RSA · 17 model pairs · {variant} · "
                f"Δρ_sym = {d_sym:.3f}"},
            "shapes": shapes,
            "annotations": annotations,
            "yaxis": {"title": "", "tickmode": "array",
                      "tickvals": y_pos, "ticktext": labels,
                      "tickfont": {"size": 9},
                      "showgrid": False, "zeroline": False},
        }

    def _visibility_for(variant: str) -> list:
        ok = {i for i in trace_index_by_variant[variant] if i is not None}
        return [i in ok for i in range(n_traces)]

    def _showlegend_for(variant: str) -> list:
        ok = {i for i in trace_index_by_variant[variant] if i is not None}
        return [i in ok for i in range(n_traces)]

    buttons = []
    for variant in variants:
        buttons.append({
            "label": variant,
            "method": "update",
            "args": [
                {"visible": _visibility_for(variant),
                 "showlegend": _showlegend_for(variant)},
                _layout_update(variant),
            ],
        })

    layout = _layout(
        xaxis={"title": "Spearman ρ", "range": [0.0, 0.95],
               "gridcolor": "#f0f0f0", "showgrid": True},
        margin={"l": 280, "r": 130, "t": 90, "b": 50},
        height=580,
        legend={"orientation": "v", "yanchor": "top", "y": 0.99,
                "x": 1.02, "font": {"size": 10},
                "bgcolor": "rgba(255,255,255,0.8)"},
    )
    layout.update(_layout_update(default))
    layout["updatemenus"] = [{
        "type": "buttons",
        "buttons": buttons,
        "direction": "right",
        "x": 1.0, "y": 1.18,
        "xanchor": "right", "yanchor": "top",
        "pad": {"l": 6, "r": 6, "t": 4, "b": 4},
        "bgcolor": "#faf7ee",
        "bordercolor": "#b08d57",
        "borderwidth": 1,
        "font": {"size": 11},
        "active": 0,
        "showactive": True,
    }]
    return {"data": data, "layout": layout}


def fig_rsa_bare_attested_slope(s313: dict) -> dict:
    """Slope chart: bare ρ → attested ρ for each of the 17 pairs.

    Within-tradition pairs (WEIRD/Sinic/bilingual) drawn thick + opaque so
    the steep gain stands out; cross-tradition pairs drawn thin +
    semi-transparent to recede into the background.
    """
    bare_pairs = {(_pair_label(p)): p for p in s313["bare"]["pairs"]}
    att_pairs = {(_pair_label(p)): p for p in s313["attested"]["pairs"]}
    labels = [p for p in bare_pairs if p in att_pairs]

    traces = []
    for g in GROUP_ORDER:
        items = []
        for lab in labels:
            bp = bare_pairs[lab]
            ap = att_pairs[lab]
            if bp["group"] != g:
                continue
            items.append((lab, float(bp["rho"]), float(ap["rho"])))
        if not items:
            continue
        is_within = g in ("within_weird", "within_sinic", "within_bilingual")
        line_width = 2.2 if is_within else 0.9
        opacity = 0.9 if is_within else 0.3
        marker_size = 8 if is_within else 5

        for k, (lab, y_bare, y_att) in enumerate(items):
            traces.append({
                "type": "scatter", "mode": "lines+markers",
                "x": ["bare", "attested"],
                "y": [y_bare, y_att],
                "line": {"color": _GROUP_COLOR[g], "width": line_width},
                "marker": {"size": marker_size, "color": _GROUP_COLOR[g]},
                "opacity": opacity,
                "showlegend": (k == 0),
                "name": GROUP_LABEL[g],
                "hovertemplate": (
                    f"{lab}<br>bare = {y_bare:.3f} → "
                    f"attested = {y_att:.3f}<extra></extra>"
                ),
            })

    return {
        "data": traces,
        "layout": _layout(
            title="§3.1.3 — Bare → attested ρ slope (17 pairs)",
            xaxis={"title": "", "tickangle": 0},
            yaxis={"title": "Spearman ρ", "range": [0.15, 0.95]},
            height=460,
            legend={"orientation": "h", "yanchor": "bottom",
                    "y": -0.15, "x": 0, "font": {"size": 10}},
        ),
    }


# --------------------------------------------------------------------------
# §3.1.4 — Categorical probe summary forest (simple)

def _mean_projection(per_template: list) -> list:
    """Mean PC1 projection across the 5 paraphrase templates."""
    if not per_template:
        return []
    n = len(per_template[0].get("pc1_projection", []))
    if n == 0:
        return []
    sums = [0.0] * n
    counts = [0] * n
    for pt in per_template:
        proj = pt.get("pc1_projection", [])
        for i, v in enumerate(proj):
            if i < n and v is not None:
                sums[i] += float(v)
                counts[i] += 1
    return [s / c if c else 0.0 for s, c in zip(sums, counts)]


def fig_categorical_probe_explorer(s314: dict) -> dict:
    """Interactive explorer: drop-down to switch test, 10 model curves shown.

    Each curve is the mean PC1 projection across the 5 paraphrase templates
    for that (test, model). A dashed vertical line marks the doctrinally
    expected break. Hover shows the model and the projection value.
    """
    tests = s314.get("tests", {})
    test_keys = list(tests.keys())
    if not test_keys:
        return {"data": [], "layout": _layout(title="(no probe data)")}

    test_titles = {tid: tests[tid].get("label", tid) for tid in test_keys}

    # Pre-compute mean projections per (test, model).
    model_order = []
    seen = set()
    for tid in test_keys:
        for m in tests[tid].get("per_model", {}):
            if m not in seen:
                model_order.append(m)
                seen.add(m)

    data = []
    # Build flat list: for each test, for each model -> one trace.
    trace_index_by_test = {}
    cursor = 0
    for tid in test_keys:
        t = tests[tid]
        cats_en = t.get("categories_en") or [f"cat {i+1}" for i in range(11)]
        n_cat = len(cats_en)
        x = list(range(1, n_cat + 1))
        indices = []
        for m in model_order:
            mblob = t.get("per_model", {}).get(m)
            if not mblob:
                indices.append(None)
                continue
            proj = _mean_projection(mblob.get("per_template", []))
            if not proj or len(proj) != n_cat:
                indices.append(None)
                continue
            grp = model_group(m)
            color = _GROUP_COLOR.get(grp, "#999")
            trace = {
                "type": "scatter",
                "mode": "lines+markers",
                "x": x,
                "y": proj,
                "name": m,
                "legendgroup": grp,
                "line": {"color": color, "width": 1.6},
                "marker": {"color": color, "size": 6,
                            "line": {"width": 0.5, "color": "#333"}},
                "hovertemplate": (
                    f"<b>{m}</b><br>"
                    "%{x}. %{customdata}<br>"
                    "PC1 = %{y:.3f}<extra></extra>"
                ),
                "customdata": cats_en,
                "visible": (tid == "test_3_age_contractual_capacity"
                            or (tid == test_keys[0]
                                and "test_3_age_contractual_capacity" not in test_keys)),
            }
            data.append(trace)
            indices.append(cursor)
            cursor += 1
        trace_index_by_test[tid] = indices

    n_traces = len(data)

    def _visible_mask(active_tid: str) -> list:
        idx_set = {i for i in trace_index_by_test.get(active_tid, []) if i is not None}
        return [i in idx_set for i in range(n_traces)]

    def _layout_update(tid: str) -> dict:
        t = tests[tid]
        cats_en = t.get("categories_en") or [f"cat {i+1}" for i in range(11)]
        cats_zh = t.get("categories_zh") or [""] * len(cats_en)
        ticktext = [f"{en}<br><span style='color:#888;font-size:9px'>{zh}</span>"
                    if zh else en for en, zh in zip(cats_en, cats_zh)]
        eg = t.get("expected_gap_index")
        polarity = t.get("polarity", "positive")
        shapes = []
        annotations = []
        if eg is not None and eg >= 0 and polarity == "positive":
            shapes.append({
                "type": "line",
                "x0": eg + 1.5, "x1": eg + 1.5,
                "y0": 0, "y1": 1,
                "xref": "x", "yref": "paper",
                "line": {"color": "#b08d57", "width": 2, "dash": "dash"},
            })
            annotations.append({
                "x": eg + 1.5, "y": 1.02,
                "xref": "x", "yref": "paper",
                "text": f"<i>expected break:</i> {cats_en[eg]} → {cats_en[eg+1]}",
                "showarrow": False,
                "xanchor": "center", "yanchor": "bottom",
                "font": {"size": 10, "color": "#7c5c2e"},
            })
        elif polarity == "negative":
            annotations.append({
                "x": 0.5, "y": 1.02,
                "xref": "paper", "yref": "paper",
                "text": "<i>negative control:</i> no doctrinal break expected",
                "showarrow": False,
                "xanchor": "center", "yanchor": "bottom",
                "font": {"size": 10, "color": "#7c5c2e"},
            })
        return {
            "title": {"text": f"§3.1.4 — Categorical probe · {test_titles[tid]}"},
            "xaxis": {"tickvals": list(range(1, len(cats_en) + 1)),
                      "ticktext": ticktext,
                      "tickangle": -35,
                      "tickfont": {"size": 10}},
            "shapes": shapes,
            "annotations": annotations,
        }

    buttons = []
    default_tid = ("test_3_age_contractual_capacity"
                   if "test_3_age_contractual_capacity" in test_keys
                   else test_keys[0])
    for tid in test_keys:
        buttons.append({
            "label": test_titles[tid][:48],
            "method": "update",
            "args": [{"visible": _visible_mask(tid)}, _layout_update(tid)],
        })

    initial_layout = _layout(
        title=f"§3.1.4 — Categorical probe · {test_titles[default_tid]}",
        xaxis={"title": "ordered legal category"},
        yaxis={"title": "mean PC1 projection (across 5 paraphrase templates)"},
        height=520,
        margin={"l": 60, "r": 25, "t": 90, "b": 130},
        showlegend=True,
        legend={"orientation": "h", "y": -0.35, "x": 0.5, "xanchor": "center",
                "font": {"size": 10}},
    )
    initial_layout.update(_layout_update(default_tid))
    initial_layout["updatemenus"] = [{
        "type": "dropdown",
        "buttons": buttons,
        "direction": "down",
        "x": 1.0, "y": 1.18,
        "xanchor": "right", "yanchor": "top",
        "pad": {"l": 6, "r": 6, "t": 4, "b": 4},
        "bgcolor": "#faf7ee",
        "bordercolor": "#b08d57",
        "borderwidth": 1,
        "font": {"size": 11},
        "showactive": True,
    }]
    # Add a separate label annotation for the dropdown.
    initial_layout.setdefault("annotations", []).append({
        "x": 1.0, "y": 1.22,
        "xref": "paper", "yref": "paper",
        "text": "<i>test:</i>",
        "showarrow": False,
        "xanchor": "right", "yanchor": "bottom",
        "font": {"size": 11, "color": "#7c5c2e"},
    })
    return {"data": data, "layout": initial_layout}


def fig_categorical_probe_forest(s314: dict) -> dict:
    """Per-test mean ensemble ρ across models, with the modal max-gap
    position label as hover."""
    tests = s314.get("tests", {})
    rows = []
    for tid, t in tests.items():
        s = t.get("summary", {})
        if "mean_ensemble_rho" in s:
            rows.append((tid, float(s["mean_ensemble_rho"]),
                          t.get("label", tid), t.get("borderline", False)))
    rows.sort(key=lambda r: -r[1])
    return {
        "data": [{
            "type": "bar", "orientation": "h",
            "x": [r[1] for r in rows],
            "y": [r[2] for r in rows],
            "marker": {
                "color": [PLOT_COLORS["warn"] if r[3] else PLOT_COLORS["accent"]
                           for r in rows],
            },
            "hovertext": [f"{r[2]}<br>ρ̄ = {r[1]:.3f}" for r in rows],
            "hoverinfo": "text",
        }],
        "layout": _layout(
            title="§3.1.4 — Categorical probe · mean ensemble ρ per test",
            xaxis={"title": "Spearman ρ̄ (ensemble across models)"},
            yaxis={"title": "", "automargin": True},
            margin={"l": 300, "r": 100, "t": 60, "b": 50},
            height=max(360, 50 * len(rows) + 120),
        ),
    }
