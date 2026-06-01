"""Plotly figure factories for Experiment 2 (§3.2).

Each `fig_*(...)` function returns a `{"data": [...], "layout": {...}}` dict.

Figures provided:
    fig_axes_ranking      §3.2.4 — bar chart of cross-tradition ρ̄ per axis
    fig_axes_forest       §3.2.3 — per-pair ρ per axis with group colouring
    fig_axes_boxplot      §3.2.3 — box of ρ per axis with overlay points
    fig_orthogonality     §3.2.2 — 6×6 mean inter-axis cosine
    fig_sanity_heatmap    §3.2.1 — sanity pass per axis × model
    fig_axes_bare_attested §3.2.4 — within & cross ρ̄ side-by-side
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
from data.loader_32 import (  # noqa: E402
    AXES_ORDER, AXIS_LABELS, WEIRD_MODELS, SINIC_MODELS,
    GROUP_ORDER, GROUP_LABEL,
)


_GROUP_COLOR = {
    "within_weird":     PLOT_COLORS["weird"],
    "within_sinic":     PLOT_COLORS["sinic"],
    "within_bilingual": PLOT_COLORS["bilingual"],
    "cross":            PLOT_COLORS["cross"],
}


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


def _axis_label(axis: str) -> str:
    return AXIS_LABELS.get(axis, axis.replace("_", " ↔ "))


# --------------------------------------------------------------------------
# §3.2.4 — Cross-tradition ρ̄ ranking, bare vs attested

def fig_axes_ranking(s324: dict, variant: str = "attested",
                      sort_ascending: bool = True) -> dict:
    """Horizontal bar chart of cross-tradition ρ̄ per axis.

    Most-divergent axis on top (lowest ρ̄ = most divergent under
    `sort_ascending=True`).
    """
    means = s324[variant]["cross_rho_mean_per_axis"]
    pairs = [(a, float(means[a])) for a in AXES_ORDER if a in means]
    pairs.sort(key=lambda r: r[1], reverse=not sort_ascending)
    axes = [r[0] for r in pairs]
    rhos = [r[1] for r in pairs]
    labels = [_axis_label(a) for a in axes]
    return {
        "data": [{
            "type": "bar", "orientation": "h",
            "x": rhos, "y": labels,
            "marker": {"color": PLOT_COLORS["accent_dark"]},
            "text": [f"{r:.3f}" for r in rhos],
            "textposition": "outside",
            "hovertemplate": "%{y}<br>ρ̄_cross = %{x:.3f}<extra></extra>",
        }],
        "layout": _layout(
            title=f"§3.2.4 — Cross-tradition ρ̄ per axis ({variant})",
            xaxis={"title": "mean Spearman ρ (cross-tradition pairs)",
                    "range": [0, max(rhos) * 1.15 + 0.05]},
            yaxis={"title": "", "automargin": True},
            margin={"l": 170, "r": 30, "t": 50, "b": 50},
            height=400,
        ),
    }


def fig_axes_ranking_toggle(s324: dict) -> dict:
    """Cross-tradition ρ̄ per axis, with a bare/attested toggle.

    Both variants are pre-computed; toggling re-orders the axis ranking
    (most divergent on top) for the selected variant.
    """
    variants = ["attested", "bare"]
    default = "attested"
    per_variant: dict[str, dict] = {}
    for variant in variants:
        means = s324[variant]["cross_rho_mean_per_axis"]
        pairs = [(a, float(means[a])) for a in AXES_ORDER if a in means]
        pairs.sort(key=lambda r: r[1])  # ascending: most divergent on top
        axes = [r[0] for r in pairs]
        rhos = [r[1] for r in pairs]
        labels = [_axis_label(a) for a in axes]
        per_variant[variant] = {"labels": labels, "rhos": rhos}

    d0 = per_variant[default]
    data = [{
        "type": "bar", "orientation": "h",
        "x": d0["rhos"], "y": d0["labels"],
        "marker": {"color": PLOT_COLORS["accent_dark"]},
        "text": [f"{r:.3f}" for r in d0["rhos"]],
        "textposition": "outside",
        "hovertemplate": "%{y}<br>ρ̄_cross = %{x:.3f}<extra></extra>",
    }]

    buttons = []
    max_x = max(
        max(per_variant[v]["rhos"]) for v in variants
    ) * 1.15 + 0.05
    for variant in variants:
        d = per_variant[variant]
        buttons.append({
            "label": variant,
            "method": "update",
            "args": [
                {"x": [d["rhos"]], "y": [d["labels"]],
                 "text": [[f"{r:.3f}" for r in d["rhos"]]]},
                {"title": {"text":
                    f"§3.2.4 — Cross-tradition ρ̄ per axis ({variant})"}},
            ],
        })

    layout = _layout(
        title=f"§3.2.4 — Cross-tradition ρ̄ per axis ({default})",
        xaxis={"title": "mean Spearman ρ (cross-tradition pairs)",
               "range": [0, max_x]},
        yaxis={"title": "", "automargin": True},
        margin={"l": 170, "r": 60, "t": 90, "b": 50},
        height=440,
    )
    layout["updatemenus"] = [{
        "type": "buttons",
        "buttons": buttons,
        "direction": "right",
        "x": 1.0, "y": 1.16,
        "xanchor": "right", "yanchor": "top",
        "pad": {"l": 6, "r": 6, "t": 4, "b": 4},
        "bgcolor": "#faf7ee",
        "bordercolor": "#b08d57",
        "borderwidth": 1,
        "font": {"size": 11},
        "active": 0,
        "showactive": True,
    }]
    layout["annotations"] = [{
        "x": 1.0, "y": 1.22,
        "xref": "paper", "yref": "paper",
        "text": "<i>encoding:</i>",
        "showarrow": False,
        "xanchor": "right", "yanchor": "bottom",
        "font": {"size": 11, "color": "#7c5c2e"},
    }]
    return {"data": data, "layout": layout}


def fig_axes_ranking_compare(s324: dict) -> dict:
    """Side-by-side bare vs attested ρ̄ per axis."""
    means_b = s324["bare"]["cross_rho_mean_per_axis"]
    means_a = s324["attested"]["cross_rho_mean_per_axis"]
    axes = list(AXES_ORDER)
    labels = [_axis_label(a) for a in axes]
    bare = [float(means_b.get(a, 0)) for a in axes]
    att = [float(means_a.get(a, 0)) for a in axes]
    return {
        "data": [
            {"type": "bar", "name": "bare",
             "x": labels, "y": bare,
             "marker": {"color": PLOT_COLORS["control"]},
             "text": [f"{r:.3f}" for r in bare], "textposition": "outside",
             "textfont": {"size": 10}},
            {"type": "bar", "name": "attested",
             "x": labels, "y": att,
             "marker": {"color": PLOT_COLORS["accent_dark"]},
             "text": [f"{r:.3f}" for r in att], "textposition": "outside",
             "textfont": {"size": 10}},
        ],
        "layout": _layout(
            title="§3.2.4 — Cross-tradition ρ̄ per axis · bare vs attested",
            xaxis={"title": "", "tickangle": -20, "automargin": True},
            yaxis={"title": "mean ρ_cross", "range": [0, 0.55]},
            barmode="group",
            margin={"l": 60, "r": 25, "t": 50, "b": 110},
            height=460,
            legend={"orientation": "h", "yanchor": "bottom",
                    "y": -0.32, "x": 0.35, "font": {"size": 11}},
        ),
    }


# --------------------------------------------------------------------------
# §3.2.3 — Per-pair ρ, forest / box

def _per_pair_by_axis(per_pair: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {a: [] for a in AXES_ORDER}
    for p in per_pair:
        if p["axis"] in out:
            out[p["axis"]].append(p)
    return out


def fig_axes_boxplot(s323: dict, variant: str = "attested") -> dict:
    """Box plot of per-pair ρ per axis, with overlay scatter coloured by group."""
    per_pair = s323[variant]["per_pair"]
    by_axis = _per_pair_by_axis(per_pair)

    box_traces = []
    overlay_traces: dict[str, dict] = {g: {"x": [], "y": [], "text": []}
                                         for g in GROUP_ORDER}
    for axis in AXES_ORDER:
        entries = by_axis[axis]
        rhos = [float(e["rho"]) for e in entries]
        label = _axis_label(axis)
        box_traces.append({
            "type": "box",
            "y": rhos, "x": [label] * len(rhos),
            "name": label, "showlegend": False,
            "boxpoints": False,
            "marker": {"color": PLOT_COLORS["border"]},
            "line": {"color": PLOT_COLORS["accent_dark"], "width": 1.0},
            "fillcolor": "rgba(176,141,87,0.18)",
            "width": 0.45,
        })
        for e in entries:
            g = e["group"]
            if g not in overlay_traces:
                continue
            overlay_traces[g]["x"].append(label)
            overlay_traces[g]["y"].append(float(e["rho"]))
            overlay_traces[g]["text"].append(
                f"{e['model_a']} × {e['model_b']}<br>ρ = {float(e['rho']):.3f}"
            )

    traces = list(box_traces)
    for g in GROUP_ORDER:
        if not overlay_traces[g]["x"]:
            continue
        traces.append({
            "type": "scatter", "mode": "markers",
            "x": overlay_traces[g]["x"], "y": overlay_traces[g]["y"],
            "marker": {"size": 7, "color": _GROUP_COLOR[g],
                       "line": {"color": "#222", "width": 0.4}},
            "name": GROUP_LABEL[g],
            "hovertext": overlay_traces[g]["text"], "hoverinfo": "text",
        })

    return {
        "data": traces,
        "layout": _layout(
            title=f"§3.2.3 — Per-pair ρ per axis ({variant})",
            xaxis={"title": "", "tickangle": -20, "automargin": True},
            yaxis={"title": "Spearman ρ", "range": [-0.2, 1.0]},
            margin={"l": 55, "r": 25, "t": 50, "b": 110},
            height=480,
            legend={"orientation": "h", "yanchor": "bottom",
                    "y": -0.35, "x": 0, "font": {"size": 11}},
        ),
    }


# --------------------------------------------------------------------------
# §3.2.2 — Inter-axis cosine (orthogonality)

_ORTHO_COLORSCALE = [
    [0.0, "#2c5f9a"], [0.25, "#a8c1d8"], [0.5, "#faf7ee"],
    [0.75, "#e9d6a0"], [1.0, "#a43a3a"],
]


def _reorder_axis_matrix(raw_axes: list, raw_matrix: list) -> list:
    """Reorder a square cosine matrix to AXES_ORDER, padding the diagonal."""
    idx_by_axis = {a: i for i, a in enumerate(raw_axes)}
    matrix = [[0.0] * len(AXES_ORDER) for _ in AXES_ORDER]
    for r, a1 in enumerate(AXES_ORDER):
        for c, a2 in enumerate(AXES_ORDER):
            if a1 in idx_by_axis and a2 in idx_by_axis:
                matrix[r][c] = float(raw_matrix[idx_by_axis[a1]][idx_by_axis[a2]])
            elif r == c:
                matrix[r][c] = 1.0
    return matrix


def fig_orthogonality(s322: dict, model: str = "BGE-EN-large",
                       variant: str = "attested") -> dict:
    """6×6 inter-axis cosine matrix with a dropdown to switch model.

    All available models for the variant are pre-computed and stored as
    parallel heatmap traces; the dropdown toggles which one is visible.
    """
    block = s322[variant]
    models = list(block.keys())
    if not models:
        return {"data": [], "layout": _layout(title="(no §3.2.2 data)")}
    default = model if model in block else models[0]
    labels = [_axis_label(a) for a in AXES_ORDER]

    data = []
    for m in models:
        entry = block[m]
        matrix = _reorder_axis_matrix(list(entry["axes"]), entry["cosine_matrix"])
        data.append({
            "type": "heatmap",
            "z": matrix,
            "x": labels, "y": labels,
            "zmin": -1, "zmax": 1, "zmid": 0,
            "colorscale": _ORTHO_COLORSCALE,
            "showscale": True,
            "colorbar": {"title": "cosine", "thickness": 12, "len": 0.7},
            "hovertemplate": f"<b>{m}</b><br>%{{y}} × %{{x}}: %{{z:.3f}}<extra></extra>",
            "name": m,
            "visible": (m == default),
        })

    n = len(models)
    buttons = []
    for idx, m in enumerate(models):
        buttons.append({
            "label": m,
            "method": "update",
            "args": [
                {"visible": [i == idx for i in range(n)]},
                {"title": {"text": f"§3.2.2 — Inter-axis cosine · {m} ({variant})"}},
            ],
        })

    layout = _layout(
        title=f"§3.2.2 — Inter-axis cosine · {default} ({variant})",
        xaxis={"title": "", "tickangle": -20, "automargin": True},
        yaxis={"title": "", "autorange": "reversed", "automargin": True},
        margin={"l": 130, "r": 25, "t": 80, "b": 90},
        height=480,
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
# §3.2.1 — Sanity pass per axis × model

def fig_sanity_heatmap(s321: dict, variant: str = "attested") -> dict:
    """Heatmap (#axes × #models) of positive_correct / n_pairs_total."""
    block = s321[variant]
    models = list(block.keys())
    matrix = []
    for axis in AXES_ORDER:
        row = []
        for m in models:
            ax = block[m]["axes"].get(axis, {})
            pos = ax.get("positive_correct", 0)
            total = ax.get("n_pairs_total", 1) or 1
            row.append(float(pos) / float(total))
        matrix.append(row)
    return {
        "data": [{
            "type": "heatmap",
            "z": matrix,
            "x": models,
            "y": [_axis_label(a) for a in AXES_ORDER],
            "zmin": 0, "zmax": 1,
            "colorscale": [[0, "#faf7ee"], [0.5, "#e9d6a0"], [1, "#5a8f3a"]],
            "colorbar": {"title": "pass fraction",
                          "thickness": 12, "len": 0.7},
            "hovertemplate": "%{y} · %{x}: %{z:.2f}<extra></extra>",
        }],
        "layout": _layout(
            title=f"§3.2.1 — Axis sanity (positive_correct / n_total, {variant})",
            xaxis={"title": "", "tickangle": -35, "automargin": True},
            yaxis={"title": "", "automargin": True},
            margin={"l": 170, "r": 25, "t": 50, "b": 130},
            height=420,
        ),
    }


# --------------------------------------------------------------------------
# §3.2.5 — Between-group differences (top divergent terms per axis)

def fig_top_divergent_terms_explorer(s325: dict, variant: str = "attested",
                                       top_k: int = 10) -> dict:
    """Per-axis horizontal bar chart of the top-K cross-tradition divergent
    terms, with paired W (Western-trained) and S (Chinese-trained)
    projections. Drop-down switches axis.
    """
    pack = s325.get(variant, {})
    if not pack:
        return {"data": [], "layout": _layout(title="(no §3.2.5 data)")}

    axes_with_data = [a for a in AXES_ORDER
                      if isinstance(pack.get(a), dict)
                      and pack[a].get("top_K_divergent")]
    if not axes_with_data:
        return {"data": [], "layout": _layout(title="(no top-divergent data)")}

    per_axis = {}
    for axis in axes_with_data:
        blob = pack[axis]
        items = blob.get("top_K_divergent", [])[:top_k]
        items_sorted = sorted(items, key=lambda r: abs(float(r.get("delta", 0))))
        labels = [
            f"{it.get('en','?')} · {it.get('zh','')}"
            for it in items_sorted
        ]
        w_scores = [float(it.get("w_score", 0)) for it in items_sorted]
        s_scores = [float(it.get("s_score", 0)) for it in items_sorted]
        deltas = [float(it.get("delta", 0)) for it in items_sorted]
        domains = [it.get("domain", "") for it in items_sorted]
        per_axis[axis] = {
            "labels":  labels,
            "w":       w_scores,
            "s":       s_scores,
            "deltas":  deltas,
            "domains": domains,
            "mean_abs": blob.get("delta_mean_abs"),
            "max_abs":  blob.get("delta_max_abs"),
        }

    default_axis = "natural_positive" if "natural_positive" in per_axis else axes_with_data[0]
    d0 = per_axis[default_axis]

    w_color = PLOT_COLORS["weird"]
    s_color = PLOT_COLORS["sinic"]

    data = [
        {
            "type": "bar",
            "orientation": "h",
            "y": d0["labels"],
            "x": d0["w"],
            "name": "Western-trained mean projection",
            "marker": {"color": w_color,
                        "line": {"color": "#333", "width": 0.4}},
            "customdata": [[d, f"|Δ| = {abs(dlt):.3f}"]
                            for d, dlt in zip(d0["domains"], d0["deltas"])],
            "hovertemplate": (
                "<b>%{y}</b><br>"
                "domain: %{customdata[0]}<br>"
                "W projection = %{x:.3f}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
        },
        {
            "type": "bar",
            "orientation": "h",
            "y": d0["labels"],
            "x": d0["s"],
            "name": "Chinese-trained mean projection",
            "marker": {"color": s_color,
                        "line": {"color": "#333", "width": 0.4}},
            "customdata": [[d, f"|Δ| = {abs(dlt):.3f}"]
                            for d, dlt in zip(d0["domains"], d0["deltas"])],
            "hovertemplate": (
                "<b>%{y}</b><br>"
                "domain: %{customdata[0]}<br>"
                "S projection = %{x:.3f}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
        },
    ]

    def _annotations_for(axis: str) -> list:
        d = per_axis[axis]
        mean_abs = d.get("mean_abs")
        max_abs = d.get("max_abs")
        bits = []
        if mean_abs is not None:
            bits.append(f"mean |Δ| = {float(mean_abs):.3f}")
        if max_abs is not None:
            bits.append(f"max |Δ| = {float(max_abs):.3f}")
        return [{
            "x": 0.5, "y": 1.02,
            "xref": "paper", "yref": "paper",
            "text": f"<i>{' · '.join(bits)}</i>" if bits else "",
            "showarrow": False,
            "xanchor": "center", "yanchor": "bottom",
            "font": {"size": 11, "color": "#7c5c2e"},
        }]

    buttons = []
    for axis in axes_with_data:
        d = per_axis[axis]
        buttons.append({
            "label": _axis_label(axis),
            "method": "update",
            "args": [
                {
                    "x": [d["w"], d["s"]],
                    "y": [d["labels"], d["labels"]],
                    "customdata": [
                        [[dm, f"|Δ| = {abs(dlt):.3f}"]
                          for dm, dlt in zip(d["domains"], d["deltas"])],
                        [[dm, f"|Δ| = {abs(dlt):.3f}"]
                          for dm, dlt in zip(d["domains"], d["deltas"])],
                    ],
                },
                {
                    "title": {"text":
                        f"§3.2.5 — Top {top_k} cross-tradition divergent terms · "
                        f"{_axis_label(axis)} ({variant})"},
                    "annotations": _annotations_for(axis) + [{
                        "x": 1.0, "y": 1.22,
                        "xref": "paper", "yref": "paper",
                        "text": "<i>axis:</i>",
                        "showarrow": False,
                        "xanchor": "right", "yanchor": "bottom",
                        "font": {"size": 11, "color": "#7c5c2e"},
                    }],
                },
            ],
        })

    layout = _layout(
        title=(f"§3.2.5 — Top {top_k} cross-tradition divergent terms · "
               f"{_axis_label(default_axis)} ({variant})"),
        xaxis={"title": "mean axis projection (signed)",
               "zeroline": True, "zerolinecolor": "#b08d57",
               "zerolinewidth": 1.5},
        yaxis={"title": "", "automargin": True, "tickfont": {"size": 11}},
        barmode="group",
        bargap=0.18,
        bargroupgap=0.08,
        height=max(520, 40 * top_k + 160),
        margin={"l": 220, "r": 25, "t": 90, "b": 70},
        showlegend=True,
        legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"},
    )
    layout["annotations"] = _annotations_for(default_axis) + [{
        "x": 1.0, "y": 1.22,
        "xref": "paper", "yref": "paper",
        "text": "<i>axis:</i>",
        "showarrow": False,
        "xanchor": "right", "yanchor": "bottom",
        "font": {"size": 11, "color": "#7c5c2e"},
    }]
    layout["updatemenus"] = [{
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
    return {"data": data, "layout": layout}
