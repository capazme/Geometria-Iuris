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


def fig_orthogonality(s322: dict, model: str = "BGE-EN-large",
                       variant: str = "attested") -> dict:
    """6×6 cosine of axis vectors for one model. Diagonal = 1.

    Reads `s322[variant][model]["cosine_matrix"]` (the canonical schema)
    and re-orders rows/cols to `AXES_ORDER` based on the embedded `axes`
    list.
    """
    block = s322[variant]
    if model not in block:
        model = next(iter(block.keys()))
    entry = block[model]
    raw_axes = list(entry["axes"])
    raw_matrix = entry["cosine_matrix"]
    idx_by_axis = {a: i for i, a in enumerate(raw_axes)}
    matrix = [[0.0] * len(AXES_ORDER) for _ in AXES_ORDER]
    for r, a1 in enumerate(AXES_ORDER):
        for c, a2 in enumerate(AXES_ORDER):
            if a1 in idx_by_axis and a2 in idx_by_axis:
                matrix[r][c] = float(raw_matrix[idx_by_axis[a1]][idx_by_axis[a2]])
            elif r == c:
                matrix[r][c] = 1.0
    labels = [_axis_label(a) for a in AXES_ORDER]
    return {
        "data": [{
            "type": "heatmap",
            "z": matrix,
            "x": labels, "y": labels,
            "zmin": -1, "zmax": 1, "zmid": 0,
            "colorscale": _ORTHO_COLORSCALE,
            "colorbar": {"title": "cosine", "thickness": 12, "len": 0.7},
            "hovertemplate": "%{y} × %{x}: %{z:.3f}<extra></extra>",
        }],
        "layout": _layout(
            title=f"§3.2.2 — Inter-axis cosine · {model} ({variant})",
            xaxis={"title": "", "tickangle": -20, "automargin": True},
            yaxis={"title": "", "autorange": "reversed", "automargin": True},
            margin={"l": 130, "r": 25, "t": 50, "b": 90},
            height=460,
        ),
    }


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
