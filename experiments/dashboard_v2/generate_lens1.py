"""Generator for dashboard_v2/lens1.html.

Reads the Lens I result JSONs, distance dumps and Mantel null distributions
and emits a single self-contained HTML page with the same visual language
as dashboard_v2/index.html. No interpretive claims are made: every section
describes what is computed, leaves the reading for Ch. 4.

Run:

    python experiments/dashboard_v2/generate_lens1.py

The script is idempotent: running it twice produces identical output.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import shared_ui as ui  # noqa: E402

LENS1 = REPO / "experiments" / "lens_1_relational" / "results"
LENS4 = REPO / "experiments" / "lens_4_values" / "results"
OUT = HERE / "lens1.html"

WEIRD_MODELS = ("BGE-EN-large", "E5-large", "FreeLaw-EN")
SINIC_MODELS = ("BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH")
BILINGUAL_MODELS = (
    ("BGE-M3-EN", "BGE-M3-ZH"),
    ("Qwen3-0.6B-EN", "Qwen3-0.6B-ZH"),
)
ALL_MODELS = (
    list(WEIRD_MODELS) + list(SINIC_MODELS)
    + [m for pair in BILINGUAL_MODELS for m in pair]
)

MODEL_GROUP_LABEL = {m: "WEIRD"     for m in WEIRD_MODELS}
MODEL_GROUP_LABEL.update({m: "Sinic" for m in SINIC_MODELS})
for pair in BILINGUAL_MODELS:
    for m in pair:
        MODEL_GROUP_LABEL[m] = "bilingue"


# --------------------------------------------------------------------------
# On-the-fly signal computation from RDMs

def _mann_whitney_less(x, y):
    """Mann-Whitney U with alternative='less' and rank-biserial r.

    Reimplemented from numpy only (avoids a scipy dependency at generator
    time). The p-value is a normal approximation with tie correction. For
    the sample sizes involved here (~8k intra, ~52k inter) the asymptotic
    approximation is effectively exact.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = x.size
    ny = y.size
    combined = np.concatenate([x, y])
    # Average ranks for ties: argsort + average rank assignment
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(combined, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    # Average ranks within tie groups
    sorted_vals = combined[order]
    _, inv, counts = np.unique(sorted_vals, return_inverse=True, return_counts=True)
    cum = np.concatenate(([0], np.cumsum(counts)))
    avg_rank_per_group = (cum[:-1] + cum[1:] + 1) / 2.0
    ranks[order] = avg_rank_per_group[inv]
    rank_sum_x = ranks[:nx].sum()
    u1 = rank_sum_x - nx * (nx + 1) / 2.0
    u2 = nx * ny - u1
    # rank-biserial r for alternative='less' (x systematically < y → r > 0)
    r = 1.0 - 2.0 * u1 / (nx * ny)
    # Normal approximation p-value (one-sided, 'less' → U1 small)
    mu = nx * ny / 2.0
    # Tie correction
    t = counts[counts > 1]
    tie_term = (t ** 3 - t).sum()
    N = combined.size
    sigma2 = (nx * ny / 12.0) * ((N + 1) - tie_term / (N * (N - 1)))
    sigma = float(np.sqrt(max(sigma2, 1e-30)))
    # continuity correction
    z = (u1 + 0.5 - mu) / sigma
    # one-sided 'less': p = Φ(z)
    from math import erf, sqrt
    p = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return {"statistic": float(u1), "u2": float(u2), "effect_r": float(r),
            "p_value": float(p), "n_x": nx, "n_y": ny,
            "median_x": float(np.median(x)), "median_y": float(np.median(y))}


def _compute_signal_for_model(rdm, domains):
    """From a 350×350 RDM and an array of 350 domain labels, compute:

    - the raw intra-domain / inter-domain distance arrays,
    - the Mann-Whitney U + effect_r,
    - the 7×7 domain topology matrix (mean distance per domain pair).
    """
    n = rdm.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    same = domains[iu] == domains[ju]
    intra = rdm[iu[same], ju[same]].astype(np.float32)
    inter = rdm[iu[~same], ju[~same]].astype(np.float32)
    stats = _mann_whitney_less(intra, inter)
    # Domain topology: 7×7 grid of mean distances (includes diagonal).
    unique_domains = sorted(set(domains.tolist()))
    idx_by_domain = {d: np.where(domains == d)[0] for d in unique_domains}
    K = len(unique_domains)
    topo = np.zeros((K, K), dtype=np.float32)
    for i, da in enumerate(unique_domains):
        idx_a = idx_by_domain[da]
        for j, db in enumerate(unique_domains):
            idx_b = idx_by_domain[db]
            if i == j:
                # within-domain: use upper triangle only
                if len(idx_a) < 2:
                    topo[i, j] = 0.0
                    continue
                sub = rdm[np.ix_(idx_a, idx_a)]
                tri_i, tri_j = np.triu_indices(len(idx_a), k=1)
                topo[i, j] = sub[tri_i, tri_j].mean()
            elif i < j:
                sub = rdm[np.ix_(idx_a, idx_b)]
                v = sub.mean()
                topo[i, j] = v
                topo[j, i] = v
    return {
        "intra":    intra,
        "inter":    inter,
        "stats":    stats,
        "topology": {"domains": unique_domains, "matrix": topo.tolist()},
    }


def _compute_legal_vs_control(vecs, core_idx, control_idx):
    """Compute the 61075 core-core pair distances and the 35000 core-control
    cross-pair distances from a single model's vectors, then run
    Mann-Whitney (alternative='less') with rank-biserial r.

    Vectors are L2-normalized at pre-compute time; cosine distance = 1 - dot.
    """
    core = vecs[core_idx]
    ctrl = vecs[control_idx]
    # core-core: upper triangle of cosine distance (350 terms → 61075 pairs)
    sim_core = core @ core.T
    iu, ju = np.triu_indices(len(core), k=1)
    legal = (1.0 - sim_core[iu, ju]).astype(np.float32)
    # core-control: all 350 × 100 cross distances = 35000
    sim_cc = core @ ctrl.T
    control = (1.0 - sim_cc).ravel().astype(np.float32)
    return _mann_whitney_less(legal, control)


def load_all():
    with (LENS1 / "lens1_results.json").open() as f:
        results = json.load(f)
    with (LENS1 / "categorical_probe.json").open() as f:
        probe = json.load(f)
    with (LENS4 / "lens4_results.json").open() as f:
        lens4 = json.load(f)
    terms = lens4["terms_core"]
    domains = np.array([t["domain"] for t in terms])

    # Signal per model computed on-the-fly from each RDM — covers all 10
    # models (3 WEIRD + 3 Sinic + 2 BGE-M3 + 2 Qwen3), unlike the JSON
    # section_31 which only precomputed the three WEIRD models.
    per_model_signal = {}
    for m in ALL_MODELS:
        rdm_path = LENS1 / "rdms" / f"{m}.npz"
        if not rdm_path.exists():
            continue
        rdm = np.load(rdm_path)["rdm"]
        per_model_signal[m] = _compute_signal_for_model(rdm, domains)

    # Legal-vs-control effect size for all 10 models, computed on-the-fly
    # from the model's own vectors.npy + the tier labels in index.json.
    # The pre-computed `distances/*.npz` dumps only covered the 3 WEIRD
    # models; re-deriving from raw vectors extends coverage to the
    # remaining 7 (Sinic + bilinguals) without rerunning the pipeline.
    with (REPO / "experiments" / "data" / "processed" / "embeddings" / "index.json").open() as f:
        emb_index = json.load(f)
    core_idx_global = np.array([i for i, t in enumerate(emb_index) if t.get("tier") == "core"])
    ctrl_idx_global = np.array([i for i, t in enumerate(emb_index) if t.get("tier") == "control"])
    emb_root = REPO / "experiments" / "data" / "processed" / "embeddings"
    legal_vs_control = {}
    for m in ALL_MODELS:
        vecs_path = emb_root / m / "vectors.npy"
        if not vecs_path.exists():
            continue
        vecs = np.load(vecs_path)
        legal_vs_control[m] = _compute_legal_vs_control(vecs, core_idx_global, ctrl_idx_global)

    # All 17 pair-wise null + bootstrap distributions (Mantel + term-level
    # resampling). Directory layout: `{A}_x_{B}.npz` with keys `null`
    # (1000,) and `bootstrap` (1000,).
    null_distributions = {}
    dist_dir = LENS1 / "distributions"
    for npz_path in sorted(dist_dir.glob("*_x_*.npz")):
        name = npz_path.stem  # e.g. 'BGE-EN-large_x_BGE-ZH-large'
        a, _, b = name.rpartition("_x_")
        arr = np.load(npz_path)
        null_distributions[(a, b)] = {"null": arr["null"], "bootstrap": arr["bootstrap"]}

    # Review CSV (kept for completeness; currently unused in the page body).
    review_path = LENS1 / "background_review.csv"
    review_rows = []
    if review_path.exists():
        with review_path.open() as f:
            reader = csv.DictReader(f)
            review_rows = list(reader)

    return results, probe, per_model_signal, legal_vs_control, null_distributions, review_rows


# --------------------------------------------------------------------------
# Plotly helpers (pure-dict, no plotly-py dependency)

def _base_layout(**overrides):
    layout = {**ui.PLOTLY_LAYOUT_DEFAULTS}
    layout["xaxis"] = {**ui.PLOTLY_AXIS_DEFAULTS}
    layout["yaxis"] = {**ui.PLOTLY_AXIS_DEFAULTS}
    layout.update(overrides)
    return layout


def _violin_fillcolor(model):
    grp = MODEL_GROUP_LABEL.get(model, "WEIRD")
    if grp == "WEIRD":
        return {"intra": ui.PLOT_COLORS["cream"], "inter": "#d5e1ef",
                "intra_line": ui.PLOT_COLORS["accent_dark"], "inter_line": ui.PLOT_COLORS["weird"]}
    if grp == "Sinic":
        return {"intra": "#f7e4df", "inter": "#f0cac0",
                "intra_line": "#a43a3a", "inter_line": "#7a2e2e"}
    # bilingue
    return {"intra": "#e8f0e0", "inter": "#cfdbc2",
            "intra_line": "#5a8f3a", "inter_line": "#3a6b1f"}


def fig_intra_vs_inter(per_model_signal):
    """Split violin per model: intra-domain (left) vs inter-domain (right)
    cosine distances. Embeds the full raw arrays (no sub-sampling) for
    all 10 models.
    """
    traces = []
    model_order = [m for m in ALL_MODELS if m in per_model_signal]
    for idx, m in enumerate(model_order):
        d = per_model_signal[m]
        colors = _violin_fillcolor(m)
        traces.append({
            "type": "violin",
            "y": d["intra"].tolist(),
            "x": [m] * len(d["intra"]),
            "name": "intra-dominio",
            "legendgroup": "intra",
            "showlegend": idx == 0,
            "side": "negative",
            "line": {"color": colors["intra_line"], "width": 1},
            "fillcolor": colors["intra"],
            "opacity": 0.9,
            "points": False,
            "meanline": {"visible": True, "color": colors["intra_line"]},
            "spanmode": "hard",
            "hovertemplate": "intra · %{x}<br>d=%{y:.3f}<extra></extra>",
        })
        traces.append({
            "type": "violin",
            "y": d["inter"].tolist(),
            "x": [m] * len(d["inter"]),
            "name": "inter-dominio",
            "legendgroup": "inter",
            "showlegend": idx == 0,
            "side": "positive",
            "line": {"color": colors["inter_line"], "width": 1},
            "fillcolor": colors["inter"],
            "opacity": 0.85,
            "points": False,
            "meanline": {"visible": True, "color": colors["inter_line"]},
            "spanmode": "hard",
            "hovertemplate": "inter · %{x}<br>d=%{y:.3f}<extra></extra>",
        })
    layout = _base_layout(
        height=520,
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "modello"},
               "categoryorder": "array", "categoryarray": model_order,
               "tickangle": -25, "automargin": True},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "distanza coseno"}, "range": [0, None]},
        violinmode="overlay",
        violingap=0.25,
        legend={"orientation": "h", "y": -0.22, "x": 0.5, "xanchor": "center"},
        margin={"l": 55, "r": 25, "t": 30, "b": 90},
    )
    return {"data": traces, "layout": layout}


def fig_legal_vs_control_effect(legal_vs_control):
    """Bar plot: effect_r of legal-vs-control per WEIRD model."""
    names = list(legal_vs_control.keys())
    values = [legal_vs_control[m]["effect_r"] for m in names]
    colors = [ui.PLOT_COLORS["weird"] for _ in names]
    text = [f"r = {v:+.3f}" for v in values]
    trace = {
        "type": "bar",
        "x": names,
        "y": values,
        "marker": {"color": colors, "line": {"color": ui.PLOT_COLORS["accent_dark"], "width": 0.8}},
        "text": text,
        "textposition": "outside",
        "hovertemplate": "%{x}<br>r = %{y:+.3f}<extra></extra>",
    }
    y_lo = min(values + [0.0]) - 0.08
    y_hi = max(values + [0.0]) + 0.08
    layout = _base_layout(
        height=320,
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "modello"},
               "tickangle": -25, "automargin": True},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS,
               "title": {"text": "effect size r  (core giuridico vs. controllo)"},
               "zeroline": True, "zerolinecolor": "#999",
               "range": [y_lo, y_hi]},
        showlegend=False,
    )
    return {"data": [trace], "layout": layout}


def fig_domain_topology(per_model_signal):
    """K×K heatmap per model with a dropdown selector (covers all 10)."""
    models = [m for m in ALL_MODELS if m in per_model_signal]
    traces = []
    buttons = []
    for i, m in enumerate(models):
        topo = per_model_signal[m]["topology"]
        domains = topo["domains"]
        matrix = topo["matrix"]
        traces.append({
            "type": "heatmap",
            "z": matrix,
            "x": domains,
            "y": domains,
            "colorscale": [
                [0.0, "#1a3b5c"], [0.25, "#4e7da8"], [0.5, "#bda684"],
                [0.75, "#d4a85c"], [1.0, "#7a2e2e"],
            ],
            "visible": i == 0,
            "colorbar": {"title": {"text": "d̄", "side": "right"},
                         "thickness": 12, "len": 0.85},
            "hovertemplate": f"{m}<br>%{{y}} × %{{x}}<br>d̄ = %{{z:.3f}}<extra></extra>",
            "showscale": True,
        })
        visibility = [j == i for j in range(len(models))]
        buttons.append({"label": f"{m}  ({MODEL_GROUP_LABEL[m]})", "method": "update",
                        "args": [{"visible": visibility},
                                 {"title.text": f"Topologia per dominio · {m} ({MODEL_GROUP_LABEL[m]})"}]})
    layout = _base_layout(
        height=500,
        title={"text": f"Topologia per dominio · {models[0]} ({MODEL_GROUP_LABEL[models[0]]})",
               "font": {"size": 13}},
        margin={"l": 140, "r": 30, "t": 70, "b": 130},
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": None, "tickangle": -30, "automargin": True},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": None, "autorange": "reversed", "automargin": True},
        updatemenus=[{"buttons": buttons, "type": "dropdown", "direction": "down",
                      "x": 0.0, "xanchor": "left", "y": 1.15, "yanchor": "top",
                      "bgcolor": "#fff", "bordercolor": ui.PLOT_COLORS["border"]}],
    )
    return {"data": traces, "layout": layout}


def fig_rsa_forest(section_314):
    """Forest plot of 17 pair-wise ρ with 95% CI, colored by group."""
    groups = [
        ("within_weird",    "intra-WEIRD",     ui.PLOT_COLORS["weird"]),
        ("within_sinic",    "intra-Sinic",     ui.PLOT_COLORS["sinic"]),
        ("within_bilingual","intra-bilingue",  ui.PLOT_COLORS["bilingual"]),
        ("cross_tradition", "cross-tradizione", ui.PLOT_COLORS["cross"]),
    ]
    traces = []
    y_labels = []
    y_pos = 0
    # We lay pairs bottom-up, so y_pos grows while we iterate top-to-bottom.
    # Collect them first then assign y values in reverse.
    rows = []
    for key, label, color in groups:
        pairs = section_314.get(key, [])
        for p in pairs:
            rows.append({
                "label": f"{p['model_a']} × {p['model_b']}",
                "group": label,
                "rho": p["rho"],
                "ci_low": p["ci_low"],
                "ci_high": p["ci_high"],
                "p_holm": p.get("p_holm", p.get("p_value")),
                "color": color,
            })
    # Order: top-to-bottom = groups order. Plotly y is numeric.
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - i  # top row gets largest y
        y_labels.append((y, r["label"]))
    # Traces: one per group for legend clarity.
    by_group = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r["group"], []).append((n - i, r))
    for group_label, entries in by_group.items():
        ys = [y for y, _ in entries]
        xs = [r["rho"] for _, r in entries]
        err_plus = [r["ci_high"] - r["rho"] for _, r in entries]
        err_minus = [r["rho"] - r["ci_low"] for _, r in entries]
        labels = [r["label"] for _, r in entries]
        text = [f"{r['label']}<br>ρ = {r['rho']:+.3f}<br>"
                f"CI 95% = [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]<br>"
                f"p = {r['p_holm']:.3g}"
                for _, r in entries]
        color = entries[0][1]["color"]
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": xs,
            "y": ys,
            "error_x": {"type": "data", "symmetric": False,
                        "array": err_plus, "arrayminus": err_minus,
                        "color": color, "thickness": 1.4, "width": 6},
            "marker": {"color": color, "size": 9, "line": {"color": "#fff", "width": 1}},
            "name": group_label,
            "text": text,
            "hovertemplate": "%{text}<extra></extra>",
        })
    layout = _base_layout(
        height=max(420, 28 * n + 80),
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "Spearman ρ  (block-bootstrap CI 95%)"},
               "zeroline": True, "zerolinecolor": "#999"},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": None,
               "tickmode": "array",
               "tickvals": [y for y, _ in y_labels],
               "ticktext": [lbl for _, lbl in y_labels],
               "automargin": True},
        legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"},
        margin={"l": 220, "r": 30, "t": 30, "b": 80},
    )
    return {"data": traces, "layout": layout}


def _pair_group_label(pair, section_314):
    """Return (group_label_italian, observed_rho) for a pair if found
    anywhere in section_314; (None, None) otherwise."""
    group_italian = {
        "within_weird":     "intra-WEIRD",
        "within_sinic":     "intra-Sinic",
        "within_bilingual": "intra-bilingue",
        "cross_tradition":  "cross-tradizione",
    }
    for key, label in group_italian.items():
        for p in section_314.get(key, []):
            if (p["model_a"], p["model_b"]) == pair or (p["model_b"], p["model_a"]) == pair:
                return label, p["rho"]
    return None, None


def fig_null_distributions_dropdown(null_distributions, section_314, default_pair=None):
    """Mantel null distribution panel with a dropdown over all 17 pairs.

    Each pair contributes two traces: the histogram of its 1000 null ρ and
    a vertical line at its observed ρ. A dropdown shows one pair at a time.
    Returns (figure_dict, list_of_pair_summaries).
    """
    # Keep pair order grouped by category, for the dropdown ordering.
    category_order = ("within_weird", "within_sinic", "within_bilingual", "cross_tradition")
    ordered_pairs = []  # list of (pair_tuple, group_label, observed_rho, p_value)
    for cat in category_order:
        for p in section_314.get(cat, []):
            key = (p["model_a"], p["model_b"])
            rev = (p["model_b"], p["model_a"])
            if key in null_distributions:
                pair_key = key
            elif rev in null_distributions:
                pair_key = rev
            else:
                continue
            group_it = {"within_weird": "intra-WEIRD", "within_sinic": "intra-Sinic",
                        "within_bilingual": "intra-bilingue", "cross_tradition": "cross-tradizione"}[cat]
            arr = np.asarray(null_distributions[pair_key]["null"], dtype=float)
            n_at_or_above = int(np.sum(arr >= p["rho"]))
            p_mantel = (1 + n_at_or_above) / (1 + len(arr))
            ordered_pairs.append((pair_key, group_it, float(p["rho"]), float(p_mantel), arr))

    # Default pair: the canonical cross BGE-EN × BGE-ZH if present, else first.
    default_idx = 0
    canonical = default_pair or ("BGE-EN-large", "BGE-ZH-large")
    for i, (pk, *_ ) in enumerate(ordered_pairs):
        if pk == canonical or pk == canonical[::-1]:
            default_idx = i
            break

    group_color = {
        "intra-WEIRD":       ui.PLOT_COLORS["weird"],
        "intra-Sinic":       ui.PLOT_COLORS["sinic"],
        "intra-bilingue":    ui.PLOT_COLORS["bilingual"],
        "cross-tradizione":  ui.PLOT_COLORS["cross"],
    }

    traces = []
    for i, (pair_key, group_it, obs_rho, p_mantel, arr) in enumerate(ordered_pairs):
        color = group_color[group_it]
        traces.append({
            "type": "histogram",
            "x": arr.tolist(),
            "nbinsx": 40,
            "marker": {"color": "#cfd9e7", "line": {"color": color, "width": 0.5}},
            "name": "null (permutazioni)",
            "hovertemplate": "ρ_null ∈ %{x}<br>n = %{y}<extra></extra>",
            "visible": i == default_idx,
            "showlegend": False,
        })
        y_top = max(80, len(arr) // 8)
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": [obs_rho, obs_rho],
            "y": [0, y_top],
            "line": {"color": color, "width": 3, "dash": "solid"},
            "name": f"ρ osservato = {obs_rho:+.3f}",
            "hovertemplate": f"ρ osservato = {obs_rho:+.3f}<br>p = {p_mantel:.3g}<extra></extra>",
            "visible": i == default_idx,
            "showlegend": False,
        })

    # Dropdown buttons (one per pair).
    buttons = []
    n_traces = len(traces)
    for i, (pair_key, group_it, obs_rho, p_mantel, _arr) in enumerate(ordered_pairs):
        vis = [False] * n_traces
        vis[2 * i] = True
        vis[2 * i + 1] = True
        buttons.append({
            "label": f"{pair_key[0]} × {pair_key[1]}  ·  {group_it}",
            "method": "update",
            "args": [
                {"visible": vis},
                {"title.text": (f"Distribuzione nulla Mantel · {pair_key[0]} × {pair_key[1]} "
                                f"({group_it}) · ρ = {obs_rho:+.3f} · p = {p_mantel:.3g}")},
            ],
        })

    default = ordered_pairs[default_idx]
    layout = _base_layout(
        height=420,
        title={"text": (f"Distribuzione nulla Mantel · {default[0][0]} × {default[0][1]} "
                        f"({default[1]}) · ρ = {default[2]:+.3f} · p = {default[3]:.3g}"),
               "font": {"size": 13}},
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "Spearman ρ sotto permutazione"}},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "conteggio (B = 1000)"}, "autorange": True},
        barmode="overlay",
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
        updatemenus=[{"buttons": buttons, "type": "dropdown", "direction": "down",
                      "x": 0.0, "xanchor": "left", "y": 1.18, "yanchor": "top",
                      "bgcolor": "#fff", "bordercolor": ui.PLOT_COLORS["border"]}],
    )
    summaries = [{"pair": pk, "group": gi, "rho": rh, "p": pm}
                 for pk, gi, rh, pm, _ in ordered_pairs]
    return {"data": traces, "layout": layout}, summaries


def fig_categorical_probe_forest(probe):
    """Forest plot of per-model ensemble ρ for the 5 categorical probes,
    with a colour per tradition (WEIRD / Sinic / bilingual)."""
    tests = probe["tests"]
    test_order = [
        ("test_1_age_imputability",         "T1 · imputabilità penale (età)"),
        ("test_2_magnitude_negative_control", "T2 · magnitude (controllo negativo)"),
        ("test_3_age_contractual_capacity", "T3 · capacità contrattuale (età)"),
        ("test_4_offence_severity",         "T4 · gravità del reato"),
        ("test_5_disposal_severity",        "T5 · severità della pena"),
    ]

    def _color(m):
        grp = MODEL_GROUP_LABEL.get(m, "WEIRD")
        if grp == "WEIRD":    return ui.PLOT_COLORS["weird"]
        if grp == "Sinic":    return ui.PLOT_COLORS["sinic"]
        if grp == "bilingue": return ui.PLOT_COLORS["bilingual"]
        return ui.PLOT_COLORS["control"]

    traces = []
    model_order = ALL_MODELS
    for t_key, t_label in test_order:
        if t_key not in tests:
            continue
        t = tests[t_key]
        xs, ys, colors, text = [], [], [], []
        for m in model_order:
            if m not in t["per_model"]:
                continue
            ens = t["per_model"][m]["ensemble"]
            xs.append(ens["mean_rho"])
            ys.append(t_label)
            colors.append(_color(m))
            text.append(f"{m} ({MODEL_GROUP_LABEL.get(m, '')})<br>"
                        f"ρ̄ = {ens['mean_rho']:+.3f}<br>"
                        f"max-gap = {ens['mean_max_gap']:.3f}")
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": xs,
            "y": ys,
            "marker": {"color": colors, "size": 10, "line": {"color": "#fff", "width": 1}, "opacity": 0.9},
            "text": text,
            "hovertemplate": "%{text}<extra></extra>",
            "name": t_label,
            "showlegend": False,
        })
    # Legend via proxy traces (one per tradition).
    for grp, col, lbl in [
        ("WEIRD",    ui.PLOT_COLORS["weird"],     "modelli WEIRD"),
        ("Sinic",    ui.PLOT_COLORS["sinic"],     "modelli Sinic"),
        ("bilingue", ui.PLOT_COLORS["bilingual"], "modelli bilingui"),
    ]:
        traces.append({
            "type": "scatter", "mode": "markers",
            "x": [None], "y": [None],
            "marker": {"color": col, "size": 10},
            "name": lbl, "showlegend": True,
        })
    layout = _base_layout(
        height=360,
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS,
               "title": {"text": "Spearman ρ ensemble  (template-mediato, 5 template)"},
               "zeroline": True, "zerolinecolor": "#999", "range": [-0.2, 1.0]},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": None, "automargin": True},
        margin={"l": 260, "r": 30, "t": 30, "b": 90},
        legend={"orientation": "h", "y": -0.24, "x": 0.5, "xanchor": "center"},
    )
    return {"data": traces, "layout": layout}


# --------------------------------------------------------------------------
# Prose sections (all non-interpretive)

def section_domanda():
    return (
        ui.section_open("domanda", "Domanda e metodo")
        + '<p class="lead">Ogni modello di linguaggio, ricevuto lo stesso lessico giuridico, costruisce una propria mappa interna in cui i termini più affini risultano più vicini e i più lontani più distanti. Lens I misura <em>il grado di accordo fra queste mappe</em>. Il confronto avviene tra modelli, non direttamente tra tradizioni giuridiche: ciò che viene quantificato è quanto due modelli, ciascuno con la sua geometria, ordinano allo stesso modo le stesse coppie di termini.</p>'
        + '<p>Per ogni modello si costruisce una ' + ui.metric_chip("g-rdm", "RDM") + ', cioè la tabella delle distanze fra tutte le coppie di termini nella sua mappa. La correlazione di '
        + ui.metric_chip("g-spearman", "Spearman") + ' fra due RDM restituisce un unico numero in \\([-1,+1]\\): misura quanto le 61&thinsp;075 coppie sono ordinate nello stesso modo dai due modelli. Un valore vicino a +1 significa ordinamenti quasi identici; un valore vicino a 0 significa ordinamenti scorrelati; un valore negativo significa ordinamenti invertiti. A ogni correlazione si affiancano una '
        + ui.metric_chip("g-mantel", "p-value Mantel") + ' (1000 permutazioni, per escludere che il valore osservato sia un artefatto casuale) e un '
        + ui.metric_chip("g-boot", "intervallo di confidenza bootstrap") + ' (1000 ricampionamenti fatti a livello di termine, per quantificare l\'incertezza del valore osservato).</p>'
        + ui.disclaimer(
            "<strong>Encoder bare.</strong> Ogni termine è codificato nella sua forma nuda, senza contesto d\'uso. "
            "I dati contestualizzati (enunciati ricavati dall\'e-Legislation di Hong Kong) sono in pre-computazione; "
            "quando disponibili, questa pagina sarà accompagnata da una versione parallela calcolata sugli stessi termini "
            "ma immersi nelle frasi della legislazione."
        )
        + ui.section_close()
    )


def section_pipeline():
    stages = [
        ("Termine",  "<strong>Termine.</strong> Il lessico di riferimento è il Bilingual Legal Glossary del Department of Justice di Hong Kong: 350 termini giuridici selezionati, 50 per ciascuno dei 7 domini (costituzionale, civile, penale, e così via), ciascuno disponibile nella forma inglese e nella forma cinese ufficiale. È il materiale comune su cui tutti i modelli vengono interrogati."),
        ("Codifica", "<strong>Codifica.</strong> Ogni termine viene dato in pasto a otto modelli (3 addestrati su corpora anglofoni, 3 su corpora sinofoni, 2 bilingui). Ciascun modello restituisce, per ogni termine, un " + ui.metric_chip("g-vector", "vettore") + ", cioè una stringa di numeri che rappresenta la posizione del termine nella mappa interna del modello. I modelli bilingui ne producono due per termine, uno per la forma inglese e uno per la forma cinese."),
        ("Geometria","<strong>Geometria.</strong> Dai vettori di un modello si ricava la sua " + ui.metric_chip("g-rdm", "matrice di dissimilarità 350×350") + ": in ogni cella c\'è la " + ui.metric_chip("g-cosdist", "distanza coseno") + " fra due termini, cioè quanto le loro direzioni nella mappa del modello divergono. La matrice è la fotografia completa della geometria relazionale che quel modello ha costruito a partire dal lessico giuridico."),
        ("Confronto","<strong>Confronto.</strong> Si prendono a due a due le 17 coppie di modelli rilevanti e, per ogni coppia, si calcola la correlazione di " + ui.metric_chip("g-spearman", "Spearman") + " fra le rispettive RDM. Ottenendo così una tabella di 17 numeri, ciascuno in \\([-1,+1]\\), che sintetizza il grado di concordanza fra le mappe di ciascuna coppia (3 intra-WEIRD, 3 intra-Sinic, 9 cross-tradizione, 2 intra-bilingue)."),
        ("Verifica", "<strong>Verifica.</strong> Per ogni correlazione si stabilisce se è attribuibile al caso o no. Il " + ui.metric_chip("g-mantel", "test di Mantel") + " mescola 1000 volte le etichette dei termini in una delle due matrici e ricalcola la correlazione, ottenendo una distribuzione di riferimento a cui confrontare il valore osservato. Il " + ui.metric_chip("g-boot", "block bootstrap") + " ricampiona 1000 volte i termini (non le coppie) per costruire un intervallo di confidenza al 95%."),
    ]
    return (
        ui.section_open("pipeline", "Pipeline")
        + '<p>Ogni numero riportato in questa pagina è il risultato di cinque operazioni svolte in sequenza. Cliccare su uno stadio ne espande il dettaglio tecnico.</p>'
        + ui.pipeline_diagram(stages)
        + ui.section_close()
    )


def _fmt_p(p):
    if p <= 1e-300:
        return "p &lt; 1e−300"
    if p < 1e-6:
        return f"p &lt; {p:.1e}"
    return f"p = {p:.3g}"


def section_domain_signal(per_model_signal, legal_vs_control):
    table_rows = []
    for m in ALL_MODELS:
        if m not in per_model_signal:
            continue
        s = per_model_signal[m]["stats"]
        lvc = legal_vs_control.get(m)
        lvc_cell = f"{lvc['effect_r']:+.3f}" if lvc is not None else "<span style='color:#aaa'>—</span>"
        table_rows.append([
            f"{m}  <span style='color:#888;font-size:0.82em'>({MODEL_GROUP_LABEL[m]})</span>",
            f"{s['median_x']:.3f}",
            f"{s['median_y']:.3f}",
            f"{s['effect_r']:+.3f}",
            _fmt_p(s["p_value"]),
            lvc_cell,
        ])
    violin_fig = fig_intra_vs_inter(per_model_signal)
    legal_fig = fig_legal_vs_control_effect(legal_vs_control)
    return (
        ui.section_open("311", "§3.1.1 — Distances within and between legal domains")
        + '<p>Prima di confrontare mappe diverse, occorre accertarsi che ciascuna mappa distingua almeno le partizioni più elementari del lessico giuridico: i sette domini (costituzionale, civile, penale, ecc.). Se un modello non colloca sistematicamente i termini di uno stesso dominio più vicini fra loro che a termini di domini differenti, vuol dire che non percepisce il diritto come un insieme articolato di rami; in quel caso qualunque misura più fine costruita sopra la sua geometria sarebbe priva di fondamento. Questa sezione è perciò il test di pre-requisito per le misure che seguono.</p>'
        + '<p>Si considerano due popolazioni di distanze:</p>'
        + '<ul><li><em>intra-dominio</em> (8&thinsp;575 coppie): entrambi i termini appartengono allo stesso dominio;</li>'
        + '<li><em>inter-dominio</em> (52&thinsp;500 coppie): i due termini appartengono a domini diversi.</li></ul>'
        + '<p>Per ogni coppia si calcola la ' + ui.metric_chip("g-cosdist", "distanza coseno") + ' nella mappa del modello. Le due popolazioni si confrontano con il test U di Mann-Whitney, un test non parametrico che non assume una forma particolare della distribuzione e si limita a chiedere se una delle due popolazioni tende ad avere valori sistematicamente più bassi dell\'altra. Alla statistica del test si accompagna una misura standard della forza dell\'effetto, la '
        + ui.metric_chip("g-effect-r", "r rank-biserial") + ', che varia in \\([-1,+1]\\): r vicino a 0 significa che le due popolazioni si sovrappongono e sono indistinguibili; r positivo significa che le distanze intra-dominio sono sistematicamente più piccole di quelle inter-dominio (il dominio viene colto); r negativo il contrario.</p>'
        + '<p>Un secondo test, costruito in modo analogo, confronta le 61&thinsp;075 coppie dei 350 termini giuridici con le 35&thinsp;000 coppie "termine giuridico × termine di controllo non giuridico" (350 × 100). La statistica di riferimento è la stessa (U di Mann-Whitney, effect r rank-biserial): r positivo significa che le coppie intra-legali sono in media più vicine delle coppie legale–non-legale (il modello colloca il lessico giuridico in una regione più coesa dello spazio rispetto al campione di controllo); r negativo significa il contrario; r = 0 significa popolazioni indistinguibili. Entrambi i test sono ora calcolati direttamente dai vettori del modello e coprono tutti e dieci i modelli (3 WEIRD + 3 Sinic + 2 BGE-M3 + 2 Qwen3).</p>'
        + '<h3>Distribuzioni intra-dominio vs inter-dominio (10 modelli, dati integrali)</h3>'
        + ui.plotly_embed(violin_fig, "fig-311-violin", 540)
        + ui.plot_caption("Per ciascun modello la figura giustappone due popolazioni di distanze: a sinistra (tinte chiare) le coppie intra-dominio, a destra (tinte scure) le coppie inter-dominio. La linea interna segna la media. I colori codificano il gruppo del modello (blu = WEIRD, rosso = Sinic, verde = bilingue). Più le due sagome risultano separate in verticale, più il modello distingue le coppie dello stesso ramo da quelle di rami diversi. I dati sono integrali: 8&thinsp;575 punti intra + 52&thinsp;500 punti inter per modello.")
        + '<h3>Statistiche per modello</h3>'
        + ui.data_table(
            ["Modello (gruppo)", "mediana intra", "mediana inter",
             "effect r (intra vs inter)", "p (Mann-Whitney)",
             "effect r (core vs controllo)"],
            table_rows,
            col_classes=["", "num", "num", "num strong", "num", "num strong"],
        )
        + '<h3>Effect size core giuridico vs. controllo non giuridico</h3>'
        + ui.plotly_embed(legal_fig, "fig-311-bar", 340)
        + ui.plot_caption("Ogni barra riporta la r rank-biserial per un modello. r positivo: le coppie di termini giuridici sono in media più vicine delle coppie legale–non-legale; r negativo: il contrario; r = 0: le due popolazioni di distanze sono indistinguibili. Il grafico copre tutti e dieci i modelli, ciascuno calcolato direttamente dal proprio file di vettori.")
        + ui.section_close()
    )


def section_domain_topology(per_model_signal):
    fig = fig_domain_topology(per_model_signal)
    return (
        ui.section_open("312", "§3.1.2 — Maps of distance across legal domains")
        + '<p>La §3.1.1 restituisce un unico numero per ciascun modello (r: il dominio è colto o no). Qui si apre il quadro: invece di aggregare tutte le coppie, si aggregano <em>per coppia di domini</em>. Il risultato è una tabella 7×7 di prossimità fra rami del diritto, leggibile come una carta delle distanze fra branche disciplinari secondo la mappa di un singolo modello.</p>'
        + '<p>Ogni cella contiene la media delle '
        + ui.metric_chip("g-cosdist", "distanze coseno") + ' fra tutte le coppie di termini in cui il primo termine appartiene al dominio della riga e il secondo al dominio della colonna. La diagonale (dominio × sé stesso) misura la <em>coesione interna</em> del dominio: valori bassi significano che i termini di quel ramo sono ben raggruppati nella mappa del modello. Fuori diagonale si legge la distanza media fra due rami distinti: valori bassi significano che il modello li colloca vicini, valori alti che li colloca distanti.</p>'
        + ui.plotly_embed(fig, "fig-312-heatmap", 520)
        + ui.plot_caption("Il menù a tendina in alto a sinistra consente di cambiare modello. La heatmap copre tutti e dieci i modelli (3 WEIRD + 3 Sinic + 2 BGE-M3 + 2 Qwen3), ciascuno calcolato direttamente dalla sua RDM 350×350. I valori sono distanze coseno (0 = vettori con la stessa direzione, 1 = ortogonali, 2 = direzioni opposte).")
        + ui.section_close()
    )


def section_rsa(section_314, null_distributions):
    s = section_314["summary"]
    forest = fig_rsa_forest(section_314)
    null_fig, summaries = fig_null_distributions_dropdown(null_distributions, section_314)
    # Pick the canonical pair's summary for the caption below the figure.
    default_summary = next(
        (s_ for s_ in summaries
         if s_["pair"] == ("BGE-EN-large", "BGE-ZH-large")
         or s_["pair"] == ("BGE-ZH-large", "BGE-EN-large")),
        summaries[0],
    )

    summary_table = ui.data_table(
        ["Categoria", "Simbolo", "ρ̄", "N coppie"],
        [
            ["intra-WEIRD",    "<em>ρ̄<sub>W</sub></em>",
                f"{s['mean_rho_within_weird']:+.3f}", "3"],
            ["intra-Sinic",    "<em>ρ̄<sub>S</sub></em>",
                f"{s['mean_rho_within_sinic']:+.3f}", "3"],
            ["intra-bilingue", "<em>ρ̄<sub>β</sub></em>",
                f"{s['mean_rho_within_bilingual']:+.3f}", "2"],
            ["cross-tradizione (monolinguale)", "<em>ρ̄<sub>cross</sub></em>",
                f"{s['mean_rho_cross']:+.3f}", "9"],
        ],
        col_classes=["", "", "num strong", "num"],
    )
    diffs_html = (
        '<p><strong>Differenze aritmetiche</strong> (solo sottrazioni tra medie, senza interpretazione):</p>'
        f'<ul>'
        f'<li>ρ̄<sub>W</sub> − ρ̄<sub>cross</sub> = {s["mean_rho_within_weird"] - s["mean_rho_cross"]:+.3f}</li>'
        f'<li>ρ̄<sub>S</sub> − ρ̄<sub>cross</sub> = {s["mean_rho_within_sinic"] - s["mean_rho_cross"]:+.3f}</li>'
        f'<li>ρ̄<sub>β</sub> − ρ̄<sub>cross</sub> = {s["mean_rho_within_bilingual"] - s["mean_rho_cross"]:+.3f}</li>'
        f'<li>ρ̄<sub>W</sub> − ρ̄<sub>β</sub> = {s["mean_rho_within_weird"] - s["mean_rho_within_bilingual"]:+.3f}</li>'
        f'</ul>'
    )

    return (
        ui.section_open("313", "§3.1.3 — Agreement between pairs of models")
        + '<p>È il cuore di Lens I. Due modelli diversi producono vettori di dimensione diversa: non si possono confrontare direttamente termine per termine, perché abitano spazi incommensurabili. La Representational Similarity Analysis aggira il problema spostando il confronto a un livello superiore: non si confrontano i vettori, si confrontano le <em>mappe di vicinanze</em> che i modelli costruiscono fra gli stessi termini. Due modelli "concordano" se, pur vivendo in spazi diversi, ordinano le 61&thinsp;075 coppie di termini dalla più vicina alla più lontana nello stesso modo.</p>'
        + '<p>In pratica: per ogni coppia di modelli (A, B) si prendono le rispettive '
        + ui.metric_chip("g-rdm", "RDM") + ' (la tabella delle distanze fra tutti i termini in ciascuna mappa) e si calcola la '
        + ui.metric_chip("g-spearman", "correlazione di Spearman") + ' fra le due. La ρ di Spearman restituisce un numero in \\([-1,+1]\\): vicino a +1 quando le due classifiche delle coppie coincidono, vicino a 0 quando sono indipendenti, negativo quando sono invertite. Con 8 modelli si ottengono 17 coppie rilevanti: 3 intra-WEIRD, 3 intra-Sinic, 9 cross-tradizione (tutte le combinazioni WEIRD × Sinic), 2 intra-bilingue (EN × ZH all\'interno di BGE-M3 e Qwen3). Ogni correlazione è corredata da '
        + ui.metric_chip("g-mantel", "p-value Mantel") + ' e '
        + ui.metric_chip("g-boot", "intervallo di confidenza bootstrap") + ' al 95%.</p>'

        + '<h3>Forest plot delle 17 correlazioni</h3>'
        + ui.plotly_embed(forest, "fig-313-forest", 560)
        + ui.plot_caption("Ogni punto è la ρ osservata fra i due modelli della coppia; la barra orizzontale è l\'intervallo di confidenza al 95% calcolato con block bootstrap a livello di termine. Il colore codifica la categoria della coppia. Passando il cursore sopra ciascun punto compare la p-value corretta per confronti multipli (Holm).")

        + '<h3>Come si decide se ρ è "reale" o potrebbe essere casuale (test di Mantel)</h3>'
        + '<p>Data una ρ osservata, ad esempio ρ = 0.2, come si stabilisce se quel valore è una proprietà effettiva delle due mappe o se potrebbe emergere per caso? La procedura (Mantel 1967) è la seguente: si conservano intatte le due RDM ma, in una delle due, si permutano casualmente le etichette dei 350 termini; si ricalcola ρ su questa versione scompaginata; si ripete l\'operazione 1000 volte. Si ottiene così una distribuzione di riferimento delle ρ compatibili con l\'ipotesi di assenza di relazione. La p-value è la frazione di ρ permutate che risultano uguali o superiori alla ρ osservata. Il menù a tendina in alto consente di selezionare qualunque delle 17 coppie: l\'istogramma mostra le 1000 ρ permutate, la linea verticale segna la ρ osservata di quella specifica coppia.</p>'
        + ui.plotly_embed(null_fig, "fig-313-null", 440)
        + ui.plot_caption(f"Coppia di default all\'apertura: {default_summary['pair'][0]} × {default_summary['pair'][1]} ({default_summary['group']}); ρ osservata = {default_summary['rho']:+.3f}; p = (1 + numero di ρ permutate ≥ ρ osservata) / (1 + B) = {default_summary['p']:.3g}. La stessa procedura è stata applicata a tutte le 17 coppie: in tutte la p risulta ≤ 0.001, soglia minima raggiungibile con B = 1000 permutazioni. Dropdown in alto a sinistra per ispezionare le altre coppie.")

        + '<h3>Come si quantifica l\'incertezza di ρ (block bootstrap a livello di termine)</h3>'
        + '<p>Per un test tradizionale (t-test) le osservazioni devono essere indipendenti. Le celle di una RDM non lo sono: ogni termine entra in 349 coppie (una con ciascuno degli altri 349). Trattare le 61&thinsp;075 celle come indipendenti sottostima l\'incertezza e produce intervalli di confidenza troppo stretti. La correzione standard (Nili et al. 2014) consiste nel ricampionare i <em>termini</em>, non le coppie: si estrae un campione casuale di termini con reimmissione, si rifiltrano entrambe le RDM sui termini estratti, si ricalcola ρ; ripetendo 1000 volte si ottiene una distribuzione della ρ e i suoi percentili 2.5% e 97.5% definiscono l\'intervallo di confidenza al 95%.</p>'

        + '<h3>Medie per categoria</h3>'
        + summary_table
        + diffs_html
        + '<p>Lo scarto osservato fra ρ̄<sub>W</sub> e ρ̄<sub>cross</sub> ammonta a ' + f"{s['cross_tradition_drop']:+.3f}" + ' punti di Spearman. La categoria intra-bilingue ρ̄<sub>β</sub> è una misura di controllo: i modelli bilingui (BGE-M3, Qwen3) codificano la forma inglese e la forma cinese di ogni termine dentro lo stesso spazio, con la stessa architettura. Confrontare ρ̄<sub>β</sub> con ρ̄<sub>cross</sub> serve a separare aritmeticamente due fattori che nel cross-tradizione sono sovrapposti: il fatto che due modelli distinti producono due geometrie distinte, e il fatto che due lingue legate a due tradizioni producono due geometrie distinte. Questa pagina si limita a esporre le differenze; il peso relativo dei due fattori non viene qui deciso.</p>'
        + ui.section_close()
    )


def _probe_test_details_html(t_key: str, t: dict, label: str, polarity: str) -> str:
    """Render a single collapsible details.entry for one probe test,
    exposing: legal threshold, 11 ordered categories (EN + ZH side by side),
    the 5 EN templates, the 5 ZH templates, and the pre-registered break
    position.
    """
    cats_en = t.get("categories_en", [])
    cats_zh = t.get("categories_zh", [])
    tmpls_en = t.get("templates_en", [])
    tmpls_zh = t.get("templates_zh", [])
    exp_gap = t.get("expected_gap_index")
    exp_brk_en = t.get("expected_break_en") or []
    exp_brk_zh = t.get("expected_break_zh") or []
    legal = t.get("legal_threshold") or "—"
    borderline = t.get("borderline", False)
    borderline_note = t.get("borderline_note")
    dist_mid = t.get("distance_from_midpoint")
    polarity_badge = (
        '<span style="background:#e5ecf4;color:#2c5f9a;font-variant:small-caps;letter-spacing:0.05em;'
        'padding:0.1rem 0.5rem;border-radius:10px;font-size:0.72rem;margin-left:0.5rem;">positivo</span>'
        if polarity == "positivo"
        else '<span style="background:#f4e5e5;color:#a43a3a;font-variant:small-caps;letter-spacing:0.05em;'
             'padding:0.1rem 0.5rem;border-radius:10px;font-size:0.72rem;margin-left:0.5rem;">controllo negativo</span>'
    )

    # Categories side-by-side table with the expected break marked.
    cat_rows = []
    for i, (ce, cz) in enumerate(zip(cats_en, cats_zh)):
        is_pre = i == exp_gap - 1 if isinstance(exp_gap, int) else False
        is_post = i == exp_gap if isinstance(exp_gap, int) else False
        idx_cell = f"<td class='num'>{i}</td>"
        en_cell = f"<td>{ce}</td>"
        zh_cell = f'<td class="zh">{cz}</td>'
        if is_pre:
            en_cell = f'<td style="border-bottom:2px solid #b08d57;">{ce}</td>'
            zh_cell = f'<td class="zh" style="border-bottom:2px solid #b08d57;">{cz}</td>'
            idx_cell = f'<td class="num" style="border-bottom:2px solid #b08d57;">{i}</td>'
        if is_post:
            en_cell = f'<td style="border-top:2px solid #b08d57;"><strong>{ce}</strong></td>'
            zh_cell = f'<td class="zh" style="border-top:2px solid #b08d57;"><strong>{cz}</strong></td>'
            idx_cell = f'<td class="num" style="border-top:2px solid #b08d57;"><strong>{i}</strong></td>'
        cat_rows.append(f"<tr>{idx_cell}{en_cell}{zh_cell}</tr>")
    cats_table = (
        '<table class="data"><thead><tr><th>Indice</th><th>Categoria EN</th><th>Categoria ZH</th></tr></thead>'
        '<tbody>' + "".join(cat_rows) + '</tbody></table>'
    )

    # Templates list. Highlight {category} inline.
    def _tmpl_li(s: str, is_zh: bool = False) -> str:
        highlighted = s.replace(
            "{category}",
            '<span style="background:#faf7ee;color:#8a6d3b;padding:0.05rem 0.35rem;'
            'border:1px solid #e5e2d8;border-radius:3px;font-style:italic;">{category}</span>',
        )
        cls = " class='zh'" if is_zh else ""
        return f'<li{cls} style="padding:0.25rem 0;">{highlighted}</li>'
    tmpls_en_html = "".join(_tmpl_li(s) for s in tmpls_en)
    tmpls_zh_html = "".join(_tmpl_li(s, is_zh=True) for s in tmpls_zh)

    body = (
        f'<p><strong>Etichetta</strong>: {t.get("label", "")}  {polarity_badge}</p>'
        f'<p><strong>Soglia giuridica pre-registrata</strong>: {legal}</p>'
        f'<p><strong>Posizione attesa del salto (expected_gap_index)</strong>: '
        f'tra l\'indice {exp_gap - 1 if isinstance(exp_gap, int) else "—"} e l\'indice {exp_gap if isinstance(exp_gap, int) else "—"}'
        + (f', ossia tra « {exp_brk_en[0]} » e « {exp_brk_en[1]} » (EN) '
           f'oppure «&nbsp;<span class="zh">{exp_brk_zh[0]}</span>&nbsp;» e «&nbsp;<span class="zh">{exp_brk_zh[1]}</span>&nbsp;» (ZH).'
           if len(exp_brk_en) == 2 and len(exp_brk_zh) == 2 else ".")
        + '</p>'
        + (f'<p><strong>Distanza dal mezzo della scala</strong>: {dist_mid}  '
           f'<span style="color:#777;font-size:0.88em;">'
           f'(la scala a 11 categorie ha il mezzo all\'indice 4; un salto atteso in quella posizione sarebbe confuso con l\'artefatto strutturale del max-gap; '
           f'la pre-registrazione richiede distanza ≥ 2, rispettata qui.)</span></p>'
           if isinstance(dist_mid, int) else "")
        + (f'<div class="disclaimer"><strong>Nota borderline</strong>: {borderline_note}</div>'
           if borderline and borderline_note else "")
        + '<h4 style="margin-top:1rem;">Le 11 categorie ordinate</h4>'
        + '<p style="font-size:0.88rem;color:#555;">La riga dorata orizzontale indica il confine giuridico pre-registrato: '
          "la categoria in grassetto è il primo termine sopra la soglia; quella immediatamente sopra è l'ultimo termine sotto la soglia.</p>"
        + cats_table
        + '<h4 style="margin-top:1rem;">I 5 template frasali (EN)</h4>'
        + '<p style="font-size:0.88rem;color:#555;">Ogni template contiene il placeholder '
          '<code>{category}</code> sostituito, a run-time, con ciascuna delle 11 categorie; '
          'il modello produce 11 vettori per template, ciascuno proiettato sull\'asse PC1 delle 11 posizioni.</p>'
        + f'<ul style="list-style:decimal;padding-left:1.5rem;">{tmpls_en_html}</ul>'
        + '<h4 style="margin-top:1rem;">I 5 template frasali (ZH)</h4>'
        + f'<ul style="list-style:decimal;padding-left:1.5rem;">{tmpls_zh_html}</ul>'
    )
    summary_main = f"<span style='font-variant:small-caps;letter-spacing:0.04em;'>{label}</span>"
    return ui.details_entry(f"probe-{t_key}", summary_main, "", body)


def section_categorical_probe(probe):
    tests = probe["tests"]
    fig = fig_categorical_probe_forest(probe)

    # Table: per test, aggregate ensemble stats
    rows = []
    test_labels = [
        ("test_1_age_imputability",           "T1 · età e imputabilità penale",    "positivo"),
        ("test_2_magnitude_negative_control", "T2 · ordine di grandezza (controllo)","negativo"),
        ("test_3_age_contractual_capacity",   "T3 · età e capacità contrattuale",  "positivo"),
        ("test_4_offence_severity",           "T4 · gravità del reato",            "positivo"),
        ("test_5_disposal_severity",          "T5 · severità della pena",          "positivo"),
    ]
    for t_key, label, polarity in test_labels:
        if t_key not in tests:
            continue
        t = tests[t_key]
        s = t["summary"]
        rows.append([
            label,
            polarity,
            f"{s['mean_ensemble_rho']:+.3f}",
            f"{s['mean_ensemble_max_gap']:.3f}",
            f"{s['n_models_exact_hit']}/{s['n_models_total']}",
            f"{s['n_models_near_hit']}/{s['n_models_total']}",
        ])
    probe_table = ui.data_table(
        ["Test", "Polarità attesa", "ρ̄ ensemble", "max-gap ensemble", "exact hits", "near hits"],
        rows,
        col_classes=["", "", "num", "num", "num", "num"],
    )

    # Details accordion: one collapsible per test with full content
    # (11 categories EN+ZH, 5 templates EN + 5 templates ZH, legal threshold,
    # pre-registered break position). Expanding a test reveals exactly the
    # materials the model was interrogated with.
    details_blocks = []
    for t_key, label, polarity in test_labels:
        if t_key not in tests:
            continue
        details_blocks.append(_probe_test_details_html(t_key, tests[t_key], label, polarity))

    return (
        ui.section_open("314", "§3.1.4 — Pre-registered ordinal probes")
        + '<p>Una sonda parametrica pre-registrata è un esperimento congegnato in due momenti distinti. Prima di guardare i dati si fissano, per iscritto: (i) una scala di 11 categorie ordinate con un significato giuridico chiaro (ad esempio 11 reati in ordine crescente di gravità, oppure 11 età in anni); (ii) un asse di proiezione, costruito con una famiglia di 5 frasi modello in cui la categoria compare in una posizione sintattica fissa ("X è punito con..."); (iii) un confine giuridicamente rilevante nella scala (ad esempio la soglia fra summary e indictable, o l\'età di 10 o 18 anni). Solo dopo si interroga il modello e si verifica se, proiettando le 11 categorie sull\'asse, il "salto" maggiore fra categorie consecutive cade proprio al confine pre-registrato.</p>'
        + '<p>Le grandezze calcolate sono:</p>'
        + '<ul>'
        + '<li><strong>ρ ensemble</strong>: media, sui 5 template, della correlazione di Spearman fra l\'ordine delle categorie lungo l\'asse di proiezione e l\'ordine giuridico pre-registrato. Vicino a +1: il modello replica l\'ordine; vicino a 0: nessuna relazione; negativo: ordine invertito;</li>'
        + '<li><strong>max-gap</strong>: la posizione del massimo scarto fra categorie consecutive sull\'asse, messa a confronto con la posizione attesa (il confine giuridico);</li>'
        + '<li><strong>exact hits / near hits</strong>: numero di modelli per cui il max-gap cade esattamente al confine atteso o, rispettivamente, entro una posizione di scarto.</li>'
        + '</ul>'
        + '<p>Il test 2 (ordine di grandezza di cifre in denaro) è un <em>controllo negativo</em>: non è un asse giuridicamente pre-registrato e non dovrebbe produrre ρ elevate. Serve da soglia di riferimento: se anche in un test senza struttura giuridica pre-registrata emergessero ρ alte, il paradigma sarebbe sospetto.</p>'

        + '<h3>Risultati per test e per modello</h3>'
        + ui.plotly_embed(fig, "fig-314-forest", 340)
        + ui.plot_caption("Ogni punto è la ρ ensemble di un singolo modello per un singolo test; il colore distingue il gruppo del modello (blu WEIRD, rosso Sinic, verde bilingue). Valori più prossimi a 1 significano che l\'ordine sulla proiezione riproduce l\'ordine giuridico pre-registrato; valori prossimi a 0 significano assenza di corrispondenza. I modelli bilingui sono rappresentati dalle loro due incarnazioni linguistiche: BGE-M3-EN codifica i template inglesi, BGE-M3-ZH i template cinesi, e analogamente per Qwen3-0.6B.")

        + '<h3>Aggregato per test</h3>'
        + probe_table

        + '<h3>Materiali pre-registrati di ciascun test</h3>'
        + '<p>Ogni test apre un accordion che espone <em>integralmente</em> il materiale interrogato: la soglia giuridica di riferimento, le 11 categorie ordinate nelle due lingue (con il confine pre-registrato contrassegnato da una linea dorata fra le due righe pertinenti), e i 5 template frasali nei quali la categoria è iniettata al posto del placeholder <code>{category}</code>. Tutto questo è fissato <em>prima</em> della run; la run stessa si limita a calcolare proiezioni e salti.</p>'
        + "".join(details_blocks)
        + '<p style="margin-top:1rem;color:#777;font-size:0.88rem;">Il dettaglio per singolo template (n = 5 per modello per test, ρ + max-gap individuali) è conservato in <code>categorical_probe.json</code>; il file YAML pre-registrato che fissa categorie, template e confini è <code>experiments/lens_1_relational/categorical_probe_expected.yaml</code>.</p>'
        + ui.section_close()
    )


def section_reproducibility(results_meta):
    d = results_meta
    return (
        ui.section_open("tecnica", "Tecnica riproducibile")
        + '<p>Tutti i numeri in questa pagina provengono dalla stessa esecuzione, condotta a partire dallo stesso snapshot di dati, dagli stessi modelli e dagli stessi semi pseudo-casuali (un seme fissa l\'esito dei campionamenti casuali in modo che la run sia ripetibile bit per bit).</p>'
        + ui.data_table(
            ["Parametro", "Valore"],
            [
                ["Data della run",                                    d["date"]],
                ["N permutazioni del test di Mantel",                 str(d["n_perm"])],
                ["N iterazioni del block bootstrap",                  str(d["n_boot"])],
                ["k (classificatore k-NN per la selezione background)", str(d["k_nn"])],
                ["Modelli WEIRD",                                     ", ".join(d["weird_models"])],
                ["Modelli Sinic",                                     ", ".join(d["sinic_models"])],
                ["Modelli bilingui",                                  ", ".join(d["bilingual_models"])],
                ["Tempo di esecuzione (secondi)",                     f"{d['elapsed_seconds']:.1f}"],
            ],
            col_classes=["", "num"],
        )
        + '<p><strong>Codice</strong>. Modulo principale: <code>experiments/lens_1_relational/lens1.py</code>. '
        + 'Funzioni rilevanti: <code>run_section_311</code>, <code>run_section_31</code>, <code>run_section_314</code>, '
        + '<code>run_section_315</code>. I risultati numerici sono archiviati in <code>experiments/lens_1_relational/results/lens1_results.json</code>; '
        + 'le distribuzioni nulle e bootstrap per ogni coppia di modelli in <code>results/distributions/</code>; le distribuzioni '
        + 'di distanze intra-dominio, inter-dominio, core giuridico e controllo per modello in <code>results/distances/</code>; '
        + 'i risultati della sonda categoriale in <code>results/categorical_probe.json</code>.</p>'
        + ui.section_close()
    )


def section_footer_page():
    body = (
        'Source code e storia dei commit: <a href="https://github.com/capazme/GeometriaIuris">'
        'github.com/capazme/GeometriaIuris</a> · dataset anchor <code>8480ea5</code>, '
        'assi <code>ad70775</code>, archivio D9 <code>e5adc12</code>. '
        'Trace di decisione: <code>experiments/lens_1_relational/trace.md</code>. '
        'Versione encoder bare.'
    )
    return body


# --------------------------------------------------------------------------
# Main

def build():
    results, probe, per_model_signal, legal_vs_control, null_distributions, _ = load_all()
    title = "Geometria Iuris · §3.1 — Comparing how models organise the legal lexicon"
    subtitle = "RSA sui 350 termini del HK DoJ Glossary · encoder bare · " + results["meta"]["date"][:10]

    nav_items = [
        ("#domanda",   "Domanda"),
        ("#pipeline",  "Pipeline"),
        ("#311",       "§3.1.1 distanze"),
        ("#312",       "§3.1.2 mappa"),
        ("#313",       "§3.1.3 accordo"),
        ("#314",       "§3.1.4 sonde"),
        ("#tecnica",   "Tecnica"),
        ("#glossary",  "Glossario"),
    ]

    html_parts = [
        ui.page_head(title, subtitle),
        ui.sticky_nav(nav_items, back_link=("index.html", "↑ index")),
        ui.open_main(),
        section_domanda(),
        section_pipeline(),
        section_domain_signal(per_model_signal, legal_vs_control),
        section_domain_topology(per_model_signal),
        section_rsa(results["section_314"], null_distributions),
        section_categorical_probe(probe),
        section_reproducibility(results["meta"]),
        ui.glossary_section([
            "g-vector", "g-cosdist", "g-rdm", "g-spearman",
            "g-mantel", "g-boot", "g-effect-r",
            "g-crosstrad", "g-withintrad", "g-bilingual",
        ]),
        ui.page_footer(section_footer_page()),
    ]

    OUT.write_text("".join(html_parts), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build()
