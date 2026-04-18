"""Generator for dashboard_v2/lens4.html.

Reads the Lens IV result JSON, the `scores/` folder with per-model per-axis
projection scores, and `value_axes.yaml`, then emits a single
self-contained HTML page matching dashboard_v2/index.html's visual
language. Non-interpretive throughout.

Run:

    python experiments/dashboard_v2/generate_lens4.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import shared_ui as ui  # noqa: E402

LENS4 = REPO / "experiments" / "lens_4_values" / "results"
AXES_YAML = REPO / "experiments" / "lens_4_values" / "value_axes.yaml"
OUT = HERE / "lens4.html"

WEIRD_MODELS = ("BGE-EN-large", "E5-large", "FreeLaw-EN", "BGE-M3-EN", "Qwen3-0.6B-EN")
SINIC_MODELS = ("BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH", "BGE-M3-ZH", "Qwen3-0.6B-ZH")
BILINGUAL_BASES = ("BGE-M3", "Qwen3-0.6B")

MODEL_GROUP_LABEL = {
    **{m: "WEIRD" for m in ("BGE-EN-large", "E5-large", "FreeLaw-EN")},
    **{m: "Sinic" for m in ("BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH")},
    **{m: "bilingue" for m in ("BGE-M3-EN", "BGE-M3-ZH", "Qwen3-0.6B-EN", "Qwen3-0.6B-ZH")},
}


def _classify_pair(model_a: str, model_b: str, group_from_lens4: str) -> str:
    """Relabel a pair emitted by lens4. When both halves of the pair come
    from the same bilingual base (e.g. BGE-M3-EN × BGE-M3-ZH) we move the
    pair from "cross" to a new "within_bilingual" bucket — that is the
    causal-control pair, not a cross-tradition pair.
    """
    for base in BILINGUAL_BASES:
        en_label = f"{base}-EN"
        zh_label = f"{base}-ZH"
        if {model_a, model_b} == {en_label, zh_label}:
            return "within_bilingual"
    return group_from_lens4

AXES_ORDER = (
    "individual_collective",
    "rights_duties",
    "public_private",
    "state_market",
    "natural_positive",
    "status_contract",
)

AXIS_LABELS = {
    "individual_collective": "individual ↔ collective",
    "rights_duties":         "rights ↔ duties",
    "public_private":        "public ↔ private",
    "state_market":          "state ↔ market",
    "natural_positive":      "natural ↔ positive",
    "status_contract":       "status ↔ contract",
}


# --------------------------------------------------------------------------
# Data loading

def parse_value_axes(path: Path) -> dict[str, dict[str, list[list[str]]]]:
    """Minimal YAML parser sufficient for the structured pole-pair file.

    Each axis block has `axis_name:` then `en_pairs:` / `zh_pairs:` each
    followed by 10 lines of `    - [pos, neg]`. Avoids a PyYAML dependency.
    """
    text = path.read_text(encoding="utf-8")
    axes: dict[str, dict[str, list[list[str]]]] = {}
    current_axis = None
    current_list = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-z_]+):\s*$", line)
        if m and line[0] != " " and m.group(1) not in ("en_pairs", "zh_pairs"):
            current_axis = m.group(1)
            axes[current_axis] = {"en_pairs": [], "zh_pairs": []}
            current_list = None
            continue
        m = re.match(r"^  (en_pairs|zh_pairs):\s*$", line)
        if m:
            current_list = m.group(1)
            continue
        m = re.match(r"^\s+-\s*\[(.*?),\s*(.*?)\]\s*$", line)
        if m and current_axis and current_list:
            pos, neg = m.group(1).strip(), m.group(2).strip()
            axes[current_axis][current_list].append([pos, neg])
    return axes


def load_all():
    with (LENS4 / "lens4_results.json").open() as f:
        results = json.load(f)
    axes = parse_value_axes(AXES_YAML)
    scores: dict[tuple[str, str], np.ndarray] = {}
    for model in list(WEIRD_MODELS) + list(SINIC_MODELS):
        for axis in AXES_ORDER:
            p = LENS4 / "scores" / f"{model}_{axis}.npy"
            if p.exists():
                scores[(model, axis)] = np.load(p)
    return results, axes, scores


# --------------------------------------------------------------------------
# Plotly figures

def _base_layout(**overrides):
    layout = {**ui.PLOTLY_LAYOUT_DEFAULTS}
    layout["xaxis"] = {**ui.PLOTLY_AXIS_DEFAULTS}
    layout["yaxis"] = {**ui.PLOTLY_AXIS_DEFAULTS}
    layout.update(overrides)
    return layout


def fig_sanity_heatmap(section_331):
    """6×6 sanity-pass heatmap: models × axes, each cell = sanity_pass / 20."""
    models = list(WEIRD_MODELS) + list(SINIC_MODELS)
    z = []
    text = []
    for m in models:
        axes = section_331["per_model"][m]["axes"]
        z.append([axes[a]["sanity_pass"] / axes[a]["sanity_total"] for a in AXES_ORDER])
        text.append([f"{axes[a]['sanity_pass']}/{axes[a]['sanity_total']}" for a in AXES_ORDER])
    trace = {
        "type": "heatmap",
        "z": z,
        "x": [AXIS_LABELS[a] for a in AXES_ORDER],
        "y": models,
        "colorscale": [[0.0, "#7a2e2e"], [0.7, "#bda684"], [0.95, "#cfe0c6"], [1.0, "#3a6b1f"]],
        "zmin": 0.75,
        "zmax": 1.0,
        "text": text,
        "texttemplate": "%{text}",
        "textfont": {"size": 11, "color": "#1a1a1a"},
        "colorbar": {"title": {"text": "pass", "side": "right"}, "thickness": 12, "len": 0.85,
                     "tickformat": ".0%"},
        "hovertemplate": "%{y}<br>%{x}<br>sanity = %{text}<extra></extra>",
    }
    layout = _base_layout(
        height=360,
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": None, "tickangle": -25, "automargin": True},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": None, "autorange": "reversed", "automargin": True},
        margin={"l": 160, "r": 30, "t": 40, "b": 100},
    )
    return {"data": [trace], "layout": layout}


def fig_orthogonality(section_331):
    """6×6 symmetric cosine heatmap between axes, with model dropdown."""
    models = list(WEIRD_MODELS) + list(SINIC_MODELS)
    traces = []
    buttons = []
    for i, m in enumerate(models):
        ortho = section_331["per_model"][m]["orthogonality"]
        z = np.eye(len(AXES_ORDER))
        for r, a1 in enumerate(AXES_ORDER):
            for c, a2 in enumerate(AXES_ORDER):
                if r == c:
                    continue
                key1 = f"{a1}_vs_{a2}"
                key2 = f"{a2}_vs_{a1}"
                if key1 in ortho:
                    z[r][c] = ortho[key1]
                elif key2 in ortho:
                    z[r][c] = ortho[key2]
        traces.append({
            "type": "heatmap",
            "z": z.tolist(),
            "x": [AXIS_LABELS[a] for a in AXES_ORDER],
            "y": [AXIS_LABELS[a] for a in AXES_ORDER],
            "colorscale": [[0.0, "#7a2e2e"], [0.25, "#bda684"], [0.5, "#f4efe1"],
                           [0.75, "#9fb4d0"], [1.0, "#1a3b5c"]],
            "zmid": 0,
            "zmin": -0.6,
            "zmax": 1.0,
            "visible": i == 0,
            "colorbar": {"title": {"text": "cos", "side": "right"}, "thickness": 12, "len": 0.85},
            "hovertemplate": f"{m}<br>%{{y}} ↔ %{{x}}<br>cos = %{{z:.3f}}<extra></extra>",
        })
        visibility = [j == i for j in range(len(models))]
        buttons.append({"label": m, "method": "update",
                        "args": [{"visible": visibility},
                                 {"title.text": f"Coseni inter-asse · {m}"}]})
    layout = _base_layout(
        height=460,
        title={"text": f"Coseni inter-asse · {models[0]}", "font": {"size": 13}},
        margin={"l": 180, "r": 30, "t": 60, "b": 130},
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "tickangle": -25, "automargin": True, "title": None},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "autorange": "reversed", "automargin": True, "title": None},
        updatemenus=[{"buttons": buttons, "type": "dropdown", "direction": "down",
                      "x": 0.0, "xanchor": "left", "y": 1.13, "yanchor": "top",
                      "bgcolor": "#fff", "bordercolor": ui.PLOT_COLORS["border"]}],
    )
    return {"data": traces, "layout": layout}


def fig_axes_forest(per_pair):
    """2×3 faceted forest plot: one subplot per axis, pair-wise ρ with CI.

    Bilingual pairs (BGE-M3-EN × BGE-M3-ZH and Qwen3-0.6B-EN × Qwen3-0.6B-ZH)
    are reclassified here from lens4's "cross" bucket into a dedicated
    "within_bilingual" bucket, to be read as causal controls rather than
    generic cross-tradition pairs.
    """
    by_axis: dict[str, list[dict]] = {a: [] for a in AXES_ORDER}
    for p in per_pair:
        entry = dict(p)
        entry["group"] = _classify_pair(p["model_a"], p["model_b"], p["group"])
        by_axis[p["axis"]].append(entry)

    traces = []
    annotations = []

    group_colors = {
        "within_weird":     ui.PLOT_COLORS["weird"],
        "within_sinic":     ui.PLOT_COLORS["sinic"],
        "within_bilingual": ui.PLOT_COLORS["bilingual"],
        "cross":            ui.PLOT_COLORS["cross"],
    }
    group_labels = {
        "within_weird":     "intra-WEIRD",
        "within_sinic":     "intra-Sinic",
        "within_bilingual": "intra-bilingue (controllo β)",
        "cross":            "cross-tradizione",
    }

    for idx, axis in enumerate(AXES_ORDER):
        entries = by_axis[axis]
        group_rank = {"within_weird": 0, "within_sinic": 1, "within_bilingual": 2, "cross": 3}
        entries_sorted = sorted(
            entries,
            key=lambda e: (group_rank[e["group"]], e["model_a"], e["model_b"]),
        )
        n = len(entries_sorted)
        ys = list(range(n, 0, -1))

        for group_key in ("within_weird", "within_sinic", "within_bilingual", "cross"):
            sub = [(y, e) for y, e in zip(ys, entries_sorted) if e["group"] == group_key]
            if not sub:
                continue
            xs = [e["rho"] for _, e in sub]
            ys_sub = [y for y, _ in sub]
            err_plus = [e["ci_high"] - e["rho"] for _, e in sub]
            err_minus = [e["rho"] - e["ci_low"] for _, e in sub]
            text = [f"{e['model_a']} × {e['model_b']}<br>ρ = {e['rho']:+.3f}<br>"
                    f"CI 95% = [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]" for _, e in sub]
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": xs,
                "y": ys_sub,
                "error_x": {"type": "data", "symmetric": False,
                            "array": err_plus, "arrayminus": err_minus,
                            "color": group_colors[group_key], "thickness": 1.3, "width": 4},
                "marker": {"color": group_colors[group_key], "size": 7,
                           "line": {"color": "#fff", "width": 0.8}},
                "name": group_labels[group_key],
                "legendgroup": group_key,
                "showlegend": idx == 0,
                "text": text,
                "hovertemplate": "%{text}<extra></extra>",
                "xaxis": f"x{idx+1}",
                "yaxis": f"y{idx+1}",
            })

    # Now build the 2×3 grid layout. Plotly default: xaxis1 = xaxis
    # domains for 3 cols, 2 rows; horizontal spacing 0.06, vertical 0.18
    xdom = [(0.0, 0.30), (0.36, 0.66), (0.72, 1.0)]
    ydom = [(0.55, 1.0), (0.0, 0.45)]

    # Determine maximum pairs in any axis facet (with 10 models: 10+10+25 = 45,
    # minus 2 that moved to within_bilingual = 43 per facet).
    max_per_axis = max((len(v) for v in by_axis.values()), default=15)
    layout = _base_layout(
        height=max(780, 16 * max_per_axis + 220),
        margin={"l": 30, "r": 20, "t": 60, "b": 70},
        showlegend=True,
        legend={"orientation": "h", "y": -0.06, "x": 0.5, "xanchor": "center"},
    )
    for idx, axis in enumerate(AXES_ORDER):
        col = idx % 3
        row = idx // 3
        x_id = f"xaxis{idx+1}"
        y_id = f"yaxis{idx+1}"
        layout[x_id] = {**ui.PLOTLY_AXIS_DEFAULTS, "domain": list(xdom[col]),
                        "anchor": f"y{idx+1}",
                        "range": [-0.25, 0.95], "zeroline": True, "zerolinecolor": "#999"}
        layout[y_id] = {**ui.PLOTLY_AXIS_DEFAULTS, "domain": list(ydom[row]),
                        "anchor": f"x{idx+1}",
                        "showticklabels": False, "zeroline": False,
                        "range": [0, max_per_axis + 2]}
        xc = 0.5 * (xdom[col][0] + xdom[col][1])
        yc = ydom[row][1] + 0.02
        annotations.append({
            "text": f"<b>{AXIS_LABELS[axis]}</b>",
            "x": xc, "y": yc, "xref": "paper", "yref": "paper",
            "showarrow": False,
            "font": {"size": 12, "color": ui.PLOT_COLORS["accent_dark"]},
        })
    layout["annotations"] = annotations
    return {"data": traces, "layout": layout}


def fig_ranking_bar(summary_per_axis):
    """Horizontal bar: cross ρ̄ per axis, sorted ascending (most divergent top).

    Overlays within ρ̄ as a ghost marker for reference.
    """
    axes_sorted = sorted(AXES_ORDER, key=lambda a: summary_per_axis[a]["mean_cross_rho"])
    labels = [AXIS_LABELS[a] for a in axes_sorted]
    cross_vals = [summary_per_axis[a]["mean_cross_rho"] for a in axes_sorted]
    within_vals = [summary_per_axis[a]["mean_within_rho"] for a in axes_sorted]
    deltas = [within_vals[i] - cross_vals[i] for i in range(len(axes_sorted))]

    trace_cross = {
        "type": "bar",
        "orientation": "h",
        "y": labels,
        "x": cross_vals,
        "marker": {"color": ui.PLOT_COLORS["cross"], "line": {"color": "#fff", "width": 0.5}},
        "text": [f"ρ̄_cross = {v:+.3f}" for v in cross_vals],
        "textposition": "inside",
        "insidetextanchor": "start",
        "name": "ρ̄ cross",
        "hovertemplate": "%{y}<br>ρ̄ cross = %{x:+.3f}<extra></extra>",
    }
    trace_within = {
        "type": "scatter",
        "mode": "markers",
        "y": labels,
        "x": within_vals,
        "marker": {"color": ui.PLOT_COLORS["weird"], "size": 11, "symbol": "diamond",
                   "line": {"color": "#fff", "width": 1}},
        "name": "ρ̄ within",
        "hovertemplate": "%{y}<br>ρ̄ within = %{x:+.3f}<extra></extra>",
    }
    layout = _base_layout(
        height=360,
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "Spearman ρ̄"},
               "zeroline": True, "zerolinecolor": "#999", "range": [-0.05, 0.75]},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "automargin": True, "title": None},
        margin={"l": 180, "r": 30, "t": 30, "b": 60},
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
    )
    return {"data": [trace_cross, trace_within], "layout": layout}, deltas


def fig_divergent_dumbbell(scores, terms, n_top=18):
    """For each axis, identify the `n_top` terms with the largest |W̄ − S̄|
    and draw horizontal dumbbells: two dots (W̄, S̄) joined by a thin line.

    Returns a dict of figure by axis for the dropdown.
    """
    data_traces = []
    buttons = []
    per_axis_y = {}

    models_w = list(WEIRD_MODELS)
    models_s = list(SINIC_MODELS)

    # Precompute per-axis ordering and data
    all_data = {}
    for axis in AXES_ORDER:
        w_scores = np.stack([scores[(m, axis)] for m in models_w])
        s_scores = np.stack([scores[(m, axis)] for m in models_s])
        w_mean = w_scores.mean(axis=0)
        s_mean = s_scores.mean(axis=0)
        gap = np.abs(w_mean - s_mean)
        top = np.argsort(-gap)[:n_top]
        # Sort selected by gap magnitude (largest at top visually)
        top = top[np.argsort(-(w_mean[top] - s_mean[top]))]
        labels = [f"{terms[i]['en']} · {terms[i]['zh']}" for i in top]
        all_data[axis] = {"idx": top, "w": w_mean[top], "s": s_mean[top], "labels": labels}

    for ax_i, axis in enumerate(AXES_ORDER):
        d = all_data[axis]
        n = len(d["w"])
        ys = list(range(n, 0, -1))
        # Line trace
        lines_x = []
        lines_y = []
        for y, w, s in zip(ys, d["w"], d["s"]):
            lines_x += [w, s, None]
            lines_y += [y, y, None]
        data_traces.append({
            "type": "scatter", "mode": "lines",
            "x": lines_x, "y": lines_y,
            "line": {"color": "#bbb", "width": 1.2},
            "name": "connettore",
            "showlegend": False,
            "hoverinfo": "skip",
            "visible": ax_i == 0,
        })
        # WEIRD marker trace
        data_traces.append({
            "type": "scatter", "mode": "markers",
            "x": list(d["w"]), "y": ys,
            "marker": {"color": ui.PLOT_COLORS["weird"], "size": 10,
                       "line": {"color": "#fff", "width": 1}},
            "name": "ρ̄ WEIRD",
            "text": [f"{lbl}<br>W̄ = {w:+.3f}" for lbl, w in zip(d["labels"], d["w"])],
            "hovertemplate": "%{text}<extra></extra>",
            "showlegend": ax_i == 0,
            "visible": ax_i == 0,
        })
        # Sinic marker trace
        data_traces.append({
            "type": "scatter", "mode": "markers",
            "x": list(d["s"]), "y": ys,
            "marker": {"color": ui.PLOT_COLORS["sinic"], "size": 10,
                       "line": {"color": "#fff", "width": 1}, "symbol": "square"},
            "name": "ρ̄ Sinic",
            "text": [f"{lbl}<br>S̄ = {s:+.3f}" for lbl, s in zip(d["labels"], d["s"])],
            "hovertemplate": "%{text}<extra></extra>",
            "showlegend": ax_i == 0,
            "visible": ax_i == 0,
        })
        per_axis_y[axis] = d["labels"]

    # Build dropdown buttons that toggle visibility + tick labels
    traces_per_axis = 3
    n_total = len(AXES_ORDER) * traces_per_axis
    for ax_i, axis in enumerate(AXES_ORDER):
        vis = [False] * n_total
        for k in range(traces_per_axis):
            vis[ax_i * traces_per_axis + k] = True
        buttons.append({
            "label": AXIS_LABELS[axis],
            "method": "update",
            "args": [
                {"visible": vis},
                {
                    "yaxis.tickvals": list(range(len(per_axis_y[axis]), 0, -1)),
                    "yaxis.ticktext": per_axis_y[axis],
                    "title.text": f"Termini con il maggiore W̄ − S̄ · {AXIS_LABELS[axis]}",
                },
            ],
        })

    first_axis = AXES_ORDER[0]
    first_labels = per_axis_y[first_axis]
    layout = _base_layout(
        height=max(420, 24 * len(first_labels) + 120),
        title={"text": f"Termini con il maggiore W̄ − S̄ · {AXIS_LABELS[first_axis]}",
               "font": {"size": 13}},
        xaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": {"text": "punteggio medio lungo l'asse"},
               "zeroline": True, "zerolinecolor": "#999", "range": [-0.35, 0.35]},
        yaxis={**ui.PLOTLY_AXIS_DEFAULTS, "title": None, "automargin": True,
               "tickmode": "array",
               "tickvals": list(range(len(first_labels), 0, -1)),
               "ticktext": first_labels},
        margin={"l": 320, "r": 30, "t": 60, "b": 70},
        legend={"orientation": "h", "y": -0.13, "x": 0.5, "xanchor": "center"},
        updatemenus=[{"buttons": buttons, "type": "dropdown", "direction": "down",
                      "x": 0.0, "xanchor": "left", "y": 1.13, "yanchor": "top",
                      "bgcolor": "#fff", "bordercolor": ui.PLOT_COLORS["border"]}],
    )
    return {"data": data_traces, "layout": layout}


# --------------------------------------------------------------------------
# Prose sections

def section_domanda():
    return (
        ui.section_open("domanda", "Domanda e metodo")
        + '<p class="lead">Lens IV sceglie sei antitesi concettuali di lunga tradizione (individuale/collettivo, diritti/doveri, pubblico/privato, stato/mercato, naturale/positivo, status/contratto) e le trasforma in sei direzioni misurabili nello spazio interno di ciascun modello. Ogni termine, una volta proiettato su una di queste direzioni, riceve un punteggio in \\([-1,+1]\\) che ne indica la tendenza verso l\'uno o l\'altro polo. La misura riportata qui quantifica il grado di accordo fra due modelli sull\'ordinamento dei 350 termini lungo ciascuna direzione: il confronto avviene fra modelli, non direttamente fra tradizioni giuridiche.</p>'
        + '<p>Ogni asse è costruito alla <a class="metric" data-target="g-kozlowski">maniera di Kozlowski</a> (Kozlowski, Taddy e Evans 2019, <em>American Sociological Review</em>) come media delle differenze fra dieci coppie di poli (es. <em>individual / collective</em>, <em>citizen / state</em>, e otto altre coppie della stessa antitesi). La ragione per usare dieci coppie e non una sola è che ogni singola coppia porta con sé le idiosincrasie delle due parole scelte; mediando dieci coppie che esprimono la stessa opposizione concettuale, il risultato è una direzione più stabile, meno dipendente dalla scelta lessicale di partenza. Le dieci coppie inglesi sono usate dai modelli WEIRD, le dieci coppie cinesi dai modelli Sinic: i due assi per lo stesso concetto sono così co-costruiti nella rispettiva lingua, non tradotti uno dall\'altro.</p>'
        + '<p>A ciascun termine si associa, su ogni asse, un punteggio calcolato come '
        + ui.metric_chip("g-cosdist", "coseno") + ' fra il '
        + ui.metric_chip("g-vector", "vettore") + ' del termine e il vettore-asse (entrambi normalizzati). Si ottengono così, per ogni modello e ogni asse, 350 punteggi. Dati due modelli A e B, la correlazione di '
        + ui.metric_chip("g-spearman", "Spearman") + ' fra le loro due serie di 350 punteggi restituisce un unico numero in \\([-1,+1]\\): misura quanto i due modelli ordinano i termini allo stesso modo lungo quell\'asse. Valore vicino a +1: ordinamenti pressoché identici; valore vicino a 0: ordinamenti scorrelati; valore negativo: ordinamenti invertiti.</p>'
        + ui.disclaimer(
            "<strong>Encoder bare.</strong> Ogni termine è codificato nella sua forma nuda, senza contesto d\'uso. "
            "Una versione contestualizzata, ottenuta immergendo ciascun termine nelle frasi dell\'e-Legislation di Hong Kong, è in pre-computazione."
        )
        + ui.section_close()
    )


def section_pipeline():
    stages = [
        ("Poli",      "<strong>Poli.</strong> Per ciascuna delle sei antitesi concettuali si predispongono dieci coppie di parole-polo nella lingua del modello: dieci coppie inglesi per i modelli WEIRD e dieci coppie cinesi per i modelli Sinic. In totale 6 assi × 10 coppie × 2 lingue = 120 coppie di poli, selezionate sulla base di Kozlowski, Taddy e Evans 2019, Legrand 1996, Merryman e Pérez-Perdomo 2007."),
        ("Assi",      "<strong>Assi.</strong> Per ogni modello e ogni asse si calcolano le dieci differenze fra i due vettori di ciascuna coppia (polo positivo meno polo negativo) e se ne prende la media; il vettore risultante, normalizzato, è il <em>vettore-asse</em>: una freccia nello spazio del modello che punta dal polo negativo al polo positivo dell\'antitesi."),
        ("Proiezione","<strong>Proiezione.</strong> A ogni termine si associa un punteggio sull\'asse, calcolato come coseno fra il " + ui.metric_chip("g-vector", "vettore") + " del termine e il vettore-asse. Poiché entrambi sono normalizzati, il coseno equivale al prodotto scalare e cade in \\([-1,+1]\\): punteggio positivo indica che il termine tende verso il polo positivo dell\'asse, punteggio negativo verso il polo negativo, zero verso la zona neutra fra i due poli."),
        ("Correlazione", "<strong>Correlazione.</strong> Per ogni coppia ordinata di modelli (A, B) e ogni asse si calcola la correlazione di " + ui.metric_chip("g-spearman", "Spearman") + " fra le due serie di 350 punteggi. Ρ vicino a +1 significa che A e B ordinano i termini dallo stesso polo nello stesso modo; con 10 modelli (5 sul lato EN + 5 sul lato ZH) si ottengono 45 coppie × 6 assi = 270 correlazioni."),
        ("Aggregazione", "<strong>Aggregazione.</strong> Per ogni asse le 45 correlazioni si raggruppano in quattro categorie: " + ui.metric_chip("g-crosstrad", "cross") + " (le 25 coppie EN × ZH di tradizioni diverse), " + ui.metric_chip("g-withintrad", "within") + " intra-WEIRD (10) e intra-Sinic (10), e due coppie di controllo bilingue (BGE-M3-EN × BGE-M3-ZH e Qwen3-0.6B-EN × Qwen3-0.6B-ZH), in cui lo stesso modello codifica entrambe le lingue. Un test di permutazione rimescola le etichette di tradizione dei modelli e ricalcola la differenza Δ = ρ̄<sub>within</sub> − ρ̄<sub>cross</sub> per produrre una p-value."),
    ]
    return (
        ui.section_open("pipeline", "Pipeline")
        + '<p>Ogni numero di questa pagina è il risultato di cinque operazioni in sequenza. Cliccare su uno stadio ne espande il dettaglio tecnico.</p>'
        + ui.pipeline_diagram(stages)
        + ui.section_close()
    )


def section_axes_construction(axes_yaml, section_331):
    parts = [ui.section_open("331", "§3.3.1 — Sei assi, 120 coppie di poli")]
    parts.append('<p>Questa sezione mostra la costruzione materiale degli assi: per ciascuna delle sei antitesi concettuali sono elencate le dieci coppie di poli in inglese (usate dai tre modelli WEIRD) e le dieci coppie di poli in cinese (usate dai tre modelli Sinic). Accanto si riporta un controllo di coerenza, il <em>sanity check</em>: si costruisce il vettore-asse usando solo nove delle dieci coppie (leave-one-out) e si verifica che la decima coppia, esclusa dal calcolo, sia correttamente orientata rispetto al vettore risultante. Poiché ciascuna coppia ha un polo positivo e uno negativo, e si lasciano fuori a turno tutte e dieci, per asse si collezionano 20 verifiche: dieci sui poli positivi e dieci sui poli negativi.</p>')

    for axis in AXES_ORDER:
        en_pairs = axes_yaml.get(axis, {}).get("en_pairs", [])
        zh_pairs = axes_yaml.get(axis, {}).get("zh_pairs", [])
        en_list = "".join(
            f'<li><span class="pos">{p[0]}</span><span class="sep">↔</span><span class="neg">{p[1]}</span></li>'
            for p in en_pairs
        )
        zh_list = "".join(
            f'<li><span class="pos zh">{p[0]}</span><span class="sep">↔</span><span class="neg zh">{p[1]}</span></li>'
            for p in zh_pairs
        )
        body = (
            '<div class="pairs">'
            f'<div class="col"><h5>EN · 10 coppie</h5><ul>{en_list}</ul></div>'
            f'<div class="col"><h5>ZH · 10 coppie</h5><ul>{zh_list}</ul></div>'
            '</div>'
        )
        # Sanity per-model summary
        rows = []
        for m in list(WEIRD_MODELS) + list(SINIC_MODELS):
            sanity = section_331["per_model"][m]["axes"][axis]
            rows.append([m,
                         f"{sanity['positive_correct']}/10",
                         f"{sanity['negative_correct']}/10",
                         f"{sanity['sanity_pass']}/{sanity['sanity_total']}"])
        body += ui.data_table(
            ["Modello", "poli positivi corretti", "poli negativi corretti", "sanity totale"],
            rows,
            col_classes=["", "num", "num", "num strong"],
        )
        parts.append(ui.details_entry(
            f"axis-{axis}", AXIS_LABELS[axis], f"asse {AXES_ORDER.index(axis) + 1} di 6", body,
        ))

    parts.append('<h3>Heatmap sanity-pass per modello × asse</h3>')
    parts.append(ui.plotly_embed(fig_sanity_heatmap(section_331), "fig-331-sanity", 380))
    parts.append(ui.plot_caption(
        "Ogni cella riporta il rapporto <em>pass / totale</em>: il totale è 20 (dieci poli positivi più dieci poli negativi), "
        "il numeratore il numero di poli che, quando la loro coppia viene esclusa dal calcolo del vettore-asse, risultano "
        "comunque orientati dalla parte attesa. Valori prossimi a 20/20 segnalano un asse coerente lungo tutte le dieci coppie; "
        "valori più bassi segnalano coppie che contribuiscono in modo meno omogeneo alla direzione finale."
    ))
    parts.append(ui.section_close())
    return "".join(parts)


def section_orthogonality(section_331):
    fig = fig_orthogonality(section_331)
    return (
        ui.section_open("331b", "§3.3.1b — Ortogonalità inter-asse")
        + '<p>Le sei antitesi sono state scelte a priori dal ricercatore come direzioni teoricamente distinte. '
        + 'Ma nulla garantisce, a costruzione avvenuta, che siano anche direzioni <em>materialmente</em> distinte '
        + 'nello spazio di un modello: due assi diversi sul piano concettuale potrebbero, nella geometria interna di un encoder, '
        + 'cadere lungo la stessa linea o quasi. Questa sezione verifica la sovrapposizione mettendo ciascuna coppia di vettori-asse '
        + 'una di fronte all\'altra e misurandone il coseno. Il risultato è una matrice 6×6 simmetrica con 1 sulla diagonale '
        + '(ogni asse coincide con se stesso).</p>'
        + '<p>Regola di lettura: un coseno prossimo a 0 segnala che i due assi puntano in direzioni (quasi) indipendenti; '
        + 'un coseno prossimo a +1 che i due assi puntano nella stessa direzione (i due nomi, in quel modello, coprono la stessa geometria); '
        + 'un coseno prossimo a −1 che puntano in direzioni opposte. Il valore conta per la leggibilità dei risultati nelle sezioni successive: '
        + 'correlazioni calcolate su assi fortemente sovrapposti andrebbero lette con cautela, perché riproducono lo stesso segnale sotto etichette diverse.</p>'
        + ui.plotly_embed(fig, "fig-331b-ortho", 480)
        + ui.plot_caption("Il menù a tendina in alto a sinistra seleziona il modello. I valori in cella sono i coseni grezzi fra vettori-asse, "
                          "in \\([-1,+1]\\). La diagonale è fissa a 1 per costruzione.")
        + ui.section_close()
    )


def section_alignment(section_332):
    forest = fig_axes_forest(section_332["per_pair"])
    s = section_332["summary_per_axis"]
    rows = []
    for axis in AXES_ORDER:
        v = s[axis]
        rows.append([
            AXIS_LABELS[axis],
            f"{v['mean_cross_rho']:+.3f}",
            f"{v['mean_within_rho']:+.3f}",
            f"{v['mean_within_rho'] - v['mean_cross_rho']:+.3f}",
            f"p = {v['perm_p_value']:.4g}",
            f"{v['effect_r']:+.3f}",
        ])
    summary_table = ui.data_table(
        ["Asse", "ρ̄ cross (25)", "ρ̄ within (20)", "Δ", "perm p", "effect r"],
        rows,
        col_classes=["", "num", "num", "num strong", "num", "num"],
    )
    return (
        ui.section_open("332", "§3.3.2 — Allineamento cross-linguistico")
        + '<p>È il cuore di Lens IV. Per ogni asse si dispone, per ciascun modello, di una lista ordinata di 350 termini dal punteggio più negativo al più positivo. '
        + 'La domanda è: quanto due modelli ordinano i termini nello stesso modo lungo lo stesso asse? '
        + 'Per rispondere si prendono a due a due tutti i modelli (45 coppie complessive con dieci modelli) e per ciascuna coppia si calcola la '
        + 'correlazione di ' + ui.metric_chip("g-spearman", "Spearman") + ' fra le rispettive serie di 350 punteggi. '
        + 'Ρ vicino a +1: ordinamenti pressoché identici; Ρ vicino a 0: ordinamenti scorrelati; Ρ negativo: ordinamenti invertiti. '
        + 'Il risultato è un forest plot a sei pannelli (uno per asse), ciascuno con 45 correlazioni suddivise in quattro categorie: '
        + '10 intra-WEIRD (tra i cinque modelli del lato EN), 10 intra-Sinic (tra i cinque del lato ZH), '
        + '23 cross-tradizione (combinazioni EN × ZH di modelli diversi), e 2 coppie di controllo bilingue (BGE-M3-EN × BGE-M3-ZH e Qwen3-0.6B-EN × Qwen3-0.6B-ZH), '
        + 'in cui lo stesso modello codifica entrambe le lingue ed è quindi un controllo architetturale.</p>'
        + '<h3>Forest plot per asse (2×3)</h3>'
        + ui.plotly_embed(forest, "fig-332-forest", 860)
        + ui.plot_caption("Ogni punto è la ρ osservata per la coppia di modelli indicata; la barra orizzontale è l\'intervallo di confidenza al 95% "
                          "calcolato con bootstrap a livello di termine (10&thinsp;000 ricampionamenti dei 350 termini). "
                          "Colori: blu = intra-WEIRD, vermiglio = intra-Sinic, verde = intra-bilingue (controllo β), bronzo = cross-tradizione. "
                          "Tutti e sei i pannelli condividono la stessa scala x per consentire il confronto visivo fra assi.")
        + '<h3>Aggregato per asse</h3>'
        + summary_table
        + '<p>Lettura della tabella. Per ogni asse si riportano: la media delle correlazioni cross-tradizione (ρ̄ cross, 25 coppie con dieci modelli, incluse le 2 coppie intra-bilingui che lens4.py conteggia come cross); '
        + 'la media delle 20 correlazioni within (10 intra-WEIRD più 10 intra-Sinic); '
        + 'la differenza aritmetica Δ = ρ̄<sub>within</sub> − ρ̄<sub>cross</sub>; la <code>perm p</code>; l\'<em>effect r</em>. '
        + 'La <code>perm p</code> è la p-value di un test di permutazione: si rimescolano casualmente le etichette di tradizione (WEIRD / Sinic) dei modelli '
        + 'e per ogni rimescolamento si ricalcola Δ come se quelle etichette fossero quelle vere; si ripete 10&thinsp;000 volte per ottenere una distribuzione '
        + 'di riferimento compatibile con l\'ipotesi che le etichette non contino, e la p è la frazione di Δ permutati uguali o superiori a quello osservato. '
        + 'Il minimo rappresentabile con 10&thinsp;000 permutazioni è ≈ 0.0001. '
        + 'L\'<em>effect r</em> è una misura standardizzata della dimensione dello scarto, in \\([-1,+1]\\): vicino a 0 lo scarto è trascurabile rispetto alla dispersione, '
        + 'vicino a 1 è ampio rispetto alla dispersione.</p>'
        + ui.section_close()
    )


def section_ranking(section_332):
    fig, _ = fig_ranking_bar(section_332["summary_per_axis"])
    return (
        ui.section_open("333", "§3.3.3 — Gerarchia di divergenza")
        + '<p>Questa sezione riordina i sei assi secondo un solo numero: la ρ̄ cross, cioè la media delle 9 correlazioni '
        + 'cross-tradizione dell\'asse, già riportata in §3.3.2. Gli assi sono disposti dall\'alto in basso per ρ̄ cross crescente: '
        + 'in alto compaiono gli assi su cui la concordanza media fra modelli WEIRD e Sinic è più bassa (ρ̄ cross vicino a zero), '
        + 'in basso gli assi su cui la concordanza media è più alta. La quantità descritta è il grado di accordo fra modelli secondo questa misura, '
        + 'non il "peso culturale" dell\'asse: la gerarchia dice quale asse presenta il disaccordo maggiore fra le due tradizioni di modelli lungo la scala ρ̄ cross, senza attribuzioni di causa.</p>'
        + '<p>Il diamante blu sovrapposto a ciascuna barra indica, per riferimento, la ρ̄ within dello stesso asse (media delle 6 correlazioni intra-gruppo). '
        + 'La distanza orizzontale fra barra e diamante è il Δ = within − cross della tabella di §3.3.2.</p>'
        + ui.plotly_embed(fig, "fig-333-bar", 380)
        + ui.plot_caption("Barra bronzo: ρ̄ cross (media delle 9 correlazioni cross-tradizione). Diamante blu: ρ̄ within (media delle 6 correlazioni intra-gruppo). "
                          "Una linea verticale marca lo zero sull\'asse x.")
        + ui.section_close()
    )


def section_divergent(scores, terms):
    fig = fig_divergent_dumbbell(scores, terms, n_top=18)
    return (
        ui.section_open("334", "§3.3.4 — Termini più divergenti per asse")
        + '<p>Mentre §3.3.2 e §3.3.3 aggregano su tutti i 350 termini, questa sezione scende al singolo termine. '
        + 'Procedura: fissato un asse, per ciascun termine si calcola la media dei punteggi che i cinque modelli del lato EN '
        + '(tre monolinguali inglesi più due bilingui BGE-M3-EN e Qwen3-0.6B-EN) gli hanno assegnato lungo quell\'asse (W̄), '
        + 'e la media dei punteggi che i cinque modelli del lato ZH (tre monolinguali cinesi più BGE-M3-ZH e Qwen3-0.6B-ZH) gli hanno assegnato (S̄). '
        + 'Lo scarto |W̄ − S̄| quantifica, per quel termine e quell\'asse, di quanto i due gruppi di modelli si spostano in media l\'uno rispetto all\'altro. '
        + 'Il grafico seleziona i 18 termini con lo scarto più grande e li rappresenta come <em>dumbbell</em>: due punti, '
        + 'uno per W̄ (cerchio blu) e uno per S̄ (quadrato vermiglio), uniti da un filo. La descrizione è puramente meccanica: '
        + 'si elencano i termini su cui i due gruppi di modelli, in media, dissentono di più secondo questa misura, senza interpretazione giuridica '
        + 'del motivo del dissenso.</p>'
        + ui.plotly_embed(fig, "fig-334-dumbbell", 620)
        + ui.plot_caption("Il menù a tendina in alto seleziona l\'asse. La scala x è centrata a zero: valori positivi indicano tendenza "
                          "verso il polo positivo (la prima parola del nome dell\'asse, es. <em>individual</em> in <em>individual ↔ collective</em>), "
                          "valori negativi verso il polo negativo. La lunghezza del filo fra i due punti è |W̄ − S̄|.")
        + ui.section_close()
    )


def section_reproducibility(results_meta):
    d = results_meta
    return (
        ui.section_open("tecnica", "Tecnica riproducibile")
        + '<p>Tutti i numeri riportati in questa pagina provengono dalla stessa esecuzione, condotta sullo stesso snapshot di dati, '
        + 'sugli stessi modelli e con gli stessi semi pseudo-casuali (un seme fissa l\'esito dei campionamenti casuali '
        + 'in modo che la run sia ripetibile bit per bit).</p>'
        + ui.data_table(
            ["Parametro", "Valore"],
            [
                ["Data della run",                          d["date"]],
                ["N iterazioni del bootstrap",              str(d["n_boot"])],
                ["Termini core (N per correlazione)",       str(d["n_core"])],
                ["Pool totale (core + background + controllo)", str(d["n_pool"])],
                ["Assi",                                    ", ".join(d["axes"])],
                ["Modelli WEIRD",                           ", ".join(d["weird_models"])],
                ["Modelli Sinic",                           ", ".join(d["sinic_models"])],
                ["Tempo di esecuzione (secondi)",           f"{d['elapsed_seconds']:.1f}"],
            ],
            col_classes=["", "num"],
        )
        + '<p><strong>Codice</strong>. Modulo principale: <code>experiments/lens_4_values/lens4.py</code>. '
        + 'Definizione degli assi (120 coppie di poli): <code>experiments/lens_4_values/value_axes.yaml</code>. '
        + 'Risultati aggregati: <code>experiments/lens_4_values/results/lens4_results.json</code>. '
        + 'Punteggi grezzi per ogni termine, ogni modello e ogni asse: <code>results/scores/{model}_{axis}.npy</code>, '
        + 'per un totale di 6 modelli × 6 assi = 36 file (ciascuno un array di 350 punteggi).</p>'
        + ui.section_close()
    )


def section_footer_page():
    return (
        'Source code e storia dei commit: <a href="https://github.com/capazme/GeometriaIuris">'
        'github.com/capazme/GeometriaIuris</a> · dataset anchor <code>8480ea5</code>, '
        'espansione assi <code>ad70775</code>, archivio D9 <code>e5adc12</code>. '
        'Trace di decisione: <code>experiments/lens_4_values/trace.md</code>. '
        'Versione encoder bare.'
    )


# --------------------------------------------------------------------------
# Main

def build():
    results, axes_yaml, scores = load_all()
    title = "Geometria Iuris · Lens IV — proiezione su assi di valori"
    subtitle = "Kozlowski axes · 6 assi × 10 coppie × 2 lingue · encoder bare · " + results["meta"]["date"][:10]

    nav_items = [
        ("#domanda",   "Domanda"),
        ("#pipeline",  "Pipeline"),
        ("#331",       "§3.3.1 assi"),
        ("#331b",      "§3.3.1b ortogonalità"),
        ("#332",       "§3.3.2 forest"),
        ("#333",       "§3.3.3 gerarchia"),
        ("#334",       "§3.3.4 divergenti"),
        ("#tecnica",   "Tecnica"),
        ("#glossary",  "Glossario"),
    ]

    html_parts = [
        ui.page_head(title, subtitle),
        ui.sticky_nav(nav_items, back_link=("index.html", "↑ index")),
        ui.open_main(),
        section_domanda(),
        section_pipeline(),
        section_axes_construction(axes_yaml, results["section_331"]),
        section_orthogonality(results["section_331"]),
        section_alignment(results["section_332"]),
        section_ranking(results["section_332"]),
        section_divergent(scores, results["terms_core"]),
        section_reproducibility(results["meta"]),
        ui.glossary_section([
            "g-vector", "g-cosdist", "g-spearman", "g-kozlowski",
            "g-boot", "g-crosstrad", "g-withintrad",
        ]),
        ui.page_footer(section_footer_page()),
    ]

    OUT.write_text("".join(html_parts), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build()
