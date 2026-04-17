"""
Lens I §3.1 robustness — Layer-aware Lens I sensitivity sweep.

Recomputes the Lens I aggregate ρ̄ pipeline (within-WEIRD, within-Sinic,
cross-tradition) at multiple transformer depths for each of the six models
in the 3+3 design, **reusing the per-layer pooled vector cache produced by
Lens III** (no new encoding required). The question this module answers is
whether the headline Lens I result (Δρ ≈ 0.260) is an artefact of the
final-layer extraction choice or whether the same cross-tradition geometric
contraction is observable at multiple depths and, if so, where it peaks.

Procedure
---------
For each model, the cached tensor at
``lens_3_stratigraphy/results/layer_vectors/{label}.npz`` has shape
``(397, L+1, dim)``. Layer 0 is the embedding layer; layers 1..L are the
transformer blocks. The pooling strategy applied during caching matches the
model's native one (CLS for BGE/Dmeta, mean for E5/FreeLaw/Text2vec), so
the per-layer vectors are directly comparable to the final-layer pooled
output that Lens I currently uses.

The probe samples six target depths per model — embedding, L/4, L/2, 2L/3,
5L/6, L — and at each one computes a per-model RDM via cosine dissimilarity
on L2-normalised vectors. The 3+3 design then yields, at each layer:

  - within-WEIRD ρ̄ = mean Spearman over the 3 WEIRD-pair RDMs
  - within-Sinic ρ̄ = mean Spearman over the 3 Sinic-pair RDMs
  - cross-tradition ρ̄ = mean Spearman over the 9 cross pairs
  - Δρ = mean_within_weird − mean_cross  (matching `lens1.py` line 401)

Cross-model alignment requires that all six models be evaluated at *some*
layer for each step, but transformers have different total depths (24, 24,
22, 24, 24, 12 here). The script aligns by *fraction-of-depth* (0.0, 0.25,
0.5, 0.667, 0.833, 1.0), rounding to the nearest integer per model.

Output
------
A JSON report `results/layer_sensitivity.json` containing per-layer per-pair
ρ values, the four aggregate scalars at each fractional depth, and a small
summary block reporting the depth at which Δρ peaks (per model and overall).
A static HTML dashboard at `results/figures/html/layer_sensitivity.html`
visualises the result with a Plotly line chart.

References
----------
Tenney, I., Das, D., & Pavlick, E. (2019). "BERT Rediscovers the Classical
  NLP Pipeline." ACL.
Ethayarajh, K. (2019). "How Contextual are Contextualized Word
  Representations? Comparing the Geometry of BERT, ELMo, and GPT-2
  Embeddings." EMNLP.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="Mean of empty slice")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.statistical import compute_rdm, upper_tri  # noqa: E402
from shared.html_style import (  # noqa: E402
    CSS, HEAD_LINKS, C_BLUE, C_VERMIL, C_GREEN, page_head,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIRD_LABELS = ["BGE-EN-large", "E5-large", "FreeLaw-EN"]
SINIC_LABELS = ["BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"]
ALL_LABELS = WEIRD_LABELS + SINIC_LABELS

LAYER_CACHE_DIR = REPO_ROOT / "lens_3_stratigraphy" / "results" / "layer_vectors"

# Six target fractional depths covering embedding → low-mid → middle →
# Sofroniew 2/3 → post-Lens-III phase-transition zone → final.
DEPTH_FRACTIONS = [0.0, 0.25, 0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0]
DEPTH_LABELS = ["0.00 (emb)", "0.25", "0.50", "0.67 (2/3)", "0.83 (5/6)", "1.00 (final)"]


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_layer_cache(label: str) -> np.ndarray:
    """Load per-layer pooled vectors for one model. Shape: (397, L+1, dim)."""
    path = LAYER_CACHE_DIR / f"{label}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing layer cache: {path}")
    data = np.load(path)
    return data["layers"]


def _nan_to_none(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialisation."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_nan_to_none(v) for v in obj)
    return obj


def fraction_to_layer_index(fraction: float, n_layers: int) -> int:
    """
    Map a fractional depth in [0, 1] to a transformer layer index.

    The cache has L+1 layers indexed 0..L (layer 0 = embedding,
    layer L = final transformer block). fraction=0 → 0, fraction=1 → L,
    intermediate values are rounded to the nearest integer.
    """
    L = n_layers - 1  # last layer index
    return int(round(fraction * L))


# ---------------------------------------------------------------------------
# Per-layer RDM and aggregation
# ---------------------------------------------------------------------------

def compute_layer_rdm(layer_vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalise a (N, dim) slice and compute its cosine RDM.

    The cache stores raw pooled hidden states; sentence-transformers
    L2-normalises only at the final layer for many models, so the per-layer
    vectors are not necessarily unit-norm. We renormalise defensively.
    """
    vecs = layer_vectors.astype(np.float32, copy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-12, None)
    return compute_rdm(vecs)


def rsa_pair(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """
    Spearman ρ on the upper triangles of two RDMs.

    Returns NaN if either input is constant (e.g., for CLS-pooled models at
    layer 0, where every term collapses to the same single learned [CLS]
    embedding before contextualisation, yielding an all-zero RDM). The NaN
    is preserved upstream and reported as ``null`` in the JSON.
    """
    tri_a = upper_tri(rdm_a)
    tri_b = upper_tri(rdm_b)
    if np.std(tri_a) < 1e-12 or np.std(tri_b) < 1e-12:
        return float("nan")
    return float(spearmanr(tri_a, tri_b).statistic)


def aggregate_at_depth(
    rdms_weird: dict[str, np.ndarray],
    rdms_sinic: dict[str, np.ndarray],
) -> dict:
    """
    Compute the four Lens I aggregate scalars given two dicts of RDMs
    keyed by model label, one per tradition.
    """
    weird_labels = list(rdms_weird.keys())
    sinic_labels = list(rdms_sinic.keys())

    within_weird_pairs = []
    for i in range(len(weird_labels)):
        for j in range(i + 1, len(weird_labels)):
            la, lb = weird_labels[i], weird_labels[j]
            rho = rsa_pair(rdms_weird[la], rdms_weird[lb])
            within_weird_pairs.append({"a": la, "b": lb, "rho": rho})

    within_sinic_pairs = []
    for i in range(len(sinic_labels)):
        for j in range(i + 1, len(sinic_labels)):
            la, lb = sinic_labels[i], sinic_labels[j]
            rho = rsa_pair(rdms_sinic[la], rdms_sinic[lb])
            within_sinic_pairs.append({"a": la, "b": lb, "rho": rho})

    cross_pairs = []
    for la in weird_labels:
        for lb in sinic_labels:
            rho = rsa_pair(rdms_weird[la], rdms_sinic[lb])
            cross_pairs.append({"a": la, "b": lb, "rho": rho})

    def _stats(pairs):
        rhos = np.array([p["rho"] for p in pairs], dtype=float)
        n_valid = int(np.sum(~np.isnan(rhos)))
        if n_valid == 0:
            return float("nan"), 0, len(pairs)
        return float(np.nanmean(rhos)), n_valid, len(pairs)

    rho_w, n_w_valid, n_w_total = _stats(within_weird_pairs)
    rho_s, n_s_valid, n_s_total = _stats(within_sinic_pairs)
    rho_x, n_x_valid, n_x_total = _stats(cross_pairs)
    # Match the Lens I convention (lens1.py:401): drop = within_weird - cross
    drop = rho_w - rho_x

    return {
        "within_weird_pairs": within_weird_pairs,
        "within_sinic_pairs": within_sinic_pairs,
        "cross_pairs": cross_pairs,
        "mean_rho_within_weird": rho_w,
        "mean_rho_within_sinic": rho_s,
        "mean_rho_cross": rho_x,
        "cross_tradition_drop": drop,
        "n_valid_pairs": {
            "within_weird": [n_w_valid, n_w_total],
            "within_sinic": [n_s_valid, n_s_total],
            "cross": [n_x_valid, n_x_total],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[layers] Loading per-layer caches for 6 models …")
    caches = {label: load_layer_cache(label) for label in ALL_LABELS}
    n_layers_per_model = {label: caches[label].shape[1] for label in ALL_LABELS}
    for label, n in n_layers_per_model.items():
        print(f"  {label}: {n} layers (indices 0..{n - 1})")

    print("\n[layers] Sweeping over six fractional depths …")
    per_depth: list[dict] = []

    for frac, frac_label in zip(DEPTH_FRACTIONS, DEPTH_LABELS):
        # Resolve actual layer indices per model
        layer_indices = {
            label: fraction_to_layer_index(frac, n_layers_per_model[label])
            for label in ALL_LABELS
        }

        # Compute per-model RDMs at this depth
        rdms_weird = {}
        for label in WEIRD_LABELS:
            idx = layer_indices[label]
            rdms_weird[label] = compute_layer_rdm(caches[label][:, idx, :])
        rdms_sinic = {}
        for label in SINIC_LABELS:
            idx = layer_indices[label]
            rdms_sinic[label] = compute_layer_rdm(caches[label][:, idx, :])

        agg = aggregate_at_depth(rdms_weird, rdms_sinic)
        agg["fraction"] = frac
        agg["fraction_label"] = frac_label
        agg["layer_indices"] = layer_indices
        per_depth.append(agg)

        print(
            f"  depth {frac_label:>14}  "
            f"within_W={agg['mean_rho_within_weird']:+.4f}  "
            f"within_S={agg['mean_rho_within_sinic']:+.4f}  "
            f"cross={agg['mean_rho_cross']:+.4f}  "
            f"Δρ={agg['cross_tradition_drop']:+.4f}"
        )

    # Find the depth that maximises Δρ (ignoring NaN, which can occur at
    # layer 0 for CLS-pooled models — see rsa_pair docstring).
    drops = [d["cross_tradition_drop"] for d in per_depth]
    drops_arr = np.array(drops, dtype=float)
    if np.all(np.isnan(drops_arr)):
        peak_idx = 0
    else:
        peak_idx = int(np.nanargmax(drops_arr))
    peak_depth = per_depth[peak_idx]

    # Per-model layer trajectories: at each fractional depth, also report
    # each model's pairwise ρ to the other models in its tradition (so we
    # can see whether the WEIRD/Sinic phase shapes converge across models).
    per_model_layer_rho = {label: [] for label in ALL_LABELS}
    for d in per_depth:
        # mean ρ each model achieves with its same-tradition partners
        for p in d["within_weird_pairs"]:
            per_model_layer_rho[p["a"]].append((d["fraction"], p["rho"], p["b"]))
            per_model_layer_rho[p["b"]].append((d["fraction"], p["rho"], p["a"]))
        for p in d["within_sinic_pairs"]:
            per_model_layer_rho[p["a"]].append((d["fraction"], p["rho"], p["b"]))
            per_model_layer_rho[p["b"]].append((d["fraction"], p["rho"], p["a"]))

    report = {
        "meta": {
            "module": "lens_1_relational/layer_sensitivity.py",
            "thesis_section": "§3.1 (robustness sweep, complementary to §3.1.3 Lens III)",
            "date": datetime.now().isoformat(timespec="seconds"),
            "weird_models": WEIRD_LABELS,
            "sinic_models": SINIC_LABELS,
            "depth_fractions": DEPTH_FRACTIONS,
            "depth_labels": DEPTH_LABELS,
            "n_terms": int(caches[WEIRD_LABELS[0]].shape[0]),
            "layers_per_model": n_layers_per_model,
        },
        "per_depth": per_depth,
        "summary": {
            "peak_depth_fraction": peak_depth["fraction"],
            "peak_depth_label": peak_depth["fraction_label"],
            "peak_drop": peak_depth["cross_tradition_drop"],
            "peak_within_weird": peak_depth["mean_rho_within_weird"],
            "peak_within_sinic": peak_depth["mean_rho_within_sinic"],
            "peak_cross": peak_depth["mean_rho_cross"],
            "final_depth_drop": per_depth[-1]["cross_tradition_drop"],
            "final_depth_within_weird": per_depth[-1]["mean_rho_within_weird"],
            "final_depth_within_sinic": per_depth[-1]["mean_rho_within_sinic"],
            "final_depth_cross": per_depth[-1]["mean_rho_cross"],
            "drop_at_each_depth": drops,
        },
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_sensitivity.json"
    # Convert NaN → None recursively before JSON serialisation (json.dump
    # otherwise emits a literal `NaN`, which is not valid JSON).
    out_path.write_text(
        json.dumps(_nan_to_none(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[layers] Report → {out_path}")
    print(
        f"[layers] Peak Δρ = {peak_depth['cross_tradition_drop']:+.4f} "
        f"at depth {peak_depth['fraction_label']}"
    )

    _render_html(report)


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def _render_html(report: dict) -> None:
    """Render a static HTML dashboard with a Plotly line chart of Δρ vs depth."""
    out_dir = Path(__file__).parent / "results" / "figures" / "html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_sensitivity.html"

    fractions = report["meta"]["depth_fractions"]
    labels = report["meta"]["depth_labels"]
    summary = report["summary"]
    per_depth = report["per_depth"]

    def _nan_to_jsnone(xs):
        return [None if (isinstance(x, float) and np.isnan(x)) else x for x in xs]

    rho_w = _nan_to_jsnone([d["mean_rho_within_weird"] for d in per_depth])
    rho_s = _nan_to_jsnone([d["mean_rho_within_sinic"] for d in per_depth])
    rho_x = _nan_to_jsnone([d["mean_rho_cross"] for d in per_depth])
    drops = _nan_to_jsnone([d["cross_tradition_drop"] for d in per_depth])
    drop_max = max((d for d in drops if d is not None), default=1.0)

    # Plotly figure 1: aggregate ρ̄ traces
    fig1 = {
        "data": [
            {
                "x": fractions, "y": rho_w, "mode": "lines+markers",
                "name": "within-WEIRD ρ̄",
                "line": {"color": C_BLUE, "width": 3},
                "marker": {"size": 9},
            },
            {
                "x": fractions, "y": rho_s, "mode": "lines+markers",
                "name": "within-Sinic ρ̄",
                "line": {"color": C_VERMIL, "width": 3},
                "marker": {"size": 9},
            },
            {
                "x": fractions, "y": rho_x, "mode": "lines+markers",
                "name": "cross-tradition ρ̄",
                "line": {"color": C_GREEN, "width": 3},
                "marker": {"size": 9},
            },
        ],
        "layout": {
            "title": "Aggregate ρ̄ across transformer depth",
            "xaxis": {
                "title": "Fractional depth",
                "tickmode": "array",
                "tickvals": fractions,
                "ticktext": labels,
                "range": [-0.05, 1.05],
            },
            "yaxis": {"title": "Mean Spearman ρ", "range": [0, 0.7]},
            "height": 380,
            "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
            "legend": {"x": 0.02, "y": 0.98},
        },
    }

    # Plotly figure 2: Δρ trajectory
    fig2 = {
        "data": [
            {
                "x": fractions, "y": drops, "mode": "lines+markers",
                "name": "Δρ (within-WEIRD − cross)",
                "line": {"color": "#222", "width": 3},
                "marker": {"size": 10},
                "fill": "tozeroy",
                "fillcolor": "rgba(0,114,178,0.10)",
            },
        ],
        "layout": {
            "title": "Cross-tradition drop (Δρ) across depth",
            "xaxis": {
                "title": "Fractional depth",
                "tickmode": "array",
                "tickvals": fractions,
                "ticktext": labels,
                "range": [-0.05, 1.05],
            },
            "yaxis": {"title": "Δρ", "range": [0, drop_max * 1.25]},
            "height": 320,
            "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
            "shapes": [{
                "type": "line",
                "x0": summary["peak_depth_fraction"],
                "x1": summary["peak_depth_fraction"],
                "y0": 0, "y1": summary["peak_drop"],
                "line": {"color": C_VERMIL, "width": 2, "dash": "dot"},
            }],
            "annotations": [{
                "x": summary["peak_depth_fraction"],
                "y": summary["peak_drop"],
                "text": f"peak Δρ = {summary['peak_drop']:.3f}",
                "showarrow": True, "arrowhead": 2, "ax": 30, "ay": -30,
                "font": {"color": C_VERMIL},
            }],
        },
    }

    def _fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "<i>n/a</i>"
        return f"{x:+.4f}"

    # Per-depth table
    rows = []
    for d in per_depth:
        rows.append(
            f"<tr>"
            f"<td><b>{d['fraction_label']}</b></td>"
            f"<td>{_fmt(d['mean_rho_within_weird'])}</td>"
            f"<td>{_fmt(d['mean_rho_within_sinic'])}</td>"
            f"<td>{_fmt(d['mean_rho_cross'])}</td>"
            f"<td><b>{_fmt(d['cross_tradition_drop'])}</b></td>"
            f"</tr>"
        )
    table = (
        '<table class="data">'
        '<thead><tr><th>Depth</th><th>within-WEIRD ρ̄</th>'
        '<th>within-Sinic ρ̄</th><th>cross ρ̄</th><th>Δρ</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    # Per-model layer index resolution table
    layer_rows = []
    layers_per_model = report["meta"]["layers_per_model"]
    for label in ALL_LABELS:
        L = layers_per_model[label] - 1
        idxs = [
            fraction_to_layer_index(f, layers_per_model[label])
            for f in fractions
        ]
        idx_str = " · ".join(f"{i}/{L}" for i in idxs)
        layer_rows.append(
            f"<tr><td><b>{label}</b></td><td>{L + 1}</td><td>{idx_str}</td></tr>"
        )
    layer_table = (
        '<table class="data">'
        '<thead><tr><th>Model</th><th>Total layers</th>'
        '<th>Sampled layer indices (per fractional depth)</th></tr></thead>'
        f"<tbody>{''.join(layer_rows)}</tbody></table>"
    )

    # Compose page
    head = page_head("Lens I — Layer Sensitivity")
    body = f"""<body>
<h1>Lens I &mdash; Layer Sensitivity (§3.1 robustness)</h1>
<p class="subtitle">Recomputes the Lens I aggregate ρ̄ at six transformer
depths per model, reusing the Lens III per-layer cache. Tests robustness of
the headline Δρ ≈ 0.260 to extraction depth and reconciles with the §3.1.3
phase transition.</p>

<div class="metrics">
  <div class="metric blue">
    <div class="value">{summary['peak_drop']:+.3f}</div>
    <div class="label">Peak Δρ</div>
  </div>
  <div class="metric vermil">
    <div class="value">{summary['peak_depth_label']}</div>
    <div class="label">Peak depth</div>
  </div>
  <div class="metric green">
    <div class="value">{summary['final_depth_drop']:+.3f}</div>
    <div class="label">Δρ at final layer</div>
  </div>
  <div class="metric">
    <div class="value">6</div>
    <div class="label">Models &times; 6 depths</div>
  </div>
</div>

<div class="card">
  <h2>Question</h2>
  <div class="question">
    <b>Is the headline Lens I result an artefact of the final-layer
    extraction choice?</b> If the cross-tradition Δρ peaks at the final
    layer and decays monotonically with depth, the answer is yes — the
    measurement is depth-specific. If Δρ is broadly stable across the
    upper half of each model and peaks in the same depth zone identified
    by Lens III as the &ldquo;legal-meaning crystallisation&rdquo; layer
    (~80–100% of L), the headline result is robust and the layer choice
    is principled.
  </div>
</div>

<div class="card">
  <h2>Aggregate ρ̄ trajectory</h2>
  <div id="plt_agg" style="width:100%;height:380px;"></div>
  <div id="plt_drop" style="width:100%;height:320px;"></div>
</div>

<div class="card">
  <h2>Per-depth aggregates</h2>
  {table}
  <p class="note-sm">Δρ definition: <code>mean_rho_within_weird − mean_rho_cross</code>
  (matching <code>lens1.py</code> line 401, the published Lens I baseline).
  Each ρ̄ is the mean of 3 within-tradition or 9 cross-tradition pairwise
  Spearman correlations on RDM upper triangles.</p>
</div>

<div class="card">
  <h2>Layer index resolution per model</h2>
  {layer_table}
  <p class="note-sm">Models have different total depth (24, 24, 22, 24, 24,
  12), so the same fractional depth maps to a different absolute layer per
  model. Index 0 is the embedding layer, index L is the final transformer
  block. The cached vectors at each layer are pooled with the model's native
  strategy (CLS for BGE/Dmeta, mean for E5/FreeLaw/Text2vec) and L2-renormalised
  before RDM construction.</p>
</div>

<div class="card">
  <h2>Reading</h2>
  <div class="finding">
    <b>Lens I + Lens III consistency.</b>
    Lens III (§3.1.3) reports a phase transition in legal meaning
    crystallisation in the last 15–20% of layers. If the layer sensitivity
    sweep here shows Δρ rising sharply over the same depth range, the
    Lens I result and the Lens III phase transition are two views of the
    same underlying phenomenon: legal meaning is acquired late in the
    transformer stack, and the cross-tradition contraction documented by
    Lens I is a property of those late, semantically loaded representations.
  </div>
</div>

<script>
const figs = {{
  plt_agg: {json.dumps(fig1)},
  plt_drop: {json.dumps(fig2)}
}};
Plotly.newPlot('plt_agg', figs.plt_agg.data, figs.plt_agg.layout, {{responsive: true}});
Plotly.newPlot('plt_drop', figs.plt_drop.data, figs.plt_drop.layout, {{responsive: true}});
</script>
</body>"""

    html = f"<!DOCTYPE html>\n<html lang='en'>\n{head}\n{body}\n</html>"
    out_path.write_text(html, encoding="utf-8")
    print(f"[layers] Dashboard → {out_path}")


if __name__ == "__main__":
    main()
