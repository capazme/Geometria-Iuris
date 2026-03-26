"""
NTA Exploration — Interactive HTML visualization.

Generates a self-contained HTML page for exploring the NTA exploration results
(101 candidate terms × 6 models). Designed for manual term selection.

Usage:
    cd experiments/
    python -m lens_3_stratigraphy.nta_exploration_viz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

# Okabe-Ito palette
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_SKY = "#56B4E9"
C_VERMIL = "#D55E00"
C_PURPLE = "#CC79A7"
C_BLACK = "#000000"
C_GREY = "#999999"

MODEL_COLORS = {
    "BGE-EN-large": C_BLUE,
    "E5-large": C_ORANGE,
    "FreeLaw-EN": C_GREEN,
    "BGE-ZH-large": C_SKY,
    "Text2vec-large-ZH": C_VERMIL,
    "Dmeta-ZH": C_PURPLE,
}

DOMAIN_COLORS = {
    "administrative": "#4e79a7",
    "civil": "#f28e2b",
    "constitutional": "#e15759",
    "criminal": "#76b7b2",
    "environmental_tech": "#ff9da7",
    "governance": "#9c755f",
    "international": "#59a14f",
    "jurisprudence": "#bab0ac",
    "labor_social": "#edc948",
    "procedure": "#b07aa1",
    "rights": "#86bcb6",
    "control": "#999999",
}

SHORT_LABELS = {
    "BGE-EN-large": "BGE-EN",
    "E5-large": "E5",
    "FreeLaw-EN": "FreeLaw",
    "BGE-ZH-large": "BGE-ZH",
    "Text2vec-large-ZH": "Text2vec",
    "Dmeta-ZH": "Dmeta",
}


def _score_term(term_data: dict, k: int) -> dict:
    """Compute interest scores for a single term."""
    evol = term_data["domain_evolution"]
    domain = term_data["domain"]

    ctrl_values = [e["n_control"] for e in evol]
    legal_values = [e["n_legal"] for e in evol]
    peak_ctrl = max(ctrl_values)
    peak_ctrl_layer = evol[ctrl_values.index(peak_ctrl)]["layer"]
    final_ctrl = ctrl_values[-1]
    initial_ctrl = ctrl_values[0]

    final_doms = evol[-1]["domains"]
    own_domain_final = final_doms.get(domain, 0)
    n_domains_final = len([d for d, c in final_doms.items() if c > 0 and d != "control"])

    volatility = sum(
        abs(ctrl_values[i + 1] - ctrl_values[i])
        for i in range(len(ctrl_values) - 1)
    )

    # Crystallization layer: first layer where ctrl drops to 0 and stays 0
    crystal_layer = None
    for i in range(len(ctrl_values)):
        if all(v == 0 for v in ctrl_values[i:]):
            crystal_layer = evol[i]["layer"]
            break

    return {
        "initial_ctrl": initial_ctrl,
        "peak_ctrl": peak_ctrl,
        "peak_ctrl_layer": peak_ctrl_layer,
        "final_ctrl": final_ctrl,
        "own_domain_final": own_domain_final,
        "n_domains_final": n_domains_final,
        "volatility": volatility,
        "ctrl_trajectory": ctrl_values,
        "legal_trajectory": legal_values,
        "crystal_layer": crystal_layer,
    }


def _compute_precomputed_data(nta_results: dict, k: int) -> dict:
    """Pre-compute all derived data for the HTML."""
    models = list(nta_results.keys())

    term_scores: dict[str, list[dict]] = {}
    for model_label, dd_data in nta_results.items():
        for term_name, term_data in dd_data["terms"].items():
            if term_name not in term_scores:
                term_scores[term_name] = []
            score = _score_term(term_data, k)
            score["model"] = model_label
            score["domain"] = term_data["domain"]
            score["zh"] = term_data.get("zh", "")
            score["sample_layers"] = dd_data["sample_layers"]
            term_scores[term_name].append(score)

    # Ranking
    ranking = []
    for term_name, scores in term_scores.items():
        avg_peak = float(np.mean([s["peak_ctrl"] for s in scores]))
        avg_vol = float(np.mean([s["volatility"] for s in scores]))
        avg_final_ctrl = float(np.mean([s["final_ctrl"] for s in scores]))
        avg_n_domains = float(np.mean([s["n_domains_final"] for s in scores]))
        avg_own_dom = float(np.mean([s["own_domain_final"] for s in scores]))
        avg_initial_ctrl = float(np.mean([s["initial_ctrl"] for s in scores]))
        interest = avg_peak * 2 + avg_vol + avg_final_ctrl * 3 + avg_n_domains

        # Per-model crystallization layers
        crystal_layers = [s["crystal_layer"] for s in scores if s["crystal_layer"] is not None]
        avg_crystal = round(float(np.mean(crystal_layers)), 1) if crystal_layers else None
        n_crystallized = len(crystal_layers)

        ranking.append({
            "term": term_name,
            "domain": scores[0]["domain"],
            "zh": scores[0]["zh"],
            "avg_peak_ctrl": round(avg_peak, 2),
            "avg_final_ctrl": round(avg_final_ctrl, 2),
            "avg_initial_ctrl": round(avg_initial_ctrl, 2),
            "avg_volatility": round(avg_vol, 2),
            "avg_n_domains": round(avg_n_domains, 2),
            "avg_own_dom": round(avg_own_dom, 2),
            "score": round(interest, 2),
            "avg_crystal_layer": avg_crystal,
            "n_crystallized": n_crystallized,
            "per_model": {
                s["model"]: {
                    "ctrl_trajectory": s["ctrl_trajectory"],
                    "legal_trajectory": s["legal_trajectory"],
                    "peak_ctrl": s["peak_ctrl"],
                    "final_ctrl": s["final_ctrl"],
                    "initial_ctrl": s["initial_ctrl"],
                    "sample_layers": s["sample_layers"],
                    "crystal_layer": s["crystal_layer"],
                    "peak_ctrl_layer": s["peak_ctrl_layer"],
                    "own_domain_final": s["own_domain_final"],
                }
                for s in scores
            },
        })

    ranking.sort(key=lambda x: -x["score"])

    # Cross-tradition divergence
    weird_labels = [m for m in models if "ZH" not in m and "Dmeta" not in m]
    sinic_labels = [m for m in models if "ZH" in m or "Dmeta" in m]

    divergences = []
    for term_name, scores in term_scores.items():
        weird_final = float(np.mean(
            [s["own_domain_final"] for s in scores if s["model"] in weird_labels]
        ))
        sinic_final = float(np.mean(
            [s["own_domain_final"] for s in scores if s["model"] in sinic_labels]
        ))
        weird_ctrl = float(np.mean(
            [s["final_ctrl"] for s in scores if s["model"] in weird_labels]
        ))
        sinic_ctrl = float(np.mean(
            [s["final_ctrl"] for s in scores if s["model"] in sinic_labels]
        ))
        weird_peak = float(np.mean(
            [s["peak_ctrl"] for s in scores if s["model"] in weird_labels]
        ))
        sinic_peak = float(np.mean(
            [s["peak_ctrl"] for s in scores if s["model"] in sinic_labels]
        ))
        div = abs(weird_final - sinic_final) + abs(weird_ctrl - sinic_ctrl) * 2
        divergences.append({
            "term": term_name,
            "domain": term_scores[term_name][0]["domain"],
            "divergence": round(div, 2),
            "weird_own_dom": round(weird_final, 2),
            "sinic_own_dom": round(sinic_final, 2),
            "weird_final_ctrl": round(weird_ctrl, 2),
            "sinic_final_ctrl": round(sinic_ctrl, 2),
            "weird_peak_ctrl": round(weird_peak, 2),
            "sinic_peak_ctrl": round(sinic_peak, 2),
        })
    divergences.sort(key=lambda x: -x["divergence"])

    # Heatmap data: terms × layers, ctrl count averaged across models
    # Use first model's sample_layers as reference
    ref_layers = nta_results[models[0]]["sample_layers"]
    heatmap_terms = []
    heatmap_matrix = []
    for r in ranking:
        row = []
        for li in range(len(ref_layers)):
            vals = []
            for m in models:
                pm = r["per_model"].get(m)
                if pm and li < len(pm["ctrl_trajectory"]):
                    vals.append(pm["ctrl_trajectory"][li])
            row.append(round(float(np.mean(vals)), 2) if vals else 0)
        heatmap_terms.append(r["term"])
        heatmap_matrix.append(row)

    # Domain summary: avg ctrl trajectory per domain
    domains_set = sorted(set(r["domain"] for r in ranking))
    domain_summary = {}
    for d in domains_set:
        d_terms = [r for r in ranking if r["domain"] == d]
        n_layers = len(ref_layers)
        avg_traj = []
        for li in range(n_layers):
            vals = []
            for r in d_terms:
                for m in models:
                    pm = r["per_model"].get(m)
                    if pm and li < len(pm["ctrl_trajectory"]):
                        vals.append(pm["ctrl_trajectory"][li])
            avg_traj.append(round(float(np.mean(vals)), 3) if vals else 0)
        domain_summary[d] = {
            "n_terms": len(d_terms),
            "avg_ctrl_trajectory": avg_traj,
            "avg_score": round(float(np.mean([r["score"] for r in d_terms])), 2),
        }

    return {
        "ranking": ranking,
        "divergences": divergences,
        "models": models,
        "k": k,
        "model_colors": {m: MODEL_COLORS.get(m, C_GREY) for m in models},
        "model_short": SHORT_LABELS,
        "domain_colors": DOMAIN_COLORS,
        "weird_labels": weird_labels,
        "sinic_labels": sinic_labels,
        "heatmap": {
            "terms": heatmap_terms,
            "layers": ref_layers,
            "matrix": heatmap_matrix,
        },
        "domain_summary": domain_summary,
    }


def build_exploration_html(nta_results: dict, k: int = 7) -> str:
    """Build self-contained interactive HTML for NTA exploration."""
    data = _compute_precomputed_data(nta_results, k)
    data_js = json.dumps(data, ensure_ascii=False)
    return _html_template(data_js)


def _html_template(data_js: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NTA Exploration — §3.1.3c Term Selection</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
  :root {{
    --bg: #f8f9fa; --fg: #1a1a2e; --card: #fff; --border: #e2e8f0;
    --accent: #0072B2; --accent-light: #e3f2fd; --accent-dark: #005a8c;
    --muted: #64748b; --success: #059669; --warn: #d97706;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--fg);
    padding: 24px 28px; line-height: 1.6; max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 2px; font-weight: 700; }}
  .subtitle {{ color: var(--muted); font-size: 0.88rem; margin-bottom: 20px; }}

  /* Nav tabs */
  .tabs {{
    display: flex; gap: 3px; margin-bottom: 20px; flex-wrap: wrap;
    border-bottom: 2px solid var(--border); padding-bottom: 0;
  }}
  .tab-btn {{
    padding: 9px 20px; border: none; border-bottom: 3px solid transparent;
    background: none; cursor: pointer; font-size: 0.85rem; color: var(--muted);
    font-weight: 500; transition: all 0.15s; border-radius: 6px 6px 0 0;
  }}
  .tab-btn:hover {{ color: var(--accent); background: var(--accent-light); }}
  .tab-btn.active {{
    color: var(--accent); border-bottom-color: var(--accent);
    font-weight: 600; background: var(--accent-light);
  }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}

  /* Explanatory notes */
  .note {{
    background: #f0f7ff; border-left: 4px solid var(--accent);
    padding: 14px 18px; margin-bottom: 18px; border-radius: 0 6px 6px 0;
    font-size: 0.84rem; line-height: 1.7;
  }}
  .note h3 {{ margin: 0 0 6px 0; font-size: 0.92rem; color: var(--accent-dark); }}
  .note p {{ margin: 5px 0; }}
  .note code {{
    background: #e2e8f0; padding: 1px 5px; border-radius: 3px;
    font-size: 0.82em; font-family: 'SF Mono', Consolas, monospace;
  }}
  .note .formula {{
    background: #fff; border: 1px solid var(--border); padding: 8px 14px;
    border-radius: 4px; margin: 8px 0; font-family: inherit;
  }}
  .note ul {{ margin: 6px 0 6px 20px; }}
  .note li {{ margin: 3px 0; }}

  /* Cards */
  .card {{
    background: var(--card); border-radius: 10px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 18px;
    border: 1px solid var(--border);
  }}
  .card h3 {{ font-size: 1.05rem; margin-bottom: 10px; font-weight: 600; }}
  .card-subtitle {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 14px; }}

  /* Ranking table */
  .rank-table {{
    width: 100%; border-collapse: collapse; font-size: 0.8rem;
    background: var(--card); border-radius: 8px; overflow: hidden;
  }}
  .rank-table th {{
    background: #f1f5f9; padding: 9px 10px; text-align: left;
    border-bottom: 2px solid var(--border); cursor: pointer;
    user-select: none; position: sticky; top: 0; z-index: 10;
    font-weight: 600; font-size: 0.78rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.3px;
  }}
  .rank-table th:hover {{ background: #e2e8f0; }}
  .rank-table th .sort-arrow {{ font-size: 0.65em; margin-left: 2px; opacity: 0.6; }}
  .rank-table td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; }}
  .rank-table tbody tr:hover {{ background: var(--accent-light); cursor: pointer; }}
  .rank-table tbody tr.selected {{ background: #bbdefb !important; }}
  .domain-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.72em; color: #fff; white-space: nowrap; font-weight: 500;
  }}
  .score-bar {{
    display: inline-block; height: 12px; border-radius: 6px;
    background: linear-gradient(90deg, var(--accent), var(--accent-dark));
    opacity: 0.65; vertical-align: middle; margin-right: 5px;
  }}
  .zh-text {{ color: #94a3b8; font-size: 0.82em; margin-left: 5px; }}
  .mini-traj {{
    display: inline-flex; align-items: flex-end; gap: 1px;
    height: 18px; vertical-align: middle; margin-left: 6px;
  }}
  .mini-traj-bar {{
    width: 4px; border-radius: 1px; transition: height 0.2s;
  }}

  /* Trajectory detail */
  .detail-placeholder {{
    text-align: center; color: var(--muted); padding: 50px;
    font-style: italic; font-size: 0.9rem;
  }}
  .traj-grid {{
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;
  }}
  @media (max-width: 1100px) {{ .traj-grid {{ grid-template-columns: 1fr 1fr; }} }}
  @media (max-width: 700px) {{ .traj-grid {{ grid-template-columns: 1fr; }} }}
  .traj-chart {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden;
  }}
  .traj-chart .model-label {{
    padding: 6px 12px; font-size: 0.78rem; font-weight: 600;
    background: #f8fafc; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }}
  .traj-chart .model-label .model-stats {{
    font-weight: 400; color: var(--muted); font-size: 0.75rem;
  }}

  /* Filters */
  .filter-row {{
    display: flex; gap: 12px; align-items: center; margin-bottom: 14px; flex-wrap: wrap;
  }}
  .filter-row select, .filter-row input {{
    padding: 6px 12px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.82rem; background: var(--card);
  }}
  .filter-row input[type="text"] {{ width: 220px; }}
  .filter-row label {{ font-size: 0.82rem; color: var(--muted); font-weight: 500; }}

  /* Stat boxes */
  .stat-row {{ display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }}
  .stat-box {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 18px; flex: 1; min-width: 110px;
    text-align: center;
  }}
  .stat-box .stat-val {{
    font-size: 1.6rem; font-weight: 700; color: var(--accent);
    letter-spacing: -0.5px;
  }}
  .stat-box .stat-label {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; }}

  /* Two-column layout */
  .split-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 900px) {{ .split-cols {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>NTA Exploration — Candidate Term Selection</h1>
<p class="subtitle">§3.1.3c Neighborhood Trajectory Analysis — 101 candidate terms across 6 models — k=7 nearest neighbors</p>

<div class="stat-row" id="stats"></div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('tabRanking', this)">Ranking</button>
  <button class="tab-btn" onclick="showTab('tabTrajectories', this)">Trajectories</button>
  <button class="tab-btn" onclick="showTab('tabHeatmap', this)">Heatmap</button>
  <button class="tab-btn" onclick="showTab('tabDomains', this)">By Domain</button>
  <button class="tab-btn" onclick="showTab('tabScatter', this)">Scatter</button>
  <button class="tab-btn" onclick="showTab('tabDivergence', this)">WEIRD vs Sinic</button>
  <button class="tab-btn" onclick="showTab('tabOverlay', this)">Overlay</button>
</div>

<!-- ═══════════════ Tab 1: Ranking ═══════════════ -->
<div id="tabRanking" class="panel active">
  <div class="note">
    <h3>What is this table?</h3>
    <p>Each of the 101 candidate terms was run through the <b>Neighborhood Trajectory Analysis</b>
    (NTA) on all 6 models. For each term, we track its $k=7$ nearest neighbors across sampled
    transformer layers (L0, L4, L8, ..., L24). The neighbor pool includes <b>397 core legal
    terms</b> and <b>100 control terms</b> (Swadesh-like everyday words).</p>
    <p>The columns measure different aspects of how a term's neighborhood evolves with depth:</p>
    <ul>
      <li><b>Peak Ctrl</b> — Maximum number of control (non-legal) neighbors at any layer, averaged
        across 6 models. High values mean the term starts in "general-purpose" semantic space.</li>
      <li><b>Final Ctrl</b> — Control neighbors remaining at the final layer. $>0$ means the
        model never fully resolves the term into legal space (persistent polysemy).</li>
      <li><b>Volatility</b> — Total absolute change in ctrl count across layer transitions
        $\\sum_\\ell |\\text{{ctrl}}_{{\\ell+1}} - \\text{{ctrl}}_\\ell|$. High = turbulent trajectory.</li>
      <li><b>Dom Final</b> — Number of distinct legal domains among the final-layer neighbors.
        High = term sits at a cross-domain intersection.</li>
      <li><b>Own Dom</b> — How many of the final $k$ neighbors share the term's own domain.
        High = the model has strongly categorized it within its domain.</li>
      <li><b>Crystal</b> — Average layer at which ctrl drops to 0 permanently ("crystallization").
        Earlier = the term is recognized as legal sooner. "—" = never fully crystallizes.</li>
    </ul>
    <div class="formula">
      $$\\text{{Score}} = 2 \\cdot \\overline{{\\text{{PeakCtrl}}}} + \\overline{{\\text{{Volatility}}}} + 3 \\cdot \\overline{{\\text{{FinalCtrl}}}} + \\overline{{\\text{{DomFinal}}}}$$
    </div>
    <p>Click any row to see that term's full trajectory in the <b>Trajectories</b> tab.</p>
  </div>

  <div class="filter-row">
    <label>Domain:</label>
    <select id="domainFilter" onchange="applyFilters()"><option value="all">All domains</option></select>
    <label>Search:</label>
    <input type="text" id="searchFilter" placeholder="Type to filter..." oninput="applyFilters()">
    <label style="margin-left:auto;font-size:0.78rem;color:#94a3b8">
      <span id="filteredCount"></span> terms shown
    </label>
  </div>
  <div style="max-height:70vh;overflow-y:auto;border-radius:8px;border:1px solid var(--border)">
    <table class="rank-table" id="rankTable">
      <thead><tr id="rankHeader"></tr></thead>
      <tbody id="rankBody"></tbody>
    </table>
  </div>
</div>

<!-- ═══════════════ Tab 2: Trajectories ═══════════════ -->
<div id="tabTrajectories" class="panel">
  <div class="note">
    <h3>Reading the trajectory charts</h3>
    <p>Each chart shows one model's view of the selected term. The x-axis is the transformer
    layer (sampled at L0, L4, L8, ...). The y-axis counts how many of the $k=7$ nearest
    neighbors are <b>legal</b> (colored area) vs <b>control/non-legal</b> (gray area).
    The two areas always sum to $k=7$.</p>
    <p>A term that starts with many control neighbors and ends with all legal neighbors
    demonstrates the model's process of <b>semantic specialization</b>: surface layers see
    the term as general-purpose, deeper layers recognize its legal specificity.</p>
    <p>The combined chart at the bottom overlays all 6 models' control curves. Convergence
    across models suggests a robust pattern; divergence suggests cultural or architectural
    differences in how models encode the term.</p>
  </div>
  <div class="card">
    <div id="termDetail" class="detail-placeholder">
      Select a term from the Ranking tab to explore its layer-by-layer trajectory.
    </div>
  </div>
</div>

<!-- ═══════════════ Tab 3: Heatmap ═══════════════ -->
<div id="tabHeatmap" class="panel">
  <div class="note">
    <h3>Control-neighbor heatmap</h3>
    <p>Each cell shows the <b>average number of control neighbors</b> (across all 6 models)
    for a given term at a given layer. Terms are sorted by their interest score (most
    interesting at top).</p>
    <p><b>Reading the pattern:</b> A column of dark cells at L0 that fades rightward means
    most terms start with many control neighbors and shed them as depth increases. Terms that
    remain dark across all layers are persistently polysemic — the models never fully resolve
    them as legal-specific. These are often the most interesting for the thesis.</p>
  </div>
  <div class="card">
    <div id="heatmapChart" style="height:900px"></div>
  </div>
</div>

<!-- ═══════════════ Tab 4: By Domain ═══════════════ -->
<div id="tabDomains" class="panel">
  <div class="note">
    <h3>Domain-level patterns</h3>
    <p>The top chart shows the <b>average control-neighbor trajectory per domain</b>, aggregated
    across all terms in that domain and all 6 models. This reveals which legal domains
    are recognized earliest (ctrl drops fast) and which remain entangled with everyday
    language longest (ctrl stays high).</p>
    <p>The bottom chart shows a bar comparison of mean interest score by domain. Domains
    with higher average scores contain terms that are, on average, more polysemic and
    more interesting for the thesis.</p>
  </div>
  <div class="card">
    <div id="domainTrajChart" style="height:400px"></div>
  </div>
  <div class="card">
    <div id="domainBarChart" style="height:350px"></div>
  </div>
</div>

<!-- ═══════════════ Tab 5: Scatter ═══════════════ -->
<div id="tabScatter" class="panel">
  <div class="note">
    <h3>Scatter plots — two complementary views</h3>
    <p><b>Left: Peak vs Final ctrl.</b> Terms in the <b>top-right</b> quadrant start with
    many control neighbors AND retain them — persistent polysemy. Top-left = high initial
    polysemy but fully resolved. Bottom-right = low initial polysemy but some control
    neighbors remain (unexpected). Bottom-left = always clearly legal.</p>
    <p><b>Right: Volatility vs Peak ctrl.</b> Terms in the top-right are both highly polysemic
    AND turbulent — their neighborhoods change dramatically across layers. These are the
    most narratively interesting for §3.1.3c.</p>
    <p>Bubble size encodes the interest score. Color encodes domain. Hover for details.</p>
  </div>
  <div class="split-cols">
    <div class="card"><div id="scatterPeakFinal" style="height:450px"></div></div>
    <div class="card"><div id="scatterVolPeak" style="height:450px"></div></div>
  </div>
  <div class="card">
    <h3>Crystallization layer distribution</h3>
    <p class="card-subtitle">
      At which layer does each term lose its last control neighbor permanently?
      Terms that never crystallize ("Never") retain non-legal neighbors even at the deepest layer.
    </p>
    <div id="crystalHistogram" style="height:350px"></div>
  </div>
</div>

<!-- ═══════════════ Tab 6: WEIRD vs Sinic ═══════════════ -->
<div id="tabDivergence" class="panel">
  <div class="note">
    <h3>Cross-tradition divergence</h3>
    <p>This tab compares how <b>WEIRD</b> (BGE-EN, E5, FreeLaw) and <b>Sinic</b>
    (BGE-ZH, Text2vec, Dmeta) model families treat the same legal concept.
    Recall that Sinic models receive the <b>Chinese translation</b> of each term.</p>
    <p>The divergence score combines two differences:</p>
    <div class="formula">
      $$\\text{{Div}} = |\\overline{{\\text{{OwnDom}}}}_W - \\overline{{\\text{{OwnDom}}}}_S|
      + 2 \\cdot |\\overline{{\\text{{FinalCtrl}}}}_W - \\overline{{\\text{{FinalCtrl}}}}_S|$$
    </div>
    <p>High divergence means the two cultural-linguistic traditions encode the concept
    in structurally different semantic neighborhoods. This is directly relevant to
    the thesis claim about cultural normative structures.</p>
    <p><b>Left scatter:</b> Each point is a term. X = WEIRD final ctrl, Y = Sinic final ctrl.
    Points on the diagonal are treated similarly by both traditions. Points far from the
    diagonal show cross-cultural encoding differences.</p>
  </div>
  <div class="split-cols">
    <div class="card"><div id="divScatter" style="height:420px"></div></div>
    <div class="card"><div id="divBarChart" style="height:420px"></div></div>
  </div>
  <div class="card">
    <table class="rank-table" id="divTable">
      <thead><tr>
        <th>#</th><th>Term</th><th>Domain</th><th>Divergence</th>
        <th>WEIRD own_dom</th><th>Sinic own_dom</th>
        <th>WEIRD ctrl</th><th>Sinic ctrl</th>
      </tr></thead>
      <tbody id="divBody"></tbody>
    </table>
  </div>
</div>

<!-- ═══════════════ Tab 7: Overlay ═══════════════ -->
<div id="tabOverlay" class="panel">
  <div class="note">
    <h3>All trajectories at a glance</h3>
    <p>Each line represents one term's control-neighbor count across layers, for the selected
    model (or averaged). Color encodes domain. Use the domain filter to highlight one domain
    and fade the rest. This bird's-eye view reveals global patterns: do all terms follow
    a similar "high ctrl at L0, low ctrl at L24" arc, or are there distinct trajectory types?</p>
  </div>
  <div class="card">
    <div class="filter-row">
      <label>Model:</label>
      <select id="overlayModel" onchange="drawOverlay()">
        <option value="avg">Average (all models)</option>
      </select>
      <label>Highlight domain:</label>
      <select id="overlayDomain" onchange="drawOverlay()"><option value="all">All</option></select>
    </div>
    <div id="overlayChart" style="height:550px"></div>
  </div>
</div>


<script>
var D = {data_js};

function showTab(id, btn) {{
  document.querySelectorAll('.panel').forEach(function(el) {{ el.classList.remove('active'); }});
  document.querySelectorAll('.tab-btn').forEach(function(el) {{ el.classList.remove('active'); }});
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  var drawMap = {{
    'tabOverlay': drawOverlay,
    'tabDivergence': drawDivergence,
    'tabHeatmap': drawHeatmap,
    'tabDomains': drawDomains,
    'tabScatter': drawScatter,
  }};
  if (drawMap[id]) setTimeout(drawMap[id], 60);
}}

/* ── Stats ── */
function renderStats() {{
  var el = document.getElementById('stats');
  var maxScore = D.ranking[0].score;
  var sum = 0;
  D.ranking.forEach(function(r) {{ sum += r.score; }});
  var avgScore = (sum / D.ranking.length).toFixed(1);
  var domains = [];
  D.ranking.forEach(function(r) {{ if (domains.indexOf(r.domain) === -1) domains.push(r.domain); }});
  var nCrystal = 0;
  D.ranking.forEach(function(r) {{ if (r.n_crystallized === D.models.length) nCrystal++; }});

  var items = [
    [D.ranking.length, 'Terms'],
    [D.models.length, 'Models'],
    [domains.length, 'Domains'],
    [maxScore, 'Max Score'],
    [avgScore, 'Avg Score'],
    [nCrystal, 'Fully Crystallized'],
  ];
  el.textContent = '';
  items.forEach(function(item) {{
    var box = document.createElement('div');
    box.className = 'stat-box';
    var val = document.createElement('div');
    val.className = 'stat-val';
    val.textContent = item[0];
    var lab = document.createElement('div');
    lab.className = 'stat-label';
    lab.textContent = item[1];
    box.appendChild(val);
    box.appendChild(lab);
    el.appendChild(box);
  }});
}}

/* ── Ranking table ── */
var sortCol = 'score';
var sortAsc = false;
var selectedTerm = null;

var COLS = [
  {{ key: 'rank', label: '#', sortable: false }},
  {{ key: 'term', label: 'Term', sortable: true }},
  {{ key: 'domain', label: 'Domain', sortable: true }},
  {{ key: 'avg_peak_ctrl', label: 'Peak Ctrl', sortable: true }},
  {{ key: 'avg_final_ctrl', label: 'Final Ctrl', sortable: true }},
  {{ key: 'avg_volatility', label: 'Volatility', sortable: true }},
  {{ key: 'avg_n_domains', label: 'Dom Final', sortable: true }},
  {{ key: 'avg_own_dom', label: 'Own Dom', sortable: true }},
  {{ key: 'avg_crystal_layer', label: 'Crystal', sortable: true }},
  {{ key: 'score', label: 'Score', sortable: true }},
  {{ key: 'sparkline', label: 'Ctrl Trajectory', sortable: false }},
];

function renderHeader() {{
  var tr = document.getElementById('rankHeader');
  tr.textContent = '';
  COLS.forEach(function(c) {{
    var th = document.createElement('th');
    var arrow = c.key === sortCol ? (sortAsc ? ' \\u25B2' : ' \\u25BC') : '';
    th.textContent = c.label;
    if (arrow) {{
      var span = document.createElement('span');
      span.className = 'sort-arrow';
      span.textContent = arrow;
      th.appendChild(span);
    }}
    if (c.sortable) th.onclick = (function(k) {{ return function() {{ sortBy(k); }}; }})(c.key);
    tr.appendChild(th);
  }});
}}

function sortBy(col) {{
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = (col === 'term' || col === 'domain'); }}
  renderTable();
}}

function getFiltered() {{
  var domF = document.getElementById('domainFilter').value;
  var search = document.getElementById('searchFilter').value.toLowerCase();
  return D.ranking.filter(function(r) {{
    if (domF !== 'all' && r.domain !== domF) return false;
    if (search && r.term.toLowerCase().indexOf(search) === -1) return false;
    return true;
  }});
}}

function buildSparkline(r) {{
  // Average ctrl trajectory across all models
  var firstM = D.models[0];
  var pm = r.per_model[firstM];
  if (!pm) return document.createTextNode('');
  var nL = pm.ctrl_trajectory.length;
  var avg = new Array(nL).fill(0);
  var count = 0;
  D.models.forEach(function(m) {{
    var p = r.per_model[m];
    if (p && p.ctrl_trajectory.length === nL) {{
      p.ctrl_trajectory.forEach(function(v, i) {{ avg[i] += v; }});
      count++;
    }}
  }});
  if (count > 0) avg = avg.map(function(v) {{ return v / count; }});

  var container = document.createElement('span');
  container.className = 'mini-traj';
  avg.forEach(function(v) {{
    var bar = document.createElement('span');
    bar.className = 'mini-traj-bar';
    var h = Math.round(v / D.k * 16);
    bar.style.height = Math.max(1, h) + 'px';
    var intensity = Math.round(v / D.k * 200);
    bar.style.background = 'rgb(' + intensity + ',' + Math.round(intensity * 0.6) + ',0)';
    container.appendChild(bar);
  }});
  return container;
}}

function renderTable() {{
  renderHeader();
  var data = getFiltered();
  document.getElementById('filteredCount').textContent = data.length;

  data.sort(function(a, b) {{
    var va = a[sortCol], vb = b[sortCol];
    // Handle null crystal layers
    if (va === null) va = sortAsc ? 999 : -1;
    if (vb === null) vb = sortAsc ? 999 : -1;
    if (typeof va === 'string') {{
      var cmp = va.localeCompare(vb);
      return sortAsc ? cmp : -cmp;
    }}
    return sortAsc ? va - vb : vb - va;
  }});

  var maxScore = D.ranking[0].score;
  var tbody = document.getElementById('rankBody');
  tbody.textContent = '';

  data.forEach(function(r, i) {{
    var dc = D.domain_colors[r.domain] || '#888';
    var barW = Math.round(r.score / maxScore * 50);
    var tr = document.createElement('tr');
    if (r.term === selectedTerm) tr.className = 'selected';
    tr.onclick = (function(t) {{ return function() {{ selectTerm(t); }}; }})(r.term);

    var td0 = document.createElement('td');
    td0.textContent = i + 1;
    td0.style.color = '#94a3b8';
    tr.appendChild(td0);

    var td1 = document.createElement('td');
    var b = document.createElement('b');
    b.textContent = r.term;
    td1.appendChild(b);
    var zh = document.createElement('span');
    zh.className = 'zh-text';
    zh.textContent = r.zh;
    td1.appendChild(zh);
    tr.appendChild(td1);

    var td2 = document.createElement('td');
    var badge = document.createElement('span');
    badge.className = 'domain-badge';
    badge.style.background = dc;
    badge.textContent = r.domain;
    td2.appendChild(badge);
    tr.appendChild(td2);

    var numCols = ['avg_peak_ctrl', 'avg_final_ctrl', 'avg_volatility', 'avg_n_domains', 'avg_own_dom'];
    numCols.forEach(function(col) {{
      var td = document.createElement('td');
      td.textContent = r[col];
      tr.appendChild(td);
    }});

    // Crystal layer
    var tdC = document.createElement('td');
    if (r.avg_crystal_layer !== null) {{
      tdC.textContent = 'L' + r.avg_crystal_layer;
      tdC.style.color = r.avg_crystal_layer <= 8 ? '#059669' : '#d97706';
    }} else {{
      tdC.textContent = '\\u2014';
      tdC.style.color = '#ef4444';
    }}
    tr.appendChild(tdC);

    // Score with bar
    var tdS = document.createElement('td');
    var bar = document.createElement('span');
    bar.className = 'score-bar';
    bar.style.width = barW + 'px';
    tdS.appendChild(bar);
    tdS.appendChild(document.createTextNode(' ' + r.score));
    tr.appendChild(tdS);

    // Sparkline
    var tdSp = document.createElement('td');
    tdSp.appendChild(buildSparkline(r));
    tr.appendChild(tdSp);

    tbody.appendChild(tr);
  }});
}}

function applyFilters() {{ renderTable(); }}

function initDomainFilter() {{
  var domains = [];
  D.ranking.forEach(function(r) {{ if (domains.indexOf(r.domain) === -1) domains.push(r.domain); }});
  domains.sort();
  var sel = document.getElementById('domainFilter');
  domains.forEach(function(d) {{
    var opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    sel.appendChild(opt);
  }});
  var sel2 = document.getElementById('overlayDomain');
  domains.forEach(function(d) {{
    var opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    sel2.appendChild(opt);
  }});
  var sel3 = document.getElementById('overlayModel');
  D.models.forEach(function(m) {{
    var opt = document.createElement('option');
    opt.value = m; opt.textContent = D.model_short[m] || m;
    sel3.appendChild(opt);
  }});
}}

/* ── Term detail ── */
function selectTerm(term) {{
  selectedTerm = term;
  renderTable();
  var r = D.ranking.find(function(x) {{ return x.term === term; }});
  if (!r) return;

  showTab('tabTrajectories', document.querySelectorAll('.tab-btn')[1]);
  var el = document.getElementById('termDetail');
  el.textContent = '';

  // Header
  var dc = D.domain_colors[r.domain] || '#888';
  var h3 = document.createElement('h3');
  var badge = document.createElement('span');
  badge.className = 'domain-badge';
  badge.style.background = dc;
  badge.textContent = r.domain;
  h3.appendChild(badge);
  h3.appendChild(document.createTextNode(' ' + r.term + ' '));
  var zhS = document.createElement('span');
  zhS.className = 'zh-text';
  zhS.textContent = r.zh;
  h3.appendChild(zhS);
  el.appendChild(h3);

  // Summary stats
  var p = document.createElement('p');
  p.style.cssText = 'font-size:0.82rem;color:#64748b;margin:8px 0 16px';
  var crystalStr = r.avg_crystal_layer !== null ? ('L' + r.avg_crystal_layer + ' (' + r.n_crystallized + '/6 models)') : 'Never';
  p.textContent = 'Score: ' + r.score +
    ' | Peak ctrl: ' + r.avg_peak_ctrl + ' | Final ctrl: ' + r.avg_final_ctrl +
    ' | Volatility: ' + r.avg_volatility + ' | Domains at final: ' + r.avg_n_domains +
    ' | Own domain: ' + r.avg_own_dom + ' | Crystallization: ' + crystalStr;
  el.appendChild(p);

  // Grid
  var grid = document.createElement('div');
  grid.className = 'traj-grid';
  D.models.forEach(function(m) {{
    var pm = r.per_model[m];
    if (!pm) return;
    var mc = D.model_colors[m] || '#888';
    var short = D.model_short[m] || m;
    var chartDiv = document.createElement('div');
    chartDiv.className = 'traj-chart';
    var label = document.createElement('div');
    label.className = 'model-label';
    label.style.borderLeft = '3px solid ' + mc;
    var nameSpan = document.createTextNode(short);
    label.appendChild(nameSpan);
    var statsSpan = document.createElement('span');
    statsSpan.className = 'model-stats';
    var cStr = pm.crystal_layer !== null ? 'L' + pm.crystal_layer : 'Never';
    statsSpan.textContent = 'peak=' + pm.peak_ctrl + ' @L' + pm.peak_ctrl_layer + ' | cryst=' + cStr;
    label.appendChild(statsSpan);
    chartDiv.appendChild(label);
    var plotDiv = document.createElement('div');
    plotDiv.id = 'traj_' + m.replace(/[^a-zA-Z0-9]/g, '_');
    plotDiv.style.height = '200px';
    chartDiv.appendChild(plotDiv);
    grid.appendChild(chartDiv);
  }});
  el.appendChild(grid);

  var combinedWrap = document.createElement('div');
  combinedWrap.style.marginTop = '16px';
  var combinedDiv = document.createElement('div');
  combinedDiv.id = 'traj_combined';
  combinedDiv.style.height = '280px';
  combinedWrap.appendChild(combinedDiv);
  el.appendChild(combinedWrap);

  setTimeout(function() {{
    D.models.forEach(function(m) {{
      var pm = r.per_model[m];
      if (!pm) return;
      var divId = 'traj_' + m.replace(/[^a-zA-Z0-9]/g, '_');
      var mc = D.model_colors[m] || '#888';
      Plotly.newPlot(divId, [
        {{ x: pm.sample_layers, y: pm.ctrl_trajectory, mode: 'lines+markers', name: 'Control',
          line: {{ color: '#94a3b8', width: 2 }}, marker: {{ size: 5 }},
          fill: 'tozeroy', fillcolor: 'rgba(148,163,184,0.15)' }},
        {{ x: pm.sample_layers, y: pm.legal_trajectory, mode: 'lines+markers', name: 'Legal',
          line: {{ color: mc, width: 2 }}, marker: {{ size: 5 }},
          fill: 'tozeroy', fillcolor: mc + '18' }},
      ], {{
        margin: {{ l: 35, r: 15, t: 8, b: 32 }},
        xaxis: {{ title: {{ text: 'Layer', font: {{ size: 10 }} }}, dtick: 4 }},
        yaxis: {{ range: [0, D.k + 0.5], dtick: 1, title: {{ text: 'Count', font: {{ size: 10 }} }} }},
        showlegend: true, legend: {{ x: 1, y: 1, xanchor: 'right', font: {{ size: 9 }} }},
        template: 'simple_white', hovermode: 'x unified',
      }}, {{ responsive: true, displayModeBar: false }});
    }});

    var traces = [];
    D.models.forEach(function(m) {{
      var pm = r.per_model[m];
      if (!pm) return;
      traces.push({{ x: pm.sample_layers, y: pm.ctrl_trajectory,
        mode: 'lines+markers', name: D.model_short[m] || m,
        line: {{ color: D.model_colors[m] || '#888', width: 2 }}, marker: {{ size: 5 }} }});
    }});
    Plotly.newPlot('traj_combined', traces, {{
      title: {{ text: 'Control neighbors — all models compared', font: {{ size: 12 }} }},
      margin: {{ l: 40, r: 20, t: 35, b: 40 }},
      xaxis: {{ title: 'Layer' }}, yaxis: {{ title: 'Control count', range: [0, D.k + 0.5], dtick: 1 }},
      template: 'simple_white', hovermode: 'x unified',
      legend: {{ orientation: 'h', y: -0.22, font: {{ size: 10 }} }},
    }}, {{ responsive: true }});
  }}, 60);
}}

/* ── Heatmap ── */
function drawHeatmap() {{
  var hm = D.heatmap;
  Plotly.newPlot('heatmapChart', [{{
    type: 'heatmap',
    z: hm.matrix,
    x: hm.layers.map(function(l) {{ return 'L' + l; }}),
    y: hm.terms,
    colorscale: [
      [0, '#f0f9ff'], [0.15, '#bae6fd'], [0.3, '#7dd3fc'],
      [0.5, '#f59e0b'], [0.7, '#ea580c'], [1, '#dc2626']
    ],
    hovertemplate: '<b>%{{y}}</b><br>Layer %{{x}}: %{{z:.1f}} ctrl neighbors<extra></extra>',
    colorbar: {{ title: {{ text: 'Avg ctrl', side: 'right' }}, thickness: 15 }},
  }}], {{
    title: {{ text: 'Control neighbors per term per layer (avg across 6 models)', font: {{ size: 13 }} }},
    margin: {{ l: 160, r: 60, t: 40, b: 50 }},
    xaxis: {{ title: 'Layer', side: 'bottom' }},
    yaxis: {{ autorange: 'reversed', tickfont: {{ size: 8 }} }},
    height: Math.max(600, hm.terms.length * 9),
  }}, {{ responsive: true }});
}}

/* ── Domains ── */
function drawDomains() {{
  var ds = D.domain_summary;
  var domains = Object.keys(ds).sort();
  var layers = D.heatmap.layers;

  // Trajectory chart
  var traces = domains.map(function(d) {{
    return {{
      x: layers, y: ds[d].avg_ctrl_trajectory,
      mode: 'lines+markers', name: d + ' (n=' + ds[d].n_terms + ')',
      line: {{ color: D.domain_colors[d] || '#888', width: 2 }},
      marker: {{ size: 5 }},
    }};
  }});
  Plotly.newPlot('domainTrajChart', traces, {{
    title: {{ text: 'Average control-neighbor trajectory by domain', font: {{ size: 13 }} }},
    margin: {{ l: 45, r: 20, t: 40, b: 45 }},
    xaxis: {{ title: 'Layer' }},
    yaxis: {{ title: 'Avg control neighbors', range: [0, 4] }},
    template: 'simple_white', hovermode: 'x unified',
    legend: {{ font: {{ size: 10 }}, orientation: 'h', y: -0.2 }},
  }}, {{ responsive: true }});

  // Bar chart
  var sorted = domains.slice().sort(function(a, b) {{ return ds[b].avg_score - ds[a].avg_score; }});
  Plotly.newPlot('domainBarChart', [{{
    type: 'bar',
    x: sorted.map(function(d) {{ return d; }}),
    y: sorted.map(function(d) {{ return ds[d].avg_score; }}),
    marker: {{ color: sorted.map(function(d) {{ return D.domain_colors[d] || '#888'; }}) }},
    hovertemplate: '%{{x}}<br>Avg score: %{{y:.1f}}<br>Terms: ' +
      sorted.map(function(d) {{ return ds[d].n_terms; }}).join(',') +
      '<extra></extra>',
    text: sorted.map(function(d) {{ return 'n=' + ds[d].n_terms; }}),
    textposition: 'outside', textfont: {{ size: 10, color: '#64748b' }},
  }}], {{
    title: {{ text: 'Average interest score by domain', font: {{ size: 13 }} }},
    margin: {{ l: 45, r: 20, t: 40, b: 80 }},
    xaxis: {{ tickangle: -30 }},
    yaxis: {{ title: 'Avg score' }},
    template: 'simple_white',
  }}, {{ responsive: true }});
}}

/* ── Scatter ── */
function drawScatter() {{
  var maxScore = D.ranking[0].score;

  // Peak vs Final ctrl
  var domains = [];
  D.ranking.forEach(function(r) {{ if (domains.indexOf(r.domain) === -1) domains.push(r.domain); }});
  var traces1 = domains.map(function(d) {{
    var items = D.ranking.filter(function(r) {{ return r.domain === d; }});
    return {{
      x: items.map(function(r) {{ return r.avg_peak_ctrl; }}),
      y: items.map(function(r) {{ return r.avg_final_ctrl; }}),
      text: items.map(function(r) {{ return r.term; }}),
      mode: 'markers', name: d,
      marker: {{
        color: D.domain_colors[d] || '#888',
        size: items.map(function(r) {{ return 6 + r.score / maxScore * 14; }}),
        opacity: 0.75,
        line: {{ width: 1, color: '#fff' }},
      }},
      hovertemplate: '<b>%{{text}}</b><br>Peak: %{{x:.1f}}<br>Final: %{{y:.1f}}<extra>' + d + '</extra>',
    }};
  }});
  Plotly.newPlot('scatterPeakFinal', traces1, {{
    title: {{ text: 'Peak ctrl vs Final ctrl', font: {{ size: 12 }} }},
    margin: {{ l: 45, r: 20, t: 40, b: 45 }},
    xaxis: {{ title: 'Peak ctrl (avg)', zeroline: true }},
    yaxis: {{ title: 'Final ctrl (avg)', zeroline: true }},
    template: 'simple_white', hovermode: 'closest',
    legend: {{ font: {{ size: 9 }}, orientation: 'h', y: -0.2 }},
    shapes: [{{
      type: 'line', x0: 0, x1: 7, y0: 0, y1: 7,
      line: {{ color: '#e2e8f0', width: 1, dash: 'dot' }},
    }}],
  }}, {{ responsive: true }});

  // Volatility vs Peak
  var traces2 = domains.map(function(d) {{
    var items = D.ranking.filter(function(r) {{ return r.domain === d; }});
    return {{
      x: items.map(function(r) {{ return r.avg_peak_ctrl; }}),
      y: items.map(function(r) {{ return r.avg_volatility; }}),
      text: items.map(function(r) {{ return r.term; }}),
      mode: 'markers', name: d,
      marker: {{
        color: D.domain_colors[d] || '#888',
        size: items.map(function(r) {{ return 6 + r.score / maxScore * 14; }}),
        opacity: 0.75,
        line: {{ width: 1, color: '#fff' }},
      }},
      hovertemplate: '<b>%{{text}}</b><br>Peak: %{{x:.1f}}<br>Vol: %{{y:.1f}}<extra>' + d + '</extra>',
    }};
  }});
  Plotly.newPlot('scatterVolPeak', traces2, {{
    title: {{ text: 'Volatility vs Peak ctrl', font: {{ size: 12 }} }},
    margin: {{ l: 45, r: 20, t: 40, b: 45 }},
    xaxis: {{ title: 'Peak ctrl (avg)' }},
    yaxis: {{ title: 'Volatility (avg)' }},
    template: 'simple_white', hovermode: 'closest',
    legend: {{ font: {{ size: 9 }}, orientation: 'h', y: -0.2 }},
  }}, {{ responsive: true }});

  // Crystallization histogram
  var crystalData = [];
  var neverCount = 0;
  D.ranking.forEach(function(r) {{
    if (r.avg_crystal_layer !== null) crystalData.push(r.avg_crystal_layer);
    else neverCount++;
  }});
  var trH = [{{
    type: 'histogram', x: crystalData,
    xbins: {{ start: 0, end: 26, size: 2 }},
    marker: {{ color: 'rgba(0,114,178,0.6)', line: {{ width: 1, color: '#0072B2' }} }},
    name: 'Crystallized',
    hovertemplate: 'Layer %{{x:.0f}}: %{{y}} terms<extra></extra>',
  }}];
  var annotations = [];
  if (neverCount > 0) {{
    annotations.push({{
      x: 26, y: neverCount * 0.8, text: neverCount + ' terms<br>never crystallize',
      showarrow: true, arrowhead: 2, ax: -40, ay: -30,
      font: {{ size: 11, color: '#ef4444' }},
    }});
  }}
  Plotly.newPlot('crystalHistogram', trH, {{
    title: {{ text: 'Crystallization layer distribution (avg across models)', font: {{ size: 12 }} }},
    margin: {{ l: 45, r: 30, t: 40, b: 45 }},
    xaxis: {{ title: 'Layer where ctrl permanently reaches 0', dtick: 4 }},
    yaxis: {{ title: 'Number of terms' }},
    template: 'simple_white',
    annotations: annotations,
  }}, {{ responsive: true }});
}}

/* ── Divergence ── */
function drawDivergence() {{
  var top30 = D.divergences.slice(0, 30);

  // Scatter: WEIRD vs Sinic final ctrl
  var domains = [];
  D.divergences.forEach(function(d) {{ if (domains.indexOf(d.domain) === -1) domains.push(d.domain); }});
  var scTraces = domains.map(function(dom) {{
    var items = D.divergences.filter(function(d) {{ return d.domain === dom; }});
    return {{
      x: items.map(function(d) {{ return d.weird_final_ctrl; }}),
      y: items.map(function(d) {{ return d.sinic_final_ctrl; }}),
      text: items.map(function(d) {{ return d.term; }}),
      mode: 'markers', name: dom,
      marker: {{
        color: D.domain_colors[dom] || '#888', size: 9, opacity: 0.7,
        line: {{ width: 1, color: '#fff' }},
      }},
      hovertemplate: '<b>%{{text}}</b><br>WEIRD ctrl: %{{x:.1f}}<br>Sinic ctrl: %{{y:.1f}}<extra>' + dom + '</extra>',
    }};
  }});
  Plotly.newPlot('divScatter', scTraces, {{
    title: {{ text: 'Final ctrl: WEIRD vs Sinic', font: {{ size: 12 }} }},
    margin: {{ l: 45, r: 20, t: 40, b: 45 }},
    xaxis: {{ title: 'WEIRD final ctrl', range: [-0.3, D.k] }},
    yaxis: {{ title: 'Sinic final ctrl', range: [-0.3, D.k] }},
    template: 'simple_white', hovermode: 'closest',
    legend: {{ font: {{ size: 9 }}, orientation: 'h', y: -0.22 }},
    shapes: [{{ type: 'line', x0: 0, x1: D.k, y0: 0, y1: D.k,
      line: {{ color: '#e2e8f0', width: 1, dash: 'dot' }} }}],
  }}, {{ responsive: true }});

  // Bar chart
  Plotly.newPlot('divBarChart', [{{
    type: 'bar', orientation: 'h',
    y: top30.map(function(d) {{ return d.term; }}).reverse(),
    x: top30.map(function(d) {{ return d.divergence; }}).reverse(),
    marker: {{ color: top30.map(function(d) {{ return D.domain_colors[d.domain] || '#888'; }}).reverse() }},
    hovertemplate: '<b>%{{y}}</b><br>Div: %{{x:.2f}}<extra></extra>',
  }}], {{
    title: {{ text: 'Top 30 cross-tradition divergence', font: {{ size: 12 }} }},
    margin: {{ l: 130, r: 20, t: 40, b: 40 }},
    xaxis: {{ title: 'Divergence' }}, template: 'simple_white',
  }}, {{ responsive: true }});

  // Table
  var tbody = document.getElementById('divBody');
  tbody.textContent = '';
  top30.forEach(function(d, i) {{
    var dc = D.domain_colors[d.domain] || '#888';
    var tr = document.createElement('tr');
    tr.onclick = (function(t) {{ return function() {{ selectTerm(t); }}; }})(d.term);
    var vals = [i + 1, d.term, d.domain, d.divergence,
      d.weird_own_dom, d.sinic_own_dom, d.weird_final_ctrl, d.sinic_final_ctrl];
    vals.forEach(function(val, ci) {{
      var td = document.createElement('td');
      if (ci === 1) {{ var b = document.createElement('b'); b.textContent = val; td.appendChild(b); }}
      else if (ci === 2) {{
        var badge = document.createElement('span');
        badge.className = 'domain-badge'; badge.style.background = dc;
        badge.textContent = val; td.appendChild(badge);
      }} else {{ td.textContent = val; }}
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
}}

/* ── Overlay ── */
function drawOverlay() {{
  var modelSel = document.getElementById('overlayModel').value;
  var domSel = document.getElementById('overlayDomain').value;

  var traces = [];
  D.ranking.forEach(function(r) {{
    var ctrl, layers;
    if (modelSel === 'avg') {{
      var firstModel = D.models[0];
      layers = r.per_model[firstModel] ? r.per_model[firstModel].sample_layers : [];
      var nLayers = layers.length;
      ctrl = new Array(nLayers).fill(0);
      var count = 0;
      D.models.forEach(function(m) {{
        var pm = r.per_model[m];
        if (!pm) return;
        var t = pm.ctrl_trajectory;
        if (t.length === nLayers) {{ t.forEach(function(v, idx) {{ ctrl[idx] += v; }}); count++; }}
      }});
      if (count > 0) ctrl = ctrl.map(function(v) {{ return v / count; }});
    }} else {{
      var pm = r.per_model[modelSel];
      if (!pm) return;
      ctrl = pm.ctrl_trajectory; layers = pm.sample_layers;
    }}
    var dc = D.domain_colors[r.domain] || '#888';
    var isHL = domSel === 'all' || r.domain === domSel;
    traces.push({{
      x: layers, y: ctrl, mode: 'lines', name: r.term,
      line: {{ color: dc, width: isHL ? 1.5 : 0.5 }},
      opacity: isHL ? 0.8 : 0.12,
      hovertemplate: r.term + ' (' + r.domain + ')<br>L%{{x}}: %{{y:.1f}} ctrl<extra></extra>',
      showlegend: false,
    }});
  }});

  var titleText = modelSel === 'avg'
    ? 'Control-neighbor trajectories — averaged across models'
    : 'Control-neighbor trajectories — ' + (D.model_short[modelSel] || modelSel);
  Plotly.newPlot('overlayChart', traces, {{
    title: {{ text: titleText, font: {{ size: 13 }} }},
    margin: {{ l: 45, r: 30, t: 40, b: 45 }},
    xaxis: {{ title: 'Layer' }},
    yaxis: {{ title: 'Control neighbors', range: [-0.2, D.k + 0.5] }},
    template: 'simple_white', hovermode: 'closest',
  }}, {{ responsive: true }});
}}

/* ── Init ── */
renderStats();
initDomainFilter();
renderTable();
</script>
</body>
</html>"""


def main() -> None:
    json_path = RESULTS_DIR / "nta_exploration.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found. Run nta_exploration.py first.")
        return

    print(f"Loading {json_path}...")
    with open(json_path, encoding="utf-8") as f:
        nta_results = json.load(f)

    print("Building interactive HTML...")
    html = build_exploration_html(nta_results, k=7)

    out_path = RESULTS_DIR / "nta_exploration.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
