"""
generate_html.py — Genera un report HTML interattivo autocontenuto.

Legge results.json e produce un singolo file HTML con:
- Grafici matplotlib → base64 PNG inline
- Tabelle ordinabili e scatter interattivi → vanilla JS + Canvas
- Dati embeddati come JSON inline
- Nessuna dipendenza esterna (no CDN, no framework)

Usage:
    python -m src.visualization.generate_html [--input output/results.json] [--output output/visualization.html]
"""
# ─── Generazione HTML: approccio tecnico ─────────────────────────────
# I grafici complessi (heatmap, dendrogrammi) vengono renderizzati con
# matplotlib e codificati in base64 PNG inline. Gli elementi interattivi
# (scatter, tabelle ordinabili) usano vanilla JS per non dipendere da
# librerie esterne. Il report è autocontenuto: può essere aperto
# offline da qualsiasi browser moderno.
# ─────────────────────────────────────────────────────────────────────

import base64
import io
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

logger = logging.getLogger(__name__)

# Colori principali
WEIRD_COLOR = "#4C72B0"
SINIC_COLOR = "#DD8452"


def _fig_to_base64(fig: plt.Figure) -> str:
    """Render matplotlib figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =====================================================================
# Rendering functions for each section
# =====================================================================

def _render_rdm_heatmaps(rsa: dict) -> str:
    """Render side-by-side RDM heatmaps as base64 PNG."""
    labels = rsa.get("labels", [])
    rdm_w = np.array(rsa["rdm_weird"])
    rdm_s = np.array(rsa["rdm_sinic"])

    short = [l[:15] for l in labels]

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Scaling indipendente per ciascuna matrice: le RDM hanno range
    # molto diversi (WEIRD ~0.07-0.26, Sinic ~0.0-0.75). Una scala
    # fissa [0,1] rende la WEIRD illeggibile. Il confronto quantitativo
    # è affidato al Spearman r; le heatmap servono a mostrare la
    # struttura *interna* di ciascuno spazio.
    triu_w = rdm_w[np.triu_indices(rdm_w.shape[0], k=1)]
    triu_s = rdm_s[np.triu_indices(rdm_s.shape[0], k=1)]

    sns.heatmap(rdm_w, ax=ax1, xticklabels=short, yticklabels=short,
                cmap="viridis", vmin=triu_w.min(), vmax=triu_w.max(),
                square=True, cbar_kws={"shrink": 0.8})
    ax1.set_title(f"WEIRD RDM (range {triu_w.min():.2f}–{triu_w.max():.2f})",
                  fontsize=11, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=5)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax1.get_yticklabels(), rotation=0)

    sns.heatmap(rdm_s, ax=ax2, xticklabels=short, yticklabels=short,
                cmap="viridis", vmin=triu_s.min(), vmax=triu_s.max(),
                square=True, cbar_kws={"shrink": 0.8})
    ax2.set_title(f"Sinic RDM (range {triu_s.min():.2f}–{triu_s.max():.2f})",
                  fontsize=11, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=5)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)

    fig.suptitle(
        f"Matrici di Dissimilarità Rappresentazionale (RDM)\n"
        f"Spearman r = {rsa['spearman_r']:.4f}, p = {rsa['p_value']:.4f}",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])

    return _fig_to_base64(fig)


def _render_rsa_scatter(rsa: dict) -> str:
    """Render RDM correlation scatter as base64 PNG."""
    rdm_w = np.array(rsa["rdm_weird"])
    rdm_s = np.array(rsa["rdm_sinic"])
    n = rdm_w.shape[0]
    triu = np.triu_indices(n, k=1)
    vec_w = rdm_w[triu]
    vec_s = rdm_s[triu]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(vec_w, vec_s, alpha=0.3, s=8, color=WEIRD_COLOR)
    ax.set_xlabel("Distanza coseno (WEIRD)", fontsize=11)
    ax.set_ylabel("Distanza coseno (Sinic)", fontsize=11)
    ax.set_title(f"Correlazione RDM (r={rsa['spearman_r']:.4f})", fontsize=12)
    lims = [0, max(vec_w.max(), vec_s.max()) * 1.05]
    ax.plot(lims, lims, "--", color="grey", linewidth=0.8, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()

    return _fig_to_base64(fig)


def _render_gw_transport(gw: dict) -> str:
    """Render GW transport plan heatmap."""
    tp = np.array(gw["transport_plan"])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(tp, ax=ax, cmap="YlOrRd", square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title(
        f"Piano di Trasporto Gromov-Wasserstein\n"
        f"Distanza = {gw['distance']:.6f}, p = {gw['p_value']:.4f}",
        fontsize=12, fontweight="bold")
    ax.set_xlabel("Termini Sinic")
    ax.set_ylabel("Termini WEIRD")
    fig.tight_layout()

    return _fig_to_base64(fig)


def _render_dendrograms(clust: dict) -> str:
    """Render side-by-side dendrograms."""
    Z_w = np.array(clust["linkage_weird"])
    Z_s = np.array(clust["linkage_sinic"])
    labels = clust.get("labels", [])

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    dendrogram(Z_w, labels=labels, leaf_rotation=45, leaf_font_size=6,
               ax=ax1, color_threshold=0)
    ax1.set_title("WEIRD", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Distanza")

    dendrogram(Z_s, labels=labels, leaf_rotation=45, leaf_font_size=6,
               ax=ax2, color_threshold=0)
    ax2.set_title("Sinic", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Distanza")

    fm_str = " | ".join(
        f"k={r['k']}: FM={r['fm_index']:.3f} (p={r['p_value']:.3f})"
        for r in clust["fm_results"]
    )
    fig.suptitle(
        f"Clustering Gerarchico — Confronto Dendrogrammi\n{fm_str}",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])

    return _fig_to_base64(fig)


def _render_fm_chart(clust: dict) -> str:
    """Render FM index vs k chart."""
    fm_results = clust["fm_results"]
    ks = [r["k"] for r in fm_results]
    fms = [r["fm_index"] for r in fm_results]
    ps = [r["p_value"] for r in fm_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(ks)), fms, color=WEIRD_COLOR, alpha=0.8)

    for i, (k, fm, p) in enumerate(zip(ks, fms, ps)):
        sig = "*" if p < 0.05 else ""
        ax.text(i, fm + 0.02, f"{fm:.3f}{sig}", ha="center", fontsize=9)

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Fowlkes-Mallows Index")
    ax.set_title("Indice FM per diversi valori di k", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()

    return _fig_to_base64(fig)


# =====================================================================
# CSS
# =====================================================================

CSS = """
:root {
    --weird: #4C72B0;
    --sinic: #DD8452;
    --bg: #fafafa;
    --card-bg: #ffffff;
    --border: #e0e0e0;
    --text: #333333;
    --text-light: #666666;
    --accent: #2c3e50;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: "Segoe UI", "Noto Sans SC", "SimHei", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}
.container { max-width: 1300px; margin: 0 auto; padding: 20px; }
header {
    background: var(--accent);
    color: white;
    padding: 30px 0;
    text-align: center;
}
header h1 { font-size: 1.8em; margin-bottom: 5px; }
header p { opacity: 0.85; font-size: 0.95em; }

/* Tabs */
.tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    background: white;
    border-bottom: 2px solid var(--border);
    padding: 10px 10px 0;
    position: sticky;
    top: 0;
    z-index: 100;
}
.tab-btn {
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-bottom: none;
    background: #f5f5f5;
    cursor: pointer;
    border-radius: 6px 6px 0 0;
    font-size: 0.85em;
    transition: background 0.2s;
}
.tab-btn:hover { background: #e8e8e8; }
.tab-btn.active {
    background: white;
    border-bottom: 2px solid white;
    margin-bottom: -2px;
    font-weight: 600;
    color: var(--accent);
}
.tab-content { display: none; padding: 20px 0; }
.tab-content.active { display: block; }

/* Cards */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.card h3 { margin-bottom: 12px; color: var(--accent); }

/* Metric boxes */
.metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-box .value {
    font-size: 1.8em;
    font-weight: 700;
    color: var(--accent);
}
.metric-box .label {
    font-size: 0.85em;
    color: var(--text-light);
    margin-top: 4px;
}
.metric-box .sub {
    font-size: 0.75em;
    color: var(--text-light);
    margin-top: 2px;
}

/* Images */
.plot-img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10px auto;
    border-radius: 4px;
}

/* Tables */
table.sortable {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
}
table.sortable th {
    background: var(--accent);
    color: white;
    padding: 8px 12px;
    text-align: left;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
}
table.sortable th:hover { background: #3d5a80; }
table.sortable th::after { content: ' ⇕'; opacity: 0.5; }
table.sortable th.sort-asc::after { content: ' ↑'; opacity: 1; }
table.sortable th.sort-desc::after { content: ' ↓'; opacity: 1; }
table.sortable td {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
}
table.sortable tr:hover { background: #f0f4f8; }

/* Jaccard bar */
.jaccard-bar {
    display: inline-block;
    height: 14px;
    border-radius: 3px;
    vertical-align: middle;
}

/* Decomposition cards */
.decomp-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.decomp-card h4 { margin-bottom: 8px; }
.decomp-cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.decomp-col h5 {
    font-size: 0.85em;
    margin-bottom: 6px;
    padding: 4px 8px;
    border-radius: 4px;
    color: white;
}
.decomp-col.weird h5 { background: var(--weird); }
.decomp-col.sinic h5 { background: var(--sinic); }
.neighbor-list {
    font-size: 0.82em;
    list-style: none;
    padding: 0;
}
.neighbor-list li {
    padding: 2px 0;
    border-bottom: 1px solid #f0f0f0;
}
.neighbor-list li span.dist {
    float: right;
    color: var(--text-light);
    font-size: 0.9em;
}

/* Guide callouts */
.guide {
    background: #f0f4f8;
    border-left: 4px solid var(--accent);
    padding: 14px 18px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.92em;
    line-height: 1.7;
    color: #444;
}
.guide strong { color: var(--accent); }
.guide em { color: #555; font-style: italic; }
.guide-title {
    font-weight: 700;
    color: var(--accent);
    font-size: 0.95em;
    margin-bottom: 6px;
    display: block;
}
.guide ul { padding-left: 18px; margin: 6px 0 0; }
.guide li { margin-bottom: 4px; }

/* Formulas */
.formula {
    display: block;
    background: #e8edf2;
    border: 1px solid #d0d8e0;
    border-radius: 4px;
    padding: 8px 14px;
    margin: 10px 0;
    font-family: "Cambria Math", "STIX Two Math", "Latin Modern Math", Georgia, serif;
    font-size: 1.05em;
    text-align: center;
    color: #2c3e50;
    line-height: 1.9;
    letter-spacing: 0.3px;
    overflow-x: auto;
}
.formula var { font-style: italic; }
.formula .fn { font-style: normal; }

/* Reference links */
.ref-link {
    display: inline-block;
    font-size: 0.82em;
    color: #2980b9;
    text-decoration: none;
    border-bottom: 1px dotted #2980b9;
    margin: 0 2px;
}
.ref-link:hover { color: #1a5276; border-bottom-style: solid; }
.refs {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #d0d8e0;
    font-size: 0.85em;
    color: #555;
}
.refs strong { font-size: 0.95em; }

/* Methodology */
.methodology p { margin-bottom: 12px; }
.methodology h4 { margin: 16px 0 8px; color: var(--accent); }
.methodology ul { padding-left: 20px; margin-bottom: 12px; }

/* Responsive */
@media (max-width: 768px) {
    .decomp-cols { grid-template-columns: 1fr; }
    .metrics { grid-template-columns: 1fr 1fr; }
    .tab-btn { font-size: 0.75em; padding: 6px 10px; }
}
"""


# =====================================================================
# JavaScript
# =====================================================================

JS = """
// Tab switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
}

// Sortable tables
function initSortableTables() {
    document.querySelectorAll('table.sortable').forEach(table => {
        const headers = table.querySelectorAll('th');
        headers.forEach((th, colIdx) => {
            th.addEventListener('click', () => {
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const isAsc = th.classList.contains('sort-asc');

                headers.forEach(h => { h.classList.remove('sort-asc', 'sort-desc'); });
                th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

                rows.sort((a, b) => {
                    let va = a.cells[colIdx].getAttribute('data-sort') || a.cells[colIdx].textContent;
                    let vb = b.cells[colIdx].getAttribute('data-sort') || b.cells[colIdx].textContent;
                    let na = parseFloat(va), nb = parseFloat(vb);
                    if (!isNaN(na) && !isNaN(nb)) {
                        return isAsc ? nb - na : na - nb;
                    }
                    return isAsc ? vb.localeCompare(va) : va.localeCompare(vb);
                });

                rows.forEach(r => tbody.appendChild(r));
            });
        });
    });
}

// Axes scatter (Canvas-based)
function drawAxesScatter(canvasId, weirdScores, sinicScores, labels) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = 50;

    const allVals = Object.values(weirdScores).concat(Object.values(sinicScores));
    const mn = Math.min(...allVals) - 0.05;
    const mx = Math.max(...allVals) + 0.05;

    function sx(v) { return pad + (v - mn) / (mx - mn) * (W - 2 * pad); }
    function sy(v) { return H - pad - (v - mn) / (mx - mn) * (H - 2 * pad); }

    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        let v = mn + (mx - mn) * i / 4;
        ctx.beginPath();
        ctx.moveTo(sx(v), pad); ctx.lineTo(sx(v), H - pad);
        ctx.moveTo(pad, sy(v)); ctx.lineTo(W - pad, sy(v));
        ctx.stroke();
    }

    // Diagonal
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(sx(mn), sy(mn));
    ctx.lineTo(sx(mx), sy(mx));
    ctx.stroke();
    ctx.setLineDash([]);

    // Points
    const points = [];
    labels.forEach(label => {
        const wx = weirdScores[label], wy = sinicScores[label];
        if (wx === undefined || wy === undefined) return;
        const px = sx(wx), py = sy(wy);
        points.push({label, px, py, wx, wy});

        ctx.fillStyle = '#4C72B0';
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
    });

    // Axes labels
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Score WEIRD →', W / 2, H - 10);
    ctx.save();
    ctx.translate(14, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Score Sinic →', 0, 0);
    ctx.restore();

    // Tooltip
    const tooltip = document.createElement('div');
    tooltip.style.cssText = 'position:absolute;background:rgba(0,0,0,0.8);color:white;padding:4px 8px;border-radius:4px;font-size:12px;pointer-events:none;display:none;z-index:200;';
    canvas.parentNode.style.position = 'relative';
    canvas.parentNode.appendChild(tooltip);

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        let found = null;
        for (const p of points) {
            if (Math.hypot(mx - p.px, my - p.py) < 8) { found = p; break; }
        }
        if (found) {
            tooltip.style.display = 'block';
            tooltip.style.left = (found.px + 10) + 'px';
            tooltip.style.top = (found.py - 20) + 'px';
            tooltip.textContent = found.label + ': W=' + found.wx.toFixed(3) + ' S=' + found.wy.toFixed(3);
        } else {
            tooltip.style.display = 'none';
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initSortableTables();
    switchTab('overview');
});
"""


# =====================================================================
# HTML generation
# =====================================================================

def _sig_label(p: float) -> str:
    if p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.4f} **"
    elif p < 0.05:
        return f"p = {p:.4f} *"
    else:
        return f"p = {p:.4f} (n.s.)"


def _jaccard_color(j: float) -> str:
    """Color for Jaccard bar: low=red, mid=yellow, high=green."""
    if j < 0.2:
        return "#e74c3c"
    elif j < 0.4:
        return "#e67e22"
    elif j < 0.6:
        return "#f1c40f"
    else:
        return "#2ecc71"


def generate_html(results: dict, output_path: Path) -> Path:
    """
    Generate self-contained interactive HTML report from results dict.

    Parameters
    ----------
    results : dict
        The full results dictionary (from results.json).
    output_path : Path
        Path for the output HTML file.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract sections — results.json has structure:
    # {"metadata": {...}, "experiments": {"1_rsa": {...}, "2_gromov_wasserstein": {...}, ...}}
    experiments = results.get("experiments", {})
    rsa = experiments.get("1_rsa", {})
    gw = experiments.get("2_gromov_wasserstein", {})
    axes = experiments.get("3_axes", {})
    clust = experiments.get("4_clustering", {})
    nda = experiments.get("5_nda", {})
    nda_a = nda.get("part_a_neighborhoods", {})
    nda_b = nda.get("part_b_decompositions", {})
    metadata = results.get("metadata", {})

    # ─── Pre-render matplotlib plots ───
    plots = {}
    if rsa.get("rdm_weird"):
        plots["rdm_heatmaps"] = _render_rdm_heatmaps(rsa)
        plots["rsa_scatter"] = _render_rsa_scatter(rsa)
    if gw.get("transport_plan"):
        plots["gw_transport"] = _render_gw_transport(gw)
    if clust.get("linkage_weird"):
        plots["dendrograms"] = _render_dendrograms(clust)
        plots["fm_chart"] = _render_fm_chart(clust)

    # ─── Build HTML sections ───
    overview_html = _build_overview(rsa, gw, axes, clust, nda_a, nda_b, metadata)
    rsa_html = _build_rsa_tab(rsa, plots)
    gw_html = _build_gw_tab(gw, plots)
    axes_html = _build_axes_tab(axes)
    clust_html = _build_clustering_tab(clust, plots)
    nda_a_html = _build_nda_a_tab(nda_a)
    nda_b_html = _build_nda_b_tab(nda_b)
    methodology_html = _build_methodology_tab()

    # ─── Assemble ───
    tabs = [
        ("overview", "Panoramica", overview_html),
        ("rsa", "1. RSA", rsa_html),
        ("gw", "2. Gromov-Wasserstein", gw_html),
        ("axes", "3. Assi Valoriali", axes_html),
        ("clustering", "4. Clustering", clust_html),
        ("nda_a", "5A. Vicinati", nda_a_html),
        ("nda_b", "5B. Decomposizioni", nda_b_html),
        ("methodology", "Metodologia", methodology_html),
    ]

    tab_buttons = "\n".join(
        f'<button class="tab-btn{"  active" if i == 0 else ""}" '
        f'data-tab="{tid}" onclick="switchTab(\'{tid}\')">{label}</button>'
        for i, (tid, label, _) in enumerate(tabs)
    )

    tab_contents = "\n".join(
        f'<div id="tab-{tid}" class="tab-content{"  active" if i == 0 else ""}">'
        f'{content}</div>'
        for i, (tid, _, content) in enumerate(tabs)
    )

    version = metadata.get("pipeline_version", "2.0")
    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CLS Pipeline — Report Interattivo</title>
<style>{CSS}</style>
</head>
<body>
<header>
<h1>CLS Pipeline v{version} — Analisi Semantica Cross-Linguistica</h1>
<p>Confronto strutturale tra spazi embedding WEIRD e Sinic per concetti giuridici</p>
</header>

<div class="container">
<div class="tabs">{tab_buttons}</div>
{tab_contents}
</div>

<script>{JS}</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML visualization generated: %s", output_path)
    return output_path


# =====================================================================
# Tab builders
# =====================================================================

def _build_overview(rsa, gw, axes, clust, nda_a, nda_b, metadata):
    """Build the overview dashboard tab."""
    n_terms = metadata.get("n_terms", "?")

    rsa_r = rsa.get("spearman_r", 0)
    rsa_p = rsa.get("p_value", 1)
    gw_d = gw.get("distance", 0)
    gw_p = gw.get("p_value", 1)
    nda_j = nda_a.get("mean_jaccard", 0)
    nda_p = nda_a.get("p_value", 1)

    axes_data = axes.get("axes", [])
    axes_summary = ""
    for ax in axes_data:
        axes_summary += (
            f'<div class="metric-box">'
            f'<div class="value">{ax.get("spearman_r", 0):.3f}</div>'
            f'<div class="label">Asse: {ax.get("axis_name", "?")}</div>'
            f'<div class="sub">{_sig_label(ax.get("spearman_p", 1))}</div>'
            f'</div>'
        )

    fm_results = clust.get("fm_results", [])
    fm_summary = ""
    for fm in fm_results:
        fm_summary += (
            f'<div class="metric-box">'
            f'<div class="value">{fm.get("fm_index", 0):.3f}</div>'
            f'<div class="label">FM (k={fm.get("k", "?")})</div>'
            f'<div class="sub">{_sig_label(fm.get("p_value", 1))}</div>'
            f'</div>'
        )

    return f"""
    <div class="card">
    <h3>Dashboard — Risultati Principali</h3>
    <div class="guide">
        <span class="guide-title">Come leggere questa pagina</span>
        Ogni riquadro qui sotto riassume un esperimento. I numeri indicano il grado
        di somiglianza (o divergenza) tra il modo in cui un modello linguistico
        <strong>occidentale</strong> (WEIRD) e uno <strong>sinico</strong> organizzano
        i medesimi {n_terms} concetti giuridici. Sotto ciascun numero compare il
        <em>p-value</em>: se p &lt; 0.05, la differenza osservata non è spiegabile
        con il solo caso (è statisticamente significativa).
    </div>
    <div class="metrics">
        <div class="metric-box">
            <div class="value">{rsa_r:.4f}</div>
            <div class="label">RSA Spearman r</div>
            <div class="sub">{_sig_label(rsa_p)}</div>
        </div>
        <div class="metric-box">
            <div class="value">{gw_d:.4f}</div>
            <div class="label">Distanza GW</div>
            <div class="sub">{_sig_label(gw_p)}</div>
        </div>
        <div class="metric-box">
            <div class="value">{nda_j:.4f}</div>
            <div class="label">Jaccard Media (NDA)</div>
            <div class="sub">{_sig_label(nda_p)}</div>
        </div>
        {axes_summary}
        {fm_summary}
    </div>
    <div class="guide">
        <span class="guide-title">Chiave di lettura rapida</span>
        <ul>
            <li><strong>RSA (Spearman r)</strong> — Va da -1 a +1. Valori vicini a +1 significano
                che i due modelli percepiscono le <em>stesse</em> coppie di concetti come
                vicine o lontane: le geometrie interne si somigliano. Valori bassi indicano
                che le strutture semantiche divergono.</li>
            <li><strong>Distanza GW</strong> — Quanto "lavoro" serve per deformare una struttura
                nell'altra. Valori prossimi a 0 = strutture quasi identiche. Valori alti =
                le relazioni tra concetti sono organizzate in modo diverso.</li>
            <li><strong>Jaccard Media (NDA)</strong> — Va da 0 a 1. Misura quanti "vicini
                semantici" ha in comune ciascun concetto nei due modelli. 1 = stessi vicini,
                0 = vicini completamente diversi.</li>
            <li><strong>Assi valoriali</strong> — Correlazione nell'ordine di rango quando
                i concetti sono proiettati su dimensioni culturali (es. individuale↔collettivo).</li>
            <li><strong>FM (Fowlkes-Mallows)</strong> — Quanto i raggruppamenti tassonomici
                coincidono tra i due modelli. FM = 1 = stesse categorie; FM = 0 = categorie
                completamente diverse. k indica il numero di gruppi.</li>
        </ul>
    </div>
    </div>

    <div class="card">
    <h3>Sintesi Narrativa</h3>
    <p>L'analisi confronta le strutture semantiche di {n_terms} concetti giuridici
       attraverso 5 esperimenti complementari, ciascuno con una prospettiva diversa
       sulla divergenza WEIRD/Sinic:</p>
    <ul style="padding-left:20px;margin-top:8px;">
        <li><strong>RSA</strong> (r={rsa_r:.3f}): correlazione tra le geometrie interne dei due spazi —
            verifica se le "distanze concettuali" sono preservate tra le due tradizioni.</li>
        <li><strong>GW</strong> (d={gw_d:.4f}): distanza strutturale via trasporto ottimale —
            quantifica lo sforzo necessario per trasformare una struttura nell'altra.</li>
        <li><strong>Assi valoriali</strong>: proiezione su dimensioni culturali (metodo Kozlowski) —
            ciascun concetto riceve un punteggio lungo assi come individuale↔collettivo.</li>
        <li><strong>Clustering</strong>: confronto tassonomico — i due modelli raggruppano i concetti
            nelle stesse famiglie?</li>
        <li><strong>NDA</strong> (J={nda_j:.3f}): analisi dei vicinati semantici e
            decomposizioni normative — quali concetti gravitano intorno a ciascun termine?</li>
    </ul>
    </div>
    """


def _build_rsa_tab(rsa, plots):
    heatmap_img = ""
    if "rdm_heatmaps" in plots:
        heatmap_img = f'<img class="plot-img" src="data:image/png;base64,{plots["rdm_heatmaps"]}" alt="RDM Heatmaps">'
    scatter_img = ""
    if "rsa_scatter" in plots:
        scatter_img = f'<img class="plot-img" src="data:image/png;base64,{plots["rsa_scatter"]}" alt="RDM Scatter">'

    rsa_r = rsa.get('spearman_r', 0)
    rsa_p = rsa.get('p_value', 1)

    # Interpretazione dinamica
    if abs(rsa_r) > 0.7:
        rsa_interp = "Le due tradizioni organizzano le relazioni tra concetti giuridici in modo molto simile: le coppie di concetti percepite come vicine in una tradizione lo sono anche nell'altra."
    elif abs(rsa_r) > 0.4:
        rsa_interp = "Le geometrie concettuali mostrano una somiglianza moderata: alcune relazioni sono condivise, ma altre divergono significativamente tra le due tradizioni."
    elif abs(rsa_r) > 0.2:
        rsa_interp = "La somiglianza strutturale è debole: i due sistemi organizzano le relazioni tra concetti giuridici in modo sostanzialmente diverso."
    else:
        rsa_interp = "Le strutture concettuali sono quasi indipendenti: non c'è evidenza che le due tradizioni percepiscano le stesse relazioni tra concetti."

    return f"""
    <div class="card">
    <h3>Esperimento 1: Representational Similarity Analysis (RSA)</h3>
    <div class="guide">
        <span class="guide-title">Cosa misura questo esperimento</span>
        Immaginiamo di prendere tutti i {rsa.get('n_terms', '?')} concetti giuridici e di misurare
        quanto ciascuna coppia è "distante" secondo un modello linguistico. Ad esempio:
        <em>"diritto"</em> e <em>"dovere"</em> sono vicini? <em>"libertà"</em> e <em>"tortura"</em>
        sono lontani? Questa operazione produce una <strong>matrice di dissimilarità</strong> (RDM):
        una tabella in cui ogni cella indica la distanza semantica tra due concetti.
        <br><br>
        L'RSA confronta le due matrici — quella del modello WEIRD e quella del modello Sinic —
        e chiede: <em>l'ordine delle distanze è lo stesso?</em> Se "diritto" e "dovere" sono
        vicini per entrambi i modelli, e "libertà" e "tortura" sono lontani per entrambi,
        allora le strutture si somigliano.
        <span class="formula">
            <var>d</var>(<var>a</var>, <var>b</var>) = 1 &minus;
            <span class="fn">cos</span>(<var>a</var>, <var>b</var>) = 1 &minus;
            (<var>a</var> &middot; <var>b</var>) / (‖<var>a</var>‖ &middot; ‖<var>b</var>‖)
            &emsp;&emsp;
            <var>&rho;</var> = 1 &minus; 6&sum;<var>d<sub>i</sub></var>&sup2; / <var>n</var>(<var>n</var>&sup2; &minus; 1)
        </span>
        <div class="refs">
            <strong>Riferimenti:</strong>
            <a class="ref-link" href="https://doi.org/10.3389/neuro.06.004.2008" target="_blank">Kriegeskorte, Mur &amp; Bandettini (2008) "Representational Similarity Analysis", <em>Frontiers in Systems Neuroscience</em>, 2, 4</a> &mdash;
            <a class="ref-link" href="https://doi.org/10.1158/0008-5472.CAN-67-0477" target="_blank">Mantel (1967) "The Detection of Disease Clustering", <em>Cancer Research</em>, 27(2)</a>
        </div>
    </div>
    <div class="metrics">
        <div class="metric-box">
            <div class="value">{rsa_r:.4f}</div>
            <div class="label">Spearman r</div>
        </div>
        <div class="metric-box">
            <div class="value">{_sig_label(rsa_p)}</div>
            <div class="label">Test di Mantel</div>
        </div>
        <div class="metric-box">
            <div class="value">{rsa.get('n_pairs', 0):,}</div>
            <div class="label">Coppie confrontate</div>
        </div>
    </div>
    <div class="guide">
        <span class="guide-title">Interpretazione</span>
        {rsa_interp}
        Il <em>p-value</em> del test di Mantel ({_sig_label(rsa_p)}) indica se questa
        correlazione è statisticamente significativa o potrebbe essere dovuta al caso.
    </div>
    </div>

    <div class="card">
    <h3>Heatmap — Matrici di Dissimilarità</h3>
    <div class="guide">
        <span class="guide-title">Come leggere questa figura</span>
        Ogni heatmap è una matrice quadrata: righe e colonne rappresentano gli stessi concetti.
        Il <strong>colore</strong> di ogni cella indica la distanza semantica tra i due concetti
        corrispondenti: <em>colori scuri</em> (viola/nero) = concetti molto distanti;
        <em>colori chiari</em> (giallo/verde) = concetti vicini.
        La diagonale è sempre chiara (ogni concetto è identico a sé stesso).
        <br><br>
        Se le due heatmap mostrano <strong>pattern simili</strong> (stesse zone chiare e scure),
        significa che i due modelli organizzano i concetti allo stesso modo.
        Differenze nei pattern rivelano dove le due tradizioni divergono.
        <br><br>
        <strong>Nota sulle scale</strong>: le due heatmap usano scale di colore
        <em>indipendenti</em> (il range numerico è indicato nel titolo di ciascuna).
        Questo è necessario perché i due modelli producono distanze in range molto diversi:
        il modello WEIRD tende a produrre distanze coseno più basse (i concetti giuridici
        sono più "compressi" nello spazio), mentre il modello Sinic li distribuisce su un
        range più ampio. Lo scaling indipendente permette di leggere la struttura interna
        di entrambe le matrici; il confronto quantitativo tra le due è affidato
        al coefficiente Spearman r riportato sopra.
    </div>
    {heatmap_img}
    </div>

    <div class="card">
    <h3>Scatter — Correlazione tra distanze</h3>
    <div class="guide">
        <span class="guide-title">Come leggere questa figura</span>
        Ogni punto rappresenta una <strong>coppia di concetti</strong>. L'asse orizzontale
        mostra la distanza tra quei due concetti nel modello WEIRD; l'asse verticale la distanza
        nel modello Sinic. Se i punti si distribuiscono lungo la
        <em>linea diagonale tratteggiata</em>, le distanze coincidono nei due modelli.
        <br><br>
        Punti lontani dalla diagonale indicano coppie di concetti per cui i due modelli
        sono in disaccordo: ad esempio, due concetti percepiti come vicini dal modello
        occidentale ma lontani da quello sinico, o viceversa.
    </div>
    {scatter_img}
    </div>
    """


def _build_gw_tab(gw, plots):
    transport_img = ""
    if "gw_transport" in plots:
        transport_img = f'<img class="plot-img" src="data:image/png;base64,{plots["gw_transport"]}" alt="GW Transport">'

    interp = gw.get("interpretation", "")
    interp_it = ("Alta anisomorfia strutturale" if interp == "high_anisomorphism"
                 else "Isomorfismo relativo")

    gw_d = gw.get('distance', 0)
    gw_p = gw.get('p_value', 1)

    return f"""
    <div class="card">
    <h3>Esperimento 2: Distanza di Gromov-Wasserstein</h3>
    <div class="guide">
        <span class="guide-title">Cosa misura questo esperimento</span>
        Immaginiamo le relazioni tra i concetti giuridici come una rete di distanze
        — una sorta di "mappa topografica" dello spazio semantico. Ogni modello
        linguistico produce la propria mappa. La distanza di Gromov-Wasserstein
        risponde alla domanda: <em>quanto bisogna "deformare" una mappa per farla
        coincidere con l'altra?</em>
        <br><br>
        A differenza dell'RSA (che confronta solo l'ordine delle distanze),
        il metodo GW cerca l'<strong>allineamento ottimale</strong> tra le due
        strutture, come cercare di sovrapporre due mappe di paesi diversi
        ruotandole, scalandole e deformandole il meno possibile.
        Una distanza GW bassa indica che le strutture sono quasi isomorfe;
        alta indica che sono organizzate in modo fondamentalmente diverso.
        <span class="formula">
            <var>GW</var>(<var>C</var><sub>1</sub>, <var>C</var><sub>2</sub>) =
            min<sub><var>T</var></sub> &sum;<sub><var>i,j,k,l</var></sub>
            <var>L</var>(<var>C</var><sub>1</sub><sup><var>ik</var></sup>,
            <var>C</var><sub>2</sub><sup><var>jl</var></sup>)
            <var>T<sub>ij</sub></var> <var>T<sub>kl</sub></var>
            &emsp; con &emsp;
            <var>L</var>(<var>x</var>, <var>y</var>) = (<var>x</var> &minus; <var>y</var>)&sup2;
        </span>
        <div class="refs">
            <strong>Riferimenti:</strong>
            <a class="ref-link" href="https://doi.org/10.1561/2200000073" target="_blank">Peyr&eacute; &amp; Cuturi (2019) "Computational Optimal Transport", <em>Foundations and Trends in ML</em>, 11(5-6)</a> &mdash;
            <a class="ref-link" href="https://aclanthology.org/D18-1214/" target="_blank">Alvarez-Melis &amp; Jaakkola (2018) "Gromov-Wasserstein Alignment of Word Embedding Spaces", <em>EMNLP</em></a> &mdash;
            <a class="ref-link" href="https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html" target="_blank">Cuturi (2013) "Sinkhorn Distances", <em>NeurIPS</em></a>
        </div>
    </div>
    <div class="metrics">
        <div class="metric-box">
            <div class="value">{gw_d:.6f}</div>
            <div class="label">Distanza GW</div>
        </div>
        <div class="metric-box">
            <div class="value">{_sig_label(gw_p)}</div>
            <div class="label">Test di permutazione</div>
        </div>
        <div class="metric-box">
            <div class="value">{interp_it}</div>
            <div class="label">Interpretazione</div>
        </div>
    </div>
    <div class="guide">
        <span class="guide-title">Interpretazione</span>
        {"La distanza osservata è relativamente bassa, suggerendo che le strutture concettuali dei due modelli sono più simili di quanto ci si aspetterebbe per caso. Le due tradizioni giuridiche, pur diverse, organizzano le relazioni tra concetti con una geometria parzialmente condivisa." if gw_d < 0.1 else "La distanza osservata è elevata, indicando che le strutture concettuali dei due modelli richiedono una deformazione sostanziale per essere allineate. Questo suggerisce che le due tradizioni giuridiche organizzano le relazioni tra concetti in modo strutturalmente diverso."}
    </div>
    </div>

    <div class="card">
    <h3>Piano di Trasporto</h3>
    <div class="guide">
        <span class="guide-title">Come leggere questa figura</span>
        Questa heatmap mostra il <strong>piano di trasporto</strong>: il modo ottimale
        di "associare" i concetti di un modello a quelli dell'altro. Ogni riga
        rappresenta un concetto nello spazio WEIRD; ogni colonna lo stesso concetto
        nello spazio Sinic. Le celle con colori <em>più intensi</em> (rosso scuro)
        indicano accoppiamenti forti: il concetto nella riga WEIRD viene "trasportato"
        prevalentemente verso il concetto nella colonna Sinic.
        <br><br>
        Se la heatmap mostra una <strong>diagonale dominante</strong> (colori intensi
        lungo la diagonale), significa che ciascun concetto viene abbinato
        principalmente a sé stesso nell'altro modello: buona corrispondenza.
        Colori diffusi fuori dalla diagonale indicano che l'allineamento non è
        uno-a-uno: i concetti si "mischiano" tra le due tradizioni.
    </div>
    {transport_img}
    </div>
    """


def _build_axes_tab(axes):
    axes_data = axes.get("axes", [])
    labels = []
    canvas_scripts = []
    cards = []

    for i, ax in enumerate(axes_data):
        ax_name = ax.get("axis_name", f"axis_{i}")
        w_scores = ax.get("weird_scores", {})
        s_scores = ax.get("sinic_scores", {})
        rho = ax.get("spearman_r", 0)
        p = ax.get("spearman_p", 1)
        ci = ax.get("bootstrap_ci", {})
        ci_str = ""
        if ci:
            ci_str = f"CI 95% = [{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}]"

        canvas_id = f"axes-canvas-{i}"
        if not labels:
            labels = list(w_scores.keys())

        # Build sortable table for this axis
        rows = []
        for label in sorted(w_scores.keys()):
            ws = w_scores.get(label, 0)
            ss = s_scores.get(label, 0)
            delta = ws - ss
            rows.append(
                f'<tr><td>{label}</td>'
                f'<td data-sort="{ws:.6f}">{ws:.4f}</td>'
                f'<td data-sort="{ss:.6f}">{ss:.4f}</td>'
                f'<td data-sort="{abs(delta):.6f}" style="color:{"#c0392b" if abs(delta) > 0.1 else "inherit"}">{delta:+.4f}</td></tr>'
            )

        table_html = f"""
        <table class="sortable">
        <thead><tr><th>Termine</th><th>WEIRD</th><th>Sinic</th><th>Δ</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
        </table>
        """

        # Canvas + data for scatter
        w_json = json.dumps(w_scores)
        s_json = json.dumps(s_scores)
        l_json = json.dumps(list(w_scores.keys()))
        canvas_scripts.append(
            f"drawAxesScatter('{canvas_id}', {w_json}, {s_json}, {l_json});"
        )

        cards.append(f"""
        <div class="card">
        <h3>Asse: {ax_name}</h3>
        <p>Spearman ρ = {rho:.4f}, {_sig_label(p)}. {ci_str}</p>
        <div style="text-align:center;margin:16px 0;">
            <canvas id="{canvas_id}" width="500" height="500" style="border:1px solid #e0e0e0;border-radius:4px;max-width:100%;"></canvas>
        </div>
        <details><summary style="cursor:pointer;margin-bottom:8px;">Tabella dettagliata (click per espandere)</summary>
        {table_html}
        </details>
        </div>
        """)

    script = f"<script>document.addEventListener('DOMContentLoaded', () => {{ {''.join(canvas_scripts)} }});</script>"

    return f"""
    <div class="card">
    <h3>Esperimento 3: Proiezione su Assi Valoriali (Kozlowski)</h3>
    <div class="guide">
        <span class="guide-title">Cosa misura questo esperimento</span>
        Alcune dimensioni culturali — come <em>individuale vs. collettivo</em> o
        <em>formale vs. informale</em> — sono implicite nel modo in cui usiamo le parole.
        Questo esperimento le rende esplicite: per ogni dimensione, si costruisce un
        <strong>asse</strong> nello spazio semantico usando coppie di antonimi
        (es. "individuo/collettivo", "indipendenza/conformità"), poi si misura
        dove ogni concetto giuridico si colloca lungo quell'asse.
        <br><br>
        Ogni modello costruisce il <strong>proprio asse</strong> nella propria lingua:
        il modello WEIRD usa coppie inglesi, il modello Sinic usa coppie cinesi.
        Questo evita il bias da traduzione — non stiamo imponendo che "freedom" e
        "自由" definiscano la stessa direzione.
        <br><br>
        La <strong>correlazione Spearman</strong> (ρ) misura se l'<em>ordine di rango</em>
        dei concetti lungo l'asse è lo stesso nei due modelli: ad esempio, se "contratto"
        è più "individualistico" di "legge" in entrambi i modelli.
        <span class="formula">
            <var>asse</var> = <span class="fn">norm</span>( 1/<var>k</var> &sum;<sub><var>i</var>=1..<var>k</var></sub>
            [<var>e</var>(<var>a<sub>i</sub></var>) &minus; <var>e</var>(<var>b<sub>i</sub></var>)] )
            &emsp;&emsp;
            <span class="fn">score</span>(<var>t</var>) = <span class="fn">cos</span>(<var>e</var>(<var>t</var>), <var>asse</var>)
        </span>
        <div class="refs">
            <strong>Riferimenti:</strong>
            <a class="ref-link" href="https://doi.org/10.1177/0003122419877135" target="_blank">Kozlowski, Taddy &amp; Evans (2019) "The Geometry of Culture", <em>American Sociological Review</em>, 84(5), 905-949</a>
        </div>
    </div>
    </div>

    <div class="card">
    <div class="guide">
        <span class="guide-title">Come leggere i grafici scatter</span>
        In ogni grafico, ciascun <strong>punto</strong> rappresenta un concetto giuridico.
        L'asse orizzontale mostra il punteggio nel modello WEIRD, quello verticale nel
        modello Sinic. <em>Passando il mouse sopra un punto</em> si visualizza il nome
        del concetto e i relativi punteggi.
        <br><br>
        Se i punti sono allineati lungo la <strong>diagonale tratteggiata</strong>, i due
        modelli concordano nell'ordine. Punti lontani dalla diagonale rappresentano
        concetti su cui le due tradizioni divergono: ad esempio, un concetto percepito
        come "individualistico" dalla tradizione occidentale ma "collettivistico" da
        quella sinica.
        <br><br>
        La <strong>tabella dettagliata</strong> sotto ogni grafico mostra i punteggi
        esatti e la colonna <strong>Δ</strong> (delta): la differenza tra i due modelli.
        Valori Δ grandi (evidenziati in rosso) indicano i concetti più divergenti.
        Le colonne sono ordinabili cliccando sull'intestazione.
    </div>
    </div>
    {"".join(cards)}
    {script}
    """


def _build_clustering_tab(clust, plots):
    dendro_img = ""
    if "dendrograms" in plots:
        dendro_img = f'<img class="plot-img" src="data:image/png;base64,{plots["dendrograms"]}" alt="Dendrograms">'
    fm_img = ""
    if "fm_chart" in plots:
        fm_img = f'<img class="plot-img" src="data:image/png;base64,{plots["fm_chart"]}" alt="FM Chart">'

    fm_rows = ""
    for fm in clust.get("fm_results", []):
        interp = "Simili" if fm.get("fm_index", 0) >= 0.5 else "Divergenti"
        fm_rows += (
            f'<tr><td>{fm.get("k", "?")}</td>'
            f'<td data-sort="{fm.get("fm_index", 0):.6f}">{fm.get("fm_index", 0):.4f}</td>'
            f'<td data-sort="{fm.get("p_value", 1):.6f}">{_sig_label(fm.get("p_value", 1))}</td>'
            f'<td>{interp}</td></tr>'
        )

    return f"""
    <div class="card">
    <h3>Esperimento 4: Clustering Gerarchico + Fowlkes-Mallows</h3>
    <div class="guide">
        <span class="guide-title">Cosa misura questo esperimento</span>
        Se chiedessimo a un giurista di tradizione occidentale e a uno di tradizione
        sinica di raggruppare gli stessi concetti in famiglie, otterremmo le stesse
        categorie? Questo esperimento risponde a questa domanda computazionalmente.
        <br><br>
        Per ciascun modello, i concetti vengono raggruppati automaticamente in base alla
        loro vicinanza semantica (clustering gerarchico). Poi si confrontano i raggruppamenti
        con l'<strong>indice Fowlkes-Mallows (FM)</strong>: un valore che misura quanto
        le categorie coincidono. FM = 1 significa raggruppamenti identici; FM = 0 significa
        raggruppamenti completamente diversi; FM = 0.5 indica una sovrapposizione parziale.
        <br><br>
        Il confronto è ripetuto per diversi numeri di gruppi (<strong>k</strong>): con pochi
        gruppi (k basso) si catturano le macro-categorie, con molti gruppi (k alto) le
        distinzioni più fini.
        <span class="formula">
            <var>FM</var> = &radic;(<var>PPV</var> &times; <var>TPR</var>)
            &emsp; dove &emsp;
            <var>PPV</var> = <var>TP</var> / (<var>TP</var> + <var>FP</var>)
            &emsp; e &emsp;
            <var>TPR</var> = <var>TP</var> / (<var>TP</var> + <var>FN</var>)
        </span>
        <div class="refs">
            <strong>Riferimenti:</strong>
            <a class="ref-link" href="https://doi.org/10.1080/01621459.1983.10478008" target="_blank">Fowlkes &amp; Mallows (1983) "A Method for Comparing Two Hierarchical Clusterings", <em>JASA</em>, 78(383)</a> &mdash;
            Ward (1963) "Hierarchical Grouping to Optimize an Objective Function", <em>JASA</em>, 58(301)
        </div>
    </div>

    <div class="guide">
        <span class="guide-title">Come leggere la tabella</span>
        <ul>
            <li><strong>k</strong> — numero di gruppi in cui sono divisi i concetti</li>
            <li><strong>FM Index</strong> — grado di concordanza (0 = nulla, 1 = perfetta).
                Valori superiori a 0.5 indicano concordanza; inferiori indicano divergenza.</li>
            <li><strong>p-value</strong> — significatività statistica. Se significativo (p &lt; 0.05),
                la concordanza (o divergenza) osservata non è dovuta al caso.</li>
        </ul>
    </div>
    <table class="sortable">
    <thead><tr><th>k</th><th>FM Index</th><th>p-value</th><th>Interpretazione</th></tr></thead>
    <tbody>{fm_rows}</tbody>
    </table>
    </div>

    <div class="card">
    <h3>Dendrogrammi</h3>
    <div class="guide">
        <span class="guide-title">Come leggere i dendrogrammi</span>
        Un dendrogramma è un <strong>albero di parentela</strong> tra concetti. In basso ci sono
        i singoli termini; man mano che si sale, i termini vengono raggruppati in famiglie
        sempre più ampie. L'altezza a cui due rami si uniscono indica quanto sono
        <em>distanti</em> semanticamente: rami che si uniscono in basso rappresentano
        concetti molto vicini; rami che si uniscono in alto rappresentano concetti lontani.
        <br><br>
        Confrontando il dendrogramma WEIRD con quello Sinic, si può osservare direttamente
        quali concetti sono raggruppati insieme in una tradizione ma separati nell'altra.
    </div>
    {dendro_img}
    </div>

    <div class="card">
    <h3>FM Index per diversi valori di k</h3>
    <div class="guide">
        <span class="guide-title">Come leggere questo grafico</span>
        Ogni barra rappresenta l'indice FM per un diverso livello di granularità (k).
        La <em>linea tratteggiata</em> orizzontale a 0.5 segna la soglia: barre
        sopra questa linea indicano che i raggruppamenti concordano più di quanto
        atteso per caso. L'asterisco (*) indica significatività statistica.
    </div>
    {fm_img}
    </div>
    """


def _build_nda_a_tab(nda_a):
    per_term = nda_a.get("per_term", [])

    rows = []
    for item in per_term:
        j = item.get("jaccard", 0)
        color = _jaccard_color(j)
        bar_width = max(j * 200, 2)
        weird_n = ", ".join(item.get("weird_neighbors", [])[:5])
        sinic_n = ", ".join(item.get("sinic_neighbors", [])[:5])
        shared_n = ", ".join(item.get("shared_neighbors", [])[:5])

        rows.append(
            f'<tr>'
            f'<td>{item.get("term", "?")}</td>'
            f'<td data-sort="{j:.6f}">'
            f'<span class="jaccard-bar" style="width:{bar_width}px;background:{color};"></span> '
            f'{j:.3f}</td>'
            f'<td style="font-size:0.8em;">{weird_n}</td>'
            f'<td style="font-size:0.8em;">{sinic_n}</td>'
            f'<td style="font-size:0.8em;">{shared_n}</td>'
            f'</tr>'
        )

    k_val = nda_a.get('k', 10)
    mean_j = nda_a.get('mean_jaccard', 0)

    return f"""
    <div class="card">
    <h3>Esperimento 5A: Confronto dei Vicinati Semantici</h3>
    <div class="guide">
        <span class="guide-title">Cosa misura questo esperimento</span>
        Per ogni concetto giuridico, il modello linguistico individua i <strong>{k_val}
        concetti più simili</strong> — il suo "vicinato semantico". Ad esempio, nel
        modello occidentale i vicini di "contratto" potrebbero essere "accordo", "patto",
        "obbligazione"; in quello sinico potrebbero essere "accordo", "responsabilità",
        "relazione" — concetti in parte diversi.
        <br><br>
        L'<strong>indice di Jaccard</strong> misura la sovrapposizione tra i due vicinati:
        quanti vicini sono in comune? Jaccard = 1 significa che i {k_val} vicini sono
        identici; Jaccard = 0 che non ce n'è nemmeno uno in comune.
        Concetti con Jaccard basso sono potenziali <strong>"falsi amici giuridici"</strong>:
        parole che sembrano riferirsi allo stesso concetto ma che, nei rispettivi contesti
        culturali, si associano a idee diverse.
        <span class="formula">
            <var>J</var>(<var>A</var>, <var>B</var>) =
            |<var>A</var> &cap; <var>B</var>| / |<var>A</var> &cup; <var>B</var>|
            &emsp; dove &emsp;
            <var>A</var> = <span class="fn">kNN</span><sub>WEIRD</sub>(<var>t</var>),&ensp;
            <var>B</var> = <span class="fn">kNN</span><sub>Sinic</sub>(<var>t</var>)
        </span>
        <div class="refs">
            <strong>Riferimenti:</strong>
            <a class="ref-link" href="https://arxiv.org/abs/2411.08687" target="_blank">Haemmerli et al. (2024) "Neighborhood Divergence Analysis", <em>arXiv:2411.08687</em></a>
        </div>
    </div>
    <div class="metrics">
        <div class="metric-box">
            <div class="value">{mean_j:.4f}</div>
            <div class="label">Jaccard Media</div>
        </div>
        <div class="metric-box">
            <div class="value">{_sig_label(nda_a.get('p_value', 1))}</div>
            <div class="label">Test di permutazione</div>
        </div>
        <div class="metric-box">
            <div class="value">{nda_a.get('n_core_terms', '?')}</div>
            <div class="label">Termini core analizzati</div>
        </div>
    </div>
    <div class="guide">
        <span class="guide-title">Interpretazione complessiva</span>
        {"La Jaccard media è bassa (sotto 0.3): la maggior parte dei concetti ha vicinati semantici molto diversi nei due modelli. Questo suggerisce che i due sistemi giuridici associano idee diverse agli stessi termini — un risultato rilevante per la traduzione giuridica e il diritto comparato." if mean_j < 0.3 else "La Jaccard media è moderata: alcuni concetti condividono vicinati semantici tra le due tradizioni, mentre altri divergono significativamente. I concetti con Jaccard basso meritano attenzione come potenziali fonti di malintesi cross-culturali." if mean_j < 0.6 else "La Jaccard media è alta: la maggior parte dei concetti ha vicinati simili nei due modelli, suggerendo un nucleo semantico condiviso tra le due tradizioni giuridiche."}
    </div>
    </div>

    <div class="card">
    <h3>Tabella dei vicinati — ordinabile per colonna</h3>
    <div class="guide">
        <span class="guide-title">Come leggere la tabella</span>
        <ul>
            <li><strong>Termine</strong> — il concetto giuridico analizzato.</li>
            <li><strong>Jaccard</strong> — sovrapposizione dei vicinati (0-1). La barra
                colorata offre un'indicazione visiva: <span style="color:#e74c3c;">rosso</span>
                = bassa, <span style="color:#e67e22;">arancione</span> = moderata,
                <span style="color:#f1c40f;">giallo</span> = discreta,
                <span style="color:#2ecc71;">verde</span> = alta.</li>
            <li><strong>Vicini WEIRD</strong> — i concetti più simili secondo il modello occidentale.</li>
            <li><strong>Vicini Sinic</strong> — i concetti più simili secondo il modello sinico.</li>
            <li><strong>Condivisi</strong> — i concetti che compaiono in entrambi i vicinati.</li>
        </ul>
        Le colonne sono ordinabili cliccando sull'intestazione. Ordinate per
        Jaccard crescente per trovare i concetti più divergenti (potenziali "falsi amici").
    </div>
    <table class="sortable">
    <thead><tr><th>Termine</th><th>Jaccard</th><th>Vicini WEIRD</th><th>Vicini Sinic</th><th>Condivisi</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
    """


def _build_nda_b_tab(nda_b):
    decomps = nda_b.get("decompositions", [])
    cards = []

    for d in decomps:
        w_neighbors = d.get("weird_neighbors", [])
        s_neighbors = d.get("sinic_neighbors", [])

        w_list = "".join(
            f'<li>{n.get("label", "?")}<span class="dist">{n.get("cosine_distance", 0):.3f}</span></li>'
            for n in w_neighbors[:10]
        )
        s_list = "".join(
            f'<li>{n.get("label", "?")}<span class="dist">{n.get("cosine_distance", 0):.3f}</span></li>'
            for n in s_neighbors[:10]
        )

        j = d.get("jaccard", 0)
        color = _jaccard_color(j)

        cards.append(f"""
        <div class="decomp-card">
        <h4>{d.get("en_formula", "")} | {d.get("zh_formula", "")}</h4>
        <p style="color:var(--text-light);font-size:0.9em;margin-bottom:8px;">
            {d.get("jurisprudential_question", "")}
        </p>
        <p>Jaccard: <span class="jaccard-bar" style="width:{max(j*150,2)}px;background:{color};"></span>
           <strong>{j:.3f}</strong></p>
        <div class="decomp-cols">
            <div class="decomp-col weird">
                <h5>WEIRD — Vicini del residuo</h5>
                <ul class="neighbor-list">{w_list}</ul>
            </div>
            <div class="decomp-col sinic">
                <h5>Sinic — Vicini del residuo</h5>
                <ul class="neighbor-list">{s_list}</ul>
            </div>
        </div>
        </div>
        """)

    mean_j = nda_b.get("mean_jaccard", 0)

    return f"""
    <div class="card">
    <h3>Esperimento 5B: Decomposizioni Normative</h3>
    <div class="guide">
        <span class="guide-title">Cosa misura questo esperimento</span>
        Questo esperimento pone <strong>domande giurisprudenziali</strong> direttamente
        al modello linguistico, usando un'operazione chiamata <em>aritmetica vettoriale</em>.
        L'intuizione è semplice: se sottraiamo il concetto B dal concetto A, il
        "residuo" cattura ciò che A ha in più rispetto a B — la sua <em>differenza specifica</em>.
        <br><br>
        <strong>Esempio</strong>: "giustizia − legge = ?" chiede al modello: <em>cosa resta
        della giustizia quando togliamo la componente legale?</em> I concetti più vicini al
        residuo rivelano cosa il modello associa a quella differenza — ad esempio,
        "equità", "morale", "coscienza". Se il modello WEIRD e quello Sinic producono
        residui diversi, significa che le due tradizioni concepiscono quella relazione
        concettuale in modo diverso.
        <br><br>
        La <strong>Jaccard</strong> tra i vicini dei residui misura la concordanza: valori
        bassi indicano interpretazioni divergenti della stessa relazione giurisprudenziale.
        Jaccard media: <strong>{mean_j:.3f}</strong>
        <span class="formula">
            <var>r</var> = <span class="fn">norm</span>(<var>e</var>(<var>A</var>) &minus; <var>e</var>(<var>B</var>))
            &emsp;&emsp;
            <span class="fn">vicini</span>(<var>r</var>) = <span class="fn">kNN</span>(<var>r</var>, <var>k</var>)
        </span>
        <div class="refs">
            <strong>Riferimenti:</strong>
            <a class="ref-link" href="https://aclanthology.org/N13-1090/" target="_blank">Mikolov et al. (2013) "Linguistic Regularities in Continuous Space Word Representations", <em>NAACL-HLT</em></a> &mdash;
            <a class="ref-link" href="https://arxiv.org/abs/2411.08687" target="_blank">Haemmerli et al. (2024) "Neighborhood Divergence Analysis", <em>arXiv:2411.08687</em></a>
        </div>
    </div>
    </div>

    <div class="card">
    <div class="guide">
        <span class="guide-title">Come leggere le schede qui sotto</span>
        Ogni scheda rappresenta una <strong>decomposizione</strong>: una domanda
        giurisprudenziale codificata come operazione vettoriale.
        <ul>
            <li>L'<strong>intestazione</strong> mostra la formula (A − B) in entrambe le lingue.</li>
            <li>La <strong>domanda giurisprudenziale</strong> spiega cosa si sta indagando.</li>
            <li>La <strong>Jaccard</strong> indica quanto i due modelli concordano sulla risposta.</li>
            <li>Le due colonne (<span style="color:var(--weird);">WEIRD</span> e
                <span style="color:var(--sinic);">Sinic</span>) mostrano i 10 concetti più vicini
                al residuo in ciascun modello. Il numero a destra è la distanza: valori
                bassi = concetti molto associati al residuo.</li>
        </ul>
        Se le due colonne contengono concetti simili, i modelli "rispondono" allo
        stesso modo. Se contengono concetti diversi, le due tradizioni interpretano
        quella relazione giuridica in modo distinto.
    </div>
    </div>
    {"".join(cards)}
    """


def _build_methodology_tab():
    return """
    <div class="card methodology">
    <h3>Nota Metodologica — Guida per il Giurista</h3>
    <p>Questo report presenta i risultati di un'analisi computazionale che confronta
       le strutture semantiche di concetti giuridici in due tradizioni culturali:
       <strong>WEIRD</strong> (Western, Educated, Industrialized, Rich, Democratic) e
       <strong>Sinic</strong> (tradizione giuridica cinese).</p>

    <h4>1. Cosa sono gli "embedding"?</h4>
    <p>Un modello linguistico addestrato su grandi quantità di testo — documenti
       giuridici, articoli, enciclopedie, legislazione — impara a rappresentare
       ogni parola o espressione come un <strong>punto in uno spazio matematico</strong>
       ad alta dimensione (centinaia di coordinate). Le parole con significati
       simili finiscono in punti vicini; quelle con significati diversi finiscono lontane.</p>
    <p>L'analogia più intuitiva è una <strong>mappa</strong>: così come su una mappa
       geografica le città vicine sono disegnate vicine, nella "mappa semantica"
       del modello i concetti affini (es. "contratto" e "accordo") si trovano
       in posizioni adiacenti, mentre concetti distanti (es. "contratto" e "tortura")
       sono separati da grandi distanze.</p>
    <p>Due modelli addestrati su corpora diversi — uno prevalentemente inglese/occidentale,
       l'altro prevalentemente cinese — producono <strong>mappe diverse</strong>. Questa
       analisi confronta sistematicamente le due mappe per capire dove convergono
       e dove divergono.</p>

    <h4>2. Come si leggono i risultati?</h4>
    <ul>
        <li><strong>RSA (Esperimento 1)</strong> — Costruisce una "tabella delle distanze"
            tra tutti i concetti in ciascun modello, poi verifica se le due tabelle
            sono correlate. <em>Analogia</em>: come chiedere a un giurista italiano e
            a uno cinese di ordinare tutte le coppie di concetti da "più simili"
            a "più diversi", e verificare se le graduatorie coincidono.</li>
        <li><strong>Gromov-Wasserstein (Esperimento 2)</strong> — Cerca l'allineamento
            ottimale tra le due strutture, misurando la "distorsione" minima necessaria.
            <em>Analogia</em>: come tentare di sovrapporre due mappe di costellazioni
            — si possono ruotare e scalare, ma se le stelle sono disposte diversamente,
            servirà una deformazione significativa.</li>
        <li><strong>Assi valoriali (Esperimento 3)</strong> — Proietta i concetti su
            dimensioni culturali esplicite (es. individuale↔collettivo), costruite
            indipendentemente per ogni lingua. <em>Analogia</em>: come posizionare
            i concetti su un termometro che misura quanto sono "individualistici"
            o "collettivistici", e confrontare le posizioni nei due modelli.</li>
        <li><strong>Clustering (Esperimento 4)</strong> — Raggruppa automaticamente i
            concetti in famiglie e verifica se le famiglie coincidono. <em>Analogia</em>:
            come verificare se un sistema di classificazione giuridico occidentale
            (diritti reali, diritti personali, ecc.) corrisponde alle categorie
            implicite nel sistema sinico.</li>
        <li><strong>NDA — Parte A (Esperimento 5A)</strong> — Per ogni concetto, confronta
            i "vicini più prossimi" nei due modelli. <em>Analogia</em>: chiedere a
            un giurista occidentale e a uno cinese: "a cosa associ il concetto di
            proprietà?" e confrontare le risposte.</li>
        <li><strong>NDA — Parte B (Esperimento 5B)</strong> — Usa l'aritmetica vettoriale
            per porre domande giurisprudenziali al modello (es. "giustizia − legge = ?").
            <em>Analogia</em>: chiedere ai due giuristi: "cosa resta della giustizia
            se togliamo la componente legale?" e confrontare le risposte.</li>
    </ul>

    <h4>3. Significatività statistica: il ruolo del p-value</h4>
    <p>Ogni risultato è accompagnato da un <strong>p-value</strong>, che risponde alla domanda:
       <em>questo risultato potrebbe essere dovuto al caso?</em></p>
    <p>Il metodo utilizzato è il <strong>test di permutazione</strong>: si rimescolano i dati
       migliaia di volte in modo casuale, e si verifica quante volte il risultato casuale
       è altrettanto estremo di quello osservato. Se succede raramente (meno del 5% delle
       volte), il risultato è considerato significativo. La formula usata include la
       correzione di Phipson &amp; Smyth che garantisce che il p-value non sia mai
       esattamente zero:</p>
    <span class="formula" style="text-align:center;">
        <var>p</var> = (&sum;<sub><var>i</var>=1..<var>B</var></sub>
        <strong>1</strong>[<var>t</var>*<sub><var>i</var></sub> &ge; <var>t</var><sub>obs</sub>] + 1)
        / (<var>B</var> + 1)
    </span>
    <ul>
        <li><strong>*</strong> p < 0.05 — significativo (meno del 5% di probabilità che sia
            dovuto al caso)</li>
        <li><strong>**</strong> p < 0.01 — molto significativo (meno dell'1%)</li>
        <li><strong>***</strong> p < 0.001 — altamente significativo (meno dello 0.1%)</li>
        <li><strong>n.s.</strong> — non significativo (il risultato potrebbe essere casuale)</li>
    </ul>
    <p>La scelta del test non parametrico (permutazione) anziché parametrico (t-test)
       è motivata dal fatto che gli embedding linguistici non seguono distribuzioni
       normali e presentano dipendenze strutturali tra le osservazioni.</p>
    <p style="font-size:0.9em;color:#555;margin-top:6px;">
       <strong>Rif.:</strong>
       <a class="ref-link" href="https://doi.org/10.2202/1544-6115.1585" target="_blank">Phipson &amp; Smyth (2010) "Permutation P-values Should Never Be Zero", <em>Stat. Appl. Genet. Mol. Biol.</em></a> &mdash;
       <a class="ref-link" href="https://doi.org/10.1214/aos/1176344552" target="_blank">Efron (1979) "Bootstrap Methods: Another Look at the Jackknife", <em>Annals of Statistics</em>, 7(1)</a> &mdash;
       Good (2005) <em>Permutation, Parametric, and Bootstrap Tests of Hypotheses</em>, 3rd ed., Springer
    </p>

    <h4>4. Glossario dei termini tecnici</h4>
    <ul>
        <li><strong>Cosine distance</strong> — Misura della dissimilarità tra due vettori
            basata sull'angolo tra di essi. Va da 0 (identici) a 1 (opposti). Insensibile
            alla "lunghezza" del vettore, misura solo la direzione semantica.
            <br><span class="formula" style="margin:4px 0;font-size:0.95em;">
            <var>d</var>(<var>a</var>, <var>b</var>) = 1 &minus;
            (<var>a</var> &middot; <var>b</var>) / (‖<var>a</var>‖ &middot; ‖<var>b</var>‖)
            &emsp; &isin; [0, 1]
            </span></li>
        <li><strong>Spearman &rho; (rho)</strong> — Coefficiente di correlazione basato sui ranghi,
            non sui valori assoluti. Misura se due classifiche sono concordi: &rho; = +1
            significa ordine identico, &rho; = 0 nessuna relazione, &rho; = &minus;1 ordine inverso.
            <br><span class="formula" style="margin:4px 0;font-size:0.95em;">
            <var>&rho;</var> = 1 &minus; 6&sum;<var>d<sub>i</sub></var>&sup2;
            / <var>n</var>(<var>n</var>&sup2; &minus; 1)
            &emsp; dove <var>d<sub>i</sub></var> = differenza tra i ranghi
            </span></li>
        <li><strong>Jaccard</strong> — Indice di sovrapposizione tra due insiemi: il rapporto
            tra gli elementi in comune e gli elementi totali. Va da 0 (nessun elemento
            in comune) a 1 (insiemi identici).
            <br><span class="formula" style="margin:4px 0;font-size:0.95em;">
            <var>J</var>(<var>A</var>, <var>B</var>) =
            |<var>A</var> &cap; <var>B</var>| / |<var>A</var> &cup; <var>B</var>|
            &emsp; &isin; [0, 1]
            </span></li>
        <li><strong>Fowlkes-Mallows (FM)</strong> — Misura la concordanza tra due raggruppamenti.
            È la media geometrica tra precisione (PPV) e sensibilità (TPR).
            <br><span class="formula" style="margin:4px 0;font-size:0.95em;">
            <var>FM</var> = &radic;(<var>PPV</var> &times; <var>TPR</var>)
            &emsp; &isin; [0, 1]
            </span></li>
        <li><strong>Bootstrap</strong> — Tecnica statistica che stima l'incertezza di un risultato
            ricampionando ripetutamente i dati <em>con rimpiazzo</em>, senza assumere una distribuzione
            teorica.
            (<a class="ref-link" href="https://doi.org/10.1214/aos/1176344552" target="_blank">Efron, 1979</a>)</li>
        <li><strong>k-NN</strong> — "k vicini più prossimi" (k-Nearest Neighbors): i k concetti
            più simili a un dato concetto nello spazio embedding, misurati per distanza coseno.</li>
        <li><strong>RDM</strong> — Matrice di Dissimilarità Rappresentazionale: tabella quadrata
            <var>N</var>&times;<var>N</var> simmetrica, in cui la cella (<var>i</var>,<var>j</var>)
            contiene la distanza coseno tra il concetto <var>i</var> e il concetto <var>j</var>.
            (<a class="ref-link" href="https://doi.org/10.3389/neuro.06.004.2008" target="_blank">Kriegeskorte et al., 2008</a>)</li>
        <li><strong>Gromov-Wasserstein</strong> — Distanza tra spazi metrici basata sul trasporto
            ottimale. Minimizza la distorsione nell'accoppiamento tra le strutture interne.
            (<a class="ref-link" href="https://doi.org/10.1561/2200000073" target="_blank">Peyr&eacute; &amp; Cuturi, 2019</a>)</li>
    </ul>

    <h4>5. Limitazioni</h4>
    <p>I modelli di embedding catturano <strong>pattern statistici</strong> nei testi di
       addestramento, non necessariamente la dottrina giuridica ufficiale.
       I risultati riflettono la <em>semantica distribuita</em> — il modo in cui i
       concetti vengono usati nel linguaggio corrente — e non cosa significano
       formalmente in senso giurisprudenziale. Le conclusioni vanno quindi intese
       come indicazioni sulla <em>percezione culturale</em> dei concetti giuridici,
       non come giudizi sulla loro definizione normativa.</p>
    <p>Inoltre, la selezione dei termini e delle coppie di antonimi influenza i risultati.
       Questo strumento è progettato per generare ipotesi e orientare l'indagine
       comparatistica, non per sostituire l'analisi dottrinale.</p>
    </div>
    """


# =====================================================================
# CLI entry point
# =====================================================================

def main():
    """Generate HTML visualization from results.json."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interactive HTML visualization from CLS results",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to results.json",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path for output HTML file",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Find project root
    root = Path(__file__).resolve().parent.parent.parent
    input_path = args.input or root / "output" / "results.json"
    output_path = args.output or root / "output" / "visualization.html"

    if not input_path.exists():
        logger.error("Results file not found: %s", input_path)
        logger.error("Run the pipeline first: python -m src.cli run")
        return 1

    logger.info("Reading results from: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    generate_html(results, output_path)
    logger.info("HTML visualization saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
