"""
html_builder.py — Costruisce il report HTML interattivo completo.

Assembla tutti i grafici Plotly e le sezioni in un singolo HTML
self-contained con tabs, navigazione e stili moderni.
"""
# ─── Report HTML self-contained ─────────────────────────────────────
# L'HTML generato include:
# - Plotly.js (CDN o minified inline per offline)
# - CSS moderno con variabili CSS
# - JS vanilla per tabs e ordinamento tabelle
# - Grafici Plotly embedded come div
# ─────────────────────────────────────────────────────────────────────

import json
import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np

from ..config import (
    COLORS,
    significance_label,
    jaccard_color,
)

logger = logging.getLogger(__name__)


def build_html_report(
    results: dict,
    output_path: Path,
    light_mode: bool = False,
) -> Path:
    """
    Costruisce report HTML interattivo completo.

    Parameters
    ----------
    results : dict
        Dizionario risultati (da results.json).
    output_path : Path
        Percorso file output.
    light_mode : bool
        Se True, omette dati raw pesanti (RDM, transport plan).

    Returns
    -------
    Path
        Percorso del file generato.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = results.get("metadata", {})
    experiments = results.get("experiments", {})

    # Estrai dati esperimenti
    rsa = experiments.get("1_rsa", {})
    gw = experiments.get("2_gromov_wasserstein", {})
    axes = experiments.get("3_axes", {})
    clust = experiments.get("4_clustering", {})
    nda = experiments.get("5_nda", {})
    umap_data = experiments.get("supplementary_umap", {})

    nda_a = nda.get("part_a_neighborhoods", {})
    nda_b = nda.get("part_b_decompositions", {})

    # Importa funzioni Plotly (con fallback)
    try:
        from .plotly_charts import (
            create_rsa_heatmap,
            create_rsa_scatter,
            create_gw_transport,
            create_axes_scatter,
            create_clustering_dendrogram,
            create_nda_scatter,
            create_nda_network,
            create_umap_scatter,
        )
        plotly_available = True
    except ImportError:
        plotly_available = False
        logger.warning("Plotly non disponibile, report senza grafici interattivi")

    # ─── Genera grafici Plotly ──────────────────────────────────────
    plotly_charts = {}

    if plotly_available:
        # RSA
        if rsa.get("rdm_weird") and not light_mode:
            try:
                plotly_charts["rsa_heatmap"] = create_rsa_heatmap(
                    np.array(rsa["rdm_weird"]),
                    np.array(rsa["rdm_sinic"]),
                    rsa.get("labels", []),
                    rsa.get("spearman_r", 0),
                    rsa.get("p_value", 1),
                )
                plotly_charts["rsa_scatter"] = create_rsa_scatter(
                    np.array(rsa["rdm_weird"]),
                    np.array(rsa["rdm_sinic"]),
                    rsa.get("labels", []),
                    rsa.get("spearman_r", 0),
                )
            except Exception as e:
                logger.warning("Errore generazione grafici RSA: %s", e)

        # GW
        if gw.get("transport_plan") and not light_mode:
            try:
                plotly_charts["gw_transport"] = create_gw_transport(
                    np.array(gw["transport_plan"]),
                    rsa.get("labels", []),
                    gw.get("distance", 0),
                    gw.get("p_value", 1),
                )
            except Exception as e:
                logger.warning("Errore generazione grafici GW: %s", e)

        # Axes
        if axes.get("axes"):
            try:
                plotly_charts["axes_scatter"] = create_axes_scatter(axes["axes"])
            except Exception as e:
                logger.warning("Errore generazione grafici Axes: %s", e)

        # Clustering
        if clust.get("fm_results"):
            try:
                plotly_charts["clustering_fm"] = create_clustering_dendrogram(clust["fm_results"])
            except Exception as e:
                logger.warning("Errore generazione grafici Clustering: %s", e)

        # NDA
        if nda_a.get("per_term"):
            try:
                plotly_charts["nda_scatter"] = create_nda_scatter(
                    nda_a["per_term"],
                    nda_a.get("mean_jaccard", 0),
                    nda_a.get("k", 15),
                )
                plotly_charts["nda_network"] = create_nda_network(nda_a["per_term"])
            except Exception as e:
                logger.warning("Errore generazione grafici NDA: %s", e)

        # UMAP
        if umap_data.get("coordinates"):
            try:
                plotly_charts["umap_scatter"] = create_umap_scatter(
                    umap_data["coordinates"].get("weird", []),
                    umap_data["coordinates"].get("sinic", []),
                )
            except Exception as e:
                logger.warning("Errore generazione grafici UMAP: %s", e)

    # ─── Costruisci sezioni HTML ────────────────────────────────────
    overview_html = _build_overview_section(rsa, gw, axes, clust, nda_a, nda_b, metadata)
    rsa_html = _build_rsa_section(rsa, plotly_charts)
    gw_html = _build_gw_section(gw, plotly_charts)
    axes_html = _build_axes_section(axes, plotly_charts)
    clust_html = _build_clustering_section(clust, plotly_charts)
    nda_a_html = _build_nda_a_section(nda_a, plotly_charts)
    nda_b_html = _build_nda_b_section(nda_b)
    umap_html = _build_umap_section(umap_data, plotly_charts)
    methodology_html = _build_methodology_section()

    # ─── Assembla HTML finale ───────────────────────────────────────
    tabs = [
        ("overview", "Panoramica", overview_html),
        ("rsa", "1. RSA", rsa_html),
        ("gw", "2. Gromov-Wasserstein", gw_html),
        ("axes", "3. Assi Valoriali", axes_html),
        ("clustering", "4. Clustering", clust_html),
        ("nda_a", "5A. Vicinati", nda_a_html),
        ("nda_b", "5B. Decomposizioni", nda_b_html),
        ("umap", "UMAP", umap_html),
        ("methodology", "Metodologia", methodology_html),
    ]

    tab_buttons = "\n".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f'data-tab="{tid}" onclick="switchTab(\'{tid}\')">{label}</button>'
        for i, (tid, label, _) in enumerate(tabs)
    )

    tab_contents = "\n".join(
        f'<div id="tab-{tid}" class="tab-content{" active" if i == 0 else ""}">'
        f'{content}</div>'
        for i, (tid, _, content) in enumerate(tabs)
    )

    version = metadata.get("pipeline_version", "3.0")

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CLS Pipeline v{version} — Report Interattivo</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>{_get_css()}</style>
</head>
<body>
<header>
<h1>CLS Pipeline v{version}</h1>
<p>Analisi Semantica Cross-Linguistica — Confronto WEIRD vs Sinic</p>
</header>

<div class="container">
<div class="tabs">{tab_buttons}</div>
{tab_contents}
</div>

<script>{_get_js()}</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report generated: %s", output_path)

    return output_path


def _build_overview_section(rsa, gw, axes, clust, nda_a, nda_b, metadata) -> str:
    """Costruisce sezione panoramica."""
    n_terms = rsa.get("n_terms") or nda_a.get("n_core_terms") or metadata.get("n_terms", "?")

    rsa_r = rsa.get("spearman_r", 0)
    rsa_p = rsa.get("p_value", 1)
    gw_d = gw.get("distance", 0)
    gw_p = gw.get("p_value", 1)
    nda_j = nda_a.get("mean_jaccard", 0)
    nda_p = nda_a.get("p_value", 1)
    nda_k = nda_a.get("k", 15)

    axes_cards = ""
    for ax in axes.get("axes", []):
        ci = ax.get("bootstrap_ci", {})
        ci_str = f"[{ci.get('ci_lower', 0):.2f}, {ci.get('ci_upper', 0):.2f}]" if ci else ""
        axes_cards += f"""
        <div class="metric-box">
            <div class="value">{ax.get('spearman_r', 0):.3f}</div>
            <div class="label">{ax.get('axis_name', 'Axis')}</div>
            <div class="sub">{significance_label(ax.get('spearman_p', 1))}</div>
        </div>
        """

    fm_cards = ""
    for fm in clust.get("fm_results", []):
        fm_cards += f"""
        <div class="metric-box">
            <div class="value">{fm.get('fm_index', 0):.3f}</div>
            <div class="label">FM (k={fm.get('k', '?')})</div>
            <div class="sub">{significance_label(fm.get('p_value', 1))}</div>
        </div>
        """

    return f"""
    <div class="card">
        <h3>Dashboard Risultati Principali</h3>
        <div class="metrics">
            <div class="metric-box">
                <div class="value">{rsa_r:.4f}</div>
                <div class="label">RSA Spearman ρ</div>
                <div class="sub">{significance_label(rsa_p)}</div>
            </div>
            <div class="metric-box">
                <div class="value">{gw_d:.4f}</div>
                <div class="label">Distanza GW</div>
                <div class="sub">{significance_label(gw_p)}</div>
            </div>
            <div class="metric-box">
                <div class="value">{nda_j:.4f}</div>
                <div class="label">Jaccard Media (k={nda_k})</div>
                <div class="sub">{significance_label(nda_p)}</div>
            </div>
            {axes_cards}
            {fm_cards}
        </div>
        <div class="guide">
            <strong>N termini analizzati:</strong> {n_terms}<br>
            <strong>Seed:</strong> {metadata.get('random_seed', 42)}<br>
            <strong>Pipeline version:</strong> {metadata.get('pipeline_version', '?')}
        </div>
    </div>
    """


def _build_rsa_section(rsa, charts) -> str:
    """Costruisce sezione RSA."""
    chart_html = charts.get("rsa_heatmap", "<p><em>Heatmap non disponibile (modalità light o Plotly mancante)</em></p>")
    scatter_html = charts.get("rsa_scatter", "")

    return f"""
    <div class="card">
        <h3>Esperimento 1: RSA (Representational Similarity Analysis)</h3>
        <div class="guide">
            <strong>Metodo:</strong> Confronta le matrici di dissimilarità (RDM) dei due spazi
            tramite correlazione Spearman. Test: Mantel permutation.
        </div>
        <div class="metrics">
            <div class="metric-box">
                <div class="value">{rsa.get('spearman_r', 0):.4f}</div>
                <div class="label">Spearman ρ</div>
            </div>
            <div class="metric-box">
                <div class="value">{significance_label(rsa.get('p_value', 1))}</div>
                <div class="label">Test di Mantel</div>
            </div>
            <div class="metric-box">
                <div class="value">{rsa.get('n_pairs', 0):,}</div>
                <div class="label">Coppie analizzate</div>
            </div>
        </div>
    </div>
    <div class="card">
        <h3>Heatmap RDM (hover per dettagli)</h3>
        {chart_html}
    </div>
    <div class="card">
        <h3>Correlazione RDM</h3>
        {scatter_html}
    </div>
    """


def _build_gw_section(gw, charts) -> str:
    """Costruisce sezione Gromov-Wasserstein."""
    chart_html = charts.get("gw_transport", "<p><em>Transport plan non disponibile</em></p>")

    return f"""
    <div class="card">
        <h3>Esperimento 2: Distanza di Gromov-Wasserstein</h3>
        <div class="guide">
            <strong>Metodo:</strong> Misura la distanza strutturale tra gli spazi via trasporto
            ottimale entropico (Sinkhorn).
        </div>
        <div class="metrics">
            <div class="metric-box">
                <div class="value">{gw.get('distance', 0):.6f}</div>
                <div class="label">Distanza GW</div>
            </div>
            <div class="metric-box">
                <div class="value">{significance_label(gw.get('p_value', 1))}</div>
                <div class="label">Test permutazione</div>
            </div>
        </div>
    </div>
    <div class="card">
        <h3>Piano di Trasporto (hover per dettagli)</h3>
        {chart_html}
    </div>
    """


def _build_axes_section(axes, charts) -> str:
    """Costruisce sezione Assi Valoriali."""
    chart_html = charts.get("axes_scatter", "<p><em>Grafico non disponibile</em></p>")

    axes_table = ""
    for ax in axes.get("axes", []):
        ci = ax.get("bootstrap_ci", {})
        ci_str = f"[{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}]" if ci else "—"
        axes_table += f"""
        <tr>
            <td>{ax.get('axis_name', '?')}</td>
            <td>{ax.get('spearman_r', 0):.4f}</td>
            <td>{ci_str}</td>
            <td>{significance_label(ax.get('spearman_p', 1))}</td>
        </tr>
        """

    return f"""
    <div class="card">
        <h3>Esperimento 3: Proiezione su Assi Valoriali (Kozlowski)</h3>
        <div class="guide">
            <strong>Metodo:</strong> Proietta i termini su assi semantici costruiti da coppie
            di antonimi. Confronta i ranghi tra i due spazi.
        </div>
        <table class="sortable">
            <thead>
                <tr><th>Asse</th><th>Spearman ρ</th><th>CI 95%</th><th>p-value</th></tr>
            </thead>
            <tbody>{axes_table}</tbody>
        </table>
    </div>
    <div class="card">
        <h3>Scatter Interattivo (usa dropdown per cambiare asse)</h3>
        {chart_html}
    </div>
    """


def _build_clustering_section(clust, charts) -> str:
    """Costruisce sezione Clustering."""
    chart_html = charts.get("clustering_fm", "<p><em>Grafico non disponibile</em></p>")

    return f"""
    <div class="card">
        <h3>Esperimento 4: Clustering Gerarchico</h3>
        <div class="guide">
            <strong>Metodo:</strong> Genera dendrogrammi indipendenti (Ward) e confronta
            le partizioni a diversi k con l'indice Fowlkes-Mallows.
        </div>
        {chart_html}
    </div>
    """


def _build_nda_a_section(nda_a, charts) -> str:
    """Costruisce sezione NDA Part A."""
    scatter_html = charts.get("nda_scatter", "<p><em>Grafico non disponibile</em></p>")
    network_html = charts.get("nda_network", "")

    return f"""
    <div class="card">
        <h3>Esperimento 5A: Confronto Vicinati k-NN</h3>
        <div class="guide">
            <strong>Metodo:</strong> Per ogni termine core, confronta i k vicini più prossimi
            nei due spazi usando l'indice di Jaccard.
        </div>
        <div class="metrics">
            <div class="metric-box">
                <div class="value">{nda_a.get('mean_jaccard', 0):.4f}</div>
                <div class="label">Jaccard Media</div>
            </div>
            <div class="metric-box">
                <div class="value">{significance_label(nda_a.get('p_value', 1))}</div>
                <div class="label">Test permutazione</div>
            </div>
            <div class="metric-box">
                <div class="value">k = {nda_a.get('k', '?')}</div>
                <div class="label">Numero vicini</div>
            </div>
        </div>
    </div>
    <div class="card">
        <h3>Distribuzione Jaccard (hover per dettagli)</h3>
        {scatter_html}
    </div>
    <div class="card">
        {network_html}
    </div>
    """


def _build_nda_b_section(nda_b) -> str:
    """Costruisce sezione NDA Part B."""
    decomps = nda_b.get("decompositions", [])

    cards = []
    for d in decomps:
        j = d.get("jaccard", 0)
        j_color = jaccard_color(j)

        weird_list = "".join(
            f'<li>{n.get("label", "?")} <span style="color:#888;">({n.get("cosine_distance", 0):.3f})</span></li>'
            for n in d.get("weird_neighbors", [])[:8]
        )
        sinic_list = "".join(
            f'<li>{n.get("label", "?")} <span style="color:#888;">({n.get("cosine_distance", 0):.3f})</span></li>'
            for n in d.get("sinic_neighbors", [])[:8]
        )

        cards.append(f"""
        <div class="decomp-card">
            <h4>{d.get('en_formula', '')} | {d.get('zh_formula', '')}</h4>
            <p style="color:#666;font-size:0.9em;">{d.get('jurisprudential_question', '')}</p>
            <p>Jaccard: <span style="display:inline-block;width:{j*150}px;height:12px;background:{j_color};border-radius:3px;"></span> <b>{j:.3f}</b></p>
            <div class="decomp-cols">
                <div class="decomp-col weird">
                    <h5>WEIRD neighbors</h5>
                    <ul class="neighbor-list">{weird_list}</ul>
                </div>
                <div class="decomp-col sinic">
                    <h5>Sinic neighbors</h5>
                    <ul class="neighbor-list">{sinic_list}</ul>
                </div>
            </div>
        </div>
        """)

    return f"""
    <div class="card">
        <h3>Esperimento 5B: Decomposizioni Normative</h3>
        <div class="guide">
            <strong>Metodo:</strong> Calcola il residuo vettoriale r = e(A) - e(B) e cerca
            i vicini del residuo nei due spazi.
        </div>
    </div>
    {"".join(cards)}
    """


def _build_umap_section(umap_data, charts) -> str:
    """Costruisce sezione UMAP."""
    chart_html = charts.get("umap_scatter", "<p><em>Grafico non disponibile</em></p>")

    return f"""
    <div class="card">
        <h3>Supplementary: UMAP Visualization</h3>
        <div class="guide">
            <strong>UMAP</strong> riduce la dimensionalità a 2D preservando la struttura locale.
            Hover sui punti per vedere i dettagli.
        </div>
        {chart_html}
    </div>
    """


def _build_methodology_section() -> str:
    """Costruisce sezione metodologia (abbreviata)."""
    return """
    <div class="card">
        <h3>Nota Metodologica</h3>
        <p>Tutti i p-value sono calcolati con <strong>test di permutazione</strong> (non parametrici).
        I 5 esperimenti testano ipotesi distinte: correlazione di rango (RSA), isomorfismo metrico (GW),
        proiezione assiologica (Axes), concordanza tassonomica (FM), sovrapposizione vicinati (NDA).</p>

        <h4>Significatività</h4>
        <ul>
            <li><strong>***</strong> p &lt; 0.001</li>
            <li><strong>**</strong> p &lt; 0.01</li>
            <li><strong>*</strong> p &lt; 0.05</li>
            <li><strong>n.s.</strong> non significativo</li>
        </ul>

        <h4>Riferimenti</h4>
        <ul>
            <li>Kriegeskorte et al. (2008) - RSA</li>
            <li>Peyré & Cuturi (2019) - Optimal Transport</li>
            <li>Kozlowski et al. (2019) - Semantic Axes</li>
            <li>Fowlkes & Mallows (1983) - FM Index</li>
            <li>Haemmerli et al. (2024) - NDA</li>
        </ul>
    </div>
    """


def _get_css() -> str:
    """Restituisce CSS per il report."""
    return """
:root {
    --weird: #4C72B0;
    --sinic: #DD8452;
    --bg: #fafafa;
    --card-bg: #ffffff;
    --border: #e0e0e0;
    --text: #333333;
    --accent: #2c3e50;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: "Segoe UI", "Noto Sans SC", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
header {
    background: var(--accent);
    color: white;
    padding: 24px;
    text-align: center;
}
header h1 { font-size: 1.6em; margin-bottom: 4px; }
header p { opacity: 0.85; font-size: 0.95em; }

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
    padding: 8px 14px;
    border: 1px solid var(--border);
    border-bottom: none;
    background: #f5f5f5;
    cursor: pointer;
    border-radius: 6px 6px 0 0;
    font-size: 0.85em;
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

.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}
.card h3 { margin-bottom: 12px; color: var(--accent); }

.metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}
.metric-box {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.metric-box .value { font-size: 1.6em; font-weight: 700; color: var(--accent); }
.metric-box .label { font-size: 0.85em; color: #666; margin-top: 4px; }
.metric-box .sub { font-size: 0.75em; color: #888; }

.guide {
    background: #f0f4f8;
    border-left: 4px solid var(--accent);
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.9em;
}

table.sortable { width: 100%; border-collapse: collapse; font-size: 0.85em; }
table.sortable th {
    background: var(--accent);
    color: white;
    padding: 8px 12px;
    text-align: left;
    cursor: pointer;
}
table.sortable th:hover { background: #3d5a80; }
table.sortable td { padding: 6px 12px; border-bottom: 1px solid var(--border); }
table.sortable tr:hover { background: #f0f4f8; }

.decomp-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.decomp-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.decomp-col h5 {
    font-size: 0.85em;
    margin-bottom: 6px;
    padding: 4px 8px;
    border-radius: 4px;
    color: white;
}
.decomp-col.weird h5 { background: var(--weird); }
.decomp-col.sinic h5 { background: var(--sinic); }
.neighbor-list { list-style: none; padding: 0; font-size: 0.82em; }
.neighbor-list li { padding: 2px 0; border-bottom: 1px solid #f0f0f0; }

@media (max-width: 768px) {
    .decomp-cols { grid-template-columns: 1fr; }
    .metrics { grid-template-columns: 1fr 1fr; }
}
"""


def _get_js() -> str:
    """Restituisce JavaScript per il report."""
    return """
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
}

function initSortableTables() {
    document.querySelectorAll('table.sortable').forEach(table => {
        const headers = table.querySelectorAll('th');
        headers.forEach((th, colIdx) => {
            th.addEventListener('click', () => {
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const isAsc = th.classList.contains('sort-asc');
                headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');
                rows.sort((a, b) => {
                    let va = a.cells[colIdx].getAttribute('data-sort') || a.cells[colIdx].textContent;
                    let vb = b.cells[colIdx].getAttribute('data-sort') || b.cells[colIdx].textContent;
                    let na = parseFloat(va), nb = parseFloat(vb);
                    if (!isNaN(na) && !isNaN(nb)) return isAsc ? nb - na : na - nb;
                    return isAsc ? vb.localeCompare(va) : va.localeCompare(vb);
                });
                rows.forEach(r => tbody.appendChild(r));
            });
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initSortableTables();
    switchTab('overview');
});
"""
