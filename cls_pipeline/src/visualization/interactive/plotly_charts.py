"""
plotly_charts.py — Grafici Plotly per visualizzazione interattiva.

Ogni funzione restituisce un oggetto Plotly Figure o il JSON/HTML
necessario per l'embedding nel report.
"""
# ─── Plotly per interattività ───────────────────────────────────────
# A differenza di matplotlib (statico), Plotly permette:
# - Hover con dettagli (termine, distanza, etc.)
# - Zoom/pan nativi
# - Filtri interattivi (slider, checkbox)
# L'HTML generato è self-contained (include plotly.js minified).
# ─────────────────────────────────────────────────────────────────────

import json
import logging
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Check if plotly is available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly non disponibile, grafici interattivi disabilitati")


from ..config import (
    COLORS,
    PLOTLY_COLORS,
    PLOTLY_LAYOUT,
    DOMAIN_COLORS,
    jaccard_color,
    significance_label,
)


def _check_plotly():
    """Verifica disponibilità Plotly."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly non installato. Installa con: pip install plotly")


def create_rsa_heatmap(
    rdm_weird: np.ndarray,
    rdm_sinic: np.ndarray,
    labels: list[str],
    spearman_r: float,
    p_value: float,
) -> str:
    """
    Crea heatmap interattiva delle RDM.

    Hover mostra: termine_i, termine_j, distanza WEIRD, distanza Sinic.

    Returns
    -------
    str
        HTML con div Plotly embedded.
    """
    _check_plotly()

    n = len(labels)

    # Crea testo hover personalizzato
    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(
                f"<b>{labels[i]}</b> ↔ <b>{labels[j]}</b><br>"
                f"WEIRD: {rdm_weird[i,j]:.4f}<br>"
                f"Sinic: {rdm_sinic[i,j]:.4f}<br>"
                f"Δ: {rdm_weird[i,j] - rdm_sinic[i,j]:+.4f}"
            )
        hover_text.append(row)

    # Figura con subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["WEIRD RDM", "Sinic RDM"],
        horizontal_spacing=0.1,
    )

    # WEIRD heatmap
    fig.add_trace(
        go.Heatmap(
            z=rdm_weird,
            x=labels,
            y=labels,
            colorscale="Viridis",
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
            colorbar=dict(title="Distance", x=0.45),
        ),
        row=1, col=1,
    )

    # Sinic heatmap
    fig.add_trace(
        go.Heatmap(
            z=rdm_sinic,
            x=labels,
            y=labels,
            colorscale="Viridis",
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
            colorbar=dict(title="Distance", x=1.0),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"RDM Heatmaps (N={n})<br><sup>Spearman ρ = {spearman_r:.4f}, {significance_label(p_value)}</sup>",
            x=0.5,
        ),
        height=700,
        **PLOTLY_LAYOUT,
    )

    # Nascondi tick labels per leggibilità (troppe)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_rsa_scatter(
    rdm_weird: np.ndarray,
    rdm_sinic: np.ndarray,
    labels: list[str],
    spearman_r: float,
) -> str:
    """
    Crea scatter plot interattivo della correlazione RDM.

    Ogni punto è una coppia di termini. Hover mostra i termini.

    Returns
    -------
    str
        HTML con div Plotly embedded.
    """
    _check_plotly()

    n = rdm_weird.shape[0]
    triu = np.triu_indices(n, k=1)

    vec_w = rdm_weird[triu]
    vec_s = rdm_sinic[triu]

    # Crea etichette per hover
    pair_labels = [
        f"{labels[i]} ↔ {labels[j]}"
        for i, j in zip(triu[0], triu[1])
    ]

    fig = go.Figure()

    # Scatter con density (usa scattergl per performance)
    fig.add_trace(
        go.Scattergl(
            x=vec_w,
            y=vec_s,
            mode="markers",
            marker=dict(
                size=3,
                color=PLOTLY_COLORS["weird"],
                opacity=0.3,
            ),
            text=pair_labels,
            hovertemplate="<b>%{text}</b><br>WEIRD: %{x:.4f}<br>Sinic: %{y:.4f}<extra></extra>",
            name="Term pairs",
        )
    )

    # Linea identità
    max_val = max(vec_w.max(), vec_s.max()) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="grey", dash="dash"),
            name="Identity",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"RDM Correlation (ρ = {spearman_r:.4f}, N pairs = {len(vec_w):,})",
        xaxis_title="Cosine distance (WEIRD)",
        yaxis_title="Cosine distance (Sinic)",
        height=600,
        **PLOTLY_LAYOUT,
    )

    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val], scaleanchor="x", scaleratio=1)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_gw_transport(
    transport_plan: np.ndarray,
    labels: list[str],
    gw_distance: float,
    p_value: float,
    threshold_percentile: float = 99,
) -> str:
    """
    Crea heatmap interattiva del transport plan con slider soglia.

    Returns
    -------
    str
        HTML con div Plotly embedded.
    """
    _check_plotly()

    n = transport_plan.shape[0]

    # Hover text
    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(
                f"<b>WEIRD:</b> {labels[i]}<br>"
                f"<b>Sinic:</b> {labels[j]}<br>"
                f"<b>Mass:</b> {transport_plan[i,j]:.2e}"
            )
        hover_text.append(row)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=transport_plan,
            x=labels,
            y=labels,
            colorscale="YlOrRd",
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
            colorbar=dict(title="Transport mass"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"GW Transport Plan<br><sup>Distance = {gw_distance:.6f}, {significance_label(p_value)}</sup>",
            x=0.5,
        ),
        xaxis_title="Sinic terms",
        yaxis_title="WEIRD terms",
        height=700,
        **PLOTLY_LAYOUT,
    )

    # Nascondi tick labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_axes_scatter(
    axes_results: list[dict],
) -> str:
    """
    Crea scatter interattivo per ogni asse valoriale.

    Dropdown permette di selezionare l'asse da visualizzare.

    Parameters
    ----------
    axes_results : list[dict]
        Lista con keys: axis_name, weird_scores, sinic_scores, spearman_r, spearman_p.

    Returns
    -------
    str
        HTML con div Plotly embedded.
    """
    _check_plotly()

    if not axes_results:
        return "<p>No axes data available</p>"

    fig = go.Figure()

    # Aggiungi trace per ogni asse (solo il primo visibile)
    for i, ax_result in enumerate(axes_results):
        weird_scores = ax_result.get("weird_scores", {})
        sinic_scores = ax_result.get("sinic_scores", {})
        axis_name = ax_result.get("axis_name", f"Axis {i}")
        rho = ax_result.get("spearman_r", 0)

        labels = list(weird_scores.keys())
        x = [weird_scores[l] for l in labels]
        y = [sinic_scores[l] for l in labels]
        delta = [weird_scores[l] - sinic_scores[l] for l in labels]

        # Colore per delta
        colors = [COLORS["sinic"] if abs(d) > 0.1 else COLORS["weird"] for d in delta]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=8, color=colors, opacity=0.7),
                text=labels,
                hovertemplate="<b>%{text}</b><br>WEIRD: %{x:.4f}<br>Sinic: %{y:.4f}<br>Δ: %{customdata:+.4f}<extra></extra>",
                customdata=delta,
                name=axis_name,
                visible=(i == 0),
            )
        )

    # Linea identità (sempre visibile)
    all_vals = []
    for ax_result in axes_results:
        all_vals.extend(ax_result.get("weird_scores", {}).values())
        all_vals.extend(ax_result.get("sinic_scores", {}).values())

    if all_vals:
        min_val = min(all_vals) - 0.05
        max_val = max(all_vals) + 0.05
    else:
        min_val, max_val = -1, 1

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="grey", dash="dash"),
            name="Identity",
            hoverinfo="skip",
        )
    )

    # Dropdown per selezionare asse
    buttons = []
    n_axes = len(axes_results)
    for i, ax_result in enumerate(axes_results):
        visibility = [False] * n_axes + [True]  # Tutti nascosti tranne identità
        visibility[i] = True  # Mostra solo questo asse

        buttons.append(
            dict(
                label=ax_result.get("axis_name", f"Axis {i}"),
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Axis: {ax_result.get('axis_name', '')} (ρ = {ax_result.get('spearman_r', 0):.4f})"},
                ],
            )
        )

    fig.update_layout(
        title=f"Axis: {axes_results[0].get('axis_name', '')} (ρ = {axes_results[0].get('spearman_r', 0):.4f})",
        xaxis_title="Projection score (WEIRD)",
        yaxis_title="Projection score (Sinic)",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        height=600,
        **PLOTLY_LAYOUT,
    )

    fig.update_xaxes(range=[min_val, max_val])
    fig.update_yaxes(range=[min_val, max_val], scaleanchor="x", scaleratio=1)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_clustering_dendrogram(
    fm_results: list[dict],
) -> str:
    """
    Crea bar chart interattivo dell'indice FM.

    Hover mostra dettagli statistici.

    Returns
    -------
    str
        HTML con div Plotly embedded.
    """
    _check_plotly()

    ks = [r["k"] for r in fm_results]
    fms = [r["fm_index"] for r in fm_results]
    ps = [r["p_value"] for r in fm_results]

    colors = [COLORS["significant"] if p < 0.05 else COLORS["not_significant"] for p in ps]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[f"k={k}" for k in ks],
            y=fms,
            marker_color=colors,
            text=[f"{fm:.3f}" for fm in fms],
            textposition="outside",
            hovertemplate="<b>k=%{x}</b><br>FM = %{y:.4f}<br>%{customdata}<extra></extra>",
            customdata=[significance_label(p) for p in ps],
        )
    )

    # Linea soglia 0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="FM = 0.5")

    fig.update_layout(
        title="Fowlkes-Mallows Index by Number of Clusters",
        xaxis_title="Number of clusters",
        yaxis_title="FM Index",
        yaxis_range=[0, 1.1],
        height=400,
        **PLOTLY_LAYOUT,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_nda_scatter(
    per_term_results: list[dict],
    mean_jaccard: float,
    k: int,
) -> str:
    """
    Crea scatter Jaccard con hover che mostra i vicini.

    Ordina per Jaccard (basso a sinistra).

    Returns
    -------
    str
        HTML con div Plotly embedded.
    """
    _check_plotly()

    # Ordina per Jaccard
    sorted_results = sorted(per_term_results, key=lambda r: r["jaccard"])

    terms = [r.get("term", r.get("label", "")) for r in sorted_results]
    jaccards = [r["jaccard"] for r in sorted_results]
    colors = [jaccard_color(j) for j in jaccards]

    # Testo hover con vicini
    hover_texts = []
    for r in sorted_results:
        weird_n = ", ".join(r.get("weird_neighbors", [])[:3])
        sinic_n = ", ".join(r.get("sinic_neighbors", [])[:3])
        shared_n = ", ".join(r.get("shared_neighbors", [])[:3])
        hover_texts.append(
            f"<b>{r.get('term', r.get('label', ''))}</b><br>"
            f"Jaccard: {r['jaccard']:.3f}<br>"
            f"WEIRD neighbors: {weird_n}<br>"
            f"Sinic neighbors: {sinic_n}<br>"
            f"Shared: {shared_n}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(terms))),
            y=jaccards,
            mode="markers",
            marker=dict(size=8, color=colors, opacity=0.8),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Linea media
    fig.add_hline(
        y=mean_jaccard,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean = {mean_jaccard:.3f}",
    )

    fig.update_layout(
        title=f"Jaccard Index per Term (k={k}, N={len(terms)})<br><sup>Sorted by Jaccard (lowest left)</sup>",
        xaxis_title="Term rank",
        yaxis_title="Jaccard index",
        yaxis_range=[0, 1.05],
        height=500,
        **PLOTLY_LAYOUT,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_nda_network(
    per_term_results: list[dict],
    top_n: int = 15,
) -> str:
    """
    Crea network interattivo dei false friends.

    Usa vis.js per network (più leggero di Plotly per grafi).
    Fallback a Plotly scatter se vis.js non disponibile.

    Returns
    -------
    str
        HTML con network embedded.
    """
    # Per semplicità, usa una tabella interattiva invece del network
    # (vis.js richiede libreria esterna)

    sorted_results = sorted(per_term_results, key=lambda r: r["jaccard"])[:top_n]

    rows = []
    for r in sorted_results:
        weird_n = ", ".join(r.get("weird_neighbors", [])[:5])
        sinic_n = ", ".join(r.get("sinic_neighbors", [])[:5])
        shared_n = ", ".join(r.get("shared_neighbors", [])[:5])
        j = r["jaccard"]
        j_color = jaccard_color(j)

        rows.append(f"""
            <tr>
                <td><b>{r.get('term', r.get('label', ''))}</b></td>
                <td><span style="display:inline-block;width:{j*150}px;height:12px;background:{j_color};border-radius:3px;"></span> {j:.3f}</td>
                <td style="font-size:0.85em;">{weird_n}</td>
                <td style="font-size:0.85em;">{sinic_n}</td>
                <td style="font-size:0.85em;">{shared_n}</td>
            </tr>
        """)

    return f"""
        <h4>Top {top_n} "False Friends" (lowest Jaccard)</h4>
        <table class="sortable">
            <thead>
                <tr>
                    <th>Term</th>
                    <th>Jaccard</th>
                    <th>WEIRD neighbors</th>
                    <th>Sinic neighbors</th>
                    <th>Shared</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    """


def create_umap_scatter(
    coords_weird: list[dict],
    coords_sinic: list[dict],
    domains: Optional[list[str]] = None,
) -> str:
    """
    Crea scatter UMAP interattivo con toggle etichette.

    Parameters
    ----------
    coords_weird, coords_sinic : list[dict]
        Liste di {label, x, y} per ogni spazio.
    domains : list[str], optional
        Domini per colorazione.

    Returns
    -------
    str
        HTML con div Plotly embedded.
    """
    _check_plotly()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["WEIRD", "Sinic"],
        horizontal_spacing=0.1,
    )

    # WEIRD
    labels_w = [p["label"] for p in coords_weird]
    x_w = [p["x"] for p in coords_weird]
    y_w = [p["y"] for p in coords_weird]

    if domains:
        colors_w = [DOMAIN_COLORS.get(domains[i], "#CCCCCC") for i in range(len(coords_weird))]
    else:
        colors_w = COLORS["weird"]

    fig.add_trace(
        go.Scatter(
            x=x_w,
            y=y_w,
            mode="markers",
            marker=dict(size=6, color=colors_w, opacity=0.7),
            text=labels_w,
            hovertemplate="<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            name="WEIRD",
        ),
        row=1, col=1,
    )

    # Sinic
    labels_s = [p["label"] for p in coords_sinic]
    x_s = [p["x"] for p in coords_sinic]
    y_s = [p["y"] for p in coords_sinic]

    if domains:
        colors_s = [DOMAIN_COLORS.get(domains[i], "#CCCCCC") for i in range(len(coords_sinic))]
    else:
        colors_s = COLORS["sinic"]

    fig.add_trace(
        go.Scatter(
            x=x_s,
            y=y_s,
            mode="markers",
            marker=dict(size=6, color=colors_s, opacity=0.7),
            text=labels_s,
            hovertemplate="<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            name="Sinic",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"UMAP Projection (N={len(coords_weird)} terms)",
        height=600,
        showlegend=False,
        **PLOTLY_LAYOUT,
    )

    fig.update_xaxes(title_text="UMAP 1")
    fig.update_yaxes(title_text="UMAP 2")

    return fig.to_html(full_html=False, include_plotlyjs=False)
