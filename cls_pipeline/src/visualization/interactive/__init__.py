"""
interactive/ — Modulo per generazione HTML interattivo con Plotly.

Genera visualizzazioni interattive con hover, zoom, filtri.
Il report HTML è self-contained (nessuna dipendenza CDN).
"""

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
from .html_builder import build_html_report

__all__ = [
    # Plotly charts
    "create_rsa_heatmap",
    "create_rsa_scatter",
    "create_gw_transport",
    "create_axes_scatter",
    "create_clustering_dendrogram",
    "create_nda_scatter",
    "create_nda_network",
    "create_umap_scatter",
    # HTML builder
    "build_html_report",
]
