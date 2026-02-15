"""
png/ — Modulo per generazione PNG publication-ready.

Ogni sotto-modulo gestisce le visualizzazioni di un esperimento:
- rsa_plots.py: Clustered heatmap, inter-domain matrix, RDM scatter
- gw_plots.py: Transport histogram, top alignments
- axes_plots.py: Scatter con CI, forest plot
- clustering_plots.py: Truncated dendrogram, FM chart
- nda_plots.py: Network graph, Jaccard histogram
- umap_plots.py: UMAP con smart label placement (adjustText)
"""

from .common import save_figure, setup_style
from .rsa_plots import (
    plot_clustered_heatmap,
    plot_inter_domain_matrix,
    plot_rdm_correlation,
    plot_null_distribution,
)
from .gw_plots import (
    plot_transport_histogram,
    plot_top_alignments,
    plot_transport_thresholded,
)
from .axes_plots import (
    plot_axes_scatter_ci,
    plot_forest_plot,
)
from .clustering_plots import (
    plot_truncated_dendrogram,
    plot_fm_chart,
)
from .nda_plots import (
    plot_jaccard_histogram,
    plot_false_friends_network,
    plot_decomposition_comparison,
)
from .umap_plots import (
    plot_umap_smart_labels,
)
from .multi_model_plots import (
    plot_multi_model_heatmap,
    plot_multi_model_forest,
    plot_multi_model_null_distributions,
)

__all__ = [
    # Common
    "save_figure",
    "setup_style",
    # RSA
    "plot_clustered_heatmap",
    "plot_inter_domain_matrix",
    "plot_rdm_correlation",
    "plot_null_distribution",
    # GW
    "plot_transport_histogram",
    "plot_top_alignments",
    "plot_transport_thresholded",
    # Axes
    "plot_axes_scatter_ci",
    "plot_forest_plot",
    # Clustering
    "plot_truncated_dendrogram",
    "plot_fm_chart",
    # NDA
    "plot_jaccard_histogram",
    "plot_false_friends_network",
    "plot_decomposition_comparison",
    # UMAP
    "plot_umap_smart_labels",
    # Multi-model
    "plot_multi_model_heatmap",
    "plot_multi_model_forest",
    "plot_multi_model_null_distributions",
]
