"""Experiment modules for CLS Pipeline v2.0."""

from .exp_rsa import (
    RSAResult,
    run_rsa,
    compute_rdm,
    plot_rdm_heatmaps,
)
from .exp_gw import (
    GWResult,
    gromov_wasserstein_distance,
)
from .exp_axes import (
    AxesExperimentResult,
    AxesComparisonResult,
    build_kozlowski_axis,
    project_on_axis,
    run_axes_experiment,
    plot_axes_comparison,
)
from .exp_clustering import (
    ClusteringResult,
    ClusteringExperimentResult,
    FMResult,
    hierarchical_clustering,
    run_clustering_experiment,
    plot_dendrograms,
)
from .exp_nda import (
    NDAExperimentResult,
    NDAPartAResult,
    NDAPartBResult,
    run_nda_part_a,
    run_nda_part_b,
    plot_nda_results,
)
from .statistical import (
    PermutationResult,
    BootstrapCIResult,
    permutation_test,
    bootstrap_ci,
    mantel_test,
)

# Keep old module names importable for backward compatibility during transition
from .module_c_umap import (
    UMAPResult,
    umap_reduce,
    plot_umap,
)

__all__ = [
    # Experiment 1: RSA
    "RSAResult",
    "run_rsa",
    "compute_rdm",
    "plot_rdm_heatmaps",
    # Experiment 2: GW
    "GWResult",
    "gromov_wasserstein_distance",
    # Experiment 3: Axes
    "AxesExperimentResult",
    "AxesComparisonResult",
    "build_kozlowski_axis",
    "project_on_axis",
    "run_axes_experiment",
    "plot_axes_comparison",
    # Experiment 4: Clustering
    "ClusteringResult",
    "ClusteringExperimentResult",
    "FMResult",
    "hierarchical_clustering",
    "run_clustering_experiment",
    "plot_dendrograms",
    # Experiment 5: NDA
    "NDAExperimentResult",
    "NDAPartAResult",
    "NDAPartBResult",
    "run_nda_part_a",
    "run_nda_part_b",
    "plot_nda_results",
    # Statistical utilities
    "PermutationResult",
    "BootstrapCIResult",
    "permutation_test",
    "bootstrap_ci",
    "mantel_test",
    # UMAP (visualization utility, not an experiment)
    "UMAPResult",
    "umap_reduce",
    "plot_umap",
]
