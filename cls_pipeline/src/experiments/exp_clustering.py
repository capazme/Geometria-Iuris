"""
exp_clustering.py — Experiment 4: Hierarchical Clustering + Fowlkes-Mallows.

Generates independent dendrograms for WEIRD and Sinic spaces, compares
taxonomy agreement using FM index across multiple k values with
permutation tests for significance.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import fowlkes_mallows_score

from .statistical import PermutationResult

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result of hierarchical clustering for one space."""
    linkage_matrix: np.ndarray
    labels: list[str]


@dataclass
class FMResult:
    """Fowlkes-Mallows result for a single k value."""
    k: int
    fm_index: float
    p_value: float
    n_permutations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "fm_index": self.fm_index,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "interpretation": (
                "divergent_taxonomies" if self.fm_index < 0.5
                else "similar_taxonomies"
            ),
        }


@dataclass
class ClusteringExperimentResult:
    """Complete result of Experiment 4."""
    clustering_weird: ClusteringResult
    clustering_sinic: ClusteringResult
    fm_results: list[FMResult]
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_terms": len(self.labels),
            "fm_results": [r.to_dict() for r in self.fm_results],
        }


def hierarchical_clustering(
    vectors: np.ndarray,
    labels: list[str],
    method: str = "ward",
) -> ClusteringResult:
    """
    Perform hierarchical agglomerative clustering.

    Parameters
    ----------
    vectors : np.ndarray
        Embedding matrix (N x D).
    labels : list[str]
        Term labels.
    method : str
        Linkage method.

    Returns
    -------
    ClusteringResult
        Linkage matrix and labels.
    """
    Z = linkage(vectors, method=method)
    logger.info("Hierarchical clustering (%s): %d terms", method, len(labels))
    return ClusteringResult(linkage_matrix=Z, labels=labels)


def _compute_fm(Z_weird, Z_sinic, k):
    """Compute FM index for a given k."""
    c_w = fcluster(Z_weird, k, criterion="maxclust")
    c_s = fcluster(Z_sinic, k, criterion="maxclust")
    return fowlkes_mallows_score(c_w, c_s)


def run_clustering_experiment(
    emb_weird: np.ndarray,
    emb_sinic: np.ndarray,
    labels: list[str],
    method: str = "ward",
    k_values: list[int] | None = None,
    n_permutations: int = 5000,
    seed: int = 42,
) -> ClusteringExperimentResult:
    """
    Run clustering experiment with multi-k FM index and permutation tests.

    Parameters
    ----------
    emb_weird : np.ndarray
        WEIRD embeddings (N x D).
    emb_sinic : np.ndarray
        Sinic embeddings (N x D).
    labels : list[str]
        Term labels.
    method : str
        Linkage method.
    k_values : list[int] | None
        List of cluster counts to test. Default: [3, 5, 7, 10].
    n_permutations : int
        Number of permutations per k.
    seed : int
        Random seed.

    Returns
    -------
    ClusteringExperimentResult
        Clustering results and FM indices with p-values.
    """
    if k_values is None:
        k_values = [3, 5, 7, 10]

    n = len(labels)
    # Filter k values that are valid
    k_values = [k for k in k_values if 2 <= k < n]

    clust_w = hierarchical_clustering(emb_weird, labels, method)
    clust_s = hierarchical_clustering(emb_sinic, labels, method)

    fm_results = []
    rng = np.random.RandomState(seed)

    for k in k_values:
        fm_observed = _compute_fm(clust_w.linkage_matrix, clust_s.linkage_matrix, k)

        # Permutation test: permute cluster assignments of one space
        null_dist = np.empty(n_permutations)
        c_w = fcluster(clust_w.linkage_matrix, k, criterion="maxclust")

        for i in range(n_permutations):
            # Permute the sinic clustering labels
            c_s_perm = rng.permutation(
                fcluster(clust_s.linkage_matrix, k, criterion="maxclust")
            )
            null_dist[i] = fowlkes_mallows_score(c_w, c_s_perm)

        p_value = (np.sum(null_dist >= fm_observed) + 1) / (n_permutations + 1)

        logger.info(
            "Clustering k=%d: FM=%.4f, p=%.4f",
            k, fm_observed, p_value,
        )

        fm_results.append(FMResult(
            k=k,
            fm_index=float(fm_observed),
            p_value=float(p_value),
            n_permutations=n_permutations,
        ))

    return ClusteringExperimentResult(
        clustering_weird=clust_w,
        clustering_sinic=clust_s,
        fm_results=fm_results,
        labels=labels,
    )


def plot_dendrograms(
    result: ClusteringExperimentResult,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (18, 8),
    dpi: int = 300,
    weird_label: str = "WEIRD",
    sinic_label: str = "Sinic",
) -> Path:
    """
    Generate side-by-side dendrogram comparison.

    Parameters
    ----------
    result : ClusteringExperimentResult
        Clustering experiment result.
    output_dir : Path | None
        Output directory.
    figsize : tuple
        Figure size.
    dpi : int
        Plot resolution.
    weird_label : str
        WEIRD model label.
    sinic_label : str
        Sinic model label.

    Returns
    -------
    Path
        Path to saved PNG.
    """
    out = output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    dendrogram(
        result.clustering_weird.linkage_matrix,
        labels=result.clustering_weird.labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax1,
        color_threshold=0,
    )
    ax1.set_title(weird_label)
    ax1.set_ylabel("Distance")

    dendrogram(
        result.clustering_sinic.linkage_matrix,
        labels=result.clustering_sinic.labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax2,
        color_threshold=0,
    )
    ax2.set_title(sinic_label)
    ax2.set_ylabel("Distance")

    fm_str = " | ".join(
        f"k={r.k}: FM={r.fm_index:.3f} (p={r.p_value:.3f})"
        for r in result.fm_results
    )
    fig.suptitle(
        f"Hierarchical Clustering Comparison\n{fm_str}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = out / "clustering_dendrograms.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    logger.info("Dendrograms saved: %s", path)
    return path
