"""
exp_clustering.py — Experiment 4: Hierarchical Clustering + Fowlkes-Mallows.

Generates independent dendrograms for WEIRD and Sinic spaces, compares
taxonomy agreement using FM index across multiple k values with
permutation tests for significance.
"""
# ─── Clustering gerarchico come test di tassonomia ───────────────────
# Domanda: i due modelli linguistici raggruppano i concetti giuridici
# nella stessa tassonomia? Si costruiscono dendrogrammi indipendenti
# per WEIRD e Sinic, si tagliano a vari k (numero di cluster), e si
# confrontano le partizioni risultanti con l'indice di Fowlkes-Mallows.
#
# FM = sqrt(PPV × TPR): media geometrica di precision e recall sulle
# coppie di elementi assegnati allo stesso cluster. FM=1 = partizioni
# identiche; FM basso = tassonomie divergenti.
#
# Si usa multi-k (3, 5, 7, 10) per verificare la robustezza del
# risultato: un singolo k potrebbe essere un artefatto della scelta
# arbitraria della granularità.
# Rif.: Fowlkes & Mallows (1983) "A Method for Comparing Two
#        Hierarchical Clusterings", JASA, 78(383), 553-569.
# ─────────────────────────────────────────────────────────────────────

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
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
            "significant": self.p_value < 0.05,
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
            "labels": self.labels,
            "linkage_weird": self.clustering_weird.linkage_matrix.tolist(),
            "linkage_sinic": self.clustering_sinic.linkage_matrix.tolist(),
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
    # Metodo Ward: minimizza la varianza intra-cluster ad ogni passo di
    # fusione. È il metodo più adatto quando si vogliono cluster compatti
    # e di dimensione simile — proprietà desiderabile per tassonomie
    # giuridiche dove i domini hanno dimensioni comparabili.
    Z = linkage(vectors, method=method)
    logger.info("Hierarchical clustering (%s): %d terms", method, len(labels))
    return ClusteringResult(linkage_matrix=Z, labels=labels)


def _compute_fm(Z_weird, Z_sinic, k):
    """Compute FM index for a given k."""
    # FM = sqrt(PPV × TPR) dove PPV e TPR sono calcolati sulle coppie:
    # - PPV: delle coppie co-assegnate in WEIRD, quante lo sono anche in Sinic?
    # - TPR: delle coppie co-assegnate in Sinic, quante lo sono anche in WEIRD?
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

    n_jobs = os.cpu_count() or 4

    for k in k_values:
        fm_observed = _compute_fm(clust_w.linkage_matrix, clust_s.linkage_matrix, k)

        # Test di permutazione: si permutano le etichette cluster dello
        # spazio sinico. Sotto l'ipotesi nulla, le assegnazioni cluster
        # dei due spazi sono indipendenti (nessun accordo tassonomico).
        #
        # Parallelizzazione con joblib per ogni k.
        c_w = fcluster(clust_w.linkage_matrix, k, criterion="maxclust")
        c_s = fcluster(clust_s.linkage_matrix, k, criterion="maxclust")

        def _single_permutation(perm_seed: int) -> float:
            """Esegue una singola permutazione FM."""
            rng_local = np.random.RandomState(perm_seed)
            c_s_perm = rng_local.permutation(c_s)
            return fowlkes_mallows_score(c_w, c_s_perm)

        perm_seeds = rng.randint(0, 2**31, size=n_permutations)

        null_dist = np.array(
            Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_single_permutation)(s) for s in perm_seeds
            )
        )

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
        leaf_rotation=45,
        leaf_font_size=7,
        ax=ax1,
        color_threshold=0,
    )
    ax1.set_title(weird_label)
    ax1.set_ylabel("Distance")

    dendrogram(
        result.clustering_sinic.linkage_matrix,
        labels=result.clustering_sinic.labels,
        leaf_rotation=45,
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
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])

    path = out / "clustering_dendrograms.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Dendrograms saved: %s", path)
    return path
