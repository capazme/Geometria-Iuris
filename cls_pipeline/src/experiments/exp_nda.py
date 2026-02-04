"""
exp_nda.py — Experiment 5: Neighborhood Divergence Analysis (NDA).

Part A: Compares k-NN neighborhoods for each concept across WEIRD and Sinic
    spaces using Jaccard similarity.
Part B: Normative decompositions — vector arithmetic operations that test
    specific jurisprudential hypotheses.

References
----------
arXiv:2411.08687 (2024) for NNGS; Mikolov et al. (2013) for vector arithmetic.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

from .statistical import PermutationResult

logger = logging.getLogger(__name__)


# =========================================================================
# Part A: Neighborhood comparison
# =========================================================================

@dataclass
class TermNeighborhoodResult:
    """Neighborhood comparison for a single term."""
    term_id: str
    label: str
    jaccard: float
    weird_neighbors: list[str]
    sinic_neighbors: list[str]
    shared_neighbors: list[str]


@dataclass
class NDAPartAResult:
    """Result of NDA Part A (neighborhood comparison)."""
    term_results: list[TermNeighborhoodResult]
    mean_jaccard: float
    p_value: float
    n_permutations: int
    k: int

    def to_dict(self) -> dict[str, Any]:
        sorted_results = sorted(self.term_results, key=lambda r: r.jaccard)
        return {
            "mean_jaccard": self.mean_jaccard,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "k": self.k,
            "n_core_terms": len(self.term_results),
            "false_friends": [
                {
                    "term": r.label,
                    "jaccard": r.jaccard,
                    "weird_neighbors": r.weird_neighbors,
                    "sinic_neighbors": r.sinic_neighbors,
                }
                for r in sorted_results[:10]
            ],
            "per_term": [
                {"term": r.label, "jaccard": r.jaccard}
                for r in sorted_results
            ],
        }


def _find_knn_labels(
    query_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_labels: list[str],
    k: int,
) -> list[str]:
    """Find k-NN labels for a query vector in a corpus."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(corpus_labels)), metric="cosine")
    nn.fit(corpus_vectors)
    distances, indices = nn.kneighbors(query_vector.reshape(1, -1))

    # Exclude the query itself if it's in the corpus
    neighbors = []
    for idx in indices[0]:
        if len(neighbors) >= k:
            break
        neighbors.append(corpus_labels[idx])
    return neighbors[:k]


def run_nda_part_a(
    emb_weird_core: np.ndarray,
    emb_sinic_core: np.ndarray,
    core_labels: list[str],
    emb_weird_all: np.ndarray,
    emb_sinic_all: np.ndarray,
    all_labels: list[str],
    k: int = 10,
    n_permutations: int = 5000,
    seed: int = 42,
) -> NDAPartAResult:
    """
    Run NDA Part A: k-NN neighborhood comparison.

    Parameters
    ----------
    emb_weird_core : np.ndarray
        WEIRD embeddings for core terms (N_core x D).
    emb_sinic_core : np.ndarray
        Sinic embeddings for core terms (N_core x D).
    core_labels : list[str]
        Core term labels.
    emb_weird_all : np.ndarray
        WEIRD embeddings for full corpus (M x D).
    emb_sinic_all : np.ndarray
        Sinic embeddings for full corpus (M x D).
    all_labels : list[str]
        Full corpus labels.
    k : int
        Number of neighbors.
    n_permutations : int
        Permutations for p-value.
    seed : int
        Random seed.

    Returns
    -------
    NDAPartAResult
        Per-term Jaccard scores, mean, and p-value.
    """
    n_core = len(core_labels)

    # Build k-NN models for both spaces
    nn_weird = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_weird.fit(emb_weird_all)

    nn_sinic = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_sinic.fit(emb_sinic_all)

    term_results = []

    for i in range(n_core):
        # Find k-NN in each space
        _, idx_w = nn_weird.kneighbors(emb_weird_core[i].reshape(1, -1))
        _, idx_s = nn_sinic.kneighbors(emb_sinic_core[i].reshape(1, -1))

        # Get neighbor labels, excluding self
        neigh_w = [all_labels[j] for j in idx_w[0] if all_labels[j] != core_labels[i]][:k]
        neigh_s = [all_labels[j] for j in idx_s[0] if all_labels[j] != core_labels[i]][:k]

        # Jaccard on concept labels
        set_w = set(neigh_w)
        set_s = set(neigh_s)
        intersection = set_w & set_s
        union = set_w | set_s
        jaccard = len(intersection) / len(union) if union else 0.0

        term_results.append(TermNeighborhoodResult(
            term_id=core_labels[i],
            label=core_labels[i],
            jaccard=jaccard,
            weird_neighbors=neigh_w,
            sinic_neighbors=neigh_s,
            shared_neighbors=sorted(intersection),
        ))

    mean_jaccard = np.mean([r.jaccard for r in term_results])

    # Permutation test: permute concept-embedding assignments in one space
    rng = np.random.RandomState(seed)
    null_dist = np.empty(n_permutations)

    for p in range(n_permutations):
        perm = rng.permutation(len(all_labels))
        emb_sinic_perm = emb_sinic_all[perm]

        nn_perm = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn_perm.fit(emb_sinic_perm)

        jaccards_perm = []
        for i in range(n_core):
            _, idx_w = nn_weird.kneighbors(emb_weird_core[i].reshape(1, -1))
            _, idx_s = nn_perm.kneighbors(emb_sinic_core[i].reshape(1, -1))

            neigh_w = [all_labels[j] for j in idx_w[0] if all_labels[j] != core_labels[i]][:k]
            neigh_s = [all_labels[j] for j in idx_s[0] if all_labels[j] != core_labels[i]][:k]

            set_w = set(neigh_w)
            set_s = set(neigh_s)
            union = set_w | set_s
            jaccards_perm.append(len(set_w & set_s) / len(union) if union else 0.0)

        null_dist[p] = np.mean(jaccards_perm)

        if (p + 1) % 500 == 0:
            logger.info("NDA Part A permutation %d/%d", p + 1, n_permutations)

    p_value = (np.sum(null_dist >= mean_jaccard) + 1) / (n_permutations + 1)

    logger.info(
        "NDA Part A: mean_jaccard=%.4f, p=%.4f (%d terms, k=%d)",
        mean_jaccard, p_value, n_core, k,
    )

    return NDAPartAResult(
        term_results=term_results,
        mean_jaccard=float(mean_jaccard),
        p_value=float(p_value),
        n_permutations=n_permutations,
        k=k,
    )


# =========================================================================
# Part B: Normative decompositions
# =========================================================================

@dataclass
class DecompositionResult:
    """Result for a single normative decomposition."""
    decomposition_id: str
    operation: str
    en_formula: str
    zh_formula: str
    jurisprudential_question: str
    weird_neighbors: list[tuple[str, float]]
    sinic_neighbors: list[tuple[str, float]]
    jaccard: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.decomposition_id,
            "operation": self.operation,
            "en_formula": self.en_formula,
            "zh_formula": self.zh_formula,
            "jurisprudential_question": self.jurisprudential_question,
            "weird_neighbors": [
                {"label": l, "cosine_distance": float(d)}
                for l, d in self.weird_neighbors
            ],
            "sinic_neighbors": [
                {"label": l, "cosine_distance": float(d)}
                for l, d in self.sinic_neighbors
            ],
            "jaccard": self.jaccard,
        }


@dataclass
class NDAPartBResult:
    """Result of NDA Part B (normative decompositions)."""
    decompositions: list[DecompositionResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_decompositions": len(self.decompositions),
            "decompositions": [d.to_dict() for d in self.decompositions],
            "mean_jaccard": float(np.mean([d.jaccard for d in self.decompositions])),
        }


def _vector_subtract_knn(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_labels: list[str],
    k: int = 10,
) -> list[tuple[str, float]]:
    """Compute residual = a - b, normalize, find k-NN."""
    residual = vec_a - vec_b
    norm = np.linalg.norm(residual)
    if norm > 0:
        residual = residual / norm

    nn = NearestNeighbors(n_neighbors=min(k, len(corpus_labels)), metric="cosine")
    nn.fit(corpus_vectors)
    distances, indices = nn.kneighbors(residual.reshape(1, -1))

    return [(corpus_labels[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]


def run_nda_part_b(
    normative_decompositions: list[dict],
    embed_fn_weird: callable,
    embed_fn_sinic: callable,
    corpus_weird: np.ndarray,
    corpus_sinic: np.ndarray,
    corpus_labels: list[str],
    k: int = 10,
) -> NDAPartBResult:
    """
    Run NDA Part B: normative decompositions.

    Parameters
    ----------
    normative_decompositions : list[dict]
        List of decomposition definitions.
    embed_fn_weird : callable
        Function to embed English text list -> np.ndarray.
    embed_fn_sinic : callable
        Function to embed Chinese text list -> np.ndarray.
    corpus_weird : np.ndarray
        Full WEIRD corpus embeddings.
    corpus_sinic : np.ndarray
        Full Sinic corpus embeddings.
    corpus_labels : list[str]
        Corpus labels.
    k : int
        Number of neighbors.

    Returns
    -------
    NDAPartBResult
        Results for each decomposition.
    """
    results = []

    for decomp in normative_decompositions:
        logger.info("Decomposition: %s - %s", decomp["id"], decomp["jurisprudential_question"])

        # Embed operands
        emb_a_w = embed_fn_weird([decomp["en_a"]])[0]
        emb_b_w = embed_fn_weird([decomp["en_b"]])[0]
        emb_a_s = embed_fn_sinic([decomp["zh_a"]])[0]
        emb_b_s = embed_fn_sinic([decomp["zh_b"]])[0]

        # k-NN of residual in each space
        weird_nn = _vector_subtract_knn(emb_a_w, emb_b_w, corpus_weird, corpus_labels, k)
        sinic_nn = _vector_subtract_knn(emb_a_s, emb_b_s, corpus_sinic, corpus_labels, k)

        # Jaccard between neighbor sets
        set_w = {label for label, _ in weird_nn}
        set_s = {label for label, _ in sinic_nn}
        union = set_w | set_s
        jaccard = len(set_w & set_s) / len(union) if union else 0.0

        en_formula = f"{decomp['en_a']} - {decomp['en_b']}"
        zh_formula = f"{decomp['zh_a']} - {decomp['zh_b']}"

        logger.info(
            "  %s | %s: Jaccard=%.3f",
            en_formula, zh_formula, jaccard,
        )

        results.append(DecompositionResult(
            decomposition_id=decomp["id"],
            operation=decomp["operation"],
            en_formula=en_formula,
            zh_formula=zh_formula,
            jurisprudential_question=decomp["jurisprudential_question"],
            weird_neighbors=weird_nn,
            sinic_neighbors=sinic_nn,
            jaccard=jaccard,
        ))

    return NDAPartBResult(decompositions=results)


# =========================================================================
# Combined result
# =========================================================================

@dataclass
class NDAExperimentResult:
    """Combined NDA result (Parts A + B)."""
    part_a: NDAPartAResult
    part_b: NDAPartBResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_a_neighborhoods": self.part_a.to_dict(),
            "part_b_decompositions": self.part_b.to_dict(),
        }


def plot_nda_results(
    result: NDAExperimentResult,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 300,
) -> Path:
    """
    Generate NDA visualization: Jaccard distribution + decomposition comparison.

    Parameters
    ----------
    result : NDAExperimentResult
        NDA experiment result.
    output_dir : Path | None
        Output directory.
    figsize : tuple
        Figure size.
    dpi : int
        Plot resolution.

    Returns
    -------
    Path
        Path to saved PNG.
    """
    out = output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Part A: Jaccard distribution
    jaccards = [r.jaccard for r in result.part_a.term_results]
    labels_sorted = [r.label for r in sorted(result.part_a.term_results, key=lambda r: r.jaccard)]
    jaccards_sorted = sorted(jaccards)

    ax1.barh(range(len(jaccards_sorted)), jaccards_sorted, color="#4C72B0", alpha=0.8)
    ax1.set_yticks(range(len(labels_sorted)))
    ax1.set_yticklabels([l[:20] for l in labels_sorted], fontsize=6)
    ax1.set_xlabel("Jaccard Index")
    ax1.set_title(
        f"Part A: Neighborhood Overlap\n"
        f"Mean Jaccard={result.part_a.mean_jaccard:.3f}, "
        f"p={result.part_a.p_value:.3f}",
        fontsize=10,
    )
    ax1.axvline(x=result.part_a.mean_jaccard, color="red", linestyle="--", linewidth=1)

    # Part B: Decomposition Jaccards
    decomp_labels = [d.en_formula for d in result.part_b.decompositions]
    decomp_jaccards = [d.jaccard for d in result.part_b.decompositions]

    ax2.barh(range(len(decomp_jaccards)), decomp_jaccards, color="#DD8452", alpha=0.8)
    ax2.set_yticks(range(len(decomp_labels)))
    ax2.set_yticklabels(decomp_labels, fontsize=9)
    ax2.set_xlabel("Jaccard Index")
    ax2.set_title("Part B: Normative Decompositions", fontsize=10)

    fig.suptitle("Neighborhood Divergence Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = out / "nda_analysis.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    logger.info("NDA plot saved: %s", path)
    return path
