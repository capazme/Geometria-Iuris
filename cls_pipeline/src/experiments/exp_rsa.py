"""
exp_rsa.py — Experiment 1: RSA + Mantel Test.

Computes Representational Dissimilarity Matrices (RDM) for WEIRD and Sinic
embedding spaces, then measures structural similarity via Spearman correlation
with significance assessed by Mantel permutation test.

References
----------
Kriegeskorte et al. (2008), Frontiers in Systems Neuroscience.
"""
# ─── RSA come confronto "di secondo ordine" ──────────────────────────
# L'RSA non confronta direttamente i vettori embedding (che vivono in
# spazi di dimensione e orientamento diversi), ma le *geometrie interne*
# dei due spazi. Si costruisce una matrice di dissimilarità (RDM) per
# ciascun modello, poi si misura quanto le due RDM sono correlate.
# È un confronto strutturale: se due concetti sono "vicini" in uno
# spazio, lo sono anche nell'altro?
# Rif.: Kriegeskorte, Mur & Bandettini (2008) "Representational
#        Similarity Analysis", Frontiers in Systems Neuroscience, 2, 4.
# ─────────────────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

from .statistical import mantel_test, PermutationResult

logger = logging.getLogger(__name__)


@dataclass
class RSAResult:
    """Result of RSA analysis."""
    rdm_weird: np.ndarray
    rdm_sinic: np.ndarray
    spearman_r: float
    p_value: float
    n_permutations: int
    n_pairs: int
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spearman_r": self.spearman_r,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "n_pairs": self.n_pairs,
            "n_terms": len(self.labels),
            "labels": self.labels,
            "rdm_weird": self.rdm_weird.tolist(),
            "rdm_sinic": self.rdm_sinic.tolist(),
            "interpretation": (
                "significant_dissimilarity" if self.p_value < 0.05
                else "no_significant_difference"
            ),
        }


def compute_rdm(vectors: np.ndarray) -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix (cosine distance).

    Parameters
    ----------
    vectors : np.ndarray
        Embedding matrix (N x D).

    Returns
    -------
    np.ndarray
        Symmetric N x N cosine distance matrix.
    """
    # Distanza del coseno: d(a,b) = 1 - cos(a,b). È invariante alla norma
    # dei vettori, quindi misura solo la direzione nello spazio semantico.
    # Questo è essenziale per gli embedding, dove la norma può variare
    # per motivi non semantici (frequenza del token, batch di addestramento).
    n = vectors.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_dist(vectors[i], vectors[j])
            rdm[i, j] = d
            rdm[j, i] = d
    return rdm


def run_rsa(
    emb_weird: np.ndarray,
    emb_sinic: np.ndarray,
    labels: list[str],
    n_permutations: int = 10000,
    seed: int = 42,
) -> RSAResult:
    """
    Run RSA analysis with Mantel test.

    Parameters
    ----------
    emb_weird : np.ndarray
        WEIRD embeddings (N x D).
    emb_sinic : np.ndarray
        Sinic embeddings (N x D).
    labels : list[str]
        Term labels (length N).
    n_permutations : int
        Number of permutations for Mantel test.
    seed : int
        Random seed.

    Returns
    -------
    RSAResult
        RDMs, Spearman r, p-value.
    """
    n = emb_weird.shape[0]
    assert emb_sinic.shape[0] == n, "Both embeddings must have same number of terms"
    assert len(labels) == n, "Labels must match embedding count"

    logger.info("Computing RDMs for %d terms...", n)

    # Pipeline in 3 passi:
    # 1. Calcola le RDM (matrici di dissimilarità) per ciascun spazio
    # 2. Correla le RDM con Spearman (confronto di rango tra geometrie)
    # 3. Valuta significatività con test di Mantel (permutazione righe/colonne)
    rdm_w = compute_rdm(emb_weird)
    rdm_s = compute_rdm(emb_sinic)

    n_pairs = n * (n - 1) // 2
    logger.info("RDMs computed: %d x %d, %d unique pairs", n, n, n_pairs)

    mantel_result = mantel_test(rdm_w, rdm_s, n_permutations=n_permutations, seed=seed)

    logger.info(
        "RSA: Spearman r=%.4f, p=%.4f",
        mantel_result.observed, mantel_result.p_value,
    )

    return RSAResult(
        rdm_weird=rdm_w,
        rdm_sinic=rdm_s,
        spearman_r=mantel_result.observed,
        p_value=mantel_result.p_value,
        n_permutations=n_permutations,
        n_pairs=n_pairs,
        labels=labels,
    )


def plot_rdm_heatmaps(
    result: RSAResult,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (18, 8),
    dpi: int = 300,
    weird_label: str = "WEIRD",
    sinic_label: str = "Sinic",
) -> Path:
    """
    Generate side-by-side RDM heatmaps.

    Parameters
    ----------
    result : RSAResult
        RSA analysis result.
    output_dir : Path | None
        Output directory.
    figsize : tuple
        Figure size.
    dpi : int
        Plot resolution.
    weird_label : str
        Label for WEIRD model.
    sinic_label : str
        Label for Sinic model.

    Returns
    -------
    Path
        Path to saved PNG.
    """
    out = output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Etichette troncate a 15 caratteri per leggibilità nelle heatmap
    short_labels = [l[:15] for l in result.labels]

    # Colormap "viridis": scelta per accessibilità (leggibile anche in
    # scala di grigi e da soggetti con daltonismo).
    # Scaling indipendente per RDM: le due matrici possono avere range
    # molto diversi (es. WEIRD ~0.07-0.26, Sinic ~0.0-0.75).
    # Il confronto quantitativo è affidato al Spearman r, non al colore.
    n = result.rdm_weird.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    triu_w = result.rdm_weird[triu_idx]
    triu_s = result.rdm_sinic[triu_idx]

    sns.heatmap(
        result.rdm_weird, ax=ax1,
        xticklabels=short_labels, yticklabels=short_labels,
        cmap="viridis", vmin=triu_w.min(), vmax=triu_w.max(),
        square=True, cbar_kws={"shrink": 0.8},
    )
    ax1.set_title(f"{weird_label} RDM (range {triu_w.min():.2f}–{triu_w.max():.2f})")
    ax1.tick_params(axis="both", labelsize=6)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax1.get_yticklabels(), rotation=0)

    sns.heatmap(
        result.rdm_sinic, ax=ax2,
        xticklabels=short_labels, yticklabels=short_labels,
        cmap="viridis", vmin=triu_s.min(), vmax=triu_s.max(),
        square=True, cbar_kws={"shrink": 0.8},
    )
    ax2.set_title(f"{sinic_label} RDM (range {triu_s.min():.2f}–{triu_s.max():.2f})")
    ax2.tick_params(axis="both", labelsize=6)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)

    fig.suptitle(
        f"Representational Dissimilarity Matrices\n"
        f"Spearman r = {result.spearman_r:.4f}, p = {result.p_value:.4f} "
        f"(Mantel test, {result.n_permutations:,} permutations)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    path = out / "rsa_rdm_heatmaps.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("RSA heatmaps saved: %s", path)
    return path
