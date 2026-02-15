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
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from .statistical import mantel_test, block_bootstrap_rdm_ci, BootstrapCIResult

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
    r_squared: float = 0.0
    bootstrap_ci: BootstrapCIResult | None = None
    null_distribution: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "spearman_r": self.spearman_r,
            "r_squared": self.r_squared,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "n_pairs": self.n_pairs,
            "n_terms": len(self.labels),
            "labels": self.labels,
            "rdm_weird": self.rdm_weird.tolist(),
            "rdm_sinic": self.rdm_sinic.tolist(),
            "significant": self.p_value < 0.05,
        }
        if self.bootstrap_ci is not None:
            d["bootstrap_ci"] = {
                "estimate": self.bootstrap_ci.estimate,
                "ci_lower": self.bootstrap_ci.ci_lower,
                "ci_upper": self.bootstrap_ci.ci_upper,
                "n_bootstrap": self.bootstrap_ci.n_bootstrap,
                "alpha": self.bootstrap_ci.alpha,
            }
        if self.null_distribution is not None:
            d["null_distribution"] = self.null_distribution.tolist()
        return d


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
    # Usa scipy.spatial.distance.pdist (C compilato) per efficienza:
    # con N=500 termini → 124.750 coppie, ~100× più veloce del doppio ciclo.
    return squareform(pdist(vectors, metric="cosine"))


def run_rsa(
    emb_weird: np.ndarray,
    emb_sinic: np.ndarray,
    labels: list[str],
    n_permutations: int = 10000,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> RSAResult:
    """
    Run RSA analysis with Mantel test, effect size, and bootstrap CI.

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
    n_bootstrap : int
        Number of bootstrap resamples for CI on Spearman r.
    seed : int
        Random seed.

    Returns
    -------
    RSAResult
        RDMs, Spearman r, r², p-value, bootstrap CI, null distribution.
    """
    n = emb_weird.shape[0]
    assert emb_sinic.shape[0] == n, "Both embeddings must have same number of terms"
    assert len(labels) == n, "Labels must match embedding count"

    logger.info("Computing RDMs for %d terms...", n)

    # Pipeline in 4 passi:
    # 1. Calcola le RDM (matrici di dissimilarità) per ciascun spazio
    # 2. Correla le RDM con Spearman (confronto di rango tra geometrie)
    # 3. Valuta significatività con test di Mantel (permutazione righe/colonne)
    # 4. Stima CI via block bootstrap (ricampiona termini, non coppie)
    rdm_w = compute_rdm(emb_weird)
    rdm_s = compute_rdm(emb_sinic)

    n_pairs = n * (n - 1) // 2
    logger.info("RDMs computed: %d x %d, %d unique pairs", n, n, n_pairs)

    mantel_result = mantel_test(rdm_w, rdm_s, n_permutations=n_permutations, seed=seed)

    r = mantel_result.observed
    r_sq = r ** 2

    logger.info(
        "RSA: Spearman r=%.4f, r²=%.4f, p=%.4f",
        r, r_sq, mantel_result.p_value,
    )

    # Block bootstrap CI: ricampiona *termini interi* (non coppie di distanze)
    # per rispettare la struttura di dipendenza delle RDM. Ogni termine
    # contribuisce a (n-1) distanze → le coppie non sono indipendenti.
    # Il pair bootstrap sottostima la varianza e produce CI anti-conservativi.
    # Rif.: Nili et al. (2014) PLoS Comp. Bio., 10(4), e1003553.
    ci_result = block_bootstrap_rdm_ci(
        emb_weird,
        emb_sinic,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    logger.info(
        "RSA bootstrap CI: [%.4f, %.4f] (%d resamples)",
        ci_result.ci_lower, ci_result.ci_upper, n_bootstrap,
    )

    return RSAResult(
        rdm_weird=rdm_w,
        rdm_sinic=rdm_s,
        spearman_r=r,
        p_value=mantel_result.p_value,
        n_permutations=n_permutations,
        n_pairs=n_pairs,
        labels=labels,
        r_squared=r_sq,
        bootstrap_ci=ci_result,
        null_distribution=mantel_result.null_distribution,
    )
