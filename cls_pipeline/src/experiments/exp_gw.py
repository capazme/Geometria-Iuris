"""
exp_gw.py — Experiment 2: Gromov-Wasserstein Distance + Permutation Test.

Measures structural distortion between WEIRD and Sinic embedding spaces
via Optimal Transport, with significance assessed by permutation test.

References
----------
Alvarez-Melis & Jaakkola (2018), EMNLP.
"""
# ─── Gromov-Wasserstein: confronto strutturale senza allineamento ────
# A differenza dell'RSA (che correla distanze), GW cerca il trasporto
# ottimale tra due spazi metrici *senza richiedere che vivano nello
# stesso spazio*. Misura quanto la struttura delle distanze interne
# deve essere "distorta" per allineare i due spazi. Una distanza GW
# bassa indica isomorfismo strutturale; alta indica anisomorfismo.
# Rif.: Alvarez-Melis & Jaakkola (2018) "Gromov-Wasserstein Alignment
#        of Word Embedding Spaces", EMNLP.
# Rif.: Peyré & Cuturi (2019) "Computational Optimal Transport",
#        Foundations and Trends in Machine Learning, 11(5-6).
# ─────────────────────────────────────────────────────────────────────

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import ot
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity

from .statistical import PermutationResult

logger = logging.getLogger(__name__)


@dataclass
class GWResult:
    """Result of Gromov-Wasserstein computation."""
    distance: float
    transport_plan: np.ndarray
    p_value: float
    n_permutations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "transport_plan": self.transport_plan.tolist(),
            "transport_plan_shape": list(self.transport_plan.shape),
            "transport_plan_summary": {
                "min": float(self.transport_plan.min()),
                "max": float(self.transport_plan.max()),
                "mean": float(self.transport_plan.mean()),
            },
            "significant": bool(self.p_value < 0.05),
        }


def _cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute intra-space cosine distance matrix (1 - similarity)."""
    return 1.0 - cosine_similarity(vectors)


def _compute_gw(
    C1: np.ndarray,
    C2: np.ndarray,
    entropic_reg: float = 5e-3,
    use_sinkhorn: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Compute GW distance from pre-computed cost matrices.

    Returns (distance, transport_plan).
    """
    # Distribuzioni uniformi: ogni termine ha lo stesso peso.
    # Non c'è ragione a priori per pesare diversamente i concetti giuridici.
    n = C1.shape[0]
    m = C2.shape[0]
    p = np.ones(n) / n
    q = np.ones(m) / m

    C1 = np.ascontiguousarray(C1, dtype=np.float64)
    C2 = np.ascontiguousarray(C2, dtype=np.float64)

    # Regolarizzazione entropica (Cuturi, 2013 "Sinkhorn Distances"):
    # rende il problema differenziabile e risolvibile in tempo O(n² log n)
    # anziché O(n³ log n). epsilon > 0 "sfoca" il piano di trasporto;
    # valori piccoli (5e-3) mantengono alta la qualità dell'approssimazione.
    # Loss quadratica ("square_loss"): penalizza distorsioni proporzionalmente
    # al quadrato della differenza, amplificando le discrepanze strutturali.
    if use_sinkhorn and entropic_reg > 0:
        transport_plan, gw_log = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q,
            loss_fun="square_loss",
            epsilon=entropic_reg,
            log=True,
        )
        gw_dist = gw_log["gw_dist"]
    else:
        transport_plan, gw_log = ot.gromov.gromov_wasserstein(
            C1, C2, p, q,
            loss_fun="square_loss",
            log=True,
        )
        gw_dist = gw_log["gw_dist"]

    return float(gw_dist), transport_plan


def gromov_wasserstein_distance(
    vectors_weird: np.ndarray,
    vectors_sinic: np.ndarray,
    entropic_reg: float = 5e-3,
    use_sinkhorn: bool = True,
    n_permutations: int = 5000,
    seed: int = 42,
) -> GWResult:
    """
    Compute Gromov-Wasserstein distance with permutation test.

    Parameters
    ----------
    vectors_weird : np.ndarray
        WEIRD embeddings (n x d1).
    vectors_sinic : np.ndarray
        Sinic embeddings (m x d2).
    entropic_reg : float
        Entropic regularization for Sinkhorn.
    use_sinkhorn : bool
        If True, use entropic-regularized GW.
    n_permutations : int
        Number of permutations for p-value.
    seed : int
        Random seed.

    Returns
    -------
    GWResult
        GW distance, transport plan, and p-value.
    """
    # Matrici di costo intra-spazio (distanza del coseno interna a ciascun modello)
    C1 = _cosine_distance_matrix(vectors_weird)
    C2 = _cosine_distance_matrix(vectors_sinic)

    # Distanza GW osservata
    gw_dist, transport_plan = _compute_gw(C1, C2, entropic_reg, use_sinkhorn)
    logger.info("GW distance (observed): %.6f", gw_dist)

    # Test di permutazione per GW: si permutano righe/colonne della matrice
    # di costo sinica (equivalente a rimescolare le associazioni concetto-embedding).
    # La distribuzione nulla rappresenta distanze GW sotto l'ipotesi che
    # non ci sia corrispondenza strutturale tra i due spazi.
    #
    # Parallelizzazione con joblib: ogni permutazione è indipendente e GW
    # è computazionalmente pesante, quindi il parallelismo dà un grande speedup.

    def _single_permutation(perm_seed: int) -> float:
        """Esegue una singola permutazione GW."""
        rng_local = np.random.RandomState(perm_seed)
        perm = rng_local.permutation(vectors_sinic.shape[0])
        C2_perm = C2[np.ix_(perm, perm)]
        dist, _ = _compute_gw(C1, C2_perm, entropic_reg, use_sinkhorn)
        return dist

    # Genera seed deterministici per ogni permutazione
    rng = np.random.RandomState(seed)
    perm_seeds = rng.randint(0, 2**31, size=n_permutations)

    n_jobs = os.cpu_count() or 4
    logger.info("GW: running %d permutations on %d cores...", n_permutations, n_jobs)

    null_dist = np.array(
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_single_permutation)(s) for s in perm_seeds
        )
    )

    # p-value: proporzione di distanze permutate <= osservata.
    # Se la distanza osservata è significativamente *bassa* rispetto al nullo,
    # i due spazi sono più simili di quanto atteso per caso.
    p_value = (np.sum(null_dist <= gw_dist) + 1) / (n_permutations + 1)

    logger.info(
        "GW permutation test: distance=%.6f, p=%.4f (%d permutations)",
        gw_dist, p_value, n_permutations,
    )

    return GWResult(
        distance=gw_dist,
        transport_plan=transport_plan,
        p_value=p_value,
        n_permutations=n_permutations,
    )
