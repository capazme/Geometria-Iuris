"""
statistical.py — Shared statistical utilities for CLS experiments.

Provides permutation tests, bootstrap confidence intervals, and the
Mantel test used across multiple experiment modules.
"""
# ─── Perché test di permutazione e non test parametrici? ───────────────
# Gli embedding linguistici producono distribuzioni fortemente non normali
# e con dipendenze strutturali tra le osservazioni (ogni coppia di vettori
# è correlata a tutte le altre). I test parametrici (t-test, F-test)
# presuppongono normalità e indipendenza: qui entrambe le assunzioni
# cadono. I test di permutazione generano una distribuzione nulla
# *empirica* permutando i dati stessi, senza ipotesi distribuzionali.
# Rif.: Good (2005) "Permutation, Parametric, and Bootstrap Tests of
#        Hypotheses", 3rd ed., Springer.
# Rif.: Efron (1979) "Bootstrap Methods: Another Look at the Jackknife",
#        Annals of Statistics, 7(1), 1-26.
# ───────────────────────────────────────────────────────────────────────

import logging
import os
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class PermutationResult:
    """Result of a permutation test."""
    observed: float
    p_value: float
    n_permutations: int
    null_distribution: np.ndarray


@dataclass
class BootstrapCIResult:
    """Result of bootstrap confidence interval estimation."""
    estimate: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    alpha: float


def permutation_test(
    observed_stat: float,
    data: np.ndarray,
    stat_fn: callable,
    n_permutations: int = 5000,
    seed: int = 42,
    alternative: str = "greater",
) -> PermutationResult:
    """
    Generic permutation test.

    Parameters
    ----------
    observed_stat : float
        The observed test statistic.
    data : np.ndarray
        Data to permute (1D array or matrix rows).
    stat_fn : callable
        Function that takes permuted data and returns a scalar statistic.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.
    alternative : str
        "greater" (default): p = P(stat_perm >= observed).
        "less": p = P(stat_perm <= observed).
        "two-sided": p = P(|stat_perm| >= |observed|).

    Returns
    -------
    PermutationResult
        Observed statistic, p-value, and null distribution.
    """
    logger.info(
        "Test di permutazione: statistica osservata=%.4f, alternative='%s', n_perm=%d",
        observed_stat, alternative, n_permutations,
    )
    rng = np.random.RandomState(seed)
    null_dist = np.empty(n_permutations)

    log_interval = max(n_permutations // 5, 1)
    for i in range(n_permutations):
        perm_data = rng.permutation(data) if data.ndim == 1 else data[rng.permutation(len(data))]
        null_dist[i] = stat_fn(perm_data)
        if (i + 1) % log_interval == 0:
            logger.info("  Permutazione %d/%d (%.0f%%)", i + 1, n_permutations, 100 * (i + 1) / n_permutations)

    # Le tre alternative definiscono la direzione del test:
    # - "greater": la statistica osservata è insolitamente alta?
    # - "less": è insolitamente bassa?
    # - "two-sided": è estrema in entrambe le direzioni?
    #
    # Correzione "+1" al numeratore e denominatore (Phipson & Smyth, 2010,
    # "Permutation P-values Should Never Be Zero", Stat. Appl. Genet. Mol. Biol.):
    # evita p-value esattamente zero e corregge il bias dovuto al conteggio
    # discreto, garantendo che p >= 1/(n_perm+1).
    if alternative == "greater":
        p_value = (np.sum(null_dist >= observed_stat) + 1) / (n_permutations + 1)
    elif alternative == "less":
        p_value = (np.sum(null_dist <= observed_stat) + 1) / (n_permutations + 1)
    else:  # two-sided
        p_value = (np.sum(np.abs(null_dist) >= np.abs(observed_stat)) + 1) / (n_permutations + 1)

    logger.info(
        "Test di permutazione completato: osservato=%.4f, p=%.4f, "
        "distribuzione nulla: media=%.4f, std=%.4f, [min=%.4f, max=%.4f]",
        observed_stat, p_value,
        null_dist.mean(), null_dist.std(), null_dist.min(), null_dist.max(),
    )

    return PermutationResult(
        observed=observed_stat,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution=null_dist,
    )


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCIResult:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    stat_fn : callable
        Function that takes data and returns a scalar statistic.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    seed : int
        Random seed.

    Returns
    -------
    BootstrapCIResult
        Point estimate and confidence interval.
    """
    # Bootstrap di Efron (1979): metodo nonparametrico per stimare
    # l'intervallo di confidenza di una statistica senza assunzioni
    # sulla distribuzione sottostante. Si ricampiona *con rimpiazzo*
    # dal campione originale, calcolando la statistica su ogni
    # ricampionamento. I percentili della distribuzione bootstrap
    # forniscono i limiti dell'intervallo (metodo dei percentili).
    rng = np.random.RandomState(seed)
    estimate = stat_fn(data)

    boot_stats = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_stats[i] = stat_fn(data[idx])

    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    logger.info(
        "Bootstrap CI: estimate=%.4f, [%.4f, %.4f] (%d resamples)",
        estimate, ci_lower, ci_upper, n_bootstrap,
    )

    return BootstrapCIResult(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def mantel_test(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> PermutationResult:
    """
    Mantel test: permutation-based significance test for matrix correlation.

    Computes Spearman correlation between upper-triangular elements of
    two distance matrices, then tests significance by permuting rows/columns.

    Parameters
    ----------
    rdm_a : np.ndarray
        First distance matrix (N x N), symmetric.
    rdm_b : np.ndarray
        Second distance matrix (N x N), symmetric.
    n_permutations : int
        Number of permutations for p-value.
    seed : int
        Random seed.

    Returns
    -------
    PermutationResult
        Spearman r (observed), p-value, and null distribution.

    References
    ----------
    Mantel, N. (1967). The detection of disease clustering and a
    generalized regression approach. Cancer Research, 27(2), 209-220.
    """
    n = rdm_a.shape[0]
    assert rdm_a.shape == rdm_b.shape == (n, n), "RDMs must be square and same size"

    # Si estraggono solo gli elementi del triangolo superiore: la matrice
    # è simmetrica e la diagonale è sempre zero, quindi non porta informazione.
    triu_idx = np.triu_indices(n, k=1)
    vec_a = rdm_a[triu_idx]
    vec_b = rdm_b[triu_idx]

    # Correlazione di Spearman (rango): scelta al posto di Pearson perché
    # non assume linearità — ci interessa che l'*ordine* delle distanze
    # sia preservato tra i due spazi, non la proporzionalità esatta.
    # Rif.: Mantel (1967) "The detection of disease clustering and a
    #        generalized regression approach", Cancer Research, 27(2).
    r_observed, _ = spearmanr(vec_a, vec_b)

    # Test di permutazione: si permutano *simultaneamente* righe e colonne
    # della seconda matrice (rdm_b[perm, perm]) per preservare la simmetria
    # della RDM. Una permutazione solo delle righe spezzerebbe la struttura
    # simmetrica e genererebbe matrici invalide come distribuzione nulla.
    #
    # Parallelizzazione con joblib: ogni permutazione è indipendente.

    def _single_permutation(perm_seed: int) -> float:
        """Esegue una singola permutazione Mantel."""
        rng_local = np.random.RandomState(perm_seed)
        perm = rng_local.permutation(n)
        rdm_b_perm = rdm_b[np.ix_(perm, perm)]
        vec_b_perm = rdm_b_perm[triu_idx]
        r_perm, _ = spearmanr(vec_a, vec_b_perm)
        return r_perm

    rng = np.random.RandomState(seed)
    perm_seeds = rng.randint(0, 2**31, size=n_permutations)

    n_jobs = os.cpu_count() or 4
    logger.info("Mantel test: running %d permutations on %d cores...", n_permutations, n_jobs)

    null_dist = np.array(
        Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_single_permutation)(s) for s in perm_seeds
        )
    )

    # p-value: proporzione di r permutati >= r osservato (+ correzione Phipson)
    p_value = (np.sum(null_dist >= r_observed) + 1) / (n_permutations + 1)

    logger.info(
        "Mantel test: r=%.4f, p=%.4f (%d permutations, %d pairs)",
        r_observed, p_value, n_permutations, len(vec_a),
    )

    return PermutationResult(
        observed=r_observed,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution=null_dist,
    )
