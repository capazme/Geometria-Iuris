"""
statistical.py — Shared statistical utilities for CLS experiments.

Provides permutation tests, bootstrap confidence intervals, and the
Mantel test used across multiple experiment modules.
"""

import logging
from dataclasses import dataclass

import numpy as np
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
    rng = np.random.RandomState(seed)
    null_dist = np.empty(n_permutations)

    for i in range(n_permutations):
        perm_data = rng.permutation(data) if data.ndim == 1 else data[rng.permutation(len(data))]
        null_dist[i] = stat_fn(perm_data)

    if alternative == "greater":
        p_value = (np.sum(null_dist >= observed_stat) + 1) / (n_permutations + 1)
    elif alternative == "less":
        p_value = (np.sum(null_dist <= observed_stat) + 1) / (n_permutations + 1)
    else:  # two-sided
        p_value = (np.sum(np.abs(null_dist) >= np.abs(observed_stat)) + 1) / (n_permutations + 1)

    logger.info(
        "Permutation test: observed=%.4f, p=%.4f (%d permutations)",
        observed_stat, p_value, n_permutations,
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

    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    vec_a = rdm_a[triu_idx]
    vec_b = rdm_b[triu_idx]

    # Observed Spearman correlation
    r_observed, _ = spearmanr(vec_a, vec_b)

    # Permutation test: permute rows/columns of rdm_b
    rng = np.random.RandomState(seed)
    null_dist = np.empty(n_permutations)

    for i in range(n_permutations):
        perm = rng.permutation(n)
        rdm_b_perm = rdm_b[np.ix_(perm, perm)]
        vec_b_perm = rdm_b_perm[triu_idx]
        r_perm, _ = spearmanr(vec_a, vec_b_perm)
        null_dist[i] = r_perm

    # p-value: proportion of permuted r >= observed r
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
