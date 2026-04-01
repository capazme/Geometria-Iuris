"""
Shared statistical utilities for the Geometria Iuris pipeline.

All functions operate on numpy arrays representing either embedding matrices
(N, dim) or Relational Dissimilarity Matrices (N, N).

Design decisions and mathematical justification:
  lens_1_relational/trace.md  — D2 (Spearman), D3 (Mantel), D4 (block bootstrap), D5 (Mann-Whitney)
  shared/math_trace.md        — R1-R3, A1-A2, S1-S6
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MannWhitneyResult:
    statistic: float       # U statistic
    p_value: float
    effect_r: float        # rank-biserial correlation ∈ [-1, 1]
    n_x: int
    n_y: int
    median_x: float
    median_y: float


@dataclass
class MantelResult:
    rho: float
    p_value: float
    r_squared: float
    null_distribution: np.ndarray  # shape (n_perm,)


@dataclass
class BootstrapCI:
    low: float
    high: float
    distribution: np.ndarray  # shape (n_boot,)


@dataclass
class RSAResult:
    rho: float
    p_value: float
    r_squared: float
    ci: BootstrapCI
    null_distribution: np.ndarray


def holm_correction(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    More powerful than Bonferroni while still controlling FWER.
    Returns adjusted p-values (capped at 1.0).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = min(cummax, 1.0)
    return adjusted


@dataclass
class PermutationGroupResult:
    """Result of a permutation test comparing two groups of scalar values."""
    observed_diff: float   # mean(group_a) - mean(group_b)
    p_value: float
    effect_r: float        # rank-biserial correlation (same as MW)
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float


def permutation_test_groups(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_perm: int = 10_000,
    alternative: str = "less",
    seed: int = 42,
) -> PermutationGroupResult:
    """
    Permutation test on the difference of means between two groups.

    Pools all values, randomly re-assigns labels, and recomputes the
    difference n_perm times. More appropriate than Mann-Whitney for
    very small samples (e.g., n=9 vs n=6).

    Parameters
    ----------
    group_a, group_b : 1-D arrays of scalar values
    alternative : 'less' (H1: mean_a < mean_b), 'greater', 'two-sided'
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n_a, n_b = len(a), len(b)
    pooled = np.concatenate([a, b])
    obs_diff = float(a.mean() - b.mean())

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rng.shuffle(pooled)
        null[i] = pooled[:n_a].mean() - pooled[n_a:].mean()

    # Phipson & Smyth (2010): p = (b + 1) / (m + 1)
    if alternative == "less":
        b_count = int((null <= obs_diff).sum())
    elif alternative == "greater":
        b_count = int((null >= obs_diff).sum())
    else:
        b_count = int((np.abs(null) >= abs(obs_diff)).sum())
    p_value = (b_count + 1) / (n_perm + 1)

    # Rank-biserial effect size (same formula as MW)
    from scipy.stats import mannwhitneyu as _mwu
    try:
        u = _mwu(a, b, alternative=alternative).statistic
        effect_r = 1.0 - 2.0 * u / (n_a * n_b)
    except ValueError:
        effect_r = 0.0

    return PermutationGroupResult(
        observed_diff=obs_diff,
        p_value=p_value,
        effect_r=float(effect_r),
        n_a=n_a, n_b=n_b,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
    )


# ---------------------------------------------------------------------------
# RDM construction
# ---------------------------------------------------------------------------

def compute_rdm(vecs: np.ndarray) -> np.ndarray:
    """
    Compute a Relational Dissimilarity Matrix from L2-normalized vectors.

    RDM[i, j] = 1 - cosine_similarity(vecs[i], vecs[j])
               = 1 - vecs[i] · vecs[j]   (valid because ||vecs|| = 1)

    Parameters
    ----------
    vecs : (N, dim) float32, L2-normalized

    Returns
    -------
    rdm : (N, N) float32, symmetric, diagonal = 0, values ∈ [0, 2]
    """
    sim = vecs @ vecs.T
    np.clip(sim, -1.0, 1.0, out=sim)   # guard against float32 drift beyond [-1, 1]
    rdm = (1.0 - sim).astype(np.float32)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def upper_tri(rdm: np.ndarray) -> np.ndarray:
    """
    Extract the upper triangle (diagonal excluded, k=1) of a square matrix.

    Returns N*(N-1)//2 unique pairwise distances — one value per term pair.
    """
    rows, cols = np.triu_indices(len(rdm), k=1)
    return rdm[rows, cols]


# ---------------------------------------------------------------------------
# §3.1.1 — Domain signal tests
# ---------------------------------------------------------------------------

def mannwhitney_with_r(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "less",
) -> MannWhitneyResult:
    """
    Mann-Whitney U test with rank-biserial correlation effect size.

    Tests whether values in x tend to be smaller than values in y.

    effect_r = 1 - 2U / (n_x * n_y)
             = (concordant_pairs - discordant_pairs) / total_pairs
    Ranges [-1, 1]: +1 = x always < y, 0 = no difference.

    Parameters
    ----------
    x, y        : 1-D distance arrays (need not be equal length)
    alternative : 'less' | 'greater' | 'two-sided'

    Returns
    -------
    MannWhitneyResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = mannwhitneyu(x, y, alternative=alternative)
    u = res.statistic
    effect_r = 1.0 - 2.0 * u / (len(x) * len(y))
    # Floor p-value to avoid reporting exact 0.0 (scipy float underflow)
    p_val = float(max(res.pvalue, np.finfo(float).tiny))
    return MannWhitneyResult(
        statistic=float(u),
        p_value=p_val,
        effect_r=float(effect_r),
        n_x=len(x),
        n_y=len(y),
        median_x=float(np.median(x)),
        median_y=float(np.median(y)),
    )


# ---------------------------------------------------------------------------
# §3.1.4 — RSA: Mantel test + block bootstrap CI
# ---------------------------------------------------------------------------

def mantel_test(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> MantelResult:
    """
    Mantel test: permutation-based significance test for RDM correlation.

    Permutes rows and columns of rdm_b jointly (preserving the distance-matrix
    structure) to generate a null distribution of Spearman rho values.
    p_value = #{rho_perm >= rho_obs} / n_perm

    Parameters
    ----------
    rdm_a, rdm_b : (N, N) symmetric distance matrices
    n_perm       : permutations (default 1000)
    seed         : random seed for reproducibility

    Returns
    -------
    MantelResult
    """
    tri_a = upper_tri(rdm_a)
    tri_b = upper_tri(rdm_b)
    rho_obs = float(spearmanr(tri_a, tri_b).statistic)

    rng = np.random.default_rng(seed)
    n = len(rdm_b)
    null = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        pi = rng.permutation(n)
        null[i] = spearmanr(tri_a, upper_tri(rdm_b[np.ix_(pi, pi)])).statistic

    # Phipson & Smyth (2010): p = (b + 1) / (m + 1) where b = number of
    # null values >= observed, m = number of permutations. This ensures
    # p is never exactly zero and is slightly conservative.
    b = int((null >= rho_obs).sum())
    p_bounded = (b + 1) / (n_perm + 1)

    return MantelResult(
        rho=rho_obs,
        p_value=p_bounded,
        r_squared=float(rho_obs ** 2),
        null_distribution=null,
    )


def block_bootstrap_rsa(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> BootstrapCI:
    """
    Block bootstrap confidence interval for RSA (Spearman rho).

    Resamples term indices with replacement — not pair indices — to respect
    the dependency structure: each term appears in N-1 pairs, so pairs are
    not independent observations (Nili et al. 2014).

    The same index set is applied to both RDMs so the same term pairs are
    compared in both models for each bootstrap iteration.

    Reference: Nili et al. (2014) PLoS Computational Biology 10(4): e1003553.

    Parameters
    ----------
    rdm_a, rdm_b : (N, N) symmetric distance matrices
    n_boot       : bootstrap iterations (default 1000)
    seed         : random seed
    ci_level     : confidence level (default 0.95 → [2.5, 97.5] percentiles)

    Returns
    -------
    BootstrapCI
    """
    n = len(rdm_a)
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0
    boots = np.empty(n_boot, dtype=np.float32)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub_a = rdm_a[np.ix_(idx, idx)]
        sub_b = rdm_b[np.ix_(idx, idx)]
        boots[i] = spearmanr(upper_tri(sub_a), upper_tri(sub_b)).statistic

    lo, hi = np.percentile(boots, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return BootstrapCI(low=float(lo), high=float(hi), distribution=boots)


# ---------------------------------------------------------------------------
# Generic bootstrap CI (reusable across experiments)
# ---------------------------------------------------------------------------

@dataclass
class GenericBootstrapCI:
    estimate: float
    ci_low: float
    ci_high: float
    distribution: np.ndarray  # (n_boot,)


def bootstrap_ci_generic(
    data: np.ndarray,
    stat_fn: callable,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> GenericBootstrapCI:
    """
    Row-resample bootstrap CI for an arbitrary statistic.

    Parameters
    ----------
    data : (N, ...) — rows are observations, resampled with replacement
    stat_fn : callable — data -> scalar
    n_boot : int
    ci_level : float
    seed : int

    Returns
    -------
    GenericBootstrapCI
    """
    data = np.asarray(data)
    estimate = float(stat_fn(data))
    rng = np.random.default_rng(seed)
    n = len(data)
    alpha = (1.0 - ci_level) / 2.0
    boots = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[i] = stat_fn(data[idx])

    lo, hi = np.percentile(boots, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return GenericBootstrapCI(
        estimate=estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        distribution=boots,
    )


def rsa(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_perm: int = 1000,
    n_boot: int = 1000,
    seed: int = 42,
) -> RSAResult:
    """
    Full RSA pipeline: Spearman rho + Mantel test + block bootstrap CI.

    Parameters
    ----------
    rdm_a, rdm_b : (N, N) symmetric distance matrices
    n_perm       : permutations for Mantel test
    n_boot       : bootstrap iterations for CI
    seed         : shared seed (used independently by Mantel and bootstrap)

    Returns
    -------
    RSAResult
    """
    mantel = mantel_test(rdm_a, rdm_b, n_perm=n_perm, seed=seed)
    ci = block_bootstrap_rsa(rdm_a, rdm_b, n_boot=n_boot, seed=seed)
    return RSAResult(
        rho=mantel.rho,
        p_value=mantel.p_value,
        r_squared=mantel.r_squared,
        ci=ci,
        null_distribution=mantel.null_distribution,
    )
