"""
ch3-measurability — self-contained library.

All shared statistical, embedding and audit utilities used by the scripts
in this folder live here, so the directory is reproducible from inputs +
scripts alone (no dependency on the parent THESIS repository).

The functions are organised into four sections, each documented inline:

  1. Hashing + I/O helpers
  2. Mathematical statistics: RDM, RSA, Mantel, bootstrap, Mann-Whitney
  3. Embedding client (sentence-transformers wrapper, SHA-keyed disk cache)
  4. Convenience aggregates (RSAResult, GenericBootstrapCI dataclasses)

For a working example of every function, see ``notebooks/analysis.ipynb``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import yaml
from scipy.stats import mannwhitneyu, spearmanr

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Hashing + I/O helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def renormalize(arr: np.ndarray) -> np.ndarray:
    """Explicitly L2-renormalize rows of a (N, dim) array. Some encoders
    (notably Qwen3) emit vectors with slight scale drift even when the
    ``normalize_embeddings`` flag is set."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.clip(norms, 1e-12, None)).astype(np.float32)


def load_config(path: Path) -> dict:
    with Path(path).open() as fh:
        return yaml.safe_load(fh)






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

# ---------------------------------------------------------------------------
# 3. Embedding client (sentence-transformers wrapper, SHA-keyed disk cache)
# ---------------------------------------------------------------------------



from sentence_transformers import SentenceTransformer



# ---------------------------------------------------------------------------
# Precomputed embedding loader (populated by shared/precompute.py)
# ---------------------------------------------------------------------------

def load_precomputed(
    model_label: str,
    embeddings_dir: str | Path,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load precomputed embeddings and the shared term index.

    Parameters
    ----------
    model_label : str
        Short model label as defined in config.yaml (e.g. "BGE-EN-large").
    embeddings_dir : str or Path
        Directory produced by ``shared/precompute.py``.
        Typically ``experiments/data/processed/embeddings/``.

    Returns
    -------
    vectors : np.ndarray
        Float32 array of shape ``(N, dim)``, L2-normalized.
        ``vectors[i]`` is the embedding for ``index[i]``.
    index : list[dict]
        Ordered list of term records, each with keys
        ``en``, ``zh_canonical``, ``domain``, ``tier``.

    Raises
    ------
    FileNotFoundError
        If the model directory or index file does not exist.
    """
    embeddings_dir = Path(embeddings_dir)
    index_path = embeddings_dir / "index.json"
    vec_path = embeddings_dir / model_label / "vectors.npy"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            "Run `python shared/precompute.py` first."
        )
    if not vec_path.exists():
        raise FileNotFoundError(
            f"Vectors not found: {vec_path}\n"
            f"Run `python shared/precompute.py --models {model_label}` first."
        )

    index: list[dict] = json.load(index_path.open(encoding="utf-8"))
    vectors: np.ndarray = np.load(vec_path)
    return vectors, index


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single embedding model, as defined in config.yaml."""

    id: str            # HuggingFace model identifier
    label: str         # Short human-readable label (used in plots and tables)
    lang: str          # "en" (WEIRD tradition) or "zh" (Sinic tradition)
    dim: int           # Output embedding dimension
    instruction: str   # Prefix prepended to each text before encoding (empty = none)
    note: str          # Free-text annotation (not used programmatically)


class EmbeddingClient:
    """
    Unified embedding client for all models in the Geometria Iuris pipeline.

    Loads model specifications from config.yaml and exposes a single ``embed``
    method that returns L2-normalized embedding arrays. Models are loaded lazily
    and kept in memory; computed embeddings are stored on disk as .npy files
    keyed by a SHA-256 digest of the (model_id, texts) pair.

    Parameters
    ----------
    config_path : str or Path
        Path to models/config.yaml.
    cache_dir : str or Path or None
        Directory for on-disk embedding cache. Defaults to the path in config.yaml
        (resolved relative to config_path). Pass None to disable caching.
    device : str
        PyTorch device string ("cpu" or "cuda").
    batch_size : int or None
        Override the batch_size from config.yaml.

    Examples
    --------
    >>> client = EmbeddingClient("experiments/models/config.yaml")
    >>> vecs = client.embed(["mens rea", "habeas corpus"], "BAAI/bge-large-en-v1.5")
    >>> vecs.shape
    (2, 1024)
    >>> import numpy as np; np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    True
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        config_path = Path(config_path).resolve()
        with config_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Parse model specifications from all tradition groups
        self._specs: dict[str, ModelSpec] = {}
        self._groups: dict[str, list[str]] = {}
        for group in ("weird", "sinic", "bilingual"):
            ids: list[str] = []
            for entry in raw.get(group, []):
                spec = ModelSpec(
                    id=entry["id"],
                    label=entry["label"],
                    lang=entry["lang"],
                    dim=entry["dim"],
                    instruction=entry.get("instruction", ""),
                    note=entry.get("note", ""),
                )
                self._specs[spec.id] = spec
                ids.append(spec.id)
            self._groups[group] = ids

        # Embedding settings
        emb_cfg = raw.get("embedding", {})
        self._normalize: bool = emb_cfg.get("normalize", True)
        # Device priority: constructor arg > config.yaml > auto-detect
        self._device: str = device or emb_cfg.get("device") or self._detect_device()
        self._batch_size: int = batch_size or emb_cfg.get("batch_size", 32)

        # Disk cache setup
        cache_cfg = raw.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True)
        if cache_dir is not None:
            resolved_cache: Path | None = Path(cache_dir)
        elif cache_enabled:
            raw_dir = cache_cfg.get("dir", "")
            if raw_dir:
                # Path in config is relative to the project root
                resolved_cache = config_path.parent.parent / raw_dir
            else:
                resolved_cache = (
                    config_path.parent.parent
                    / "data"
                    / "processed"
                    / "embeddings_cache"
                )
        else:
            resolved_cache = None

        self._cache_dir: Path | None = resolved_cache
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory model registry (populated on first use)
        self._loaded: dict[str, SentenceTransformer] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        model_id: str,
        *,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Return L2-normalized embeddings for ``texts`` using ``model_id``.

        Parameters
        ----------
        texts : list[str]
            Input texts. Ordering is preserved in the output array.
        model_id : str
            HuggingFace model identifier as listed in config.yaml.
        use_cache : bool
            Read from and write to the on-disk cache.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(texts), dim)``, L2-normalized.
            cosine(u, v) = u · v holds after normalization.
        """
        if model_id not in self._specs:
            raise ValueError(
                f"Unknown model '{model_id}'.\n"
                f"Available: {list(self._specs)}"
            )

        if use_cache and self._cache_dir is not None:
            cached = self._load_cache(model_id, texts)
            if cached is not None:
                logger.debug("Cache hit: %s (%d texts)", model_id, len(texts))
                return cached

        model = self._get_model(model_id)
        spec = self._specs[model_id]

        # Prepend instruction if the model requires it
        inputs = (
            [spec.instruction + t for t in texts]
            if spec.instruction
            else texts
        )

        logger.info("Encoding %d texts with %s ...", len(texts), spec.label)
        vecs: np.ndarray = model.encode(
            inputs,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(texts) > 100,
        )
        vecs = vecs.astype(np.float32)

        if use_cache and self._cache_dir is not None:
            self._save_cache(model_id, texts, vecs)

        return vecs

    def embed_all(
        self,
        texts: list[str],
        *,
        group: Literal["weird", "sinic", "all"] = "all",
        use_cache: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Embed ``texts`` with every model in ``group``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from model_id to embedding array of shape ``(N, dim)``.
        """
        model_ids = (
            list(self._specs) if group == "all" else self._groups[group]
        )
        return {
            mid: self.embed(texts, mid, use_cache=use_cache)
            for mid in model_ids
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def all_specs(self) -> list[ModelSpec]:
        """All model specs (WEIRD first, then Sinic)."""
        return list(self._specs.values())

    @property
    def weird_specs(self) -> list[ModelSpec]:
        """WEIRD model specs in config order."""
        return [self._specs[mid] for mid in self._groups["weird"]]

    @property
    def sinic_specs(self) -> list[ModelSpec]:
        """Sinic model specs in config order."""
        return [self._specs[mid] for mid in self._groups["sinic"]]

    @property
    def bilingual_specs(self) -> list[ModelSpec]:
        """Bilingual control model specs in config order."""
        return [self._specs[mid] for mid in self._groups.get("bilingual", [])]

    def group_ids(self, group: Literal["weird", "sinic", "bilingual"]) -> list[str]:
        """Model IDs for a given tradition group."""
        return list(self._groups.get(group, []))

    def unload_model(self, model_id: str) -> None:
        """Remove a loaded model from memory (useful on RAM-constrained machines)."""
        if model_id in self._loaded:
            del self._loaded[model_id]
            logger.info("Unloaded model '%s'", model_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_model(self, model_id: str) -> SentenceTransformer:
        if model_id not in self._loaded:
            logger.info("Loading model '%s' ...", model_id)
            self._loaded[model_id] = SentenceTransformer(
                model_id, device=self._device, trust_remote_code=True,
            )
        return self._loaded[model_id]

    def _cache_key(self, model_id: str, texts: list[str]) -> str:
        """
        Deterministic cache filename for a (model, texts) pair.

        Key = SHA-256 digest (truncated to 20 hex chars) of the JSON-serialized
        payload. The model label is prepended for human readability.
        Order of texts is significant: embed(["a","b"]) and embed(["b","a"])
        produce different cache files.
        """
        payload = json.dumps(
            {"model": model_id, "texts": texts},
            ensure_ascii=False,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        label = self._specs[model_id].label.replace("/", "_")
        return f"{label}_{digest}.npy"

    def _load_cache(self, model_id: str, texts: list[str]) -> np.ndarray | None:
        path = self._cache_dir / self._cache_key(model_id, texts)  # type: ignore[operator]
        if path.exists():
            return np.load(path)
        return None

    def _save_cache(
        self, model_id: str, texts: list[str], vecs: np.ndarray
    ) -> None:
        path = self._cache_dir / self._cache_key(model_id, texts)  # type: ignore[operator]
        np.save(path, vecs)
        logger.debug("Cached: %s", path.name)
