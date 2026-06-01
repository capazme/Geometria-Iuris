"""
Pre-check 2 — Polysemy / representation robustness (Gamma, 2026-04-11).

Tests whether the Lens I headline result Δρ ≈ 0.260 (within-tradition vs
cross-tradition RSA correlation) is robust to the choice of aggregation
function when a legal term is represented by a cloud of variants rather
than by a single bare embedding.

This pre-check decides the urgency of D-A (contextualised extraction with
mean pooling). If Δρ is stable across {mean, medoid, first PC, bare term},
D-A is a preference rather than a necessity and the existing pipeline is
defensible. If Δρ is fragile, D-A becomes critical and the Lens I result
must be reported alongside a sensitivity analysis over aggregation methods.

Procedure
---------
For each of the 397 core terms and each of the six models, encode five
synthetic variants of the term (one bare form and four template surface
variations), then derive four term representations from the resulting
cloud:
  A. Mean        (normalised arithmetic mean of the 5 vectors)
  B. Medoid      (variant with maximal average cosine to the others)
  C. First PC    (top PC of the 5-vector cloud, oriented toward the mean)
  D. Bare term   (variant 1 alone)

For each representation, rebuild the 397 × 397 RDM per model, compute the
standard Lens I aggregates (within-WEIRD ρ̄, within-Sinic ρ̄, cross-tradition
ρ̄) over the 3+3 design, and report Δρ = ρ̄_within − ρ̄_cross.

Decision thresholds
-------------------
  ROBUST     : max |Δρ(X) − Δρ(Y)| over all pairs (A, B, C, D) < 0.05
  FRAGILE    : max |Δρ(X) − Δρ(Y)| > 0.10
  MODERATE   : in between

If FRAGILE, adopt Option Alpha (full revision, D-A with proper
contextualised extraction and multi-prototype sensitivity). If ROBUST,
Option Beta remains on the table. Between: Alpha-lite.

Background
----------
- Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2018). "Linear
  Algebraic Structure of Word Senses with Applications to Polysemy." TACL.
- Chronis, G., & Erk, K. (2020). "When is a bishop not like a rook?
  Multi-prototype BERT Embeddings for Estimating Semantic Relationships."
  CoNLL.

Output
------
JSON report in pre_checks/results/precheck_2_polysemy.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.embeddings import EmbeddingClient  # noqa: E402
from shared.statistical import compute_rdm, upper_tri  # noqa: E402


# ---------------------------------------------------------------------------
# Template variants (one cloud per term, per language)
# ---------------------------------------------------------------------------

EN_TEMPLATES = [
    "{term}",
    "the legal term {term}",
    "the concept of {term} in law",
    "{term}, as used in legislation,",
    "the court considered the {term}",
]

ZH_TEMPLATES = [
    "{term}",
    "法律術語{term}",
    "法律上的{term}概念",
    "法例中的{term}",
    "法院考慮了{term}",
]


# ---------------------------------------------------------------------------
# Model groups (mirrors the 3+3 design of models/config.yaml)
# ---------------------------------------------------------------------------

WEIRD_MODELS = [
    ("BAAI/bge-large-en-v1.5", "BGE-EN-large"),
    ("intfloat/e5-large-v2", "E5-large"),
    ("freelawproject/modernbert-embed-base_finetune_512", "FreeLaw-EN"),
]

SINIC_MODELS = [
    ("BAAI/bge-large-zh-v1.5", "BGE-ZH-large"),
    ("GanymedeNil/text2vec-large-chinese", "Text2vec-large-ZH"),
    ("DMetaSoul/Dmeta-embedding-zh", "Dmeta-ZH"),
]


# ---------------------------------------------------------------------------
# Aggregation functions (operate on an (n_variants, dim) cloud)
# ---------------------------------------------------------------------------

def agg_mean(cloud: np.ndarray) -> np.ndarray:
    v = cloud.mean(axis=0)
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-12)


def agg_medoid(cloud: np.ndarray) -> np.ndarray:
    sims = cloud @ cloud.T
    np.fill_diagonal(sims, -np.inf)
    avg_sim = sims.mean(axis=1)
    idx = int(np.argmax(avg_sim))
    return cloud[idx]


def agg_first_pc(cloud: np.ndarray) -> np.ndarray:
    centered = cloud - cloud.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    # Orient so PC1 has positive inner product with the mean direction
    mean_dir = cloud.mean(axis=0)
    if pc1 @ mean_dir < 0:
        pc1 = -pc1
    norm = np.linalg.norm(pc1)
    return pc1 / max(norm, 1e-12)


def agg_bare(cloud: np.ndarray) -> np.ndarray:
    # Variant 1 is the bare term (unit-normalised already from the client)
    v = cloud[0]
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-12)


AGGREGATORS = {
    "mean": agg_mean,
    "medoid": agg_medoid,
    "first_pc": agg_first_pc,
    "bare": agg_bare,
}


# ---------------------------------------------------------------------------
# Encoding and RDM construction
# ---------------------------------------------------------------------------

def build_term_cloud(
    terms: list[str],
    templates: list[str],
) -> tuple[list[str], int, int]:
    """
    Return a flat list of all (term, variant) strings and the shape info.
    Order: for term i, variants 0..V-1 are at positions i*V + 0 .. i*V + V-1.
    """
    flat: list[str] = []
    for t in terms:
        for tpl in templates:
            flat.append(tpl.format(term=t))
    return flat, len(terms), len(templates)


def aggregate_cloud(
    flat_vectors: np.ndarray,
    n_terms: int,
    n_variants: int,
    aggregator_name: str,
) -> np.ndarray:
    """Collapse the flat (n_terms * n_variants, dim) matrix to (n_terms, dim)."""
    dim = flat_vectors.shape[1]
    out = np.empty((n_terms, dim), dtype=np.float32)
    agg_fn = AGGREGATORS[aggregator_name]
    for i in range(n_terms):
        cloud = flat_vectors[i * n_variants : (i + 1) * n_variants]
        out[i] = agg_fn(cloud)
    # Re-normalise defensively
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.clip(norms, 1e-12, None)
    return out.astype(np.float32)


def spearman_rdm(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman rho between the upper triangles of two RDMs."""
    a = upper_tri(rdm_a)
    b = upper_tri(rdm_b)
    rho, _ = spearmanr(a, b)
    return float(rho)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = REPO_ROOT / "models" / "config.yaml"
    client = EmbeddingClient(str(config_path))

    # Load core terms from index
    embeddings_dir = REPO_ROOT / "data" / "processed" / "embeddings"
    index = json.load((embeddings_dir / "index.json").open(encoding="utf-8"))
    core_mask = [i for i, e in enumerate(index) if e.get("tier") == "core"]
    print(f"[polysemy] {len(core_mask)} core terms detected")

    en_terms = [index[i]["en"] for i in core_mask]
    zh_terms = [index[i]["zh_canonical"] for i in core_mask]

    # Build the flat input lists
    en_flat, n_terms_en, n_variants_en = build_term_cloud(en_terms, EN_TEMPLATES)
    zh_flat, n_terms_zh, n_variants_zh = build_term_cloud(zh_terms, ZH_TEMPLATES)
    assert n_terms_en == n_terms_zh == len(core_mask)
    assert n_variants_en == n_variants_zh == 5
    n_terms = n_terms_en
    n_variants = n_variants_en

    # Encode all variants, all models
    print(
        f"[polysemy] encoding {n_terms} terms × {n_variants} variants = "
        f"{n_terms * n_variants} inputs per model"
    )
    weird_flat: dict[str, np.ndarray] = {}
    sinic_flat: dict[str, np.ndarray] = {}
    for model_id, label in WEIRD_MODELS:
        print(f"[polysemy]   WEIRD encoding: {label}")
        weird_flat[label] = client.embed(en_flat, model_id, use_cache=True)
    for model_id, label in SINIC_MODELS:
        print(f"[polysemy]   Sinic encoding: {label}")
        sinic_flat[label] = client.embed(zh_flat, model_id, use_cache=True)

    # For each aggregator, build per-model term representations and RDMs
    per_aggregator: dict[str, dict] = {}
    for agg_name in AGGREGATORS:
        print(f"[polysemy] aggregator = {agg_name}")
        weird_rdms: dict[str, np.ndarray] = {}
        sinic_rdms: dict[str, np.ndarray] = {}
        for label in weird_flat:
            vecs = aggregate_cloud(weird_flat[label], n_terms, n_variants, agg_name)
            weird_rdms[label] = compute_rdm(vecs)
        for label in sinic_flat:
            vecs = aggregate_cloud(sinic_flat[label], n_terms, n_variants, agg_name)
            sinic_rdms[label] = compute_rdm(vecs)

        # Within-WEIRD pairs (3 choose 2 = 3)
        weird_pairs = []
        for a, b in combinations(sorted(weird_rdms.keys()), 2):
            rho = spearman_rdm(weird_rdms[a], weird_rdms[b])
            weird_pairs.append({"a": a, "b": b, "rho": rho})
        within_weird_mean = float(np.mean([p["rho"] for p in weird_pairs]))

        # Within-Sinic pairs
        sinic_pairs = []
        for a, b in combinations(sorted(sinic_rdms.keys()), 2):
            rho = spearman_rdm(sinic_rdms[a], sinic_rdms[b])
            sinic_pairs.append({"a": a, "b": b, "rho": rho})
        within_sinic_mean = float(np.mean([p["rho"] for p in sinic_pairs]))

        # Cross-tradition pairs (3 × 3 = 9)
        cross_pairs = []
        for w_label in sorted(weird_rdms.keys()):
            for s_label in sorted(sinic_rdms.keys()):
                rho = spearman_rdm(weird_rdms[w_label], sinic_rdms[s_label])
                cross_pairs.append({"weird": w_label, "sinic": s_label, "rho": rho})
        cross_mean = float(np.mean([p["rho"] for p in cross_pairs]))

        within_mean = (within_weird_mean + within_sinic_mean) / 2.0
        delta_rho = within_mean - cross_mean

        per_aggregator[agg_name] = {
            "within_weird_mean": within_weird_mean,
            "within_sinic_mean": within_sinic_mean,
            "within_mean": within_mean,
            "cross_mean": cross_mean,
            "delta_rho": delta_rho,
            "weird_pairs": weird_pairs,
            "sinic_pairs": sinic_pairs,
            "cross_pairs": cross_pairs,
        }
        print(
            f"  within-WEIRD={within_weird_mean:.4f}  "
            f"within-Sinic={within_sinic_mean:.4f}  "
            f"cross={cross_mean:.4f}  Δρ={delta_rho:+.4f}"
        )

    # Decision
    deltas = {name: r["delta_rho"] for name, r in per_aggregator.items()}
    max_pairwise_diff = 0.0
    max_pair = None
    for a, b in combinations(deltas.keys(), 2):
        diff = abs(deltas[a] - deltas[b])
        if diff > max_pairwise_diff:
            max_pairwise_diff = diff
            max_pair = (a, b)

    if max_pairwise_diff < 0.05:
        status = "ROBUST"
        narrative = (
            "Lens I Δρ is stable across all four aggregation methods. D-A "
            "(contextualised extraction with mean pooling) is a methodological "
            "preference rather than a necessity for the headline result. "
            "Option Beta remains viable."
        )
    elif max_pairwise_diff > 0.10:
        status = "FRAGILE"
        narrative = (
            "Lens I Δρ is fragile across aggregation methods. The headline "
            "result is partially an artefact of the aggregation choice. D-A "
            "is critical: the full contextualised pipeline with a proper "
            "sensitivity analysis over aggregators is necessary. Option Alpha "
            "is required."
        )
    else:
        status = "MODERATE"
        narrative = (
            "Lens I Δρ is moderately sensitive to aggregation. D-A is "
            "justified but not urgent. The existing pipeline can stand as "
            "primary with the aggregation sensitivity reported alongside. "
            "Option Alpha-lite is appropriate."
        )

    report = {
        "pre_check": "2_polysemy",
        "date": "2026-04-11",
        "n_terms": n_terms,
        "n_variants": n_variants,
        "templates": {"en": EN_TEMPLATES, "zh": ZH_TEMPLATES},
        "weird_models": [label for _, label in WEIRD_MODELS],
        "sinic_models": [label for _, label in SINIC_MODELS],
        "per_aggregator": per_aggregator,
        "summary_delta_rho": deltas,
        "max_pairwise_diff": max_pairwise_diff,
        "max_pair": max_pair,
        "thresholds": {"robust_max_diff": 0.05, "fragile_min_diff": 0.10},
        "aggregate": {"status": status, "narrative": narrative},
    }

    out_path = REPO_ROOT / "pre_checks" / "results" / "precheck_2_polysemy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[polysemy] Status: {status}")
    print(f"[polysemy] Max pairwise Δρ diff: {max_pairwise_diff:.4f}")
    print(f"[polysemy] Report written to {out_path}")


if __name__ == "__main__":
    main()
