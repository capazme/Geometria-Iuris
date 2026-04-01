"""
Lens V — Semantic Neighborhoods.

Implements the full Lens V analysis pipeline (Ch.3 §3.2):

  §3.2.1  Neighborhood overlap: cross- vs within-tradition Jaccard
  §3.2.2  Juridical false friends: top-20 divergent terms
  §3.2.3  Domain divergence: Kruskal-Wallis + pairwise Mann-Whitney

Usage
-----
    cd experiments/
    python -m lens_5_neighborhoods.lens5                    # full run
    python -m lens_5_neighborhoods.lens5 --section 3.2.1    # overlap only
    python -m lens_5_neighborhoods.lens5 --k 15 --n-perm 1000
    python -m lens_5_neighborhoods.lens5 --no-viz

Outputs
-------
    lens_5_neighborhoods/results/lens5_results.json
    lens_5_neighborhoods/results/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import kruskal

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.embeddings import load_precomputed
from shared.statistical import mannwhitney_with_r

RESULTS_DIR = Path(__file__).parent / "results"
EMB_DIR = ROOT / "data" / "processed" / "embeddings"
CONFIG_PATH = ROOT / "models" / "config.yaml"

# 12 a priori false friend candidates from domain_mapping_rules.md
A_PRIORI_FALSE_FRIENDS = [
    "contract", "law", "rights", "equity", "person", "sovereignty",
    "punishment", "judge", "property", "mediation", "precedent", "due process",
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> tuple[list[str], list[str]]:
    """Return (weird_labels, sinic_labels) from config.yaml."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    weird = [m["label"] for m in raw.get("weird", [])]
    sinic = [m["label"] for m in raw.get("sinic", [])]
    return weird, sinic


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all_vecs(labels: list[str]) -> tuple[
    dict[str, np.ndarray],
    list[dict],
    list[int],
    list[dict],
]:
    """
    Load precomputed vectors for all models.

    Returns
    -------
    vecs_all : {label: (9472, dim)} full embedding matrices
    index    : 9472 term records
    core_idx : positional indices of the 397 core terms
    terms_core : 397 core term records with en, zh_canonical, domain
    """
    _, index = load_precomputed(labels[0], EMB_DIR)
    core_idx = [i for i, t in enumerate(index) if t["tier"] == "core" and t["domain"]]
    terms_core = [index[i] for i in core_idx]

    vecs_all: dict[str, np.ndarray] = {}
    for label in labels:
        vecs, _ = load_precomputed(label, EMB_DIR)
        vecs_all[label] = vecs

    return vecs_all, index, core_idx, terms_core


# ---------------------------------------------------------------------------
# k-NN computation
# ---------------------------------------------------------------------------

def _compute_knn(
    vecs_all: np.ndarray,
    core_idx: list[int],
    k: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute k nearest neighbors for each core term in the full pool.

    Parameters
    ----------
    vecs_all : (N_pool, dim) L2-normalized
    core_idx : positions of core terms in the pool
    k : number of neighbors

    Returns
    -------
    knn : (n_core, k) int64 — pool indices of neighbors (NOT core indices)
    knn_sims : (n_core, k) float64 — cosine similarities to each neighbor
    """
    core_vecs = vecs_all[core_idx]                     # (n_core, dim)
    sims = core_vecs @ vecs_all.T                      # (n_core, N_pool)

    # Mask self-similarity: set sim to -inf for each core term's own position
    core_idx_arr = np.array(core_idx)
    for i, ci in enumerate(core_idx_arr):
        sims[i, ci] = -np.inf

    # Top-k by descending similarity
    # argsort ascending → take last k → reverse for descending order
    top_k = np.argsort(sims, axis=1)[:, -k:][:, ::-1]

    # Extract the cosine similarities for the selected neighbors
    knn_sims = np.take_along_axis(sims, top_k, axis=1)

    return top_k.astype(np.int64), knn_sims


# ---------------------------------------------------------------------------
# Jaccard computation
# ---------------------------------------------------------------------------

def _jaccard_pair(knn_a: np.ndarray, knn_b: np.ndarray) -> np.ndarray:
    """
    Per-term Jaccard similarity between two k-NN arrays.

    Parameters
    ----------
    knn_a, knn_b : (n_core, k) int64

    Returns
    -------
    jaccard : (n_core,) float in [0, 1]
    """
    n = len(knn_a)
    jaccard = np.empty(n, dtype=np.float64)
    for i in range(n):
        set_a = set(knn_a[i].tolist())
        set_b = set(knn_b[i].tolist())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard[i] = intersection / union if union > 0 else 0.0
    return jaccard


def _permutation_test_jaccard(
    knn_a: np.ndarray,
    knn_b: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Permutation test for mean Jaccard similarity.

    Null hypothesis: aligned pairs (term i in model A ↔ term i in model B)
    have the same mean Jaccard as randomly shuffled pairs.

    Permutes core indices: knn_a[π(i)] vs knn_b[i].

    Returns
    -------
    obs_mean : observed mean Jaccard
    p_value  : one-sided p (Phipson & Smyth bounded)
    null_dist : (n_perm,) null distribution of mean Jaccard
    """
    obs_mean = float(_jaccard_pair(knn_a, knn_b).mean())

    rng = np.random.default_rng(seed)
    n_core = len(knn_a)
    null = np.empty(n_perm, dtype=np.float64)

    for p in range(n_perm):
        pi = rng.permutation(n_core)
        null[p] = _jaccard_pair(knn_a[pi], knn_b).mean()

    # One-sided: how often does the null produce Jaccard >= observed?
    # (aligned pairs should have HIGHER Jaccard than random)
    # So p = P(null >= obs) tests H0: alignment doesn't help
    p_raw = float((null >= obs_mean).mean())
    p_value = max(p_raw, 1.0 / n_perm)   # Phipson & Smyth bound

    return obs_mean, p_value, null


# ---------------------------------------------------------------------------
# §3.2.1 — Neighborhood overlap
# ---------------------------------------------------------------------------

def run_section_321(
    weird_labels: list[str],
    sinic_labels: list[str],
    knn_all: dict[str, np.ndarray],
    k: int,
    n_perm: int,
) -> dict:
    """
    §3.2.1 — Neighborhood overlap: cross vs within-tradition Jaccard.

    Computes per-term Jaccard for:
    - 9 cross-tradition pairs (3 WEIRD × 3 Sinic)
    - 3 within-WEIRD pairs (C(3,2))
    - 3 within-Sinic pairs (C(3,2))

    Each pair: per-term Jaccard + permutation test on mean.
    Summary: Mann-Whitney comparing cross vs within distributions.
    """
    print(f"\n[§3.2.1] Neighborhood overlap (k={k}, n_perm={n_perm})")

    # Per-term Jaccard arrays saved to NPZ for granular visualizations
    npz_dir = RESULTS_DIR / "jaccard_per_term"
    npz_dir.mkdir(parents=True, exist_ok=True)

    def _run_pairs(pairs: list[tuple[str, str]], group_name: str) -> list[dict]:
        results = []
        for la, lb in pairs:
            t0 = time.perf_counter()
            jacc = _jaccard_pair(knn_all[la], knn_all[lb])
            obs_mean, p_val, null = _permutation_test_jaccard(
                knn_all[la], knn_all[lb], n_perm=n_perm,
            )
            elapsed = time.perf_counter() - t0
            print(f"    {la} × {lb}:  J̄={obs_mean:.4f}  p={p_val:.4f}  ({elapsed:.1f}s)")
            # Save per-term Jaccard array for granular viz
            np.save(npz_dir / f"{la}_x_{lb}.npy", jacc)
            results.append({
                "model_a": la,
                "model_b": lb,
                "group": group_name,
                "mean_jaccard": round(obs_mean, 4),
                "std_jaccard": round(float(jacc.std()), 4),
                "median_jaccard": round(float(np.median(jacc)), 4),
                "p_value": round(p_val, 4),
            })
        return results

    cross_pairs = list(product(weird_labels, sinic_labels))
    within_weird_pairs = list(combinations(weird_labels, 2))
    within_sinic_pairs = list(combinations(sinic_labels, 2))

    print("  Cross-tradition pairs:")
    cross_results = _run_pairs(cross_pairs, "cross")

    print("  Within-WEIRD pairs:")
    within_weird_results = _run_pairs(within_weird_pairs, "within_weird")

    print("  Within-Sinic pairs:")
    within_sinic_results = _run_pairs(within_sinic_pairs, "within_sinic")

    # Summary: cross vs within mean Jaccard (Mann-Whitney)
    cross_means = np.array([r["mean_jaccard"] for r in cross_results])
    within_means = np.array(
        [r["mean_jaccard"] for r in within_weird_results + within_sinic_results]
    )
    mw = mannwhitney_with_r(cross_means, within_means, alternative="less")
    print(f"\n  Summary: cross J̄={cross_means.mean():.4f}  "
          f"within J̄={within_means.mean():.4f}  "
          f"r={mw.effect_r:+.4f}  p={mw.p_value:.4f}")

    return {
        "k": k,
        "n_perm": n_perm,
        "cross_tradition": cross_results,
        "within_weird": within_weird_results,
        "within_sinic": within_sinic_results,
        "summary": {
            "mean_cross": round(float(cross_means.mean()), 4),
            "mean_within": round(float(within_means.mean()), 4),
            "mw_statistic": round(mw.statistic, 2),
            "mw_p_value": mw.p_value,
            "mw_effect_r": round(mw.effect_r, 4),
        },
    }


# ---------------------------------------------------------------------------
# §3.2.2 — False friends
# ---------------------------------------------------------------------------

def _majority_neighbors(
    knn_all: dict[str, np.ndarray],
    labels: list[str],
    term_idx: int,
    index: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Majority-vote top-N neighbors for a given term across a group of models.

    For each model in the group, collect the k neighbors of term_idx.
    Count how often each neighbor appears across models. Return the top_n
    most frequent, with their EN labels and domains.
    """
    counter: Counter = Counter()
    for label in labels:
        for ni in knn_all[label][term_idx]:
            counter[int(ni)] += 1

    top = counter.most_common(top_n)
    return [
        {
            "en": index[ni]["en"],
            "domain": index[ni].get("domain") or "control",
            "tier": index[ni]["tier"],
            "votes": count,
        }
        for ni, count in top
    ]


def _neighborhood_quality(knn_sims: np.ndarray) -> np.ndarray:
    """Mean cosine similarity to k-NN per term. Shape: (n_core,)."""
    return knn_sims.mean(axis=1)


def run_section_322(
    cross_results: list[dict],
    knn_all: dict[str, np.ndarray],
    knn_sims_all: dict[str, np.ndarray],
    terms_core: list[dict],
    index: list[dict],
    weird_labels: list[str],
    sinic_labels: list[str],
    k: int,
) -> dict:
    """
    §3.2.2 — Juridical false friends.

    Two-stage ranking:
    1. Neighborhood quality filter — exclude terms without dense neighborhoods
       in both traditions (mean cosine similarity to k-NN, min of WEIRD/Sinic).
    2. Rank surviving terms by mean cross-tradition Jaccard (ascending).

    Top-20 from the filtered pool = genuine false friends.
    """
    print("\n[§3.2.2] Juridical false friends")
    n_core = len(terms_core)

    # ── Stage 1: Neighborhood quality ──
    # Per-term mean cosine similarity to k-NN, per model
    quality_per_model = {
        label: _neighborhood_quality(knn_sims_all[label])
        for label in weird_labels + sinic_labels
    }
    # Average per tradition
    weird_q = np.mean([quality_per_model[l] for l in weird_labels], axis=0)
    sinic_q = np.mean([quality_per_model[l] for l in sinic_labels], axis=0)
    # Conservative: min of the two traditions
    quality = np.minimum(weird_q, sinic_q)  # (n_core,)

    quality_cutoff = float(np.percentile(quality, 25))
    quality_mask = quality >= quality_cutoff
    n_passed = int(quality_mask.sum())
    print(f"  Neighborhood quality: cutoff={quality_cutoff:.4f} "
          f"(Q1, 25th pctl), {n_passed}/{n_core} terms pass")

    # ── Stage 2: Cross-tradition Jaccard on filtered terms ──
    per_term_jaccard = np.zeros(n_core, dtype=np.float64)
    n_cross = 0
    for r in cross_results:
        la, lb = r["model_a"], r["model_b"]
        jacc = _jaccard_pair(knn_all[la], knn_all[lb])
        per_term_jaccard += jacc
        n_cross += 1
    per_term_jaccard /= n_cross

    # Rank only filtered terms: ascending Jaccard, then descending quality
    filtered_idx = np.where(quality_mask)[0]
    filtered_jaccard = per_term_jaccard[filtered_idx]
    filtered_quality = quality[filtered_idx]
    # lexsort: last key is primary → (secondary=-quality, primary=jaccard)
    rank_order = np.lexsort((-filtered_quality, filtered_jaccard))
    ranked_filtered = filtered_idx[rank_order]

    top_20 = []
    for rank, ti in enumerate(ranked_filtered[:20]):
        term = terms_core[ti]
        weird_nb = _majority_neighbors(knn_all, weird_labels, ti, index, top_n=5)
        sinic_nb = _majority_neighbors(knn_all, sinic_labels, ti, index, top_n=5)
        entry = {
            "rank": rank + 1,
            "en": term["en"],
            "zh": term["zh_canonical"],
            "domain": term["domain"],
            "mean_jaccard": round(float(per_term_jaccard[ti]), 4),
            "divergence": round(1.0 - float(per_term_jaccard[ti]), 4),
            "quality": round(float(quality[ti]), 4),
            "quality_weird": round(float(weird_q[ti]), 4),
            "quality_sinic": round(float(sinic_q[ti]), 4),
            "weird_neighbors": weird_nb,
            "sinic_neighbors": sinic_nb,
        }
        top_20.append(entry)
        print(f"    #{rank+1:2d}  {term['en']:<25s}  J̄={per_term_jaccard[ti]:.4f}  "
              f"q={quality[ti]:.4f}  [{term['domain']}]")

    # Save per-term arrays for viz
    npz_dir = RESULTS_DIR / "jaccard_per_term"
    npz_dir.mkdir(parents=True, exist_ok=True)
    np.save(npz_dir / "mean_cross_tradition.npy", per_term_jaccard)
    np.save(npz_dir / "neighborhood_quality.npy", quality)

    # All 397 terms with Jaccard + quality (for full-table viz)
    all_terms_data = []
    for ti in range(n_core):
        t = terms_core[ti]
        all_terms_data.append({
            "en": t["en"],
            "zh": t["zh_canonical"],
            "domain": t["domain"],
            "mean_jaccard": round(float(per_term_jaccard[ti]), 4),
            "divergence": round(1.0 - float(per_term_jaccard[ti]), 4),
            "quality": round(float(quality[ti]), 4),
            "quality_weird": round(float(weird_q[ti]), 4),
            "quality_sinic": round(float(sinic_q[ti]), 4),
            "passed_filter": bool(quality_mask[ti]),
        })

    return {
        "k": k,
        "top_20": top_20,
        "all_terms": all_terms_data,
        "per_term_mean_jaccard": {
            "mean": round(float(per_term_jaccard.mean()), 4),
            "std": round(float(per_term_jaccard.std()), 4),
            "min": round(float(per_term_jaccard.min()), 4),
            "max": round(float(per_term_jaccard.max()), 4),
        },
        "quality_filter": {
            "metric": "min(mean_cosine_sim_WEIRD, mean_cosine_sim_Sinic)",
            "cutoff_percentile": 25,
            "cutoff_value": round(quality_cutoff, 4),
            "n_total": n_core,
            "n_passed": n_passed,
            "n_excluded": n_core - n_passed,
            "quality_distribution": {
                "mean": round(float(quality.mean()), 4),
                "std": round(float(quality.std()), 4),
                "min": round(float(quality.min()), 4),
                "q25": round(float(np.percentile(quality, 25)), 4),
                "median": round(float(np.median(quality)), 4),
                "q75": round(float(np.percentile(quality, 75)), 4),
                "max": round(float(quality.max()), 4),
            },
        },
    }


# ---------------------------------------------------------------------------
# §3.2.3 — Domain divergence
# ---------------------------------------------------------------------------

def run_section_323(
    cross_results: list[dict],
    knn_all: dict[str, np.ndarray],
    terms_core: list[dict],
) -> dict:
    """
    §3.2.3 — Which domains diverge most?

    Groups mean cross-tradition Jaccard by domain.
    Kruskal-Wallis H test. If significant: pairwise Mann-Whitney with Bonferroni.
    """
    print("\n[§3.2.3] Domain divergence")
    n_core = len(terms_core)

    # Per-term mean cross-tradition Jaccard
    per_term_jaccard = np.zeros(n_core, dtype=np.float64)
    n_cross = 0
    for r in cross_results:
        la, lb = r["model_a"], r["model_b"]
        jacc = _jaccard_pair(knn_all[la], knn_all[lb])
        per_term_jaccard += jacc
        n_cross += 1
    per_term_jaccard /= n_cross

    # Group by domain
    domains = [t["domain"] for t in terms_core]
    domain_labels = sorted(set(domains))
    domain_arr = np.array(domains)

    domain_groups: dict[str, np.ndarray] = {}
    for d in domain_labels:
        mask = domain_arr == d
        domain_groups[d] = per_term_jaccard[mask]

    # Print per-domain stats
    domain_stats: dict[str, dict] = {}
    for d in domain_labels:
        g = domain_groups[d]
        domain_stats[d] = {
            "n": len(g),
            "mean_jaccard": round(float(g.mean()), 4),
            "std_jaccard": round(float(g.std()), 4),
            "median_jaccard": round(float(np.median(g)), 4),
        }
        print(f"    {d:<20s}  n={len(g):3d}  J̄={g.mean():.4f}  "
              f"med={np.median(g):.4f}  std={g.std():.4f}")

    # Kruskal-Wallis H test
    groups_list = [domain_groups[d] for d in domain_labels]
    h_stat, kw_p = kruskal(*groups_list)
    print(f"\n  Kruskal-Wallis: H={h_stat:.2f}  p={kw_p:.4e}")

    # Pairwise Mann-Whitney with Bonferroni if significant
    pairwise: list[dict] = []
    n_comparisons = len(domain_labels) * (len(domain_labels) - 1) // 2
    if kw_p < 0.05:
        print(f"  Pairwise Mann-Whitney (Bonferroni, {n_comparisons} comparisons):")
        for d1, d2 in combinations(domain_labels, 2):
            mw = mannwhitney_with_r(
                domain_groups[d1], domain_groups[d2], alternative="two-sided",
            )
            p_adj = min(mw.p_value * n_comparisons, 1.0)
            sig = "*" if p_adj < 0.05 else ""
            pairwise.append({
                "domain_a": d1,
                "domain_b": d2,
                "statistic": round(mw.statistic, 2),
                "p_raw": round(mw.p_value, 6),
                "p_bonferroni": round(p_adj, 6),
                "effect_r": round(mw.effect_r, 4),
                "significant": bool(p_adj < 0.05),
            })
            if sig:
                print(f"    {d1} vs {d2}:  r={mw.effect_r:+.3f}  "
                      f"p_adj={p_adj:.4f} {sig}")
    else:
        print("  Not significant — skipping pairwise tests.")

    # Per-domain raw values for viz (box plots with actual distributions)
    domain_values: dict[str, list[float]] = {
        d: [round(float(v), 4) for v in domain_groups[d]]
        for d in domain_labels
    }

    # Domain × pair heatmap data: mean Jaccard per (domain, model-pair)
    pair_domain_heatmap: list[dict] = []
    for r in cross_results:
        la, lb = r["model_a"], r["model_b"]
        jacc = _jaccard_pair(knn_all[la], knn_all[lb])
        row: dict = {"pair": f"{la} × {lb}"}
        for d in domain_labels:
            mask = domain_arr == d
            row[d] = round(float(jacc[mask].mean()), 4)
        pair_domain_heatmap.append(row)

    return {
        "domain_stats": domain_stats,
        "domain_values": domain_values,
        "pair_domain_heatmap": pair_domain_heatmap,
        "kruskal_wallis": {
            "H": round(float(h_stat), 4),
            "p_value": float(kw_p),
            "significant": bool(kw_p < 0.05),
        },
        "pairwise": pairwise,
        "n_comparisons": n_comparisons,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Lens V — Semantic Neighborhoods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--section",
        choices=["3.2.1", "3.2.2", "3.2.3", "all"],
        default="all",
        help="Which section(s) to run",
    )
    parser.add_argument("--k", type=int, default=15,
                        help="k for k-NN neighborhoods")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Permutations for significance test")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip figure generation after pipeline")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weird_labels, sinic_labels = _load_config()
    all_labels = weird_labels + sinic_labels

    print("=" * 60)
    print("Lens V — Semantic Neighborhoods")
    print(f"  WEIRD : {weird_labels}")
    print(f"  Sinic : {sinic_labels}")
    print(f"  k={args.k}  n_perm={args.n_perm}")
    print("=" * 60)

    # Load all vectors
    print("\n[load] Loading embeddings...")
    t0 = time.perf_counter()
    vecs_all, index, core_idx, terms_core = _load_all_vecs(all_labels)
    print(f"  {len(all_labels)} models loaded, pool={len(index)}, "
          f"core={len(core_idx)}  ({time.perf_counter()-t0:.1f}s)")

    # Compute k-NN for all models (once, reused across sections)
    print(f"\n[knn] Computing k={args.k} nearest neighbors...")
    t0 = time.perf_counter()
    knn_all: dict[str, np.ndarray] = {}
    knn_sims_all: dict[str, np.ndarray] = {}
    for label in all_labels:
        knn_all[label], knn_sims_all[label] = _compute_knn(
            vecs_all[label], core_idx, k=args.k,
        )
        print(f"    {label}  knn shape={knn_all[label].shape}")
    print(f"  Done ({time.perf_counter()-t0:.1f}s)")

    output: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "k": args.k,
            "n_perm": args.n_perm,
            "n_core": len(core_idx),
            "n_pool": len(index),
            "weird_models": weird_labels,
            "sinic_models": sinic_labels,
        }
    }

    t_start = time.perf_counter()

    if args.section in ("3.2.1", "all"):
        output["section_321"] = run_section_321(
            weird_labels, sinic_labels, knn_all,
            k=args.k, n_perm=args.n_perm,
        )

    cross_results = output.get("section_321", {}).get("cross_tradition", [])

    if args.section in ("3.2.2", "all"):
        if not cross_results:
            print("\n[§3.2.2] Skipped — requires §3.2.1 cross_tradition results.")
        else:
            output["section_322"] = run_section_322(
                cross_results, knn_all, knn_sims_all,
                terms_core, index,
                weird_labels, sinic_labels, k=args.k,
            )

    if args.section in ("3.2.3", "all"):
        if not cross_results:
            print("\n[§3.2.3] Skipped — requires §3.2.1 cross_tradition results.")
        else:
            output["section_323"] = run_section_323(
                cross_results, knn_all, terms_core,
            )

    total = time.perf_counter() - t_start
    output["meta"]["elapsed_seconds"] = round(total, 1)

    out_path = RESULTS_DIR / "lens5_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done in {total:.0f}s  →  {out_path}")

    if not args.no_viz:
        from lens_5_neighborhoods.viz import run_viz
        run_viz(RESULTS_DIR, output)


if __name__ == "__main__":
    main()
