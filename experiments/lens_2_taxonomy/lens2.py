"""
Lens II — Emergent Taxonomy.

Implements the Lens II analysis pipeline (Ch.4 §4.4 — Horizons):

  §4.4.1  Taxonomic recovery: FM at k=7 vs human domain labels
  §4.4.2  Taxonomic horizons: FM(k) curves, k=2..20
  §4.4.3  Cross-tradition taxonomic agreement: FM between WEIRD/Sinic partitions

Usage
-----
    cd experiments/
    python -m lens_2_taxonomy.lens2                    # full run
    python -m lens_2_taxonomy.lens2 --section 4.4.1    # recovery only
    python -m lens_2_taxonomy.lens2 --n-perm 200       # fast dev run
    python -m lens_2_taxonomy.lens2 --no-viz

Outputs
-------
    lens_2_taxonomy/results/lens2_results.json
    lens_2_taxonomy/results/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.statistical import permutation_test_groups

RESULTS_DIR = Path(__file__).parent / "results"
RDM_DIR = ROOT / "lens_1_relational" / "results" / "rdms"
EMB_INDEX_PATH = ROOT / "data" / "processed" / "embeddings" / "index.json"
CONFIG_PATH = ROOT / "models" / "config.yaml"


# ---------------------------------------------------------------------------
# Config & data loading
# ---------------------------------------------------------------------------

def _load_config() -> tuple[list[str], list[str]]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    weird = [m["label"] for m in raw.get("weird", [])]
    sinic = [m["label"] for m in raw.get("sinic", [])]
    return weird, sinic


def _load_rdm(label: str) -> np.ndarray:
    path = RDM_DIR / f"{label}.npz"
    rdm = np.load(path)["rdm"].astype(np.float64)
    np.clip(rdm, 0, None, out=rdm)  # fix float32 rounding artifacts
    return rdm


def _load_human_labels() -> tuple[list[str], np.ndarray]:
    with EMB_INDEX_PATH.open(encoding="utf-8") as f:
        index = json.load(f)
    core = [t for t in index if t.get("tier") == "core" and t.get("domain")]
    domains_sorted = sorted(set(t["domain"] for t in core))
    domain_to_int = {d: i for i, d in enumerate(domains_sorted)}
    labels = np.array([domain_to_int[t["domain"]] for t in core], dtype=np.int64)
    return domains_sorted, labels


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _build_linkage(rdm: np.ndarray, method: str = "average") -> np.ndarray:
    condensed = squareform(rdm, checks=False)
    return linkage(condensed, method=method)


def _cluster_at_k(Z: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Z, t=k, criterion="maxclust")


# ---------------------------------------------------------------------------
# Fowlkes-Mallows Index (vectorized numpy, no sklearn)
# ---------------------------------------------------------------------------

def _fm_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fowlkes-Mallows index between two integer label arrays.

    FM = TP / sqrt((TP + FP) * (TP + FN))
    where TP/FP/FN are counts of concordant/discordant pairs.
    """
    n = len(a)
    # Co-assignment matrices (upper triangle only)
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    tp = int((same_a & same_b & mask).sum())
    fp = int((~same_a & same_b & mask).sum())
    fn = int((same_a & ~same_b & mask).sum())

    denom = np.sqrt(float((tp + fp) * (tp + fn)))
    return tp / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _permutation_test_fm(
    human_labels: np.ndarray,
    model_labels: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Permutation test for FM significance.

    Shuffles human_labels n_perm times, recomputes FM each time.
    Returns (observed_fm, p_value, null_distribution).
    """
    obs = _fm_score(human_labels, model_labels)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shuffled = rng.permutation(human_labels)
        null[i] = _fm_score(shuffled, model_labels)
    # Phipson & Smyth (2010): p = (b + 1) / (m + 1)
    b = int((null >= obs).sum())
    p_value = (b + 1) / (n_perm + 1)
    return obs, p_value, null


# ---------------------------------------------------------------------------
# §4.4.1 — Taxonomic recovery
# ---------------------------------------------------------------------------

def run_section_441(
    all_labels: list[str],
    weird_labels: list[str],
    sinic_labels: list[str],
    human_labels: np.ndarray,
    n_perm: int,
) -> dict:
    print("\n[§4.4.1] Taxonomic recovery (k=7)")

    null_dir = RESULTS_DIR / "null_dists"
    null_dir.mkdir(parents=True, exist_ok=True)

    per_model: dict[str, dict] = {}
    for label in all_labels:
        tradition = "WEIRD" if label in weird_labels else "Sinic"
        t0 = time.perf_counter()

        rdm = _load_rdm(label)
        Z_avg = _build_linkage(rdm, method="average")
        Z_cmp = _build_linkage(rdm, method="complete")

        model_labels_avg = _cluster_at_k(Z_avg, 7)
        model_labels_cmp = _cluster_at_k(Z_cmp, 7)

        fm_avg, p_val, null = _permutation_test_fm(
            human_labels, model_labels_avg, n_perm=n_perm,
        )
        fm_cmp = _fm_score(human_labels, model_labels_cmp)

        np.save(null_dir / f"{label}_k7.npy", null)

        elapsed = time.perf_counter() - t0
        print(f"    {label} ({tradition}):  FM={fm_avg:.4f}  "
              f"p={p_val:.4f}  null_μ={null.mean():.4f}  ({elapsed:.1f}s)")

        per_model[label] = {
            "tradition": tradition,
            "fm": round(fm_avg, 4),
            "p_value": round(p_val, 4),
            "null_mean": round(float(null.mean()), 4),
            "null_std": round(float(null.std()), 4),
            "fm_complete_linkage": round(fm_cmp, 4),
        }

    return {"k": 7, "n_perm": n_perm, "per_model": per_model}


# ---------------------------------------------------------------------------
# §4.4.2 — Taxonomic horizons (FM curves)
# ---------------------------------------------------------------------------

def run_section_442(
    all_labels: list[str],
    weird_labels: list[str],
    sinic_labels: list[str],
    human_labels: np.ndarray,
    Zs: dict[str, np.ndarray],
    k_min: int,
    k_max: int,
) -> dict:
    print(f"\n[§4.4.2] Taxonomic horizons (k={k_min}..{k_max})")
    k_range = list(range(k_min, k_max + 1))

    # FM vs human for each model
    human_curves: dict[str, list[float]] = {}
    for label in all_labels:
        curve = []
        for k in k_range:
            ml = _cluster_at_k(Zs[label], k)
            curve.append(round(_fm_score(human_labels, ml), 4))
        human_curves[label] = curve
        peak_k = k_range[int(np.argmax(curve))]
        print(f"    {label}:  peak FM at k={peak_k} ({max(curve):.4f})")

    # Cross-tradition FM curves
    cross_pairs = list(product(weird_labels, sinic_labels))
    cross_curves: dict[str, list[float]] = {}
    for la, lb in cross_pairs:
        curve = []
        for k in k_range:
            cl_a = _cluster_at_k(Zs[la], k)
            cl_b = _cluster_at_k(Zs[lb], k)
            curve.append(round(_fm_score(cl_a, cl_b), 4))
        key = f"{la}_x_{lb}"
        cross_curves[key] = curve

    # Within-tradition FM curves (for comparison in viz)
    within_weird_curves: dict[str, list[float]] = {}
    for la, lb in combinations(weird_labels, 2):
        curve = []
        for k in k_range:
            cl_a = _cluster_at_k(Zs[la], k)
            cl_b = _cluster_at_k(Zs[lb], k)
            curve.append(round(_fm_score(cl_a, cl_b), 4))
        within_weird_curves[f"{la}_x_{lb}"] = curve

    within_sinic_curves: dict[str, list[float]] = {}
    for la, lb in combinations(sinic_labels, 2):
        curve = []
        for k in k_range:
            cl_a = _cluster_at_k(Zs[la], k)
            cl_b = _cluster_at_k(Zs[lb], k)
            curve.append(round(_fm_score(cl_a, cl_b), 4))
        within_sinic_curves[f"{la}_x_{lb}"] = curve

    return {
        "k_range": k_range,
        "human_curves": human_curves,
        "cross_tradition_curves": cross_curves,
        "within_weird_curves": within_weird_curves,
        "within_sinic_curves": within_sinic_curves,
    }


# ---------------------------------------------------------------------------
# §4.4.3 — Cross-tradition taxonomic agreement
# ---------------------------------------------------------------------------

def run_section_443(
    weird_labels: list[str],
    sinic_labels: list[str],
    Zs: dict[str, np.ndarray],
) -> dict:
    print("\n[§4.4.3] Cross-tradition taxonomic agreement (k=7)")

    def _fm_pairs(pairs: list[tuple[str, str]], group: str) -> list[dict]:
        results = []
        for la, lb in pairs:
            cl_a = _cluster_at_k(Zs[la], 7)
            cl_b = _cluster_at_k(Zs[lb], 7)
            fm = _fm_score(cl_a, cl_b)
            results.append({"model_a": la, "model_b": lb, "fm": round(fm, 4)})
            print(f"    {la} × {lb}:  FM={fm:.4f}  [{group}]")
        return results

    cross_pairs = list(product(weird_labels, sinic_labels))
    within_weird_pairs = list(combinations(weird_labels, 2))
    within_sinic_pairs = list(combinations(sinic_labels, 2))

    print("  Cross-tradition:")
    cross = _fm_pairs(cross_pairs, "cross")
    print("  Within-WEIRD:")
    within_w = _fm_pairs(within_weird_pairs, "within_weird")
    print("  Within-Sinic:")
    within_s = _fm_pairs(within_sinic_pairs, "within_sinic")

    # Permutation test: cross vs all within (more appropriate than MW for n=9 vs n=6)
    cross_vals = np.array([r["fm"] for r in cross])
    within_vals = np.array([r["fm"] for r in within_w + within_s])
    pt = permutation_test_groups(cross_vals, within_vals, alternative="less")

    print(f"\n  Summary: cross FM̄={cross_vals.mean():.4f}  "
          f"within FM̄={within_vals.mean():.4f}  "
          f"r={pt.effect_r:+.4f}  p={pt.p_value:.4f}")

    return {
        "k": 7,
        "cross_tradition": cross,
        "within_weird": within_w,
        "within_sinic": within_s,
        "summary": {
            "mean_fm_cross": round(float(cross_vals.mean()), 4),
            "mean_fm_within_weird": round(float(np.mean([r["fm"] for r in within_w])), 4),
            "mean_fm_within_sinic": round(float(np.mean([r["fm"] for r in within_s])), 4),
            "mean_fm_within": round(float(within_vals.mean()), 4),
        },
        "perm_cross_vs_within": {
            "p_value": pt.p_value,
            "effect_r": round(pt.effect_r, 4),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Lens II — Emergent Taxonomy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--section",
        choices=["4.4.1", "4.4.2", "4.4.3", "all"],
        default="all",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weird_labels, sinic_labels = _load_config()
    all_labels = weird_labels + sinic_labels
    domain_names, human_labels = _load_human_labels()

    print("=" * 60)
    print("Lens II — Emergent Taxonomy")
    print(f"  WEIRD : {weird_labels}")
    print(f"  Sinic : {sinic_labels}")
    print(f"  Domains: {domain_names}")
    print(f"  n_perm={args.n_perm}  k={args.k_min}..{args.k_max}")
    print("=" * 60)

    # Pre-build linkage matrices (reused across sections)
    print("\n[load] Building linkage matrices from Lens I RDMs...")
    t0 = time.perf_counter()
    Zs: dict[str, np.ndarray] = {}
    for label in all_labels:
        rdm = _load_rdm(label)
        Zs[label] = _build_linkage(rdm, method="average")
        print(f"    {label}  Z shape={Zs[label].shape}")
    print(f"  Done ({time.perf_counter()-t0:.1f}s)")

    output: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "n_perm": args.n_perm,
            "k_anchor": 7,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "linkage_method": "average",
            "n_core_terms": len(human_labels),
            "weird_models": weird_labels,
            "sinic_models": sinic_labels,
            "domains": domain_names,
            "domain_counts": {
                d: int((human_labels == i).sum())
                for i, d in enumerate(domain_names)
            },
        },
    }

    t_start = time.perf_counter()

    if args.section in ("4.4.1", "all"):
        output["section_441"] = run_section_441(
            all_labels, weird_labels, sinic_labels,
            human_labels, n_perm=args.n_perm,
        )

    if args.section in ("4.4.2", "all"):
        output["section_442"] = run_section_442(
            all_labels, weird_labels, sinic_labels,
            human_labels, Zs, k_min=args.k_min, k_max=args.k_max,
        )

    if args.section in ("4.4.3", "all"):
        output["section_443"] = run_section_443(
            weird_labels, sinic_labels, Zs,
        )

    total = time.perf_counter() - t_start
    output["meta"]["elapsed_seconds"] = round(total, 1)

    out_path = RESULTS_DIR / "lens2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done in {total:.0f}s  →  {out_path}")

    if not args.no_viz:
        from lens_2_taxonomy.viz import run_viz
        run_viz(RESULTS_DIR, output)


if __name__ == "__main__":
    main()
