"""
Lens II §4.4 robustness check on the contextualised pool.

Recomputes §4.4.1 (FM at k=7 vs human taxonomy) and §4.4.3
(cross-tradition FM: within-WEIRD, within-Sinic, cross) on the
contextualised term pool built by
`lens_5_neighborhoods/build_contextualized_pool.py`, and reports side
by side with the bare-pool numbers.

The pipeline matches `lens_2_taxonomy/lens2.py` exactly in the clustering
step:
  - average-linkage hierarchical clustering on the (1 - cosine) RDM
  - fcluster at k=7
  - Fowlkes-Mallows index vs human domain labels
  - Fowlkes-Mallows cross-model (within-W, within-S, cross)

Output: results/clustering_ctx_robustness.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.embeddings import load_precomputed  # noqa: E402
from shared.statistical import compute_rdm  # noqa: E402


WEIRD_LABELS = ["BGE-EN-large", "E5-large", "FreeLaw-EN"]
SINIC_LABELS = ["BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"]
ALL_LABELS = WEIRD_LABELS + SINIC_LABELS

EMB_BARE = REPO_ROOT / "data" / "processed" / "embeddings"
EMB_CTX = REPO_ROOT / "data" / "processed" / "embeddings_contextualized"

K_HUMAN = 7  # 7 human domains in the published §4.4.1


def core_indices_and_labels(index: list[dict]) -> tuple[list[int], np.ndarray, list[str]]:
    """Return core positional indices, integer-encoded human domain labels, and domain order."""
    core_idx = [i for i, t in enumerate(index) if t["tier"] == "core" and t["domain"]]
    core_terms = [index[i] for i in core_idx]
    domains_sorted = sorted(set(t["domain"] for t in core_terms))
    domain_to_int = {d: i for i, d in enumerate(domains_sorted)}
    labels = np.array([domain_to_int[t["domain"]] for t in core_terms], dtype=np.int64)
    return core_idx, labels, domains_sorted


def build_rdm_from_pool(pool_root: Path, label: str, core_idx: list[int]) -> np.ndarray:
    """Load pool vectors for one model, slice the core, and compute (1 - cosine) RDM."""
    vecs, _ = load_precomputed(label, pool_root)
    return compute_rdm(vecs[core_idx]).astype(np.float64)


def cluster_at_k(rdm: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage hierarchical clustering, cut at k clusters."""
    # Clip float32 artefacts before squareform
    np.clip(rdm, 0, None, out=rdm)
    condensed = squareform(rdm, checks=False)
    Z = linkage(condensed, method="average")
    return fcluster(Z, t=k, criterion="maxclust")


def fm_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fowlkes-Mallows index.
    FM = TP / sqrt((TP + FP) * (TP + FN))
    """
    n = len(a)
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    tp = int((same_a & same_b & mask).sum())
    fp = int((~same_a & same_b & mask).sum())
    fn = int((same_a & ~same_b & mask).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn)))
    return tp / denom if denom > 0 else 0.0


def run_clustering_on_pool(pool_root: Path, label: str, human_labels: np.ndarray, core_idx: list[int]) -> dict:
    print(f"\n[{label}] Clustering on {pool_root.relative_to(REPO_ROOT)} …")
    cluster_labels: dict[str, np.ndarray] = {}
    per_model: dict[str, dict] = {}

    for model in ALL_LABELS:
        rdm = build_rdm_from_pool(pool_root, model, core_idx)
        cl = cluster_at_k(rdm, k=K_HUMAN)
        cluster_labels[model] = cl
        fm_vs_human = fm_score(human_labels, cl)
        per_model[model] = {"fm_vs_human_k7": round(float(fm_vs_human), 4)}
        print(f"  {model:20s}  FM(vs human, k=7) = {fm_vs_human:.4f}")

    # §4.4.3: cross-model FM
    within_weird_pairs = list(combinations(WEIRD_LABELS, 2))
    within_sinic_pairs = list(combinations(SINIC_LABELS, 2))
    cross_pairs = list(product(WEIRD_LABELS, SINIC_LABELS))

    def fm_between(la: str, lb: str) -> float:
        return fm_score(cluster_labels[la], cluster_labels[lb])

    within_weird = [
        {"a": la, "b": lb, "fm": round(fm_between(la, lb), 4)}
        for la, lb in within_weird_pairs
    ]
    within_sinic = [
        {"a": la, "b": lb, "fm": round(fm_between(la, lb), 4)}
        for la, lb in within_sinic_pairs
    ]
    cross = [
        {"a": la, "b": lb, "fm": round(fm_between(la, lb), 4)}
        for la, lb in cross_pairs
    ]

    fm_w = float(np.mean([p["fm"] for p in within_weird]))
    fm_s = float(np.mean([p["fm"] for p in within_sinic]))
    fm_x = float(np.mean([p["fm"] for p in cross]))
    fm_mean_vs_human = float(np.mean([v["fm_vs_human_k7"] for v in per_model.values()]))

    print(
        f"\n[{label}] §4.4.3 cross-model FM:\n"
        f"  within-WEIRD FM̄ = {fm_w:.4f}\n"
        f"  within-Sinic FM̄ = {fm_s:.4f}\n"
        f"  cross        FM̄ = {fm_x:.4f}\n"
        f"  mean FM vs human (6 models, k=7) = {fm_mean_vs_human:.4f}"
    )

    return {
        "per_model": per_model,
        "within_weird_pairs": within_weird,
        "within_sinic_pairs": within_sinic,
        "cross_pairs": cross,
        "mean_fm_vs_human": fm_mean_vs_human,
        "mean_fm_within_weird": fm_w,
        "mean_fm_within_sinic": fm_s,
        "mean_fm_cross": fm_x,
        "drop_symmetric": (fm_w + fm_s) / 2 - fm_x,
    }


def main() -> None:
    print("=" * 70)
    print("Lens II §4.4 robustness — bare pool vs contextualised pool")
    print("=" * 70)

    _, index = load_precomputed(WEIRD_LABELS[0], EMB_BARE)
    core_idx, human_labels, domains = core_indices_and_labels(index)
    print(f"  {len(core_idx)} core terms across {len(domains)} human domains")

    bare = run_clustering_on_pool(EMB_BARE, "bare", human_labels, core_idx)
    ctx = run_clustering_on_pool(EMB_CTX, "ctx", human_labels, core_idx)

    print("\n" + "=" * 70)
    print("SIDE BY SIDE (Lens II §4.4)")
    print("=" * 70)
    print(f"{'quantity':<35s}  {'bare':>10s}  {'ctx':>10s}  {'Δ':>10s}")
    for key, lbl in [
        ("mean_fm_vs_human", "FM vs human (mean of 6)"),
        ("mean_fm_within_weird", "within-WEIRD FM̄"),
        ("mean_fm_within_sinic", "within-Sinic FM̄"),
        ("mean_fm_cross",        "cross FM̄"),
        ("drop_symmetric",       "ΔFM symmetric"),
    ]:
        b = bare[key]
        c = ctx[key]
        print(f"  {lbl:<33s}  {b:10.4f}  {c:10.4f}  {c - b:+10.4f}")

    report = {
        "meta": {
            "module": "lens_2_taxonomy/clustering_ctx_robustness.py",
            "thesis_section": "§4.4 robustness check — contextualised pool",
            "date": datetime.now().isoformat(timespec="seconds"),
            "weird_models": WEIRD_LABELS,
            "sinic_models": SINIC_LABELS,
            "k_human": K_HUMAN,
            "n_core": len(core_idx),
            "n_human_domains": len(domains),
            "human_domains": domains,
            "bare_pool": str(EMB_BARE.relative_to(REPO_ROOT)),
            "ctx_pool": str(EMB_CTX.relative_to(REPO_ROOT)),
        },
        "bare_pool": bare,
        "ctx_pool": ctx,
    }

    out_path = REPO_ROOT / "lens_2_taxonomy" / "results" / "clustering_ctx_robustness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ctx-lens2] Report → {out_path}")


if __name__ == "__main__":
    main()
