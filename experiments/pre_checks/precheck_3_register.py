"""
Pre-check 3 — Register homogeneity / top-PC dependence (Gamma, 2026-04-11).

Tests whether the Lens I headline result Δρ ≈ 0.260 is driven by the top
principal components of the core-term embedding cloud. Top PCs of a
contextualised embedding space are known to encode frequency, anisotropy,
and stylistic register rather than fine-grained semantic content
(Mu & Viswanath 2018; Ethayarajh 2019). If Δρ collapses after removing
the top few PCs, the signal is register-confounded and D-C (confound
projection, with LEACE as the state-of-the-art method) is critical. If
Δρ is stable or strengthens, the signal lives in non-principal directions
and D-C is less urgent.

Procedure
---------
For each of the six models, load the precomputed 397 × dim core-term
matrix. For each k in {0, 1, 3, 5, 10}:
  1. Compute PCA on the 397 × dim matrix.
  2. Remove (project out) the top k PCs:
       v_clean = v − Σ_{i=1..k} (v · p_i) p_i
  3. Re-normalise.
  4. Build the RDM and compute within/cross Spearman ρ̄.

Report Δρ as a function of k, plus the fraction of variance removed.

Decision thresholds
-------------------
  ROBUST    : Δρ(k=10) within 0.05 of Δρ(k=0). D-C not critical.
  FRAGILE   : Δρ(k=3) drops by more than 0.10 relative to Δρ(k=0).
              D-C critical (LEACE needed).
  MODERATE  : between.

Output
------
JSON report in pre_checks/results/precheck_3_register.json
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

from shared.embeddings import load_precomputed  # noqa: E402
from shared.statistical import compute_rdm, upper_tri  # noqa: E402


WEIRD_LABELS = ["BGE-EN-large", "E5-large", "FreeLaw-EN"]
SINIC_LABELS = ["BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"]

K_VALUES = [0, 1, 3, 5, 10]


def remove_top_pcs(vectors: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """
    Remove the top-k principal components from a (N, dim) matrix.

    Returns the cleaned and re-normalised matrix, plus the fraction of
    total variance that was removed.
    """
    if k == 0:
        return vectors.copy(), 0.0

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # SVD is numerically stable and gives us singular values directly
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    total_var = float((s ** 2).sum())
    top_k_var = float((s[:k] ** 2).sum())
    frac_removed = top_k_var / max(total_var, 1e-12)

    # Project out the top k components
    top_pcs = vt[:k]  # shape (k, dim)
    projection = (vectors @ top_pcs.T) @ top_pcs  # shape (N, dim)
    cleaned = vectors - projection

    # Re-normalise (Levy-Goldberg-Dagan 2015 warning: nonuniform rescale)
    norms = np.linalg.norm(cleaned, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    cleaned_normed = (cleaned / norms).astype(np.float32)
    return cleaned_normed, frac_removed


def spearman_rdm(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    a = upper_tri(rdm_a)
    b = upper_tri(rdm_b)
    rho, _ = spearmanr(a, b)
    return float(rho)


def load_core_vectors(label: str) -> np.ndarray:
    """Load precomputed vectors and slice to core terms (tier == 'core')."""
    embeddings_dir = REPO_ROOT / "data" / "processed" / "embeddings"
    vecs, index = load_precomputed(label, embeddings_dir)
    core_idx = [i for i, e in enumerate(index) if e.get("tier") == "core"]
    return vecs[core_idx]


def run_for_k(
    weird_vecs: dict[str, np.ndarray],
    sinic_vecs: dict[str, np.ndarray],
    k: int,
) -> dict:
    """Run the Lens I ρ̄ aggregation for a given k."""
    weird_rdms = {}
    sinic_rdms = {}
    var_removed: dict[str, float] = {}

    for label, v in weird_vecs.items():
        cleaned, frac = remove_top_pcs(v, k)
        var_removed[label] = frac
        weird_rdms[label] = compute_rdm(cleaned)
    for label, v in sinic_vecs.items():
        cleaned, frac = remove_top_pcs(v, k)
        var_removed[label] = frac
        sinic_rdms[label] = compute_rdm(cleaned)

    # Within-WEIRD
    weird_pairs = []
    for a, b in combinations(sorted(weird_rdms.keys()), 2):
        rho = spearman_rdm(weird_rdms[a], weird_rdms[b])
        weird_pairs.append({"a": a, "b": b, "rho": rho})
    within_weird_mean = float(np.mean([p["rho"] for p in weird_pairs]))

    # Within-Sinic
    sinic_pairs = []
    for a, b in combinations(sorted(sinic_rdms.keys()), 2):
        rho = spearman_rdm(sinic_rdms[a], sinic_rdms[b])
        sinic_pairs.append({"a": a, "b": b, "rho": rho})
    within_sinic_mean = float(np.mean([p["rho"] for p in sinic_pairs]))

    # Cross-tradition
    cross_pairs = []
    for w in sorted(weird_rdms.keys()):
        for s in sorted(sinic_rdms.keys()):
            rho = spearman_rdm(weird_rdms[w], sinic_rdms[s])
            cross_pairs.append({"weird": w, "sinic": s, "rho": rho})
    cross_mean = float(np.mean([p["rho"] for p in cross_pairs]))

    within_mean = (within_weird_mean + within_sinic_mean) / 2.0
    delta_rho = within_mean - cross_mean

    return {
        "k": k,
        "within_weird_mean": within_weird_mean,
        "within_sinic_mean": within_sinic_mean,
        "within_mean": within_mean,
        "cross_mean": cross_mean,
        "delta_rho": delta_rho,
        "variance_removed_per_model": var_removed,
        "weird_pairs": weird_pairs,
        "sinic_pairs": sinic_pairs,
        "cross_pairs": cross_pairs,
    }


def main() -> None:
    print("[register] loading precomputed core vectors ...")
    weird_vecs = {label: load_core_vectors(label) for label in WEIRD_LABELS}
    sinic_vecs = {label: load_core_vectors(label) for label in SINIC_LABELS}

    for label, v in {**weird_vecs, **sinic_vecs}.items():
        print(f"  {label}: shape {v.shape}")

    print("[register] running Lens I pipeline at each k ...")
    per_k = []
    for k in K_VALUES:
        result = run_for_k(weird_vecs, sinic_vecs, k)
        per_k.append(result)
        print(
            f"  k={k:2d}  within-W={result['within_weird_mean']:.4f}  "
            f"within-S={result['within_sinic_mean']:.4f}  "
            f"cross={result['cross_mean']:.4f}  Δρ={result['delta_rho']:+.4f}"
        )

    # Decision based on Δρ trajectory
    delta_by_k = {r["k"]: r["delta_rho"] for r in per_k}
    delta_0 = delta_by_k[0]
    delta_3 = delta_by_k[3]
    delta_10 = delta_by_k[10]

    drop_at_3 = delta_0 - delta_3
    drop_at_10 = delta_0 - delta_10

    if abs(delta_10 - delta_0) < 0.05:
        status = "ROBUST"
        narrative = (
            "Lens I Δρ is stable after removing the top 10 principal "
            "components of the core-term cloud. The cross-tradition drop "
            "does not live in the dominant directions of the embedding "
            "space, so it is not driven by register-level or anisotropy "
            "confounds. D-C is a methodological preference, not a necessity. "
            "The existing pipeline stands with respect to this critique."
        )
    elif drop_at_3 > 0.10:
        status = "FRAGILE"
        narrative = (
            "Lens I Δρ drops substantially after removing only the top 3 "
            "principal components. The headline signal is dominated by the "
            "leading PCs of the embedding cloud, which are likely to encode "
            "register, frequency, and anisotropy rather than specifically "
            "legal semantic content. D-C is critical: a principled concept-"
            "erasure procedure (LEACE, Belrose et al. 2023) is needed before "
            "the Lens I result can be interpreted as a measurement of legal "
            "semantic structure rather than stylistic confound."
        )
    else:
        status = "MODERATE"
        narrative = (
            "Lens I Δρ is partially dependent on the top PCs of the core "
            "term cloud. The signal survives projection-based cleaning but "
            "is attenuated. D-C is justified and should be implemented, but "
            "the existing pipeline is not invalidated; it should be reported "
            "alongside a cleaned version as a sensitivity analysis."
        )

    report = {
        "pre_check": "3_register",
        "date": "2026-04-11",
        "k_values": K_VALUES,
        "weird_models": WEIRD_LABELS,
        "sinic_models": SINIC_LABELS,
        "per_k": per_k,
        "summary_delta_rho_by_k": delta_by_k,
        "drop_at_k3": drop_at_3,
        "drop_at_k10": drop_at_10,
        "thresholds": {
            "robust_max_diff_k10": 0.05,
            "fragile_min_drop_k3": 0.10,
        },
        "aggregate": {"status": status, "narrative": narrative},
    }

    out_path = REPO_ROOT / "pre_checks" / "results" / "precheck_3_register.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[register] Status: {status}")
    print(f"[register] Δρ(k=0) = {delta_0:+.4f}")
    print(f"[register] Δρ(k=3) = {delta_3:+.4f}  (drop = {drop_at_3:+.4f})")
    print(f"[register] Δρ(k=10) = {delta_10:+.4f}  (drop = {drop_at_10:+.4f})")
    print(f"[register] Report written to {out_path}")


if __name__ == "__main__":
    main()
