#!/usr/bin/env python3
"""
Extension D — Δρ_sym vs %bg robustness curve.

For each mix level p ∈ {0%, 10%, 25%, 50%, 75%} of bg injection:
  - Repeat N_replicates (default 10) times:
    - Sample a 364-term pool: round(364*(1-p)) core randomly + round(364*p) bg
      (bg must have K_min≥4 to be attested-comparable to core).
    - Compute RDMs for 2 EN-side + 2 ZH-side models (BGE primary slots).
    - Compute the 4 cross-tradition pairs RSA (Spearman ρ), within-W (1 pair),
      within-S (1 pair). With B=1000 Mantel + B=1000 bootstrap (smaller than
      headline B=10000 for tractability across N_pool replicates).
  - Aggregate: mean ± std Δρ_sym across replicates.

Output: ext/D_robustness/robustness_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import compute_rdm, upper_tri  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--en-models", nargs="+",
                        default=["BGE-EN-large", "BGE-M3-EN"])
    parser.add_argument("--zh-models", nargs="+",
                        default=["BGE-ZH-large", "BGE-M3-ZH"])
    parser.add_argument("--pcts", nargs="+", type=float,
                        default=[0.0, 0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)
    emb = REPO_ROOT / cfg["paths"]["embeddings"]

    # Load core attested for the 4 models
    core_att: dict[str, np.ndarray] = {}
    for label in args.en_models + args.zh_models:
        core_att[label] = np.load(emb / label / "vecs_attested.npy").astype(np.float32)
    N_core = core_att[args.en_models[0]].shape[0]
    print(f"core size: {N_core}")

    # Load bg attested, mask to K_min >= 4 + non-zero in all 4 labels
    bg_idx = json.loads((emb / "bg/index.json").read_text())
    k_min = np.array([t["k_min"] for t in bg_idx])
    nonzero_mask = np.ones(len(bg_idx), dtype=bool)
    bg_att: dict[str, np.ndarray] = {}
    for label in args.en_models + args.zh_models:
        att_path = emb / "bg" / label / "vecs_attested.npy"
        if not att_path.exists():
            raise FileNotFoundError(f"{label} bg attested not found at {att_path}")
        v = np.load(att_path).astype(np.float32)
        nonzero_mask &= np.linalg.norm(v, axis=1) > 1e-6
        bg_att[label] = v

    bg_eligible = (k_min >= 4) & nonzero_mask
    n_bg_eligible = int(bg_eligible.sum())
    print(f"bg eligible (K_min≥4 + non-zero attested all 4 labels): {n_bg_eligible}")
    bg_eligible_idx = np.where(bg_eligible)[0]

    rng = np.random.default_rng(args.seed)
    cross_pairs = list(product(args.en_models, args.zh_models))
    en_within = list(combinations(args.en_models, 2))
    zh_within = list(combinations(args.zh_models, 2))

    results_per_pct: list[dict] = []
    for p in args.pcts:
        n_bg = int(round(N_core * p))
        n_core = N_core - n_bg
        if n_bg > n_bg_eligible:
            print(f"p={p:.2f} requires n_bg={n_bg} > eligible {n_bg_eligible}; skipping")
            continue
        print(f"\n=== p={p:.0%}  (n_core={n_core}, n_bg={n_bg}) ===")
        replicates: list[dict] = []
        for r in range(args.n_replicates):
            core_sel = rng.choice(N_core, size=n_core, replace=False)
            bg_sel = rng.choice(bg_eligible_idx, size=n_bg, replace=False) if n_bg > 0 else np.array([], dtype=int)
            # Build per-label combined matrix
            rdms: dict[str, np.ndarray] = {}
            for label in args.en_models + args.zh_models:
                core_vecs = core_att[label][core_sel]
                bg_vecs = bg_att[label][bg_sel] if n_bg > 0 else np.zeros((0, core_vecs.shape[1]), dtype=np.float32)
                combined = np.vstack([core_vecs, bg_vecs])
                rdms[label] = compute_rdm(combined)
            rho_W = float(np.mean([
                spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
                for a, b in en_within
            ])) if en_within else None
            rho_S = float(np.mean([
                spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
                for a, b in zh_within
            ])) if zh_within else None
            rho_C = float(np.mean([
                spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
                for a, b in cross_pairs
            ]))
            d_sym = (rho_W + rho_S) / 2 - rho_C if rho_W is not None and rho_S is not None else None
            replicates.append({
                "rep": r, "rho_W": round(rho_W, 4), "rho_S": round(rho_S, 4),
                "rho_cross": round(rho_C, 4), "delta_sym": round(d_sym, 4),
            })
        # Aggregate
        dsyms = [rp["delta_sym"] for rp in replicates]
        rWs = [rp["rho_W"] for rp in replicates]
        rSs = [rp["rho_S"] for rp in replicates]
        rCs = [rp["rho_cross"] for rp in replicates]
        results_per_pct.append({
            "p_bg": p,
            "n_core": n_core,
            "n_bg": n_bg,
            "n_replicates": args.n_replicates,
            "mean_rho_W": round(float(np.mean(rWs)), 4),
            "mean_rho_S": round(float(np.mean(rSs)), 4),
            "mean_rho_cross": round(float(np.mean(rCs)), 4),
            "mean_delta_sym": round(float(np.mean(dsyms)), 4),
            "std_delta_sym": round(float(np.std(dsyms)), 4),
            "ci_low_delta_sym": round(float(np.percentile(dsyms, 2.5)), 4),
            "ci_high_delta_sym": round(float(np.percentile(dsyms, 97.5)), 4),
            "replicates": replicates,
        })
        print(f"  Δρ_sym mean={np.mean(dsyms):.4f}  std={np.std(dsyms):.4f}  "
              f"CI95=[{np.percentile(dsyms,2.5):.4f}, {np.percentile(dsyms,97.5):.4f}]")

    out_dir = RUN_DIR / "ext" / "D_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "robustness_curve.json").open("w") as fh:
        json.dump({
            "meta": {
                "en_models": args.en_models,
                "zh_models": args.zh_models,
                "n_replicates": args.n_replicates,
                "seed": args.seed,
                "n_core_total": N_core,
                "n_bg_eligible": n_bg_eligible,
                "core_K_range": "4-8",
                "bg_K_filter": "K_min>=4",
                "metric": "cosine RDM, Spearman ρ on upper triangles (NO Mantel/bootstrap here)",
                "notes": "Permutation/CI is across pool-replicates, not within a single pool.",
            },
            "results": results_per_pct,
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {(out_dir / 'robustness_curve.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
