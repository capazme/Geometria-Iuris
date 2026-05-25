#!/usr/bin/env python3
"""
Extension H — K saturation curve.

For each K-bucket in {1, 2, 3, 4-7, 8} (using bg attested embeddings),
compute ρ̄_cross attested on the 9 cross-tradition WEIRD×Sinic pairs using
the bg terms in that bucket as the pool, then compare with the core
(K∈{4-8}) baseline of 0.246.

Output: ext/H_K_saturation/k_saturation.json + .csv
        ext/H_K_saturation/k_saturation_plot_data.json

Requires bg attested encoded for at least 2 representative models per side.
With single-model encodings we still get a baseline curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
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
                        default=["BGE-EN-large", "BGE-M3-EN"],
                        help="EN-side models with bg attested (default 2 primary)")
    parser.add_argument("--zh-models", nargs="+",
                        default=["BGE-ZH-large", "BGE-M3-ZH"],
                        help="ZH-side models with bg attested (default 2 primary)")
    parser.add_argument("--buckets", nargs="+", type=str,
                        default=["1", "2", "3", "4-7", "8"],
                        help="K_min buckets to evaluate")
    parser.add_argument("--min-pool", type=int, default=50,
                        help="Min bg count per bucket to include")
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)
    emb = REPO_ROOT / cfg["paths"]["embeddings"]
    bg_idx = json.loads((emb / "bg/index.json").read_text())
    k_mins = np.array([t["k_min"] for t in bg_idx])

    def bucket_to_mask(b: str) -> np.ndarray:
        if "-" in b:
            lo, hi = (int(x) for x in b.split("-"))
            return (k_mins >= lo) & (k_mins <= hi)
        v = int(b)
        return k_mins == v

    pairs = list(product(args.en_models, args.zh_models))

    # Per-bucket: load bg attested for each label, subset, compute RDMs, RSA each pair.
    results: list[dict] = []
    for b in args.buckets:
        mask = bucket_to_mask(b)
        n = int(mask.sum())
        if n < args.min_pool:
            print(f"bucket K={b}: only {n} bg terms, skipping (min={args.min_pool})")
            continue
        print(f"\n--- bucket K={b}  (n={n}) ---")

        # Load attested for each label, subset to mask
        vecs: dict[str, np.ndarray] = {}
        for label in args.en_models + args.zh_models:
            att_path = emb / "bg" / label / "vecs_attested.npy"
            if not att_path.exists():
                print(f"  WARNING: {label} has no bg vecs_attested.npy — skipping")
                continue
            v = np.load(att_path).astype(np.float32)[mask]
            # Drop zero rows (no contexts in that lang)
            nonzero = np.linalg.norm(v, axis=1) > 1e-6
            v = v[nonzero]
            vecs[label] = v
            print(f"  {label}: {v.shape[0]}/{n} valid (non-zero) attested rows")

        # Intersect indices: only bg with non-zero attested in BOTH lang sides
        # Trick: keep track of which bg positions made it through for each label
        valid_per_label: dict[str, np.ndarray] = {}
        for label in args.en_models + args.zh_models:
            att_path = emb / "bg" / label / "vecs_attested.npy"
            if not att_path.exists():
                continue
            v = np.load(att_path).astype(np.float32)[mask]
            valid_per_label[label] = np.linalg.norm(v, axis=1) > 1e-6

        common = np.ones(n, dtype=bool)
        for v in valid_per_label.values():
            common &= v
        n_common = int(common.sum())
        print(f"  common nonzero (all labels): {n_common}/{n}")
        if n_common < 30:
            print(f"  too few common ({n_common}); skipping")
            continue

        # Re-load and subset to common
        vecs = {}
        for label in args.en_models + args.zh_models:
            att_path = emb / "bg" / label / "vecs_attested.npy"
            if not att_path.exists():
                continue
            v = np.load(att_path).astype(np.float32)[mask][common]
            vecs[label] = v

        rdms = {label: compute_rdm(v) for label, v in vecs.items()}
        rhos: list[float] = []
        for la, lb in pairs:
            if la not in rdms or lb not in rdms:
                continue
            rho = float(spearmanr(upper_tri(rdms[la]), upper_tri(rdms[lb])).statistic)
            rhos.append(rho)
            print(f"  {la} × {lb}  ρ={rho:+.4f}")
        results.append({
            "K_bucket": b,
            "n_bg": n,
            "n_common_nonzero": n_common,
            "rhos_cross_pairs": [round(r, 4) for r in rhos],
            "mean_rho_cross": round(float(np.mean(rhos)), 4) if rhos else None,
            "std_rho_cross": round(float(np.std(rhos)), 4) if rhos else None,
        })

    # Add the run #4 core baseline (K∈{4-8}, attested) for comparison
    core_rho = 0.246  # from run #4 attested headline
    out_dir = RUN_DIR / "ext" / "H_K_saturation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "k_saturation.json").open("w") as fh:
        json.dump({
            "meta": {
                "en_models": args.en_models,
                "zh_models": args.zh_models,
                "pairs_n": len(pairs),
                "core_reference_attested_cross_rho": core_rho,
                "core_K_range": "4-8",
            },
            "buckets": results,
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {(out_dir / 'k_saturation.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
