#!/usr/bin/env python3
"""
Extension X — Δρ_sym vs %control robustness curve (dual of D).

Inject control terms (everyday vocabulary, NOT legal) into the 364-term
core pool at increasing percentages. Recompute Δρ_sym. Control terms
have no attested encoding (no HK Cap. attestation), so this experiment
operates on **bare** embeddings only.

Baseline Δρ_sym bare on 364 core = 0.165 (run #4 headline).

Expectation: Δρ_sym → ~0 as %control grows, demonstrating discriminative
validity (the tradition signal vanishes when the pool is not legal).

Output: ext/X_control_robustness/control_robustness_curve.json
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
                        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.27])
    parser.add_argument("--n-replicates", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)
    emb = REPO_ROOT / cfg["paths"]["embeddings"]

    # BARE encodings only (control have no attested)
    core_vecs: dict[str, np.ndarray] = {}
    ctrl_vecs: dict[str, np.ndarray] = {}
    for label in args.en_models + args.zh_models:
        core_vecs[label] = np.load(emb / label / "vecs_bare.npy").astype(np.float32)
        ctrl_vecs[label] = np.load(emb / "control_bare" / label / "vecs.npy").astype(np.float32)
    N_core = core_vecs[args.en_models[0]].shape[0]
    N_ctrl = ctrl_vecs[args.en_models[0]].shape[0]
    print(f"core size: {N_core},  control size: {N_ctrl}")

    rng = np.random.default_rng(args.seed)
    cross_pairs = list(product(args.en_models, args.zh_models))
    en_within = list(combinations(args.en_models, 2))
    zh_within = list(combinations(args.zh_models, 2))

    results: list[dict] = []
    for p in args.pcts:
        n_ctrl_use = int(round(N_core * p))
        n_core_use = N_core - n_ctrl_use
        if n_ctrl_use > N_ctrl:
            print(f"p={p:.2f} requires n_ctrl={n_ctrl_use} > available {N_ctrl}; skipping")
            continue
        print(f"\n=== p={p:.0%}  (n_core={n_core_use}, n_ctrl={n_ctrl_use}) ===")
        deltas: list[float] = []
        rWs: list[float] = []
        rSs: list[float] = []
        rCs: list[float] = []
        for r in range(args.n_replicates):
            core_sel = rng.choice(N_core, size=n_core_use, replace=False)
            ctrl_sel = (rng.choice(N_ctrl, size=n_ctrl_use, replace=False)
                        if n_ctrl_use > 0 else np.array([], dtype=int))
            rdms: dict[str, np.ndarray] = {}
            for label in args.en_models + args.zh_models:
                core_part = core_vecs[label][core_sel]
                ctrl_part = (ctrl_vecs[label][ctrl_sel] if n_ctrl_use > 0
                             else np.zeros((0, core_part.shape[1]), dtype=np.float32))
                combined = np.vstack([core_part, ctrl_part])
                rdms[label] = compute_rdm(combined)
            rho_W = float(np.mean([
                spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
                for a, b in en_within
            ]))
            rho_S = float(np.mean([
                spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
                for a, b in zh_within
            ]))
            rho_C = float(np.mean([
                spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
                for a, b in cross_pairs
            ]))
            d_sym = (rho_W + rho_S) / 2 - rho_C
            deltas.append(d_sym)
            rWs.append(rho_W); rSs.append(rho_S); rCs.append(rho_C)
        results.append({
            "p_control": p,
            "n_core": n_core_use,
            "n_control": n_ctrl_use,
            "n_replicates": args.n_replicates,
            "mean_rho_W": round(float(np.mean(rWs)), 4),
            "mean_rho_S": round(float(np.mean(rSs)), 4),
            "mean_rho_cross": round(float(np.mean(rCs)), 4),
            "mean_delta_sym": round(float(np.mean(deltas)), 4),
            "std_delta_sym": round(float(np.std(deltas)), 4),
            "ci_low_delta_sym": round(float(np.percentile(deltas, 2.5)), 4),
            "ci_high_delta_sym": round(float(np.percentile(deltas, 97.5)), 4),
        })
        print(f"  Δρ_sym mean={np.mean(deltas):.4f}  std={np.std(deltas):.4f}  "
              f"CI95=[{np.percentile(deltas,2.5):.4f}, {np.percentile(deltas,97.5):.4f}]")

    out_dir = RUN_DIR / "ext" / "X_control_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "control_robustness_curve.json").open("w") as fh:
        json.dump({
            "meta": {
                "encoding_variant": "bare",
                "en_models": args.en_models,
                "zh_models": args.zh_models,
                "n_replicates": args.n_replicates,
                "seed": args.seed,
                "core_pool_size": N_core,
                "control_pool_size": N_ctrl,
                "note": "Dual of extension D: control terms have no attested encoding (no HK Cap. attestation), so this curve operates on BARE embeddings. Baseline (p=0%) recovers Δρ_sym bare on 364 core.",
            },
            "results": results,
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {(out_dir / 'control_robustness_curve.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
