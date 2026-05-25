#!/usr/bin/env python3
"""
Extension F — Confidence-stratified Δρ_sym.

Using k-NN assignment from ext A (bg → domain via core), split bg into
high-confidence (top decile) and low-confidence (bottom decile) strata.
For each stratum, sample N bg terms (matched count) and inject them into
the 364 core; compute Δρ_sym. Compare with control (random bg).

Output: ext/F_confidence/confidence_strata.json
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


def compute_delta_sym(rdms: dict, en_models: list[str], zh_models: list[str]) -> dict:
    en_within = list(combinations(en_models, 2))
    zh_within = list(combinations(zh_models, 2))
    cross = list(product(en_models, zh_models))
    rho_W = float(np.mean([
        spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
        for a, b in en_within
    ])) if en_within else 0.0
    rho_S = float(np.mean([
        spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
        for a, b in zh_within
    ])) if zh_within else 0.0
    rho_C = float(np.mean([
        spearmanr(upper_tri(rdms[a]), upper_tri(rdms[b])).statistic
        for a, b in cross
    ]))
    return {
        "rho_W": rho_W, "rho_S": rho_S, "rho_cross": rho_C,
        "delta_sym": (rho_W + rho_S) / 2 - rho_C,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--en-models", nargs="+",
                        default=["BGE-EN-large", "BGE-M3-EN"])
    parser.add_argument("--zh-models", nargs="+",
                        default=["BGE-ZH-large", "BGE-M3-ZH"])
    parser.add_argument("--n-inject", type=int, default=91,  # ~25% of 364
                        help="Number of bg to inject per stratum")
    parser.add_argument("--n-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)
    emb = REPO_ROOT / cfg["paths"]["embeddings"]

    knn_path = RUN_DIR / "ext/A_bg_knn/background_assignments.json"
    if not knn_path.exists():
        raise SystemExit("Run ext_A_knn_bg.py first")
    knn = json.loads(knn_path.read_text())

    bg_idx = json.loads((emb / "bg/index.json").read_text())
    k_min = np.array([t["k_min"] for t in bg_idx])
    en_by_bg = {t["en"]: i for i, t in enumerate(bg_idx)}

    # Filter bg to K_min>=4 + non-zero attested for all 4 labels
    nonzero = np.ones(len(bg_idx), dtype=bool)
    bg_att: dict[str, np.ndarray] = {}
    for label in args.en_models + args.zh_models:
        v = np.load(emb / "bg" / label / "vecs_attested.npy").astype(np.float32)
        nonzero &= np.linalg.norm(v, axis=1) > 1e-6
        bg_att[label] = v
    eligible = (k_min >= 4) & nonzero
    eligible_idx_set = set(np.where(eligible)[0].tolist())

    # Sort bg by confidence
    bg_conf: list[tuple[int, float]] = []
    for a in knn["assignments"]:
        i = en_by_bg.get(a["en"])
        if i is None or i not in eligible_idx_set:
            continue
        bg_conf.append((i, a["confidence"]))
    bg_conf.sort(key=lambda x: x[1])
    n = len(bg_conf)
    print(f"eligible bg sorted by confidence: {n}")
    low_decile = [i for i, _ in bg_conf[: n // 10]]
    high_decile = [i for i, _ in bg_conf[-n // 10:]]
    print(f"low decile: {len(low_decile)} bg, high decile: {len(high_decile)}")

    # Load core attested
    core_att = {label: np.load(emb / label / "vecs_attested.npy").astype(np.float32)
                for label in args.en_models + args.zh_models}
    N_core = core_att[args.en_models[0]].shape[0]

    rng = np.random.default_rng(args.seed)

    def replicate_stratum(stratum_idx: list[int], n_inject: int, n_rep: int) -> dict:
        if len(stratum_idx) < n_inject:
            print(f"  WARNING: stratum has {len(stratum_idx)} bg, request {n_inject}; will resample with replacement")
            replace = True
        else:
            replace = False
        deltas: list[float] = []
        rWs: list[float] = []
        rSs: list[float] = []
        rCs: list[float] = []
        for _ in range(n_rep):
            chosen = rng.choice(np.array(stratum_idx), size=n_inject, replace=replace)
            rdms: dict[str, np.ndarray] = {}
            for label in args.en_models + args.zh_models:
                combined = np.vstack([core_att[label], bg_att[label][chosen]])
                rdms[label] = compute_rdm(combined)
            r = compute_delta_sym(rdms, args.en_models, args.zh_models)
            deltas.append(r["delta_sym"])
            rWs.append(r["rho_W"])
            rSs.append(r["rho_S"])
            rCs.append(r["rho_cross"])
        return {
            "n_inject": n_inject,
            "n_replicates": n_rep,
            "mean_delta_sym": round(float(np.mean(deltas)), 4),
            "std_delta_sym": round(float(np.std(deltas)), 4),
            "ci_low_delta_sym": round(float(np.percentile(deltas, 2.5)), 4),
            "ci_high_delta_sym": round(float(np.percentile(deltas, 97.5)), 4),
            "mean_rho_W": round(float(np.mean(rWs)), 4),
            "mean_rho_S": round(float(np.mean(rSs)), 4),
            "mean_rho_cross": round(float(np.mean(rCs)), 4),
        }

    print("\n--- baseline: core-only ---")
    rdms_core = {label: compute_rdm(core_att[label])
                 for label in args.en_models + args.zh_models}
    baseline = compute_delta_sym(rdms_core, args.en_models, args.zh_models)
    baseline = {k: round(v, 4) for k, v in baseline.items()}
    print(f"  baseline Δρ_sym={baseline['delta_sym']}")

    print("\n--- high-confidence stratum (top decile by k-NN confidence) ---")
    high = replicate_stratum(high_decile, args.n_inject, args.n_replicates)
    print(f"  Δρ_sym mean={high['mean_delta_sym']}  std={high['std_delta_sym']}")

    print("\n--- low-confidence stratum (bottom decile) ---")
    low = replicate_stratum(low_decile, args.n_inject, args.n_replicates)
    print(f"  Δρ_sym mean={low['mean_delta_sym']}  std={low['std_delta_sym']}")

    print("\n--- random control (eligible bg sampled uniformly) ---")
    all_eligible = sorted(eligible_idx_set)
    rand_ctrl = replicate_stratum(all_eligible, args.n_inject, args.n_replicates)
    print(f"  Δρ_sym mean={rand_ctrl['mean_delta_sym']}  std={rand_ctrl['std_delta_sym']}")

    out_dir = RUN_DIR / "ext" / "F_confidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "confidence_strata.json").open("w") as fh:
        json.dump({
            "meta": {
                "en_models": args.en_models,
                "zh_models": args.zh_models,
                "n_inject_per_stratum": args.n_inject,
                "n_replicates": args.n_replicates,
                "seed": args.seed,
                "n_bg_eligible": len(eligible_idx_set),
                "n_low_decile": len(low_decile),
                "n_high_decile": len(high_decile),
            },
            "baseline_core_only": baseline,
            "high_confidence_bg_injected": high,
            "low_confidence_bg_injected": low,
            "random_control_bg_injected": rand_ctrl,
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {(out_dir / 'confidence_strata.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
