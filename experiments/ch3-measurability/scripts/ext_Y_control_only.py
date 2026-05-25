#!/usr/bin/env python3
"""
Extension Y — Cross-tradition ρ on control-only pool (sanity inverse).

Compute the 17 RSA pairs on the 100 control terms (everyday vocabulary).
Expectation: Δρ_sym ≈ 0 — the tradition signal should NOT exist on
non-legal lexicon.

Bare-only (control have no attested encoding).

Output: ext/Y_control_only/control_only_rsa.json
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
    args = parser.parse_args()
    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)
    emb = REPO_ROOT / cfg["paths"]["embeddings"]

    en_labels = [m["label"] for m in cfg["models_weird"]] + \
                [f"{m['label']}-EN" for m in cfg["models_bilingual"]]
    zh_labels = [m["label"] for m in cfg["models_sinic"]] + \
                [f"{m['label']}-ZH" for m in cfg["models_bilingual"]]

    # Load control bare for each model
    vecs: dict[str, np.ndarray] = {}
    for label in en_labels + zh_labels:
        vecs[label] = np.load(emb / "control_bare" / label / "vecs.npy").astype(np.float32)
    rdms = {label: compute_rdm(v) for label, v in vecs.items()}

    cross_pairs = list(product(en_labels[:3], zh_labels[:3]))  # 3 WEIRD × 3 Sinic = 9
    en_within = list(combinations(en_labels[:3], 2))  # 3 within-WEIRD
    zh_within = list(combinations(zh_labels[:3], 2))  # 3 within-Sinic
    bilingual = [(f"{m['label']}-EN", f"{m['label']}-ZH") for m in cfg["models_bilingual"]]

    def rsa_pair(la: str, lb: str) -> float:
        return float(spearmanr(upper_tri(rdms[la]), upper_tri(rdms[lb])).statistic)

    rho_W_pairs = [(la, lb, rsa_pair(la, lb)) for la, lb in en_within]
    rho_S_pairs = [(la, lb, rsa_pair(la, lb)) for la, lb in zh_within]
    rho_C_pairs = [(la, lb, rsa_pair(la, lb)) for la, lb in cross_pairs]
    rho_B_pairs = [(la, lb, rsa_pair(la, lb)) for la, lb in bilingual]

    rho_W = float(np.mean([r for _, _, r in rho_W_pairs]))
    rho_S = float(np.mean([r for _, _, r in rho_S_pairs]))
    rho_C = float(np.mean([r for _, _, r in rho_C_pairs]))
    rho_B = float(np.mean([r for _, _, r in rho_B_pairs]))

    delta_sym = (rho_W + rho_S) / 2 - rho_C
    print(f"Control-only pool (N=100):")
    print(f"  within-WEIRD ρ̄ = {rho_W:.4f}")
    print(f"  within-Sinic ρ̄ = {rho_S:.4f}")
    print(f"  cross ρ̄        = {rho_C:.4f}")
    print(f"  bilingual ρ̄   = {rho_B:.4f}")
    print(f"  Δρ_sym         = {delta_sym:.4f}")
    print()
    print("Comparison:")
    print(f"  Δρ_sym on 364 core bare (run #4):         0.165")
    print(f"  Δρ_sym on 364 core attested (run #4):     0.543")
    print(f"  Δρ_sym on 100 control bare (this test):  {delta_sym:.4f}")

    out_dir = RUN_DIR / "ext" / "Y_control_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "control_only_rsa.json").open("w") as fh:
        json.dump({
            "meta": {
                "encoding_variant": "bare",
                "n_control": 100,
                "en_labels": en_labels,
                "zh_labels": zh_labels,
            },
            "summary": {
                "mean_rho_within_weird": round(rho_W, 4),
                "mean_rho_within_sinic": round(rho_S, 4),
                "mean_rho_cross_tradition": round(rho_C, 4),
                "mean_rho_within_bilingual": round(rho_B, 4),
                "delta_rho_symmetric": round(delta_sym, 4),
            },
            "comparison": {
                "delta_sym_core_bare_run4": 0.165,
                "delta_sym_core_attested_run4": 0.543,
                "delta_sym_control_bare": round(delta_sym, 4),
            },
            "within_weird_pairs": [{"model_a": a, "model_b": b, "rho": round(r, 4)} for a, b, r in rho_W_pairs],
            "within_sinic_pairs": [{"model_a": a, "model_b": b, "rho": round(r, 4)} for a, b, r in rho_S_pairs],
            "cross_pairs": [{"model_a": a, "model_b": b, "rho": round(r, 4)} for a, b, r in rho_C_pairs],
            "bilingual_pairs": [{"model_a": a, "model_b": b, "rho": round(r, 4)} for a, b, r in rho_B_pairs],
        }, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {(out_dir / 'control_only_rsa.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
