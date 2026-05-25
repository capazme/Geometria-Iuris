#!/usr/bin/env python3
"""
Step 3-4 — Lens I (§3.1) for run #4 post-BLP.

Operates on the run-#4 pool (364 core terms, 7 domains, no background/control
tiers — these were dropped in the post-BLP curation). Implements the three
sub-sections that the run-#4 pool supports:

  §3.1.1  intra-domain vs inter-domain distances (Mann-Whitney U + r) — per
          WEIRD model. Legal-vs-control is omitted: no control tier in run #4.
  §3.1.2  domain topology K×K matrix (K=7) — per model
  §3.1.3  RSA on 17 pre-registered model pairs:
            9 cross-tradition + 3 within-WEIRD + 3 within-Sinic + 2 within-
            bilingual. Each pair: Spearman ρ, Mantel test (B=10000),
            block bootstrap CI (B=10000). Holm correction across the 17 p.

§3.1.4 (categorical probe) is delegated to scripts/03b_categorical.py
(separately because it has its own pre-registration record).

Usage
-----
    python3 03_structure_experiment.py --variant bare        # bare encodings
    python3 03_structure_experiment.py --variant attested    # attested encodings
    python3 03_structure_experiment.py --variant bare --n-perm 1000 --n-boot 1000   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    compute_rdm,
    holm_correction,
    mannwhitney_with_r,
    rsa,
    upper_tri,
)


def _rsa_pair_worker(args: tuple) -> dict:
    """Worker for ProcessPoolExecutor — re-imports shared.statistical and
    loads its own RDMs from disk to avoid pickling overhead. Deterministic
    because seed is fixed per call (independent of worker scheduling)."""
    la, lb, group, rdm_dir, save_dir, n_perm, n_boot, seed = args
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from _lib import rsa as _rsa
    rdm_a = np.load(_P(rdm_dir) / f"{la}.npz")["rdm"]
    rdm_b = np.load(_P(rdm_dir) / f"{lb}.npz")["rdm"]
    result = _rsa(rdm_a, rdm_b, n_perm=n_perm, n_boot=n_boot, seed=seed)
    np.savez_compressed(
        _P(save_dir) / f"{la}_x_{lb}.npz",
        null=result.null_distribution,
        bootstrap=result.ci.distribution,
    )
    return {
        "model_a": la,
        "model_b": lb,
        "group": group,
        "rho": round(result.rho, 4),
        "r_squared": round(result.r_squared, 4),
        "p_value": float(result.p_value),
        "ci_low": round(result.ci.low, 4),
        "ci_high": round(result.ci.high, 4),
    }


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_index(emb_dir: Path) -> list[dict]:
    with (emb_dir / "index.json").open() as fh:
        return json.load(fh)


def load_vecs(emb_dir: Path, label: str, variant: str) -> np.ndarray:
    fname = "vecs_bare.npy" if variant == "bare" else "vecs_attested.npy"
    return np.load(emb_dir / label / fname).astype(np.float32)


def intra_inter_split(rdm: np.ndarray, domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = len(rdm)
    rows, cols = np.triu_indices(n, k=1)
    dom = np.array(domains)
    same = dom[rows] == dom[cols]
    tri = rdm[rows, cols]
    return tri[same], tri[~same]


def domain_topology(rdm: np.ndarray, domains: list[str]) -> tuple[list[str], np.ndarray]:
    labels = sorted(set(domains))
    dom = np.array(domains)
    k = len(labels)
    topo = np.zeros((k, k), dtype=np.float32)
    for i, d1 in enumerate(labels):
        idx1 = np.where(dom == d1)[0]
        for j, d2 in enumerate(labels):
            idx2 = np.where(dom == d2)[0]
            sub = rdm[np.ix_(idx1, idx2)]
            if i == j:
                topo[i, j] = float(upper_tri(sub).mean()) if len(idx1) > 1 else 0.0
            else:
                topo[i, j] = float(sub.mean())
    return labels, topo


def mw_to_dict(mw) -> dict:
    return {
        "statistic": round(mw.statistic, 2),
        "p_value": float(mw.p_value),
        "effect_r": round(mw.effect_r, 4),
        "n_x": int(mw.n_x),
        "n_y": int(mw.n_y),
        "median_x": round(mw.median_x, 4),
        "median_y": round(mw.median_y, 4),
    }


def rsa_to_dict(la: str, lb: str, result) -> dict:
    return {
        "model_a": la,
        "model_b": lb,
        "rho": round(result.rho, 4),
        "r_squared": round(result.r_squared, 4),
        "p_value": float(result.p_value),
        "ci_low": round(result.ci.low, 4),
        "ci_high": round(result.ci.high, 4),
    }


def section_311(rdms: dict[str, np.ndarray], weird: list[str], domains: list[str]) -> dict:
    """Intra-vs-inter Mann-Whitney for the 3 WEIRD models."""
    per_model: dict[str, dict] = {}
    for label in weird:
        intra, inter = intra_inter_split(rdms[label], domains)
        mw = mannwhitney_with_r(intra, inter, alternative="less")
        per_model[label] = mw_to_dict(mw)
        print(f"  §3.1.1 {label}: intra med={mw.median_x:.3f}  "
              f"inter med={mw.median_y:.3f}  r={mw.effect_r:+.3f}  p={mw.p_value:.2e}")
    return {"per_model": per_model}


def section_312(rdms: dict[str, np.ndarray], domains: list[str],
                save_dir: Path) -> dict:
    """K×K topology per model."""
    per_model: dict[str, dict] = {}
    for label, rdm in rdms.items():
        labels_d, topo = domain_topology(rdm, domains)
        per_model[label] = {"domains": labels_d, "matrix": topo.tolist()}
        np.savez_compressed(save_dir / f"{label}.npz", topology=topo, domains=np.array(labels_d))
    return {"per_model": per_model}


def section_313(rdms: dict[str, np.ndarray],
                pairs_cross: list[list[str]],
                pairs_within_weird: list[list[str]],
                pairs_within_sinic: list[list[str]],
                pairs_within_bilingual: list[list[str]],
                n_perm: int, n_boot: int, seed: int,
                save_dir: Path,
                rdm_dir: Path,
                n_workers: int = 0) -> dict:
    """17 RSA pairs with Mantel + bootstrap; Holm correction across all.
    Parallelized: each pair runs in its own worker process (deterministic
    because seed is fixed per call). n_workers=0 → max(1, cpu_count - 1)."""
    all_pairs_buckets = [
        ("within_weird",       pairs_within_weird),
        ("within_sinic",       pairs_within_sinic),
        ("cross_tradition",    pairs_cross),
        ("within_bilingual",   pairs_within_bilingual),
    ]
    pair_tuples: list[tuple] = []
    pair_to_group: dict[tuple, str] = {}
    for group, pairs in all_pairs_buckets:
        for la, lb in pairs:
            args = (la, lb, group, str(rdm_dir), str(save_dir),
                    int(n_perm), int(n_boot), int(seed))
            pair_tuples.append(args)
            pair_to_group[(la, lb)] = group

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = min(n_workers, len(pair_tuples))
    print(f"  §3.1.3 parallel: {len(pair_tuples)} pairs across {n_workers} workers")

    t0 = time.perf_counter()
    all_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_rsa_pair_worker, args): (args[0], args[1])
                   for args in pair_tuples}
        for fut in as_completed(futures):
            la, lb = futures[fut]
            d = fut.result()
            all_results.append(d)
            dt = time.perf_counter() - t0
            print(f"    [{dt:5.0f}s] {la} × {lb}  ρ={d['rho']:+.3f}  "
                  f"r²={d['r_squared']:.3f}  "
                  f"CI=[{d['ci_low']:.3f},{d['ci_high']:.3f}]  "
                  f"p={d['p_value']:.4f}")
    print(f"  §3.1.3 parallel done in {time.perf_counter() - t0:.0f}s")

    # Re-bucket in canonical order matching cfg pairs
    def canonical_order(group: str, pairs: list[list[str]]) -> list[dict]:
        out: list[dict] = []
        for la, lb in pairs:
            for r in all_results:
                if r["model_a"] == la and r["model_b"] == lb and r["group"] == group:
                    out.append(r)
                    break
        return out

    bucket_results: dict[str, list[dict]] = {
        "within_weird":       canonical_order("within_weird",     pairs_within_weird),
        "within_sinic":       canonical_order("within_sinic",     pairs_within_sinic),
        "cross_tradition":    canonical_order("cross_tradition",  pairs_cross),
        "within_bilingual":   canonical_order("within_bilingual", pairs_within_bilingual),
    }
    all_results_ordered = (bucket_results["within_weird"]
                           + bucket_results["within_sinic"]
                           + bucket_results["cross_tradition"]
                           + bucket_results["within_bilingual"])

    raw_ps = [r["p_value"] for r in all_results_ordered]
    adj_ps = holm_correction(raw_ps)
    for r, p_adj in zip(all_results_ordered, adj_ps):
        r["p_holm"] = round(float(p_adj), 6)
    all_results = all_results_ordered

    rho_w = float(np.mean([r["rho"] for r in bucket_results["within_weird"]]))
    rho_s = float(np.mean([r["rho"] for r in bucket_results["within_sinic"]]))
    rho_c = float(np.mean([r["rho"] for r in bucket_results["cross_tradition"]]))
    rho_b = (float(np.mean([r["rho"] for r in bucket_results["within_bilingual"]]))
             if bucket_results["within_bilingual"] else None)

    summary = {
        "mean_rho_within_weird":     round(rho_w, 4),
        "mean_rho_within_sinic":     round(rho_s, 4),
        "mean_rho_cross_tradition":  round(rho_c, 4),
        "delta_rho_symmetric":       round((rho_w + rho_s) / 2 - rho_c, 4),
    }
    if rho_b is not None:
        summary["mean_rho_within_bilingual"] = round(rho_b, 4)
    print(f"\n  Summary: ρ̄_W={rho_w:.3f}  ρ̄_S={rho_s:.3f}  "
          f"ρ̄_cross={rho_c:.3f}  Δρ_sym={summary['delta_rho_symmetric']:.3f}"
          + (f"  ρ̄_bilingual={rho_b:.3f}" if rho_b is not None else ""))

    return {**bucket_results, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--variant", choices=["bare", "attested"], required=True)
    parser.add_argument("--n-perm", type=int, default=None,
                        help="Override Mantel B (default: config.n_perm_mantel)")
    parser.add_argument("--n-boot", type=int, default=None,
                        help="Override bootstrap B (default: config.n_boot)")
    parser.add_argument("--no-rsa", action="store_true",
                        help="Skip §3.1.3 (RSA only); useful for smoke")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker count for parallel RSA; 0 = cpu_count-1")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    emb_dir = REPO_ROOT / cfg["paths"]["embeddings"]
    out_root = REPO_ROOT / cfg["paths"][f"lens1_{args.variant}"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rdms").mkdir(exist_ok=True)
    (out_root / "topology").mkdir(exist_ok=True)
    (out_root / "distributions").mkdir(exist_ok=True)

    n_perm = args.n_perm if args.n_perm is not None else int(cfg["n_perm_mantel"])
    n_boot = args.n_boot if args.n_boot is not None else int(cfg["n_boot"])
    seed = int(cfg["seed"])

    # All labels we need RDMs for
    weird = [m["label"] for m in cfg["models_weird"]]
    sinic = [m["label"] for m in cfg["models_sinic"]]
    bilingual_en = [f"{m['label']}-EN" for m in cfg["models_bilingual"]]
    bilingual_zh = [f"{m['label']}-ZH" for m in cfg["models_bilingual"]]
    all_labels = weird + sinic + bilingual_en + bilingual_zh

    print("=" * 60)
    print(f"Lens I — run #4 post-BLP — variant={args.variant}")
    print(f"  WEIRD : {weird}")
    print(f"  Sinic : {sinic}")
    print(f"  Bilingual: {bilingual_en} | {bilingual_zh}")
    print(f"  n_perm={n_perm}  n_boot={n_boot}  seed={seed}")
    print("=" * 60)

    # Load index + per-model embeddings
    index = load_index(emb_dir)
    if len(index) != 364:
        raise AssertionError(f"index has {len(index)} entries (expected 364)")
    domains = [t["domain"] for t in index]

    # Build RDMs (cosine distance) for every label
    rdms: dict[str, np.ndarray] = {}
    for label in all_labels:
        vecs = load_vecs(emb_dir, label, args.variant)
        if vecs.shape[0] != 364:
            raise AssertionError(f"{label} has {vecs.shape[0]} rows (expected 364)")
        # Drop zero-norm rows? With L2-norm embeddings they should never be zero;
        # but attested with all-zero contexts would be — abort on detection.
        zero_rows = int((np.linalg.norm(vecs, axis=1) < 1e-6).sum())
        if zero_rows:
            raise AssertionError(f"{label} has {zero_rows} zero-norm rows — encoding bug")
        rdms[label] = compute_rdm(vecs)
        np.savez_compressed(out_root / "rdms" / f"{label}.npz", rdm=rdms[label])
        print(f"  RDM[{label}] shape={rdms[label].shape}")

    out: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "variant": args.variant,
            "n_perm": n_perm,
            "n_boot": n_boot,
            "seed": seed,
            "n_terms": 364,
            "domains": sorted(set(domains)),
            "models_weird": weird,
            "models_sinic": sinic,
            "models_bilingual_en": bilingual_en,
            "models_bilingual_zh": bilingual_zh,
        }
    }

    t0 = time.perf_counter()

    print("\n[§3.1.1] intra-vs-inter")
    out["section_311"] = section_311(rdms, weird, domains)

    print("\n[§3.1.2] domain topology")
    out["section_312"] = section_312(rdms, domains, out_root / "topology")

    if not args.no_rsa:
        print("\n[§3.1.3] RSA")
        out["section_313"] = section_313(
            rdms,
            cfg["pairs_cross"], cfg["pairs_within_weird"],
            cfg["pairs_within_sinic"], cfg["pairs_within_bilingual"],
            n_perm=n_perm, n_boot=n_boot, seed=seed,
            save_dir=out_root / "distributions",
            rdm_dir=out_root / "rdms",
            n_workers=args.workers,
        )

    elapsed = time.perf_counter() - t0
    out["meta"]["elapsed_seconds"] = round(elapsed, 1)

    out_path = out_root / "experiment_1_results.json"
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nDone in {elapsed:.0f}s -> {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
