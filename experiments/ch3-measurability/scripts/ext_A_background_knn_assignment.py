#!/usr/bin/env python3
"""
Extension A — k-NN domain assignment for background terms.

For each of the 9045 bg terms, find the k=7 nearest neighbours in the
364-term core pool (cosine distance, BGE-EN-large bare embeddings as the
primary EN-side reference), majority-vote their domain, record vote
fraction as confidence.

Output:
  ext/A_bg_knn/background_assignments.csv
  ext/A_bg_knn/background_assignments.json   (full per-term record)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--primary-model", default="BGE-EN-large",
                        help="Model whose bare embeddings drive the k-NN")
    parser.add_argument("--k", type=int, default=7)
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    emb = REPO_ROOT / cfg["paths"]["embeddings"]
    core_idx = json.loads((emb / "index.json").read_text())
    bg_idx = json.loads((emb / "bg/index.json").read_text())

    core_vecs = np.load(emb / args.primary_model / "vecs_bare.npy").astype(np.float32)
    bg_vecs = np.load(emb / "bg" / args.primary_model / "vecs_bare.npy").astype(np.float32)
    core_domains = np.array([t["domain"] for t in core_idx])

    print(f"core={core_vecs.shape}, bg={bg_vecs.shape}, k={args.k}, primary={args.primary_model}")

    # Cosine distance, smaller = closer
    D = cdist(bg_vecs, core_vecs, metric="cosine")
    nearest_idx = np.argsort(D, axis=1)[:, : args.k]

    out_rows: list[dict] = []
    domain_counts: Counter = Counter()
    conf_list: list[float] = []
    for i, t in enumerate(bg_idx):
        nbrs = core_domains[nearest_idx[i]]
        votes = Counter(nbrs)
        top_domain, top_count = votes.most_common(1)[0]
        confidence = top_count / args.k
        domain_counts[top_domain] += 1
        conf_list.append(confidence)
        out_rows.append({
            "en": t["en"],
            "zh": t["zh"],
            "k_min": t["k_min"],
            "k_en": t["k_en"],
            "k_zh": t["k_zh"],
            "assigned_domain": top_domain,
            "confidence": round(confidence, 4),
            "vote_breakdown": dict(votes),
        })

    out_dir = RUN_DIR / "ext" / "A_bg_knn"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "background_assignments.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["en", "zh", "k_en", "k_zh", "k_min", "assigned_domain",
                    "confidence", "vote_breakdown"])
        for r in out_rows:
            w.writerow([r["en"], r["zh"], r["k_en"], r["k_zh"], r["k_min"],
                        r["assigned_domain"], r["confidence"],
                        json.dumps(r["vote_breakdown"], ensure_ascii=False)])
    with (out_dir / "background_assignments.json").open("w") as fh:
        json.dump({
            "meta": {
                "primary_model": args.primary_model,
                "k": args.k,
                "n_bg": len(bg_idx),
                "n_core": len(core_idx),
                "metric": "cosine",
                "domain_distribution": dict(domain_counts),
                "confidence_mean": round(float(np.mean(conf_list)), 4),
                "confidence_median": round(float(np.median(conf_list)), 4),
                "confidence_high_decile": round(float(np.quantile(conf_list, 0.9)), 4),
                "confidence_low_decile": round(float(np.quantile(conf_list, 0.1)), 4),
                "n_low_confidence": int(sum(1 for c in conf_list if c < 4/args.k)),
            },
            "assignments": out_rows,
        }, fh, indent=2, ensure_ascii=False)

    print("\nDomain distribution (bg → core):")
    for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:20s} {n:5d}")
    print(f"\nMean confidence: {np.mean(conf_list):.3f}  median: {np.median(conf_list):.3f}")
    print(f"Output: {(out_dir / 'background_assignments.csv').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
