#!/usr/bin/env python3
"""
Extension G — Automated false-friends detector across bg pool.

For each bg term with K_en≥2 AND K_zh≥2, compute cosine similarity between
the attested EN vector (BGE-EN-large) and the attested ZH vector
(BGE-ZH-large). Low similarity = cross-tradition divergence (a "false
friend": same lemma but different meaning constellation across traditions).

Also reports a baseline using a bilingual model (BGE-M3 EN ↔ ZH) where the
same encoder produces both vectors — if a bg looks divergent even within
BGE-M3, the divergence is in the legal usage, not in the encoder choice.

Output: ext/G_false_friends/false_friends.csv (sorted ascending by cos sim)
        ext/G_false_friends/false_friends.json (full record + meta)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--en-model", default="BGE-EN-large")
    parser.add_argument("--zh-model", default="BGE-ZH-large")
    parser.add_argument("--bilingual-en", default="BGE-M3-EN")
    parser.add_argument("--bilingual-zh", default="BGE-M3-ZH")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K most-divergent bg to print")
    parser.add_argument("--min-K", type=int, default=2,
                        help="Min K_en and K_zh for a bg to be considered")
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)
    emb = REPO_ROOT / cfg["paths"]["embeddings"]
    bg_idx = json.loads((emb / "bg/index.json").read_text())

    k_en = np.array([t["k_en"] for t in bg_idx])
    k_zh = np.array([t["k_zh"] for t in bg_idx])

    en_att = np.load(emb / "bg" / args.en_model / "vecs_attested.npy").astype(np.float32)
    zh_att = np.load(emb / "bg" / args.zh_model / "vecs_attested.npy").astype(np.float32)

    bi_en_path = emb / "bg" / args.bilingual_en / "vecs_attested.npy"
    bi_zh_path = emb / "bg" / args.bilingual_zh / "vecs_attested.npy"
    has_bilingual = bi_en_path.exists() and bi_zh_path.exists()
    if has_bilingual:
        bi_en = np.load(bi_en_path).astype(np.float32)
        bi_zh = np.load(bi_zh_path).astype(np.float32)
    else:
        bi_en = bi_zh = None

    eligible = (k_en >= args.min_K) & (k_zh >= args.min_K) \
        & (np.linalg.norm(en_att, axis=1) > 1e-6) \
        & (np.linalg.norm(zh_att, axis=1) > 1e-6)
    n_eligible = int(eligible.sum())
    print(f"eligible bg (K_en≥{args.min_K} AND K_zh≥{args.min_K}, both attested): {n_eligible}")

    rows: list[dict] = []
    for i in np.where(eligible)[0]:
        cos_cross = float(en_att[i] @ zh_att[i])
        cos_bi = None
        if has_bilingual and np.linalg.norm(bi_en[i]) > 1e-6 and np.linalg.norm(bi_zh[i]) > 1e-6:
            cos_bi = float(bi_en[i] @ bi_zh[i])
        rows.append({
            "en": bg_idx[i]["en"],
            "zh": bg_idx[i]["zh"],
            "k_en": int(k_en[i]),
            "k_zh": int(k_zh[i]),
            "k_min": int(min(k_en[i], k_zh[i])),
            f"cos_{args.en_model}_vs_{args.zh_model}": round(cos_cross, 4),
            f"cos_{args.bilingual_en}_vs_{args.bilingual_zh}": round(cos_bi, 4) if cos_bi is not None else None,
        })
    rows.sort(key=lambda r: r[f"cos_{args.en_model}_vs_{args.zh_model}"])

    out_dir = RUN_DIR / "ext" / "G_false_friends"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "false_friends.csv").open("w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (out_dir / "false_friends.json").open("w") as fh:
        json.dump({
            "meta": {
                "en_model": args.en_model, "zh_model": args.zh_model,
                "bilingual_pair": [args.bilingual_en, args.bilingual_zh] if has_bilingual else None,
                "n_eligible_bg": n_eligible,
                "min_K": args.min_K,
            },
            "rows": rows,
        }, fh, indent=2, ensure_ascii=False)

    cross_key = f"cos_{args.en_model}_vs_{args.zh_model}"
    print(f"\nTop-{args.top_k} most-divergent bg (lowest cosine cross-tradition):")
    print(f"{'en':30s} {'zh':14s} {'K_en':>4s} {'K_zh':>4s} {'cross':>7s}  bilingual")
    for r in rows[: args.top_k]:
        bi = r[f"cos_{args.bilingual_en}_vs_{args.bilingual_zh}"]
        bi_s = f"{bi:+.3f}" if bi is not None else " n/a"
        print(f"{r['en'][:30]:30s} {r['zh'][:14]:14s} "
              f"{r['k_en']:>4d} {r['k_zh']:>4d} {r[cross_key]:>7.3f}  {bi_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
