"""
k-NN domain labelling for the strict-pool background candidates.

For each term in `data/processed/strict_pool_snapshot.json` that lives in
the background tier, finds the k=7 nearest neighbours among the current
350 core terms (using BGE-EN-large bare vectors, post-D2 re-encode on
cleaned forms) and assigns the predicted domain by majority vote.

Output: `data/review/strict_pool_labelled.json`, one record per
strict-pool background candidate, with predicted_domain + k-NN vote
distribution + attestation counts. This file feeds the per-domain LLM
curation in D4 step 2.

D4 step 1 of `experiments/trace_firthian_pivot.md`.

Usage
-----
    python data/label_strict_pool.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS = REPO_ROOT / "data" / "processed" / "legal_terms.json"
SNAPSHOT = REPO_ROOT / "data" / "processed" / "strict_pool_snapshot.json"
TERM_CONTEXTS = REPO_ROOT / "data" / "processed" / "elegislation" / "term_contexts.jsonl"
BGE_EN_VECTORS = REPO_ROOT / "data" / "processed" / "embeddings" / "BGE-EN-large" / "vectors.npy"
INDEX_JSON = REPO_ROOT / "data" / "processed" / "embeddings" / "index.json"
OUT_PATH = REPO_ROOT / "data" / "review" / "strict_pool_labelled.json"

K = 7  # k-NN majority vote


def main() -> int:
    print("Loading legal_terms.json ...")
    lt = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))["terms"]
    index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))

    print("Loading BGE-EN-large bare vectors (post-D2 cleanup) ...")
    vecs = np.load(BGE_EN_VECTORS).astype(np.float32)
    # Already L2-normalized at write time, but normalize defensively
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-12, None)
    print(f"  vectors shape: {vecs.shape}")

    print("Loading strict pool snapshot ...")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    strict_bg = snap["strict_background_indices"]
    strict_core = snap["strict_core_indices"]
    print(f"  strict-pool background: {len(strict_bg)}")
    print(f"  strict-pool core:       {len(strict_core)}")

    # Identify current 350 core indices (from index.json — which is master-aligned post-D6)
    core_indices = [i for i, t in enumerate(index) if t.get("tier") == "core" and t.get("domain")]
    print(f"  current core (from index.json): {len(core_indices)}")

    # Collect domain labels for the 350 core
    core_domains = [index[i]["domain"] for i in core_indices]

    # Build core-only vector matrix and KDTree-style NN search
    print("Computing k-NN for each strict-bg candidate against 350 core ...")
    core_vecs = vecs[core_indices]  # (350, 1024)

    # Cosine similarity = dot product on normalized vectors
    # For each candidate, similarity to all 350 core, take top-K
    print("Loading attested context counts ...")
    contexts = {}
    with TERM_CONTEXTS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            contexts[rec["term_idx"]] = rec

    labelled = []
    for j, ti in enumerate(strict_bg):
        candidate_vec = vecs[ti]  # (1024,)
        sims = core_vecs @ candidate_vec  # (350,)
        top_k_idx = np.argpartition(-sims, K)[:K]
        # Sort top-K by sim descending
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        top_k_domains = [core_domains[idx] for idx in top_k_idx]
        top_k_sims = [float(sims[idx]) for idx in top_k_idx]

        votes = Counter(top_k_domains)
        predicted = votes.most_common(1)[0][0]
        confidence = votes[predicted] / K
        margin = (votes.most_common(2)[0][1] - votes.most_common(2)[1][1]) / K if len(votes) > 1 else 1.0

        rec = contexts.get(ti)
        n_en = len(rec.get("en_contexts", [])) if rec else 0
        n_zh = len(rec.get("zh_contexts", [])) if rec else 0

        labelled.append({
            "term_idx": ti,
            "en": index[ti]["en"],
            "zh_canonical": index[ti]["zh_canonical"],
            "en_clean": index[ti]["en_clean"],
            "zh_clean": index[ti]["zh_clean"],
            "predicted_domain": predicted,
            "vote_distribution": dict(votes),
            "confidence": round(confidence, 3),
            "margin": round(margin, 3),
            "top_neighbour_sim": round(top_k_sims[0], 4),
            "k_attested_en": n_en,
            "k_attested_zh": n_zh,
        })

        if (j + 1) % 500 == 0:
            print(f"  {j+1}/{len(strict_bg)} labelled ...")

    # Per-domain count
    print("\n=== Predicted domain distribution (background candidates) ===")
    pred_dist = Counter(r["predicted_domain"] for r in labelled)
    for d in sorted(pred_dist):
        print(f"  {d:18s}: {pred_dist[d]}")
    print(f"  total: {sum(pred_dist.values())}")

    # High vs low confidence breakdown
    high_conf = sum(1 for r in labelled if r["confidence"] >= 0.71)  # ≥5/7 votes
    mid_conf = sum(1 for r in labelled if 0.43 <= r["confidence"] < 0.71)
    low_conf = sum(1 for r in labelled if r["confidence"] < 0.43)
    print(f"\nConfidence: high (≥5/7): {high_conf}, mid (3-4/7): {mid_conf}, low (<3/7): {low_conf}")

    # Per-domain confidence breakdown
    print("\n=== Per-predicted-domain confidence breakdown ===")
    for d in sorted(pred_dist):
        domain_recs = [r for r in labelled if r["predicted_domain"] == d]
        h = sum(1 for r in domain_recs if r["confidence"] >= 0.71)
        m = sum(1 for r in domain_recs if 0.43 <= r["confidence"] < 0.71)
        l = sum(1 for r in domain_recs if r["confidence"] < 0.43)
        print(f"  {d:18s}: total={len(domain_recs):4d}  high={h:4d}  mid={m:4d}  low={l:4d}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "meta": {
            "date": "2026-05-01",
            "k": K,
            "method": "k-NN majority vote on BGE-EN-large bare vectors (post-D2 cleanup)",
            "n_candidates": len(labelled),
            "anchor_core_size": len(core_indices),
        },
        "candidates": labelled,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
