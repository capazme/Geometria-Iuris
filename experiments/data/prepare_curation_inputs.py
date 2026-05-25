"""
Prepare per-domain input files for the D4 step 2 curation agents.

For each of 7 domains, writes a JSON input file containing:
  - 50 current core terms in that domain (with k_en, k_zh attestation counts)
  - Top backfill candidates predicted to that domain (post-D4 step 1 k-NN
    labelling), ranked by confidence × (k_en + k_zh), capped at CAP=500.

Filter: candidates with confidence < 0.43 (i.e. <3/7 k-NN votes) are
included only if the predicted-domain pool would otherwise be smaller
than 100. This gives the agent enough material while not flooding it
with low-quality predictions.

D4 step 2 input prep, per `experiments/trace_firthian_pivot.md`.

Usage
-----
    python data/prepare_curation_inputs.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS = REPO_ROOT / "data" / "processed" / "legal_terms.json"
TERM_CONTEXTS = REPO_ROOT / "data" / "processed" / "elegislation" / "term_contexts.jsonl"
LABELLED = REPO_ROOT / "data" / "review" / "strict_pool_labelled.json"
INDEX_JSON = REPO_ROOT / "data" / "processed" / "embeddings" / "index.json"
OUT_DIR = REPO_ROOT / "data" / "review"

CAP_PER_DOMAIN = 500
MIN_HIGHMID_BEFORE_INCLUDING_LOW = 100  # if high+mid candidates < this, include low too


def main() -> int:
    print("Loading inputs ...")
    lt = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))["terms"]
    index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    labelled = json.loads(LABELLED.read_text(encoding="utf-8"))["candidates"]

    contexts = {}
    with TERM_CONTEXTS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            contexts[rec["term_idx"]] = rec

    # Build pool position map: (en, zh_canonical) → idx in 9472 pool
    pool_map = {(t["en"], t["zh_canonical"]): i for i, t in enumerate(index)}

    # Group current core by domain, with attestation counts
    print("\n=== Current core per domain (with attestation counts) ===")
    domain_to_core: dict[str, list[dict]] = defaultdict(list)
    for t in lt:
        if t.get("tier") != "core" or not t.get("domain"):
            continue
        d = t["domain"]
        pi = pool_map.get((t["en"], t["zh_canonical"]))
        if pi is None:
            print(f"  WARNING: core term not in pool: {t['en']!r}")
            continue
        rec = contexts.get(pi)
        n_en = len(rec.get("en_contexts", [])) if rec else 0
        n_zh = len(rec.get("zh_contexts", [])) if rec else 0
        domain_to_core[d].append({
            "term_idx": pi,
            "en": t["en"],
            "zh_canonical": t["zh_canonical"],
            "en_clean": t.get("en_clean") or t["en"],
            "zh_clean": t.get("zh_clean") or t["zh_canonical"],
            "k_en": n_en,
            "k_zh": n_zh,
            "passes_strict_gate": n_en >= 4 and n_zh >= 4,
        })

    for d in sorted(domain_to_core):
        n = len(domain_to_core[d])
        n_strict = sum(1 for t in domain_to_core[d] if t["passes_strict_gate"])
        print(f"  {d:18s}: {n} core ({n_strict} strict K≥4, {n - n_strict} sub-K≥4)")

    # Group candidates by predicted domain
    print("\n=== Backfill candidates per domain (post k-NN) ===")
    domain_to_cands: dict[str, list[dict]] = defaultdict(list)
    for c in labelled:
        domain_to_cands[c["predicted_domain"]].append(c)

    # For each domain: rank by confidence × (k_en + k_zh), apply cap
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in sorted(domain_to_core):
        core = domain_to_core[d]
        cands = domain_to_cands.get(d, [])

        # Filter: include high+mid; if fewer than threshold, include low too
        high_mid = [c for c in cands if c["confidence"] >= 0.43]
        if len(high_mid) < MIN_HIGHMID_BEFORE_INCLUDING_LOW:
            print(f"  {d}: only {len(high_mid)} high/mid; including low candidates too")
            filtered = cands
        else:
            filtered = high_mid

        # Sort by confidence DESC, then by k_en + k_zh DESC, then by k_en DESC
        filtered.sort(
            key=lambda c: (-c["confidence"], -(c["k_attested_en"] + c["k_attested_zh"]), -c["k_attested_en"])
        )
        # Cap
        candidates_out = filtered[:CAP_PER_DOMAIN]

        # Project to clean output schema
        candidates_clean = [{
            "term_idx": c["term_idx"],
            "en": c["en"],
            "zh_canonical": c["zh_canonical"],
            "en_clean": c["en_clean"],
            "zh_clean": c["zh_clean"],
            "k_en": c["k_attested_en"],
            "k_zh": c["k_attested_zh"],
            "knn_confidence": c["confidence"],
            "knn_vote_distribution": c["vote_distribution"],
            "knn_top_neighbour_sim": c["top_neighbour_sim"],
        } for c in candidates_out]

        out_path = OUT_DIR / f"curation_input_{d}.json"
        out_path.write_text(json.dumps({
            "domain": d,
            "target_count": 50,
            "current_core": core,
            "backfill_candidates": candidates_clean,
            "candidate_pool_stats": {
                "total_predicted_to_domain": len(cands),
                "high_mid_confidence": len([c for c in cands if c["confidence"] >= 0.43]),
                "passed_to_agent": len(candidates_clean),
            },
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {d:18s}: core={len(core)} candidates_passed={len(candidates_clean)} → {out_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
