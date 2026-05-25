"""
Regenerate the per-directory index.json files from legal_terms.json (master).

The three embedding directories — embeddings/, embeddings_contextualized/,
embeddings_ctx_attested/ — each carry an index.json that mirrors the
9472-row master pool. The vector matrices vectors.npy in each directory
are aligned by row-position to this index. After any edit to
legal_terms.json (e.g. tier flips, addition of *_clean fields), the three
index.json files diverge from the master if not regenerated.

This utility re-emits the index.json in each directory with the same
9472-row order as legal_terms.json, including the fields needed by
downstream scripts:
    en, zh_canonical, en_clean, zh_clean, tier, domain.

D6 of `experiments/trace_firthian_pivot.md` (2026-05-01).

Usage
-----
    python data/sync_indexes.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS = REPO_ROOT / "data" / "processed" / "legal_terms.json"

EMBEDDING_DIRS = [
    REPO_ROOT / "data" / "processed" / "embeddings",
    REPO_ROOT / "data" / "processed" / "embeddings_contextualized",
    REPO_ROOT / "data" / "processed" / "embeddings_ctx_attested",
]

INDEX_FIELDS = ("en", "zh_canonical", "en_clean", "zh_clean", "tier", "domain")


def build_index(terms: list[dict]) -> list[dict]:
    """Project legal_terms.json to the index.json field schema, preserving order."""
    return [
        {f: t.get(f) for f in INDEX_FIELDS}
        for t in terms
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Preview only, no write")
    args = p.parse_args()

    payload = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))
    terms = payload["terms"]
    print(f"Master: {LEGAL_TERMS.relative_to(REPO_ROOT)}  ({len(terms)} terms)")

    new_index = build_index(terms)

    # Sanity: tier breakdown of master
    from collections import Counter
    tiers = Counter(t.get("tier") for t in terms)
    print(f"  master tiers: {dict(tiers)}")

    for d in EMBEDDING_DIRS:
        idx_path = d / "index.json"
        rel = idx_path.relative_to(REPO_ROOT)
        if not d.exists():
            print(f"  [skip] {rel}: directory does not exist")
            continue

        if idx_path.exists():
            old = json.loads(idx_path.read_text(encoding="utf-8"))
            old_tiers = Counter(t.get("tier") for t in old)
            print(f"  {rel}: pre-sync tiers = {dict(old_tiers)} ({len(old)} terms)")
        else:
            print(f"  {rel}: did not exist")

        if args.dry_run:
            continue

        idx_path.write_text(
            json.dumps(new_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  {rel}: synced ({len(new_index)} terms)")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
