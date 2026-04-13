"""
Build a term-to-context index from the parsed e-Legislation corpus.

For each of the 9472 terms in legal_terms.json, finds all sections in the
e-Legislation corpus where the term appears (case-insensitive for EN,
exact match for ZH). Extracts a context window around each occurrence and
writes the results for downstream contextualised embedding computation.

This replaces the synthetic template approach in build_contextualized_pool.py
with attested legal usage from Hong Kong legislation.

Algorithm: single-pass over sections (inverted approach) rather than
per-term scanning, giving O(S * T_avg) where T_avg is the average number
of terms per section, instead of O(T * S) regex scans.

Output
------
data/processed/elegislation/
    term_contexts.jsonl     one record per term with attested contexts
    coverage.json           per-term hit counts + overall coverage stats

Usage
-----
    python data/build_term_contexts.py [--max-contexts N] [--min-context-len N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
LEGAL_TERMS = ROOT / "data" / "processed" / "legal_terms.json"
SECTIONS_JSONL = ROOT / "data" / "processed" / "elegislation" / "sections.jsonl"
OUT_DIR = ROOT / "data" / "processed" / "elegislation"

CONTEXT_WINDOW = 300


def build_context_snippet(text: str, match_start: int, match_end: int, window: int = CONTEXT_WINDOW) -> str:
    """Extract a context window around a match, respecting word boundaries."""
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    if start > 0:
        space = text.rfind(" ", start, match_start)
        if space > 0:
            start = space + 1
    if end < len(text):
        space = text.find(" ", match_end, end)
        if space > 0:
            end = space
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def main(args: argparse.Namespace) -> int:
    max_contexts = args.max_contexts
    min_context_len = args.min_context_len

    # Load terms
    logger.info("Loading legal_terms.json ...")
    data = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))
    terms = data["terms"] if isinstance(data, dict) else data
    logger.info("  %d terms loaded", len(terms))

    # Build lookup structures for fast matching
    # EN: lowercased term → list of (term_idx, original_term)
    # Split into single-word and multi-word for different strategies
    en_single: dict[str, list[int]] = defaultdict(list)  # word → [term_idx, ...]
    en_multi: list[tuple[int, str, str]] = []  # (term_idx, lower_term, original)

    for i, t in enumerate(terms):
        en = t["en"]
        en_lower = en.lower()
        words = en_lower.split()
        if len(words) == 1:
            en_single[en_lower].append(i)
        else:
            en_multi.append((i, en_lower, en))

    # ZH: build Aho-Corasick-like prefix index for substring matching
    # Simple approach: group by first character for fast filtering
    zh_by_char: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i, t in enumerate(terms):
        zh = t["zh_canonical"]
        if zh:
            zh_by_char[zh[0]].append((i, zh))

    logger.info("  EN single-word terms: %d unique words", len(en_single))
    logger.info("  EN multi-word terms: %d", len(en_multi))
    logger.info("  ZH terms: %d (indexed by %d first-chars)", sum(len(v) for v in zh_by_char.values()), len(zh_by_char))

    # Load sections
    logger.info("Loading sections.jsonl ...")
    sections: list[dict] = []
    with SECTIONS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            sections.append(json.loads(line))
    logger.info("  %d sections loaded", len(sections))

    # Accumulators
    en_contexts: dict[int, list[dict]] = defaultdict(list)
    zh_contexts: dict[int, list[dict]] = defaultdict(list)

    # Single pass over sections
    logger.info("Scanning sections (inverted approach) ...")
    for si, sec in enumerate(sections):
        cap = sec["cap"]
        sid = sec["section_id"]

        # --- EN matching ---
        en_text = sec.get("en_text", "")
        if en_text and len(en_text) >= min_context_len:
            en_lower = en_text.lower()
            # Tokenize once for single-word lookup
            en_words = set(re.findall(r"[a-z]+(?:[-'][a-z]+)*", en_lower))

            # Single-word terms: set intersection
            for word in en_words:
                if word in en_single:
                    for idx in en_single[word]:
                        if len(en_contexts[idx]) < max_contexts:
                            pos = en_lower.find(word)
                            if pos >= 0:
                                snippet = build_context_snippet(en_text, pos, pos + len(word))
                                en_contexts[idx].append({"cap": cap, "section_id": sid, "context": snippet})

            # Multi-word terms: substring check (fast pre-filter with `in`)
            for idx, lower_term, orig in en_multi:
                if len(en_contexts[idx]) < max_contexts and lower_term in en_lower:
                    pos = en_lower.find(lower_term)
                    if pos >= 0:
                        snippet = build_context_snippet(en_text, pos, pos + len(lower_term))
                        en_contexts[idx].append({"cap": cap, "section_id": sid, "context": snippet})

        # --- ZH matching ---
        zh_text = sec.get("zh_text", "")
        if zh_text and len(zh_text) >= min_context_len:
            # For each character in the text, check if any term starts with it
            chars_in_text = set(zh_text)
            for ch in chars_in_text:
                if ch in zh_by_char:
                    for idx, zh_term in zh_by_char[ch]:
                        if len(zh_contexts[idx]) < max_contexts and zh_term in zh_text:
                            pos = zh_text.find(zh_term)
                            snippet = build_context_snippet(zh_text, pos, pos + len(zh_term))
                            zh_contexts[idx].append({"cap": cap, "section_id": sid, "context": snippet})

        if (si + 1) % 10000 == 0:
            logger.info("  %d/%d sections processed ...", si + 1, len(sections))

    en_hit_count = sum(1 for v in en_contexts.values() if v)
    zh_hit_count = sum(1 for v in zh_contexts.values() if v)
    logger.info("  EN: %d/%d terms found (%.1f%%)", en_hit_count, len(terms),
                100 * en_hit_count / len(terms))
    logger.info("  ZH: %d/%d terms found (%.1f%%)", zh_hit_count, len(terms),
                100 * zh_hit_count / len(terms))

    # Write term_contexts.jsonl
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "term_contexts.jsonl"
    both_hit = 0
    records_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(terms):
            en_ctxs = en_contexts.get(i, [])
            zh_ctxs = zh_contexts.get(i, [])
            if en_ctxs or zh_ctxs:
                if en_ctxs and zh_ctxs:
                    both_hit += 1
                record = {
                    "term_idx": i,
                    "term_en": t["en"],
                    "term_zh": t["zh_canonical"],
                    "domain": t.get("domain"),
                    "tier": t["tier"],
                    "en_contexts": en_ctxs,
                    "zh_contexts": zh_ctxs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

    logger.info("\nContexts: %s (%d records)", out_path, records_written)

    # Coverage stats
    coverage = {
        "total_terms": len(terms),
        "en_terms_found": en_hit_count,
        "zh_terms_found": zh_hit_count,
        "both_found": both_hit,
        "en_coverage_pct": round(100 * en_hit_count / len(terms), 1),
        "zh_coverage_pct": round(100 * zh_hit_count / len(terms), 1),
        "both_coverage_pct": round(100 * both_hit / len(terms), 1),
        "max_contexts_per_term": max_contexts,
        "min_context_len": min_context_len,
        "by_tier": {},
        "by_domain": {},
    }

    for tier in ("core", "background", "control"):
        tier_idx = [i for i, t in enumerate(terms) if t["tier"] == tier]
        en_found = sum(1 for i in tier_idx if en_contexts.get(i))
        zh_found = sum(1 for i in tier_idx if zh_contexts.get(i))
        coverage["by_tier"][tier] = {
            "total": len(tier_idx),
            "en_found": en_found,
            "zh_found": zh_found,
        }

    domain_counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "en": 0, "zh": 0})
    for i, t in enumerate(terms):
        if t["tier"] == "core" and t.get("domain"):
            d = t["domain"]
            domain_counts[d]["total"] += 1
            if en_contexts.get(i):
                domain_counts[d]["en"] += 1
            if zh_contexts.get(i):
                domain_counts[d]["zh"] += 1
    coverage["by_domain"] = dict(domain_counts)

    cov_path = OUT_DIR / "coverage.json"
    cov_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False))
    logger.info("Coverage: %s", cov_path)

    logger.info("\n=== Coverage Summary ===")
    logger.info("  EN: %d/%d (%.1f%%)", en_hit_count, len(terms), coverage["en_coverage_pct"])
    logger.info("  ZH: %d/%d (%.1f%%)", zh_hit_count, len(terms), coverage["zh_coverage_pct"])
    logger.info("  Both: %d/%d (%.1f%%)", both_hit, len(terms), coverage["both_coverage_pct"])

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-contexts", type=int, default=8,
                        help="Max context snippets per term per language (default: 8)")
    parser.add_argument("--min-context-len", type=int, default=20,
                        help="Minimum section text length to consider (default: 20 chars)")
    sys.exit(main(parser.parse_args()))
