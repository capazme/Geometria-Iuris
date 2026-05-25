"""
Materialise the post-BLP canonical dataset file `legal_terms_postBLP.json`.

This is the single self-contained file the downstream encoding + experiments
pipeline reads. It merges:
  - 364 KEEP terms from `postBLP_curation_longlist.csv`
  - ZH overrides from `zh_overrides_postBLP.json`
  - Per-term K_en/K_zh from `coverage_postBLP.json`
  - Base fields (en_clean, zh_canonical, term_idx) from `legal_terms.json`

Schema (per-term)
-----------------
- term_idx                       : int, position in the 9472-pool index
- domain                         : str, one of 7 domains
- tier_postBLP                   : "core"
- en                             : str, canonical EN headword
- en_clean                       : str, glossary-cleaned EN (used for matching)
- zh_canonical                   : str, DOJ-glossary canonical form (citation)
- zh_clean                       : str, Firthian-cleaned ZH (kept for audit)
- zh_clean_postBLP               : str, FINAL ZH form for encoding (post override)
- zh_postBLP_override_applied    : bool
- zh_postBLP_override_rationale  : str | null (only if override applied)
- k_en_postBLP                   : int, attested EN contexts (max 8)
- k_zh_postBLP                   : int, attested ZH contexts (max 8)
- k_min_postBLP                  : int, min(k_en, k_zh)

Usage
-----
    python3 experiments/data/build_legal_terms_postBLP.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import date

ROOT = Path("/Users/gpuzio/Desktop/CODE/THESIS/experiments")
PROC = ROOT / "data" / "processed"
ELEG = PROC / "elegislation"

LEGAL_TERMS = PROC / "legal_terms.json"
CSV_LONGLIST = PROC / "postBLP_curation_longlist.csv"
ZH_OVERRIDES = PROC / "zh_overrides_postBLP.json"
COVERAGE = ELEG / "coverage_postBLP.json"
OUT = PROC / "legal_terms_postBLP.json"


def main() -> int:
    print("Loading sources …")
    lt = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))["terms"]
    by_en_zh: dict[tuple[str, str], int] = {}
    for i, t in enumerate(lt):
        en = t.get("en", "")
        for zh_field in ("zh_clean", "zh_canonical"):
            by_en_zh.setdefault((en, t.get(zh_field, "")), i)

    keep_rows: list[dict] = []
    with CSV_LONGLIST.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["curation_decision"] == "KEEP":
                keep_rows.append(r)
    print(f"  KEEP terms in longlist: {len(keep_rows)}")

    ov_raw = json.loads(ZH_OVERRIDES.read_text(encoding="utf-8"))
    overrides: dict[tuple[str, str], dict] = {}
    drops_in_override: set[tuple[str, str]] = set()
    for k, v in ov_raw["overrides"].items():
        domain, en = k.split("|", 1)
        if v.get("level") == "DROP":
            drops_in_override.add((domain, en))
        elif v.get("zh_override"):
            overrides[(domain, en)] = v

    cov = json.loads(COVERAGE.read_text(encoding="utf-8"))
    coverage_per_idx: dict[int, dict] = {int(k): v for k, v in cov["per_term"].items()}

    print(f"  ZH overrides: {len(overrides)} apply, {len(drops_in_override)} DROP")
    print(f"  coverage entries: {len(coverage_per_idx)}")

    out_terms = []
    unmatched = []
    skipped_drop_in_override = 0

    for r in keep_rows:
        if (r["domain"], r["en"]) in drops_in_override:
            skipped_drop_in_override += 1
            continue
        idx = by_en_zh.get((r["en"], r["zh"]))
        if idx is None:
            unmatched.append(r)
            continue
        t = lt[idx]
        ov = overrides.get((r["domain"], r["en"]))
        zh_postBLP = ov["zh_override"] if ov else (t.get("zh_clean") or t.get("zh_canonical", ""))
        cov_info = coverage_per_idx.get(idx, {})
        out_terms.append({
            "term_idx": idx,
            "domain": r["domain"],
            "tier_postBLP": "core",
            "en": t.get("en", ""),
            "en_clean": t.get("en_clean") or t.get("en", ""),
            "zh_canonical": t.get("zh_canonical", ""),
            "zh_clean": t.get("zh_clean", ""),
            "zh_clean_postBLP": zh_postBLP,
            "zh_postBLP_override_applied": bool(ov),
            "zh_postBLP_override_rationale": ov.get("rationale") if ov else None,
            "k_en_postBLP": cov_info.get("k_en", 0),
            "k_zh_postBLP": cov_info.get("k_zh", 0),
            "k_min_postBLP": cov_info.get("k_min", 0),
        })

    if unmatched:
        print(f"  [!] {len(unmatched)} KEEP rows unmatched in legal_terms.json:")
        for r in unmatched[:5]:
            print(f"      {r['domain']} | {r['en']} | {r['zh']}")
    if skipped_drop_in_override:
        print(f"  Skipped {skipped_drop_in_override} KEEP rows that have level=DROP in zh_overrides")

    # stats
    per_dom: dict[str, list[dict]] = defaultdict(list)
    for t in out_terms:
        per_dom[t["domain"]].append(t)

    domain_stats = {}
    for dom in sorted(per_dom):
        ts = per_dom[dom]
        kmins = [t["k_min_postBLP"] for t in ts]
        domain_stats[dom] = {
            "n": len(ts),
            "n_K4_strict": sum(1 for k in kmins if k >= 4),
            "k_min_median": statistics.median(kmins),
            "k_min_mean": round(statistics.mean(kmins), 2),
        }

    out = {
        "_meta": {
            "version": "1.0",
            "created": str(date.today()),
            "description": (
                "Post-BLP canonical core dataset for Geometria Iuris Ch.3 "
                "(Methodology of Legal Sciences thesis, LUISS). Each entry "
                "is a HK Cap. bilingual legal term selected by manual "
                "curation from the K_postBLP≥4 long list (1271 candidates) "
                "with ZH wrong-sense glosses corrected against the corpus."
            ),
            "selection_criteria": {
                "corpus": "HK Cap. e-Legislation",
                "post_BLP_threshold_year": 1989,
                "min_K_target": 4,
                "max_contexts_per_lang": 8,
                "domains": 7,
                "balance_band_target": "55-65 per domain (achieved 41-60)",
            },
            "stats": {
                "n_total": len(out_terms),
                "n_K4_strict": sum(1 for t in out_terms if t["k_min_postBLP"] >= 4),
                "n_with_zh_override": sum(1 for t in out_terms if t["zh_postBLP_override_applied"]),
                "per_domain": domain_stats,
            },
            "audit_trail": {
                "longlist_csv": "postBLP_curation_longlist.csv",
                "zh_overrides_json": "zh_overrides_postBLP.json",
                "curation_audit_json": "postBLP_curation_audit.json",
                "context_file_jsonl": "elegislation/term_contexts_postBLP.jsonl",
                "coverage_json": "elegislation/coverage_postBLP.json",
                "trace_doc_md": "../trace_postBLP_curation.md",
                "curation_script_py": "../apply_postBLP_curation.py",
                "extraction_script_py": "../build_postBLP_contexts.py",
            },
            "encoding_instructions": {
                "en_field": "en_clean",
                "zh_field": "zh_clean_postBLP",
                "contexts_source": "elegislation/term_contexts_postBLP.jsonl",
                "aggregation": "mean over attested contexts per language; no synthetic padding",
                "note": "When zh_postBLP_override_applied=true, zh_clean_postBLP differs from zh_clean; the override form is the form actually attested in the post-1989 corpus and should be used as the encoding input.",
            },
        },
        "terms": out_terms,
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    size_mb = OUT.stat().st_size / 1_000_000
    print(f"\nWrote {OUT.name} ({size_mb:.2f} MB, {len(out_terms)} terms)")

    print("\nPer-domain summary:")
    print(f"  {'domain':18}  {'n':>3}  {'K≥4':>4}  k_min med  k_min mean")
    for dom, s in domain_stats.items():
        print(f"  {dom:18}  {s['n']:3}  {s['n_K4_strict']:4}  {s['k_min_median']:>9.0f}  {s['k_min_mean']:>10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
