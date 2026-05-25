"""
Re-extract attested contexts restricted to post-1989 (post-BLP) Caps for the
408 KEEP terms in `postBLP_curation_longlist.csv`.

Adapts the inverted-pass algorithm from `build_term_contexts.py` but:
  1. pre-filters sections by Cap.year ≥ 1989 (D7 of trace_postBLP_curation.md)
  2. restricts the term universe to the 408 KEEP terms only
  3. caps contexts at MAX_CONTEXTS=8 per term per language (D5 Firthian)

Inputs
------
- experiments/data/processed/postBLP_curation_longlist.csv
- experiments/data/processed/legal_terms.json (term_idx, en_clean, zh_clean)
- experiments/data/processed/elegislation/sections.jsonl (corpus)
- experiments/data/processed/cap_enactment_years.json (Cap → year)

Outputs
-------
- experiments/data/processed/elegislation/term_contexts_postBLP.jsonl
- experiments/data/processed/elegislation/coverage_postBLP.json

Usage
-----
    python3 experiments/data/build_postBLP_contexts.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/gpuzio/Desktop/CODE/THESIS/experiments")
PROC = ROOT / "data" / "processed"
ELEG = PROC / "elegislation"

CSV_LONGLIST = PROC / "postBLP_curation_longlist.csv"
CAP_YEARS = PROC / "cap_enactment_years.json"
SECTIONS_JSONL = ELEG / "sections.jsonl"
LEGAL_TERMS = PROC / "legal_terms.json"
ZH_OVERRIDES = PROC / "zh_overrides_postBLP.json"

OUT_CONTEXTS = ELEG / "term_contexts_postBLP.jsonl"
OUT_COVERAGE = ELEG / "coverage_postBLP.json"

POST_BLP_THRESHOLD = 1989
MAX_CONTEXTS = 8
MIN_CONTEXT_LEN = 60


# -------------------------------- helpers ---------------------------------- #

def cap_year(cap: str, verified: dict[str, int], ranges: dict[str, int]) -> int:
    """Resolve Cap → year via verified lookup or heuristic range."""
    cap = str(cap).strip()
    if cap in verified:
        return verified[cap]
    digits = "".join(c for c in cap if c.isdigit())
    if not digits:
        return 0
    n = int(digits)
    for spec, year in ranges.items():
        if spec.startswith("<="):
            if n <= int(spec[2:]):
                return year
        elif spec.startswith(">"):
            if n > int(spec[1:]):
                return year
        elif "-" in spec:
            a, b = spec.split("-")
            if int(a) <= n <= int(b):
                return year
    return 0


def build_snippet(text: str, match_start: int, match_end: int, window: int = 200) -> str:
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
    s = text[start:end].strip()
    if start > 0:
        s = "..." + s
    if end < len(text):
        s = s + "..."
    return s


# ---------------------------------- main ----------------------------------- #

def main() -> int:
    t0 = time.time()
    print("Loading inputs ...")
    raw = json.loads(CAP_YEARS.read_text(encoding="utf-8"))
    verified = {str(k): int(v) for k, v in raw["verified"].items()}
    ranges = raw["_meta"]["heuristic_ranges"]
    print(f"  cap_enactment_years.json: {len(verified)} verified Caps")

    keep_rows: list[dict] = []
    with CSV_LONGLIST.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["curation_decision"] == "KEEP":
                keep_rows.append(r)
    print(f"  KEEP terms: {len(keep_rows)}")

    # ZH overrides overlay: map (domain, en) → new zh matching string.
    # Entries with level=DROP remove the term from the active KEEP set entirely.
    zh_override_map: dict[tuple[str, str], str] = {}
    drop_overrides: set[tuple[str, str]] = set()
    if ZH_OVERRIDES.exists():
        ov = json.loads(ZH_OVERRIDES.read_text(encoding="utf-8"))
        for k, v in ov["overrides"].items():
            domain, en = k.split("|", 1)
            level = v.get("level")
            if level == "DROP":
                drop_overrides.add((domain, en))
            elif v.get("zh_override"):
                zh_override_map[(domain, en)] = v["zh_override"]
        print(f"  ZH overrides loaded: {len(zh_override_map)} apply, {len(drop_overrides)} DROP")
    else:
        print(f"  [!] no ZH override file: {ZH_OVERRIDES}")

    # Apply DROP overrides to keep_rows
    pre_drop = len(keep_rows)
    keep_rows = [r for r in keep_rows if (r["domain"], r["en"]) not in drop_overrides]
    print(f"  KEEP after override DROP: {len(keep_rows)} (dropped {pre_drop - len(keep_rows)})")

    lt = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))["terms"]
    by_en_zh = {}
    for i, t in enumerate(lt):
        en = t.get("en", "")
        for zh_field in ("zh_clean", "zh_canonical"):
            zh = t.get(zh_field, "")
            by_en_zh.setdefault((en, zh), i)
    keep_idx_to_row: dict[int, dict] = {}
    keep_term_idx: set[int] = set()
    for r in keep_rows:
        idx = by_en_zh.get((r["en"], r["zh"]))
        if idx is None:
            print(f"  [!] unmatched: ({r['domain']}) en={r['en']!r} zh={r['zh']!r}")
            continue
        keep_idx_to_row[idx] = r
        keep_term_idx.add(idx)
    print(f"  KEEP terms matched in legal_terms: {len(keep_term_idx)}")

    # Build matching structures restricted to KEEP terms.
    # ZH matching uses the override form when present; otherwise falls back
    # to zh_clean / zh_canonical.
    en_single: dict[str, list[int]] = defaultdict(list)
    en_multi: list[tuple[int, str]] = []
    zh_by_char: dict[str, list[tuple[int, str]]] = defaultdict(list)
    idx_to_zh_used: dict[int, str] = {}

    for i in keep_term_idx:
        t = lt[i]
        row = keep_idx_to_row[i]
        en = (t.get("en_clean") or t["en"]).lower()
        words = en.split()
        if len(words) == 1:
            en_single[en].append(i)
        else:
            en_multi.append((i, en))
        # Choose ZH form: override > zh_clean > zh_canonical
        ovk = (row["domain"], row["en"])
        zh = zh_override_map.get(ovk)
        if not zh:
            zh = t.get("zh_clean") or t.get("zh_canonical", "")
        idx_to_zh_used[i] = zh
        if zh:
            zh_by_char[zh[0]].append((i, zh))

    print(f"  EN single-word terms: {len(en_single)}, multi-word: {len(en_multi)}")
    print(f"  ZH terms indexed by {len(zh_by_char)} first-chars")

    # ---- single pass over sections, with Cap.year pre-filter ---- #
    print(f"\nScanning sections.jsonl with Cap.year >= {POST_BLP_THRESHOLD} pre-filter ...")
    en_contexts: dict[int, list[dict]] = defaultdict(list)
    zh_contexts: dict[int, list[dict]] = defaultdict(list)
    cap_year_cache: dict[str, int] = {}
    n_sections = 0
    n_sections_post = 0
    n_terms_full_en = 0
    n_terms_full_zh = 0

    word_re = re.compile(r"[a-z]+(?:[-'][a-z]+)*")

    with SECTIONS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            sec = json.loads(line)
            n_sections += 1
            cap = sec["cap"]
            if cap not in cap_year_cache:
                cap_year_cache[cap] = cap_year(cap, verified, ranges)
            y = cap_year_cache[cap]
            if y < POST_BLP_THRESHOLD:
                continue
            n_sections_post += 1
            sid = sec["section_id"]

            # EN
            en_text = sec.get("en_text", "")
            if en_text and len(en_text) >= MIN_CONTEXT_LEN:
                en_lower = en_text.lower()
                words = set(word_re.findall(en_lower))
                # single-word matches
                for w in words:
                    if w in en_single:
                        for idx in en_single[w]:
                            if len(en_contexts[idx]) < MAX_CONTEXTS:
                                pos = en_lower.find(w)
                                if pos >= 0:
                                    en_contexts[idx].append({
                                        "cap": cap, "cap_year": y, "section_id": sid,
                                        "context": build_snippet(en_text, pos, pos + len(w)),
                                    })
                # multi-word matches
                for idx, lower_term in en_multi:
                    if len(en_contexts[idx]) < MAX_CONTEXTS and lower_term in en_lower:
                        pos = en_lower.find(lower_term)
                        if pos >= 0:
                            en_contexts[idx].append({
                                "cap": cap, "cap_year": y, "section_id": sid,
                                "context": build_snippet(en_text, pos, pos + len(lower_term)),
                            })

            # ZH
            zh_text = sec.get("zh_text", "")
            if zh_text and len(zh_text) >= MIN_CONTEXT_LEN:
                chars = set(zh_text)
                for ch in chars:
                    if ch in zh_by_char:
                        for idx, zh_term in zh_by_char[ch]:
                            if len(zh_contexts[idx]) < MAX_CONTEXTS and zh_term in zh_text:
                                pos = zh_text.find(zh_term)
                                zh_contexts[idx].append({
                                    "cap": cap, "cap_year": y, "section_id": sid,
                                    "context": build_snippet(zh_text, pos, pos + len(zh_term)),
                                })

            if n_sections % 10000 == 0:
                print(f"  {n_sections} sections scanned, {n_sections_post} post-BLP")

    n_terms_full_en = sum(1 for v in en_contexts.values() if len(v) >= MAX_CONTEXTS)
    n_terms_full_zh = sum(1 for v in zh_contexts.values() if len(v) >= MAX_CONTEXTS)
    print(f"\n  total sections scanned: {n_sections}")
    print(f"  post-BLP sections (year>={POST_BLP_THRESHOLD}): {n_sections_post}")
    print(f"  KEEP terms reaching MAX_CONTEXTS={MAX_CONTEXTS} (en/zh): {n_terms_full_en}/{n_terms_full_zh}")

    # ---- write outputs ---- #
    print(f"\nWriting outputs ...")
    coverage_per_term = {}
    out_records = []

    for idx in sorted(keep_term_idx):
        t = lt[idx]
        row = keep_idx_to_row[idx]
        en_ctxs = en_contexts.get(idx, [])
        zh_ctxs = zh_contexts.get(idx, [])
        zh_used = idx_to_zh_used.get(idx, "")
        zh_orig = t.get("zh_clean") or t.get("zh_canonical", "")
        rec = {
            "term_idx": idx,
            "term_en": t["en"],
            "term_zh": t.get("zh_canonical", ""),
            "term_en_match": t.get("en_clean") or t["en"],
            "term_zh_match": zh_used,
            "term_zh_overridden": zh_used != zh_orig,
            "term_zh_original": zh_orig if zh_used != zh_orig else None,
            "domain": row["domain"],
            "tier_postBLP": "core",
            "k_en_postBLP": len(en_ctxs),
            "k_zh_postBLP": len(zh_ctxs),
            "k_min_postBLP": min(len(en_ctxs), len(zh_ctxs)),
            "en_contexts": en_ctxs,
            "zh_contexts": zh_ctxs,
        }
        out_records.append(rec)
        coverage_per_term[idx] = {
            "en": t["en"],
            "zh": t.get("zh_clean") or t.get("zh_canonical", ""),
            "domain": row["domain"],
            "k_en": len(en_ctxs),
            "k_zh": len(zh_ctxs),
            "k_min": min(len(en_ctxs), len(zh_ctxs)),
        }

    with OUT_CONTEXTS.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {OUT_CONTEXTS.name}: {len(out_records)} records")

    # Domain rollups
    by_dom = defaultdict(list)
    for info in coverage_per_term.values():
        by_dom[info["domain"]].append(info)

    domain_stats = {}
    for dom in sorted(by_dom):
        items = by_dom[dom]
        ken = [i["k_en"] for i in items]
        kzh = [i["k_zh"] for i in items]
        kmin = [i["k_min"] for i in items]
        domain_stats[dom] = {
            "n_terms": len(items),
            "k_en":  {"min": min(ken),  "median": statistics.median(ken),  "mean": round(statistics.mean(ken), 1),  "max": max(ken)},
            "k_zh":  {"min": min(kzh),  "median": statistics.median(kzh),  "mean": round(statistics.mean(kzh), 1),  "max": max(kzh)},
            "k_min": {"min": min(kmin), "median": statistics.median(kmin), "mean": round(statistics.mean(kmin), 1), "max": max(kmin)},
            "n_below_K4_en":   sum(1 for i in items if i["k_en"]   < 4),
            "n_below_K4_zh":   sum(1 for i in items if i["k_zh"]   < 4),
            "n_below_K4_either": sum(1 for i in items if i["k_min"] < 4),
            "n_zero_en":       sum(1 for i in items if i["k_en"] == 0),
            "n_zero_zh":       sum(1 for i in items if i["k_zh"] == 0),
        }

    coverage_out = {
        "_meta": {
            "threshold_year": POST_BLP_THRESHOLD,
            "max_contexts": MAX_CONTEXTS,
            "min_context_len": MIN_CONTEXT_LEN,
            "n_keep_terms": len(keep_term_idx),
            "n_sections_total": n_sections,
            "n_sections_post_BLP": n_sections_post,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "per_domain": domain_stats,
        "per_term": coverage_per_term,
    }
    with OUT_COVERAGE.open("w", encoding="utf-8") as f:
        json.dump(coverage_out, f, ensure_ascii=False, indent=2)
    print(f"  {OUT_COVERAGE.name}")

    # Print summary
    print("\n" + "=" * 100)
    print("Per-domain coverage (post-BLP only, K_max=8):")
    print("=" * 100)
    print(f"{'domain':18}  {'n':>3}  {'k_en':>21}  {'k_zh':>21}  {'k_min':>21}  K<4_either")
    for dom, s in domain_stats.items():
        en  = s["k_en"];  zh = s["k_zh"]; km = s["k_min"]
        line = (f"{dom:18}  {s['n_terms']:3}  "
                f"min={en['min']:>2} med={en['median']:>4.0f} mean={en['mean']:>4.1f} max={en['max']:>2}  "
                f"min={zh['min']:>2} med={zh['median']:>4.0f} mean={zh['mean']:>4.1f} max={zh['max']:>2}  "
                f"min={km['min']:>2} med={km['median']:>4.0f} mean={km['mean']:>4.1f} max={km['max']:>2}  "
                f"{s['n_below_K4_either']:3}")
        print(line)

    # Overall
    all_kmin = [i["k_min"] for i in coverage_per_term.values()]
    print(f"\nOverall: n={len(all_kmin)}, k_min median={statistics.median(all_kmin):.0f}, "
          f"K_min<4: {sum(1 for k in all_kmin if k<4)}/{len(all_kmin)}")
    print(f"Elapsed: {round(time.time()-t0,1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
