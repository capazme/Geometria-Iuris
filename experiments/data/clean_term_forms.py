"""
Add `en_clean` and `zh_clean` fields to legal_terms.json.

These are the canonical surface forms used by the experimental pipeline
for: (1) corpus matching when retrieving attested contexts, (2) bare
embedding encoding, (3) attested context-window encoding.

The original `en` and `zh_canonical` fields are preserved as the
catalogued DOJ glossary entries (citation form). The `*_clean` fields
strip glossary-editorial metadata that is not part of the legal term
itself.

D1 of `experiments/trace_firthian_pivot.md` (2026-05-01).

Cleanup rules
-------------

ZH (aggressive):
- Strip `（[^）]*）` annotative parentheses (e.g. `（香港）`)
- Truncate at `※` or `☛` (DOJ "compare" / "see also" markers)
- Split on `;` or `/`, keep first part
- Strip `〔[^〕]*〕` glossary brackets (paraphrasis explanations)
- Strip terminal `的` when the resulting string is at least 2 characters
  (handles adverbial `憲制的` → `憲制`; preserves `目的` because trimming
  would leave only one character)

EN (conservative):
- Split on `;`, keep first part (handles `"X; abbreviation"` → `"X"`)
- Otherwise verbatim

Usage
-----
    python data/clean_term_forms.py --dry-run        # preview only
    python data/clean_term_forms.py --apply          # writes file + backup
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS_PATH = REPO_ROOT / "data" / "processed" / "legal_terms.json"


# Regex patterns
RE_PARENS_ZH = re.compile(r'（[^）]*）')   # full-width Chinese parens
RE_BRACKETS_ZH = re.compile(r'〔[^〕]*〕')  # full-width Chinese brackets
RE_MARKER_ZH = re.compile(r'[※☛].*')      # truncate at first marker
RE_SPLIT_SEP = re.compile(r'[;/]')        # split on ; or /
RE_PARENS_HALF = re.compile(r'\([^)]*\)') # half-width parens (rare in zh field)


def clean_zh(s: str) -> str:
    """Apply ZH cleanup rules. Returns cleaned form or original if cleanup
    would empty the string."""
    if not s:
        return s
    original = s
    # 1. Truncate at ※ or ☛
    s = RE_MARKER_ZH.sub('', s)
    # 2. Strip annotative parentheses (full + half width)
    s = RE_PARENS_ZH.sub('', s)
    s = RE_PARENS_HALF.sub('', s)
    # 3. Strip glossary brackets
    s = RE_BRACKETS_ZH.sub('', s)
    # 4. Split on ; or /, keep first
    parts = RE_SPLIT_SEP.split(s, maxsplit=1)
    s = parts[0]
    # 5. Strip whitespace
    s = s.strip()
    # 6. Strip terminal 的 if result remains ≥2 chars
    while s.endswith('的') and len(s) > 2:
        s = s[:-1]
    s = s.strip()
    # Safety: if cleanup emptied the string, fall back to original
    if not s:
        return original
    return s


def clean_en(s: str) -> str:
    """Apply EN cleanup rules. Conservative: split on ; only."""
    if not s:
        return s
    original = s
    parts = s.split(';', maxsplit=1)
    s = parts[0].strip()
    if not s:
        return original
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Add en_clean / zh_clean to legal_terms.json")
    p.add_argument('--dry-run', action='store_true', help="Preview only, no write")
    p.add_argument('--apply', action='store_true', help="Write file + backup")
    p.add_argument('--full-report', action='store_true', help="Show every change (default: top 30 per language)")
    args = p.parse_args()

    if not (args.dry_run or args.apply):
        print("ERROR: pass --dry-run or --apply")
        return 1

    payload = json.loads(LEGAL_TERMS_PATH.read_text(encoding='utf-8'))
    terms = payload['terms']
    print(f"Loaded {len(terms)} terms from {LEGAL_TERMS_PATH.relative_to(REPO_ROOT)}")

    changed_zh = []  # list of (idx, en, zh_orig, zh_clean, tier, domain)
    changed_en = []
    unchanged_count = 0

    for i, t in enumerate(terms):
        en_orig = t.get('en', '')
        zh_orig = t.get('zh_canonical', '')
        en_c = clean_en(en_orig)
        zh_c = clean_zh(zh_orig)
        if en_c != en_orig:
            changed_en.append((i, en_orig, en_c, t.get('tier'), t.get('domain')))
        if zh_c != zh_orig:
            changed_zh.append((i, en_orig, zh_orig, zh_c, t.get('tier'), t.get('domain')))
        if en_c == en_orig and zh_c == zh_orig:
            unchanged_count += 1
        # In any case, attach the new fields
        t['en_clean'] = en_c
        t['zh_clean'] = zh_c

    print(f"\n=== Summary ===")
    print(f"  EN changed:  {len(changed_en)} terms")
    print(f"  ZH changed:  {len(changed_zh)} terms")
    print(f"  unchanged:   {unchanged_count} terms")

    # Tier breakdown of changes
    from collections import Counter
    tier_zh = Counter(c[4] for c in changed_zh)
    tier_en = Counter(c[3] for c in changed_en)
    print(f"\n=== ZH changes by tier ===")
    for tier, n in tier_zh.most_common():
        print(f"  {tier}: {n}")
    print(f"=== EN changes by tier ===")
    for tier, n in tier_en.most_common():
        print(f"  {tier}: {n}")

    # Show core-tier changes in full
    print(f"\n=== ZH changes on CORE tier (full list) ===")
    core_zh = [c for c in changed_zh if c[4] == 'core']
    print(f"  {len(core_zh)} terms")
    for idx, en, orig, clean, _, dom in core_zh:
        print(f"    [{dom:14s}] {en!r:35s}  {orig!r}  →  {clean!r}")

    print(f"\n=== EN changes on CORE tier (full list) ===")
    core_en = [c for c in changed_en if c[3] == 'core']
    print(f"  {len(core_en)} terms")
    for idx, orig, clean, _, dom in core_en:
        print(f"    [{dom:14s}] {orig!r}  →  {clean!r}")

    # Full report flag for background scanning
    if args.full_report:
        print(f"\n=== ZH changes on BACKGROUND tier (sample 30 of {len(changed_zh)-len(core_zh)}) ===")
        bg_zh = [c for c in changed_zh if c[4] != 'core'][:30]
        for idx, en, orig, clean, tier, _ in bg_zh:
            print(f"    [{tier}] {en!r:30s}  {orig!r}  →  {clean!r}")

    if args.dry_run:
        print(f"\n[DRY RUN] No file written. Use --apply to commit.")
        return 0

    # APPLY mode: write backup, then file
    backup_path = LEGAL_TERMS_PATH.parent / f"legal_terms.json.bak_pre_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(LEGAL_TERMS_PATH, backup_path)
    print(f"\nBackup written: {backup_path.relative_to(REPO_ROOT)}")

    LEGAL_TERMS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"Updated: {LEGAL_TERMS_PATH.relative_to(REPO_ROOT)}")
    print(f"Added fields: en_clean, zh_clean")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
