"""
Demote the 23 keep_kunder4 doctrinal-exception terms from core to background,
without backfill.

Rationale: enforce a single linear criterion (K≥4 in both languages) across
the entire core, accepting per-domain imbalance (45-50 per domain) instead
of weakening the criterion or backfilling with poorly-fitting candidates.

The 23 terms remain in legal_terms.json with full audit trace (tier_history,
firthian_curation_decision='kunder4_demoted_no_backfill') for transparency
and possible future use.

D4 step 4 final revision (linear-and-coherent, no backfill, 2026-05-02).

Usage
-----
    python data/demote_kunder4.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS = REPO_ROOT / "data" / "processed" / "legal_terms.json"
DATE = "2026-05-02"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    payload = json.loads(LEGAL_TERMS.read_text(encoding='utf-8'))
    terms = payload['terms']

    kunder4 = [(i, t) for i, t in enumerate(terms) if t.get('firthian_curation_decision') == 'keep_kunder4']
    print(f"Found {len(kunder4)} keep_kunder4 entries to demote")

    print("\nPer-domain count:")
    by_dom = Counter(t['domain'] for _, t in kunder4)
    for d in sorted(by_dom):
        print(f"  {d}: {by_dom[d]}")

    print("\nTerms to demote (term_idx, en, current k_en/k_zh):")
    for ti, t in kunder4:
        print(f"  {ti:5d}  {t['en']!r:35s} domain={t['domain']:18s}")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
        return 0

    backup_path = LEGAL_TERMS.parent / f"legal_terms.json.bak_pre_kunder4_demote_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(LEGAL_TERMS, backup_path)
    print(f"\nBackup: {backup_path.relative_to(REPO_ROOT)}")

    for ti, t in kunder4:
        t.setdefault('tier_history', []).append({
            'date': DATE, 'from': 'core', 'to': 'background',
            'reason': 'kunder4_demote_no_backfill',
        })
        t['tier'] = 'background'
        t['firthian_kunder4_origin_domain'] = t['domain']
        t['domain'] = None
        t['firthian_curation_decision'] = 'kunder4_demoted_no_backfill'

    final_core = [t for t in terms if t.get('tier') == 'core']
    dom_count = Counter(t.get('domain') for t in final_core)
    print(f"\nFinal core: {len(final_core)} terms")
    print(f"Per-domain distribution:")
    for d in sorted(dom_count):
        print(f"  {d}: {dom_count[d]}")

    LEGAL_TERMS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"\nSaved: {LEGAL_TERMS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
