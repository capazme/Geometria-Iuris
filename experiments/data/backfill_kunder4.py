"""
Replace the 23 keep_kunder4 doctrinal exceptions with strict-pool backfill
candidates to enforce a single uniform K≥4 criterion across all 350 core.

Per-domain procedure:
  1. For each kunder4 term in the current core, demote (core → background).
  2. From strict_pool_labelled.json filter candidates predicted to that
     domain that are not currently in core and pass K≥4 in both languages.
  3. Rank by knn_confidence DESC, then k_en + k_zh DESC.
  4. Promote the top N (where N = original kunder4 count for that domain).

This restores 50/domain balance without exception slots.

The 23 demoted terms remain in legal_terms.json as background with
audit trace (`firthian_curation_decision: 'kunder4_backfilled_out'`).

D4 step 4 addendum (linear-and-coherent revision, 2026-05-02).

Usage
-----
    python data/backfill_kunder4.py [--dry-run]
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
LABELLED = REPO_ROOT / "data" / "review" / "strict_pool_labelled.json"
DATE = "2026-05-02"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    payload = json.loads(LEGAL_TERMS.read_text(encoding='utf-8'))
    terms = payload['terms']
    labelled = json.loads(LABELLED.read_text(encoding='utf-8'))['candidates']

    # Identify current keep_kunder4 entries (terms where firthian_curation_decision == 'keep_kunder4')
    kunder4_terms = []
    for i, t in enumerate(terms):
        if t.get('firthian_curation_decision') == 'keep_kunder4':
            kunder4_terms.append((i, t))

    print(f"Found {len(kunder4_terms)} keep_kunder4 entries to backfill")

    # Per-domain count
    by_domain = Counter()
    for i, t in kunder4_terms:
        by_domain[t['domain']] += 1
    print(f"\nPer-domain kunder4 counts (= backfill needed):")
    for d in sorted(by_domain):
        print(f"  {d}: {by_domain[d]}")

    # Find candidates per domain: predicted_domain match, in background, not already in core, K≥4
    current_core_idx = {i for i, t in enumerate(terms) if t.get('tier') == 'core'}

    cand_by_domain = {}
    for c in labelled:
        ti = c['term_idx']
        d = c['predicted_domain']
        if ti in current_core_idx:
            continue
        if c['k_attested_en'] < 4 or c['k_attested_zh'] < 4:
            continue
        cand_by_domain.setdefault(d, []).append(c)

    # Sort each domain's candidates by confidence DESC, then attestation DESC
    for d in cand_by_domain:
        cand_by_domain[d].sort(key=lambda c: (-c['confidence'], -(c['k_attested_en'] + c['k_attested_zh']), -c['k_attested_en']))

    # Build backfill plan
    backfill_plan = {}  # domain → list of candidates to promote
    for d, n in by_domain.items():
        avail = cand_by_domain.get(d, [])
        if len(avail) < n:
            print(f"  ⚠ {d}: need {n}, only {len(avail)} available")
            backfill_plan[d] = avail
        else:
            backfill_plan[d] = avail[:n]

    print(f"\nBackfill candidates selected:")
    for d in sorted(backfill_plan):
        print(f"\n  {d}:")
        for c in backfill_plan[d]:
            print(f"    + term_idx {c['term_idx']:5d}  {c['en']!r:30s}  k={c['k_attested_en']}/{c['k_attested_zh']}  conf={c['confidence']}")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
        return 0

    # Backup
    backup_path = LEGAL_TERMS.parent / f"legal_terms.json.bak_pre_kunder4_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(LEGAL_TERMS, backup_path)
    print(f"\nBackup: {backup_path.relative_to(REPO_ROOT)}")

    # Apply: demote kunder4 → background
    for ti, t in kunder4_terms:
        t.setdefault('tier_history', []).append({
            'date': DATE, 'from': 'core', 'to': 'background',
            'reason': 'kunder4_backfill_demote',
        })
        t['tier'] = 'background'
        # Preserve domain in audit trace, clear active domain
        t['firthian_kunder4_origin_domain'] = t['domain']
        t['domain'] = None
        t['firthian_curation_decision'] = 'kunder4_backfilled_out'

    # Apply: promote backfill candidates
    promoted = 0
    for d, cands in backfill_plan.items():
        for c in cands:
            ti = c['term_idx']
            t = terms[ti]
            if t.get('tier') != 'background':
                print(f"  ⚠ promote term_idx {ti}: current tier is {t.get('tier')}, expected 'background'")
                continue
            t.setdefault('tier_history', []).append({
                'date': DATE, 'from': 'background', 'to': 'core',
                'reason': 'kunder4_backfill_promote',
            })
            t['tier'] = 'core'
            t['domain'] = d
            t['firthian_curation_decision'] = 'kunder4_backfill_promote'
            t['firthian_curation_rationale'] = f"Strict-pool backfill replacing demoted keep_kunder4 entry. K_en={c['k_attested_en']}, K_zh={c['k_attested_zh']}, knn_conf={c['confidence']}."
            t['firthian_sub_area'] = 'backfill'  # generic sub_area; can be refined later
            promoted += 1

    # Verify final core
    final_core = [t for t in terms if t.get('tier') == 'core']
    final_dom = Counter(t.get('domain') for t in final_core)
    print(f"\nFinal core size: {len(final_core)}")
    print(f"Per-domain distribution:")
    for d in sorted(final_dom):
        print(f"  {d}: {final_dom[d]}")

    LEGAL_TERMS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"\nSaved: {LEGAL_TERMS.relative_to(REPO_ROOT)}")
    print(f"Demoted: {len(kunder4_terms)}, promoted: {promoted}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
