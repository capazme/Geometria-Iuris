"""
Apply per-domain Firthian curation decisions to legal_terms.json.

For each of 7 domains, reads `data/review/firthian_decisions_<domain>.json`
and applies:
  - Drops: term[tier] = 'background', domain cleared, tier_history note added
  - Promotes: term[tier] = 'core', domain set to the curation domain

keep_strict and keep_kunder4 entries remain as core (no change needed).

D4 step 4 of `experiments/trace_firthian_pivot.md`.

Audit fields added per term:
  - tier_history: list of {date, from, to, reason}
  - firthian_curation_decision: 'keep_strict' | 'keep_kunder4' | 'drop' | 'promote'
  - firthian_curation_rationale: copied from decision file (for kunder4/drop/promote)
  - firthian_sub_area: copied from decision file (for keep_*/promote)

Usage
-----
    python data/apply_firthian_decisions.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS = REPO_ROOT / "data" / "processed" / "legal_terms.json"
REVIEW = REPO_ROOT / "data" / "review"

DOMAINS = ['administrative', 'civil', 'constitutional', 'criminal', 'international', 'labor_social', 'procedure']
DATE = "2026-05-02"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    payload = json.loads(LEGAL_TERMS.read_text(encoding='utf-8'))
    terms = payload['terms']
    print(f"Loaded {len(terms)} terms")

    if not args.dry_run:
        backup_path = LEGAL_TERMS.parent / f"legal_terms.json.bak_pre_firthian_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(LEGAL_TERMS, backup_path)
        print(f"Backup: {backup_path.relative_to(REPO_ROOT)}")

    # Plan changes (term_idx → operation)
    plan_drop = {}  # term_idx → (domain, rationale)
    plan_promote = {}  # term_idx → (domain, rationale, sub_area)
    plan_keep_strict = {}  # term_idx → (domain, sub_area)
    plan_keep_kunder4 = {}  # term_idx → (domain, sub_area, rationale)

    for d in DOMAINS:
        j = json.loads((REVIEW / f'firthian_decisions_{d}.json').read_text())
        dec = j['decisions']
        for e in dec['keep_strict']:
            plan_keep_strict[e['term_idx']] = (d, e['sub_area'])
        for e in dec['keep_kunder4']:
            plan_keep_kunder4[e['term_idx']] = (d, e['sub_area'], e['rationale'])
        for e in dec['drop']:
            plan_drop[e['term_idx']] = (d, e['rationale'])
        for e in dec['promote']:
            plan_promote[e['term_idx']] = (d, e.get('rationale',''), e['sub_area'])

    print(f"\nPlan summary:")
    print(f"  keep_strict (no change):  {len(plan_keep_strict)}")
    print(f"  keep_kunder4 (no change): {len(plan_keep_kunder4)}")
    print(f"  drop (core → background): {len(plan_drop)}")
    print(f"  promote (background → core): {len(plan_promote)}")

    # Sanity: kept + drop should equal current core, promote should equal needed
    current_core = {i for i, t in enumerate(terms) if t.get('tier') == 'core'}
    print(f"\nCurrent core: {len(current_core)}")
    print(f"  kept (strict + kunder4): {len(plan_keep_strict) + len(plan_keep_kunder4)}")
    print(f"  drop: {len(plan_drop)}")
    print(f"  expected total: {len(plan_keep_strict) + len(plan_keep_kunder4) + len(plan_drop)} (should equal current core 350)")

    # Apply changes
    n_drop_applied = 0
    n_promo_applied = 0
    n_keep_decorated = 0

    for ti, (domain, rationale) in plan_drop.items():
        t = terms[ti]
        if t.get('tier') != 'core':
            print(f"  ⚠ drop term_idx {ti}: current tier is {t.get('tier')}, expected 'core'")
            continue
        t.setdefault('tier_history', []).append({
            'date': DATE, 'from': 'core', 'to': 'background',
            'reason': 'firthian_curation_drop',
        })
        t['tier'] = 'background'
        t['firthian_curation_decision'] = 'drop'
        t['firthian_curation_rationale'] = rationale
        t['firthian_curation_domain_origin'] = domain  # was in this domain
        # Keep domain field for now (audit trace), or clear it. We'll clear it.
        t['domain'] = None
        n_drop_applied += 1

    for ti, (domain, rationale, sub_area) in plan_promote.items():
        t = terms[ti]
        if t.get('tier') != 'background':
            print(f"  ⚠ promote term_idx {ti}: current tier is {t.get('tier')}, expected 'background'")
            continue
        t.setdefault('tier_history', []).append({
            'date': DATE, 'from': 'background', 'to': 'core',
            'reason': 'firthian_curation_promote',
        })
        t['tier'] = 'core'
        t['domain'] = domain
        t['firthian_curation_decision'] = 'promote'
        t['firthian_curation_rationale'] = rationale
        t['firthian_sub_area'] = sub_area
        n_promo_applied += 1

    for ti, (domain, sub_area) in plan_keep_strict.items():
        t = terms[ti]
        t['firthian_curation_decision'] = 'keep_strict'
        t['firthian_sub_area'] = sub_area
        n_keep_decorated += 1

    for ti, (domain, sub_area, rationale) in plan_keep_kunder4.items():
        t = terms[ti]
        t['firthian_curation_decision'] = 'keep_kunder4'
        t['firthian_curation_rationale'] = rationale
        t['firthian_sub_area'] = sub_area
        n_keep_decorated += 1

    print(f"\nApplied:")
    print(f"  drops: {n_drop_applied}")
    print(f"  promotes: {n_promo_applied}")
    print(f"  keep decorations: {n_keep_decorated}")

    # Verify final core
    final_core = [t for t in terms if t.get('tier') == 'core']
    from collections import Counter
    final_dom = Counter(t.get('domain') for t in final_core)
    print(f"\nFinal core size: {len(final_core)}")
    print(f"Final per-domain distribution:")
    for d in sorted(final_dom):
        print(f"  {d}: {final_dom[d]}")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
        return 0

    LEGAL_TERMS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"\nSaved: {LEGAL_TERMS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
