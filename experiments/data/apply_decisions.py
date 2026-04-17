"""Apply the 7 per-domain curation decisions to legal_terms.json.

For each decisions_<domain>.json:
  - action == "drop": matching core terms have tier flipped to "background"
  - action == "promote": matching background terms have tier flipped to
    "core" and domain set to the target domain

A timestamped backup of legal_terms.json is written before any change.
A human-readable report is written to data/processed/rebalance_report.md.

Usage:
    python experiments/data/apply_decisions.py
    python experiments/data/apply_decisions.py --dry-run   # preview only
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "data" / "processed"
REVIEW = REPO / "experiments" / "data" / "review"
LEGAL = DATA / "legal_terms.json"
REPORT = DATA / "rebalance_report.md"

DOMAINS = [
    "civil", "criminal", "procedure", "international",
    "constitutional", "administrative", "labor_social",
]


def domain_counts(terms, tier="core"):
    return Counter(t["domain"] for t in terms if t.get("tier") == tier)


def apply(dry_run: bool = False) -> None:
    legal = json.loads(LEGAL.read_text())
    terms = legal["terms"]
    en_index = {t["en"]: t for t in terms}  # terms are unique by EN

    pre_counts = domain_counts(terms, "core")
    pre_total = sum(pre_counts.values())

    drops_applied = []
    promotes_applied = []
    errors = []

    for domain in DOMAINS:
        dec = json.loads((REVIEW / f"decisions_{domain}.json").read_text())
        action = dec["action"]

        if action == "drop":
            for entry in dec["drop"]:
                en = entry["en"]
                t = en_index.get(en)
                if t is None:
                    errors.append(f"[{domain}] drop: EN '{en}' not in dataset")
                    continue
                if t.get("tier") != "core" or t.get("domain") != domain:
                    errors.append(
                        f"[{domain}] drop: EN '{en}' is not core/{domain} "
                        f"(was tier={t.get('tier')}, domain={t.get('domain')})"
                    )
                    continue
                t["tier"] = "background"
                drops_applied.append((domain, en, entry.get("rationale", "")))

        elif action == "promote":
            for entry in dec["promote"]:
                en = entry["en"]
                t = en_index.get(en)
                if t is None:
                    errors.append(f"[{domain}] promote: EN '{en}' not in dataset")
                    continue
                if t.get("tier") != "background":
                    errors.append(
                        f"[{domain}] promote: EN '{en}' is not background "
                        f"(was tier={t.get('tier')}, domain={t.get('domain')})"
                    )
                    continue
                t["tier"] = "core"
                t["domain"] = domain
                promotes_applied.append((domain, en, entry.get("rationale", "")))
        else:
            errors.append(f"[{domain}] unknown action: {action}")

    post_counts = domain_counts(terms, "core")
    post_total = sum(post_counts.values())

    # Sanity checks
    expected_per_domain = 50
    bad_domains = [d for d in DOMAINS if post_counts[d] != expected_per_domain]

    print(f"Pre  core total: {pre_total}")
    print(f"Post core total: {post_total}")
    print(f"Drops applied:    {len(drops_applied)}")
    print(f"Promotes applied: {len(promotes_applied)}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    if bad_domains:
        print(f"\nDomain mismatch: {bad_domains}")
        for d in bad_domains:
            print(f"  {d}: {post_counts[d]} (expected {expected_per_domain})")

    if errors or bad_domains:
        print("\nABORT: not writing. Fix errors above first.")
        return

    print("\n--- per-domain counts (core) ---")
    for d in DOMAINS:
        print(f"  {d:<18} {pre_counts[d]:>4} → {post_counts[d]:>4}  (Δ {post_counts[d] - pre_counts[d]:+d})")

    if dry_run:
        print("\n[dry-run] no file written.")
        return

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = LEGAL.with_suffix(f".json.bak_{ts}")
    shutil.copy2(LEGAL, backup)
    print(f"\nBackup: {backup.relative_to(REPO)}")

    # Write new legal_terms.json
    LEGAL.write_text(json.dumps(legal, ensure_ascii=False, indent=2))
    print(f"Wrote: {LEGAL.relative_to(REPO)}")

    # Write human-readable report
    lines = [
        "# Core rebalance report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Backup**: `{backup.name}`",
        f"**Script**: `experiments/data/apply_decisions.py`",
        f"**Trace**: see `experiments/trace_pivot_2lens.md` D5.",
        "",
        "## Summary",
        "",
        f"- Total core before: **{pre_total}**",
        f"- Total core after:  **{post_total}**",
        f"- Dropped (core → background): **{len(drops_applied)}**",
        f"- Promoted (background → core): **{len(promotes_applied)}**",
        f"- Target per domain: **{expected_per_domain}** (γ policy)",
        "",
        "## Per-domain counts",
        "",
        "| Domain | Before | After | Δ |",
        "|---|---:|---:|---:|",
    ]
    for d in DOMAINS:
        delta = post_counts[d] - pre_counts[d]
        lines.append(f"| {d} | {pre_counts[d]} | {post_counts[d]} | {delta:+d} |")

    lines += ["", "## Drops (core → background)", ""]
    for d in DOMAINS:
        d_drops = [(en, r) for dom, en, r in drops_applied if dom == d]
        if not d_drops:
            continue
        lines += [f"### {d} ({len(d_drops)})", ""]
        for en, r in d_drops:
            lines.append(f"- **{en}** — {r}")
        lines.append("")

    lines += ["## Promotions (background → core)", ""]
    for d in DOMAINS:
        d_proms = [(en, r) for dom, en, r in promotes_applied if dom == d]
        if not d_proms:
            continue
        lines += [f"### {d} ({len(d_proms)})", ""]
        for en, r in d_proms:
            lines.append(f"- **{en}** — {r}")
        lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"Wrote: {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply(dry_run=args.dry_run)
