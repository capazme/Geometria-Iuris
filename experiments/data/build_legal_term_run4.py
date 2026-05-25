"""
Produce `legal_term_run4.json` — minimal run-ready dataset.

Schema (per term): {en, zh, domain, tier, doj_divisions, source}.
Only the fields actually consumed by the encoders / experiments are kept.
`zh` is the post-BLP form (override applied if present; cleaned Firthian
zh_clean otherwise). Filter: only the 364 post-BLP KEEP terms.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/Users/gpuzio/Desktop/CODE/THESIS/experiments")
PROC = ROOT / "data" / "processed"

LEGAL_TERMS = PROC / "legal_terms.json"
CSV_LONGLIST = PROC / "postBLP_curation_longlist.csv"
ZH_OVERRIDES = PROC / "zh_overrides_postBLP.json"
OUT = PROC / "legal_term_run4.json"


def main() -> int:
    lt = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))["terms"]

    by_en_zh: dict[tuple[str, str], int] = {}
    for i, t in enumerate(lt):
        en = t.get("en", "")
        for zh_field in ("zh_clean", "zh_canonical"):
            by_en_zh.setdefault((en, t.get(zh_field, "")), i)

    keep_rows = [r for r in csv.DictReader(CSV_LONGLIST.open(encoding="utf-8"))
                 if r["curation_decision"] == "KEEP"]

    ov_raw = json.loads(ZH_OVERRIDES.read_text(encoding="utf-8"))
    overrides: dict[tuple[str, str], str] = {}
    drops: set[tuple[str, str]] = set()
    for k, v in ov_raw["overrides"].items():
        domain, en = k.split("|", 1)
        if v.get("level") == "DROP":
            drops.add((domain, en))
        elif v.get("zh_override"):
            overrides[(domain, en)] = v["zh_override"]

    out_terms = []
    for r in keep_rows:
        if (r["domain"], r["en"]) in drops:
            continue
        idx = by_en_zh.get((r["en"], r["zh"]))
        if idx is None:
            continue
        src = lt[idx]
        zh = overrides.get((r["domain"], r["en"])) or src.get("zh_clean") or src.get("zh_canonical", "")
        out_terms.append({
            "en": src.get("en_clean") or src.get("en", ""),
            "zh": zh,
            "domain": r["domain"],
            "tier": "core",
            "doj_divisions": src.get("doj_divisions", []),
            "source": src.get("source", ""),
        })

    OUT.write_text(json.dumps({"terms": out_terms}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    size_mb = OUT.stat().st_size / 1_000_000
    print(f"Wrote {OUT.name} ({size_mb:.2f} MB, {len(out_terms)} terms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
