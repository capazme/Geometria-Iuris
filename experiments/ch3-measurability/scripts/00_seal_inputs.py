#!/usr/bin/env python3
"""
Step 0 — Seal inputs for run #4 (post-BLP final).

Copies the seven frozen input files into experiments/ch3-measurability/inputs/,
computes SHA-256 of each snapshot, writes manifest.json, and verifies
the integrity gate (364 KEEP terms in legal_term_run4.json, 358/364 with
k_min_postBLP >= 4).

Idempotent: re-running overwrites the snapshots and rewrites manifest.json
deterministically (the hashes only change if the source files change).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
INPUTS_DIR = RUN_DIR / "inputs"

SOURCES: list[tuple[Path, str]] = [
    # Core 364 pool (post-BLP curated terms — the headline pool of §3)
    (REPO_ROOT / "experiments" / "data" / "processed" / "legal_term_run4.json",
     "core_terms_snapshot.json"),
    # ZH overrides applied during curation (audit trail)
    (REPO_ROOT / "experiments" / "data" / "processed" / "zh_overrides_postBLP.json",
     "zh_overrides_snapshot.json"),
    # Cap. enactment years (used to filter post-1989 BLP enactments)
    (REPO_ROOT / "experiments" / "data" / "processed" / "cap_enactment_years.json",
     "cap_enactment_years_snapshot.json"),
    # Per-term attested contexts (post-1989, K≥1) — for attested encoding
    (REPO_ROOT / "experiments" / "data" / "processed" / "elegislation" / "term_contexts_postBLP.jsonl",
     "term_contexts_bilingual_snapshot.jsonl"),
    # Per-term context coverage (K_min by term)
    (REPO_ROOT / "experiments" / "data" / "processed" / "elegislation" / "coverage_postBLP.json",
     "context_coverage_snapshot.json"),
    # Curation rationale document
    (REPO_ROOT / "experiments" / "data" / "trace_postBLP_curation.md",
     "trace_curation_snapshot.md"),
    # 6 Kozlowski-style value axes for Experiment 2
    (REPO_ROOT / "experiments" / "lens_4_values" / "value_axes.yaml",
     "value_axes_snapshot.yaml"),
    # 100 control terms (everyday vocabulary) — bare-only by design (no HK Cap. attestation)
    # Extracted in step 02c. Sealed here as a snapshot for audit.
    # NOTE: control_terms_snapshot.json is built by 02c_encode_control.py from
    # the legacy legal_terms.json (tier='control'); see that script for details.
    # 9.045 background terms (legalish residual). Snapshot built by 02_encode_background.py.
    # bg_terms_snapshot.json and bg_contexts_snapshot.jsonl are sealed there.
]


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def copy_and_hash(src: Path, dst: Path) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"Missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src.relative_to(REPO_ROOT)),
        "snapshot": str(dst.relative_to(REPO_ROOT)),
        "size_bytes": dst.stat().st_size,
        "sha256": sha256_of(dst),
    }


def verify_terms(legal_terms_snapshot: Path) -> dict:
    with legal_terms_snapshot.open() as fh:
        data = json.load(fh)
    terms = data["terms"] if isinstance(data, dict) and "terms" in data else data
    if not isinstance(terms, list):
        raise ValueError("legal_term_run4 snapshot has unexpected schema")
    n_total = len(terms)
    tiers: dict[str, int] = {}
    domains: dict[str, int] = {}
    for t in terms:
        tiers[t.get("tier", "?")] = tiers.get(t.get("tier", "?"), 0) + 1
        domains[t.get("domain", "?")] = domains.get(t.get("domain", "?"), 0) + 1
    if n_total != 364:
        raise AssertionError(f"Expected 364 terms, found {n_total}")
    if tiers.get("core") != 364:
        raise AssertionError(f"Expected 364 core, found {tiers}")
    return {"n_total": n_total, "tiers": tiers, "domains": domains}


def verify_coverage(coverage_snapshot: Path) -> dict:
    with coverage_snapshot.open() as fh:
        cov = json.load(fh)
    per_term = cov["per_term"]
    kmins = [v["k_min"] for v in per_term.values()]
    n_ge4 = sum(1 for k in kmins if k >= 4)
    n_lt4 = sum(1 for k in kmins if k < 4)
    if len(per_term) != 364:
        raise AssertionError(f"Coverage per_term has {len(per_term)} entries, expected 364")
    return {
        "n_total": len(per_term),
        "n_k_ge_4": n_ge4,
        "n_k_lt_4": n_lt4,
        "meta": cov.get("_meta"),
    }


def main() -> int:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    snapshots: list[dict] = []
    for src, dst_name in SOURCES:
        snapshots.append(copy_and_hash(src, INPUTS_DIR / dst_name))

    terms_check = verify_terms(INPUTS_DIR / "core_terms_snapshot.json")
    coverage_check = verify_coverage(INPUTS_DIR / "context_coverage_snapshot.json")

    hashes = {s["snapshot"]: s["sha256"] for s in snapshots}
    if len(set(hashes.values())) != len(hashes):
        raise AssertionError("Snapshots have duplicate hashes — input list is wrong")

    manifest_path = RUN_DIR / "manifest.json"
    # Merge with existing manifest to preserve fields written by downstream
    # scripts (control_snapshot, bg_snapshot, outputs, embeddings, …).
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"run_id": "ch3-measurability"}

    manifest["step"] = "00_seal_inputs"
    manifest["sealed_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["snapshots"] = snapshots
    manifest["verification"] = {
        "core_terms": terms_check,
        "context_coverage": coverage_check,
    }

    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"sealed {len(snapshots)} snapshots into {INPUTS_DIR.relative_to(REPO_ROOT)}")
    print(f"manifest -> {manifest_path.relative_to(REPO_ROOT)}")
    print(f"terms: {terms_check['n_total']} core ({terms_check['domains']})")
    print(f"coverage: {coverage_check['n_k_ge_4']}/{coverage_check['n_total']} with k_min>=4 "
          f"({coverage_check['n_k_lt_4']} below)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
