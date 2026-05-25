"""
Apply manual `zh_clean` overrides for terms whose DOJ glossary entry maps
to a wrong-sense ZH form (mismatched semantically to the EN headword).

Scope: 3 terms in the constitutional domain flagged by the per-domain
curation agent (D4 step 2):

  term_idx 72   `human rights`  zh_canonical='《歐洲人權公約》' → zh_clean='人權'
  term_idx 2230 `liberty`       zh_canonical='任何一方均可提出申請' → zh_clean='自由'
  term_idx 4113 `committee`     zh_canonical='受託監管人' → zh_clean='委員會'

Each override is a paraphrasis of the canonical bare ZH form for the EN
sense (not from the DOJ glossary). The original `zh_canonical` is
preserved for citation; only `zh_clean` is overridden.

After applying: re-encode the 5 ZH-side model slots for these 3 rows
only, re-run sync_indexes.py, and refresh term_contexts.jsonl for these
terms. The constitutional decision JSON should be updated accordingly.

D4 step 3 sub-decision (post per-domain curation, 2026-05-01).

Usage
-----
    python data/apply_zh_overrides.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS = REPO_ROOT / "data" / "processed" / "legal_terms.json"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "processed" / "embeddings"
CONFIG = REPO_ROOT / "models" / "config.yaml"

# (term_idx, expected_en, new_zh_clean)
OVERRIDES = [
    (72, "human rights", "人權"),
    (2230, "liberty", "自由"),
    (4113, "committee", "委員會"),
]

# ZH-side model slots to patch (5 total)
ZH_MODEL_SLOTS = [
    ("BGE-ZH-large", "BAAI/bge-large-zh-v1.5"),
    ("Text2vec-large-ZH", "GanymedeNil/text2vec-large-chinese"),
    ("Dmeta-ZH", "DMetaSoul/Dmeta-embedding-zh"),
    ("BGE-M3-ZH", "BAAI/bge-m3"),
    ("Qwen3-0.6B-ZH", "Qwen/Qwen3-Embedding-0.6B"),
]


def main() -> int:
    print("Step 1: Update legal_terms.json zh_clean for 3 override terms ...")
    payload = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))
    terms = payload["terms"]

    # Idempotent: skip Step 1 if already overridden
    already_done = all(terms[ti].get("zh_clean_overridden") and terms[ti]["zh_clean"] == new_zh
                       for ti, _, new_zh in OVERRIDES)
    if already_done:
        print("  All 3 terms already have zh_clean_overridden=True with target values. Skipping Step 1.")
    else:
        backup_path = LEGAL_TERMS.parent / f"legal_terms.json.bak_pre_zh_override_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(LEGAL_TERMS, backup_path)
        print(f"  Backup: {backup_path.relative_to(REPO_ROOT)}")

        for ti, expected_en, new_zh in OVERRIDES:
            t = terms[ti]
            if t["en"] != expected_en:
                raise RuntimeError(f"Sanity fail: term_idx {ti} en={t['en']!r}, expected {expected_en!r}")
            old = t.get("zh_clean", "")
            t["zh_clean"] = new_zh
            t["zh_clean_overridden"] = True
            t["zh_clean_override_reason"] = "DOJ glossary zh_canonical maps to wrong-sense ZH form; canonical bare ZH applied"
            print(f"  term_idx {ti} ({expected_en}): {old!r} → {new_zh!r}")

        LEGAL_TERMS.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {LEGAL_TERMS.relative_to(REPO_ROOT)}")

    print("\nStep 2: Re-encode targeted rows in 5 ZH-side model slots ...")
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from shared.embeddings import EmbeddingClient

    client = EmbeddingClient(str(CONFIG), device="cpu")

    new_zh_strings = [new_zh for _, _, new_zh in OVERRIDES]
    target_indices = [ti for ti, _, _ in OVERRIDES]

    # Skip a slot if its 3 target rows already differ meaningfully from the pre-D2 backup
    # (= already patched in a prior run). Threshold: 0.1 in L2 norm.
    bak_dir = REPO_ROOT / "data" / "processed" / "embeddings.bak_pre_clean_20260501_222806"

    for label, model_id in ZH_MODEL_SLOTS:
        vec_path = EMBEDDINGS_DIR / label / "vectors.npy"
        if not vec_path.exists():
            print(f"  WARNING: {vec_path.relative_to(REPO_ROOT)} not found, skip")
            continue

        # Idempotency check
        bak_path = bak_dir / label / "vectors.npy"
        if bak_path.exists():
            cur = np.load(vec_path)
            bak = np.load(bak_path)
            diffs = [float(np.linalg.norm(cur[ti] - bak[ti])) for ti in target_indices]
            if all(d > 0.1 for d in diffs):
                print(f"  {label}: already patched (row diffs vs bak: {[round(d,3) for d in diffs]}), skip")
                continue

        print(f"  Re-encoding {label} for {len(new_zh_strings)} strings ...")
        new_vecs = client.embed(new_zh_strings, model_id, use_cache=False).astype(np.float32)
        # Defensive normalization (EmbeddingClient already L2-normalizes)
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        new_vecs = new_vecs / np.clip(norms, 1e-12, None)

        # Load existing matrix, patch rows, save back
        vecs = np.load(vec_path)
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)
        for i, ti in enumerate(target_indices):
            vecs[ti] = new_vecs[i]
        np.save(vec_path, vecs)
        print(f"    Patched rows {target_indices}, saved {vec_path.relative_to(REPO_ROOT)}")

    print("\nStep 3: Done. Run `sync_indexes.py` and `build_term_contexts.py` next to refresh metadata + attestation counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
