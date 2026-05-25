"""
Pass-2 manual `zh_clean` overrides for gloss-artefact terms identified in
D4 step 3 user review. The DOJ glossary `zh_canonical` for these terms
is a qualified phrase, sub-type, or sentence rather than the bare
canonical ZH form, causing zero corpus matching even though the bare
form is heavily attested.

Same operational pattern as `apply_zh_overrides.py` (3 constitutional
drift terms, executed first), extended to 12 additional terms across
civil, labor_social, procedure, and constitutional domains.

D4 step 3 sub-decision (extension of pass-1 override, 2026-05-02).

Usage
-----
    python data/apply_zh_overrides_pass2.py
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

# (term_idx, expected_en, new_zh_clean, gloss_artefact_type)
OVERRIDES = [
    (107, "negligence", "疏忽", "qualifier+lemma → bare"),
    (168, "tort", "侵權", "sub-type → genus"),
    (177, "trustee", "受託人", "paraphrase → bare"),
    (32, "copyright", "版權", "sub-type → bare"),
    (43, "defamation", "誹謗", "qualifier+lemma → bare"),
    (54, "employee", "僱員", "qualifier+lemma → bare"),
    (353, "trade union", "工會", "paraphrase → bare"),
    (7, "affidavit", "誓章", "long-phrase → bare"),
    (99, "mediation", "調解", "sub-type → genus"),
    (246, "cause of action", "訴因", "sentence → bare"),
    (356, "rule of law", "法治", "literal-translation → idiomatic bare"),
    (361, "Resolution of the Legislative Council", "立法會決議", "old-name → modern HKSAR name"),
]

ZH_MODEL_SLOTS = [
    ("BGE-ZH-large", "BAAI/bge-large-zh-v1.5"),
    ("Text2vec-large-ZH", "GanymedeNil/text2vec-large-chinese"),
    ("Dmeta-ZH", "DMetaSoul/Dmeta-embedding-zh"),
    ("BGE-M3-ZH", "BAAI/bge-m3"),
    ("Qwen3-0.6B-ZH", "Qwen/Qwen3-Embedding-0.6B"),
]


def main() -> int:
    print(f"Step 1: Update zh_clean for {len(OVERRIDES)} gloss-artefact terms ...")
    payload = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))
    terms = payload["terms"]

    already_done = all(terms[ti].get("zh_clean_overridden") and terms[ti]["zh_clean"] == new_zh
                       for ti, _, new_zh, _ in OVERRIDES)
    if already_done:
        print("  All overrides already applied. Skipping Step 1.")
    else:
        backup_path = LEGAL_TERMS.parent / f"legal_terms.json.bak_pre_zh_override_pass2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(LEGAL_TERMS, backup_path)
        print(f"  Backup: {backup_path.relative_to(REPO_ROOT)}")

        for ti, expected_en, new_zh, kind in OVERRIDES:
            t = terms[ti]
            if t["en"] != expected_en:
                raise RuntimeError(f"Sanity fail: term_idx {ti} en={t['en']!r}, expected {expected_en!r}")
            old = t.get("zh_clean", "")
            t["zh_clean"] = new_zh
            t["zh_clean_overridden"] = True
            t["zh_clean_override_reason"] = f"DOJ glossary gloss-artefact ({kind}); canonical bare ZH applied"
            print(f"  term_idx {ti} ({expected_en}): {old!r} → {new_zh!r}  [{kind}]")

        LEGAL_TERMS.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {LEGAL_TERMS.relative_to(REPO_ROOT)}")

    print(f"\nStep 2: Re-encode {len(OVERRIDES)} target rows in 5 ZH-side slots ...")
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from shared.embeddings import EmbeddingClient

    client = EmbeddingClient(str(CONFIG), device="cpu")

    new_zh_strings = [new_zh for _, _, new_zh, _ in OVERRIDES]
    target_indices = [ti for ti, _, _, _ in OVERRIDES]

    bak_dir = REPO_ROOT / "data" / "processed" / "embeddings.bak_pre_clean_20260501_222806"

    for label, model_id in ZH_MODEL_SLOTS:
        vec_path = EMBEDDINGS_DIR / label / "vectors.npy"
        if not vec_path.exists():
            print(f"  WARNING: {vec_path.relative_to(REPO_ROOT)} not found, skip")
            continue

        # Idempotency: skip if all 12 target rows already differ meaningfully from pre-D2 backup
        bak_path = bak_dir / label / "vectors.npy"
        if bak_path.exists():
            cur = np.load(vec_path)
            bak = np.load(bak_path)
            diffs = [float(np.linalg.norm(cur[ti] - bak[ti])) for ti in target_indices]
            if all(d > 0.1 for d in diffs):
                print(f"  {label}: already patched (all 12 row diffs > 0.1), skip")
                continue
            unpatched = [ti for ti, d in zip(target_indices, diffs) if d <= 0.1]
            print(f"  {label}: {len(unpatched)}/{len(target_indices)} rows still need patching ...")
        else:
            print(f"  {label}: re-encoding all {len(target_indices)} rows ...")

        new_vecs = client.embed(new_zh_strings, model_id, use_cache=False).astype(np.float32)
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        new_vecs = new_vecs / np.clip(norms, 1e-12, None)

        vecs = np.load(vec_path)
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)
        for i, ti in enumerate(target_indices):
            vecs[ti] = new_vecs[i]
        np.save(vec_path, vecs)
        print(f"    {label}: patched {len(target_indices)} rows, saved")

    print(f"\nStep 2 done. Run sync_indexes.py + build_term_contexts.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
