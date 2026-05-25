#!/usr/bin/env python3
"""
Step 2c — Encode the 100 control terms (everyday, non-legal vocabulary)
with all 10 models for §3.1.1 legal-vs-control comparison.

Control terms are pulled from the legacy `legal_terms.json` (tier='control';
they are NOT in `legal_term_run4.json` because run #4's pool is core-only).
They have no HK Cap. attestation by design, so we produce **bare encodings
only** — there is no attested-context analogue for ordinary words.

Output:
  embeddings/control_bare/index.json
  embeddings/control_bare/{model}/vecs.npy      (100, dim) L2-norm
  embeddings/control_bare/{model}/meta.json

The manifest is extended with the control snapshot + per-model output hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import EmbeddingClient  # noqa: E402

LEGACY_TERMS = REPO_ROOT / "experiments" / "data" / "processed" / "legal_terms.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def renormalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.clip(norms, 1e-12, None)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    inputs_dir = REPO_ROOT / cfg["paths"]["inputs"]
    emb_root = REPO_ROOT / cfg["paths"]["embeddings"]
    control_out = emb_root / "control_bare"
    control_out.mkdir(parents=True, exist_ok=True)

    # Snapshot the control terms from the legacy file
    with LEGACY_TERMS.open() as fh:
        legacy = json.load(fh)
    controls = [t for t in legacy["terms"] if t.get("tier") == "control"]
    if len(controls) != 100:
        raise AssertionError(f"Expected 100 control terms, found {len(controls)}")

    snapshot_path = inputs_dir / "control_terms_snapshot.json"
    with snapshot_path.open("w") as fh:
        json.dump({"terms": controls}, fh, indent=2, ensure_ascii=False)
    snapshot_sha = sha256_of(snapshot_path)
    print(f"snapshot: {snapshot_path.relative_to(REPO_ROOT)}  sha256={snapshot_sha[:16]}…")

    # Index.json for control pool — sibling to embeddings/index.json
    index = [
        {"en": t["en"], "zh": t.get("zh_clean") or t["zh_canonical"],
         "domain": t.get("domain"), "tier": "control"}
        for t in controls
    ]
    with (control_out / "index.json").open("w") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    en_texts = [t.get("en_clean") or t["en"] for t in controls]
    zh_texts = [t.get("zh_clean") or t["zh_canonical"] for t in controls]

    # Run plan: one encoding per model — EN for WEIRD + bilingual-EN, ZH for Sinic + bilingual-ZH
    jobs: list[tuple[str, str, str, list[str], int]] = []
    for m in cfg["models_weird"]:
        jobs.append((m["label"], m["id"], "en", en_texts, int(m["dim"])))
    for m in cfg["models_sinic"]:
        jobs.append((m["label"], m["id"], "zh", zh_texts, int(m["dim"])))
    for m in cfg["models_bilingual"]:
        jobs.append((f"{m['label']}-EN", m["id"], "en", en_texts, int(m["dim"])))
        jobs.append((f"{m['label']}-ZH", m["id"], "zh", zh_texts, int(m["dim"])))

    client = EmbeddingClient(
        config_path=REPO_ROOT / "experiments" / "models" / "config.yaml",
        device=cfg.get("device", "cpu"),
        batch_size=cfg.get("batch_size", 64),
    )

    t0 = time.perf_counter()
    prev_id: str | None = None
    for label, model_id, lang, texts, dim in jobs:
        out_dir = control_out / label
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.get("unload_between_models", False) and prev_id and prev_id != model_id:
            client.unload_model(prev_id)
            import gc; gc.collect()
        print(f"[{label}] encoding {len(texts)} control terms (lang={lang}, dim={dim})...")
        t1 = time.perf_counter()
        vecs = client.embed(texts, model_id, use_cache=True).astype(np.float32)
        vecs = renormalize(vecs)
        dt = time.perf_counter() - t1
        norms = np.linalg.norm(vecs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise AssertionError(f"{label} controls L2-norm deviation max={np.abs(norms-1.0).max():g}")
        np.save(out_dir / "vecs.npy", vecs)
        meta = {
            "model_id": model_id, "model_label": label, "lang": lang,
            "dim": dim, "n_terms": len(texts), "dtype": "float32",
            "l2_norm": True, "date": date.today().isoformat(),
            "elapsed_s": round(dt, 2),
            "snapshot_sha256_controls": snapshot_sha,
            "kind": "control_bare",
        }
        with (out_dir / "meta.json").open("w") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        print(f"  -> {out_dir.relative_to(REPO_ROOT)}  ({dt:.1f}s)")
        prev_id = model_id

    # Extend manifest.json with control snapshot
    manifest_path = RUN_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("control_snapshot", {
        "source": str(LEGACY_TERMS.relative_to(REPO_ROOT)),
        "snapshot": str(snapshot_path.relative_to(REPO_ROOT)),
        "size_bytes": snapshot_path.stat().st_size,
        "sha256": snapshot_sha,
        "n_terms": len(controls),
        "added_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    manifest["control_snapshot"]["sha256"] = snapshot_sha
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\ntotal: {time.perf_counter() - t0:.1f}s; manifest extended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
