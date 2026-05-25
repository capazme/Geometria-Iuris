#!/usr/bin/env python3
"""
Step 2d — Encode background terms for the extension experiments (D, E, F, G, H, A).

Background terms (tier='background' in `legal_terms.json`, ~9045 entries) are
the legalish residual the run-#4 post-BLP curation did NOT promote to core.
They are NOT in `legal_term_run4.json`. We snapshot them, encode them, and
expose them to the extension experiments.

  embeddings/bg/index.json                 ordered list of {en, zh, K_en, K_zh}
  embeddings/bg/{model}/vecs_bare.npy      (N_bg, dim) all rows valid (every bg has en/zh)
  embeddings/bg/{model}/vecs_attested.npy  (N_bg, dim) row valid iff K_lang >= 1; else zero
  embeddings/bg/{model}/coverage.json      per-term n_contexts_en/zh + k_min

Snapshots in inputs/:
  inputs/bg_terms_snapshot.json     the 9045 bg records (subset of legal_terms.json)
  inputs/bg_contexts_snapshot.jsonl the matching context records (subset of term_contexts.jsonl)

Manifest is extended with both snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
LEGACY_CTXS = REPO_ROOT / "experiments" / "data" / "processed" / "elegislation" / "term_contexts.jsonl"


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
    parser.add_argument("--max-contexts", type=int, default=8,
                        help="Max attested contexts per bg term per lang")
    parser.add_argument("--max-context-chars", type=int, default=120)
    parser.add_argument("--bare-only", action="store_true",
                        help="Skip attested encoding (fast smoke)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to subset of model labels")
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    inputs_dir = REPO_ROOT / cfg["paths"]["inputs"]
    out_root = REPO_ROOT / cfg["paths"]["embeddings"] / "bg"
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Snapshot bg terms + their contexts ---
    print("[snapshot] loading legacy terms + contexts...")
    with LEGACY_TERMS.open() as fh:
        legacy = json.load(fh)["terms"]
    ctxs_by_idx: dict[int, dict] = {}
    with LEGACY_CTXS.open() as fh:
        for line in fh:
            rec = json.loads(line)
            ctxs_by_idx[rec["term_idx"]] = rec

    bg_records: list[dict] = []
    bg_ctxs: list[dict] = []
    for i, t in enumerate(legacy):
        if t.get("tier") != "background":
            continue
        bg_records.append({
            "legacy_idx": i,
            "en": t["en"],
            "zh": t.get("zh_clean") or t["zh_canonical"],
            "en_clean": t.get("en_clean") or t["en"],
            "zh_clean": t.get("zh_clean") or t["zh_canonical"],
            "domain": t.get("domain"),
            "doj_divisions": t.get("doj_divisions"),
        })
        rec = ctxs_by_idx.get(i, {})
        bg_ctxs.append({
            "legacy_idx": i,
            "en": t["en"],
            "k_en": len(rec.get("en_contexts", [])),
            "k_zh": len(rec.get("zh_contexts", [])),
            "en_contexts": rec.get("en_contexts", [])[:args.max_contexts],
            "zh_contexts": rec.get("zh_contexts", [])[:args.max_contexts],
        })

    bg_snapshot = inputs_dir / "bg_terms_snapshot.json"
    with bg_snapshot.open("w") as fh:
        json.dump({"terms": bg_records}, fh, indent=2, ensure_ascii=False)
    bg_snap_sha = sha256_of(bg_snapshot)

    bg_ctxs_snapshot = inputs_dir / "bg_contexts_snapshot.jsonl"
    with bg_ctxs_snapshot.open("w") as fh:
        for r in bg_ctxs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    bg_ctxs_snap_sha = sha256_of(bg_ctxs_snapshot)

    print(f"[snapshot] {len(bg_records)} bg terms sealed (sha256={bg_snap_sha[:12]}…)")
    print(f"[snapshot] {len(bg_ctxs)} bg context records sealed (sha256={bg_ctxs_snap_sha[:12]}…)")

    # --- Index for bg pool ---
    index = []
    for r, c in zip(bg_records, bg_ctxs):
        index.append({
            "en": r["en"], "zh": r["zh"], "domain": r["domain"],
            "k_en": c["k_en"], "k_zh": c["k_zh"], "k_min": min(c["k_en"], c["k_zh"]),
        })
    with (out_root / "index.json").open("w") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    n_bg = len(bg_records)
    en_texts = [r["en_clean"] for r in bg_records]
    zh_texts = [r["zh_clean"] for r in bg_records]

    # --- Build run plan: 10 jobs (3 WEIRD + 3 Sinic + 2 bilingual × 2 lati) ---
    jobs: list[tuple[str, str, str, list[str], int]] = []
    for m in cfg["models_weird"]:
        jobs.append((m["label"], m["id"], "en", en_texts, int(m["dim"])))
    for m in cfg["models_sinic"]:
        jobs.append((m["label"], m["id"], "zh", zh_texts, int(m["dim"])))
    for m in cfg["models_bilingual"]:
        jobs.append((f"{m['label']}-EN", m["id"], "en", en_texts, int(m["dim"])))
        jobs.append((f"{m['label']}-ZH", m["id"], "zh", zh_texts, int(m["dim"])))

    if args.models is not None:
        jobs = [j for j in jobs if j[0] in args.models]

    client = EmbeddingClient(
        config_path=REPO_ROOT / "experiments" / "models" / "config.yaml",
        device=cfg.get("device", "cpu"),
        batch_size=cfg.get("batch_size", 64),
    )

    overall_t0 = time.perf_counter()
    prev_id: str | None = None
    for label, model_id, lang, bare_texts, dim in jobs:
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.get("unload_between_models", False) and prev_id and prev_id != model_id:
            client.unload_model(prev_id)
            import gc; gc.collect()

        print(f"\n=== {label} (lang={lang}, dim={dim}) ===")
        t0 = time.perf_counter()
        bare = client.embed(bare_texts, model_id, use_cache=True).astype(np.float32)
        bare = renormalize(bare)
        dt_bare = time.perf_counter() - t0
        print(f"  bare encoded in {dt_bare:.1f}s, shape={bare.shape}")
        np.save(out_dir / "vecs_bare.npy", bare)

        attested = np.zeros((n_bg, dim), dtype=np.float32)
        k_used: list[int] = []
        dt_att = 0.0
        if not args.bare_only:
            ctx_key = "en_contexts" if lang == "en" else "zh_contexts"
            flat: list[str] = []
            offsets: list[tuple[int, int]] = []
            cursor = 0
            for c in bg_ctxs:
                ctxs = [s["context"][:args.max_context_chars] for s in c[ctx_key]]
                offsets.append((cursor, cursor + len(ctxs)))
                flat.extend(ctxs)
                cursor += len(ctxs)
                k_used.append(len(ctxs))

            if flat:
                print(f"  attested: encoding {len(flat)} contexts...")
                t1 = time.perf_counter()
                flat_vecs = client.embed(flat, model_id, use_cache=False).astype(np.float32)
                dt_att = time.perf_counter() - t1
                print(f"    encoded in {dt_att:.1f}s")
                for j, (s, e) in enumerate(offsets):
                    if e == s:
                        continue
                    mean_vec = flat_vecs[s:e].mean(axis=0)
                    nrm = np.linalg.norm(mean_vec)
                    if nrm > 1e-12:
                        attested[j] = (mean_vec / nrm).astype(np.float32)
            np.save(out_dir / "vecs_attested.npy", attested)

        coverage = {
            "lang": lang,
            "n_bg": n_bg,
            "n_with_context": int(sum(1 for k in k_used if k >= 1)) if k_used else 0,
            "n_K_ge_4": int(sum(1 for k in k_used if k >= 4)) if k_used else 0,
            "n_K_lt_1": int(sum(1 for k in k_used if k == 0)) if k_used else n_bg,
            "per_term": [
                {"en": r["en"], "zh": r["zh"], "n_contexts": int(k)}
                for r, k in zip(bg_records, k_used)
            ] if k_used else [],
        }
        with (out_dir / "coverage.json").open("w") as fh:
            json.dump(coverage, fh, indent=2, ensure_ascii=False)

        meta = {
            "model_id": model_id, "model_label": label, "lang": lang,
            "dim": dim, "n_bg": n_bg, "dtype": "float32", "l2_norm": True,
            "date": date.today().isoformat(),
            "elapsed_bare_s": round(dt_bare, 2),
            "elapsed_attested_s": round(dt_att, 2),
            "snapshot_sha256_bg": bg_snap_sha,
            "kind": "bg",
        }
        with (out_dir / "meta.json").open("w") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        prev_id = model_id

    print(f"\nTotal bg encoding: {time.perf_counter() - overall_t0:.1f}s")

    # Update manifest
    manifest_path = RUN_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bg_snapshot"] = {
        "source_terms": str(LEGACY_TERMS.relative_to(REPO_ROOT)),
        "source_contexts": str(LEGACY_CTXS.relative_to(REPO_ROOT)),
        "snapshot_terms": str(bg_snapshot.relative_to(REPO_ROOT)),
        "snapshot_contexts": str(bg_ctxs_snapshot.relative_to(REPO_ROOT)),
        "sha256_terms": bg_snap_sha,
        "sha256_contexts": bg_ctxs_snap_sha,
        "n_terms": n_bg,
        "added_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("manifest extended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
