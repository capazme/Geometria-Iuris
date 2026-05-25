#!/usr/bin/env python3
"""
Step 2 — Re-encoding for run #4 (post-BLP final).

For each of the 10 encoding jobs (3 WEIRD + 3 Sinic + 2 bilingual × {EN, ZH}),
produces:

  embeddings/{label}/
    vecs_bare.npy        shape (364, dim), float32, L2-normalized
    vecs_attested.npy    shape (364, dim), float32, L2-normalized
    coverage.json        per-term n_contexts_attested + Cap. provenance
    meta.json            model + dim + dtype + dates + sha256

Also writes embeddings/index.json: the canonical 364-term ordering.

The pool is loaded from inputs/core_terms_snapshot.json (frozen by
step 0). Contexts are joined from inputs/term_contexts_bilingual_snapshot.jsonl
by the English headword (unique over 364). For 6 terms the K<4 floor is
not met in one language (Cap.-internal); we encode on the available
contexts and flag them in coverage.json.

Idempotent: if `vecs_bare.npy` and `vecs_attested.npy` already exist with a
meta.json that matches the current snapshot hash, the model is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import EmbeddingClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_config(config_path: Path) -> dict:
    with config_path.open() as fh:
        return yaml.safe_load(fh)


def load_terms(snapshot: Path) -> list[dict]:
    with snapshot.open() as fh:
        data = json.load(fh)
    terms = data["terms"] if isinstance(data, dict) and "terms" in data else data
    if len(terms) != 364:
        raise AssertionError(f"Expected 364 terms, got {len(terms)}")
    return terms


def load_contexts(snapshot: Path) -> dict[str, dict]:
    by_en: dict[str, dict] = {}
    with snapshot.open() as fh:
        for line in fh:
            rec = json.loads(line)
            by_en[rec["term_en"]] = rec
    return by_en


def collect_per_term_contexts(
    terms: list[dict],
    contexts: dict[str, dict],
    lang: str,
    max_n: int,
    max_chars: int,
) -> tuple[list[list[str]], list[int], list[list[dict]]]:
    """For each of the 364 terms, return (truncated context strings,
    actual n_used, per-context provenance) in the lang of interest."""
    key = f"{lang}_contexts"
    per_term: list[list[str]] = []
    n_used: list[int] = []
    prov: list[list[dict]] = []
    for t in terms:
        rec = contexts.get(t["en"])
        if rec is None:
            raise KeyError(f"No context record for term en={t['en']!r}")
        ctxs = rec.get(key, [])[:max_n]
        per_term.append([c["context"][:max_chars] for c in ctxs])
        n_used.append(len(ctxs))
        prov.append([{"cap": c.get("cap"), "year": c.get("cap_year"),
                      "section": c.get("section_id")} for c in ctxs])
    return per_term, n_used, prov


def encode_attested(
    client: EmbeddingClient,
    model_id: str,
    per_term: list[list[str]],
    n_terms: int,
    dim: int,
) -> np.ndarray:
    """Encode the flat list of contexts, mean-aggregate per term,
    L2-normalize the mean."""
    flat: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for ctxs in per_term:
        offsets.append((cursor, cursor + len(ctxs)))
        flat.extend(ctxs)
        cursor += len(ctxs)

    logger.info("    flat contexts to encode: %d", len(flat))
    if len(flat) == 0:
        raise AssertionError("Empty flat context list")
    t0 = time.perf_counter()
    flat_vecs = client.embed(flat, model_id, use_cache=False).astype(np.float32)
    logger.info("    encoded in %.1fs", time.perf_counter() - t0)

    out = np.zeros((n_terms, dim), dtype=np.float32)
    for j, (s, e) in enumerate(offsets):
        if e == s:
            continue
        mean_vec = flat_vecs[s:e].mean(axis=0)
        nrm = np.linalg.norm(mean_vec)
        if nrm > 1e-12:
            mean_vec = mean_vec / nrm
        out[j] = mean_vec.astype(np.float32)
    return out


def write_outputs(
    out_dir: Path,
    label: str,
    model_id: str,
    lang_tag: str,
    dim: int,
    vecs_bare: np.ndarray,
    vecs_attested: np.ndarray,
    terms: list[dict],
    n_attested: list[int],
    prov: list[list[dict]],
    snapshot_sha: str,
    elapsed_bare: float,
    elapsed_attested: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vecs_bare.npy", vecs_bare)
    np.save(out_dir / "vecs_attested.npy", vecs_attested)

    coverage = {
        "lang": lang_tag,
        "n_terms": len(terms),
        "n_terms_k_ge_4": int(sum(1 for k in n_attested if k >= 4)),
        "n_terms_k_lt_4": int(sum(1 for k in n_attested if k < 4)),
        "min_n_attested": int(min(n_attested)),
        "max_n_attested": int(max(n_attested)),
        "median_n_attested": float(np.median(n_attested)),
        "mean_n_attested": float(np.mean(n_attested)),
        "per_term": [
            {
                "en": t["en"],
                "zh": t["zh"],
                "domain": t["domain"],
                "n_contexts_bare": 1,
                "n_contexts_attested": int(n),
                "caps": [p.get("cap") for p in pr],
                "k_below_floor": bool(n < 4),
            }
            for t, n, pr in zip(terms, n_attested, prov)
        ],
    }
    with (out_dir / "coverage.json").open("w") as fh:
        json.dump(coverage, fh, indent=2, ensure_ascii=False)

    meta = {
        "model_id": model_id,
        "model_label": label,
        "lang": lang_tag,
        "dim": dim,
        "n_terms": len(terms),
        "dtype": "float32",
        "l2_norm": True,
        "date": date.today().isoformat(),
        "elapsed_bare_s": round(elapsed_bare, 2),
        "elapsed_attested_s": round(elapsed_attested, 2),
        "snapshot_sha256_terms": snapshot_sha,
    }
    with (out_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def renormalize(arr: np.ndarray) -> np.ndarray:
    """Explicitly L2-renormalize rows (some models — notably Qwen3 — emit
    vectors with a slight scale drift around 1.0 even when the encoder
    config says normalize=True)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.clip(norms, 1e-12, None)).astype(np.float32)


def verify_norms(arr: np.ndarray, name: str, tol: float = 1e-4) -> None:
    norms = np.linalg.norm(arr, axis=1)
    nonzero = norms > 1e-12
    if not nonzero.all():
        bad = int((~nonzero).sum())
        logger.warning("    %s has %d zero-norm rows", name, bad)
    deviation = np.abs(norms[nonzero] - 1.0).max() if nonzero.any() else 0.0
    if deviation > tol:
        raise AssertionError(f"{name} L2-norm deviates by {deviation:g} (tol={tol})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model labels (default: all 10 jobs)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a model if vecs_bare.npy + vecs_attested.npy + meta.json match snapshot sha")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    inputs_dir = REPO_ROOT / cfg["paths"]["inputs"]
    out_root = REPO_ROOT / cfg["paths"]["embeddings"]
    out_root.mkdir(parents=True, exist_ok=True)

    terms_snapshot = inputs_dir / "core_terms_snapshot.json"
    ctxs_snapshot = inputs_dir / "term_contexts_bilingual_snapshot.jsonl"
    snapshot_sha = sha256_of(terms_snapshot)

    terms = load_terms(terms_snapshot)
    contexts = load_contexts(ctxs_snapshot)
    if len(contexts) != 364:
        raise AssertionError(f"context jsonl has {len(contexts)} records, expected 364")

    # Shared index.json — single source of truth for term ordering
    index = [
        {"en": t["en"], "zh": t["zh"], "domain": t["domain"], "tier": t.get("tier")}
        for t in terms
    ]
    with (out_root / "index.json").open("w") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    logger.info("index.json written (%d entries)", len(index))

    en_texts = [t["en"] for t in terms]
    zh_texts = [t["zh"] for t in terms]

    max_n = int(cfg["attested_max_contexts"])
    max_chars = int(cfg["attested_max_context_chars"])
    en_per_term, en_n, en_prov = collect_per_term_contexts(
        terms, contexts, "en", max_n, max_chars,
    )
    zh_per_term, zh_n, zh_prov = collect_per_term_contexts(
        terms, contexts, "zh", max_n, max_chars,
    )
    logger.info("EN attested coverage: min=%d median=%g max=%d mean=%.2f",
                min(en_n), np.median(en_n), max(en_n), np.mean(en_n))
    logger.info("ZH attested coverage: min=%d median=%g max=%d mean=%.2f",
                min(zh_n), np.median(zh_n), max(zh_n), np.mean(zh_n))

    # Build the 10-job run plan: (job_label, model_id, lang, bare_texts, ctx_per_term, ctx_n, ctx_prov, dim)
    jobs: list[tuple[str, str, str, list[str], list[list[str]], list[int], list[list[dict]], int]] = []
    for m in cfg["models_weird"]:
        jobs.append((m["label"], m["id"], "en", en_texts,
                     en_per_term, en_n, en_prov, int(m["dim"])))
    for m in cfg["models_sinic"]:
        jobs.append((m["label"], m["id"], "zh", zh_texts,
                     zh_per_term, zh_n, zh_prov, int(m["dim"])))
    for m in cfg["models_bilingual"]:
        jobs.append((f"{m['label']}-EN", m["id"], "en", en_texts,
                     en_per_term, en_n, en_prov, int(m["dim"])))
        jobs.append((f"{m['label']}-ZH", m["id"], "zh", zh_texts,
                     zh_per_term, zh_n, zh_prov, int(m["dim"])))

    if args.models is not None:
        jobs = [j for j in jobs if j[0] in args.models]
    if not jobs:
        logger.error("No jobs matched --models %s", args.models)
        return 1

    logger.info("Total encoding jobs: %d", len(jobs))

    # Use the master models/config.yaml for the EmbeddingClient (it has model
    # specs + cache). Override the device from run4 config.
    client_config = REPO_ROOT / "experiments" / "models" / "config.yaml"
    client = EmbeddingClient(
        config_path=client_config,
        device=cfg.get("device", "cpu"),
        batch_size=cfg.get("batch_size", 64),
    )

    overall_t0 = time.perf_counter()
    prev_model_id: str | None = None
    for label, model_id, lang, bare_texts, ctx_per_term, ctx_n, ctx_prov, dim in jobs:
        out_dir = out_root / label
        meta_path = out_dir / "meta.json"
        if (args.skip_existing
                and (out_dir / "vecs_bare.npy").exists()
                and (out_dir / "vecs_attested.npy").exists()
                and meta_path.exists()):
            try:
                m = json.loads(meta_path.read_text())
                if m.get("snapshot_sha256_terms") == snapshot_sha:
                    logger.info("[%s] up-to-date; skipping", label)
                    continue
            except Exception:
                pass

        logger.info("=== %s (model=%s, lang=%s, dim=%d) ===",
                    label, model_id, lang, dim)

        if (cfg.get("unload_between_models", False)
                and prev_model_id is not None
                and prev_model_id != model_id):
            client.unload_model(prev_model_id)
            import gc
            gc.collect()

        t0 = time.perf_counter()
        bare = client.embed(bare_texts, model_id, use_cache=True).astype(np.float32)
        bare = renormalize(bare)
        elapsed_bare = time.perf_counter() - t0
        verify_norms(bare, f"{label}/bare")
        logger.info("  bare encoded in %.1fs, shape=%s", elapsed_bare, bare.shape)

        t1 = time.perf_counter()
        attested = encode_attested(client, model_id, ctx_per_term, len(terms), dim)
        attested = renormalize(attested)
        elapsed_attested = time.perf_counter() - t1
        verify_norms(attested, f"{label}/attested")
        logger.info("  attested aggregated in %.1fs, shape=%s", elapsed_attested, attested.shape)

        write_outputs(
            out_dir, label, model_id, lang, dim,
            bare, attested, terms,
            ctx_n, ctx_prov,
            snapshot_sha, elapsed_bare, elapsed_attested,
        )
        logger.info("  -> %s", out_dir.relative_to(REPO_ROOT))
        prev_model_id = model_id

    logger.info("=== Total encoding time: %.1fs ===", time.perf_counter() - overall_t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
