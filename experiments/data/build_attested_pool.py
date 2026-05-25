"""
Build contextualised embeddings from attested HK Cap. e-Legislation contexts.

D5 of `experiments/trace_firthian_pivot.md`.

Architecture (post linear-and-coherent revision):
- Operates on the 327-term Firthian-strict core only (terms with K≥4 attested
  contexts in both languages, on glossary-cleaned forms).
- For each (term, model-language) pair: mean-aggregate over min(N_attested, 8)
  attested contexts. NO synthetic padding.
- Non-core rows in the output matrix are zero-filled (matrix shape preserved
  at 9472 × dim for compatibility with bare embeddings).
- Fails loudly on any core term with N_attested < 4 (D3+D4 hard gate
  guarantees this never happens; the assert protects against silent regression).

Output
------
data/processed/embeddings_ctx_attested/
    index.json                   shared 9472-row index (synced from master)
    {model_label}/
        vectors.npy              shape (9472, dim), float32, L2-normalized
                                 327 valid rows + 9145 zero rows
        meta.json                provenance: model_id, lang, n_core_attested,
                                 attestation_stats (min/median/max N per lang)

Usage
-----
    python data/build_attested_pool.py [--models LABEL ...] [--device DEVICE]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.embeddings import EmbeddingClient  # noqa: E402

LEGAL_TERMS = REPO_ROOT / "data" / "processed" / "legal_terms.json"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "processed" / "embeddings"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "embeddings_ctx_attested"
TERM_CONTEXTS = REPO_ROOT / "data" / "processed" / "elegislation" / "term_contexts.jsonl"

WEIRD_MODELS = [
    ("BAAI/bge-large-en-v1.5", "BGE-EN-large", "en"),
    ("intfloat/e5-large-v2", "E5-large", "en"),
    ("freelawproject/modernbert-embed-base_finetune_512", "FreeLaw-EN", "en"),
]
SINIC_MODELS = [
    ("BAAI/bge-large-zh-v1.5", "BGE-ZH-large", "zh"),
    ("GanymedeNil/text2vec-large-chinese", "Text2vec-large-ZH", "zh"),
    ("DMetaSoul/Dmeta-embedding-zh", "Dmeta-ZH", "zh"),
]
BILINGUAL_MODELS = [
    ("BAAI/bge-m3", "BGE-M3", "bi"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B", "bi"),
]

MIN_N_ATTESTED = 4
MAX_N_ATTESTED = 8
MAX_CONTEXT_CHARS = 120


def load_attested_contexts() -> dict[int, dict]:
    contexts: dict[int, dict] = {}
    with TERM_CONTEXTS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            contexts[rec["term_idx"]] = rec
    return contexts


def core_indices(legal_terms: list[dict]) -> list[int]:
    return [i for i, t in enumerate(legal_terms) if t.get("tier") == "core" and t.get("domain")]


def collect_contexts(
    core_idx: list[int],
    contexts: dict[int, dict],
    lang: str,
) -> tuple[list[list[str]], list[int]]:
    """Returns (per_term_contexts, n_attested_per_term). Asserts K≥MIN_N_ATTESTED."""
    key = f"{lang}_contexts"
    per_term: list[list[str]] = []
    n_per_term: list[int] = []
    for ti in core_idx:
        rec = contexts.get(ti, {})
        ctxs = rec.get(key, [])
        n = len(ctxs)
        if n < MIN_N_ATTESTED:
            raise AssertionError(
                f"Core term_idx {ti} has only {n} {lang} attested contexts "
                f"(< {MIN_N_ATTESTED}); D3+D4 hard gate violated."
            )
        n_use = min(n, MAX_N_ATTESTED)
        per_term.append([c["context"][:MAX_CONTEXT_CHARS] for c in ctxs[:n_use]])
        n_per_term.append(n_use)
    return per_term, n_per_term


def encode_pool(
    client: EmbeddingClient,
    model_id: str,
    output_label: str,
    lang: str,
    n_total_terms: int,
    core_idx: list[int],
    per_term_contexts: list[list[str]],
    n_per_term: list[int],
    dim: int,
) -> tuple[np.ndarray, dict]:
    print(f"  [{output_label}] Encoding attested means for {len(core_idx)} core terms ...")

    all_contexts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for ctxs in per_term_contexts:
        offsets.append((cursor, cursor + len(ctxs)))
        all_contexts.extend(ctxs)
        cursor += len(ctxs)
    print(f"    total flat contexts: {len(all_contexts)}")

    t0 = time.perf_counter()
    flat_vecs = client.embed(all_contexts, model_id, use_cache=False).astype(np.float32)
    norms = np.linalg.norm(flat_vecs, axis=1, keepdims=True)
    flat_vecs = flat_vecs / np.clip(norms, 1e-12, None)
    dt = time.perf_counter() - t0
    print(f"    encoded in {dt:.1f}s")

    out = np.zeros((n_total_terms, dim), dtype=np.float32)
    for j, ti in enumerate(core_idx):
        s, e = offsets[j]
        term_vecs = flat_vecs[s:e]
        mean_vec = term_vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-12:
            mean_vec = mean_vec / norm
        out[ti] = mean_vec.astype(np.float32)

    stats = {
        "n_core_attested": len(core_idx),
        "n_total_rows": n_total_terms,
        "min_n_attested": int(min(n_per_term)),
        "median_n_attested": float(np.median(n_per_term)),
        "max_n_attested": int(max(n_per_term)),
        "mean_n_attested": float(np.mean(n_per_term)),
        "lang": lang,
    }
    return out, stats


def main(args: argparse.Namespace) -> int:
    print("[D5] Loading legal_terms.json ...")
    payload = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))
    terms = payload["terms"]
    core_idx = core_indices(terms)
    print(f"  total terms: {len(terms)}, core size: {len(core_idx)}")

    print("[D5] Loading term_contexts.jsonl ...")
    contexts = load_attested_contexts()
    print(f"  {len(contexts)} terms with at least one context")

    print("[D5] Validating EN core attestation ...")
    en_per_term, en_n = collect_contexts(core_idx, contexts, "en")
    print("[D5] Validating ZH core attestation ...")
    zh_per_term, zh_n = collect_contexts(core_idx, contexts, "zh")
    print(f"  EN N: min={min(en_n)}, median={int(np.median(en_n))}, max={max(en_n)}, mean={float(np.mean(en_n)):.2f}")
    print(f"  ZH N: min={min(zh_n)}, median={int(np.median(zh_n))}, max={max(zh_n)}, mean={float(np.mean(zh_n)):.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bare_index_path = EMBEDDINGS_DIR / "index.json"
    if bare_index_path.exists():
        index = json.loads(bare_index_path.read_text(encoding="utf-8"))
        (OUTPUT_DIR / "index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    all_models = WEIRD_MODELS + SINIC_MODELS
    target_labels = set(args.models) if args.models else None

    run_plan: list[tuple[str, str, str, list[list[str]], list[int]]] = []
    for mid, label, lang in all_models:
        if target_labels and label not in target_labels:
            continue
        if lang == "en":
            run_plan.append((mid, label, "en", en_per_term, en_n))
        else:
            run_plan.append((mid, label, "zh", zh_per_term, zh_n))

    for mid, label, _ in BILINGUAL_MODELS:
        if target_labels and label not in target_labels:
            continue
        run_plan.append((mid, f"{label}-EN", "en", en_per_term, en_n))
        run_plan.append((mid, f"{label}-ZH", "zh", zh_per_term, zh_n))

    if args.dry_run:
        print(f"\n[DRY RUN] {len(run_plan)} jobs planned:")
        for mid, label, lang, _, _ in run_plan:
            print(f"  {mid.split('/')[-1]} → {label} ({lang})")
        return 0

    config_path = REPO_ROOT / "models" / "config.yaml"
    client = EmbeddingClient(str(config_path), device=args.device or "cpu")

    def lookup_dim(label: str) -> int:
        meta_path = EMBEDDINGS_DIR / label / "meta.json"
        if meta_path.exists():
            return int(json.loads(meta_path.read_text())["dim"])
        return 1024

    n_total_terms = len(terms)
    overall_t0 = time.perf_counter()
    for mid, label, lang, per_term_ctx, n_per_term in run_plan:
        dim = lookup_dim(label)
        print(f"\n=== {label} (lang={lang}, dim={dim}) ===")
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        vectors, stats = encode_pool(
            client, mid, label, lang, n_total_terms, core_idx,
            per_term_ctx, n_per_term, dim,
        )
        np.save(out_dir / "vectors.npy", vectors)
        meta = {
            "model_id": mid,
            "model_label": label,
            "lang": lang,
            "dim": dim,
            "n_total_rows": n_total_terms,
            "n_core_attested": len(core_idx),
            "context_source": "hk_elegislation_strict_no_padding",
            "min_n_attested": stats["min_n_attested"],
            "median_n_attested": stats["median_n_attested"],
            "mean_n_attested": stats["mean_n_attested"],
            "max_n_attested": stats["max_n_attested"],
            "max_context_chars": MAX_CONTEXT_CHARS,
            "date": "2026-05-02",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"  Saved: {(out_dir / 'vectors.npy').relative_to(REPO_ROOT)}")

    print(f"\n=== Total time: {time.perf_counter() - overall_t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Subset of model labels to run")
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
