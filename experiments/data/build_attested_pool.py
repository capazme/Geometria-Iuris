"""
Build contextualised embeddings from attested e-Legislation contexts.

For each term in the 9472-term pool:
  - If attested contexts exist in term_contexts.jsonl: encode those (up to 8)
  - Otherwise: fall back to the 8 synthetic legal templates

Mean-aggregates per-term vectors into a single contextualised representation,
identical to build_contextualized_pool.py but grounded in real legislative use.

Output
------
data/processed/embeddings_ctx_attested/
    {model_label}/
        vectors.npy     shape (9472, dim), float32, L2-normalized
        meta.json       provenance metadata

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

EMBEDDINGS_DIR = REPO_ROOT / "data" / "processed" / "embeddings"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "embeddings_ctx_attested"
TERM_CONTEXTS = REPO_ROOT / "data" / "processed" / "elegislation" / "term_contexts.jsonl"

# Models: (model_id, output_label, lang)
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

# Synthetic fallback templates (same as build_contextualized_pool.py)
EN_TEMPLATES = [
    "{term}",
    "the legal term {term}",
    "{term} in Hong Kong legislation",
    "the meaning of {term}",
    "the court considered the {term}",
    "the doctrine of {term}",
    "an action concerning the {term}",
    "the legal effect of the {term}",
]
ZH_TEMPLATES = [
    "{term}",
    "法律術語{term}",
    "香港法例中的{term}",
    "{term}的法律意義",
    "法庭考慮了{term}",
    "{term}原則",
    "涉及{term}的訴訟",
    "{term}的法律效力",
]

N_CONTEXTS = 8  # target number of contexts per term
MAX_CONTEXT_CHARS = 120  # truncate long contexts to keep decoder models fast


def load_attested_contexts() -> dict[int, dict]:
    """Load term_contexts.jsonl into a dict keyed by term_idx."""
    contexts: dict[int, dict] = {}
    if TERM_CONTEXTS.exists():
        with TERM_CONTEXTS.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                contexts[rec["term_idx"]] = rec
    return contexts


def build_sentences_for_term(
    term_idx: int,
    term_en: str,
    term_zh: str,
    lang: str,
    attested: dict[int, dict],
) -> list[str]:
    """Build up to N_CONTEXTS sentences for a term, preferring attested contexts."""
    rec = attested.get(term_idx)
    sentences: list[str] = []

    if lang == "en" and rec and rec.get("en_contexts"):
        for ctx in rec["en_contexts"][:N_CONTEXTS]:
            sentences.append(ctx["context"][:MAX_CONTEXT_CHARS])
    elif lang == "zh" and rec and rec.get("zh_contexts"):
        for ctx in rec["zh_contexts"][:N_CONTEXTS]:
            sentences.append(ctx["context"][:MAX_CONTEXT_CHARS])

    # Pad with synthetic templates if not enough attested contexts
    if len(sentences) < N_CONTEXTS:
        templates = EN_TEMPLATES if lang == "en" else ZH_TEMPLATES
        term = term_en if lang == "en" else term_zh
        for tpl in templates:
            if len(sentences) >= N_CONTEXTS:
                break
            sent = tpl.format(term=term)
            if sent not in sentences:
                sentences.append(sent)

    return sentences[:N_CONTEXTS]


def encode_pool(
    client: EmbeddingClient,
    model_id: str,
    output_label: str,
    lang: str,
    index: list[dict],
    attested: dict[int, dict],
) -> np.ndarray:
    """Encode all terms with attested+fallback contexts, return (N, dim) matrix."""
    n_terms = len(index)
    n_attested = 0
    n_synthetic = 0

    # Build all sentences for all terms
    all_sentences: list[list[str]] = []
    for i, term in enumerate(index):
        sents = build_sentences_for_term(
            i, term["en"], term["zh_canonical"], lang, attested,
        )
        all_sentences.append(sents)
        # Track provenance
        rec = attested.get(i)
        ctx_key = "en_contexts" if lang == "en" else "zh_contexts"
        if rec and rec.get(ctx_key):
            n_attested += 1
        else:
            n_synthetic += 1

    print(f"    attested: {n_attested}/{n_terms} ({100*n_attested/n_terms:.0f}%)  "
          f"synthetic fallback: {n_synthetic}")

    # Encode in batches per context slot (same strategy as build_contextualized_pool.py)
    accum: np.ndarray | None = None
    for ci in range(N_CONTEXTS):
        t0 = time.perf_counter()
        texts = [sents[ci] if ci < len(sents) else sents[-1] for sents in all_sentences]
        vecs = client.embed(texts, model_id, use_cache=False).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)
        if accum is None:
            accum = vecs.astype(np.float64)
        else:
            accum += vecs.astype(np.float64)
        dt = time.perf_counter() - t0
        print(f"    context {ci+1}/{N_CONTEXTS}: {n_terms} sents in {dt:.1f}s")

    assert accum is not None
    mean_vecs = (accum / N_CONTEXTS).astype(np.float32)
    norms = np.linalg.norm(mean_vecs, axis=1, keepdims=True)
    mean_vecs = mean_vecs / np.clip(norms, 1e-12, None)
    return mean_vecs


def main(args: argparse.Namespace) -> int:
    print("[attested-pool] Loading shared term index ...")
    index_path = EMBEDDINGS_DIR / "index.json"
    index: list[dict] = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"  {len(index)} terms loaded")

    print("[attested-pool] Loading attested contexts ...")
    attested = load_attested_contexts()
    print(f"  {len(attested)} terms with attested contexts")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    config_path = REPO_ROOT / "models" / "config.yaml"
    client = EmbeddingClient(str(config_path), device=args.device or None)
    overall_t0 = time.perf_counter()

    # Build model run plan
    run_plan: list[tuple[str, str, str]] = []  # (model_id, output_label, lang)

    all_models = WEIRD_MODELS + SINIC_MODELS + BILINGUAL_MODELS
    requested = set(args.models) if args.models else None

    for model_id, label, lang in all_models:
        if lang == "bi":
            entries = [(model_id, f"{label}-EN", "en"), (model_id, f"{label}-ZH", "zh")]
        else:
            entries = [(model_id, label, lang)]
        for mid, olabel, olang in entries:
            if requested and olabel not in requested:
                continue
            run_plan.append((mid, olabel, olang))

    if args.dry_run:
        print(f"\n[DRY RUN] {len(run_plan)} jobs planned:")
        for mid, olabel, olang in run_plan:
            print(f"  {mid} → {olabel} ({olang})")
        return 0

    prev_model_id: str | None = None
    for model_id, output_label, lang in run_plan:
        print(f"\n[attested-pool] === {output_label} ({lang}) ===")
        out_dir = OUTPUT_DIR / output_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vectors.npy"

        if out_path.exists() and not args.force:
            print(f"  [skip] {out_path} already exists (use --force to overwrite)")
            continue

        # Unload previous model if different
        if prev_model_id and prev_model_id != model_id:
            client.unload_model(prev_model_id)
            import gc; gc.collect()

        vecs = encode_pool(client, model_id, output_label, lang, index, attested)
        np.save(out_path, vecs)

        meta = {
            "label": output_label,
            "model_id": model_id,
            "lang": lang,
            "n_terms": len(index),
            "n_contexts": N_CONTEXTS,
            "context_source": "attested+synthetic_fallback",
            "attested_source": "hk_elegislation",
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"  → {out_path} (shape {vecs.shape})")
        prev_model_id = model_id

    elapsed = time.perf_counter() - overall_t0
    print(f"\n[attested-pool] Done. Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[attested-pool] Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", metavar="LABEL",
                        help="Output labels to encode (default: all 10)")
    parser.add_argument("--device", default=None, help="cpu, mps, cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    sys.exit(main(parser.parse_args()))
