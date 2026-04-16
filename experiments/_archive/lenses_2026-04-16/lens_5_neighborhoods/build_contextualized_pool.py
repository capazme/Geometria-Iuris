"""
Build a contextualised counterpart of the 9472-term bare-term embedding pool.

For each of the six models, encodes every term in the pool with eight legally
plausible templated variants in the matching language, mean-aggregates the
eight per-term vectors into a single contextualised representation, and
writes the resulting (9472, dim) matrix to
``data/processed/embeddings_contextualized/{label}/vectors.npy``.

The output pool is the apples-to-apples target for the false-friends polysemy
test (`false_friends_polysemy.py`): both the query and the neighbour pool
share the same contextualisation.

Cost: ~76,000 sentences/model × 6 models ≈ 24 minutes on MPS (benchmarked
2026-04-11). Re-runs are free thanks to the EmbeddingClient SHA-256 disk
cache.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.embeddings import EmbeddingClient  # noqa: E402

EMBEDDINGS_DIR = REPO_ROOT / "data" / "processed" / "embeddings"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "embeddings_contextualized"

WEIRD_MODELS = [
    ("BAAI/bge-large-en-v1.5", "BGE-EN-large"),
    ("intfloat/e5-large-v2", "E5-large"),
    ("freelawproject/modernbert-embed-base_finetune_512", "FreeLaw-EN"),
]
SINIC_MODELS = [
    ("BAAI/bge-large-zh-v1.5", "BGE-ZH-large"),
    ("GanymedeNil/text2vec-large-chinese", "Text2vec-large-ZH"),
    ("DMetaSoul/Dmeta-embedding-zh", "Dmeta-ZH"),
]
BILINGUAL_MODELS = [
    ("BAAI/bge-m3", "BGE-M3"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B"),
]

# Same 8 templates as false_friends_polysemy.py — ensures the query vectors
# and the pool vectors come from the same family of contextualised
# representations.
EN_VARIANT_TEMPLATES = [
    "{term}",
    "the legal term {term}",
    "{term} in Hong Kong legislation",
    "the meaning of {term}",
    "the court considered the {term}",
    "the doctrine of {term}",
    "an action concerning the {term}",
    "the legal effect of the {term}",
]
ZH_VARIANT_TEMPLATES = [
    "{term}",
    "法律術語{term}",
    "香港法例中的{term}",
    "{term}的法律意義",
    "法庭考慮了{term}",
    "{term}原則",
    "涉及{term}的訴訟",
    "{term}的法律效力",
]


def build_pool_for_model(
    client: EmbeddingClient,
    model_id: str,
    label: str,
    terms: list[str],
    templates: list[str],
) -> np.ndarray:
    """
    Encode every term × every template, mean-aggregate per term, return
    a unit-normalised (N, dim) matrix.

    The encoder is called once *per template*, not once per term, so the
    inner loop is N templates and the outer loop is the per-template batch
    of N terms. This minimises model-switching overhead and lets the cache
    deduplicate cleanly (one cache key per template, not one per term).
    """
    n_terms = len(terms)
    accum: np.ndarray | None = None

    for ti, tpl in enumerate(templates):
        t0 = time.perf_counter()
        texts = [tpl.format(term=t) for t in terms]
        vecs = client.embed(texts, model_id, use_cache=True).astype(np.float32)
        # Defensive renormalisation; the EmbeddingClient already L2-normalises
        # but we want to be sure each individual variant is unit-norm before
        # mean-aggregation.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)

        if accum is None:
            accum = vecs.astype(np.float64)
        else:
            accum += vecs.astype(np.float64)

        dt = time.perf_counter() - t0
        print(
            f"    template {ti + 1}/{len(templates)}: "
            f"{n_terms} sents in {dt:6.2f}s ({n_terms / dt:6.1f} sent/s)"
        )

    assert accum is not None
    mean_vecs = (accum / len(templates)).astype(np.float32)
    # Re-normalise after mean aggregation
    norms = np.linalg.norm(mean_vecs, axis=1, keepdims=True)
    mean_vecs = mean_vecs / np.clip(norms, 1e-12, None)
    return mean_vecs


def main() -> None:
    print("[ctx-pool] Loading shared term index …")
    index_path = EMBEDDINGS_DIR / "index.json"
    index: list[dict] = json.loads(index_path.read_text(encoding="utf-8"))
    en_terms = [r["en"] for r in index]
    zh_terms = [r["zh_canonical"] for r in index]
    print(f"[ctx-pool]   {len(index)} terms loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Copy index reference for reproducibility
    (OUTPUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config_path = REPO_ROOT / "models" / "config.yaml"
    client = EmbeddingClient(str(config_path))

    overall_t0 = time.perf_counter()

    for model_id, label in WEIRD_MODELS:
        print(f"\n[ctx-pool] === {label} (EN) ===")
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vectors.npy"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue
        vecs = build_pool_for_model(
            client, model_id, label, en_terms, EN_VARIANT_TEMPLATES
        )
        np.save(out_path, vecs)
        meta = {
            "label": label,
            "n_terms": len(index),
            "n_templates": len(EN_VARIANT_TEMPLATES),
            "templates": EN_VARIANT_TEMPLATES,
            "aggregator": "mean",
            "lang": "en",
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  → {out_path} (shape {vecs.shape})")

    for model_id, label in SINIC_MODELS:
        print(f"\n[ctx-pool] === {label} (ZH) ===")
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vectors.npy"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue
        vecs = build_pool_for_model(
            client, model_id, label, zh_terms, ZH_VARIANT_TEMPLATES
        )
        np.save(out_path, vecs)
        meta = {
            "label": label,
            "n_terms": len(index),
            "n_templates": len(ZH_VARIANT_TEMPLATES),
            "templates": ZH_VARIANT_TEMPLATES,
            "aggregator": "mean",
            "lang": "zh",
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  → {out_path} (shape {vecs.shape})")

    # Bilingual control models: encode BOTH EN and ZH for each model
    for model_id, label in BILINGUAL_MODELS:
        for lang_tag, terms, templates in [
            ("EN", en_terms, EN_VARIANT_TEMPLATES),
            ("ZH", zh_terms, ZH_VARIANT_TEMPLATES),
        ]:
            out_label = f"{label}-{lang_tag}"
            print(f"\n[ctx-pool] === {out_label} (bilingual {lang_tag}) ===")
            out_dir = OUTPUT_DIR / out_label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "vectors.npy"
            if out_path.exists():
                print(f"  [skip] {out_path} already exists")
                continue
            vecs = build_pool_for_model(
                client, model_id, label, terms, templates
            )
            np.save(out_path, vecs)
            meta = {
                "label": out_label,
                "n_terms": len(index),
                "n_templates": len(templates),
                "templates": templates,
                "aggregator": "mean",
                "lang": lang_tag.lower(),
            }
            (out_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  → {out_path} (shape {vecs.shape})")

    elapsed = time.perf_counter() - overall_t0
    print(f"\n[ctx-pool] Done. Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[ctx-pool] Pool root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
