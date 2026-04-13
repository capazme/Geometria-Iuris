"""
Pre-compute and persist embeddings for all terms in legal_terms.json.

For each of the 6 models (3 WEIRD + 3 Sinic), embeds all N terms using the
appropriate language field:
  - WEIRD (lang=en): term["en"]
  - Sinic (lang=zh): term["zh_canonical"]

Results are saved as float32 .npy arrays alongside a shared term index and
per-model provenance metadata. The term order in vectors.npy is identical to
index.json, enabling direct index-based lookup in all Lens experiments.

Output layout
-------------
data/processed/embeddings/
    index.json              ordered list of {en, zh_canonical, domain, tier}
    {model_label}/
        vectors.npy         shape (N, dim), float32, L2-normalized
        meta.json           {model_id, lang, dim, n_terms, date, source_sha256}

Run
---
    python shared/precompute.py [--models MODEL_LABEL ...] [--device cpu|mps|cuda]

Examples
--------
    # All 6 models, auto-detected device
    python shared/precompute.py

    # Only WEIRD models on MPS
    python shared/precompute.py --models BGE-EN-large E5-large FreeLaw-EN --device mps

    # Dry run: show what would be computed without encoding
    python shared/precompute.py --dry-run
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.embeddings import EmbeddingClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LEGAL_TERMS = ROOT / "data" / "processed" / "legal_terms.json"
EMBEDDINGS_DIR = ROOT / "data" / "processed" / "embeddings"
CONFIG = ROOT / "models" / "config.yaml"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def build_index(terms: list[dict]) -> list[dict]:
    """Extract the minimal per-term metadata used by Lens experiments."""
    return [
        {
            "en": t["en"],
            "zh_canonical": t["zh_canonical"],
            "domain": t.get("domain"),
            "tier": t["tier"],
        }
        for t in terms
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


# ---------------------------------------------------------------------------
# Per-model embedding
# ---------------------------------------------------------------------------

def embed_model(
    client: EmbeddingClient,
    model_id: str,
    output_label: str,
    lang_tag: str,
    texts: list[str],
    dim: int,
    out_dir: Path,
    source_sha256: str,
) -> None:
    """Embed texts with a single model and save to out_dir/output_label/."""
    model_dir = out_dir / output_label
    meta_path = model_dir / "meta.json"
    vec_path = model_dir / "vectors.npy"

    # Skip if already computed for the same source file
    if vec_path.exists() and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing.get("source_sha256") == source_sha256:
            logger.info("[%s] Already computed — skipping.", output_label)
            return

    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Embedding %d texts (lang=%s, dim=%d) ...",
                output_label, len(texts), lang_tag, dim)
    t0 = time.perf_counter()

    vecs = client.embed(texts, model_id, use_cache=False)

    elapsed = time.perf_counter() - t0
    logger.info("[%s] Done in %.1fs  shape=%s", output_label, elapsed, vecs.shape)

    np.save(vec_path, vecs)

    meta = {
        "model_id": model_id,
        "model_label": output_label,
        "lang": lang_tag,
        "dim": dim,
        "n_terms": len(texts),
        "date": date.today().isoformat(),
        "elapsed_s": round(elapsed, 2),
        "source_sha256": source_sha256,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("[%s] Saved to %s", output_label, model_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    logger.info("Loading legal_terms.json ...")
    data = json.loads(LEGAL_TERMS.read_text(encoding="utf-8"))
    terms = data["terms"] if isinstance(data, dict) else data
    source_sha256 = sha256_file(LEGAL_TERMS)
    logger.info("  %d terms loaded (sha256=%s...)", len(terms), source_sha256[:12])

    index = build_index(terms)
    en_texts = [t["en"] for t in terms]
    zh_texts = [t["zh_canonical"] for t in terms]

    # Save shared index
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = EMBEDDINGS_DIR / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    logger.info("Index written: %s (%d entries)", index_path, len(index))

    client = EmbeddingClient(CONFIG, device=args.device or None)

    # Determine which models to run
    all_specs = {s.label: s for s in client.all_specs}
    labels = args.models if args.models else list(all_specs)

    unknown = [l for l in labels if l not in all_specs]
    if unknown:
        logger.error("Unknown model labels: %s. Available: %s", unknown, list(all_specs))
        return 1

    # Build the run plan: list of (model_id, output_label, lang_tag, texts, dim)
    run_plan: list[tuple[str, str, str, list[str], int]] = []
    for label in labels:
        spec = all_specs[label]
        if spec.lang == "bi":
            # Bilingual models produce two embedding sets (EN + ZH)
            run_plan.append((spec.id, f"{label}-EN", "en", en_texts, spec.dim))
            run_plan.append((spec.id, f"{label}-ZH", "zh", zh_texts, spec.dim))
        elif spec.lang == "en":
            run_plan.append((spec.id, label, "en", en_texts, spec.dim))
        else:
            run_plan.append((spec.id, label, "zh", zh_texts, spec.dim))

    if args.dry_run:
        logger.info("\n[DRY RUN] %d embedding jobs planned:", len(run_plan))
        for mid, olabel, ltag, texts, dim in run_plan:
            logger.info("  %s → %s (%s, %d texts, dim=%d)",
                        mid.split("/")[-1], olabel, ltag, len(texts), dim)
        return 0

    logger.info("Device: %s", client._device)

    prev_model_id: str | None = None
    for model_id, output_label, lang_tag, texts, dim in run_plan:
        # Unload previous model if --unload-between is set (saves RAM)
        if args.unload_between and prev_model_id and prev_model_id != model_id:
            client.unload_model(prev_model_id)
            import gc; gc.collect()

        try:
            embed_model(
                client, model_id, output_label, lang_tag,
                texts, dim, EMBEDDINGS_DIR, source_sha256,
            )
        except Exception as exc:
            logger.error("[%s] FAILED: %s", output_label, exc)
            return 1
        prev_model_id = model_id

    logger.info("\nAll done. Embeddings in: %s", EMBEDDINGS_DIR)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", metavar="LABEL",
        help="Model labels to embed (default: all 6). "
             "E.g. --models BGE-EN-large E5-large",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device: cpu, mps, cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be computed without encoding",
    )
    parser.add_argument(
        "--unload-between", action="store_true",
        help="Unload each model after use to save RAM (recommended on 16GB machines)",
    )
    sys.exit(main(parser.parse_args()))
