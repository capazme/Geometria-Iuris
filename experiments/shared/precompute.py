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
    model_label: str,
    texts: list[str],
    out_dir: Path,
    source_sha256: str,
) -> None:
    spec = next(s for s in client.all_specs if s.label == model_label)
    model_dir = out_dir / model_label
    meta_path = model_dir / "meta.json"
    vec_path = model_dir / "vectors.npy"

    # Skip if already computed for the same source file
    if vec_path.exists() and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing.get("source_sha256") == source_sha256:
            logger.info("[%s] Already computed — skipping.", model_label)
            return

    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Embedding %d texts (lang=%s, dim=%d) ...",
                model_label, len(texts), spec.lang, spec.dim)
    t0 = time.perf_counter()

    # Embed in one call; the client's disk cache is bypassed here because we
    # write to a structured directory instead of the SHA-256 cache.
    vecs = client.embed(texts, spec.id, use_cache=False)

    elapsed = time.perf_counter() - t0
    logger.info("[%s] Done in %.1fs  shape=%s", model_label, elapsed, vecs.shape)

    np.save(vec_path, vecs)

    meta = {
        "model_id": spec.id,
        "model_label": model_label,
        "lang": spec.lang,
        "dim": spec.dim,
        "n_terms": len(texts),
        "date": date.today().isoformat(),
        "elapsed_s": round(elapsed, 2),
        "source_sha256": source_sha256,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("[%s] Saved to %s", model_label, model_dir)


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

    if args.dry_run:
        logger.info("\n[DRY RUN] Would embed %d EN + %d ZH texts across %d models.",
                    len(en_texts), len(zh_texts),
                    len(args.models) if args.models else 6)
        return 0

    client = EmbeddingClient(CONFIG, device=args.device or None)
    logger.info("Device: %s", client._device)

    # Determine which models to run
    all_specs = {s.label: s for s in client.all_specs}
    labels = args.models if args.models else list(all_specs)

    unknown = [l for l in labels if l not in all_specs]
    if unknown:
        logger.error("Unknown model labels: %s. Available: %s", unknown, list(all_specs))
        return 1

    for label in labels:
        spec = all_specs[label]
        texts = en_texts if spec.lang == "en" else zh_texts
        try:
            embed_model(client, label, texts, EMBEDDINGS_DIR, source_sha256)
        except Exception as exc:
            logger.error("[%s] FAILED: %s", label, exc)
            return 1

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
    sys.exit(main(parser.parse_args()))
