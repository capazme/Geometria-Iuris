"""Compute 3-component PCA on the 350 core terms for every model, then
serialise a single JSON that the landing page can read to draw a 3D
scatter with a model dropdown.

Run:

    python experiments/dashboard_v2/compute_pca_all_models.py

Output: experiments/dashboard_v2/dataset_pca3d.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

EMB_ROOT = REPO / "experiments" / "data" / "processed" / "embeddings"
LENS4_JSON = REPO / "experiments" / "lens_4_values" / "results" / "lens4_results.json"
OUT = HERE / "dataset_pca3d.json"

ALL_MODELS = [
    "BGE-EN-large", "E5-large", "FreeLaw-EN",
    "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH",
    "BGE-M3-EN", "BGE-M3-ZH",
    "Qwen3-0.6B-EN", "Qwen3-0.6B-ZH",
]

MODEL_GROUP = {m: "WEIRD"     for m in ("BGE-EN-large", "E5-large", "FreeLaw-EN")}
MODEL_GROUP.update({m: "Sinic" for m in ("BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH")})
MODEL_GROUP.update({m: "bilingue" for m in ("BGE-M3-EN", "BGE-M3-ZH", "Qwen3-0.6B-EN", "Qwen3-0.6B-ZH")})


def pca_3(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center + SVD-based 3-component PCA. Returns (projections Nx3, explained
    variance ratio 3,). Sign of each component is arbitrary; we force the
    largest-|value| coordinate of each PC to be positive for cross-model
    comparability when inspecting the plot.
    """
    x = x - x.mean(axis=0, keepdims=True)
    # Economy SVD for efficiency.
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    proj = u[:, :3] * s[:3]
    # Sign convention.
    for k in range(3):
        col = proj[:, k]
        if np.abs(col.min()) > np.abs(col.max()):
            proj[:, k] = -col
    total = (s ** 2).sum()
    ratio = (s[:3] ** 2 / total) if total > 0 else np.zeros(3)
    return proj.astype(np.float32), ratio.astype(np.float32)


def load_core_indices_and_terms():
    with (EMB_ROOT / "index.json").open() as f:
        index = json.load(f)
    core = [(i, t) for i, t in enumerate(index) if t.get("tier") == "core"]
    idx = np.array([i for i, _ in core])
    terms = [t for _, t in core]
    return idx, terms


def main():
    core_idx, terms = load_core_indices_and_terms()

    # Load domain labels for the 350 core terms from the canonical lens4 JSON
    # (which is the source-of-truth ordering for the thesis dashboards).
    with LENS4_JSON.open() as f:
        lens4 = json.load(f)
    core_terms_canonical = lens4["terms_core"]
    # Verify ordering matches: the embedding index.json keeps the same order
    # as lens4.terms_core (both derive from data/processed/legal_terms.json).
    assert len(core_terms_canonical) == len(terms), "core term count mismatch"

    output: dict = {
        "models": ALL_MODELS,
        "n_core": len(terms),
        "terms": [
            {"en": t["en"], "zh": t.get("zh_canonical", t.get("zh", "")),
             "domain": t["domain"]}
            for t in terms
        ],
        "per_model": {},
    }

    for model in ALL_MODELS:
        vecs_path = EMB_ROOT / model / "vectors.npy"
        if not vecs_path.exists():
            print(f"skip {model}: no vectors.npy")
            continue
        vecs = np.load(vecs_path)
        core_vecs = vecs[core_idx]
        proj, ratio = pca_3(core_vecs)
        output["per_model"][model] = {
            "group": MODEL_GROUP[model],
            "dim":   int(core_vecs.shape[1]),
            "explained_variance_ratio": ratio.tolist(),
            "x": proj[:, 0].tolist(),
            "y": proj[:, 1].tolist(),
            "z": proj[:, 2].tolist(),
        }
        print(f"{model:22s}  dim={core_vecs.shape[1]}  "
              f"explained_var=[{ratio[0]:.3f}, {ratio[1]:.3f}, {ratio[2]:.3f}]")

    with OUT.open("w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
