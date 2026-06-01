"""
Pre-check 1 — Numeracy sanity check (Gamma, adversarial revision 2026-04-11).

Tests whether the six sentence-transformers used in the Geometria Iuris
pipeline represent numerical magnitude in an ordinally meaningful way.

This check decides the form of the parametric stress test D-D (lens 1
§3.1.5). If the models fail to distinguish numerical values in an ordered
way, D-D must be redesigned as a categorical probe; if they distinguish
them, the numerical form is viable.

Background
----------
Wallace, E., Wang, Y., Li, S., Singh, S., & Gardner, M. (2019).
"Do NLP Models Know Numbers? Probing Numeracy in Embeddings." EMNLP.
Showed that BERT-family subword tokenizers represent numbers poorly.
BGE, E5, FreeLaw-modernbert, text2vec-large-chinese, Dmeta-embedding-zh
are all BERT-family; BGE-large-zh is also BERT-family.

Procedure
---------
For each model, encode a sequence of templates differing only in the
numerical value X. Compute:
  (a) Spearman correlation between X and the projection onto the first
      principal component of the sequence.
  (b) Ordinal monotonicity ratio: fraction of X values for which
      cos(emb(X), emb(X+step_small)) > cos(emb(X), emb(X+step_large)).

Decision thresholds
-------------------
PASS:   rho(X, PC1) >= 0.5  AND  ordinal monotonicity >= 0.75
PARTIAL: neither criterion strictly met but at least one above the threshold
FAIL:   both criteria below their thresholds

Output
------
JSON report in pre_checks/results/precheck_1_numeracy.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.embeddings import EmbeddingClient  # noqa: E402


# Template X values. Chosen to span the range relevant to D-D's Test 1
# (age and imputability). Spacing is uneven to stress both small and large
# differences.
X_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 35, 50, 70]


# Templates, one per language, minimally varying only X.
EN_TEMPLATE = "a person aged {x} years"
ZH_TEMPLATE = "一個 {x} 歲的人"


def build_sequence(template: str, x_values: list[int]) -> list[str]:
    """Build a sequence of strings by substituting X into the template."""
    return [template.format(x=x) for x in x_values]


def pc1_projection(vectors: np.ndarray) -> np.ndarray:
    """
    Project each vector in `vectors` onto the first principal component of
    the sequence. Sign-orient the PC so that its projection is monotonically
    increasing with the sequence index (i.e. aligned with the natural
    ordering of X), so that the sign ambiguity of PCA does not affect the
    direction of the Spearman correlation.
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # Compute PC1 via SVD (more stable than eigendecomposition for small N)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]  # shape: (dim,)
    proj = centered @ pc1
    # Sign-align so the projection is, on average, monotone with index
    # (we just flip sign if rho(index, proj) is negative; this is not data
    # snooping because the sign of a PC is arbitrary)
    idx = np.arange(len(proj))
    r, _ = spearmanr(idx, proj)
    if r < 0:
        proj = -proj
    return proj


def ordinal_monotonicity(vectors: np.ndarray, x_values: list[int]) -> float:
    """
    Compute the fraction of interior indices i for which the model
    distinguishes near neighbours from far neighbours in the expected way:

        cos(emb(X_i), emb(X_{i+1})) > cos(emb(X_i), emb(X_{-1}))

    where X_{-1} is the last X in the sequence (maximally far).
    """
    n = len(vectors)
    # Since vectors are L2-normalized, cosine = dot product
    sims_to_next = np.array(
        [vectors[i] @ vectors[i + 1] for i in range(n - 1)]
    )
    sims_to_far = np.array(
        [vectors[i] @ vectors[-1] for i in range(n - 1)]
    )
    return float((sims_to_next > sims_to_far).mean())


def evaluate_model(
    client: EmbeddingClient,
    model_id: str,
    lang: str,
) -> dict:
    """Run the numeracy check for a single model and return its metrics."""
    template = EN_TEMPLATE if lang == "en" else ZH_TEMPLATE
    texts = build_sequence(template, X_VALUES)
    vectors = client.embed(texts, model_id, use_cache=True)
    vectors = vectors.astype(np.float32)

    # Ensure L2 normalization (should already be guaranteed by client)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-12, None)

    # Metric (a): Spearman rho between X and PC1 projection
    pc1_proj = pc1_projection(vectors)
    rho, pvalue = spearmanr(X_VALUES, pc1_proj)

    # Metric (b): ordinal monotonicity
    monotonicity = ordinal_monotonicity(vectors, X_VALUES)

    # Decision
    rho_pass = abs(float(rho)) >= 0.5
    mono_pass = monotonicity >= 0.75
    if rho_pass and mono_pass:
        status = "PASS"
    elif rho_pass or mono_pass:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return {
        "model_id": model_id,
        "lang": lang,
        "template": template,
        "x_values": X_VALUES,
        "spearman_rho_x_pc1": float(rho),
        "spearman_pvalue": float(pvalue),
        "ordinal_monotonicity": monotonicity,
        "rho_passes_threshold": rho_pass,
        "monotonicity_passes_threshold": mono_pass,
        "status": status,
    }


def main() -> None:
    config_path = REPO_ROOT / "models" / "config.yaml"
    client = EmbeddingClient(str(config_path))

    # All six models, with language indicator
    models = [
        ("BAAI/bge-large-en-v1.5", "en"),
        ("intfloat/e5-large-v2", "en"),
        ("freelawproject/modernbert-embed-base_finetune_512", "en"),
        ("BAAI/bge-large-zh-v1.5", "zh"),
        ("GanymedeNil/text2vec-large-chinese", "zh"),
        ("DMetaSoul/Dmeta-embedding-zh", "zh"),
    ]

    per_model = []
    for model_id, lang in models:
        print(f"[numeracy] running {model_id} ({lang}) ...")
        result = evaluate_model(client, model_id, lang)
        per_model.append(result)
        print(
            f"  rho(X, PC1) = {result['spearman_rho_x_pc1']:+.4f}, "
            f"monotonicity = {result['ordinal_monotonicity']:.4f}, "
            f"status = {result['status']}"
        )

    # Aggregate decision
    statuses = [r["status"] for r in per_model]
    n_pass = sum(s == "PASS" for s in statuses)
    n_partial = sum(s == "PARTIAL" for s in statuses)
    n_fail = sum(s == "FAIL" for s in statuses)

    if n_pass >= 5:
        overall = "PASS"
        narrative = (
            "Numeracy is reliable across the pipeline. D-D (parametric stress "
            "test) can proceed in its numerical form on all passing models."
        )
    elif n_pass + n_partial >= 3:
        overall = "PARTIAL_PASS"
        narrative = (
            "Numeracy is reliable on some models but not others. D-D should "
            "run in numerical form on PASS models and in categorical fallback "
            "form on non-PASS models, with per-model status reported alongside "
            "the result."
        )
    else:
        overall = "FAIL"
        narrative = (
            "Numeracy is unreliable across the pipeline. D-D must be redesigned "
            "as a categorical probe only (see D-D trace, numeracy precondition "
            "section, for the categorical fallback template sets)."
        )

    report = {
        "pre_check": "1_numeracy",
        "date": "2026-04-11",
        "thresholds": {
            "rho_x_pc1_min": 0.5,
            "ordinal_monotonicity_min": 0.75,
        },
        "x_values": X_VALUES,
        "templates": {"en": EN_TEMPLATE, "zh": ZH_TEMPLATE},
        "per_model": per_model,
        "aggregate": {
            "n_pass": n_pass,
            "n_partial": n_partial,
            "n_fail": n_fail,
            "overall": overall,
            "narrative": narrative,
        },
    }

    out_path = REPO_ROOT / "pre_checks" / "results" / "precheck_1_numeracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[numeracy] Overall: {overall}")
    print(f"[numeracy] Report written to {out_path}")


if __name__ == "__main__":
    main()
