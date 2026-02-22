"""
k-NN domain assignment for background terms (§8.1).

Assigns each background term to a legal domain by majority vote among its
k nearest core terms in embedding space. Used in §8.1 to test whether the
embedding signal is strong enough to self-organize unlabelled legal terms.

Design decisions: lens_1_relational/trace.md — D_BG1 (k-NN), D_BG2 (k=7)
Math reference:   shared/math_trace.md — §B1
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np


def assign_domains(
    vecs_bg: np.ndarray,
    vecs_core: np.ndarray,
    labels_core: list[str],
    k: int = 7,
) -> list[dict]:
    """
    Assign domain labels to background terms via k-NN majority vote.

    For each background term b:
      1. Compute cosine similarity with all core terms: sim = vecs_bg[b] @ vecs_core.T
      2. Take the k most similar core terms (argsort descending)
      3. Majority vote on their domain labels → assigned_domain
      4. confidence = count(majority_domain) / k

    Parameters
    ----------
    vecs_bg     : (N_bg, dim) L2-normalized background embeddings
    vecs_core   : (N_core, dim) L2-normalized core embeddings
    labels_core : domain label per core term, length N_core
    k           : neighbors for majority vote (default 7, odd → no ties)

    Returns
    -------
    list of dict per background term:
        assigned_domain : str
        confidence      : float ∈ [0, 1] — proportion of k neighbors agreeing
        neighbor_domains: list[str] — k neighbor domains, similarity order
        neighbor_sims   : list[float]
        neighbor_indices: list[int] — indices into vecs_core / labels_core
    """
    labels_arr = np.array(labels_core)
    sims = vecs_bg @ vecs_core.T    # (N_bg, N_core) — one matrix multiply

    results = []
    for i in range(len(vecs_bg)):
        top_k = np.argsort(sims[i])[-k:][::-1]
        top_labels = labels_arr[top_k].tolist()
        top_sims = sims[i][top_k].tolist()

        counts = Counter(top_labels)
        assigned, count = counts.most_common(1)[0]

        results.append({
            "assigned_domain": assigned,
            "confidence": round(count / k, 4),
            "neighbor_domains": top_labels,
            "neighbor_sims": [round(float(s), 4) for s in top_sims],
            "neighbor_indices": top_k.tolist(),
        })
    return results


def build_review_csv(
    terms_bg: list[dict],
    assignments: list[dict],
    terms_core: list[dict],
    output_path: Path,
) -> None:
    """
    Export domain assignments to a CSV structured for jurist review.

    Sorted by assigned_domain ASC, confidence DESC — so the reviewer works
    domain by domain (high-confidence cases first) and scans for outliers
    rather than evaluating each term from scratch.

    Columns
    -------
    en, zh, assigned_domain, confidence,
    neighbor_1 … neighbor_k  (format: "term [domain] sim=X.XXX"),
    annotation  (empty — reviewer fills with ✓ or the correct domain name)
    """
    k = len(assignments[0]["neighbor_domains"])
    rows = []

    for term, asgn in zip(terms_bg, assignments):
        row = {
            "en": term["en"],
            "zh": term.get("zh_canonical", ""),
            "assigned_domain": asgn["assigned_domain"],
            "confidence": asgn["confidence"],
        }
        for j in range(k):
            nb = terms_core[asgn["neighbor_indices"][j]]
            row[f"neighbor_{j + 1}"] = (
                f"{nb['en']} [{asgn['neighbor_domains'][j]}]"
                f" sim={asgn['neighbor_sims'][j]:.3f}"
            )
        row["annotation"] = ""
        rows.append(row)

    rows.sort(key=lambda r: (r["assigned_domain"], -r["confidence"]))

    fieldnames = (
        ["en", "zh", "assigned_domain", "confidence"]
        + [f"neighbor_{j + 1}" for j in range(k)]
        + ["annotation"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_annotations(csv_path: Path) -> dict:
    """
    Load a reviewed CSV and compute accuracy metrics.

    Accepted "correct" values (case-insensitive): ✓ ok correct sì si yes y 1
    Any other non-empty value is treated as a domain correction.

    Returns
    -------
    dict:
        total            : int
        annotated        : int
        correct          : int
        accuracy_overall : float | None
        per_domain       : {domain: {total, correct, accuracy}}
        errors           : list of {en, zh, assigned, corrected_to}
    """
    CORRECT_TOKENS = {"✓", "ok", "correct", "sì", "si", "yes", "y", "1"}

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    annotated = [r for r in rows if r["annotation"].strip()]
    correct_set = {
        id(r) for r in annotated
        if r["annotation"].strip().lower() in CORRECT_TOKENS
    }

    per_domain: dict[str, dict] = {}
    for r in annotated:
        d = r["assigned_domain"]
        per_domain.setdefault(d, {"total": 0, "correct": 0})
        per_domain[d]["total"] += 1
        if id(r) in correct_set:
            per_domain[d]["correct"] += 1
    for v in per_domain.values():
        v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] else 0.0

    errors = [
        {
            "en": r["en"],
            "zh": r["zh"],
            "assigned": r["assigned_domain"],
            "corrected_to": r["annotation"].strip(),
        }
        for r in annotated
        if id(r) not in correct_set
    ]

    n_correct = len(correct_set)
    return {
        "total": len(rows),
        "annotated": len(annotated),
        "correct": n_correct,
        "accuracy_overall": round(n_correct / len(annotated), 4) if annotated else None,
        "per_domain": per_domain,
        "errors": errors,
    }
