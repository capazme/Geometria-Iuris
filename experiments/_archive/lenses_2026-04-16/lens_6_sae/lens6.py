"""
Lens VI analysis: map SAE features to legal domains.

Loads trained SAE activations and the term index, then computes:
  - Domain enrichment per feature (Fisher's exact test, Holm-corrected)
  - Domain Selectivity Index (DSI) per feature
  - Feature coverage per domain
  - Top-activating terms per feature

Output
------
results/
    domain_enrichment.json    per-feature domain enrichment results
    feature_summary.json      aggregate statistics
    top_features.json         top-N features by DSI with term lists

Usage
-----
    python lens_6_sae/lens6.py [--expansion 4] [--k 32] [--top-n 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
EMB_DIR = REPO_ROOT / "data" / "processed" / "embeddings"


def make_suffix(model_label: str, expansion: int, k: int) -> str:
    return f"_{model_label}_x{expansion}_k{k}"


def load_data(
    model_label: str, expansion: int, k: int
) -> tuple[np.ndarray, list[dict], list[str]]:
    """Load activations and term index. Return (activations, index, domains)."""
    suffix = make_suffix(model_label, expansion, k)
    act_path = RESULTS_DIR / f"activations{suffix}.npy"
    index_path = EMB_DIR / "index.json"

    activations = np.load(act_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    # Extract unique non-None domains
    domains = sorted(set(t.get("domain") or "" for t in index) - {""})
    return activations, index, domains


def domain_enrichment(
    activations: np.ndarray,
    index: list[dict],
    domains: list[str],
    top_n: int = 50,
) -> list[dict]:
    """For each feature, test domain enrichment of its top-activating terms.

    Uses Fisher's exact test (one-sided) for each (feature, domain) pair.
    Returns a list of per-feature results, sorted by best DSI.
    """
    n_terms, n_features = activations.shape

    # Build domain label array (empty string for unlabeled terms)
    term_domains = np.array([t.get("domain") or "" for t in index])
    labeled_mask = term_domains != ""
    n_labeled = labeled_mask.sum()

    # Domain counts in full labeled set
    domain_counts = {d: int((term_domains == d).sum()) for d in domains}
    print(f"[Lens VI] {n_labeled} labeled terms across {len(domains)} domains")
    for d in domains:
        print(f"  {d}: {domain_counts[d]}")

    results = []
    n_tests = n_features * len(domains)

    for fi in range(n_features):
        feat_acts = activations[:, fi]

        # Skip dead features
        if feat_acts.max() == 0:
            continue

        # Top-N activating terms (by activation value)
        top_idx = np.argsort(feat_acts)[::-1][:top_n]
        top_domains = term_domains[top_idx]
        top_labeled = top_domains[top_domains != ""]

        # Domain distribution in top-N
        domain_dist = {}
        for d in domains:
            n_in_top = int((top_labeled == d).sum())
            domain_dist[d] = n_in_top

        # Fisher's exact test for each domain
        enrichments = []
        for d in domains:
            a = domain_dist[d]                                 # in top AND domain d
            b = len(top_labeled) - a                           # in top AND NOT domain d
            c = domain_counts[d] - a                           # NOT in top AND domain d
            d_val = (n_labeled - domain_counts[d]) - b         # NOT in top AND NOT domain d

            _, p_value = stats.fisher_exact(
                [[a, b], [c, d_val]], alternative="greater"
            )
            odds_ratio = (a * d_val) / max(b * c, 1)
            enrichments.append({
                "domain": d,
                "count_in_top": a,
                "expected": domain_counts[d] * len(top_labeled) / n_labeled
                if n_labeled > 0 else 0,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
            })

        # Domain Selectivity Index: Herfindahl-Hirschman concentration index
        # DSI = sum(p_i^2) where p_i = proportion of domain i in top-N labeled
        # DSI = 1 means all mass in one domain; DSI = 1/K means uniform.
        # Note: unbalanced domain sizes mean DSI is not directly comparable
        # across domains — a feature enriched for a small domain (labor=30)
        # will have higher DSI than one equally enriched for a large domain
        # (civil=136) at the same count ratio.
        if len(top_labeled) > 0:
            proportions = np.array(
                [domain_dist[d] / len(top_labeled) for d in domains]
            )
            dsi = float(np.sum(proportions ** 2))
        else:
            dsi = 0.0

        # Best domain
        best_domain = max(enrichments, key=lambda e: e["count_in_top"])

        # Top terms with metadata
        top_terms = []
        for idx in top_idx[:20]:
            t = index[idx]
            top_terms.append({
                "term_idx": int(idx),
                "en": t["en"],
                "domain": t.get("domain") or "none",
                "activation": float(feat_acts[idx]),
            })

        results.append({
            "feature_idx": fi,
            "dsi": float(dsi),
            "best_domain": best_domain["domain"],
            "best_domain_count": best_domain["count_in_top"],
            "n_labeled_in_top": int(len(top_labeled)),
            "domain_distribution": domain_dist,
            "enrichments": enrichments,
            "top_terms": top_terms,
            "mean_activation": float(feat_acts[feat_acts > 0].mean())
            if feat_acts.max() > 0 else 0.0,
            "n_activating": int((feat_acts > 0).sum()),
        })

    # Sort by DSI (most selective first)
    results.sort(key=lambda r: r["dsi"], reverse=True)
    return results


def holm_correction(results: list[dict]) -> list[dict]:
    """Apply Holm-Bonferroni step-down correction with cummax enforcement."""
    # Collect all (feature_idx, domain, p_value) triples
    all_pvals = []
    for r in results:
        for e in r["enrichments"]:
            all_pvals.append((r["feature_idx"], e["domain"], e["p_value"]))

    if not all_pvals:
        return results

    # Sort by p-value (ascending)
    all_pvals.sort(key=lambda x: x[2])
    n_total = len(all_pvals)

    # Compute raw adjusted p-values: p_adj[i] = p[i] * (n - rank_i)
    raw_adj = [pval * (n_total - rank) for rank, (_, _, pval) in enumerate(all_pvals)]

    # Enforce monotonicity (step-down cummax): each adjusted p must be
    # >= all previous ones in the sorted order. Without this, the Holm
    # procedure can produce decreasing adjusted p-values, which is invalid.
    for i in range(1, len(raw_adj)):
        raw_adj[i] = max(raw_adj[i], raw_adj[i - 1])

    # Cap at 1.0 and build lookup
    adjusted = {}
    for i, (fi, domain, _) in enumerate(all_pvals):
        adjusted[(fi, domain)] = min(raw_adj[i], 1.0)

    # Write back
    for r in results:
        for e in r["enrichments"]:
            key = (r["feature_idx"], e["domain"])
            e["p_adjusted"] = adjusted.get(key, 1.0)
            e["significant"] = e["p_adjusted"] < 0.05

    return results


def compute_summary(results: list[dict], domains: list[str]) -> dict:
    """Compute aggregate statistics across all features."""
    n_features = len(results)
    if n_features == 0:
        return {}

    dsi_values = [r["dsi"] for r in results]

    # Count features significantly enriched per domain
    domain_feature_counts = {d: 0 for d in domains}
    n_significant = 0
    for r in results:
        for e in r["enrichments"]:
            if e.get("significant", False):
                domain_feature_counts[e["domain"]] += 1
                n_significant += 1

    # Unique features with at least one significant enrichment
    features_with_sig = sum(
        1 for r in results
        if any(e.get("significant", False) for e in r["enrichments"])
    )

    return {
        "n_active_features": n_features,
        "n_features_with_significant_enrichment": features_with_sig,
        "n_significant_tests": n_significant,
        "dsi_mean": float(np.mean(dsi_values)),
        "dsi_median": float(np.median(dsi_values)),
        "dsi_max": float(np.max(dsi_values)),
        "dsi_p90": float(np.percentile(dsi_values, 90)),
        "domain_feature_counts": domain_feature_counts,
    }


def main(args: argparse.Namespace) -> int:
    model_label = args.model
    print(f"[Lens VI] Loading data for {model_label} ...")
    activations, index, domains = load_data(model_label, args.expansion, args.k)
    print(f"[Lens VI] Activations: {activations.shape}")
    print(f"[Lens VI] Domains: {domains}")

    print("\n[Lens VI] Computing domain enrichment ...")
    results = domain_enrichment(activations, index, domains, top_n=args.top_n)
    print(f"[Lens VI] {len(results)} active features analyzed")

    print("[Lens VI] Applying Holm-Bonferroni correction ...")
    results = holm_correction(results)

    summary = compute_summary(results, domains)
    summary["model_label"] = model_label

    # Print highlights
    print(f"\n[Lens VI] === Summary ({model_label}) ===")
    print(f"  Active features: {summary['n_active_features']}")
    print(f"  Features with significant enrichment: "
          f"{summary['n_features_with_significant_enrichment']}")
    print(f"  DSI: mean={summary['dsi_mean']:.3f}, "
          f"median={summary['dsi_median']:.3f}, "
          f"max={summary['dsi_max']:.3f}, "
          f"p90={summary['dsi_p90']:.3f}")
    print(f"  Domain feature counts:")
    for d, c in sorted(summary["domain_feature_counts"].items(),
                       key=lambda x: -x[1]):
        print(f"    {d}: {c} features")

    # Print top-10 most selective features
    print(f"\n[Lens VI] === Top 10 most selective features ===")
    for r in results[:10]:
        top_3 = [t["en"] for t in r["top_terms"][:3]]
        sig_domains = [
            e["domain"] for e in r["enrichments"] if e.get("significant")
        ]
        print(
            f"  Feature {r['feature_idx']:4d}: DSI={r['dsi']:.3f}, "
            f"best={r['best_domain']} ({r['best_domain_count']}/{r['n_labeled_in_top']}), "
            f"sig={sig_domains}, "
            f"top=[{', '.join(top_3)}]"
        )

    # Save results
    suffix = make_suffix(model_label, args.expansion, args.k)

    enrichment_path = RESULTS_DIR / f"domain_enrichment{suffix}.json"
    with open(enrichment_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Lens VI] Saved: {enrichment_path}")

    summary_path = RESULTS_DIR / f"feature_summary{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Lens VI] Saved: {summary_path}")

    # Top features (compact version for quick inspection)
    top_features = []
    for r in results[:args.top_n]:
        top_features.append({
            "feature_idx": r["feature_idx"],
            "dsi": r["dsi"],
            "best_domain": r["best_domain"],
            "best_domain_count": r["best_domain_count"],
            "n_labeled_in_top": r["n_labeled_in_top"],
            "significant_domains": [
                e["domain"] for e in r["enrichments"]
                if e.get("significant")
            ],
            "top_terms": [
                {"en": t["en"], "domain": t["domain"]}
                for t in r["top_terms"][:10]
            ],
        })

    top_path = RESULTS_DIR / f"top_features{suffix}.json"
    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2, ensure_ascii=False)
    print(f"[Lens VI] Saved: {top_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="BGE-EN-large",
                        help="Embedding label (directory name under embeddings/)")
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--top-n", type=int, default=50,
                        help="Top-N terms per feature for enrichment analysis")
    sys.exit(main(parser.parse_args()))
