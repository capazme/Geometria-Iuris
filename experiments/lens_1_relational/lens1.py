"""
Lens I — Relational Distance Structure.

Implements the full Lens I analysis pipeline (Ch.3 §3.1):

  §3.1.1  Background term domain assignment (k-NN signal test)
          Intra-domain vs inter-domain distances (Mann-Whitney U)
          Legal vs control signal (Mann-Whitney U)
  §3.1.2  Domain topology K×K matrix
  §3.1.4  Within-tradition RSA robustness + Cross-tradition RSA

Usage
-----
    cd experiments/
    python -m lens_1_relational.lens1                       # full run, all defaults
    python -m lens_1_relational.lens1 --n-perm 200          # faster (dev/debug)
    python -m lens_1_relational.lens1 --section 3.1.1       # only background + signal
    python -m lens_1_relational.lens1 --section 3.1.4       # only RSA

Outputs
-------
    lens_1_relational/results/lens1_results.json
    lens_1_relational/results/background_review.csv   (open in Excel/Numbers to annotate)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.embeddings import load_precomputed
from shared.statistical import (
    RSAResult,
    compute_rdm,
    holm_correction,
    mannwhitney_with_r,
    rsa,
    upper_tri,
)
from lens_1_relational.domain_assignment import (
    assign_domains,
    build_review_csv,
)

RESULTS_DIR = Path(__file__).parent / "results"
EMB_DIR = ROOT / "data" / "processed" / "embeddings"
CONFIG_PATH = ROOT / "models" / "config.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Return (weird_labels, sinic_labels, bilingual_pairs) from config.yaml.

    bilingual_pairs: list of (en_label, zh_label) for bilingual control models.
    Each bilingual model produces two embedding dirs: {label}-EN and {label}-ZH.
    """
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    weird = [m["label"] for m in raw.get("weird", [])]
    sinic = [m["label"] for m in raw.get("sinic", [])]
    bilingual = [
        (f"{m['label']}-EN", f"{m['label']}-ZH")
        for m in raw.get("bilingual", [])
    ]
    return weird, sinic, bilingual


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_terms_by_tier(index: list[dict]) -> tuple[list, list[int], list[int], list[int]]:
    """
    Split the full term index into tier-specific lists and positional indices.

    Returns
    -------
    terms_core, core_idx, bg_idx, ctrl_idx
    where *_idx are positions into the original index / vectors array.
    """
    core_idx = [i for i, t in enumerate(index) if t["tier"] == "core" and t["domain"]]
    bg_idx   = [i for i, t in enumerate(index) if t["tier"] == "background"]
    ctrl_idx = [i for i, t in enumerate(index) if t["tier"] == "control"]
    terms_core = [index[i] for i in core_idx]
    return terms_core, core_idx, bg_idx, ctrl_idx


def _load_vecs_for_model(
    label: str,
    core_idx: list[int],
    bg_idx: list[int],
    ctrl_idx: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed embeddings and split by tier."""
    vecs, _ = load_precomputed(label, EMB_DIR)
    return vecs[core_idx], vecs[bg_idx], vecs[ctrl_idx]


# ---------------------------------------------------------------------------
# Domain analysis helpers (§3.1)
# ---------------------------------------------------------------------------

def _intra_inter_split(
    rdm_core: np.ndarray,
    domains: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split upper-triangle distances into intra-domain and inter-domain sets.

    Returns (intra_dists, inter_dists).
    """
    n = len(rdm_core)
    rows, cols = np.triu_indices(n, k=1)
    dom_arr = np.array(domains)
    same = dom_arr[rows] == dom_arr[cols]
    tri = rdm_core[rows, cols]
    return tri[same], tri[~same]


def _domain_topology(
    rdm_core: np.ndarray,
    domains: list[str],
) -> tuple[list[str], np.ndarray]:
    """
    Build a K×K matrix of mean inter-domain cosine distances.

    Diagonal entries = mean intra-domain distance (upper triangle only).
    Off-diagonal entries = mean distance between all pairs across two domains.

    Returns (domain_labels_sorted, K×K matrix).
    """
    domain_labels = sorted(set(domains))
    dom_arr = np.array(domains)
    k = len(domain_labels)
    topo = np.zeros((k, k), dtype=np.float32)

    for i, d1 in enumerate(domain_labels):
        idx1 = np.where(dom_arr == d1)[0]
        for j, d2 in enumerate(domain_labels):
            idx2 = np.where(dom_arr == d2)[0]
            sub = rdm_core[np.ix_(idx1, idx2)]
            if i == j:
                topo[i, j] = float(upper_tri(sub).mean()) if len(idx1) > 1 else 0.0
            else:
                topo[i, j] = float(sub.mean())

    return domain_labels, topo


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _rsa_to_dict(label_a: str, label_b: str, result: RSAResult) -> dict:
    return {
        "model_a": label_a,
        "model_b": label_b,
        "rho": round(result.rho, 4),
        "p_value": round(result.p_value, 4),
        "r_squared": round(result.r_squared, 4),
        "ci_low": round(result.ci.low, 4),
        "ci_high": round(result.ci.high, 4),
    }


def _mw_to_dict(res) -> dict:
    return {
        "statistic": round(res.statistic, 2),
        "p_value": res.p_value,
        "effect_r": round(res.effect_r, 4),
        "n_x": res.n_x,
        "n_y": res.n_y,
        "median_x": round(res.median_x, 4),
        "median_y": round(res.median_y, 4),
    }


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def run_section_311(
    weird_labels: list[str],
    k: int = 7,
    primary_model: str | None = None,
) -> dict:
    """
    §3.1.1 — Background term domain assignment via k-NN.

    Uses the primary WEIRD model (first in config, BGE-EN-large) for the
    main assignment. Returns summary stats; saves background_review.csv.
    """
    primary = primary_model or weird_labels[0]
    print(f"\n[§3.1.1] Background k-NN assignment  (k={k}, model={primary})")

    vecs, index = load_precomputed(primary, EMB_DIR)
    terms_core, core_idx, bg_idx, ctrl_idx = _load_terms_by_tier(index)
    terms_bg = [index[i] for i in bg_idx]

    vecs_core = vecs[core_idx]
    vecs_bg   = vecs[bg_idx]
    labels_core = [t["domain"] for t in terms_core]

    t0 = time.perf_counter()
    assignments = assign_domains(vecs_bg, vecs_core, labels_core, k=k)
    elapsed = time.perf_counter() - t0
    print(f"  assigned {len(assignments)} terms in {elapsed:.1f}s")

    csv_path = RESULTS_DIR / "background_review.csv"
    build_review_csv(terms_bg, assignments, terms_core, csv_path)
    print(f"  review CSV → {csv_path.relative_to(ROOT.parent)}")

    # Summary statistics
    from collections import Counter
    domain_counts = Counter(a["assigned_domain"] for a in assignments)
    confidences = np.array([a["confidence"] for a in assignments])
    low_conf = int((confidences < 4 / k).sum())

    print(f"  domain distribution: {dict(sorted(domain_counts.items()))}")
    print(f"  confidence — mean={confidences.mean():.2f}  "
          f"median={np.median(confidences):.2f}  "
          f"low (<{4/k:.2f}): {low_conf} ({100*low_conf/len(assignments):.1f}%)")

    return {
        "model": primary,
        "k": k,
        "n_background": len(assignments),
        "domain_counts": dict(domain_counts),
        "confidence_mean": round(float(confidences.mean()), 4),
        "confidence_median": round(float(np.median(confidences)), 4),
        "low_confidence_n": low_conf,
        "low_confidence_threshold": round(4 / k, 4),
        "review_csv": str(csv_path),
    }


def run_section_31(
    weird_labels: list[str],
) -> dict:
    """
    §3.1 — Relational distance structure (WEIRD models only).

    §3.1.1  Intra-domain vs inter-domain distances (Mann-Whitney U)
    §3.1.1  Legal (core) vs control distances (Mann-Whitney U)
    §3.1.2  Domain topology K×K matrix
    """
    print("\n[§3.1] Domain signal tests (WEIRD models)")

    # Load index once (shared across all WEIRD models)
    _, index = load_precomputed(weird_labels[0], EMB_DIR)
    terms_core, core_idx, bg_idx, ctrl_idx = _load_terms_by_tier(index)
    domains = [t["domain"] for t in terms_core]
    n_core  = len(core_idx)
    n_ctrl  = len(ctrl_idx)

    per_model: dict[str, dict] = {}

    for label in weird_labels:
        print(f"  {label}")
        vecs_core, vecs_bg, vecs_ctrl = _load_vecs_for_model(
            label, core_idx, bg_idx, ctrl_idx
        )

        # --- §3.1.1 intra vs inter-domain ---
        rdm_core = compute_rdm(vecs_core)
        intra, inter = _intra_inter_split(rdm_core, domains)
        mw_81 = mannwhitney_with_r(intra, inter, alternative="less")
        print(f"    §3.1.1  intra(med={mw_81.median_x:.3f}) vs "
              f"inter(med={mw_81.median_y:.3f})  "
              f"r={mw_81.effect_r:+.3f}  p={mw_81.p_value:.2e}")

        # --- §3.1.1 legal vs control ---
        combined = np.vstack([vecs_core, vecs_ctrl])
        rdm_lc = compute_rdm(combined)
        legal_legal = upper_tri(rdm_lc[:n_core, :n_core])
        legal_ctrl  = rdm_lc[:n_core, n_core:].flatten()
        mw_82 = mannwhitney_with_r(legal_legal, legal_ctrl, alternative="less")
        print(f"    §3.1.1  legal(med={mw_82.median_x:.3f}) vs "
              f"ctrl(med={mw_82.median_y:.3f})  "
              f"r={mw_82.effect_r:+.3f}  p={mw_82.p_value:.2e}")

        # --- §3.1.2 domain topology ---
        domain_labels, topo = _domain_topology(rdm_core, domains)

        np.savez_compressed(
            RESULTS_DIR / "distances" / f"{label}.npz",
            intra=intra, inter=inter, legal=legal_legal, control=legal_ctrl,
        )

        per_model[label] = {
            "intra_vs_inter": _mw_to_dict(mw_81),
            "legal_vs_control": _mw_to_dict(mw_82),
            "domain_topology": {
                "domains": domain_labels,
                "matrix": topo.tolist(),
            },
        }

    return {"per_model": per_model, "n_core": n_core, "n_control": n_ctrl}


def run_rsa_pairs(
    labels_a: list[str],
    labels_b: list[str],
    rdms: dict[str, np.ndarray],
    section_name: str,
    n_perm: int,
    n_boot: int,
    save_dir: Path | None = None,
) -> list[dict]:
    """Run RSA for all pairs (labels_a × labels_b) and return serializable results."""
    pairs = list(combinations(labels_a, 2)) if labels_a is labels_b else list(product(labels_a, labels_b))
    results = []
    for la, lb in pairs:
        t0 = time.perf_counter()
        result = rsa(rdms[la], rdms[lb], n_perm=n_perm, n_boot=n_boot)
        elapsed = time.perf_counter() - t0
        if save_dir is not None:
            np.savez_compressed(
                save_dir / f"{la}_x_{lb}.npz",
                null=result.null_distribution,
                bootstrap=result.ci.distribution,
            )
        d = _rsa_to_dict(la, lb, result)
        print(f"    {la} × {lb}  ρ={d['rho']:+.3f}  "
              f"r²={d['r_squared']:.3f}  "
              f"CI=[{d['ci_low']:.3f},{d['ci_high']:.3f}]  "
              f"p={d['p_value']:.4f}  ({elapsed:.0f}s)")
        results.append(d)
    return results


def run_section_314(
    weird_labels: list[str],
    sinic_labels: list[str],
    n_perm: int,
    n_boot: int,
    bilingual_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """
    §3.1.4 Within-tradition RSA + Cross-tradition RSA.

    Builds one RDM per model (core terms only) then computes:
      - 3 within-WEIRD pairs
      - 3 within-Sinic pairs
      - 9 cross-tradition pairs
      - N within-bilingual pairs (β control: same model, EN↔ZH)
    """
    bilingual_pairs = bilingual_pairs or []
    print(f"\n[§3.1.4] RSA  (n_perm={n_perm}, n_boot={n_boot})")

    # Collect all labels that need RDMs
    bilingual_labels = [en for en, zh in bilingual_pairs] + [zh for en, zh in bilingual_pairs]
    all_labels = weird_labels + sinic_labels + bilingual_labels
    rdms: dict[str, np.ndarray] = {}

    print("  Computing RDMs...")
    _, index = load_precomputed(weird_labels[0], EMB_DIR)
    _, core_idx, _, _ = _load_terms_by_tier(index)

    rdm_dir = RESULTS_DIR / "rdms"
    for label in all_labels:
        vecs, _ = load_precomputed(label, EMB_DIR)
        rdms[label] = compute_rdm(vecs[core_idx])
        np.savez_compressed(rdm_dir / f"{label}.npz", rdm=rdms[label])
        print(f"    {label}  RDM shape={rdms[label].shape}")

    dist_dir = RESULTS_DIR / "distributions"
    print("  Within-WEIRD RSA:")
    within_weird = run_rsa_pairs(
        weird_labels, weird_labels, rdms,
        "within_weird", n_perm, n_boot, save_dir=dist_dir,
    )

    print("  Within-Sinic RSA:")
    within_sinic = run_rsa_pairs(
        sinic_labels, sinic_labels, rdms,
        "within_sinic", n_perm, n_boot, save_dir=dist_dir,
    )

    print("  Cross-tradition RSA:")
    cross = run_rsa_pairs(
        weird_labels, sinic_labels, rdms,
        "cross", n_perm, n_boot, save_dir=dist_dir,
    )

    # β control: within-bilingual pairs (same model, EN↔ZH)
    within_bilingual: list[dict] = []
    if bilingual_pairs:
        print("  Within-bilingual RSA (β control):")
        for en_label, zh_label in bilingual_pairs:
            pair_results = run_rsa_pairs(
                [en_label], [zh_label], rdms,
                "within_bilingual", n_perm, n_boot, save_dir=dist_dir,
            )
            within_bilingual.extend(pair_results)

    # Holm correction across ALL Mantel p-values
    all_pairs = within_weird + within_sinic + cross + within_bilingual
    raw_ps = [r["p_value"] for r in all_pairs]
    adj_ps = holm_correction(raw_ps)
    for r, p_adj in zip(all_pairs, adj_ps):
        r["p_holm"] = round(p_adj, 6)

    # Summary: intra-tradition vs cross-tradition mean rho
    rho_weird  = np.mean([r["rho"] for r in within_weird])
    rho_sinic  = np.mean([r["rho"] for r in within_sinic])
    rho_cross  = np.mean([r["rho"] for r in cross])
    print(f"\n  Summary:  within-WEIRD ρ̄={rho_weird:.3f}  "
          f"within-Sinic ρ̄={rho_sinic:.3f}  cross ρ̄={rho_cross:.3f}")
    drop = rho_weird - rho_cross
    print(f"  Cross-tradition drop (WEIRD→cross): Δρ = {drop:.3f}")

    summary = {
        "mean_rho_within_weird": round(float(rho_weird), 4),
        "mean_rho_within_sinic": round(float(rho_sinic), 4),
        "mean_rho_cross": round(float(rho_cross), 4),
        "cross_tradition_drop": round(float(drop), 4),
    }

    result = {
        "within_weird": within_weird,
        "within_sinic": within_sinic,
        "cross_tradition": cross,
    }

    if within_bilingual:
        rho_bi = np.mean([r["rho"] for r in within_bilingual])
        print(f"  β control: within-bilingual ρ̄={rho_bi:.3f}")
        print(f"  Gap persists? bilingual ρ̄ ({rho_bi:.3f}) vs monolingual cross ρ̄ ({rho_cross:.3f})")
        summary["mean_rho_within_bilingual"] = round(float(rho_bi), 4)
        result["within_bilingual"] = within_bilingual

    result["summary"] = summary
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Lens I — Relational Distance Structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--section",
        choices=["3.1.1", "3.1", "3.1.4", "all"],
        default="all",
        help="Which section(s) to run",
    )
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Permutations for Mantel test")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap iterations for CI")
    parser.add_argument("--k", type=int, default=7,
                        help="k for k-NN domain assignment (§3.1.1)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip figure generation after pipeline")
    parser.add_argument("--emb-dir", default=None,
                        help="Override embedding directory (default: data/processed/embeddings)")
    args = parser.parse_args(argv)

    # Allow overriding the embedding directory (e.g. for attested-context pool)
    global EMB_DIR
    if args.emb_dir:
        EMB_DIR = ROOT / args.emb_dir
        print(f"  [override] EMB_DIR = {EMB_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rdms").mkdir(exist_ok=True)
    (RESULTS_DIR / "distances").mkdir(exist_ok=True)
    (RESULTS_DIR / "distributions").mkdir(exist_ok=True)
    weird_labels, sinic_labels, bilingual_pairs = _load_config()

    print("=" * 60)
    print("Lens I — Relational Distance Structure")
    print(f"  WEIRD : {weird_labels}")
    print(f"  Sinic : {sinic_labels}")
    if bilingual_pairs:
        print(f"  Bilingual β: {bilingual_pairs}")
    print(f"  n_perm={args.n_perm}  n_boot={args.n_boot}  k={args.k}")
    print("=" * 60)

    output: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "n_perm": args.n_perm,
            "n_boot": args.n_boot,
            "k_nn": args.k,
            "weird_models": weird_labels,
            "sinic_models": sinic_labels,
            "bilingual_models": [f"{en}/{zh}" for en, zh in bilingual_pairs],
        }
    }

    t_start = time.perf_counter()

    if args.section in ("3.1.1", "all"):
        output["section_311"] = run_section_311(weird_labels, k=args.k)

    if args.section in ("3.1", "all"):
        output["section_31"] = run_section_31(weird_labels)

    if args.section in ("3.1.4", "all"):
        output["section_314"] = run_section_314(
            weird_labels, sinic_labels,
            n_perm=args.n_perm, n_boot=args.n_boot,
            bilingual_pairs=bilingual_pairs,
        )

    total = time.perf_counter() - t_start
    output["meta"]["elapsed_seconds"] = round(total, 1)

    out_path = RESULTS_DIR / "lens1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done in {total:.0f}s  →  {out_path.relative_to(ROOT.parent)}")

    if not args.no_viz:
        from lens_1_relational.viz import run_viz
        run_viz(RESULTS_DIR, output)


if __name__ == "__main__":
    main()
