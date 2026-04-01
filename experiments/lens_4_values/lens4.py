"""
Lens IV — Value Axis Projection.

Implements the full Lens IV analysis pipeline (Ch.3 §3.3):

  §3.3.1  Axis construction — building value dimensions from antonym pairs
  §3.3.2  Cross-tradition alignment — Spearman rho + bootstrap CI per axis per pair
  §3.3.3  Which axes diverge most?

Usage
-----
    cd experiments/
    python -m lens_4_values.lens4                    # full run
    python -m lens_4_values.lens4 --section 3.3.1    # axis quality only
    python -m lens_4_values.lens4 --n-boot 1000
    python -m lens_4_values.lens4 --no-viz

Outputs
-------
    lens_4_values/results/lens4_results.json
    lens_4_values/results/figures/
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
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.embeddings import EmbeddingClient, load_precomputed
from shared.statistical import (
    GenericBootstrapCI,
    bootstrap_ci_generic,
    mannwhitney_with_r,
    permutation_test_groups,
)

RESULTS_DIR = Path(__file__).parent / "results"
EMB_DIR = ROOT / "data" / "processed" / "embeddings"
CONFIG_PATH = ROOT / "models" / "config.yaml"
AXES_PATH = Path(__file__).parent / "value_axes.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> tuple[list[str], list[str], dict[str, str]]:
    """Return (weird_labels, sinic_labels, label_to_lang) from config.yaml."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    weird = [m["label"] for m in raw.get("weird", [])]
    sinic = [m["label"] for m in raw.get("sinic", [])]
    label_to_lang: dict[str, str] = {}
    for group in ("weird", "sinic"):
        for m in raw.get(group, []):
            label_to_lang[m["label"]] = m["lang"]
    return weird, sinic, label_to_lang


def _load_model_id_map() -> dict[str, str]:
    """Return {label: hf_model_id} from config.yaml."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    mapping: dict[str, str] = {}
    for group in ("weird", "sinic"):
        for m in raw.get(group, []):
            mapping[m["label"]] = m["id"]
    return mapping


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all_vecs(labels: list[str]) -> tuple[
    dict[str, np.ndarray],
    list[dict],
    list[int],
    list[dict],
]:
    """
    Load precomputed vectors for all models.

    Returns
    -------
    vecs_all : {label: (9472, dim)} full embedding matrices
    index    : 9472 term records
    core_idx : positional indices of the 397 core terms
    terms_core : 397 core term records
    """
    _, index = load_precomputed(labels[0], EMB_DIR)
    core_idx = [i for i, t in enumerate(index) if t["tier"] == "core" and t["domain"]]
    terms_core = [index[i] for i in core_idx]

    vecs_all: dict[str, np.ndarray] = {}
    for label in labels:
        vecs, _ = load_precomputed(label, EMB_DIR)
        vecs_all[label] = vecs

    return vecs_all, index, core_idx, terms_core


def _load_value_axes() -> dict[str, dict]:
    """Load 3 axes with en_pairs + zh_pairs from value_axes.yaml."""
    with AXES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Index lookup helpers
# ---------------------------------------------------------------------------

def _build_en_lookup(index: list[dict]) -> dict[str, int]:
    """Map lowercase EN term -> first index in pool."""
    lookup: dict[str, int] = {}
    for i, t in enumerate(index):
        key = t["en"].lower()
        if key not in lookup:
            lookup[key] = i
    return lookup


def _build_zh_lookup(index: list[dict]) -> dict[str, int]:
    """Map zh_canonical (traditional) -> first index in pool."""
    lookup: dict[str, int] = {}
    for i, t in enumerate(index):
        zh = t.get("zh_canonical", "")
        if zh and zh not in lookup:
            lookup[zh] = i
    return lookup


def _simplified_to_traditional(text: str) -> str:
    """Convert simplified Chinese to traditional for index lookup."""
    try:
        import opencc
        converter = opencc.OpenCC("s2t")
        return converter.convert(text)
    except ImportError:
        return text


# ---------------------------------------------------------------------------
# Axis construction (Kozlowski 2019)
# ---------------------------------------------------------------------------

def _get_pair_vectors(
    pairs: list[list[str]],
    lang: str,
    vecs_pool: np.ndarray,
    en_lookup: dict[str, int],
    zh_lookup: dict[str, int],
    embed_client: EmbeddingClient | None,
    model_id: str | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Resolve antonym pairs to embedding vectors.

    First tries precomputed pool lookup. Falls back to on-the-fly embedding
    via EmbeddingClient if the term is not in the pool.
    """
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for pos_term, neg_term in pairs:
        vecs_pair = []
        for term in (pos_term, neg_term):
            idx = None
            if lang == "en":
                idx = en_lookup.get(term.lower())
            elif lang == "zh":
                trad = _simplified_to_traditional(term)
                idx = zh_lookup.get(trad)
                if idx is None:
                    idx = zh_lookup.get(term)

            if idx is not None:
                vecs_pair.append(vecs_pool[idx])
            elif embed_client is not None and model_id is not None:
                vec = embed_client.embed([term], model_id)[0]
                vecs_pair.append(vec)
            else:
                vecs_pair.append(None)

        if vecs_pair[0] is not None and vecs_pair[1] is not None:
            result.append((vecs_pair[0], vecs_pair[1]))
    return result


def _build_axis(
    pair_vectors: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Kozlowski difference-vector method:
    axis = L2_normalize(mean(embed(pos_i) - embed(neg_i)))
    """
    diffs = np.array([pos - neg for pos, neg in pair_vectors])
    mean_diff = diffs.mean(axis=0)
    norm = np.linalg.norm(mean_diff)
    if norm > 0:
        mean_diff /= norm
    return mean_diff


def _project_terms(
    vecs_core: np.ndarray,
    axis_vec: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of core term vectors with axis vector -> (n_core,)."""
    return (vecs_core @ axis_vec).astype(np.float64)


# ---------------------------------------------------------------------------
# §3.3.1 — Axis construction quality
# ---------------------------------------------------------------------------

def run_section_331(
    axes_all: dict[str, dict[str, np.ndarray]],
    pair_info: dict[str, dict[str, dict]],
    all_labels: list[str],
    axis_names: list[str],
) -> dict:
    """
    §3.3.1 — Axis construction quality.

    Reports: antonym pair count per axis, sanity check (positive pairs project > 0),
    inter-axis cosine similarity (orthogonality diagnostic).
    """
    print("\n[§3.3.1] Axis construction quality")

    per_model: dict[str, dict] = {}
    for label in all_labels:
        model_axes = axes_all[label]
        model_info = pair_info[label]

        axis_report: dict[str, dict] = {}
        for ax_name in axis_names:
            info = model_info[ax_name]
            axis_report[ax_name] = {
                "n_pairs_used": info["n_pairs_used"],
                "n_pairs_total": info["n_pairs_total"],
                "positive_correct": info["positive_correct"],
                "negative_correct": info["negative_correct"],
                "sanity_pass": info["positive_correct"] + info["negative_correct"],
                "sanity_total": info["n_pairs_used"] * 2,
            }

        # Inter-axis cosine similarity
        orthogonality: dict[str, float] = {}
        for a1, a2 in combinations(axis_names, 2):
            cos_sim = float(model_axes[a1] @ model_axes[a2])
            orthogonality[f"{a1}_vs_{a2}"] = round(cos_sim, 4)

        per_model[label] = {
            "axes": axis_report,
            "orthogonality": orthogonality,
        }

        # Print summary
        for ax_name in axis_names:
            r = axis_report[ax_name]
            print(f"  {label:20s}  {ax_name:25s}  pairs={r['n_pairs_used']}/{r['n_pairs_total']}  "
                  f"sanity={r['sanity_pass']}/{r['sanity_total']}")

    return {"per_model": per_model}


# ---------------------------------------------------------------------------
# §3.3.2 — Cross-tradition alignment
# ---------------------------------------------------------------------------

def _spearman_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> GenericBootstrapCI:
    """
    Spearman rho between two score vectors, with row-resample bootstrap CI.

    The statistic function computes Spearman rho on resampled (score_a, score_b) pairs.
    """
    stacked = np.column_stack([scores_a, scores_b])

    def stat_fn(data: np.ndarray) -> float:
        return float(spearmanr(data[:, 0], data[:, 1]).statistic)

    return bootstrap_ci_generic(stacked, stat_fn, n_boot=n_boot, seed=seed)


def run_section_332(
    scores_all: dict[str, dict[str, np.ndarray]],
    weird_labels: list[str],
    sinic_labels: list[str],
    axis_names: list[str],
    n_boot: int = 1000,
) -> dict:
    """
    §3.3.2 — Cross-tradition alignment: Spearman rho + bootstrap CI.

    For each of 15 pairs x 3 axes: Spearman rho + bootstrap CI.
    Summary: Mann-Whitney on cross rho vs within rho per axis.
    """
    print(f"\n[§3.3.2] Cross-tradition alignment (n_boot={n_boot})")

    cross_pairs = list(product(weird_labels, sinic_labels))
    within_weird_pairs = list(combinations(weird_labels, 2))
    within_sinic_pairs = list(combinations(sinic_labels, 2))

    all_pairs = (
        [(la, lb, "cross") for la, lb in cross_pairs]
        + [(la, lb, "within_weird") for la, lb in within_weird_pairs]
        + [(la, lb, "within_sinic") for la, lb in within_sinic_pairs]
    )

    per_pair: list[dict] = []
    for la, lb, group in all_pairs:
        for ax_name in axis_names:
            t0 = time.perf_counter()
            sa = scores_all[la][ax_name]
            sb = scores_all[lb][ax_name]

            rho_raw = float(spearmanr(sa, sb).statistic)
            ci = _spearman_bootstrap(sa, sb, n_boot=n_boot)
            elapsed = time.perf_counter() - t0

            entry = {
                "model_a": la,
                "model_b": lb,
                "group": group,
                "axis": ax_name,
                "rho": round(rho_raw, 4),
                "ci_low": round(ci.ci_low, 4),
                "ci_high": round(ci.ci_high, 4),
                "bootstrap_estimate": round(ci.estimate, 4),
            }
            per_pair.append(entry)
            print(f"    {la:15s} × {lb:15s}  {ax_name:25s}  "
                  f"ρ={rho_raw:+.4f}  CI=[{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]  "
                  f"({elapsed:.1f}s)")

    # Summary per axis: permutation test cross vs within
    summary_per_axis: dict[str, dict] = {}
    for ax_name in axis_names:
        cross_rhos = np.array([
            e["rho"] for e in per_pair
            if e["axis"] == ax_name and e["group"] == "cross"
        ])
        within_rhos = np.array([
            e["rho"] for e in per_pair
            if e["axis"] == ax_name and e["group"] in ("within_weird", "within_sinic")
        ])
        pt = permutation_test_groups(cross_rhos, within_rhos, alternative="less")
        summary_per_axis[ax_name] = {
            "mean_cross_rho": round(float(cross_rhos.mean()), 4),
            "mean_within_rho": round(float(within_rhos.mean()), 4),
            "perm_p_value": pt.p_value,
            "effect_r": round(pt.effect_r, 4),
        }
        print(f"\n  [{ax_name}] cross ρ̄={cross_rhos.mean():.4f}  "
              f"within ρ̄={within_rhos.mean():.4f}  "
              f"r={pt.effect_r:+.4f}  p={pt.p_value:.4f}")

    return {
        "n_boot": n_boot,
        "per_pair": per_pair,
        "summary_per_axis": summary_per_axis,
    }


# ---------------------------------------------------------------------------
# §3.3.3 — Which axes diverge most?
# ---------------------------------------------------------------------------

def run_section_333(section_332_results: dict) -> dict:
    """
    §3.3.3 — Compare axes: which axis has the lowest cross-tradition rho?

    Descriptive comparison only. Kruskal-Wallis was removed because the three
    axes are not orthogonal (cosine similarity up to ~0.5) and the 9 cross-
    tradition values per axis come from the same 9 model pairs, violating the
    independence assumption required by KW.
    """
    print("\n[§3.3.3] Which axes diverge most? (descriptive)")

    per_pair = section_332_results["per_pair"]
    axis_names = sorted(set(e["axis"] for e in per_pair))

    cross_rhos_by_axis: dict[str, np.ndarray] = {}
    for ax_name in axis_names:
        rhos = np.array([
            e["rho"] for e in per_pair
            if e["axis"] == ax_name and e["group"] == "cross"
        ])
        cross_rhos_by_axis[ax_name] = rhos
        print(f"  {ax_name:25s}  n={len(rhos)}  mean ρ={rhos.mean():.4f}  "
              f"std={rhos.std():.4f}")

    # Rank axes by mean cross rho (ascending = most divergent first)
    ranked = sorted(
        [(ax, float(cross_rhos_by_axis[ax].mean())) for ax in axis_names],
        key=lambda x: x[1],
    )
    print(f"\n  Ranking (most divergent first): "
          + ", ".join(f"{ax} ({rho:.3f})" for ax, rho in ranked))

    return {
        "cross_rhos_by_axis": {
            ax: {
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "values": [round(float(v), 4) for v in arr],
            }
            for ax, arr in cross_rhos_by_axis.items()
        },
        "ranking_most_divergent_first": [
            {"axis": ax, "mean_cross_rho": rho} for ax, rho in ranked
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Lens IV — Value Axis Projection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--section",
        choices=["3.3.1", "3.3.2", "3.3.3", "all"],
        default="all",
        help="Which section(s) to run",
    )
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Bootstrap iterations for CI")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip figure generation after pipeline")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weird_labels, sinic_labels, label_to_lang = _load_config()
    all_labels = weird_labels + sinic_labels
    label_to_model_id = _load_model_id_map()

    print("=" * 60)
    print("Lens IV — Value Axis Projection")
    print(f"  WEIRD : {weird_labels}")
    print(f"  Sinic : {sinic_labels}")
    print(f"  n_boot={args.n_boot}")
    print("=" * 60)

    # Load all vectors + index
    print("\n[load] Loading embeddings...")
    t0 = time.perf_counter()
    vecs_all, index, core_idx, terms_core = _load_all_vecs(all_labels)
    n_core = len(core_idx)
    print(f"  {len(all_labels)} models loaded, pool={len(index)}, "
          f"core={n_core}  ({time.perf_counter()-t0:.1f}s)")

    # Build lookups
    en_lookup = _build_en_lookup(index)
    zh_lookup = _build_zh_lookup(index)

    # Load axis definitions
    axes_def = _load_value_axes()
    axis_names = list(axes_def.keys())
    print(f"  Axes: {axis_names}")

    # Initialize EmbeddingClient for on-the-fly fallback
    embed_client = EmbeddingClient(CONFIG_PATH, device="cpu")

    # Build axes for all models
    print("\n[axes] Building value axes...")
    axes_all: dict[str, dict[str, np.ndarray]] = {}
    pair_info: dict[str, dict[str, dict]] = {}

    for label in all_labels:
        lang = label_to_lang[label]
        model_id = label_to_model_id[label]
        pair_key = "en_pairs" if lang == "en" else "zh_pairs"
        axes_all[label] = {}
        pair_info[label] = {}

        for ax_name in axis_names:
            pairs = axes_def[ax_name][pair_key]
            pvecs = _get_pair_vectors(
                pairs, lang, vecs_all[label],
                en_lookup, zh_lookup, embed_client, model_id,
            )

            if not pvecs:
                print(f"  WARNING: {label}/{ax_name} — no pairs resolved!")
                axes_all[label][ax_name] = np.zeros(vecs_all[label].shape[1])
                pair_info[label][ax_name] = {
                    "n_pairs_used": 0, "n_pairs_total": len(pairs),
                    "positive_correct": 0, "negative_correct": 0,
                }
                continue

            axis_vec = _build_axis(pvecs)
            axes_all[label][ax_name] = axis_vec

            # Sanity check: positive terms should project > 0, negative < 0
            pos_correct = sum(1 for pos, neg in pvecs if float(pos @ axis_vec) > 0)
            neg_correct = sum(1 for pos, neg in pvecs if float(neg @ axis_vec) < 0)

            pair_info[label][ax_name] = {
                "n_pairs_used": len(pvecs),
                "n_pairs_total": len(pairs),
                "positive_correct": pos_correct,
                "negative_correct": neg_correct,
            }

        print(f"  {label}: {len(axis_names)} axes built")

    # Project core terms onto all axes
    print("\n[project] Projecting core terms...")
    scores_all: dict[str, dict[str, np.ndarray]] = {}
    for label in all_labels:
        core_vecs = vecs_all[label][core_idx]
        scores_all[label] = {}
        for ax_name in axis_names:
            scores_all[label][ax_name] = _project_terms(
                core_vecs, axes_all[label][ax_name],
            )
        print(f"  {label}: {n_core} terms × {len(axis_names)} axes projected")

    # Run sections
    output: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "n_boot": args.n_boot,
            "n_core": n_core,
            "n_pool": len(index),
            "axes": axis_names,
            "weird_models": weird_labels,
            "sinic_models": sinic_labels,
        }
    }

    t_start = time.perf_counter()

    if args.section in ("3.3.1", "all"):
        output["section_331"] = run_section_331(
            axes_all, pair_info, all_labels, axis_names,
        )

    if args.section in ("3.3.2", "all"):
        output["section_332"] = run_section_332(
            scores_all, weird_labels, sinic_labels, axis_names,
            n_boot=args.n_boot,
        )

    if args.section in ("3.3.3", "all"):
        if "section_332" not in output:
            print("\n[§3.3.3] Skipped — requires §3.3.2 results.")
        else:
            output["section_333"] = run_section_333(output["section_332"])

    total = time.perf_counter() - t_start
    output["meta"]["elapsed_seconds"] = round(total, 1)

    # Save per-axis projection scores for viz
    scores_dir = RESULTS_DIR / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for label in all_labels:
        for ax_name in axis_names:
            np.save(
                scores_dir / f"{label}_{ax_name}.npy",
                scores_all[label][ax_name],
            )

    # Save term metadata for viz
    output["terms_core"] = [
        {"en": t["en"], "zh": t["zh_canonical"], "domain": t["domain"]}
        for t in terms_core
    ]

    out_path = RESULTS_DIR / "lens4_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done in {total:.0f}s  →  {out_path}")

    if not args.no_viz:
        from lens_4_values.viz import run_viz
        run_viz(RESULTS_DIR, output)


if __name__ == "__main__":
    main()
