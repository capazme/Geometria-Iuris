#!/usr/bin/env python3
"""
Step 5 — Lens IV (§3.2) for run #4 post-BLP.

Runs both encoding variants (bare + attested) in a single invocation since
axes construction is cheap. Produces:

  experiment_2_axes/results_bare/experiment_2_results.json
  experiment_2_axes/results_bare/scores/{label}_{axis}.npy
  experiment_2_axes/results_attested/experiment_2_results.json
  experiment_2_axes/results_attested/scores/{label}_{axis}.npy

Sub-sections (re-indexed §3.2 vs legacy §3.3):
  §3.2.1  Axis construction quality (sanity: positive_correct, negative_correct)
  §3.2.2  Orthogonality 6×6 cosine matrix per model
  §3.2.3  Per-axis cross-tradition alignment (Spearman ρ + bootstrap CI)
          45 model pairs × 6 axes = 270 entries.
  §3.2.4  Ranking: most-divergent axis cross-tradition (mean ρ̄ ascending)
  §3.2.5  Term-level divergence: per-term Δ score between W̄ and S̄ scores
          (top-N divergent terms surfaced per axis)

Pair definitions (5 EN-side + 5 ZH-side):
  EN-side: 3 WEIRD mono + 2 bilingual-EN
  ZH-side: 3 Sinic mono + 2 bilingual-ZH

  cross           : 5 × 5  = 25
  within_weird    : C(5,2) = 10
  within_sinic    : C(5,2) = 10
  Total           :         45

Usage
-----
    python3 05_axes_experiment.py --variant both    # bare + attested (default)
    python3 05_axes_experiment.py --variant bare    # only bare
    python3 05_axes_experiment.py --variant attested
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

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import EmbeddingClient  # noqa: E402
from _lib import bootstrap_ci_generic  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_index(emb_dir: Path) -> list[dict]:
    with (emb_dir / "index.json").open() as fh:
        return json.load(fh)


def load_vecs(emb_dir: Path, label: str, variant: str) -> np.ndarray:
    fname = "vecs_bare.npy" if variant == "bare" else "vecs_attested.npy"
    return np.load(emb_dir / label / fname).astype(np.float32)


def load_axes(axes_path: Path) -> dict[str, dict]:
    with axes_path.open() as fh:
        return yaml.safe_load(fh)


def build_en_lookup(index: list[dict]) -> dict[str, int]:
    return {t["en"].lower(): i for i, t in enumerate(index)}


def build_zh_lookup(index: list[dict]) -> dict[str, int]:
    return {t["zh"]: i for i, t in enumerate(index) if t.get("zh")}


def to_traditional(text: str) -> str:
    try:
        import opencc
        return opencc.OpenCC("s2t").convert(text)
    except ImportError:
        return text


def resolve_pair_vectors(
    pairs: list[list[str]],
    lang: str,
    vecs_pool: np.ndarray,
    en_lookup: dict[str, int],
    zh_lookup: dict[str, int],
    client: EmbeddingClient,
    model_id: str,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """For each [pos, neg] antonym pair, resolve to a vec via pool lookup
    first, on-the-fly encoding fallback otherwise."""
    out: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for pos_term, neg_term in pairs:
        vecs_pair = []
        for term in (pos_term, neg_term):
            idx = None
            if lang == "en":
                idx = en_lookup.get(term.lower())
            else:
                trad = to_traditional(term)
                idx = zh_lookup.get(trad)
                if idx is None:
                    idx = zh_lookup.get(term)
            if idx is not None:
                vecs_pair.append(vecs_pool[idx])
            else:
                v = client.embed([term], model_id, use_cache=True)[0].astype(np.float32)
                vecs_pair.append(v)
        out.append((vecs_pair[0], vecs_pair[1], pos_term, neg_term))
    return out


def build_axis(pair_vectors: list[tuple[np.ndarray, np.ndarray, str, str]]) -> np.ndarray:
    diffs = np.array([pos - neg for pos, neg, _, _ in pair_vectors])
    mean_diff = diffs.mean(axis=0)
    nrm = np.linalg.norm(mean_diff)
    if nrm > 1e-12:
        mean_diff = mean_diff / nrm
    return mean_diff.astype(np.float32)


def project_terms(vecs: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (vecs @ axis).astype(np.float64)


def axis_sanity(pair_vectors, axis_vec) -> tuple[int, int]:
    pos_correct = sum(1 for pos, _, _, _ in pair_vectors if float(pos @ axis_vec) > 0)
    neg_correct = sum(1 for _, neg, _, _ in pair_vectors if float(neg @ axis_vec) < 0)
    return pos_correct, neg_correct


def spearman_ci(sa: np.ndarray, sb: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, np.ndarray]:
    stacked = np.column_stack([sa, sb])
    rho_obs = float(spearmanr(sa, sb).statistic)

    def stat_fn(data: np.ndarray) -> float:
        return float(spearmanr(data[:, 0], data[:, 1]).statistic)

    ci = bootstrap_ci_generic(stacked, stat_fn, n_boot=n_boot, seed=seed)
    return rho_obs, float(ci.ci_low), float(ci.ci_high), ci.distribution


def run_variant(
    cfg: dict,
    variant: str,
    emb_dir: Path,
    axes_def: dict,
    en_pairs_labels: list[str],
    zh_pairs_labels: list[str],
    client: EmbeddingClient,
    label_to_model_id: dict[str, str],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)
    (out_dir / "axes").mkdir(exist_ok=True)
    (out_dir / "distributions").mkdir(exist_ok=True)

    n_boot = int(cfg["n_boot"])
    seed = int(cfg["seed"])

    index = load_index(emb_dir)
    en_lookup = build_en_lookup(index)
    zh_lookup = build_zh_lookup(index)
    domains = [t["domain"] for t in index]
    n_terms = len(index)

    axis_names = list(axes_def.keys())

    all_labels = en_pairs_labels + zh_pairs_labels
    lang_of: dict[str, str] = {l: "en" for l in en_pairs_labels}
    lang_of.update({l: "zh" for l in zh_pairs_labels})

    print(f"\n[experiment_2_axes/{variant}] Building axes for {len(all_labels)} models...")
    vecs_all: dict[str, np.ndarray] = {}
    axes_all: dict[str, dict[str, np.ndarray]] = {}
    pair_info: dict[str, dict[str, dict]] = {}
    for label in all_labels:
        lang = lang_of[label]
        model_id = label_to_model_id[label]
        vecs = load_vecs(emb_dir, label, variant)
        vecs_all[label] = vecs
        axes_all[label] = {}
        pair_info[label] = {}
        pair_key = "en_pairs" if lang == "en" else "zh_pairs"
        for ax_name in axis_names:
            pairs = axes_def[ax_name][pair_key]
            pv = resolve_pair_vectors(pairs, lang, vecs, en_lookup, zh_lookup,
                                      client, model_id)
            axis_vec = build_axis(pv)
            axes_all[label][ax_name] = axis_vec
            pos_ok, neg_ok = axis_sanity(pv, axis_vec)
            pair_info[label][ax_name] = {
                "n_pairs_used": len(pv),
                "n_pairs_total": len(pairs),
                "positive_correct": int(pos_ok),
                "negative_correct": int(neg_ok),
            }
            np.save(out_dir / "axes" / f"{label}_{ax_name}.npy", axis_vec)
        print(f"  {label}: {len(axis_names)} axes built")

    # Project all terms onto axes
    scores_all: dict[str, dict[str, np.ndarray]] = {}
    for label in all_labels:
        scores_all[label] = {}
        for ax_name in axis_names:
            s = project_terms(vecs_all[label], axes_all[label][ax_name])
            scores_all[label][ax_name] = s
            np.save(out_dir / "scores" / f"{label}_{ax_name}.npy", s)

    # ----- §3.2.1 Axis sanity -----
    section_321: dict[str, dict] = {}
    for label in all_labels:
        ax_report = {ax: pair_info[label][ax] for ax in axis_names}
        section_321[label] = {"axes": ax_report}

    # ----- §3.2.2 Orthogonality 6×6 -----
    section_322: dict[str, dict] = {}
    for label in all_labels:
        cos = np.zeros((len(axis_names), len(axis_names)), dtype=np.float32)
        for i, a in enumerate(axis_names):
            for j, b in enumerate(axis_names):
                cos[i, j] = float(axes_all[label][a] @ axes_all[label][b])
        section_322[label] = {
            "axes": axis_names,
            "cosine_matrix": cos.tolist(),
        }

    # ----- §3.2.3 Per-axis alignment (45 pairs × 6 axes) -----
    cross_pairs = list(product(en_pairs_labels, zh_pairs_labels))
    within_en_pairs = list(combinations(en_pairs_labels, 2))
    within_zh_pairs = list(combinations(zh_pairs_labels, 2))
    bucketed: list[tuple[str, str, str]] = (
        [(a, b, "cross") for a, b in cross_pairs]
        + [(a, b, "within_weird") for a, b in within_en_pairs]
        + [(a, b, "within_sinic") for a, b in within_zh_pairs]
    )
    print(f"\n[experiment_2_axes/{variant}] §3.2.3: {len(bucketed)} pairs × {len(axis_names)} axes "
          f"= {len(bucketed)*len(axis_names)} entries (B={n_boot})")
    per_pair: list[dict] = []
    for la, lb, group in bucketed:
        for ax_name in axis_names:
            t0 = time.perf_counter()
            sa = scores_all[la][ax_name]
            sb = scores_all[lb][ax_name]
            rho, ci_lo, ci_hi, dist = spearman_ci(sa, sb, n_boot=n_boot, seed=seed)
            per_pair.append({
                "model_a": la,
                "model_b": lb,
                "group": group,
                "axis": ax_name,
                "rho": round(rho, 4),
                "ci_low": round(ci_lo, 4),
                "ci_high": round(ci_hi, 4),
            })
            if group == "cross":
                np.savez_compressed(
                    out_dir / "distributions" / f"{la}_x_{lb}_{ax_name}.npz",
                    bootstrap=dist,
                )

    # ----- §3.2.4 ranking most divergent axis -----
    cross_means: dict[str, float] = {}
    for ax_name in axis_names:
        rhos = [p["rho"] for p in per_pair if p["group"] == "cross" and p["axis"] == ax_name]
        cross_means[ax_name] = float(np.mean(rhos))
    ranking = sorted(cross_means.items(), key=lambda x: x[1])
    section_324 = {
        "cross_rho_mean_per_axis": {k: round(v, 4) for k, v in cross_means.items()},
        "ranking_most_divergent_first": [
            {"axis": ax, "mean_cross_rho": round(rho, 4)} for ax, rho in ranking
        ],
    }
    print("  ranking (most divergent first):",
          ", ".join(f"{a} {r:.3f}" for a, r in ranking))

    # ----- §3.2.5 term-level divergence -----
    # For each axis, mean WEIRD score per term vs mean Sinic score per term,
    # then surface top-K most-divergent terms.
    section_325: dict[str, dict] = {}
    K_top = 20
    for ax_name in axis_names:
        w_mean = np.mean([scores_all[l][ax_name] for l in en_pairs_labels], axis=0)
        s_mean = np.mean([scores_all[l][ax_name] for l in zh_pairs_labels], axis=0)
        delta = w_mean - s_mean
        top_idx = np.argsort(-np.abs(delta))[:K_top]
        top = [
            {
                "en": index[i]["en"],
                "zh": index[i]["zh"],
                "domain": index[i]["domain"],
                "w_score": round(float(w_mean[i]), 4),
                "s_score": round(float(s_mean[i]), 4),
                "delta": round(float(delta[i]), 4),
            }
            for i in top_idx
        ]
        section_325[ax_name] = {
            "delta_mean_abs": round(float(np.mean(np.abs(delta))), 4),
            "delta_max_abs": round(float(np.max(np.abs(delta))), 4),
            "top_K_divergent": top,
        }

    return {
        "meta": {
            "variant": variant,
            "n_terms": n_terms,
            "n_boot": n_boot,
            "seed": seed,
            "axes": axis_names,
            "labels_en_pairs": en_pairs_labels,
            "labels_zh_pairs": zh_pairs_labels,
        },
        "section_321": section_321,
        "section_322": section_322,
        "section_323": {"per_pair": per_pair},
        "section_324": section_324,
        "section_325": section_325,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--variant", choices=["bare", "attested", "both"], default="both")
    parser.add_argument("--n-boot", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.n_boot is not None:
        cfg["n_boot"] = args.n_boot
    emb_dir = REPO_ROOT / cfg["paths"]["embeddings"]
    inputs_dir = REPO_ROOT / cfg["paths"]["inputs"]

    axes_def = load_axes(inputs_dir / "value_axes_snapshot.yaml")
    print(f"Loaded {len(axes_def)} axes: {list(axes_def.keys())}")

    en_pairs_labels = [m["label"] for m in cfg["models_weird"]] + \
                      [f"{m['label']}-EN" for m in cfg["models_bilingual"]]
    zh_pairs_labels = [m["label"] for m in cfg["models_sinic"]] + \
                      [f"{m['label']}-ZH" for m in cfg["models_bilingual"]]
    print(f"EN-side labels (use en_pairs): {en_pairs_labels}")
    print(f"ZH-side labels (use zh_pairs): {zh_pairs_labels}")

    # Map labels to model IDs for on-the-fly fallback
    label_to_model_id: dict[str, str] = {}
    for m in cfg["models_weird"] + cfg["models_sinic"]:
        label_to_model_id[m["label"]] = m["id"]
    for m in cfg["models_bilingual"]:
        label_to_model_id[f"{m['label']}-EN"] = m["id"]
        label_to_model_id[f"{m['label']}-ZH"] = m["id"]

    client = EmbeddingClient(
        config_path=REPO_ROOT / "experiments" / "models" / "config.yaml",
        device=cfg.get("device", "cpu"),
    )

    variants = ["bare", "attested"] if args.variant == "both" else [args.variant]
    overall_t0 = time.perf_counter()
    for variant in variants:
        out_dir = REPO_ROOT / cfg["paths"][f"lens4_{variant}"]
        print("\n" + "=" * 60)
        print(f"Lens IV — run #4 post-BLP — variant={variant}")
        print("=" * 60)
        t0 = time.perf_counter()
        result = run_variant(
            cfg=cfg, variant=variant, emb_dir=emb_dir, axes_def=axes_def,
            en_pairs_labels=en_pairs_labels, zh_pairs_labels=zh_pairs_labels,
            client=client, label_to_model_id=label_to_model_id,
            out_dir=out_dir,
        )
        result["meta"]["date"] = datetime.now().isoformat(timespec="seconds")
        result["meta"]["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
        with (out_dir / "experiment_2_results.json").open("w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nDone {variant} in {result['meta']['elapsed_seconds']}s "
              f"-> {(out_dir / 'experiment_2_results.json').relative_to(REPO_ROOT)}")
    print(f"\nTotal axes_experiment: {time.perf_counter() - overall_t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
