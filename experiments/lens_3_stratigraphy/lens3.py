"""
Lens III — Layer Stratigraphy.

Implements the full Lens III analysis pipeline (Ch.3 §3.1.3):

  §3.1.3a  Single-term behavior: drift + Jaccard across layers
  §3.1.3b  Global structure: domain signal emergence + RSA convergence
  §3.1.3c  NTA: Neighborhood Trajectory Analysis (qualitative layer-by-layer k-NN)

Usage
-----
    cd experiments/
    python -m lens_3_stratigraphy.lens3                        # full run
    python -m lens_3_stratigraphy.lens3 --section 3.1.3a       # drift + Jaccard only
    python -m lens_3_stratigraphy.lens3 --section 3.1.3b       # domain signal + RSA only
    python -m lens_3_stratigraphy.lens3 --section nta           # NTA only
    python -m lens_3_stratigraphy.lens3 --force                # bypass cache
    python -m lens_3_stratigraphy.lens3 --no-viz               # skip figures

Outputs
-------
    lens_3_stratigraphy/results/lens3_results.json
    lens_3_stratigraphy/results/layer_vectors/{model}.npz
    lens_3_stratigraphy/results/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.embeddings import load_precomputed
from shared.statistical import compute_rdm, mannwhitney_with_r, upper_tri
from lens_3_stratigraphy.layer_extraction import extract_per_layer, verify_final_layer

RESULTS_DIR = Path(__file__).parent / "results"
EMB_DIR = ROOT / "data" / "processed" / "embeddings"
CONFIG_PATH = ROOT / "models" / "config.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> tuple[list[str], list[str]]:
    """Return (weird_labels, sinic_labels) from config.yaml."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    weird = [m["label"] for m in raw.get("weird", [])]
    sinic = [m["label"] for m in raw.get("sinic", [])]
    return weird, sinic


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_core_terms() -> tuple[list[dict], list[int]]:
    """Load term index and return (core_terms, core_indices)."""
    _, index = load_precomputed("BGE-EN-large", EMB_DIR)
    core_idx = [i for i, t in enumerate(index) if t["tier"] == "core" and t["domain"]]
    terms_core = [index[i] for i in core_idx]
    return terms_core, core_idx


def _get_model_lang(label: str) -> str:
    """Return 'en' or 'zh' for a given model label."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for group in ("weird", "sinic"):
        for entry in raw.get(group, []):
            if entry["label"] == label:
                return entry["lang"]
    raise ValueError(f"Model '{label}' not found in {CONFIG_PATH}")


def _terms_for_model(terms_core: list[dict], label: str) -> list[str]:
    """Return the correct text list (EN or ZH) based on model language."""
    lang = _get_model_lang(label)
    if lang == "zh":
        return [t["zh_canonical"] for t in terms_core]
    return [t["en"] for t in terms_core]


# ---------------------------------------------------------------------------
# §3.1.3a — Single-term: intra-domain vs inter-domain distance split
# ---------------------------------------------------------------------------

def _intra_inter_split(
    rdm: np.ndarray,
    domains: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Split upper-triangle distances into intra-domain and inter-domain sets."""
    n = len(rdm)
    rows, cols = np.triu_indices(n, k=1)
    dom_arr = np.array(domains)
    same = dom_arr[rows] == dom_arr[cols]
    tri = rdm[rows, cols]
    return tri[same], tri[~same]


# ---------------------------------------------------------------------------
# §3.1.3a — Single-term analysis: drift + Jaccard
# ---------------------------------------------------------------------------

def _compute_drift(layer_vecs: np.ndarray) -> np.ndarray:
    """
    Compute per-term cosine drift between consecutive layers.

    Parameters
    ----------
    layer_vecs : (N, L+1, dim)

    Returns
    -------
    (N, L) drift matrix — drift[t, l] = cosine_distance(vec[t, l], vec[t, l+1])
    """
    n_terms, n_states, _ = layer_vecs.shape
    n_transitions = n_states - 1
    drift = np.zeros((n_terms, n_transitions), dtype=np.float32)
    for t in range(n_terms):
        for l in range(n_transitions):
            drift[t, l] = cosine_distance(layer_vecs[t, l], layer_vecs[t, l + 1])
    return drift


def _compute_jaccard(layer_vecs: np.ndarray, k: int = 7) -> np.ndarray:
    """
    Compute per-term k-NN Jaccard distance between consecutive layers.

    jaccard[t, l] = 1 - |kNN(t,l) ∩ kNN(t,l+1)| / |kNN(t,l) ∪ kNN(t,l+1)|

    Parameters
    ----------
    layer_vecs : (N, L+1, dim) L2-normalized
    k : int — number of neighbors

    Returns
    -------
    (N, L) Jaccard distance matrix
    """
    n_terms, n_states, _ = layer_vecs.shape
    n_transitions = n_states - 1
    jaccard = np.zeros((n_terms, n_transitions), dtype=np.float32)

    for l in range(n_transitions):
        # Similarity matrices at layer l and l+1
        sim_l = layer_vecs[:, l, :] @ layer_vecs[:, l, :].T
        sim_l1 = layer_vecs[:, l + 1, :] @ layer_vecs[:, l + 1, :].T

        for t in range(n_terms):
            # Exclude self (set self-sim to -inf)
            s_l = sim_l[t].copy()
            s_l[t] = -np.inf
            s_l1 = sim_l1[t].copy()
            s_l1[t] = -np.inf

            nn_l = set(np.argsort(s_l)[-k:])
            nn_l1 = set(np.argsort(s_l1)[-k:])

            intersection = len(nn_l & nn_l1)
            union = len(nn_l | nn_l1)
            jaccard[t, l] = 1.0 - intersection / union if union > 0 else 0.0

    return jaccard


def _aggregate_by_domain(
    matrix: np.ndarray,
    domains: list[str],
) -> dict[str, list[float]]:
    """
    Average matrix rows by domain.

    Parameters
    ----------
    matrix : (N, L) — rows are terms, columns are layers/transitions
    domains : list[str] — domain label per term

    Returns
    -------
    {domain: [mean_per_transition...]}
    """
    dom_arr = np.array(domains)
    result = {}
    for d in sorted(set(domains)):
        mask = dom_arr == d
        result[d] = matrix[mask].mean(axis=0).tolist()
    return result


def _top_drift_terms(
    drift: np.ndarray,
    terms: list[dict],
    n: int = 10,
) -> list[dict]:
    """Return the top-n terms by total drift (sum across all transitions)."""
    total = drift.sum(axis=1)
    top_idx = np.argsort(total)[-n:][::-1]
    return [
        {
            "en": terms[i]["en"],
            "domain": terms[i]["domain"],
            "total_drift": round(float(total[i]), 4),
            "drift_curve": [round(float(d), 4) for d in drift[i]],
        }
        for i in top_idx
    ]


def run_section_313a(
    all_labels: list[str],
    terms_core: list[dict],
    core_idx: list[int],
    k: int = 7,
    device: str = "mps",
    force: bool = False,
) -> dict:
    """
    §3.1.3a — Single-term behavior: drift + Jaccard across layers.
    """
    print("\n[§3.1.3a] Single-term analysis (drift + Jaccard)")
    domains = [t["domain"] for t in terms_core]

    per_model: dict[str, dict] = {}

    for label in all_labels:
        print(f"\n  {label}")
        t0 = time.perf_counter()
        texts = _terms_for_model(terms_core, label)

        layer_vecs = extract_per_layer(label, texts, device=device, force=force)
        n_terms, n_states, dim = layer_vecs.shape
        print(f"    shape: ({n_terms}, {n_states}, {dim})")

        verify_final_layer(label, layer_vecs, core_idx)

        print("    Computing drift...")
        drift = _compute_drift(layer_vecs)
        print(f"    Drift range: [{drift.min():.4f}, {drift.max():.4f}]")

        print(f"    Computing Jaccard (k={k})...")
        jaccard = _compute_jaccard(layer_vecs, k=k)
        print(f"    Jaccard range: [{jaccard.min():.4f}, {jaccard.max():.4f}]")

        drift_by_domain = _aggregate_by_domain(drift, domains)
        jaccard_by_domain = _aggregate_by_domain(jaccard, domains)
        top_terms = _top_drift_terms(drift, terms_core)

        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s")

        per_model[label] = {
            "n_layers": n_states - 1,
            "n_states": n_states,
            "drift_mean": round(float(drift.mean()), 4),
            "drift_max": round(float(drift.max()), 4),
            "jaccard_mean": round(float(jaccard.mean()), 4),
            "drift_by_domain": drift_by_domain,
            "jaccard_by_domain": jaccard_by_domain,
            "top_drift_terms": top_terms,
        }

    return {"per_model": per_model, "k": k}


# ---------------------------------------------------------------------------
# §3.1.3b — Structural analysis: domain signal + RSA convergence
# ---------------------------------------------------------------------------

def run_section_313b(
    all_labels: list[str],
    terms_core: list[dict],
    core_idx: list[int],
    device: str = "mps",
    force: bool = False,
) -> dict:
    """
    §3.1.3b — Global structure: domain signal emergence + RSA convergence.
    """
    print("\n[§3.1.3b] Structural analysis (domain signal + RSA convergence)")
    domains = [t["domain"] for t in terms_core]

    per_model: dict[str, dict] = {}

    for label in all_labels:
        print(f"\n  {label}")
        t0 = time.perf_counter()
        texts = _terms_for_model(terms_core, label)

        layer_vecs = extract_per_layer(label, texts, device=device, force=force)
        n_terms, n_states, dim = layer_vecs.shape

        # RDM at final layer (reference for RSA convergence)
        rdm_final = compute_rdm(layer_vecs[:, -1, :])
        tri_final = upper_tri(rdm_final)

        domain_signal_r: list[float] = []
        rsa_vs_final: list[float] = []

        for l in range(n_states):
            rdm_l = compute_rdm(layer_vecs[:, l, :])

            # Domain signal: Mann-Whitney intra vs inter
            intra, inter = _intra_inter_split(rdm_l, domains)
            mw = mannwhitney_with_r(intra, inter, alternative="less")
            domain_signal_r.append(round(float(mw.effect_r), 4))

            # RSA convergence: Spearman between layer l and final
            tri_l = upper_tri(rdm_l)
            if l == n_states - 1:
                rho = 1.0  # identity
            else:
                rho = float(spearmanr(tri_l, tri_final).statistic)
            rsa_vs_final.append(round(rho, 4))

            if l % 6 == 0 or l == n_states - 1:
                print(f"    layer {l:2d}:  r={domain_signal_r[-1]:+.3f}  "
                      f"ρ(vs final)={rsa_vs_final[-1]:.3f}")

        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s")

        per_model[label] = {
            "n_layers": n_states - 1,
            "n_states": n_states,
            "domain_signal_r": domain_signal_r,
            "rsa_vs_final_rho": rsa_vs_final,
            "domain_signal_r_final": domain_signal_r[-1],
            "rsa_convergence_layer_50pct": _find_threshold(rsa_vs_final, 0.5),
            "rsa_convergence_layer_90pct": _find_threshold(rsa_vs_final, 0.9),
        }

    return {"per_model": per_model}


def _find_threshold(values: list[float], threshold: float) -> int | None:
    """Find the first layer index where the value exceeds the threshold."""
    for i, v in enumerate(values):
        if v >= threshold:
            return i
    return None


# ---------------------------------------------------------------------------
# §3.1.3c — Neighborhood Trajectory Analysis (NTA) — trace D5
# ---------------------------------------------------------------------------

NTA_TARGET_TERMS = [
    "negligence",       # civil — but general English meaning is moral/everyday
    "sovereignty",      # constitutional — but also international / political
    "corruption",       # criminal — but also governance / moral
    "comity",           # international — but also everyday courtesy
    "adoption",         # civil (legal) — but also everyday meaning
    "strike",           # labor — but also military / everyday
    "disclosure",       # procedure — but also financial / everyday
    "franchise",        # constitutional — but also commercial
]


def _load_pool_terms() -> tuple[list[dict], list[int]]:
    """Load core + control terms as the NTA neighbor pool."""
    _, index = load_precomputed("BGE-EN-large", EMB_DIR)
    pool_idx = [
        i for i, t in enumerate(index)
        if (t["tier"] == "core" and t["domain"]) or t["tier"] == "control"
    ]
    pool_terms = [index[i] for i in pool_idx]
    return pool_terms, pool_idx


def run_nta(
    model_label: str,
    terms_core: list[dict],
    core_idx: list[int],
    target_terms: list[str] | None = None,
    k: int = 7,
    device: str = "cpu",
    sample_layers: list[int] | None = None,
    force: bool = False,
) -> dict:
    """
    §3.1.3c — Neighborhood Trajectory Analysis (NTA).

    For each target term, traces the k-NN neighborhood across sampled layers.
    The neighbor pool includes core legal terms (397) and control terms (100
    Swadesh-like non-legal words), allowing detection of the legal/non-legal
    boundary shift as representational depth increases.

    Parameters
    ----------
    model_label : str
    terms_core : list[dict] — core terms with en, zh_canonical, domain
    core_idx : list[int] — indices into the full precomputed vectors
    target_terms : list[str] — EN names to analyze (exact match)
    k : int — number of neighbors
    device : str — PyTorch device
    sample_layers : list[int] | None — which layers to show (None = auto)
    force : bool — bypass cache

    Returns
    -------
    dict with per-term trajectory data
    """
    if target_terms is None:
        target_terms = NTA_TARGET_TERMS

    # Build pool = core + control
    pool_terms, pool_idx = _load_pool_terms()
    n_pool = len(pool_terms)
    n_core = len(terms_core)

    print(f"\n[NTA] {model_label}, k={k}, {len(target_terms)} terms, "
          f"pool={n_pool} (core={n_core} + control={n_pool - n_core})")

    # Extract layer vectors for entire pool
    pool_texts = _terms_for_model(pool_terms, model_label)
    cache_label = f"{model_label}_pool"
    layer_vecs = extract_per_layer(
        model_label, pool_texts, device=device,
        cache_label=cache_label, force=force,
    )
    n_terms_actual, n_states, dim = layer_vecs.shape

    # Build metadata arrays for the pool
    en_names = [t["en"] for t in pool_terms]
    domains = [t.get("domain") or "control" for t in pool_terms]
    tiers = [t["tier"] for t in pool_terms]

    # Map target term names to pool indices
    name_to_idx = {t["en"].lower(): i for i, t in enumerate(pool_terms)}
    targets = []
    for name in target_terms:
        idx = name_to_idx.get(name.lower())
        if idx is not None:
            targets.append((name, idx))
        else:
            print(f"  [skip] '{name}' not found in pool")

    if sample_layers is None:
        n_samples = min(n_states, 7)
        sample_layers = sorted(set(
            [int(round(i * (n_states - 1) / (n_samples - 1)))
             for i in range(n_samples)]
        ))

    result: dict = {
        "model": model_label,
        "k": k,
        "sample_layers": sample_layers,
        "pool_size": n_pool,
        "pool_core": n_core,
        "pool_control": n_pool - n_core,
        "terms": {},
    }

    for term_name, t_idx in targets:
        print(f"  {term_name} (idx={t_idx}, domain={domains[t_idx]})")
        layers_data = []
        prev_nn_set: set[int] | None = None

        for l in sample_layers:
            vec_t = layer_vecs[t_idx, l, :]
            sims = layer_vecs[:, l, :] @ vec_t
            sims[t_idx] = -np.inf

            top_k_idx = np.argsort(sims)[-k:][::-1]
            nn_set = set(top_k_idx.tolist())

            neighbors = []
            for rank, ni in enumerate(top_k_idx):
                entry = {
                    "rank": rank + 1,
                    "en": en_names[ni],
                    "domain": domains[ni],
                    "tier": tiers[ni],
                    "sim": round(float(sims[ni]), 4),
                }
                if prev_nn_set is not None:
                    if ni not in prev_nn_set:
                        entry["status"] = "entered"
                neighbors.append(entry)

            exited = []
            if prev_nn_set is not None:
                for ni in prev_nn_set:
                    if ni not in nn_set:
                        exited.append({
                            "en": en_names[ni],
                            "domain": domains[ni],
                            "tier": tiers[ni],
                        })

            layer_entry = {
                "layer": l,
                "neighbors": neighbors,
            }
            if exited:
                layer_entry["exited"] = exited

            layers_data.append(layer_entry)
            prev_nn_set = nn_set

        # Domain/tier composition per layer
        domain_evolution = []
        for ld in layers_data:
            dom_counts: dict[str, int] = {}
            n_control = 0
            n_legal = 0
            for nb in ld["neighbors"]:
                d = nb["domain"]
                dom_counts[d] = dom_counts.get(d, 0) + 1
                if nb["tier"] == "control":
                    n_control += 1
                else:
                    n_legal += 1
            domain_evolution.append({
                "layer": ld["layer"],
                "domains": dom_counts,
                "n_legal": n_legal,
                "n_control": n_control,
            })

        result["terms"][term_name] = {
            "idx": t_idx,
            "domain": domains[t_idx],
            "zh": pool_terms[t_idx]["zh_canonical"],
            "layers": layers_data,
            "domain_evolution": domain_evolution,
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Lens III — Layer Stratigraphy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--section",
        choices=["3.1.3a", "3.1.3b", "nta", "all"],
        default="all",
        help="Which section(s) to run",
    )
    parser.add_argument("--k", type=int, default=7,
                        help="k for k-NN Jaccard (§3.1.3a)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip figure generation after pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and re-extract all layer vectors")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (cpu recommended; mps is non-deterministic)")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    weird_labels, sinic_labels = _load_config()
    all_labels = weird_labels + sinic_labels

    terms_core, core_idx = _load_core_terms()

    print("=" * 60)
    print("Lens III — Layer Stratigraphy")
    print(f"  WEIRD : {weird_labels}")
    print(f"  Sinic : {sinic_labels}")
    print(f"  Core terms: {len(terms_core)}")
    print(f"  k={args.k}  device={args.device}")
    print("=" * 60)

    output: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "k": args.k,
            "n_core": len(terms_core),
            "weird_models": weird_labels,
            "sinic_models": sinic_labels,
            "device": args.device,
        }
    }

    t_start = time.perf_counter()

    if args.section in ("3.1.3a", "all"):
        output["section_313a"] = run_section_313a(
            all_labels, terms_core, core_idx,
            k=args.k, device=args.device, force=args.force,
        )

    if args.section in ("3.1.3b", "all"):
        output["section_313b"] = run_section_313b(
            all_labels, terms_core, core_idx,
            device=args.device, force=args.force,
        )

    if args.section in ("nta", "all"):
        nta_results = {}
        for label in all_labels:
            nta_results[label] = run_nta(
                label, terms_core, core_idx,
                k=args.k, device=args.device, force=args.force,
            )
        output["nta"] = nta_results

    total = time.perf_counter() - t_start
    output["meta"]["elapsed_seconds"] = round(total, 1)

    out_path = RESULTS_DIR / "lens3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done in {total:.0f}s  →  {out_path}")

    if not args.no_viz:
        from lens_3_stratigraphy.viz import run_viz
        run_viz(RESULTS_DIR, output)


if __name__ == "__main__":
    main()
