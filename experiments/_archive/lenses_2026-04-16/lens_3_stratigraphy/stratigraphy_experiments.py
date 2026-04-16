"""
Stratigraphy Experiments — 6 additional layer-by-layer analyses.

All use cached layer vectors from lens3.py. No new model inference needed.

Experiments:
  1. Domain Emergence Order — per-domain signal r across layers
  2. Cross-Lingual Layer Alignment — RSA between WEIRD/Sinic layer pairs
  3. k-NN Classification Accuracy — domain prediction accuracy per layer
  4. Anisotropy Profile — mean pairwise cosine similarity per layer
  5. Domain Migration Paths — majority-domain trajectory per term
  6. Phase Transition Detection — change points in domain signal curve

Usage:
    cd experiments/
    python -m lens_3_stratigraphy.stratigraphy_experiments
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.statistical import compute_rdm, mannwhitney_with_r, upper_tri
from shared.embeddings import load_precomputed
from lens_3_stratigraphy.lens3 import _load_config, _load_core_terms

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = RESULTS_DIR / "layer_vectors"
EMB_DIR = ROOT / "data" / "processed" / "embeddings"

CROSS_PAIRS = [
    ("BGE-EN-large", "BGE-ZH-large"),
    ("E5-large", "Text2vec-large-ZH"),
    ("FreeLaw-EN", "Dmeta-ZH"),
]

DOMAIN_COLORS = {
    "administrative": "#4e79a7", "civil": "#f28e2b",
    "constitutional": "#e15759", "criminal": "#76b7b2",
    "environmental_tech": "#ff9da7", "governance": "#9c755f",
    "international": "#59a14f", "jurisprudence": "#bab0ac",
    "labor_social": "#edc948", "procedure": "#b07aa1",
    "rights": "#86bcb6",
}

SHORT = {
    "BGE-EN-large": "BGE-EN", "E5-large": "E5", "FreeLaw-EN": "FreeLaw",
    "BGE-ZH-large": "BGE-ZH", "Text2vec-large-ZH": "Text2vec", "Dmeta-ZH": "Dmeta",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_vecs(model: str, pool: bool = False) -> np.ndarray:
    suffix = "_pool" if pool else ""
    path = CACHE_DIR / f"{model}{suffix}.npz"
    return np.load(path)["layers"]


def _load_pool_tiers() -> list[str]:
    """Return tier labels ('core' or 'control') for each pool term."""
    _, index = load_precomputed("BGE-EN-large", EMB_DIR)
    pool = [
        t for t in index
        if (t["tier"] == "core" and t.get("domain")) or t["tier"] == "control"
    ]
    return [t["tier"] for t in pool]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Precompute RDM upper triangles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def precompute_tris(models: list[str], terms_core: list[dict]) -> dict:
    """Compute RDM upper triangles for all models × all layers."""
    print("\n[Phase 0] Precomputing RDM upper triangles...")
    n = len(terms_core)
    rows, cols = np.triu_indices(n, k=1)
    cache = {}
    for model in models:
        vecs = _load_vecs(model)
        n_t, n_states, dim = vecs.shape
        tris = []
        for l in range(n_states):
            rdm = compute_rdm(vecs[:, l, :])
            tris.append(rdm[rows, cols].astype(np.float32))
        cache[model] = {"n_states": n_states, "tris": tris}
        print(f"  {model}: {n_states} states, {len(tris[0])} pairs")
    return cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 1: Domain Emergence Order
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp_domain_emergence(
    models: list[str],
    terms_core: list[dict],
    tri_cache: dict,
    threshold: float = 0.05,
) -> dict:
    """Per-domain signal r at each layer. Find emergence order."""
    print("\n[Exp 1] Domain Emergence Order")
    domains = [t["domain"] for t in terms_core]
    unique_domains = sorted(set(domains))
    dom_arr = np.array(domains)
    n = len(terms_core)
    rows, cols = np.triu_indices(n, k=1)

    masks = {}
    for d in unique_domains:
        m = dom_arr == d
        masks[d] = {"intra": m[rows] & m[cols], "inter": m[rows] ^ m[cols]}

    results = {}
    for model in models:
        t0 = time.perf_counter()
        n_states = tri_cache[model]["n_states"]
        tris = tri_cache[model]["tris"]

        per_domain = {}
        for d in unique_domains:
            r_curve = []
            for l in range(n_states):
                tri = tris[l]
                intra = tri[masks[d]["intra"]]
                inter = tri[masks[d]["inter"]]
                if len(intra) > 1 and len(inter) > 1:
                    mw = mannwhitney_with_r(intra, inter, alternative="less")
                    r_curve.append(round(float(mw.effect_r), 4))
                else:
                    r_curve.append(0.0)

            emergence = None
            for i, r in enumerate(r_curve):
                if r > threshold:
                    emergence = i
                    break

            per_domain[d] = {
                "r_curve": r_curve,
                "emergence_layer": emergence,
                "max_r": round(float(max(r_curve)), 4),
                "max_r_layer": int(np.argmax(r_curve)),
            }

        elapsed = time.perf_counter() - t0
        print(f"  {model}: {elapsed:.1f}s")
        results[model] = {"n_states": n_states, "per_domain": per_domain}

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 2: Cross-Lingual Layer Alignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp_cross_lingual(tri_cache: dict) -> dict:
    """RSA between every layer pair for WEIRD-Sinic model pairs."""
    print("\n[Exp 2] Cross-Lingual Layer Alignment")
    results = {}
    for m_w, m_s in CROSS_PAIRS:
        t0 = time.perf_counter()
        tris_w = tri_cache[m_w]["tris"]
        tris_s = tri_cache[m_s]["tris"]
        n_w = len(tris_w)
        n_s = len(tris_s)

        matrix = np.zeros((n_w, n_s), dtype=np.float32)
        for lw in range(n_w):
            for ls in range(n_s):
                matrix[lw, ls] = float(spearmanr(tris_w[lw], tris_s[ls]).statistic)

        elapsed = time.perf_counter() - t0
        pair_key = f"{SHORT[m_w]} vs {SHORT[m_s]}"
        print(f"  {pair_key}: {n_w}x{n_s} = {n_w*n_s} correlations in {elapsed:.1f}s")

        results[pair_key] = {
            "weird_model": m_w,
            "sinic_model": m_s,
            "n_weird_layers": n_w,
            "n_sinic_layers": n_s,
            "matrix": matrix.round(4).tolist(),
            "max_rho": round(float(matrix.max()), 4),
            "diagonal_rho": [
                round(float(matrix[i, i]), 4)
                for i in range(min(n_w, n_s))
            ],
        }

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 3: k-NN Classification Accuracy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp_knn_classification(
    models: list[str],
    terms_core: list[dict],
    k: int = 7,
) -> dict:
    """Domain classification via k-NN majority vote at each layer."""
    print("\n[Exp 3] k-NN Classification Accuracy")
    domains = [t["domain"] for t in terms_core]
    unique_domains = sorted(set(domains))
    dom_idx = {d: i for i, d in enumerate(unique_domains)}
    true_labels = np.array([dom_idx[d] for d in domains])
    n = len(terms_core)
    n_dom = len(unique_domains)

    results = {}
    for model in models:
        t0 = time.perf_counter()
        vecs = _load_vecs(model)
        n_t, n_states, dim = vecs.shape

        accuracy_curve = []
        per_domain_acc = []
        confusion_at = {}

        # Select layers for confusion matrices: L0, ~33%, ~66%, final
        cm_layers = sorted(set([0, n_states // 3, 2 * n_states // 3, n_states - 1]))

        for l in range(n_states):
            sim = vecs[:, l, :] @ vecs[:, l, :].T
            np.fill_diagonal(sim, -np.inf)

            # k-NN majority vote
            top_k = np.argpartition(sim, -k, axis=1)[:, -k:]
            predicted = np.zeros(n, dtype=int)
            for t in range(n):
                nn_doms = true_labels[top_k[t]]
                counts = np.bincount(nn_doms, minlength=n_dom)
                predicted[t] = int(np.argmax(counts))

            acc = float((predicted == true_labels).mean())
            accuracy_curve.append(round(acc, 4))

            # Per-domain accuracy
            dom_acc = {}
            for di, d in enumerate(unique_domains):
                mask = true_labels == di
                if mask.sum() > 0:
                    dom_acc[d] = round(float((predicted[mask] == di).mean()), 4)
            per_domain_acc.append(dom_acc)

            # Confusion matrix at selected layers
            if l in cm_layers:
                cm = np.zeros((n_dom, n_dom), dtype=int)
                for t in range(n):
                    cm[true_labels[t], predicted[t]] += 1
                confusion_at[l] = {
                    "matrix": cm.tolist(),
                    "domains": unique_domains,
                }

        elapsed = time.perf_counter() - t0
        print(f"  {model}: final_acc={accuracy_curve[-1]:.3f}, {elapsed:.1f}s")

        results[model] = {
            "n_states": n_states,
            "accuracy_curve": accuracy_curve,
            "per_domain_acc_final": per_domain_acc[-1],
            "confusion_at": confusion_at,
            "k": k,
        }

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 4: Anisotropy Profile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp_anisotropy(
    models: list[str],
    terms_core: list[dict],
) -> dict:
    """Mean pairwise cosine similarity per layer, legal vs control."""
    print("\n[Exp 4] Anisotropy Profile")
    n_core = len(terms_core)

    # Try to load pool tiers for legal-vs-control split
    try:
        tiers = _load_pool_tiers()
        has_pool = True
        core_mask = np.array([t == "core" for t in tiers])
        ctrl_mask = np.array([t == "control" for t in tiers])
        print(f"  Pool: {core_mask.sum()} core + {ctrl_mask.sum()} control")
    except Exception:
        has_pool = False
        print("  Pool cache not available, using core-only")

    results = {}
    for model in models:
        t0 = time.perf_counter()

        if has_pool:
            try:
                vecs = _load_vecs(model, pool=True)
            except FileNotFoundError:
                vecs = _load_vecs(model)
                has_pool_model = False
            else:
                has_pool_model = True
        else:
            vecs = _load_vecs(model)
            has_pool_model = False

        n_t, n_states, dim = vecs.shape

        legal_aniso = []
        ctrl_aniso = []
        overall_aniso = []

        for l in range(n_states):
            v = vecs[:, l, :]
            sim = v @ v.T
            np.fill_diagonal(sim, np.nan)

            if has_pool_model:
                # Legal terms only
                sim_legal = sim[np.ix_(core_mask, core_mask)]
                legal_aniso.append(round(float(np.nanmean(sim_legal)), 4))
                # Control terms only
                sim_ctrl = sim[np.ix_(ctrl_mask, ctrl_mask)]
                ctrl_aniso.append(round(float(np.nanmean(sim_ctrl)), 4))
                overall_aniso.append(round(float(np.nanmean(sim)), 4))
            else:
                overall_aniso.append(round(float(np.nanmean(sim)), 4))
                legal_aniso.append(overall_aniso[-1])

        elapsed = time.perf_counter() - t0
        print(f"  {model}: {elapsed:.1f}s")

        entry = {
            "n_states": n_states,
            "overall": overall_aniso,
            "legal": legal_aniso,
        }
        if ctrl_aniso:
            entry["control"] = ctrl_aniso
        results[model] = entry

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 5: Domain Migration Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp_domain_migration(
    models: list[str],
    terms_core: list[dict],
    k: int = 7,
) -> dict:
    """Track majority domain of k-NN across layers for each term."""
    print("\n[Exp 5] Domain Migration Paths")
    domains = [t["domain"] for t in terms_core]
    unique_domains = sorted(set(domains))
    dom_idx = {d: i for i, d in enumerate(unique_domains)}
    true_labels = np.array([dom_idx[d] for d in domains])
    n = len(terms_core)

    results = {}
    for model in models:
        t0 = time.perf_counter()
        vecs = _load_vecs(model)
        n_t, n_states, dim = vecs.shape

        # Matrix: (n_terms, n_states) — majority domain index
        migration = np.zeros((n, n_states), dtype=int)
        for l in range(n_states):
            sim = vecs[:, l, :] @ vecs[:, l, :].T
            np.fill_diagonal(sim, -np.inf)
            top_k = np.argpartition(sim, -k, axis=1)[:, -k:]
            for t in range(n):
                nn_doms = true_labels[top_k[t]]
                counts = np.bincount(nn_doms, minlength=len(unique_domains))
                migration[t, l] = int(np.argmax(counts))

        # Count transitions per layer
        transition_counts = []
        for l in range(n_states - 1):
            changes = int((migration[:, l] != migration[:, l + 1]).sum())
            transition_counts.append(changes)

        # Find terms with most domain changes
        n_changes = np.array([
            len(set(migration[t].tolist())) - 1 + sum(
                1 for l in range(n_states - 1)
                if migration[t, l] != migration[t, l + 1]
            )
            for t in range(n)
        ])
        top_changers_idx = np.argsort(n_changes)[-20:][::-1]
        top_changers = []
        for idx in top_changers_idx:
            path = [unique_domains[migration[idx, l]] for l in range(n_states)]
            top_changers.append({
                "term": terms_core[idx]["en"],
                "domain": domains[idx],
                "path": path,
                "n_transitions": int(sum(
                    1 for l in range(n_states - 1)
                    if migration[idx, l] != migration[idx, l + 1]
                )),
            })

        elapsed = time.perf_counter() - t0
        print(f"  {model}: {elapsed:.1f}s")

        results[model] = {
            "n_states": n_states,
            "domains": unique_domains,
            "migration": migration.tolist(),
            "term_names": [t["en"] for t in terms_core],
            "true_domains": domains,
            "transition_counts": transition_counts,
            "top_changers": top_changers,
        }

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 6: Phase Transition Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp_phase_transitions(
    models: list[str],
    terms_core: list[dict],
    tri_cache: dict,
) -> dict:
    """Detect change points in domain signal r curve."""
    print("\n[Exp 6] Phase Transition Detection")
    domains = [t["domain"] for t in terms_core]
    n = len(terms_core)
    rows, cols = np.triu_indices(n, k=1)
    dom_arr = np.array(domains)
    same = dom_arr[rows] == dom_arr[cols]

    results = {}
    for model in models:
        tris = tri_cache[model]["tris"]
        n_states = tri_cache[model]["n_states"]

        # Global domain signal r at each layer
        r_curve = []
        for l in range(n_states):
            tri = tris[l]
            intra = tri[same]
            inter = tri[~same]
            mw = mannwhitney_with_r(intra, inter, alternative="less")
            r_curve.append(round(float(mw.effect_r), 4))

        # First derivative: Δr
        delta_r = [
            round(r_curve[l + 1] - r_curve[l], 4)
            for l in range(n_states - 1)
        ]

        # Second derivative: ΔΔr
        delta2_r = [
            round(delta_r[l + 1] - delta_r[l], 4)
            for l in range(len(delta_r) - 1)
        ]

        # Change point: layer with maximum absolute Δr
        if delta_r:
            abs_delta = [abs(d) for d in delta_r]
            cp_layer = int(np.argmax(abs_delta))
            cp_magnitude = delta_r[cp_layer]
        else:
            cp_layer = None
            cp_magnitude = 0.0

        # Acceleration peak: layer with maximum absolute ΔΔr
        if delta2_r:
            abs_delta2 = [abs(d) for d in delta2_r]
            accel_layer = int(np.argmax(abs_delta2))
            accel_magnitude = delta2_r[accel_layer]
        else:
            accel_layer = None
            accel_magnitude = 0.0

        # Saturation layer: first layer where r reaches 90% of max
        max_r = max(r_curve) if r_curve else 0
        sat_layer = None
        if max_r > 0:
            for i, r in enumerate(r_curve):
                if r >= 0.9 * max_r:
                    sat_layer = i
                    break

        print(f"  {model}: change_point=L{cp_layer} (Δr={cp_magnitude:+.4f}), "
              f"saturation=L{sat_layer}")

        results[model] = {
            "n_states": n_states,
            "r_curve": r_curve,
            "delta_r": delta_r,
            "delta2_r": delta2_r,
            "change_point_layer": cp_layer,
            "change_point_delta": cp_magnitude,
            "accel_layer": accel_layer,
            "accel_magnitude": accel_magnitude,
            "saturation_layer": sat_layer,
            "max_r": round(max_r, 4),
        }

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_html(all_results: dict) -> str:
    """Build self-contained interactive HTML for all 6 experiments."""
    # Slim down migration data for JSON (full matrix is large)
    slim = json.loads(json.dumps(all_results))
    for model, data in slim.get("exp5_migration", {}).items():
        # Keep only first 100 terms + top changers in the migration matrix
        if "migration" in data:
            data["migration_truncated"] = data["migration"][:100]
            data["term_names_truncated"] = data["term_names"][:100]
            data["true_domains_truncated"] = data["true_domains"][:100]
            del data["migration"]
            del data["term_names"]
            del data["true_domains"]

    data_js = json.dumps(slim, ensure_ascii=False)
    return _html_template(data_js)


def _html_template(data_js: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stratigraphy Experiments — Deep Layer Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
  :root {{
    --bg: #f8f9fa; --fg: #1a1a2e; --card: #fff; --border: #e2e8f0;
    --accent: #0072B2; --accent-light: #e3f2fd; --muted: #64748b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--fg);
    padding: 24px 28px; line-height: 1.6; max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 2px; font-weight: 700; }}
  .subtitle {{ color: var(--muted); font-size: 0.88rem; margin-bottom: 20px; }}
  .tabs {{
    display: flex; gap: 3px; margin-bottom: 20px; flex-wrap: wrap;
    border-bottom: 2px solid var(--border);
  }}
  .tab-btn {{
    padding: 9px 16px; border: none; border-bottom: 3px solid transparent;
    background: none; cursor: pointer; font-size: 0.82rem; color: var(--muted);
    font-weight: 500; transition: all 0.15s; border-radius: 6px 6px 0 0;
  }}
  .tab-btn:hover {{ color: var(--accent); background: var(--accent-light); }}
  .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; background: var(--accent-light); }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .note {{
    background: #f0f7ff; border-left: 4px solid var(--accent);
    padding: 14px 18px; margin-bottom: 18px; border-radius: 0 6px 6px 0;
    font-size: 0.84rem; line-height: 1.7;
  }}
  .note h3 {{ margin: 0 0 6px 0; font-size: 0.92rem; color: #005a8c; }}
  .note p {{ margin: 5px 0; }}
  .note ul {{ margin: 6px 0 6px 20px; }}
  .note li {{ margin: 3px 0; }}
  .note code {{ background: #e2e8f0; padding: 1px 5px; border-radius: 3px; font-size: 0.82em; }}
  .note .formula {{ background: #fff; border: 1px solid var(--border); padding: 8px 14px; border-radius: 4px; margin: 8px 0; }}
  .card {{
    background: var(--card); border-radius: 10px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 18px;
    border: 1px solid var(--border);
  }}
  .card h3 {{ font-size: 1.0rem; margin-bottom: 10px; font-weight: 600; }}
  .card-sub {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 14px; }}
  .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .split3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  @media (max-width: 900px) {{ .split, .split3 {{ grid-template-columns: 1fr; }} }}
  .filter-row {{
    display: flex; gap: 12px; align-items: center; margin-bottom: 14px; flex-wrap: wrap;
  }}
  .filter-row select {{ padding: 5px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.82rem; }}
  .filter-row label {{ font-size: 0.82rem; color: var(--muted); font-weight: 500; }}
</style>
</head>
<body>

<h1>Stratigraphy Experiments — Deep Layer Analysis</h1>
<p class="subtitle">6 experiments on per-layer representations — 397 core terms across 6 models — cached data, no new inference</p>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('t1',this)">1. Emergence</button>
  <button class="tab-btn" onclick="showTab('t2',this)">2. Cross-Lingual</button>
  <button class="tab-btn" onclick="showTab('t3',this)">3. Classification</button>
  <button class="tab-btn" onclick="showTab('t4',this)">4. Anisotropy</button>
  <button class="tab-btn" onclick="showTab('t5',this)">5. Migration</button>
  <button class="tab-btn" onclick="showTab('t6',this)">6. Phase Transitions</button>
</div>

<!-- ═══ Tab 1: Domain Emergence ═══ -->
<div id="t1" class="panel active">
  <div class="note">
    <h3>Experiment 1 — Domain Emergence Order</h3>
    <p>For each legal domain $d$, we compute a <b>domain-specific signal</b> at every layer.
    This isolates pairs where both terms belong to domain $d$ (intra) vs pairs where one
    term is in $d$ and the other is not (inter):</p>
    <div class="formula">
      $$r_d(\\ell) = 1 - \\frac{{2\\,U_d(\\ell)}}{{n_{{\\text{{intra}}}} \\cdot n_{{\\text{{inter}}}}}}$$
    </div>
    <p>The <b>emergence layer</b> is the first layer where $r_d > 0.05$. Earlier emergence
    means the model encodes that domain's internal structure sooner. This reveals whether
    there is a natural ordering in how legal domains crystallize — do foundational domains
    (constitutional, criminal) emerge before technical ones (procedure, administrative)?</p>
    <p>The bar chart at the bottom shows the emergence order averaged across all 6 models.</p>
  </div>
  <div class="card">
    <div class="filter-row">
      <label>Model:</label>
      <select id="e1Model" onchange="drawE1()"></select>
    </div>
    <div id="e1Curves" style="height:400px"></div>
  </div>
  <div class="card">
    <div id="e1Bars" style="height:350px"></div>
  </div>
</div>

<!-- ═══ Tab 2: Cross-Lingual ═══ -->
<div id="t2" class="panel">
  <div class="note">
    <h3>Experiment 2 — Cross-Lingual Layer Alignment</h3>
    <p>For each pair of models (one WEIRD, one Sinic), we compute the
    <b>RSA correlation</b> between every combination of layers:</p>
    <div class="formula">
      $$\\rho(\\ell_W, \\ell_S) = \\text{{Spearman}}\\big(\\text{{upper\\_tri}}(\\text{{RDM}}_W^{{(\\ell_W)}}),\\;
      \\text{{upper\\_tri}}(\\text{{RDM}}_S^{{(\\ell_S)}})\\big)$$
    </div>
    <p>The heatmap shows $\\rho$ for all layer combinations. If the maximum $\\rho$ lies
    on the diagonal, the models process legal semantics at the same depth. An off-diagonal
    maximum means one tradition needs more/fewer layers to reach the same relational structure.</p>
    <p>Three pairs are shown: same-family (BGE-EN vs BGE-ZH), same-pooling (E5 vs Text2vec),
    and asymmetric (FreeLaw 22L vs Dmeta 12L).</p>
  </div>
  <div class="split3" id="e2Container"></div>
</div>

<!-- ═══ Tab 3: Classification ═══ -->
<div id="t3" class="panel">
  <div class="note">
    <h3>Experiment 3 — k-NN Domain Classification</h3>
    <p>At each layer, every term is classified into a domain by <b>majority vote</b>
    among its $k=7$ nearest neighbors. If 4+ of the 7 neighbors belong to domain $d$,
    the term is predicted as $d$.</p>
    <p>The accuracy curve shows what fraction of terms are correctly classified at each
    layer. This complements the domain signal $r$ (which measures distributional separation)
    with a <b>pointwise prediction</b> metric — can the geometry alone recover the
    domain label of each individual term?</p>
    <p>The confusion matrices show which domains get confused with each other at early,
    middle, and late layers.</p>
  </div>
  <div class="card">
    <div id="e3Accuracy" style="height:400px"></div>
  </div>
  <div class="card">
    <h3>Per-domain accuracy at final layer</h3>
    <div id="e3DomainBars" style="height:300px"></div>
  </div>
  <div class="card">
    <h3>Confusion matrices (first WEIRD model)</h3>
    <p class="card-sub">Rows = true domain, columns = predicted domain. Darker = more terms. Normalized by row.</p>
    <div id="e3CMs"></div>
  </div>
</div>

<!-- ═══ Tab 4: Anisotropy ═══ -->
<div id="t4" class="panel">
  <div class="note">
    <h3>Experiment 4 — Anisotropy Profile</h3>
    <p>Anisotropy measures how <b>uniformly distributed</b> the representations are in the
    embedding space. We compute the <b>mean pairwise cosine similarity</b> at each layer:</p>
    <div class="formula">
      $$\\text{{aniso}}(\\ell) = \\frac{{1}}{{\\binom{{n}}{{2}}}} \\sum_{{i < j}} \\cos(\\mathbf{{h}}_i^{{(\\ell)}}, \\mathbf{{h}}_j^{{(\\ell)}})$$
    </div>
    <p>High anisotropy (mean sim near 1) means all vectors point in similar directions —
    the space is "narrow." Low anisotropy (mean sim near 0) means vectors are spread out.
    Ethayarajh (2019) showed that BERT becomes increasingly anisotropic in deeper layers.</p>
    <p>We compare <b>legal terms</b> (397 core) vs <b>control terms</b> (100 non-legal).
    If legal terms become more anisotropic than control terms, the model is actively
    constructing a tighter "legal subspace" in deeper layers.</p>
  </div>
  <div class="card">
    <div class="filter-row">
      <label>Model:</label>
      <select id="e4Model" onchange="drawE4()"></select>
    </div>
    <div id="e4Chart" style="height:400px"></div>
  </div>
  <div class="card">
    <div id="e4AllModels" style="height:350px"></div>
  </div>
</div>

<!-- ═══ Tab 5: Migration ═══ -->
<div id="t5" class="panel">
  <div class="note">
    <h3>Experiment 5 — Domain Migration Paths</h3>
    <p>For each term, at each layer, we determine the <b>majority domain</b> of its $k=7$
    nearest neighbors. The sequence of majority domains across layers reveals the term's
    <b>migration path</b> — which legal categories does the model "try on" before settling?</p>
    <p>The heatmap shows domain identity (color) at each layer for each term. A column of
    uniform color means the model assigns the same domain from the start. Horizontal color
    changes show domain transitions. Terms that traverse multiple domains are the most
    interesting for the thesis — they show the model engaging in something analogous to
    Tarello's systematic interpretation.</p>
    <p>The bar chart at bottom counts how many terms change their majority domain at each
    layer transition. Peaks indicate layers where major domain reassignment occurs.</p>
  </div>
  <div class="card">
    <div class="filter-row">
      <label>Model:</label>
      <select id="e5Model" onchange="drawE5()"></select>
    </div>
    <div id="e5Heatmap" style="height:500px"></div>
  </div>
  <div class="card">
    <h3>Domain transition counts per layer</h3>
    <p class="card-sub">How many terms change their majority domain at each layer transition?</p>
    <div id="e5Transitions" style="height:300px"></div>
  </div>
  <div class="card">
    <h3>Most migrating terms</h3>
    <p class="card-sub">Top 20 terms with most domain transitions across layers.</p>
    <div id="e5TopChangers"></div>
  </div>
</div>

<!-- ═══ Tab 6: Phase Transitions ═══ -->
<div id="t6" class="panel">
  <div class="note">
    <h3>Experiment 6 — Phase Transition Detection</h3>
    <p>We look for <b>discontinuities</b> in the domain signal curve $r(\\ell)$.
    The first derivative $\\Delta r(\\ell) = r(\\ell+1) - r(\\ell)$ measures the rate of
    change. The second derivative $\\Delta^2 r$ measures acceleration.</p>
    <p>A large $|\\Delta r|$ at layer $\\ell$ indicates a <b>phase transition</b> — the
    domain structure undergoes rapid reorganization at that depth. If all models show a
    phase transition at the same relative depth (e.g., ~60% of total layers), this
    suggests a universal architectural property. If transitions occur at different
    relative depths, the processing schedule is model-specific.</p>
    <p>The <b>saturation layer</b> is where $r$ first reaches 90% of its maximum value.</p>
  </div>
  <div class="card">
    <div id="e6Signal" style="height:350px"></div>
  </div>
  <div class="split">
    <div class="card"><div id="e6Delta" style="height:300px"></div></div>
    <div class="card"><div id="e6Delta2" style="height:300px"></div></div>
  </div>
  <div class="card">
    <h3>Summary</h3>
    <div id="e6Summary"></div>
  </div>
</div>

<script>
var D = {data_js};
var MC = {{
  'BGE-EN-large':'#0072B2','E5-large':'#E69F00','FreeLaw-EN':'#009E73',
  'BGE-ZH-large':'#56B4E9','Text2vec-large-ZH':'#D55E00','Dmeta-ZH':'#CC79A7'
}};
var SH = {json.dumps(SHORT)};
var DC = {json.dumps(DOMAIN_COLORS)};
var MODELS = D.models;

function showTab(id, btn) {{
  document.querySelectorAll('.panel').forEach(function(el){{el.classList.remove('active')}});
  document.querySelectorAll('.tab-btn').forEach(function(el){{el.classList.remove('active')}});
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  var drawMap = {{t1:drawE1,t2:drawE2,t3:drawE3,t4:drawE4,t5:drawE5,t6:drawE6}};
  if (drawMap[id]) setTimeout(drawMap[id], 60);
}}

function fillSelect(id, models) {{
  var sel = document.getElementById(id);
  if (sel.options.length > 0) return;
  models.forEach(function(m) {{
    var opt = document.createElement('option');
    opt.value = m; opt.textContent = SH[m]||m;
    sel.appendChild(opt);
  }});
}}

/* ── Exp 1: Domain Emergence ── */
function drawE1() {{
  fillSelect('e1Model', MODELS);
  var model = document.getElementById('e1Model').value || MODELS[0];
  var e1 = D.exp1_emergence[model];
  var domains = Object.keys(e1.per_domain).sort();
  var traces = domains.map(function(d) {{
    var pd = e1.per_domain[d];
    return {{
      x: Array.from({{length:pd.r_curve.length}}, function(_,i){{return i}}),
      y: pd.r_curve, mode:'lines+markers', name: d,
      line: {{color: DC[d]||'#888', width:2}}, marker:{{size:4}},
    }};
  }});
  Plotly.newPlot('e1Curves', traces, {{
    title:{{text:'Per-domain signal r across layers — '+SH[model], font:{{size:13}}}},
    margin:{{l:50,r:20,t:40,b:45}},
    xaxis:{{title:'Layer'}}, yaxis:{{title:'Domain signal r'}},
    template:'simple_white', hovermode:'x unified',
    legend:{{font:{{size:9}}, orientation:'h', y:-0.18}},
    shapes:[{{type:'line',x0:0,x1:e1.n_states-1,y0:0.05,y1:0.05,
      line:{{color:'#ccc',dash:'dot',width:1}}}}],
  }}, {{responsive:true}});

  // Bar chart: emergence order averaged across models
  var avgEmergence = {{}};
  domains.forEach(function(d) {{
    var vals = [];
    MODELS.forEach(function(m) {{
      var em = D.exp1_emergence[m].per_domain[d].emergence_layer;
      if (em !== null) vals.push(em);
    }});
    avgEmergence[d] = vals.length > 0
      ? vals.reduce(function(a,b){{return a+b}},0)/vals.length : null;
  }});
  var sorted = domains.filter(function(d){{return avgEmergence[d]!==null}})
    .sort(function(a,b){{return avgEmergence[a]-avgEmergence[b]}});
  var notEmerged = domains.filter(function(d){{return avgEmergence[d]===null}});
  sorted = sorted.concat(notEmerged);

  Plotly.newPlot('e1Bars', [{{
    type:'bar', orientation:'h',
    y: sorted.map(function(d){{return d}}).reverse(),
    x: sorted.map(function(d){{return avgEmergence[d]||0}}).reverse(),
    marker:{{color: sorted.map(function(d){{return DC[d]||'#888'}}).reverse()}},
    text: sorted.map(function(d){{
      return avgEmergence[d]!==null ? 'L'+avgEmergence[d].toFixed(1) : 'Never';
    }}).reverse(),
    textposition:'outside', textfont:{{size:10}},
    hovertemplate:'%{{y}}: L%{{x:.1f}}<extra></extra>',
  }}], {{
    title:{{text:'Domain emergence order (avg across 6 models, threshold r>0.05)',font:{{size:12}}}},
    margin:{{l:120,r:60,t:40,b:40}},
    xaxis:{{title:'Avg emergence layer'}},
    template:'simple_white',
  }}, {{responsive:true}});
}}

/* ── Exp 2: Cross-Lingual ── */
function drawE2() {{
  var container = document.getElementById('e2Container');
  container.textContent = '';
  var pairs = Object.keys(D.exp2_crosslingual);
  pairs.forEach(function(pair, pi) {{
    var d = D.exp2_crosslingual[pair];
    var div = document.createElement('div');
    div.className = 'card';
    var plotDiv = document.createElement('div');
    plotDiv.id = 'e2_' + pi;
    plotDiv.style.height = '380px';
    div.appendChild(plotDiv);
    container.appendChild(div);

    setTimeout(function() {{
      Plotly.newPlot('e2_'+pi, [{{
        type:'heatmap', z:d.matrix,
        x: Array.from({{length:d.n_sinic_layers}}, function(_,i){{return 'L'+i}}),
        y: Array.from({{length:d.n_weird_layers}}, function(_,i){{return 'L'+i}}),
        colorscale:'RdBu', zmid:0,
        hovertemplate:'WEIRD L%{{y}}, Sinic L%{{x}}<br>\\u03C1 = %{{z:.3f}}<extra></extra>',
        colorbar:{{title:'\\u03C1', thickness:12}},
      }}], {{
        title:{{text:pair+' (max \\u03C1='+d.max_rho+')', font:{{size:12}}}},
        margin:{{l:50,r:50,t:40,b:50}},
        xaxis:{{title:SH[d.sinic_model]+' layer', side:'bottom'}},
        yaxis:{{title:SH[d.weird_model]+' layer'}},
      }}, {{responsive:true}});
    }}, 80 + pi*50);
  }});
}}

/* ── Exp 3: Classification ── */
function drawE3() {{
  // Accuracy curves
  var traces = MODELS.map(function(m) {{
    var e = D.exp3_classification[m];
    return {{
      x: Array.from({{length:e.accuracy_curve.length}}, function(_,i){{return i}}),
      y: e.accuracy_curve, mode:'lines+markers', name:SH[m],
      line:{{color:MC[m], width:2}}, marker:{{size:4}},
    }};
  }});
  Plotly.newPlot('e3Accuracy', traces, {{
    title:{{text:'k-NN classification accuracy across layers',font:{{size:13}}}},
    margin:{{l:50,r:20,t:40,b:45}},
    xaxis:{{title:'Layer'}}, yaxis:{{title:'Accuracy', range:[0,1]}},
    template:'simple_white', hovermode:'x unified',
    legend:{{font:{{size:10}}, orientation:'h', y:-0.18}},
  }}, {{responsive:true}});

  // Per-domain bars at final layer (first model)
  var firstM = MODELS[0];
  var domAcc = D.exp3_classification[firstM].per_domain_acc_final;
  var doms = Object.keys(domAcc).sort(function(a,b){{return domAcc[b]-domAcc[a]}});
  Plotly.newPlot('e3DomainBars', [{{
    type:'bar', x:doms, y:doms.map(function(d){{return domAcc[d]}}),
    marker:{{color:doms.map(function(d){{return DC[d]||'#888'}})}},
    text:doms.map(function(d){{return (domAcc[d]*100).toFixed(0)+'%'}}),
    textposition:'outside', textfont:{{size:10}},
  }}], {{
    title:{{text:'Per-domain accuracy at final layer ('+SH[firstM]+')',font:{{size:12}}}},
    margin:{{l:50,r:20,t:40,b:80}},
    xaxis:{{tickangle:-30}}, yaxis:{{title:'Accuracy',range:[0,1.1]}},
    template:'simple_white',
  }}, {{responsive:true}});

  // Confusion matrices
  var cmDiv = document.getElementById('e3CMs');
  cmDiv.textContent = '';
  var cmData = D.exp3_classification[firstM].confusion_at;
  var cmLayers = Object.keys(cmData).sort(function(a,b){{return a-b}});
  var grid = document.createElement('div');
  grid.className = 'split3';

  cmLayers.forEach(function(l, ci) {{
    var cm = cmData[l];
    var mat = cm.matrix;
    var doms2 = cm.domains;
    // Normalize by row
    var norm = mat.map(function(row) {{
      var s = row.reduce(function(a,b){{return a+b}},0);
      return s > 0 ? row.map(function(v){{return Math.round(v/s*100)/100}}) : row;
    }});

    var card = document.createElement('div');
    card.className = 'card';
    card.style.padding = '10px';
    var plotDiv = document.createElement('div');
    plotDiv.id = 'cm_' + ci;
    plotDiv.style.height = '300px';
    card.appendChild(plotDiv);
    grid.appendChild(card);

    setTimeout(function() {{
      Plotly.newPlot('cm_'+ci, [{{
        type:'heatmap', z:norm.slice().reverse(),
        x:doms2, y:doms2.slice().reverse(),
        colorscale:'Blues',
        hovertemplate:'True: %{{y}}<br>Pred: %{{x}}<br>%{{z:.2f}}<extra></extra>',
        showscale: ci === cmLayers.length-1,
      }}], {{
        title:{{text:'Layer '+l, font:{{size:11}}}},
        margin:{{l:80,r:20,t:35,b:60}},
        xaxis:{{tickangle:-45, tickfont:{{size:8}}}},
        yaxis:{{tickfont:{{size:8}}}},
      }}, {{responsive:true}});
    }}, 100 + ci*50);
  }});
  cmDiv.appendChild(grid);
}}

/* ── Exp 4: Anisotropy ── */
function drawE4() {{
  fillSelect('e4Model', MODELS);
  var model = document.getElementById('e4Model').value || MODELS[0];
  var e = D.exp4_anisotropy[model];
  var layers = Array.from({{length:e.legal.length}}, function(_,i){{return i}});

  var traces = [
    {{x:layers, y:e.legal, mode:'lines+markers', name:'Legal (core)',
      line:{{color:'#0072B2',width:2}}, marker:{{size:4}}}},
  ];
  if (e.control) {{
    traces.push({{x:layers, y:e.control, mode:'lines+markers', name:'Control',
      line:{{color:'#999',width:2,dash:'dash'}}, marker:{{size:4}}}});
  }}
  traces.push({{x:layers, y:e.overall, mode:'lines', name:'Overall',
    line:{{color:'#ccc',width:1}}, opacity:0.5}});

  Plotly.newPlot('e4Chart', traces, {{
    title:{{text:'Anisotropy profile — '+SH[model], font:{{size:13}}}},
    margin:{{l:50,r:20,t:40,b:45}},
    xaxis:{{title:'Layer'}}, yaxis:{{title:'Mean pairwise cosine similarity'}},
    template:'simple_white', hovermode:'x unified',
    legend:{{font:{{size:10}}}},
  }}, {{responsive:true}});

  // All models: legal anisotropy
  var allTraces = MODELS.map(function(m) {{
    var ed = D.exp4_anisotropy[m];
    var ls = Array.from({{length:ed.legal.length}}, function(_,i){{return i/(ed.legal.length-1)}});
    return {{x:ls, y:ed.legal, mode:'lines+markers', name:SH[m],
      line:{{color:MC[m],width:2}}, marker:{{size:3}}}};
  }});
  Plotly.newPlot('e4AllModels', allTraces, {{
    title:{{text:'Legal term anisotropy — all models (x = relative depth)',font:{{size:12}}}},
    margin:{{l:50,r:20,t:40,b:45}},
    xaxis:{{title:'Relative depth (0=embedding, 1=final)'}},
    yaxis:{{title:'Mean cosine similarity (legal terms)'}},
    template:'simple_white', hovermode:'x unified',
    legend:{{font:{{size:10}}, orientation:'h', y:-0.18}},
  }}, {{responsive:true}});
}}

/* ── Exp 5: Migration ── */
function drawE5() {{
  fillSelect('e5Model', MODELS);
  var model = document.getElementById('e5Model').value || MODELS[0];
  var e = D.exp5_migration[model];
  var doms = e.domains;
  var migration = e.migration_truncated;
  var names = e.term_names_truncated;
  var n_states = e.n_states;

  // Build discrete colorscale
  var nDom = doms.length;
  var cscale = [];
  doms.forEach(function(d,i) {{
    var c = DC[d]||'#888';
    cscale.push([i/nDom, c]);
    cscale.push([(i+1)/nDom, c]);
  }});

  var labels = names.map(function(n,i) {{
    return n + ' [' + e.true_domains_truncated[i] + ']';
  }});

  Plotly.newPlot('e5Heatmap', [{{
    type:'heatmap', z:migration,
    x: Array.from({{length:n_states}}, function(_,i){{return 'L'+i}}),
    y: labels,
    colorscale: cscale, zmin:0, zmax:nDom,
    hovertemplate:'%{{y}}<br>Layer %{{x}}: '+
      '<extra></extra>',
    showscale:false,
  }}], {{
    title:{{text:'Majority-domain trajectory (first 100 terms) — '+SH[model],font:{{size:12}}}},
    margin:{{l:180,r:30,t:40,b:50}},
    xaxis:{{title:'Layer', side:'bottom'}},
    yaxis:{{autorange:'reversed', tickfont:{{size:7}}}},
    height: Math.max(400, migration.length * 6),
  }}, {{responsive:true}});

  // Transition counts
  Plotly.newPlot('e5Transitions', [{{
    type:'bar',
    x: Array.from({{length:e.transition_counts.length}}, function(_,i){{return 'L'+i+'\\u2192L'+(i+1)}}),
    y: e.transition_counts,
    marker:{{color:'rgba(0,114,178,0.6)', line:{{width:1, color:'#0072B2'}}}},
  }}], {{
    title:{{text:'Terms changing majority domain per transition — '+SH[model],font:{{size:12}}}},
    margin:{{l:45,r:20,t:40,b:60}},
    xaxis:{{tickangle:-30}}, yaxis:{{title:'# terms changing domain'}},
    template:'simple_white',
  }}, {{responsive:true}});

  // Top changers table
  var tcDiv = document.getElementById('e5TopChangers');
  tcDiv.textContent = '';
  var table = document.createElement('table');
  table.style.cssText = 'width:100%;border-collapse:collapse;font-size:0.8rem';
  var thead = document.createElement('thead');
  var hr = document.createElement('tr');
  ['#','Term','Domain','Transitions','Path'].forEach(function(h) {{
    var th = document.createElement('th');
    th.textContent = h;
    th.style.cssText = 'padding:6px 8px;text-align:left;border-bottom:2px solid #e2e8f0;font-size:0.75rem;color:#64748b';
    hr.appendChild(th);
  }});
  thead.appendChild(hr);
  table.appendChild(thead);
  var tbody = document.createElement('tbody');
  e.top_changers.forEach(function(tc, i) {{
    var tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid #f1f5f9';
    var vals = [i+1, tc.term, tc.domain, tc.n_transitions];
    vals.forEach(function(v, ci) {{
      var td = document.createElement('td');
      td.style.padding = '5px 8px';
      if (ci === 2) {{
        var badge = document.createElement('span');
        badge.style.cssText = 'display:inline-block;padding:1px 6px;border-radius:8px;font-size:0.72em;color:#fff;background:'+(DC[v]||'#888');
        badge.textContent = v;
        td.appendChild(badge);
      }} else {{
        td.textContent = v;
      }}
      tr.appendChild(td);
    }});
    // Path as colored dots
    var tdP = document.createElement('td');
    tdP.style.padding = '5px 8px';
    tc.path.forEach(function(d, pi) {{
      var dot = document.createElement('span');
      dot.style.cssText = 'display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 1px;background:'+(DC[d]||'#888');
      dot.title = 'L'+pi+': '+d;
      tdP.appendChild(dot);
    }});
    tr.appendChild(tdP);
    tbody.appendChild(tr);
  }});
  table.appendChild(tbody);
  tcDiv.appendChild(table);
}}

/* ── Exp 6: Phase Transitions ── */
function drawE6() {{
  // Signal r curves
  var traces = MODELS.map(function(m) {{
    var e = D.exp6_phase[m];
    return {{
      x: Array.from({{length:e.r_curve.length}}, function(_,i){{return i}}),
      y: e.r_curve, mode:'lines+markers', name:SH[m],
      line:{{color:MC[m],width:2}}, marker:{{size:4}},
    }};
  }});
  // Add change point markers
  MODELS.forEach(function(m) {{
    var e = D.exp6_phase[m];
    if (e.change_point_layer !== null) {{
      traces.push({{
        x:[e.change_point_layer], y:[e.r_curve[e.change_point_layer]],
        mode:'markers', name:SH[m]+' CP',
        marker:{{size:14, symbol:'star', color:MC[m], line:{{width:1,color:'#fff'}}}},
        showlegend:false, hovertemplate:SH[m]+' change point: L%{{x}}<extra></extra>',
      }});
    }}
  }});
  Plotly.newPlot('e6Signal', traces, {{
    title:{{text:'Domain signal r with change points (\\u2605)',font:{{size:13}}}},
    margin:{{l:50,r:20,t:40,b:45}},
    xaxis:{{title:'Layer'}}, yaxis:{{title:'Domain signal r'}},
    template:'simple_white', hovermode:'closest',
    legend:{{font:{{size:9}}, orientation:'h', y:-0.18}},
  }}, {{responsive:true}});

  // Delta r
  var dTraces = MODELS.map(function(m) {{
    var e = D.exp6_phase[m];
    return {{
      x: Array.from({{length:e.delta_r.length}}, function(_,i){{return i}}),
      y: e.delta_r, mode:'lines+markers', name:SH[m],
      line:{{color:MC[m],width:1.5}}, marker:{{size:3}},
    }};
  }});
  Plotly.newPlot('e6Delta', dTraces, {{
    title:{{text:'\\u0394r (first derivative)',font:{{size:12}}}},
    margin:{{l:50,r:20,t:40,b:45}},
    xaxis:{{title:'Layer transition'}}, yaxis:{{title:'\\u0394r'}},
    template:'simple_white', hovermode:'x unified',
    legend:{{font:{{size:9}}}}, shapes:[{{type:'line',x0:0,x1:25,y0:0,y1:0,line:{{color:'#ccc',dash:'dot'}}}}],
  }}, {{responsive:true}});

  // Delta2 r
  var d2Traces = MODELS.map(function(m) {{
    var e = D.exp6_phase[m];
    return {{
      x: Array.from({{length:e.delta2_r.length}}, function(_,i){{return i}}),
      y: e.delta2_r, mode:'lines+markers', name:SH[m],
      line:{{color:MC[m],width:1.5}}, marker:{{size:3}},
    }};
  }});
  Plotly.newPlot('e6Delta2', d2Traces, {{
    title:{{text:'\\u0394\\u00B2r (second derivative)',font:{{size:12}}}},
    margin:{{l:50,r:20,t:40,b:45}},
    xaxis:{{title:'Layer'}}, yaxis:{{title:'\\u0394\\u00B2r'}},
    template:'simple_white', hovermode:'x unified',
    legend:{{font:{{size:9}}}}, shapes:[{{type:'line',x0:0,x1:25,y0:0,y1:0,line:{{color:'#ccc',dash:'dot'}}}}],
  }}, {{responsive:true}});

  // Summary table
  var sumDiv = document.getElementById('e6Summary');
  sumDiv.textContent = '';
  var table = document.createElement('table');
  table.style.cssText = 'width:100%;border-collapse:collapse;font-size:0.82rem';
  var thead = document.createElement('thead');
  var hr = document.createElement('tr');
  ['Model','Layers','Max r','Change Point','\\u0394r at CP','Saturation (90%)','Accel Peak'].forEach(function(h) {{
    var th = document.createElement('th');
    th.textContent = h;
    th.style.cssText = 'padding:8px;text-align:left;border-bottom:2px solid #e2e8f0;font-size:0.75rem;color:#64748b';
    hr.appendChild(th);
  }});
  thead.appendChild(hr);
  table.appendChild(thead);
  var tbody = document.createElement('tbody');
  MODELS.forEach(function(m) {{
    var e = D.exp6_phase[m];
    var tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid #f1f5f9';
    var vals = [
      SH[m], e.n_states-1, e.max_r,
      e.change_point_layer !== null ? 'L'+e.change_point_layer : '\\u2014',
      e.change_point_delta !== 0 ? (e.change_point_delta > 0 ? '+':'')+e.change_point_delta.toFixed(4) : '\\u2014',
      e.saturation_layer !== null ? 'L'+e.saturation_layer : '\\u2014',
      e.accel_layer !== null ? 'L'+e.accel_layer : '\\u2014',
    ];
    vals.forEach(function(v) {{
      var td = document.createElement('td');
      td.style.padding = '6px 8px';
      td.textContent = v;
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
  table.appendChild(tbody);
  sumDiv.appendChild(table);
}}

/* ── Init ── */
drawE1();
</script>
</body>
</html>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    t_start = time.perf_counter()
    weird_labels, sinic_labels = _load_config()
    all_models = weird_labels + sinic_labels
    terms_core, core_idx = _load_core_terms()

    print("=" * 60)
    print("Stratigraphy Experiments — Deep Layer Analysis")
    print(f"  Models: {[SHORT[m] for m in all_models]}")
    print(f"  Core terms: {len(terms_core)}")
    print("=" * 60)

    # Phase 0: Precompute RDM upper triangles
    tri_cache = precompute_tris(all_models, terms_core)

    # Run all 6 experiments
    results = {"models": all_models}
    results["exp1_emergence"] = exp_domain_emergence(all_models, terms_core, tri_cache)
    results["exp2_crosslingual"] = exp_cross_lingual(tri_cache)
    results["exp3_classification"] = exp_knn_classification(all_models, terms_core)
    results["exp4_anisotropy"] = exp_anisotropy(all_models, terms_core)
    results["exp5_migration"] = exp_domain_migration(all_models, terms_core)
    results["exp6_phase"] = exp_phase_transitions(all_models, terms_core, tri_cache)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"All experiments done in {elapsed:.0f}s")

    # Save JSON
    json_path = RESULTS_DIR / "stratigraphy_experiments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON -> {json_path}")

    # Build HTML
    html = build_html(results)
    html_path = RESULTS_DIR / "stratigraphy_experiments.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML -> {html_path}")


if __name__ == "__main__":
    main()
