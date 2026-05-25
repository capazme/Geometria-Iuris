"""Loader for the 9 extensions in `experiments/ch3-measurability/ext/`.

Returns compact JSON for each of:
  A — bg k-NN domain assignments              (background_assignments.json)
  D — Δρ_sym vs %background curve             (robustness_curve.json)
  E — axes out-of-sample coherence            (coherence.json)
  F — confidence-stratified bg injection      (confidence_strata.json)
  G — false-friends bilingual divergence      (false_friends.json + csv)
  H — K saturation curve                      (k_saturation.json)
  X — Δρ_sym vs %control curve                (control_robustness_curve.json)
  Y — control-only Δρ_sym (bare)              (control_only_rsa.json)
  Z — tier hierarchy (core / bg / control)    (tier_hierarchy.json)

The narrative classification (headline-strengthening vs caveat) lives in
the Robustness page, not here.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXT = REPO_ROOT / "experiments" / "ch3-measurability" / "ext"


def _load_json(rel_path: str) -> dict:
    with (EXT / rel_path).open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Individual loaders.

def load_A_bg_knn() -> dict:
    """A — k-NN domain assignment of the 9 045 background terms."""
    return _load_json("A_bg_knn/background_assignments.json")


def load_D_robustness() -> dict:
    """D — Δρ_sym attested vs %bg injection (0, 10, 25, 50, 75)."""
    return _load_json("D_robustness/robustness_curve.json")


def load_E_axes_oos() -> dict:
    """E — out-of-sample axis projection coherence on the 9 045 bg."""
    return _load_json("E_axes_oos/coherence.json")


def load_F_confidence() -> dict:
    """F — Δρ_sym stratified by k-NN confidence (low / high / random)."""
    return _load_json("F_confidence/confidence_strata.json")


def load_G_false_friends() -> dict:
    """G — same-lemma divergence: cross-encoder cosine vs bilingual cosine."""
    return _load_json("G_false_friends/false_friends.json")


def load_H_k_saturation() -> dict:
    """H — ρ_cross attested as a function of minimum K (K=1..8)."""
    return _load_json("H_K_saturation/k_saturation.json")


def load_X_control_robustness() -> dict:
    """X — Δρ_sym bare vs %control injection (dual of D)."""
    return _load_json("X_control_robustness/control_robustness_curve.json")


def load_Y_control_only() -> dict:
    """Y — Δρ_sym bare computed on the 100 everyday-language control terms.

    The canonical reframing: this returns 0.156, indistinguishable from the
    bare core baseline (0.165). The legal signal is the attested-bare gap
    on the core (0.543 − 0.165 = 0.378), not the attested absolute.
    """
    return _load_json("Y_control_only/control_only_rsa.json")


def load_Z_tier_hierarchy() -> dict:
    """Z — median(core-core) < median(core-bg) < median(core-control)?
    Reports per-model monotonic hierarchy boolean (3/10 in run #4)."""
    return _load_json("Z_tier_hierarchy/tier_hierarchy.json")


# --------------------------------------------------------------------------
# Headline number extractors — convenience accessors used by the Robustness
# page to fill the canonical tables and Y-caveat callout.

def y_caveat_numbers() -> dict:
    """Pack the 4 numbers that the Y caveat callout displays:
    attested 0.543, bare 0.165, bare-control 0.156, gap 0.378."""
    y = load_Y_control_only()
    cmp_ = y["comparison"]
    bare_core = float(cmp_["delta_sym_core_bare_run4"])
    attested_core = float(cmp_["delta_sym_core_attested_run4"])
    bare_control = float(cmp_["delta_sym_control_bare"])
    return {
        "attested_core": attested_core,                       # 0.543
        "bare_core":     bare_core,                           # 0.165
        "bare_control":  bare_control,                        # 0.156 (run #4: 0.1557)
        "legal_gap":     round(attested_core - bare_core, 3), # 0.378
    }


def d_robustness_table() -> list[dict]:
    """D table for the curve: one entry per p_bg level."""
    return [
        {
            "p_bg":           float(r["p_bg"]),
            "mean_delta_sym": float(r["mean_delta_sym"]),
            "ci_low":         float(r["ci_low_delta_sym"]),
            "ci_high":        float(r["ci_high_delta_sym"]),
            "mean_rho_cross": float(r["mean_rho_cross"]),
            "mean_rho_W":     float(r["mean_rho_W"]),
            "mean_rho_S":     float(r["mean_rho_S"]),
            "n_bg":           int(r["n_bg"]),
            "n_replicates":   int(r["n_replicates"]),
        }
        for r in load_D_robustness()["results"]
    ]


def x_robustness_table() -> list[dict]:
    """X table for the dual curve."""
    return [
        {
            "p_control":      float(r["p_control"]),
            "mean_delta_sym": float(r["mean_delta_sym"]),
            "ci_low":         float(r["ci_low_delta_sym"]),
            "ci_high":        float(r["ci_high_delta_sym"]),
            "n_control":      int(r["n_control"]),
            "n_replicates":   int(r["n_replicates"]),
        }
        for r in load_X_control_robustness()["results"]
    ]


def h_saturation_table() -> list[dict]:
    """H table for the saturation curve."""
    return [
        {
            "K":               r["K_bucket"],
            "mean_rho_cross":  float(r["mean_rho_cross"]),
            "std_rho_cross":   float(r["std_rho_cross"]),
            "n_bg":            int(r["n_bg"]),
            "n_common_nonzero": int(r["n_common_nonzero"]),
        }
        for r in load_H_k_saturation()["buckets"]
    ]


def f_confidence_table() -> list[dict]:
    """F table: baseline + 3 injection strata."""
    f = load_F_confidence()
    base = f["baseline_core_only"]
    rows = [
        {
            "stratum":        "baseline (core only)",
            "mean_delta_sym": float(base["delta_sym"]),
            "ci_low":         None,
            "ci_high":        None,
        },
    ]
    for key, label in [
        ("high_confidence_bg_injected", "high-confidence bg"),
        ("low_confidence_bg_injected",  "low-confidence bg"),
        ("random_control_bg_injected",  "random bg (control)"),
    ]:
        if key in f:
            r = f[key]
            rows.append({
                "stratum":        label,
                "mean_delta_sym": float(r["mean_delta_sym"]),
                "ci_low":         float(r["ci_low_delta_sym"]),
                "ci_high":        float(r["ci_high_delta_sym"]),
            })
    return rows


def z_tier_table() -> list[dict]:
    """Z table: per-model median (core-core / core-bg / core-control)
    + monotonic hierarchy boolean."""
    z = load_Z_tier_hierarchy()
    rows = []
    for model, blob in z["per_model"].items():
        med = blob["median"]
        rows.append({
            "model":            model,
            "median_core_core": float(med["core_core"]),
            "median_core_bg":   float(med["core_bg"]),
            "median_core_ctrl": float(med["core_control"]),
            "monotonic":        bool(blob.get("monotonic_hierarchy", False)),
        })
    return rows


def g_false_friends_top(n: int = 12) -> list[dict]:
    """G top false-friend rows, ranked by `cos_bilingual − cos_cross_encoder`
    (largest divergence first).

    Each row: {en, zh, k_min, cos_cross, cos_bilingual}.
    """
    g = load_G_false_friends()
    rows = g["rows"]
    # Field names depend on the encoders configured — pull dynamically.
    keys = list(rows[0].keys()) if rows else []
    cross_key = next((k for k in keys if k.startswith("cos_BGE-EN") and "BGE-ZH" in k), None)
    bili_key = next((k for k in keys if k.startswith("cos_BGE-M3-EN")), None)
    if not (cross_key and bili_key):
        return []
    enriched = []
    for r in rows:
        cross = r.get(cross_key)
        bili = r.get(bili_key)
        if cross is None or bili is None:
            continue
        enriched.append({
            "en":            r["en"],
            "zh":            r["zh"],
            "k_min":         int(r.get("k_min", 0)),
            "cos_cross":     float(cross),
            "cos_bilingual": float(bili),
            "divergence":    float(bili) - float(cross),
        })
    enriched.sort(key=lambda d: d["divergence"], reverse=True)
    return enriched[:n]


# --------------------------------------------------------------------------
# Bulk loader.

def load_all() -> dict:
    """Load every extension in one call (used by `pages/robustness_caveats.build`)."""
    return {
        "A": load_A_bg_knn(),
        "D": load_D_robustness(),
        "E": load_E_axes_oos(),
        "F": load_F_confidence(),
        "G": load_G_false_friends(),
        "H": load_H_k_saturation(),
        "X": load_X_control_robustness(),
        "Y": load_Y_control_only(),
        "Z": load_Z_tier_hierarchy(),
    }
