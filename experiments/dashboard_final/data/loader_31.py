"""Loader for Experiment 1 (§3.1) — distance structure.

Reads `experiments/ch3-measurability/experiment_1_structure/results_{bare,attested}/`:
  - `experiment_1_results.json`      §3.1.1 intra-vs-inter, §3.1.2 7×7 topology,
                                     §3.1.3 RSA pairs, §3.1.1 legal-vs-control
  - `legal_vs_control.json`          standalone copy of §3.1.1 (bare only)
  - `categorical_probe.json`         §3.1.4 ordinal probe

The loader only consumes the JSON files: it does not touch the binary
`rdms/*.npz` (per-model 364×364), `distributions/*.npz` (Mantel null), or
`topology/*.npz` (also 7×7 but binary-encoded). The 7×7 topology RDM is
fully available inline inside `section_312.per_model.<m>.matrix`, which is
sufficient for the dashboard's needs.

Schema produced by `load_all()` documented at end of module.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CH3 = REPO_ROOT / "experiments" / "ch3-measurability" / "experiment_1_structure"
BARE_DIR = CH3 / "results_bare"
ATT_DIR = CH3 / "results_attested"


# --------------------------------------------------------------------------
# Model constants — stable across all figure builders and pages.

WEIRD_MODELS = ("BGE-EN-large", "E5-large", "FreeLaw-EN")
SINIC_MODELS = ("BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH")
BILINGUAL_MODELS = (
    "BGE-M3-EN", "BGE-M3-ZH",
    "Qwen3-0.6B-EN", "Qwen3-0.6B-ZH",
)
BILINGUAL_PAIRS = (
    ("BGE-M3-EN", "BGE-M3-ZH"),
    ("Qwen3-0.6B-EN", "Qwen3-0.6B-ZH"),
)

# Canonical display order: top row = English-side encoders, bottom row =
# Chinese-side encoders. Bilingual EN/ZH siblings are adjacent.
ALL_MODELS_ORDERED = (
    "BGE-EN-large", "E5-large", "FreeLaw-EN",
    "BGE-M3-EN",    "Qwen3-0.6B-EN",
    "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH",
    "BGE-M3-ZH",    "Qwen3-0.6B-ZH",
)


def model_group(m: str) -> str:
    if m in WEIRD_MODELS:
        return "weird"
    if m in SINIC_MODELS:
        return "sinic"
    return "bilingual"


def _classify_pair(a: str, b: str) -> str:
    """One of: within_weird, within_sinic, within_bilingual, cross."""
    pair = frozenset((a, b))
    for ea, eb in BILINGUAL_PAIRS:
        if pair == frozenset((ea, eb)):
            return "within_bilingual"
    a_g = model_group(a)
    b_g = model_group(b)
    if a_g == "weird" and b_g == "weird":
        return "within_weird"
    if a_g == "sinic" and b_g == "sinic":
        return "within_sinic"
    return "cross"


# --------------------------------------------------------------------------
# JSON readers.

def _load_results(variant: str) -> dict:
    """variant ∈ {'bare', 'attested'}."""
    base = BARE_DIR if variant == "bare" else ATT_DIR
    with (base / "experiment_1_results.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _flatten_section_313(s313: dict) -> list[dict]:
    """Collect the 4 sub-buckets into one list of 17 enriched dicts.

    Each entry carries: model_a, model_b, group (re-classified to ensure
    `within_bilingual` is its own bucket), rho, r_squared, p_value,
    ci_low, ci_high.
    """
    out: list[dict] = []
    for bucket in ("within_weird", "within_sinic",
                   "cross_tradition", "within_bilingual"):
        for entry in s313.get(bucket, []):
            d = dict(entry)
            d["group"] = _classify_pair(d["model_a"], d["model_b"])
            out.append(d)
    return out


# --------------------------------------------------------------------------
# Section loaders.

def load_section_311() -> dict:
    """§3.1.1 intra-vs-inter Mann-Whitney + legal-vs-control.

    Returns:
        {
          "bare": {
            "intra_inter": {<model>: {statistic, p_value, effect_r, n_x,
                                       n_y, median_x, median_y}, ...},
                                       # WEIRD only (3 models)
            "legal_vs_control": {<model>: {... same fields ...}, ...},
                                       # all 10 models
            "meta_lvc": {date, n_legal, n_control, control_kind, ...}
          },
          "attested": {... same shape ...},
          "meta": {... experiment_1 meta ...}
        }
    """
    bare = _load_results("bare")
    att = _load_results("attested")
    return {
        "bare": {
            "intra_inter":     dict(bare["section_311"]["per_model"]),
            "legal_vs_control": dict(bare["section_311_legal_vs_control"]["per_model"]),
            "meta_lvc":        dict(bare["section_311_legal_vs_control"]["meta"]),
        },
        "attested": {
            "intra_inter":     dict(att["section_311"]["per_model"]),
            # Attested has no legal-vs-control: controls have no HK Cap.
            # attestation, so the comparison is bare-only by design.
            "legal_vs_control": {},
            "meta_lvc":        {},
        },
        "meta": dict(bare["meta"]),
    }


def load_section_312(primary_model: str = "BGE-EN-large") -> dict:
    """§3.1.2 7×7 inter-domain topology RDM, per model.

    Returns:
        {
          "bare": {
            "primary": <model>,
            "domains": [...],
            "matrix":  [[...], ...],         # 7×7 primary RDM
            "all_models": {<m>: {"domains": [...], "matrix": [...]}, ...}
          },
          "attested": {... same shape ...},
          "meta": {...}
        }
    """
    bare = _load_results("bare")
    att = _load_results("attested")

    def _pack(results: dict, primary: str) -> dict:
        per = results["section_312"]["per_model"]
        if primary not in per:
            primary = next(iter(per.keys()))
        return {
            "primary":    primary,
            "domains":    list(per[primary]["domains"]),
            "matrix":     [list(row) for row in per[primary]["matrix"]],
            "all_models": {
                m: {
                    "domains": list(blob["domains"]),
                    "matrix":  [list(row) for row in blob["matrix"]],
                }
                for m, blob in per.items()
            },
        }

    return {
        "bare":     _pack(bare, primary_model),
        "attested": _pack(att, primary_model),
        "meta":     dict(bare["meta"]),
    }


def consensus_topology(s312: dict, variant: str = "bare") -> dict:
    """Mean of the K×K matrices across all models, with element-wise min/max.

    Returns `{"matrix", "matrix_min", "matrix_max", "domains", "n_models"}`.
    """
    import numpy as np
    pack = s312[variant]
    domains = list(pack["domains"])
    mats = []
    for m, blob in pack["all_models"].items():
        if list(blob["domains"]) != domains:
            order = [blob["domains"].index(d) for d in domains]
            mat = np.asarray(blob["matrix"], dtype=np.float64)
            mat = mat[np.ix_(order, order)]
        else:
            mat = np.asarray(blob["matrix"], dtype=np.float64)
        mats.append(mat)
    stack = np.stack(mats, axis=0)
    return {
        "domains":    domains,
        "matrix":     stack.mean(axis=0).tolist(),
        "matrix_min": stack.min(axis=0).tolist(),
        "matrix_max": stack.max(axis=0).tolist(),
        "n_models":   int(stack.shape[0]),
    }


def load_section_313() -> dict:
    """§3.1.3 RSA — 17 model pairs (3 WEIRD + 3 Sinic + 9 cross + 2 bilingual).

    Returns:
        {
          "bare": {
            "summary": {mean_rho_within_weird, mean_rho_within_sinic,
                        mean_rho_cross_tradition, delta_rho_symmetric,
                        mean_rho_within_bilingual},
            "pairs":   [17 × {model_a, model_b, group, rho, r_squared,
                              p_value, ci_low, ci_high}],
          },
          "attested": {... same shape ...},
          "meta":     {... experiment_1 meta ...}
        }
    """
    bare = _load_results("bare")
    att = _load_results("attested")
    return {
        "bare": {
            "summary": dict(bare["section_313"]["summary"]),
            "pairs":   _flatten_section_313(bare["section_313"]),
        },
        "attested": {
            "summary": dict(att["section_313"]["summary"]),
            "pairs":   _flatten_section_313(att["section_313"]),
        },
        "meta": dict(bare["meta"]),
    }


def load_section_314() -> dict:
    """§3.1.4 categorical probe (pool-independent, linked from run #3).

    Returns the raw JSON: meta + tests. Each test entry contains
    categories_en/zh, expected_break_*, expected_gap_index, per_model
    (per template + ensemble), and a summary.
    """
    path = BARE_DIR / "categorical_probe.json"
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    tests = {}
    for tid, t in raw.get("tests", {}).items():
        per_model = {}
        for m, mblob in t.get("per_model", {}).items():
            per_model[m] = {
                "label":    mblob.get("label", m),
                "lang":     mblob.get("lang"),
                "ensemble": dict(mblob.get("ensemble", {})),
            }
        tests[tid] = {
            "label":                  t.get("label"),
            "polarity":               t.get("polarity"),
            "legal_threshold":        t.get("legal_threshold"),
            "categories_en":          list(t.get("categories_en", [])),
            "categories_zh":          list(t.get("categories_zh", [])),
            "templates_en":           list(t.get("templates_en", [])),
            "templates_zh":           list(t.get("templates_zh", [])),
            "expected_break_en":      list(t.get("expected_break_en", [])),
            "expected_break_zh":      list(t.get("expected_break_zh", [])),
            "expected_gap_index":     t.get("expected_gap_index"),
            "distance_from_midpoint": t.get("distance_from_midpoint"),
            "borderline":             bool(t.get("borderline", False)),
            "borderline_note":        t.get("borderline_note"),
            "per_model":              per_model,
            "summary":                dict(t.get("summary", {})),
        }
    return {"meta": dict(raw.get("meta", {})), "tests": tests}


# --------------------------------------------------------------------------
# Bulk loader.

def load_all() -> dict:
    """Load every §3.1 block in one call (used by `pages/experiment_31.build`)."""
    return {
        "s311": load_section_311(),
        "s312": load_section_312(),
        "s313": load_section_313(),
        "s314": load_section_314(),
    }


# --------------------------------------------------------------------------
# Group ordering for legends.

GROUP_ORDER = ("within_weird", "within_sinic", "within_bilingual", "cross")

GROUP_LABEL = {
    "within_weird":     "within-WEIRD",
    "within_sinic":     "within-Sinic",
    "within_bilingual": "within-bilingual (β control)",
    "cross":            "cross-tradition",
}
