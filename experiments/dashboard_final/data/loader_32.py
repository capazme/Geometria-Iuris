"""Loader for Experiment 2 (§3.2) — value axes alignment.

Reads `experiments/ch3-measurability/experiment_2_axes/results_{bare,attested}/`:
  - `experiment_2_results.json`     §3.2.1 sanity, §3.2.2 orthogonality,
                                    §3.2.3 per-pair ρ on 6 axes,
                                    §3.2.4 cross-tradition ranking,
                                    §3.2.5 (bare) or extension hooks

Axes layout (in the JSON `meta.axes`, used as canonical order):
    individual_collective, rights_duties, public_private,
    state_market, natural_positive, status_contract

Schema produced by `load_all()` documented at end of module.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CH3 = REPO_ROOT / "experiments" / "ch3-measurability" / "experiment_2_axes"
BARE_DIR = CH3 / "results_bare"
ATT_DIR = CH3 / "results_attested"


# --------------------------------------------------------------------------
# Model + axis constants.

WEIRD_MODELS = (
    "BGE-EN-large", "E5-large", "FreeLaw-EN",
    "BGE-M3-EN", "Qwen3-0.6B-EN",
)
SINIC_MODELS = (
    "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH",
    "BGE-M3-ZH", "Qwen3-0.6B-ZH",
)
BILINGUAL_BASES = ("BGE-M3", "Qwen3-0.6B")

AXES_ORDER = (
    "individual_collective",
    "rights_duties",
    "public_private",
    "state_market",
    "natural_positive",
    "status_contract",
)

AXIS_LABELS = {
    "individual_collective": "individual ↔ collective",
    "rights_duties":         "rights ↔ duties",
    "public_private":        "public ↔ private",
    "state_market":          "state ↔ market",
    "natural_positive":      "natural ↔ positive",
    "status_contract":       "status ↔ contract",
}


def classify_pair(model_a: str, model_b: str, group_from_lens: str) -> str:
    """Reclassify the bilingual same-encoder pair as `within_bilingual`."""
    for base in BILINGUAL_BASES:
        if {model_a, model_b} == {f"{base}-EN", f"{base}-ZH"}:
            return "within_bilingual"
    return group_from_lens


# --------------------------------------------------------------------------
# JSON readers.

def _load_results(variant: str) -> dict:
    """variant ∈ {'bare', 'attested'}."""
    base = BARE_DIR if variant == "bare" else ATT_DIR
    with (base / "experiment_2_results.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _enrich_per_pair(per_pair: list[dict]) -> list[dict]:
    """Apply `classify_pair` so the bilingual pair appears as its own group."""
    out = []
    for p in per_pair:
        e = dict(p)
        e["group"] = classify_pair(p["model_a"], p["model_b"], p.get("group", ""))
        out.append(e)
    return out


# --------------------------------------------------------------------------
# Section loaders.

def load_section_321() -> dict:
    """§3.2.1 axis sanity (per model × axis: n_pairs_used / positive_correct / …)."""
    bare = _load_results("bare")
    att = _load_results("attested")
    return {
        "bare":     dict(bare["section_321"]),
        "attested": dict(att["section_321"]),
        "meta":     dict(bare["meta"]),
    }


def load_section_322() -> dict:
    """§3.2.2 axes independence — inter-axis cosine per model."""
    bare = _load_results("bare")
    att = _load_results("attested")
    return {
        "bare":     dict(bare["section_322"]),
        "attested": dict(att["section_322"]),
        "meta":     dict(bare["meta"]),
    }


def load_section_323() -> dict:
    """§3.2.3 per-pair ρ on 6 axes (45 pairs × 6 axes = 270 entries)."""
    bare = _load_results("bare")
    att = _load_results("attested")
    return {
        "bare":     {"per_pair": _enrich_per_pair(bare["section_323"]["per_pair"])},
        "attested": {"per_pair": _enrich_per_pair(att["section_323"]["per_pair"])},
        "meta":     dict(bare["meta"]),
    }


def load_section_324() -> dict:
    """§3.2.4 cross-tradition divergence ranking per axis.

    Returns:
        {
          "bare": {
            "cross_rho_mean_per_axis": {<axis>: float},
            "ranking_most_divergent_first": [<axis>, ...]
          },
          "attested": {... same shape ...},
          "meta": {...}
        }
    """
    bare = _load_results("bare")
    att = _load_results("attested")
    return {
        "bare":     dict(bare["section_324"]),
        "attested": dict(att["section_324"]),
        "meta":     dict(bare["meta"]),
    }


def load_section_325() -> dict:
    """§3.2.5 between-group differences (top divergent terms per axis).

    Schema depends on the experiment script — typically a dict of axis →
    list of {term, w_mean, s_mean, gap}. The dashboard treats it as opaque
    and renders whatever is present.
    """
    bare = _load_results("bare")
    att = _load_results("attested")
    return {
        "bare":     dict(bare.get("section_325", {})),
        "attested": dict(att.get("section_325", {})),
        "meta":     dict(bare["meta"]),
    }


# --------------------------------------------------------------------------
# Helpers.

def find_pair(per_pair: list[dict], model_a: str, model_b: str,
              axis: str) -> dict | None:
    target = {model_a, model_b}
    for entry in per_pair:
        if entry["axis"] != axis:
            continue
        if {entry["model_a"], entry["model_b"]} == target:
            return entry
    return None


def per_axis_rho_distribution(per_pair: list[dict]) -> dict[str, dict]:
    """Bucket the per-pair ρ for each axis, with sub-buckets per group."""
    out: dict[str, dict] = {a: {"all": [], "groups": {}} for a in AXES_ORDER}
    for entry in per_pair:
        axis = entry["axis"]
        if axis not in out:
            continue
        out[axis]["all"].append(float(entry["rho"]))
        g = entry["group"]
        out[axis]["groups"].setdefault(g, []).append(float(entry["rho"]))
    return out


# --------------------------------------------------------------------------
# Bulk loader.

def load_all() -> dict:
    """Load every §3.2 block in one call (used by `pages/experiment_32.build`)."""
    return {
        "s321": load_section_321(),
        "s322": load_section_322(),
        "s323": load_section_323(),
        "s324": load_section_324(),
        "s325": load_section_325(),
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
