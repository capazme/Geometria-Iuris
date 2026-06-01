"""Loaders for the frozen input snapshots used by the lexicon page.

Reads the input snapshots under `experiments/ch3-measurability/inputs/`
and reshapes them into compact dicts ready for HTML rendering. None of
these returns the raw embeddings or the heavy per-context payloads:
the loaders truncate contexts to a readable length and cap the number
of samples per term, so the resulting HTML stays in the tens of KB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DASH_ROOT = _HERE.parent
_REPO_ROOT = _DASH_ROOT.parent.parent

INPUTS_DIR = _REPO_ROOT / "experiments" / "ch3-measurability" / "inputs"


_DOMAIN_ORDER = [
    "administrative",
    "civil",
    "constitutional",
    "criminal",
    "international",
    "labor_social",
    "procedure",
]

_DOMAIN_LABEL = {
    "administrative": "Administrative",
    "civil":          "Civil",
    "constitutional": "Constitutional",
    "criminal":       "Criminal",
    "international":  "International",
    "labor_social":   "Labour &amp; social",
    "procedure":      "Procedure",
}


def domain_label(d: str) -> str:
    return _DOMAIN_LABEL.get(d, d.replace("_", " ").title())


# --------------------------------------------------------------------------
# Core terms (364)

def load_core_terms() -> list[dict]:
    """Return the 364 curated legal terms in their snapshot order."""
    with (INPUTS_DIR / "core_terms_snapshot.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("terms", []))


def core_terms_by_domain() -> dict[str, list[dict]]:
    """Group the 364 terms by domain, preserving canonical domain order."""
    terms = load_core_terms()
    out: dict[str, list[dict]] = {d: [] for d in _DOMAIN_ORDER}
    for t in terms:
        d = t.get("domain", "")
        out.setdefault(d, []).append(t)
    return out


# --------------------------------------------------------------------------
# Control terms (100 Swadesh)

def load_control_terms() -> list[dict]:
    """Return the 100 everyday-language control terms."""
    with (INPUTS_DIR / "control_terms_snapshot.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("terms", []))


# --------------------------------------------------------------------------
# Term contexts (real ordinance passages)

def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def load_term_contexts(max_en: int = 2, max_zh: int = 2,
                        ctx_len_en: int = 320,
                        ctx_len_zh: int = 160) -> dict[str, dict]:
    """Return {term_en_lowercase → {term_en, term_zh, en[], zh[]}}.

    Each entry keeps the `max_en` longest English contexts and `max_zh`
    longest Chinese contexts, each truncated to its respective length.
    Preferring longer contexts produces more legible passages — some
    matches in the JSONL collapse to "…term…" when the surrounding
    window was already narrow.
    """
    path = INPUTS_DIR / "term_contexts_bilingual_snapshot.jsonl"
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            term_en = d.get("term_en") or ""
            if not term_en:
                continue
            key = term_en.lower()

            def _pick(contexts, n: int, length: int):
                pool = [c for c in contexts if isinstance(c, dict)]
                pool.sort(key=lambda c: -len((c.get("context") or "").strip()))
                picked = []
                for c in pool[:n]:
                    picked.append({
                        "cap":     c.get("cap"),
                        "year":    c.get("cap_year"),
                        "section": c.get("section_id"),
                        "text":    _truncate(c.get("context", ""), length),
                    })
                return picked

            en_ctx = _pick(d.get("en_contexts", []), max_en, ctx_len_en)
            zh_ctx = _pick(d.get("zh_contexts", []), max_zh, ctx_len_zh)
            out[key] = {
                "term_en": term_en,
                "term_zh": d.get("term_zh") or "",
                "domain":  d.get("domain") or "",
                "k_en":    d.get("k_en_postBLP"),
                "k_zh":    d.get("k_zh_postBLP"),
                "en":      en_ctx,
                "zh":      zh_ctx,
            }
    return out


# --------------------------------------------------------------------------
# Value axes — antonym pairs

_AXES_ORDER = (
    "individual_collective",
    "rights_duties",
    "public_private",
    "state_market",
    "natural_positive",
    "status_contract",
)


def _parse_yaml_simple(text: str) -> dict:
    """Minimal YAML parser for the value_axes_snapshot.yaml schema.

    The file is deterministic: top-level keys (axis names) each contain
    `en_pairs:` and `zh_pairs:` which are lists of `[pos, neg]` pairs.
    Falls back to PyYAML if available.
    """
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    # Hand-rolled fallback (only handles the exact schema we ship).
    out: dict = {}
    current_axis: str | None = None
    current_list: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            name = line.rstrip(":").strip()
            out[name] = {"en_pairs": [], "zh_pairs": []}
            current_axis = name
            current_list = None
        elif line.startswith("  ") and not line.startswith("    "):
            key = line.strip().rstrip(":")
            current_list = key
        elif line.startswith("    -") and current_axis and current_list:
            entry = line.strip().lstrip("-").strip()
            if entry.startswith("[") and entry.endswith("]"):
                pair = [p.strip().strip("'\"") for p in entry[1:-1].split(",")]
                if len(pair) == 2:
                    out[current_axis][current_list].append(pair)
    return out


def load_value_axes() -> dict[str, dict]:
    """Return {axis → {en_pairs: [[pos, neg], ...], zh_pairs: [...]}}."""
    path = INPUTS_DIR / "value_axes_snapshot.yaml"
    with path.open(encoding="utf-8") as f:
        raw = _parse_yaml_simple(f.read())
    return {a: raw[a] for a in _AXES_ORDER if a in raw}


# --------------------------------------------------------------------------
# Categorical probes — categories + templates

def load_probe_inputs() -> dict[str, dict]:
    """Return per-test {label, categories_en, categories_zh, templates_en,
    templates_zh, expected_break_en, expected_break_zh, expected_gap_index,
    polarity}.
    """
    path = (_REPO_ROOT / "experiments" / "ch3-measurability"
            / "experiment_1_structure" / "results_bare"
            / "categorical_probe.json")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict] = {}
    for tid, t in raw.get("tests", {}).items():
        out[tid] = {
            "label":             t.get("label", tid),
            "polarity":          t.get("polarity"),
            "legal_threshold":   t.get("legal_threshold"),
            "categories_en":     list(t.get("categories_en", [])),
            "categories_zh":     list(t.get("categories_zh", [])),
            "templates_en":      list(t.get("templates_en", [])),
            "templates_zh":      list(t.get("templates_zh", [])),
            "expected_break_en": list(t.get("expected_break_en", [])),
            "expected_break_zh": list(t.get("expected_break_zh", [])),
            "expected_gap_index": t.get("expected_gap_index"),
            "distance_from_midpoint": t.get("distance_from_midpoint"),
            "borderline":        bool(t.get("borderline", False)),
        }
    return out
