"""
Lens I §3.1.5 — Categorical parametric probe (pre-registered rebuild).

This is the Phase 0 rebuild of the original §3.1.5 categorical probe,
redesigned in response to the 2026-04-11 adversarial review which
identified two structural defects in the earlier version:

  1. **Midpoint artefact** on 6-category tests (old Tests 4 and 5):
     with only 6 categories, the PC1 max-gap falls at the linguistic
     midpoint by construction, and a legal threshold that happens to
     coincide with the midpoint cannot be distinguished from the
     artefact.

  2. **Post-hoc tuning** of the expected break position for Test 3
     (HARKing): the original run moved the expected position after
     seeing the first results.

The rebuilt probe fixes both defects:

  - **All 5 tests use 11 categories**. Midpoint gap index for an 11-cat
    sequence is 4. Positive tests require their expected legal break
    to be at a gap index with distance >= 2 from the midpoint, so the
    PC1 max-gap heuristic is not confounded with the midpoint by
    construction.

  - **All test specifications are pre-registered** in
    ``categorical_probe_expected.yaml`` and loaded at runtime. The
    probe script itself contains no hardcoded expected positions. The
    YAML is committed to git before the probe runs, producing an
    auditable pre-registration.

The probe runs, for each of the 5 pre-registered tests, the 3+3 model
design on the EN and ZH templated sequences. Per (model, test) the
output is:

  - Spearman ρ(category index, PC1 projection) for each of the 5
    paraphrase templates, reported as ensemble mean ± SD
  - Modal max-gap position across the 5 paraphrases, with its frequency
  - Exact-hit: modal position == pre-registered expected_gap_index
  - Near-hit (±1): |modal - expected| <= 1

Output JSON and HTML are strictly descriptive (algorithm + metric
definitions + raw numbers). No "perfect validation" language; no
interpretation of the results. The thesis chapter is where readings
happen.

References
----------
Sclar, M., et al. (2024). "Quantifying Language Models' Sensitivity to
  Spurious Features in Prompt Design." ICLR.
Wallace, E., Wang, Y., Li, S., Singh, S., & Gardner, M. (2019). "Do NLP
  Models Know Numbers? Probing Numeracy in Embeddings." EMNLP.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.embeddings import EmbeddingClient  # noqa: E402
from shared.html_style import (  # noqa: E402
    CSS, HEAD_LINKS, C_BLUE, C_VERMIL, C_GREEN, page_head,
)


EXPECTED_YAML = Path(__file__).parent / "categorical_probe_expected.yaml"

WEIRD_MODELS = [
    ("BAAI/bge-large-en-v1.5", "BGE-EN-large"),
    ("intfloat/e5-large-v2", "E5-large"),
    ("freelawproject/modernbert-embed-base_finetune_512", "FreeLaw-EN"),
    # Bilingual models on the EN side (each is used with EN templates only in
    # this slot, in parallel with its ZH-side twin below).
    ("BAAI/bge-m3", "BGE-M3-EN"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B-EN"),
]
SINIC_MODELS = [
    ("BAAI/bge-large-zh-v1.5", "BGE-ZH-large"),
    ("GanymedeNil/text2vec-large-chinese", "Text2vec-large-ZH"),
    ("DMetaSoul/Dmeta-embedding-zh", "Dmeta-ZH"),
    # Bilingual models on the ZH side.
    ("BAAI/bge-m3", "BGE-M3-ZH"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B-ZH"),
]


# ---------------------------------------------------------------------------
# Pre-registration loader
# ---------------------------------------------------------------------------

def load_pre_registered_tests() -> dict:
    """
    Load the pre-registered test specifications from the committed YAML.

    The probe script MUST NOT modify these specifications at runtime.
    Any amendment must go through an explicit edit of the YAML file and
    be documented in the trace.
    """
    if not EXPECTED_YAML.exists():
        raise FileNotFoundError(
            f"Pre-registration YAML not found: {EXPECTED_YAML}. "
            "This file must be committed to git before the probe runs."
        )
    with EXPECTED_YAML.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Validate the midpoint-distance constraint for positive tests
    meta = data["meta"]
    midpoint = meta["midpoint_gap_index"]
    distance_threshold = meta["distance_threshold"]
    for test_id, spec in data["tests"].items():
        if spec["polarity"] != "positive":
            continue
        dist = spec.get("distance_from_midpoint")
        if dist is None:
            continue
        if dist < distance_threshold and not spec.get("borderline", False):
            raise ValueError(
                f"Test {test_id} is positive with distance_from_midpoint={dist} "
                f"< threshold {distance_threshold}, but is not marked borderline. "
                "This violates the pre-registration design constraint."
            )

    return data


# ---------------------------------------------------------------------------
# Per-template analytic procedure
# ---------------------------------------------------------------------------

def project_onto_pc1_signed(vectors: np.ndarray) -> np.ndarray:
    """
    Project a sequence of vectors onto its own first principal component,
    with the sign of the PC chosen so that the projection is, on average,
    monotonically increasing with the sequence index.
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    proj = centered @ pc1
    idx = np.arange(len(proj))
    rho, _ = spearmanr(idx, proj)
    if rho is not None and not np.isnan(rho) and rho < 0:
        proj = -proj
    return proj.astype(np.float32)


def consecutive_pc1_gaps(pc1_proj: np.ndarray) -> np.ndarray:
    """Absolute differences between consecutive PC1 projections, range-normalised."""
    diffs = np.abs(np.diff(pc1_proj))
    total_range = float(pc1_proj.max() - pc1_proj.min())
    if total_range < 1e-12:
        return np.zeros_like(diffs)
    return (diffs / total_range).astype(np.float32)


def analyse_one_template(
    client: EmbeddingClient,
    model_id: str,
    categories: list[str],
    template: str,
) -> dict:
    """Encode the templated category sequence and compute PC1 + max-gap metrics."""
    texts = [template.format(category=c) for c in categories]
    vecs = client.embed(texts, model_id, use_cache=True).astype(np.float32)

    # Defensive renormalisation
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-12, None)

    pc1_proj = project_onto_pc1_signed(vecs)
    indices = np.arange(len(categories))
    rho, pvalue = spearmanr(indices, pc1_proj)
    gaps = consecutive_pc1_gaps(pc1_proj)

    if len(gaps) == 0:
        max_gap = 0.0
        max_gap_position = -1
    else:
        max_gap = float(gaps.max())
        max_gap_position = int(np.argmax(gaps))

    return {
        "template": template,
        "spearman_rho": float(rho) if rho is not None else float("nan"),
        "spearman_pvalue": float(pvalue) if pvalue is not None else float("nan"),
        "pc1_projection": pc1_proj.tolist(),
        "consecutive_gaps_normalised": gaps.tolist(),
        "max_gap": max_gap,
        "max_gap_position": max_gap_position,
        "max_gap_between": (
            [categories[max_gap_position], categories[max_gap_position + 1]]
            if max_gap_position >= 0
            else []
        ),
    }


# ---------------------------------------------------------------------------
# Per-model aggregation across paraphrase templates
# ---------------------------------------------------------------------------

def run_test_for_model(
    client: EmbeddingClient,
    model_id: str,
    label: str,
    lang: str,
    categories: list[str],
    templates: list[str],
    expected_gap_index: int,
) -> dict:
    """Run the probe on every template paraphrase for one model and aggregate."""
    per_template = []
    for ti, tpl in enumerate(templates):
        r = analyse_one_template(client, model_id, categories, tpl)
        r["template_index"] = ti
        per_template.append(r)

    rhos = np.array(
        [r["spearman_rho"] for r in per_template], dtype=float
    )
    max_gaps = np.array(
        [r["max_gap"] for r in per_template], dtype=float
    )
    max_gap_positions = [r["max_gap_position"] for r in per_template]
    pos_counter = Counter(max_gap_positions)
    modal_pos, modal_freq = pos_counter.most_common(1)[0]

    n_at_expected = (
        sum(1 for p in max_gap_positions if p == expected_gap_index)
        if expected_gap_index >= 0 else 0
    )
    n_within_one = (
        sum(1 for p in max_gap_positions if abs(p - expected_gap_index) <= 1)
        if expected_gap_index >= 0 else 0
    )

    modal_is_exact = (expected_gap_index >= 0 and modal_pos == expected_gap_index)
    modal_is_near = (expected_gap_index >= 0 and abs(modal_pos - expected_gap_index) <= 1)

    return {
        "model_id": model_id,
        "label": label,
        "lang": lang,
        "n_templates": len(templates),
        "per_template": per_template,
        "ensemble": {
            "mean_rho": float(np.nanmean(rhos)),
            "std_rho": float(np.nanstd(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
            "min_rho": float(np.nanmin(rhos)),
            "max_rho": float(np.nanmax(rhos)),
            "mean_max_gap": float(np.nanmean(max_gaps)),
            "modal_max_gap_position": int(modal_pos),
            "modal_max_gap_freq": int(modal_freq),
            "modal_max_gap_between": (
                [categories[modal_pos], categories[modal_pos + 1]]
                if 0 <= modal_pos < len(categories) - 1
                else []
            ),
            "expected_gap_index": int(expected_gap_index),
            "n_templates_at_expected": int(n_at_expected),
            "n_templates_within_one": int(n_within_one),
            "modal_is_exact": bool(modal_is_exact),
            "modal_is_near": bool(modal_is_near),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data = load_pre_registered_tests()
    meta = data["meta"]
    tests = data["tests"]
    print(f"[probe] Pre-registration loaded from {EXPECTED_YAML}")
    print(f"[probe]   date: {meta['date']}")
    print(f"[probe]   n_tests: {len(tests)}  n_cats: {meta['n_categories_per_test']}"
          f"  midpoint gap: {meta['midpoint_gap_index']}")

    config_path = REPO_ROOT / "models" / "config.yaml"
    client = EmbeddingClient(str(config_path))

    report = {
        "meta": {
            "module": "lens_1_relational/categorical_probe.py",
            "thesis_section": "§3.1.5 — Parametric validation (pre-registered rebuild)",
            "date": datetime.now().isoformat(timespec="seconds"),
            "weird_models": [m[1] for m in WEIRD_MODELS],
            "sinic_models": [m[1] for m in SINIC_MODELS],
            "pre_registration_file": str(EXPECTED_YAML.relative_to(REPO_ROOT)),
            "pre_registration_meta": meta,
            "n_tests": len(tests),
        },
        "tests": {},
    }

    for test_id, spec in tests.items():
        print(f"\n[probe] === {test_id}  ({spec['label']}) ===")
        print(f"[probe]   polarity: {spec['polarity']}")
        if spec["polarity"] == "positive":
            print(f"[probe]   expected break (EN): {spec['expected_break_en']} "
                  f"at gap idx {spec['expected_gap_index']} "
                  f"(distance {spec['distance_from_midpoint']} from midpoint)")
            if spec.get("borderline"):
                print(f"[probe]   BORDERLINE: {spec.get('borderline_note', '')[:120]}...")

        per_model: dict[str, dict] = {}

        for model_id, label in WEIRD_MODELS:
            print(f"[probe]   WEIRD: {label}")
            r = run_test_for_model(
                client, model_id, label, "en",
                spec["categories_en"], spec["templates_en"],
                expected_gap_index=spec["expected_gap_index"],
            )
            per_model[label] = r
            ens = r["ensemble"]
            hit = "exact" if ens["modal_is_exact"] else ("near" if ens["modal_is_near"] else "—")
            print(
                f"    rho ensemble = {ens['mean_rho']:+.3f} ± {ens['std_rho']:.3f}  "
                f"modal break pos = {ens['modal_max_gap_position']} "
                f"(freq {ens['modal_max_gap_freq']}/{r['n_templates']}) "
                f"between {ens['modal_max_gap_between']}  [{hit}]"
            )

        for model_id, label in SINIC_MODELS:
            print(f"[probe]   Sinic: {label}")
            r = run_test_for_model(
                client, model_id, label, "zh",
                spec["categories_zh"], spec["templates_zh"],
                expected_gap_index=spec["expected_gap_index"],
            )
            per_model[label] = r
            ens = r["ensemble"]
            hit = "exact" if ens["modal_is_exact"] else ("near" if ens["modal_is_near"] else "—")
            print(
                f"    rho ensemble = {ens['mean_rho']:+.3f} ± {ens['std_rho']:.3f}  "
                f"modal break pos = {ens['modal_max_gap_position']} "
                f"(freq {ens['modal_max_gap_freq']}/{r['n_templates']}) "
                f"between {ens['modal_max_gap_between']}  [{hit}]"
            )

        # Aggregate across models
        all_rhos = [per_model[lbl]["ensemble"]["mean_rho"] for lbl in per_model]
        all_max_gaps = [per_model[lbl]["ensemble"]["mean_max_gap"] for lbl in per_model]
        n_exact = sum(1 for lbl in per_model if per_model[lbl]["ensemble"]["modal_is_exact"])
        n_near = sum(1 for lbl in per_model if per_model[lbl]["ensemble"]["modal_is_near"])

        report["tests"][test_id] = {
            "label": spec["label"],
            "polarity": spec["polarity"],
            "legal_threshold": spec.get("legal_threshold"),
            "categories_en": spec["categories_en"],
            "categories_zh": spec["categories_zh"],
            "templates_en": spec["templates_en"],
            "templates_zh": spec["templates_zh"],
            "expected_break_en": spec.get("expected_break_en", []),
            "expected_break_zh": spec.get("expected_break_zh", []),
            "expected_gap_index": int(spec["expected_gap_index"]),
            "distance_from_midpoint": spec.get("distance_from_midpoint"),
            "borderline": spec.get("borderline", False),
            "borderline_note": spec.get("borderline_note"),
            "per_model": per_model,
            "summary": {
                "mean_ensemble_rho": float(np.nanmean(all_rhos)),
                "median_ensemble_rho": float(np.nanmedian(all_rhos)),
                "mean_ensemble_max_gap": float(np.nanmean(all_max_gaps)),
                "n_models_exact_hit": int(n_exact),
                "n_models_near_hit": int(n_near),
                "n_models_total": len(per_model),
            },
        }

    out_path = REPO_ROOT / "lens_1_relational" / "results" / "categorical_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_nan_to_none(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[probe] Report → {out_path}")

    _render_html(report)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _nan_to_none(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_nan_to_none(v) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# HTML dashboard (descriptive-only: algorithms + metrics + raw results)
# ---------------------------------------------------------------------------

def _fmt(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "<i>n/a</i>"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _render_html(report: dict) -> None:
    out_dir = Path(__file__).parent / "results" / "figures" / "html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "categorical_probe.html"

    head = page_head("§3.1.5 — Parametric probe (pre-registered)")

    test_cards = []
    for tid, t in report["tests"].items():
        polarity_color = C_GREEN if t["polarity"] == "positive" else C_VERMIL
        polarity_label = "POSITIVE" if t["polarity"] == "positive" else "NEGATIVE CONTROL"

        # Per-model rows
        rows = []
        for label, m in t["per_model"].items():
            ens = m["ensemble"]
            tradition_class = "rsa-weird" if m["lang"] == "en" else "rsa-sinic"
            modal_between = ens["modal_max_gap_between"]
            modal_text = (
                f"{modal_between[0]} → {modal_between[1]}"
                if len(modal_between) == 2 else "—"
            )

            hit_marker = ""
            if t["polarity"] == "positive":
                if ens["modal_is_exact"]:
                    hit_marker = ' style="background:#d8f0d8;"'
                elif ens["modal_is_near"]:
                    hit_marker = ' style="background:#f0f8e8;"'

            rows.append(
                f'<tr class="{tradition_class}"{hit_marker}>'
                f"<td><b>{label}</b></td>"
                f"<td>{_fmt(ens['mean_rho'])} ± {ens['std_rho']:.3f}</td>"
                f"<td>{_fmt(ens['min_rho'])} … {_fmt(ens['max_rho'])}</td>"
                f"<td>{_fmt(ens['mean_max_gap'], signed=False)}</td>"
                f"<td>{ens['modal_max_gap_position']}</td>"
                f"<td>{modal_text}</td>"
                f"<td>{ens['modal_max_gap_freq']}/{m['n_templates']}</td>"
                f"</tr>"
            )

        models_table = (
            '<table class="data">'
            "<thead><tr><th>Model</th><th>ensemble ρ̄ ± SD</th>"
            "<th>min … max</th><th>mean max-gap</th>"
            "<th>modal gap idx</th><th>modal break (categories)</th>"
            "<th>freq</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

        # Categories listing
        cats_en_html = " · ".join(f"<code>{c}</code>" for c in t["categories_en"])
        cats_zh_html = " · ".join(f"<code>{c}</code>" for c in t["categories_zh"])

        # Expected break block
        if t["polarity"] == "positive":
            dist = t.get("distance_from_midpoint")
            dist_str = f"distance {dist} from midpoint (gap idx 4)" if dist is not None else ""
            expected_html = (
                f'<div class="method"><b>Pre-registered expected break</b>: '
                f"EN <code>{' → '.join(t['expected_break_en'])}</code>; "
                f"ZH <code>{' → '.join(t['expected_break_zh'])}</code>; "
                f"gap index <code>{t['expected_gap_index']}</code> ({dist_str}).</div>"
            )
            if t.get("borderline") and t.get("borderline_note"):
                expected_html += (
                    f'<div class="method" style="border-left-color:#888;">'
                    f"<b>Borderline note</b>: {t['borderline_note']}</div>"
                )
        else:
            expected_html = (
                '<div class="method"><b>Negative control</b>: '
                "no legal threshold exists for this sequence. Expected break "
                "position is <code>none</code>.</div>"
            )

        s = t["summary"]
        summary_line = ""
        if t["polarity"] == "positive":
            summary_line = (
                f'<p style="font-size:0.85rem; color:#444; margin-top:10px;">'
                f"Exact modal hit: <b>{s['n_models_exact_hit']}/{s['n_models_total']}</b>; "
                f"near-hit (±1): <b>{s['n_models_near_hit']}/{s['n_models_total']}</b>; "
                f"mean ensemble ρ̄: <b>{_fmt(s['mean_ensemble_rho'])}</b>; "
                f"mean ensemble max-gap: <b>{_fmt(s['mean_ensemble_max_gap'], signed=False)}</b>."
                "</p>"
            )
        else:
            summary_line = (
                f'<p style="font-size:0.85rem; color:#444; margin-top:10px;">'
                f"Mean ensemble ρ̄ across 6 models: <b>{_fmt(s['mean_ensemble_rho'])}</b>; "
                f"mean ensemble max-gap: <b>{_fmt(s['mean_ensemble_max_gap'], signed=False)}</b>."
                "</p>"
            )

        test_cards.append(f"""
<div class="card" id="{tid}">
  <h2>
    <span style="color:{polarity_color};font-size:0.78rem;text-transform:uppercase;
                 letter-spacing:0.05em;">{polarity_label}</span><br>
    {tid} &mdash; {t['label']}
  </h2>
  {expected_html}
  <p style="font-size:0.84rem; color:#444;">
    <b>Categories (EN)</b>: {cats_en_html}<br>
    <b>Categories (ZH)</b>: {cats_zh_html}
  </p>
  <p style="font-size:0.84rem; color:#444;">
    <b>Paraphrase templates</b>: 5 per language, listed in the
    pre-registration YAML. Each template is encoded for each model and
    the aggregate statistics across paraphrases are reported below.
  </p>
  {models_table}
  {summary_line}
</div>""")

    meta = report["meta"]
    pre_meta = meta.get("pre_registration_meta", {})

    body = f"""<body>
<h1>§3.1.5 &mdash; Parametric validation of the §3.1 instrument</h1>
<p class="subtitle">Pre-registered categorical probe. Five tests, each
using 11 ordinal categories, encoded through 5 paraphrase templates per
language on the 3+3 model design. Expected break positions are pre-registered
in <code>categorical_probe_expected.yaml</code> (committed to git before this
run). The probe reports per-model modal break position vs the pre-registered
expectation and computes exact-hit and ±1 near-hit counts.</p>

<div class="card">
  <h2>Algorithm</h2>
  <div class="method">
    <b>Step 1 — Encoding</b>: for each pre-registered test, each of the
    11 categories is inserted into each of 5 paraphrase templates and
    encoded with each of the 6 models (3 WEIRD on EN templates, 3 Sinic
    on ZH templates). Per (test, model, template) the output is an
    11 × dim matrix of L2-normalised vectors.
  </div>
  <div class="method">
    <b>Step 2 — PC1 projection</b>: the 11-vector cloud is centred and
    the first right-singular vector of the centred matrix is extracted
    via SVD. Each vector is projected onto this PC1. The sign of the PC
    is flipped if the resulting projection has negative Spearman ρ with
    the sequence index, so the projection is monotonically increasing
    with category order by construction (sign is not informative).
  </div>
  <div class="method">
    <b>Step 3 — Ordinal monotonicity</b>: Spearman ρ between the category
    ordinal index (0..10) and the PC1 projection. Reported per template;
    aggregated across the 5 paraphrase templates as mean ± SD.
  </div>
  <div class="method">
    <b>Step 4 — Max-gap position</b>: the 10 consecutive differences of
    the PC1 projection are range-normalised; the position of the largest
    gap is recorded. Aggregated across the 5 paraphrase templates as
    the modal (most frequent) position with its frequency count.
  </div>
  <div class="method">
    <b>Step 5 — Hit classification</b>: for positive tests, the
    per-model modal gap position is compared to the pre-registered
    expected gap index. <code>modal_is_exact</code> iff equal;
    <code>modal_is_near</code> iff |modal − expected| ≤ 1.
  </div>
  <div class="method">
    <b>Design constraint</b>: all 11-category sequences have midpoint
    gap index 4 (between elements 4 and 5). Positive tests are
    pre-registered to have expected gap index at distance ≥ 2 from the
    midpoint, so a modal break at the expected position cannot be
    attributed to the generic PC1-midpoint effect. Tests with distance
    &lt; 2 are flagged <code>borderline</code> and their interpretation
    is restricted.
  </div>
</div>

<div class="card">
  <h2>Metric definitions</h2>
  <div class="method">
    <b>Spearman ρ</b>: rank correlation between two variables, invariant
    to monotone rescaling. Range [−1, +1]. +1 means the two rank orders
    agree exactly.
  </div>
  <div class="method">
    <b>PC1 projection</b>: scalar coordinate of each vector along the
    first principal component of the cloud. With 11 points, PC1 captures
    the dominant direction of variation.
  </div>
  <div class="method">
    <b>max-gap position</b>: integer index in [0, 9] identifying which
    consecutive pair of categories is furthest apart in the PC1
    projection. Range-normalised to [0, 1] for reporting.
  </div>
  <div class="method">
    <b>modal_max_gap_freq</b>: number of paraphrase templates (out of 5)
    for which the max-gap falls at the same position. Higher values
    indicate the result is robust to templating variation.
  </div>
</div>

<div class="card">
  <h2>Pre-registration metadata</h2>
  <p style="font-size:0.85rem;">
    Pre-registration file:
    <code>{meta.get('pre_registration_file', '')}</code><br>
    Date: <code>{pre_meta.get('date', '')}</code><br>
    n_categories_per_test: <code>{pre_meta.get('n_categories_per_test', '')}</code><br>
    midpoint gap index: <code>{pre_meta.get('midpoint_gap_index', '')}</code><br>
    distance threshold (min distance from midpoint): <code>{pre_meta.get('distance_threshold', '')}</code><br>
    n_paraphrase_templates: <code>{pre_meta.get('n_paraphrase_templates', '')}</code><br>
    pre_registered_by: <code>{pre_meta.get('pre_registered_by', '')}</code>
  </p>
</div>

{''.join(test_cards)}
</body>"""

    html = f"<!DOCTYPE html>\n<html lang='en'>\n{head}\n{body}\n</html>"
    out_path.write_text(html, encoding="utf-8")
    print(f"[probe] Dashboard → {out_path}")


if __name__ == "__main__":
    main()
