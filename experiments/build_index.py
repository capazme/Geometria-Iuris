"""
Build an index HTML page linking all Lens interactive dashboards.

Usage:
    cd experiments/
    python build_index.py
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.html_style import (
    CSS, HEAD_LINKS, C_BLUE, C_ORANGE, C_GREEN, C_VERMIL, C_SKY, C_PURPLE,
)

ROOT = Path(__file__).parent

LENSES = [
    {
        "id": "lens1",
        "number": "I",
        "title": "Relational Distance Structure",
        "section": "§3.1",
        "color": C_BLUE,
        "html": "lens_1_relational/results/figures/html/lens1_interactive.html",
        "json": "lens_1_relational/results/lens1_results.json",
        "description": (
            "Tests whether embedding models encode legal domain structure "
            "and whether the relational geometry is preserved across "
            "WEIRD and Sinic traditions (RSA on 397 core terms × 6 models)."
        ),
    },
    {
        "id": "lens3",
        "number": "III",
        "title": "Layer Stratigraphy",
        "section": "§3.1.3",
        "color": C_GREEN,
        "html": "lens_3_stratigraphy/results/figures/html/lens3_interactive.html",
        "json": "lens_3_stratigraphy/results/lens3_results.json",
        "extras": [
            ("Stratigraphy experiments",
             "lens_3_stratigraphy/results/stratigraphy_experiments.html"),
            ("NTA exploration",
             "lens_3_stratigraphy/results/nta_exploration.html"),
        ],
        "description": (
            "Measures at which transformer layer legal meaning crystallises. "
            "Tracks domain signal, representation drift, and neighborhood "
            "trajectory across all hidden layers for 6 models."
        ),
    },
    {
        "id": "lens4",
        "number": "IV",
        "title": "Value Axis Projection",
        "section": "§3.3",
        "color": C_VERMIL,
        "html": "lens_4_values/results/figures/html/lens4_interactive.html",
        "json": "lens_4_values/results/lens4_results.json",
        "description": (
            "Projects legal concepts onto three value axes "
            "(individual/collective, rights/duties, public/private) "
            "via Kozlowski difference-vectors and measures cross-tradition "
            "alignment per axis."
        ),
    },
    {
        "id": "lens5",
        "number": "V",
        "title": "Semantic Neighborhoods",
        "section": "§3.2",
        "color": C_ORANGE,
        "html": "lens_5_neighborhoods/results/figures/html/lens5_interactive.html",
        "json": "lens_5_neighborhoods/results/lens5_results.json",
        "description": (
            "Computes per-term k-NN Jaccard overlap between model pairs "
            "to identify 'false friends' and measure which legal domains "
            "diverge most across traditions."
        ),
    },
]


def _load_metrics(lens: dict) -> dict:
    """Extract key display metrics from a lens result JSON."""
    jpath = ROOT / lens["json"]
    if not jpath.exists():
        return {}
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)

    lid = lens["id"]
    metrics: dict[str, str] = {}

    if lid == "lens1":
        s = data.get("section_314", {}).get("summary", {})
        if s:
            metrics["Within-WEIRD ρ"] = f"{s['mean_rho_within_weird']:.3f}"
            metrics["Within-Sinic ρ"] = f"{s['mean_rho_within_sinic']:.3f}"
            metrics["Cross ρ"] = f"{s['mean_rho_cross']:.3f}"
            metrics["Δρ"] = f"{s['cross_tradition_drop']:.3f}"

    elif lid == "lens3":
        models = data.get("section_313b", {}).get("per_model", {})
        if models:
            r_vals = [m["domain_signal_r"][-1] for m in models.values()
                      if "domain_signal_r" in m]
            if r_vals:
                metrics["Domain r (final, mean)"] = f"{sum(r_vals)/len(r_vals):.3f}"
            metrics["Models"] = str(len(models))
            first = next(iter(models.values()), {})
            metrics["Layers"] = str(len(first.get("domain_signal_r", [])))

    elif lid == "lens4":
        s332 = data.get("section_332", {})
        if s332:
            axes = s332.get("per_axis", {})
            for ax_name, ax_data in axes.items():
                cross = ax_data.get("cross_tradition", {})
                rho = cross.get("mean_rho")
                if rho is not None:
                    short = ax_name.replace("_", "/")
                    metrics[f"Cross ρ ({short})"] = f"{rho:.3f}"

    elif lid == "lens5":
        s321 = data.get("section_321", {})
        if s321:
            metrics["Cross J (mean)"] = f"{s321.get('cross_tradition_mean_jaccard', 0):.3f}"
            metrics["Within-WEIRD J"] = f"{s321.get('within_weird_mean_jaccard', 0):.3f}"
            metrics["Within-Sinic J"] = f"{s321.get('within_sinic_mean_jaccard', 0):.3f}"

    meta = data.get("meta", {})
    date = meta.get("date", "")[:10]
    if date:
        metrics["Run date"] = date

    return metrics


def _metric_card(label: str, value: str) -> str:
    return (f'<div class="idx-metric"><span class="idx-mv">{value}</span>'
            f'<span class="idx-ml">{label}</span></div>')


def _lens_card(lens: dict) -> str:
    metrics = _load_metrics(lens)
    color = lens["color"]
    html_path = lens["html"]
    exists = (ROOT / html_path).exists()

    metrics_html = "".join(_metric_card(k, v) for k, v in metrics.items())

    extras_html = ""
    for label, path in lens.get("extras", []):
        if (ROOT / path).exists():
            extras_html += (
                f' <a class="idx-extra" href="{path}">{label}</a>'
            )

    link = (f'<a class="idx-open" href="{html_path}">Open dashboard →</a>'
            if exists else '<span class="idx-missing">Not generated yet</span>')

    return f"""
    <div class="idx-card" style="border-left-color: {color};">
      <div class="idx-header">
        <span class="idx-num" style="color: {color};">Lens {lens['number']}</span>
        <span class="idx-section">{lens['section']}</span>
      </div>
      <h2 class="idx-title">{lens['title']}</h2>
      <p class="idx-desc">{lens['description']}</p>
      <div class="idx-metrics">{metrics_html}</div>
      <div class="idx-actions">{link}{extras_html}</div>
    </div>"""


def build_index() -> Path:
    cards = "\n".join(_lens_card(l) for l in LENSES)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Geometria Iuris — Experiment Dashboard</title>
{HEAD_LINKS}
<style>
{CSS}

/* Index-specific styles */
.idx-hero {{
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: #fff; padding: 32px 28px 24px; margin: -20px -28px 24px;
  border-radius: 0 0 12px 12px;
}}
.idx-hero h1 {{ font-size: 1.6rem; margin: 0; color: #fff; }}
.idx-hero .subtitle {{ color: rgba(255,255,255,0.7); margin-bottom: 0; }}
.idx-hero-stats {{ display: flex; gap: 24px; margin-top: 16px; }}
.idx-hero-stat {{ text-align: center; }}
.idx-hero-stat .val {{ font-size: 1.8rem; font-weight: 700; }}
.idx-hero-stat .lbl {{ font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: 0.05em; opacity: 0.7; }}

.idx-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
@media (max-width: 900px) {{ .idx-grid {{ grid-template-columns: 1fr; }} }}

.idx-card {{
  background: #fff; border: 1px solid #e0e0e0; border-left: 5px solid #ccc;
  border-radius: 8px; padding: 20px 22px; transition: box-shadow 0.15s;
}}
.idx-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
.idx-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
.idx-num {{ font-weight: 700; font-size: 0.85rem; text-transform: uppercase;
            letter-spacing: 0.04em; }}
.idx-section {{ font-size: 0.78rem; color: #999; }}
.idx-title {{ font-size: 1.1rem; margin: 0 0 6px 0; color: #222; }}
.idx-desc {{ font-size: 0.84rem; color: #555; margin: 0 0 12px 0; line-height: 1.5; }}

.idx-metrics {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
.idx-metric {{
  background: #f8f8f8; border-radius: 6px; padding: 8px 12px;
  display: flex; flex-direction: column; align-items: center; min-width: 80px;
}}
.idx-mv {{ font-size: 1.05rem; font-weight: 700; color: #333; }}
.idx-ml {{ font-size: 0.68rem; color: #999; text-transform: uppercase;
           letter-spacing: 0.03em; margin-top: 2px; }}

.idx-actions {{ display: flex; gap: 10px; align-items: center; }}
.idx-open {{
  display: inline-block; padding: 7px 16px; background: #222; color: #fff;
  border-radius: 6px; text-decoration: none; font-size: 0.82rem; font-weight: 500;
  transition: background 0.15s;
}}
.idx-open:hover {{ background: #444; }}
.idx-extra {{
  display: inline-block; padding: 7px 14px; background: #f0f0f0; color: #555;
  border-radius: 6px; text-decoration: none; font-size: 0.78rem;
  transition: background 0.15s;
}}
.idx-extra:hover {{ background: #e0e0e0; color: #222; }}
.idx-missing {{ font-size: 0.82rem; color: #ccc; font-style: italic; }}

.idx-footer {{
  margin-top: 28px; padding-top: 14px; border-top: 1px solid #e0e0e0;
  font-size: 0.78rem; color: #aaa; text-align: center;
}}
</style>
</head>
<body>

<div class="idx-hero">
  <h1>Geometria Iuris</h1>
  <p class="subtitle">Measuring Legal Meaning Across Cultural Normative Structures
  in Embedding Spaces</p>
  <div class="idx-hero-stats">
    <div class="idx-hero-stat">
      <div class="val">4</div><div class="lbl">Lenses</div>
    </div>
    <div class="idx-hero-stat">
      <div class="val">6</div><div class="lbl">Models</div>
    </div>
    <div class="idx-hero-stat">
      <div class="val">9,472</div><div class="lbl">Terms</div>
    </div>
    <div class="idx-hero-stat">
      <div class="val">397</div><div class="lbl">Core terms</div>
    </div>
  </div>
</div>

<div class="idx-grid">
{cards}
</div>

<div class="idx-footer">
  Thesis: Metodologia delle Scienze Giuridiche (LUISS) &middot;
  3 WEIRD models (EN) &times; 3 Sinic models (ZH) &middot;
  Source: HK DOJ Bilingual Legal Glossary
</div>

</body>
</html>"""

    out = ROOT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Index → {out}")
    return out


if __name__ == "__main__":
    build_index()
