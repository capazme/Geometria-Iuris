"""
NTA exploration run — all candidate terms × 6 models.

Outputs a readable report to results/nta_exploration.txt for manual selection
of the most interesting terms to include in the thesis.

Usage:
    cd experiments/
    python -m lens_3_stratigraphy.nta_exploration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lens_3_stratigraphy.lens3 import (
    _load_config,
    _load_core_terms,
    run_nta,
)

RESULTS_DIR = Path(__file__).parent / "results"

# 101 candidate terms — fundamental to legal theory, polysemic, or archetypical
CANDIDATES = [
    # Theory of law / constitutional foundations
    "rule of law", "freedom", "equality", "sovereignty", "constitution",
    "constitutional", "human rights", "habeas corpus", "citizenship",
    "suffrage", "franchise", "amendment", "veto", "prerogative",
    "martial law", "legislation",
    # Criminal archetypes
    "murder", "homicide", "manslaughter", "assault", "battery",
    "theft", "larceny", "robbery", "felony", "conspiracy",
    "corruption", "bribery", "forgery", "perjury", "blackmail",
    "extortion", "imprisonment", "parole", "probation", "remand",
    "conviction", "acquittal", "prosecution", "sentencing",
    "actus reus", "mens rea", "recklessness",
    # Civil law fundamentals
    "tort", "negligence", "trespass", "nuisance", "defamation",
    "restitution", "divorce", "adoption", "inheritance", "succession",
    "mortgage", "fiduciary", "trustee", "bailment", "easement",
    "copyright", "patent", "pledge", "subrogation", "novation",
    "tortious", "tortfeasor", "torture",
    # International law
    "treaty", "asylum", "extradition", "genocide", "comity",
    "ratification", "refugee", "diplomatic", "covenant",
    "international law", "jus cogens", "pacta sunt servanda",
    # Procedure
    "mediation", "conciliation", "disclosure", "discovery",
    "burden of proof", "res judicata", "locus standi", "subpoena",
    "hearsay", "affidavit", "cause of action",
    # Labor
    "strike", "dismissal", "harassment", "trade union", "employment",
    # Administrative
    "eminent domain", "ultra vires", "ombudsman",
    # Cross-domain / polysemic
    "accessory", "accession", "approbation", "indictment",
]


def _score_term(term_data: dict, k: int) -> dict:
    """Compute interest scores for a single term across its layers."""
    evol = term_data["domain_evolution"]
    domain = term_data["domain"]

    ctrl_values = [e["n_control"] for e in evol]
    legal_values = [e["n_legal"] for e in evol]
    peak_ctrl = max(ctrl_values)
    peak_ctrl_layer = evol[ctrl_values.index(peak_ctrl)]["layer"]
    final_ctrl = ctrl_values[-1]
    initial_ctrl = ctrl_values[0]

    # Domain convergence: how many of the final k neighbors share the term's domain?
    final_doms = evol[-1]["domains"]
    own_domain_final = final_doms.get(domain, 0)

    # Domain diversity at final layer
    n_domains_final = len([d for d, c in final_doms.items() if c > 0 and d != "control"])

    # Trajectory volatility: total change in ctrl count across layers
    volatility = sum(abs(ctrl_values[i+1] - ctrl_values[i]) for i in range(len(ctrl_values)-1))

    return {
        "initial_ctrl": initial_ctrl,
        "peak_ctrl": peak_ctrl,
        "peak_ctrl_layer": peak_ctrl_layer,
        "final_ctrl": final_ctrl,
        "own_domain_final": own_domain_final,
        "n_domains_final": n_domains_final,
        "volatility": volatility,
        "ctrl_trajectory": ctrl_values,
    }


def format_report(nta_results: dict, k: int) -> str:
    """Format exploration results as a readable text report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("NTA EXPLORATION REPORT — Candidate term selection for §3.1.3c")
    lines.append("=" * 90)
    lines.append(f"Terms: {len(CANDIDATES)} | Models: {len(nta_results)} | k={k}")
    lines.append("")

    # Collect scores across all models
    term_scores: dict[str, list[dict]] = {}

    for model_label, dd_data in nta_results.items():
        for term_name, term_data in dd_data["terms"].items():
            if term_name not in term_scores:
                term_scores[term_name] = []
            score = _score_term(term_data, k)
            score["model"] = model_label
            score["domain"] = term_data["domain"]
            score["zh"] = term_data.get("zh", "")
            term_scores[term_name].append(score)

    # ── Section 1: Most interesting terms (ranked by cross-model interest) ──
    lines.append("")
    lines.append("-" * 90)
    lines.append("RANKING: Most interesting terms (sorted by cross-model trajectory interest)")
    lines.append("-" * 90)
    lines.append("")
    lines.append("Interest = peak_ctrl (how many ctrl neighbors at worst) + volatility")
    lines.append("         + final_ctrl (unresolved polysemy) + domain_spread (cross-domain)")
    lines.append("")

    ranking: list[tuple[str, float, dict]] = []
    for term_name, scores in term_scores.items():
        avg_peak = np.mean([s["peak_ctrl"] for s in scores])
        avg_vol = np.mean([s["volatility"] for s in scores])
        avg_final_ctrl = np.mean([s["final_ctrl"] for s in scores])
        avg_n_domains = np.mean([s["n_domains_final"] for s in scores])
        # Composite interest score
        interest = avg_peak * 2 + avg_vol + avg_final_ctrl * 3 + avg_n_domains
        ranking.append((term_name, interest, {
            "domain": scores[0]["domain"],
            "zh": scores[0]["zh"],
            "avg_peak_ctrl": round(avg_peak, 1),
            "avg_final_ctrl": round(avg_final_ctrl, 1),
            "avg_volatility": round(avg_vol, 1),
            "avg_n_domains_final": round(avg_n_domains, 1),
        }))

    ranking.sort(key=lambda x: -x[1])

    lines.append(f"{'#':>3} {'Term':<28} {'Domain':<16} {'PeakCtrl':>8} "
                 f"{'FinalCtrl':>9} {'Volat':>6} {'DomF':>5} {'Score':>6}")
    lines.append("─" * 90)
    for i, (name, score, info) in enumerate(ranking, 1):
        lines.append(
            f"{i:3d} {name:<28} {info['domain']:<16} "
            f"{info['avg_peak_ctrl']:>8.1f} {info['avg_final_ctrl']:>9.1f} "
            f"{info['avg_volatility']:>6.1f} {info['avg_n_domains_final']:>5.1f} "
            f"{score:>6.1f}"
        )

    # ── Section 2: Per-term detail (trajectory per model) ──
    lines.append("")
    lines.append("")
    lines.append("=" * 90)
    lines.append("DETAIL: Per-term trajectory across all models")
    lines.append("=" * 90)

    # Show in ranking order
    for term_name, _, _ in ranking:
        scores = term_scores[term_name]
        domain = scores[0]["domain"]
        zh = scores[0]["zh"]
        lines.append("")
        lines.append(f"── {term_name} ({domain}) {zh} ──")

        for s in scores:
            model = s["model"]
            traj = s["ctrl_trajectory"]
            traj_str = " → ".join(str(v) for v in traj)
            final_doms_str = ""
            lines.append(
                f"  {model:<22} ctrl=[{traj_str}]  "
                f"peak={s['peak_ctrl']} @L{s['peak_ctrl_layer']}  "
                f"final: own_dom={s['own_domain_final']}/{k}  "
                f"n_domains={s['n_domains_final']}"
            )

    # ── Section 3: Cross-tradition divergence ──
    lines.append("")
    lines.append("")
    lines.append("=" * 90)
    lines.append("CROSS-TRADITION: Terms where WEIRD and Sinic models diverge most")
    lines.append("=" * 90)
    lines.append("")

    weird_labels = [s["model"] for s in term_scores[CANDIDATES[0]]
                    if "ZH" not in s["model"] and "Dmeta" not in s["model"]]
    sinic_labels = [s["model"] for s in term_scores[CANDIDATES[0]]
                    if "ZH" in s["model"] or "Dmeta" in s["model"]]

    divergences: list[tuple[str, float]] = []
    for term_name, scores in term_scores.items():
        weird_final = np.mean([s["own_domain_final"] for s in scores
                               if s["model"] in weird_labels])
        sinic_final = np.mean([s["own_domain_final"] for s in scores
                               if s["model"] in sinic_labels])
        weird_ctrl = np.mean([s["final_ctrl"] for s in scores
                              if s["model"] in weird_labels])
        sinic_ctrl = np.mean([s["final_ctrl"] for s in scores
                              if s["model"] in sinic_labels])
        div = abs(weird_final - sinic_final) + abs(weird_ctrl - sinic_ctrl) * 2
        divergences.append((term_name, div))

    divergences.sort(key=lambda x: -x[1])

    lines.append(f"{'#':>3} {'Term':<28} {'Domain':<16} {'Divergence':>10}")
    lines.append("─" * 60)
    for i, (name, div) in enumerate(divergences[:30], 1):
        domain = term_scores[name][0]["domain"]
        lines.append(f"{i:3d} {name:<28} {domain:<16} {div:>10.1f}")

    return "\n".join(lines)


def main() -> None:
    weird_labels, sinic_labels = _load_config()
    all_labels = weird_labels + sinic_labels
    terms_core, core_idx = _load_core_terms()

    print(f"Running NTA exploration: {len(CANDIDATES)} terms × {len(all_labels)} models")

    nta_results = {}
    for label in all_labels:
        nta_results[label] = run_nta(
            label, terms_core, core_idx,
            target_terms=CANDIDATES,
            k=7, device="cpu",
        )

    # Save raw JSON
    json_path = RESULTS_DIR / "nta_exploration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nta_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw JSON → {json_path}")

    # Save readable report
    report = format_report(nta_results, k=7)
    report_path = RESULTS_DIR / "nta_exploration.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report   → {report_path}")


if __name__ == "__main__":
    main()
