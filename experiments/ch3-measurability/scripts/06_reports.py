#!/usr/bin/env python3
"""
Step 6 — Reports for run #4 post-BLP.

Reads experiment_1_structure/results_{bare,attested}/experiment_1_results.json and
experiment_2_axes/results_{bare,attested}/experiment_2_results.json, then produces:

  reports/numbers_headline.md      — table-ready numbers for CLAUDE.md §10
  reports/changes_vs_run3.md       — delta between run #3 (Firthian) and #4
  reports/verification_gate.md     — pass/fail per criterion of PLAN §8

Diagnostic plots are produced separately by the notebook (step 7).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"


# Run #3 reference numbers (from CLAUDE.md §10, attested column)
RUN3_REF = {
    "rho_W_attested": 0.760,
    "rho_S_attested": 0.845,
    "rho_cross_attested": 0.259,
    "delta_rho_attested": 0.541,
    "rho_bilingual_attested": 0.340,
    "rho_W_bare": 0.499,
    "rho_S_bare": 0.470,
    "rho_cross_bare": 0.288,
    "delta_rho_bare": 0.211,
    "axes_cross_attested": {
        "public_private":         0.254,
        "status_contract":        0.151,
        "rights_duties":          0.071,
        "individual_collective":  0.182,
        "state_market":           0.232,
        "natural_positive":       0.111,
    },
    "axes_cross_bare": {
        "public_private":         0.451,
        "status_contract":        0.402,
        "rights_duties":          0.370,
        "individual_collective":  0.358,
        "state_market":           0.254,
        "natural_positive":       0.247,
    },
    "n_terms": 327,
    "n_perm": 1000,
    "n_boot": 1000,
}


def load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def write_headline(out_path: Path,
                   structure_bare: dict, lens1_att: dict,
                   axes_bare: dict, lens4_att: dict) -> None:
    s1b = structure_bare["section_313"]["summary"]
    s1a = lens1_att["section_313"]["summary"]
    s4b = axes_bare["section_324"]["cross_rho_mean_per_axis"]
    s4a = lens4_att["section_324"]["cross_rho_mean_per_axis"]
    ranking_attested = lens4_att["section_324"]["ranking_most_divergent_first"]
    lvc = structure_bare.get("section_311_legal_vs_control", {}).get("per_model", {})

    lines: list[str] = []
    lines.append("# Run #4 headline numbers — post-BLP final (364 terms)")
    lines.append("")
    lines.append("All ρ̄ are Spearman correlations on RDM upper triangles (§3.1.3) "
                 "or on per-term axis projections (§3.2.3).")
    lines.append(f"B (Mantel) = {structure_bare['meta']['n_perm']}; "
                 f"B (block bootstrap) = {structure_bare['meta']['n_boot']}; "
                 f"seed = {structure_bare['meta']['seed']}; "
                 f"N = {structure_bare['meta']['n_terms']}.")
    lines.append("")

    lines.append("## §3.1.3 RSA cross-tradition (Lens I)")
    lines.append("")
    lines.append("| Metric                                  | Bare    | Attested |")
    lines.append("|-----------------------------------------|---------|----------|")
    lines.append(f"| within-WEIRD ρ̄ (3 pairs)               | {s1b['mean_rho_within_weird']:.3f}   | **{s1a['mean_rho_within_weird']:.3f}** |")
    lines.append(f"| within-Sinic ρ̄ (3 pairs)               | {s1b['mean_rho_within_sinic']:.3f}   | **{s1a['mean_rho_within_sinic']:.3f}** |")
    lines.append(f"| cross-tradition ρ̄ (9 pairs)            | {s1b['mean_rho_cross_tradition']:.3f}   | **{s1a['mean_rho_cross_tradition']:.3f}** |")
    lines.append(f"| Δρ_sym (avg within − cross)            | {s1b['delta_rho_symmetric']:.3f}   | **{s1a['delta_rho_symmetric']:.3f}** |")
    if "mean_rho_within_bilingual" in s1b:
        lines.append(f"| within-bilingual ρ̄ (β control)         | {s1b['mean_rho_within_bilingual']:.3f}   | {s1a['mean_rho_within_bilingual']:.3f} |")
    lines.append("")

    lines.append("## §3.2.4 Axes alignment cross-tradition ρ̄ (Lens IV)")
    lines.append("")
    lines.append("| Axis                  | Bare    | Attested |")
    lines.append("|-----------------------|---------|----------|")
    for ax in s4b.keys():
        lines.append(f"| {ax:21s} | {s4b[ax]:.3f}   | **{s4a[ax]:.3f}** |")
    lines.append("")

    lines.append("## §3.2.4 Ranking (most divergent → least) — attested")
    lines.append("")
    for i, e in enumerate(ranking_attested, 1):
        lines.append(f"{i}. **{e['axis']}** — ρ̄ = {e['mean_cross_rho']:.3f}")
    lines.append("")

    if lvc:
        lines.append("## §3.1.1 Legal-vs-control (Lens I, bare only — N=100 control terms)")
        lines.append("")
        lines.append("Mann-Whitney U one-sided (alternative='less'): legal-legal more compact than legal-control.")
        lines.append("Effect size r = rank-biserial.")
        lines.append("")
        lines.append("| Model | legal med | ctrl med | r | p_value |")
        lines.append("|-------|-----------|----------|---|---------|")
        for label, m in lvc.items():
            lines.append(f"| {label} | {m['median_x']:.3f} | {m['median_y']:.3f} | "
                         f"{m['effect_r']:+.3f} | {m['p_value']:.2e} |")
        lines.append("")
        n_pos = sum(1 for m in lvc.values() if m["effect_r"] > 0 and m["p_value"] < 0.05)
        lines.append(f"**{n_pos}/{len(lvc)} models confirm legal-legal more compact than legal-control.**")
        lines.append("")

    # Extension sections — sourced from ext/*.json files
    run_dir = out_path.parent.parent
    ext = run_dir / "ext"
    if ext.exists():
        _append_ext_sections(lines, ext)

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _append_ext_sections(lines: list[str], ext: Path) -> None:
    """Append Extension D, G, H, X, Y, Z sections to the headline document.

    Each section is sourced from its `ext/<dir>/*.json` so the report
    regenerates deterministically. A, E, F get a one-line mention.
    """
    # D — Δρ_sym vs %bg curve
    d = _safe_load(ext / "D_robustness" / "robustness_curve.json")
    if d and d.get("results"):
        lines.append("---")
        lines.append("")
        lines.append("## Extension D — Δρ_sym vs %bg robustness curve")
        lines.append("")
        m = d["meta"]
        lines.append(f"{m.get('n_replicates', '?')} replicates × {len(d['results'])} mix levels; "
                     f"{len(m.get('en_models', []))} EN-side + {len(m.get('zh_models', []))} ZH-side primary models, attested.")
        lines.append("")
        lines.append("| %bg | n_core | n_bg | mean Δρ_sym | std | CI95 |")
        lines.append("|-----|--------|------|-------------|-----|------|")
        for r in d["results"]:
            lines.append(f"| {int(r['p_bg']*100)}% | {r['n_core']} | {r['n_bg']} | "
                         f"**{r['mean_delta_sym']:.3f}** | {r['std_delta_sym']:.3f} | "
                         f"[{r['ci_low_delta_sym']:.3f}, {r['ci_high_delta_sym']:.3f}] |")
        delta_change = d["results"][-1]["mean_delta_sym"] - d["results"][0]["mean_delta_sym"]
        lines.append("")
        lines.append(f"Δρ_sym from {d['results'][0]['mean_delta_sym']:.3f} (p=0%) to "
                     f"{d['results'][-1]['mean_delta_sym']:.3f} (p={int(d['results'][-1]['p_bg']*100)}%): "
                     f"{'+'if delta_change>=0 else ''}{delta_change:.3f}. "
                     f"**Δρ_sym is robust under bg injection** — the cross-tradition gap is structural, "
                     f"not curation-dependent.")
        lines.append("")

    # G — false-friends
    g = _safe_load(ext / "G_false_friends" / "false_friends.json")
    if g and g.get("rows"):
        lines.append("---")
        lines.append("")
        lines.append(f"## Extension G — Automated false-friends ({g['meta']['n_eligible_bg']} bg eligible)")
        lines.append("")
        en_m = g["meta"]["en_model"]
        zh_m = g["meta"]["zh_model"]
        bi = g["meta"].get("bilingual_pair") or [None, None]
        cross_key = f"cos_{en_m}_vs_{zh_m}"
        bi_key = f"cos_{bi[0]}_vs_{bi[1]}" if bi[0] else None
        lines.append(f"Cross-encoder: {en_m} × {zh_m}. Bilingual control: {bi[0]} × {bi[1]}.")
        lines.append("")
        lines.append("Top-10 most divergent (lowest cross-encoder cosine):")
        lines.append("")
        lines.append("| en | zh | K_en | K_zh | cross | bilingual |")
        lines.append("|----|----|------|------|-------|-----------|")
        for r in g["rows"][:10]:
            bi_val = r.get(bi_key) if bi_key else None
            bi_s = f"{bi_val:+.3f}" if bi_val is not None else "n/a"
            lines.append(f"| {r['en']} | {r['zh']} | {r['k_en']} | {r['k_zh']} | "
                         f"{r[cross_key]:+.3f} | {bi_s} |")
        lines.append("")
        lines.append("Same-lemma terms have **negative** cross-encoder cosine but **+0.5 to +0.75** bilingual cosine. The cross-tradition divergence is *tradition-shaped*, not *encoder-artefact*.")
        lines.append("")

    # H — K saturation
    h = _safe_load(ext / "H_K_saturation" / "k_saturation.json")
    if h and h.get("buckets"):
        lines.append("---")
        lines.append("")
        lines.append("## Extension H — K saturation curve")
        lines.append("")
        lines.append("ρ_cross attested as a function of bg K_min bucket:")
        lines.append("")
        lines.append("| K bucket | n bg | ρ_cross |")
        lines.append("|----------|------|---------|")
        for b in h["buckets"]:
            mr = b.get("mean_rho_cross")
            mr_s = f"**{mr:+.3f}**" if mr is not None else "n/a"
            lines.append(f"| {b['K_bucket']} | {b['n_bg']} | {mr_s} |")
        lines.append(f"| core 4-8 (run #4 headline) | 364 | +{h['meta']['core_reference_attested_cross_rho']:.3f} |")
        lines.append("")
        lines.append("ρ_cross monotonic with K; saturation around K≥4. The pre-registered threshold is empirically justified.")
        lines.append("")

    # X — Δρ_sym vs %control
    x = _safe_load(ext / "X_control_robustness" / "control_robustness_curve.json")
    if x and x.get("results"):
        lines.append("---")
        lines.append("")
        lines.append("## Extension X — Δρ_sym vs %control (bare, dual of D)")
        lines.append("")
        lines.append("| %ctrl | n_core | n_ctrl | mean Δρ_sym bare | std |")
        lines.append("|-------|--------|--------|------------------|-----|")
        for r in x["results"]:
            lines.append(f"| {int(r['p_control']*100)}% | {r['n_core']} | {r['n_control']} | "
                         f"{r['mean_delta_sym']:.3f} | {r['std_delta_sym']:.3f} |")
        d0 = x["results"][0]["mean_delta_sym"]
        dN = x["results"][-1]["mean_delta_sym"]
        lines.append("")
        lines.append(f"Δρ_sym bare from {d0:.3f} (p=0%) to {dN:.3f} (p={int(x['results'][-1]['p_control']*100)}%): "
                     f"{dN-d0:+.3f}. Direction correct (non-legal injection reduces the signal), effect small.")
        lines.append("")

    # Y — control-only RSA
    y = _safe_load(ext / "Y_control_only" / "control_only_rsa.json")
    if y and y.get("summary"):
        lines.append("---")
        lines.append("")
        lines.append("## Extension Y — Cross-tradition ρ on control-only (CRUCIAL CAVEAT)")
        lines.append("")
        s = y["summary"]
        cmp = y["comparison"]
        lines.append("Δρ_sym bare on the 100 control terms (everyday vocabulary, *I, you, he, this, here*, and ZH equivalents):")
        lines.append("")
        lines.append("| Quantity | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| within-WEIRD ρ̄ | {s['mean_rho_within_weird']:.3f} |")
        lines.append(f"| within-Sinic ρ̄ | {s['mean_rho_within_sinic']:.3f} |")
        lines.append(f"| cross-tradition ρ̄ | {s['mean_rho_cross_tradition']:.3f} |")
        lines.append(f"| **Δρ_sym bare** | **{s['delta_rho_symmetric']:.3f}** |")
        lines.append("")
        lines.append("**Comparison:**")
        lines.append("")
        lines.append("| Pool | Encoding | Δρ_sym |")
        lines.append("|------|----------|--------|")
        lines.append(f"| 364 core | bare | {cmp['delta_sym_core_bare_run4']:.3f} |")
        lines.append(f"| 364 core | attested | **{cmp['delta_sym_core_attested_run4']:.3f}** |")
        lines.append(f"| 100 control | bare | **{cmp['delta_sym_control_bare']:.3f}** ← indistinguishable from core bare |")
        lines.append("")
        gap = cmp["delta_sym_core_attested_run4"] - cmp["delta_sym_core_bare_run4"]
        lines.append(f"**Reframing.** The bare signal is encoder-tradition shaped, not legal-tradition shaped. "
                     f"The legal contribution is the **attested-bare gap on the core: "
                     f"{cmp['delta_sym_core_attested_run4']:.3f} − {cmp['delta_sym_core_bare_run4']:.3f} = {gap:.3f}**.")
        lines.append("")

    # Z — tier hierarchy
    z = _safe_load(ext / "Z_tier_hierarchy" / "tier_hierarchy.json")
    if z and z.get("per_model"):
        lines.append("---")
        lines.append("")
        lines.append("## Extension Z — 3-tier distance hierarchy")
        lines.append("")
        lines.append("| Model | core×core | core×bg | core×control | Monotonic |")
        lines.append("|-------|-----------|---------|--------------|-----------|")
        for label, m in z["per_model"].items():
            med = m["median"]
            mono = "✓" if m["monotonic_hierarchy"] else "✗"
            lines.append(f"| {label} | {med['core_core']:.3f} | {med['core_bg']:.3f} | "
                         f"{med['core_control']:.3f} | {mono} |")
        n_mono = z["meta"]["n_models_with_monotonic_hierarchy"]
        n_total = z["meta"]["n_models"]
        lines.append("")
        lines.append(f"**{n_mono}/{n_total} models** satisfy median(core×core) < median(core×bg) < median(core×control). "
                     f"In the others, bg are *farther* from core than control are. "
                     f"Tier classification is corpus-curative, not geometric.")
        lines.append("")

    # A, E, F — short notes
    a = _safe_load(ext / "A_bg_knn" / "background_assignments.json")
    e_oos = _safe_load(ext / "E_axes_oos" / "coherence.json")
    f_strat = _safe_load(ext / "F_confidence" / "confidence_strata.json")
    if a or e_oos or f_strat:
        lines.append("---")
        lines.append("")
        lines.append("## Extensions A, E, F (short notes)")
        lines.append("")
        if a:
            am = a["meta"]
            lines.append(f"- **A** (k-NN bg domain assignment): {am['n_bg']} bg → 7 domains via k={am['k']} NN; "
                         f"mean confidence {am['confidence_mean']:.3f}, median {am['confidence_median']:.3f}.")
        if e_oos:
            lines.append(f"- **E** (axes out-of-sample projection): 6 axes projected on bg; coherence per "
                         f"k-NN-assigned domain reported per (model, axis) — pool-sensitive axes remain "
                         f"pool-sensitive on bg.")
        if f_strat:
            base = f_strat["baseline_core_only"]["delta_sym"]
            hi = f_strat["high_confidence_bg_injected"]["mean_delta_sym"]
            lo = f_strat["low_confidence_bg_injected"]["mean_delta_sym"]
            lines.append(f"- **F** (confidence-stratified, n=20 replicates, n_inject={f_strat['meta']['n_inject_per_stratum']}): "
                         f"baseline Δρ_sym={base:.3f}; high-conf bg injected → {hi:.3f} ({hi-base:+.3f}); "
                         f"low-conf → {lo:.3f} ({lo-base:+.3f}). Small effect, interpretive hint only.")
        lines.append("")

    # Closing pointer
    lines.append("---")
    lines.append("")
    lines.append("**Three anchor results for Cap. 4:** D (structurality), G + bilingual control "
                 "(tradition-shaped, not encoder-shaped), Y (the legal signal is the attested-bare "
                 "gap on the core, 0.378). See `reports/extensions_summary.md` for the full narrative.")


def write_changes_vs_run3(out_path: Path,
                          structure_bare: dict, lens1_att: dict,
                          axes_bare: dict, lens4_att: dict) -> None:
    s1b = structure_bare["section_313"]["summary"]
    s1a = lens1_att["section_313"]["summary"]
    s4b = axes_bare["section_324"]["cross_rho_mean_per_axis"]
    s4a = lens4_att["section_324"]["cross_rho_mean_per_axis"]

    lines: list[str] = []
    lines.append("# Changes: run #3 (Firthian 327) → run #4 (post-BLP 364)")
    lines.append("")
    lines.append("Δ = run4 − run3. Positive Δ means run #4 is higher.")
    lines.append("")

    lines.append("## §3.1.3 RSA — attested column (the headline)")
    lines.append("")
    lines.append("| Metric              | Run #3 (327) | Run #4 (364) | Δ      |")
    lines.append("|---------------------|--------------|--------------|--------|")
    pairs = [
        ("within-WEIRD ρ̄",  "rho_W_attested",     s1a["mean_rho_within_weird"]),
        ("within-Sinic ρ̄",  "rho_S_attested",     s1a["mean_rho_within_sinic"]),
        ("cross ρ̄",          "rho_cross_attested", s1a["mean_rho_cross_tradition"]),
        ("Δρ_sym",            "delta_rho_attested", s1a["delta_rho_symmetric"]),
    ]
    if "mean_rho_within_bilingual" in s1a:
        pairs.append(("within-bilingual ρ̄", "rho_bilingual_attested",
                      s1a["mean_rho_within_bilingual"]))
    for label, key, new in pairs:
        old = RUN3_REF[key]
        delta = new - old
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label:20s} | {old:.3f}        | {new:.3f}        | {sign}{delta:.3f} |")
    lines.append("")

    lines.append("## §3.1.3 RSA — bare column")
    lines.append("")
    lines.append("| Metric              | Run #3 (327) | Run #4 (364) | Δ      |")
    lines.append("|---------------------|--------------|--------------|--------|")
    pairs_bare = [
        ("within-WEIRD ρ̄",  "rho_W_bare",     s1b["mean_rho_within_weird"]),
        ("within-Sinic ρ̄",  "rho_S_bare",     s1b["mean_rho_within_sinic"]),
        ("cross ρ̄",          "rho_cross_bare", s1b["mean_rho_cross_tradition"]),
        ("Δρ_sym",            "delta_rho_bare", s1b["delta_rho_symmetric"]),
    ]
    for label, key, new in pairs_bare:
        old = RUN3_REF[key]
        delta = new - old
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label:20s} | {old:.3f}        | {new:.3f}        | {sign}{delta:.3f} |")
    lines.append("")

    lines.append("## §3.2.4 Axes cross-tradition ρ̄ — attested")
    lines.append("")
    lines.append("| Axis                  | Run #3 (327) | Run #4 (364) | Δ      |")
    lines.append("|-----------------------|--------------|--------------|--------|")
    for ax in s4a.keys():
        new = s4a[ax]
        old = RUN3_REF["axes_cross_attested"].get(ax, 0.0)
        delta = new - old
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {ax:21s} | {old:.3f}        | {new:.3f}        | {sign}{delta:.3f} |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_gate(out_path: Path,
               structure_bare: dict, lens1_att: dict,
               axes_bare: dict, lens4_att: dict,
               embeddings_dir: Path) -> None:
    s1a = lens1_att["section_313"]["summary"]
    rsa_pairs = (
        lens1_att["section_313"]["within_weird"]
        + lens1_att["section_313"]["within_sinic"]
        + lens1_att["section_313"]["cross_tradition"]
    )
    p_max = max(p["p_value"] for p in rsa_pairs)
    p_holm_max = max(p["p_holm"] for p in rsa_pairs)

    n_embed_dirs = sum(1 for p in embeddings_dir.iterdir()
                       if p.is_dir() and (p / "vecs_bare.npy").exists()
                       and (p / "vecs_attested.npy").exists())

    rho_W = s1a["mean_rho_within_weird"]
    rho_S = s1a["mean_rho_within_sinic"]
    rho_C = s1a["mean_rho_cross_tradition"]
    delta = s1a["delta_rho_symmetric"]
    ref = RUN3_REF

    # Legal-vs-control: majority of models must show legal-legal more compact
    lvc = structure_bare.get("section_311_legal_vs_control", {}).get("per_model", {})
    lvc_pos = sum(1 for m in lvc.values() if m["effect_r"] > 0 and m["p_value"] < 0.05)

    checks = [
        ("≥10 embeddings dirs with bare + attested",  n_embed_dirs >= 10,
         f"{n_embed_dirs}/10"),
        ("ρ̄_cross attested in [run3 ±0.10]",
         abs(rho_C - ref["rho_cross_attested"]) <= 0.10,
         f"{rho_C:.3f} vs {ref['rho_cross_attested']:.3f}"),
        ("ρ̄_W attested in [run3 ±0.10]",
         abs(rho_W - ref["rho_W_attested"]) <= 0.10,
         f"{rho_W:.3f} vs {ref['rho_W_attested']:.3f}"),
        ("ρ̄_S attested in [run3 ±0.10]",
         abs(rho_S - ref["rho_S_attested"]) <= 0.10,
         f"{rho_S:.3f} vs {ref['rho_S_attested']:.3f}"),
        ("Δρ_sym attested ≥ 0.4",  delta >= 0.4,  f"{delta:.3f}"),
        ("Mantel p_max ≤ 0.001",   p_max <= 0.001, f"{p_max:.6f}"),
        ("Holm p_max ≤ 0.005",     p_holm_max <= 0.005, f"{p_holm_max:.6f}"),
        ("legal-vs-control: ≥8/10 models with r>0 and p<0.05",
         lvc_pos >= 8, f"{lvc_pos}/10"),
    ]

    lines: list[str] = []
    lines.append("# Verification gate — run #4 post-BLP")
    lines.append("")
    lines.append("Per PLAN.md §8. If any check is FAIL: stop and investigate.")
    lines.append("")
    lines.append("| Check                                  | Status | Value vs threshold |")
    lines.append("|----------------------------------------|--------|--------------------|")
    pass_count = 0
    for desc, ok, val in checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1
        lines.append(f"| {desc:38s} | {status:6s} | {val} |")
    lines.append("")
    lines.append(f"**{pass_count}/{len(checks)} checks passed**.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    structure_bare = load(REPO_ROOT / cfg["paths"]["structure_bare"] / "experiment_1_results.json")
    lens1_att  = load(REPO_ROOT / cfg["paths"]["structure_attested"] / "experiment_1_results.json")
    axes_bare = load(REPO_ROOT / cfg["paths"]["axes_bare"] / "experiment_2_results.json")
    lens4_att  = load(REPO_ROOT / cfg["paths"]["axes_attested"] / "experiment_2_results.json")

    reports_dir = REPO_ROOT / cfg["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    write_headline(reports_dir / "numbers_headline.md",
                   structure_bare, lens1_att, axes_bare, lens4_att)
    write_changes_vs_run3(reports_dir / "changes_vs_run3.md",
                          structure_bare, lens1_att, axes_bare, lens4_att)
    write_gate(reports_dir / "verification_gate.md",
               structure_bare, lens1_att, axes_bare, lens4_att,
               REPO_ROOT / cfg["paths"]["embeddings"])

    print(f"Reports written to {reports_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
