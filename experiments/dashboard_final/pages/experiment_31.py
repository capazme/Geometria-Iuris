"""Experiment 3.1 — distance structure of the legal lexicon.

Four sub-sections: §3.1.1 distances within and between domains;
§3.1.2 maps of inter-domain distance; §3.1.3 agreement between model
pairs (RSA); §3.1.4 ordered legal categories.

Each sub-section follows the five-step concentric pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import shared_ui as ui  # noqa: E402
from apparatus import apparatus_block  # noqa: E402
from data import loader_31  # noqa: E402
from figures import exp31 as figs  # noqa: E402


# --------------------------------------------------------------------------
# Intro

def _intro(meta: dict) -> str:
    return ui.section_open("intro", "Experiment §3.1 — Distance structure") + f"""
<p class="lead">
Frozen 2026-05-17 — {meta.get('n_terms', 364)} core legal terms
sampled from Hong Kong ordinances co-drafted under the Bilingual Laws
Project (post-1989, structurally bilingual, not equal-authenticity
fictions), distributed across 7 legal domains (administrative, civil,
constitutional, criminal, international, labor, procedure). Ten encoder
models — three WEIRD, three Sinic, four bilingual — encode each term in
two ways: <em>attested</em>, the primary signal, as the mean of K ≥ 4
real ordinance contexts; and <em>bare</em>, a methodological baseline,
as the lemma in isolation. Attested is what the experiments analyse;
bare grounds the Y caveat that isolates legal content from
encoder-tradition bias.
</p>

<p>
The Experiment asks three nested questions. Are legal terms more
similar to each other than they are to everyday vocabulary, and do
they cluster by domain (§3.1.1)? What does the inter-domain map look
like, and do models agree on it (§3.1.2)? When two models produce
distance maps over the same 364 terms, do they agree more when they
share a language tradition than when they do not (§3.1.3)? And do
embeddings recover the relative ordering of pre-registered legal
categories (§3.1.4)?
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# §3.1.1

def _section_311(s311: dict) -> str:
    fig_lc = figs.fig_legal_control_bar(s311, variant="bare")
    fig_ii = figs.fig_intra_inter_bar(s311, variant="bare")
    lvc = s311["bare"]["legal_vs_control"]
    n_pass = sum(1 for m, b in lvc.items()
                 if float(b.get("effect_r", 0)) > 0 and float(b.get("p_value", 1)) < 0.05)
    n_total = len(lvc)
    return ui.section_open("s311",
                            "§3.1.1 · Distances within and between legal domains") + \
        ui.scenario_block(
            "A judge classifies disputes by domain — civil, criminal, "
            "constitutional. Embedding models, when asked the same of "
            "the vocabulary, should produce tighter clusters within a "
            "domain than between domains."
        ) + \
        ui.result_block(
            f"<strong>{n_pass} of {n_total} models</strong> place legal "
            "vocabulary closer together than everyday-language control "
            "vocabulary (rank-biserial r &gt; 0, p &lt; 0.05). The two "
            "exceptions are diagnostic: FreeLaw-EN (fine-tuned on a "
            "legal corpus, loses the term-class contrast) and "
            "Qwen3-0.6B-EN (small multilingual, under-specialised on "
            "English). On the 3 WEIRD models the same pattern holds "
            "for intra-vs-inter domain comparisons."
        ) + \
        ui.plot_block(fig_lc, "fig-311-legal-control", height_px=420,
                       caption="Mann-Whitney rank-biserial r per model. "
                                "Positive r = legal-legal distances "
                                "below legal-control. Two negative bars "
                                "(FreeLaw-EN, Qwen3-0.6B-EN) are "
                                "informative, not failures.") + \
        ui.plot_block(fig_ii, "fig-311-intra-inter", height_px=360,
                       caption="Intra-domain vs inter-domain distances "
                                "on the three WEIRD models. Positive r "
                                "= intra-domain compactness.") + \
        ui.takehome_block(
            "Legal vocabulary clusters by domain in eight of ten "
            "encoders. The two negative exceptions reveal limits of "
            "the instrument, not of the lexicon. Chapter 4 §4.1 cites "
            "this as the baseline result that the rest of the chapter "
            "depends on."
        ) + \
        apparatus_block(
            formula=(
                "r = 1 − 2U / (n<sub>x</sub> n<sub>y</sub>)"
            ),
            stats=[("test", "Mann-Whitney U one-sided"),
                   ("n_legal", "66 066 pairs"),
                   ("n_control", "36 400 pairs"),
                   ("pass", f"{n_pass} / {n_total}")],
            meta=("Effect r is the rank-biserial transform of U; r &gt; 0 "
                  "indicates median(legal-legal) &lt; median(legal-control). "
                  "All ten p-values reported in "
                  "<code>section_311_legal_vs_control.per_model</code>."),
            code_ref=[("experiments/ch3-measurability/experiment_1_structure/"
                       "results_bare/", "legal_vs_control.json")],
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.1.2

def _section_312(s312: dict) -> str:
    fig_single = figs.fig_topology_heatmap(s312, model="BGE-EN-large",
                                            variant="attested")
    fig_small = figs.fig_topology_smallmultiples(s312, variant="attested")
    return ui.section_open("s312",
                            "§3.1.2 · Maps of distance between legal domains") + \
        ui.scenario_block(
            "What does inter-domain distance look like, when seven "
            "domains of HK law are projected by an encoder? Do "
            "criminal and procedural law sit close, as a doctrinal "
            "intuition would predict? Do encoders agree on the map?"
        ) + \
        ui.result_block(
            "The 7 × 7 inter-domain map shows recurrent structure: "
            "constitutional, civil, and criminal domains form a tight "
            "core; international and procedural law sit on the periphery; "
            "labor and administrative law occupy intermediate positions. "
            "Ten encoders produce visually similar maps — quantified "
            "by the §3.1.3 RSA test below."
        ) + \
        ui.plot_block(fig_single, "fig-312-single", height_px=460,
                       caption="BGE-EN-large attested · mean cosine "
                                "distance between every pair of "
                                "core terms whose domains are (row, col).") + \
        ui.plot_block(fig_small, "fig-312-small", height_px=520,
                       caption="Same 7 × 7 map, one panel per model "
                                "(attested). The qualitative pattern "
                                "recurs across all ten encoders.") + \
        ui.takehome_block(
            "The inter-domain map is the unit of analysis for §3.1.3: "
            "RSA correlates these K × K matrices across model pairs. "
            "Chapter 4 §4.1 reads the recurrence of the same coarse "
            "layout across encoders as a stability claim about the "
            "geometry."
        ) + \
        apparatus_block(
            formula=(
                "M[i,j] = mean<sub>terms a ∈ domain i,<br>"
                "terms b ∈ domain j</sub> cosine(emb(a), emb(b))"
            ),
            stats=[("domains",   "7"),
                   ("matrix",    "7 × 7"),
                   ("metric",    "cosine on L2-normalised pooled embeddings"),
                   ("populated", "all 10 models")],
            meta=("Diagonal entries are upper-triangle means (no "
                  "self-distance). Off-diagonal entries are full-block "
                  "means. Matrices are symmetric within sub-section "
                  "rounding."),
            code_ref=[("experiments/ch3-measurability/scripts/",
                       "experiment_1_structure.py")],
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.1.3 — the headline section, with link to Robustness Y

def _section_313(s313: dict) -> str:
    fig_forest_att = figs.fig_rsa_forest(s313, variant="attested")
    fig_slope = figs.fig_rsa_bare_attested_slope(s313)
    sum_att = s313["attested"]["summary"]
    sum_bare = s313["bare"]["summary"]
    return ui.section_open("s313",
                            "§3.1.3 · Agreement between pairs of models (RSA)") + \
        ui.scenario_block(
            "When two encoders trained on the same language tradition "
            "produce maps of the same 364 terms, do they agree on the "
            "shape of the map? And when they belong to different "
            "traditions, does their agreement drop?"
        ) + \
        ui.result_block(
            f"On attested encodings: within-WEIRD ρ̄ = "
            f"<strong>{sum_att['mean_rho_within_weird']:.3f}</strong>, "
            f"within-Sinic ρ̄ = "
            f"<strong>{sum_att['mean_rho_within_sinic']:.3f}</strong>, "
            f"cross-tradition ρ̄ = "
            f"<strong>{sum_att['mean_rho_cross_tradition']:.3f}</strong>. "
            f"The symmetric gap Δρ_sym = "
            f"<strong>{sum_att['delta_rho_symmetric']:.3f}</strong>. "
            "The within-bilingual control ρ̄ = "
            f"{sum_att['mean_rho_within_bilingual']:.3f} is close to "
            "the cross-tradition mean — confirming the gap is not an "
            "encoder-pair artefact."
        ) + \
        ui.plot_block(fig_forest_att, "fig-313-forest", height_px=560,
                       caption="17 model pairs, attested encoding, "
                                "ordered by group. Error bars are 95% CI "
                                "(block bootstrap, B = 10 000). All 17 "
                                "p-values are at the permutation floor "
                                "(B = 10 000).") + \
        ui.plot_block(fig_slope, "fig-313-slope", height_px=440,
                       caption="Bare → attested ρ shift for each of the "
                                "17 pairs. Within-WEIRD and within-Sinic "
                                "pairs gain substantially; cross-tradition "
                                "pairs gain little.") + \
        ui.disclaimer(
            "<strong>The absolute number 0.543 is not the finding by itself.</strong> "
            f"The bare Δρ_sym on the same 364 core terms is "
            f"{sum_bare['delta_rho_symmetric']:.3f}: a methodological "
            "baseline with no legal content, reported only so that the "
            "Y caveat can subtract it. On 100 everyday-language control "
            "terms the bare baseline returns 0.156, statistically "
            "indistinguishable, confirming the baseline is encoder-shaped. "
            "The legal signal is the gap that attestation opens on the "
            "core: 0.378 = 0.543 − 0.165. See "
            '<a href="robustness_caveats.html#Y-caveat">'
            "Robustness § Y caveat</a> for the full reframing."
        ) + \
        ui.takehome_block(
            "Two-tradition divergence is a structural property of the "
            "attested geometry: Chapter 4 §4.1 cites this section as "
            "the headline; §4.2 cites the Y caveat as its primary limit."
        ) + \
        apparatus_block(
            formula=(
                "Δρ<sub>sym</sub> = "
                "(ρ̄<sub>W</sub> + ρ̄<sub>S</sub>) / 2 − ρ̄<sub>cross</sub>"
            ),
            stats=[("Δρ_sym attested",
                    f"{sum_att['delta_rho_symmetric']:.3f}"),
                   ("Δρ_sym bare",
                    f"{sum_bare['delta_rho_symmetric']:.3f}"),
                   ("Mantel B",      "10 000"),
                   ("Bootstrap B",   "10 000"),
                   ("Holm K",        "17"),
                   ("p_max attested", "≤ 1.7e-3")],
            meta=("RSA on upper-triangle Spearman ρ over per-pair 364×364 "
                  "cosine RDMs. 95% CIs from term-level block bootstrap "
                  "(Nili et al. 2014). Mantel p ≤ 1e-4 for all 17 pairs."),
            code_ref=[("experiments/ch3-measurability/experiment_1_structure/"
                       "results_attested/", "experiment_1_results.json")],
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.1.4 — categorical probe

def _section_314(s314: dict) -> str:
    fig = figs.fig_categorical_probe_forest(s314)
    tests_data = s314.get("tests", {})
    n_tests = len(tests_data)

    test_1 = tests_data.get("test_1_age_imputability", {})
    test_3 = tests_data.get("test_3_age_contractual_capacity", {})
    test_4 = tests_data.get("test_4_offence_severity", {})
    test_5 = tests_data.get("test_5_disposal_severity", {})
    test_2 = tests_data.get("test_2_magnitude_negative_control", {})

    def _rho(t: dict) -> float:
        return float(t.get("summary", {}).get("mean_ensemble_rho", 0))

    def _hits(t: dict) -> int:
        return int(t.get("summary", {}).get("n_models_exact_hit", 0))

    def _n_models(t: dict) -> int:
        return int(t.get("summary", {}).get("n_models_total", 0))

    n_total = _n_models(test_3) if test_3 else 10
    total_exact = sum(_hits(t) for t in tests_data.values()
                       if t.get("polarity") == "positive")
    n_positive_tests = sum(1 for t in tests_data.values()
                            if t.get("polarity") == "positive")

    return ui.section_open("s314",
                            "§3.1.4 · Ordered legal categories and meaningful breakpoints") + \
        ui.scenario_block(
            "Legal doctrine positions concepts along ordered continua "
            "marked by doctrinally significant transitions. Criminal "
            "law marks the <em>doli incapax</em> threshold between "
            "infancy and imputability; contract law marks the onset of "
            "capacity at the age of majority; procedure marks the "
            "summary / indictable divide and the determinate / "
            "indeterminate disposal boundary. If embedding geometry "
            "tracks legal meaning, the largest cosine gap along the "
            "principal component of an ordered sequence should fall at "
            "the doctrinally expected break, not at a linguistic "
            "midpoint or a random position."
        ) + \
        ui.result_block(
            f"Five pre-registered ordinal probes, each templated across "
            f"eleven legal categories and five paraphrase variants per "
            f"language, are evaluated against {n_total} encoders. The "
            f"ensemble mean ρ̄ on the age-contractual-capacity probe is "
            f"<strong>{_rho(test_3):.3f}</strong>; on disposal severity "
            f"<strong>{_rho(test_5):.3f}</strong>; on offence severity "
            f"<strong>{_rho(test_4):.3f}</strong>; on the <em>doli "
            f"incapax</em> threshold (borderline: expected break sits "
            f"close to the eleven-position midpoint) "
            f"<strong>{_rho(test_1):.3f}</strong>. A negative-control "
            f"probe on contract value returns "
            f"<strong>{_rho(test_2):.3f}</strong>, confirming that the "
            f"positive signal is doctrinal, not generic to ordered "
            f"sequences. The breakpoint placement is curation-independent: "
            f"the probe operates on templated category sequences rather "
            f"than on the curated lexicon, so its result transfers across "
            f"pool variations."
        ) + \
        ui.plot_block(fig, "fig-314-probe", height_px=360,
                       caption="Ensemble mean ρ̄ per test, averaged across "
                                "ten encoders. The orange bar marks the "
                                "borderline doli incapax probe, where the "
                                "expected break sits within one position "
                                "of the linguistic midpoint and modal-gap "
                                "placement is partially confounded with "
                                "sequence symmetry.") + \
        ui.takehome_block(
            "Encoders recover the relative ordering of pre-registered "
            "legal categories and place the largest gap at the "
            "doctrinally expected position on three of the four positive "
            "tests, while the negative control returns no signal. "
            "Chapter 4 §4.1 reads §3.1.4 as the evidence that the "
            "geometry is not noise: the doctrinal break is where the "
            "doctrine says it is, and the modal placement is robust to "
            "pool curation."
        ) + \
        apparatus_block(
            formula=(
                "ρ = Spearman(category_index<sub>1..11</sub>, "
                "PC1_coordinate); &nbsp; "
                "max_gap_index = argmax<sub>i</sub> "
                "|emb(category<sub>i+1</sub>) − emb(category<sub>i</sub>)|"
            ),
            stats=[("n_tests",            str(n_tests)),
                   ("positive tests",     str(n_positive_tests)),
                   ("categories / test",  "11"),
                   ("templates / test",   "5 EN + 5 ZH"),
                   ("models",             str(n_total)),
                   ("exact-hit count (positive tests)",
                    f"{total_exact} / {n_positive_tests * n_total}")],
            meta=("Each probe encodes eleven templated category "
                  "sentences in both languages, projects the resulting "
                  "RDM onto its first principal component, and asks "
                  "whether the cosine gaps along the ordered sequence "
                  "peak at the doctrinally expected position. The "
                  "borderline flag triggers when the expected gap index "
                  "sits within one position of the sequence midpoint, "
                  "where modal-hit counting is confounded with sequence "
                  "symmetry."),
            code_ref=[("experiments/ch3-measurability/experiment_1_structure/"
                       "results_bare/", "categorical_probe.json")],
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# build()

def build() -> str:
    data = loader_31.load_all()
    parts = [
        ui.page_head(
            title="Experiment §3.1 — Distance structure",
            subtitle="How the legal lexicon clusters by domain, and "
                     "whether encoders agree on the map.",
            crumb="Chapter 3 · Experiment 1",
        ),
        ui.sticky_nav(current_href="experiment_31.html"),
        ui.open_main(),
        _intro(data["s311"]["meta"]),
        _section_311(data["s311"]),
        _section_312(data["s312"]),
        _section_313(data["s313"]),
        _section_314(data["s314"]),
        ui.linear_nav(
            prev=("how_it_works.html", "How it works"),
            next_=("experiment_32.html", "Experiment §3.2"),
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
