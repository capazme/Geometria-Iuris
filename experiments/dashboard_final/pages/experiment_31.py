"""Experiment §3.1 — distance structure of the legal lexicon.

Four sub-sections of the thesis: §3.1.1 distances within and between
domains; §3.1.2 maps of inter-domain distance; §3.1.3 agreement
between model pairs; §3.1.4 ordered legal categories.
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
{meta.get('n_terms', 364)} legal terms drawn from the Hong Kong
ordinances co-drafted under the Bilingual Laws Project (§2.2), spread
across seven domains (administrative, civil, constitutional, criminal,
international, labour, procedure). Ten language models process each
term in two ways: <em>attested</em> — the average of the four or more
real ordinance passages in which the term appears — and <em>bare</em>,
the term in isolation. The attested encoding is the result the
experiment analyses; the bare encoding stays alongside as a baseline
for the control-pool subtraction discussed on the Robustness page.
</p>

<p>
The section answers four nested questions. Within a single model, are
terms in the same legal domain closer to each other than to terms in
other domains (§3.1.1)? What is the shape of the seven-by-seven map of
inter-domain distance, and does it recur across models (§3.1.2)? When
two models produce distance maps of the same 364 terms, do they agree
more when they share a language tradition than when they do not
(§3.1.3)? And do the models recover the relative ordering of legal
categories that the doctrine defines as graded (§3.1.4)?
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
            "A lawyer groups terms by legal domain — civil, criminal, "
            "constitutional — without thinking about it. Asked the same "
            "of the lexicon, do language models reproduce the grouping? "
            "And do they keep the legal lexicon, taken as a whole, "
            "distinct from the everyday vocabulary against which it is "
            "supposed to specialise?"
        ) + \
        ui.result_block(
            f"<strong>{n_pass} of {n_total} models</strong> place the "
            "legal lexicon closer together than the everyday-language "
            "control vocabulary, with rank-biserial r above zero and "
            "p below 0.05. The two non-conforming readings are "
            "diagnostic rather than failures: FreeLaw-EN is fine-tuned "
            "on a legal corpus and therefore applies its legal prior "
            "to ordinary words as well; Qwen3-0.6B-EN is small and "
            "multilingual, with English representations not resolved "
            "enough to support the term-class contrast. Within the "
            "three Western-trained models, the same pattern holds for "
            "intra-domain versus inter-domain distance."
        ) + \
        ui.plot_block(fig_lc, "fig-311-legal-control", height_px=420,
                       caption="Rank-biserial r for the legal-versus-control "
                                "Mann-Whitney test, one bar per model. "
                                "Positive r means legal-legal distances "
                                "sit below legal-control distances. The "
                                "two negative bars belong to FreeLaw-EN "
                                "and Qwen3-0.6B-EN, the two diagnostic "
                                "cases of §4.2 of the thesis.") + \
        ui.plot_block(fig_ii, "fig-311-intra-inter", height_px=360,
                       caption="Intra-domain versus inter-domain distance "
                                "on the three Western-trained models. "
                                "Positive r marks intra-domain "
                                "compactness.") + \
        ui.takehome_block(
            "The legal lexicon is internally domain-organised in every "
            "model of the panel and externally distinguishable from "
            "everyday vocabulary in eight of ten. The two exceptions "
            "delimit the encoder regimes on which the diagnostic cannot "
            "be relied upon (§4.2)."
        ) + \
        apparatus_block(
            formula=(
                "r = 1 − 2U / (n<sub>x</sub> n<sub>y</sub>)"
            ),
            stats=[("test",      "Mann-Whitney U, one-sided"),
                   ("legal pairs",   "66 066"),
                   ("control pairs", "36 400"),
                   ("models passing", f"{n_pass} / {n_total}")],
            meta=("Rank-biserial r is the transform of U from "
                  "[0, n<sub>x</sub> n<sub>y</sub>] to [−1, +1]: positive "
                  "r means the first distribution sits below the second."),
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.1.2

def _section_312(s312: dict) -> str:
    fig_topology = figs.fig_topology_smallmultiples(s312, variant="attested")
    return ui.section_open("s312",
                            "§3.1.2 · Maps of distance between legal domains") + \
        ui.scenario_block(
            "What does the geometry of the seven domains of Hong Kong "
            "law look like, when a model projects them as a 7 × 7 map "
            "of average inter-domain distance? Do procedure and "
            "administrative sit at the centre, criminal and "
            "international at the periphery, as doctrinal intuition "
            "would predict? Do models agree on the shape of the map?"
        ) + \
        ui.result_block(
            "Every domain is, on average, closer to itself than to any "
            "of the other six. Procedure and administrative anchor the "
            "centre of the map (lowest mean off-diagonal distance to "
            "the rest); criminal and international sit at the "
            "periphery; labour and constitutional take intermediate "
            "positions. The qualitative pattern recurs across the ten "
            "models; the §3.1.3 measurement that follows turns the "
            "visual recurrence into a number."
        ) + \
        ui.plot_block(fig_topology, "fig-312-topology", height_px=540,
                       caption="The 7 × 7 inter-domain map. Each cell "
                                "is the mean cosine distance between "
                                "terms in the row domain and terms in "
                                "the column domain (attested encoding). "
                                "Use the dropdown above the chart to "
                                "switch between the ten models — the "
                                "same diagonal-darkest pattern recurs "
                                "in every reading.") + \
        ui.takehome_block(
            "The inter-domain map has a stable qualitative shape "
            "across the ten models. The recurrence is the geometric "
            "premise on which the §3.1.3 measurement of agreement "
            "between pairs of models rests."
        ) + \
        apparatus_block(
            formula=(
                "M[i,j] = mean<sub>terms a ∈ domain i,<br>"
                "terms b ∈ domain j</sub> cosine(emb(a), emb(b))"
            ),
            stats=[("domains",   "7"),
                   ("matrix",    "7 × 7"),
                   ("metric",    "cosine on L2-normalised vectors"),
                   ("models", "10")],
            meta=("Diagonal entries are upper-triangle means, "
                  "excluding self-distances; off-diagonal entries are "
                  "full-block means."),
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.1.3 — agreement between pairs of models

def _section_313(s313: dict) -> str:
    fig_forest_att = figs.fig_rsa_forest(s313, variant="attested")
    fig_slope = figs.fig_rsa_bare_attested_slope(s313)
    sum_att = s313["attested"]["summary"]
    sum_bare = s313["bare"]["summary"]
    return ui.section_open("s313",
                            "§3.1.3 · Agreement between pairs of models") + \
        ui.scenario_block(
            "When two models trained on the same language tradition "
            "produce distance maps of the same 364 terms, do they "
            "agree on the shape of the map? When they belong to "
            "different traditions, does their agreement drop? The "
            "answer is the principal quantitative reading of the "
            "chapter."
        ) + \
        ui.result_block(
            "On attested encodings, the average rank correlation "
            "between pairs of Western-trained models is "
            f"<strong>{sum_att['mean_rho_within_weird']:.3f}</strong>, "
            "the average between pairs of Chinese-trained models is "
            f"<strong>{sum_att['mean_rho_within_sinic']:.3f}</strong>, "
            "and the average across nine pairs that span the two "
            f"traditions is <strong>{sum_att['mean_rho_cross_tradition']:.3f}</strong>. "
            "The symmetric within-versus-cross gap, computed as the "
            "average of the two within-tradition means minus the "
            f"cross-tradition mean, is <strong>{sum_att['delta_rho_symmetric']:.3f}</strong>. "
            "The two bilingual readings (a single model embedding both "
            "languages of input) sit at "
            f"{sum_att['mean_rho_within_bilingual']:.3f}, in the same "
            "band as the cross-tradition pairs and well below either "
            "within-tradition floor: holding the model identity fixed "
            "and varying only the language of input does not close the "
            "cross-tradition gap."
        ) + \
        ui.plot_block(fig_forest_att, "fig-313-forest", height_px=560,
                       caption="Seventeen pre-registered model pairs, "
                                "attested encoding. Error bars are 95% "
                                "confidence intervals from term-level "
                                "block bootstrap (B = 10 000). All "
                                "seventeen Mantel p-values lie at the "
                                "permutation floor; the Holm-adjusted "
                                "maximum is 0.0017.") + \
        ui.plot_block(fig_slope, "fig-313-slope", height_px=440,
                       caption="Bare-to-attested ρ trajectory for each "
                                "of the seventeen pairs. Within-"
                                "tradition pairs rise steeply under "
                                "contextualisation on Hong Kong "
                                "ordinance passages; cross-tradition "
                                "pairs trace nearly flat lines.") + \
        ui.disclaimer(
            f"<strong>The 0.543 figure is not the legal-meaning gap on "
            "its own.</strong> The same construction on bare encodings "
            f"of the 364 terms returns "
            f"{sum_bare['delta_rho_symmetric']:.3f}, and the same "
            "construction on 100 everyday-language control terms "
            "returns 0.156 — statistically indistinguishable from the "
            "bare gap on the legal core. The bare gap is therefore "
            "shaped by the models themselves, not by legal vocabulary. "
            "The legal-meaning contribution is the difference that "
            f"attestation adds on the legal core: 0.378 = "
            f"{sum_att['delta_rho_symmetric']:.3f} − "
            f"{sum_bare['delta_rho_symmetric']:.3f}. See the "
            '<a href="robustness_caveats.html#control-pool-subtraction">'
            "Robustness page</a> for the full decomposition."
        ) + \
        ui.takehome_block(
            "Within-tradition agreement on the attested 364-term core "
            "is high and tight; cross-tradition agreement is markedly "
            "lower. The gap is real but its interpretable size is the "
            "0.378 attested-bare difference, not the 0.543 attested "
            "absolute. §4.1 of the thesis reads §3.1.3 as the "
            "principal cross-tradition finding; §4.2 reads the "
            "control-pool subtraction as its primary limit."
        ) + \
        apparatus_block(
            formula=(
                "Δρ<sub>sym</sub> = "
                "(ρ̄<sub>W</sub> + ρ̄<sub>S</sub>) / 2 − ρ̄<sub>cross</sub>"
            ),
            stats=[("Δρ<sub>sym</sub> attested",
                    f"{sum_att['delta_rho_symmetric']:.3f}"),
                   ("Δρ<sub>sym</sub> bare",
                    f"{sum_bare['delta_rho_symmetric']:.3f}"),
                   ("Mantel B",      "10 000"),
                   ("Bootstrap B",   "10 000"),
                   ("Holm K",        "17"),
                   ("p<sub>max</sub> (Holm)", "0.0017")],
            meta=("Spearman ρ on the upper triangle of each model's "
                  "364 × 364 cosine distance matrix; 95% confidence "
                  "intervals from term-level block bootstrap "
                  "(Nili et al. 2014). See §2.4 of the thesis for the "
                  "full statistical apparatus."),
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
            "Legal doctrine grades concepts along ordered continua "
            "marked by doctrinally significant transitions: criminal "
            "law marks the <em>doli incapax</em> threshold between "
            "infancy and imputability; contract law marks the onset of "
            "capacity at the age of majority; procedure marks the "
            "summary / indictable divide and the determinate / "
            "indeterminate disposal boundary. If embedding geometry "
            "tracks legal meaning, the largest cosine gap along the "
            "principal component of an ordered sequence should fall at "
            "the doctrinally expected break, not at a linguistic "
            "midpoint."
        ) + \
        ui.result_block(
            f"Five pre-registered ordinal probes, each templated across "
            f"eleven legal categories and five paraphrase variants per "
            f"language, are evaluated against {n_total} models. The "
            f"ensemble mean Spearman ρ on the age-and-contractual-"
            f"capacity probe is "
            f"<strong>{_rho(test_3):.3f}</strong>; on disposal "
            f"severity <strong>{_rho(test_5):.3f}</strong>; on offence "
            f"severity <strong>{_rho(test_4):.3f}</strong>; on the "
            f"<em>doli incapax</em> threshold "
            f"<strong>{_rho(test_1):.3f}</strong>, a borderline test "
            f"whose expected break sits close to the eleven-position "
            f"midpoint. A negative-control probe on contract value "
            f"returns <strong>{_rho(test_2):.3f}</strong>, confirming "
            f"that the positive signal is doctrinal, not generic to "
            f"ordered sequences. The probe operates on templated "
            f"sentences rather than on the curated lexicon, so its "
            f"reading is independent of pool curation."
        ) + \
        ui.plot_block(fig, "fig-314-probe", height_px=360,
                       caption="Ensemble mean Spearman ρ per test, "
                                "averaged across ten models. The orange "
                                "bar marks the borderline doli incapax "
                                "probe, where the expected break sits "
                                "within one position of the linguistic "
                                "midpoint.") + \
        ui.takehome_block(
            "Where a legal threshold marks a discontinuity that is "
            "also linguistically marked — by a calque, by a "
            "morphological boundary, by a lexicalised opposition — the "
            "embedding registers the threshold as the geometric "
            "breakpoint. Models do not discover orderings autonomously, "
            "and they do not validate a breakpoint that the lexicon "
            "does not carry."
        ) + \
        apparatus_block(
            formula=(
                "ρ = Spearman(category_index<sub>1..11</sub>, "
                "PC1_coordinate); &nbsp; "
                "max_gap_index = argmax<sub>i</sub> "
                "|emb(category<sub>i+1</sub>) − emb(category<sub>i</sub>)|"
            ),
            stats=[("tests",            str(n_tests)),
                   ("positive tests",     str(n_positive_tests)),
                   ("categories / test",  "11"),
                   ("templates / test",   "5 EN + 5 ZH"),
                   ("models",             str(n_total)),
                   ("exact hits (positive tests)",
                    f"{total_exact} / {n_positive_tests * n_total}")],
            meta=("Each probe encodes eleven templated category "
                  "sentences in both languages, projects the resulting "
                  "vectors onto their first principal component, and "
                  "asks whether the cosine gaps peak at the doctrinally "
                  "expected position."),
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
                     "whether models agree on the shape of the map.",
            crumb="Chapter 3 · §3.1",
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
