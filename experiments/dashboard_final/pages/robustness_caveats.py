"""Robustness & caveats page — generates `output/robustness_caveats.html`.

Five readings drawn from the inferential discipline of §4.2 of the
thesis: control-pool subtraction, pool perturbation, same-lemma term
proof, expected failure modes, bilingual causal control.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import shared_ui as ui  # noqa: E402
from apparatus import apparatus_block  # noqa: E402
from data import loader_extensions as ext_loader  # noqa: E402
from figures import extensions as ext_fig  # noqa: E402


def _intro() -> str:
    return ui.section_open("intro", "Robustness &amp; caveats") + """
<p class="lead">
The §3.1.3 cross-tradition reading of 0.543 is the measured quantity,
not the interpretable one. This page collects the five passages of
inferential discipline that §4.2 of the thesis declares as the form
in which the measurement is usable: the subtraction that isolates the
legal-meaning contribution from the model-tradition baseline; the
behaviour of the principal quantity under pool perturbation; the
term-level reading on same-lemma pairs; the two diagnostic regimes
on which the §3.1.1 instrument cannot be relied upon; and the
bilingual control that rules out a model-identity confound.
</p>
""" + ui.section_close()


# ==========================================================================
# (a) Control-pool subtraction — the most prominent reading

def _section_control_pool() -> str:
    nums = ext_loader.y_caveat_numbers()
    return ui.section_open("control-pool-subtraction",
                            "The control-pool subtraction") + \
        ui.scenario_block(
            "The §3.1.3 reading of 0.543 measures the symmetric "
            "within-versus-cross tradition gap as the pipeline "
            "actually computes it, on the attested encodings of the "
            "364-term legal core. It does not, by itself, isolate the "
            "contribution that the legal attestation adds. What part "
            "of the 0.543 is owed to the legal contextualisation on "
            "Hong Kong ordinances, and what part is a baseline that "
            "any two cross-tradition models would carry into the "
            "comparison before the legal context has been applied?"
        ) + \
        ui.result_block(
            "The same construction repeated on the bare encodings of "
            "the 364 legal terms returns "
            f"<strong>{nums['bare_core']:.3f}</strong>: a "
            "model-tradition baseline with no legal content. The same "
            "construction repeated on 100 everyday-language control "
            "terms returns "
            f"<strong>{nums['bare_control']:.3f}</strong>, "
            "statistically indistinguishable from the bare baseline on "
            "the legal core. The bare gap is therefore "
            "model-tradition-shaped, not legal-tradition-shaped. The "
            "legal-meaning contribution is the difference that "
            "attestation adds on the legal core: "
            f"<strong>{nums['legal_gap']:.3f} = "
            f"{nums['attested_core']:.3f} − {nums['bare_core']:.3f}</strong>."
        ) + \
        ui.number_callout(
            f"{nums['legal_gap']:.3f}",
            ("Legal-meaning signal · attested − bare on the 364 "
             "curated terms · "
             f"<strong>{nums['attested_core']:.3f} − "
             f"{nums['bare_core']:.3f}</strong>. "
             "This is the share of the §3.1.3 number that can be "
             "attributed to contextualisation on Hong Kong "
             "ordinances, against a model-tradition baseline of "
             "approximately 0.16."),
        ) + \
        ui.data_table(
            columns=("Reading", "Δρ<sub>sym</sub>", "Interpretation"),
            rows=[
                ("Bare, 364 legal terms",
                 f"{nums['bare_core']:.3f}",
                 "model-tradition baseline with no legal content"),
                ("Bare, 100 everyday-language control terms",
                 f"{nums['bare_control']:.3f}",
                 "indistinguishable from the baseline above: confirms "
                 "the bare gap is shaped by the models, not by legal "
                 "vocabulary"),
                ("Attested, 364 legal terms",
                 f"{nums['attested_core']:.3f}",
                 "the §3.1.3 result the experiment analyses"),
                ("Legal-meaning contribution",
                 f"{nums['legal_gap']:.3f}",
                 f"attested − bare = {nums['attested_core']:.3f} − "
                 f"{nums['bare_core']:.3f}, the share attributable to "
                 "contextualisation on Hong Kong ordinances"),
            ],
            col_classes=("", "num strong", ""),
            row_classes=("", "", "", "highlight"),
        ) + \
        ui.takehome_block(
            "The 0.543 figure and the 0.378 figure belong together. "
            f"The thesis cites the absolute {nums['attested_core']:.3f} "
            "as the principal §3.1.3 measurement and the "
            f"{nums['legal_gap']:.3f} as the interpretable legal-"
            "meaning contribution. §4.2 of the thesis carries this "
            "decomposition as the primary methodological limit of the "
            "experiment."
        ) + \
        apparatus_block(
            formula=(
                "legal contribution = Δρ<sub>sym</sub><sup>attested</sup>(legal core) "
                "− Δρ<sub>sym</sub><sup>bare</sup>(legal core) "
                f"= {nums['attested_core']:.3f} − {nums['bare_core']:.3f} "
                f"= {nums['legal_gap']:.3f}"
            ),
            stats=[
                ("attested, legal core", f"{nums['attested_core']:.3f}"),
                ("bare, legal core",     f"{nums['bare_core']:.3f}"),
                ("bare, control",        f"{nums['bare_control']:.3f}"),
                ("contribution",         f"{nums['legal_gap']:.3f}"),
            ],
            meta=("Control terms are everyday vocabulary — pronouns, "
                  "deixis, basic common nouns — that have no Hong Kong "
                  "ordinance attestation by design, so the only "
                  "comparable reading on them is bare-on-bare."),
        ) + \
        ui.section_close()


# ==========================================================================
# (b) Pool perturbation — robustness of the principal quantity

def _section_pool_perturbation() -> str:
    D_table = ext_loader.d_robustness_table()
    fig = ext_fig.fig_D_robustness_curve(D_table)
    last = D_table[-1]
    return ui.section_open("pool-perturbation",
                            "Robustness under pool perturbation") + \
        ui.scenario_block(
            "How robust is the §3.1.3 reading if the curated 364-term "
            "pool is partly replaced by uncurated background legal "
            "vocabulary that the corpus does include but that the "
            "manual curation did not vet?"
        ) + \
        ui.result_block(
            "The symmetric within-versus-cross tradition gap moves "
            f"from <strong>{D_table[0]['mean_delta_sym']:.3f}</strong> "
            "with no background injected to "
            f"<strong>{last['mean_delta_sym']:.3f}</strong> with 75% "
            "of the pool replaced by background legal terms. The "
            "trajectory drifts upward, not downward: the gap does not "
            "decay with contamination from neighbouring legal "
            "vocabulary, it strengthens slightly. The cross-tradition "
            "reading is therefore a property of legal vocabulary at "
            "large, not an artefact of the particular 364-term "
            "selection on which §3.1.3 was computed."
        ) + \
        ui.plot_block(fig, "fig-pool-perturbation", height_px=440,
                       caption="Symmetric within-versus-cross tradition "
                                "gap as a function of the background "
                                "share of the pool. Mean ± 95% "
                                "confidence interval across ten pool "
                                "replicates per injection level.") + \
        ui.takehome_block(
            "The principal §3.1.3 reading survives heavy curation "
            "perturbation. The §4.1 synthesis cites this as the "
            "robustness condition under which the cross-tradition "
            "claim travels beyond the curated pool."
        ) + \
        apparatus_block(
            formula=(
                "Δρ<sub>sym</sub>(p) = "
                "(ρ̄<sub>W</sub>(p) + ρ̄<sub>S</sub>(p)) / 2 "
                "− ρ̄<sub>cross</sub>(p)"
            ),
            stats=[
                ("at 0% background",  f"{D_table[0]['mean_delta_sym']:.3f}"),
                ("at 25% background", f"{D_table[2]['mean_delta_sym']:.3f}"),
                ("at 75% background", f"{last['mean_delta_sym']:.3f}"),
                ("replicates / level", str(D_table[0]['n_replicates'])),
            ],
            meta=("Confidence intervals across pool replicates at each "
                  "injection level. No level whose interval excludes "
                  "the no-injection baseline."),
            collapsible=True,
        ) + \
        ui.section_close()


# ==========================================================================
# (c) Same-lemma divergence — term-level evidence

def _section_same_lemma() -> str:
    G_rows = ext_loader.g_false_friends_top(8)
    fig = ext_fig.fig_G_false_friends_scatter(G_rows)
    most = G_rows[0] if G_rows else None
    example_rows = G_rows[:4]
    return ui.section_open("same-lemma-divergence",
                            "Same-lemma divergence at term level") + \
        ui.scenario_block(
            "Consider a legal term and its direct Chinese translation "
            "— say, <em>specimen signature</em> and 簽名樣本 "
            "(literally <em>signature sample</em>): the same legal "
            "object, a reference exemplar of an authorised signer's "
            "handwriting deposited with a bank or court. When one "
            "model trained on English legal corpora and one model "
            "trained on Chinese legal corpora encode the pair, do "
            "they recognise the two strings as the same concept? "
            "And does a single model that handles both languages at "
            "once agree? If the two tradition-specialised models "
            "disagree but a single bilingual model agrees, the "
            "disagreement cannot be a calibration mismatch between "
            "models. It must lie in what each tradition-specialised "
            "model has internalised."
        ) + \
        ui.result_block(
            "On the same set of same-lemma legal pairs, two models "
            "trained one on English legal corpora and one on Chinese "
            "legal corpora (BGE-EN-large × BGE-ZH-large) report "
            "cosine similarity in the band "
            "<strong>−0.11 to +0.10</strong>: at best mild alignment, "
            "often outright anti-correlation. A single bilingual "
            "model that places both languages in one semantic space "
            "(BGE-M3-EN × BGE-M3-ZH) reports cosine in the band "
            "<strong>+0.27 to +0.87</strong> on the very same pairs. "
            "The disagreement is not a property of the model "
            "architectures: when the tradition layer is removed by a "
            "shared model, the lemmas align."
        ) + \
        ui.plot_block(fig, "fig-same-lemma", height_px=480,
                       caption="Each point is one same-lemma pair "
                                "(English headword paired with its "
                                "canonical Chinese translation in the "
                                "Hong Kong DOJ bilingual glossary). "
                                "Cosine under the two tradition-"
                                "specialised models on the x-axis, "
                                "cosine under the single bilingual "
                                "model on the y-axis. The cloud sits "
                                "consistently above the y = x diagonal.") + \
        ui.data_table(
            columns=("English term", "Chinese term",
                      "Cross-tradition cosine", "Bilingual cosine"),
            rows=[(r["en"], r["zh"],
                    f"{r['cos_cross']:+.3f}",
                    f"{r['cos_bilingual']:+.3f}")
                   for r in example_rows],
            col_classes=("", "", "num", "num strong"),
        ) + \
        (ui.disclaimer(
            f"Lead example: <strong>{most['en']}</strong> / "
            f"{most['zh']}. The two tradition-specialised models "
            f"place the pair at cosine "
            f"<strong>{most['cos_cross']:+.3f}</strong> "
            f"(near-orthogonal or weakly anti-aligned). The single "
            f"bilingual model places it at "
            f"<strong>{most['cos_bilingual']:+.3f}</strong> (strongly "
            f"aligned). The instrument is not failing: it is reporting "
            f"a difference that the tradition-specialised training "
            f"itself has internalised."
         ) if most else "") + \
        ui.takehome_block(
            "Cross-tradition divergence on the very same legal lemma "
            "is a property of how each tradition's training corpus "
            "encodes the concept, not a calibration mismatch between "
            "model architectures. When both languages flow through a "
            "single model, the divergence resolves. §4.1 of the "
            "thesis reads this as the term-level proof of the §3.1.3 "
            "agreement claim."
        ) + \
        apparatus_block(
            stats=[
                ("eligible same-lemma pairs", "4 156"),
                ("attestation filter",
                  "at least two ordinance contexts per term, each language"),
                ("tradition-specialised models",
                  "BGE-EN-large × BGE-ZH-large"),
                ("bilingual model",
                  "BGE-M3-EN × BGE-M3-ZH"),
                ("cross-tradition cosine range", "[−0.106, +0.100]"),
                ("bilingual cosine range",       "[+0.273, +0.867]"),
            ],
            meta=("Same-lemma pairs are English headwords paired with "
                  "their canonical Chinese translation in the Hong Kong "
                  "DOJ bilingual glossary, restricted to terms with at "
                  "least two ordinance attestations in each language. "
                  "Cosine is computed on the mean of the attested "
                  "context vectors."),
            collapsible=True,
        ) + \
        ui.section_close()


# ==========================================================================
# (d) Expected failure modes — what §3.1.1 cannot do

def _section_failure_modes() -> str:
    return ui.section_open("expected-failures",
                            "Expected failure modes") + \
        ui.scenario_block(
            "Two readings in Chapter 3 do not align with what the rest "
            "of the panel reports. §3.1.1 of the thesis identifies "
            "two model regimes on which the legal-versus-control "
            "diagnostic cannot be relied upon; §3.1.4 reports a "
            "negative-control probe whose result is the absence of "
            "the very signal the probe was designed not to find. Both "
            "are limits of the instrument, declared explicitly by "
            "§4.2 of the thesis."
        ) + \
        ui.result_block(
            "<strong>FreeLaw-EN, the model fine-tuned on a United "
            "States legal corpus, fails the §3.1.1 legal-versus-"
            "control test in the sign-reversed direction</strong>: "
            "rank-biserial r = −0.121, with the legal-control median "
            "sitting below the legal-legal median, so the alternative "
            "hypothesis of the one-sided test is rejected. The reason "
            "is structural: a model steeped in legal English applies "
            "its legal prior to ordinary vocabulary as well, treating "
            "<em>I</em>, <em>you</em>, <em>here</em> under the same "
            "representational regime as <em>trustee</em>, <em>lien</em>, "
            "<em>registration</em>. The diagnostic operates on general-"
            "purpose models, not on already-fine-tuned ones."
        ) + \
        ui.result_block(
            "<strong>The §3.1.4 negative-control probe on contract "
            "value finds no doctrinal break</strong>, as the law "
            "prescribes: common law imposes the writing requirement "
            "on contracts for the sale of land regardless of "
            "consideration, so the sequence from <em>symbolic</em> to "
            "<em>massive</em> contract value should not register a "
            "stable structural break. It does not. The English-side "
            "readings cluster the modal break at the linguistic "
            "midpoint (the generic artefact the pre-registration "
            "anticipated for a uniform sequence); the Chinese-side "
            "readings disaggregate without converging on any "
            "alternative break. The probe is not finding a signal "
            "where the law expects none — which is, on this design, "
            "exactly the success criterion."
        ) + \
        ui.takehome_block(
            "Both readings make the instrument's negative space "
            "visible: the §3.1.1 diagnostic does not apply to "
            "legally-fine-tuned models, and the §3.1.4 probe behaves "
            "as the law prescribes when the law prescribes no "
            "threshold. §4.2 of the thesis carries these as the "
            "explicit limits on the affirmative reading of §4.1."
        ) + \
        apparatus_block(
            stats=[
                ("FreeLaw-EN, legal-vs-control r",        "−0.121"),
                ("FreeLaw-EN, p (one-sided)",             "1.0"),
                ("Negative control, ensemble Spearman ρ", "0.651"),
                ("Negative control, modal break",
                  "at the linguistic midpoint, no doctrinal anchor"),
            ],
            meta=("FreeLaw-EN passes the §3.1.1 within-domain test "
                  "(r = +0.214 bare, +0.258 attested) but fails the "
                  "legal-versus-control test; the fine-tuning collapses "
                  "the term-class boundary the second test is "
                  "designed to detect."),
            collapsible=True,
        ) + \
        ui.section_close()


# ==========================================================================
# (e) Bilingual control — encoder-identity counterfactual

def _section_bilingual_control() -> str:
    return ui.section_open("bilingual-control",
                            "The bilingual control") + \
        ui.scenario_block(
            "A sceptic might argue that the cross-tradition gap of "
            "§3.1.3 is an artefact of model identity: nine cross-"
            "tradition pairs, however symmetrically chosen, are still "
            "nine pairs of distinct models, trained by different "
            "teams on different curations. To rule out the model-"
            "identity confound, two bilingual models (BGE-M3 and "
            "Qwen3-Embedding-0.6B) embed the entire 364-term lexicon "
            "twice — once in English, once in Chinese — and the same "
            "agreement statistic is computed on the pair (English "
            "side × Chinese side) of each."
        ) + \
        ui.result_block(
            "The two bilingual readings cluster at "
            "<strong>ρ̄ = 0.316</strong> — within the cross-tradition "
            "band of 0.246 and statistically indistinguishable from "
            "it, well below the within-tradition floors of 0.712 "
            "(Western-trained) and 0.868 (Chinese-trained) by more "
            "than four times the typical confidence-interval width. "
            "Holding model identity fixed and varying only the "
            "language of input does not close the cross-tradition "
            "gap. The factor that explains the §3.1.3 reading "
            "therefore cannot be model identity; it must be something "
            "the models have absorbed from the corpora on which they "
            "were trained."
        ) + \
        ui.data_table(
            columns=("Pair", "Spearman ρ", "Reading"),
            rows=[
                ("Within Western-trained (3 pairs)",
                 "0.712", "models agreeing within tradition"),
                ("Within Chinese-trained (3 pairs)",
                 "0.868", "models agreeing within tradition"),
                ("Cross-tradition (9 pairs)",
                 "0.246", "models from the two traditions, compared"),
                ("Bilingual control (2 pairs)",
                 "0.316", "same model, two languages of input"),
            ],
            col_classes=("", "num strong", ""),
            row_classes=("", "", "", "highlight"),
        ) + \
        ui.takehome_block(
            "The bilingual control is the causal counterfactual of "
            "§2.3 of the thesis: by holding model identity fixed, it "
            "isolates the corpus-tradition component of the cross-"
            "tradition gap. The reading delivers: the gap survives "
            "the manipulation."
        ) + \
        apparatus_block(
            stats=[
                ("within Western-trained",  "0.712"),
                ("within Chinese-trained",  "0.868"),
                ("cross-tradition (9 pairs)", "0.246"),
                ("bilingual control (2 pairs)", "0.316"),
            ],
            meta=("Bilingual control pairs: BGE-M3 read on its "
                  "English side against itself read on its Chinese "
                  "side, and the same for Qwen3-Embedding-0.6B. "
                  "Source: §3.1.3 of the thesis."),
            collapsible=True,
        ) + \
        ui.section_close()


# ==========================================================================
# build()

def build() -> str:
    parts = [
        ui.page_head(
            title="Robustness &amp; caveats",
            subtitle="Five readings drawn from the inferential "
                     "discipline of §4.2 of the thesis.",
            crumb="Chapter 3 · Robustness",
        ),
        ui.sticky_nav(current_href="robustness_caveats.html"),
        ui.open_main(),
        _intro(),
        _section_control_pool(),
        _section_pool_perturbation(),
        _section_same_lemma(),
        _section_failure_modes(),
        _section_bilingual_control(),
        ui.linear_nav(
            prev=("experiment_32.html", "Experiment §3.2"),
            next_=None,
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
