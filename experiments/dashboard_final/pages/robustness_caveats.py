"""Robustness & caveats page — generates `output/robustness_caveats.html`.

Houses the seven non-headline extensions plus the Y caveat as the
section's anchor. Two parts:

  1. Headline-strengthening   D, G, H
  2. Caveats                   F, X, Y (anchor), Z, with A and E as appendix
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
    return ui.section_open("intro", "Robustness & caveats") + """
<p class="lead">
Three anchor results carry the chapter (see Home). Seven smaller
investigations either strengthen those results or qualify them; this
page houses all seven, alongside a methodological reframing of the
headline number — the Y caveat — that the rest of the dashboard refers
back to.
</p>

<p>
The opening half (D, G, H) shows that the cross-tradition gap survives
pool perturbation, that the divergence operates at the level of
individual same-lemma terms, and that the empirical K threshold for
attestation is grounded. The closing half (F, X, Y, Z) discloses
limits: stratification by k-NN confidence, dual injection of control
terms, the reframing of the legal signal as the attested-bare gap, and
the partial failure of a naive tier hierarchy.
</p>
""" + ui.section_close()


# ==========================================================================
# Headline-strengthening: D, G, H

def _section_D() -> str:
    D_table = ext_loader.d_robustness_table()
    fig = ext_fig.fig_D_robustness_curve(D_table)
    last = D_table[-1]
    return ui.section_open("D-robustness",
                            "D · Δρ_sym stability under background injection") + \
        ui.scenario_block(
            "How robust is the gap if the lexicon is partly contaminated "
            "with background legal terms — terms that the curator did not "
            "vet by hand, but that the corpus does include?"
        ) + \
        ui.result_block(
            f"The cross-tradition gap Δρ_sym attested moves from "
            f"<strong>{D_table[0]['mean_delta_sym']:.3f}</strong> at 0% "
            f"background to <strong>{last['mean_delta_sym']:.3f}</strong> "
            f"at 75% background. The signal does not collapse — it "
            f"rises slightly. The pattern is structural."
        ) + \
        ui.plot_block(fig, "fig-D-robustness", height_px=440,
                       caption="Mean ± 95% CI across 10 pool replicates "
                                "per p_bg level. Baseline (p_bg = 0) "
                                "matches the headline Δρ_sym attested.") + \
        ui.takehome_block(
            "Pool curation moves the absolute number slightly but does "
            "not erase the cross-tradition gap. Chapter 4 may cite the "
            "headline Δρ_sym attested with confidence: the gap persists "
            "regardless of which background terms are added to the pool."
        ) + \
        apparatus_block(
            formula=(
                "Δρ<sub>sym</sub>(p) = "
                "(ρ̄<sub>W</sub>(p) + ρ̄<sub>S</sub>(p)) / 2 "
                "− ρ̄<sub>cross</sub>(p)"
            ),
            stats=[
                ("Δρ at 0%",  f"{D_table[0]['mean_delta_sym']:.3f}"),
                ("Δρ at 25%", f"{D_table[2]['mean_delta_sym']:.3f}"),
                ("Δρ at 75%", f"{last['mean_delta_sym']:.3f}"),
                ("replicates / level", str(D_table[0]['n_replicates'])),
            ],
            meta=("Pool replicates over background subsamples; CI is "
                  "across replicates, not within a single pool (no "
                  "Mantel inside this curve)."),
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_D_robustness.py", "main()")],
        ) + \
        ui.section_close()


def _section_G() -> str:
    G_rows = ext_loader.g_false_friends_top(8)
    fig = ext_fig.fig_G_false_friends_scatter(G_rows)
    most = G_rows[0] if G_rows else None
    example_rows = G_rows[:4]
    return ui.section_open("G-false-friends",
                            "G · Same-lemma divergence (term-level proof)") + \
        ui.scenario_block(
            "Consider a term like <em>specimen signature</em> in English "
            "law and its direct Chinese translation 簽名樣本 (literally "
            "<em>signature sample</em>): both designate the same legal "
            "object, a reference exemplar of an authorised signer's "
            "handwriting deposited with a bank or court. If we ask two "
            "separate models, each trained on a single language tradition, "
            "to encode the pair, do they recognise the two strings as the "
            "same concept? And if we ask a single model that encodes both "
            "languages simultaneously, does it agree? The question matters: "
            "if the two encoders disagree but a single bilingual model "
            "agrees, the disagreement cannot be an encoder calibration "
            "artefact. It must lie in what each language-specialised "
            "encoder has absorbed from its own tradition."
        ) + \
        ui.result_block(
            "On the same set of same-lemma legal pairs, two encoders "
            "trained one on English legal corpora and one on Chinese "
            "legal corpora (BGE-EN-large × BGE-ZH-large) report cosine "
            "similarity in the band <strong>−0.11 to +0.10</strong>: at "
            "best mild alignment, often outright anti-correlation. "
            "A single bilingual encoder that places both languages in one "
            "semantic space (BGE-M3-EN × BGE-M3-ZH) reports cosine in the "
            "band <strong>+0.27 to +0.87</strong> on the very same pairs. "
            "The disagreement is not a property of the encoder "
            "architectures: when the tradition layer is removed by a "
            "shared encoder, the lemmas align."
        ) + \
        ui.plot_block(fig, "fig-G-false-friends", height_px=480,
                       caption="Each point is one same-lemma pair "
                                "(English headword paired with its "
                                "canonical Chinese translation in the "
                                "HK DOJ bilingual glossary). The x-axis "
                                "is the cosine under the two "
                                "tradition-specialised encoders; the "
                                "y-axis is the cosine under the single "
                                "bilingual encoder. The cloud sits "
                                "consistently above the y = x diagonal: "
                                "bilingual cosine exceeds cross-encoder "
                                "cosine on essentially every pair.") + \
        ui.data_table(
            columns=("English term", "Chinese term",
                      "cos cross-tradition", "cos bilingual"),
            rows=[(r["en"], r["zh"],
                    f"{r['cos_cross']:+.3f}",
                    f"{r['cos_bilingual']:+.3f}")
                   for r in example_rows],
            col_classes=("", "", "num", "num strong"),
        ) + \
        (ui.disclaimer(
            f"Lead example: <strong>{most['en']}</strong> / "
            f"{most['zh']}. The two language-specialised encoders place "
            f"the pair at cosine <strong>{most['cos_cross']:+.3f}</strong> "
            f"(near-orthogonal or weakly anti-aligned). The single "
            f"bilingual encoder places it at "
            f"<strong>{most['cos_bilingual']:+.3f}</strong> (strongly "
            f"aligned). The instrument is not failing: it is reporting a "
            f"difference that the tradition-specialised training itself "
            f"has internalised."
         ) if most else "") + \
        ui.takehome_block(
            "Cross-tradition divergence on the very same legal lemma is "
            "a property of how each tradition's language and corpus "
            "encode the concept, not a calibration mismatch between "
            "encoder architectures. When both languages flow through a "
            "single model, the divergence resolves. Chapter 4 should "
            "cite G as the term-level proof of the headline RSA claim: "
            "the encoder is a witness of the tradition, and the "
            "divergence the witness reports is doctrinal, not technical."
        ) + \
        apparatus_block(
            formula=(
                "cos(en, zh) = (emb<sub>en</sub> · emb<sub>zh</sub>) / "
                "(‖emb<sub>en</sub>‖ × ‖emb<sub>zh</sub>‖) "
                "on L2-normalised pooled vectors"
            ),
            stats=[
                ("eligible same-lemma pairs", "4 156"),
                ("min K filter", "K ≥ 2 attestations per term, each language"),
                ("tradition-specialised encoders",
                  "BGE-EN-large × BGE-ZH-large"),
                ("bilingual encoder",
                  "BGE-M3-EN × BGE-M3-ZH"),
                ("cross-tradition cos range", "[−0.106, +0.100]"),
                ("bilingual cos range",       "[+0.273, +0.867]"),
            ],
            meta=("Same-lemma pairs are English headwords paired with "
                  "their canonical Chinese translation in the Hong Kong "
                  "DOJ bilingual glossary, restricted to terms with at "
                  "least two ordinance attestations in each language. "
                  "Cosine is computed on the pooled mean of attested "
                  "context embeddings. No Mantel test is required on "
                  "this slice: the claim is a contrast between two "
                  "encodings of the same term set."),
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_G_false_friends.py", "main()")],
        ) + \
        ui.section_close()


def _section_H() -> str:
    H_table = ext_loader.h_saturation_table()
    fig = ext_fig.fig_H_k_saturation(H_table)
    return ui.section_open("H-k-saturation",
                            "H · K saturation: why K ≥ 4?") + \
        ui.scenario_block(
            "Each attested embedding for a term is the mean of K real "
            "ordinance contexts. If K is too small, the representation "
            "is dominated by noise; if K is large, it is dominated by "
            "the corpus statistic. What K makes the cross-tradition "
            "signal stable?"
        ) + \
        ui.result_block(
            "At K = 1 the cross-tradition ρ̄ is negative "
            "(<strong>−0.13</strong>) — the signal is below noise. "
            "At K = 4 it reaches the saturation band; it climbs "
            "modestly through K = 8 (<strong>+0.22</strong>). "
            "The K ≥ 4 threshold used throughout the chapter is "
            "empirically justified."
        ) + \
        ui.plot_block(fig, "fig-H-k-saturation", height_px=440,
                       caption="ρ̄_cross attested on a single bilingual "
                                "comparison (BGE-EN-large × BGE-ZH-large) "
                                "with the background pool partitioned by "
                                "K_min. The K ≥ 4 band frames the operating "
                                "regime adopted in §2.3.") + \
        ui.takehome_block(
            "K ≥ 4 is not a convention: at lower K the cross-tradition "
            "signal is anti-correlated with itself across pool replicates. "
            "Chapter 2 §2.3 cites H as the operational justification for "
            "the threshold."
        ) + \
        apparatus_block(
            stats=[
                ("ρ̄_cross at K=1", f"{H_table[0]['mean_rho_cross']:+.3f}"),
                ("ρ̄_cross at K=4-7",
                 f"+{[r['mean_rho_cross'] for r in H_table if r['K']=='4-7'][0]:.3f}"),
                ("ρ̄_cross at K=8", f"+{H_table[-1]['mean_rho_cross']:.3f}"),
                ("core reference", "0.246"),
            ],
            meta=("All buckets evaluated on the same encoder pair; only "
                  "the bg subsample varies. n_pairs = 1 per bucket "
                  "(single encoder pair)."),
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_H_k_saturation.py", "main()")],
        ) + \
        ui.section_close()


# ==========================================================================
# Y caveat — the anchor of this page

def _section_Y() -> str:
    nums = ext_loader.y_caveat_numbers()
    return ui.section_open("Y-caveat",
                            "Y · The legal signal is the gap, not the absolute") + \
        ui.scenario_block(
            "The headline Δρ_sym attested = 0.543 describes the "
            "cross-tradition gap in the attested encoding of the legal "
            "lexicon. Is this gap a property of legal meaning, or is "
            "part of it already present in the encoders before any "
            "ordinance context is added? To isolate the legal "
            "contribution, the experiment computes a methodological "
            "baseline: the same Δρ_sym on the bare encoding (lemma in "
            "isolation, no context), both on the 364 core terms and on "
            "100 non-legal control words (pronouns, deixis, common "
            "nouns)."
        ) + \
        ui.result_block(
            "The bare baseline on the core returns "
            f"<strong>{nums['bare_core']:.3f}</strong>: an "
            "encoder-tradition signal with no legal content. The same "
            "baseline computed on 100 non-legal control words returns "
            f"<strong>{nums['bare_control']:.3f}</strong>, statistically "
            "indistinguishable from the core baseline, confirming that "
            "the bare gap is shaped by the encoders, not by the legal "
            "vocabulary. The legal signal is the contribution that "
            "attestation adds on top: "
            f"<strong>{nums['legal_gap']:.3f} = {nums['attested_core']:.3f} − "
            f"{nums['bare_core']:.3f}</strong>."
        ) + \
        ui.number_callout(
            f"{nums['legal_gap']:.3f}",
            ("Legal-meaning signal · attested core − bare core · "
             f"<strong>{nums['attested_core']:.3f} − "
             f"{nums['bare_core']:.3f}</strong>. "
             "This is the quantity attributable to context-bound "
             "legal attestation (the HK ordinances), against an "
             "encoder-tradition baseline of approximately 0.16."),
        ) + \
        ui.data_table(
            columns=("Signal", "Δρ_sym", "Interpretation"),
            rows=[
                ("Encoder-tradition baseline (bare, 364 core)",
                 f"{nums['bare_core']:.3f}",
                 "cross-tradition gap with no legal context, "
                 "intrinsic to the encoders"),
                ("Verification of the baseline (bare, 100 controls)",
                 f"{nums['bare_control']:.3f}",
                 "indistinguishable from the core baseline: the bare "
                 "gap is encoder-shaped, not legal-shaped"),
                ("Full pipeline output (attested, 364 core)",
                 f"{nums['attested_core']:.3f}",
                 "the result the experiments analyse"),
                ("Legal-meaning signal (isolated)",
                 f"{nums['legal_gap']:.3f}",
                 f"attested − bare = {nums['attested_core']:.3f} − "
                 f"{nums['bare_core']:.3f}, the contribution attributable "
                 f"to HK ordinance attestation"),
            ],
            col_classes=("", "num strong", ""),
            row_classes=("", "", "", "highlight"),
        ) + \
        ui.takehome_block(
            "Chapter 4 §4.1 should cite both numbers — the "
            f"absolute attested ({nums['attested_core']:.3f}) and the "
            f"legal-meaning gap ({nums['legal_gap']:.3f}) — and §4.2 "
            "should disclose the Y caveat as the primary methodological "
            "limit of the experiment."
        ) + \
        apparatus_block(
            formula=(
                "legal signal = Δρ<sub>sym</sub><sup>attested</sup>(core) "
                "− Δρ<sub>sym</sub><sup>bare</sup>(core) "
                "= 0.543 − 0.165 = 0.378"
            ),
            stats=[
                ("attested core", f"{nums['attested_core']:.3f}"),
                ("bare core",     f"{nums['bare_core']:.3f}"),
                ("bare control",  f"{nums['bare_control']:.3f}"),
                ("legal gap",     f"{nums['legal_gap']:.3f}"),
            ],
            meta=("Control terms are everyday vocabulary "
                  "(pronouns, deixis, common nouns) — they have no HK "
                  "ordinance attestation by design, so the comparison "
                  "is necessarily bare-on-bare. Run #3 reported the same "
                  "qualitative pattern."),
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_Y_control_only.py", "main()")],
            sources=[
                "Caveat phrased in the thesis inferential discipline.",
                "Adopted as primary §4.2 limit.",
            ],
        ) + \
        ui.section_close()


# ==========================================================================
# Secondary caveats: F, X, Z

def _section_F() -> str:
    F_table = ext_loader.f_confidence_table()
    fig = ext_fig.fig_F_confidence_bars(F_table)
    return ui.section_open("F-confidence",
                            "F · Confidence-stratified injection") + \
        ui.scenario_block(
            "When background terms are injected into the pool, do "
            "high-confidence ones — clearly belonging to a single legal "
            "domain — boost the signal more than low-confidence ones, "
            "or less?"
        ) + \
        ui.result_block(
            "Low-confidence (semantically ambiguous) background "
            "increases the signal slightly; high-confidence "
            "(categorical) background suppresses it slightly. "
            "Both shifts are within ± 0.03, with overlapping CIs. "
            "Interpret as a hint, not a finding."
        ) + \
        ui.plot_block(fig, "fig-F-confidence", height_px=400,
                       caption="Baseline = core-only. Injection strata "
                                "n_inject = 91, n_replicates = 20. "
                                "Error bars are 95% CI across replicates.") + \
        ui.takehome_block(
            "Counter-intuitively, the signal sits at the boundary of the "
            "pool, not in its centre. Effect size is small; Chapter 4 "
            "may cite F as an interpretive hint but not as evidence."
        ) + \
        apparatus_block(
            stats=[(r["stratum"], f"{r['mean_delta_sym']:.3f}")
                   for r in F_table],
            meta=("Strata sized to the deciles of k-NN domain-assignment "
                  "confidence (extension A); see <em>ext_A_bg_knn</em>."),
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_F_confidence.py", "main()")],
            collapsible=True,
        ) + \
        ui.section_close()


def _section_X() -> str:
    X_table = ext_loader.x_robustness_table()
    fig = ext_fig.fig_X_control_curve(X_table)
    return ui.section_open("X-control",
                            "X · Dual robustness — injection of control terms") + \
        ui.scenario_block(
            "D injected legal background terms (still legal, just "
            "un-curated). What happens if we inject genuinely non-legal "
            "vocabulary into the bare pool?"
        ) + \
        ui.result_block(
            f"Δρ_sym bare drops monotonically from "
            f"<strong>{X_table[0]['mean_delta_sym']:.3f}</strong> at 0% "
            f"control to <strong>{X_table[-1]['mean_delta_sym']:.3f}</strong> "
            "at the upper limit. The direction is right; the magnitude "
            "is small because the baseline itself is small."
        ) + \
        ui.plot_block(fig, "fig-X-control-robustness", height_px=400,
                       caption="Dual of D, computed on bare encodings "
                                "only (controls have no HK ordinance "
                                "attestation). 15 replicates per level.") + \
        ui.takehome_block(
            "Bare Δρ_sym is sensitive to pool contamination but only "
            "weakly. Chapter 4 cites D and X together as a paired "
            "robustness statement: attested is stable upward, bare "
            "decays downward."
        ) + \
        apparatus_block(
            stats=[("Δρ_sym at 0% control",
                    f"{X_table[0]['mean_delta_sym']:.3f}"),
                   ("Δρ_sym at upper limit",
                    f"{X_table[-1]['mean_delta_sym']:.3f}"),
                   ("replicates / level",
                    str(X_table[0]['n_replicates']))],
            meta=("Control pool capped at 100; the largest p_control "
                  "actually reachable is approximately 27%."),
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_X_control_robustness.py", "main()")],
            collapsible=True,
        ) + \
        ui.section_close()


def _section_Z() -> str:
    Z_table = ext_loader.z_tier_table()
    fig = ext_fig.fig_Z_tier_medians(Z_table)
    n_monotonic = sum(1 for r in Z_table if r["monotonic"])
    return ui.section_open("Z-tier-hierarchy",
                            "Z · The naive tier hierarchy does not hold") + \
        ui.scenario_block(
            "Intuition says: core terms should cluster more tightly than "
            "background terms, which in turn should cluster more tightly "
            "than the everyday-language control. The corpus pipeline was "
            "designed under that expectation. Does the geometry agree?"
        ) + \
        ui.result_block(
            f"Only <strong>{n_monotonic} of {len(Z_table)}</strong> "
            "models satisfy the monotonic hierarchy "
            "median(core-core) &lt; median(core-bg) &lt; median(core-control). "
            "The other models place background terms farther from core "
            "than the control vocabulary. Tier classification is a "
            "curatorial property of the corpus, not a geometric one."
        ) + \
        ui.plot_block(fig, "fig-Z-tier-hierarchy", height_px=480,
                       caption="Per-model median cosine distance on the "
                                "three pair populations. ✓ marks models "
                                "with monotonic hierarchy.") + \
        ui.takehome_block(
            "Background and control occupy distinct, model-dependent "
            "regions of the embedding. Chapter 4 §4.2 should disclose Z "
            "as an honest limit: the corpus tiers are operational, not "
            "semantic."
        ) + \
        apparatus_block(
            stats=[("n models",   str(len(Z_table))),
                   ("monotonic",  f"{n_monotonic} / {len(Z_table)}"),
                   ("test",       "Mann-Whitney on cosine medians")],
            meta=("Distances computed on bare encodings (controls are "
                  "bare-only). Hierarchy boolean as reported by "
                  "<code>ext_Z_tier_hierarchy</code>."),
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_Z_tier_hierarchy.py", "main()")],
            collapsible=True,
        ) + \
        ui.section_close()


# ==========================================================================
# Appendix: A, E (utilities)

def _appendix() -> str:
    A = ext_loader.load_A_bg_knn()
    E = ext_loader.load_E_axes_oos()
    fig_A = ext_fig.fig_A_bg_domain_distribution(A)
    return ui.section_open("appendix", "Appendix · supporting utilities") + """
<h3>A · k-NN domain assignment of the 9 045 background terms</h3>

<p>
The k-NN assigner labels each background term with the legal domain
voted by its seven nearest core neighbours. Mean assignment confidence
is 0.515, with a long-tailed distribution biased toward
<em>procedure</em> and <em>criminal</em>. The output supports F's
confidence stratification, but is itself a reusable resource for
future pool expansion.
</p>
""" + ui.plot_block(fig_A, "fig-A-bg-domains", height_px=400,
                     caption="Domain distribution of the 9 045 bg terms, "
                              "k = 7 (BGE-EN-large).") + \
        apparatus_block(
            stats=[("n_bg", f"{A['meta']['n_bg']:,}"),
                   ("k",   str(A['meta']['k'])),
                   ("metric", A['meta'].get('metric', 'cosine'))],
            meta="See CSV under <code>ext/A_bg_knn/background_assignments.csv</code>.",
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_A_bg_knn.py", "main()")],
            collapsible=True,
        ) + """
<h3>E · Out-of-sample coherence of the 6 axes on background</h3>

<p>
The six Kozlowski axes constructed on the curated 364-term pool were
projected onto the 9 045 background terms (attested). Within-domain
mean / std are stable for the three pool-robust axes
(<em>individual_collective</em>, <em>public_private</em>,
<em>natural_positive</em>) and noisier for the three pool-sensitive
ones (<em>rights_duties</em>, <em>status_contract</em>,
<em>state_market</em>). The pool-sensitivity flag of §3.2.4 generalises
beyond the curated pool.
</p>
""" + apparatus_block(
            stats=[("n_bg", f"{E['meta']['n_bg']:,}"),
                   ("axes", "6"),
                   ("variant", E['meta'].get('variant', 'attested'))],
            meta="Per-domain mean / std stored in <code>per_model_per_axis_per_domain</code> "
                 "of <code>ext/E_axes_oos/coherence.json</code>.",
            code_ref=[("experiments/ch3-measurability/scripts/"
                       "ext_E_axes_oos.py", "main()")],
            collapsible=True,
        ) + ui.section_close()


# ==========================================================================
# build()

def build() -> str:
    parts = [
        ui.page_head(
            title="Robustness & caveats",
            subtitle="Seven extensions that strengthen — and qualify — "
                     "the headline numbers.",
            crumb="Chapter 3 · Extensions A–Z",
        ),
        ui.sticky_nav(current_href="robustness_caveats.html"),
        ui.open_main(),
        _intro(),
        _section_D(),
        _section_G(),
        _section_H(),
        _section_Y(),
        _section_F(),
        _section_X(),
        _section_Z(),
        _appendix(),
        ui.linear_nav(
            prev=("experiment_32.html", "Experiment §3.2"),
            next_=None,
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
