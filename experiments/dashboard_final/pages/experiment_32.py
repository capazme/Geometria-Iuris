"""Experiment §3.2 — projecting legal terms onto value axes.

Five sub-sections of the thesis: §3.2.1 axis construction; §3.2.2
inter-axis independence; §3.2.3 agreement on the ranking; §3.2.4
cross-linguistic agreement; §3.2.5 between-group differences.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import shared_ui as ui  # noqa: E402
from apparatus import apparatus_block  # noqa: E402
from data import loader_32  # noqa: E402
from figures import exp32 as figs  # noqa: E402


_AXIS_LABELS = loader_32.AXIS_LABELS


def _axis_table_rows(means: dict, ordering: list[str]) -> list:
    return [(_AXIS_LABELS.get(a, a), f"{float(means[a]):.3f}") for a in ordering]


# --------------------------------------------------------------------------
# Intro

def _intro(meta: dict) -> str:
    return ui.section_open("intro", "Experiment §3.2 — Value axes") + """
<p class="lead">
The second experiment projects the 364 legal terms onto six axes
drawn from legal doctrine and political theory:
<em>individual ↔ collective</em>, <em>rights ↔ duties</em>,
<em>public ↔ private</em>, <em>state ↔ market</em>,
<em>natural ↔ positive</em>, <em>status ↔ contract</em>. Each axis is
built from ten antonym pairs (Kozlowski, Taddy &amp; Evans, 2019) in
each language — English on the Western-trained side, Chinese on the
Chinese-trained side. The axis is the L2-normalised mean of the ten
pole-difference vectors; each term receives a signed score on each
axis as its cosine with the axis direction.
</p>

<p>
The section answers four nested questions. Are the axes coherent —
do the ten antonym pairs that built each axis project on the side
they were chosen to represent (§3.2.1)? Are the six axes independent
directions in the embedding space, or do they partly overlap
(§3.2.2)? When two models score the same 364 terms on the same axis,
do their rankings agree more within a tradition than across (§3.2.3,
§3.2.4)? And on which axes do the two traditions diverge most
(§3.2.4, §3.2.5)?
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# §3.2.1

def _section_321(s321: dict) -> str:
    fig = figs.fig_sanity_heatmap(s321, variant="attested")
    return ui.section_open("s321",
                            "§3.2.1 · Building an axis from pairs of opposites") + \
        ui.scenario_block(
            "Legal doctrine treats contracts as the antithesis of "
            "status, rights as the antithesis of duties. If a "
            "language model encodes legal vocabulary, the ten antonym "
            "pairs that define each axis should mostly project on the "
            "side they were chosen to represent."
        ) + \
        ui.result_block(
            "On the six axes and six monolingual models, the great "
            "majority of cells return ten out of ten (or nine out of "
            "ten) antonym pairs aligned with their nominal pole; no "
            "cell falls below seven of ten. The few soft cells flag "
            "axes where one specific pole word sits closer to its "
            "opposite under the contextualisation of Hong Kong "
            "ordinances — informative about doctrinal placement, not "
            "about axis formation."
        ) + \
        ui.plot_block(fig, "fig-321-sanity", height_px=420,
                       caption='Sanity heatmap: ratio of antonym pairs '
                                'aligned with their nominal pole, one '
                                'cell per axis × model (attested '
                                'encoding). The ten English and ten '
                                'Chinese antonym pairs that build each '
                                'axis are listed verbatim under '
                                '<a href="lexicon.html#axes">Inside the '
                                'inputs</a>.') + \
        ui.takehome_block(
            "Axis construction is sound on the majority of cells. "
            "The few low-pass cells are the early warning of the "
            "pool-sensitivity reported in §3.2.4."
        ) + \
        apparatus_block(
            formula=(
                "axis<sub>+</sub> − axis<sub>−</sub> = "
                "mean<sub>k = 1..10</sub> "
                "(emb(positive<sub>k</sub>) − emb(negative<sub>k</sub>))"
            ),
            stats=[("axes",     "6"),
                   ("antonym pairs / axis", "10"),
                   ("languages", "English + Chinese"),
                   ("models",    "5 EN + 5 ZH")],
            meta=("Pass means at least half the ten pairs project on "
                  "the side they were chosen for. The averaging step "
                  "follows Kozlowski, Taddy &amp; Evans (2019)."),
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.2.2

def _section_322(s322: dict) -> str:
    fig = figs.fig_orthogonality(s322, model="BGE-EN-large",
                                  variant="attested")
    return ui.section_open("s322",
                            "§3.2.2 · Axes independence") + \
        ui.scenario_block(
            "<em>Public ↔ private</em> and <em>state ↔ market</em> "
            "sound related; <em>rights ↔ duties</em> and "
            "<em>individual ↔ collective</em> do too. If two axes "
            "occupied the same direction in embedding space, "
            "projecting the 364 terms on both would be redundant. How "
            "much do the six axes share?"
        ) + \
        ui.result_block(
            "On the representative reading (BGE-EN-large, attested), "
            "the fifteen off-diagonal cosines range between −0.21 and "
            "+0.34 in signed value, with mean magnitude 0.13. No pair "
            "is collinear; no pair is exactly orthogonal. "
            "<em>Individual / collective</em> and "
            "<em>rights / duties</em> are the most aligned (<em>cos</em> "
            "≈ +0.34, the &ldquo;individual&rdquo; pole near the "
            "&ldquo;rights&rdquo; pole); <em>natural / positive</em> "
            "is the closest of the six to a direction independent of "
            "the others."
        ) + \
        ui.plot_block(fig, "fig-322-ortho", height_px=480,
                       caption="Inter-axis cosine matrix, attested "
                                "encoding. The diagonal is unity by "
                                "construction; off-diagonal values are "
                                "signed cosines. Use the dropdown to "
                                "switch model — the qualitative shape "
                                "of the matrix recurs across "
                                "traditions, the magnitudes vary.") + \
        ui.takehome_block(
            "Six axes occupy six distinct directions: the per-axis "
            "readings of §3.2.4 are not six restatements of the same "
            "measurement. Doctrinal proximity (individual / rights, "
            "state / status, public / state) shows up as moderate but "
            "bounded alignment, never as collinearity."
        ) + \
        apparatus_block(
            formula="cos(axis<sub>i</sub>, axis<sub>j</sub>) = "
                    "axis<sub>i</sub> · axis<sub>j</sub> / "
                    "(‖axis<sub>i</sub>‖ × ‖axis<sub>j</sub>‖)",
            stats=[("axes", "6"),
                   ("matrix", "6 × 6 symmetric"),
                   ("display model", "BGE-EN-large")],
            meta=("Off-diagonal range on the representative model: "
                  "[−0.21, +0.34], mean magnitude 0.13. The full per-model "
                  "matrices are referenced from §3.2.2 of the thesis."),
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.2.3

def _section_323(s323: dict) -> str:
    fig = figs.fig_axes_boxplot(s323, variant="attested")
    pp = s323["attested"]["per_pair"]
    n_pairs_per_axis = len(pp) // 6
    return ui.section_open("s323",
                            "§3.2.3 · Agreement on the ranking") + \
        ui.scenario_block(
            "Each model assigns a signed score to each of the 364 "
            "terms on each of the six axes. When two models are "
            "compared on the same axis, their rankings of the 364 "
            "terms should correlate strongly within a tradition and "
            "more weakly across — if the axes track tradition-specific "
            "legal meaning."
        ) + \
        ui.result_block(
            f"Distribution of Spearman ρ across {n_pairs_per_axis} "
            "pairs per axis, attested encoding. Within-tradition "
            "rankings agree strongly on most axes; cross-tradition "
            "rankings agree more weakly — and the spread varies "
            "systematically by axis. The §3.1.3 cohort-level pattern "
            "transfers to the per-axis level."
        ) + \
        ui.plot_block(fig, "fig-323-box", height_px=480,
                       caption="Box plot of per-pair ρ for each axis, "
                                "attested. Points are individual model "
                                "pairs, coloured by group.") + \
        ui.takehome_block(
            "Per-pair ρ distributions are tight within tradition and "
            "broader across. The single-axis projection picks up the "
            "same structural agreement that §3.1.3 reads on the full "
            "distance map."
        ) + \
        apparatus_block(
            stats=[("pairs total", str(len(pp))),
                   ("pairs / axis", f"{n_pairs_per_axis}"),
                   ("axes",         "6"),
                   ("metric",       "Spearman ρ on the 364-term rank vector")],
            meta="Each per-pair entry stores a 95% confidence interval "
                 "from term-level block bootstrap (B = 10 000).",
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.2.4 — the substantive ranking

def _section_324(s324: dict) -> str:
    fig_ranking = figs.fig_axes_ranking_toggle(s324)
    fig_cmp = figs.fig_axes_ranking_compare(s324)
    means_att = s324["attested"]["cross_rho_mean_per_axis"]
    ranking = sorted(means_att.items(), key=lambda r: r[1])
    most = ranking[0]
    least = ranking[-1]
    return ui.section_open("s324",
                            "§3.2.4 · Cross-linguistic agreement — which axes diverge most?") + \
        ui.scenario_block(
            "Among the six axes, which is the <em>least</em> shared "
            "across traditions — the one on which Western-trained and "
            "Chinese-trained models rank the 364 terms most "
            "differently? And by which margin? Doctrine has its own "
            "intuitions: natural-versus-positive law, "
            "state-versus-market allocation, individual-versus-"
            "collective normative weight all carry tradition-specific "
            "framing."
        ) + \
        ui.result_block(
            f"On attested encodings, the <strong>most divergent</strong> "
            f"axis is <em>{most[0].replace('_', ' ↔ ')}</em> with "
            f"cross-tradition mean ρ = <strong>{most[1]:.3f}</strong>. "
            f"The <strong>least divergent</strong> is "
            f"<em>{least[0].replace('_', ' ↔ ')}</em> with mean ρ = "
            f"<strong>{least[1]:.3f}</strong>. The ranking is "
            "curation-sensitive on three of the six axes: "
            "<em>individual / collective</em>, <em>public / private</em> "
            "and <em>natural / positive</em> hold their cross-tradition "
            "mean under pool perturbation, while "
            "<em>rights / duties</em>, <em>status / contract</em> and "
            "<em>state / market</em> shift substantially when the "
            "curated pool shifts. The pool sensitivity is itself the "
            "methodological reading of §4.2 of the thesis."
        ) + \
        ui.plot_block(fig_ranking, "fig-324-ranking", height_px=440,
                       caption="Cross-tradition mean ρ per axis, most "
                                "divergent on top. Use the toggle "
                                "above the chart to switch between "
                                "attested (default) and bare encoding "
                                "— five of the six axes diverge more "
                                "under attestation; rights / duties "
                                "is the only axis on which the two "
                                "encodings coincide.") + \
        ui.plot_block(fig_cmp, "fig-324-cmp", height_px=480,
                       caption="The same data, bare and attested side "
                                "by side. The exception "
                                "(rights / duties) is visible as the "
                                "axis where the two bars overlap.") + \
        ui.data_table(
            columns=("Axis", "Cross-tradition ρ̄ (attested)"),
            rows=_axis_table_rows(means_att, [r[0] for r in ranking]),
            col_classes=("", "num strong"),
        ) + \
        ui.disclaimer(
            "<strong>Pool-sensitivity warning.</strong> "
            "On <em>rights / duties</em>, <em>status / contract</em> "
            "and <em>state / market</em> the cross-tradition ρ̄ shifts "
            "by more than 0.05 under realistic pool perturbations, "
            "so the rank order of these three axes is not invariant. "
            "On <em>individual / collective</em>, <em>public / "
            "private</em> and <em>natural / positive</em> the ρ̄ is "
            "stable under the same perturbations. Read the ranking as "
            "an ordering within an internally-scaled column, not as a "
            "cross-axis comparison of magnitudes (§4.2)."
        ) + \
        ui.takehome_block(
            "Three axes carry a stable cross-tradition signature, "
            "three are sensitive to curation. The reading is "
            "ordinal — the ranking is the substantive measurement — "
            "not metric: rights / duties at 0.394 is not &ldquo;twice "
            "as agreed&rdquo; as individual / collective at 0.186."
        ) + \
        apparatus_block(
            stats=[("most divergent",  f"{most[0]} · {most[1]:.3f}"),
                   ("least divergent", f"{least[0]} · {least[1]:.3f}"),
                   ("axes total",      "6"),
                   ("cross pairs / axis", "9 monolingual (3 EN × 3 ZH)")],
            meta=("Each cell is the mean Spearman ρ between the 364-term "
                  "rankings of one English-side model and one Chinese-side "
                  "model on the same axis."),
            sources=[
                "Kozlowski, Taddy &amp; Evans (2019) — axis-construction recipe.",
            ],
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.2.5 — between-group differences (term-level exhibit)

def _section_325(s325: dict) -> str:
    fig_terms = figs.fig_top_divergent_terms_explorer(s325,
                                                       variant="attested",
                                                       top_k=10)
    return ui.section_open("s325",
                            "§3.2.5 · Between-group differences") + \
        ui.scenario_block(
            "Beyond an axis-level ρ, the substantive question is: "
            "which individual terms anchor the cross-tradition "
            "divergence? Their identity tells a doctrinal story that "
            "the aggregate ρ cannot."
        ) + \
        ui.result_block(
            "On natural / positive, the most divergent terms are "
            "<em>prejudice</em>, <em>discrimination</em>, "
            "<em>punishment</em>, <em>religion</em>, <em>perjury</em>: "
            "the Western-trained reading sends them toward the "
            "&ldquo;natural&rdquo; pole (the offence is wrong before "
            "being criminalised); the Chinese-trained reading sends "
            "the same five toward the &ldquo;positive&rdquo; pole "
            "(the offence is the statutory text that criminalises). "
            "On individual / collective, the divergence reverses "
            "polarity: terms of bilateral private-law relation "
            "(<em>compensation</em>, <em>obligation</em>, "
            "<em>counterparty</em>) sit on the individual side for "
            "the Western-trained models and on the collective side "
            "for the Chinese-trained ones. On rights / duties, both "
            "readings of the term <em>freedom</em> / 自由 collapse to "
            "the same Chinese lemma but project a quarter-axis apart "
            "(|Δ| ≈ 0.30). Use the axis dropdown to browse the top "
            "ten divergent terms on each axis."
        ) + \
        ui.plot_block(fig_terms, "fig-325-terms", height_px=560,
                       caption="Per-axis top-ten cross-tradition "
                                "divergent terms, sorted by |Δ| from "
                                "largest at the top. Blue bar: mean "
                                "projection of the term across the "
                                "three Western-trained models. Red "
                                "bar: mean projection across the "
                                "three Chinese-trained models. |Δ| is "
                                "in the hover. Use the axis dropdown "
                                "to switch between the six axes "
                                "(attested encoding).") + \
        ui.takehome_block(
            "Term-level and axis-level divergence converge: where the "
            "axis ρ̄ is lowest, the top divergent terms project on "
            "opposite poles; where the axis ρ̄ is highest, the "
            "divergence concentrates on a single term ("
            "<em>freedom</em> / 自由). §3.2.5 of the thesis is the "
            "exhibit catalogue."
        ) + \
        apparatus_block(
            stats=[("axes", "6"),
                   ("top divergent terms / axis", "5"),
                   ("ranking quantity",
                    "|Δ(t,a)| = |W(t,a) − S(t,a)|")],
            meta=("|Δ| values are not comparable across axes: each "
                  "axis carries its own projection scale. The intra-"
                  "axis ranking is meaningful; the cross-axis "
                  "magnitude comparison is not (§4.2)."),
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# build()

def build() -> str:
    data = loader_32.load_all()
    parts = [
        ui.page_head(
            title="Experiment §3.2 — Value axes",
            subtitle="Projecting 364 legal terms onto six axes drawn "
                     "from doctrine and political theory.",
            crumb="Chapter 3 · §3.2",
        ),
        ui.sticky_nav(current_href="experiment_32.html"),
        ui.open_main(),
        _intro(data["s321"]["meta"]),
        _section_321(data["s321"]),
        _section_322(data["s322"]),
        _section_323(data["s323"]),
        _section_324(data["s324"]),
        _section_325(data["s325"]),
        ui.linear_nav(
            prev=("experiment_31.html", "Experiment §3.1"),
            next_=("robustness_caveats.html", "Robustness &amp; caveats"),
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
