"""Experiment 3.2 — projecting legal terms onto value axes.

Five sub-sections: §3.2.1 axis construction & sanity; §3.2.2 axes
independence; §3.2.3 agreement on the ranking; §3.2.4 cross-linguistic
agreement; §3.2.5 between-group differences.
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
    axes_str = ", ".join(loader_32.AXES_ORDER)
    return ui.section_open("intro", "Experiment §3.2 — Value axes") + f"""
<p class="lead">
The second experiment projects the 364 legal terms onto six pre-defined
value axes drawn from legal doctrine and political theory:
<em>individual ↔ collective</em>, <em>rights ↔ duties</em>,
<em>public ↔ private</em>, <em>state ↔ market</em>,
<em>natural ↔ positive</em>, <em>status ↔ contract</em>. Each axis is
constructed (Kozlowski et al. 2019) as the centroid of cosine
differences over a list of antonymic seed pairs, in both English and
Chinese. Each term then receives a signed score on each axis.
</p>

<p>
The Experiment asks three nested questions. Are the axes coherent —
do the seed pairs vote for the same direction within an axis (§3.2.1)?
Are they independent — do different axes capture different conceptual
dimensions (§3.2.2)? When encoders project the same 364 terms on the
same six axes, do models in the same tradition agree more than models
across traditions (§3.2.3, §3.2.4)? And on which axes do the two
traditions diverge most (§3.2.5)?
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# §3.2.1

def _section_321(s321: dict) -> str:
    fig = figs.fig_sanity_heatmap(s321, variant="attested")
    return ui.section_open("s321",
                            "§3.2.1 · Building an axis from pairs of opposites") + \
        ui.scenario_block(
            "Doctrine speaks of contracts as the antithesis of status; "
            "of rights as the antithesis of duties. If an embedding "
            "model captures the legal vocabulary, the antonymic seed "
            "pairs that define each axis should mostly vote for the "
            "same direction."
        ) + \
        ui.result_block(
            "Most axis × model combinations achieve high sanity-pass "
            "fractions (5 / 5 seed pairs aligned, or 4 / 5). A small "
            "number of cells fall below 3 / 5 — typically Sinic models "
            "on axes whose English seed pairs translate awkwardly. The "
            "diagnostic is informative: pool-sensitivity of §3.2.4 "
            "begins here."
        ) + \
        ui.plot_block(fig, "fig-321-sanity", height_px=420,
                       caption="Heatmap of positive_correct / "
                                "n_pairs_total per axis × model "
                                "(attested encoding).") + \
        ui.takehome_block(
            "Axis construction is sound on the majority of cells. The "
            "few low-pass cells flag which axes will turn out "
            "pool-sensitive in §3.2.4."
        ) + \
        apparatus_block(
            formula=(
                "axis<sub>+</sub> − axis<sub>−</sub> = "
                "mean<sub>k ∈ seeds</sub> "
                "(emb(positive<sub>k</sub>) − emb(negative<sub>k</sub>))"
            ),
            stats=[("axes",     "6"),
                   ("seed pairs / axis", "≤ 20"),
                   ("languages", "EN + ZH (parallel pairs)"),
                   ("models",    "5 EN + 5 ZH")],
            meta=("Sanity check: each seed pair contributes a "
                  "positive_correct (its English positive should score "
                  "higher than its negative). Pass = positive_correct "
                  "≥ 0.5 × n_pairs_used."),
            code_ref=[("experiments/ch3-measurability/scripts/",
                       "experiment_2_axes.py")],
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
            "sound related. If they were captured by the same direction "
            "in embedding space, scoring a term on both would be "
            "redundant. How orthogonal are the six axes?"
        ) + \
        ui.result_block(
            "The 6 × 6 inter-axis cosine matrix is dominated by values "
            "between −0.2 and +0.4. No two axes are collinear; no two "
            "are perfectly orthogonal either. <em>Rights-duties</em> and "
            "<em>individual-collective</em> are the most correlated "
            "pair, as doctrine predicts; <em>natural-positive</em> and "
            "<em>state-market</em> are the most independent."
        ) + \
        ui.plot_block(fig, "fig-322-ortho", height_px=460,
                       caption="Inter-axis cosine for BGE-EN-large "
                                "(attested). Diagonal = 1 by construction; "
                                "off-diagonal values are signed cosines "
                                "between axis vectors.") + \
        ui.takehome_block(
            "Six axes carry six distinct directions of variation. "
            "Doctrinal proximity (rights / duties, individual / "
            "collective) shows up as moderate but bounded correlation. "
            "Chapter 4 §4.2 cites this as evidence the axes are "
            "linearly independent enough to be reported separately."
        ) + \
        apparatus_block(
            formula="cos(axis<sub>i</sub>, axis<sub>j</sub>) = "
                    "axis<sub>i</sub> · axis<sub>j</sub> / "
                    "(‖axis<sub>i</sub>‖ × ‖axis<sub>j</sub>‖)",
            stats=[("axes", "6"),
                   ("matrix", "6 × 6 symmetric"),
                   ("display model", "BGE-EN-large")],
            meta="Per-model inter-axis matrix stored in "
                 "<code>section_322.&lt;model&gt;.cosine_matrix</code>. "
                 "Diagonal not enforced post hoc — it is 1.0 by axis-vector "
                 "self-cosine.",
            code_ref=[("experiments/ch3-measurability/experiment_2_axes/"
                       "results_attested/", "experiment_2_results.json")],
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
            "Each encoder ranks the 364 legal terms along each of the "
            "six axes. When two encoders are compared on the same axis, "
            "their rankings should correlate strongly within a tradition "
            "and weakly across traditions — if the axes track "
            "tradition-specific legal meaning."
        ) + \
        ui.result_block(
            f"Distributions of Spearman ρ across {n_pairs_per_axis} "
            "pairs per axis, attested. Within-tradition rankings agree "
            "strongly on most axes; cross-tradition rankings agree "
            "weakly — and the spread varies systematically by axis "
            "(visible in §3.2.4)."
        ) + \
        ui.plot_block(fig, "fig-323-box", height_px=480,
                       caption="Box plot of per-pair ρ for each axis "
                                "(attested). Points are individual "
                                "model-pair entries, coloured by group.") + \
        ui.takehome_block(
            "Per-pair ρ distributions are tight within tradition, "
            "broader across. The structure transfers from §3.1.3 (RSA "
            "on the full RDM) to §3.2 (RSA on each one-dimensional "
            "projection)."
        ) + \
        apparatus_block(
            stats=[("pairs total", str(len(pp))),
                   ("pairs / axis", f"{n_pairs_per_axis}"),
                   ("axes",         "6"),
                   ("metric",       "Spearman ρ on 364-vector ranks")],
            meta="Each per-pair entry stores 95% CI from bootstrap "
                 "B = 10 000 (block bootstrap on terms).",
            code_ref=[("experiments/ch3-measurability/experiment_2_axes/"
                       "results_attested/", "experiment_2_results.json")],
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.2.4 — the substantive ranking

def _section_324(s324: dict) -> str:
    fig_cmp = figs.fig_axes_ranking_compare(s324)
    fig_att = figs.fig_axes_ranking(s324, variant="attested",
                                     sort_ascending=True)
    means_att = s324["attested"]["cross_rho_mean_per_axis"]
    ranking = sorted(means_att.items(), key=lambda r: r[1])
    most = ranking[0]
    least = ranking[-1]
    return ui.section_open("s324",
                            "§3.2.4 · Cross-linguistic agreement (which axes diverge most?)") + \
        ui.scenario_block(
            "Among the six axes, which is the <em>least</em> shared "
            "across traditions — the one where WEIRD and Sinic models "
            "rank the 364 terms most differently? Doctrine has its own "
            "intuitions: natural-vs-positive law, rights-vs-duties, "
            "state-vs-market all carry tradition-specific weight."
        ) + \
        ui.result_block(
            f"On attested encodings, the <strong>most divergent</strong> "
            f"axis is <em>{most[0].replace('_', ' ↔ ')}</em> with "
            f"cross-tradition ρ̄ = <strong>{most[1]:.3f}</strong>. "
            f"The <strong>least divergent</strong> is "
            f"<em>{least[0].replace('_', ' ↔ ')}</em> with "
            f"ρ̄ = <strong>{least[1]:.3f}</strong>. "
            "The ranking is curation-sensitive: three axes "
            "(individual-collective, public-private, natural-positive) "
            "hold their cross-tradition ρ̄ under pool perturbation, "
            "while three (rights-duties, status-contract, state-market) "
            "shift substantially when the curated pool shifts. The "
            "axis-level finding is the sensitivity itself, not a fixed "
            "ordering."
        ) + \
        ui.plot_block(fig_att, "fig-324-att", height_px=400,
                       caption="Cross-tradition ρ̄ per axis (attested). "
                                "Most divergent on top.") + \
        ui.plot_block(fig_cmp, "fig-324-cmp", height_px=480,
                       caption="Bare vs attested side-by-side. The "
                                "bare ranking is closer to a uniform "
                                "distribution; attestation amplifies "
                                "the spread on some axes and not others.") + \
        ui.data_table(
            columns=("Axis", "ρ̄_cross (attested)"),
            rows=_axis_table_rows(means_att, [r[0] for r in ranking]),
            col_classes=("", "num strong"),
        ) + \
        ui.disclaimer(
            "<strong>Pool sensitivity warning.</strong> "
            "Three axes (rights-duties, status-contract, state-market) "
            "show rank-order instability under pool perturbation; the "
            "remaining three (individual-collective, public-private, "
            "natural-positive) hold steady. Axis-ranking claims should "
            "be made on the pool-robust set only, or explicitly "
            "qualified for the pool-sensitive ones."
        ) + \
        ui.takehome_block(
            "Three axes (the pool-robust set) carry a stable "
            "cross-tradition signature; three are sensitive to "
            "curation. Chapter 4 §4.1 reports the pool-robust ranking; "
            "§4.2 discloses the pool-sensitivity of the others."
        ) + \
        apparatus_block(
            stats=[("most divergent",  f"{most[0]} · {most[1]:.3f}"),
                   ("least divergent", f"{least[0]} · {least[1]:.3f}"),
                   ("axes total",      "6"),
                   ("cross pairs / axis", "9 (3 WEIRD × 3 Sinic)")],
            meta=("Bare and attested rankings stored under "
                  "<code>section_324.cross_rho_mean_per_axis</code> "
                  "(values shown above)."),
            code_ref=[("experiments/ch3-measurability/experiment_2_axes/"
                       "results_attested/", "experiment_2_results.json")],
            sources=[
                "Kozlowski et al. (2019) — axis construction method.",
            ],
            collapsible=True,
        ) + \
        ui.section_close()


# --------------------------------------------------------------------------
# §3.2.5 — between-group differences (qualitative)

def _section_325(s325: dict) -> str:
    bare = s325.get("bare", {})
    attested = s325.get("attested", {})
    has_terms = bool(bare) or bool(attested)
    body = (
        "<p>The experiment also reports, per axis, the legal terms whose "
        "WEIRD-mean and Sinic-mean projection differ most. These are the "
        "qualitative anchors that complement the quantitative ρ̄ of "
        "§3.2.4.</p>"
        if has_terms else
        "<p><em>Per-axis top-divergent term lists are computed at build "
        "time from <code>section_325</code>. If the field is empty in "
        "the loaded JSON, this section becomes a stub; consult the "
        "thesis text §3.2.5 for the term-level reading.</em></p>"
    )
    return ui.section_open("s325",
                            "§3.2.5 · Between-group differences") + \
        ui.scenario_block(
            "Beyond an axis-level ρ̄, the substantive question is: which "
            "individual terms anchor the cross-tradition divergence? "
            "Their identity tells a doctrinal story."
        ) + \
        ui.result_block(
            "Per-axis top-divergent terms identify the loci of "
            "tradition-specific weight: see §3.2.5 in the thesis text "
            "for the doctrinal reading. The G extension (Robustness "
            "page) supplies a complementary term-level proof on "
            "same-lemma vocabulary."
        ) + \
        body + \
        ui.takehome_block(
            "Term-level divergence and axis-level divergence converge: "
            "Chapter 4 §4.1 quotes individual terms to make the "
            "tradition story concrete."
        ) + \
        apparatus_block(
            stats=[("source", "section_325 of experiment_2_results.json"),
                   ("see also", "Robustness G (same-lemma proof)")],
            meta="Term-level lists are intentionally not enumerated on "
                 "the dashboard to avoid steering the doctrinal reading "
                 "into a fixed narrative; the thesis text walks through "
                 "them in context.",
            code_ref=[("experiments/ch3-measurability/experiment_2_axes/",
                       "results_*/experiment_2_results.json")],
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
            subtitle="Projecting 364 legal terms onto six doctrinal axes, "
                     "across ten encoders and two language traditions.",
            crumb="Chapter 3 · Experiment 2",
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
            next_=("robustness_caveats.html", "Robustness & caveats"),
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
