"""How it works — pipeline narrative end-to-end.

Generates `output/how_it_works.html`. Walks a reader through the
five-stage measurement pipeline, with a stopover on the bare/attested
distinction (load-bearing for the Y caveat) and a worked Kozlowski
example.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import shared_ui as ui  # noqa: E402


def _intro() -> str:
    return ui.section_open("intro", "How the measurement works") + """
<p class="lead">
Chapter 3 turns a methodological question — <em>can legal meaning be
measured?</em> — into two pre-registered experiments on a fixed
lexicon. To follow the results, it helps to picture the pipeline that
turns 364 legal terms into a number like Δρ_sym = 0.543. Five
stages take you from the ordinance PDFs of the Hong Kong legislative
archive to the headline RSA agreement statistic.
</p>
""" + ui.section_close()


def _pipeline_section() -> str:
    stages = [
        ("Ordinances",
         "<strong>Source:</strong> Hong Kong ordinances enacted after 1989, "
         "co-drafted in English and Chinese under the Bilingual Laws "
         "Project. Each ordinance is a parallel pair of authoritative texts. "
         "The pipeline indexes every textual occurrence of every legal "
         "term, with paragraph and sentence offsets."),
        ("Contexts",
         "<strong>K-attestation:</strong> for each term that appears at "
         "least K = 4 times in real ordinances, we extract K context "
         "windows (paragraph-sized, anchored on the term). The window "
         "is the unit of attestation: the lemma is seen <em>in real "
         "use</em>, not in isolation. Terms below K = 4 are dropped — "
         "the threshold's empirical justification is in extension H."),
        ("Encoders",
         "<strong>10 models × 2 encodings:</strong> each encoder processes "
         "the term twice. <em>Attested</em> (the result): the mean of "
         "the K context embeddings, each context passed to the encoder "
         "with the lemma's position marked. This is the ordinance-grounded "
         "vector that the experiments analyse. <em>Bare</em> (a "
         "methodological control): the lemma in isolation, no context. "
         "Reported alongside attested so the Y caveat can isolate the "
         "legal contribution from the encoder-tradition baseline."),
        ("RDM",
         "<strong>Representational Dissimilarity Matrix:</strong> for each "
         "(model, encoding) pair, build a 364 × 364 symmetric matrix where "
         "cell (i, j) is the cosine distance between term i and term j. "
         "The RDM is the geometric portrait of how that model sees the "
         "lexicon. Twenty RDMs total (10 models × 2 encodings)."),
        ("Comparison",
         "<strong>RSA + axis projection:</strong> §3.1 compares RDMs in "
         "pairs (17 pairs, Spearman ρ on upper triangles) to get "
         "Δρ_sym. §3.2 projects each RDM onto six value axes, then "
         "compares the projected rankings instead. Both reductions land "
         "the same structural finding: within-tradition encoders agree "
         "more than cross-tradition ones."),
    ]
    return ui.section_open("pipeline", "Five stages") + """
<p>
Click a stage to see what the script actually does at that step. The
chain of files at each stage is preserved in
<code>experiments/ch3-measurability/</code>.
</p>
""" + ui.pipeline_diagram(stages) + ui.section_close()


def _bare_attested_section() -> str:
    return ui.section_open("bare-attested",
                            "Bare vs attested — load-bearing distinction") + """
<p>
The single most important methodological move in Chapter 3 is the
split between the bare and the attested encoding. Without it, the
Y caveat — which is what isolates the legal signal from the encoder
baseline — cannot be stated.
</p>

<h3>Bare encoding</h3>
<p>
The lemma is sent to the encoder as a single string, with no context.
The output is a vector that reflects everything the encoder has
internalised about that string from its training corpus. For
language-tradition encoders, that vector inherits the tradition.
</p>

<h3>Attested encoding</h3>
<p>
For each of K context windows, the encoder produces a vector for the
sentence; we take the mean. The result is the lemma's embedding
<em>in real use</em>, biased by what HK ordinances do with the term.
The attestation step is where the corpus's specific legal context
enters the geometry.
</p>

<h3>Why the distinction matters</h3>
<p>
A naive reading of Δρ_sym attested = 0.543 says: <em>this is what
cross-tradition legal divergence looks like, measured</em>. The Y
caveat (Robustness page) shows that on 100 control terms the bare
Δρ_sym is 0.156 — indistinguishable from the bare core. The legal
signal lives in the gap that attestation opens up, not in the
attested absolute. Throughout the chapter we cite both numbers.
</p>
""" + ui.disclaimer(
        "<strong>Role of bare in the thesis.</strong> "
        "Attested is the result. Bare is a methodological baseline "
        "computed to isolate the legal signal in the Y caveat "
        "(Robustness page). The legal signal is the attested-bare gap "
        "on the 364 core terms: 0.378 = 0.543 − 0.165. The dashboard "
        "reports bare only for this isolation purpose; it is not a "
        "second headline finding."
    ) + ui.section_close()


def _kozlowski_section() -> str:
    return ui.section_open("kozlowski",
                            "What a value axis looks like — worked example") + """
<p>
§3.2 projects every term onto six pre-defined value axes. Each axis is
constructed by the Kozlowski-Taddy-Evans procedure (2019): you give
the encoder a list of ~20 antonymic pairs (rights vs duties,
freedom vs constraint, claim vs obligation, …) and let it produce
a direction vector — the centroid of the differences. A term's score
on the axis is then its cosine with that vector. Positive scores sit
on the &ldquo;rights&rdquo; pole; negative scores sit on the
&ldquo;duties&rdquo; pole.
</p>

<p>
The construction has two properties worth dwelling on. First, the axis
inherits the encoder's bias: a Sinic encoder will give the same English
pair list a slightly different direction vector than a WEIRD encoder
would. That difference is exactly what §3.2.3 and §3.2.4 measure.
Second, the seed pairs are bilingual — each axis has an English and a
Chinese version. The Chinese axis is not a translation of the English
axis; it is built from Chinese seed pairs the doctrine itself supplies.
</p>

<p>
Of the six axes, three are robust to pool curation
(<em>individual ↔ collective</em>, <em>public ↔ private</em>,
<em>natural ↔ positive</em>) and three are sensitive
(<em>rights ↔ duties</em>, <em>status ↔ contract</em>,
<em>state ↔ market</em>). The sensitive axes shift when the curated
pool shifts; the robust ones hold steady. §3.2.4 reports both sets;
§4.2 of the thesis discloses the sensitivity as a methodological limit
rather than a finding.
</p>
""" + ui.section_close()


def _reading_path() -> str:
    return ui.section_open("reading-path",
                            "Two reading paths") + """
<p>
The remaining pages can be read in two orders.
</p>

<p>
<strong>The result-first path</strong> goes Home → Robustness &amp;
caveats → Experiment §3.1 → Experiment §3.2. The Home page summarises
the three anchor results; Robustness opens with the Y caveat that
reframes the headline number; the two experiment pages then provide
the technical detail and the figures. A reader pressed for time gets
the substantive content this way.
</p>

<p>
<strong>The method-first path</strong> goes Home → Methodology → How it
works → §3.1 → §3.2 → Robustness. This order matches the thesis text
and lets the reader build the apparatus before seeing the numbers. The
committee member who reads the thesis alongside the dashboard will
find this order more comfortable.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# build()

def build() -> str:
    parts = [
        ui.page_head(
            title="How it works — the measurement pipeline",
            subtitle="From ordinance PDFs to Δρ_sym in five stages, "
                     "with a stopover on bare vs attested.",
            crumb="Chapter 2 · Pipeline",
            include_plotly=False,
        ),
        ui.sticky_nav(current_href="how_it_works.html"),
        ui.open_main(),
        _intro(),
        _pipeline_section(),
        _bare_attested_section(),
        _kozlowski_section(),
        _reading_path(),
        ui.linear_nav(
            prev=("methodology.html", "Methodology"),
            next_=("experiment_31.html", "Experiment §3.1"),
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
