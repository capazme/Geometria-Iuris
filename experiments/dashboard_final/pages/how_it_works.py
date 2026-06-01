"""How it works — pipeline narrative end-to-end.

Generates `output/how_it_works.html`. Walks the reader from the Hong
Kong ordinance archive to the §3.1.3 cross-tradition reading in five
stages, with a stopover on the bare/attested distinction and a worked
example of the §3.2 axis construction.
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
Chapter 3 of the thesis turns a methodological question —
<em>is legal meaning measurable?</em> — into two experiments on a
fixed legal lexicon. To follow the §3.1 and §3.2 readings, it helps
to picture the pipeline that turns 364 legal terms into the
cross-tradition agreement of §3.1.3. Five stages take you from the
ordinance text of the Hong Kong archive to the rank correlation
between two models, with a stop on the bare-versus-attested
distinction and on the §3.2 axis construction.
</p>
""" + ui.section_close()


def _pipeline_section() -> str:
    stages = [
        ("Ordinances",
         "<strong>Source:</strong> Hong Kong ordinances enacted from "
         "1989 onwards, co-drafted in English and Chinese under the "
         "Bilingual Laws Project. Each ordinance is a parallel pair "
         "of authoritative texts. The pipeline indexes every textual "
         "occurrence of every legal term, with paragraph and sentence "
         "offsets."),
        ("Contexts",
         "<strong>Real-use windows:</strong> for each term that "
         "appears at least four times in the ordinance text, the "
         "pipeline extracts the four (or more) context windows in "
         "which the term occurs. The window is the unit of "
         "attestation: the term is seen <em>in real use</em>, not in "
         "isolation. Terms with fewer than four contexts are dropped "
         "(§2.3 of the thesis justifies the threshold)."),
        ("Models",
         "<strong>Ten models × two encodings:</strong> each language "
         "model processes the term twice. <em>Attested</em> "
         "(the result): the mean of the context vectors, each "
         "context passed to the model with the term's position "
         "marked. This is the ordinance-grounded vector that the "
         "experiments analyse. <em>Bare</em> (a methodological "
         "baseline): the term in isolation, no context. Reported "
         "alongside attested so that the control-pool subtraction "
         "(Robustness page) can isolate the legal-meaning "
         "contribution from the model-tradition baseline."),
        ("Distance maps",
         "<strong>The 364 × 364 cosine distance matrix:</strong> "
         "for each model and each encoding, build the symmetric "
         "matrix whose cell (i, j) is the cosine distance between "
         "term i and term j. The matrix is the geometric portrait of "
         "how that model sees the lexicon. Twenty maps in total "
         "(ten models × two encodings)."),
        ("Comparison",
         "<strong>Agreement and projection:</strong> §3.1 compares "
         "distance maps in pairs (seventeen pre-registered model "
         "pairs, Spearman ρ on the upper triangles) and reports the "
         "symmetric within-versus-cross tradition gap. §3.2 projects "
         "each model's 364 vectors onto six axes built from antonym "
         "pairs and compares the projected rankings instead. Both "
         "reductions land the same structural finding: models within "
         "a language tradition agree more on the legal lexicon than "
         "models across traditions."),
    ]
    return ui.section_open("pipeline", "Five stages") + """
<p>
Click a stage to see what the pipeline does at that step. The five
stages correspond to the apparatus assembled in §2.1 (lexicon),
§2.2 (corpus), §2.3 (models) and §2.4 (statistical tools) of the
thesis.
</p>
""" + ui.pipeline_diagram(stages) + ui.section_close()


def _bare_attested_section() -> str:
    return ui.section_open("bare-attested",
                            "Bare and attested — the two encodings") + """
<p>
The single methodological move that frames everything Chapter 3
reads is the split between the bare and the attested encoding. They
sit alongside each other throughout the experiments; their
difference is what makes the cross-tradition reading legally
interpretable.
</p>

<h3>Bare encoding</h3>
<p>
The term is sent to the language model as a single string, with no
context. The output is a vector that reflects whatever the model has
internalised about that string from its training corpus. For
tradition-specialised models, the vector inherits the tradition
along with the lexicon.
</p>

<h3>Attested encoding</h3>
<p>
For each context window in which the term appears, the model
produces a vector for the surrounding passage; the pipeline takes
the mean and re-normalises to unit length. The result is the term's
embedding <em>in real use</em>, shaped by what Hong Kong ordinances
do with it. The attestation step is where the corpus's specific
legal context enters the geometry.
</p>

<h3>Why both?</h3>
<p>
The §3.1.3 reading of 0.543 is computed on the attested encoding.
The same construction run on the bare encoding returns 0.165 — and
the same construction run on a 100-term everyday-language control
set returns 0.156, statistically indistinguishable from the bare
legal core. The bare gap is shaped by the models themselves, not
by legal vocabulary. The legal-meaning contribution is what
attestation adds on the legal core: 0.378 = 0.543 − 0.165. The full
decomposition is on the
<a href="robustness_caveats.html#control-pool-subtraction">Robustness
page</a>.
</p>
""" + ui.section_close()


def _kozlowski_section() -> str:
    return ui.section_open("kozlowski",
                            "What an axis looks like — a worked example") + """
<p>
§3.2 of the thesis projects each of the 364 terms onto six axes
built from antonym pairs. Each axis is constructed by the procedure
that Kozlowski, Taddy and Evans set out in 2019 for the sociological
analysis of meaning: you give the model a list of ten antonym pairs
(rights / duties, freedom / constraint, claim / obligation, …) and
let it produce a direction vector — the L2-normalised mean of the
ten pole-difference vectors. A term's score on the axis is its
cosine with that vector. Positive scores sit on one pole; negative
scores on the other.
</p>

<p>
Two properties of the construction are worth dwelling on. First,
the axis inherits the model's bias: a Chinese-trained model will
give the same Chinese pair list a slightly different direction
vector than a Western-trained model would give to its English
counterpart, and that difference is exactly what §3.2.3 and §3.2.4
measure. Second, the ten antonym pairs are not parallel
translations across the two languages: each side draws its pairs
from the doctrinal vocabulary of its own tradition (rights / duties
on the English side, 權 / 義 on the Chinese side; natural / positive
law on the English side, 天理 / 國法 on the Chinese side).
</p>

<p>
Of the six axes, three are robust to pool curation
(<em>individual ↔ collective</em>, <em>public ↔ private</em>,
<em>natural ↔ positive</em>) and three are sensitive
(<em>rights ↔ duties</em>, <em>status ↔ contract</em>,
<em>state ↔ market</em>): the sensitive axes shift in rank when the
curated pool shifts; the robust ones hold steady. §3.2.4 of the
thesis reports both sets; §4.2 carries the sensitivity as the
methodological limit of the per-axis ranking, not as a finding.
</p>
""" + ui.section_close()


def _reading_path() -> str:
    return ui.section_open("reading-path", "Two reading paths") + """
<p>
The remaining pages can be read in two orders.
</p>

<p>
<strong>The result-first path</strong> goes Home → Robustness &amp;
caveats → Experiment §3.1 → Experiment §3.2. The Home page lists
the chapter's principal readings; Robustness opens with the
control-pool subtraction that reframes the §3.1.3 absolute as the
legal-meaning contribution; the two experiment pages then provide
the technical detail and the figures. A reader pressed for time
reaches the substantive content this way.
</p>

<p>
<strong>The method-first path</strong> goes Home → Methodology →
How it works → §3.1 → §3.2 → Robustness. This order matches the
thesis text and lets the reader build the apparatus before seeing
the numbers — the comfortable order for a committee member reading
the thesis alongside the dashboard.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# build()

def build() -> str:
    parts = [
        ui.page_head(
            title="How it works — the measurement pipeline",
            subtitle="From ordinance text to the §3.1.3 reading in "
                     "five stages, with a stop on bare versus "
                     "attested.",
            crumb="Chapters 2 · Pipeline",
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
