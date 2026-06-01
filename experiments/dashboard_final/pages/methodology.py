"""Methodology page — models, dataset, statistical toolkit.

Generates `output/methodology.html`. Companion to Chapters 1-2 of the
thesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import shared_ui as ui  # noqa: E402


# --------------------------------------------------------------------------
# Models table

MODELS = [
    # (label, hf_id, family, dim, languages, tradition)
    ("BGE-EN-large",       "BAAI/bge-large-en-v1.5",
     "BGE",        1024, "en",       "Western-trained"),
    ("E5-large",           "intfloat/e5-large-v2",
     "E5",         1024, "en",       "Western-trained"),
    ("FreeLaw-EN",         "OpenLegalAI/legalbench-bge-large-en-v1.5",
     "BGE (legal-FT)", 1024, "en",   "Western-trained"),
    ("BGE-ZH-large",       "BAAI/bge-large-zh-v1.5",
     "BGE",        1024, "zh",       "Chinese-trained"),
    ("Text2vec-large-ZH",  "shibing624/text2vec-large-chinese",
     "Text2vec",   1024, "zh",       "Chinese-trained"),
    ("Dmeta-ZH",           "DMetaSoul/Dmeta-embedding",
     "Dmeta",       768, "zh",       "Chinese-trained"),
    ("BGE-M3-EN",          "BAAI/bge-m3 (EN side)",
     "BGE-M3",     1024, "en (bilingual)", "Bilingual"),
    ("BGE-M3-ZH",          "BAAI/bge-m3 (ZH side)",
     "BGE-M3",     1024, "zh (bilingual)", "Bilingual"),
    ("Qwen3-0.6B-EN",      "Qwen/Qwen3-Embedding-0.6B (EN side)",
     "Qwen",       1024, "en (multilingual)", "Bilingual"),
    ("Qwen3-0.6B-ZH",      "Qwen/Qwen3-Embedding-0.6B (ZH side)",
     "Qwen",       1024, "zh (multilingual)", "Bilingual"),
]


# --------------------------------------------------------------------------
# Intro (bridges §1 to §2)

def _intro() -> str:
    return ui.section_open("intro", "From the conceptual conditions to the apparatus") + """
<p class="lead">
Chapter 1 of the thesis derives the conceptual conditions under which
legal meaning can be observed at scale: meaning as use, language as
space, geometry as legal instrument (§1.2–§1.4). Chapter 2 turns those
conditions into an apparatus. This page summarises the apparatus — the
language models, the lexicon, and the statistical tools — that the two
experiments in §3.1 and §3.2 rely on.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# Models section

def _models_section() -> str:
    rows = []
    for lab, hf, fam, dim, langs, trad in MODELS:
        rows.append((lab, fam, str(dim), langs,
                      f'<code>{hf}</code>', trad))
    return ui.section_open("models", "Ten language models") + """
<p>
Ten language models span three families and two language traditions.
The selection is deliberately heterogeneous: it includes a legal
fine-tune (<em>FreeLaw-EN</em>) to test whether legal specialisation
helps; a small multilingual model (<em>Qwen3-0.6B</em>) to test the
limits of generalisation; and a single bilingual model
(<em>BGE-M3</em>) deployed twice — once on the English side, once on
the Chinese — to provide a same-model bilingual control used in
§3.1.3 and §3.2.4.
</p>

<p>
The Western-trained vs Chinese-trained contrast is the language-tradition
contrast that organises the empirical chapters. The thesis is not about
that contrast <em>per se</em>: it is methodological. The choice of Hong
Kong ordinances co-drafted in English and Chinese (§2.2) makes the
contrast naturally available, and the design uses it as the test for
whether embedded geometry is sensitive to tradition.
</p>
""" + ui.data_table(
        columns=("Label", "Family", "Dim", "Lang", "Reference", "Tradition"),
        rows=rows,
        col_classes=("strong", "", "num", "", "", ""),
    ) + ui.section_close()


# --------------------------------------------------------------------------
# Dataset section

def _dataset_section() -> str:
    return ui.section_open("dataset", "Dataset · 364 + 9 045 + 100 terms") + """
<p>
The dataset has three tiers. Each term is a parallel English/Chinese
pair drawn from the Hong Kong DOJ bilingual legal glossary, and each
has been re-attested against the post-1989 ordinances co-drafted under
the Bilingual Laws Project. The filter retains a term only if it
appears at least four times in real ordinance contexts; the
threshold is justified empirically in §3 of the thesis. The complete
364-term lexicon and the 100 control words are browsable verbatim
under <a href="lexicon.html#lexicon">Inside the inputs</a>, each
term expandable to two real Hong Kong ordinance passages.
</p>

<h3>Core · 364 terms</h3>
<p>
The curated, vetted core of the legal lexicon. Distributed across 7
domains (administrative, civil, constitutional, criminal,
international, labor &amp; social, procedure) with band-balanced
support (41 – 60 attestations per domain). Each term is encoded twice
by each model: <em>bare</em> (the lemma in isolation) and
<em>attested</em> (the mean embedding of its real ordinance contexts).
</p>

<h3>Background · 9 045 terms</h3>
<p>
The remainder of the bilingual glossary that satisfies the four-context
threshold but was not hand-vetted. Used in the robustness analyses of
§3 to test the principal results under pool perturbation, and as the
input for a k-nearest-neighbour assignment that labels each background
term with the domain voted by its seven nearest core neighbours.
</p>

<h3>Control · 100 terms</h3>
<p>
Everyday-language vocabulary with no legal content: pronouns, deixis,
common nouns (<em>I, you, he, this, here, water, day, year, etc.</em>).
Encoded bare only — controls have no Hong Kong ordinance attestation by
design. The control pool grounds the §3.1.1 legal-vs-control test and
underpins the control-pool subtraction in §3.1.3: the bare Δρ on the
control pool is statistically indistinguishable from the bare Δρ on the
core, isolating the contribution that attestation in legal context
adds.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# Statistical toolkit · six methods as compact glossary

def _toolkit_section() -> str:
    return ui.section_open("toolkit", "Statistical toolkit") + """
<p>
Six tools, each with a specific role and a specific limit. They are
introduced here in compact form; their full application — including
the inferential discipline that separates measure, interpretation and
limit — is the subject of §2.4 of the thesis.
</p>

<h3>Representational Similarity Analysis (RSA)</h3>
<p>
For each language model, build a Representational Dissimilarity Matrix
(RDM): a 364 × 364 symmetric matrix where cell (i, j) is the cosine
distance between term i and term j as encoded by that model. Comparing
two language models then means computing the Spearman ρ between the
upper triangles of their two RDMs. ρ = 1 if they rank pairs
identically; ρ = 0 if they are uncorrelated.
<em>Reference: Kriegeskorte, Mur &amp; Bandettini (2008).</em>
</p>

<h3>Mantel test (B = 10 000)</h3>
<p>
The null hypothesis for an RSA ρ is that the two RDMs are unrelated.
The Mantel test draws a permutation distribution by shuffling one
RDM's rows (and matching columns) and recomputing ρ. In §3.1.3 the
test applied to all 17 pre-registered model pairs returns
p ≤ 1 × 10⁻⁴ on every pair.
</p>

<h3>Holm correction (K = 17)</h3>
<p>
The 17 pre-registered model pairs in §3.1.3 mean 17 simultaneous tests.
The Holm–Bonferroni correction guards against multiple-testing
inflation; the corrected p<sub>max</sub> across all 17 pairs is
≤ 1.7 × 10⁻³.
</p>

<h3>Block bootstrap on terms (B = 10 000)</h3>
<p>
The 66 066 upper-triangle entries of an RDM are not independent: each
term contributes to 363 of them. The block bootstrap resamples
<em>terms</em> (not pairs) and re-subsets the RDM, then recomputes ρ —
yielding term-level confidence intervals that honour the dependence
structure. <em>Reference: Nili et al. (2014).</em>
</p>

<h3>Kozlowski axis construction</h3>
<p>
For each value axis (e.g. <em>individual ↔ collective</em>) curate up
to twenty antonymic seed pairs and compute the centroid difference
vector: positive minus negative, averaged over pairs and normalised to
unit length. A term's score on the axis is its cosine with the axis
vector. The six axes of §3.2 are built this way.
<em>Reference: Kozlowski, Taddy &amp; Evans (2019).</em>
</p>

<h3>Mann-Whitney U with rank-biserial r</h3>
<p>
For two distance distributions (e.g. legal-legal vs legal-control)
compute the non-parametric Mann-Whitney U. The associated effect size
is the rank-biserial r = 1 − 2U / (n<sub>x</sub> n<sub>y</sub>); r &gt; 0 means the
first distribution sits below the second. Used in §3.1.1.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# build()

def build() -> str:
    parts = [
        ui.page_head(
            title="Methodology · models, dataset, toolkit",
            subtitle="Chapters 1–2 of the dissertation",
            crumb="Chapters 1–2",
            include_plotly=False,
        ),
        ui.sticky_nav(current_href="methodology.html"),
        ui.open_main(),
        _intro(),
        _models_section(),
        _dataset_section(),
        _toolkit_section(),
        ui.linear_nav(
            prev=("index.html", "Home"),
            next_=("how_it_works.html", "How it works"),
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
