"""Methodology page — models, dataset, statistical toolkit.

Generates `output/methodology.html`. No Plotly figures; pure HTML cards
and tables. Companion to §2 of the thesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import shared_ui as ui  # noqa: E402
from apparatus import apparatus_block  # noqa: E402


# --------------------------------------------------------------------------
# Models table

MODELS = [
    # (label, hf_id, family, dim, languages, tradition)
    ("BGE-EN-large",       "BAAI/bge-large-en-v1.5",
     "BGE",        1024, "en",       "WEIRD"),
    ("E5-large",           "intfloat/e5-large-v2",
     "E5",         1024, "en",       "WEIRD"),
    ("FreeLaw-EN",         "OpenLegalAI/legalbench-bge-large-en-v1.5",
     "BGE (legal-FT)", 1024, "en",   "WEIRD"),
    ("BGE-ZH-large",       "BAAI/bge-large-zh-v1.5",
     "BGE",        1024, "zh",       "Sinic"),
    ("Text2vec-large-ZH",  "shibing624/text2vec-large-chinese",
     "Text2vec",   1024, "zh",       "Sinic"),
    ("Dmeta-ZH",           "DMetaSoul/Dmeta-embedding",
     "Dmeta",       768, "zh",       "Sinic"),
    ("BGE-M3-EN",          "BAAI/bge-m3 (EN side)",
     "BGE-M3",     1024, "en (bilingual)", "Bilingual"),
    ("BGE-M3-ZH",          "BAAI/bge-m3 (ZH side)",
     "BGE-M3",     1024, "zh (bilingual)", "Bilingual"),
    ("Qwen3-0.6B-EN",      "Qwen/Qwen3-Embedding-0.6B (EN side)",
     "Qwen",       1024, "en (multilingual)", "Bilingual"),
    ("Qwen3-0.6B-ZH",      "Qwen/Qwen3-Embedding-0.6B (ZH side)",
     "Qwen",       1024, "zh (multilingual)", "Bilingual"),
]


def _models_section() -> str:
    rows = []
    for lab, hf, fam, dim, langs, trad in MODELS:
        rows.append((lab, fam, str(dim), langs,
                      f'<code>{hf}</code>', trad))
    return ui.section_open("models", "Ten encoder models") + """
<p>
Ten embedding encoders span three model families and two language
traditions. The selection is deliberately heterogeneous: it includes a
legal fine-tune (<em>FreeLaw-EN</em>) to test whether legal specialisation
helps; a small multilingual model (<em>Qwen3-0.6B</em>) to test the
limits of generalisation; and a single bilingual model
(<em>BGE-M3</em>) deployed twice — once on the English side, once on
the Chinese — to provide a same-encoder control (β control in §3.1.3).
</p>

<p>
WEIRD versus Sinic is shorthand for the language-tradition contrast.
The thesis is not about WEIRD versus Sinic <em>per se</em>; the choice
of HK ordinances co-drafted in English and Chinese makes the contrast
naturally available, and so the design uses it as the test for whether
embedded geometry is sensitive to tradition.
</p>
""" + ui.data_table(
        columns=("Label", "Family", "Dim", "Lang", "HuggingFace ID", "Tradition"),
        rows=rows,
        col_classes=("strong", "", "num", "", "", ""),
    ) + ui.section_close()


# --------------------------------------------------------------------------
# Dataset section

def _dataset_section() -> str:
    return ui.section_open("dataset", "Dataset · 364 + 9 045 + 100 terms") + """
<p>
The dataset has three tiers, frozen at run #4 (2026-05-17). Each term
is a parallel English/Chinese pair, drawn from the Hong Kong DOJ
bilingual legal glossary, and each has been re-attested against the
post-1989 ordinances co-drafted under the Bilingual Laws Project. The
filter retains a term only if it appears at least K = 4 times in real
ordinance contexts (the operational threshold whose empirical
justification lives in Robustness extension H).
</p>

<h3>Core · 364 terms</h3>
<p>
The curated, vetted core of the legal lexicon. Distributed across 7
domains (administrative, civil, constitutional, criminal,
international, labor &amp; social, procedure) with band-balanced
support (41 – 60 attestations per domain). Each term is encoded twice
by each model: bare (the lemma in isolation) and attested (the mean
embedding of its K real ordinance contexts).
</p>

<h3>Background · 9 045 terms</h3>
<p>
The remainder of the bilingual glossary that satisfies the K ≥ 4
threshold but was not hand-vetted. Used in robustness extensions (D, E,
F) to test the chapter's headline claims under pool perturbation. A
k-NN assigner (extension A) labels each background term with the
domain voted by its seven nearest core neighbours.
</p>

<h3>Control · 100 terms</h3>
<p>
Everyday-language vocabulary with no legal content: pronouns, deixis,
common nouns (<em>I, you, he, this, here, water, day, year, etc.</em>).
Encoded bare only — controls have no HK ordinance attestation by
design. The control pool grounds the §3.1.1 legal-vs-control test and,
critically, the Y caveat: the bare Δρ_sym on controls is statistically
indistinguishable from the bare Δρ_sym on the core, isolating the
attestation contribution.
</p>
""" + apparatus_block(
        stats=[("core",        "364"),
               ("background",  "9 045"),
               ("control",     "100"),
               ("domains",     "7"),
               ("K threshold", "≥ 4"),
               ("source",      "HK DOJ bilingual glossary · post-BLP ordinances")],
        meta="Frozen inputs in <code>experiments/ch3-measurability/inputs/</code>, "
             "SHA-256 hashed in <code>manifest.json</code>.",
        code_ref=[("experiments/ch3-measurability/scripts/",
                   "data_build.py")],
        collapsible=True,
    ) + ui.section_close()


# --------------------------------------------------------------------------
# Statistical toolkit

def _toolkit_section() -> str:
    return ui.section_open("toolkit", "Statistical toolkit") + """
<p>
Six tools, each with a specific role and a specific limit. The
chapter's headline numbers are reproducible up to the seed
(<code>seed = 42</code>); the manifest hashes guarantee the inputs.
</p>

<h3>Representational Similarity Analysis (RSA)</h3>
<p>
For each encoder, build a Representational Dissimilarity Matrix
(RDM): a 364 × 364 symmetric matrix where cell (i, j) is the cosine
distance between term i and term j as encoded by that model. Comparing
two encoders means: compute the Spearman ρ between the upper triangles
of their two RDMs. ρ = 1 if they rank pairs identically; ρ = 0 if
they are uncorrelated.
</p>
""" + apparatus_block(
        formula=(
            "ρ<sub>RSA</sub>(M<sub>1</sub>, M<sub>2</sub>) = "
            "Spearman(uppertri(RDM<sub>1</sub>), uppertri(RDM<sub>2</sub>))"
        ),
        stats=[("RDM size",           "364 × 364"),
               ("triangle entries",   "66 066"),
               ("metric",             "cosine on L2-normalised pooled vectors")],
        meta="Standard reference: Kriegeskorte, Mur & Bandettini (2008) <em>Front. "
             "Syst. Neurosci.</em>",
        collapsible=True,
    ) + """
<h3>Mantel test (B = 10 000)</h3>
<p>
The null hypothesis for an RSA ρ is that the two RDMs are
unrelated. The Mantel test draws a permutation distribution by
shuffling one RDM's rows (and matching columns) and recomputing ρ.
With B = 10 000 permutations all 17 model-pair p-values floor at
p ≤ 1 × 10⁻⁴ — the test is well-powered.
</p>

<h3>Holm correction (K = 17)</h3>
<p>
Comparing 17 model pairs means 17 simultaneous tests. The Holm-Bonferroni
correction guards against the multiple-testing inflation. The
corrected p_max across all 17 pairs is ≤ 1.7 × 10⁻³ — still safely
below the conventional 5 × 10⁻³ threshold.
</p>

<h3>Block bootstrap on terms (B = 10 000)</h3>
<p>
The 66 066 upper-triangle entries of an RDM are not independent: each
term contributes to 363 of them. Standard bootstrap inflates the
confidence intervals. The block bootstrap resamples <em>terms</em> and
re-subsets the RDM, then recomputes ρ — yielding term-level CIs that
honour the dependence structure.
</p>
""" + apparatus_block(
        meta="Reference: Nili et al. (2014) <em>PLoS Comput. Biol.</em> "
             "10(4) e1003553.",
        stats=[("bootstrap B", "10 000"), ("resampled", "terms (not pairs)")],
        collapsible=True,
    ) + """
<h3>Kozlowski axis construction</h3>
<p>
For each value axis (e.g. <em>individual ↔ collective</em>), curate ~20
antonymic seed pairs and compute the centroid difference vector:
positive minus negative, averaged over pairs. The axis is the unit
vector of that centroid. A term's score on the axis is its cosine with
the axis vector.
</p>
""" + apparatus_block(
        formula=(
            "axis = mean<sub>k</sub> "
            "(emb(pos<sub>k</sub>) − emb(neg<sub>k</sub>)) / "
            "‖mean<sub>k</sub> (…)‖"
        ),
        meta="Reference: Kozlowski, Taddy & Evans (2019) <em>American "
             "Sociological Review</em> 84(5): 905-949.",
        stats=[("axes",       "6"),
               ("seed pairs", "≤ 20 / axis"),
               ("languages",  "EN + ZH (parallel)")],
        collapsible=True,
    ) + """
<h3>Mann-Whitney U with rank-biserial r</h3>
<p>
For two distance distributions (e.g. legal-legal vs legal-control),
compute the non-parametric Mann-Whitney U. The associated effect size
is the rank-biserial r = 1 − 2U / (n_x × n_y); r &gt; 0 means the
first distribution sits below the second. Used in §3.1.1.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# Reproducibility section

def _reproducibility_section() -> str:
    return ui.section_open("reproducibility", "Reproducibility") + """
<p>
The run is finalised on 2026-05-17. Every input — the term snapshots,
the value-axis YAML, the per-model embedding matrices — is SHA-256
hashed and registered in <code>manifest.json</code>; the 50 hashes are
preserved alongside the JSON results. The execution is deterministic
(seed = 42, CPU float32) and the verification gate (8 / 8 PASS,
documented on Home) re-checks internal consistency every time the
build is regenerated, guarding against silent regression.
</p>

<p>
The dashboard itself is regenerated by running
<code>python3 experiments/dashboard_final/build.py</code> from the
repository root. All HTML files in <code>output/</code> are derived
artefacts; the source of truth is the JSON in
<code>experiments/ch3-measurability/</code>.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# build()

def build() -> str:
    parts = [
        ui.page_head(
            title="Methodology · models, dataset, toolkit",
            subtitle="Companion to Chapter 2 — the instrument that "
                     "the experiments use.",
            crumb="Chapter 2 · Methodology",
            include_plotly=False,
        ),
        ui.sticky_nav(current_href="methodology.html"),
        ui.open_main(),
        _models_section(),
        _dataset_section(),
        _toolkit_section(),
        _reproducibility_section(),
        ui.linear_nav(
            prev=("index.html", "Home"),
            next_=("how_it_works.html", "How it works"),
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
