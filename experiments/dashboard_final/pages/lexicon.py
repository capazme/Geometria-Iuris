"""Inside the inputs — the lexicon, control, antonym pairs, probe templates.

Generates `output/lexicon.html`. The page surfaces the frozen input
snapshots so that a lawyer reading the dashboard can inspect, in
plain HTML, what the language models were actually given:

  - The 364 legal terms, grouped by domain, each expandable to two
    real ordinance passages (English + Chinese, with Cap. and section
    references).
  - The 100 everyday-language control terms.
  - The categorical probe inputs (11 ordered categories + 5 paraphrase
    templates per test, both languages).
  - The ten antonym pairs that build each of the six value axes
    (both languages).

No Plotly here — the page is text-heavy by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import html  # noqa: E402

import shared_ui as ui  # noqa: E402
from data import loader_inputs as inputs  # noqa: E402


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


# --------------------------------------------------------------------------
# Intro

def _intro() -> str:
    return ui.section_open("intro", "Look inside the inputs") + """
<p class="lead">
The dashboard so far has shown what the language models <em>produced</em>:
distance maps, agreement scores, axis rankings, probe curves. This page
is the other side — what the models <em>received</em>. The 364 legal
terms with their real ordinance contexts, the 100 control words, the
five categorical probes with their templated sentences, and the ten
antonym pairs that built each of the six value axes. Every entry is
verbatim from the input snapshots that §2.1, §2.3, §3.1.4 and
§3.2.1 of the thesis describe.
</p>
""" + ui.section_close()


# --------------------------------------------------------------------------
# §2.1 — the 364 legal terms (by domain), each expandable to contexts

def _lexicon_section() -> str:
    terms_by_dom = inputs.core_terms_by_domain()
    contexts = inputs.load_term_contexts()
    parts = [
        ui.section_open("lexicon",
                         "The 364 legal terms — by domain, with real ordinance contexts"),
        """
<p>
Every term is bilingual (English / Chinese), labelled with the domain
into which §2.1 of the thesis sorts it and with the divisions of the
Hong Kong Department of Justice that maintain it. Click a term to
reveal up to two real passages from the Hong Kong ordinances that
contain it (one English-side, one Chinese-side, when available),
prefixed with the Cap. and section references.
</p>
""",
    ]
    for d, ts in terms_by_dom.items():
        if not ts:
            continue
        label = inputs.domain_label(d)
        n = len(ts)
        parts.append(
            f'<details class="domain-block">'
            f'<summary>{label} '
            f'<span class="domain-count">({n} terms)</span></summary>'
            f'<div class="inner">'
        )
        for t in ts:
            en = t.get("en", "?")
            zh = t.get("zh", "")
            doj = ", ".join(t.get("doj_divisions", []) or [])
            ctx = contexts.get(en.lower(), {})
            en_ctxs = ctx.get("en", [])
            zh_ctxs = ctx.get("zh", [])
            k_en = ctx.get("k_en")
            k_zh = ctx.get("k_zh")
            meta_bits = []
            if doj:
                meta_bits.append(f"DOJ: {_esc(doj)}")
            if k_en is not None and k_zh is not None:
                meta_bits.append(
                    f"K = {k_en} EN · {k_zh} ZH attestations"
                )
            meta = " &nbsp;·&nbsp; ".join(meta_bits)
            ctx_html = ""
            if en_ctxs or zh_ctxs:
                lines = ['<div class="contexts">']
                for c in en_ctxs:
                    ref = (f"Cap. {c.get('cap','?')} "
                           f"{c.get('section','')} ({c.get('year','')})")
                    lines.append(
                        f'<div class="ctx en">'
                        f'<span class="lang-tag">EN</span>'
                        f'<span class="ref">{_esc(ref)}</span>'
                        f'{_esc(c.get("text",""))}</div>'
                    )
                for c in zh_ctxs:
                    ref = (f"Cap. {c.get('cap','?')} "
                           f"{c.get('section','')} ({c.get('year','')})")
                    lines.append(
                        f'<div class="ctx zh">'
                        f'<span class="lang-tag">ZH</span>'
                        f'<span class="ref">{_esc(ref)}</span>'
                        f'{_esc(c.get("text",""))}</div>'
                    )
                lines.append("</div>")
                ctx_html = "\n".join(lines)
            else:
                ctx_html = (
                    '<div class="contexts">'
                    '<em>No bilingual contexts retained for this lemma in '
                    'the snapshot.</em></div>'
                )
            parts.append(
                f'<details class="term-row">'
                f'<summary>'
                f'<span class="term-en">{_esc(en)}</span>'
                f'<span class="term-zh">{_esc(zh)}</span>'
                f'<span class="term-meta">{meta}</span>'
                f'</summary>'
                f'{ctx_html}'
                f'</details>'
            )
        parts.append("</div></details>")
    parts.append(ui.section_close())
    return "".join(parts)


# --------------------------------------------------------------------------
# §2.1 — the 100 everyday-language control terms

def _control_section() -> str:
    ctrl = inputs.load_control_terms()
    if not ctrl:
        return ""
    parts = [
        ui.section_open("control",
                         "The 100 everyday-language control terms"),
        """
<p>
The control pool grounds the §3.1.1 legal-versus-control diagnostic and
the §3.1.3 control-pool subtraction discussed on the Robustness page.
The terms are drawn from the Swadesh basic-vocabulary list:
pronouns, deictics, basic common nouns. They have no Hong Kong
ordinance attestation by design — the only comparable reading the
experiment runs on them is the bare encoding.
</p>
<div class="control-grid">
""",
    ]
    for t in ctrl:
        en = t.get("en", "?")
        zh = t.get("zh", "")
        parts.append(
            f'<div class="item">'
            f'<span class="en">{_esc(en)}</span>'
            f'<span class="zh">{_esc(zh)}</span>'
            f'</div>'
        )
    parts.append("</div>")
    parts.append(ui.section_close())
    return "".join(parts)


# --------------------------------------------------------------------------
# §3.1.4 — probe templates

def _probes_section() -> str:
    probes = inputs.load_probe_inputs()
    parts = [
        ui.section_open("probes",
                         "Categorical probes — eleven categories and five paraphrase templates per test"),
        """
<p>
The §3.1.4 probes test whether the geometry preserves a doctrinally
ordered sequence and places the largest gap at the legislatively
expected break. For each test, the inputs are eleven category words
in a defensible order plus five sentence templates that hold the
syntactic frame constant while the category word varies. The
template slot is shown as
<span class="slot">{category}</span>.
The doctrinally expected break is highlighted on the category row.
</p>
""",
    ]
    for tid, t in probes.items():
        label = t.get("label", tid)
        polarity = t.get("polarity", "positive")
        threshold = t.get("legal_threshold") or ""
        eg = t.get("expected_gap_index")
        borderline = t.get("borderline", False)
        cats_en = t.get("categories_en", [])
        cats_zh = t.get("categories_zh", [])
        tmpl_en = t.get("templates_en", [])
        tmpl_zh = t.get("templates_zh", [])

        polarity_tag = "negative control" if polarity == "negative" else (
            "positive · borderline" if borderline else "positive"
        )

        parts.append(f'<div class="probe-block" id="probe-{tid}">')
        parts.append(f'<h4>{_esc(label)} '
                     f'<span class="domain-count">({polarity_tag})</span>'
                     f'</h4>')
        if threshold:
            parts.append(
                f'<div class="threshold">{_esc(threshold)}</div>'
            )

        # Categories — EN row.
        def _cat_row(cats: list, lang: str) -> str:
            bits = []
            for i, c in enumerate(cats):
                cls = "expected" if (eg is not None
                                       and (i == eg or i == eg + 1)
                                       and polarity == "positive") else ""
                content = (f'<span class="{cls}">{_esc(c)}</span>'
                           if cls else _esc(c))
                bits.append(f'<span class="cat-idx">{i+1}.</span>{content}')
            joiner = " &nbsp;→&nbsp; "
            return f'<div class="cats"><strong>{lang}:</strong> ' \
                   f'{joiner.join(bits)}</div>'

        if cats_en:
            parts.append(_cat_row(cats_en, "EN"))
        if cats_zh:
            parts.append(_cat_row(cats_zh, "ZH"))

        # Templates.
        if tmpl_en:
            parts.append('<div class="templates"><strong>EN templates:</strong><ol>')
            for tpl in tmpl_en:
                rendered = _esc(tpl).replace(
                    "{category}",
                    '<span class="slot">{category}</span>',
                )
                parts.append(f'<li>{rendered}</li>')
            parts.append("</ol></div>")
        if tmpl_zh:
            parts.append('<div class="templates"><strong>ZH templates:</strong><ol>')
            for tpl in tmpl_zh:
                rendered = _esc(tpl).replace(
                    "{category}",
                    '<span class="slot">{category}</span>',
                )
                parts.append(f'<li>{rendered}</li>')
            parts.append("</ol></div>")

        parts.append("</div>")
    parts.append(ui.section_close())
    return "".join(parts)


# --------------------------------------------------------------------------
# §3.2.1 — antonym pairs per axis

_AXIS_TITLES = {
    "individual_collective": "Individual ↔ Collective",
    "rights_duties":         "Rights ↔ Duties",
    "public_private":        "Public ↔ Private",
    "state_market":          "State ↔ Market",
    "natural_positive":      "Natural ↔ Positive",
    "status_contract":       "Status ↔ Contract",
}


def _axes_section() -> str:
    axes = inputs.load_value_axes()
    parts = [
        ui.section_open("axes",
                         "Value axes — the ten antonym pairs that build each axis"),
        """
<p>
Each of the six axes of §3.2 is built as the mean of ten antonym-pair
difference vectors, normalised to unit length (Kozlowski, Taddy &amp;
Evans, 2019). The recipe is bilingual: each axis has an English version
built from ten English pairs and a Chinese version built from ten
Chinese pairs drawn from the doctrinal vocabulary of the corresponding
tradition. The pairs are listed below in their snapshot order; the
first entry is the positive pole, the second the negative pole.
</p>
""",
    ]
    for axis, blob in axes.items():
        en = blob.get("en_pairs", [])
        zh = blob.get("zh_pairs", [])
        title = _AXIS_TITLES.get(axis, axis.replace("_", " ↔ "))
        parts.append(
            f'<details class="domain-block" id="axis-{axis}" open>'
            f'<summary>{title} '
            f'<span class="domain-count">({len(en)} EN + {len(zh)} ZH pairs)'
            f'</span></summary>'
            f'<div class="inner"><div class="axis-pairs">'
        )
        # EN column.
        parts.append('<div class="pairs-column">'
                     '<h4>English antonym pairs</h4><ol>')
        for pair in en:
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            pos, neg = pair[0], pair[1]
            parts.append(
                f'<li><span class="pos">{_esc(pos)}</span>'
                f'<span class="sep">vs</span>'
                f'<span class="neg">{_esc(neg)}</span></li>'
            )
        parts.append("</ol></div>")
        # ZH column.
        parts.append('<div class="pairs-column">'
                     '<h4>Chinese antonym pairs</h4><ol>')
        for pair in zh:
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            pos, neg = pair[0], pair[1]
            parts.append(
                f'<li><span class="pos">{_esc(pos)}</span>'
                f'<span class="sep">vs</span>'
                f'<span class="neg">{_esc(neg)}</span></li>'
            )
        parts.append("</ol></div></div></div></details>")
    parts.append(ui.section_close())
    return "".join(parts)


# --------------------------------------------------------------------------
# build()

def build() -> str:
    parts = [
        ui.page_head(
            title="Inside the inputs",
            subtitle="The 364 legal terms, the 100 control words, the "
                     "categorical probes, the antonym pairs.",
            crumb="Chapter 2 · Inputs",
            include_plotly=False,
        ),
        ui.sticky_nav(current_href="lexicon.html"),
        ui.open_main(),
        _intro(),
        _lexicon_section(),
        _control_section(),
        _probes_section(),
        _axes_section(),
        ui.linear_nav(
            prev=("how_it_works.html", "How it works"),
            next_=("experiment_31.html", "Experiment §3.1"),
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
