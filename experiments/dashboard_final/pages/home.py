"""Home page — welcome + navigation.

Generates `output/index.html`. Pure HTML (no Plotly figure on this page).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import shared_ui as ui  # noqa: E402


def _welcome() -> str:
    return ui.section_open("welcome", "Welcome") + """
<p class="lead">
A companion to Chapter 3 of <em>Geometria Iuris: Measuring Legal Meaning
Across Cultural Normative Structures in Embedding Spaces</em> (LUISS,
Methodology of Legal Science). The dashboard mirrors the thesis section by
section; every quantity on these pages comes from §3.1, §3.2 or §4 of the
dissertation, and is named with the same notation.
</p>

<p>
The thesis asks a single question (§1.5):
<em>is legal meaning susceptible to measurement?</em> The cross-tradition
design that organises the empirical chapters is the test bench of the
instrument, not its purpose; the question is methodological, not
comparative.
</p>
""" + ui.section_close()


def _where_to_read() -> str:
    return ui.section_open("where-to-read", "Where to read what") + ui.anchor_cards([
        ("§1 – §2",
         "Methodology",
         'How the question becomes empirical: meaning as use, geometry as '
         'record. <a href="methodology.html">Read &rsaquo;</a>'),
        ("§2.3 – §2.4",
         "How it works",
         'The instrument: language models, distance maps, value axes. '
         '<a href="how_it_works.html">Read &rsaquo;</a>'),
        ("§3.1",
         "Distance structure",
         'Whether the legal lexicon organises itself coherently, and whether '
         'two traditions agree on the organisation. '
         '<a href="experiment_31.html">Read &rsaquo;</a>'),
        ("§3.2",
         "Value axes",
         'Whether two traditions agree on the axes that order legal '
         'vocabulary. <a href="experiment_32.html">Read &rsaquo;</a>'),
        ("§4",
         "Limits and controls",
         'Where the instrument fails, and what its readings cannot warrant. '
         '<a href="robustness_caveats.html">Read &rsaquo;</a>'),
    ]) + ui.section_close()


def build() -> str:
    parts = [
        ui.page_head(
            title="Geometria Iuris — Chapter 3 companion dashboard",
            subtitle="A reading aid to the dissertation",
            crumb="Methodology of Legal Science · LUISS",
            include_plotly=False,
        ),
        ui.sticky_nav(current_href="index.html"),
        ui.open_main(),
        _welcome(),
        _where_to_read(),
        ui.linear_nav(
            prev=None,
            next_=("methodology.html", "Methodology"),
            first_page=True,
        ),
        ui.page_footer(ui._default_footer()),
    ]
    return "".join(parts)
