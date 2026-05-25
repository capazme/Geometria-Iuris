"""Apparatus block — the L2 technical card at the bottom of each sub-section.

Single full-width box with a left bronze border, the label
"Technical apparatus" in small caps, then (optionally) a centred formula,
a one-line stats summary, a meta line (p-value, CI, n_permutations,
bootstrap details), code references, and doctrinal sources. Compact:
4-6 lines total.

Visual styling lives in `shared_ui.CSS_MAIN` under `.apparatus`.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def apparatus_block(
    *,
    formula: str | Sequence[str] | None = None,
    stats: Sequence[tuple[str, str]] | None = None,
    meta: str | None = None,
    code_ref: Iterable[tuple[str, str]] | None = None,
    sources: Sequence[str] | None = None,
    collapsible: bool = False,
    summary_label: str = "Open technical detail",
) -> str:
    """Render the technical apparatus block.

    Args:
        formula: A single formula string (rendered verbatim — Unicode +
            HTML, no LaTeX) or a list of formula strings.
        stats: list of (label, value) pairs rendered inline as
            `<strong>label</strong> = value · …`.
        meta: a single-line meta sentence — typically p-value, CI,
            number of permutations / bootstrap iterations.
        code_ref: list of (file, function) pairs rendered as
            `code: file:function`.
        sources: list of doctrinal citations (free-form HTML) — typically
            1-3 short references.
        collapsible: if True, wrap the contents in a `<details>` element
            so the technical apparatus collapses by default (useful when
            multiple apparati appear on the same page).
        summary_label: text shown next to the disclosure triangle when
            `collapsible=True`.
    """
    inner: list[str] = []

    if formula is not None:
        eqs = [formula] if isinstance(formula, str) else list(formula)
        for eq in eqs:
            inner.append(f'<div class="formula">{eq}</div>')

    if stats:
        rendered = " &nbsp;·&nbsp; ".join(
            f"<strong>{_esc(label)}</strong> = {value}"
            for label, value in stats
        )
        inner.append(f'<p class="stats">{rendered}</p>')

    if meta:
        inner.append(f'<p class="meta">{meta}</p>')

    if code_ref:
        refs = " &nbsp;·&nbsp; ".join(
            f'<code>{_esc(file)}:{_esc(fn)}</code>'
            for file, fn in code_ref
        )
        inner.append(f'<p class="meta">Code: {refs}</p>')

    if sources:
        joined = " &nbsp;·&nbsp; ".join(sources)
        inner.append(f'<p class="meta">Sources: {joined}</p>')

    inner_html = "".join(inner)
    label = '<span class="lab">Technical apparatus</span>'

    if collapsible:
        body = (
            '<details>'
            f'<summary>{_esc(summary_label)}</summary>'
            f'{inner_html}'
            '</details>'
        )
        return f'<div class="apparatus">{label}{body}</div>\n'

    return f'<div class="apparatus">{label}{inner_html}</div>\n'
