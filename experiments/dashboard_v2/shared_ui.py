"""Shared UI helpers for the Geometria Iuris dashboards.

Design system extracted verbatim from dashboard_v2/index.html so that the
lens pages share a single source of truth for palette, typography and
components. Functions return HTML fragments as strings; the page
generators concatenate them.
"""

from __future__ import annotations

import html
import json
from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Palette (mirrors the Okabe-inspired slots used across plots and text)

PLOT_COLORS = {
    "weird":      "#2c5f9a",
    "sinic":      "#a43a3a",
    "bilingual":  "#5a8f3a",
    "cross":      "#8a6d3b",
    "control":    "#7f7f7f",
    "accent":     "#b08d57",
    "accent_dark":"#8a6d3b",
    "ink":        "#1a1a1a",
    "cream":      "#faf7ee",
    "border":     "#e5e2d8",
    "panel":      "#fff",
}

MODEL_GROUP = {
    "BGE-EN-large":      "weird",
    "E5-large":          "weird",
    "FreeLaw-EN":        "weird",
    "BGE-ZH-large":      "sinic",
    "Text2vec-large-ZH": "sinic",
    "Dmeta-ZH":          "sinic",
    "BGE-M3-EN":         "bilingual",
    "BGE-M3-ZH":         "bilingual",
    "Qwen3-0.6B-EN":     "bilingual",
    "Qwen3-0.6B-ZH":     "bilingual",
}


# --------------------------------------------------------------------------
# CSS and JS extracted from index.html (identical rules, so all three pages
# share exactly the same visual language).

CSS_MAIN = r"""
  :root {
    --bg: #f6f5f1;
    --ink: #1a1a1a;
    --muted: #777;
    --accent: #b08d57;
    --accent-dark: #8a6d3b;
    --panel: #fff;
    --border: #e5e2d8;
    --cream: #faf7ee;
    --good: #2e7d32;
    --warn: #c76a00;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: 'Iowan Old Style', 'Charter', Georgia, serif; line-height: 1.6;
    font-size: 16px; }
  body { scroll-behavior: smooth; }

  header.masthead { padding: 3rem 2rem 1.5rem; text-align: center; background: var(--ink); color: var(--bg); }
  header.masthead h1 { margin: 0 0 0.3rem; font-size: 1.8rem; font-weight: 500; letter-spacing: 0.01em; }
  header.masthead p { margin: 0; color: #c9c5b8; font-style: italic; font-size: 0.95rem; }

  nav.toc { position: sticky; top: 0; background: var(--ink); z-index: 50;
    padding: 0.55rem 2rem; border-bottom: 1px solid #333; }
  nav.toc ul { list-style: none; margin: 0; padding: 0; display: flex; gap: 1.3rem;
    flex-wrap: wrap; max-width: 1000px; margin: 0 auto; font-size: 0.82rem; }
  nav.toc a { color: #c9c5b8; text-decoration: none; padding: 0.2rem 0;
    border-bottom: 2px solid transparent; transition: all 0.15s; }
  nav.toc a:hover { color: var(--accent); border-bottom-color: var(--accent); }

  main { max-width: 900px; margin: 0 auto; padding: 2.5rem 2rem 5rem; }

  section { margin: 3rem 0; }
  section:first-child { margin-top: 0; }

  h2 { font-size: 1.35rem; font-weight: 600; margin: 0 0 1rem;
    padding-bottom: 0.4rem; border-bottom: 2px solid var(--accent); }
  h3 { font-size: 1.08rem; font-weight: 600; margin: 1.5rem 0 0.5rem; color: #333; }
  h4 { font-size: 0.95rem; font-weight: 600; margin: 1.2rem 0 0.4rem; color: #555;
    font-variant: small-caps; letter-spacing: 0.08em; }
  p { margin: 0 0 0.9rem; text-align: justify; }
  p.lead { font-size: 1.05rem; color: #333; }

  .pipeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem;
    margin: 1.5rem 0; }
  .stage { background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
    padding: 0.9rem 0.7rem; text-align: center; cursor: pointer; transition: all 0.15s;
    position: relative; }
  .stage:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 3px 8px rgba(0,0,0,0.07); }
  .stage .n { display: inline-block; font-size: 0.7rem; font-weight: 700; color: var(--accent);
    background: var(--cream); padding: 0.08rem 0.5rem; border-radius: 10px; margin-bottom: 0.4rem; }
  .stage .label { font-size: 0.82rem; font-weight: 600; line-height: 1.3; color: var(--ink); }
  .stage .arrow { position: absolute; right: -0.6rem; top: 50%; transform: translateY(-50%);
    width: 1rem; height: 1rem; border-top: 2px solid var(--accent); border-right: 2px solid var(--accent);
    transform: translateY(-50%) rotate(45deg); opacity: 0.5; z-index: 1; }
  .stage:last-child .arrow { display: none; }

  .stage-detail { display: none; background: var(--cream); border-left: 3px solid var(--accent);
    padding: 1rem 1.2rem; margin-top: 0.5rem; font-size: 0.92rem; }
  .stage-detail.open { display: block; }

  details.entry { background: var(--panel); border: 1px solid var(--border); border-radius: 3px;
    margin: 0.4rem 0; transition: all 0.12s; }
  details.entry[open] { border-color: var(--accent); box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
  details.entry > summary { padding: 0.75rem 1rem; cursor: pointer; list-style: none;
    display: flex; align-items: baseline; gap: 0.8rem; font-weight: 600; }
  details.entry > summary::-webkit-details-marker { display: none; }
  details.entry > summary::before { content: '▸'; color: var(--accent);
    display: inline-block; transition: transform 0.15s; font-size: 0.8rem; }
  details.entry[open] > summary::before { transform: rotate(90deg); }
  details.entry .sym { color: var(--muted); font-weight: 400; font-size: 0.9rem;
    font-style: italic; margin-left: auto; }
  details.entry .body { padding: 0 1.2rem 1rem 2rem; color: #333; }
  details.entry .formula { background: var(--cream); border-radius: 2px;
    padding: 0.5rem 0.8rem; margin: 0.6rem 0; text-align: center; font-size: 1.05em; }
  details.entry .body dl { display: grid; grid-template-columns: 130px 1fr;
    gap: 0.35rem 0.8rem; font-size: 0.9rem; margin: 0.8rem 0 0.3rem; }
  details.entry .body dt { color: var(--muted); font-variant: small-caps;
    letter-spacing: 0.04em; font-weight: 500; }
  details.entry .body dd { margin: 0; }

  table.data { border-collapse: collapse; width: 100%; margin: 1rem 0;
    font-size: 0.9rem; background: var(--panel); }
  table.data th, table.data td { padding: 0.55rem 0.8rem; text-align: left;
    border-bottom: 1px solid var(--border); }
  table.data th { font-variant: small-caps; letter-spacing: 0.05em; color: #555;
    font-weight: 600; background: var(--cream); border-bottom: 2px solid var(--accent); }
  table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.data td.strong { font-weight: 700; color: var(--accent-dark); }
  table.data tr:hover { background: var(--cream); }

  a.metric { color: var(--accent-dark); text-decoration: none;
    border-bottom: 1px dotted var(--accent); font-style: italic;
    transition: all 0.1s; cursor: pointer; }
  a.metric:hover { background: var(--cream); color: var(--ink);
    border-bottom-style: solid; }

  ol.steps { list-style: none; counter-reset: step; padding: 0; margin: 1rem 0; }
  ol.steps li { counter-increment: step; position: relative; padding-left: 2.6rem;
    margin-bottom: 1.1rem; }
  ol.steps li::before { content: counter(step); position: absolute; left: 0; top: 0;
    width: 1.9rem; height: 1.9rem; background: var(--ink); color: var(--bg);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; font-family: Georgia; }
  ol.steps li strong { color: var(--ink); }
  ol.steps li .caption { display: block; color: var(--muted); font-size: 0.88rem; margin-top: 0.2rem; }
  ol.steps li .math { background: var(--cream); border-left: 2px solid var(--accent);
    padding: 0.45rem 0.8rem; margin-top: 0.5rem; font-size: 0.92rem; }

  .pairs { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin: 0.8rem 0; }
  .pairs .col { background: var(--cream); padding: 0.7rem 0.9rem; border-radius: 3px;
    font-size: 0.88rem; }
  .pairs .col h5 { margin: 0 0 0.5rem; font-size: 0.78rem; color: var(--accent-dark);
    font-variant: small-caps; letter-spacing: 0.06em; }
  .pairs .col ul { list-style: none; margin: 0; padding: 0; }
  .pairs .col li { padding: 0.12rem 0; font-variant-numeric: tabular-nums; }
  .pairs .col .pos { color: var(--ink); font-weight: 500; }
  .pairs .col .sep { color: var(--muted); margin: 0 0.4rem; }
  .pairs .col .neg { color: #666; }
  .zh { font-family: 'Source Han Serif', 'Noto Serif CJK SC', 'STSong', serif; }

  .disclaimer { background: #fff8e5; border-left: 3px solid #d4a017;
    padding: 0.9rem 1.2rem; margin: 1.5rem 0; font-size: 0.92rem; color: #6a4f12;
    border-radius: 2px; }

  .button-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
  .button-row a { flex: 1; min-width: 260px; background: var(--ink); color: var(--bg);
    padding: 1rem 1.3rem; border-radius: 3px; text-decoration: none;
    transition: all 0.15s; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
  .button-row a:hover { background: var(--accent-dark); transform: translateY(-1px); }
  .button-row a strong { display: block; color: var(--accent); margin-bottom: 0.25rem; font-size: 1.02rem; }
  .button-row a span { font-size: 0.85rem; color: #d4d0c2; }

  footer { text-align: center; color: var(--muted); font-size: 0.83rem;
    padding: 2rem; border-top: 1px solid var(--border); margin-top: 4rem; }
  footer code { background: var(--panel); padding: 0.1em 0.35em; border-radius: 2px;
    font-family: 'Source Code Pro', Menlo, monospace; font-size: 0.9em; }
  footer a { color: var(--accent-dark); }

  .highlight { background: #fdf2d0; transition: background 2s; }

  .plot { width: 100%; background: var(--panel); border: 1px solid var(--border);
    border-radius: 3px; margin: 1rem 0; }
  .plot-caption { font-size: 0.82rem; color: var(--muted); margin-top: -0.4rem;
    margin-bottom: 1.2rem; }

  .plot-controls { display: flex; gap: 0.8rem; align-items: center;
    margin: 0.6rem 0; font-size: 0.88rem; flex-wrap: wrap; }
  .plot-controls label { color: var(--muted); font-variant: small-caps;
    letter-spacing: 0.05em; font-size: 0.8rem; }
  .plot-controls select { padding: 0.25rem 0.5rem; border: 1px solid var(--border);
    border-radius: 3px; background: var(--panel); font-family: inherit; font-size: 0.85rem;
    color: var(--ink); }
"""


JS_MAIN = r"""
document.addEventListener("DOMContentLoaded", () => {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "\\(", right: "\\)", display: false},
        {left: "\\[", right: "\\]", display: true},
      ],
      throwOnError: false,
    });
  }

  window.toggleStage = (el) => {
    const stages = [...document.querySelectorAll(".stage")];
    const idx = stages.indexOf(el);
    const details = document.querySelectorAll("#stage-details .stage-detail");
    details.forEach((d, i) => d.classList.toggle("open", i === idx && !el.classList.contains("active")));
    stages.forEach(s => s.classList.remove("active"));
    if (!details[idx].classList.contains("open")) return;
    el.classList.add("active");
  };

  document.querySelectorAll("a.metric[data-target]").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = document.getElementById(link.dataset.target);
      if (!target) return;
      target.setAttribute("open", "");
      target.scrollIntoView({behavior: "smooth", block: "center"});
      target.classList.add("highlight");
      setTimeout(() => target.classList.remove("highlight"), 1500);
    });
  });
});
"""


# --------------------------------------------------------------------------
# Plotly default layout for the lens pages.

PLOTLY_LAYOUT_DEFAULTS = {
    "paper_bgcolor": "#fff",
    "plot_bgcolor":  "#fbfaf5",
    "font":          {"family": "Iowan Old Style, Charter, Georgia, serif", "size": 12, "color": "#1a1a1a"},
    "hoverlabel":    {"bgcolor": "#fff", "bordercolor": "#b08d57", "font": {"size": 12}},
    "margin":        {"l": 55, "r": 25, "t": 30, "b": 50},
}

PLOTLY_AXIS_DEFAULTS = {
    "zeroline":  False,
    "gridcolor": "#e5e2d8",
    "linecolor": "#b08d57",
    "ticks":     "outside",
    "tickcolor": "#b08d57",
}


# --------------------------------------------------------------------------
# Fragment helpers. Each returns a string; the generators concatenate.

def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def page_head(title: str, subtitle: str, include_plotly: bool = True) -> str:
    """Open `<!DOCTYPE html>` through `<header class="masthead">`."""
    plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>' if include_plotly else ""
    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<style>
{CSS_MAIN}
</style>
{plotly_tag}
</head>
<body>

<header class="masthead">
  <h1>{_esc(title)}</h1>
  <p>{subtitle}</p>
</header>
"""


def sticky_nav(items: Sequence[tuple[str, str]], back_link: tuple[str, str] | None = None) -> str:
    """Render the sticky TOC.

    `items` is a list of (href, label) pairs for internal anchors.
    `back_link` is an optional (href, label) for an external link marked with ↗.
    """
    parts = ['<nav class="toc"><ul>']
    for href, label in items:
        parts.append(f'<li><a href="{_esc(href)}">{_esc(label)}</a></li>')
    if back_link:
        href, label = back_link
        parts.append(f'<li><a href="{_esc(href)}">{_esc(label)} ↗</a></li>')
    parts.append("</ul></nav>\n")
    return "".join(parts)


def open_main() -> str:
    return "<main>\n"


def close_main() -> str:
    return "</main>\n"


def section_open(section_id: str, heading: str) -> str:
    return f'<!-- ================================================================== -->\n<section id="{_esc(section_id)}">\n<h2>{heading}</h2>\n'


def section_close() -> str:
    return "</section>\n"


def pipeline_diagram(stages: Sequence[tuple[str, str]]) -> str:
    """Render the 5-stage clickable pipeline.

    `stages` is a list of (short_label, detail_html) pairs. Details render as
    expandable paragraphs below the row.
    """
    n = len(stages)
    stage_html = ['<div class="pipeline">']
    for i, (label, _) in enumerate(stages):
        arrow = '<div class="arrow"></div>' if i < n - 1 else ""
        stage_html.append(
            f'<div class="stage" onclick="toggleStage(this)">'
            f'<div class="n">{i+1}</div><div class="label">{label}</div>{arrow}'
            f'</div>'
        )
    stage_html.append("</div>")
    detail_html = ['<div id="stage-details">']
    for i, (_, detail) in enumerate(stages):
        detail_html.append(f'<div class="stage-detail" data-idx="{i}">{detail}</div>')
    detail_html.append("</div>")
    return "\n".join(stage_html + detail_html) + "\n"


def metric_chip(target_id: str, label: str) -> str:
    return f'<a class="metric" data-target="{_esc(target_id)}">{label}</a>'


def steps_list(steps: Sequence[tuple[str, str, str | None]]) -> str:
    """Render an `ol.steps` block.

    Each step is (title_html, caption_html, math_html_or_None).
    """
    parts = ['<ol class="steps">']
    for title, caption, math in steps:
        inner = f'<strong>{title}</strong>'
        if caption:
            inner += f'<span class="caption">{caption}</span>'
        if math:
            inner += f'<div class="math">{math}</div>'
        parts.append(f'<li>{inner}</li>')
    parts.append("</ol>\n")
    return "".join(parts)


def data_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[str]],
    col_classes: Sequence[str] | None = None,
) -> str:
    """Render a styled table.

    `col_classes` is per-column: empty string or e.g. 'num', 'num strong'.
    """
    if col_classes is None:
        col_classes = [""] * len(columns)
    th = "".join(f"<th>{c}</th>" for c in columns)
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="{col_classes[i]}">{v}</td>' if col_classes[i] else f"<td>{v}</td>"
            for i, v in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f'<table class="data">\n<thead><tr>{th}</tr></thead>\n<tbody>\n' + "\n".join(body) + "\n</tbody></table>\n"


def disclaimer(html_body: str) -> str:
    return f'<div class="disclaimer">{html_body}</div>\n'


def details_entry(entry_id: str, summary_main: str, symbol: str, body_html: str) -> str:
    """Render a glossary-style accordion entry."""
    sym = f'<span class="sym">{symbol}</span>' if symbol else ""
    return (
        f'<details class="entry" id="{_esc(entry_id)}">\n'
        f'<summary>{summary_main} {sym}</summary>\n'
        f'<div class="body">{body_html}</div>\n'
        f'</details>\n'
    )


def plotly_embed(fig_dict: dict, div_id: str, height_px: int = 420) -> str:
    """Emit a Plotly `<div>` + instantiation script.

    `fig_dict` is a plain dict with `data` and `layout`. `height_px` controls
    the div height; Plotly is instantiated with `responsive: true` and the
    mode bar disabled.
    """
    data_json = json.dumps(fig_dict.get("data", []), separators=(",", ":"), default=_json_default)
    layout_json = json.dumps(fig_dict.get("layout", {}), separators=(",", ":"), default=_json_default)
    return (
        f'<div id="{_esc(div_id)}" class="plot" style="height:{height_px}px;"></div>\n'
        f'<script>Plotly.newPlot("{_esc(div_id)}", {data_json}, {layout_json}, '
        f'{{displayModeBar:false, responsive:true}});</script>\n'
    )


def _json_default(obj):
    try:
        import numpy as np  # local import so shared_ui works without numpy if unused
    except ImportError:
        raise TypeError(f"cannot serialise {type(obj)}")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj)}")


def plot_caption(text: str) -> str:
    return f'<p class="plot-caption">{text}</p>\n'


def page_footer(body_html: str, js_extra: str = "") -> str:
    """Close `<main>`, emit the footer and the boot script."""
    extra = f"<script>{js_extra}</script>\n" if js_extra else ""
    return (
        "</main>\n\n"
        f"<footer>{body_html}</footer>\n\n"
        '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>\n'
        f"<script>\n{JS_MAIN}\n</script>\n"
        f"{extra}"
        "</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# Canonical glossary entries used across pages.

GLOSSARY_CORE = {
    "g-vector": dict(
        summary="Vettore (embedding)",
        symbol="v ∈ ℝ<sup>d</sup>",
        body=(
            "<p>Un modello linguistico trasforma ogni termine in una sequenza di numeri chiamata "
            "<em>vettore</em> o <em>embedding</em>. Il numero di numeri (la <em>dimensione</em>, "
            "di solito 768 o 1024) è fissato dal modello. Dopo la codifica il vettore viene "
            "normalizzato in modo che la sua lunghezza sia esattamente 1: conta solo la "
            "direzione, non la magnitudine.</p>"
            "<dl><dt>Simbolo</dt><dd>\\( v_i \\in \\mathbb{R}^d \\) con \\( \\lVert v_i \\rVert_2 = 1 \\).</dd></dl>"
        ),
    ),
    "g-cosdist": dict(
        summary="Distanza coseno",
        symbol="1 − cos(u, v) ∈ [0, 2]",
        body=(
            "<p>Misura quanto due vettori puntano in direzioni diverse. Quando due termini "
            "hanno significati simili secondo il modello, i loro vettori puntano in direzioni "
            "simili e la distanza coseno è vicina a 0.</p>"
            "<div class=\"formula\">\\( d_{\\cos}(u, v) = 1 - u \\cdot v \\)</div>"
        ),
    ),
    "g-rdm": dict(
        summary="Matrice di dissimilarità (RDM)",
        symbol="D ∈ ℝ<sup>N×N</sup>",
        body=(
            "<p>Per un pool di N termini e un modello fissato, la RDM è la matrice N×N che "
            "contiene, nella cella (i, j), la "
            + metric_chip("g-cosdist", "distanza coseno")
            + " tra i vettori del termine <em>i</em> e del termine <em>j</em>. Simmetrica, "
            "zero sulla diagonale. Con N=350: 61&nbsp;075 celle indipendenti.</p>"
        ),
    ),
    "g-spearman": dict(
        summary="Correlazione di Spearman",
        symbol="ρ ∈ [−1, +1]",
        body=(
            "<p>Misura quanto due liste di valori, ordinate per rango, sono d'accordo. ρ=+1 "
            "se l'ordinamento è identico, ρ=−1 se è invertito, ρ≈0 se non c'è relazione "
            "monotona.</p>"
            "<div class=\"formula\">\\( \\rho = 1 - \\dfrac{6 \\sum d_i^2}{n(n^2 - 1)} \\)</div>"
        ),
    ),
    "g-mantel": dict(
        summary="Mantel permutation test",
        symbol="p-value",
        body=(
            "<p>Procedura per verificare che una correlazione tra due RDM non derivi dal "
            "caso. Le etichette di una delle due matrici vengono mescolate B volte (qui "
            "B=1000); la p-value è la frazione di permutazioni che produce una correlazione "
            "uguale o superiore a quella osservata.</p>"
            "<div class=\"formula\">\\( p = \\dfrac{1 + \\#\\{b : \\rho^{(b)} \\geq \\rho_{\\text{obs}}\\}}{1 + B} \\)</div>"
            "<dl><dt>Range</dt><dd>\\( p \\in [1/(B+1), 1] \\); qui minimo = 0.001.</dd></dl>"
        ),
    ),
    "g-boot": dict(
        summary="Block bootstrap (intervallo di confidenza)",
        symbol="CI 95%",
        body=(
            "<p>Si estraggono con ripetizione N termini dai 350, si ricostruiscono le sotto-RDM "
            "corrispondenti e si ricalcola la correlazione. 1000 iterazioni producono una "
            "distribuzione; il 2,5° e il 97,5° percentile definiscono l'intervallo di confidenza "
            "al 95%. Il campionamento avviene per <strong>termine</strong>, non per coppia "
            "(Nili et al. 2014).</p>"
        ),
    ),
    "g-kozlowski": dict(
        summary="Asse di valori alla Kozlowski",
        symbol="a ∈ ℝ<sup>d</sup>",
        body=(
            "<p>Un asse di valori è una direzione nello spazio vettoriale che cattura "
            "un'opposizione concettuale. Si costruisce scegliendo P coppie di poli; l'asse "
            "è la media delle differenze tra i vettori delle coppie, normalizzata a modulo 1.</p>"
            "<div class=\"formula\">\\( a = \\mathrm{L2}\\!\\left(\\dfrac{1}{P} \\sum_{p=1}^{P} (v^{+}_{p} - v^{-}_{p})\\right) \\)</div>"
            "<p>Ogni termine riceve un punteggio sull'asse via "
            + metric_chip("g-cosdist", "distanza coseno")
            + " (più precisamente prodotto scalare, essendo i vettori normalizzati).</p>"
        ),
    ),
    "g-effect-r": dict(
        summary="Effect size rank-biserial",
        symbol="r ∈ [−1, +1]",
        body=(
            "<p>Dimensione dell'effetto associata al test di Mann-Whitney. r = 1 − 2U / (n₁·n₂) "
            "(forma semplificata). Valore positivo = la distribuzione <em>x</em> ha ranghi più "
            "bassi della distribuzione <em>y</em>; |r| ≥ 0.5 è convenzionalmente considerato "
            "un effetto grande.</p>"
        ),
    ),
    "g-crosstrad": dict(
        summary="Media cross-tradizione",
        symbol="ρ̄<sub>cross</sub>",
        body=(
            "<p>Media aritmetica delle 9 correlazioni di Spearman tra i 3 modelli WEIRD e i 3 "
            "modelli Sinic.</p>"
            "<div class=\"formula\">\\( \\bar{\\rho}_{\\text{cross}} = \\dfrac{1}{9} \\sum_{w \\in W} \\sum_{s \\in S} \\rho(w, s) \\)</div>"
        ),
    ),
    "g-withintrad": dict(
        summary="Media intra-tradizione",
        symbol="ρ̄<sub>W</sub>, ρ̄<sub>S</sub>",
        body=(
            "<p>Media delle correlazioni tra coppie della stessa tradizione (3 per WEIRD, "
            "3 per Sinic). Fornisce il limite superiore della comparabilità attesa tra modelli "
            "che condividono la stessa matrice culturale d'addestramento.</p>"
        ),
    ),
    "g-bilingual": dict(
        summary="Controllo causale bilingue",
        symbol="ρ̄<sub>β</sub>",
        body=(
            "<p>I modelli bilingui (BGE-M3, Qwen3) codificano EN e ZH nel medesimo spazio. "
            "ρ̄<sub>β</sub> è la media sui due modelli delle correlazioni tra la propria RDM-EN e "
            "la propria RDM-ZH. Il confronto con "
            + metric_chip("g-crosstrad", "ρ̄<sub>cross</sub>")
            + " separa l'effetto di architettura dall'effetto di tradizione giuridica.</p>"
        ),
    ),
}


def glossary_section(entry_ids: Sequence[str], extra_entries: Mapping[str, dict] | None = None) -> str:
    """Render a glossary section picking entries from GLOSSARY_CORE and
    optional page-specific extras."""
    pool = dict(GLOSSARY_CORE)
    if extra_entries:
        pool.update(extra_entries)
    parts = [
        section_open("glossary", "Glossario delle metriche"),
        '<p>Ogni metrica è spiegata due volte: in parole, e in formula. Gli elementi '
        'nel testo scritti in <a class="metric">corsivo dorato</a> rimandano qui.</p>',
    ]
    for eid in entry_ids:
        if eid not in pool:
            continue
        e = pool[eid]
        parts.append(details_entry(eid, e["summary"], e.get("symbol", ""), e["body"]))
    parts.append(section_close())
    return "\n".join(parts)
