"""Shared UI helpers for the Geometria Iuris dashboard (final).

Self-contained HTML output: Plotly loaded from a vendored
`assets/plotly.min.js`, no CDN, no KaTeX. Formulae use Unicode (Δρ_sym,
ρ̄, etc.) and `<sub>/<sup>`.

Each sub-section of an experimental page follows the same five-step
concentric pattern:

    1. scenario_block       legal scenario, italic, 1-2 sentences
    2. result_block         verdict in plain words, 1 sentence
    3. plot_block           annotated Plotly figure
    4. takehome_block       1 sentence, pointer to Cap.4
    5. apparatus_block      L2 technical apparatus (formula + stats + code)

`apparatus_block` is defined in `apparatus.py`.
"""

from __future__ import annotations

import html
import json
from typing import Iterable, Sequence


# --------------------------------------------------------------------------
# Palette — Okabe-inspired slots, identical to v3.

PLOT_COLORS = {
    "weird":       "#2c5f9a",
    "sinic":       "#a43a3a",
    "bilingual":   "#5a8f3a",
    "cross":       "#8a6d3b",
    "control":     "#7f7f7f",
    "accent":      "#b08d57",
    "accent_dark": "#8a6d3b",
    "ink":         "#1a1a1a",
    "cream":       "#faf7ee",
    "border":      "#e5e2d8",
    "panel":       "#fff",
    "takehome":    "#2c5f9a",
    "apparatus":   "#6a4f12",
    "scenario":    "#8a6d3b",
    "good":        "#2e7d32",
    "warn":        "#c76a00",
}


# --------------------------------------------------------------------------
# CSS — single source of truth. Written verbatim into `assets/style.css`
# by `build.py` and linked from every page.

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
    --takehome: #2c5f9a;
    --scenario: #8a6d3b;
    --apparatus: #6a4f12;
    --good: #2e7d32;
    --warn: #c76a00;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: 'Iowan Old Style', 'Charter', Georgia, serif; line-height: 1.6;
    font-size: 16px; }
  body { scroll-behavior: smooth; }

  header.masthead { padding: 3rem 2rem 1.5rem; text-align: center;
    background: var(--ink); color: var(--bg); }
  header.masthead h1 { margin: 0 0 0.3rem; font-size: 1.75rem; font-weight: 500;
    letter-spacing: 0.01em; }
  header.masthead p { margin: 0; color: #c9c5b8; font-style: italic; font-size: 0.95rem; }
  header.masthead .crumb { font-size: 0.78rem; color: #8a8472; letter-spacing: 0.08em;
    font-variant: small-caps; margin-bottom: 0.6rem; }

  nav.toc { position: sticky; top: 0; background: var(--ink); z-index: 50;
    padding: 0.55rem 2rem; border-bottom: 1px solid #333; }
  nav.toc ul { list-style: none; margin: 0 auto; padding: 0; display: flex;
    gap: 1.3rem; flex-wrap: wrap; max-width: 1000px; font-size: 0.82rem; }
  nav.toc a { color: #c9c5b8; text-decoration: none; padding: 0.2rem 0;
    border-bottom: 2px solid transparent; transition: all 0.15s; }
  nav.toc a:hover, nav.toc a.current { color: var(--accent);
    border-bottom-color: var(--accent); }

  main { max-width: 880px; margin: 0 auto; padding: 2.5rem 2rem 5rem; }

  section { margin: 3rem 0; }
  section:first-child { margin-top: 0; }

  h2 { font-size: 1.35rem; font-weight: 600; margin: 0 0 1rem;
    padding-bottom: 0.4rem; border-bottom: 2px solid var(--accent); }
  h3 { font-size: 1.08rem; font-weight: 600; margin: 1.5rem 0 0.5rem; color: #333; }
  h4 { font-size: 0.95rem; font-weight: 600; margin: 1.2rem 0 0.4rem; color: #555;
    font-variant: small-caps; letter-spacing: 0.08em; }

  p { margin: 0 0 0.9rem; text-align: justify; }
  p.lead { font-size: 1.05rem; color: #333; }

  .step-label { font-size: 0.66rem; font-variant: small-caps;
    letter-spacing: 0.1em; color: var(--accent-dark); font-weight: 700;
    margin: 1.6rem 0 0.4rem; display: block; }

  /* Scenario — italic blockquote, 1-2 sentences setting a legal context */
  .scenario { background: var(--cream); border-left: 3px solid var(--scenario);
    padding: 0.9rem 1.2rem; margin: 0.6rem 0 1.4rem;
    font-style: italic; color: #3a2f12; border-radius: 0 2px 2px 0;
    font-size: 0.98rem; line-height: 1.6; }
  .scenario .lab { font-size: 0.66rem; font-variant: small-caps;
    letter-spacing: 0.1em; color: var(--scenario); font-weight: 700;
    display: block; margin-bottom: 0.35rem; font-style: normal; }

  /* Result — 1-sentence verdict, plain words, no jargon */
  .result { background: var(--panel); border: 1px solid var(--border);
    border-left: 3px solid var(--good); border-radius: 0 2px 2px 0;
    padding: 0.7rem 1.1rem; margin: 0.6rem 0 1.2rem;
    font-size: 1rem; font-weight: 500; color: var(--ink); }
  .result .lab { font-size: 0.66rem; font-variant: small-caps;
    letter-spacing: 0.1em; color: var(--good); font-weight: 700;
    display: block; margin-bottom: 0.3rem; }

  /* Question (alternative to scenario/result for sub-sections that frame
     the experimental question rather than a courtroom scenario) */
  .question { color: var(--ink); margin: 0 0 1.4rem; font-size: 1rem;
    line-height: 1.65; }
  .question strong { color: var(--ink); }

  /* Procedure — ordered step list with embedded numeric examples */
  ol.procedure { list-style: none; counter-reset: step;
    padding: 0; margin: 0.5rem 0 1.4rem; }
  ol.procedure > li { counter-increment: step; position: relative;
    padding-left: 2.4rem; margin-bottom: 1rem; line-height: 1.55;
    font-size: 0.95rem; }
  ol.procedure > li::before { content: counter(step); position: absolute;
    left: 0; top: 0.05rem; width: 1.7rem; height: 1.7rem;
    background: var(--cream); color: var(--accent-dark);
    border: 1px solid var(--accent); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.8rem; font-family: Georgia, serif; }
  ol.procedure > li .example { display: block; margin-top: 0.4rem;
    background: var(--cream); border-left: 2px solid var(--accent);
    padding: 0.4rem 0.7rem; font-size: 0.88rem; color: #444;
    border-radius: 0 2px 2px 0; }
  ol.procedure > li .example strong { color: var(--accent-dark);
    font-variant-numeric: tabular-nums; }

  .plot-frame { background: var(--panel); border: 1px solid var(--border);
    border-radius: 3px; margin: 0.8rem 0 0.4rem; }
  .plot-caption { font-size: 0.84rem; color: var(--muted);
    margin: 0.3rem 0 1.3rem; line-height: 1.5; }

  /* Meaning of numbers — what each metric measures, with its limits */
  .meaning { background: var(--panel); border: 1px solid var(--border);
    border-left: 3px solid var(--takehome); border-radius: 0 2px 2px 0;
    padding: 0.85rem 1.1rem; margin: 1.2rem 0; }
  .meaning .lab { font-size: 0.66rem; font-variant: small-caps;
    letter-spacing: 0.1em; color: var(--takehome); font-weight: 700;
    display: block; margin-bottom: 0.4rem; }
  .meaning dl { display: grid; grid-template-columns: minmax(140px, max-content) 1fr;
    gap: 0.4rem 1rem; margin: 0; font-size: 0.92rem; }
  .meaning dt { color: var(--takehome); font-weight: 600;
    font-variant-numeric: tabular-nums;
    font-family: 'Source Code Pro', Menlo, monospace; font-size: 0.85rem; }
  .meaning dd { margin: 0; color: var(--ink); line-height: 1.55; }
  .meaning .limit { display: block; margin-top: 0.25rem;
    color: #555; font-size: 0.85rem; font-style: italic; }
  .meaning .limit::before { content: "What it does NOT say · "; font-style: normal;
    color: var(--muted); font-variant: small-caps; letter-spacing: 0.06em;
    font-weight: 700; font-size: 0.78rem; }

  /* Take-home — single-sentence pointer to Cap.4 */
  .takehome { background: var(--panel); border: 1px solid var(--border);
    border-left: 3px solid var(--takehome); border-radius: 0 2px 2px 0;
    padding: 0.85rem 1.1rem; margin: 1.2rem 0 1.6rem;
    font-size: 0.96rem; color: var(--ink); line-height: 1.55; }
  .takehome .lab { font-size: 0.66rem; font-variant: small-caps;
    letter-spacing: 0.1em; color: var(--takehome); font-weight: 700;
    display: block; margin-bottom: 0.3rem; }

  /* Apparatus — technical L2 box, full width, bronze left border */
  .apparatus { background: var(--panel); border: 1px solid var(--border);
    border-left: 3px solid var(--apparatus); border-radius: 0 2px 2px 0;
    padding: 0.85rem 1.1rem; margin: 1.2rem 0 2rem;
    font-size: 0.88rem; color: #333; }
  .apparatus .lab { font-size: 0.66rem; font-variant: small-caps;
    letter-spacing: 0.1em; color: var(--apparatus); font-weight: 700;
    display: block; margin-bottom: 0.5rem; }
  .apparatus .formula { background: var(--cream); border-radius: 2px;
    padding: 0.45rem 0.7rem; margin: 0.4rem 0 0.6rem; text-align: center;
    font-size: 0.95em; font-family: 'Source Code Pro', Menlo, monospace; }
  .apparatus .stats { margin: 0.3rem 0; font-variant-numeric: tabular-nums; }
  .apparatus .stats strong { color: var(--apparatus); font-weight: 600; }
  .apparatus .meta { font-size: 0.8rem; color: #666; margin-top: 0.5rem;
    line-height: 1.5; }
  .apparatus code { font-family: 'Source Code Pro', Menlo, monospace;
    font-size: 0.85em; background: var(--cream); padding: 0.1em 0.35em;
    border-radius: 2px; color: #444; }
  .apparatus details > summary { cursor: pointer; outline: none;
    color: var(--apparatus); font-variant: small-caps; letter-spacing: 0.06em;
    font-weight: 700; font-size: 0.78rem; }
  .apparatus details[open] > summary { margin-bottom: 0.5rem; }

  /* Pipeline diagram (clickable 5-stage strip) */
  .pipeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem;
    margin: 1.5rem 0; }
  .stage { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.9rem 0.7rem; text-align: center;
    cursor: pointer; transition: all 0.15s; position: relative; }
  .stage:hover { border-color: var(--accent); transform: translateY(-2px);
    box-shadow: 0 3px 8px rgba(0,0,0,0.07); }
  .stage.active { border-color: var(--accent); background: var(--cream); }
  .stage .n { display: inline-block; font-size: 0.7rem; font-weight: 700;
    color: var(--accent); background: var(--cream);
    padding: 0.08rem 0.5rem; border-radius: 10px; margin-bottom: 0.4rem; }
  .stage .label { font-size: 0.82rem; font-weight: 600; line-height: 1.3;
    color: var(--ink); }
  .stage .arrow { position: absolute; right: -0.6rem; top: 50%;
    width: 1rem; height: 1rem;
    border-top: 2px solid var(--accent); border-right: 2px solid var(--accent);
    transform: translateY(-50%) rotate(45deg); opacity: 0.5; z-index: 1; }
  .stage:last-child .arrow { display: none; }
  .stage-detail { display: none; background: var(--cream);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.2rem; margin-top: 0.5rem; font-size: 0.92rem; }
  .stage-detail.open { display: block; }

  /* Verification gate badge (used on the Home page) */
  .gate-badge { display: inline-flex; align-items: center; gap: 0.6rem;
    background: var(--cream); border: 1px solid var(--good);
    color: var(--good); padding: 0.45rem 1rem; border-radius: 999px;
    font-size: 0.88rem; font-weight: 600; letter-spacing: 0.02em;
    margin: 0.6rem 0; }
  .gate-badge .dot { width: 0.55rem; height: 0.55rem; border-radius: 50%;
    background: var(--good); display: inline-block; }

  /* Anchor cards (used on Home for the three anchor results) */
  .anchor-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem; margin: 1.5rem 0 2rem; }
  .anchor-card { background: var(--panel); border: 1px solid var(--border);
    border-top: 3px solid var(--accent); border-radius: 3px;
    padding: 1rem 1.1rem 1.1rem; }
  .anchor-card .tag { font-size: 0.68rem; font-variant: small-caps;
    letter-spacing: 0.08em; color: var(--accent-dark); font-weight: 700;
    margin-bottom: 0.4rem; display: block; }
  .anchor-card h3 { margin: 0 0 0.5rem; font-size: 1rem; color: var(--ink); }
  .anchor-card p { font-size: 0.92rem; margin: 0; line-height: 1.55; }

  /* Tables */
  table.data { border-collapse: collapse; width: 100%; margin: 1rem 0;
    font-size: 0.9rem; background: var(--panel); }
  table.data th, table.data td { padding: 0.55rem 0.8rem; text-align: left;
    border-bottom: 1px solid var(--border); }
  table.data th { font-variant: small-caps; letter-spacing: 0.05em; color: #555;
    font-weight: 600; background: var(--cream);
    border-bottom: 2px solid var(--accent); }
  table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.data td.strong { font-weight: 700; color: var(--accent-dark); }
  table.data tr.highlight { background: #fff8e5; }
  table.data tr.highlight td.strong { color: #6a4f12; }
  table.data tr:hover { background: var(--cream); }

  /* Highlighted single-number callout (used on Robustness Y caveat) */
  .number-callout { display: flex; align-items: baseline; gap: 0.8rem;
    background: #fff8e5; border: 1px solid #d4a017; border-radius: 4px;
    padding: 1rem 1.4rem; margin: 1.4rem 0; }
  .number-callout .value { font-size: 2.4rem; font-weight: 700;
    color: var(--accent-dark); font-variant-numeric: tabular-nums;
    line-height: 1; }
  .number-callout .descr { font-size: 0.92rem; color: #4a3f1c;
    line-height: 1.5; }
  .number-callout .descr strong { color: var(--accent-dark); }

  /* Disclaimer */
  .disclaimer { background: #fff8e5; border-left: 3px solid #d4a017;
    padding: 0.9rem 1.2rem; margin: 1.5rem 0; font-size: 0.92rem;
    color: #6a4f12; border-radius: 2px; }

  /* Linear nav (prev/next at page bottom) */
  nav.linear { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
    margin: 4rem 0 0.5rem; border-top: 1px solid var(--border);
    padding-top: 1.5rem; }
  nav.linear a { display: block; padding: 0.95rem 1.1rem; background: var(--panel);
    border: 1px solid var(--border); border-radius: 3px; text-decoration: none;
    color: var(--ink); transition: all 0.15s; }
  nav.linear a:hover { border-color: var(--accent); background: var(--cream);
    transform: translateY(-1px); }
  nav.linear a.prev { text-align: left; }
  nav.linear a.next { text-align: right; grid-column: 2; }
  nav.linear a .dir { display: block; font-size: 0.72rem; color: var(--muted);
    font-variant: small-caps; letter-spacing: 0.06em; margin-bottom: 0.15rem; }
  nav.linear a .lab { font-weight: 600; color: var(--accent-dark); font-size: 0.98rem; }
  nav.linear .spacer { background: transparent; border: none; }
  nav.linear.first-page { grid-template-columns: minmax(0, 480px);
    justify-content: center; }
  nav.linear.first-page a.next { grid-column: 1; text-align: center; }

  footer { text-align: center; color: var(--muted); font-size: 0.83rem;
    padding: 2rem; border-top: 1px solid var(--border); margin-top: 4rem; }
  footer code { background: var(--panel); padding: 0.1em 0.35em;
    border-radius: 2px; font-family: 'Source Code Pro', Menlo, monospace;
    font-size: 0.9em; }
  footer a { color: var(--accent-dark); }

  @media (max-width: 700px) {
    .pipeline { grid-template-columns: 1fr; }
    .stage .arrow { display: none; }
    nav.linear { grid-template-columns: 1fr; }
    nav.linear a.next { grid-column: 1; }
    main { padding: 1.5rem 1.1rem 4rem; }
    .anchor-grid { grid-template-columns: 1fr; }
    .number-callout { flex-direction: column; gap: 0.3rem; }
  }

  /* Lexicon page — input snapshots */
  details.domain-block {
    margin: 0.6rem 0 1.1rem; border: 1px solid var(--border);
    border-radius: 4px; background: var(--panel); overflow: hidden; }
  details.domain-block > summary {
    cursor: pointer; padding: 0.8rem 1.1rem; font-weight: 600;
    font-size: 1.02rem; color: var(--accent-dark);
    background: var(--cream); border-bottom: 1px solid var(--border);
    list-style: none; }
  details.domain-block > summary::-webkit-details-marker { display: none; }
  details.domain-block > summary::before {
    content: "▸ "; display: inline-block; transition: transform 0.15s;
    color: var(--accent); margin-right: 0.3rem; }
  details.domain-block[open] > summary::before { transform: rotate(90deg); }
  details.domain-block > .inner { padding: 0.6rem 1.1rem 1.1rem; }
  details.domain-block .domain-count { color: var(--muted);
    font-weight: 400; font-size: 0.9em; margin-left: 0.4rem; }

  details.term-row {
    border-bottom: 1px solid var(--border); padding: 0.4rem 0; }
  details.term-row:last-child { border-bottom: none; }
  details.term-row > summary {
    cursor: pointer; padding: 0.35rem 0.4rem; font-size: 0.95rem;
    list-style: none; display: flex; flex-wrap: wrap; gap: 0.6rem;
    align-items: baseline; }
  details.term-row > summary::-webkit-details-marker { display: none; }
  details.term-row > summary::before {
    content: "+"; display: inline-block; width: 1em; color: var(--accent);
    font-weight: 700; }
  details.term-row[open] > summary::before { content: "−"; }
  details.term-row .term-en { font-weight: 600; color: var(--ink); }
  details.term-row .term-zh { color: var(--accent-dark);
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", serif; }
  details.term-row .term-meta { color: var(--muted); font-size: 0.85em;
    margin-left: auto; font-style: italic; }
  details.term-row .contexts {
    margin: 0.6rem 0 0.4rem 1.4rem; padding: 0.6rem 0.9rem;
    background: var(--cream); border-left: 3px solid var(--accent);
    border-radius: 2px; }
  details.term-row .ctx {
    margin: 0.3rem 0; line-height: 1.55; font-size: 0.9rem; }
  details.term-row .ctx .ref { font-weight: 600;
    color: var(--accent-dark); font-size: 0.85em;
    margin-right: 0.4rem; white-space: nowrap; }
  details.term-row .ctx .lang-tag {
    display: inline-block; font-size: 0.7em; font-weight: 700;
    color: var(--muted); margin-right: 0.4rem;
    border: 1px solid var(--border); padding: 0 0.3em; border-radius: 2px; }
  details.term-row .ctx.zh { font-family: "Noto Sans CJK SC",
    "Microsoft YaHei", "PingFang SC", serif; }

  /* Axis pairs view */
  .axis-pairs { display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem; margin: 0.8rem 0; }
  .axis-pairs .pairs-column {
    background: var(--cream); border-left: 3px solid var(--accent);
    padding: 0.6rem 0.9rem; border-radius: 2px; }
  .axis-pairs .pairs-column h4 {
    margin: 0 0 0.5rem; font-size: 0.95rem; color: var(--accent-dark); }
  .axis-pairs ol { margin: 0; padding-left: 1.4rem;
    font-size: 0.92rem; line-height: 1.7; }
  .axis-pairs ol li .pos { font-weight: 600; color: var(--ink); }
  .axis-pairs ol li .sep { color: var(--muted); margin: 0 0.3em; }
  .axis-pairs ol li .neg { color: var(--accent-dark); font-style: italic; }

  /* Probe template view */
  .probe-block {
    margin: 0.8rem 0; padding: 0.8rem 1rem;
    background: var(--cream); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 2px; }
  .probe-block h4 { margin: 0 0 0.4rem; font-size: 0.98rem;
    color: var(--accent-dark); }
  .probe-block .threshold {
    font-size: 0.86rem; color: var(--muted); font-style: italic;
    margin-bottom: 0.5rem; }
  .probe-block .cats {
    font-size: 0.9rem; line-height: 1.7;
    padding: 0.4rem 0.6rem; background: #fff;
    border-radius: 2px; margin-bottom: 0.5rem; }
  .probe-block .cats .cat-idx {
    font-weight: 600; color: var(--accent); font-size: 0.82em;
    margin-right: 0.3em; }
  .probe-block .cats .expected {
    background: rgba(176, 141, 87, 0.18); padding: 0 0.2em;
    border-radius: 2px; font-weight: 600; }
  .probe-block .templates ol {
    margin: 0; padding-left: 1.5rem; font-size: 0.9rem;
    line-height: 1.7; color: var(--ink); }
  .probe-block .templates ol li { margin-bottom: 0.25rem; }
  .probe-block .templates ol li .slot {
    background: rgba(176, 141, 87, 0.25);
    padding: 0 0.25em; border-radius: 2px;
    font-style: italic; color: var(--accent-dark); }

  /* Compact control list */
  .control-grid { display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 0.4rem; font-size: 0.9rem; margin: 0.6rem 0; }
  .control-grid .item {
    padding: 0.3rem 0.5rem; background: var(--cream);
    border-left: 2px solid var(--accent);
    border-radius: 2px; }
  .control-grid .item .en { font-weight: 500; color: var(--ink); }
  .control-grid .item .zh { color: var(--accent-dark); margin-left: 0.3em;
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", serif; }
"""


JS_MAIN = r"""
document.addEventListener("DOMContentLoaded", () => {
  window.toggleStage = (el) => {
    const stages = [...document.querySelectorAll(".stage")];
    const idx = stages.indexOf(el);
    const details = document.querySelectorAll("#stage-details .stage-detail");
    const wasActive = el.classList.contains("active");
    details.forEach(d => d.classList.remove("open"));
    stages.forEach(s => s.classList.remove("active"));
    if (!wasActive && details[idx]) {
      details[idx].classList.add("open");
      el.classList.add("active");
    }
  };
});
"""


# --------------------------------------------------------------------------
# Plotly defaults.

PLOTLY_LAYOUT_DEFAULTS = {
    "paper_bgcolor": "#fff",
    "plot_bgcolor":  "#fbfaf5",
    "font":          {"family": "Iowan Old Style, Charter, Georgia, serif",
                      "size": 12, "color": "#1a1a1a"},
    "hoverlabel":    {"bgcolor": "#fff", "bordercolor": "#b08d57",
                      "font": {"size": 12}},
    "margin":        {"l": 55, "r": 25, "t": 50, "b": 50},
}

PLOTLY_AXIS_DEFAULTS = {
    "zeroline":  False,
    "gridcolor": "#e5e2d8",
    "linecolor": "#b08d57",
    "ticks":     "outside",
    "tickcolor": "#b08d57",
    "tickfont":  {"size": 11},
}


# --------------------------------------------------------------------------
# Navigation chain (the six pages).

NAV_ITEMS: list[tuple[str, str]] = [
    ("index.html",              "Home"),
    ("methodology.html",       "Methodology"),
    ("how_it_works.html",      "How it works"),
    ("lexicon.html",            "Inside the inputs"),
    ("experiment_31.html",     "Experiment §3.1"),
    ("experiment_32.html",     "Experiment §3.2"),
    ("robustness_caveats.html", "Robustness & caveats"),
]


# --------------------------------------------------------------------------
# HTML scaffolding.

def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def page_head(
    title: str,
    subtitle: str,
    *,
    include_plotly: bool = True,
    crumb: str | None = None,
) -> str:
    """Open `<!DOCTYPE html>` through `<header class="masthead">`.

    `crumb` is rendered above the title in small caps (e.g. "Chapter 3 · §3.1").
    `include_plotly` toggles the local `<script src="assets/plotly.min.js">`.
    """
    plotly_tag = ('<script src="assets/plotly.min.js"></script>'
                  if include_plotly else "")
    crumb_html = (f'<div class="crumb">{_esc(crumb)}</div>'
                  if crumb else "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="assets/style.css">
{plotly_tag}
</head>
<body>

<header class="masthead">
  {crumb_html}
  <h1>{_esc(title)}</h1>
  <p>{subtitle}</p>
</header>
"""


def sticky_nav(current_href: str | None = None) -> str:
    """Render the sticky TOC linking the six pages.

    `current_href` (if given) receives the `current` class — visual
    highlight in the bar.
    """
    parts = ['<nav class="toc"><ul>']
    for href, label in NAV_ITEMS:
        cls = ' class="current"' if href == current_href else ""
        parts.append(f'<li><a{cls} href="{_esc(href)}">{_esc(label)}</a></li>')
    parts.append("</ul></nav>\n")
    return "".join(parts)


def open_main() -> str:
    return "<main>\n"


def close_main() -> str:
    return "</main>\n"


def section_open(section_id: str, heading: str) -> str:
    return (f'<!-- {"=" * 70} -->\n'
            f'<section id="{_esc(section_id)}">\n<h2>{heading}</h2>\n')


def section_close() -> str:
    return "</section>\n"


# --------------------------------------------------------------------------
# Concentric pattern — five blocks.

def scenario_block(html_body: str) -> str:
    """Italic legal scenario, 1-2 sentences."""
    return (
        '<div class="scenario">'
        '<span class="lab">Legal scenario</span>'
        f'{html_body}</div>\n'
    )


def result_block(html_body: str) -> str:
    """One-sentence verdict, plain words."""
    return (
        '<div class="result">'
        '<span class="lab">Result in words</span>'
        f'{html_body}</div>\n'
    )


def question_block(html_body: str) -> str:
    """Experimental question (alternative to scenario for setup sections)."""
    return (
        '<span class="step-label">Question</span>\n'
        f'<p class="question">{html_body}</p>\n'
    )


def procedure_block(steps: Sequence[tuple[str, str | None]]) -> str:
    """Numbered procedure as an ordered list."""
    parts = ['<span class="step-label">Procedure</span>\n',
             '<ol class="procedure">\n']
    for entry in steps:
        if isinstance(entry, str):
            step_html, example_html = entry, None
        else:
            step_html, example_html = entry
        parts.append('<li>')
        parts.append(step_html)
        if example_html:
            parts.append(f'<div class="example">{example_html}</div>')
        parts.append('</li>\n')
    parts.append('</ol>\n')
    return "".join(parts)


def meaning_block(
    rows: Sequence[tuple[str, str, str | None]],
    label: str = "What the numbers mean",
) -> str:
    """Definition list: each metric, its reading, and its limit."""
    parts = ['<div class="meaning">\n',
             f'<span class="lab">{_esc(label)}</span>\n',
             '<dl>\n']
    for entry in rows:
        if len(entry) == 2:
            symbol, says = entry
            limit = None
        else:
            symbol, says, limit = entry
        parts.append(f'<dt>{symbol}</dt><dd>{says}')
        if limit:
            parts.append(f'<span class="limit">{limit}</span>')
        parts.append('</dd>\n')
    parts.append('</dl>\n</div>\n')
    return "".join(parts)


def takehome_block(html_body: str) -> str:
    """One-sentence take-home, pointer to Cap.4."""
    return (
        '<div class="takehome">'
        '<span class="lab">Take-home</span>'
        f'{html_body}</div>\n'
    )


def plot_block(
    fig_dict: dict,
    div_id: str,
    *,
    height_px: int | None = None,
    caption: str = "",
) -> str:
    """Render a Plotly figure inside a `.plot-frame`, with optional caption."""
    layout = dict(fig_dict.get("layout", {}))
    layout_height = layout.get("height")
    if height_px is None:
        height_px = (layout_height
                     if isinstance(layout_height, (int, float)) else 420)
    layout["height"] = height_px
    data_json = json.dumps(fig_dict.get("data", []),
                           separators=(",", ":"), default=_json_default)
    layout_json = json.dumps(layout, separators=(",", ":"),
                             default=_json_default)
    cap_html = (f'<p class="plot-caption">{caption}</p>\n'
                if caption else "")
    return (
        f'<div id="{_esc(div_id)}" class="plot-frame" '
        f'style="height:{height_px}px;"></div>\n'
        f'<script>Plotly.newPlot("{_esc(div_id)}", {data_json}, '
        f'{layout_json}, {{displayModeBar:false, responsive:true}});</script>\n'
        f'{cap_html}'
    )


def annotate_finding(
    fig_dict: dict,
    x: float,
    y: float,
    text: str,
    *,
    xref: str = "x",
    yref: str = "y",
    ax: float = 40,
    ay: float = -40,
    bgcolor: str = "#fff8e5",
    bordercolor: str = "#d4a017",
) -> dict:
    """Append a Plotly annotation to `fig_dict.layout.annotations`."""
    layout = fig_dict.setdefault("layout", {})
    annotations = list(layout.get("annotations", []))
    annotations.append({
        "x": x, "y": y, "xref": xref, "yref": yref,
        "text": text,
        "showarrow": True, "arrowhead": 3, "arrowwidth": 1.3,
        "arrowcolor": bordercolor,
        "ax": ax, "ay": ay,
        "bgcolor": bgcolor, "bordercolor": bordercolor,
        "borderwidth": 1, "borderpad": 4,
        "font": {"size": 11, "color": "#6a4f12"},
        "align": "left",
    })
    layout["annotations"] = annotations
    return fig_dict


# --------------------------------------------------------------------------
# Linear nav, footer.

def linear_nav(
    prev: tuple[str, str] | None,
    next_: tuple[str, str] | None,
    *,
    first_page: bool = False,
) -> str:
    """Bottom bar with prev/next links."""
    if first_page and next_ is not None and prev is None:
        href, label = next_
        return (
            '<nav class="linear first-page">\n'
            f'<a class="next" href="{_esc(href)}">'
            f'<span class="dir">Next →</span>'
            f'<span class="lab">{_esc(label)}</span></a>\n'
            '</nav>\n'
        )
    parts = ['<nav class="linear">']
    if prev is not None:
        href, label = prev
        parts.append(
            f'<a class="prev" href="{_esc(href)}">'
            f'<span class="dir">← Previous</span>'
            f'<span class="lab">{_esc(label)}</span></a>'
        )
    else:
        parts.append('<div class="spacer"></div>')
    if next_ is not None:
        href, label = next_
        parts.append(
            f'<a class="next" href="{_esc(href)}">'
            f'<span class="dir">Next →</span>'
            f'<span class="lab">{_esc(label)}</span></a>'
        )
    else:
        parts.append('<div class="spacer"></div>')
    parts.append('</nav>\n')
    return "".join(parts)


def page_footer(body_html: str) -> str:
    return (
        "</main>\n\n"
        f"<footer>{body_html}</footer>\n\n"
        f"<script>\n{JS_MAIN}\n</script>\n"
        "</body>\n</html>\n"
    )


def _default_footer() -> str:
    return (
        'Geometria Iuris — companion to the dissertation. '
        'LUISS · Methodology of Legal Science.'
    )


# --------------------------------------------------------------------------
# Pipeline diagram (clickable 5-stage strip).

def pipeline_diagram(stages: Sequence[tuple[str, str]]) -> str:
    """Render a 5-stage clickable pipeline."""
    n = len(stages)
    stage_html = ['<div class="pipeline">']
    for i, (label, _) in enumerate(stages):
        arrow = '<div class="arrow"></div>' if i < n - 1 else ""
        stage_html.append(
            f'<div class="stage" onclick="toggleStage(this)">'
            f'<div class="n">{i+1}</div>'
            f'<div class="label">{_esc(label)}</div>{arrow}'
            f'</div>'
        )
    stage_html.append("</div>")
    detail_html = ['<div id="stage-details">']
    for i, (_, detail) in enumerate(stages):
        detail_html.append(
            f'<div class="stage-detail" data-idx="{i}">{detail}</div>'
        )
    detail_html.append("</div>")
    return "\n".join(stage_html + detail_html) + "\n"


# --------------------------------------------------------------------------
# Tables, callouts, disclaimers, gate badge, anchor cards.

def data_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[str]],
    col_classes: Sequence[str] | None = None,
    row_classes: Sequence[str] | None = None,
) -> str:
    if col_classes is None:
        col_classes = [""] * len(columns)
    rows = list(rows)
    if row_classes is None:
        row_classes = [""] * len(rows)
    th = "".join(f"<th>{c}</th>" for c in columns)
    body = []
    for row, rc in zip(rows, row_classes):
        cells = "".join(
            (f'<td class="{col_classes[i]}">{v}</td>'
             if col_classes[i] else f"<td>{v}</td>")
            for i, v in enumerate(row)
        )
        tr_open = f'<tr class="{rc}">' if rc else "<tr>"
        body.append(f"{tr_open}{cells}</tr>")
    return (
        f'<table class="data">\n<thead><tr>{th}</tr></thead>\n<tbody>\n'
        + "\n".join(body)
        + "\n</tbody></table>\n"
    )


def number_callout(value: str, descr_html: str) -> str:
    """Big-number callout, used on Robustness for the 0.378 anchor."""
    return (
        '<div class="number-callout">'
        f'<div class="value">{_esc(value)}</div>'
        f'<div class="descr">{descr_html}</div>'
        '</div>\n'
    )


def disclaimer(html_body: str) -> str:
    return f'<div class="disclaimer">{html_body}</div>\n'


def gate_badge(html_body: str) -> str:
    return (
        '<span class="gate-badge">'
        '<span class="dot"></span>'
        f'{html_body}</span>'
    )


def anchor_cards(
    cards: Sequence[tuple[str, str, str]],
) -> str:
    """`cards` is a list of (tag, title, body_html)."""
    parts = ['<div class="anchor-grid">']
    for tag, title, body in cards:
        parts.append(
            '<div class="anchor-card">'
            f'<span class="tag">{_esc(tag)}</span>'
            f'<h3>{_esc(title)}</h3>'
            f'<p>{body}</p>'
            '</div>'
        )
    parts.append('</div>\n')
    return "".join(parts)


# --------------------------------------------------------------------------
# JSON serialiser that handles numpy types lazily.

def _json_default(obj):
    try:
        import numpy as np
    except ImportError:
        raise TypeError(f"cannot serialise {type(obj)}")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"cannot serialise {type(obj)}")
