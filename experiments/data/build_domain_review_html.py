"""Generate a reviewable HTML of core terms for a given legal domain.

The HTML lists every core term in the selected domain with its bilingual
canonical form, ZH variants, HK DOJ source divisions, and up to two
attested contexts per language drawn from the e-Legislation corpus. It
is a read-only document: the lawyer uses it to decide which terms to
keep, demote, or flag for further review. The decision is recorded
manually (outside this script) in the pivot trace.

Usage:
    python experiments/data/build_domain_review_html.py --domain administrative
    python experiments/data/build_domain_review_html.py --domain labor_social
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "data" / "processed"
OUT = REPO / "experiments" / "data" / "review"
OUT.mkdir(parents=True, exist_ok=True)


def _load_contexts() -> dict[tuple[str, str], dict]:
    """Index term_contexts.jsonl by (term_en, term_zh)."""
    path = DATA / "elegislation" / "term_contexts.jsonl"
    idx: dict[tuple[str, str], dict] = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            rec.setdefault("en_contexts", [])
            rec.setdefault("zh_contexts", [])
            idx[(rec["term_en"], rec["term_zh"])] = rec
    return idx


def _fmt_context(c: dict, max_chars: int = 260) -> str:
    cap = c.get("cap", "?")
    section = c.get("section_id", "?")
    text = c.get("context", "")
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return (
        f'<div class="ctx"><span class="cap">Cap. {html.escape(str(cap))} '
        f'§{html.escape(str(section))}</span>'
        f'<p>{html.escape(text)}</p></div>'
    )


def build_html(domain: str) -> Path:
    legal = json.loads((DATA / "legal_terms.json").read_text())
    terms = [t for t in legal["terms"] if t.get("tier") == "core" and t.get("domain") == domain]
    terms.sort(key=lambda t: t["en"].lower())

    contexts = _load_contexts()

    rows_html = []
    for t in terms:
        en = t.get("en", "")
        zh = t.get("zh_canonical", "")
        variants = t.get("zh_variants") or []
        divisions = t.get("doj_divisions") or []
        source = t.get("source", "")

        ctx = contexts.get((en, zh), {})
        en_ctx = (ctx.get("en_contexts") or [])[:2]
        zh_ctx = (ctx.get("zh_contexts") or [])[:2]

        en_blocks = "".join(_fmt_context(c) for c in en_ctx) or '<div class="ctx empty">— no attested EN context —</div>'
        zh_blocks = "".join(_fmt_context(c) for c in zh_ctx) or '<div class="ctx empty">— no attested ZH context —</div>'

        variants_str = ", ".join(html.escape(v) for v in variants) if variants else "—"
        divisions_str = ", ".join(html.escape(d) for d in divisions) if divisions else "—"

        search_blob = " ".join([en, zh, *variants, *divisions, source]).lower()
        search_blob = re.sub(r"\s+", " ", search_blob)

        rows_html.append(f"""
<details class="term" data-search="{html.escape(search_blob, quote=True)}">
  <summary>
    <span class="en">{html.escape(en)}</span>
    <span class="zh">{html.escape(zh)}</span>
    <span class="meta">{html.escape(source)}</span>
  </summary>
  <div class="body">
    <dl>
      <dt>ZH variants</dt><dd>{variants_str}</dd>
      <dt>DOJ divisions</dt><dd>{divisions_str}</dd>
    </dl>
    <div class="contexts">
      <h4>Attested EN contexts</h4>{en_blocks}
      <h4>Attested ZH contexts</h4>{zh_blocks}
    </div>
  </div>
</details>
""")

    n = len(terms)
    title = f"Core terms review — domain: {domain} ({n})"

    style = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #f6f5f1; color: #1a1a1a; font-family: 'Iowan Old Style', 'Charter', Georgia, serif; }
header { position: sticky; top: 0; background: #1a1a1a; color: #f6f5f1; padding: 1.2rem 2rem; z-index: 10; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
header h1 { margin: 0; font-size: 1.2rem; font-weight: 500; letter-spacing: 0.02em; }
header p { margin: 0.3rem 0 0; font-size: 0.85rem; opacity: 0.8; font-style: italic; }
#search { margin: 0.6rem 0 0; padding: 0.45rem 0.8rem; width: 100%; max-width: 500px; border: 1px solid #555; border-radius: 3px; background: #222; color: #f6f5f1; font-size: 0.9rem; font-family: inherit; }
#search::placeholder { color: #999; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 4rem; }
.term { background: #fff; border-left: 3px solid #b08d57; margin-bottom: 0.5rem; border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.term.hidden { display: none; }
.term summary { padding: 0.7rem 1rem; cursor: pointer; list-style: none; display: flex; gap: 1.2rem; align-items: baseline; flex-wrap: wrap; }
.term summary::-webkit-details-marker { display: none; }
.term summary::before { content: '▸'; color: #b08d57; transition: transform 0.15s; display: inline-block; margin-right: 0.2rem; }
.term[open] summary::before { transform: rotate(90deg); }
.term .en { font-weight: 600; font-size: 1.02rem; color: #1a1a1a; }
.term .zh { color: #444; font-size: 1rem; font-family: 'Source Han Serif', 'Noto Serif CJK SC', 'STSong', serif; }
.term .meta { color: #999; font-size: 0.8rem; margin-left: auto; font-style: italic; }
.term .body { padding: 0 1.2rem 1rem; border-top: 1px solid #eee; }
.term dl { display: grid; grid-template-columns: 130px 1fr; gap: 0.35rem 1rem; margin: 0.8rem 0 1rem; font-size: 0.88rem; }
.term dt { color: #777; font-variant: small-caps; letter-spacing: 0.03em; }
.term dd { margin: 0; color: #333; }
.contexts h4 { margin: 1rem 0 0.4rem; font-size: 0.82rem; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.ctx { background: #faf9f5; padding: 0.6rem 0.9rem; margin: 0.3rem 0; border-left: 2px solid #ddd; font-size: 0.87rem; }
.ctx.empty { color: #aaa; font-style: italic; border-left-color: transparent; }
.ctx .cap { display: inline-block; color: #b08d57; font-weight: 600; font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.25rem; }
.ctx p { margin: 0; color: #333; line-height: 1.5; }
footer { text-align: center; color: #999; font-size: 0.8rem; padding: 1rem; }
"""

    script = """
const search = document.getElementById('search');
const terms = document.querySelectorAll('.term');
search.addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  let visible = 0;
  terms.forEach(t => {
    const match = !q || t.dataset.search.includes(q);
    t.classList.toggle('hidden', !match);
    if (match) visible++;
  });
  document.getElementById('count').textContent = visible;
});
document.getElementById('expand-all').addEventListener('click', (e) => { e.preventDefault(); terms.forEach(t => t.open = true); });
document.getElementById('collapse-all').addEventListener('click', (e) => { e.preventDefault(); terms.forEach(t => t.open = false); });
"""

    body = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{style}</style>
</head><body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>Review for pivot D5.1 — confirm which entries belong in the core set.</p>
  <input id="search" type="search" placeholder="Filter by EN, ZH, DOJ division, variant…">
  <div style="margin-top:0.5rem; font-size: 0.8rem; opacity: 0.7;">
    <span id="count">{n}</span> terms shown &nbsp;·&nbsp;
    <a href="#" id="expand-all" style="color:#b08d57;">expand all</a> &nbsp;/&nbsp;
    <a href="#" id="collapse-all" style="color:#b08d57;">collapse all</a>
  </div>
</header>
<main>
{''.join(rows_html)}
</main>
<footer>Generated from legal_terms.json + e-Legislation attested contexts. Geometria Iuris · 2026-04-16</footer>
<script>{script}</script>
</body></html>
"""

    out = OUT / f"review_{domain}.html"
    out.write_text(body)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="administrative",
                        help="Domain to review (administrative, civil, ...)")
    args = parser.parse_args()
    path = build_html(args.domain)
    print(f"Wrote {path}")
