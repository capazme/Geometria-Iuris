"""
Build a single self-contained HTML dashboard combining all lens experiments.

Parses the 6 existing HTML files (4 main lenses + 2 supplementary) and
assembles them into one file with two-level navigation (lens tabs + inner tabs).

Usage:
    cd experiments/
    python build_combined.py
"""

from __future__ import annotations

import re
from pathlib import Path

from shared.html_style import (
    CSS, HEAD_LINKS, C_BLUE, C_ORANGE, C_GREEN, C_VERMIL,
)
from build_index import LENSES, _load_metrics, _metric_card

ROOT = Path(__file__).parent

# ── Source definitions ──────────────────────────────────────────────────────

MAIN_SOURCES = [
    {"prefix": "L1", "label": "I — Relational Distance",
     "path": ROOT / "lens_1_relational/results/figures/html/lens1_interactive.html",
     "color": C_BLUE},
    {"prefix": "L2", "label": "II — Emergent Taxonomy",
     "path": ROOT / "lens_2_taxonomy/results/figures/html/lens2_interactive.html",
     "color": "#CC79A7"},
    {"prefix": "L3", "label": "III — Layer Stratigraphy",
     "path": ROOT / "lens_3_stratigraphy/results/figures/html/lens3_interactive.html",
     "color": C_GREEN},
    {"prefix": "L4", "label": "IV — Value Axes",
     "path": ROOT / "lens_4_values/results/figures/html/lens4_interactive.html",
     "color": C_VERMIL},
    {"prefix": "L5", "label": "V — Neighborhoods",
     "path": ROOT / "lens_5_neighborhoods/results/figures/html/lens5_interactive.html",
     "color": C_ORANGE},
]

SUPP_SOURCES = [
    {"prefix": "NTA", "label": "NTA Exploration", "color": "#64748b",
     "path": ROOT / "lens_3_stratigraphy/results/nta_exploration.html",
     # Functions called from HTML event handlers (must be global)
     "global_fns": ["applyFilters", "drawOverlay", "sortBy", "selectTerm"],
     # Draw hooks: panel_id (pre-namespace) → function name
     "draw_map": {
         "tabOverlay": "drawOverlay", "tabDivergence": "drawDivergence",
         "tabHeatmap": "drawHeatmap", "tabDomains": "drawDomains",
         "tabScatter": "drawScatter",
     },
     # Init lines (called once on first show)
     "init_lines": ["renderStats();", "initDomainFilter();", "renderTable();"],
    },
    {"prefix": "STRAT", "label": "Layer Experiments", "color": "#64748b",
     "path": ROOT / "lens_3_stratigraphy/results/stratigraphy_experiments.html",
     "global_fns": ["drawE1", "drawE4", "drawE5"],
     "draw_map": {
         "t1": "drawE1", "t2": "drawE2", "t3": "drawE3",
         "t4": "drawE4", "t5": "drawE5", "t6": "drawE6",
     },
     "init_lines": ["drawE1();"],
    },
]


# ── ID extraction & namespacing ─────────────────────────────────────────────

def _collect_ids(html: str) -> list[str]:
    """Extract all id='...' / id=\"...\" values, sorted longest-first."""
    ids = re.findall(r'id=["\']([^"\']+)["\']', html)
    return sorted(set(ids), key=len, reverse=True)


def _namespace(content: str, ids: list[str], prefix: str,
               script_mode: bool = False) -> str:
    """Replace every occurrence of each ID with PREFIX_ID.

    script_mode=True also replaces bare string literals 'ID' and "ID".
    """
    for oid in ids:
        nid = f"{prefix}_{oid}"
        # HTML attributes
        content = content.replace(f'id="{oid}"', f'id="{nid}"')
        content = content.replace(f"id='{oid}'", f"id='{nid}'")
        # Tab switching
        content = content.replace(f"showTab('{oid}'", f"showTab('{nid}'")
        content = content.replace(f'showTab("{oid}"', f'showTab("{nid}"')
        # getElementById
        content = content.replace(f"getElementById('{oid}')", f"getElementById('{nid}')")
        content = content.replace(f'getElementById("{oid}")', f'getElementById("{nid}")')
        # Plotly
        content = content.replace(f"Plotly.newPlot('{oid}'", f"Plotly.newPlot('{nid}'")
        content = content.replace(f'Plotly.newPlot("{oid}"', f'Plotly.newPlot("{nid}"')
        # String concatenation patterns like 'e2_'+pi or 'traj_'+model
        content = content.replace(f"'{oid}_'", f"'{nid}_'")
        content = content.replace(f'"{oid}_"', f'"{nid}_"')
        # Object key in drawMap: {t1:drawE1, ...} → {STRAT_t1:drawE1, ...}
        content = content.replace(f"{{{oid}:", f"{{{nid}:")
        content = content.replace(f",{oid}:", f",{nid}:")
        if script_mode:
            # Generic bare string literals (for fillSelect('e1Model') etc.)
            content = content.replace(f"'{oid}'", f"'{nid}'")
            content = content.replace(f'"{oid}"', f'"{nid}"')
    return content


def _namespace_figs_keys(figs_block: str, prefix: str) -> str:
    """Prefix all figs object keys (like plt_forest → L1_plt_forest)."""
    return re.sub(
        r'^(\s+)(plt_\w+)(\s*:)',
        rf'\g<1>{prefix}_\g<2>\g<3>',
        figs_block,
        flags=re.MULTILINE,
    )


# ── Extraction: main lens files ────────────────────────────────────────────

def _extract_main(html: str, prefix: str) -> tuple[str, str, str]:
    """
    Extract from a main lens HTML file.

    Returns (body_html, figs_entries, extra_scripts).
    - body_html: everything between <body> and the <script> containing const figs
    - figs_entries: the inner content of const figs = { ... }; (keys prefixed)
    - extra_scripts: any inline <script> blocks in the body (e.g. ntaSwitchModel)
    """
    # Split at <body>
    _, after_body = html.split("<body>", 1)
    before_close = after_body.split("</body>")[0]

    # Find the <script> block containing "const figs"
    # Split into body content + script blocks
    parts = re.split(r"<script>", before_close)
    body_html = parts[0]  # everything before the first <script>

    extra_scripts = ""
    figs_block = ""

    for part in parts[1:]:
        script_content = part.split("</script>")[0]
        if "const figs" in script_content:
            # Extract figs entries
            match = re.search(
                r"const figs = \{(.*?)\};\s*\n",
                script_content,
                re.DOTALL,
            )
            if match:
                figs_block = match.group(1)
        else:
            # Extra inline script (e.g. ntaSwitchModel in L3)
            extra_scripts += f"\n{script_content}\n"

    # Collect IDs from body
    ids = _collect_ids(body_html)
    body_html = _namespace(body_html, ids, prefix)
    if extra_scripts.strip():
        extra_scripts = _namespace(extra_scripts, ids, prefix)

    # Namespace figs keys and the div references in the body
    figs_block = _namespace_figs_keys(figs_block, prefix)

    # Also namespace the plot div IDs in the body (plt_ → PREFIX_plt_)
    figs_keys = re.findall(r"(plt_\w+)\s*:", figs_block)
    for fk in figs_keys:
        # The key is already prefixed in figs_block; prefix the body HTML div
        orig_key = fk.replace(f"{prefix}_", "")
        body_html = body_html.replace(
            f'id="{orig_key}"', f'id="{prefix}_{orig_key}"'
        )

    return body_html.strip(), figs_block, extra_scripts.strip()


# ── Extraction: supplementary files ─────────────────────────────────────────

def _extract_supp(src: dict) -> str:
    """
    Build an iframe srcdoc for a supplementary HTML file.

    Returns the full escaped HTML content for embedding in srcdoc.
    This avoids all JS scope/namespace issues — the supplementary file
    runs in its own browsing context with zero conflicts.
    """
    html = src["path"].read_text(encoding="utf-8")
    # Escape for srcdoc attribute: & → &amp;, " → &quot;
    escaped = html.replace("&", "&amp;").replace('"', "&quot;")
    return escaped


# ── CSS: supplementary-specific additions ───────────────────────────────────

SUPP_CSS = """
/* Supplementary CSS vars (NTA, Stratigraphy) */
:root {
  --bg: #fafafa; --fg: #1a1a2e; --card: #fff; --border: #e2e8f0;
  --accent: #0072B2; --accent-light: #e3f2fd; --accent-dark: #005a8c;
  --muted: #64748b; --success: #059669; --warn: #d97706;
}

/* NTA-specific */
.rank-table { width:100%; border-collapse:collapse; font-size:0.8rem; background:var(--card);
  border-radius:8px; overflow:hidden; }
.rank-table th { background:#f1f5f9; padding:9px 10px; text-align:left;
  border-bottom:2px solid var(--border); cursor:pointer; user-select:none;
  position:sticky; top:0; z-index:10; font-weight:600; font-size:0.78rem;
  color:var(--muted); text-transform:uppercase; letter-spacing:0.3px; }
.rank-table th:hover { background:#e2e8f0; }
.rank-table th .sort-arrow { font-size:0.65em; margin-left:2px; opacity:0.6; }
.rank-table td { padding:7px 10px; border-bottom:1px solid #f1f5f9; }
.rank-table tbody tr:hover { background:var(--accent-light); cursor:pointer; }
.rank-table tbody tr.selected { background:#bbdefb !important; }
.domain-badge { display:inline-block; padding:2px 8px; border-radius:10px;
  font-size:0.72em; color:#fff; white-space:nowrap; font-weight:500; }
.score-bar { display:inline-block; height:12px; border-radius:6px;
  background:linear-gradient(90deg,var(--accent),var(--accent-dark));
  opacity:0.65; vertical-align:middle; margin-right:5px; }
.zh-text { color:#94a3b8; font-size:0.82em; margin-left:5px; }
.mini-traj { display:inline-flex; align-items:flex-end; gap:1px;
  height:18px; vertical-align:middle; margin-left:6px; }
.mini-traj-bar { width:4px; border-radius:1px; }
.detail-placeholder { text-align:center; color:var(--muted); padding:50px;
  font-style:italic; font-size:0.9rem; }
.traj-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
@media (max-width:1100px) { .traj-grid { grid-template-columns:1fr 1fr; } }
@media (max-width:700px) { .traj-grid { grid-template-columns:1fr; } }
.traj-chart { background:var(--card); border:1px solid var(--border);
  border-radius:8px; overflow:hidden; }
.traj-chart .model-label { padding:6px 12px; font-size:0.78rem; font-weight:600;
  background:#f8fafc; border-bottom:1px solid var(--border);
  display:flex; justify-content:space-between; align-items:center; }
.traj-chart .model-label .model-stats { font-weight:400; color:var(--muted); font-size:0.75rem; }
.filter-row { display:flex; gap:12px; align-items:center; margin-bottom:14px; flex-wrap:wrap; }
.filter-row select, .filter-row input { padding:6px 12px; border:1px solid var(--border);
  border-radius:6px; font-size:0.82rem; background:var(--card); }
.filter-row input[type="text"] { width:220px; }
.filter-row label { font-size:0.82rem; color:var(--muted); font-weight:500; }
.stat-row { display:flex; gap:10px; margin-bottom:18px; flex-wrap:wrap; }
.stat-box { background:var(--card); border:1px solid var(--border);
  border-radius:8px; padding:12px 18px; flex:1; min-width:110px; text-align:center; }
.stat-box .stat-val { font-size:1.6rem; font-weight:700; color:var(--accent); }
.stat-box .stat-label { font-size:0.72rem; color:var(--muted); text-transform:uppercase; }
.split-cols { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:900px) { .split-cols { grid-template-columns:1fr; } }
.card-subtitle { font-size:0.82rem; color:var(--muted); margin-bottom:14px; }
.note .formula { background:#fff; border:1px solid var(--border);
  padding:8px 14px; border-radius:4px; margin:8px 0; }

/* Stratigraphy-specific */
.split { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.split3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }
@media (max-width:900px) { .split, .split3 { grid-template-columns:1fr; } }
.card-sub { font-size:0.82rem; color:var(--muted); margin-bottom:14px; }
"""

# ── Home section ────────────────────────────────────────────────────────────

def _build_home() -> str:
    """Build the Home tab content with hero + navigable lens cards."""
    cards_html = ""
    for lens in LENSES:
        metrics = _load_metrics(lens)
        metrics_html = "".join(_metric_card(k, v) for k, v in metrics.items())
        color = lens["color"]
        # Map lens IDs to section IDs for onclick
        section_map = {
            "lens1": "sec_L1", "lens2": "sec_L2", "lens3": "sec_L3",
            "lens4": "sec_L4", "lens5": "sec_L5",
        }
        sec_id = section_map.get(lens["id"], "home")
        onclick = f"document.querySelector(&quot;.lens-tab[data-sec='{sec_id}']&quot;).click()"

        extras_html = ""

        cards_html += f"""
    <div class="idx-card" style="border-left-color:{color}; cursor:pointer;"
         onclick="{onclick}">
      <div class="idx-header">
        <span class="idx-num" style="color:{color};">Lens {lens['number']}</span>
        <span class="idx-section">{lens['section']}</span>
      </div>
      <h2 class="idx-title">{lens['title']}</h2>
      <p class="idx-desc">{lens['description']}</p>
      <div class="idx-metrics">{metrics_html}</div>
      <div class="idx-actions">
        <span class="idx-open">Open dashboard &rarr;</span>{extras_html}
      </div>
    </div>"""

    return f"""
<div class="idx-hero">
  <h1>Geometria Iuris</h1>
  <p class="subtitle">Measuring Legal Meaning Across Cultural Normative Structures
  in Embedding Spaces</p>
  <div class="idx-hero-stats">
    <div class="idx-hero-stat"><div class="val">5</div><div class="lbl">Lenses</div></div>
    <div class="idx-hero-stat"><div class="val">6</div><div class="lbl">Models</div></div>
    <div class="idx-hero-stat"><div class="val">9,472</div><div class="lbl">Terms</div></div>
    <div class="idx-hero-stat"><div class="val">397</div><div class="lbl">Core terms</div></div>
  </div>
</div>
<div class="idx-grid">{cards_html}
</div>
<div class="idx-footer">
  Thesis: Metodologia delle Scienze Giuridiche (LUISS) &middot;
  3 WEIRD models (EN) &times; 3 Sinic models (ZH) &middot;
  Source: HK DOJ Bilingual Legal Glossary
</div>"""


# ── Top-level navigation CSS ───────────────────────────────────────────────

TOP_NAV_CSS = """
/* Top-level lens tabs */
.lens-tabs {
  display: flex; gap: 2px; margin-bottom: 0; flex-wrap: wrap;
  border-bottom: 3px solid #e0e0e0; padding: 0; background: #fff;
  position: sticky; top: 0; z-index: 100;
}
.lens-tab {
  padding: 11px 18px; border: none; border-bottom: 4px solid transparent;
  background: none; cursor: pointer; font-size: 0.82rem; color: #888;
  font-weight: 600; transition: all 0.15s; white-space: nowrap;
}
.lens-tab:hover { color: #333; background: #f5f5f5; }
.lens-tab.active { color: #222; border-bottom-color: #222; background: #fafafa; }

/* Lens sections */
.lens-section { display: none; padding-top: 16px; }
.lens-section.active { display: block; }
"""

# ── Index-specific CSS (from build_index.py) ────────────────────────────────

INDEX_CSS = """
.idx-hero {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: #fff; padding: 32px 28px 24px; margin: -20px -28px 24px;
  border-radius: 0 0 12px 12px;
}
.idx-hero h1 { font-size: 1.6rem; margin: 0; color: #fff; }
.idx-hero .subtitle { color: rgba(255,255,255,0.7); margin-bottom: 0; }
.idx-hero-stats { display: flex; gap: 24px; margin-top: 16px; }
.idx-hero-stat { text-align: center; }
.idx-hero-stat .val { font-size: 1.8rem; font-weight: 700; }
.idx-hero-stat .lbl { font-size: 0.75rem; text-transform: uppercase;
                      letter-spacing: 0.05em; opacity: 0.7; }
.idx-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 900px) { .idx-grid { grid-template-columns: 1fr; } }
.idx-card {
  background: #fff; border: 1px solid #e0e0e0; border-left: 5px solid #ccc;
  border-radius: 8px; padding: 20px 22px; transition: box-shadow 0.15s;
}
.idx-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.idx-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.idx-num { font-weight: 700; font-size: 0.85rem; text-transform: uppercase;
          letter-spacing: 0.04em; }
.idx-section { font-size: 0.78rem; color: #999; }
.idx-title { font-size: 1.1rem; margin: 0 0 6px 0; color: #222; }
.idx-desc { font-size: 0.84rem; color: #555; margin: 0 0 12px 0; line-height: 1.5; }
.idx-metrics { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.idx-metric { background: #f8f8f8; border-radius: 6px; padding: 8px 12px;
  display: flex; flex-direction: column; align-items: center; min-width: 80px; }
.idx-mv { font-size: 1.05rem; font-weight: 700; color: #333; }
.idx-ml { font-size: 0.68rem; color: #999; text-transform: uppercase;
         letter-spacing: 0.03em; margin-top: 2px; }
.idx-actions { display: flex; gap: 10px; align-items: center; }
.idx-open { display: inline-block; padding: 7px 16px; background: #222; color: #fff;
  border-radius: 6px; text-decoration: none; font-size: 0.82rem; font-weight: 500; }
.idx-extra { display: inline-block; padding: 7px 14px; background: #f0f0f0; color: #555;
  border-radius: 6px; text-decoration: none; font-size: 0.78rem; cursor: pointer; }
.idx-extra:hover { background: #e0e0e0; color: #222; }
.idx-footer { margin-top: 28px; padding-top: 14px; border-top: 1px solid #e0e0e0;
  font-size: 0.78rem; color: #aaa; text-align: center; }
"""

# ── Unified JS ──────────────────────────────────────────────────────────────

UNIFIED_JS = """\
const rendered = new Set();

function renderPlotsInPanel(panel) {
  panel.querySelectorAll("[id]").forEach(function(el) {
    if (!rendered.has(el.id) && figs[el.id]) {
      Plotly.newPlot(el.id, figs[el.id].data, figs[el.id].layout, {responsive: true});
      rendered.add(el.id);
    }
  });
}

function showLens(lensId, btn) {
  document.querySelectorAll('.lens-section').forEach(function(el) {
    el.classList.remove('active');
  });
  document.querySelectorAll('.lens-tab').forEach(function(el) {
    el.classList.remove('active');
  });
  document.getElementById(lensId).classList.add('active');
  btn.classList.add('active');

  // Render plots in the active inner panel
  var activePanel = document.querySelector('#' + lensId + ' .panel.active');
  if (activePanel) {
    setTimeout(function() { renderPlotsInPanel(activePanel); }, 50);
  }

  // Resize any already-rendered Plotly charts (fixes hidden-div sizing)
  setTimeout(function() {
    document.querySelectorAll('#' + lensId + ' .js-plotly-plot').forEach(function(el) {
      Plotly.Plots.resize(el);
    });
  }, 100);

  // Deferred init for supplementary sections (runs once)
  if (window._initHooks && window._initHooks[lensId]) {
    setTimeout(window._initHooks[lensId], 150);
    delete window._initHooks[lensId];
  }

  // Re-render KaTeX in newly visible section
  if (typeof renderMathInElement === 'function') {
    setTimeout(function() {
      renderMathInElement(document.getElementById(lensId), {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false}
        ]
      });
    }, 200);
  }
}

function showTab(panelId, btn) {
  var section = btn.closest('.lens-section');
  section.querySelectorAll('.panel').forEach(function(el) {
    el.classList.remove('active');
  });
  section.querySelectorAll('.tab-btn').forEach(function(el) {
    el.classList.remove('active');
  });
  document.getElementById(panelId).classList.add('active');
  btn.classList.add('active');

  // Render main-lens Plotly plots
  setTimeout(function() {
    renderPlotsInPanel(document.getElementById(panelId));
  }, 50);

  // Supplementary draw hooks
  if (window._drawHooks && window._drawHooks[panelId]) {
    setTimeout(window._drawHooks[panelId], 60);
  }
}

// Render plots in the initially active panel of the home section
var firstActive = document.querySelector('.lens-section.active .panel.active');
if (firstActive) renderPlotsInPanel(firstActive);
"""


# ── Assembly ────────────────────────────────────────────────────────────────

def build_combined() -> Path:
    all_figs_entries: list[str] = []
    sections_html: list[str] = []
    extra_scripts: list[str] = []
    supp_iframes: list[str] = []

    # ── Main lens files ──
    for src in MAIN_SOURCES:
        print(f"  Parsing {src['prefix']}: {src['path'].name}")
        html = src["path"].read_text(encoding="utf-8")
        body, figs_entries, extras = _extract_main(html, src["prefix"])
        all_figs_entries.append(figs_entries)
        sec_id = f"sec_{src['prefix']}"
        sections_html.append(
            f'<div id="{sec_id}" class="lens-section">\n{body}\n</div>'
        )
        if extras:
            extra_scripts.append(extras)

    # ── Supplementary files: inject as extra inner tabs inside Lens III ──
    supp_panels = []
    supp_tab_buttons = []
    for src in SUPP_SOURCES:
        print(f"  Parsing {src['prefix']}: {src['path'].name}")
        escaped = _extract_supp(src)
        panel_id = f"L3_{src['prefix']}"
        supp_panels.append(
            f'<div id="{panel_id}" class="panel">\n'
            f'  <iframe srcdoc="{escaped}" '
            f'style="width:100%;height:calc(100vh - 120px);border:none;" '
            f'loading="lazy"></iframe>\n'
            f'</div>'
        )
        supp_tab_buttons.append(
            f'<button class="tab-btn" '
            f'onclick="showTab(\'{panel_id}\', this)">{src["label"]}</button>'
        )

    # Inject supplementary panels and tab buttons into the L3 section
    if supp_panels:
        l3_idx = next(
            i for i, s in enumerate(sections_html)
            if 'id="sec_L3"' in s
        )
        l3_html = sections_html[l3_idx]
        # Append extra tab buttons to the tabs div
        tabs_close = "</div>"  # closing tag of .tabs div
        extra_btns = "\n".join(supp_tab_buttons)
        l3_html = l3_html.replace(
            tabs_close, f"{extra_btns}\n{tabs_close}", 1,
        )
        # Append iframe panels before the closing </div> of the lens-section
        extra_panels = "\n".join(supp_panels)
        l3_html = l3_html.rstrip()
        # Remove trailing </div> and re-add after panels
        if l3_html.endswith("</div>"):
            l3_html = l3_html[:-6] + f"\n{extra_panels}\n</div>"
        sections_html[l3_idx] = l3_html

    # ── Build top-level tabs (main lenses only) ──
    tab_buttons = [
        '<button class="lens-tab active" data-sec="home" '
        'onclick="showLens(\'home\', this)">Home</button>'
    ]
    for src in MAIN_SOURCES:
        sec_id = f"sec_{src['prefix']}"
        tab_buttons.append(
            f'<button class="lens-tab" data-sec="{sec_id}" '
            f'onclick="showLens(\'{sec_id}\', this)">{src["label"]}</button>'
        )
    tabs_bar = '<div class="lens-tabs">\n  ' + "\n  ".join(tab_buttons) + "\n</div>"

    # ── Build figs dict ──
    merged_figs = ",\n".join(e.strip().rstrip(",") for e in all_figs_entries if e.strip())
    figs_script = f"const figs = {{\n{merged_figs}\n}};"

    # ── Home section ──
    home_html = f'<div id="home" class="lens-section active">\n{_build_home()}\n</div>'

    # ── Extra scripts from main lenses ──
    extra_block = ""
    if extra_scripts:
        extra_block = "<script>\n" + "\n".join(extra_scripts) + "\n</script>"

    # ── Supplementary IIFEs ──
    supp_block = ""  # supplementary files embedded via iframe srcdoc

    # ── Assemble ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Geometria Iuris — Complete Experiment Dashboard</title>
{HEAD_LINKS}
<style>
{CSS}
{TOP_NAV_CSS}
{INDEX_CSS}
{SUPP_CSS}
</style>
</head>
<body>

{tabs_bar}

{home_html}

{"".join(sections_html)}

{extra_block}

{supp_block}

<script>
{figs_script}

{UNIFIED_JS}
</script>

</body>
</html>"""

    out = ROOT / "geometria_iuris_dashboard.html"
    out.write_text(html, encoding="utf-8")

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\n  Output: {out}")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  Sections: {len(MAIN_SOURCES)} lenses + Home"
          f" ({len(SUPP_SOURCES)} supplementary inside Lens III)")

    # Sanity: check for duplicate IDs
    all_ids = re.findall(r'id="([^"]+)"', html)
    dupes = [x for x in set(all_ids) if all_ids.count(x) > 1]
    if dupes:
        print(f"\n  WARNING: {len(dupes)} duplicate IDs found: {dupes[:10]}")
    else:
        print(f"  IDs:    {len(all_ids)} unique, 0 duplicates")

    return out


if __name__ == "__main__":
    print("Building combined dashboard...\n")
    build_combined()
