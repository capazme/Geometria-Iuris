"""
Shared HTML style system for all Lens interactive dashboards.

Provides consistent CSS, JS (lazy rendering), and HTML head/structure
so that all Lens HTML files share the same visual identity.
"""

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette
# ---------------------------------------------------------------------------

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_SKY    = "#56B4E9"
C_VERMIL = "#D55E00"
C_PURPLE = "#CC79A7"
C_BLACK  = "#000000"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.3.1.min.js"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = f"""
:root {{
  --blue: {C_BLUE}; --orange: {C_ORANGE}; --green: {C_GREEN};
  --sky: {C_SKY}; --vermil: {C_VERMIL}; --purple: {C_PURPLE};
}}
* {{ box-sizing: border-box; }}
body {{ font-family: "Inter", "Segoe UI", system-ui, sans-serif; margin: 0;
       padding: 20px 28px; background: #fafafa; color: #1a1a1a; line-height: 1.55; }}
h1 {{ font-size: 1.35rem; margin: 0 0 4px 0; }}
.subtitle {{ color: #666; font-size: 0.85rem; margin-bottom: 16px; }}

/* Tabs */
.tabs {{ display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap;
         border-bottom: 2px solid #e0e0e0; padding-bottom: 0; }}
.tab-btn {{
  padding: 9px 18px; border: none; border-bottom: 3px solid transparent;
  background: none; cursor: pointer; font-size: 0.85rem; color: #666;
  font-weight: 500; transition: all 0.15s;
}}
.tab-btn:hover {{ color: #222; background: #f0f0f0; }}
.tab-btn.active {{ color: var(--blue); border-bottom-color: var(--blue); background: none; }}
.panel {{ display: none; }}
.panel.active {{ display: block; }}

/* Cards & boxes */
.card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
         padding: 18px 22px; margin-bottom: 16px; }}
.card h2 {{ font-size: 1.05rem; margin: 0 0 6px 0; color: #333; }}
.card h3 {{ font-size: 0.95rem; margin: 14px 0 6px 0; color: #444; }}
.card p {{ margin: 4px 0; font-size: 0.88rem; color: #444; }}

.question {{ border-left: 4px solid var(--blue); padding: 12px 16px;
             background: #f0f7ff; margin-bottom: 14px; border-radius: 0 6px 6px 0; }}
.question b {{ color: var(--blue); }}

.finding {{ border-left: 4px solid var(--green); padding: 12px 16px;
            background: #f0faf4; margin-bottom: 14px; border-radius: 0 6px 6px 0; }}
.finding b {{ color: var(--green); }}

.warning {{ border-left: 4px solid var(--vermil); padding: 12px 16px;
            background: #fff5f0; margin-bottom: 14px; border-radius: 0 6px 6px 0; }}
.warning b {{ color: var(--vermil); }}

.method {{ border-left: 4px solid #aaa; padding: 10px 16px; background: #f8f8f8;
           margin-bottom: 14px; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #555; }}

/* Key metric cards */
.metrics {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 16px; }}
.metric {{ flex: 1; min-width: 140px; background: #fff; border: 1px solid #e0e0e0;
           border-radius: 8px; padding: 14px 16px; text-align: center; }}
.metric .value {{ font-size: 1.6rem; font-weight: 700; margin: 4px 0; }}
.metric .label {{ font-size: 0.78rem; color: #888; text-transform: uppercase;
                  letter-spacing: 0.04em; }}
.metric.blue .value {{ color: var(--blue); }}
.metric.vermil .value {{ color: var(--vermil); }}
.metric.green .value {{ color: var(--green); }}
.metric.orange .value {{ color: var(--orange); }}

/* Tables */
table.data {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 10px 0; }}
table.data th {{ background: #f5f5f5; padding: 8px 10px; text-align: left;
                 border-bottom: 2px solid #ddd; font-weight: 600; color: #555; }}
table.data td {{ padding: 7px 10px; border-bottom: 1px solid #eee; }}
table.data tr:hover {{ background: #fafafa; }}
td.anomaly, td.anomaly b {{ color: var(--vermil); }}
tr.rsa-weird td:first-child {{ color: var(--blue); font-weight: 600; }}
tr.rsa-sinic td:first-child {{ color: var(--vermil); font-weight: 600; }}
tr.rsa-cross td:first-child {{ color: var(--green); font-weight: 600; }}

/* Layout helpers */
.plot-label {{ font-size: 0.82rem; color: #888; margin: 12px 0 4px 0; font-weight: 500; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.note-sm {{ font-size: 0.8rem; color: #888; margin-top: 4px; }}

/* NTA styles (Lens III) */
.nta-select {{ padding: 6px 12px; font-size: 0.85rem; border: 1px solid #ccc;
               border-radius: 4px; margin-bottom: 12px; }}
.nta-table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; margin: 8px 0; }}
.nta-table th {{ background: #f5f5f5; padding: 6px 8px; text-align: left;
                 border-bottom: 2px solid #ddd; font-weight: 600; }}
.nta-table td {{ padding: 5px 8px; border-bottom: 1px solid #eee; }}
.nta-entered {{ color: var(--green); font-weight: 600; }}
.nta-exited {{ color: var(--vermil); font-style: italic; }}
.nta-control {{ color: #999; font-style: italic; }}
"""


# ---------------------------------------------------------------------------
# HTML head (CDN links)
# ---------------------------------------------------------------------------

HEAD_LINKS = f"""\
<script src="{PLOTLY_CDN}"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>"""


# ---------------------------------------------------------------------------
# JS — lazy tab rendering
# ---------------------------------------------------------------------------

LAZY_JS = """\
const rendered = new Set();
function renderPlotsInPanel(panel) {
  panel.querySelectorAll("[id^='plt_']").forEach(function(el) {
    if (!rendered.has(el.id) && figs[el.id]) {
      Plotly.newPlot(el.id, figs[el.id].data, figs[el.id].layout, {responsive: true});
      rendered.add(el.id);
    }
  });
}

function showTab(id, btn) {
  document.querySelectorAll(".panel").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  btn.classList.add("active");
  setTimeout(function() { renderPlotsInPanel(document.getElementById(id)); }, 50);
}

renderPlotsInPanel(document.querySelector(".panel.active"));"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def page_head(title: str) -> str:
    """Return complete <head> block."""
    return f"""<head>
<meta charset="utf-8">
<title>{title}</title>
{HEAD_LINKS}
<style>{CSS}</style>
</head>"""


def tabs_bar(tabs: list[tuple[str, str]], first_active: bool = True) -> str:
    """Build tab button bar. tabs = [(panel_id, label), ...]."""
    parts = []
    for i, (pid, label) in enumerate(tabs):
        active = " active" if (i == 0 and first_active) else ""
        parts.append(
            f'  <button class="tab-btn{active}" '
            f"onclick=\"showTab('{pid}', this)\">{label}</button>"
        )
    return '<div class="tabs">\n' + "\n".join(parts) + "\n</div>"


def plots_script(plots: dict[str, str]) -> str:
    """Build <script> block with figs dict + lazy rendering JS."""
    entries = ",\n".join(f"  {k}: {v}" for k, v in plots.items())
    return f"<script>\nconst figs = {{\n{entries}\n}};\n\n{LAZY_JS}\n</script>"


def format_p(p: float) -> str:
    """Format p-value for HTML display."""
    if p == 0.0:
        return "&lt; 10<sup>-300</sup>"
    if p < 0.0001:
        return f"{p:.1e}"
    if p >= 0.99:
        return f"{p:.1f}"
    return f"{p:.4f}"
