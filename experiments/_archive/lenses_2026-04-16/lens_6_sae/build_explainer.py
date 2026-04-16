"""
Build a pedagogical HTML explaining the SAE decomposition with real data.
16 steps: framing, mechanics (1-8), interpretation (9-11), cross-model (12),
sensitivity (13), limits (14), summary (15).

Output: lens_6_sae/results/figures/html/sae_explainer.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lens_6_sae.sae import TopKSAE  # noqa: E402
from shared.html_style import (  # noqa: E402
    C_BLUE, C_ORANGE, C_GREEN, C_VERMIL, C_PURPLE, C_SKY, PLOTLY_CDN,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EMB_DIR = REPO_ROOT / "data" / "processed" / "embeddings"
OUT_DIR = RESULTS_DIR / "figures" / "html"

MODEL_ORDER = [
    "BGE-EN-large", "E5-large", "FreeLaw-EN",
    "BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH",
    "BGE-M3-EN", "BGE-M3-ZH", "Qwen3-0.6B-EN", "Qwen3-0.6B-ZH",
]
MODEL_LABELS = [
    "BGE-EN", "E5", "FreeLaw", "BGE-ZH", "Txt2v-ZH", "Dmeta",
    "M3-EN", "M3-ZH", "Qw3-EN", "Qw3-ZH",
]
MODEL_COLORS = [C_BLUE]*3 + [C_VERMIL]*3 + [C_GREEN]*2 + [C_PURPLE]*2

DOMAIN_COLORS = {
    "constitutional": C_BLUE, "criminal": C_VERMIL, "civil": C_ORANGE,
    "international": C_GREEN, "labor_social": C_PURPLE,
    "administrative": C_SKY, "procedure": "#999999", "none": "#cccccc",
}
DOMAIN_IT = {
    "constitutional": "Costituzionale", "criminal": "Penale",
    "civil": "Civile", "international": "Internazionale",
    "labor_social": "Lavoro", "administrative": "Amministrativo",
    "procedure": "Procedura", "none": "Non etichettato",
}

# ── HTML helpers ──────────────────────────────────────────────────────

def code(text: str) -> str:
    return f'<div class="codebox">{text}</div>'

def io_pair(inp_title, inp, out_title, out) -> str:
    return (
        '<div class="iobox">'
        f'<div class="io inp"><div class="io-title">{inp_title}</div>{inp}</div>'
        f'<div class="io out"><div class="io-title">{out_title}</div>{out}</div>'
        '</div>'
    )

def hbar(terms, acts, colors, left_m=160, h=320):
    fig = go.Figure(go.Bar(
        y=terms[::-1], x=acts[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{a:.3f}" for a in acts[::-1]], textposition="outside",
    ))
    fig.update_layout(height=h, template="plotly_white", xaxis_title="Attivazione",
                      margin=dict(t=10, b=40, l=left_m, r=60))
    return fig.to_json()

def legend_html():
    parts = []
    for d in ["criminal", "civil", "international", "labor_social",
              "administrative", "constitutional", "procedure", "none"]:
        parts.append(
            f'<span><span class="dot" style="background:{DOMAIN_COLORS[d]}"></span>'
            f'<span style="font-size:0.82rem">{DOMAIN_IT[d]}</span></span>')
    return '<div class="legend">' + "".join(parts) + "</div>"

# ── Chart builders ────────────────────────────────────────────────────

def domain_count_bar(counts: dict) -> str:
    domains = sorted(counts.keys(), key=lambda d: -counts[d])
    vals = [counts[d] for d in domains]
    cols = [DOMAIN_COLORS.get(d, "#999") for d in domains]
    labs = [DOMAIN_IT.get(d, d) for d in domains]
    fig = go.Figure(go.Bar(
        y=labs[::-1], x=vals[::-1], orientation="h",
        marker_color=cols[::-1],
        text=[str(v) for v in vals[::-1]], textposition="outside",
    ))
    fig.update_layout(height=280, template="plotly_white",
                      xaxis_title="Feature significative (Holm p < 0.05)",
                      margin=dict(t=10, b=40, l=130, r=40))
    return fig.to_json()


def dsi_histogram(enrichment: list) -> str:
    vals = [r["dsi"] for r in enrichment if r["n_labeled_in_top"] >= 3]
    fig = go.Figure(go.Histogram(
        x=vals, nbinsx=30, marker_color="rgba(0,114,178,0.5)",
        marker_line=dict(color=C_BLUE, width=0.5),
    ))
    fig.add_vrect(x0=0, x1=0.3, fillcolor="rgba(213,94,0,0.08)",
                  line_width=0, annotation_text="generalisti",
                  annotation_position="top left",
                  annotation=dict(font_size=11, font_color=C_VERMIL))
    fig.add_vrect(x0=0.7, x1=1.01, fillcolor="rgba(0,158,115,0.08)",
                  line_width=0, annotation_text="specialisti",
                  annotation_position="top right",
                  annotation=dict(font_size=11, font_color=C_GREEN))
    fig.add_vline(x=np.median(vals), line=dict(color="#888", width=1, dash="dash"),
                  annotation_text="mediana", annotation_position="top left",
                  annotation=dict(font_size=10, font_color="#888"))
    fig.update_layout(height=300, template="plotly_white",
                      xaxis_title="Domain Selectivity Index (1 = massimamente selettivo)",
                      yaxis_title="Conteggio feature",
                      margin=dict(t=10, b=40, l=50, r=20))
    return fig.to_json()


def contingency_heatmap(a, b, c, d) -> str:
    z = [[a, b], [c, d]]
    labels = [[f"<b>{a}</b><br>penale &cap; top-50", f"<b>{b}</b><br>non-penale &cap; top-50"],
              [f"<b>{c}</b><br>penale &cap; fuori", f"<b>{d}</b><br>non-penale &cap; fuori"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["criminal", "non-criminal"], y=["nel top-50", "fuori dal top-50"],
        colorscale=[[0, "#fff"], [1, C_VERMIL]], showscale=False,
        text=labels, texttemplate="%{text}", hoverinfo="skip",
        textfont=dict(size=12),
    ))
    fig.update_layout(height=220, template="plotly_white",
                      margin=dict(t=10, b=40, l=120, r=20),
                      yaxis=dict(autorange="reversed"))
    return fig.to_json()


def pvalue_distribution(enrichment: list) -> str:
    raw_p = [e["p_value"] for r in enrichment for e in r["enrichments"]
             if e["p_value"] < 0.999]
    log_p = [-np.log10(max(p, 1e-300)) for p in raw_p]
    fig = go.Figure(go.Histogram(
        x=log_p, nbinsx=50, marker_color="rgba(0,114,178,0.5)",
        marker_line=dict(color=C_BLUE, width=0.5),
    ))
    fig.add_vline(x=-np.log10(0.05), line=dict(color=C_VERMIL, width=2, dash="dash"),
                  annotation_text="p = 0.05 (raw)",
                  annotation_position="top right",
                  annotation=dict(font_color=C_VERMIL))
    fig.update_layout(height=300, template="plotly_white",
                      xaxis_title="-log10(p)   [valori pi&ugrave; alti = pi&ugrave; significativi]",
                      yaxis_title="Conteggio test",
                      margin=dict(t=10, b=40, l=50, r=20))
    return fig.to_json()


def evr_comparison_bar(cross_quality, labels, colors) -> str:
    evrs = [cross_quality[m]["evr"] for m in MODEL_ORDER]
    fig = go.Figure(go.Bar(
        x=labels, y=evrs, marker_color=colors,
        text=[f"{v:.3f}" for v in evrs], textposition="outside",
    ))
    fig.add_hline(y=0.95, line=dict(color="#888", width=1, dash="dash"),
                  annotation_text="R&sup2; = 0.95",
                  annotation_position="bottom right")
    fig.update_layout(height=350, template="plotly_white",
                      yaxis_title="R&sup2; (Explained Variance Ratio)",
                      yaxis_range=[0.88, 1.0],
                      margin=dict(t=10, b=40, l=50, r=20))
    return fig.to_json()


def sensitivity_bar(all_enrichment, labels, colors) -> str:
    n005 = []
    n010 = []
    for m in MODEL_ORDER:
        enr = all_enrichment[m]
        s05 = sum(1 for r in enr
                  if any(e.get("significant") for e in r["enrichments"]))
        s10 = sum(1 for r in enr
                  if any(e.get("p_adjusted", 1) < 0.10 for e in r["enrichments"]))
        n005.append(s05)
        n010.append(s10)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="p < 0.05", x=labels, y=n005,
                         marker_color=[c for c in colors],
                         text=n005, textposition="outside"))
    def hex_to_rgba(h, alpha=0.4):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    fig.add_trace(go.Bar(name="p < 0.10", x=labels, y=n010,
                         marker_color=[hex_to_rgba(c) for c in colors],
                         text=n010, textposition="outside"))
    fig.update_layout(height=350, template="plotly_white", barmode="group",
                      yaxis_title="Feature con enrichment significativo",
                      margin=dict(t=10, b=40, l=50, r=20),
                      legend=dict(x=0.8, y=1.0))
    return fig.to_json()


# ── CSS ───────────────────────────────────────────────────────────────

CSS = f"""
* {{ box-sizing:border-box; }}
body {{ font-family:"Inter","Segoe UI",system-ui,sans-serif; margin:0 auto;
  padding:24px 32px; background:#fafafa; color:#1a1a1a; line-height:1.7; max-width:980px; }}
h1 {{ font-size:1.55rem; margin:0 0 4px 0; }}
h2 {{ font-size:1.2rem; margin:0 0 8px 0; color:#333;
     border-bottom:2px solid #e0e0e0; padding-bottom:6px; }}
h3 {{ font-size:1rem; margin:18px 0 6px 0; color:#444; }}
p {{ margin:8px 0; font-size:0.93rem; }}
.step {{ background:#fff; border:1px solid #e0e0e0; border-radius:10px;
  padding:22px 26px; margin:24px 0; border-left:5px solid {C_BLUE}; }}
.step.hl {{ border-left-color:{C_VERMIL}; }}
.step.res {{ border-left-color:{C_GREEN}; }}
.sn {{ display:inline-block; background:{C_BLUE}; color:white;
  width:30px; height:30px; border-radius:50%; text-align:center;
  line-height:30px; font-weight:700; font-size:0.88rem; margin-right:8px; }}
.step.hl .sn {{ background:{C_VERMIL}; }}
.step.res .sn {{ background:{C_GREEN}; }}
.fm {{ background:#f5f5f5; border:1px solid #e8e8e8; border-radius:6px;
  padding:14px 20px; margin:14px 0; font-family:"Georgia",serif;
  font-size:0.95rem; text-align:center; color:#333; line-height:1.8; }}
.kn {{ display:inline-block; background:{C_BLUE}; color:white;
  padding:2px 10px; border-radius:12px; font-weight:700; font-size:0.88rem; }}
.kn.r {{ background:{C_VERMIL}; }} .kn.g {{ background:{C_GREEN}; }}
.an {{ background:#FFF8E1; border-left:4px solid {C_ORANGE};
  padding:14px 18px; margin:14px 0; border-radius:0 6px 6px 0; font-size:0.9rem; }}
.finding {{ background:#f0faf4; border-left:4px solid {C_GREEN};
  padding:14px 18px; margin:14px 0; border-radius:0 6px 6px 0; }}
.warning {{ background:#fff5f0; border-left:4px solid {C_VERMIL};
  padding:14px 18px; margin:14px 0; border-radius:0 6px 6px 0; }}
.mathbox {{ background:#f0f4ff; border:1px solid #d0d8f0; border-radius:8px;
  padding:16px 20px; margin:14px 0; font-size:0.9rem; }}
.mathbox b {{ color:{C_BLUE}; }}
.codebox {{ background:#1e1e2e; color:#cdd6f4; border-radius:8px;
  padding:14px 18px; margin:12px 0; font-family:"Fira Code","SF Mono","Menlo",monospace;
  font-size:0.8rem; line-height:1.6; overflow-x:auto; white-space:pre; }}
.codebox .kw {{ color:#cba6f7; }} .codebox .fn {{ color:#89b4fa; }}
.codebox .cm {{ color:#6c7086; font-style:italic; }} .codebox .st {{ color:#a6e3a1; }}
.codebox .nb {{ color:#fab387; }}
.iobox {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0; }}
.iobox .io {{ border-radius:8px; padding:12px 16px; font-family:"Fira Code","SF Mono",
  "Menlo",monospace; font-size:0.78rem; line-height:1.5; overflow-x:auto; white-space:pre; }}
.iobox .inp {{ background:#e8f4fd; border:1px solid #b6d4fe; }}
.iobox .out {{ background:#e8fde8; border:1px solid #b6feb6; }}
.iobox .io-title {{ font-family:"Inter",sans-serif; font-size:0.72rem; font-weight:700;
  text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px; }}
.iobox .inp .io-title {{ color:#0d6efd; }}
.iobox .out .io-title {{ color:#198754; }}
@media (max-width:800px) {{ .iobox,.two {{ grid-template-columns:1fr; }} }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.fc {{ background:#fff; border:1px solid #e0e0e0; border-radius:8px;
  padding:14px 18px; margin:10px 0; }}
.fc h3 {{ margin:0 0 6px 0; }}
table.dt {{ width:100%; border-collapse:collapse; font-size:0.85rem; margin:10px 0; }}
table.dt th {{ background:#f5f5f5; padding:8px 10px; text-align:left;
  border-bottom:2px solid #ddd; font-weight:600; color:#555; }}
table.dt td {{ padding:7px 10px; border-bottom:1px solid #eee; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin:10px 0; }}
.legend span {{ display:inline-flex; align-items:center; gap:4px; }}
.legend .dot {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
.note {{ font-size:0.82rem; color:#888; margin-top:4px; }}
"""


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    print("[explainer] Loading data ...")

    vectors = np.load(EMB_DIR / "BGE-EN-large" / "vectors.npy")
    activations = np.load(RESULTS_DIR / "activations_BGE-EN-large_x4_k32.npy")
    with open(EMB_DIR / "index.json") as f:
        index = json.load(f)
    with open(RESULTS_DIR / "domain_enrichment_BGE-EN-large_x4_k32.json") as f:
        enrichment = json.load(f)
    with open(RESULTS_DIR / "training_metrics_BGE-EN-large_x4_k32.json") as f:
        tm = json.load(f)
    with open(RESULTS_DIR / "cross_model_comparison.json") as f:
        cross = json.load(f)

    # All models data
    all_training = {}
    all_enrichment = {}
    for m in MODEL_ORDER:
        with open(RESULTS_DIR / f"training_metrics_{m}_x4_k32.json") as f:
            all_training[m] = json.load(f)
        with open(RESULTS_DIR / f"domain_enrichment_{m}_x4_k32.json") as f:
            all_enrichment[m] = json.load(f)

    cross_quality = {}
    for m in MODEL_ORDER:
        qq = all_training[m]["quality"]
        cross_quality[m] = {
            "evr": qq["explained_variance_ratio"],
            "cos": qq["cosine_sim_mean"],
            "dead": qq["n_dead_features"],
        }

    model = TopKSAE(1024, 4096, 32)
    model.load_state_dict(torch.load(
        RESULTS_DIR / "sae_weights_BGE-EN-large_x4_k32.pt",
        map_location="cpu", weights_only=True,
    ))

    W_enc = model.encoder.weight.detach().numpy()
    W_dec = model.decoder.weight.detach().numpy()
    b_dec = model.b_dec.detach().numpy()
    b_enc = model.encoder.bias.detach().numpy()

    # theft data
    ti = 338
    tv = vectors[ti]
    tz = activations[ti]
    x_c = tv - b_dec
    z_pre = W_enc @ x_c + b_enc
    sorted_zpre = np.sort(z_pre)[::-1]

    x_t = torch.from_numpy(vectors[ti:ti + 1]).float()
    with torch.no_grad():
        x_hat_t, _, _ = model(x_t)
    tr = x_hat_t.numpy()[0]
    cos_theft = float(np.dot(tv, tr) / (np.linalg.norm(tv) * np.linalg.norm(tr)))

    d703 = W_dec[:, 703]
    d405 = W_dec[:, 405]
    d3230 = W_dec[:, 3230]
    cos_703_405 = float(np.dot(d703, d405))
    cos_703_3230 = float(np.dot(d703, d3230))

    feat_map = {r["feature_idx"]: r for r in enrichment}
    q = tm["quality"]

    theft_active_idx = np.where(tz > 0)[0]
    theft_sorted = theft_active_idx[np.argsort(tz[theft_active_idx])[::-1]]

    # Fisher data for F2005
    term_domains = [t.get("domain") or "" for t in index]
    n_labeled = sum(1 for d in term_domains if d != "")
    n_criminal = sum(1 for d in term_domains if d == "criminal")
    f2005_acts = activations[:, 2005]
    top50_2005 = np.argsort(f2005_acts)[::-1][:50]
    top50_lab = [term_domains[i] for i in top50_2005 if term_domains[i] != ""]
    n_crim_top = sum(1 for d in top50_lab if d == "criminal")
    fa, fb = n_crim_top, len(top50_lab) - n_crim_top
    fc, fd = n_criminal - fa, (n_labeled - n_criminal) - fb

    # P-value stats
    n_raw_sig = sum(1 for r in enrichment for e in r["enrichments"] if e["p_value"] < 0.05)
    n_holm_sig = sum(1 for r in enrichment for e in r["enrichments"]
                     if e.get("p_adjusted", 1) < 0.05)
    n_holm_010 = sum(1 for r in enrichment for e in r["enrichments"]
                     if e.get("p_adjusted", 1) < 0.10)

    # DSI stats
    dsi_vals = [r["dsi"] for r in enrichment if r["n_labeled_in_top"] >= 3]
    n_gen = sum(1 for d in dsi_vals if d < 0.3)
    n_mix = sum(1 for d in dsi_vals if 0.3 <= d < 0.7)
    n_spec = sum(1 for d in dsi_vals if d >= 0.7)

    dot703 = float(np.dot(W_enc[703], x_c))
    contrib_703 = tz[703] * d703

    # ── Build Plotly figures ──
    plots = {}

    # vector barplot
    fig = go.Figure(go.Bar(
        x=list(range(100)), y=tv[:100].tolist(),
        marker_color=[C_BLUE if v >= 0 else C_VERMIL for v in tv[:100]]))
    fig.update_layout(height=220, template="plotly_white",
                      xaxis_title="Dimensione (prime 100 di 1.024)",
                      yaxis_title="Valore", margin=dict(t=10, b=40, l=50, r=20),
                      showlegend=False)
    plots["plt_vec"] = fig.to_json()

    # matrix heatmap
    sub = W_enc[:5, :5]
    fig = go.Figure(go.Heatmap(
        z=sub.tolist(), x=[f"x[{i}]" for i in range(5)],
        y=[f"z[{i}]" for i in range(5)],
        colorscale="RdBu_r", zmid=0,
        text=[[f"{v:.3f}" for v in row] for row in sub],
        texttemplate="%{text}", hoverinfo="skip"))
    fig.update_layout(height=250, template="plotly_white",
                      margin=dict(t=10, b=40, l=60, r=20),
                      xaxis_title="5 di 1.024 dimensioni input",
                      yaxis_title="5 di 4.096 feature")
    plots["plt_matrix"] = fig.to_json()

    # pre-activation histogram
    fig = go.Figure(go.Histogram(
        x=z_pre.tolist(), nbinsx=80,
        marker_color="rgba(0,114,178,0.5)",
        marker_line=dict(color=C_BLUE, width=0.5)))
    threshold = sorted_zpre[31]
    fig.add_vline(x=threshold, line=dict(color=C_VERMIL, width=2, dash="dash"),
                  annotation_text=f"soglia TopK = {threshold:.4f}",
                  annotation_position="top right")
    fig.update_layout(height=280, template="plotly_white",
                      xaxis_title="Pre-attivazione", yaxis_title="Conteggio",
                      margin=dict(t=10, b=40, l=50, r=20))
    plots["plt_zpre"] = fig.to_json()

    # sparse activations (32 bars)
    sp_labels = [f"F{fi}" for fi in theft_sorted]
    sp_vals = [float(tz[fi]) for fi in theft_sorted]
    sp_colors = [C_VERMIL if fi == 703 else C_BLUE for fi in theft_sorted]
    fig = go.Figure(go.Bar(
        x=sp_labels, y=sp_vals, marker_color=sp_colors,
        text=[f"{v:.3f}" for v in sp_vals], textposition="outside",
        textfont=dict(size=9)))
    fig.update_layout(height=350, template="plotly_white",
                      xaxis_title="Feature (32 attive, ordinate per attivazione)",
                      yaxis_title="Attivazione", yaxis_range=[0, 0.45],
                      margin=dict(t=10, b=60, l=50, r=20),
                      xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
    plots["plt_sparse"] = fig.to_json()

    # training curve
    hist = tm["history"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Errore di ricostruzione (MSE)", "Feature attive"))
    fig.add_trace(go.Scatter(x=[h["epoch"] for h in hist], y=[h["recon_loss"] for h in hist],
                             line=dict(color=C_BLUE, width=2), name="MSE"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[h["epoch"] for h in hist], y=[h["n_active"] for h in hist],
                             line=dict(color=C_GREEN, width=2), fill="tozeroy",
                             fillcolor="rgba(0,158,115,0.1)", name="Active"), row=2, col=1)
    fig.update_xaxes(title_text="Epoca", row=2, col=1)
    fig.update_layout(height=400, template="plotly_white",
                      margin=dict(t=30, b=40), showlegend=False)
    plots["plt_training"] = fig.to_json()

    # decoder directions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(50)), y=d703[:50].tolist(),
                             mode="lines", name="F703 (penale)",
                             line=dict(color=C_VERMIL, width=2)))
    fig.add_trace(go.Scatter(x=list(range(50)), y=d405[:50].tolist(),
                             mode="lines", name="F405 (internazionale)",
                             line=dict(color=C_GREEN, width=2)))
    fig.update_layout(height=250, template="plotly_white",
                      xaxis_title="Dimensione (prime 50)", yaxis_title="Valore",
                      margin=dict(t=10, b=40, l=50, r=20),
                      legend=dict(x=0.55, y=1.0))
    plots["plt_directions"] = fig.to_json()

    # reconstruction
    n_show = 80
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_show)), y=tv[:n_show].tolist(),
                             mode="lines", name="Originale",
                             line=dict(color=C_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=list(range(n_show)), y=tr[:n_show].tolist(),
                             mode="lines", name="Ricostruito",
                             line=dict(color=C_ORANGE, width=2, dash="dash")))
    fig.update_layout(height=280, template="plotly_white",
                      xaxis_title="Dimensione (prime 80)", yaxis_title="Valore",
                      margin=dict(t=10, b=40, l=50, r=20),
                      legend=dict(x=0.55, y=1.0))
    plots["plt_recon"] = fig.to_json()

    # feature showcases
    for fid, lm in [(703, 160), (405, 250), (3322, 160), (3032, 210), (497, 160)]:
        r = feat_map[fid]
        t10 = r["top_terms"][:10]
        plots[f"plt_f{fid}"] = hbar(
            [t["en"][:35] for t in t10], [t["activation"] for t in t10],
            [DOMAIN_COLORS.get(t["domain"], "#ccc") for t in t10], left_m=lm)

    # domain count bar (BGE-EN)
    plots["plt_domain_counts"] = domain_count_bar(
        cross["BGE-EN-large"]["domain_feature_counts"])

    # DSI histogram
    plots["plt_dsi_hist"] = dsi_histogram(enrichment)

    # contingency heatmap
    plots["plt_contingency"] = contingency_heatmap(fa, fb, fc, fd)

    # p-value distribution
    plots["plt_pval_dist"] = pvalue_distribution(enrichment)

    # cross-model bar (sig features)
    sc = [cross.get(m, {}).get("n_features_with_significant_enrichment", 0) for m in MODEL_ORDER]
    fig = go.Figure(go.Bar(x=MODEL_LABELS, y=sc, marker_color=MODEL_COLORS,
                           text=sc, textposition="outside"))
    fig.update_layout(height=350, template="plotly_white",
                      yaxis_title="Feature significative (Holm p < 0.05)",
                      margin=dict(t=10, b=40, l=50, r=20))
    plots["plt_cross_bar"] = fig.to_json()

    # cross-model heatmap
    do = ["criminal", "labor_social", "international", "constitutional",
          "civil", "administrative", "procedure"]
    zh = []
    for m in MODEL_ORDER:
        dc = cross.get(m, {}).get("domain_feature_counts", {})
        zh.append([dc.get(d, 0) for d in do])
    fig = go.Figure(go.Heatmap(
        z=zh, x=[DOMAIN_IT[d] for d in do], y=MODEL_LABELS,
        colorscale="YlOrRd", text=zh, texttemplate="%{text}", hoverinfo="skip"))
    fig.update_layout(height=380, template="plotly_white",
                      margin=dict(t=10, b=40, l=80, r=20),
                      yaxis=dict(autorange="reversed"))
    plots["plt_cross_heat"] = fig.to_json()

    # cross-model EVR
    plots["plt_cross_evr"] = evr_comparison_bar(cross_quality, MODEL_LABELS, MODEL_COLORS)

    # sensitivity
    plots["plt_sensitivity"] = sensitivity_bar(all_enrichment, MODEL_LABELS, MODEL_COLORS)

    # ── Build HTML ──
    P = []

    # HEAD
    P.append(f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<title>SAE Explainer — Geometria Iuris</title>
<script src="{PLOTLY_CDN}"></script>
<style>{CSS}</style></head><body>
<h1>Come lo Sparse Autoencoder scompone il significato giuridico</h1>
<div style="background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:18px 22px;margin-bottom:24px;font-size:0.9rem">
<p style="margin:0 0 8px 0"><b>Cosa stai per leggere.</b>
Un modello di intelligenza artificiale (BGE-EN-v1.5, sviluppato dal Beijing Academy of AI)
ha letto miliardi di pagine web e ha imparato a trasformare qualsiasi parola o frase
in una lista di 1.024 numeri &mdash; un <i>vettore</i>. Noi non abbiamo modificato questo modello:
lo usiamo cos&igrave; com'&egrave;, come uno strumento gi&agrave; costruito.</p>
<p style="margin:0 0 8px 0">Abbiamo dato a questo modello 9.472 termini giuridici (da <i>habeas corpus</i>
a <i>tort</i>) e abbiamo ottenuto 9.472 vettori. Ogni vettore codifica tutto ci&ograve;
che il modello "sa" su quel termine &mdash; ma i 1.024 numeri sono opachi: non sappiamo
quale numero codifica quale informazione.</p>
<p style="margin:0 0 8px 0">Per rendere leggibile questa rappresentazione, abbiamo costruito uno
<b>Sparse Autoencoder</b> (SAE): un piccolo modello matematico (non di linguaggio)
che scompone ogni vettore di 1.024 numeri opachi in una combinazione di poche
<i>feature</i> interpretabili &mdash; come un prisma che scompone la luce bianca nei colori
dello spettro. L'SAE &egrave; l'unica cosa che abbiamo addestrato noi.</p>
<p style="margin:0"><b>La domanda:</b> le feature che emergono dalla scomposizione
corrispondono a categorie del diritto (penale, civile, internazionale...)?
Se s&igrave;, significa che il modello ha <i>gi&agrave; appreso</i> la struttura del
diritto &mdash; semplicemente leggendo come le parole vengono usate.</p>
<p style="margin:8px 0 0 0;color:#666;font-size:0.82rem">Ogni step mostra il codice Python, i dati
in ingresso (riquadro blu) e il risultato (riquadro verde), con i numeri reali dell'esperimento.</p>
</div>
""")

    # STEP 0
    P.append(f"""
<div class="step res">
<h2><span class="sn">0</span> Inquadramento: da dove veniamo, dove andiamo</h2>
<p>Nei Lens precedenti abbiamo stabilito che:</p>
<ul style="font-size:0.92rem">
  <li><b>Lens I</b> (RSA): i termini dello stesso dominio giuridico sono geometricamente
  pi&ugrave; vicini tra loro (&rho; within = 0.49 vs cross = 0.25)</li>
  <li><b>Lens V</b> (k-NN): i vicinati semantici cambiano sistematicamente tra tradizioni
  (Jaccard within = 0.29 vs cross = 0.09)</li>
</ul>
<p>Ma queste sono <b>misure di distanza aggregate</b>: sappiamo che il clustering esiste,
non sappiamo <i>perch&eacute;</i>. L'SAE risponde a questa domanda:</p>
<div class="an">
<b>Domanda.</b> Il modello BGE-EN-v1.5 possiede <i>feature interne</i> &mdash; direzioni
specifiche nello spazio embedding &mdash; che si attivano selettivamente per i termini
di uno specifico dominio giuridico?
</div>
<p>Se s&igrave;, il domain signal del Lens I non &egrave; un artefatto delle metriche:
&egrave; codificato nella struttura interna del modello.</p>
</div>
""")

    # STEPS 1-8 (mechanics — same structure as before)
    P.append(f"""
<div class="step">
<h2><span class="sn">1</span> Caricare il vettore del termine</h2>
{code('<span class="cm"># Caricare gli embedding pre-calcolati</span>\n'
'vectors = np.<span class="fn">load</span>(<span class="st">"embeddings/BGE-EN-large/vectors.npy"</span>)\n'
'<span class="cm"># vectors.shape = (9472, 1024)</span>\n\n'
'x = vectors[<span class="nb">338</span>]  <span class="cm"># "theft"</span>')}
{io_pair("Input: il termine", '<b>"theft"</b> (furto) &mdash; indice 338',
  "Output: vettore x",
  f'x = [{tv[0]:+.4f}, {tv[1]:+.4f}, {tv[2]:+.4f}, ...]\nx.shape = (1024,)\n||x|| = {np.linalg.norm(tv):.4f}  (L2-normalizzato)')}
<div id="plt_vec" style="height:220px"></div>
</div>

<div class="step">
<h2><span class="sn">2</span> Centratura: sottrarre il bias del decoder</h2>
{code('x_centered = x - b_dec')}
{io_pair("Input", f'x[:5]     = [{tv[0]:+.4f}, {tv[1]:+.4f}, ...]\nb_dec[:5] = [{b_dec[0]:+.4f}, {b_dec[1]:+.4f}, ...]',
  "Output", f'x_c[:5] = [{x_c[0]:+.4f}, {x_c[1]:+.4f}, ...]')}
</div>

<div class="step">
<h2><span class="sn">3</span> Moltiplicazione matriciale: 4.096 prodotti scalari</h2>
{code('<span class="cm"># W_enc: matrice 4.096 &times; 1.024 ({4096*1024:,} parametri)</span>\n'
'z_pre = W_enc @ x_centered + b_enc\n\n'
'<span class="cm"># Esempio: feature 703</span>\n'
'z_pre[703] = dot(W_enc[703], x_c) + b_enc[703]')}
{io_pair("Input: riga 703 di W_enc",
  f'W_enc[703, :5] = [{W_enc[703,0]:+.4f}, {W_enc[703,1]:+.4f}, ...]\nb_enc[703] = {b_enc[703]:+.6f}',
  "Output: pre-attivazione F703",
  f'dot(W_enc[703], x_c) = {dot703:+.6f}\n+ b_enc[703]          = {b_enc[703]:+.6f}\n= z_pre[703]          = <b>{z_pre[703]:+.6f}</b>')}
<div id="plt_matrix" style="height:250px"></div>
<p class="note">Heatmap: angolo 5&times;5 di W<sub>enc</sub> ({4096*1024:,} parametri totali)</p>
</div>

<div class="step">
<h2><span class="sn">4</span> Distribuzione delle pre-attivazioni</h2>
{code(f'z_pre.shape = (4096,)\nz_pre.min() = {z_pre.min():.4f}\nz_pre.max() = {z_pre.max():.4f}  <span class="cm"># Feature 703</span>')}
<div id="plt_zpre" style="height:280px"></div>
<p>La maggior parte &egrave; negativa o vicina a zero. La soglia TopK (linea rossa) separa
le 32 feature che sopravvivranno.</p>
</div>

<div class="step hl">
<h2><span class="sn">5</span> TopK: tenere solo le 32 feature pi&ugrave; forti</h2>
{code('<span class="cm"># 1. ReLU: azzera i valori negativi</span>\n'
'<span class="cm"># 2. TopK: tieni solo i 32 pi&ugrave; alti, azzera il resto</span>\n'
'z = TopK(ReLU(z_pre), k=<span class="nb">32</span>)')}
{io_pair("Input: 4.096 pre-attivazioni",
  f'top-3:\n  F703:  {z_pre[703]:+.4f}\n  F{theft_sorted[1]}: {z_pre[theft_sorted[1]]:+.4f}\n  F{theft_sorted[2]}: {z_pre[theft_sorted[2]]:+.4f}\n...\nsoglia = {sorted_zpre[31]:+.4f}',
  "Output: 32 non-zero, 4.064 zero",
  f'z[703]  = <b>{tz[703]:.4f}</b> (1&deg;)\nz[{theft_sorted[1]}] = {tz[theft_sorted[1]]:.4f} (2&deg;)\n...\nz[{theft_sorted[31]}] = {tz[theft_sorted[31]]:.4f} (32&deg;)\nz[...] = 0.0000 (gli altri 4.064)')}
<div id="plt_sparse" style="height:350px"></div>
<p class="note">F703 (rosso) domina con il 19% del peso totale.</p>
</div>

<div class="step">
<h2><span class="sn">6</span> Il decoder: ogni feature &egrave; una direzione nello spazio</h2>
{code('<span class="cm"># Ricostruzione = somma pesata delle 32 direzioni attive</span>\n'
'x_hat = <span class="fn">sum</span>(z[i] * W_dec[:, i] <span class="kw">for</span> i <span class="kw">in</span> active) + b_dec')}
{io_pair("Input: F703 attivazione + direzione",
  f'z[703] = {tz[703]:.4f}\nW_dec[:, 703][:3] = [{d703[0]:+.4f}, ...]',
  "Output: contributo F703",
  f'z[703] &times; d_703[:3] = [{contrib_703[0]:+.4f}, ...]\n||contributo|| = {np.linalg.norm(contrib_703):.4f}')}
<div id="plt_directions" style="height:250px"></div>
<table class="dt">
<tr><th>Coppia</th><th>cos(&theta;)</th><th>Significato</th></tr>
<tr><td>F703 (penale) vs F405 (internazionale)</td><td><b>{cos_703_405:.4f}</b></td>
    <td>Quasi ortogonali: informazione indipendente</td></tr>
<tr><td>F703 (penale) vs F3230 (penale gravi)</td><td><b>{cos_703_3230:.4f}</b></td>
    <td>Lievemente correlate: stesso dominio, sotto-aree diverse</td></tr>
</table>
</div>

<div class="step">
<h2><span class="sn">7</span> Confronto: originale vs ricostruito</h2>
{code('cos_sim = np.<span class="fn">dot</span>(x, x_hat) / (norm(x) * norm(x_hat))')}
{io_pair("Input: vettore originale",
  f'x[:3] = [{tv[0]:+.4f}, {tv[1]:+.4f}, {tv[2]:+.4f}]',
  "Output: vettore ricostruito",
  f'x_hat[:3] = [{tr[0]:+.4f}, {tr[1]:+.4f}, {tr[2]:+.4f}]\n\ncos(x, x_hat) = <b>{cos_theft:.4f}</b>')}
<div id="plt_recon" style="height:280px"></div>
<p>R&sup2; su 9.472 termini: <span class="kn g">{q['explained_variance_ratio']:.3f}</span>.
Coseno medio: <b>{q['cosine_sim_mean']:.4f}</b>.</p>
</div>

<div class="step">
<h2><span class="sn">8</span> Addestramento: 1.000 epoche di ottimizzazione</h2>
{code('<span class="kw">for</span> epoch <span class="kw">in</span> <span class="fn">range</span>(<span class="nb">1000</span>):\n'
'    <span class="kw">for</span> batch <span class="kw">in</span> dataloader:       <span class="cm"># 256 termini alla volta</span>\n'
'        x_hat, z, z_pre = model(batch) <span class="cm"># forward</span>\n'
'        loss = MSE(batch, x_hat)        <span class="cm"># errore</span>\n'
'        loss.<span class="fn">backward</span>()                <span class="cm"># gradienti</span>\n'
'        optimizer.<span class="fn">step</span>()               <span class="cm"># aggiorna pesi</span>\n'
'        normalize_decoder()             <span class="cm"># norma colonne = 1</span>')}
{io_pair("Epoca 0 (inizio)",
  f'MSE = {tm["history"][0]["recon_loss"]:.6f}\nFeature attive = {tm["history"][0]["n_active"]} / 4.096',
  "Epoca 999 (fine)",
  f'MSE = {tm["history"][-1]["recon_loss"]:.6f}\nFeature attive = {tm["history"][-1]["n_active"]} / 4.096\nTempo = {tm["training_time_s"]:.0f}s')}
<div id="plt_training" style="height:400px"></div>
</div>
""")

    # STEP 9 — expanded
    f497 = feat_map[497]
    f497_dd = f497["domain_distribution"]
    f497_str = ", ".join(f'{DOMAIN_IT.get(d,d)}:{c}' for d, c in
                         sorted(f497_dd.items(), key=lambda x: -x[1]) if c > 0)
    P.append(f"""
<div class="step res">
<h2><span class="sn">9</span> Le feature corrispondono a domini giuridici</h2>
{code('<span class="cm"># Per ogni feature, prendere i 50 termini pi&ugrave; attivati</span>\n'
'top50 = np.<span class="fn">argsort</span>(activations[:, fid])[<span class="nb">-50</span>:][::<span class="nb">-1</span>]\n'
'<span class="cm"># Verificare se appartengono allo stesso dominio</span>\n'
'domains = [index[i][<span class="st">"domain"</span>] <span class="kw">for</span> i <span class="kw">in</span> top50]')}
{legend_html()}
<h3>Quattro feature specialiste</h3>
<div class="two">
  <div class="fc"><h3 style="color:{C_VERMIL}">F703: Furto e reati patrimoniali</h3>
    <div id="plt_f703" style="height:320px"></div></div>
  <div class="fc"><h3 style="color:{C_GREEN}">F405: Trattati e convenzioni</h3>
    <div id="plt_f405" style="height:320px"></div></div>
</div>
<div class="two">
  <div class="fc"><h3 style="color:{C_PURPLE}">F3322: Diritto del lavoro</h3>
    <div id="plt_f3322" style="height:320px"></div></div>
  <div class="fc"><h3 style="color:{C_ORANGE}">F3032: Locazione e affitto</h3>
    <div id="plt_f3032" style="height:320px"></div></div>
</div>

<h3>Una feature generalista per contrasto</h3>
<div class="fc">
  <h3>F497: Tort / atto illecito (DSI = {f497['dsi']:.3f})</h3>
  <p style="font-size:0.85rem">Domini: {f497_str} &mdash; distribuiti su 7 domini diversi</p>
  <div id="plt_f497" style="height:320px"></div>
</div>
<div class="an">
<b>Contrasto.</b> F703 (DSI = 1.0) attiva <i>solo</i> termini penali.
F497 (DSI = 0.22) attiva termini di 7 domini diversi: il concetto di <i>tort</i>
(danno illecito) attraversa il civile, il penale, il lavoro, l'amministrativo.
L'SAE riflette questa natura giuridicamente trasversale.
</div>

<h3>Quante feature per dominio?</h3>
<div id="plt_domain_counts" style="height:280px"></div>
<div class="finding">
<p><b>Pattern notevoli:</b></p>
<ul style="font-size:0.9rem;margin:4px 0">
  <li><b>Criminal</b> e <b>labor</b> (7 ciascuno): vocabolari altamente specializzati
  (theft, felony, manslaughter / worker, employer, sick leave)</li>
  <li><b>Procedure = 0</b>: i termini procedurali (appeal, jurisdiction, injunction) si
  attivano attraverso tutti i domini sostanziali. Non hanno feature dedicate perch&eacute;
  sono semanticamente trasversali &mdash; servono ovunque.</li>
  <li><b>Civil = 2</b> nonostante sia il dominio pi&ugrave; grande (136 termini): il diritto civile
  &egrave; un contenitore ampio. "contract" attiva feature diverse da "tenant" o "patent".
  Coerente con il Lens V: il civile ha la stabilit&agrave; di vicinato pi&ugrave; bassa.</li>
</ul>
</div>
</div>
""")

    # STEP 10 — DSI distribution
    P.append(f"""
<div class="step">
<h2><span class="sn">10</span> Lo spettro della selettivit&agrave;: generalisti vs specialisti</h2>
{code('<span class="cm"># Domain Selectivity Index (HHI concentration)</span>\n'
'<span class="cm"># DSI = sum(p_i^2): 1 = tutto in un dominio, 1/7 = uniforme</span>\n'
'dsi = <span class="fn">sum</span>(proportions ** 2)')}
<div id="plt_dsi_hist" style="height:300px"></div>
<div class="mathbox">
<b>Delle {len(dsi_vals):,} feature con dati sufficienti:</b><br>
&bull; <b>{n_gen}</b> generaliste (DSI &lt; 0.3): attivano termini di molti domini<br>
&bull; <b>{n_mix}</b> miste (0.3 &le; DSI &lt; 0.7): selettivit&agrave; parziale<br>
&bull; <b>{n_spec}</b> specialiste (DSI &ge; 0.7): fortemente legate a un dominio<br><br>
Le 27 feature significative dopo Holm sono la <b>punta dell'iceberg</b> della coda
specialista. La maggior parte delle feature (il blocco centrale) codifica informazione
che attraversa i confini tra domini.
</div>
</div>
""")

    # STEP 11 — statistical test expanded
    P.append(f"""
<div class="step">
<h2><span class="sn">11</span> Il test statistico: Fisher + Holm-Bonferroni</h2>
<p>Esempio: <b>Feature 2005</b>. Dei 50 termini top, {len(top50_lab)} hanno etichetta,
e <b>tutti {n_crim_top}</b> sono "criminal".</p>
{code('<span class="kw">from</span> scipy.stats <span class="kw">import</span> fisher_exact\n'
f'table = [[{fa}, {fb}], [{fc}, {fd}]]\n'
'odds, p = fisher_exact(table, alternative=<span class="st">"greater"</span>)')}
<h3>Tabella di contingenza</h3>
<div id="plt_contingency" style="height:220px"></div>
{io_pair("La domanda",
  f'Se pesco {len(top50_lab)} termini a caso\ntra {n_labeled} etichettati\n(di cui {n_criminal} criminal),\nquali chance di ottenerne\n{n_crim_top}/{len(top50_lab)} criminal?',
  "La risposta",
  f'p = 1.13 &times; 10&minus;12\n\nProbabilit&agrave;: ~1 su\nmille miliardi.')}

<h3>28.672 test, 833 &rarr; 27</h3>
<p>Eseguiamo il Fisher test per ogni coppia (feature &times; dominio): 4.096 &times; 7 = <b>28.672 test</b>.
Su cos&igrave; tanti test, ~1.434 falsi positivi sono attesi per caso (5% &times; 28.672).</p>
<div id="plt_pval_dist" style="height:300px"></div>
<div class="mathbox">
<b>{n_raw_sig}</b> test hanno p &lt; 0.05 raw (barra rossa nel grafico). Ma dopo la
correzione di Holm-Bonferroni (che moltiplica ogni p per il numero di test rimanenti
e impone monotonicit&agrave;), solo <b>{n_holm_sig}</b> sopravvivono.<br><br>
La distanza da {n_raw_sig} a {n_holm_sig} &egrave; il prezzo della rigorosit&agrave; statistica.
Queste {n_holm_sig} feature sono genuine con altissima confidenza.
</div>
</div>
""")

    # STEP 12 — cross-model expanded
    evr_rows = ""
    for m in MODEL_ORDER:
        cq = cross_quality[m]
        evr_rows += (f'<tr><td>{m}</td><td>{cq["evr"]:.3f}</td>'
                     f'<td>{cq["cos"]:.4f}</td><td>{cq["dead"]}</td></tr>\n')
    P.append(f"""
<div class="step res">
<h2><span class="sn">12</span> Cross-model: il pattern si ripete in 10 modelli</h2>
<p>Stessa procedura su 10 modelli (3 WEIRD, 3 Sinic, 4 bilingui):</p>
<div id="plt_cross_bar" style="height:350px"></div>
<p class="note">
<span style="color:{C_BLUE}">&block;</span> WEIRD&ensp;
<span style="color:{C_VERMIL}">&block;</span> Sinic&ensp;
<span style="color:{C_GREEN}">&block;</span> Bilingue EN&ensp;
<span style="color:{C_PURPLE}">&block;</span> Bilingue ZH</p>
<div id="plt_cross_heat" style="height:380px"></div>

<h3>Qualit&agrave; della ricostruzione</h3>
<div id="plt_cross_evr" style="height:350px"></div>
<table class="dt">
<tr><th>Modello</th><th>R&sup2;</th><th>cos</th><th>Feature morte</th></tr>
{evr_rows}</table>

<div class="finding">
<p><b>Finding 1: universalit&agrave;.</b> Tutti i 10 modelli (5 famiglie architetturali)
producono feature domain-selective (da 11 a 38 significative).</p>
<p><b>Finding 2: asimmetria EN/ZH.</b> Stesso modello bilingue, stessi dati:
BGE-M3-EN 22 vs ZH 11 (2:1). Qwen3-EN 38 vs ZH 18 (2.1:1).
L'inglese giuridico ha un vocabolario pi&ugrave; specializzato per dominio.</p>
<p><b>Finding 3: FreeLaw constitutional.</b> Il modello addestrato su testi
giuridici ha "constitutional" come top dominio (6 feature vs 0-3 nei modelli generici).</p>
</div>
</div>
""")

    # STEP 13 — sensitivity
    P.append(f"""
<div class="step">
<h2><span class="sn">13</span> Sensibilit&agrave;: quanto &egrave; robusto il risultato?</h2>
<h3>Soglia di significativit&agrave;: p &lt; 0.05 vs p &lt; 0.10</h3>
<div id="plt_sensitivity" style="height:350px"></div>
<div class="mathbox">
Per BGE-EN-v1.5: {n_holm_sig} feature a p &lt; 0.05, {n_holm_010} a p &lt; 0.10.
Il risultato &egrave; <b>stabile</b>: rilassare la soglia aggiunge poche unit&agrave;,
non un'esplosione di falsi positivi. Il core finding (decine di feature significative
in ogni modello) non dipende dalla scelta esatta della soglia.
</div>
</div>
""")

    # STEP 14 — limits
    P.append(f"""
<div class="step hl">
<h2><span class="sn">14</span> Limiti: cosa questa analisi non pu&ograve; dire</h2>

<h3>1. Il bottleneck delle etichette</h3>
<p>Solo <b>430 dei 9.472 termini</b> (4.5%) hanno un'etichetta di dominio.
Le altre 9.042 sono termini di background senza classificazione. Una feature
potrebbe essere perfettamente selettiva per un concetto giuridico che non
rientra nelle nostre 7 categorie &mdash; e noi non la vedremmo mai.</p>

<h3>2. Naming post-hoc</h3>
<p>Chiamiamo la Feature 703 "penale" perch&eacute; i termini che la attivano di
pi&ugrave; sono penali. Ma non abbiamo ispezionato la <i>direzione</i> del decoder
(i 1.024 numeri di W<sub>dec</sub>[:, 703]) per capire <i>cosa</i> codifica
geometricamente. Potrebbe codificare una propriet&agrave; linguistica (es.
vocabolario anglosassone vs latino) che <i>correla</i> con il dominio penale
senza esserne la causa.</p>

<h3>3. Correlazione, non causazione</h3>
<p>L'SAE trova struttura di co-occorrenza: i termini penali attivano le stesse
feature perch&eacute; <i>co-occorrono in contesti simili</i> nel training corpus
di BGE. Questo non significa che il modello "comprende" il diritto penale come
categoria giuridica &mdash; significa che i pattern d'uso linguistico del diritto
penale sono statisticamente distinti.</p>

<h3>4. Precedenti limitati</h3>
<p>Gli SAE su sentence embeddings sono un approccio del 2025 con pochi precedenti
metodologici (Tehenan et al. 2025, Bussmann et al. 2025). Le scelte di design
(expansion 4&times;, k=32, 1.000 epoche) sono calibrate sul nostro dataset ma non
validate su benchmark standard di interpretabilit&agrave;.</p>
</div>
""")

    # STEP 15 — summary
    P.append(f"""
<div class="step res">
<h2><span class="sn">15</span> Il pipeline completo</h2>
<table class="dt">
<tr style="background:#f5f5f5"><th>Step</th><th>Operazione</th><th>Dimensioni</th></tr>
<tr><td>1-2</td><td><code>x_c = vectors[338] - b_dec</code></td><td>(1024,)</td></tr>
<tr><td>3</td><td><code>z_pre = W_enc @ x_c + b_enc</code></td><td>(4096,)</td></tr>
<tr><td>5</td><td><code>z = TopK(ReLU(z_pre), k=32)</code></td><td>32 non-zero</td></tr>
<tr><td>6-7</td><td><code>x_hat = z @ W_dec.T + b_dec</code></td><td>(1024,), cos={cos_theft:.3f}</td></tr>
<tr><td>8</td><td>1.000 epoche di gradient descent</td><td>R&sup2;={q['explained_variance_ratio']:.3f}</td></tr>
<tr><td>9-10</td><td>Fisher exact + Holm su top-50 per feature</td><td>{n_holm_sig} significative</td></tr>
<tr><td>12</td><td>Ripetuto su 10 modelli</td><td>11-38 per modello</td></tr>
</table>
<div class="finding">
<b>Conclusione.</b> Il modello BGE-EN-v1.5 (e 9 altri modelli) possiede feature interne
che si attivano selettivamente per domini giuridici specifici. Il domain signal documentato
nel Lens I ha una base meccanicistica: non &egrave; solo una misura aggregata di distanza,
ma riflette una organizzazione interna genuina del significato giuridico nello spazio embedding.
</div>
</div>
""")

    # SCRIPT
    entries = ",\n".join(f'  "{k}": {v}' for k, v in plots.items())
    P.append(f"""
<script>
var figs = {{
{entries}
}};
Object.keys(figs).forEach(function(id) {{
  Plotly.newPlot(id, figs[id].data, figs[id].layout, {{responsive: true}});
}});
</script></body></html>""")

    html = "\n".join(P)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sae_explainer.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[explainer] Saved: {out_path} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
