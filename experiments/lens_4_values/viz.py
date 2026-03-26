"""
Visualization for Lens IV — Value Axis Projection (§3.3).

PNG figures (thesis-quality, 300 DPI):
  fig_rho_forest          — §3.3.2 3-facet forest: 15 pairs × 3 axes, ρ ± CI
  fig_rho_summary         — §3.3.3 grouped bar: 3 axes, cross vs within mean ρ ± std
  fig_axis_scatter        — §3.3.2 3-panel: WEIRD-avg vs Sinic-avg per axis
  fig_divergent_dumbbell  — §3.3.2 3-panel: top-15 dumbbell WEIRD–Sinic per axis

Interactive HTML (Plotly CDN, self-contained, 5 tabs):
  build_html              — single lens4_interactive.html

Orchestrator:
  run_viz(results_dir, results) — called from lens4.main()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.html_style import (
    C_BLACK,
    C_BLUE,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_SKY,
    C_VERMIL,
    page_head,
    plots_script,
    tabs_bar,
)

GROUP_COLORS = {
    "cross":        C_GREEN,
    "within_weird": C_BLUE,
    "within_sinic": C_VERMIL,
}

DOMAIN_COLORS = {
    "administrative":  "#0072B2",
    "civil":           "#E69F00",
    "constitutional":  "#009E73",
    "criminal":        "#D55E00",
    "international":   "#56B4E9",
    "labor_social":    "#CC79A7",
    "procedure":       "#999999",
}

DPI = 300

_RC = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def _apply_style() -> None:
    plt.rcParams.update(_RC)


def _short(label: str) -> str:
    return {
        "BGE-EN-large": "BGE-EN",
        "E5-large": "E5",
        "FreeLaw-EN": "FreeLaw",
        "BGE-ZH-large": "BGE-ZH",
        "Text2vec-large-ZH": "Text2vec",
        "Dmeta-ZH": "Dmeta",
    }.get(label, label)


def _axis_label(axis: str) -> str:
    return axis.replace("_", "–")


def _load_tradition_averages(
    results: dict, scores_dir: Path, axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weird_avg, sinic_avg) score arrays for an axis."""
    weird_avg = np.mean([
        np.load(scores_dir / f"{m}_{axis}.npy")
        for m in results["meta"]["weird_models"]
    ], axis=0)
    sinic_avg = np.mean([
        np.load(scores_dir / f"{m}_{axis}.npy")
        for m in results["meta"]["sinic_models"]
    ], axis=0)
    return weird_avg, sinic_avg


# ===========================================================================
# PNG — §3.3.2 ρ forest (3 facets)
# ===========================================================================

def fig_rho_forest(results: dict, save_dir: Path) -> Path:
    """3 facet subplots (1×3): 15 rows each, ρ ± CI colored by group."""
    _apply_style()
    s332 = results["section_332"]
    per_pair = s332["per_pair"]
    axes = results["meta"]["axes"]

    fig, axs = plt.subplots(1, 3, figsize=(17, 8), sharey=False)

    for ax, axis in zip(axs, axes):
        entries = [e for e in per_pair if e["axis"] == axis]
        order = {"within_weird": 0, "within_sinic": 1, "cross": 2}
        entries.sort(key=lambda e: (order.get(e["group"], 9), e["model_a"]))
        n = len(entries)

        for i, e in enumerate(entries):
            y = n - 1 - i
            color = GROUP_COLORS.get(e["group"], "#999")
            ax.plot(e["rho"], y, "o", color=color, ms=6, zorder=3)
            ax.plot([e["ci_low"], e["ci_high"]], [y, y], "-", color=color, lw=2)

        ax.set_yticks(range(n))
        ax.set_yticklabels([
            f"{_short(e['model_a'])} × {_short(e['model_b'])}"
            for e in reversed(entries)
        ], fontsize=7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(_axis_label(axis), fontsize=10)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

        # Annotation: cross ρ̄ / within ρ̄ / MW p
        s = s332["summary_per_axis"][axis]
        ax.text(
            0.02, -0.08,
            f"cross ρ̄={s['mean_cross_rho']:.3f}  |  "
            f"within ρ̄={s['mean_within_rho']:.3f}  |  "
            f"MW p={s['mw_p_value']:.4f}",
            transform=ax.transAxes, fontsize=7, color="#555",
        )

    handles = [
        mpatches.Patch(color=C_BLUE, label="Within-WEIRD"),
        mpatches.Patch(color=C_VERMIL, label="Within-Sinic"),
        mpatches.Patch(color=C_GREEN, label="Cross-tradition"),
    ]
    axs[-1].legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    fig.suptitle("§3.3.2 — Value axis alignment: ρ ± 95% CI", fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = save_dir / "332_rho_forest.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# PNG — §3.3.3 ρ summary bars
# ===========================================================================

def fig_rho_summary(results: dict, save_dir: Path) -> Path:
    """Grouped bar: 3 axes, cross vs within mean ρ ± std."""
    _apply_style()
    s332 = results["section_332"]
    per_pair = s332["per_pair"]
    axes = results["meta"]["axes"]

    x = np.arange(len(axes))
    width = 0.35
    cross_m, cross_s, within_m, within_s = [], [], [], []

    for axis in axes:
        cr = [e["rho"] for e in per_pair if e["axis"] == axis and e["group"] == "cross"]
        wr = [e["rho"] for e in per_pair if e["axis"] == axis and e["group"] in ("within_weird", "within_sinic")]
        cross_m.append(np.mean(cr));  cross_s.append(np.std(cr))
        within_m.append(np.mean(wr)); within_s.append(np.std(wr))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, cross_m, width, yerr=cross_s,
           label="Cross-tradition", color=C_GREEN, alpha=0.85, capsize=4)
    ax.bar(x + width / 2, within_m, width, yerr=within_s,
           label="Within-tradition", color=C_BLUE, alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([_axis_label(a) for a in axes], fontsize=9)
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title("§3.3.3 — Cross vs within alignment by axis", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    out = save_dir / "333_rho_summary.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# PNG — §3.3.2 axis scatter (WEIRD-avg vs Sinic-avg)
# ===========================================================================

def fig_axis_scatter(results: dict, save_dir: Path) -> Path:
    """3 panels: WEIRD-avg vs Sinic-avg score per axis."""
    _apply_style()
    axes = results["meta"]["axes"]
    terms = results.get("terms_core", [])
    scores_dir = save_dir.parent.parent / "scores"

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, axis in zip(axs, axes):
        w, s = _load_tradition_averages(results, scores_dir, axis)
        colors = [DOMAIN_COLORS.get(t.get("domain", ""), "#999") for t in terms]
        ax.scatter(w, s, c=colors, s=12, alpha=0.6, edgecolors="none")

        # Regression + diagonal
        lim = [min(w.min(), s.min()), max(w.max(), s.max())]
        ax.plot(lim, lim, color="#ccc", linewidth=0.8, linestyle=":")
        m, b = np.polyfit(w, s, 1)
        x_line = np.linspace(w.min(), w.max(), 100)
        ax.plot(x_line, m * x_line + b, color=C_BLACK, linewidth=1, linestyle="--", alpha=0.5)

        from scipy.stats import spearmanr
        rho, _ = spearmanr(w, s)
        ax.set_xlabel("WEIRD avg score")
        ax.set_ylabel("Sinic avg score")
        ax.set_title(f"{_axis_label(axis)}  (ρ = {rho:+.3f})", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("§3.3.2 — Value axis alignment (tradition-averaged)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "332_axis_scatter.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# PNG — §3.3.2 divergent terms dumbbell
# ===========================================================================

def fig_divergent_dumbbell(results: dict, save_dir: Path) -> Path:
    """3-panel dumbbell: top-15 terms by |WEIRD-avg − Sinic-avg| per axis."""
    _apply_style()
    axes = results["meta"]["axes"]
    terms = results.get("terms_core", [])
    scores_dir = save_dir.parent.parent / "scores"
    n_show = 15

    fig, axs = plt.subplots(1, 3, figsize=(17, 7))
    for ax, axis in zip(axs, axes):
        w, s = _load_tradition_averages(results, scores_dir, axis)
        div = np.abs(w - s)
        top_idx = np.argsort(div)[-n_show:][::-1]

        for rank, idx in enumerate(top_idx):
            y = n_show - 1 - rank
            ax.plot([w[idx], s[idx]], [y, y], "-", color="#bbb", lw=1.5, zorder=1)
            ax.plot(w[idx], y, "o", color=C_BLUE, ms=7, zorder=3)
            ax.plot(s[idx], y, "o", color=C_VERMIL, ms=7, zorder=3)

        labels = [terms[i]["en"] if i < len(terms) else str(i) for i in top_idx]
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(list(reversed(labels)), fontsize=7)
        ax.set_xlabel("Projection score")
        ax.set_title(f"{_axis_label(axis)}", fontsize=10)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    handles = [
        mpatches.Patch(color=C_BLUE, label="WEIRD avg"),
        mpatches.Patch(color=C_VERMIL, label="Sinic avg"),
    ]
    axs[-1].legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("§3.3.2 — Top-15 divergent terms per axis (|WEIRD − Sinic|)", fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = save_dir / "332_divergent_dumbbell.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Interactive HTML — Plotly
# ===========================================================================

# ---- Tab 1: §3.3.1 Axis Construction ------------------------------------

def _pj_axis_construction(results: dict) -> str:
    """Heatmap: 6 models × 3 axes (sanity pass rate) + inter-axis cosine."""
    s331 = results["section_331"]
    per_model = s331["per_model"]
    model_labels = list(per_model.keys())
    axes = results["meta"]["axes"]

    # Sanity heatmap
    z = []
    text = []
    for m in model_labels:
        row, row_t = [], []
        for a in axes:
            info = per_model[m]["axes"][a]
            rate = info["sanity_pass"] / info["sanity_total"]
            row.append(rate * 100)
            row_t.append(f"{info['sanity_pass']}/{info['sanity_total']}")
        z.append(row)
        text.append(row_t)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=["Sanity pass rate (%)", "Inter-axis cosine (select model ▾)"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Heatmap(
        z=z, x=[_axis_label(a) for a in axes],
        y=[_short(m) for m in model_labels],
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale=[[0, "#d32f2f"], [0.8, "#fdd835"], [0.9, "#8bc34a"], [1, "#2e7d32"]],
        zmin=70, zmax=100,
        hovertemplate="%{y} / %{x}: %{text} (%{z:.0f}%)<extra></extra>",
        colorbar=dict(title="%", x=0.42, len=0.9),
    ), row=1, col=1)

    # Inter-axis cosine: dropdown per model
    ortho_traces = []
    for i, m in enumerate(model_labels):
        orth = per_model[m]["orthogonality"]
        # Build 3×3 matrix
        mat = np.eye(3)
        for key, val in orth.items():
            parts = key.split("_vs_")
            r = axes.index(parts[0])
            c = axes.index(parts[1])
            mat[r, c] = val
            mat[c, r] = val
        ax_labels = [_axis_label(a)[:10] for a in axes]
        text_mat = [[f"{mat[r][c]:.3f}" for c in range(3)] for r in range(3)]
        ortho_traces.append(go.Heatmap(
            z=mat.tolist(), x=ax_labels, y=ax_labels,
            text=text_mat, texttemplate="%{text}", textfont=dict(size=11),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            hovertemplate="%{y} × %{x}: cos = %{z:.3f}<extra></extra>",
            showscale=False, visible=(i == 0),
        ))

    for t in ortho_traces:
        fig.add_trace(t, row=1, col=2)

    # Dropdown for orthogonality
    n_base = 1  # heatmap trace
    buttons = []
    for i, m in enumerate(model_labels):
        vis = [True] + [False] * len(model_labels)
        vis[n_base + i] = True
        buttons.append(dict(label=_short(m), method="update",
                            args=[{"visible": vis}]))

    fig.update_layout(
        height=400,
        template="simple_white",
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.98, xanchor="right", y=1.15, yanchor="top",
        )],
    )
    return fig.to_json()


# ---- Tab 2: §3.3.2 Alignment (forest facets) ----------------------------

def _pj_forest_facets(results: dict) -> str:
    """3 facet forest subplots: per-pair ρ ± CI, one per axis."""
    s332 = results["section_332"]
    per_pair = s332["per_pair"]
    axes = results["meta"]["axes"]

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=False,
        subplot_titles=[_axis_label(a) for a in axes],
        horizontal_spacing=0.08,
    )

    for col, axis in enumerate(axes, 1):
        entries = [e for e in per_pair if e["axis"] == axis]
        order = {"within_weird": 0, "within_sinic": 1, "cross": 2}
        entries.sort(key=lambda e: (order.get(e["group"], 9), e["model_a"]))

        for group, color, label in [
            ("within_weird", C_BLUE, "Within-WEIRD"),
            ("within_sinic", C_VERMIL, "Within-Sinic"),
            ("cross", C_GREEN, "Cross-tradition"),
        ]:
            ge = [e for e in entries if e["group"] == group]
            ylabels = [f"{_short(e['model_a'])}×{_short(e['model_b'])}" for e in ge]
            rhos = [e["rho"] for e in ge]
            err_minus = [e["rho"] - e["ci_low"] for e in ge]
            err_plus = [e["ci_high"] - e["rho"] for e in ge]

            fig.add_trace(go.Scatter(
                x=rhos, y=ylabels, mode="markers",
                marker=dict(size=8, color=color),
                error_x=dict(type="data", symmetric=False,
                             array=err_plus, arrayminus=err_minus),
                name=label, legendgroup=label,
                showlegend=(col == 1),
                hovertemplate="%{y}<br>ρ = %{x:.4f}<extra>" + label + "</extra>",
            ), row=1, col=col)

        # Annotation under each facet
        s = s332["summary_per_axis"][axis]
        fig.add_annotation(
            text=(f"cross ρ̄ = {s['mean_cross_rho']:.3f} | "
                  f"within ρ̄ = {s['mean_within_rho']:.3f} | "
                  f"MW p = {s['mw_p_value']:.4f}"),
            xref=f"x{col}" if col > 1 else "x",
            yref=f"y{col}" if col > 1 else "y",
            x=0.3, y=-0.5, showarrow=False,
            font=dict(size=10, color="#666"),
        )

    fig.update_layout(
        height=600, template="simple_white",
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
    )
    return fig.to_json()


# ---- Tab 3: §3.3.2 Scatter Explorer (tradition-averaged) ----------------

def _pj_scatter_explorer(results: dict, results_dir: Path) -> str:
    """Scatter WEIRD-avg vs Sinic-avg with 3 dropdown entries (one per axis)."""
    axes = results["meta"]["axes"]
    terms = results.get("terms_core", [])
    scores_dir = results_dir / "scores"

    fig = go.Figure()
    for i, axis in enumerate(axes):
        w, s = _load_tradition_averages(results, scores_dir, axis)

        from scipy.stats import spearmanr
        rho, _ = spearmanr(w, s)

        colors = [DOMAIN_COLORS.get(t.get("domain", ""), "#999") for t in terms]
        hovers = [
            f"<b>{t['en']}</b> ({t['zh']})<br>"
            f"Domain: {t['domain']}<br>"
            f"WEIRD: {wi:.4f}<br>Sinic: {si:.4f}<br>"
            f"|Δ| = {abs(wi - si):.4f}"
            for t, wi, si in zip(terms, w, s)
        ]

        # Scatter
        fig.add_trace(go.Scatter(
            x=w.tolist(), y=s.tolist(), mode="markers",
            marker=dict(size=5, color=colors, opacity=0.6),
            hovertext=hovers, hoverinfo="text",
            name=f"{_axis_label(axis)} (ρ = {rho:+.3f})",
            visible=(i == 0),
        ))

        # Diagonal
        lim_lo = min(w.min(), s.min())
        lim_hi = max(w.max(), s.max())
        fig.add_trace(go.Scatter(
            x=[lim_lo, lim_hi], y=[lim_lo, lim_hi],
            mode="lines", line=dict(color="#ccc", dash="dot", width=1),
            showlegend=False, hoverinfo="skip",
            visible=(i == 0),
        ))

        # Regression
        m, b = np.polyfit(w, s, 1)
        x_line = np.linspace(w.min(), w.max(), 50)
        fig.add_trace(go.Scatter(
            x=x_line.tolist(), y=(m * x_line + b).tolist(),
            mode="lines", line=dict(color=C_BLACK, dash="dash", width=1),
            showlegend=False, hoverinfo="skip",
            visible=(i == 0),
        ))

    # Dropdown: 3 entries, each toggles 3 traces (scatter, diagonal, regression)
    traces_per = 3
    buttons = []
    for i, axis in enumerate(axes):
        w, s = _load_tradition_averages(results, scores_dir, axis)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(w, s)
        vis = [False] * (len(axes) * traces_per)
        for j in range(traces_per):
            vis[i * traces_per + j] = True
        buttons.append(dict(
            label=f"{_axis_label(axis)} (ρ={rho:+.3f})",
            method="update", args=[{"visible": vis}],
        ))

    fig.update_layout(
        xaxis_title="WEIRD avg score",
        yaxis_title="Sinic avg score",
        template="simple_white", height=600,
        updatemenus=[dict(buttons=buttons, direction="down",
                         x=0.98, xanchor="right", y=1.15, yanchor="top")],
    )
    return fig.to_json()


# ---- Tab 4: §3.3.3 Which Axis Diverges Most? ----------------------------

def _pj_strip_divergence(results: dict) -> str:
    """Strip/swarm: 3 columns (axes), dots = individual ρ, colored cross/within."""
    s332 = results["section_332"]
    per_pair = s332["per_pair"]
    s333 = results["section_333"]
    axes = results["meta"]["axes"]

    fig = go.Figure()

    for group, color, label in [
        ("cross", C_GREEN, "Cross-tradition"),
        ("within_weird", C_BLUE, "Within-WEIRD"),
        ("within_sinic", C_VERMIL, "Within-Sinic"),
    ]:
        xs, ys, hovers = [], [], []
        for axis in axes:
            entries = [e for e in per_pair if e["axis"] == axis and e["group"] == group]
            for e in entries:
                xs.append(_axis_label(axis))
                ys.append(e["rho"])
                hovers.append(f"{_short(e['model_a'])}×{_short(e['model_b'])}<br>ρ = {e['rho']:.4f}")

        fig.add_trace(go.Box(
            x=xs, y=ys,
            name=label, marker_color=color,
            boxpoints="all", jitter=0.4, pointpos=0,
            line=dict(width=0), fillcolor="rgba(0,0,0,0)",
            marker=dict(size=8, opacity=0.8),
            hovertext=hovers, hoverinfo="text",
        ))

    # Add mean diamonds
    for axis in axes:
        cr = [e["rho"] for e in per_pair if e["axis"] == axis and e["group"] == "cross"]
        wr = [e["rho"] for e in per_pair if e["axis"] == axis and e["group"] in ("within_weird", "within_sinic")]
        for mean_val, color, name in [
            (np.mean(cr), C_GREEN, ""),
            (np.mean(wr), C_BLUE, ""),
        ]:
            fig.add_trace(go.Scatter(
                x=[_axis_label(axis)], y=[mean_val],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color=color,
                            line=dict(width=2, color="white")),
                showlegend=False, hoverinfo="skip",
            ))

    # KW annotation
    kw = s333["kruskal_wallis"]
    fig.add_annotation(
        text=f"Kruskal-Wallis H = {kw['H']:.2f}, p = {kw['p_value']:.4f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.12, showarrow=False,
        font=dict(size=11, color="#555"),
    )

    fig.update_layout(
        yaxis_title="Spearman ρ",
        template="simple_white", height=500,
        boxmode="group",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig.to_json()


# ---- Tab 5: Drilldown — Divergent Terms (dumbbell) ----------------------

def _pj_dumbbell(results: dict, results_dir: Path) -> str:
    """Top-20 dumbbell: WEIRD vs Sinic per axis with dropdown."""
    axes = results["meta"]["axes"]
    terms = results.get("terms_core", [])
    scores_dir = results_dir / "scores"
    n_show = 20

    fig = go.Figure()
    traces_per = 3  # lines + weird dots + sinic dots

    for i, axis in enumerate(axes):
        w, s = _load_tradition_averages(results, scores_dir, axis)
        div = np.abs(w - s)
        top_idx = np.argsort(div)[-n_show:][::-1]

        labels = [terms[idx]["en"] if idx < len(terms) else str(idx)
                  for idx in top_idx]
        y_pos = list(range(n_show - 1, -1, -1))

        # Connector lines
        for rank, idx in enumerate(top_idx):
            y = y_pos[rank]
            fig.add_trace(go.Scatter(
                x=[w[idx], s[idx]], y=[y, y],
                mode="lines", line=dict(color="#ccc", width=2),
                showlegend=False, hoverinfo="skip",
                visible=(i == 0),
            ))

        # WEIRD dots
        hovers_w = [
            f"<b>{terms[idx]['en']}</b> ({terms[idx]['zh']})<br>"
            f"Domain: {terms[idx]['domain']}<br>"
            f"WEIRD avg: {w[idx]:.4f}"
            for idx in top_idx
        ]
        fig.add_trace(go.Scatter(
            x=[w[idx] for idx in top_idx], y=y_pos,
            mode="markers", marker=dict(size=9, color=C_BLUE),
            name="WEIRD avg", legendgroup="weird",
            showlegend=(i == 0), hovertext=hovers_w, hoverinfo="text",
            visible=(i == 0),
        ))

        # Sinic dots
        hovers_s = [
            f"<b>{terms[idx]['en']}</b> ({terms[idx]['zh']})<br>"
            f"Domain: {terms[idx]['domain']}<br>"
            f"Sinic avg: {s[idx]:.4f}"
            for idx in top_idx
        ]
        fig.add_trace(go.Scatter(
            x=[s[idx] for idx in top_idx], y=y_pos,
            mode="markers", marker=dict(size=9, color=C_VERMIL),
            name="Sinic avg", legendgroup="sinic",
            showlegend=(i == 0), hovertext=hovers_s, hoverinfo="text",
            visible=(i == 0),
        ))

    # Dropdown
    # Per axis: n_show connector lines + 1 WEIRD scatter + 1 Sinic scatter
    per_axis = n_show + 2
    total = len(axes) * per_axis
    buttons = []
    for i, axis in enumerate(axes):
        vis = [False] * total
        for j in range(per_axis):
            vis[i * per_axis + j] = True
        buttons.append(dict(
            label=_axis_label(axis),
            method="update",
            args=[
                {"visible": vis},
                {"yaxis.tickvals": list(range(n_show)),
                 "yaxis.ticktext": list(reversed([
                     terms[idx]["en"] if idx < len(terms) else str(idx)
                     for idx in np.argsort(
                         np.abs(
                             _load_tradition_averages(results, scores_dir, axis)[0]
                             - _load_tradition_averages(results, scores_dir, axis)[1]
                         )
                     )[-n_show:][::-1]
                 ]))},
            ],
        ))

    # Initial y-axis labels (first axis)
    w0, s0 = _load_tradition_averages(results, scores_dir, axes[0])
    top0 = np.argsort(np.abs(w0 - s0))[-n_show:][::-1]
    init_labels = list(reversed([
        terms[idx]["en"] if idx < len(terms) else str(idx)
        for idx in top0
    ]))

    fig.update_layout(
        xaxis_title="Projection score",
        yaxis=dict(tickvals=list(range(n_show)), ticktext=init_labels, tickfont=dict(size=9)),
        template="simple_white", height=650,
        updatemenus=[dict(buttons=buttons, direction="down",
                         x=0.98, xanchor="right", y=1.15, yanchor="top")],
    )
    return fig.to_json()


# ===========================================================================
# HTML builder
# ===========================================================================

def build_html(results_dir: Path, results: dict, save_path: Path) -> Path:
    print("    Building Plotly figures...")
    plots = {}
    if "section_331" in results:
        plots["construction"] = _pj_axis_construction(results)
    if "section_332" in results:
        plots["forest"] = _pj_forest_facets(results)
        plots["scatter"] = _pj_scatter_explorer(results, results_dir)
        plots["dumbbell"] = _pj_dumbbell(results, results_dir)
    if "section_333" in results:
        plots["strip"] = _pj_strip_divergence(results)
    save_path.write_text(_html_template(plots), encoding="utf-8")
    return save_path


def _html_template(plots: dict[str, str]) -> str:
    tab_defs = [
        ("construction", "§3.3.1 Axis Construction", "pConstruct"),
        ("forest",       "§3.3.2 Alignment",         "pForest"),
        ("scatter",      "§3.3.2 Scatter",            "pScatter"),
        ("strip",        "§3.3.3 Divergence",         "pStrip"),
        ("dumbbell",     "§3.3.2 Drilldown",          "pDumbbell"),
    ]

    panels_content = {
        "construction": """
<div class="question">
  <b>How are the three value axes constructed, and do they capture independent dimensions?</b>
  A "value axis" is a direction in the embedding space that represents a conceptual scale between two opposing poles.
  To build one, we select 10 pairs of contrasting words &mdash; for example, the individual/collective axis uses pairs like (person, community), (individual, society), (autonomy, solidarity), and so on.
  For each pair, the model's numerical representation (embedding) of the "positive" pole word is subtracted from that of the "negative" pole word, producing a difference vector that points from one pole toward the other.
  The 10 resulting difference vectors are then averaged and normalised to unit length, yielding a single direction in the high-dimensional space.
  Each of the 397 legal terms can then be "projected" onto this direction &mdash; that is, we measure how far each term falls along this scale.
  Terms associated with the "individual" pole receive a positive score; terms associated with the "collective" pole receive a negative score (or vice versa, depending on convention).
  Three such axes are constructed: <i>individual/collective</i>, <i>rights/duties</i>, and <i>public/private</i>.
</div>
<div class="card">
  <h2>Axis definition</h2>
  <p>Each axis direction is the L2-normalised mean of 10 difference vectors (positive pole minus negative pole):</p>
  <p>$$\\mathbf{a} = \\mathrm{L2}\\!\\left(\\frac{1}{|P|}\\sum_{i}
  \\big(\\mathbf{e}_{\\text{pos}_i} - \\mathbf{e}_{\\text{neg}_i}\\big)\\right)$$</p>
  <p><b>Sanity check (left heatmap):</b> for each of the 10 antonym pairs defining an axis, the positive member should score higher than the negative member when projected onto that axis. For example, on the individual/collective axis, "person" should score higher than "community" (toward the individual pole). The pass rate reports the fraction of pairs where this expected ordering holds. A 100% pass rate means all pairs project in the correct direction, confirming that the axis direction meaningfully captures the intended conceptual contrast. A low pass rate would signal that the axis fails to separate its defining poles &mdash; and therefore cannot be trusted to produce meaningful scores for other terms.</p>
  <p><b>Inter-axis cosine matrix (right heatmap):</b> if the three axes capture genuinely independent dimensions of legal meaning, they should be approximately perpendicular (orthogonal) in the vector space. The inter-axis cosine matrix shows the cosine similarity between each pair of axis direction vectors: values near 0 mean the two axes are nearly orthogonal &mdash; that is, they measure truly distinct aspects of legal meaning and do not overlap. Values near +1 or &minus;1 would mean two axes point in similar or opposite directions, which would indicate that they are measuring the same underlying dimension. Select a model from the dropdown to inspect its orthogonality matrix.</p>
</div>
""",
        "forest": """
<div class="question">
  <b>Do two models agree on how legal terms rank along a value axis?</b>
  Each of the 397 legal terms is projected onto a given value axis in two different models, producing two lists of 397 numerical scores.
  The Spearman rank correlation $\\rho$ (rho) measures how well the two models agree on the <i>ordering</i> of terms along that axis &mdash; not on the exact numerical scores, but on the ranks (which term scores highest, which lowest, and so on).
  A $\\rho$ of +1 means the two models produce a perfectly identical ranking; $\\rho$ = 0 means no systematic relationship between the two rankings; $\\rho$ = &minus;1 would mean perfectly inverted rankings.
  Each row in the forest plot represents one model pair, and the horizontal position of the dot shows that pair's $\\rho$ on the given axis.
</div>
<div class="card">
  <h2>Method</h2>
  <p>For each (model pair, axis) combination:</p>
  <p>$$\\rho = \\text{Spearman}\\!\\left(\\text{proj}_{\\mathbf{a}_1}(\\mathbf{e}_{t}^{(1)}),\\;
  \\text{proj}_{\\mathbf{a}_2}(\\mathbf{e}_{t}^{(2)})\\right)$$</p>
  <p><b>Confidence intervals:</b> the horizontal error bars represent the 95% confidence interval for each $\\rho$, computed via row-resample bootstrap. This means the procedure is repeated 10,000 times ($B = 10{,}000$): in each repetition, 397 terms are drawn with replacement (some terms may appear multiple times, others may be omitted), and $\\rho$ is recomputed on the resampled set. After all 10,000 repetitions, the 2.5th and 97.5th percentiles of the resulting distribution of $\\rho$ values form the 95% confidence interval. This bootstrap method, known as row-resample, respects the structure of the data: it resamples entire terms rather than individual score entries, preserving the pairing between models.</p>
  <p><b>Annotations below each axis facet:</b> three summary statistics are shown. First, the mean $\\rho$ for cross-tradition pairs (the 9 pairs formed by combining a WEIRD model with a Sinic model). Second, the mean $\\rho$ for within-tradition pairs (the 6 pairs formed within the same tradition: 3 within-WEIRD and 3 within-Sinic). Third, the result of a Mann-Whitney $U$ test, which compares the set of cross-tradition $\\rho$ values against the set of within-tradition $\\rho$ values. The Mann-Whitney $U$ test is a non-parametric test (it makes no assumptions about the shape of the distribution) that assesses whether one group tends to produce systematically higher or lower values than the other.</p>
</div>
""",
        "scatter": """
<div class="question">
  <b>Where does each legal term fall on a value axis, according to WEIRD models versus Sinic models?</b>
  Each dot represents one of the 397 core legal terms. The horizontal position (X-axis) shows the term's average projection score across the 3 WEIRD models; the vertical position (Y-axis) shows the term's average projection score across the 3 Sinic models.
  For instance, on the individual/collective axis, a term like "contract" might receive a score of +0.3 (toward the individual pole) in the WEIRD average and +0.1 in the Sinic average, placing its dot above and to the right of the origin.
  Use the dropdown to switch between the three value axes.
</div>
<div class="card">
  <h2>Reading the scatter</h2>
  <p><b>Dotted line (identity, $y = x$):</b> points lying exactly on this line represent terms where both traditions assign the same projection score. The closer a dot is to this line, the more the two traditions agree on where that term falls along the axis.</p>
  <p><b>Points far from the identity line:</b> these represent terms where the two traditions disagree on positioning. A term well above the line scores higher in the Sinic average than in the WEIRD average; a term well below the line scores higher in the WEIRD average.</p>
  <p><b>Dashed line (OLS linear fit):</b> shows the overall linear trend between the two tradition averages. If the relationship were perfect, this line would coincide with the identity line.</p>
  <p><b>Spearman $\\rho$ in the panel title:</b> summarises the overall rank agreement between the two tradition averages. See the forest plot panel for a detailed explanation of $\\rho$.</p>
  <p><b>Dot colours:</b> each dot is coloured by branch of law (e.g., constitutional, criminal, civil). Hover over any dot to see the term name (in English and Chinese), its branch of law, and the exact projection scores in both traditions.</p>
</div>
""",
        "strip": """
<div class="question">
  <b>Does the level of cross-tradition agreement vary across the three value axes?</b>
  Each dot in this strip plot (also called a swarm plot) represents one model pair's Spearman $\\rho$ on a given axis.
  The dots are grouped into three columns (one per axis: individual/collective, rights/duties, public/private) and coloured by pair type: green dots represent cross-tradition pairs (9 pairs total, each combining a WEIRD model with a Sinic model), blue dots represent within-WEIRD pairs (3 pairs, comparing WEIRD models with each other), and red dots represent within-Sinic pairs (3 pairs, comparing Sinic models with each other).
  Diamond-shaped markers show the group mean for each colour within each column.
</div>
<div class="card">
  <h2>Kruskal-Wallis test</h2>
  <p>The Kruskal-Wallis $H$ test is a non-parametric statistical test that compares whether the distributions of cross-tradition $\\rho$ values differ across the three axes. "Non-parametric" means it does not assume that the data follow a normal (bell-shaped) distribution &mdash; instead, it works by ranking all the values and comparing the average ranks across groups. It is the rank-based analogue of a one-way ANOVA (analysis of variance), a classical test for comparing group means.</p>
  <p>Concretely, the test takes all the cross-tradition $\\rho$ values (9 per axis = 27 values total), ranks them from lowest to highest, and asks: are the ranks distributed evenly across the three axes, or does one axis tend to cluster at higher or lower ranks?</p>
  <p>The $H$ statistic and $p$-value are reported in the panel annotation. A low $p$-value (conventionally below 0.05) would signal that at least one axis has a systematically different distribution of $\\rho$ values from the others &mdash; meaning that the degree of cross-tradition agreement is not uniform across all three conceptual dimensions.</p>
</div>
""",
        "dumbbell": """
<div class="question">
  <b>Which specific legal terms show the largest positioning gap between traditions on each value axis?</b>
  For each axis, the 20 terms with the largest absolute difference between their WEIRD tradition average score and their Sinic tradition average score are displayed.
  For example, if "sovereignty" has a WEIRD average of +0.4 and a Sinic average of &minus;0.1 on the public/private axis, the gap is 0.5 and the horizontal connector between the two dots will be long, making the term visually prominent in the chart.
  This chart highlights terms where the two legal traditions position a concept very differently along the same conceptual scale.
</div>
<div class="card">
  <h2>Reading the dumbbell chart</h2>
  <p><b>Term labels (left):</b> terms are sorted by decreasing gap size, with the largest gap at the top.</p>
  <p><b>Blue dot:</b> the WEIRD tradition average &mdash; that is, the mean projection score of the term across the 3 WEIRD models (E5, BGE-EN, FreeLaw).</p>
  <p><b>Red dot:</b> the Sinic tradition average &mdash; that is, the mean projection score of the term across the 3 Sinic models (BGE-ZH, Text2vec, Dmeta).</p>
  <p><b>Grey connector line:</b> its length represents the absolute difference between the two tradition averages. A longer line means a larger gap in how the two traditions position the term along the axis.</p>
  <p><b>Dropdown:</b> use the dropdown menu (top right) to switch between the three value axes (individual/collective, rights/duties, public/private). The terms shown will change because different terms may exhibit the largest tradition gaps on different axes.</p>
  <p><b>Hover:</b> move the cursor over any dot to see the term name (English and Chinese), its branch of law, and the exact numerical projection score.</p>
</div>
""",
    }

    active_tabs = [(pid, label) for key, label, pid in tab_defs if key in plots]

    panels_html = ""
    plots_keyed: dict[str, str] = {}
    first = True
    for key, label, div_id in tab_defs:
        if key not in plots:
            continue
        active = " active" if first else ""
        content = panels_content.get(key, "")
        panels_html += (
            f'<div id="{div_id}" class="panel{active}">'
            f'{content}'
            f'<div id="plt_{key}"></div></div>\n'
        )
        plots_keyed[f"plt_{key}"] = plots[key]
        first = False

    head = page_head("Lens IV — Value Axis Projection")
    tab_html = tabs_bar(active_tabs)
    script = plots_script(plots_keyed)

    return f"""<!DOCTYPE html>
<html lang="en">
{head}
<body>
<h1>Lens IV — Value Axis Projection</h1>
<p class="subtitle">§3.3 — Projection of 397 legal terms onto three value axes (individual/collective, rights/duties, public/private) across 6 models (3 WEIRD, 3 Sinic). Rank correlation and bootstrap confidence intervals.</p>
{tab_html}

{panels_html}
{script}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_viz(results_dir: Path, results: dict) -> None:
    """Generate all Lens IV figures (PNG + HTML). Called from lens4.main()."""
    png_dir = results_dir / "figures" / "png"
    html_dir = results_dir / "figures" / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    print("\n[viz] Generating PNG figures...")

    if "section_332" in results:
        p = fig_rho_forest(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_axis_scatter(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_divergent_dumbbell(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_333" in results or "section_332" in results:
        p = fig_rho_summary(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    print("[viz] Generating interactive HTML...")
    html_path = build_html(results_dir, results, html_dir / "lens4_interactive.html")
    print(f"  {html_path.name}")

    print(f"[viz] Done — {len(generated)} PNG + 1 HTML")
