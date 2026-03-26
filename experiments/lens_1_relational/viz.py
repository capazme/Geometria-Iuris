"""
Visualization for Lens I — Relational Distance Structure.

PNG figures (thesis-quality, 300 DPI):
  fig_311_domains        — §3.1.1 domain distribution bar chart
  fig_311_confidence     — §3.1.1 k-NN confidence histogram
  fig_311_intra_inter    — §3.1.1 intra vs inter violin (3-panel, WEIRD)
  fig_311_legal_control  — §3.1.1 legal vs control violin (3-panel, WEIRD)
  fig_312_topology       — §3.1.2 K×K domain topology heatmaps (3-panel)
  fig_rsa_forest         — §3.1.4 RSA forest plot (15 pairs, colored by group)
  fig_rsa_null           — §3.1.4 RSA null distributions (15-panel grid)

Interactive HTML (Plotly CDN, self-contained):
  build_html             — single lens1_interactive.html with 5 tabs

Orchestrator:
  run_viz(results_dir, results) — called from lens1.main()
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.html_style import (
    CSS, C_BLUE, C_ORANGE, C_GREEN, C_SKY, C_VERMIL, C_PURPLE, C_BLACK,
    format_p, page_head, plots_script, tabs_bar,
)

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


# ---------------------------------------------------------------------------
# §3.1.1 — Domain distribution
# ---------------------------------------------------------------------------

def fig_311_domains(results: dict, save_dir: Path) -> Path:
    """Bar chart of background term domain distribution (§3.1.1)."""
    _apply_style()
    domain_counts = results["section_311"]["domain_counts"]
    domains = sorted(domain_counts, key=domain_counts.__getitem__, reverse=True)
    counts = [domain_counts[d] for d in domains]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(domains)), counts, color=C_BLUE, alpha=0.85, width=0.6)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Term count")
    ax.set_title("§3.1.1 — Background term domain distribution", fontsize=11)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count), ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    out = save_dir / "311_domain_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_311_confidence(results_dir: Path, results: dict, save_dir: Path) -> Path:
    """Histogram of k-NN confidence scores with threshold line (§3.1.1)."""
    _apply_style()
    csv_path = results_dir / "background_review.csv"
    confidences = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            confidences.append(float(row["confidence"]))
    conf = np.array(confidences)
    threshold = results["section_311"]["low_confidence_threshold"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf, bins=40, color=C_BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(
        threshold, color=C_VERMIL, linewidth=1.5, linestyle="--",
        label=f"Threshold {threshold:.2f}  (4/k)",
    )
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title("§3.1.1 — k-NN confidence distribution", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out = save_dir / "311_confidence.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# §3.1.1 — Intra vs inter-domain distances
# ---------------------------------------------------------------------------

def fig_311_intra_inter(
    results_dir: Path, weird_labels: list[str], save_dir: Path
) -> Path:
    """3-panel violin: intra vs inter-domain distances per WEIRD model (§3.1.1)."""
    _apply_style()
    dist_dir = results_dir / "distances"

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    for ax, label in zip(axes, weird_labels):
        npz = np.load(dist_dir / f"{label}.npz")
        parts = ax.violinplot(
            [npz["intra"], npz["inter"]], positions=[0, 1],
            showmedians=True, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], [C_SKY, C_ORANGE]):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color(C_BLACK)
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Intra-domain", "Inter-domain"])
        ax.set_title(_short(label), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Cosine distance")

    fig.suptitle("§3.1.1 — Intra vs inter-domain distances (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "311_intra_inter.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# §3.1.1 — Legal vs control distances
# ---------------------------------------------------------------------------

def fig_311_legal_control(
    results_dir: Path, weird_labels: list[str], save_dir: Path
) -> Path:
    """3-panel violin: legal vs control distances per WEIRD model (§3.1.1)."""
    _apply_style()
    dist_dir = results_dir / "distances"

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    for ax, label in zip(axes, weird_labels):
        npz = np.load(dist_dir / f"{label}.npz")
        parts = ax.violinplot(
            [npz["legal"], npz["control"]], positions=[0, 1],
            showmedians=True, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], [C_BLUE, C_VERMIL]):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color(C_BLACK)
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Legal", "Control"])
        ax.set_title(_short(label), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Cosine distance")

    fig.suptitle("§3.1.1 — Legal vs control distances (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "311_legal_control.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# §3.1.2 — Domain topology
# ---------------------------------------------------------------------------

def fig_312_topology(results: dict, save_dir: Path) -> Path:
    """3-panel annotated heatmaps: K×K domain topology per WEIRD model (§3.1.2)."""
    _apply_style()
    per_model = results["section_31"]["per_model"]
    model_labels = list(per_model.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, mlabel in zip(axes, model_labels):
        topo = per_model[mlabel]["domain_topology"]
        domains = topo["domains"]
        matrix = np.array(topo["matrix"])
        k = len(domains)
        abbrev = [d[:5] for d in domains]

        im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(abbrev, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(abbrev, fontsize=7)
        ax.set_title(_short(mlabel), fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=5, color="black")

    fig.suptitle("§3.1.2 — Domain topology K×K (WEIRD models)", fontsize=11)
    plt.tight_layout()
    out = save_dir / "312_topology.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# RSA — Forest plot
# ---------------------------------------------------------------------------

def fig_rsa_forest(results: dict, save_dir: Path) -> Path:
    """Forest plot: Spearman ρ ± 95% CI for all 15 RSA pairs (§3.1.4)."""
    _apply_style()
    rsa_data = results["section_314"]

    pairs: list[dict] = []
    for group, color in [
        ("within_weird", C_BLUE),
        ("within_sinic", C_VERMIL),
        ("cross_tradition", C_GREEN),
    ]:
        for r in rsa_data[group]:
            pairs.append({
                "label": f"{_short(r['model_a'])} × {_short(r['model_b'])}",
                "rho": r["rho"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "color": color,
            })

    n = len(pairs)
    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.5 + 1.5)))

    for i, p in enumerate(pairs):
        y = n - 1 - i
        ax.plot(p["rho"], y, "o", color=p["color"], ms=7, zorder=3)
        ax.plot([p["ci_low"], p["ci_high"]], [y, y], "-", color=p["color"], lw=2.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([p["label"] for p in reversed(pairs)], fontsize=8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_xlim(-0.15, 0.85)
    ax.set_title("RSA forest plot — §3.1.4", fontsize=11)
    ax.legend(
        handles=[
            mpatches.Patch(color=C_BLUE, label="Within-WEIRD"),
            mpatches.Patch(color=C_VERMIL, label="Within-Sinic"),
            mpatches.Patch(color=C_GREEN, label="Cross-tradition"),
        ],
        loc="lower right", frameon=False, fontsize=8,
    )
    plt.tight_layout()
    out = save_dir / "rsa_forest.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# RSA — Null distributions
# ---------------------------------------------------------------------------

def fig_rsa_null(results_dir: Path, results: dict, save_dir: Path) -> Path:
    """5×3 grid of null distributions with observed ρ line (§3.1.4)."""
    _apply_style()
    dist_dir = results_dir / "distributions"
    rsa_data = results["section_314"]

    all_pairs: list[tuple[dict, str, str]] = []
    for group, color in [
        ("within_weird", C_BLUE),
        ("within_sinic", C_VERMIL),
        ("cross_tradition", C_GREEN),
    ]:
        for r in rsa_data[group]:
            all_pairs.append((r, color, group))

    n = len(all_pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))
    axes_flat = axes.flatten()

    for i, (r, color, _) in enumerate(all_pairs):
        ax = axes_flat[i]
        la, lb = r["model_a"], r["model_b"]
        fname = dist_dir / f"{la}_x_{lb}.npz"

        if not fname.exists():
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(f"{_short(la)} × {_short(lb)}", fontsize=8)
            continue

        null = np.load(fname)["null"]
        ax.hist(null, bins=50, color=color, alpha=0.6, density=True, edgecolor="none")
        ax.axvline(r["rho"], color=C_BLACK, linewidth=1.5,
                   label=f"ρ={r['rho']:.3f}")
        ax.set_title(f"{_short(la)} × {_short(lb)}", fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_xlabel("ρ (null)", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("RSA null distributions — §3.1.4", fontsize=11)
    plt.tight_layout()
    out = save_dir / "rsa_null_distributions.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Interactive HTML — Plotly helpers
# ---------------------------------------------------------------------------

def _downsample(arr: np.ndarray, max_n: int = 5000) -> np.ndarray:
    """Downsample array for Plotly violin (KDE is stable at ~5k points)."""
    if len(arr) <= max_n:
        return arr
    rng = np.random.default_rng(42)
    return rng.choice(arr, size=max_n, replace=False)


def _pj_violin_intra_inter(
    results_dir: Path, results: dict, weird_labels: list[str],
) -> str:
    """Violin: intra vs inter-domain distances, with stat annotations."""
    dist_dir = results_dir / "distances"
    per_model = results["section_31"]["per_model"]
    fig = make_subplots(cols=3, rows=1, horizontal_spacing=0.08,
                        subplot_titles=[_short(l) for l in weird_labels])
    for i, label in enumerate(weird_labels, 1):
        path = dist_dir / f"{label}.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for arr, name, color in [
            (npz["intra"], "Intra-domain", C_SKY),
            (npz["inter"], "Inter-domain", C_ORANGE),
        ]:
            fig.add_trace(go.Violin(
                y=_downsample(arr), name=name, box_visible=True,
                meanline_visible=True, fillcolor=color, line_color=C_BLACK,
                opacity=0.75, showlegend=(i == 1),
            ), row=1, col=i)
        mw = per_model[label]["intra_vs_inter"]
        fig.add_annotation(
            text=(f"r = {mw['effect_r']:+.3f}<br>"
                  f"med: {mw['median_x']:.3f} vs {mw['median_y']:.3f}"),
            xref=f"x{'' if i == 1 else i} domain",
            yref=f"y{'' if i == 1 else i} domain",
            x=0.5, y=1.0, showarrow=False,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1, borderpad=4,
        )
    fig.update_layout(
        template="simple_white", height=450, violinmode="group",
        margin=dict(t=50, b=40),
    )
    for i in range(1, 4):
        fig.update_yaxes(title_text="Cosine distance" if i == 1 else "", row=1, col=i)
    return fig.to_json()


def _pj_violin_legal_control(
    results_dir: Path, results: dict, weird_labels: list[str],
) -> str:
    """Violin: legal vs control distances, with stat annotations."""
    dist_dir = results_dir / "distances"
    per_model = results["section_31"]["per_model"]
    fig = make_subplots(cols=3, rows=1, horizontal_spacing=0.08,
                        subplot_titles=[_short(l) for l in weird_labels])
    for i, label in enumerate(weird_labels, 1):
        path = dist_dir / f"{label}.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for arr, name, color in [
            (npz["legal"], "Legal-legal", C_BLUE),
            (npz["control"], "Legal-control", C_VERMIL),
        ]:
            fig.add_trace(go.Violin(
                y=_downsample(arr), name=name, box_visible=True,
                meanline_visible=True, fillcolor=color, line_color=C_BLACK,
                opacity=0.75, showlegend=(i == 1),
            ), row=1, col=i)
        mw = per_model[label]["legal_vs_control"]
        r_val = mw["effect_r"]
        flag = " ⚠" if r_val < 0 else ""
        fig.add_annotation(
            text=(f"r = {r_val:+.3f}{flag}<br>"
                  f"med: {mw['median_x']:.3f} vs {mw['median_y']:.3f}"),
            xref=f"x{'' if i == 1 else i} domain",
            yref=f"y{'' if i == 1 else i} domain",
            x=0.5, y=1.0, showarrow=False,
            font=dict(size=11, color=C_VERMIL if r_val < 0 else "#222"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=C_VERMIL if r_val < 0 else "#ccc",
            borderwidth=1, borderpad=4,
        )
    fig.update_layout(
        template="simple_white", height=450, violinmode="group",
        margin=dict(t=50, b=40),
    )
    for i in range(1, 4):
        fig.update_yaxes(title_text="Cosine distance" if i == 1 else "", row=1, col=i)
    return fig.to_json()


def _pj_topology(results: dict) -> str:
    """3-panel K×K domain topology heatmaps."""
    per_model = results["section_31"]["per_model"]
    models = list(per_model.keys())
    fig = make_subplots(cols=3, rows=1, horizontal_spacing=0.06,
                        subplot_titles=[_short(m) for m in models])
    for i, mlabel in enumerate(models, 1):
        topo = per_model[mlabel]["domain_topology"]
        domains = topo["domains"]
        matrix = np.array(topo["matrix"])
        fig.add_trace(go.Heatmap(
            z=matrix, x=domains, y=domains, colorscale="RdYlBu_r",
            showscale=(i == 3), zmin=0.15, zmax=0.70,
            text=[[f"{v:.3f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>",
        ), row=1, col=i)
    fig.update_layout(
        template="simple_white", height=480,
        margin=dict(t=50, b=10),
    )
    return fig.to_json()


def _pj_rsa_forest(results: dict) -> str:
    """Forest plot: Spearman rho with CI, grouped by tradition pair type."""
    rsa_data = results["section_314"]

    # Build ordered list: within-WEIRD, within-Sinic, then cross
    items: list[tuple[str, str, float, float, float, str]] = []
    for group, color in [
        ("within_weird", C_BLUE),
        ("within_sinic", C_VERMIL),
        ("cross_tradition", C_GREEN),
    ]:
        for r in rsa_data[group]:
            items.append((
                f"{_short(r['model_a'])} × {_short(r['model_b'])}",
                group.replace("_", "-"),
                r["rho"], r["ci_low"], r["ci_high"], color,
            ))

    # Reverse for bottom-to-top display
    items = items[::-1]
    fig = go.Figure()
    seen: set[str] = set()
    for label, gname, rho, lo, hi, color in items:
        show = gname not in seen
        seen.add(gname)
        fig.add_trace(go.Scatter(
            x=[rho], y=[label], mode="markers",
            marker=dict(size=10, color=color),
            error_x=dict(type="data", symmetric=False,
                         array=[hi - rho], arrayminus=[rho - lo]),
            name=gname if show else "",
            legendgroup=gname, showlegend=show,
            hovertemplate=f"ρ = {rho:.3f} [{lo:.3f}, {hi:.3f}]<extra>{label}</extra>",
        ))

    # Mean markers
    summary = rsa_data["summary"]
    for val, label, color, sym in [
        (summary["mean_rho_within_weird"], "WEIRD mean", C_BLUE, "diamond"),
        (summary["mean_rho_within_sinic"], "Sinic mean", C_VERMIL, "diamond"),
        (summary["mean_rho_cross"], "Cross mean", C_GREEN, "diamond"),
    ]:
        fig.add_vline(x=val, line_dash="dot", line_color=color, opacity=0.4)

    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="Spearman ρ",
        template="simple_white", height=520,
        xaxis=dict(range=[-0.05, 0.80]),
        margin=dict(l=200, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig.to_json()


def _pj_rsa_null(results_dir: Path, results: dict) -> str:
    """Combined null distribution plot for selected RSA pairs."""
    dist_dir = results_dir / "distributions"
    rsa_data = results["section_314"]
    # Pick 3 representative pairs: best within-WEIRD, best within-Sinic, best cross
    representatives = []
    for group, color, gname in [
        ("within_weird", C_BLUE, "Within-WEIRD"),
        ("within_sinic", C_VERMIL, "Within-Sinic"),
        ("cross_tradition", C_GREEN, "Cross-tradition"),
    ]:
        best = max(rsa_data[group], key=lambda r: r["rho"])
        representatives.append((best, color, gname))

    fig = make_subplots(cols=3, rows=1, horizontal_spacing=0.06,
                        subplot_titles=[g for _, _, g in representatives])
    for i, (r, color, gname) in enumerate(representatives, 1):
        fname = dist_dir / f"{r['model_a']}_x_{r['model_b']}.npz"
        if not fname.exists():
            continue
        null = np.load(fname)["null"]
        fig.add_trace(go.Histogram(
            x=null.tolist(), nbinsx=60, marker_color=color, opacity=0.5,
            name=f"Null ({gname})", showlegend=False,
        ), row=1, col=i)
        fig.add_vline(
            x=r["rho"], line_color=C_BLACK, line_width=2.5,
            annotation_text=f"ρ = {r['rho']:.3f}",
            annotation_font_size=11, row=1, col=i,
        )
        fig.update_xaxes(title_text="ρ (null)", row=1, col=i)
    fig.update_layout(
        template="simple_white", height=320,
        margin=dict(t=50, b=50),
    )
    return fig.to_json()


def _pj_rsa_matrix(results: dict) -> str:
    """6×6 RSA correlation matrix: WEIRD models first, then Sinic."""
    rsa = results["section_314"]
    weird = results["meta"]["weird_models"]
    sinic = results["meta"]["sinic_models"]
    labels = weird + sinic
    short_labels = [_short(l) for l in labels]
    n = len(labels)

    # Build symmetric matrix from pairwise results
    rho_map: dict[tuple[str, str], float] = {}
    for group in ("within_weird", "within_sinic", "cross_tradition"):
        for r in rsa[group]:
            rho_map[(r["model_a"], r["model_b"])] = r["rho"]
            rho_map[(r["model_b"], r["model_a"])] = r["rho"]

    matrix = np.ones((n, n), dtype=np.float64)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                matrix[i, j] = rho_map.get((a, b), 0.0)

    # Annotation text
    text = [[f"{matrix[i, j]:.3f}" if i != j else "1" for j in range(n)]
            for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=short_labels, y=short_labels,
        colorscale="RdYlBu_r", zmin=0, zmax=0.75, reversescale=True,
        text=text, texttemplate="%{text}", textfont=dict(size=13),
        hovertemplate="%{y} × %{x}: ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", len=0.8),
    ))

    # Add block-structure lines
    n_weird = len(weird)
    fig.add_shape(type="rect", x0=-0.5, x1=n_weird - 0.5,
                  y0=-0.5, y1=n_weird - 0.5,
                  line=dict(color=C_BLUE, width=2.5))
    fig.add_shape(type="rect", x0=n_weird - 0.5, x1=n - 0.5,
                  y0=n_weird - 0.5, y1=n - 0.5,
                  line=dict(color=C_VERMIL, width=2.5))

    fig.update_layout(
        template="simple_white", height=440, width=520,
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis=dict(side="bottom", tickangle=0),
        yaxis=dict(autorange="reversed"),
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _signal_summary_html(results: dict) -> str:
    """Build HTML summary table for §3.1.1 signal tests."""
    per_model = results["section_31"]["per_model"]
    rows = []
    for label in per_model:
        ii = per_model[label]["intra_vs_inter"]
        lc = per_model[label]["legal_vs_control"]
        lc_flag = ' class="anomaly"' if lc["effect_r"] < 0 else ""
        rows.append(f"""<tr>
          <td><b>{_short(label)}</b></td>
          <td>{ii['median_x']:.3f}</td><td>{ii['median_y']:.3f}</td>
          <td><b>{ii['effect_r']:+.3f}</b></td><td>{format_p(ii['p_value'])}</td>
          <td>{lc['median_x']:.3f}</td><td>{lc['median_y']:.3f}</td>
          <td{lc_flag}><b>{lc['effect_r']:+.3f}</b></td>
          <td{lc_flag}>{format_p(lc['p_value'])}</td>
        </tr>""")
    return "\n".join(rows)


def _rsa_table_html(results: dict) -> str:
    """Build HTML table rows for RSA results."""
    rsa = results["section_314"]
    rows = []
    for group, group_label, css in [
        ("within_weird", "Within-WEIRD", "rsa-weird"),
        ("within_sinic", "Within-Sinic", "rsa-sinic"),
        ("cross_tradition", "Cross-tradition", "rsa-cross"),
    ]:
        for r in rsa[group]:
            rows.append(f"""<tr class="{css}">
              <td>{group_label}</td>
              <td>{_short(r['model_a'])} × {_short(r['model_b'])}</td>
              <td><b>{r['rho']:.3f}</b></td>
              <td>[{r['ci_low']:.3f}, {r['ci_high']:.3f}]</td>
              <td>{r['r_squared']:.3f}</td>
              <td>{format_p(r['p_value'])}</td>
            </tr>""")
    return "\n".join(rows)


def build_html(results_dir: Path, results: dict, save_path: Path) -> Path:
    """Build a single self-contained Plotly HTML with tab navigation."""
    weird_labels = results["meta"]["weird_models"]
    print("    Building Plotly figures...")
    plots = {
        "rsa_matrix":    _pj_rsa_matrix(results),
        "intra_inter":   _pj_violin_intra_inter(results_dir, results, weird_labels),
        "legal_control": _pj_violin_legal_control(results_dir, results, weird_labels),
        "topology":      _pj_topology(results),
        "forest":        _pj_rsa_forest(results),
        "null":          _pj_rsa_null(results_dir, results),
    }
    html = _html_template(plots, results)
    save_path.write_text(html, encoding="utf-8")
    return save_path


def _html_template(plots: dict[str, str], results: dict) -> str:
    meta = results["meta"]
    s31 = results.get("section_31", {})
    s314 = results.get("section_314", {})
    summary = s314.get("summary", {})
    n_core = s31.get("n_core", "?")
    n_ctrl = s31.get("n_control", "?")

    signal_rows = _signal_summary_html(results) if s31 else ""
    rsa_rows = _rsa_table_html(results) if s314 else ""

    rho_w = summary.get("mean_rho_within_weird", 0)
    rho_s = summary.get("mean_rho_within_sinic", 0)
    rho_x = summary.get("mean_rho_cross", 0)
    drop = summary.get("cross_tradition_drop", 0)

    _head = page_head("Lens I — Relational Distance Structure")
    _tabs = tabs_bar([
        ("overview", "Overview"),
        ("signal", "Domain Signal"),
        ("topology", "Domain Topology"),
        ("rsa", "Cross-Tradition RSA"),
    ])
    _plots_dict = {
        "plt_rsa_matrix":    plots["rsa_matrix"],
        "plt_intra_inter":   plots["intra_inter"],
        "plt_legal_control": plots["legal_control"],
        "plt_topology":      plots["topology"],
        "plt_forest":        plots["forest"],
        "plt_null":          plots["null"],
    }
    _script = plots_script(_plots_dict)

    return f"""<!DOCTYPE html>
<html lang="en">
{_head}
<body>
<h1>Lens I &mdash; Relational Distance Structure</h1>
<p class="subtitle">
  {n_core} core terms &times; 6 models (3&nbsp;WEIRD + 3&nbsp;Sinic) &middot;
  {n_ctrl} control terms &middot;
  Mantel B={meta.get("n_perm", "?"):,} &middot;
  Bootstrap B={meta.get("n_boot", "?"):,} &middot;
  Run: {meta.get("date", "?")[:10]}
</p>

{_tabs}

<!-- ==================== OVERVIEW ==================== -->
<div id="overview" class="panel active">
  <div class="card">
    <h2>Experimental setup</h2>
    <p>This experiment uses <b>embedding models</b>: neural networks that have been
    trained on very large collections of text (called "corpora") and that learn to
    represent each word or phrase as a list of numbers &mdash; a numerical vector &mdash;
    in a high-dimensional space. The key property of these models is that words
    which tend to appear in similar linguistic contexts receive similar vectors.
    Six such models are used here. Three were trained primarily on English-language
    corpora and are labelled <b>"WEIRD"</b> (an acronym introduced by Henrich et al.
    2010, standing for Western, Educated, Industrialised, Rich, Democratic, which
    characterises the cultural context of the training data). The other three were
    trained on Chinese-language corpora and are labelled <b>"Sinic"</b>.</p>

    <p>The legal vocabulary under examination consists of <b>397 core legal terms</b>
    drawn from the <b>Hong Kong Department of Justice Bilingual Legal Glossary</b>,
    a professionally curated bilingual resource that pairs English legal terms with
    their official Chinese translations. These 397 terms span <b>7 branches of law</b>:
    criminal law (e.g., "murder", "manslaughter", "robbery"),
    civil law (e.g., "tort", "negligence", "easement"),
    constitutional law (e.g., "sovereignty", "separation of powers", "judicial review"),
    international law (e.g., "treaty", "asylum", "extradition"),
    procedure (e.g., "discovery", "subpoena", "appeal"),
    labour and social law (e.g., "collective bargaining", "minimum wage", "pension"),
    and administrative law (e.g., "delegated legislation", "licensing", "judicial notice").
    In addition, <b>100 control terms</b> are included: these are words from the
    <b>Swadesh-100 list</b>, a standard set of basic, culturally universal vocabulary
    items such as "water", "fire", "hand", "sleep", "stone", and "sun". Because
    these words have no specifically legal content, they serve as a <b>non-legal
    baseline</b> against which the behaviour of legal terms can be compared.</p>

    <p>For each of the 6 models, a <b>Relational Dissimilarity Matrix (RDM)</b> is
    computed. An RDM is a large symmetric table with 397 rows and 397 columns, one
    for each legal term. Each cell records the <b>cosine distance</b> between the
    vectors of the two corresponding terms: formally,
    $\\text{{RDM}}[i,j] = 1 - \\cos(\\mathbf{{v}}_i, \\mathbf{{v}}_j)$. Cosine distance
    is a measure of how different two vectors are: a value of 0 means the two vectors
    are identical (pointing in exactly the same direction), while a value approaching 1
    means they are maximally different (though in practice, values in these models
    rarely exceed 0.7). The RDM thus captures the full pattern of semantic distances
    among all 397 legal terms as perceived by one model. Three analyses are then
    performed on these matrices:</p>
    <ol style="font-size:0.88rem; color:#444; padding-left:20px;">
      <li><b>Intra vs. inter-domain distances</b> &mdash; the 78,606 pairwise distances
      from each RDM are divided into those where both terms belong to the same branch
      of law ("intra-domain") and those where the terms belong to different branches
      ("inter-domain"). A Mann-Whitney $U$ test then asks whether these two groups
      of distances differ systematically.</li>
      <li><b>Legal vs. control distances</b> &mdash; distances among the 397 legal
      terms are compared with distances between legal terms and the 100 Swadesh-100
      control words, using the same Mann-Whitney $U$ test. This checks whether the
      model treats legal vocabulary differently from ordinary, non-legal words.</li>
      <li><b>Cross-model RSA</b> &mdash; the Spearman rank correlation between the RDMs
      of two different models is computed, measuring how similarly the two models
      organise the 397 legal terms relative to each other. The 15 possible model pairs
      are partitioned into within-WEIRD, within-Sinic, and cross-tradition comparisons.</li>
    </ol>
  </div>

  <div class="two-col">
    <div>
      <p class="plot-label">Each cell in this 6&times;6 matrix shows the <b>Spearman
      rank correlation ($\\rho$)</b> between the full 397&times;397 distance matrices
      of two models. Spearman $\\rho$ measures how well the <i>rank ordering</i> of
      all 78,006 pairwise distances agrees between two models: it ranges from $-1$
      (the rankings are perfectly inverted) through $0$ (no systematic agreement)
      to $+1$ (perfect agreement in rankings). The six models are arranged with the
      3 WEIRD models occupying the first three rows and columns, and the 3 Sinic models
      occupying the last three, creating a visible 2&times;2 block structure.</p>
      <div id="plt_rsa_matrix"></div>
      <p style="font-size:0.78rem; color:#888; margin-top:6px;">
        <span style="color:var(--blue);">&block;</span> The blue rectangle highlights
        the 3&times;3 block of WEIRD-vs-WEIRD comparisons. &nbsp;
        <span style="color:var(--vermil);">&block;</span> The red rectangle highlights
        the 3&times;3 block of Sinic-vs-Sinic comparisons. &nbsp;
        All other cells are cross-tradition comparisons (one WEIRD model paired with
        one Sinic model).
        Colour scale: darker red = higher $\\rho$ (stronger agreement), darker blue =
        lower $\\rho$ (weaker agreement).
      </p>
    </div>
    <div>
      <div class="metrics" style="flex-direction:column;">
        <div class="metric blue">
          <div class="label">Within-WEIRD &rho;&macr;</div>
          <div class="value">{rho_w:.3f}</div>
        </div>
        <div class="metric vermil">
          <div class="label">Within-Sinic &rho;&macr;</div>
          <div class="value">{rho_s:.3f}</div>
        </div>
        <div class="metric green">
          <div class="label">Cross-tradition &rho;&macr;</div>
          <div class="value">{rho_x:.3f}</div>
        </div>
        <div class="metric">
          <div class="label">Tradition gap &Delta;&rho;</div>
          <div class="value" style="color:#333;">{drop:.3f}</div>
        </div>
      </div>
      <p style="font-size:0.82rem; color:#666; margin-top:12px; line-height:1.5;">
        These four values summarise the matrix on the left.
        <b>Within-WEIRD $\\bar{{\\rho}}$</b> is the mean of the 3 correlation
        values in the blue (WEIRD-vs-WEIRD) block.
        <b>Within-Sinic $\\bar{{\\rho}}$</b> is the mean of the 3 correlation
        values in the red (Sinic-vs-Sinic) block.
        <b>Cross-tradition $\\bar{{\\rho}}$</b> is the mean of the 9 off-block
        correlations (each pairing one WEIRD model with one Sinic model).
        <b>Tradition gap $\\Delta\\rho$</b> = Within-WEIRD $\\bar{{\\rho}}$ &minus;
        Cross-tradition $\\bar{{\\rho}}$.
      </p>
    </div>
  </div>

  <div class="card">
    <h2>Summary statistics</h2>
    <p>Intra-domain vs. inter-domain: rank-biserial $r$ ranges from
    $+0.23$ to $+0.30$ across the 3 WEIRD models (all $p &lt; 10^{{-100}}$).
    The <b>rank-biserial $r$</b> quantifies the probability that a randomly
    drawn distance from the first group (intra-domain) is smaller than a
    randomly drawn distance from the second group (inter-domain), rescaled to
    the range $[-1, +1]$. For example, a value of $r = +0.30$ means that in
    roughly 65% of such random draws, the intra-domain distance is the smaller
    of the two.</p>
    <p>RSA: within-tradition mean $\\bar{{\\rho}}_{{\\text{{WEIRD}}}} = {rho_w:.2f}$,
    $\\bar{{\\rho}}_{{\\text{{Sinic}}}} = {rho_s:.2f}$;
    cross-tradition mean $\\bar{{\\rho}}_{{\\text{{cross}}}} = {rho_x:.2f}$;
    gap $\\Delta\\rho = {drop:.2f}$.
    All 15 pairwise $\\rho$ values have $p = 0.0001$ (floor at $B = 10\\,000$
    permutations).</p>
  </div>

  <div class="card">
    <h2>Note on FreeLaw-EN</h2>
    <p>FreeLaw-EN (BAAI/bge-base-en, fine-tuned on legal corpora by the
    FreeLaw Project) is included in this experiment alongside two general-purpose
    English embedding models (BGE-EN-large and E5-large). It is the only model
    with a negative rank-biserial in the legal-vs-control test ($r = -0.17$,
    $p = 1.0$): legal-legal distances are on average <i>larger</i> than
    legal-control distances. The intra-vs-inter domain test, however, remains
    positive ($r = +0.30$), with the same sign and comparable magnitude as the
    other two WEIRD models.</p>
  </div>
</div>

<!-- ==================== DOMAIN SIGNAL ==================== -->
<div id="signal" class="panel">
  <div class="question">
    <b>Intra-domain vs. inter-domain distances.</b>
    Imagine laying out all 397 legal terms and drawing a line between every
    possible pair &mdash; that gives 78,606 unique pairs (the number of entries
    in the upper triangle of the 397&times;397 distance matrix). Each pair is
    then classified into one of two groups. A pair is <b>"intra-domain"</b> if
    both terms belong to the same branch of law (for example, "murder" and
    "manslaughter" are both criminal-law terms). A pair is <b>"inter-domain"</b>
    if the two terms belong to different branches (for example, "murder" from
    criminal law and "easement" from civil law). The question is whether the
    cosine distance &mdash; the numerical measure of how far apart the two
    terms are in the model's vector space &mdash; tends to be systematically
    different for these two groups.
  </div>

  <div class="card">
    <h2>Violin plots &mdash; intra vs. inter-domain</h2>
    <p>Each of the three panels below shows the full distribution of cosine
    distances for one WEIRD embedding model, split into intra-domain pairs
    (blue) and inter-domain pairs (orange). A <b>violin plot</b> combines two
    visual elements: the <i>outer shape</i> is a density estimate (a smoothed
    histogram) where wider sections indicate that more data points fall at that
    distance value, and the <i>inner rectangle</i> is a standard box plot showing
    the interquartile range (the middle 50% of the data) with a horizontal line
    at the median. Together, these allow the reader to see both the central
    tendency and the full shape of each distribution at a glance.</p>
    <p>The <b>annotation</b> at the top of each panel reports two quantities:
    the rank-biserial $r$ (explained below) and the median cosine distance for
    each of the two groups. These numerical summaries complement the visual
    impression provided by the violins.</p>
    <p>Rank-biserial $r = 1 - 2U/(n_x \\cdot n_y)$, where $U$ is the
    Mann-Whitney statistic. $r &gt; 0$: intra-domain distances tend to be
    smaller than inter-domain distances; $r &lt; 0$: intra-domain distances tend
    to be larger; $r = 0$: no systematic difference between the two groups.</p>
  </div>
  <div id="plt_intra_inter"></div>

  <div class="card" style="margin-top: 8px;">
    <h2>Violin plots &mdash; legal vs. control</h2>
    <p>This second set of violin plots compares two different groups of
    distances: <b>legal-legal</b> distances (blue), which are the cosine
    distances between pairs of legal terms (the same 78,606 pairs from the
    RDM), and <b>legal-control</b> distances (red), which are the cosine
    distances between each of the 397 legal terms and each of the 100
    Swadesh-100 control words (39,700 pairs). The control words &mdash;
    basic, culturally universal vocabulary such as "water", "fire", "hand",
    "sleep" &mdash; have no specifically legal content and serve as a
    non-legal baseline. The violin format is the same as above (outer shape =
    density estimate, inner rectangle = box plot with median line).
    $r &gt; 0$: legal-legal distances tend to be smaller than legal-control
    distances; $r &lt; 0$: legal-legal distances tend to be larger.</p>
  </div>
  <div id="plt_legal_control"></div>

  <div class="card" style="margin-top: 8px;">
    <h2>Summary statistics</h2>
    <table class="data">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="4" style="text-align:center; border-bottom:1px solid #ddd;">Intra vs. Inter-domain</th>
          <th colspan="4" style="text-align:center; border-bottom:1px solid #ddd;">Legal vs. Control</th>
        </tr>
        <tr>
          <th>Med. intra</th><th>Med. inter</th><th>r</th><th>p</th>
          <th>Med. legal</th><th>Med. control</th><th>r</th><th>p</th>
        </tr>
      </thead>
      <tbody>
        {signal_rows}
      </tbody>
    </table>
    <p style="font-size:0.8rem; color:#888; line-height:1.5;">
      The statistical test used is the <b>one-sided Mann-Whitney $U$</b> test.
      "One-sided" means that the test specifically asks whether distances in the
      first group (intra-domain, or legal-legal) tend to be <i>smaller</i> than
      distances in the second group (inter-domain, or legal-control), rather than
      simply asking whether the two groups differ in either direction.
      The effect size is the rank-biserial $r = 1 - 2U/(n_x n_y)$.
      The $p$-values are not bounded below and are reported at machine precision
      where applicable (i.e., values smaller than approximately $10^{{-300}}$
      are reported as such).
    </p>
  </div>
</div>

<!-- ==================== TOPOLOGY ==================== -->
<div id="topology" class="panel">
  <div class="question">
    <b>Domain topology.</b> This matrix provides a bird's-eye view of how
    each embedding model positions the 7 branches of law relative to each
    other. It is a $K \\times K$ matrix (where $K = 7$, one row and one
    column per branch of law). Each cell records the <b>average cosine
    distance</b> between all pairs of terms that span the two branches
    indicated by the row and column. For example, the cell at row "criminal"
    and column "procedure" averages over all cosine distances between every
    criminal-law term and every procedural-law term. Diagonal cells (e.g.,
    "criminal" vs. "criminal") show the average distance among terms
    <i>within</i> the same branch &mdash; a measure of how spread out that
    branch is in the model's vector space (computed from the upper triangle
    only, to avoid counting each pair twice).
  </div>
  <div class="card">
    <h2>Reading the heatmap</h2>
    <p>One heatmap is shown for each of the three WEIRD models, displayed
    side by side for direct visual comparison. The <b>colour scale</b> maps
    blue to lower mean distance and red to higher mean distance. Crucially,
    the scale is <b>fixed across all three panels</b> (ranging from 0.15 to
    0.70) so that colours are directly comparable from one model to the next:
    the same shade of blue or red means the same numerical distance in every
    panel. Each cell also displays its exact numerical value as a text label.
    Hover over any cell for additional detail.</p>
  </div>
  <div id="plt_topology"></div>
</div>

<!-- ==================== RSA ==================== -->
<div id="rsa" class="panel">
  <div class="question">
    <b>Representational Similarity Analysis (RSA).</b>
    RSA (Kriegeskorte et al. 2008) is a technique originally developed in
    computational neuroscience and adapted here for comparing legal-semantic
    spaces. The core idea is as follows: comparing individual term vectors
    directly across two different embedding models is impossible, because each
    model uses its own internal coordinate system (the axes of one model's
    vector space have no correspondence with those of another). RSA sidesteps
    this problem by comparing the <i>pattern of distances</i> instead. If two
    models both place "murder" close to "manslaughter" and far from "easement",
    their 397&times;397 distance matrices will be correlated &mdash; even though
    the actual vector coordinates may be entirely different. Specifically, for
    each of the 15 possible model pairs, the <b>Spearman rank correlation
    $\\rho$</b> is computed between the upper triangles of the two RDMs (78,006
    pairwise distance values each). The 15 pairs are partitioned into three
    groups: 3 within-WEIRD (comparing two English-trained models), 3
    within-Sinic (comparing two Chinese-trained models), and 9 cross-tradition
    (comparing one English-trained model with one Chinese-trained model).
  </div>

  <div class="card">
    <h2>Statistical methods</h2>
    <p><b>Significance: Mantel permutation test</b>
    ($B = {meta.get("n_perm", "?"):,}$ permutations). The Mantel test assesses
    whether the observed correlation between two distance matrices could have
    arisen purely by chance. It works by randomly shuffling the rows and columns
    of one matrix simultaneously (which destroys any true correspondence between
    the two matrices while preserving each matrix's internal structure) and then
    recomputing the Spearman correlation. This shuffling-and-recomputing step is
    repeated $B$ times (here $B = {meta.get("n_perm", "?"):,}$), producing a
    distribution of "null" correlations &mdash; i.e., the range of correlation
    values one would expect if the two matrices were unrelated. The $p$-value is
    the proportion of these null correlations that are equal to or larger than
    the actually observed $\\rho$. If $p$ is very small, the observed correlation
    is unlikely to be a coincidence. The $p$-value is bounded below by $1/B$
    (following Phipson &amp; Smyth 2010), meaning the smallest reportable value
    with $B = {meta.get("n_perm", "?"):,}$ is $0.0001$.</p>
    <p><b>Confidence intervals: block bootstrap</b>
    ($B = {meta.get("n_boot", "?"):,}$ resamples). The confidence interval
    quantifies the precision of the estimated $\\rho$. It is computed by
    repeatedly resampling the 397 legal terms <i>with replacement</i>: in each
    resample, some terms appear multiple times and others are omitted entirely.
    For each resample, the corresponding sub-matrices are extracted from the two
    RDMs and the Spearman $\\rho$ is recomputed. This "block" resampling strategy
    (rather than resampling individual distance pairs) respects the fact that
    distances involving the same term are not statistically independent of each
    other (Nili et al. 2014). The reported 95% confidence interval spans from
    the 2.5th to the 97.5th percentile of the $B$ resampled $\\rho$ values.</p>
  </div>

  <p class="plot-label"><b>Forest plot.</b> Each row represents one of the 15
  model pairs. The <b>dot</b> marks the observed Spearman $\\rho$ for that pair.
  The <b>horizontal bar</b> extending from each dot shows the 95% bootstrap
  confidence interval: the range of $\\rho$ values within which the true
  correlation plausibly falls, given the resampling variability described above.
  If two bars do not overlap, the difference between the corresponding $\\rho$
  values is robust to resampling. <b>Vertical dotted lines</b> mark the mean
  $\\rho$ for each group. Colour coding: <span style="color:var(--blue);">blue</span>
  = within-WEIRD pairs, <span style="color:var(--vermil);">red</span> =
  within-Sinic pairs, <span style="color:var(--green);">green</span> =
  cross-tradition pairs.</p>
  <div id="plt_forest"></div>

  <div class="metrics" style="margin-top: 12px;">
    <div class="metric blue">
      <div class="label">Within-WEIRD &rho;</div>
      <div class="value">{rho_w:.3f}</div>
    </div>
    <div class="metric vermil">
      <div class="label">Within-Sinic &rho;</div>
      <div class="value">{rho_s:.3f}</div>
    </div>
    <div class="metric green">
      <div class="label">Cross-tradition &rho;</div>
      <div class="value">{rho_x:.3f}</div>
    </div>
    <div class="metric">
      <div class="label">Tradition gap &Delta;&rho;</div>
      <div class="value" style="color:#333;">{drop:.3f}</div>
    </div>
  </div>

  <p class="plot-label"><b>Null distributions.</b> Each histogram shows the
  distribution of Spearman $\\rho$ values obtained by the Mantel permutation
  test for one representative model pair (the pair with the highest observed
  $\\rho$ in each group). The coloured bars represent the "null" correlations
  &mdash; i.e., the values of $\\rho$ that arise when rows and columns of one
  RDM are randomly shuffled, destroying any genuine correspondence between the
  two matrices. The <b>black vertical line</b> marks the actually observed
  $\\rho$. When the observed $\\rho$ falls far to the right of the null
  histogram, none (or almost none) of the random permutations produced a
  correlation as large, and the corresponding $p$-value is at or near the
  floor of $1/B$.</p>
  <div id="plt_null"></div>

  <div class="card" style="margin-top: 8px;">
    <h2>Full results table</h2>
    <p style="font-size:0.82rem; color:#666; line-height:1.5; margin-bottom:8px;">
      Column definitions:
      <b>$\\rho$</b> = Spearman rank correlation between the two models' RDMs.
      <b>95% CI</b> = bootstrap confidence interval (2.5th to 97.5th percentile
      of the block-bootstrap distribution).
      <b>$r^2$</b> = coefficient of determination, representing the proportion
      of variance in one RDM's distance rankings that is accounted for by the
      other RDM's distance rankings.
      <b>$p$</b> = Mantel permutation $p$-value (bounded below by $1/B$).
    </p>
    <table class="data">
      <thead>
        <tr><th>Group</th><th>Pair</th><th>&rho;</th><th>95% CI</th><th>r&sup2;</th><th>p</th></tr>
      </thead>
      <tbody>
        {rsa_rows}
      </tbody>
    </table>
  </div>


</div>

{_script}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_viz(results_dir: Path, results: dict) -> None:
    """Generate all Lens I figures (PNG + HTML). Called from lens1.main()."""
    png_dir = results_dir / "figures" / "png"
    html_dir = results_dir / "figures" / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    weird_labels: list[str] = results["meta"]["weird_models"]
    dist_dir = results_dir / "distances"
    generated: list[Path] = []

    print("\n[viz] Generating PNG figures...")

    if "section_311" in results:
        p = fig_311_domains(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        p = fig_311_confidence(results_dir, results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_31" in results:
        if any((dist_dir / f"{m}.npz").exists() for m in weird_labels):
            p = fig_311_intra_inter(results_dir, weird_labels, png_dir)
            generated.append(p)
            print(f"  {p.name}")

            p = fig_311_legal_control(results_dir, weird_labels, png_dir)
            generated.append(p)
            print(f"  {p.name}")

        p = fig_312_topology(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

    if "section_314" in results:
        p = fig_rsa_forest(results, png_dir)
        generated.append(p)
        print(f"  {p.name}")

        ddist = results_dir / "distributions"
        if ddist.exists() and any(ddist.iterdir()):
            p = fig_rsa_null(results_dir, results, png_dir)
            generated.append(p)
            print(f"  {p.name}")

    print("[viz] Generating interactive HTML...")
    html_path = build_html(results_dir, results, html_dir / "lens1_interactive.html")
    print(f"  {html_path.name}")

    print(f"[viz] Done — {len(generated)} PNG + 1 HTML")
