"""
Apples-to-apples false-friends polysemy test.

Both the query and the neighbour pool are encoded with the same family
of contextualised templates. The contextualised pool is built once by
`build_contextualized_pool.py` (one mean-of-eight-templates vector per
term per model, ~19 minutes wall time on MPS) and stored in
`data/processed/embeddings_contextualized/`. This script consumes that
pool and:

  Phase 2 — Lens V baseline reproduction.
    Recomputes the published §3.2.1 within-WEIRD, within-Sinic, and
    cross-tradition mean Jaccard over all 397 core terms on the
    contextualised pool, then compares to the published bare-term
    baselines (≈ 0.258 / 0.327 / 0.088). A close reproduction means
    the contextualised pool is a valid alternative representation;
    a radical disagreement would mean the contextualisation has broken
    the relational structure and any false-friend test on it is moot.

  Phase 3 — apples-to-apples false-friends test.
    Recomputes per-pair Jaccard for the published top-12 false friends
    on the contextualised pool, where the query and the pool share
    the same templating. Compares to the bare-term baseline.

Output
------
`results/false_friends_polysemy.json` and a static HTML dashboard at
`results/figures/html/false_friends_polysemy.html`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.embeddings import load_precomputed  # noqa: E402
from shared.html_style import (  # noqa: E402
    CSS, HEAD_LINKS, C_BLUE, C_VERMIL, C_GREEN, C_ORANGE, page_head,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_NN = 15

WEIRD_LABELS = ["BGE-EN-large", "E5-large", "FreeLaw-EN"]
SINIC_LABELS = ["BGE-ZH-large", "Text2vec-large-ZH", "Dmeta-ZH"]
ALL_LABELS = WEIRD_LABELS + SINIC_LABELS

EMB_BARE = REPO_ROOT / "data" / "processed" / "embeddings"
EMB_CTX = REPO_ROOT / "data" / "processed" / "embeddings_contextualized"
LENS5_RESULTS = REPO_ROOT / "lens_5_neighborhoods" / "results" / "lens5_results.json"


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def load_pool(label: str, root: Path) -> tuple[np.ndarray, list[dict]]:
    """Load (vectors, index) from a precomputed pool directory."""
    return load_precomputed(label, root)


def core_indices(index: list[dict]) -> list[int]:
    """Positional indices of the 397 core terms in the 9472-term pool."""
    return [i for i, t in enumerate(index) if t["tier"] == "core" and t["domain"]]


# ---------------------------------------------------------------------------
# k-NN computation (matches lens5._compute_knn)
# ---------------------------------------------------------------------------

def compute_knn_for_indices(
    pool_vecs: np.ndarray,
    query_indices: list[int],
    k: int = K_NN,
) -> np.ndarray:
    """
    For each query position in the pool, return the top-k pool indices
    closest in cosine sense, with self-similarity masked.
    Returns shape (n_query, k).
    """
    query_vecs = pool_vecs[query_indices]
    sims = query_vecs @ pool_vecs.T
    qarr = np.array(query_indices)
    for i, qi in enumerate(qarr):
        sims[i, qi] = -np.inf
    top_k = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
    return top_k.astype(np.int64)


def per_term_jaccard(knn_a: np.ndarray, knn_b: np.ndarray) -> np.ndarray:
    """Per-term Jaccard between two k-NN arrays of shape (n, k)."""
    n = len(knn_a)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sa = set(int(x) for x in knn_a[i])
        sb = set(int(x) for x in knn_b[i])
        u = len(sa | sb)
        out[i] = (len(sa & sb) / u) if u else 0.0
    return out


# ---------------------------------------------------------------------------
# Phase 2 — baseline reproduction
# ---------------------------------------------------------------------------

def reproduce_lens5_baseline(pools: dict[str, np.ndarray], core_idx: list[int]) -> dict:
    """
    Recompute the §3.2.1 mean Jaccard summary on a given set of model pools.

    Returns dict with within_weird, within_sinic, cross, and per-pair detail.
    """
    knn_by_model = {
        label: compute_knn_for_indices(pools[label], core_idx, K_NN)
        for label in ALL_LABELS
    }

    within_weird_pairs = []
    for i in range(len(WEIRD_LABELS)):
        for j in range(i + 1, len(WEIRD_LABELS)):
            la, lb = WEIRD_LABELS[i], WEIRD_LABELS[j]
            j_arr = per_term_jaccard(knn_by_model[la], knn_by_model[lb])
            within_weird_pairs.append({"a": la, "b": lb, "mean_jaccard": float(j_arr.mean())})

    within_sinic_pairs = []
    for i in range(len(SINIC_LABELS)):
        for j in range(i + 1, len(SINIC_LABELS)):
            la, lb = SINIC_LABELS[i], SINIC_LABELS[j]
            j_arr = per_term_jaccard(knn_by_model[la], knn_by_model[lb])
            within_sinic_pairs.append({"a": la, "b": lb, "mean_jaccard": float(j_arr.mean())})

    cross_pairs = []
    for la in WEIRD_LABELS:
        for lb in SINIC_LABELS:
            j_arr = per_term_jaccard(knn_by_model[la], knn_by_model[lb])
            cross_pairs.append({"a": la, "b": lb, "mean_jaccard": float(j_arr.mean())})

    return {
        "mean_within_weird_jaccard": float(np.mean([p["mean_jaccard"] for p in within_weird_pairs])),
        "mean_within_sinic_jaccard": float(np.mean([p["mean_jaccard"] for p in within_sinic_pairs])),
        "mean_cross_jaccard": float(np.mean([p["mean_jaccard"] for p in cross_pairs])),
        "within_weird_pairs": within_weird_pairs,
        "within_sinic_pairs": within_sinic_pairs,
        "cross_pairs": cross_pairs,
        "knn_by_model": knn_by_model,
    }


# ---------------------------------------------------------------------------
# Phase 3 — false friends test
# ---------------------------------------------------------------------------

def load_top_false_friends(n: int = 12) -> list[dict]:
    data = json.loads(LENS5_RESULTS.read_text(encoding="utf-8"))
    top = data["section_322"]["top_20"][:n]
    return [
        {
            "rank": r["rank"],
            "en": r["en"],
            "zh": r["zh"],
            "domain": r["domain"],
            "published_mean_jaccard": float(r.get("mean_jaccard", 0.0)),
        }
        for r in top
    ]


def compute_false_friends_jaccard(
    pools: dict[str, np.ndarray],
    core_idx: list[int],
    index: list[dict],
    false_friends: list[dict],
) -> list[dict]:
    """
    For each false friend, compute its k=15 neighbours in each model's pool
    and aggregate per-pair Jaccard across the 9 cross-tradition combinations.
    """
    # Position of each false friend in the 9472-term pool
    by_en = {t["en"].strip(): i for i, t in enumerate(index)}

    knn_per_term: dict[int, dict[str, np.ndarray]] = {}
    for f in false_friends:
        pool_pos = by_en.get(f["en"].strip(), -1)
        f["pool_index"] = pool_pos
        knn_per_term[f["rank"]] = {}
        for label in ALL_LABELS:
            if pool_pos < 0:
                knn_per_term[f["rank"]][label] = np.array([], dtype=np.int64)
                continue
            qv = pools[label][pool_pos]
            sims = pools[label] @ qv
            sims[pool_pos] = -np.inf
            top_k = np.argpartition(-sims, K_NN)[:K_NN]
            top_k = top_k[np.argsort(-sims[top_k])]
            knn_per_term[f["rank"]][label] = top_k.astype(np.int64)

    results: list[dict] = []
    for f in false_friends:
        knn = knn_per_term[f["rank"]]
        cross_pairs = []
        for wl in WEIRD_LABELS:
            for sl in SINIC_LABELS:
                a, b = knn[wl], knn[sl]
                if a.size == 0 or b.size == 0:
                    j = float("nan")
                else:
                    sa = set(int(x) for x in a)
                    sb = set(int(x) for x in b)
                    u = len(sa | sb)
                    j = (len(sa & sb) / u) if u else 0.0
                cross_pairs.append({"weird": wl, "sinic": sl, "jaccard": j})

        within_weird = []
        for i in range(len(WEIRD_LABELS)):
            for j2 in range(i + 1, len(WEIRD_LABELS)):
                la, lb = WEIRD_LABELS[i], WEIRD_LABELS[j2]
                a, b = knn[la], knn[lb]
                if a.size == 0 or b.size == 0:
                    jj = float("nan")
                else:
                    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
                    jj = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
                within_weird.append({"a": la, "b": lb, "jaccard": jj})
        within_sinic = []
        for i in range(len(SINIC_LABELS)):
            for j2 in range(i + 1, len(SINIC_LABELS)):
                la, lb = SINIC_LABELS[i], SINIC_LABELS[j2]
                a, b = knn[la], knn[lb]
                if a.size == 0 or b.size == 0:
                    jj = float("nan")
                else:
                    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
                    jj = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
                within_sinic.append({"a": la, "b": lb, "jaccard": jj})

        cross_vals = [p["jaccard"] for p in cross_pairs if not np.isnan(p["jaccard"])]
        results.append({
            "rank": f["rank"],
            "en": f["en"],
            "zh": f["zh"],
            "domain": f["domain"],
            "published_mean_jaccard": f["published_mean_jaccard"],
            "pool_index": f["pool_index"],
            "mean_cross_jaccard": float(np.mean(cross_vals)) if cross_vals else float("nan"),
            "max_cross_jaccard": float(np.max(cross_vals)) if cross_vals else float("nan"),
            "mean_within_weird_jaccard": float(
                np.mean([p["jaccard"] for p in within_weird if not np.isnan(p["jaccard"])])
            ),
            "mean_within_sinic_jaccard": float(
                np.mean([p["jaccard"] for p in within_sinic if not np.isnan(p["jaccard"])])
            ),
            "cross_pairs": cross_pairs,
            "within_weird_pairs": within_weird,
            "within_sinic_pairs": within_sinic,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[ff] Loading bare-term and contextualised pools …")
    bare_pools: dict[str, np.ndarray] = {}
    ctx_pools: dict[str, np.ndarray] = {}
    index = None
    for label in ALL_LABELS:
        v_bare, idx = load_pool(label, EMB_BARE)
        bare_pools[label] = v_bare
        v_ctx, _ = load_pool(label, EMB_CTX)
        ctx_pools[label] = v_ctx
        if index is None:
            index = idx
        print(f"  {label}: bare={v_bare.shape}  ctx={v_ctx.shape}")

    assert index is not None
    core_idx = core_indices(index)
    print(f"\n[ff] {len(index)} pool terms, {len(core_idx)} core terms")

    # ---- Phase 2: baseline reproduction --------------------------------
    print("\n[ff] Phase 2: recomputing §3.2.1 baselines on bare and ctx pools")
    bare_baseline = reproduce_lens5_baseline(bare_pools, core_idx)
    ctx_baseline = reproduce_lens5_baseline(ctx_pools, core_idx)

    print(
        f"\n  bare-term pool baseline (sanity check, should match published):\n"
        f"    within-WEIRD J̄ = {bare_baseline['mean_within_weird_jaccard']:.4f} "
        f"(published: 0.258)\n"
        f"    within-Sinic J̄ = {bare_baseline['mean_within_sinic_jaccard']:.4f} "
        f"(published: 0.327)\n"
        f"    cross         J̄ = {bare_baseline['mean_cross_jaccard']:.4f} "
        f"(published: 0.088)\n"
    )
    print(
        f"  contextualised pool baseline:\n"
        f"    within-WEIRD J̄ = {ctx_baseline['mean_within_weird_jaccard']:.4f}\n"
        f"    within-Sinic J̄ = {ctx_baseline['mean_within_sinic_jaccard']:.4f}\n"
        f"    cross         J̄ = {ctx_baseline['mean_cross_jaccard']:.4f}\n"
    )

    # ---- Phase 3: false friends apples-to-apples ----------------------
    print("[ff] Phase 3: false friends Jaccard on both pools")
    false_friends = load_top_false_friends(12)

    bare_results = compute_false_friends_jaccard(
        bare_pools, core_idx, index, [dict(f) for f in false_friends]
    )
    ctx_results = compute_false_friends_jaccard(
        ctx_pools, core_idx, index, [dict(f) for f in false_friends]
    )

    print(
        f"\n  {'rank':>4}  {'term (en)':25s}  {'pub J̄':>8}  "
        f"{'bare-pool J̄':>13}  {'ctx-pool J̄':>13}  {'Δ':>7}"
    )
    print("  " + "-" * 78)
    for b, c in zip(bare_results, ctx_results):
        delta = c["mean_cross_jaccard"] - b["mean_cross_jaccard"]
        print(
            f"  {b['rank']:>4}  {b['en']:25s}  {b['published_mean_jaccard']:8.3f}  "
            f"{b['mean_cross_jaccard']:13.4f}  {c['mean_cross_jaccard']:13.4f}  "
            f"{delta:+7.4f}"
        )

    bare_mean = float(np.mean([r["mean_cross_jaccard"] for r in bare_results]))
    ctx_mean = float(np.mean([r["mean_cross_jaccard"] for r in ctx_results]))
    bare_max = float(np.max([r["max_cross_jaccard"] for r in bare_results]))
    ctx_max = float(np.max([r["max_cross_jaccard"] for r in ctx_results]))
    n_nonzero_ctx = sum(1 for r in ctx_results if r["mean_cross_jaccard"] > 0)

    print(
        f"\n  Aggregated over 12 false friends:\n"
        f"    bare-pool   mean cross J̄ = {bare_mean:.4f}, max single pair = {bare_max:.4f}\n"
        f"    ctx-pool    mean cross J̄ = {ctx_mean:.4f}, max single pair = {ctx_max:.4f}\n"
        f"    ctx-pool nonzero terms = {n_nonzero_ctx}/12"
    )

    report = {
        "meta": {
            "module": "lens_5_neighborhoods/false_friends_polysemy.py",
            "thesis_section": "§3.2 (apples-to-apples polysemy validation)",
            "date": datetime.now().isoformat(timespec="seconds"),
            "weird_models": WEIRD_LABELS,
            "sinic_models": SINIC_LABELS,
            "k_nn": K_NN,
            "n_false_friends": len(false_friends),
            "bare_pool": str(EMB_BARE.relative_to(REPO_ROOT)),
            "ctx_pool": str(EMB_CTX.relative_to(REPO_ROOT)),
            "ctx_aggregator": "mean of 8 templated variants per term",
            "n_terms_pool": len(index),
            "n_core_terms": len(core_idx),
        },
        "phase_2_baselines": {
            "bare_pool": {k: v for k, v in bare_baseline.items() if k != "knn_by_model"},
            "ctx_pool": {k: v for k, v in ctx_baseline.items() if k != "knn_by_model"},
            "published_reference": {
                "within_weird": 0.258,
                "within_sinic": 0.327,
                "cross": 0.088,
            },
        },
        "phase_3_false_friends": {
            "bare_pool_results": bare_results,
            "ctx_pool_results": ctx_results,
            "summary": {
                "bare_mean_cross_jaccard": bare_mean,
                "ctx_mean_cross_jaccard": ctx_mean,
                "bare_max_cross_jaccard": bare_max,
                "ctx_max_cross_jaccard": ctx_max,
                "n_terms_nonzero_in_ctx": n_nonzero_ctx,
                "n_terms_total": 12,
            },
        },
    }

    out_dir = Path(__file__).parent / "results"
    out_path = out_dir / "false_friends_polysemy.json"
    out_path.write_text(
        json.dumps(_nan_to_none(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[ff] Report → {out_path}")

    _render_html(report)


def _nan_to_none(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def _render_html(report: dict) -> None:
    out_dir = Path(__file__).parent / "results" / "figures" / "html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "false_friends_polysemy.html"

    head = page_head("Lens V — False Friends Polysemy (apples-to-apples)")

    bare_b = report["phase_2_baselines"]["bare_pool"]
    ctx_b = report["phase_2_baselines"]["ctx_pool"]
    pub = report["phase_2_baselines"]["published_reference"]
    s = report["phase_3_false_friends"]["summary"]

    # Sanity check: did the bare-pool baseline reproduce the published numbers?
    bare_match = (
        abs(bare_b["mean_within_weird_jaccard"] - pub["within_weird"]) < 0.05
        and abs(bare_b["mean_within_sinic_jaccard"] - pub["within_sinic"]) < 0.05
        and abs(bare_b["mean_cross_jaccard"] - pub["cross"]) < 0.05
    )
    sanity_class = "finding" if bare_match else "warning"

    # ctx vs bare false friends comparison
    bare_results = report["phase_3_false_friends"]["bare_pool_results"]
    ctx_results = report["phase_3_false_friends"]["ctx_pool_results"]

    rows = []
    for b, c in zip(bare_results, ctx_results):
        delta = c["mean_cross_jaccard"] - b["mean_cross_jaccard"]
        delta_color = ""
        if delta > 0.02:
            delta_color = ' style="color:#0072B2;font-weight:600;"'
        elif delta > 0:
            delta_color = ' style="color:#0072B2;"'
        elif delta < 0:
            delta_color = ' style="color:#D55E00;"'
        rows.append(
            f"<tr>"
            f"<td><b>{b['rank']}</b></td>"
            f"<td><code>{b['en']}</code></td>"
            f"<td><code>{b['zh']}</code></td>"
            f"<td>{b['domain']}</td>"
            f"<td>{b['published_mean_jaccard']:.3f}</td>"
            f"<td>{b['mean_cross_jaccard']:.4f}</td>"
            f"<td>{c['mean_cross_jaccard']:.4f}</td>"
            f"<td{delta_color}>{delta:+.4f}</td>"
            f"</tr>"
        )
    table = (
        '<table class="data">'
        "<thead><tr><th>#</th><th>EN</th><th>ZH</th><th>Domain</th>"
        "<th>Published J̄</th><th>bare-pool cross J̄</th>"
        "<th>ctx-pool cross J̄</th><th>Δ (ctx − bare)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    body = f"""<body>
<h1>Lens V &mdash; False Friends Polysemy (apples-to-apples)</h1>
<p class="subtitle">Apples-to-apples polysemy test on the published top-12
false friends. Both query and neighbour pool are encoded with the same
mean-of-eight-templates contextualisation.</p>

<div class="metrics">
  <div class="metric blue">
    <div class="value">{bare_b['mean_within_weird_jaccard']:.3f}</div>
    <div class="label">bare W (published 0.258)</div>
  </div>
  <div class="metric vermil">
    <div class="value">{bare_b['mean_within_sinic_jaccard']:.3f}</div>
    <div class="label">bare S (published 0.327)</div>
  </div>
  <div class="metric green">
    <div class="value">{bare_b['mean_cross_jaccard']:.3f}</div>
    <div class="label">bare cross (published 0.088)</div>
  </div>
  <div class="metric orange">
    <div class="value">{ctx_b['mean_cross_jaccard']:.3f}</div>
    <div class="label">ctx-pool cross J̄ (397 core)</div>
  </div>
</div>

<div class="card">
  <h2>Phase 2 — Baseline reproduction sanity check</h2>
  <div class="{sanity_class}">
    <b>Bare-pool baseline {'reproduced' if bare_match else 'DRIFTED FROM'} the published numbers.</b>
    The §3.2.1 within-WEIRD / within-Sinic / cross J̄ on the bare-term pool
    of 9472 terms come out as
    <code>{bare_b['mean_within_weird_jaccard']:.4f}</code> /
    <code>{bare_b['mean_within_sinic_jaccard']:.4f}</code> /
    <code>{bare_b['mean_cross_jaccard']:.4f}</code>,
    against published <code>{pub['within_weird']:.3f}</code> /
    <code>{pub['within_sinic']:.3f}</code> /
    <code>{pub['cross']:.3f}</code>. {'A clean reproduction confirms the pipeline computes the same neighbourhood Jaccard as the published §3.2.1.' if bare_match else 'A drift would mean the pool / index / k=15 lookup logic does not match the published Lens V code.'}
  </div>
  <h3>Contextualised pool baseline</h3>
  <p>
    On the contextualised pool, the same §3.2.1 computation gives
    <code>{ctx_b['mean_within_weird_jaccard']:.4f}</code> /
    <code>{ctx_b['mean_within_sinic_jaccard']:.4f}</code> /
    <code>{ctx_b['mean_cross_jaccard']:.4f}</code>. If the contextualised
    cross J̄ is in the same ballpark as the bare-term cross J̄, the
    contextualisation is a meaningful alternative representation that
    preserves relational structure; if it is radically higher or lower,
    the contextualisation has shifted the geometry to a degree that
    changes the §3.2 finding itself.
  </p>
</div>

<div class="card">
  <h2>Phase 3 — Apples-to-apples false friends (per-term)</h2>
  {table}
  <p class="note-sm">
    "bare-pool cross J̄" reproduces the published Lens V baseline for the
    top 12 false friends (each cell = mean over 9 W × S pairs of k=15
    Jaccard). "ctx-pool cross J̄" computes the same quantity on the
    contextualised pool, where each term is the mean of its 8 templated
    variants. "Δ (ctx − bare)" is the lift the contextualisation
    produces. Positive Δ values are blue; negative are vermilion.
  </p>
</div>

<div class="card">
  <h2>Aggregated over 12 false friends</h2>
  <table class="data">
    <thead><tr><th>Pool</th><th>mean cross J̄</th><th>max single pair</th><th>nonzero / 12</th></tr></thead>
    <tbody>
      <tr><td><b>bare</b></td>
          <td>{s['bare_mean_cross_jaccard']:.4f}</td>
          <td>{s['bare_max_cross_jaccard']:.4f}</td>
          <td>—</td></tr>
      <tr><td><b>contextualised</b></td>
          <td>{s['ctx_mean_cross_jaccard']:.4f}</td>
          <td>{s['ctx_max_cross_jaccard']:.4f}</td>
          <td>{s['n_terms_nonzero_in_ctx']} / 12</td></tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h2>Reading</h2>
  <p style="font-size:0.85rem;">
    With both query and pool contextualised, any non-zero cross J̄ here
    represents a genuine cross-tradition overlap of *contextualised
    representations*, not a cross-form artefact. This is the primary
    polysemy check for §3.2.2.
  </p>
</div>
</body>"""

    html = f"<!DOCTYPE html>\n<html lang='en'>\n{head}\n{body}\n</html>"
    out_path.write_text(html, encoding="utf-8")
    print(f"[ff] Dashboard → {out_path}")


if __name__ == "__main__":
    main()
