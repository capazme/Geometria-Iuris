"""
run_single_experiment.py — Run a single experiment from the CLS pipeline.

Usage:
    .venv/bin/python run_single_experiment.py embeddings   # Pre-load all embeddings
    .venv/bin/python run_single_experiment.py 1             # RSA + Mantel
    .venv/bin/python run_single_experiment.py 2             # Gromov-Wasserstein
    .venv/bin/python run_single_experiment.py 3             # Axiological Axes
    .venv/bin/python run_single_experiment.py 4             # Clustering
    .venv/bin/python run_single_experiment.py 5             # NDA
    .venv/bin/python run_single_experiment.py html          # Generate HTML visualization
"""

import json
import logging
import sys
import time

import numpy as np

from src.core.config_loader import load_config
from src.core.device import DeviceManager
from src.embeddings.client import EmbeddingClient


def setup():
    """Load config, dataset, and embedding client."""
    config = load_config()

    # Set logging to INFO for our code only
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("cls_pipeline").setLevel(logging.INFO)
    logging.getLogger("src").setLevel(logging.INFO)

    # Load dataset
    data_path = config.get_absolute_path("processed") / "legal_terms.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize client
    device_manager = DeviceManager(config.device.preferred)
    device_info = device_manager.detect()
    print(f"Device: {device_info.device_name} ({device_info.device_type})")

    client = EmbeddingClient(config, device_manager)

    return config, data, client


def load_embeddings(data, client):
    """Pre-load all embeddings needed by all experiments."""
    core_terms = data["core_terms"]
    background_terms = data["background_terms"]
    all_terms = core_terms + background_terms

    en_core = [t["en"] for t in core_terms]
    zh_core = [t["zh"] for t in core_terms]
    en_all = [t["en"] for t in all_terms]
    zh_all = [t["zh"] for t in all_terms]

    print(f"\nLoading embeddings for {len(en_core)} core + {len(background_terms)} background = {len(en_all)} terms")

    t0 = time.time()
    emb_weird_core = client.get_embeddings(en_core, model_type="weird")
    print(f"  WEIRD core: {emb_weird_core.shape}")

    emb_weird_all = client.get_embeddings(en_all, model_type="weird")
    print(f"  WEIRD all:  {emb_weird_all.shape}")

    emb_sinic_core = client.get_embeddings(zh_core, model_type="sinic")
    print(f"  Sinic core: {emb_sinic_core.shape}")

    emb_sinic_all = client.get_embeddings(zh_all, model_type="sinic")
    print(f"  Sinic all:  {emb_sinic_all.shape}")

    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s (cached embeddings are instant)")

    return {
        "en_core": en_core,
        "zh_core": zh_core,
        "en_all": en_all,
        "zh_all": zh_all,
        "emb_weird_core": emb_weird_core,
        "emb_sinic_core": emb_sinic_core,
        "emb_weird_all": emb_weird_all,
        "emb_sinic_all": emb_sinic_all,
    }


def run_exp1(config, data, client, emb):
    """Experiment 1: RSA + Mantel Test."""
    from src.experiments.exp_rsa import run_rsa, plot_rdm_heatmaps

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: RSA + Mantel Test")
    print("=" * 60)

    n_perm = config.experiments.rsa.n_permutations
    print(f"  Permutations: {n_perm}")

    t0 = time.time()
    result = run_rsa(
        emb["emb_weird_core"], emb["emb_sinic_core"],
        labels=emb["en_core"],
        n_permutations=n_perm,
        seed=config.random_seed,
    )
    dt = time.time() - t0

    print(f"\n  RESULTS ({dt:.1f}s):")
    print(f"  Spearman r  = {result.spearman_r:.6f}")
    print(f"  p-value     = {result.p_value:.6f}")
    print(f"  N pairs     = {result.n_pairs}")
    print(f"  N terms     = {len(result.labels)}")

    if result.p_value < 0.05:
        print(f"  --> SIGNIFICANT: RDMs are correlated (structures partly preserved)")
    else:
        print(f"  --> NOT SIGNIFICANT: No evidence of structural preservation")

    # Plot
    plots_dir = config.get_absolute_path("plots")
    plot_rdm_heatmaps(result, output_dir=plots_dir, dpi=150)
    print(f"  Plot saved to {plots_dir}/rsa_rdm_heatmaps.png")

    return result


def run_exp2(config, data, client, emb):
    """Experiment 2: Gromov-Wasserstein."""
    from src.experiments.exp_gw import gromov_wasserstein_distance

    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Gromov-Wasserstein Distance")
    print("=" * 60)

    gw_cfg = config.experiments.gromov_wasserstein
    print(f"  Permutations: {gw_cfg.n_permutations}")
    print(f"  Entropic reg: {gw_cfg.entropic_reg}")

    t0 = time.time()
    result = gromov_wasserstein_distance(
        emb["emb_weird_core"], emb["emb_sinic_core"],
        entropic_reg=gw_cfg.entropic_reg,
        use_sinkhorn=gw_cfg.use_sinkhorn,
        n_permutations=gw_cfg.n_permutations,
        seed=config.random_seed,
    )
    dt = time.time() - t0

    print(f"\n  RESULTS ({dt:.1f}s):")
    print(f"  GW distance = {result.distance:.6f}")
    print(f"  p-value     = {result.p_value:.6f}")
    print(f"  Transport plan: {result.transport_plan.shape}")

    if result.distance > 0.1:
        print(f"  --> HIGH ANISOMORPHISM (distance > 0.1)")
    else:
        print(f"  --> RELATIVE ISOMORPHISM (distance <= 0.1)")

    return result


def run_exp3(config, data, client, emb):
    """Experiment 3: Axiological Axes."""
    from src.experiments.exp_axes import run_axes_experiment, plot_axes_comparison

    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Axiological Axis Projection (Kozlowski)")
    print("=" * 60)

    axes_cfg = config.experiments.axes
    value_axes = data["value_axes"]
    print(f"  Axes: {list(value_axes.keys())}")
    print(f"  Bootstrap: {axes_cfg.n_bootstrap}")

    def embed_weird(texts):
        return client.get_embeddings(texts, model_type="weird")
    def embed_sinic(texts):
        return client.get_embeddings(texts, model_type="sinic")

    t0 = time.time()
    result = run_axes_experiment(
        emb["emb_weird_core"], emb["emb_sinic_core"],
        labels=emb["en_core"],
        value_axes=value_axes,
        embed_fn_weird=embed_weird,
        embed_fn_sinic=embed_sinic,
        n_bootstrap=axes_cfg.n_bootstrap,
        seed=config.random_seed,
    )
    dt = time.time() - t0

    print(f"\n  RESULTS ({dt:.1f}s):")
    for ax in result.axes:
        ci = ax.bootstrap_ci
        ci_str = f"  CI=[{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]" if ci else ""
        print(f"  {ax.axis_name}:")
        print(f"    Spearman rho = {ax.spearman_r:.4f}, p = {ax.spearman_p:.4f}{ci_str}")

        # Top 5 most divergent terms
        deltas = {l: ax.weird_scores[l] - ax.sinic_scores[l] for l in emb["en_core"]}
        sorted_d = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"    Top divergent: ", end="")
        print(", ".join(f"{t}({d:+.3f})" for t, d in sorted_d[:5]))

    plots_dir = config.get_absolute_path("plots")
    plot_axes_comparison(result, output_dir=plots_dir, dpi=150)
    print(f"  Plot saved to {plots_dir}/axes_projection_comparison.png")

    return result


def run_exp4(config, data, client, emb):
    """Experiment 4: Hierarchical Clustering."""
    from src.experiments.exp_clustering import run_clustering_experiment, plot_dendrograms

    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Hierarchical Clustering + Fowlkes-Mallows")
    print("=" * 60)

    clust_cfg = config.experiments.clustering
    print(f"  Method: {clust_cfg.method}")
    print(f"  k values: {clust_cfg.k_values}")
    print(f"  Permutations: {clust_cfg.n_permutations}")

    t0 = time.time()
    result = run_clustering_experiment(
        emb["emb_weird_core"], emb["emb_sinic_core"],
        labels=emb["en_core"],
        method=clust_cfg.method,
        k_values=clust_cfg.k_values,
        n_permutations=clust_cfg.n_permutations,
        seed=config.random_seed,
    )
    dt = time.time() - t0

    print(f"\n  RESULTS ({dt:.1f}s):")
    for fm in result.fm_results:
        interp = "SIMILAR" if fm.fm_index >= 0.5 else "DIVERGENT"
        print(f"  k={fm.k:2d}: FM={fm.fm_index:.4f}, p={fm.p_value:.4f} --> {interp}")

    plots_dir = config.get_absolute_path("plots")
    plot_dendrograms(result, output_dir=plots_dir, figsize=(18, 8), dpi=150)
    print(f"  Plot saved to {plots_dir}/clustering_dendrograms.png")

    return result


def run_exp5(config, data, client, emb):
    """Experiment 5: Neighborhood Divergence Analysis."""
    from src.experiments.exp_nda import run_nda_part_a, run_nda_part_b, NDAExperimentResult, plot_nda_results

    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: Neighborhood Divergence Analysis (NDA)")
    print("=" * 60)

    nda_cfg = config.experiments.nda
    print(f"  k: {nda_cfg.k}")
    print(f"  Permutations: {nda_cfg.n_permutations}")

    # Part A
    print("\n  --- Part A: k-NN Neighborhood Comparison ---")
    t0 = time.time()
    part_a = run_nda_part_a(
        emb["emb_weird_core"], emb["emb_sinic_core"],
        core_labels=emb["en_core"],
        emb_weird_all=emb["emb_weird_all"],
        emb_sinic_all=emb["emb_sinic_all"],
        all_labels=emb["en_all"],
        k=nda_cfg.k,
        n_permutations=nda_cfg.n_permutations,
        seed=config.random_seed,
    )
    dt_a = time.time() - t0

    print(f"\n  Part A RESULTS ({dt_a:.1f}s):")
    print(f"  Mean Jaccard = {part_a.mean_jaccard:.4f}")
    print(f"  p-value      = {part_a.p_value:.4f}")

    print(f"\n  Top 10 'False Friends' (lowest Jaccard):")
    sorted_terms = sorted(part_a.term_results, key=lambda r: r.jaccard)
    for r in sorted_terms[:10]:
        print(f"    {r.label:<25} J={r.jaccard:.3f}  shared={r.shared_neighbors}")

    print(f"\n  Top 5 'True Cognates' (highest Jaccard):")
    for r in sorted_terms[-5:]:
        print(f"    {r.label:<25} J={r.jaccard:.3f}  shared={r.shared_neighbors}")

    # Part B
    print("\n  --- Part B: Normative Decompositions ---")
    decompositions = data["normative_decompositions"]

    def embed_weird(texts):
        return client.get_embeddings(texts, model_type="weird")
    def embed_sinic(texts):
        return client.get_embeddings(texts, model_type="sinic")

    t0 = time.time()
    part_b = run_nda_part_b(
        decompositions,
        embed_fn_weird=embed_weird,
        embed_fn_sinic=embed_sinic,
        corpus_weird=emb["emb_weird_all"],
        corpus_sinic=emb["emb_sinic_all"],
        corpus_labels=emb["en_all"],
        k=nda_cfg.k,
    )
    dt_b = time.time() - t0

    print(f"\n  Part B RESULTS ({dt_b:.1f}s):")
    for d in part_b.decompositions:
        print(f"\n  {d.en_formula} | {d.zh_formula}")
        print(f"    Question: {d.jurisprudential_question[:80]}")
        print(f"    Jaccard: {d.jaccard:.3f}")
        print(f"    WEIRD neighbors: {', '.join(l for l, _ in d.weird_neighbors[:7])}")
        print(f"    Sinic neighbors: {', '.join(l for l, _ in d.sinic_neighbors[:7])}")

    nda_result = NDAExperimentResult(part_a=part_a, part_b=part_b)
    plots_dir = config.get_absolute_path("plots")
    plot_nda_results(nda_result, output_dir=plots_dir, dpi=150)
    print(f"\n  Plot saved to {plots_dir}/nda_analysis.png")

    return nda_result


def run_html(config, data, client, emb):
    """Generate HTML visualization from existing results.json."""
    from src.visualization.generate_html import generate_html

    print("\n" + "=" * 60)
    print("  GENERATING HTML VISUALIZATION")
    print("=" * 60)

    output_dir = config.get_absolute_path("output")
    results_path = output_dir / config.output.results_file
    html_path = output_dir / config.output.visualization_file

    if not results_path.exists():
        print(f"  ERROR: Results file not found: {results_path}")
        print("  Run the full pipeline first: python -m src.cli run")
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    generate_html(results, html_path)
    print(f"\n  HTML visualization saved to: {html_path}")
    return html_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_experiment.py <embeddings|1|2|3|4|5|html>")
        return 1

    cmd = sys.argv[1]

    # HTML generation doesn't need embeddings
    if cmd == "html":
        config = load_config()
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        run_html(config, None, None, None)
        return 0

    config, data, client = setup()
    emb = load_embeddings(data, client)

    if cmd == "embeddings":
        print("\nAll embeddings pre-loaded and cached. Run experiments with: python run_single_experiment.py 1")
        client.unload_models()
        return 0

    runners = {
        "1": run_exp1,
        "2": run_exp2,
        "3": run_exp3,
        "4": run_exp4,
        "5": run_exp5,
    }

    if cmd not in runners:
        print(f"Unknown experiment: {cmd}")
        return 1

    runners[cmd](config, data, client, emb)
    client.unload_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())
