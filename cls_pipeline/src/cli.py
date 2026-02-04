"""
cli.py — Command-line interface for CLS Pipeline v2.0.

Entry point with subcommands: run, info, clear-cache.
Orchestrates the 5-experiment pipeline and generates structured output.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .core.config_loader import load_config, Config
from .core.device import DeviceManager
from .core.hashing import compute_config_hash, compute_file_hash
from .core.output_manager import OutputManager
from .embeddings.client import EmbeddingClient
from .experiments import (
    # Experiment 1: RSA
    run_rsa,
    plot_rdm_heatmaps,
    # Experiment 2: GW
    gromov_wasserstein_distance,
    # Experiment 3: Axes
    run_axes_experiment,
    plot_axes_comparison,
    # Experiment 4: Clustering
    run_clustering_experiment,
    plot_dendrograms,
    # Experiment 5: NDA
    run_nda_part_a,
    run_nda_part_b,
    NDAExperimentResult,
    plot_nda_results,
    # UMAP (supplementary)
    umap_reduce,
    plot_umap,
)

logger = logging.getLogger("cls_pipeline")


def setup_logging(config: Config) -> None:
    """Configure logging based on config."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_dataset(config: Config) -> dict:
    """Load the structured legal terms dataset."""
    data_path = config.get_absolute_path("processed") / "legal_terms.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Run 'python -m src.data.build_dataset' first."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_core = len(data["core_terms"])
    n_bg = len(data["background_terms"])
    logger.info("Dataset loaded: %d core + %d background = %d terms", n_core, n_bg, n_core + n_bg)
    return data


def run_pipeline(config: Config, generate_html: bool = True) -> Path:
    """
    Execute the full CLS pipeline (5 experiments + UMAP visualization).

    Parameters
    ----------
    config : Config
        Pipeline configuration.
    generate_html : bool
        Whether to generate HTML visualization.

    Returns
    -------
    Path
        Path to results.json file.
    """
    separator = "=" * 70

    logger.info(separator)
    logger.info("  CLS Pipeline v%s - Cross-Lingual Semantics Analysis", config.version)
    logger.info(separator)

    # Load dataset
    data = load_dataset(config)
    core_terms = data["core_terms"]
    background_terms = data["background_terms"]
    all_terms = core_terms + background_terms
    value_axes = data["value_axes"]
    decompositions = data["normative_decompositions"]

    # Extract text lists
    en_core = [t["en"] for t in core_terms]
    zh_core = [t["zh"] for t in core_terms]
    en_all = [t["en"] for t in all_terms]
    zh_all = [t["zh"] for t in all_terms]

    # Compute hashes for reproducibility
    data_path = config.get_absolute_path("processed") / "legal_terms.json"
    input_data_hash = compute_file_hash(data_path)
    config_hash = compute_config_hash(config)

    logger.info("Input data hash: %s", input_data_hash[:16])
    logger.info("Config hash: %s", config_hash[:16])

    # Initialize device manager and embedding client
    device_manager = DeviceManager(config.device.preferred)
    device_info = device_manager.detect()
    logger.info("Device: %s (%s)", device_info.device_name, device_info.device_type)

    # Initialize output manager
    output_manager = OutputManager(config, input_data_hash, config_hash)

    # Initialize embedding client
    client = EmbeddingClient(config, device_manager)

    # Helper functions for embedding (used by axes and NDA)
    def embed_weird(texts: list[str]) -> np.ndarray:
        return client.get_embeddings(texts, model_type="weird")

    def embed_sinic(texts: list[str]) -> np.ndarray:
        return client.get_embeddings(texts, model_type="sinic")

    # ==================================================================
    # Phase 1: Extract embeddings
    # ==================================================================
    logger.info(separator)
    logger.info("  PHASE 1: Embedding Extraction")
    logger.info(separator)

    # Core term embeddings
    emb_weird_core = embed_weird(en_core)
    emb_sinic_core = embed_sinic(zh_core)

    logger.info("Core embeddings - WEIRD: %s, Sinic: %s", emb_weird_core.shape, emb_sinic_core.shape)

    # Full corpus embeddings (core + background)
    emb_weird_all = embed_weird(en_all)
    emb_sinic_all = embed_sinic(zh_all)

    logger.info("Full corpus - WEIRD: %s, Sinic: %s", emb_weird_all.shape, emb_sinic_all.shape)

    # ==================================================================
    # Experiment 1: RSA + Mantel Test
    # ==================================================================
    logger.info(separator)
    logger.info("  EXPERIMENT 1: RSA + Mantel Test")
    logger.info(separator)

    rsa_config = config.experiments.rsa
    rsa_result = run_rsa(
        emb_weird_core, emb_sinic_core,
        labels=en_core,
        n_permutations=rsa_config.n_permutations,
        seed=config.random_seed,
    )

    print(f"\n{'─' * 50}")
    print(f"  RSA: Spearman r = {rsa_result.spearman_r:.4f}")
    print(f"  Mantel p-value  = {rsa_result.p_value:.4f}")
    print(f"  N pairs         = {rsa_result.n_pairs}")
    print(f"{'─' * 50}\n")

    output_manager.set_rsa_result(rsa_result)

    if config.output.save_plots:
        plots_dir = config.get_absolute_path("plots")
        plot_rdm_heatmaps(
            rsa_result, output_dir=plots_dir,
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
        )

    # ==================================================================
    # Experiment 2: Gromov-Wasserstein Distance
    # ==================================================================
    logger.info(separator)
    logger.info("  EXPERIMENT 2: Gromov-Wasserstein Distance")
    logger.info(separator)

    gw_config = config.experiments.gromov_wasserstein
    gw_result = gromov_wasserstein_distance(
        emb_weird_core, emb_sinic_core,
        entropic_reg=gw_config.entropic_reg,
        use_sinkhorn=gw_config.use_sinkhorn,
        n_permutations=gw_config.n_permutations,
        seed=config.random_seed,
    )

    print(f"\n{'─' * 50}")
    print(f"  GW Distance:    {gw_result.distance:.6f}")
    print(f"  p-value:        {gw_result.p_value:.4f}")
    print(f"  Interpretation: {'High anisomorphism' if gw_result.distance > 0.1 else 'Relative isomorphism'}")
    print(f"{'─' * 50}\n")

    output_manager.set_gw_result(gw_result)

    # ==================================================================
    # Experiment 3: Axiological Axis Projection
    # ==================================================================
    logger.info(separator)
    logger.info("  EXPERIMENT 3: Axiological Axis Projection (Kozlowski)")
    logger.info(separator)

    axes_config = config.experiments.axes
    axes_result = run_axes_experiment(
        emb_weird_core, emb_sinic_core,
        labels=en_core,
        value_axes=value_axes,
        embed_fn_weird=embed_weird,
        embed_fn_sinic=embed_sinic,
        n_bootstrap=axes_config.n_bootstrap,
        seed=config.random_seed,
    )

    print(f"\n{'─' * 50}")
    print("  Axiological Axes Results:")
    for ax in axes_result.axes:
        ci_str = ""
        if ax.bootstrap_ci:
            ci_str = f" CI=[{ax.bootstrap_ci.ci_lower:.3f}, {ax.bootstrap_ci.ci_upper:.3f}]"
        print(f"  {ax.axis_name}: rho={ax.spearman_r:.4f}, p={ax.spearman_p:.4f}{ci_str}")
    print(f"{'─' * 50}\n")

    output_manager.set_axes_result(axes_result)

    if config.output.save_plots:
        plots_dir = config.get_absolute_path("plots")
        plot_axes_comparison(
            axes_result, output_dir=plots_dir,
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
        )

    # ==================================================================
    # Experiment 4: Hierarchical Clustering
    # ==================================================================
    logger.info(separator)
    logger.info("  EXPERIMENT 4: Hierarchical Clustering + Fowlkes-Mallows")
    logger.info(separator)

    clust_config = config.experiments.clustering
    clustering_result = run_clustering_experiment(
        emb_weird_core, emb_sinic_core,
        labels=en_core,
        method=clust_config.method,
        k_values=clust_config.k_values,
        n_permutations=clust_config.n_permutations,
        seed=config.random_seed,
    )

    print(f"\n{'─' * 50}")
    print("  Fowlkes-Mallows Index (multi-k):")
    for fm in clustering_result.fm_results:
        interp = "Similar" if fm.fm_index >= 0.5 else "Divergent"
        print(f"  k={fm.k:2d}: FM={fm.fm_index:.4f}, p={fm.p_value:.4f} ({interp})")
    print(f"{'─' * 50}\n")

    output_manager.set_clustering_result(clustering_result)

    if config.output.save_plots:
        plots_dir = config.get_absolute_path("plots")
        plot_dendrograms(
            clustering_result, output_dir=plots_dir,
            figsize=(18, 8),
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
        )

    # ==================================================================
    # Experiment 5: Neighborhood Divergence Analysis
    # ==================================================================
    logger.info(separator)
    logger.info("  EXPERIMENT 5: Neighborhood Divergence Analysis (NDA)")
    logger.info(separator)

    nda_config = config.experiments.nda

    # Part A: Neighborhood comparison
    logger.info("  Part A: k-NN Neighborhood Comparison")
    nda_part_a = run_nda_part_a(
        emb_weird_core, emb_sinic_core,
        core_labels=en_core,
        emb_weird_all=emb_weird_all,
        emb_sinic_all=emb_sinic_all,
        all_labels=en_all,
        k=nda_config.k,
        n_permutations=nda_config.n_permutations,
        seed=config.random_seed,
    )

    print(f"\n{'─' * 50}")
    print(f"  NDA Part A: Mean Jaccard = {nda_part_a.mean_jaccard:.4f}")
    print(f"  p-value = {nda_part_a.p_value:.4f}")
    print(f"\n  Top 5 'False Friends' (lowest Jaccard):")
    sorted_terms = sorted(nda_part_a.term_results, key=lambda r: r.jaccard)
    for r in sorted_terms[:5]:
        print(f"    {r.label:<25} Jaccard={r.jaccard:.3f}")
    print(f"{'─' * 50}\n")

    # Part B: Normative decompositions
    logger.info("  Part B: Normative Decompositions")
    nda_part_b = run_nda_part_b(
        decompositions,
        embed_fn_weird=embed_weird,
        embed_fn_sinic=embed_sinic,
        corpus_weird=emb_weird_all,
        corpus_sinic=emb_sinic_all,
        corpus_labels=en_all,
        k=nda_config.k,
    )

    print(f"\n{'─' * 50}")
    print("  NDA Part B: Normative Decompositions:")
    for d in nda_part_b.decompositions:
        print(f"  {d.en_formula:<25} Jaccard={d.jaccard:.3f}")
        print(f"    WEIRD neighbors: {', '.join(l for l, _ in d.weird_neighbors[:5])}")
        print(f"    Sinic neighbors: {', '.join(l for l, _ in d.sinic_neighbors[:5])}")
    print(f"{'─' * 50}\n")

    nda_result = NDAExperimentResult(part_a=nda_part_a, part_b=nda_part_b)
    output_manager.set_nda_result(nda_result)

    if config.output.save_plots:
        plots_dir = config.get_absolute_path("plots")
        plot_nda_results(
            nda_result, output_dir=plots_dir,
            dpi=config.output.plot_dpi,
        )

    # ==================================================================
    # Supplementary: UMAP Visualization
    # ==================================================================
    logger.info(separator)
    logger.info("  SUPPLEMENTARY: UMAP Visualization")
    logger.info(separator)

    umap_config = config.experiments.umap
    umap_result = umap_reduce(
        emb_weird_core, emb_sinic_core,
        en_core, en_core,
        n_neighbors=umap_config.n_neighbors,
        min_dist=umap_config.min_dist,
        metric=umap_config.metric,
        random_state=config.random_seed,
    )

    n_weird = emb_weird_core.shape[0]
    coords_weird = umap_result.coords_2d[:n_weird]
    coords_sinic = umap_result.coords_2d[n_weird:]

    output_manager.set_umap_result(coords_weird, coords_sinic, en_core, en_core)

    if config.output.save_plots:
        plots_dir = config.get_absolute_path("plots")
        plot_umap(
            umap_result,
            output_dir=plots_dir,
            figsize=config.output.plot_figsize,
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
        )

    # ==================================================================
    # Save results
    # ==================================================================
    logger.info(separator)
    logger.info("  PIPELINE COMPLETED")
    logger.info(separator)

    results_path = output_manager.save()

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  1. RSA:        r={rsa_result.spearman_r:.4f}, p={rsa_result.p_value:.4f}")
    print(f"  2. GW:         dist={gw_result.distance:.6f}, p={gw_result.p_value:.4f}")
    for ax in axes_result.axes:
        print(f"  3. Axis ({ax.axis_name[:15]}): rho={ax.spearman_r:.4f}")
    for fm in clustering_result.fm_results:
        print(f"  4. Clustering (k={fm.k}): FM={fm.fm_index:.4f}, p={fm.p_value:.4f}")
    print(f"  5. NDA:        mean_Jaccard={nda_part_a.mean_jaccard:.4f}, p={nda_part_a.p_value:.4f}")
    print(f"{'=' * 60}")
    print(f"\n  Results saved to: {results_path}")

    # Generate HTML visualization
    if generate_html:
        html_path = generate_visualization(config)
        print(f"  Visualization: {html_path}")

    print()

    # Cleanup
    client.unload_models()

    return results_path


def generate_visualization(config: Config) -> Path:
    """Generate HTML visualization from results.json."""
    output_dir = config.get_absolute_path("output")
    html_path = output_dir / config.output.visualization_file

    template_path = Path(__file__).parent.parent / "output" / "visualization.html"

    if template_path.exists():
        import shutil
        shutil.copy(template_path, html_path)

    logger.info("Visualization generated: %s", html_path)
    return html_path


def cmd_run(args: argparse.Namespace) -> int:
    """Execute 'run' subcommand."""
    config = load_config(args.config)
    setup_logging(config)

    try:
        run_pipeline(config, generate_html=not args.no_html)
        return 0
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute 'info' subcommand."""
    config = load_config(args.config)

    print(f"\nCLS Pipeline v{config.version}")
    print("=" * 40)
    print(f"\nModels:")
    for name, model in config.models.items():
        print(f"  {name}: {model.name}")
        print(f"         dimension: {model.dimension}")
        print(f"         language: {model.language}")

    print(f"\nDevice: {config.device.preferred}")
    print(f"Batch size: {config.device.batch_size}")
    print(f"Random seed: {config.random_seed}")

    print(f"\nExperiments:")
    print(f"  1. RSA:        {config.experiments.rsa.n_permutations} permutations")
    print(f"  2. GW:         reg={config.experiments.gromov_wasserstein.entropic_reg}, {config.experiments.gromov_wasserstein.n_permutations} perm")
    print(f"  3. Axes:       {config.experiments.axes.n_bootstrap} bootstrap resamples")
    print(f"  4. Clustering: k={config.experiments.clustering.k_values}, {config.experiments.clustering.n_permutations} perm")
    print(f"  5. NDA:        k={config.experiments.nda.k}, {config.experiments.nda.n_permutations} perm")

    print(f"\nPaths:")
    print(f"  Data:      {config.get_absolute_path('data')}")
    print(f"  Processed: {config.get_absolute_path('processed')}")
    print(f"  Cache:     {config.get_absolute_path('cache')}")
    print(f"  Output:    {config.get_absolute_path('output')}")
    print(f"  Models:    {config.get_absolute_path('models')}")

    from .embeddings.cache import EmbeddingCache
    cache = EmbeddingCache(config.get_absolute_path("cache"))
    stats = cache.stats()
    print(f"\nCache: {stats['count']} files ({stats['total_size_mb']} MB)")

    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    """Execute 'clear-cache' subcommand."""
    config = load_config(args.config)

    from .embeddings.cache import EmbeddingCache
    cache = EmbeddingCache(config.get_absolute_path("cache"))

    count = cache.clear()
    print(f"Cleared {count} cached embeddings")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="cls-pipeline",
        description="CLS Pipeline - Cross-Lingual Semantics Analysis",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML visualization generation",
    )
    run_parser.set_defaults(func=cmd_run)

    # info command
    info_parser = subparsers.add_parser("info", help="Show pipeline information")
    info_parser.set_defaults(func=cmd_info)

    # clear-cache command
    cache_parser = subparsers.add_parser("clear-cache", help="Clear embedding cache")
    cache_parser.set_defaults(func=cmd_clear_cache)

    args = parser.parse_args()

    if args.command is None:
        args.command = "run"
        args.no_html = False
        args.func = cmd_run

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
