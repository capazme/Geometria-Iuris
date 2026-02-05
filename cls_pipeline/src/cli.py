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
    n_ctrl = len(data.get("control_terms", []))
    n_total = n_core + n_bg + n_ctrl
    logger.info("Dataset loaded: %d core + %d background + %d control = %d terms",
                n_core, n_bg, n_ctrl, n_total)
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
    control_terms = data.get("control_terms", [])
    all_terms = core_terms + background_terms + control_terms
    value_axes = data["value_axes"]
    decompositions = data["normative_decompositions"]

    # Extract text lists
    en_core = [t["en"] for t in core_terms]
    zh_core = [t["zh"] for t in core_terms]
    en_all = [t["en"] for t in all_terms]
    zh_all = [t["zh"] for t in all_terms]
    domains_core = [t.get("domain", "unknown") for t in core_terms]

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

    # Generate plots from results
    if config.output.save_plots:
        generate_plots(config)

    # Generate HTML visualization
    if generate_html:
        html_path = generate_visualization(config)
        print(f"  Visualization: {html_path}")

    print()

    # Cleanup
    client.unload_models()

    return results_path


def generate_visualization(config: Config, light_mode: bool = False) -> Path:
    """Generate interactive HTML visualization from results.json."""
    # Usa il nuovo builder interattivo con fallback al vecchio
    try:
        from .visualization.interactive.html_builder import build_html_report
        use_new_builder = True
    except ImportError:
        from .visualization.generate_html import generate_html
        use_new_builder = False

    output_dir = config.get_absolute_path("output")
    results_path = output_dir / config.output.results_file
    html_path = output_dir / config.output.visualization_file

    if not results_path.exists():
        logger.warning("Results file not found: %s — skipping visualization", results_path)
        return html_path

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if use_new_builder:
        build_html_report(results, html_path, light_mode=light_mode)
    else:
        generate_html(results, html_path)

    logger.info("Visualization generated: %s", html_path)
    return html_path


def generate_plots(config: Config) -> None:
    """
    Generate all plots from results.json.

    This function reads the saved results and regenerates all plots,
    allowing visualization tweaks without re-running the full pipeline.
    """
    from .experiments.exp_rsa import RSAResult
    from .experiments.exp_axes import AxesExperimentResult, AxesComparisonResult
    from .experiments.statistical import BootstrapCIResult
    from .experiments.exp_clustering import ClusteringResult, ClusteringExperimentResult, FMResult
    from .experiments.exp_nda import (
        NDAExperimentResult, NDAPartAResult, NDAPartBResult,
        TermNeighborhoodResult, DecompositionResult,
    )
    from .experiments.module_c_umap import UMAPResult

    output_dir = config.get_absolute_path("output")
    results_path = output_dir / config.output.results_file
    plots_dir = config.get_absolute_path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        logger.warning("Results file not found: %s — cannot generate plots", results_path)
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    experiments = results.get("experiments", {})
    logger.info("Generating plots from %s", results_path)

    # Load dataset for domain info
    data = load_dataset(config)
    domains_core = [t.get("domain", "unknown") for t in data["core_terms"]]

    # ─── RSA Heatmaps ─────────────────────────────────────────────────────
    if "1_rsa" in experiments:
        rsa_data = experiments["1_rsa"]
        rsa_result = RSAResult(
            rdm_weird=np.array(rsa_data["rdm_weird"]),
            rdm_sinic=np.array(rsa_data["rdm_sinic"]),
            spearman_r=rsa_data["spearman_r"],
            p_value=rsa_data["p_value"],
            n_permutations=rsa_data["n_permutations"],
            n_pairs=rsa_data["n_pairs"],
            labels=rsa_data["labels"],
        )
        plot_rdm_heatmaps(
            rsa_result, output_dir=plots_dir,
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
            domains=domains_core,
        )
        logger.info("RSA plots generated")

    # ─── Axes Comparison ──────────────────────────────────────────────────
    if "3_axes" in experiments:
        axes_data = experiments["3_axes"]
        axes_list = []
        for ax in axes_data["axes"]:
            ci = None
            if ax.get("bootstrap_ci"):
                ci = BootstrapCIResult(
                    estimate=ax["bootstrap_ci"].get("estimate", ax["spearman_r"]),
                    ci_lower=ax["bootstrap_ci"]["ci_lower"],
                    ci_upper=ax["bootstrap_ci"]["ci_upper"],
                    n_bootstrap=ax["bootstrap_ci"]["n_bootstrap"],
                    alpha=ax["bootstrap_ci"].get("alpha", 0.05),
                )
            axes_list.append(AxesComparisonResult(
                axis_name=ax["axis_name"],
                weird_scores=ax.get("weird_scores", {}),
                sinic_scores=ax.get("sinic_scores", {}),
                spearman_r=ax["spearman_r"],
                spearman_p=ax["spearman_p"],
                bootstrap_ci=ci,
            ))
        axes_result = AxesExperimentResult(
            axes=axes_list,
            labels=list(axes_list[0].weird_scores.keys()) if axes_list else [],
        )
        plot_axes_comparison(
            axes_result, output_dir=plots_dir,
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
        )
        logger.info("Axes plots generated")

    # ─── Clustering Dendrograms ───────────────────────────────────────────
    if "4_clustering" in experiments:
        clust_data = experiments["4_clustering"]
        fm_results = [
            FMResult(
                k=fm["k"],
                fm_index=fm["fm_index"],
                p_value=fm["p_value"],
                n_permutations=fm.get("n_permutations", 5000),
            )
            for fm in clust_data["fm_results"]
        ]
        clustering_result = ClusteringExperimentResult(
            clustering_weird=ClusteringResult(
                linkage_matrix=np.array(clust_data["linkage_weird"]),
                labels=clust_data["labels"],
            ),
            clustering_sinic=ClusteringResult(
                linkage_matrix=np.array(clust_data["linkage_sinic"]),
                labels=clust_data["labels"],
            ),
            fm_results=fm_results,
            labels=clust_data["labels"],
        )
        plot_dendrograms(
            clustering_result, output_dir=plots_dir,
            figsize=(18, 8),
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
        )
        logger.info("Clustering plots generated")

    # ─── NDA Results ──────────────────────────────────────────────────────
    if "5_nda" in experiments:
        nda_data = experiments["5_nda"]

        # Part A — la struttura JSON usa "part_a_neighborhoods"
        part_a_data = nda_data.get("part_a_neighborhoods", nda_data.get("part_a", {}))
        term_results = [
            TermNeighborhoodResult(
                term_id=tr.get("term", tr.get("label", "")),
                label=tr.get("term", tr.get("label", "")),
                jaccard=tr["jaccard"],
                weird_neighbors=tr["weird_neighbors"],
                sinic_neighbors=tr["sinic_neighbors"],
                shared_neighbors=tr.get("shared_neighbors", []),
            )
            for tr in part_a_data.get("per_term", [])
        ]
        part_a = NDAPartAResult(
            term_results=term_results,
            mean_jaccard=part_a_data["mean_jaccard"],
            p_value=part_a_data["p_value"],
            n_permutations=part_a_data.get("n_permutations", 5000),
            k=part_a_data["k"],
        )

        # Part B — la struttura JSON usa "part_b_decompositions"
        part_b_data = nda_data.get("part_b_decompositions", nda_data.get("part_b", {}))
        decompositions = [
            DecompositionResult(
                decomposition_id=d.get("id", ""),
                operation=d.get("operation", "subtraction"),
                en_formula=d["en_formula"],
                zh_formula=d["zh_formula"],
                jurisprudential_question=d.get("jurisprudential_question", ""),
                jaccard=d["jaccard"],
                weird_neighbors=[(n["label"], n.get("cosine_distance", 0)) for n in d["weird_neighbors"]],
                sinic_neighbors=[(n["label"], n.get("cosine_distance", 0)) for n in d["sinic_neighbors"]],
            )
            for d in part_b_data.get("decompositions", [])
        ]
        part_b = NDAPartBResult(decompositions=decompositions)

        nda_result = NDAExperimentResult(part_a=part_a, part_b=part_b)
        plot_nda_results(
            nda_result, output_dir=plots_dir,
            dpi=config.output.plot_dpi,
        )
        logger.info("NDA plots generated")

    # ─── UMAP Visualization ───────────────────────────────────────────────
    if "supplementary_umap" in experiments:
        umap_data = experiments["supplementary_umap"]
        coords = umap_data["coordinates"]

        weird_coords = coords["weird"]
        sinic_coords = coords["sinic"]

        n_weird = len(weird_coords)
        n_sinic = len(sinic_coords)

        coords_2d = np.zeros((n_weird + n_sinic, 2))
        term_labels = []
        model_labels = np.zeros(n_weird + n_sinic, dtype=int)

        for i, pt in enumerate(weird_coords):
            coords_2d[i, 0] = pt["x"]
            coords_2d[i, 1] = pt["y"]
            term_labels.append(pt["label"])
            model_labels[i] = 0  # WEIRD

        for i, pt in enumerate(sinic_coords):
            coords_2d[n_weird + i, 0] = pt["x"]
            coords_2d[n_weird + i, 1] = pt["y"]
            term_labels.append(pt["label"])
            model_labels[n_weird + i] = 1  # Sinic

        umap_result = UMAPResult(
            coords_2d=coords_2d,
            model_labels=model_labels,
            term_labels=term_labels,
        )
        plot_umap(
            umap_result,
            output_dir=plots_dir,
            figsize=config.output.plot_figsize,
            dpi=config.output.plot_dpi,
            weird_label=config.models["weird"].label,
            sinic_label=config.models["sinic"].label,
        )
        logger.info("UMAP plot generated")

    print(f"\nPlots saved to: {plots_dir}")


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


def cmd_plots(args: argparse.Namespace) -> int:
    """Execute 'plots' subcommand — regenerate plots from results.json."""
    config = load_config(args.config)
    setup_logging(config)

    try:
        generate_plots_v2(
            config,
            experiments=getattr(args, "exp", None),
            formats=getattr(args, "format", ["png"]),
        )
        return 0
    except Exception as e:
        logger.exception("Plot generation failed: %s", e)
        return 1


def cmd_html(args: argparse.Namespace) -> int:
    """Execute 'html' subcommand — regenerate HTML visualization."""
    config = load_config(args.config)
    setup_logging(config)

    try:
        light_mode = getattr(args, "light", False)
        html_path = generate_visualization(config, light_mode=light_mode)
        print(f"HTML visualization generated: {html_path}")
        return 0
    except Exception as e:
        logger.exception("HTML generation failed: %s", e)
        return 1


def generate_plots_v2(
    config: Config,
    experiments: list[str] | None = None,
    formats: list[str] | None = None,
) -> None:
    """
    Generate publication-ready plots using the new visualization system.

    Parameters
    ----------
    config : Config
        Pipeline configuration.
    experiments : list[str], optional
        List of experiments to generate (e.g., ["rsa", "gw"]).
        If None, generates all.
    formats : list[str], optional
        Output formats (default: ["png"]).
    """
    if formats is None:
        formats = ["png"]

    output_dir = config.get_absolute_path("output")
    results_path = output_dir / config.output.results_file
    plots_dir = config.get_absolute_path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        logger.warning("Results file not found: %s — cannot generate plots", results_path)
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    exp_data = results.get("experiments", {})
    dpi = config.output.plot_dpi

    # Load dataset for domain info
    try:
        data = load_dataset(config)
        domains = [t.get("domain", "unknown") for t in data["core_terms"]]
    except Exception:
        domains = None

    all_experiments = ["rsa", "gw", "axes", "clustering", "nda", "umap"]
    if experiments is None:
        experiments = all_experiments

    # Import new visualization modules
    try:
        from .visualization.png import (
            plot_clustered_heatmap,
            plot_inter_domain_matrix,
            plot_rdm_correlation,
            plot_transport_histogram,
            plot_top_alignments,
            plot_axes_scatter_ci,
            plot_forest_plot,
            plot_truncated_dendrogram,
            plot_fm_chart,
            plot_jaccard_histogram,
            plot_false_friends_network,
            plot_decomposition_comparison,
            plot_umap_smart_labels,
        )
        use_new_plots = True
    except ImportError as e:
        logger.warning("New visualization module not available: %s", e)
        logger.info("Falling back to legacy plot generation")
        generate_plots(config)
        return

    # ─── RSA ─────────────────────────────────────────────────────────────
    if "rsa" in experiments and "1_rsa" in exp_data:
        rsa = exp_data["1_rsa"]
        rdm_w = np.array(rsa["rdm_weird"])
        rdm_s = np.array(rsa["rdm_sinic"])
        labels = rsa.get("labels", [])

        plot_clustered_heatmap(
            rdm_w, rdm_s, labels,
            rsa["spearman_r"], rsa["p_value"],
            plots_dir, domains=domains, dpi=dpi, formats=formats,
        )

        if domains:
            plot_inter_domain_matrix(
                rdm_w, rdm_s, domains,
                plots_dir, dpi=dpi, formats=formats,
            )

        plot_rdm_correlation(
            rdm_w, rdm_s,
            rsa["spearman_r"], rsa["p_value"],
            plots_dir, dpi=dpi, formats=formats,
        )
        logger.info("RSA plots generated (new)")

    # ─── GW ──────────────────────────────────────────────────────────────
    if "gw" in experiments and "2_gromov_wasserstein" in exp_data:
        gw = exp_data["2_gromov_wasserstein"]
        tp = np.array(gw["transport_plan"])
        labels = exp_data.get("1_rsa", {}).get("labels", [])

        plot_transport_histogram(
            tp, gw["distance"], gw["p_value"],
            plots_dir, dpi=dpi, formats=formats,
        )

        if labels:
            plot_top_alignments(
                tp, labels, plots_dir,
                dpi=dpi, formats=formats,
            )
        logger.info("GW plots generated (new)")

    # ─── Axes ────────────────────────────────────────────────────────────
    if "axes" in experiments and "3_axes" in exp_data:
        axes = exp_data["3_axes"]

        # Forest plot
        axes_for_forest = [
            {
                "axis_name": ax["axis_name"],
                "spearman_r": ax["spearman_r"],
                "spearman_p": ax.get("spearman_p", 1),
                "ci_lower": ax.get("bootstrap_ci", {}).get("ci_lower"),
                "ci_upper": ax.get("bootstrap_ci", {}).get("ci_upper"),
            }
            for ax in axes.get("axes", [])
        ]
        plot_forest_plot(axes_for_forest, plots_dir, dpi=dpi, formats=formats)

        # Individual scatter plots
        for ax in axes.get("axes", []):
            ci = ax.get("bootstrap_ci", {})
            plot_axes_scatter_ci(
                ax.get("weird_scores", {}),
                ax.get("sinic_scores", {}),
                ax["axis_name"],
                ax["spearman_r"],
                ax.get("spearman_p", 1),
                ci.get("ci_lower"),
                ci.get("ci_upper"),
                plots_dir, dpi=dpi, formats=formats,
            )
        logger.info("Axes plots generated (new)")

    # ─── Clustering ──────────────────────────────────────────────────────
    if "clustering" in experiments and "4_clustering" in exp_data:
        clust = exp_data["4_clustering"]

        plot_truncated_dendrogram(
            np.array(clust["linkage_weird"]),
            np.array(clust["linkage_sinic"]),
            clust.get("labels", []),
            clust.get("fm_results", []),
            plots_dir, dpi=dpi, formats=formats,
        )

        plot_fm_chart(
            clust.get("fm_results", []),
            plots_dir, dpi=dpi, formats=formats,
        )
        logger.info("Clustering plots generated (new)")

    # ─── NDA ─────────────────────────────────────────────────────────────
    if "nda" in experiments and "5_nda" in exp_data:
        nda = exp_data["5_nda"]
        part_a = nda.get("part_a_neighborhoods", {})
        part_b = nda.get("part_b_decompositions", {})

        if part_a.get("per_term"):
            plot_jaccard_histogram(
                part_a["per_term"],
                part_a["mean_jaccard"],
                part_a["p_value"],
                part_a["k"],
                plots_dir, dpi=dpi, formats=formats,
            )

            plot_false_friends_network(
                part_a["per_term"],
                plots_dir, dpi=dpi, formats=formats,
            )

        if part_b.get("decompositions"):
            plot_decomposition_comparison(
                part_b["decompositions"],
                plots_dir, dpi=dpi, formats=formats,
            )
        logger.info("NDA plots generated (new)")

    # ─── UMAP ────────────────────────────────────────────────────────────
    if "umap" in experiments and "supplementary_umap" in exp_data:
        umap_data = exp_data["supplementary_umap"]
        coords = umap_data.get("coordinates", {})

        weird_pts = coords.get("weird", [])
        sinic_pts = coords.get("sinic", [])

        if weird_pts and sinic_pts:
            labels = [p["label"] for p in weird_pts]
            coords_w = np.array([[p["x"], p["y"]] for p in weird_pts])
            coords_s = np.array([[p["x"], p["y"]] for p in sinic_pts])

            plot_umap_smart_labels(
                coords_w, coords_s, labels,
                plots_dir, domains=domains, dpi=dpi, formats=formats,
            )
        logger.info("UMAP plots generated (new)")

    print(f"\nPlots saved to: {plots_dir}")


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

    # plots command
    plots_parser = subparsers.add_parser("plots", help="Regenerate plots from results.json")
    plots_parser.add_argument(
        "--exp",
        nargs="+",
        choices=["rsa", "gw", "axes", "clustering", "nda", "umap"],
        default=None,
        help="Experiments to plot (default: all)",
    )
    plots_parser.add_argument(
        "--format",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=["png"],
        help="Output formats (default: png)",
    )
    plots_parser.set_defaults(func=cmd_plots)

    # html command
    html_parser = subparsers.add_parser("html", help="Regenerate HTML visualization")
    html_parser.add_argument(
        "--light",
        action="store_true",
        help="Light mode: omit heavy raw data (RDM, transport plan) for smaller file",
    )
    html_parser.set_defaults(func=cmd_html)

    args = parser.parse_args()

    if args.command is None:
        args.command = "run"
        args.no_html = False
        args.func = cmd_run

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
