"""
config_loader.py — YAML configuration parser with environment variable support.

Loads configuration from config.yaml and .env files, providing a unified
Config object for the entire pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for a single embedding model."""
    name: str
    dimension: int
    label: str
    language: str
    prefix: str = ""


@dataclass
class DeviceConfig:
    """Device configuration for PyTorch."""
    preferred: str = "auto"
    batch_size: int = 32


@dataclass
class UMAPConfig:
    """UMAP experiment configuration."""
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    n_components: int = 2
    init: str = "spectral"


@dataclass
class RSAConfig:
    """RSA experiment configuration."""
    n_permutations: int = 10000


@dataclass
class GWConfig:
    """Gromov-Wasserstein experiment configuration."""
    entropic_reg: float = 0.005
    use_sinkhorn: bool = True
    n_permutations: int = 5000


@dataclass
class AxesConfig:
    """Axiological axes experiment configuration."""
    n_bootstrap: int = 1000


@dataclass
class ClusteringConfig:
    """Clustering experiment configuration."""
    method: str = "ward"
    k_values: list[int] = field(default_factory=lambda: [3, 5, 7, 10])
    n_permutations: int = 5000


@dataclass
class NDAConfig:
    """Neighborhood Divergence Analysis configuration."""
    k: int = 10
    n_permutations: int = 5000


@dataclass
class ExperimentsConfig:
    """All experiments configuration."""
    rsa: RSAConfig = field(default_factory=RSAConfig)
    gromov_wasserstein: GWConfig = field(default_factory=GWConfig)
    axes: AxesConfig = field(default_factory=AxesConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    nda: NDAConfig = field(default_factory=NDAConfig)


@dataclass
class PathsConfig:
    """Paths configuration."""
    data: Path = field(default_factory=lambda: Path("data"))
    processed: Path = field(default_factory=lambda: Path("data/processed"))
    cache: Path = field(default_factory=lambda: Path("cache"))
    output: Path = field(default_factory=lambda: Path("output"))
    plots: Path = field(default_factory=lambda: Path("output/plots"))
    models: Path = field(default_factory=lambda: Path("models"))


@dataclass
class OutputConfig:
    """Output configuration."""
    results_file: str = "results.json"
    visualization_file: str = "visualization.html"
    save_plots: bool = True
    plot_dpi: int = 300
    plot_figsize: tuple[int, int] = (12, 8)
    plot_style: str = "whitegrid"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class Config:
    """Main configuration class for CLS Pipeline."""
    version: str
    name: str
    models: dict[str, ModelConfig]
    device: DeviceConfig
    random_seed: int
    experiments: ExperimentsConfig
    paths: PathsConfig
    output: OutputConfig
    logging: LoggingConfig
    project_root: Path
    hf_token: str | None = None

    def get_absolute_path(self, path_name: str) -> Path:
        """Get absolute path for a configured path."""
        path = getattr(self.paths, path_name)
        if path.is_absolute():
            return path
        return self.project_root / path


def _parse_model_config(data: dict) -> ModelConfig:
    """Parse model configuration from dict."""
    return ModelConfig(
        name=data["name"],
        dimension=data["dimension"],
        label=data["label"],
        language=data["language"],
        prefix=data.get("prefix", ""),
    )


def _parse_device_config(data: dict) -> DeviceConfig:
    """Parse device configuration from dict."""
    return DeviceConfig(
        preferred=data.get("preferred", "auto"),
        batch_size=data.get("batch_size", 32),
    )


def _parse_umap_config(data: dict) -> UMAPConfig:
    """Parse UMAP configuration from dict."""
    return UMAPConfig(
        n_neighbors=data.get("n_neighbors", 15),
        min_dist=data.get("min_dist", 0.1),
        metric=data.get("metric", "cosine"),
        n_components=data.get("n_components", 2),
        init=data.get("init", "spectral"),
    )


def _parse_gw_config(data: dict) -> GWConfig:
    """Parse Gromov-Wasserstein configuration from dict."""
    return GWConfig(
        entropic_reg=data.get("entropic_reg", 0.005),
        use_sinkhorn=data.get("use_sinkhorn", True),
    )


def _parse_rsa_config(data: dict) -> RSAConfig:
    """Parse RSA configuration from dict."""
    return RSAConfig(
        n_permutations=data.get("n_permutations", 10000),
    )


def _parse_axes_config(data: dict) -> AxesConfig:
    """Parse axes configuration from dict."""
    return AxesConfig(
        n_bootstrap=data.get("n_bootstrap", 1000),
    )


def _parse_clustering_config(data: dict) -> ClusteringConfig:
    """Parse clustering configuration from dict."""
    return ClusteringConfig(
        method=data.get("method", "ward"),
        k_values=data.get("k_values", [3, 5, 7, 10]),
        n_permutations=data.get("n_permutations", 5000),
    )


def _parse_nda_config(data: dict) -> NDAConfig:
    """Parse NDA configuration from dict."""
    return NDAConfig(
        k=data.get("k", 10),
        n_permutations=data.get("n_permutations", 5000),
    )


def _parse_experiments_config(data: dict) -> ExperimentsConfig:
    """Parse experiments configuration from dict."""
    return ExperimentsConfig(
        rsa=_parse_rsa_config(data.get("rsa", {})),
        gromov_wasserstein=_parse_gw_config(data.get("gromov_wasserstein", {})),
        axes=_parse_axes_config(data.get("axes", {})),
        umap=_parse_umap_config(data.get("umap", {})),
        clustering=_parse_clustering_config(data.get("clustering", {})),
        nda=_parse_nda_config(data.get("nda", {})),
    )


def _parse_paths_config(data: dict) -> PathsConfig:
    """Parse paths configuration from dict."""
    return PathsConfig(
        data=Path(data.get("data", "data")),
        processed=Path(data.get("processed", "data/processed")),
        cache=Path(data.get("cache", "cache")),
        output=Path(data.get("output", "output")),
        plots=Path(data.get("plots", "output/plots")),
        models=Path(data.get("models", "models")),
    )


def _parse_output_config(data: dict) -> OutputConfig:
    """Parse output configuration from dict."""
    figsize = data.get("plot_figsize", [12, 8])
    return OutputConfig(
        results_file=data.get("results_file", "results.json"),
        visualization_file=data.get("visualization_file", "visualization.html"),
        save_plots=data.get("save_plots", True),
        plot_dpi=data.get("plot_dpi", 300),
        plot_figsize=tuple(figsize),
        plot_style=data.get("plot_style", "whitegrid"),
    )


def _parse_logging_config(data: dict) -> LoggingConfig:
    """Parse logging configuration from dict."""
    return LoggingConfig(
        level=data.get("level", "INFO"),
        format=data.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )


def load_config(
    config_path: Path | str | None = None,
    env_path: Path | str | None = None,
) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Parameters
    ----------
    config_path : Path | str | None
        Path to config.yaml. If None, looks in project root.
    env_path : Path | str | None
        Path to .env file. If None, looks in project root.

    Returns
    -------
    Config
        Fully populated configuration object.
    """
    # Determine project root
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)
        project_root = config_path.parent

    # Load .env file
    if env_path is None:
        env_path = project_root / ".env"
    else:
        env_path = Path(env_path)

    if env_path.exists():
        load_dotenv(env_path)

    # Load YAML config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Parse all sections
    pipeline = data.get("pipeline", {})
    models_data = data.get("models", {})

    models = {
        "weird": _parse_model_config(models_data["weird"]),
        "sinic": _parse_model_config(models_data["sinic"]),
    }

    config = Config(
        version=pipeline.get("version", "2.0.0"),
        name=pipeline.get("name", "CLS Pipeline"),
        models=models,
        device=_parse_device_config(data.get("device", {})),
        random_seed=data.get("reproducibility", {}).get("random_seed", 42),
        experiments=_parse_experiments_config(data.get("experiments", {})),
        paths=_parse_paths_config(data.get("paths", {})),
        output=_parse_output_config(data.get("output", {})),
        logging=_parse_logging_config(data.get("logging", {})),
        project_root=project_root,
        hf_token=os.getenv("HF_TOKEN"),
    )

    # Create directories
    for path_name in ["data", "processed", "cache", "output", "plots", "models"]:
        path = config.get_absolute_path(path_name)
        path.mkdir(parents=True, exist_ok=True)

    return config


# Module-level singleton for convenience
_config: Config | None = None


def get_config() -> Config:
    """Get or load the global configuration singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
