"""
output_manager.py — Structured output management for results.json.

Handles collection, validation, and serialization of all pipeline results
into a structured JSON format for visualization and analysis.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class PipelineMetadata:
    """Metadata for the pipeline run."""
    pipeline_version: str
    timestamp_utc: str
    config_hash: str
    input_data_hash: str
    models: dict[str, dict[str, Any]]
    device: str
    random_seed: int


class OutputManager:
    """
    Manages collection and serialization of pipeline results.

    Collects results from each experiment and produces a structured JSON
    output file suitable for visualization and analysis.
    """

    def __init__(
        self,
        config: Any,
        input_data_hash: str,
        config_hash: str,
    ):
        self.config = config
        self.output_dir = config.get_absolute_path("output")

        self.metadata = PipelineMetadata(
            pipeline_version=config.version,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            input_data_hash=input_data_hash,
            models={
                name: {
                    "name": model.name,
                    "dimension": model.dimension,
                    "label": model.label,
                }
                for name, model in config.models.items()
            },
            device=config.device.preferred,
            random_seed=config.random_seed,
        )

        self.experiments: dict[str, Any] = {}

    def set_rsa_result(self, result) -> None:
        """Set RSA experiment result."""
        self.experiments["1_rsa"] = result.to_dict()
        logger.info("RSA result recorded: r=%.4f", result.spearman_r)

    def set_gw_result(self, result) -> None:
        """Set Gromov-Wasserstein experiment result."""
        self.experiments["2_gromov_wasserstein"] = result.to_dict()
        logger.info("GW result recorded: distance=%.6f", result.distance)

    def set_axes_result(self, result) -> None:
        """Set axiological axes experiment result."""
        self.experiments["3_axes"] = result.to_dict()
        logger.info("Axes result recorded: %d axes", len(result.axes))

    def set_clustering_result(self, result) -> None:
        """Set clustering experiment result."""
        self.experiments["4_clustering"] = result.to_dict()
        logger.info(
            "Clustering result recorded: %d k-values",
            len(result.fm_results),
        )

    def set_nda_result(self, result) -> None:
        """Set NDA experiment result."""
        self.experiments["5_nda"] = result.to_dict()
        logger.info(
            "NDA result recorded: mean_jaccard=%.4f",
            result.part_a.mean_jaccard,
        )

    def set_umap_result(
        self,
        coords_weird: np.ndarray,
        coords_sinic: np.ndarray,
        labels_weird: list[str],
        labels_sinic: list[str],
    ) -> None:
        """Set UMAP visualization result (supplementary)."""
        weird_coords = [
            {"label": label, "x": float(coords_weird[i, 0]), "y": float(coords_weird[i, 1])}
            for i, label in enumerate(labels_weird)
        ]
        sinic_coords = [
            {"label": label, "x": float(coords_sinic[i, 0]), "y": float(coords_sinic[i, 1])}
            for i, label in enumerate(labels_sinic)
        ]
        self.experiments["supplementary_umap"] = {
            "coordinates": {"weird": weird_coords, "sinic": sinic_coords},
        }
        logger.info("UMAP result recorded: %d points", len(weird_coords) + len(sinic_coords))

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "metadata": asdict(self.metadata),
            "experiments": self.experiments,
        }

    def save(self, filename: str | None = None) -> Path:
        """
        Save results to JSON file.

        Parameters
        ----------
        filename : str | None
            Output filename. Uses config default if None.

        Returns
        -------
        Path
            Path to saved file.
        """
        if filename is None:
            filename = self.config.output.results_file

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

        logger.info("Results saved to: %s", output_path)
        return output_path
