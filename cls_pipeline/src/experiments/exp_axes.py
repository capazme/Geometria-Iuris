"""
exp_axes.py — Experiment 3: Axiological Axis Projection (Kozlowski method).

Projects legal terms onto value axes constructed from multiple antonym pairs,
independently in each embedding space. Compares rank-order via Spearman
correlation with bootstrap confidence intervals.

References
----------
Kozlowski, Taddy & Evans (2019), American Sociological Review.
"""
# ─── Geometria culturale di Kozlowski ────────────────────────────────
# L'intuizione: una dimensione culturale (es. individuo↔collettivo) è
# codificata nello spazio embedding come *direzione*. Per recuperarla,
# si usano coppie multiple di antonimi (non una sola coppia) e se ne
# media la differenza vettoriale. Questo riduce il rumore idiosincratico
# di ciascuna coppia e isola la componente semantica condivisa.
#
# Ciascun modello costruisce il *proprio* asse dal *proprio* vocabolario:
# l'asse WEIRD usa coppie inglesi, l'asse Sinic usa coppie cinesi.
# Questo evita il bias da traduzione: non imponiamo che "freedom" e "自由"
# definiscano la stessa direzione.
#
# Rif.: Kozlowski, Taddy & Evans (2019) "The Geometry of Culture:
#        Analyzing the Meanings of Class through Word Embeddings",
#        American Sociological Review, 84(5), 905-949.
# ─────────────────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr

from .statistical import bootstrap_ci, BootstrapCIResult

logger = logging.getLogger(__name__)


@dataclass
class AxisResult:
    """Result for a single axis in one model space."""
    axis_name: str
    axis_vector: np.ndarray
    scores: dict[str, float]


@dataclass
class AxesComparisonResult:
    """Result of axis projection comparison across models."""
    axis_name: str
    weird_scores: dict[str, float]
    sinic_scores: dict[str, float]
    spearman_r: float
    spearman_p: float
    bootstrap_ci: BootstrapCIResult | None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "axis_name": self.axis_name,
            "weird_scores": self.weird_scores,
            "sinic_scores": self.sinic_scores,
            "spearman_r": self.spearman_r,
            "spearman_p": self.spearman_p,
        }
        if self.bootstrap_ci:
            result["bootstrap_ci"] = {
                "estimate": self.bootstrap_ci.estimate,
                "ci_lower": self.bootstrap_ci.ci_lower,
                "ci_upper": self.bootstrap_ci.ci_upper,
                "n_bootstrap": self.bootstrap_ci.n_bootstrap,
                "alpha": self.bootstrap_ci.alpha,
            }
        return result


@dataclass
class AxesExperimentResult:
    """Complete result of Experiment 3."""
    axes: list[AxesComparisonResult]
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_terms": len(self.labels),
            "n_axes": len(self.axes),
            "axes": [ax.to_dict() for ax in self.axes],
        }


def build_kozlowski_axis(pair_embeddings: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Build a value axis from multiple antonym pairs (Kozlowski method).

    axis = normalize(mean(embed(a_i) - embed(b_i) for i in 1..k))

    Parameters
    ----------
    pair_embeddings : list[tuple[np.ndarray, np.ndarray]]
        List of (positive_embedding, negative_embedding) tuples.

    Returns
    -------
    np.ndarray
        L2-normalized axis vector.
    """
    # Media delle differenze: ciascuna coppia (positivo, negativo) contribuisce
    # con un vettore direzione. Mediando su k coppie, si cancella il rumore
    # specifico di ogni singola coppia. La normalizzazione L2 rende l'asse
    # un vettore unitario, così la proiezione successiva (cosine similarity)
    # è interpretabile direttamente come score in [-1, 1].
    diffs = [a - b for a, b in pair_embeddings]
    axis = np.mean(diffs, axis=0)
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm
    return axis


def project_on_axis(
    vectors: np.ndarray,
    labels: list[str],
    axis: np.ndarray,
) -> dict[str, float]:
    """
    Project vectors onto an axis using cosine similarity.

    Parameters
    ----------
    vectors : np.ndarray
        Term embeddings (N x D).
    labels : list[str]
        Term labels.
    axis : np.ndarray
        Normalized axis vector.

    Returns
    -------
    dict[str, float]
        Label -> score mapping, scores in [-1, 1].
    """
    # Cosine similarity come proiezione direzionale: cos(v, asse) misura
    # quanto il vettore del termine è allineato con la dimensione culturale.
    # Score positivo = polo "positivo" dell'asse, negativo = polo "negativo".
    scores = {}
    for label, vec in zip(labels, vectors):
        sim = 1.0 - cosine_dist(vec, axis)
        scores[label] = float(sim)
    return scores


def run_axes_experiment(
    emb_weird: np.ndarray,
    emb_sinic: np.ndarray,
    labels: list[str],
    value_axes: dict,
    embed_fn_weird: callable,
    embed_fn_sinic: callable,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> AxesExperimentResult:
    """
    Run the axiological axis projection experiment.

    Parameters
    ----------
    emb_weird : np.ndarray
        WEIRD term embeddings (N x D).
    emb_sinic : np.ndarray
        Sinic term embeddings (N x D).
    labels : list[str]
        Term labels (length N).
    value_axes : dict
        Axis definitions with en_pairs and zh_pairs.
    embed_fn_weird : callable
        Function to embed English text list -> np.ndarray.
    embed_fn_sinic : callable
        Function to embed Chinese text list -> np.ndarray.
    n_bootstrap : int
        Number of bootstrap resamples for CI.
    seed : int
        Random seed.

    Returns
    -------
    AxesExperimentResult
        Scores, correlations, and CIs for all axes.
    """
    results = []

    for axis_name, axis_def in value_axes.items():
        logger.info("Processing axis: %s", axis_name)

        en_pairs = axis_def["en_pairs"]
        zh_pairs = axis_def["zh_pairs"]

        # Ciascun modello costruisce il proprio asse indipendentemente:
        # WEIRD usa coppie inglesi, Sinic usa coppie cinesi.
        en_pos_texts = [p[0] for p in en_pairs]
        en_neg_texts = [p[1] for p in en_pairs]
        en_pos_emb = embed_fn_weird(en_pos_texts)
        en_neg_emb = embed_fn_weird(en_neg_texts)
        weird_pairs = list(zip(en_pos_emb, en_neg_emb))

        # Embed antonym pairs for Sinic model
        zh_pos_texts = [p[0] for p in zh_pairs]
        zh_neg_texts = [p[1] for p in zh_pairs]
        zh_pos_emb = embed_fn_sinic(zh_pos_texts)
        zh_neg_emb = embed_fn_sinic(zh_neg_texts)
        sinic_pairs = list(zip(zh_pos_emb, zh_neg_emb))

        # Build axes independently
        weird_axis = build_kozlowski_axis(weird_pairs)
        sinic_axis = build_kozlowski_axis(sinic_pairs)

        # Project terms
        weird_scores = project_on_axis(emb_weird, labels, weird_axis)
        sinic_scores = project_on_axis(emb_sinic, labels, sinic_axis)

        # Spearman correlation between rank orders
        w_vals = np.array([weird_scores[l] for l in labels])
        s_vals = np.array([sinic_scores[l] for l in labels])
        rho, p_val = spearmanr(w_vals, s_vals)

        # Bootstrap CI per la correlazione di Spearman: ricampionando i
        # termini con rimpiazzo, si stima la variabilità della correlazione.
        paired_scores = np.column_stack([w_vals, s_vals])

        def spearman_stat(data):
            r, _ = spearmanr(data[:, 0], data[:, 1])
            return r

        ci_result = bootstrap_ci(
            paired_scores, spearman_stat,
            n_bootstrap=n_bootstrap, seed=seed,
        )

        logger.info(
            "Axis %s: rho=%.4f, p=%.4f, CI=[%.4f, %.4f]",
            axis_name, rho, p_val, ci_result.ci_lower, ci_result.ci_upper,
        )

        results.append(AxesComparisonResult(
            axis_name=axis_name,
            weird_scores=weird_scores,
            sinic_scores=sinic_scores,
            spearman_r=float(rho),
            spearman_p=float(p_val),
            bootstrap_ci=ci_result,
        ))

    return AxesExperimentResult(axes=results, labels=labels)


def plot_axes_comparison(
    result: AxesExperimentResult,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (18, 6),
    dpi: int = 300,
    weird_label: str = "WEIRD",
    sinic_label: str = "Sinic",
) -> Path:
    """
    Generate comparative bar charts for all axes.

    Parameters
    ----------
    result : AxesExperimentResult
        Experiment result.
    output_dir : Path | None
        Output directory.
    figsize : tuple
        Figure size per axis.
    dpi : int
        Plot resolution.
    weird_label : str
        WEIRD model label.
    sinic_label : str
        Sinic model label.

    Returns
    -------
    Path
        Path to saved PNG.
    """
    out = output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)

    n_axes = len(result.axes)
    fig, axes_list = plt.subplots(1, n_axes, figsize=(figsize[0], figsize[1]))
    if n_axes == 1:
        axes_list = [axes_list]

    sns.set_style("whitegrid")

    for idx, ax_result in enumerate(result.axes):
        ax = axes_list[idx]

        # Sort labels by WEIRD score for consistent ordering
        sorted_labels = sorted(
            result.labels,
            key=lambda l: ax_result.weird_scores.get(l, 0),
            reverse=True,
        )
        # Show top 20 and bottom 5 for readability
        if len(sorted_labels) > 25:
            show_labels = sorted_labels[:15] + sorted_labels[-10:]
        else:
            show_labels = sorted_labels

        w_scores = [ax_result.weird_scores[l] for l in show_labels]
        s_scores = [ax_result.sinic_scores[l] for l in show_labels]

        x = np.arange(len(show_labels))
        width = 0.35

        ax.barh(x - width / 2, w_scores, width, label=weird_label, color="#4C72B0")
        ax.barh(x + width / 2, s_scores, width, label=sinic_label, color="#DD8452")

        ax.set_yticks(x)
        ax.set_yticklabels([l[:20] for l in show_labels], fontsize=7)
        ax.set_xlabel("Projection Score")
        ax.set_title(
            f"{ax_result.axis_name}\n"
            f"rho={ax_result.spearman_r:.3f}, p={ax_result.spearman_p:.3f}",
            fontsize=10,
        )
        ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    fig.suptitle("Axiological Axis Projections (Kozlowski Method)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = out / "axes_projection_comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    logger.info("Axes projection plot saved: %s", path)
    return path
