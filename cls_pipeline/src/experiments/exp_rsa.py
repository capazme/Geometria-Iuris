"""
exp_rsa.py — Experiment 1: RSA + Mantel Test.

Computes Representational Dissimilarity Matrices (RDM) for WEIRD and Sinic
embedding spaces, then measures structural similarity via Spearman correlation
with significance assessed by Mantel permutation test.

References
----------
Kriegeskorte et al. (2008), Frontiers in Systems Neuroscience.
"""
# ─── RSA come confronto "di secondo ordine" ──────────────────────────
# L'RSA non confronta direttamente i vettori embedding (che vivono in
# spazi di dimensione e orientamento diversi), ma le *geometrie interne*
# dei due spazi. Si costruisce una matrice di dissimilarità (RDM) per
# ciascun modello, poi si misura quanto le due RDM sono correlate.
# È un confronto strutturale: se due concetti sono "vicini" in uno
# spazio, lo sono anche nell'altro?
# Rif.: Kriegeskorte, Mur & Bandettini (2008) "Representational
#        Similarity Analysis", Frontiers in Systems Neuroscience, 2, 4.
# ─────────────────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cosine as cosine_dist, pdist, squareform
from scipy.stats import spearmanr

from .statistical import mantel_test, PermutationResult

logger = logging.getLogger(__name__)


@dataclass
class RSAResult:
    """Result of RSA analysis."""
    rdm_weird: np.ndarray
    rdm_sinic: np.ndarray
    spearman_r: float
    p_value: float
    n_permutations: int
    n_pairs: int
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spearman_r": self.spearman_r,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "n_pairs": self.n_pairs,
            "n_terms": len(self.labels),
            "labels": self.labels,
            "rdm_weird": self.rdm_weird.tolist(),
            "rdm_sinic": self.rdm_sinic.tolist(),
            "significant": self.p_value < 0.05,
        }


def compute_rdm(vectors: np.ndarray) -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix (cosine distance).

    Parameters
    ----------
    vectors : np.ndarray
        Embedding matrix (N x D).

    Returns
    -------
    np.ndarray
        Symmetric N x N cosine distance matrix.
    """
    # Distanza del coseno: d(a,b) = 1 - cos(a,b). È invariante alla norma
    # dei vettori, quindi misura solo la direzione nello spazio semantico.
    # Questo è essenziale per gli embedding, dove la norma può variare
    # per motivi non semantici (frequenza del token, batch di addestramento).
    # Usa scipy.spatial.distance.pdist (C compilato) per efficienza:
    # con N=500 termini → 124.750 coppie, ~100× più veloce del doppio ciclo.
    return squareform(pdist(vectors, metric="cosine"))


def run_rsa(
    emb_weird: np.ndarray,
    emb_sinic: np.ndarray,
    labels: list[str],
    n_permutations: int = 10000,
    seed: int = 42,
) -> RSAResult:
    """
    Run RSA analysis with Mantel test.

    Parameters
    ----------
    emb_weird : np.ndarray
        WEIRD embeddings (N x D).
    emb_sinic : np.ndarray
        Sinic embeddings (N x D).
    labels : list[str]
        Term labels (length N).
    n_permutations : int
        Number of permutations for Mantel test.
    seed : int
        Random seed.

    Returns
    -------
    RSAResult
        RDMs, Spearman r, p-value.
    """
    n = emb_weird.shape[0]
    assert emb_sinic.shape[0] == n, "Both embeddings must have same number of terms"
    assert len(labels) == n, "Labels must match embedding count"

    logger.info("Computing RDMs for %d terms...", n)

    # Pipeline in 3 passi:
    # 1. Calcola le RDM (matrici di dissimilarità) per ciascun spazio
    # 2. Correla le RDM con Spearman (confronto di rango tra geometrie)
    # 3. Valuta significatività con test di Mantel (permutazione righe/colonne)
    rdm_w = compute_rdm(emb_weird)
    rdm_s = compute_rdm(emb_sinic)

    n_pairs = n * (n - 1) // 2
    logger.info("RDMs computed: %d x %d, %d unique pairs", n, n, n_pairs)

    mantel_result = mantel_test(rdm_w, rdm_s, n_permutations=n_permutations, seed=seed)

    logger.info(
        "RSA: Spearman r=%.4f, p=%.4f",
        mantel_result.observed, mantel_result.p_value,
    )

    return RSAResult(
        rdm_weird=rdm_w,
        rdm_sinic=rdm_s,
        spearman_r=mantel_result.observed,
        p_value=mantel_result.p_value,
        n_permutations=n_permutations,
        n_pairs=n_pairs,
        labels=labels,
    )


def plot_rdm_heatmaps(
    result: RSAResult,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (18, 8),
    dpi: int = 300,
    weird_label: str = "WEIRD",
    sinic_label: str = "Sinic",
    domains: list[str] | None = None,
) -> Path:
    """
    Generate side-by-side RDM heatmaps.

    For large matrices (N > 80), labels are hidden to preserve readability.
    A separate sampled heatmap with labels is generated for inspection.

    Parameters
    ----------
    result : RSAResult
        RSA analysis result.
    output_dir : Path | None
        Output directory.
    figsize : tuple
        Figure size.
    dpi : int
        Plot resolution.
    weird_label : str
        Label for WEIRD model.
    sinic_label : str
        Label for Sinic model.
    domains : list[str] | None
        Domain for each term (same order as labels). If provided, generates
        an additional inter-domain distance matrix.

    Returns
    -------
    Path
        Path to saved PNG.
    """
    out = output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    n = result.rdm_weird.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    triu_w = result.rdm_weird[triu_idx]
    triu_s = result.rdm_sinic[triu_idx]

    # Soglia per mostrare etichette: con N > 80, le etichette diventano
    # illeggibili e appesantiscono il rendering. Mostriamo solo la struttura.
    show_labels = n <= 80

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if show_labels:
        short_labels = [l[:15] for l in result.labels]
        xticklabels_w, yticklabels_w = short_labels, short_labels
        xticklabels_s, yticklabels_s = short_labels, short_labels
    else:
        xticklabels_w, yticklabels_w = False, False
        xticklabels_s, yticklabels_s = False, False

    sns.heatmap(
        result.rdm_weird, ax=ax1,
        xticklabels=xticklabels_w, yticklabels=yticklabels_w,
        cmap="viridis", vmin=triu_w.min(), vmax=triu_w.max(),
        square=True, cbar_kws={"shrink": 0.8},
    )
    ax1.set_title(f"{weird_label} RDM (N={n}, range {triu_w.min():.2f}–{triu_w.max():.2f})")
    if show_labels:
        ax1.tick_params(axis="both", labelsize=6)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax1.get_yticklabels(), rotation=0)

    sns.heatmap(
        result.rdm_sinic, ax=ax2,
        xticklabels=xticklabels_s, yticklabels=yticklabels_s,
        cmap="viridis", vmin=triu_s.min(), vmax=triu_s.max(),
        square=True, cbar_kws={"shrink": 0.8},
    )
    ax2.set_title(f"{sinic_label} RDM (N={n}, range {triu_s.min():.2f}–{triu_s.max():.2f})")
    if show_labels:
        ax2.tick_params(axis="both", labelsize=6)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax2.get_yticklabels(), rotation=0)

    fig.suptitle(
        f"Representational Dissimilarity Matrices\n"
        f"Spearman r = {result.spearman_r:.4f}, p = {result.p_value:.4f} "
        f"(Mantel test, {result.n_permutations:,} permutations)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    path = out / "rsa_rdm_heatmaps.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("RSA heatmaps saved: %s", path)

    # ─── Heatmap complete full-size con etichette (file separati) ─────────
    # Per N grande, generiamo singole heatmap abbastanza grandi da rendere
    # le etichette leggibili. Regola: ~0.12 pollici per termine.
    if n > 80:
        fig_size_full = max(20, n * 0.12)  # es. N=394 → ~47 pollici
        short_labels = [l[:20] for l in result.labels]  # etichette più lunghe

        # WEIRD full
        fig_w = plt.figure(figsize=(fig_size_full, fig_size_full))
        ax_w = fig_w.add_subplot(111)
        sns.heatmap(
            result.rdm_weird, ax=ax_w,
            xticklabels=short_labels, yticklabels=short_labels,
            cmap="viridis", vmin=triu_w.min(), vmax=triu_w.max(),
            square=True, cbar_kws={"shrink": 0.5},
        )
        ax_w.set_title(f"{weird_label} RDM (N={n})", fontsize=16, fontweight="bold")
        ax_w.tick_params(axis="both", labelsize=5)
        plt.setp(ax_w.get_xticklabels(), rotation=90, ha="center")
        plt.setp(ax_w.get_yticklabels(), rotation=0)
        fig_w.tight_layout()
        path_w_full = out / "rsa_rdm_weird_full.png"
        fig_w.savefig(path_w_full, dpi=150, bbox_inches="tight")  # dpi ridotto per file size
        plt.close(fig_w)
        logger.info("RSA WEIRD full heatmap saved: %s", path_w_full)

        # Sinic full
        fig_s = plt.figure(figsize=(fig_size_full, fig_size_full))
        ax_s = fig_s.add_subplot(111)
        sns.heatmap(
            result.rdm_sinic, ax=ax_s,
            xticklabels=short_labels, yticklabels=short_labels,
            cmap="viridis", vmin=triu_s.min(), vmax=triu_s.max(),
            square=True, cbar_kws={"shrink": 0.5},
        )
        ax_s.set_title(f"{sinic_label} RDM (N={n})", fontsize=16, fontweight="bold")
        ax_s.tick_params(axis="both", labelsize=5)
        plt.setp(ax_s.get_xticklabels(), rotation=90, ha="center")
        plt.setp(ax_s.get_yticklabels(), rotation=0)
        fig_s.tight_layout()
        path_s_full = out / "rsa_rdm_sinic_full.png"
        fig_s.savefig(path_s_full, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        logger.info("RSA Sinic full heatmap saved: %s", path_s_full)

    # ─── Heatmap campionata con etichette (se N > 80) ─────────────────────
    # Per consentire ispezione visiva dei termini, estraiamo un sottocampione
    # stratificato (o casuale se non ci sono domini) di ~50 termini.
    if n > 80:
        sample_size = min(50, n)
        rng = np.random.default_rng(42)

        if domains is not None and len(domains) == n:
            # Campionamento stratificato per dominio
            unique_domains = list(dict.fromkeys(domains))  # preserva ordine
            per_domain = max(1, sample_size // len(unique_domains))
            sample_idx = []
            for dom in unique_domains:
                dom_indices = [i for i, d in enumerate(domains) if d == dom]
                chosen = rng.choice(dom_indices, size=min(per_domain, len(dom_indices)), replace=False)
                sample_idx.extend(chosen.tolist())
            sample_idx = sorted(sample_idx[:sample_size])
        else:
            sample_idx = sorted(rng.choice(n, size=sample_size, replace=False).tolist())

        rdm_w_sample = result.rdm_weird[np.ix_(sample_idx, sample_idx)]
        rdm_s_sample = result.rdm_sinic[np.ix_(sample_idx, sample_idx)]
        sample_labels = [result.labels[i][:15] for i in sample_idx]

        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize)

        sns.heatmap(
            rdm_w_sample, ax=ax3,
            xticklabels=sample_labels, yticklabels=sample_labels,
            cmap="viridis", square=True, cbar_kws={"shrink": 0.8},
        )
        ax3.set_title(f"{weird_label} RDM (sample N={len(sample_idx)})")
        ax3.tick_params(axis="both", labelsize=6)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax3.get_yticklabels(), rotation=0)

        sns.heatmap(
            rdm_s_sample, ax=ax4,
            xticklabels=sample_labels, yticklabels=sample_labels,
            cmap="viridis", square=True, cbar_kws={"shrink": 0.8},
        )
        ax4.set_title(f"{sinic_label} RDM (sample N={len(sample_idx)})")
        ax4.tick_params(axis="both", labelsize=6)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax4.get_yticklabels(), rotation=0)

        fig2.suptitle(
            f"RDM Sample ({len(sample_idx)} terms) — for label inspection",
            fontsize=13, fontweight="bold",
        )
        fig2.tight_layout(rect=[0, 0.05, 1, 0.93])

        path_sample = out / "rsa_rdm_heatmaps_sample.png"
        fig2.savefig(path_sample, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        logger.info("RSA sample heatmaps saved: %s", path_sample)

    # ─── Matrice inter-dominio (se forniti domini) ────────────────────────
    # Aggreghiamo le distanze per dominio: media delle distanze tra tutti i
    # termini di un dominio e tutti i termini di un altro. Risultato: matrice
    # compatta K×K (dove K = numero di domini) che mostra relazioni macro.
    if domains is not None and len(domains) == n:
        unique_domains = list(dict.fromkeys(domains))
        k = len(unique_domains)
        domain_rdm_w = np.zeros((k, k))
        domain_rdm_s = np.zeros((k, k))

        domain_indices = {dom: [i for i, d in enumerate(domains) if d == dom] for dom in unique_domains}

        for i, dom_i in enumerate(unique_domains):
            for j, dom_j in enumerate(unique_domains):
                idx_i = domain_indices[dom_i]
                idx_j = domain_indices[dom_j]
                # Media delle distanze tra tutti i termini dei due domini
                domain_rdm_w[i, j] = result.rdm_weird[np.ix_(idx_i, idx_j)].mean()
                domain_rdm_s[i, j] = result.rdm_sinic[np.ix_(idx_i, idx_j)].mean()

        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))

        # Nomi dominio più leggibili
        domain_labels = [d.replace("_", " ").title() for d in unique_domains]

        sns.heatmap(
            domain_rdm_w, ax=ax5,
            xticklabels=domain_labels, yticklabels=domain_labels,
            cmap="viridis", square=True, annot=True, fmt=".3f",
            cbar_kws={"shrink": 0.8},
        )
        ax5.set_title(f"{weird_label} Inter-Domain RDM")
        ax5.tick_params(axis="both", labelsize=9)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax5.get_yticklabels(), rotation=0)

        sns.heatmap(
            domain_rdm_s, ax=ax6,
            xticklabels=domain_labels, yticklabels=domain_labels,
            cmap="viridis", square=True, annot=True, fmt=".3f",
            cbar_kws={"shrink": 0.8},
        )
        ax6.set_title(f"{sinic_label} Inter-Domain RDM")
        ax6.tick_params(axis="both", labelsize=9)
        plt.setp(ax6.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax6.get_yticklabels(), rotation=0)

        fig3.suptitle(
            f"Inter-Domain Distance Matrix ({k} domains)\n"
            f"Mean cosine distance between all terms of each domain pair",
            fontsize=12, fontweight="bold",
        )
        fig3.tight_layout(rect=[0, 0.03, 1, 0.92])

        path_domain = out / "rsa_rdm_interdomain.png"
        fig3.savefig(path_domain, dpi=dpi, bbox_inches="tight")
        plt.close(fig3)
        logger.info("RSA inter-domain heatmap saved: %s", path_domain)

    return path
