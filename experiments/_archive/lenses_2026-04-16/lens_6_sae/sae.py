"""
TopK Sparse Autoencoder for sentence embedding decomposition.

Trains on BGE-EN-v1.5 bare embeddings (9472 x 1024) and decomposes them
into monosemantic features. Follows Gao et al. 2024 (arXiv:2406.04093)
architecture with calibration for small dataset size.

Output
------
results/
    sae_weights.pt        trained model state dict
    activations.npy       (9472, dict_size) feature activation matrix
    training_metrics.json  loss curves, reconstruction quality, dead neurons

Usage
-----
    python lens_6_sae/sae.py [--expansion 4] [--k 32] [--epochs 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
EMB_DIR = REPO_ROOT / "data" / "processed" / "embeddings"


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder (Gao et al. 2024).

    Architecture:
        encode: z_pre = W_enc @ (x - b_dec) + b_enc
        activate: z = TopK(ReLU(z_pre))
        decode: x_hat = W_dec @ z + b_dec

    Decoder columns are unit-normalized after each optimizer step.
    """

    def __init__(self, input_dim: int, dict_size: int, k: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        # Kaiming init for encoder, decoder columns ~ unit norm
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Normalize decoder columns to unit norm."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sparse activations z, pre-activation z_pre)."""
        x_centered = x - self.b_dec
        z_pre = self.encoder(x_centered)

        # TopK: keep only top-k activations, zero out rest, apply ReLU
        topk_vals, topk_idx = torch.topk(z_pre, self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(1, topk_idx, torch.relu(topk_vals))
        return z, z_pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse features."""
        return self.decoder(z) + self.b_dec

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reconstruction, sparse activations, pre-activations)."""
        z, z_pre = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, z_pre


def compute_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z_pre: torch.Tensor,
    model: TopKSAE,
    dead_mask: torch.Tensor | None,
    aux_coeff: float = 1 / 32,
) -> tuple[torch.Tensor, dict]:
    """MSE reconstruction loss + auxiliary dead neuron loss.

    The auxiliary loss (Gao et al. 2024) encourages dead neurons to
    participate by computing reconstruction through the top-2k dead
    neurons and adding its MSE.
    """
    recon_loss = (x - x_hat).pow(2).mean()
    metrics = {"recon_loss": recon_loss.item()}

    # Auxiliary dead neuron loss
    aux_loss = torch.tensor(0.0, device=x.device)
    if dead_mask is not None and dead_mask.any():
        n_dead = dead_mask.sum().item()
        if n_dead > 0:
            dead_z_pre = z_pre[:, dead_mask]
            k_aux = min(model.k * 2, int(n_dead))
            if k_aux > 0:
                topk_vals, topk_idx = torch.topk(dead_z_pre, k_aux, dim=-1)
                dead_z = torch.zeros_like(dead_z_pre)
                dead_z.scatter_(1, topk_idx, torch.relu(topk_vals))
                dead_decoder = model.decoder.weight[:, dead_mask]
                x_hat_aux = dead_z @ dead_decoder.t() + model.b_dec
                aux_loss = (x - x_hat_aux).pow(2).mean()
        metrics["aux_loss"] = aux_loss.item()
        metrics["n_dead"] = int(n_dead)

    total = recon_loss + aux_coeff * aux_loss
    metrics["total_loss"] = total.item()
    return total, metrics


def train_sae(
    vectors: np.ndarray,
    dict_size: int,
    k: int,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    warmup_epochs: int = 100,
    dead_threshold: int = 50,
    device: str = "mps",
    seed: int = 42,
) -> tuple[TopKSAE, np.ndarray, dict]:
    """Train TopK SAE and return (model, activations, metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = vectors.shape[1]
    n_samples = vectors.shape[0]
    print(f"[SAE] Training TopK SAE: {input_dim}d -> {dict_size} features, k={k}")
    print(f"[SAE] Data: {n_samples} samples, {epochs} epochs, batch={batch_size}")
    print(f"[SAE] Device: {device}")

    X = torch.from_numpy(vectors).float()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    model = TopKSAE(input_dim, dict_size, k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    neuron_activity = torch.zeros(dict_size, device=device)
    dead_mask: torch.Tensor | None = None

    history: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        n_batches = 0
        epoch_activity = torch.zeros(dict_size, device=device)

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            x_hat, z, z_pre = model(batch_x)
            loss, batch_metrics = compute_loss(
                batch_x, x_hat, z_pre, model, dead_mask
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model._normalize_decoder()

            epoch_loss += batch_metrics["total_loss"]
            epoch_recon += batch_metrics["recon_loss"]
            n_batches += 1

            epoch_activity += (z > 0).float().sum(dim=0)

        scheduler.step()

        neuron_activity = 0.9 * neuron_activity + 0.1 * epoch_activity
        dead_mask = neuron_activity < dead_threshold

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches

        if epoch % 100 == 0 or epoch == epochs - 1:
            n_dead = dead_mask.sum().item() if dead_mask is not None else 0
            n_active = dict_size - n_dead
            elapsed = time.perf_counter() - t0
            print(
                f"  epoch {epoch:4d}/{epochs}: "
                f"loss={avg_loss:.6f} recon={avg_recon:.6f} "
                f"active={n_active}/{dict_size} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"[{elapsed:.1f}s]"
            )
            history.append({
                "epoch": epoch,
                "total_loss": avg_loss,
                "recon_loss": avg_recon,
                "n_active": n_active,
                "n_dead": n_dead,
                "lr": scheduler.get_last_lr()[0],
            })

    elapsed = time.perf_counter() - t0
    print(f"[SAE] Training complete: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Compute final activations for all samples
    model.eval()
    with torch.no_grad():
        all_z = []
        all_x_hat = []
        for start in range(0, n_samples, batch_size):
            batch_x = X[start : start + batch_size].to(device)
            x_hat, z, _ = model(batch_x)
            all_z.append(z.cpu())
            all_x_hat.append(x_hat.cpu())
        activations = torch.cat(all_z, dim=0).numpy()
        reconstructions = torch.cat(all_x_hat, dim=0).numpy()

    # Reconstruction quality metrics
    mse = np.mean((vectors - reconstructions) ** 2)
    # EVR = R^2: denominator is per-dimension centered variance, not E[x^2].
    # Sentence embeddings have a strong mean vector (anisotropy); E[x^2] and
    # np.var(x) are both ~2x the centered variance, inflating EVR.
    dim_mean = vectors.mean(axis=0)
    total_var = np.mean((vectors - dim_mean) ** 2)
    evr = 1 - mse / total_var if total_var > 0 else 0.0
    cos_sims = np.sum(vectors * reconstructions, axis=1) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(reconstructions, axis=1)
        + 1e-12
    )

    l0_per_sample = np.count_nonzero(activations, axis=1)
    feature_freq = np.count_nonzero(activations, axis=0)

    quality = {
        "mse": float(mse),
        "explained_variance_ratio": float(evr),
        "cosine_sim_mean": float(np.mean(cos_sims)),
        "cosine_sim_std": float(np.std(cos_sims)),
        "l0_mean": float(np.mean(l0_per_sample)),
        "l0_std": float(np.std(l0_per_sample)),
        "n_active_features": int(np.sum(feature_freq > 0)),
        "n_dead_features": int(np.sum(feature_freq == 0)),
        "feature_freq_median": float(np.median(feature_freq[feature_freq > 0]))
        if np.any(feature_freq > 0) else 0.0,
    }

    metrics = {
        "config": {
            "input_dim": input_dim,
            "dict_size": dict_size,
            "k": k,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_samples": n_samples,
            "seed": seed,
        },
        "quality": quality,
        "history": history,
        "training_time_s": elapsed,
    }

    print(f"\n[SAE] Reconstruction quality:")
    print(f"  MSE:  {quality['mse']:.6f}")
    print(f"  EVR:  {quality['explained_variance_ratio']:.4f}")
    print(f"  cos:  {quality['cosine_sim_mean']:.4f} +/- {quality['cosine_sim_std']:.4f}")
    print(f"  L0:   {quality['l0_mean']:.1f} +/- {quality['l0_std']:.1f}")
    print(f"  Features: {quality['n_active_features']} active, "
          f"{quality['n_dead_features']} dead")

    return model, activations, metrics


def make_suffix(model_label: str, expansion: int, k: int) -> str:
    """Canonical file suffix for a given model/config combination."""
    return f"_{model_label}_x{expansion}_k{k}"


def main(args: argparse.Namespace) -> int:
    model_label = args.model
    vec_path = EMB_DIR / model_label / "vectors.npy"
    print(f"[SAE] Loading embeddings from {vec_path}")
    vectors = np.load(vec_path)
    print(f"[SAE] Shape: {vectors.shape}, dtype: {vectors.dtype}")

    input_dim = vectors.shape[1]
    dict_size = int(args.expansion * input_dim)
    k = args.k

    model, activations, metrics = train_sae(
        vectors,
        dict_size=dict_size,
        k=k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup,
        device=args.device,
        seed=args.seed,
    )
    metrics["config"]["model_label"] = model_label

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = make_suffix(model_label, args.expansion, k)

    weights_path = RESULTS_DIR / f"sae_weights{suffix}.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"\n[SAE] Saved weights: {weights_path}")

    act_path = RESULTS_DIR / f"activations{suffix}.npy"
    np.save(act_path, activations)
    print(f"[SAE] Saved activations: {act_path} (shape {activations.shape})")

    metrics_path = RESULTS_DIR / f"training_metrics{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAE] Saved metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="BGE-EN-large",
                        help="Embedding label (directory name under embeddings/)")
    parser.add_argument("--expansion", type=int, default=4,
                        help="Expansion factor (dict_size = expansion * input_dim)")
    parser.add_argument("--k", type=int, default=32,
                        help="TopK sparsity (features active per sample)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100,
                        help="LR warmup epochs")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    sys.exit(main(parser.parse_args()))
