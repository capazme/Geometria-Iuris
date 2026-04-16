"""
Run SAE training + domain enrichment analysis for all embedding models.

Skips models that already have results (use --force to re-run).
Prints a cross-model comparison table at the end.

Usage
-----
    python lens_6_sae/run_all_models.py [--force] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ALL_MODELS = [
    "BGE-EN-large",       # WEIRD Slot 1 (dim=1024)
    "E5-large",           # WEIRD Slot 2 (dim=1024)
    "FreeLaw-EN",         # WEIRD Slot 3 (dim=768)
    "BGE-ZH-large",       # Sinic Slot 1 (dim=1024)
    "Text2vec-large-ZH",  # Sinic Slot 2 (dim=1024)
    "Dmeta-ZH",           # Sinic Slot 3 (dim=768)
    "BGE-M3-EN",          # Bilingual control EN (dim=1024)
    "BGE-M3-ZH",          # Bilingual control ZH (dim=1024)
    "Qwen3-0.6B-EN",      # Bilingual control EN (dim=1024)
    "Qwen3-0.6B-ZH",      # Bilingual control ZH (dim=1024)
]

PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python3")


def make_suffix(model_label: str, expansion: int, k: int) -> str:
    return f"_{model_label}_x{expansion}_k{k}"


def run_model(model: str, expansion: int, k: int, epochs: int,
              device: str, force: bool) -> dict | None:
    """Train SAE + run enrichment for one model. Return summary or None."""
    suffix = make_suffix(model, expansion, k)
    act_path = RESULTS_DIR / f"activations{suffix}.npy"
    summary_path = RESULTS_DIR / f"feature_summary{suffix}.json"

    if act_path.exists() and summary_path.exists() and not force:
        print(f"\n[skip] {model}: results exist (use --force to re-run)")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"[batch] Training SAE for {model} ...")
    print(f"{'='*60}")

    # Train SAE
    t0 = time.perf_counter()
    cmd_train = [
        PYTHON, "lens_6_sae/sae.py",
        "--model", model,
        "--expansion", str(expansion),
        "--k", str(k),
        "--epochs", str(epochs),
        "--device", device,
    ]
    result = subprocess.run(cmd_train, cwd=str(REPO_ROOT),
                            capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] SAE training failed for {model}")
        return None

    # Run enrichment analysis
    print(f"\n[batch] Running enrichment for {model} ...")
    cmd_enrich = [
        PYTHON, "lens_6_sae/lens6.py",
        "--model", model,
        "--expansion", str(expansion),
        "--k", str(k),
    ]
    result = subprocess.run(cmd_enrich, cwd=str(REPO_ROOT),
                            capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Enrichment failed for {model}")
        return None

    elapsed = time.perf_counter() - t0
    print(f"[batch] {model} complete in {elapsed:.0f}s")

    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def print_comparison(summaries: dict[str, dict]) -> None:
    """Print cross-model comparison table."""
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL COMPARISON (expansion=4, k=32)")
    print(f"{'='*80}")
    header = f"{'Model':25s} {'Features':>8} {'Sig':>5} {'DSI mean':>8} {'DSI p90':>8} {'Top domain':>20}"
    print(header)
    print("-" * len(header))

    for model in ALL_MODELS:
        s = summaries.get(model)
        if s is None:
            print(f"{model:25s} {'FAILED':>8}")
            continue

        # Find top domain
        dc = s.get("domain_feature_counts", {})
        top_d = max(dc, key=dc.get) if dc else "—"
        top_c = dc.get(top_d, 0)

        print(
            f"{model:25s} "
            f"{s['n_active_features']:8d} "
            f"{s['n_features_with_significant_enrichment']:5d} "
            f"{s['dsi_mean']:8.3f} "
            f"{s['dsi_p90']:8.3f} "
            f"{top_d} ({top_c})"
        )

    # Save comparison
    comp_path = RESULTS_DIR / "cross_model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n[batch] Saved: {comp_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="+", help="Subset of models to run")
    args = parser.parse_args()

    models = args.models if args.models else ALL_MODELS
    summaries: dict[str, dict] = {}

    overall_t0 = time.perf_counter()
    for model in models:
        summary = run_model(
            model, args.expansion, args.k, args.epochs,
            args.device, args.force,
        )
        if summary is not None:
            summaries[model] = summary

    elapsed = time.perf_counter() - overall_t0
    print(f"\n[batch] All models done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print_comparison(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
