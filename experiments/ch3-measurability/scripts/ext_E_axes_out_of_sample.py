#!/usr/bin/env python3
"""
Extension E — Out-of-sample axes projection.

Project the 9045 bg terms onto the 6 Kozlowski axes built on the 364 core.
Reuses experiment_2_axes/results_{bare,attested}/axes/{label}_{axis}.npy axis vectors.

Output:
  ext/E_axes_oos/scores_bg/{label}_{axis}.npy   (9045 scores per term)
  ext/E_axes_oos/coherence.json
    — per (label, axis): correlation of bg-score with k-NN-assigned-domain
      mean-score-of-core (test if axes generalize out-of-sample).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    parser.add_argument("--variant", choices=["bare", "attested"], default="bare",
                        help="Which axes to use (bare or attested)")
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    emb = REPO_ROOT / cfg["paths"]["embeddings"]
    lens4_root = REPO_ROOT / cfg["paths"][f"lens4_{args.variant}"]
    bg_idx = json.loads((emb / "bg/index.json").read_text())

    en_labels = [m["label"] for m in cfg["models_weird"]] + \
                [f"{m['label']}-EN" for m in cfg["models_bilingual"]]
    zh_labels = [m["label"] for m in cfg["models_sinic"]] + \
                [f"{m['label']}-ZH" for m in cfg["models_bilingual"]]
    all_labels = en_labels + zh_labels

    # Axis names from any axes_experiment results
    axes_experiment = json.loads((lens4_root / "experiment_2_results.json").read_text())
    axis_names = axes_experiment["meta"]["axes"]

    out_dir = RUN_DIR / "ext" / "E_axes_oos"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = out_dir / f"scores_bg_{args.variant}"
    scores_dir.mkdir(exist_ok=True)

    # Load knn assignment (built in ext A); fall back gracefully
    knn_path = RUN_DIR / "ext/A_bg_knn/background_assignments.json"
    bg_assigned: dict[str, str] = {}
    if knn_path.exists():
        knn = json.loads(knn_path.read_text())
        bg_assigned = {a["en"]: a["assigned_domain"] for a in knn["assignments"]}

    coherence: dict[str, dict] = {}
    for label in all_labels:
        bg_vecs = np.load(emb / "bg" / label / "vecs_bare.npy").astype(np.float32)
        per_axis: dict[str, dict] = {}
        for ax in axis_names:
            axis_vec = np.load(lens4_root / "axes" / f"{label}_{ax}.npy").astype(np.float32)
            bg_scores = bg_vecs @ axis_vec
            np.save(scores_dir / f"{label}_{ax}.npy", bg_scores.astype(np.float32))
            # Per assigned domain: mean bg score
            per_domain: dict[str, dict] = {}
            for dom in set(bg_assigned.values()):
                idxs = [i for i, t in enumerate(bg_idx) if bg_assigned.get(t["en"]) == dom]
                if not idxs:
                    continue
                arr = bg_scores[idxs]
                per_domain[dom] = {
                    "n": len(idxs),
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std()), 4),
                }
            per_axis[ax] = per_domain
        coherence[label] = per_axis
        print(f"  {label}: projected 9045 bg on {len(axis_names)} axes")

    with (out_dir / "coherence.json").open("w") as fh:
        json.dump({
            "meta": {
                "variant": args.variant,
                "n_bg": len(bg_idx),
                "axes": axis_names,
                "labels": all_labels,
                "domain_source": "knn_bg_assignment_BGE-EN-large_k7",
            },
            "per_model_per_axis_per_domain": coherence,
        }, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {out_dir.relative_to(REPO_ROOT)}/coherence.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
