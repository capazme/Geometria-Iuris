#!/usr/bin/env python3
"""
Step 3c — §3.1.1 legal-vs-control (Lens I, bare only).

For each of the 10 models, compares the distribution of pairwise distances
within the 364 legal core terms against the cross-block distribution
between legal core and the 100 control (everyday-language) terms.

Mann-Whitney U (one-sided 'less': legal-legal more compact than legal-ctrl)
+ rank-biserial effect size r. Bare-only by design: controls have no HK Cap.
attestation and thus no attested-context counterpart.

Output:
  experiment_1_structure/results_bare/legal_vs_control.json
  experiment_1_structure/results_bare/legal_vs_control/{model}.npz    (legal + legal_ctrl arrays)

Also patches experiment_1_structure/results_bare/experiment_1_results.json with a new section
`section_311_legal_vs_control`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    compute_rdm,
    mannwhitney_with_r,
    upper_tri,
)


def mw_dict(mw) -> dict:
    return {
        "statistic": round(mw.statistic, 2),
        "p_value": float(mw.p_value),
        "effect_r": round(mw.effect_r, 4),
        "n_x": int(mw.n_x),
        "n_y": int(mw.n_y),
        "median_x": round(mw.median_x, 4),
        "median_y": round(mw.median_y, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    args = parser.parse_args()
    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    emb_dir = REPO_ROOT / cfg["paths"]["embeddings"]
    ctrl_dir = emb_dir / "control_bare"
    out_root = REPO_ROOT / cfg["paths"]["structure_bare"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "legal_vs_control").mkdir(exist_ok=True)

    all_labels = (
        [m["label"] for m in cfg["models_weird"]]
        + [m["label"] for m in cfg["models_sinic"]]
        + [f"{m['label']}-EN" for m in cfg["models_bilingual"]]
        + [f"{m['label']}-ZH" for m in cfg["models_bilingual"]]
    )

    per_model: dict[str, dict] = {}
    for label in all_labels:
        legal = np.load(emb_dir / label / "vecs_bare.npy").astype(np.float32)
        ctrl  = np.load(ctrl_dir / label / "vecs.npy").astype(np.float32)
        if legal.shape[0] != 364 or ctrl.shape[0] != 100:
            raise AssertionError(f"{label}: legal={legal.shape[0]}, ctrl={ctrl.shape[0]}")

        rdm_legal = compute_rdm(legal)
        legal_dist = upper_tri(rdm_legal)
        cross = cdist(legal, ctrl, metric="cosine").flatten()

        mw = mannwhitney_with_r(legal_dist, cross, alternative="less")
        per_model[label] = mw_dict(mw)
        np.savez_compressed(
            out_root / "legal_vs_control" / f"{label}.npz",
            legal=legal_dist.astype(np.float32),
            legal_ctrl=cross.astype(np.float32),
        )
        print(f"  {label:22s} legal med={mw.median_x:.3f}  "
              f"ctrl med={mw.median_y:.3f}  r={mw.effect_r:+.3f}  p={mw.p_value:.2e}")

    payload = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "n_legal": 364,
            "n_control": 100,
            "variant": "bare",
            "alternative": "less",
            "metric": "cosine",
            "control_kind": "everyday-language (pronouns, deixis, common nouns)",
        },
        "per_model": per_model,
    }
    out_path = out_root / "legal_vs_control.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nWritten: {out_path.relative_to(REPO_ROOT)}")

    # Patch experiment_1_results.json bare with the new section
    lens1_path = out_root / "experiment_1_results.json"
    if lens1_path.exists():
        experiment_1 = json.loads(experiment_1_path.read_text())
        experiment_1["section_311_legal_vs_control"] = payload
        lens1_path.write_text(json.dumps(experiment_1, indent=2, ensure_ascii=False))
        print(f"Patched: {lens1_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
