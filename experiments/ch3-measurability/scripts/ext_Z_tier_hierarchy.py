#!/usr/bin/env python3
"""
Extension Z — Three-tier distance hierarchy.

For each of the 10 models, compare the cosine-distance distributions of:
  (a) core × core      (intra-pool, 364×364 upper triangle)
  (b) core × bg        (cross-block, 364 × 9045)
  (c) core × control   (cross-block, 364 × 100)

Expectation: medians (a) < (b) < (c). The bg are "legal-ish, semi-near"
and control are "non-legal, far". Confirms the operational meaning of
the three-tier classification at the embedding-geometry level.

Output: ext/Z_tier_hierarchy/tier_hierarchy.json + per-model arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import compute_rdm, mannwhitney_with_r, upper_tri  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    args = parser.parse_args()
    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    emb = REPO_ROOT / cfg["paths"]["embeddings"]
    all_labels = (
        [m["label"] for m in cfg["models_weird"]]
        + [m["label"] for m in cfg["models_sinic"]]
        + [f"{m['label']}-EN" for m in cfg["models_bilingual"]]
        + [f"{m['label']}-ZH" for m in cfg["models_bilingual"]]
    )

    per_model: dict[str, dict] = {}
    out_dir = RUN_DIR / "ext" / "Z_tier_hierarchy"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "distances").mkdir(exist_ok=True)

    for label in all_labels:
        core = np.load(emb / label / "vecs_bare.npy").astype(np.float32)
        bg = np.load(emb / "bg" / label / "vecs_bare.npy").astype(np.float32)
        ctrl = np.load(emb / "control_bare" / label / "vecs.npy").astype(np.float32)

        d_cc = upper_tri(compute_rdm(core))
        d_cb = cdist(core, bg, metric="cosine").flatten()
        d_ctr = cdist(core, ctrl, metric="cosine").flatten()

        mw_cb = mannwhitney_with_r(d_cc, d_cb, alternative="less")
        mw_cctr = mannwhitney_with_r(d_cc, d_ctr, alternative="less")
        mw_bctr = mannwhitney_with_r(d_cb, d_ctr, alternative="less")

        per_model[label] = {
            "n_pairs": {
                "core_core": int(d_cc.size),
                "core_bg": int(d_cb.size),
                "core_control": int(d_ctr.size),
            },
            "median": {
                "core_core": round(float(np.median(d_cc)), 4),
                "core_bg": round(float(np.median(d_cb)), 4),
                "core_control": round(float(np.median(d_ctr)), 4),
            },
            "mean": {
                "core_core": round(float(d_cc.mean()), 4),
                "core_bg": round(float(d_cb.mean()), 4),
                "core_control": round(float(d_ctr.mean()), 4),
            },
            "mw_core_vs_bg": {
                "r_effect": round(float(mw_cb.effect_r), 4),
                "p_value": float(mw_cb.p_value),
            },
            "mw_core_vs_control": {
                "r_effect": round(float(mw_cctr.effect_r), 4),
                "p_value": float(mw_cctr.p_value),
            },
            "mw_bg_vs_control": {
                "r_effect": round(float(mw_bctr.effect_r), 4),
                "p_value": float(mw_bctr.p_value),
            },
            "monotonic_hierarchy": bool(
                np.median(d_cc) < np.median(d_cb) < np.median(d_ctr)
            ),
        }
        np.savez_compressed(
            out_dir / "distances" / f"{label}.npz",
            core_core=d_cc.astype(np.float32),
            core_bg=d_cb.astype(np.float32),
            core_control=d_ctr.astype(np.float32),
        )
        m = per_model[label]
        print(f"{label:22s} medians c-c={m['median']['core_core']:.3f}  "
              f"c-bg={m['median']['core_bg']:.3f}  "
              f"c-ctrl={m['median']['core_control']:.3f}  "
              f"{'monotonic ✓' if m['monotonic_hierarchy'] else 'NOT monotonic ✗'}")

    n_monotonic = sum(1 for v in per_model.values() if v["monotonic_hierarchy"])
    with (out_dir / "tier_hierarchy.json").open("w") as fh:
        json.dump({
            "meta": {
                "n_models": len(all_labels),
                "n_core": 364,
                "n_bg": 9045,
                "n_control": 100,
                "expectation": "median(core-core) < median(core-bg) < median(core-control)",
                "n_models_with_monotonic_hierarchy": n_monotonic,
            },
            "per_model": per_model,
        }, fh, indent=2, ensure_ascii=False)

    print(f"\nMonotonic hierarchy holds in {n_monotonic}/{len(all_labels)} models.")
    print(f"Wrote {(out_dir / 'tier_hierarchy.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
