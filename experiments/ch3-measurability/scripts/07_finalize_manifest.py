#!/usr/bin/env python3
"""
Step 7 (finalize) — Extend manifest.json with SHA-256 hashes of all run #4
output artefacts (experiment_1_results.json, experiment_2_results.json, reports, and
the nine extensions A-H + X/Y/Z).

This closes the audit chain: inputs sealed at step 0 + outputs hashed here.
Idempotent — re-running re-hashes and overwrites the `outputs` block.

Also hashes the control_bare/ and bg/ embedding subdirs in addition to the
core 10 model dirs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"
MANIFEST = RUN_DIR / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_outputs() -> list[dict]:
    paths: list[Path] = [
        RUN_DIR / "experiment_1_structure/results_bare/experiment_1_results.json",
        RUN_DIR / "experiment_1_structure/results_attested/experiment_1_results.json",
        RUN_DIR / "experiment_1_structure/results_bare/categorical_probe.json",
        RUN_DIR / "experiment_1_structure/results_attested/categorical_probe.json",
        RUN_DIR / "experiment_2_axes/results_bare/experiment_2_results.json",
        RUN_DIR / "experiment_2_axes/results_attested/experiment_2_results.json",
        RUN_DIR / "reports/numbers_headline.md",
        RUN_DIR / "reports/changes_vs_run3.md",
        RUN_DIR / "reports/verification_gate.md",
        RUN_DIR / "reports/extensions_summary.md",
        RUN_DIR / "embeddings/index.json",
        RUN_DIR / "ext/A_bg_knn/background_assignments.json",
        RUN_DIR / "ext/D_robustness/robustness_curve.json",
        RUN_DIR / "ext/E_axes_oos/coherence.json",
        RUN_DIR / "ext/F_confidence/confidence_strata.json",
        RUN_DIR / "ext/G_false_friends/false_friends.json",
        RUN_DIR / "ext/H_K_saturation/k_saturation.json",
        RUN_DIR / "ext/X_control_robustness/control_robustness_curve.json",
        RUN_DIR / "ext/Y_control_only/control_only_rsa.json",
        RUN_DIR / "ext/Z_tier_hierarchy/tier_hierarchy.json",
    ]
    out: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        out.append({
            "path": str(p.relative_to(REPO_ROOT)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_of(p),
        })
    return out


def collect_embedding_meta() -> dict:
    """Hash the 10 core embedding dirs (one per model)."""
    emb_root = RUN_DIR / "embeddings"
    out: dict[str, dict] = {}
    for d in sorted(p for p in emb_root.iterdir() if p.is_dir()):
        if d.name in ("control_bare", "bg"):
            continue  # collected separately
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        out[d.name] = {
            "dim": meta["dim"],
            "elapsed_bare_s": meta.get("elapsed_bare_s"),
            "elapsed_attested_s": meta.get("elapsed_attested_s"),
            "vecs_bare_sha256":     sha256_of(d / "vecs_bare.npy"),
            "vecs_attested_sha256": sha256_of(d / "vecs_attested.npy"),
        }
    return out


def collect_control_embeddings() -> dict:
    """Hash the 10 control_bare/{model}/vecs.npy files."""
    ctrl_root = RUN_DIR / "embeddings" / "control_bare"
    if not ctrl_root.exists():
        return {}
    out: dict[str, dict] = {}
    for d in sorted(p for p in ctrl_root.iterdir() if p.is_dir()):
        vec_path = d / "vecs.npy"
        if not vec_path.exists():
            continue
        out[d.name] = {
            "vecs_sha256": sha256_of(vec_path),
        }
    return out


def collect_bg_embeddings() -> dict:
    """Hash the bg/{model}/vecs_bare.npy and vecs_attested.npy files."""
    bg_root = RUN_DIR / "embeddings" / "bg"
    if not bg_root.exists():
        return {}
    out: dict[str, dict] = {}
    for d in sorted(p for p in bg_root.iterdir() if p.is_dir()):
        vec_bare = d / "vecs_bare.npy"
        vec_att = d / "vecs_attested.npy"
        entry: dict[str, str] = {}
        if vec_bare.exists():
            entry["vecs_bare_sha256"] = sha256_of(vec_bare)
        if vec_att.exists():
            entry["vecs_attested_sha256"] = sha256_of(vec_att)
        if entry:
            out[d.name] = entry
    return out


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit("manifest.json missing — run 00_seal_inputs.py first")
    manifest = json.loads(MANIFEST.read_text())

    manifest["finalized_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["outputs"] = collect_outputs()
    manifest["embeddings"] = collect_embedding_meta()
    manifest["control_embeddings"] = collect_control_embeddings()
    manifest["bg_embeddings"] = collect_bg_embeddings()

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest.json updated: "
          f"{len(manifest['outputs'])} outputs + "
          f"{len(manifest['embeddings'])} core + "
          f"{len(manifest['control_embeddings'])} control + "
          f"{len(manifest['bg_embeddings'])} bg embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
