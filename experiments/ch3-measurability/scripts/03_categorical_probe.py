#!/usr/bin/env python3
"""
Step 3b — Link §3.1.4 categorical-probe results.

The categorical probe is built on templated sentences with 11 categories per
test, not on pool terms. Its results are therefore pool-independent: the same
five tests with the same five paraphrase templates produce the same numbers
whether the underlying pool is run #3's 327 Firthian-strict terms or run #4's
364 post-BLP terms. We do not re-execute the probe for run #4; we link the
run #3 output and annotate the provenance.

This script:
  1. Copies <frozen pre-registration>/categorical_probe.json
     to ch3-measurability/experiment_1_structure/results_bare/categorical_probe.json
  2. Copies <frozen pre-registration>/categorical_probe.json
     to ch3-measurability/experiment_1_structure/results_attested/categorical_probe.json
  3. Stamps a `linked_from` field with the source path and a `note` explaining
     pool-independence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments" / "ch3-measurability"


def link_one(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload.setdefault("meta", {})
    payload["meta"]["linked_from"] = str(src.relative_to(REPO_ROOT))
    payload["meta"]["linked_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["meta"]["pool_independence_note"] = (
        "Categorical probe uses templated 11-category sentence sequences "
        "(see categorical_probe_expected.yaml). The probe does not consume "
        "the run pool. Run #4 reuses the run #3 output without re-execution; "
        "any model-level encoding cache differences between run #3 and run #4 "
        "are absorbed by the EmbeddingClient SHA-keyed cache."
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  linked {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    args = parser.parse_args()

    with Path(args.config).open() as fh:
        cfg = yaml.safe_load(fh)

    # The categorical probe is BOTH pool-independent AND encoding-independent
    # (it uses templated sentences, not pool terms. Hence a single source
    # file is sufficient — we link the same file to both variant folders.
    source = REPO_ROOT / "<frozen pre-registration>/categorical_probe.json"
    sources = {"bare": source, "attested": source}
    destinations = {
        "bare":     REPO_ROOT / cfg["paths"]["structure_bare"]     / "categorical_probe.json",
        "attested": REPO_ROOT / cfg["paths"]["structure_attested"] / "categorical_probe.json",
    }
    for variant in ("bare", "attested"):
        link_one(sources[variant], destinations[variant])
    return 0


if __name__ == "__main__":
    sys.exit(main())
