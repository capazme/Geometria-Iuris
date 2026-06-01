"""
Smoke test for all six embedding models in Geometria Iuris.

Tests each model in sequence:
  1. Model loads without error
  2. Output shape matches the dim declared in config.yaml
  3. Output is L2-normalized (unit norm per row)

Run from experiments/:
    python shared/smoke_test.py

Exit code 0 if all models pass; 1 if any fail.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Suppress tqdm weight-loading bars (220KB of output per model, not a hang)
# and transformer logger noise. Only our print() calls will show.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.embeddings import EmbeddingClient, ModelSpec  # noqa: E402

PROBE = "The court shall have jurisdiction over the matter."
PROBE_ZH = "法院對此事項具有管轄權。"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"


def test_model(client: EmbeddingClient, spec: ModelSpec, probe: str) -> bool:
    print(f"\n  [{spec.label}] {spec.id}")

    print(f"  → loading weights ...", end="", flush=True)
    t_load = time.perf_counter()
    try:
        client._get_model(spec.id)
    except Exception as exc:
        print(f"\n  {FAIL} Load failed: {exc}")
        return False
    print(f" done ({time.perf_counter() - t_load:.1f}s)")

    print(f"  → first encode (may be slow on MPS: Metal JIT) ...", end="", flush=True)
    t_enc = time.perf_counter()
    try:
        vecs = client.embed([probe], spec.id, use_cache=False)
    except Exception as exc:
        print(f"\n  {FAIL} Encode failed: {exc}")
        return False
    print(f" done ({time.perf_counter() - t_enc:.1f}s)")

    # Shape check
    if vecs.shape != (1, spec.dim):
        print(f"  {FAIL} Shape mismatch: expected (1, {spec.dim}), got {vecs.shape}")
        return False

    # Normalization check
    norm = float(np.linalg.norm(vecs[0]))
    if not np.isclose(norm, 1.0, atol=1e-4):
        print(f"  {FAIL} Not unit-norm: ‖v‖ = {norm:.6f}")
        return False

    print(f"  {PASS} shape={vecs.shape}  ‖v‖={norm:.6f}")
    return True


def main() -> int:
    config = ROOT / "models" / "config.yaml"
    print(f"Config: {config}")
    print(f"Probe EN: '{PROBE}'")
    print(f"Probe ZH: '{PROBE_ZH}'")

    # Smoke test uses CPU: correctness check, not performance benchmark.
    # MPS Metal JIT compilation on first encode is silent and can appear as a hang.
    # The cache ensures MPS is only slow once during actual experiments.
    client = EmbeddingClient(config, device="cpu")
    print(f"Device: {client._device}  (forced CPU for predictable smoke test)")

    results: dict[str, bool] = {}

    print("\n── WEIRD models (English) ─────────────────────────────────────────")
    for spec in client.weird_specs:
        results[spec.label] = test_model(client, spec, PROBE)

    print("\n── Sinic models (Chinese) ─────────────────────────────────────────")
    for spec in client.sinic_specs:
        results[spec.label] = test_model(client, spec, PROBE_ZH)

    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"\n── Summary {'─' * 56}")
    for label, ok in results.items():
        icon = PASS if ok else FAIL
        print(f"  {icon}  {label}")
    print(f"\n  {passed}/{total} models passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
