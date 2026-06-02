# Geometria Iuris — Measuring Legal Meaning in Embedding Spaces

Experimental code and frozen results for the master's thesis *Geometria Iuris:
Measuring Legal Meaning Across Cultural Normative Structures in Embedding
Spaces* (LUISS, Methodology of Legal Science).

The thesis asks a methodological question: **is legal meaning measurable?** The
cross-tradition design (WEIRD vs Sinic legal language) is the experimental
apparatus for that question, not the subject of the thesis. The instrument is an
embedding space; the contribution is showing what such a space can and cannot
register about legal meaning, and with what limits.

This repository is the **code and the frozen experimental artefacts** behind the
results. It is deliberately code-centric.

## What is here

```
experiments/
├── ch3-measurability/    Frozen run #4 (post-BLP). The canonical results.
│   ├── experiment_1_structure/         geometric structure & model agreement (JSON)
│   ├── experiment_2_axes/              projection onto value axes (JSON)
│   ├── ext/                            robustness extensions (A..Z)
│   ├── scripts/                        the pipeline that produced the results
│   ├── reports/                        diagnostic plots
│   ├── figures/                        thesis figures (PNG) + matplotlib scripts that generate them
│   ├── categorical_probe_expected.yaml §3.1.4 pre-registration (frozen 2026-04-11)
│   ├── manifest.json                   50 SHA-256 hashes (reproducibility gate)
│   └── config.yaml                     single source of truth for run #4 parameters
├── dashboard_final/      Build pipeline for the static dashboard,
│                         regenerable via build.py from the results.
├── shared/               Runtime modules (embeddings client, statistics, HTML helpers)
├── pre_checks/           Adversarial pre-checks (numeracy, polysemy, register)
└── data/                 Legal-lexicon construction pipeline: 364 post-BLP
                          terms across 7 domains + 100-item Swadesh control,
                          plus the processed JSON inputs.

docs/                     GitHub Pages source: seven self-contained static HTML
                          pages mirroring dashboard_final/output/.
                          Hosted at https://capazme.github.io/Geometria-Iuris/
```

## Headline result

Run #4 is post-BLP (Hong Kong ordinances enacted under the Bilingual Laws
Project, structural bilingual co-drafting), 364 legal terms × 10 encoders, both
*bare* and *attested* readings, with a 100-item non-legal control. Verification
gate: 8/8 PASS.

The cross-tradition symmetrised divergence is **Δρ_sym (attested) = 0.543**.
A critical caveat is built into the reading: the same metric on the bare encoder
is ≈ 0.165 on the legal pool and ≈ 0.156 on the non-legal control, so the bare
gap is *encoder-tradition-shaped*, not legal-tradition-shaped. The legal signal
is therefore the **attested−bare gap ≈ 0.378**, not the attested absolute. The
full set of numbers lives in the JSON result files under
`experiments/ch3-measurability/experiment_{1,2}_*/results_{bare,attested}/`, and
in the extensions under `ext/`; the dashboard at
[capazme.github.io/Geometria-Iuris](https://capazme.github.io/Geometria-Iuris/)
visualises them with the inferential discipline (measure / interpretation /
limit) that governs their reading.

## What is NOT here, and why

- **Raw Hong Kong legislation** — licensing and size (hundreds of MB of zipped
  corpora). The build scripts document how the lexicon was derived from it.
- **Model weights** — downloaded from their original sources by name.
- **Embeddings (`.npy` / `.npz`)** — ~540 MB of intermediate vectors,
  regenerable from the models and the inputs. The small `meta.json` /
  `coverage.json` manifests that describe them are kept.
- **The thesis manuscript** — added separately.

## Reproducing / inspecting

The results are JSON and are meant to be read directly. The headline numbers are
the `section_*` blocks of the per-experiment `*_results.json` files under
`experiments/ch3-measurability/experiment_{1,2}_*/results_{bare,attested}/`. To
regenerate the static dashboard from those JSONs:

```bash
python3 experiments/dashboard_final/build.py
# writes the six HTML files into experiments/dashboard_final/output/
# the same content is mirrored in docs/ as the GitHub Pages source
```

The pipeline is Python 3 (NumPy, SciPy, Plotly; the categorical probe relies on
`numpy.linalg.svd` and `scipy.stats.spearmanr`). Computation is CPU, float32,
deterministic with fixed seeds.

### A note on pre-registration provenance

The §3.1.4 categorical probe was pre-registered before the run that produced its
figures: the expected breakpoints, the eleven-category sequences, and the
distance-from-midpoint constraint live in
`experiments/ch3-measurability/categorical_probe_expected.yaml`. The commit date
of the pre-registration (2026-04-11) is recorded in the YAML's `meta` block.
This repository starts from a clean history, so the date is documented in-file
rather than proven by git log.
