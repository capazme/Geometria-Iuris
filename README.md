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
│   ├── experiment_1_structure/   geometric structure & model agreement
│   ├── experiment_2_axes/        projection onto value axes
│   ├── ext/                      robustness extensions (A..Z)
│   ├── scripts/                  the pipeline that produced the results
│   ├── reports/                  ready-made number tables
│   ├── manifest.json             50 SHA-256 hashes (reproducibility gate)
│   └── HANDOFF.md / OVERVIEW.md  how to read the numbers
├── dashboard_final/      Six self-contained static HTML pages (committee
│                         presentation), regenerable from the results.
└── data/                 Legal-lexicon construction pipeline:
                          364 post-BLP terms across 7 domains + 100-item
                          Swadesh control, with the processed JSON inputs.
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
is therefore the **attested−bare gap ≈ 0.378**, not the attested absolute. See
`experiments/ch3-measurability/HANDOFF.md` for the full set of numbers and the
inferential discipline that governs their reading.

## What is NOT here, and why

- **Raw Hong Kong legislation** — licensing and size (hundreds of MB of zipped
  corpora). The build scripts document how the lexicon was derived from it.
- **Model weights** — downloaded from their original sources by name.
- **Embeddings (`.npy` / `.npz`)** — ~540 MB of intermediate vectors,
  regenerable from the models and the inputs. The small `meta.json` /
  `coverage.json` manifests that describe them are kept.
- **The thesis manuscript** — added separately.

## Reproducing / inspecting

The results are JSON and are meant to be read directly; the headline tables are
in `experiments/ch3-measurability/reports/`. To regenerate the presentation
dashboard:

```bash
python3 experiments/dashboard_final/build.py
# writes the six HTML files into experiments/dashboard_final/output/
```

The pipeline is Python 3 (NumPy, SciPy, Plotly; the categorical probe relies on
`numpy.linalg.svd` and `scipy.stats.spearmanr`). Computation is CPU, float32,
deterministic with fixed seeds.

### A note on pre-registration provenance

The §3.1.4 categorical probe was pre-registered before the run that produced its
figures: the expected breakpoints and the midpoint constraint live in
`experiments/lens_1_relational/categorical_probe_expected.yaml` (kept locally),
with the commit date recorded in that file and in `HANDOFF.md`. This repository
starts from a clean history, so the date is documented in-file rather than
proven by git log.
