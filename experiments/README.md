# Geometria Iuris — Experiments

Source of truth for thesis structure: `documenti/003_GeometriaIuris_Indice.docx`

**2026-04-16 — experimental scope reduced to two lenses.**
See `trace_pivot_2lens.md` for the reasoning. Lenses II, III, V, VI are
archived under `_archive/lenses_2026-04-16/` and remain available for
appendix citation.

## Structure

```
experiments/
├── data/                    Ch.2 §2.1 — dataset design
│   ├── raw/                 source files (HK DOJ XML, e-Legislation, etc.)
│   ├── processed/           built dataset and embedding cache
│   ├── parse_sources.py     parsers for raw sources
│   └── build_dataset.py     builds legal_terms.json
├── models/                  Ch.2 §2.3 — model selection
│   └── config.yaml          model panel (6 base + 2 bilingual)
├── shared/
│   ├── embeddings.py        unified embedding client
│   └── statistical.py       permutation tests, bootstrap, effect sizes
├── lens_1_relational/       Ch.3 §3.1 — RSA, relational structure
├── lens_4_values/           Ch.3 §3.3 — Kozlowski value axes
├── trace_pivot_2lens.md     pivot decision record (2026-04-16)
├── _archive/                archived experiments and legacy code
└── notebooks/               exploratory analysis and visualization
```

## Experiment map

| Experiment     | Technique                     | Question                              | Section |
|----------------|-------------------------------|---------------------------------------|---------|
| Lens I         | RSA / relational distance     | Does the instrument detect signal, replicably, with discriminating power? | §3.1 |
| Lens IV        | Kozlowski value-axis projection | On which doctrinally-chosen value dimensions do traditions diverge? | §3.3 |

## Build order

1. `data/parse_sources.py`  — parse HK DOJ XML and e-Legislation corpus
2. `data/build_dataset.py`  — assemble `legal_terms.json`
3. `shared/precompute.py`   — embed terms with each of the 8 models
4. `lens_1_relational/`     — Ch.3 §3.1 (instrument validation)
5. `lens_4_values/`         — Ch.3 §3.3 (applied measurement)
