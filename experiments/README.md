# Geometria Iuris — Experiments

Source of truth for thesis structure: `documenti/003_GeometriaIuris_Indice.docx`

## Structure

```
experiments/
├── data/                    Ch.2 §2.1 — dataset design
│   ├── raw/                 source files (HK DOJ XML, CC-CEDICT, etc.)
│   ├── processed/           built dataset and embedding cache
│   ├── parse_sources.py     parsers for raw sources
│   └── build_dataset.py     builds legal_terms.json
├── models/                  Ch.2 §2.3 — model selection
│   └── config.yaml          model selection (WEIRD + Sinic)
├── shared/
│   ├── embeddings.py        unified embedding client
│   └── statistical.py       permutation tests, bootstrap, effect sizes
├── lens_1_relational/       Exp. macro — Ch.3 §3.1
├── lens_2_taxonomy/         (→ Ch.4 §4.4 horizons)
├── lens_3_stratigraphy/     Exp. macro — Ch.3 §3.1.3
├── lens_4_values/           Exp. comparatistico — Ch.3 §3.3
├── lens_5_neighborhoods/    Exp. micro — Ch.3 §3.2
└── notebooks/               exploratory analysis and visualization
```

## Experiment map

| Experiment     | Technique                   | Question                              | Section |
|----------------|-----------------------------|---------------------------------------|---------|
| Macro (Lens I) | Relational distance (RDM)   | Do legal structures have shape?       | §3.1    |
| Macro (Lens III)| Layer stratigraphy          | At what depth does meaning live?      | §3.1.3  |
| Micro (Lens V) | Semantic neighborhoods      | Which concepts are false friends?     | §3.2    |
| Comparative (Lens IV) | Value axis projection | On which dimensions do they diverge?  | §3.3    |

## Build order

1. `data/parse_sources.py` — parse HK DOJ XML and other sources
2. `data/build_dataset.py` — assemble `legal_terms.json`
3. `shared/embeddings.py`  — embed terms with each model
4. Experiments: macro (§3.1) → micro (§3.2) → comparative (§3.3)
