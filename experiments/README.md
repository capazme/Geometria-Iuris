# Geometria Iuris — Experiments

Source of truth for thesis structure: `documenti/001_GeometriaIuris_Indice.docx`

## Structure

```
experiments/
├── data/
│   ├── raw/              source files (HK DOJ XML, CC-CEDICT, etc.)
│   ├── processed/        built dataset and embedding cache
│   ├── parse_sources.py  parsers for raw sources
│   └── build_dataset.py  builds legal_terms.json
├── models/
│   └── config.yaml       model selection (WEIRD + Sinic)
├── shared/
│   ├── embeddings.py     unified embedding client
│   └── statistical.py    permutation tests, bootstrap, effect sizes
├── lens_1_relational/    Lens I  — relational distance structure (Ch.8 + Ch.9)
├── lens_2_taxonomy/      Lens II — emergent taxonomy (Ch.8 + Ch.9)
├── lens_3_stratigraphy/  Lens III — layer stratigraphy (Ch.8 only)
├── lens_4_values/        Lens IV — value axis projection (Ch.9 only)
├── lens_5_neighborhoods/ Lens V  — semantic neighborhoods (Ch.9 only)
└── notebooks/            exploratory analysis and visualization
```

## Lens map

| Lens | Technique                   | Question                              | Chapter |
|------|-----------------------------|---------------------------------------|---------|
| I    | Relational distance (RDM)   | Do legal structures have shape?       | 8 + 9   |
| II   | Emergent taxonomy           | Do legal categories cluster?          | 8 + 9   |
| III  | Layer stratigraphy          | At what depth does meaning live?      | 8       |
| IV   | Value axis projection       | On which dimensions do they diverge?  | 9       |
| V    | Semantic neighborhoods      | Which concepts are false friends?     | 9       |

## Build order

1. `data/parse_sources.py` — parse HK DOJ XML and other sources
2. `data/build_dataset.py` — assemble `legal_terms.json`
3. `shared/embeddings.py`  — embed terms with each model
4. Lenses in order: I → II → III → IV → V
