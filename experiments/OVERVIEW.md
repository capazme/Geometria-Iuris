# Geometria Iuris — Experiment Overview

**Last updated**: 2026-04-16
**Audience**: Co-supervisor (ML engineering background)
**Purpose**: Navigate the `experiments/` folder, understand what each component does, and know the status of each experiment.

> **2026-04-16 — scope pivot.** The experimental apparatus has been reduced
> from five lenses to two (Lens I + Lens IV). Lenses II, III, V, VI are
> archived under `_archive/lenses_2026-04-16/` and remain consultable for
> appendix use. The reasoning is recorded in `trace_pivot_2lens.md`.
> The state described below reflects the reduced design; the archived
> lenses retain their own internal documentation where they now live.

---

## 1. What this thesis measures

The thesis asks an epistemological question: *can embedding geometry serve as a legitimate instrument for measuring legal meaning?* The experimental design compares two legal traditions (WEIRD and Sinic) not as an end in itself, but to generate the variation needed to validate the measurement instrument.

The core hypothesis ("DigiVolksgeist"): sentence-embedding models trained predominantly on one linguistic community encode the normative practices of that community in the geometry of their vector space. If legal meaning has measurable geometric structure, models from different traditions should produce systematically different structures for the same set of legal concepts.

The apparatus is reduced to two lenses, selected for *transparency of
methodological choices*: every arbitrary parameter in a retained
experiment is either literature-standard (Lens I) or a doctrinal choice
declared by the jurist (Lens IV). This excludes experiments whose output
depends on machine-set hyperparameters without doctrinal anchor.

---

## 2. Model design: 3+3 symmetric slots + 2 bilingual controls

Eight sentence-embedding models: six monolingual (three per tradition) plus two bilingual controls added in the Phase 2 upgrade (β).

| Slot | Role | WEIRD (EN) | Sinic (ZH) |
|------|------|-----------|------------|
| 1 | Architecture control (BGE family) | `BAAI/bge-large-en-v1.5` (1024d) | `BAAI/bge-large-zh-v1.5` (1024d) |
| 2 | STS-oriented | `intfloat/e5-large-v2` (1024d) | `GanymedeNil/text2vec-large-chinese` (1024d) |
| 3 | Legally-informed / domain | `freelawproject/modernbert-embed-base_finetune_512` (768d) | `DMetaSoul/Dmeta-embedding-zh` (768d) |
| β1 | Bilingual control (BGE family) | — | `BAAI/bge-m3` (1024d, 560M) |
| β2 | Bilingual control (Qwen family) | — | `Qwen/Qwen3-Embedding-0.6B` (1024d, 0.6B) |

The 3+3 monolingual panel produces **9 cross-tradition pairs** and
**6 within-tradition pairs**. The two bilingual models each encode both
languages in a shared space, producing **2 within-bilingual pairs** that
function as a causal control: if the cross-tradition gap persists
within a bilingual model, it is not an encoder-architecture artefact.

A finding is considered a "tradition-level signal" if it holds in **≥7/9** monolingual cross-tradition pairs *and* is confirmed by ≥1 of the 2 bilingual controls.

Config: `models/config.yaml`
Trace: `models/trace_model_selection.md`

---

## 3. Dataset

**Primary source**: Hong Kong Department of Justice Bilingual Legal Glossary (XML), enriched with section-level attested contexts from the HK e-Legislation corpus (DATA.GOV.HK).
**No machine translation**: all EN↔ZH mappings are official government translations.

Three tiers:

| Tier | Count | Role | Domain assignment |
|------|-------|------|-------------------|
| **Core** | 397 → ~430 target | Primary analysis set across both lenses | Hand-labeled into 7 legal domains |
| **Background** | ~8,975 | Held-out validation for domain signal | k-NN majority vote (k=7) from core terms |
| **Control** | 100 | Non-legal baseline (Swadesh basic vocabulary) | N/A |

The administrative-law domain is currently under-represented (12 core terms); Phase 2 identified 125 keyword-matched promotion candidates in the background pool. Target post-promotion: ~45 administrative terms. See `trace_pivot_2lens.md` D5.

**7 legal domains**: constitutional, civil, criminal, administrative, international, labor_social, procedure.

Key files:
- `data/processed/legal_terms.json` — the built dataset (JSON, one object per term)
- `data/processed/embeddings/` — precomputed bare-template embeddings per model
- `data/processed/embeddings_contextualized/` — attested-context embeddings (from e-Legislation)
- `data/build_dataset.py` — assembles `legal_terms.json`
- `data/parse_sources.py` — parses raw HK DOJ XML
- `data/parse_elegislation.py` — parses bulk e-Legislation XML (65,599 sections)
- `data/build_term_contexts.py` — builds term→contexts index
- `data/trace_dataset_design.md` — all design decisions

---

## 4. Shared infrastructure

| File | What it does |
|------|-------------|
| `shared/embeddings.py` | Unified embedding client: loads any of the 8 models (6 base + 2 bilingual), caches results, handles tokenization differences |
| `shared/statistical.py` | Permutation tests (Mantel), block bootstrap CIs, row-resample bootstrap, effect sizes (rank-biserial r) |
| `shared/precompute.py` | Batch-precomputes all embeddings, supports bilingual model expansion and `--unload-between` |
| `shared/smoke_test.py` | Sanity checks: loads each model, embeds a few terms, verifies dimensions and L2 normalization |
| `shared/math_trace.md` | Step-by-step mathematical derivations referenced by experiment traces |

**Reproducibility note**: All computation runs on **CPU only** (not MPS/GPU) for deterministic cross-platform results. The Apple MPS backend produces non-deterministic outputs for the Chinese models when batch composition changes; `--device cpu` is the enforced default.

---

## 5. Experiments — status and methods

### Overview table

| Lens | Name | Thesis § | Status | Key output |
|------|------|----------|--------|-----------|
| **I** | Relational distance | §3.1 | **Active — redesign for v2 pending** | `lens1_results.json`, RDMs, RSA matrices |
| **IV** | Value axes | §3.3 | **Active — redesign for v2 pending** | `lens4_results.json`, projection scores |

Archived (see `_archive/lenses_2026-04-16/`):
- **Lens II** — taxonomy (was §4.4 horizons, already deferred)
- **Lens III** — layer stratigraphy (was §3.1.3)
- **Lens V** — neighborhoods / false friends (was §3.2)
- **Lens VI** — Sparse Autoencoder mechanistic decomposition (Phase 2 γ addition)

---

### Lens I — Relational distance structure (`lens_1_relational/`)

**Question**: Does the instrument detect non-random relational structure in legal embedding spaces, reliably across independently trained models, with discriminating power under known variation (cross-tradition, monolingual vs bilingual control, bare vs attested context)?

**Method summary**:
1. Build a Representational Dissimilarity Matrix (RDM) per model: N×N cosine distance matrix over core terms.
2. **Domain signal** (§3.1.1): Split RDM entries into intra-domain vs inter-domain pairs. Mann-Whitney U tests whether intra < inter. Effect size: rank-biserial r.
3. **Legal vs control** (§3.1.1): Same test but splitting legal-legal vs legal-control distances.
4. **Domain topology** (§3.1.2): K×K inter-domain distance matrix.
5. **RSA** (§3.1.4): Compare RDM upper triangles across models via Spearman ρ. Significance: Mantel permutation test (B=10,000). Uncertainty: block bootstrap CI (B=10,000) at term level.
6. **Causal control** (§3.1.4): bilingual models BGE-M3 and Qwen3 encode both languages in a shared space; their EN×ZH correlation isolates the gap attributable to legal semantics rather than encoder architecture.
7. **Attested-context robustness** (§3.1.4): re-run with term vectors mean-aggregated from attested e-Legislation passages instead of synthetic templates.
8. **Parametric categorical probe** (§3.1.5): 11-category ordinal sequences with legal thresholds (age × imputability, age × contractual capacity, disposal severity), pre-registered expected break positions, modal max-gap detection on PC1. Determinate/indeterminate disposal is the clean positive result (4/6 models recover the break exactly).

**Status**: Active — core analyses computed on the pre-pivot design; redesign for v2 (integrated bilingual + attested headline, expanded dataset, de-coupling from archived Lens III) pending per `trace_pivot_2lens.md` D3.

---

### Lens IV — Value axis projection (`lens_4_values/`)

**Question**: On which doctrinally-defined value dimensions do the two traditions' models produce divergent orderings of legal concepts?

**Method summary**:
1. **Axis construction** (§3.3.1): doctrinally chosen axes, each built from ≥10 antonym pairs per language using the Kozlowski difference-vector method: axis = L2_normalize(mean(embed(pos_i) − embed(neg_i))). Pole-pairs are **independent per language** (no translation — each model builds its axis from its own language's conceptual resources). Current axes: `individual_collective`, `rights_duties`, `public_private`. Expansion to ~6 axes planned under `trace_pivot_2lens.md` D4.
2. **Inter-axis orthogonality diagnostic** (§3.3.1): cosine similarity between axis vectors. Natural correlation is permitted (Sacco's summa divisio); Gram-Schmidt orthogonalisation rejected as it would destroy the axes' semantic content.
3. **Cross-linguistic alignment** (§3.3.2): project all core terms onto each axis. Compare projections across model pairs via Spearman ρ + row-resample bootstrap CI (B=10,000). Row-resample (not block) because projection scores are per-term scalars, not pairwise distances.
4. **Axis divergence ranking** (§3.3.3): descriptive means and bootstrap CIs. The Kruskal-Wallis omnibus test was removed as statistically invalid (pseudo-replication + non-orthogonal axes, see Lens IV trace D7 revised).

**Current results (pre-pivot, 6 base monolingual models only)**:
| Axis | Cross-tradition ρ̄ | Within ρ̄ |
|------|---------------------|-----------|
| public_private | 0.402 | higher |
| rights_duties | 0.380 | higher |
| individual_collective | 0.292 | higher |

The gradient (public/private most aligned, individual/collective least aligned) matches the comparative-law intuition that more culturally-loaded axes show larger cross-tradition divergence — but this remains a level-2 interpretation pending v2 redesign.

**Status**: Active — redesign for v2 (bilingual control integration, attested-context projection, axis expansion, pole-pair audit) pending per `trace_pivot_2lens.md` D4.

---

## 6. Statistical toolkit summary

All resampling procedures use B=10,000 iterations (Nili et al. 2014).

| Method | Used in | Purpose |
|--------|---------|---------|
| Mann-Whitney U + rank-biserial r | Lens I (§3.1.1) | Domain signal, legal vs control |
| Mantel permutation test (B=10k) | Lens I (§3.1.4) | RSA significance (respects RDM dependency) |
| Block bootstrap CI (B=10k) | Lens I (§3.1.4) | RSA uncertainty (resamples terms, not pairs) |
| Row-resample bootstrap CI (B=10k) | Lens IV (§3.3.2) | Spearman ρ CI for projection scores (per-term scalars → independent) |
| Permutation test on group labels | Lens IV (§3.3.2) | Cross vs within separation with small N |
| Holm-Bonferroni correction | Both lenses | Multiple-comparison control across model pairs and (for Lens IV) axes |

**Why block bootstrap in Lens I but row-resample in Lens IV?** In Lens I, the unit of analysis is term-*pair* distances (RDM cells): each term contributes to N-1 cells, creating dependency → resample at the term level (block). In Lens IV, projection scores are per-term scalars → each score is independent given the axis → standard row-resample is correct and sufficient.

---

## 7. How to reproduce (v2 pipeline, expected)

```bash
# 1. Install dependencies
pip install -r experiments/requirements.txt

# 2. Parse sources and build dataset
python experiments/data/parse_sources.py
python experiments/data/parse_elegislation.py
python experiments/data/build_dataset.py
python experiments/data/build_term_contexts.py

# 3. Precompute embeddings (CPU only)
python experiments/shared/precompute.py --device cpu
python experiments/shared/precompute.py --device cpu --attested

# 4. Smoke test
python experiments/shared/smoke_test.py

# 5. Run the two lenses
python experiments/lens_1_relational/lens1.py
python experiments/lens_4_values/lens4.py
```

Precomputed results from the pre-pivot run are cached in each lens's `results/` directory and remain available until v2 rerun.

---

## 8. Key files quick reference

| What you're looking for | Where to find it |
|------------------------|-----------------|
| Pivot decision (2-lens redesign) | `trace_pivot_2lens.md` |
| Dataset (terms + domains + translations) | `data/processed/legal_terms.json` |
| Precomputed embeddings | `data/processed/embeddings/` and `embeddings_contextualized/` |
| Model configuration | `models/config.yaml` |
| All design decisions for Lens I | `lens_1_relational/trace.md` |
| All design decisions for Lens IV | `lens_4_values/trace.md` |
| Mathematical derivations | `shared/math_trace.md` |
| Statistical utilities | `shared/statistical.py` |
| Archived lenses (II, III, V, VI) | `_archive/lenses_2026-04-16/` |

---

## 9. Notation conventions

- **WEIRD**: Western, Educated, Industrialized, Rich, Democratic — the 3 English-trained models
- **Sinic**: the 3 Chinese-trained models
- **Bilingual**: the 2 encoders trained on both EN and ZH (BGE-M3, Qwen3)
- **Core terms**: the hand-labeled legal terms (primary analysis set)
- **RDM**: Representational Dissimilarity Matrix (N×N cosine distances)
- **RSA**: Representational Similarity Analysis (Spearman ρ between RDM upper triangles)
- **ρ**: Spearman rank correlation coefficient
- **r**: rank-biserial correlation (effect size for Mann-Whitney U)
- **B**: number of bootstrap/permutation resamples (always 10,000)
