# Geometria Iuris — Experiment Overview

**Last updated**: 2026-03-13
**Audience**: Co-supervisor (ML engineering background)
**Purpose**: Navigate the `experiments/` folder, understand what each component does, and know the status of each experiment.

---

## 1. What this thesis measures

The thesis asks an epistemological question: *can embedding geometry serve as a legitimate instrument for measuring legal meaning?* The experimental design compares two legal traditions (WEIRD and Sinic) not as an end in itself, but to generate the variation needed to validate the measurement instrument.

The core hypothesis ("DigiVolksgeist"): sentence-embedding models trained predominantly on one linguistic community encode the normative practices of that community in the geometry of their vector space. If legal meaning has measurable geometric structure, models from different traditions should produce systematically different structures for the same set of legal concepts.

---

## 2. Model design: 3+3 symmetric slots

Six sentence-embedding models, three per tradition, selected to fill symmetric "slots":

| Slot | Role | WEIRD (EN) | Sinic (ZH) |
|------|------|-----------|------------|
| 1 | Architecture control (BGE family) | `BAAI/bge-large-en-v1.5` (1024d) | `BAAI/bge-large-zh-v1.5` (1024d) |
| 2 | STS-oriented | `intfloat/multilingual-e5-large` (1024d) | `GanymedeNil/text2vec-large-chinese` (1024d) |
| 3 | Legally-informed / domain | `nlpaueb/legal-bert-base-uncased` → FreeLaw-EN (768d) | `DMetaSoul/Dmeta-embedding-zh` (768d) |

This produces **9 cross-tradition pairs** (3 WEIRD × 3 Sinic) and **6 within-tradition pairs** (3 WEIRD-WEIRD + 3 Sinic-Sinic). A finding is considered a "tradition-level signal" if it holds in **≥7/9** cross-tradition pairs.

Config: `models/config.yaml`
Trace: `models/trace_model_selection.md`

---

## 3. Dataset

**Primary source**: Hong Kong Department of Justice Bilingual Legal Glossary (XML).
**No machine translation**: all EN↔ZH mappings are official government translations.

Three tiers:

| Tier | Count | Role | Domain assignment |
|------|-------|------|-------------------|
| **Core** | 397 | Primary analysis set across all experiments | Hand-labeled into 7 legal domains |
| **Background** | ~8,975 | Held-out validation for domain signal | k-NN majority vote (k=7) from core terms |
| **Control** | 100 | Non-legal baseline (Swadesh basic vocabulary) | N/A |

**7 legal domains**: constitutional, civil, criminal, administrative, international, labor_social, procedure.

Key files:
- `data/processed/legal_terms.json` — the built dataset (JSON, one object per term)
- `data/processed/embeddings/` — precomputed embedding matrices per model
- `data/build_dataset.py` — assembles `legal_terms.json` from parsed sources
- `data/parse_sources.py` — parses raw HK DOJ XML
- `data/trace_dataset_design.md` — all design decisions

---

## 4. Shared infrastructure

| File | What it does |
|------|-------------|
| `shared/embeddings.py` | Unified embedding client: loads any of the 6 models, caches results, handles tokenization differences |
| `shared/statistical.py` | Permutation tests (Mantel), block bootstrap CIs, row-resample bootstrap, effect sizes (rank-biserial r) |
| `shared/precompute.py` | Batch-precomputes all embeddings to `data/processed/embeddings/` |
| `shared/smoke_test.py` | Sanity checks: loads each model, embeds a few terms, verifies dimensions and L2 normalization |
| `shared/math_trace.md` | Step-by-step mathematical derivations referenced by experiment traces |

**Reproducibility note**: All computation runs on **CPU only** (not MPS/GPU). We discovered that Apple MPS produces non-deterministic results for the Chinese models when batch composition changes. The `--device cpu` flag is the enforced default. See `lens_3_stratigraphy/trace.md` for details.

---

## 5. Experiments — Status and methods

### Overview table

| Lens | Name | Thesis § | Status | Key output |
|------|------|----------|--------|-----------|
| **I** | Relational distance | §3.1 | **In progress** | `lens1_results.json`, RDMs, RSA matrices |
| **II** | Taxonomy | §4.4 | **Deferred** | — (future work) |
| **III** | Layer stratigraphy | §3.1.3 | **Active** | `lens3_results.json`, layer vectors, NTA |
| **IV** | Value axes | §3.3 | **Complete** (Phase 1-2) | `lens4_results.json`, projection scores |
| **V** | Neighborhoods | §3.2 | **Active** | `lens5_results.json`, Jaccard per term |

---

### Lens I — Relational distance structure (`lens_1_relational/`)

**Question**: Do legal embedding spaces have non-random relational structure, and is that structure shared across traditions?

**Method summary**:
1. Build a Representational Dissimilarity Matrix (RDM) per model: 397×397 cosine distance matrix over core terms.
2. **Domain signal** (§3.1.1): Split RDM entries into intra-domain vs inter-domain pairs. Mann-Whitney U tests whether intra < inter (i.e., same-domain terms are closer). Effect size: rank-biserial r.
3. **Legal vs control** (§3.1.1): Same test but splitting legal-legal vs legal-control distances.
4. **Domain topology** (§3.1.2): K×K inter-domain distance matrix (which domains are geometrically close?).
5. **RSA** (§3.1.4): Compare RDM upper triangles across models via Spearman ρ. Significance: Mantel permutation test (B=10,000). Uncertainty: block bootstrap CI (B=10,000) — resamples at the term level (not pair level) to respect the dependency structure where each term appears in N-1 pairs.

**Key decisions**:
- Cosine distance (scale-invariant across model dimensions)
- Spearman ρ for RSA (rank-based, robust to different distance distributions)
- Mantel test, not t-test (RDM entries are not i.i.d.)
- Background term assignment via k-NN majority vote (k=7) with confidence threshold 4/7

**Results directory**: `lens_1_relational/` contains `rdms/`, `distances/`, `distributions/`, `figures/`, `lens1_results.json`, `background_review.csv`.

**Status**: In progress — core analyses computed, writing up.

---

### Lens III — Layer stratigraphy (`lens_3_stratigraphy/`)

**Question**: At what depth inside the transformer does legal meaning crystallize?

**Method summary**:
Extract hidden states from all L+1 layers (embedding layer through final hidden state) for each of the 6 models. Compute 4 metrics:

*§3.1.3a — Single-term level:*
- **Drift**: cosine distance between consecutive layers for each term — how much does the representation change?
- **Jaccard** (k=7): neighborhood overlap between consecutive layers — does the term's local context shift?

*§3.1.3b — Structural level:*
- **Domain signal r**: Mann-Whitney rank-biserial r at each layer — at what depth does domain structure emerge?
- **RSA ρ**: Spearman correlation between each layer's RDM and the final layer's RDM — when does the relational structure converge to its final form?

*§3.1.3c — Neighborhood Trajectory Analysis (NTA):*
For 8 polysemous terms (negligence, sovereignty, corruption, comity, adoption, strike, disclosure, franchise), track the full k-NN list at every layer with entry/exit annotations. The neighbor pool includes both core legal terms (397) and control terms (100 Swadesh), allowing detection of the legal/non-legal boundary shift across depth.

**Falsification criteria**: (1) flat domain signal r → legal meaning is not depth-dependent; (2) RSA ρ ≈ 1.0 from layer 1 → depth adds nothing; (3) uniformly low drift → model barely transforms input.

**Layer extraction**: via `model[0].auto_model` with `output_hidden_states=True`, replicating native pooling (CLS or Mean) per model. Verified that `layers[:, -1, :]` matches precomputed final-layer embeddings within float32 tolerance.

**Model layer counts**: BGE-EN/ZH, E5, Text2vec = 24 layers (25 hidden states); FreeLaw = 22 (23); Dmeta = 12 (13).

**Results directory**: `lens_3_stratigraphy/` contains `layer_vectors/`, `figures/`, `lens3_results.json`, NTA exploration files.

**Status**: Active — layer extraction and metric computation done, NTA in progress.

---

### Lens IV — Value axis projection (`lens_4_values/`)

**Question**: On which value dimensions do the two traditions diverge?

**Method summary**:
1. **Axis construction** (§3.3.1): Three value axes — `individual_collective`, `rights_duties`, `public_private` — each built from 10 antonym pairs per language using the Kozlowski difference-vector method: axis = L2_normalize(mean(embed(pos_i) − embed(neg_i))). Antonym pairs are **independent per language** (no translation — each model builds its axis from its own language's conceptual resources).
2. **Inter-axis orthogonality** (§3.3.1): Cosine similarity between axis vectors as a diagnostic. Finding: `individual_collective` and `public_private` share moderate overlap (cos θ = 0.17–0.38), consistent with comparative law's "summa divisio." `rights_duties` is near-orthogonal to both (cos θ < 0.10).
3. **Cross-linguistic alignment** (§3.3.2): Project all 397 core terms onto each axis. Compare projections across model pairs via Spearman ρ + row-resample bootstrap CI (B=10,000). Row-resample (not block) because projection scores are per-term scalars, not pairwise distances — each term's score is independent given the axis.
4. **Axis divergence ranking** (§3.3.3): Kruskal-Wallis H test on cross-tradition ρ grouped by axis. Result: H=7.20, p=0.027. **Caveat**: the 9 cross-tradition ρ per axis derive from a 3×3 model grid with partial non-independence (pseudo-replication); the test is indicative, not confirmatory.

**Errata**: A duplicate antonym pair `[权利, 义务]` was found in two axes' ZH pairs, artificially inflating inter-axis cosine. Fixed by replacing with `[私心, 公心]`. Impact: BGE-ZH inter-axis cosine dropped 0.377 → 0.205; KW p-value went from 0.124 → 0.027.

**Results directory**: `lens_4_values/` contains `figures/`, `lens4_results.json`, `scores/`.

**Status**: Complete (Phase 1-2). Phase 3-4 (divergent term deep-dives) deferred.

---

### Lens V — Semantic neighborhoods (`lens_5_neighborhoods/`)

**Question**: Which legally aligned concepts (same term in EN and ZH) have divergent semantic neighborhoods — i.e., are "semantic false friends"?

**Method summary**:
1. For each of the 397 core terms in each model, find the k=15 nearest neighbors (cosine similarity) from the full embedding pool (~9,472 terms).
2. Compute Jaccard similarity J(N_EN(t), N_ZH(t)) for each term across each of the 9 cross-tradition model pairs. Divergence score = 1−J.
3. Rank terms by **mean Jaccard across all 9 cross-tradition pairs**. Top-20 = "false friend" candidates (terms whose WEIRD and Sinic neighborhoods share the fewest neighbors).
4. Global significance: permutation test on mean Jaccard (shuffles tradition labels).
5. Domain-level divergence (§3.2.3): aggregate per-domain to identify which legal domains diverge most.

**Key decisions**:
- k=15 (not k=7): ~0.2% of pool, better Jaccard resolution than k=7 (which gives intersections of only 2-3 terms)
- Jaccard (not RBO): with fixed-k neighborhoods, rank order within the set is less important than set membership
- Ranking, not thresholding: avoids arbitrary cutoffs; data-driven identification of false friends
- Normative decomposition (vector arithmetic) was considered and **dropped** — not mapped to any thesis index section

**Results directory**: `lens_5_neighborhoods/` contains `figures/`, `jaccard_per_term/`, `lens5_results.json`.

**Status**: Active — Jaccard computation done, domain-level aggregation and false friend narratives in progress.

---

### Lens II — Taxonomy (`lens_2_taxonomy/`)

**Status**: Deferred to future work (§4.4 Horizons). The trace file exists but all decisions are empty. This experiment would have explored emergent taxonomic structure via clustering.

---

## 6. Statistical toolkit summary

All resampling procedures use B=10,000 iterations (Nili et al. 2014).

| Method | Used in | Purpose |
|--------|---------|---------|
| Mann-Whitney U + rank-biserial r | Lens I (§3.1.1) | Domain signal, legal vs control |
| Mantel permutation test (B=10k) | Lens I (§3.1.4) | RSA significance (respects RDM dependency) |
| Block bootstrap CI (B=10k) | Lens I (§3.1.4) | RSA uncertainty (resamples terms, not pairs) |
| Row-resample bootstrap CI (B=10k) | Lens IV (§3.3.2) | Spearman ρ CI for projection scores (per-term scalars → independent) |
| Kruskal-Wallis H | Lens IV (§3.3.3) | Compare cross-tradition ρ across axes |
| Permutation test on mean Jaccard | Lens V (§3.2.1) | Global neighborhood divergence significance |

**Why block bootstrap in Lens I but row-resample in Lens IV?** In Lens I, the unit of analysis is term-*pair* distances (RDM cells): each term contributes to N-1 cells, creating dependency → resample at the term level (block). In Lens IV, projection scores are per-term scalars: score_i = cos(embed(term_i), axis) → each score is independent given the axis → standard row-resample is correct and sufficient.

---

## 7. How to reproduce

```bash
# 1. Install dependencies
pip install -r experiments/requirements.txt

# 2. Parse sources and build dataset
python experiments/data/parse_sources.py
python experiments/data/build_dataset.py

# 3. Precompute embeddings (CPU only — critical for reproducibility)
python experiments/shared/precompute.py --device cpu

# 4. Smoke test
python experiments/shared/smoke_test.py

# 5. Run experiments (order: Lens I → III → V → IV)
# Each lens has its own run script in its directory.
```

Precomputed results are already cached in each lens's results directory. To verify without re-running, inspect the `*_results.json` files.

---

## 8. Key files quick reference

| What you're looking for | Where to find it |
|------------------------|-----------------|
| Dataset (terms + domains + translations) | `data/processed/legal_terms.json` |
| Precomputed embeddings | `data/processed/embeddings/` |
| Model configuration | `models/config.yaml` |
| All design decisions for experiment X | `lens_X_*/trace.md` |
| Mathematical derivations | `shared/math_trace.md` |
| Statistical utilities | `shared/statistical.py` |
| Jupyter exploration notebooks | `notebooks/` |
| Thesis chapter mapping | `README.md` (experiment map table) |

---

## 9. Notation conventions

- **WEIRD**: Western, Educated, Industrialized, Rich, Democratic — the 3 English-trained models
- **Sinic**: the 3 Chinese-trained models
- **Core terms**: the 397 hand-labeled legal terms (primary analysis set)
- **RDM**: Representational Dissimilarity Matrix (N×N cosine distances)
- **RSA**: Representational Similarity Analysis (Spearman ρ between RDM upper triangles)
- **ρ**: Spearman rank correlation coefficient
- **r**: rank-biserial correlation (effect size for Mann-Whitney U)
- **J**: Jaccard similarity
- **B**: number of bootstrap/permutation resamples (always 10,000)
