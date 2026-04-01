# Trace: Lens II — Emergent taxonomy

**Thesis chapter(s)**: Ch.4 §4.4 — Horizons
**Date**: 2026-04-01
**Status**: active

---

## Context

Lens II asks whether the geometric structure of embedding spaces spontaneously
organizes legal vocabulary into categories that correspond to human legal
taxonomy, and whether those emergent categories are shared across cultural
traditions.

This is the "horizon" lens: a compact, complete experiment that extends the
instrument's reach into taxonomic structure. The 397 core terms carry human
domain labels (7 domains: administrative, civil, constitutional, criminal,
international, labor_social, procedure). The question is whether agglomerative
clustering on each model's RDM recovers these labels.

Three thesis subsections:
- **§4.4.1**: Taxonomic recovery — FM at k=7 vs human labels
- **§4.4.2**: Taxonomic horizons — FM(k) curves, k=2..20
- **§4.4.3**: Cross-tradition taxonomic agreement — FM between WEIRD and Sinic

---

## Decision log

### D1 — Clustering algorithm

**Options considered**:
- Ward linkage: minimizes within-cluster variance, produces compact clusters
  (pro: standard default; con: mathematically requires Euclidean distances —
  the Lance-Williams update formula assumes variance minimization under L2.
  scipy does not raise an error when applied to cosine distances but produces
  incorrect dendrograms)
- Average linkage (UPGMA): arithmetic mean of all inter-cluster pairwise
  distances (pro: valid for any metric, standard in biological taxonomy,
  consistent with cosine-distance RDM; con: slightly less crisp clusters
  than Ward on Euclidean data)
- Complete linkage: maximum inter-cluster distance (pro: also metric-free,
  conservative; con: more sensitive to outliers)

**Decision**: Average linkage as primary; complete linkage computed for
robustness (stored but not tested).

**Rationale**: The RDMs are cosine distances (1 − cos_sim), stored as the
upper triangle of a 397×397 symmetric matrix. Ward linkage silently accepts
non-Euclidean condensed distance matrices in scipy but applies an update
formula that assumes Euclidean variance minimization — producing structurally
wrong dendrograms. Average linkage is metric-free and is the standard choice
in taxonomy-recovery analyses (Sokal & Michener 1958). Complete linkage serves
as a robustness check.

**Thesis text implication**: → §4.4.1 "Agglomerative hierarchical clustering
with average linkage (UPGMA) is applied to each model's cosine-distance matrix.
Ward linkage, though commonly used, was excluded because it requires Euclidean
distances — a condition violated by cosine distance."

---

### D2 — Number of clusters / resolution parameter

**Options considered**:
- Fixed k=7 only: matches the 7 human domains (pro: cleanest comparison;
  con: misses the "horizon" question)
- k=7 anchor + FM curves k=2..20: answers both "does the model recover
  human taxonomy?" and "at which granularity does it best align?" (pro:
  the curve IS the horizon; con: 19 k-values per model, but negligible
  compute time)
- k=2..50 with elbow detection: broader range (pro: thorough; con: beyond
  k=20 clusters become very small with 397 terms, FM unstable)

**Decision**: k=7 as primary anchor (§4.4.1); FM(k) curves over k=2..20
(§4.4.2).

**Rationale**: k=7 is the natural comparator — the number of human domain
labels. The FM(k) curve reveals whether the embedding space "prefers" a
different granularity. If FM peaks at k≠7, this is evidence that the geometry
encodes a different taxonomic structure than the human classification. The
range 2–20 covers all interpretable partitions (k=2 for binary splits, k=7
for domains, k=14 for sub-domains).

**Thesis text implication**: → §4.4.2 "The FM(k) curve reveals the taxonomic
horizon: the granularity at which the geometric partition best aligns with
human legal categories. A peak at k=7 would confirm that the instrument's
natural resolution matches the human taxonomy; a peak elsewhere would suggest
the geometry encodes a different organizational structure."

---

### D3 — Agreement metric with human classification

**Options considered**:
- Adjusted Rand Index (ARI): chance-corrected, range [-1,1] (pro: standard;
  con: negative values counterintuitive, correction depends on hypergeometric
  model assumptions)
- Fowlkes-Mallows Index: FM = TP / sqrt((TP+FP)(TP+FN)), range [0,1] (pro:
  interpretable as geometric mean of co-cluster precision and recall, specified
  by TechnicalOverview; con: not chance-corrected analytically, but permutation
  test provides empirical null)
- NMI: information-theoretic (pro: handles different k; con: logarithmic
  scale, less intuitive for legal audience)

**Decision**: Fowlkes-Mallows Index with permutation test (n_perm=1000).

**Rationale**: FM is directly interpretable: FM=1 means every pair of terms
that shares a human domain label also shares a model cluster (and vice versa).
The permutation test (shuffle human labels, recompute FM) provides the
empirical null distribution, making analytical chance correction unnecessary.
Implemented via numpy outer-product broadcasting (no sklearn dependency).

**Thesis text implication**: → §4.4.1 "Agreement between model partition and
human labels is measured by the Fowlkes-Mallows index, which quantifies the
geometric mean of pairwise co-assignment precision and recall. Statistical
significance is assessed by permutation: human labels are randomly shuffled
1,000 times, generating an empirical null distribution of FM values."

---

### D4 — Cross-tradition comparison method

**Options considered**:
- FM between WEIRD and Sinic partitions at k=7 for all 9 pairs, compared
  to within-tradition FM (3+3 pairs) via Mann-Whitney U (pro: directly
  parallel to Lens I RSA cross-tradition design; con: small sample n=9 vs n=6)
- Cophenetic correlation between WEIRD and Sinic dendrograms (pro: compares
  full hierarchical structure; con: complex, loses the FM(k) curve narrative)
- FM(k) curves as visual comparison only (pro: rich narrative; con: no
  inferential claim)

**Decision**: FM at k=7 for all 15 pairs with Mann-Whitney test (§4.4.3),
plus FM(k) curves as visual "horizons" narrative (§4.4.2).

**Rationale**: The Mann-Whitney test on 9 cross vs 6 within FM values is
the same statistical framework as Lens I §3.1.4 and Lens IV §3.3.3. The
FM(k) curves provide the visual narrative for §4.4.2 without requiring a
per-k statistical test. Cross-tradition FM values at k=7 are expected to be
substantially higher than human-vs-model FM (~0.45), suggesting the models
agree more with each other than with human taxonomy.

**Thesis text implication**: → §4.4.3 "The geometric taxonomies of WEIRD and
Sinic models are far more similar to each other (FM ≈ 0.85) than either is
to the human domain classification (FM ≈ 0.45). This convergence suggests
that embedding models trained on distinct corpora develop a shared geometric
organization of legal vocabulary that diverges from expert categorical
knowledge."

---

## Open questions

- None at this stage.

---

## References

- Fowlkes, E. B., & Mallows, C. L. (1983). A method for comparing two
  hierarchical clusterings. JASA 78(383), 553–569.
- Sokal, R. R., & Michener, C. D. (1958). A statistical method for evaluating
  systematic relationships. University of Kansas Science Bulletin 38, 1409–1438.
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
  zero. Statistical Applications in Genetics and Molecular Biology 9(1).
