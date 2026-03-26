# Trace: Lens V — Semantic neighborhoods

**Thesis chapter(s)**: Ch.3 §3.2 — Semantic neighborhoods (micro)
**Date**: 2026-02-27
**Status**: active

---

## Context

Lens V asks which legally aligned concepts (same term in EN and ZH via DOJ
glossary) have divergent semantic neighborhoods. A concept is a "semantic false
friend" when its nearest neighbors in the WEIRD space differ substantially from
its nearest neighbors in the Sinic space.

This is the micro-scale complement to Lens I (§3.1, macro: relational structure
of entire traditions). Lens I measures global geometry preservation; Lens V
measures per-concept neighborhood divergence.

Three thesis subsections:
- **§3.2.1**: Neighborhood overlap — when the same concept has different neighbors
- **§3.2.2**: Juridical false friends — apparent convergence, structural divergence
- **§3.2.3**: Which domains diverge most?

---

## Decision log

### D1 — k (neighborhood size)

**Options considered**:
- k=7: consistent with Lens I/III k-NN, but only 0.07% of 9,472-term pool →
  very low Jaccard resolution (union capped at 14)
- k=15: 0.16% of pool, union up to 30, better Jaccard granularity, matches
  archive `exp_nda.py` which was validated in early pipeline runs
- k=25: ~0.26% of pool, inflates overlap artificially by pulling in increasingly
  distant neighbors, dilutes the "core neighborhood" signal

**Decision**: k=15

**Rationale**: k=15 provides sufficient resolution for Jaccard similarity while
remaining defensible as a small fraction (~0.2%) of the 9,472-term pool. The
choice is consistent with the NDA experiments in the archive pipeline (v3.2)
which used k=15 and produced interpretable divergence scores. k=7 would be too
sparse for meaningful Jaccard computation (intersection rarely > 2–3 terms),
while k=25 would inflate overlap beyond the "nearest semantic neighbors" concept.

**Thesis text implication**: → §3.2.1 "We define the semantic neighborhood of
a concept as its k=15 nearest neighbors in cosine similarity space, representing
approximately 0.2% of the embedding pool. This neighborhood size provides
sufficient resolution for Jaccard similarity measurement while remaining small
enough to capture genuine semantic proximity."

---

### D2 — Overlap metric

**Options considered**:
- Jaccard similarity J(A,B) = |A∩B| / |A∪B|: standard set-overlap metric,
  bounded [0,1], symmetric, well-understood in NLP literature (pro: simplicity,
  interpretability; con: does not capture rank order within neighborhood)
- Overlap coefficient |A∩B| / min(|A|,|B|): asymmetric, degenerates to 1 when
  one set ⊂ other; inappropriate for equal-size neighborhoods
- Rank-Biased Overlap (RBO): captures rank ordering, parameter-dependent (p),
  adds complexity without clear empirical advantage for fixed-k neighborhoods

**Decision**: Jaccard similarity

**Rationale**: With fixed k=15, both neighborhoods have the same cardinality,
making Jaccard and overlap coefficient functionally different (Jaccard penalizes
non-overlap more). Jaccard is the standard choice in neighborhood-based NLP
analyses (Hamilton et al. 2016 on semantic shift; Dubossarsky et al. 2017).
RBO would add a tuning parameter (p) without clear benefit since we are not
interested in rank precision within the neighborhood — only in set membership.

Report both J (overlap) and 1−J (divergence score) for readability.

**Thesis text implication**: → §3.2.1 "Neighborhood overlap is quantified by
Jaccard similarity J(N_EN(t), N_ZH(t)) = |N_EN ∩ N_ZH| / |N_EN ∪ N_ZH|.
The complement 1−J serves as a divergence score: 1−J = 0 indicates identical
neighborhoods, 1−J = 1 indicates completely disjoint neighborhoods."

---

### D3 — Divergence threshold / ranking method

**Options considered**:
- Fixed threshold (J < 0.2 = "divergent"): arbitrary, not data-driven, threshold
  choice would need separate justification
- Ranking by mean cross-tradition Jaccard: data-driven, no distributional
  assumptions, produces an ordered list of false friend candidates
- Z-score relative to overall Jaccard distribution: assumes normality of Jaccard
  values, which is unlikely given bounded [0,1] distribution

**Decision**: Rank by mean Jaccard across 9 cross-tradition pairs (3 WEIRD × 3
Sinic). Top-20 = false friend candidates. Global significance via permutation
test on mean Jaccard (not per-term threshold).

**Rationale**: The ranking approach avoids arbitrary thresholds and produces a
directly usable list of the most divergent concepts. Using the mean across all 9
cross-tradition pairs (rather than a single pair) increases robustness: a concept
must diverge consistently across model pairs to rank high. The permutation test
establishes that the observed cross-tradition Jaccard is significantly lower than
chance (basic validity check), without requiring per-term multiple testing.

**Thesis text implication**: → §3.2.2 "We identify false friend candidates by
ranking terms by their mean Jaccard similarity across all 9 cross-tradition
model pairs. The top-20 most divergent terms — those whose WEIRD and Sinic
neighborhoods share the fewest neighbors — constitute our empirical false
friend list. This data-driven ranking avoids arbitrary divergence thresholds."

---

### D4 — Normative decomposition method (Law − State = ?)

**Options considered**:
- Vector arithmetic decompositions (e.g., Law − State, Rights − Collective) to
  reveal latent normative dimensions in the embedding space
- Drop entirely: thesis index §3.2 has three subsections (§3.2.1 overlap,
  §3.2.2 false friends, §3.2.3 domain divergence) — none maps to normative
  decomposition via vector arithmetic

**Decision**: Dropped.

**Rationale**: The thesis index (source of truth: `003_GeometriaIuris_Indice.docx`)
does not contain a subsection for vector arithmetic decompositions under §3.2.
Introducing an unmapped experiment would violate the project rule "Non introdurre
esperimenti o sezioni che non compaiano nell'indice." If normative decomposition
proves valuable, it belongs in a future revision of the index.

**Thesis text implication**: No text generated. This decision is a scope boundary:
§3.2 covers neighborhood overlap, false friends, and domain-level divergence only.

---

## Open questions

- None at this stage.

---

## References

- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word
  embeddings reveal statistical laws of semantic change. ACL 2016.
- Dubossarsky, H., Weinshall, D., & Grossman, E. (2017). Outta control:
  Laws of semantic change and inherent biases in word representation models.
  EMNLP 2017.
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
  zero: Calculating exact P-values when permutations are randomly drawn.
  Statistical Applications in Genetics and Molecular Biology, 9(1).
