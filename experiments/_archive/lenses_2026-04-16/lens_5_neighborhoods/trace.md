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

### D5 — Neighborhood quality filter for false friends

**Options considered**:
- A: A priori candidate list (12 terms pre-labeled as likely false friends) —
  validates against human intuition (pro: external anchor; con: subjective,
  potentially biased by researcher's legal background, 0/12 hit rate suggests
  the list itself is flawed)
- B: Corpus frequency filter (exclude terms below a frequency threshold) —
  proxy for "model familiarity" (pro: simple; con: frequency data is not
  available for all models, frequency ≠ embedding quality)
- C: Neighborhood quality = mean cosine similarity to k-NN — direct measure
  of embedding density, computed per model per tradition, min of WEIRD/Sinic
  means as conservative aggregator (pro: data-driven, no external data needed,
  directly measures what matters; con: adds a hyperparameter for the cutoff
  percentile)

**Decision**: Option C. Cutoff at 25th percentile (Q1) of the quality
distribution across 397 core terms.

**Rationale**: The unfiltered ranking selects 27 terms with Jaccard = 0 at the
top. These fall into two categories: (a) terms the models barely "know" — sparse
neighborhoods, low cosine similarity to neighbors — where zero overlap is an
artifact of weak embeddings; (b) terms the models know well — dense, high-
similarity neighborhoods — where zero overlap reflects genuine cross-tradition
divergence. The quality metric separates these cases without requiring external
data or subjective labeling.

Results: cutoff = 0.6402 (25th percentile). 298/397 terms pass. 8 of 27 zero-
Jaccard terms excluded (quality too low = noise), 19 retained as genuine false
friends with quality ranging 0.65–0.78. Among these, quality serves as secondary
sort: "patent" (q=0.78) ranks highest because it is well-embedded in both
traditions yet placed in completely disjoint neighborhoods.

Sensitivity: the 25th percentile is conventional (Q1 of the IQR). Alternative
cutoffs at 10th percentile (less aggressive, 358 terms pass) and 50th percentile
(more aggressive, 199 terms pass) can be reported as robustness checks.

**Thesis text implication**: → §3.2.2 "To distinguish genuine semantic
divergence from embedding artifacts, we condition the false friend ranking on
neighborhood quality — defined as the mean cosine similarity between each term
and its k nearest neighbors, averaged within each tradition and aggregated
conservatively as min(quality_WEIRD, quality_Sinic). Terms below the 25th
percentile of this quality distribution are excluded: their neighborhoods are
too sparse to support meaningful overlap comparison. The remaining 298 terms
are ranked by ascending Jaccard similarity, with quality as tiebreaker."

---

### D-FF — False-friends polysemy validation (results, 2026-04-11, corrected)
**Status**: v1 run, v2 apples-to-apples run, corrected after adversarial review

The Pre-check 2 polysemy proxy of 2026-04-11 ran a synthetic-template
context-aggregation test on all 397 core terms and showed that the
headline Lens I result is essentially unchanged when the bare-term
representation is replaced by a multi-prototype mean. That test was a
*proxy*: it acted on all 397 core terms with templated variants, not on
the specific terms most likely to manifest the polysemy failure mode.
This decision documents **two** subsequent runs — v1 and v2 — designed
as progressively stronger versions of the polysemy challenge on the
exact 12 terms identified by Lens V §3.2.2 as having maximally
divergent neighbourhoods between traditions.

#### v1 — Query-side contextualisation only (`false_friends_polysemy.py`)

**Procedure**. For each (en, zh) record in the published top-12 false
friends, eight legally plausible templated variants were constructed
in each language and encoded with the matching tradition's three
models. Per (term, model), the eight context vectors were reduced to
a single query vector by four aggregators (bare, mean, medoid, first
PC). The query vector was looked up in the **existing bare-term**
9472-term pool.

**v1 results** (sanity check confirms the pipeline computes the
published baseline):

| Aggregator | mean cross J̄ | nonzero terms | strict-zero (max=0) |
|---|---|---|---|
| bare (sanity) | 0.000 | 0 / 12 | 12 / 12 |
| mean | 0.002 | 5 / 12 | 7 / 12 |
| medoid | 0.004 | 8 / 12 | 4 / 12 |
| first PC | 0.000 | 0 / 12 | 12 / 12 |

**v1 limitation — asymmetric comparison**. The v1 query is
contextualised ("the legal term patent") but the pool against which
it is compared is bare ("patent", "remand", ...). This asymmetry
means the v1 cannot distinguish "the contextualised representation
preserves the cross-tradition divergence" from "the contextualised
query finds different bare-term neighbours because a frase looks
different from a word when looked up in a pool of words". The v1
result should not be read as a vindication of §3.2.2 — the test has
a structural floor.

#### v2 — Apples-to-apples contextualised pool (`false_friends_polysemy_v2.py`)

The v1 limitation was addressed by building a fully contextualised
9472-term pool via `build_contextualized_pool.py`: each term in each
of the 6 models is encoded with the same 8 templated variants and
mean-aggregated to produce a (9472, dim) contextualised matrix per
model. Build cost: ~19 minutes on MPS. Stored in
`data/processed/embeddings_contextualized/`.

**Phase 2 sanity check** — reproducing §3.2.1 on the bare pool vs
recomputing on the ctx pool, for all 397 core terms:

| Quantity | published | bare recomputed | ctx pool |
|---|---|---|---|
| within-WEIRD J̄ | 0.258 | 0.2575 ✓ | 0.3738 |
| within-Sinic J̄ | 0.327 | 0.3265 ✓ | 0.4028 |
| cross J̄ | 0.088 | 0.0881 ✓ | 0.1008 |

The bare recomputation reproduces the published §3.2.1 numbers
exactly. The contextualised pool raises all Jaccards: within-WEIRD
by 45%, within-Sinic by 23%, cross by 15%. Within-tradition rises
more than cross, so the cross/within gap is *preserved* under
contextualisation.

**Phase 3 top-12 false friends, apples-to-apples**
(`results/false_friends_polysemy_v2.json`):

| Pool | mean cross J̄ | max single pair | nonzero / 12 |
|---|---|---|---|
| bare | 0.0000 | 0.0000 | 0 / 12 |
| ctx | 0.0042 | 0.0345 | 5 / 12 |

Five of the 12 published top false friends lift out of strict-zero
overlap under contextualisation: subrogation (0.012),
reprobation (0.023), joinder (0.008), divorcement (0.004),
remand (0.004). The lift is real but small in absolute magnitude —
roughly two orders of magnitude below the within-tradition J̄
baselines (0.37 W, 0.40 S on the ctx pool). The false-friend status
is not "erased" by contextualisation, but it is no longer the
strict-zero floor that the bare-pool baseline suggested.

#### Ranking analysis: is the top-12 identity representation-dependent?

This was the most important finding of the v2 run and was not
anticipated by the original Lens V design. Per-term mean cross J̄ was
computed for all 397 core terms on both the bare pool and the ctx
pool, and the two rankings were compared:

- **Global Spearman ρ (per-term J̄ bare vs ctx)**: 0.878 (p < 10⁻¹²⁷).
  The two rankings are strongly correlated. Terms with low cross J̄ on
  the bare pool tend to have low cross J̄ on the ctx pool too.
- **Top-12 overlap (bare top-12 ∩ ctx top-12)**: 3 / 12.
- **Top-20 overlap**: 7 / 20. Of the 12 published false friends, only
  5 remain in the top-20 of the contextualised ranking.

The **phenomenon** "there exists a population of legal terms whose
cross-tradition neighbour overlap is near-zero" is robust (global
ρ = 0.878, many terms tied near zero in both pools). The **identity**
"these specific 12 terms are the top false friends" is not robust
(3/12 top-12 overlap). The bare-pool ranking has many terms tied at
J̄ = 0 and the tiebreaker (neighbourhood quality score) selects 12
out of this tied population; under contextualisation some of those
tied terms lift off zero, and the ranking shuffles among the
remaining near-zero terms.

**Lessons for §3.2.2 in the thesis text**:

The §3.2.2 section should report:
- The existence of a population of terms with cross J̄ ≈ 0 is the
  robust phenomenon. This is preserved across representation.
- The specific top-12 is an artefact of the tiebreaker on the
  bare-pool ranking; under contextualisation only 3/12 survive in
  the top-12. The published top-12 must be described as
  "illustrative examples of the phenomenon under the bare-term
  representation", not as a canonical list.
- The within-tradition baselines shift substantially under
  contextualisation (0.258 → 0.374 W, 0.327 → 0.403 S), but the
  cross/within gap is preserved (within rises more than cross).
- Contextualisation acts as a "common substrate" that pulls all
  models closer together; the fact that the cross/within gap
  survives is the robustness claim for §3.2.2.

**What to drop from the v1 narrative**. The original "Lens V
strongly vindicated" language was produced *before* the adversarial
review identified the v1 asymmetry. That verdict is withdrawn. The
v2 apples-to-apples test replaces it.

**Output**:
- v1: `experiments/lens_5_neighborhoods/false_friends_polysemy.py`
  and its JSON/HTML (kept on disk as historical record, unlinked
  from the dashboard after Phase 1 rebuild)
- v2: `experiments/lens_5_neighborhoods/false_friends_polysemy_v2.py`
  and its JSON
- `experiments/lens_5_neighborhoods/build_contextualized_pool.py`
  (pool construction)
- `data/processed/embeddings_contextualized/` (ctx pool, 6 models)
- Dashboard surface after Phase 2: `#robustness-ctx` tab of the
  regenerated `lens5_interactive.html`, plus a robustness row in
  the `sec_3_2.html` section page and in the consolidated
  `robustness.html` appendix.

---

### D8 — Bilingual control and attested-context results (β + δ)

**Date**: 2026-04-13
**Decision**: Add bilingual control (BGE-M3, Qwen3-0.6B) and attested-context
embeddings (HK e-Legislation) to §3.2.1 neighborhood overlap analysis.
See Lens I trace D10/D11 for full rationale.

**Results — §3.2.1 k-NN Jaccard (k=15, n_perm=1000)**:

| Category | Bare J̄ | Attested J̄ | Δ |
|---|---|---|---|
| Within-WEIRD | 0.257 | 0.328 | +0.071 |
| Within-Sinic | 0.329 | 0.351 | +0.022 |
| Cross-tradition (mono) | 0.088 | 0.063 | −0.025 |
| Within-bilingual (β) | 0.103 | 0.073 | −0.030 |

Per-pair bilingual (bare): BGE-M3 EN×ZH J̄=0.108, Qwen3 EN×ZH J̄=0.098.
Per-pair bilingual (attested): BGE-M3 EN×ZH J̄=0.066, Qwen3 EN×ZH J̄=0.081.

**Interpretation** (level 2): At the neighborhood level, the divergence
pattern is even starker than at the RSA level. Bilingual J̄ (0.103 bare,
0.073 attested) is nearly identical to cross-mono J̄ (0.088 bare, 0.063
attested) — far from within-tradition J̄ (0.293 bare, 0.339 attested).
Even the same bilingual model organizes EN and ZH legal terms into almost
entirely different semantic neighborhoods. Attested contexts amplify this:
within-tradition neighborhoods become more coherent, cross-tradition
neighborhoods become more divergent.
**Limit** (level 3): Jaccard at k=15 is sensitive to the pool composition
(9472 terms). The background terms (~8900) dominate the neighbor pool; if
a domain-specific subset were used, results might differ. The bilingual
models may have been trained with cross-lingual alignment objectives that
artificially inflate their J̄ relative to a "true" cross-tradition baseline.

**Thesis text implication**: → §3.2.1 gains the same two-layer argument as
§3.1.4: (1) bilingual control confirms the gap is not architectural, (2)
attested contexts confirm it is not an artifact of synthetic framing. The
convergence of Lens I (global) and Lens V (local) on the same conclusion
strengthens the overall methodological argument.

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
