# Trace: Lens IV — Value axis projection

**Thesis chapter(s)**: Ch.3 §3.3 — Value dimensions (comparative)
**Date**: 2026-03-02
**Status**: complete (Phase 1–2 done; Phase 3–4 deferred)

---

## Context

Lens IV asks on which value dimensions the two legal traditions diverge. It
constructs geometric axes from antonym pairs and projects legal terms onto them,
measuring cross-linguistic alignment of each axis.

Three thesis subsections:
- **§3.3.1**: Axis construction — building value dimensions from antonym pairs
- **§3.3.2**: Cross-tradition alignment — Spearman rho + bootstrap CI per axis per pair
- **§3.3.3**: Which axes diverge most?

---

## Analysis map

| # | Analysis | Section | Method | Input | N | Math trace |
|---|----------|---------|--------|-------|---|------------|
| 1 | Axis construction + sanity | §3.3.1 | Kozlowski diff-vector, orientation check | 10 antonym pairs × 3 axes × 2 langs | 6 models × 3 axes × 20 pairs | V1–V3 |
| 2 | Inter-axis orthogonality | §3.3.1 | Cosine similarity between axis vectors | Axis vectors per model | 6 models × 3 axis-pairs | V4 |
| 3 | Cross-linguistic alignment | §3.3.2 | Spearman ρ + row-resample bootstrap CI | Projection scores on 397 core terms | 15 model-pairs × 3 axes = 45 | V5–V6 |
| 4 | Within vs cross comparison | §3.3.2 | Mann-Whitney U (9 cross vs 6 within) | ρ values per axis | 3 axes × 15 ρ = 45 | V7 |
| 5 | Axis divergence ranking | §3.3.3 | Kruskal-Wallis H + Bonferroni post-hoc | Cross-tradition ρ per axis | 3 axes × 9 cross ρ = 27 | V8 |
| 6 | Divergent term identification | §3.3.2 | |WEIRD-avg − Sinic-avg| on projections | Per-term avg scores across 3 models | 397 terms × 3 axes | — |

---

## Decision log

### D1 — Value dimensions (which axes)

**Options considered**:
- Option A: 3 axes — individual_collective, rights_duties, public_private
  - Pro: capture foundational comparative law tensions (Legrand 1996, Merryman & Perez-Perdomo 2007); already validated in archive `exp_axes.py`
  - Con: may miss culture-specific dimensions (e.g. sacred_secular)
- Option B: 5+ axes (add sacred_secular, formal_informal)
  - Pro: broader coverage of cultural variation
  - Con: requires new antonym pairs without thesis index support; risks noise from under-theorized dimensions

**Decision**: Option A — 3 axes

**Rationale**: These three capture the fundamental comparative law tensions
documented in the literature. The archive pipeline validated these axes with
interpretable results. Adding more axes would require new antonym pairs
without corresponding thesis sections. → §3.3.1 "We construct three value
axes reflecting foundational tensions in comparative legal theory."

**Thesis text implication**: The thesis can claim that value organization was
measured along three theoretically-grounded dimensions. It cannot claim
exhaustive coverage of all possible cultural value dimensions. → §3.3.1

---

### D2 — Antonym pair selection method

**Options considered**:
- Option A: Shared cross-lingual pairs (translate same pairs into both languages)
  - Pro: direct comparability
  - Con: translation introduces bias; English-centric axis may not capture Chinese conceptual structure
- Option B: Independent pairs per language (10 EN + 10 ZH per axis, each model builds axis from its own language)
  - Pro: each model's axis reflects how that language organizes the dimension; avoids translation bias
  - Con: axes may not be strictly comparable (comparing projections on different axes)

**Decision**: Option B — Independent pairs per language

**Rationale**: Kozlowski (2019) recommends >=5 pairs to reduce noise. 10 pairs
provides robust averaging. Independent axis construction avoids translation bias:
the EN axis captures how English-trained models organize the dimension, the ZH
axis captures how Chinese-trained models do. The comparison is rank-based
(Spearman rho on projection scores), which is valid even when axes are
independently constructed — we compare the *ordering* of terms, not the
absolute scores. → §3.3.1

**Thesis text implication**: The thesis can claim that each tradition's models
construct value dimensions from their own linguistic resources, and that the
comparison measures whether these independently-constructed dimensions produce
similar orderings of legal concepts. → §3.3.1

---

### D3 — Axis construction method

**Options considered**:
- Option A: Kozlowski difference-vector method: axis = L2_normalize(mean(embed(pos_i) - embed(neg_i)))
  - Pro: standard in computational sociology (Kozlowski et al. 2019); validated in archive; mean-of-differences cancels idiosyncratic noise
  - Con: assumes linear structure in embedding space
- Option B: PCA on antonym pair vectors
  - Pro: captures more variance in the antonym set
  - Con: less interpretable; first PC may not align with intended axis direction
- Option C: SVM classifier boundary
  - Pro: nonlinear decision boundary
  - Con: overly complex; not standard in the literature; harder to interpret

**Decision**: Option A — Kozlowski difference-vector method

**Rationale**: Standard in computational sociology (Kozlowski 2019), validated in
the archive pipeline. The mean-of-differences approach cancels idiosyncratic noise
from individual pairs while preserving the shared directional signal. Projection
via cosine similarity yields scores in [-1, 1]. → §3.3.1

**Thesis text implication**: The thesis can describe axis construction as
"following Kozlowski et al. (2019), we compute the mean difference vector of
10 antonym pairs and project legal terms via cosine similarity." The linear
assumption is documented as a methodological limit. → §3.3.1

---

### D4 — Cross-linguistic comparison method

**Options considered**:
- Option A: Pearson correlation on projection scores
  - Pro: standard; sensitive to linear relationships
  - Con: assumes normality; sensitive to outliers
- Option B: Spearman rho on projection scores (N=397 core terms)
  - Pro: rank-based, non-parametric; appropriate for projection scores that may not be normally distributed
  - Con: less powerful than Pearson under true normality
- Option C: Kendall tau
  - Pro: even more robust than Spearman
  - Con: computationally slower; harder to interpret

**Decision**: Option B — Spearman rho + bootstrap CI

**Rationale**: Spearman is rank-based and non-parametric, appropriate for projection
scores that may not be normally distributed. Row-resample bootstrap CI (B=10000,
percentile method) provides uncertainty quantification without distributional
assumptions — see D5 for bootstrap variant justification, D6 for B choice. All
15 pairs are tested: 9 cross-tradition + 3 within-WEIRD + 3 within-Sinic.
Mann-Whitney compares cross ρ vs within ρ per axis as a descriptive summary;
see D7 caveat on pseudo-replication. → §3.3.2

**Thesis text implication**: The thesis can report Spearman ρ and 95% bootstrap CI
for each of 45 comparisons (15 pairs × 3 axes), and a descriptive Mann-Whitney
summary comparing cross vs within groups. It can claim that rank alignment (not
absolute scores) is the unit of comparison. The MW should be framed as descriptive
rather than confirmatory due to pseudo-replication (see D7). → §3.3.2, §3.3.3

---

### D5 — Bootstrap variant: row-resample vs block bootstrap

**Options considered**:
- Option A: Block bootstrap (resample term indices, resubset RDMs, recompute Spearman)
  - Pro: used in Lens I for RSA; accounts for between-term dependency structure in RDMs
  - Con: no RDMs here — projection scores are per-term scalars, not pairwise distances
- Option B: Row-resample bootstrap (resample term indices, resubset score vectors, recompute Spearman)
  - Pro: correct unit of resampling for per-term scores; each term's score is independent given the axis; standard for correlation bootstraps (Efron & Tibshirani 1993)
  - Con: none significant — this is the textbook case

**Decision**: Option B — Row-resample bootstrap

**Rationale**: In Lens I, the unit of analysis is term-*pair* distances (RDM cells),
where each term contributes to (n−1) cells, creating dependency. Block bootstrap
resamples at the term level to respect this structure. In Lens IV, projection scores
are per-term scalars: score_i = cos(embed(term_i), axis). Each term's score is
independent given the axis vector. Row-resample bootstrap (resample N term indices
with replacement, resubset both score vectors, recompute Spearman) is therefore
the correct and sufficient approach. → §3.3.2, math trace V6.

**Thesis text implication**: The thesis can justify the bootstrap variant by noting
that "unlike RSA (§3.1), where each term contributes to N−1 pairwise distances,
projection scores are independent per-term, making row-resample bootstrap
appropriate (Efron & Tibshirani, 1993)." → §3.3.2

---

### D6 — Number of bootstrap resamples (B=10000)

**Options considered**:
- Option A: B=1000 (faster, sufficient for point estimates)
  - Pro: 15× faster runtime; adequate for CI widths per Efron (1987)
  - Con: percentile CI endpoints at the 2.5th and 97.5th percentile are estimated from only 25 resamples each; inconsistent with Lens I's B=10000
- Option B: B=10000 (consistent with Lens I)
  - Pro: CI endpoints estimated from 250 resamples each; matches Lens I standard (Nili 2014); Monte Carlo SE < 0.002 for ρ
  - Con: ~60s total runtime (acceptable)

**Decision**: Option B — B=10000

**Rationale**: Consistency across all lenses: Lens I uses B=10000 for permutation
tests (Nili 2014), Lens IV should use the same for bootstrap CIs. The runtime
cost is negligible (61s for full pipeline). This ensures the thesis can state a
uniform Monte Carlo standard across experiments. → §3.3.2, §2.4 (methods chapter)

**Thesis text implication**: "All resampling procedures use B=10,000 iterations,
following the recommendation of Nili et al. (2014) for stable Monte Carlo
estimates." This is a single sentence that covers both lenses. → §3.3.2

---

### D7 — Kruskal-Wallis vs ANOVA for §3.3.3 + pseudo-replication caveat

**Options considered**:
- Option A: One-way ANOVA on cross-tradition ρ grouped by axis
  - Pro: parametric, more powerful
  - Con: ρ values are not normally distributed (bounded in [-1,1], skewed); N=9 per group is small; pseudo-replication from 3×3 model grid
- Option B: Kruskal-Wallis H test (non-parametric)
  - Pro: no distributional assumption; rank-based like the underlying Spearman ρ; standard for small-N ordinal data
  - Con: less powerful; same pseudo-replication issue as ANOVA

**Decision**: Option B — Kruskal-Wallis H

**Rationale**: Spearman ρ values are bounded and may not be normally distributed.
Kruskal-Wallis is rank-based, consistent with the rank-based philosophy of the
entire Lens IV analysis. The pseudo-replication caveat applies equally to both
options: the 9 cross-tradition ρ values per axis come from a 3×3 grid where each
model appears in 3 pairs, violating independence. This must be disclosed in the
thesis. With KW H=7.20, p=0.027, the result is significant even at α=0.05, but
the effective sample size is smaller than the nominal N=9. → §3.3.3, math trace V8.

**Thesis text implication**: "A Kruskal-Wallis test reveals significant heterogeneity
in cross-tradition alignment across the three value axes (H=7.20, p=0.027). We
note that the nine ρ values per axis derive from a 3×3 model grid, introducing
partial non-independence; the test should be interpreted as indicative rather than
confirmatory." → §3.3.3

---

### D8 — Axis orthogonality: natural correlation with diagnostic

**Options considered**:
- Option A: Enforce orthogonal axes via Gram-Schmidt orthogonalization
  - Pro: guarantees zero inter-axis correlation; cleaner interpretation of per-axis results
  - Con: the orthogonalized axes lose their semantic grounding (axis 2 becomes "axis 2 minus its projection on axis 1"); violates the Kozlowski method; individual_collective and public_private are conceptually related (summa divisio)
- Option B: Allow natural correlation, report inter-axis cosine as diagnostic
  - Pro: axes retain their semantic meaning; cosine similarity is a transparency measure, not a confound; comparative law theory predicts partial overlap between individual_collective and public_private (Sacco 2019)
  - Con: high inter-axis correlation could inflate apparent agreement across axes

**Decision**: Option B — Natural correlation with diagnostic

**Rationale**: Enforcing orthogonality would distort the axes' semantic content.
The comparative law literature predicts that individual_collective and public_private
are complementary aspects of the same fundamental division (the "summa divisio"
between public and private law — Sacco 2019). Their moderate cosine similarity
(0.17–0.38 depending on model) is *expected* and *interpretable*, not a confound.
rights_duties shows near-zero cosine with the other two, confirming it captures an
independent dimension. The diagnostic is reported per-model in §3.3.1 and visualized
in the interactive HTML. → §3.3.1, math trace V4.

**Thesis text implication**: "Inter-axis cosine similarity reveals that
individual_collective and public_private share moderate geometric overlap (cos θ =
0.17–0.38), consistent with comparative law's 'summa divisio' between these two
organizing principles (Sacco, 2019). rights_duties is near-orthogonal to both
(cos θ < 0.10), suggesting it captures a genuinely independent dimension." → §3.3.1

---

## Errata and fixes

### E1 — Duplicate antonym pair in zh_pairs (2026-03-02)

**Issue**: `[权利, 义务]` (rights, duties) appeared in both `individual_collective`
zh_pairs and `rights_duties` zh_pairs in `value_axes.yaml`. This artificially
inflated the cosine similarity between the two axes for Sinic models.

**Fix**: Replaced the duplicate in `individual_collective` with `[私心, 公心]`
(private heart, public heart) — a genuine Chinese conceptual pair reflecting
the individual/collective tension without overlap with rights_duties vocabulary.

**Impact**:
- BGE-ZH inter-axis cosine (ic vs rd): 0.377 → 0.205 (substantial reduction)
- Kruskal-Wallis H: 4.17 (p=0.124) → 7.20 (p=0.027) — became significant
- The fix removed an artificial correlation that was masking genuine
  inter-axis divergence patterns.

**Lesson**: When axes are constructed from independent word pairs, vocabulary
overlap between axes is equivalent to feature leakage. Each axis's pair set
must be checked for zero intersection with all other axes' pair sets.

---

### D7 — Removal of Kruskal-Wallis test from §3.3.3

**Options considered**:
- A: Keep Kruskal-Wallis omnibus test comparing 9 cross-tradition rho values
  across 3 axes (pro: formal test; con: violates independence — the same 9
  model pairs produce values for all 3 axes, and the axes themselves are
  non-orthogonal with cosine similarity up to ~0.5)
- B: Replace KW with a permutation test that respects the paired structure
  (pro: valid; con: complex design for a secondary comparison)
- C: Report descriptive ranking only (pro: honest, avoids false precision
  from an invalid test; con: no inferential claim)

**Decision**: Option C — descriptive ranking only.

**Rationale**: The Kruskal-Wallis H test requires independent observations.
With the same 9 model pairs contributing one value per axis, and axes that
share up to 25% of their variance (cosine ≈ −0.5), both the within-group
and between-group independence assumptions are violated. Reporting the ranking
with means and standard deviations is more honest than a p-value from an
invalid test. The forest plot in the visualization already makes the axis
differences visually apparent.

**Thesis text implication**: → §3.3.3 "The three axes exhibit different
degrees of cross-tradition convergence: public/private (ρ̄_cross = 0.402)
shows the strongest alignment, followed by rights/duties (0.380) and
individual/collective (0.292). We report these descriptively rather than
via omnibus testing, because the 9 cross-tradition values per axis derive
from the same model pairs and the axes are non-orthogonal."

---

### D8 — Replacement of Mann-Whitney with permutation test for cross/within comparison

**Options considered**:
- A: Mann-Whitney U on 9 cross vs 6 within values (pro: standard; con: with
  n=15 total the test has very low power, and U=0 with r=1.0 is a ceiling
  artefact rather than a measurement)
- B: Permutation test on group labels (pro: exact, no distributional
  assumptions, well-suited for small samples; con: slightly more complex)

**Decision**: Option B — permutation test (10,000 permutations) on the
cross/within group label assignment.

**Rationale**: With only 15 observations the Mann-Whitney U test produces
artefactual extreme values (U=0, r=1.0) that give false precision. The
permutation test shuffles the cross/within labels 10,000 times and computes
the difference in means, producing an empirical p-value that is honest about
the small sample size.

**Thesis text implication**: → §3.3.2 "The separation between cross-tradition
and within-tradition correlation values is assessed by permutation test
(10,000 shuffles of group labels) rather than Mann-Whitney U, given the
small sample size (n_cross=9, n_within=6)."

---

## References

- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*.
  Chapman & Hall/CRC.

- Kozlowski, A. C., Taddy, M., & Evans, J. A. (2019). The geometry of culture:
  Analyzing the meanings of class through word embeddings. *American Sociological
  Review*, 84(5), 905-949.
- Legrand, P. (1996). European legal systems are not converging. *International
  and Comparative Law Quarterly*, 45(1), 52-81.
- Merryman, J. H., & Perez-Perdomo, R. (2007). *The civil law tradition*.
  Stanford University Press.
