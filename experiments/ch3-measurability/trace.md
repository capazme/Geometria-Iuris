# Run #4 — post-BLP final · Parameter trace

Twelve decisions registered **before** execution. Each has options
considered, decision, rationale, and the thesis section that depends on it.

---

## D1. Model list (10 encoding jobs)

**Options.**
- A. 6 monolingual (run #2 baseline)
- B. 6 + 2 bilingual lati = 10 (run #3 baseline)
- C. 10 + Nomic-v1.5 + Conan-v2 = 12

**Decision.** B — 3 WEIRD (BGE-EN-large, E5-large, FreeLaw-EN) + 3 Sinic
(BGE-ZH-large, Text2vec-large-ZH, Dmeta-ZH) + 2 bilingual lati each
(BGE-M3-EN/ZH, Qwen3-0.6B-EN/ZH).

**Rationale.** Continuity with run #3. Nomic-v1.5 was excluded for
sentence-pair retrieval bias (not symmetric STS). Conan-v2 is API-only.
12 models would have inflated FWER without adding architectural diversity
beyond the existing 3-slot cross-terna design (BGE family, STS-tuned,
domain-tuned).

**Thesis implication.** §2.3 (Language models as cultural informants):
the 6 monolingual encoders carry the WEIRD/Sinic split; the 2 bilingual
encoders serve as causal controls (§3.1.3 within-bilingual ρ̄).

---

## D2. Encoding variants per model

**Options.**
- A. Bare only
- B. Bare + attested-postBLP
- C. Bare + attested-postBLP + attested-all-time

**Decision.** B.

**Rationale.** The 364-term pool itself is post-BLP curated (K≥4 on
post-1989 enactments). An "all-time attested" variant would mix pre-BLP
unilateral drafting (where the fictio of equal authenticity is a fiction,
not a structure) with post-BLP co-drafting; this would re-introduce the
bias the pivot was designed to eliminate. The bare/attested contrast
remains the principled axis of §3.1.3.

**Thesis implication.** §2.2 (Hong Kong as natural laboratory): the
single attested variant inherits BLP co-redaction; §3.1.3 reports two
columns (bare, attested) per pair.

---

## D3. Attestation aggregation function

**Options.**
- A. Mean of context vectors
- B. Median of context vectors
- C. First-context-only (no aggregation)
- D. Token-weighted mean

**Decision.** A — mean across `min(N_attested, 8)` context vectors,
then L2-normalize the mean.

**Rationale.** Continuity with run #3 (and with the lens-1 attested
result of `lens_1_relational/results_attested/`). Mean is the standard
Firth aggregator and is robust under the linear regime of normalized
embeddings (means of unit vectors live near the centroid of the cone).
Median is sample-quantile-based and discards the off-centroid context
information that carries the Firthian signal; first-only ignores K-1
contexts; token-weighted requires tokenizer-aware bookkeeping outside
the encoder API.

**Thesis implication.** §2.3 (operational definition of attested
embeddings); §3.1.3 attested column.

---

## D4. Mantel test permutations

**Options.**
- A. B = 1000  (run #2, run #3)
- B. B = 10000  (definitive run)
- C. B = 100000 (overkill)

**Decision.** B = 10000.

**Rationale.** With 17 model pairs and Holm correction at α = 0.05,
the smallest representable p-value at B=1000 is 0.001 — which after
Holm correction with K=17 becomes 0.017, blunting the discriminatory
power for the most significant pairs. B=10000 pushes the floor to 0.0001
(Holm-adjusted 0.0017), comfortably below standard α. B=100000 would
buy nothing additional and quadruple compute.

**Thesis implication.** §3.1.3 (Mantel p column); §3.2.3 (axis p column).

---

## D5. Bootstrap CI method and B

**Options.**
- A. Pair-bootstrap (resample 1-half of N(N-1)/2 pairs); B = 1000
- B. Block-bootstrap on terms (resample N term indices, rebuild RDMs);
  B = 10000
- C. Jackknife (leave-one-term-out)

**Decision.** B — block-bootstrap on terms, B = 10000.

**Rationale.** Term-level resampling preserves the dependency structure
of the RDM (each term appears in N-1 pairs). Nili et al. (2014) PLoS
Comp. Bio. 10(4):e1003553 §5.2 demonstrates that pair-level bootstrap
is anti-conservative for RDM correlations. Jackknife under-covers the
tail behavior; B=10000 gives stable percentile CI bounds.

**Thesis implication.** §3.1.3 (95% CI column); §3.2.3 (95% CI per axis).

---

## D6. Multiple-comparison correction

**Options.**
- A. Bonferroni (K=17)
- B. Holm-Bonferroni step-down (K=17)
- C. Benjamini-Hochberg FDR
- D. None

**Decision.** B — Holm-Bonferroni step-down with K=17 on the §3.1.3 p-values.

**Rationale.** §3.1.3 is a family of 17 pre-registered tests on the same
RDMs (3 WEIRD × 3 Sinic + 3 within-W + 3 within-S + 2 within-bilingual).
FWER control is the appropriate criterion (we want zero false claims of
"agreement above chance"). Holm dominates Bonferroni in power without
sacrificing FWER. BH/FDR would be appropriate if §3.1.3 were exploratory;
it is not. For §3.2.3 the 6 axes are independent constructs (D6+D2 in
trace_pivot_2lens.md) so no within-axis correction; cross-axis we report
descriptively only.

**Thesis implication.** §3.1.3 reports both raw p and Holm-adjusted p_holm.

---

## D7. Reproducibility — seed, device, dtype

**Options.**
- A. seed=42, CPU, float32  (deterministic)
- B. seed=42, MPS (Apple Silicon), float32  (5× faster, bit-for-bit non-
  reproducible across machines)
- C. seed=42, CPU, float64 (deterministic, 2× memory)

**Decision.** A — `seed=42`, `device=cpu`, `dtype=float32`.

**Rationale.** Final consegna requires that a reviewer with the same
sources and same code reproduces our numbers bit-for-bit. MPS is non-
deterministic for some kernels (matrix multiplication ordering in Metal
shaders) and exact match across machines is not guaranteed. float64
buys 1-2 ULP in the percentile CI which is below the reporting precision
(ρ̄ to 3 decimal places). CPU bare encoding of 364 strings × 10 models
costs minutes; CPU attested of ~3000 contexts × 10 models is the heavy
part (estimated 4-6h on M-series CPU).

**Thesis implication.** §2.4 (Statistical tools — reproducibility
guarantees); appendix A (reproducibility note).

---

## D8. Per-domain term count reporting

**Options.**
- A. Rounded 50/domain ("balanced")
- B. Actual 41-60 band ("post-BLP filter result")
- C. Both: 50 baseline + observed delta

**Decision.** B — report the observed band 41-60 (administrative=52,
civil=60, constitutional=49, criminal=54, international=41,
labor_social=60, procedure=48; mean 52.0).

**Rationale.** The 41-60 band is what the post-BLP K≥4 filter produces
when applied to the curated longlist. Forcing it to 50/domain would
require either dropping high-coverage terms (information loss) or
synthesizing low-coverage ones (re-introducing the fictio bias). The
band is small enough that any per-domain statistic (Mann-Whitney intra/
inter, 7×7 topology) remains well-conditioned.

**Thesis implication.** §2.1 (lessico construction); §3.1.1 §3.1.2 per-
domain reporting; footnote on band rationale.

---

## D9. Categorical probe tests

**Options.**
- A. Same 5 pre-registered tests as run #3
- B. Re-prereg new tests on the post-BLP pool
- C. Drop §3.1.4

**Decision.** A.

**Rationale.** §3.1.4 was pre-registered before any post-BLP curation
decision (trace_pre_registration_categorical.md, 2026-04-30). Re-
preregistering after seeing the curated pool would constitute
post-hoc fitting. The tests are: (i) sub-domain Spearman, (ii) civil-
vs-criminal Mann-Whitney, (iii) procedure-vs-substantive, (iv) rights-
vs-duties, (v) public-vs-private summa divisio. The pool change does
not invalidate them; it strengthens them (more attestational signal).

**Thesis implication.** §3.1.4 reports the same 5 ensemble ρ as run #3
but on the cleaner pool.

---

## D10. Output destination — isolation from run #3

**Options.**
- A. Overwrite `experiments/lens_*/results*` (in-place upgrade)
- B. New folder `experiments/ch3-measurability/` (isolated)
- C. Versioned folder `experiments/lens_*/results_v4/`

**Decision.** B — `experiments/ch3-measurability/`.

**Rationale.** Run #3 results (`lens_*/results_bare`, `results_attested`,
`results.bak_pre_firthian`) remain on disk for the dashboard and for
the comparison report (§3.1.3 changes_vs_run3.md). Overwriting would
make rollback expensive; per-lens versioning would scatter the audit
trail. A single run #4 folder concentrates inputs + scripts + outputs +
manifest + trace in one place.

**Thesis implication.** §3 final reporting; appendix B (reproducibility
trail).

---

## D11. Dashboard v3 loader update

**Options.**
- A. Hard-cutover: rewrite loaders to read from `ch3-measurability/`
- B. Dual-source: loader takes `--run-id` flag
- C. No change: dashboard stays on run #3 until thesis is delivered

**Decision.** C for now; A as a follow-up after run #4 verification gate.

**Rationale.** Switching the dashboard before §8 (verification gate) is
premature. After the gate passes, a 30-minute loader edit in
`dashboard_v3/data/results_31.py` and `results_32.py` swaps the path.

**Thesis implication.** Dashboard is appendix material; not on the
critical path for the chapter text.

---

## D12. Script reproducibility convention

**Options.**
- A. Hard-coded paths in each script
- B. argparse + `--config config.yaml`
- C. Environment variables

**Decision.** B — every script in `scripts/` takes `--config` with
default `ch3-measurability/config.yaml`. All paths, seeds, B values are
read from the YAML.

**Rationale.** Single source of truth. Any reviewer can rerun any
stage with a different config (e.g. `B=1000` for a fast smoke test)
without editing code. The manifest records which config was used.

**Thesis implication.** Appendix A (reproducibility); allows a reviewer
to reproduce the exact numbers cited in §3.

---

## D13. Control terms — scope of use

**Context.** The 100 control terms (everyday vocabulary: pronouns, deixis,
common nouns) were already present in the legacy `legal_terms.json` under
`tier='control'`. They were OMITTED from the initial run #4 pipeline
because `legal_term_run4.json` is core-only.

**Options.**
- A. Skip control: 364-core only, §3.1.1 legal-vs-control test is omitted.
- B. Snapshot the 100 control, encode bare (no attested by design — no HK Cap.
  attestation for everyday vocabulary), run §3.1.1 legal-vs-control.

**Decision.** B.

**Rationale.** §3.1.1 legal-vs-control is a discriminative-validity test:
do the legal embeddings form a more compact cluster than the everyday-language
embeddings? It is the natural complement of intra-vs-inter. Omitting it
would leave a methodological gap.

**Thesis implication.** §3.1.1 reports legal-vs-control Mann-Whitney per
model. 8/10 confirm signal; FreeLaw-EN and Qwen3-0.6B-EN do not (legal
fine-tuning erodes term-class boundary; multilingual under-specialization).
Reported in `experiment_1_structure/results_bare/legal_vs_control.json`. Used as the
operational scope statement of the framework in §4.2.

---

## D14. Background terms — scope of use

**Context.** The legacy `legal_terms.json` contains 9.045 background terms
(`tier='background'`), legalish residual not promoted to core during the
post-BLP curation. These have variable K (some have K≥4 in both langs,
some have K<4 in one or both). They were not part of the original run #4
plan.

**Options.**
- A. Ignore bg: run #4 = 364 core + 100 control, full stop.
- B. Snapshot bg + their contexts, encode bare for all 10 models and
  attested for 4 primary models (BGE-EN, BGE-ZH, BGE-M3-EN, BGE-M3-ZH);
  use bg as input for six robustness extensions (A, D, E, F, G, H).

**Decision.** B, with the constraint that bg never replace core in the
headline numbers — they only feed robustness extensions.

**Rationale.** A 9.045-term out-of-curation pool is a free robustness
resource. Running the headline against bg-injected pools (extension D),
projecting bg on the axes (extension E), running an automated
false-friends detector on bg (extension G), and using K-buckets of bg
to validate the K≥4 threshold (extension H) yields multi-angle evidence
on the structurality of Δρ_sym, the generalization of the axes, and
the empirical justification of pre-registered choices. Cost: ~70 min
of encoding on CPU.

**Constraint.** Bg attested encoding is limited to 4 primary models
(BGE-EN, BGE-ZH, BGE-M3-EN, BGE-M3-ZH). Qwen3-0.6B-ZH attested would
have cost ~7h on the bg pool given its slow ZH tokenizer. The 4 chosen
models cover the within-WEIRD, within-Sinic and cross-tradition pair
needed by D, F, G; H operates on the single BGE pair.

**Thesis implication.** Extensions A, D, E, F, G, H added; reported in
`reports/extensions_summary.md`. D and G are headline-strengthening;
H justifies §2.3 K≥4 empirically; E + F are cited in §4.2 with caveats.

---

## D15. Control-driven dual extensions (X, Y, Z)

**Context.** Symmetric to D14, we asked whether control terms could
extend Lens I beyond the §3.1.1 legal-vs-control test alone.

**Options.**
- A. Stop at §3.1.1 legal-vs-control.
- B. Add three controlled experiments: X = dual of D (inject control,
  expect Δρ_sym to fall); Y = run RSA on control-only pool (expect Δρ_sym
  → 0 if signal is legal-specific); Z = three-tier distance hierarchy
  (core×core vs core×bg vs core×control medians).

**Decision.** B.

**Rationale.** X gives the *dual* robustness statement to D (signal falls
under non-legal perturbation), Y is a sanity check on the discriminative
nature of the bare signal, Z is a corpus-geometry diagnostic. Together
they test the framework symmetrically.

**Key finding from Y.** Δρ_sym bare = 0.156 on the 100 control-only
pool. Δρ_sym bare = 0.165 on the 364 core. **Indistinguishable.** This
falsifies the implicit claim "the bare signal is legal-specific" and
forces a reframing (see D16).

**Thesis implication.** X, Y, Z reported in `reports/extensions_summary.md`.
Y is methodologically central and must be flagged in §4.1 framing.

---

## D16. Bare-versus-attested framing (post-Y reframing)

**Context.** Extension Y (D15) showed that Δρ_sym bare on 100 control
terms (0.156) is statistically indistinguishable from Δρ_sym bare on
the 364 core (0.165). The bare signal is therefore *encoder-tradition
shaped* (WEIRD encoders disagree with Sinic encoders even on pronouns
and deictics), not legal-tradition shaped.

**Options.**
- A. Cite Δρ_sym attested = 0.543 as the legal signal headline.
- B. Reframe: cite the **attested-bare gap** (0.543 − 0.165 = 0.378 on
  core) as the legal signal. The attested absolute is a hybrid of
  encoder-tradition (≈ 0.165 baseline) plus legal-attestation (≈ 0.378).
- C. Cite both the absolute and the gap, with explicit framing.

**Decision.** C.

**Rationale.** The attested absolute is the natural quantity to report
(it is what the experiment computes), but it is methodologically
ambiguous: a reader could attribute it entirely to encoder-tradition
unless they read Y. The attested-bare gap is the cleanly *legal* signal,
isolating contextualization on HK Cap. attestations from the encoder
baseline. Reporting both, with explicit framing, lets the reader follow
the reasoning step by step.

**Frase tipo per la prosa:**
> *"Δρ_sym attested = 0.543 measures the within-vs-cross tradition gap
> as actually computed in our pipeline. Yet on the 100 everyday-language
> control terms the same Δρ_sym bare metric returns 0.156, statistically
> indistinguishable from 0.165 on the 364 core bare. The legal-attestation
> contribution is therefore best isolated as the **attested-bare gap of
> 0.378 on the core** (0.543 − 0.165), against an encoder-tradition
> baseline of ~0.16 that is shared with non-legal vocabulary."*

**Thesis implication.** §4.1 must cite both the absolute and the gap.
§4.2 has Y as a primary caveat. CLAUDE.md §10, HANDOFF.md, and
`reports/extensions_summary.md` updated accordingly.

---

## D17. Folder naming + slug convention

**Context.** The working folder `run4_postBLP/` was a date-and-corpus
slug useful internally but ostile to thesis citation and to anyone
landing cold on the repo.

**Options.**
- A. Keep `run4_postBLP/`.
- B. Rename `ch3-measurability/` (capitulum + theme).
- C. Rename `geometria-iuris-2026/` (brand + year).
- D. Rename `hk-postBLP-364/` (corpus + curation + size).

**Decision.** B.

**Rationale.** The folder ties one-to-one with Chapter 3 of the thesis;
the capitulum is the natural anchor for a reader who arrives via the
thesis footnotes. The theme word *measurability* echoes the central
research question. Future-proof: a future run #5 with a different pool
would still belong to ch3-measurability.

**Thesis implication.** Footnote pattern: file-relative against
`experiments/ch3-measurability/`. Promotable to a standalone Zenodo
deposit with DOI when the thesis is finalized; the relative path
`../shared/` (statistical and embedding utilities) is the only cross-
repo dependency to bundle.
