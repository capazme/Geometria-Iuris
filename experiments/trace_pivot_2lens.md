# Trace: Pivot to two-lens experimental design

**Thesis chapter(s)**: Ch.3 (entire chapter restructured)
**Date**: 2026-04-16
**Status**: D1-D2 decided; D3-D6 proposed, awaiting approval

---

## Context

On 2026-04-02 the co-relatore (engineering, LUISS) raised two objections:
(1) methodology without a concrete result, (2) too many experiments. The
Phase 2 upgrade (β+γ+δ) responded with *more* experiments (Lens VI SAE)
rather than with consolidation, addressing (1) only indirectly and
actively worsening (2).

In the 2026-04-16 review the author nominated a second, more fundamental
objection to the false-friend framing of Lens V: its output depends on
machine-set hyperparameters (k in k-NN, Jaccard threshold) with no
doctrinal anchor, so a reader who recomputes with k=10 or k=20 obtains
different rankings. A measurement instrument whose output is controlled
by hidden knobs fails the Ch.1 §1.1 test (intersubjective accessibility,
telescope analogy). The same critique applies, in weaker form, to Lens
III (layer choice, breakpoint detection thresholds) and Lens VI (SAE
expansion ratio, TopK cutoff, dictionary size).

This trace records the restructuring of the experimental apparatus around
two lenses whose methodological choices are either doctrinally anchored
or literature-standard.

---

## D1 — Reduce experimental scope to Lens I + Lens IV

**Options considered**:
- **Option A**: keep 5 lenses + SAE (status quo after Phase 2).
  *Pro*: maximal evidence base. *Contro*: reproduces the problem the
  co-relatore identified; narrative cost too high to explain to a
  mixed commissione in the time available.
- **Option B**: Lens I + Lens V (RSA + false friends).
  *Pro*: false friends is the most applied deliverable. *Contro*:
  hyperparameter arbitrariness (k, threshold); inverts jurist/machine
  authority relation established in Ch.1 §1.5.
- **Option C**: Lens I + Lens IV (RSA + value axes).
  *Pro*: every arbitrary parameter in Lens IV is a *doctrinal choice*
  declared by the jurist (which poles define the axis). The machine
  does only deterministic arithmetic. Respects the division of labour
  of Ch.1 §1.5. *Contro*: less "applied-feeling" than false friends.
- **Option D**: Lens I alone.
  *Pro*: maximal minimalism. *Contro*: leaves the co-relatore's
  "methodology without result" objection unanswered.

**Decision**: **Option C** — Lens I + Lens IV.

**Rationale**: The selection criterion is *transparency of methodological
choices*: every arbitrary parameter in a retained experiment must be
(a) doctrinally anchored or literature-standard, (b) fixed before data
is seen, and (c) defended in a written decision record. Lens I satisfies
this via literature-standard choices (cosine distance, Spearman ρ,
Mantel test, block bootstrap — each with a single-sentence defence
traceable to Kriegeskorte, Nili, Phipson-Smyth). Lens IV satisfies it
via *doctrinal* anchoring: the pole-pairs that construct each axis
are legal-theoretical choices, not machine parameters. The machine's
residual role (embed → mean-diff → L2-normalize → project → rank-correlate)
is free of knobs. Lenses III, V, VI each require at least one knob
(layer, k, Jaccard threshold, expansion ratio, TopK) that no amount of
theoretical defence can fully anchor.

**Thesis text implication**: → Ch.3 structure reduces to two sections
of substance plus one interpretive section (quod numeri tacent). The
instrument validation question ("does it work?") is fully handled by
Lens I; the applied-measurement question ("what does it show?") is
fully handled by Lens IV. The division of labour matches the two
audiences of the thesis: Lens I answers the relatore (method), Lens IV
answers the co-relatore (applied result).

---

## D2 — Archive strategy for Lenses II, III, V, VI

**Options considered**:
- **Option A**: `mv` to `experiments/_archive/lenses_2026-04-16/`.
  *Pro*: reversible, preserves code and results, navigable. *Contro*:
  none relevant.
- **Option B**: delete and rely on git history. *Pro*: cleaner tree.
  *Contro*: recovery friction, harder to cite from appendix.
- **Option C**: leave in place, exclude from dashboard. *Pro*: no
  disruption. *Contro*: leaves a disordered tree; reader cannot
  distinguish live from archived work.

**Decision**: **Option A** — move to
`experiments/_archive/lenses_2026-04-16/`.

**Rationale**: Archival is informational, not destructive. The archived
lenses' results remain available for appendix citation and for future
"robustness by previous design" claims. The date-stamped subfolder
preserves the causal chain (what was archived when and why).

**Implementation** (executed this session):
- `mv experiments/lens_2_taxonomy       experiments/_archive/lenses_2026-04-16/`
- `mv experiments/lens_3_stratigraphy   experiments/_archive/lenses_2026-04-16/`
- `mv experiments/lens_5_neighborhoods  experiments/_archive/lenses_2026-04-16/`
- `mv experiments/lens_6_sae            experiments/_archive/lenses_2026-04-16/`

`build_dashboard.py` references archived lenses extensively (4 of 5
rendered sections). Rather than patch a 43k-token file that will be
rewritten from scratch when the new Lens I + IV results are ready,
the dashboard is left as-is in this commit and flagged for rewrite
in D3's execution step.

`OVERVIEW.md` and `README.md` are updated in this commit to reflect
the new scope.

**Thesis text implication**: → Index (`documenti/003_GeometriaIuris_Indice.docx`)
sections 3.1.3 (stratigraphy), 3.2 (neighborhoods), 4.4 (horizons) are
removed in the next index revision. Section 3.3 (values) may be
renumbered to 3.2 depending on the final Ch.3 structure. This
re-numbering is deferred to D3 when the final scope of Lens I's
subsections is fixed.

---

## D3 — Lens I redesign scope ("per bene") **[PROPOSED]**

Lens I's decision log is already mature (D1-D11 in
`lens_1_relational/trace.md`). The Phase 2 additions (D10 bilingual
control, D11 attested contexts) are already integrated. The redesign
is therefore narrow: remove coupling to archived lenses and promote
Phase 2 additions from "robustness sub-section" to "headline result".

**Proposed changes**:

1. **De-couple from archived Lens III**: D9 (layer sensitivity sweep)
   currently frames its result as cross-corroboration of Lens III's
   phase transition. Rewrite as self-contained Lens I robustness:
   "the cross-tradition gap is stable across layer extraction depth
   in the symmetric definition, with peak at 5L/6 (Δρ_sym 0.346) and
   final-layer value 0.237". No reference to Lens III narrative.
   `layer_vectors/` cache is retained inside `_archive/` but read
   by Lens I at runtime — or copied into `lens_1_relational/` for
   self-containment. Recommend **copy** to make Lens I self-contained.

2. **Keep §3.1.5 parametric categorical probe** as is. The D8/D8-update
   design (11-category sequences, pre-registered expected positions,
   Test 5 determinate/indeterminate as clean positive, Test 4 as
   honest negative) is well-documented and satisfies the transparency
   criterion (all category sequences pre-registered in YAML before the
   rerun). This is the single §3.1.5 sub-section that directly
   answers the co-relatore's "concrete result" request inside Lens I.

3. **Promote D10 (bilingual control) and D11 (attested contexts) to
   headline**: rewrite §3.1.4 so the primary reported number is the
   cross-tradition drop *under the bilingual control* (the causal
   control), with the monolingual result and the attested-vs-bare
   contrast reported as layered evidence. The current text treats
   these as Phase 2 additions; they are actually the strongest
   version of the finding and should carry the chapter.

4. **Rerun Lens I on the updated dataset and model panel** (see D5,
   D6): this requires recomputing RDMs only (no new method). Compute
   cost: ~6 hours on the full 8-model panel with attested contexts
   (dominated by Qwen3 attested encoding — 200× slower with long
   contexts, already known from Phase 2 progress memory).

**Open question for Lens I**: do we add a larger Qwen3 variant
(4B or 8B) as a 9th model? See D6.

---

## D4 — Lens IV redesign scope ("per bene") **[PROPOSED]**

Lens IV is the lens that most needs work. Phase 2 upgrades (β bilingual
models and δ attested contexts) have *not* been integrated into Lens IV;
the current results (ind/coll 0.292, rights/duties 0.380, public/private
0.402) are computed on the 6 base monolingual models with templated
projection scores. "Per bene" means putting Lens IV on the same
methodological footing as Lens I.

**Proposed changes**:

1. **Model panel**: extend from 6 base to 6 base + 2 bilingual
   (BGE-M3, Qwen3-0.6B), matching Lens I. This adds 6 bilingual model
   pairs to the comparison: 1 bilingual × 3 WEIRD + 1 bilingual × 3
   Sinic + 1 within-bilingual, ×2 bilingual models. Produces a
   causal-control reading analogous to Lens I D10: "the value-axis
   divergence persists under a shared-architecture bilingual encoder,
   so it is not an encoder artefact."

2. **Attested-context projection**: recompute each term's embedding
   as the mean of up to 8 attested e-Legislation contexts (the
   `embeddings_contextualized/` pool already built for Lens I,
   re-used by Lens IV). Projection onto axes becomes a projection
   of legally-attested usage, not of templated "the legal term X"
   phrases. Closes the Firthian gap in Lens IV the same way D11
   closed it for Lens I.

3. **Axis expansion and pole-pair revision**: see D6' below.
   Candidate new axes (selection pending user confirmation):
   - `formal_substantive` — formalism/substantive reasoning in legal theory
   - `state_market` — state/market, public-order vs contractual freedom
   - `substantive_procedural` — right vs remedy, Ulpian actio
   - `natural_positive` — natural law vs positive law
   - `codified_uncodified` — written code vs judicial development
   - `territorial_personal` — territorial jurisdiction vs personal law

   Each candidate axis requires: (a) doctrinal defence in the trace
   (why these poles), (b) 10+10 pole pairs per language, (c) an
   inter-axis cosine diagnostic against all existing axes to detect
   feature leakage.

4. **Pole-pair audit of existing 3 axes**: review all 60 pairs
   (10 per lang × 3 axes) for:
   - feature leakage (vocabulary appearing in another axis, as caught
     by errata E1 for 权利/义务);
   - doctrinal soundness (are these the best pairs for the concept?);
   - cross-lang symmetry of register (are the EN and ZH pairs at
     comparable levels of abstraction?).
   The audit outcome and any pair replacements are recorded as a new
   Lens IV D-entry before the rerun.

5. **Statistical apparatus**: keep row-resample bootstrap for
   projection-score Spearman ρ (Lens IV D5); keep B=10000 (Lens IV D6);
   keep natural-correlation diagnostic instead of Gram-Schmidt (Lens IV
   D8). These are already literature-standard; no change needed.

**Open question for Lens IV**: target number of axes after expansion.
Trade-off: more axes → richer comparative picture but ⊥ more
pole-pair curation work + higher multiple-comparison burden. Proposal:
**6 axes total** (3 existing + 3 new from the candidate list),
multiple-comparison correction across the 6 × 9 = 54 cross-tradition
ρ values via Holm-Bonferroni, aligned with Lens I D6.

---

## D5 — Dataset expansion **[PROPOSED]**

Current dataset (from `data/trace_dataset_design.md` and `OVERVIEW.md`):
- Core: 397 hand-labelled terms across 7 domains
- Background: ~8,975 terms (HK DOJ glossary)
- Control: 100 (Swadesh basic vocabulary)

The administrative-law domain is under-represented at 12 core terms.
Phase 2 memo flagged 125 keyword-matched candidates in the background
pool, awaiting lawyer curation.

**Proposed changes**:

1. **Promote administrative-law candidates**: user-driven curation of
   the 125 candidates to select ~33 additional admin terms, bringing
   the domain to ~45. Target total: ~430 core terms.

2. **Domain rebalancing audit**: produce a per-domain term-count
   table and identify any other under-represented domains. Decision
   on further promotion deferred to that audit.

3. **Cross-lens consistency**: dataset changes apply to both Lens I
   and Lens IV simultaneously. Old published numbers (e.g. Lens I
   Δρ=0.260, Lens IV public/private 0.402) are re-computed from
   scratch on the new dataset; the old numbers are archived with a
   note, not retroactively "corrected".

4. **Control set**: unchanged (Swadesh 100), already adequate for
   the legal-vs-control signal test.

**Open question**: whether to re-derive the 7 domain labels from the
literature (Legrand, Merryman) or keep the ad-hoc taxonomy adopted
in `data/trace_dataset_design.md`. Recommend **keep** — changing
the domain taxonomy mid-project would invalidate all prior design
decisions on domain signal.

---

## D6 — Model panel expansion **[PROPOSED]**

Current panel (from `models/config.yaml`):
- 3 WEIRD base (BGE-EN, E5-large, FreeLaw-EN)
- 3 Sinic base (BGE-ZH, Text2vec-ZH, Dmeta-ZH)
- 2 bilingual (BGE-M3 560M, Qwen3-0.6B)

User signalled interest in "bigger models". The 0.6B Qwen3 was chosen
under 16GB RAM constraint; an upgrade path to 4B or 8B exists with
quantisation.

**Proposed changes**:

1. **Add Qwen3-Embedding-4B** as a 9th model (3rd bilingual, scale
   control): tests whether the bilingual cross-tradition drop is
   sensitive to model size within the same family. Under 4-bit
   quantisation, fits in 16GB RAM; attested-context encoding will
   be slow (expect 1-2 hours per language on 9472 terms at 120-char
   context truncation).

2. **Do not add Qwen3-8B**: marginal scientific value (Qwen3-4B
   already tests the size-scaling hypothesis), unfavourable cost
   ratio. Keep as "future work" if results suggest size matters.

3. **Keep all 6 base models unchanged**: the WEIRD/Sinic 3+3 symmetry
   is load-bearing for the cross-tradition design and should not be
   disturbed at this stage.

**Alternative considered**: add E5-Mistral-7B-instruct (Wang et al.
2024) as a WEIRD-side scale control analogous to Qwen3-4B on the
Sinic side. *Pro*: symmetrical scale-control. *Contro*: breaks the
3+3 base panel symmetry by adding an asymmetric 4th WEIRD model;
complicates the cross-pair count; 7B is at the upper edge of the
RAM budget even quantised. **Not recommended** for this pivot;
flag as Phase 3 if reviewers request it.

**Open question**: whether Qwen3-4B addition is worth the 4-6 hours
of additional compute on top of the D3/D4 rerun (~6 hours). The
conservative call is to **rerun Lens I and Lens IV on the existing
8-model panel first**, obtain clean headline numbers, and add
Qwen3-4B as a follow-up if the co-relatore requests a scale-control
claim.

---

## Execution order (after D3-D6 approval)

1. Audit admin candidates (user-driven curation) → updated dataset.
2. Pole-pair audit of existing Lens IV axes → updated `value_axes.yaml`.
3. Define new axes (doctrinal justification in Lens IV trace D-entry) →
   `value_axes.yaml` extended.
4. Precompute embeddings on new dataset for all 8 models, bare + attested.
5. Rerun Lens I on new dataset, 8 models, bare + attested.
6. Rerun Lens IV on new dataset, 8 models, bare + attested, new axes.
7. Rewrite `build_dashboard.py` around Lens I + IV only.
8. Regenerate dashboard. Produce one-page executive summary for the
   co-relatore from the new numbers.

Estimated time to step 7: 2-3 working days of curation + 1-2 overnight
compute runs.

---

## References

- Phase 2 plan: `~/.claude/projects/.../memory/project_phase2_plan.md`
- Phase 2 progress: `~/.claude/projects/.../memory/project_phase2_progress.md`
- Co-relatore feedback: `~/.claude/projects/.../memory/project_corelatore_feedback.md`
- Lens I trace: `experiments/lens_1_relational/trace.md`
- Lens IV trace: `experiments/lens_4_values/trace.md`
- Model selection trace: `experiments/models/trace_model_selection.md`
- Dataset trace: `experiments/data/trace_dataset_design.md`
