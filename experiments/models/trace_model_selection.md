# Trace: Model selection

**Thesis chapter(s)**: Ch.7 §7.1 — Models as cultural informants
**Date**: 2026-02-21
**Status**: complete

---

## Context

Models are treated as cultural informants: their training corpora encode the
linguistic-normative practices of the communities that produced them (the
DigiVolksgeist hypothesis, Ch.5 §5.3). Model selection is therefore not a
neutral technical choice — it is a sampling decision with epistemological
consequences. The thesis must be transparent about which communities are being
sampled and why.

Two hard constraints apply to every model considered:

1. **Monolingual**: the model must be trained exclusively on the target language
   (English for WEIRD, Chinese for Sinic). Models trained on parallel translation
   corpora or with explicit cross-lingual alignment objectives are excluded, because
   their geometry would reflect a translation correspondence rather than an
   independent semantic space. The test for exclusion is the presence of cross-lingual
   retrieval benchmarks (MIRACL, MTEB cross-lingual) in the model's training signal
   or evaluation report — a high score on these benchmarks is evidence of cross-lingual
   alignment training, not a design choice that can be separated from the model.

2. **Bi-encoder dense retrieval architecture**: sentence-level embeddings via pooled
   transformer representations. Cross-encoders (e.g., rerankers) and sparse models
   (BM25, SPLADE) are excluded because they do not produce geometry-comparable
   vector spaces.

---

## Decision log

### D1 — Number of models per tradition
**Options considered**:
- Alt. A: Single model per tradition (1+1) — maximally parsimonious but
  non-replicable: any finding is confounded with idiosyncratic properties of
  one model's training corpus and architecture. A single model cannot distinguish
  between a robust cross-tradition signal and a corpus artifact.
- Alt. B: Two models per tradition (2+2) — marginal improvement over 1+1;
  a 3/4 majority threshold (one dissent) is insufficient to establish robustness,
  and 2 models are vulnerable to the case where both share a training corpus
  artifact.
- Alt. C: Three models per tradition (3+3) → 9 cross-tradition pairs (chosen) —
  produces a majority threshold of ≥7/9 pairs consistent for a finding to be
  declared robust, while remaining feasible computationally (9 embedding matrix
  pairs per experiment) and tractable in terms of model curation.
- Alt. D: Four or more models per tradition — diminishing returns beyond 3;
  model curation becomes the dominant methodological contribution at the expense
  of the substantive legal-semantic analysis.

**Decision**: 3 models per tradition (WEIRD) × 3 models per tradition (Sinic)
= 9 cross-tradition pairs. Robustness criterion: ≥7/9 pairs consistent.

**Rationale**:
The 3+3 design provides minimum viable robustness. Each finding reported in
Ch.9 is described as a "tradition-level signal" only if it holds in ≥7 of 9
model pairs. A single-pair finding (1+1) or a bare majority (5/9) would not
meet this threshold, making the epistemological cost of a weaker design too
high given the comparative claims the thesis advances. The design also
enables two layers of analysis: within-tradition consistency (do the 3 WEIRD
models agree with each other on Chapter 8 findings?) and cross-tradition
robustness (do the 9 pairs agree on Chapter 9 findings?). A 3+3 design is
the minimum that makes both layers meaningful.

**Thesis text implication**:
§7.1.1 (model selection rationale) introduces the 3+3 design and defines the
≥7/9 robustness threshold. The two-layer analysis structure (within-tradition
+ cross-tradition) is introduced in §7.2 (analysis plan). The threshold is
revisited in §9.1 when cross-tradition results are first reported.

---

### D2 — WEIRD model selection
**Options considered**:
- `BAAI/bge-en-icl` — excluded: uses in-context learning prompts (concatenated
  demonstrations) to condition the embedding; this mechanism conflates the query
  semantic with few-shot examples, making the geometry dependent on prompt design
  rather than on stable distributional knowledge. Inappropriate for single-term
  embedding where no demonstration context is available.
- `nomic-ai/nomic-embed-text-v1.5` — excluded (version revision): Nomic-v1.5
  requires a task prefix (`search_query:`, `search_document:`, etc.) to activate
  the correct projection head. Single legal terms do not map unambiguously to any
  task prefix, introducing a design choice that contaminated the embedding geometry.
  Nomic-v1.5 may be reconsidered in future work with a principled prefix strategy.
- `intfloat/e5-large-v2` — retained: contrastive training on MS-MARCO and NLI
  data, no cross-lingual alignment, monolingual EN, well-validated on MTEB-EN;
  STS-optimised architecture fills the second slot.
- `BAAI/bge-large-en-v1.5` — selected: large-scale (335M) RetroMAE backbone,
  monolingual EN, strong MTEB-EN performance, BGE family enables architecture
  control via symmetric BGE-ZH pairing (Slot 1).
- `freelawproject/modernbert-embed-base_finetune_512` — selected: finetuned on
  the CourtListener corpus (U.S. federal court opinions, a primary source of
  common law reasoning), ModernBERT backbone; provides a legally-informed
  informant rather than a general-corpus one.
- `sentence-transformers/all-mpnet-base-v2` — considered but excluded: general-
  purpose, smaller (110M), no legal domain coverage; dominated by the other candidates.

**Decision**: Three WEIRD models:
1. `BAAI/bge-large-en-v1.5` (BGE-EN-large, dim=1024) — architecture control
2. `intfloat/e5-large-v2` (E5-large, dim=1024) — STS-oriented
3. `freelawproject/modernbert-embed-base_finetune_512` (FreeLaw-EN, dim=768) — legally-informed

**Rationale**:
The three slots represent three distinct training philosophies, ensuring that
within-tradition consistency is not an artifact of shared training data or
architecture. Slots 1 and 2 use general-corpus models (BGE trained on retrieval
data, E5 trained on contrastive NLI/STS data); Slot 3 injects legal-domain
knowledge via corpus finetuning. If all three produce consistent geometries for
Chapter 8 findings, the convergence is robust across training regimes. The
dimension mismatch (FreeLaw 768 vs. 1024) is addressed by L2 normalization and
by the fact that RSA/GW/NDA operate on pairwise distance matrices, which are
dimension-independent.

**Thesis text implication**:
§7.1.2 (WEIRD model inventory) presents the three models with their training
regimes and justifies the exclusion of bge-en-icl and Nomic-v1.5. The cross-terna
symmetry principle (§7.1.4) explains why BGE-EN was chosen specifically as the
architecture-control anchor for the Sinic side.

---

### D3 — Sinic model selection
**Options considered**:
- `BAAI/bge-zh-v1.5` (base, 102M) — initially considered; excluded on scale
  grounds: 102M vs. 335M WEIRD models introduces a capacity asymmetry that
  confounds cross-tradition comparison with scale confound. Replaced by:
- `BAAI/bge-large-zh-v1.5` (335M) — selected: same BGE family as BGE-EN-large,
  equivalent scale (both 335M), monolingual ZH training, RetroMAE backbone.
  Provides the architecture-control pair for Slot 1.
- `shibing624/text2vec-base-chinese` — excluded: base scale (110M), asymmetric
  with E5-large-v2 (335M). Replaced by:
- `GanymedeNil/text2vec-large-chinese` (large scale, LERT-large backbone) —
  selected: CoSENT training on CNSD (Chinese Natural Language Inference +
  STS corpora), monolingual ZH, symmetric with E5-large-v2 in the STS-oriented
  slot. Slot 2.
- `uer/sbert-base-chinese-nli` — excluded: 2021 model, base scale, outdated;
  outperformed by text2vec-large-chinese on all relevant ZH STS benchmarks.
- `DMetaSoul/Dmeta-embedding-zh` — selected: finetuned on ZH legal, academic,
  and domain-rich corpora; monolingual ZH; provides the legally-informed Sinic
  informant symmetric with FreeLaw-EN. Slot 3.

**Decision**: Three Sinic models:
1. `BAAI/bge-large-zh-v1.5` (BGE-ZH-large, dim=1024) — architecture control
2. `GanymedeNil/text2vec-large-chinese` (Text2vec-large-ZH, dim=1024) — STS-oriented
3. `DMetaSoul/Dmeta-embedding-zh` (Dmeta-ZH, dim=768) — legally-informed

**Rationale**:
The Sinic terna mirrors the WEIRD terna in training philosophy:
- Slot 1 (BGE-ZH ↔ BGE-EN): same family, same scale, same backbone — any
  cross-tradition divergence at this pair cannot be attributed to architecture.
- Slot 2 (Text2vec-large ↔ E5-large): both STS-oriented, different families —
  if they agree with Slot 1, the signal is robust across training data.
- Slot 3 (Dmeta ↔ FreeLaw): both legally-informed — the pair is the most
  theory-relevant one: if two models trained on legal corpora in their respective
  traditions diverge, the divergence is most directly attributable to legal-semantic
  structure rather than general distributional differences.

The explicit cross-terna symmetry prevents the 9-pair robustness analysis from
conflating architecture effects with tradition effects.

**Thesis text implication**:
§7.1.3 (Sinic model inventory) presents the three models and their selection
rationale. §7.1.4 (cross-terna symmetry) formalises the slot structure and its
anti-confound logic.

---

### D4 — Cross-model robustness strategy
**Options considered**:
- Alt. A: Report results for one representative pair only — non-replicable;
  conceals model variance; epistemologically equivalent to 1+1 design despite
  having collected 3+3 data.
- Alt. B: Average across all 9 pairs, report aggregate — loses within-tradition
  signal; obscures the architecture-control slot comparison.
- Alt. C: Two-level reporting (Chapter 8: within-tradition; Chapter 9: cross-tradition
  aggregate with pair-level disclosure) — preserves both layers without overloading
  every result table. Chosen.

**Decision**: Two-level robustness reporting:
- **Level 1 (Chapter 8 — within-tradition)**: each experiment on WEIRD models
  is reported as three within-tradition results. A finding is declared robust
  within the WEIRD tradition if all 3 WEIRD models agree (direction + magnitude
  within a tolerance band). Disagreement across WEIRD models flags a model-
  specific artifact and is noted explicitly.
- **Level 2 (Chapter 9 — cross-tradition)**: each cross-tradition experiment
  produces 9 pair-level results. A finding is declared a tradition-level signal
  if ≥7/9 pairs are directionally consistent. The three slot-level sub-aggregates
  (Slot 1 pairs, Slot 2 pairs, Slot 3 pairs) are reported alongside the 9-pair
  aggregate to enable architecture-effect decomposition.

**Rationale**:
The two-level structure directly maps onto the thesis argument. Chapter 8 asks:
"Is there a stable WEIRD legal geometry?" — answered by within-tradition consistency.
Chapter 9 asks: "Do WEIRD and Sinic geometries diverge?" — answered by cross-tradition
pair agreement. Reporting raw pair-level scores in an appendix table (§A.2) preserves
full transparency without burdening the main text.

**Thesis text implication**:
§7.2 (analysis plan) introduces the two-level reporting strategy. §8.1 applies
Level 1 for the first time. §9.1 applies Level 2. §A.2 (appendix) contains
the full 9-pair result table for each experiment.

---

### D5 — Industrial model pair (abandoned)

**Finding**: No monolingual large-scale Chinese embedding model exists from
major industrial providers. This is a structural fact about the current state
of the Chinese AI ecosystem that has direct methodological consequences.

**Evidence gathered**:
- Cohere `embed-english-v3.0` (1024-dim, Cohere AI): strictly monolingual EN;
  one of the strongest English embedding models on MTEB-EN; would have been the
  natural WEIRD industrial candidate.
- Alibaba `Qwen3-Embedding`: explicitly multilingual by design (100+ languages);
  trained on multilingual parallel data; excluded by the monolingual constraint.
- Tencent `Seed1.5-Embedding`: multilingual (109 languages); trained with
  cross-lingual alignment objectives; excluded.
- Zhipu `embedding-3`: multilingual; excluded.
- ByteDance/Baidu: comparable large-scale embedding models either multilingual
  or not publicly released.
- Smaller ZH-only academic models exist (e.g., `shibing624/text2vec-base-chinese`)
  but are not comparable in scale to Cohere embed-english-v3.0 (no large-scale
  monolingual ZH industrial model above ~200M parameters as of February 2026).

**Structural interpretation**:
The asymmetry is not accidental. The Chinese AI industry's decision to develop
multilingual embedding models — rather than monolingual ZH models comparable
to Cohere's EN offering — reflects a commercial and geopolitical strategy
(cross-lingual retrieval for global deployment) that differs from the English-
language market's earlier pattern of monolingual specialisation. This asymmetry
in the AI ecosystem is itself a finding: the infrastructure available to encode
WEIRD legal tradition (multiple large-scale monolingual EN models from commercial
providers) differs structurally from the infrastructure available for the Sinic
tradition (reliance on academic monolingual models; industrial providers have
moved to multilingual). The thesis does not treat this asymmetry as a limitation
to be apologised for — it is a datum about the material conditions of NLP
research that warrants a brief note in §7.1.

**Decision**: Industrial model pair renounced. The 3+3 design uses academic
models only. The structural gap in the Chinese AI ecosystem is documented as a
methodological finding in §7.1, not as a limitation.

**Thesis text implication**:
§7.1 (sidebar or inline note, ~150 words): Notes that the natural experiment of
pairing a large English industrial model (Cohere) with a Chinese industrial
equivalent was not realisable because no monolingual large-scale Chinese
industrial embedding model exists as of February 2026. Interprets this as a
reflection of divergent commercial AI development strategies: EN providers
specialised (monolingual high-performance models); ZH providers globalised
(multilingual cross-lingual models). This is not framed as a weakness of the
thesis design but as a structural fact about the AI ecosystem that the thesis
operates in.

---

## Open questions

---

## References

- BAAI. BGE-large-en-v1.5. Hugging Face, 2024. https://huggingface.co/BAAI/bge-large-en-v1.5
- BAAI. BGE-large-zh-v1.5. Hugging Face, 2024. https://huggingface.co/BAAI/bge-large-zh-v1.5
- Wang, L., et al. Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5). arXiv:2212.03533, 2022.
- GanymedeNil. text2vec-large-chinese. Hugging Face, 2023. https://huggingface.co/GanymedeNil/text2vec-large-chinese
- DMetaSoul. Dmeta-embedding-zh. Hugging Face, 2024. https://huggingface.co/DMetaSoul/Dmeta-embedding-zh
- Free Law Project. modernbert-embed-base_finetune_512. Hugging Face, 2025. https://huggingface.co/freelawproject/modernbert-embed-base_finetune_512
- Muennighoff, N., et al. MTEB: Massive Text Embedding Benchmark. EACL 2023.
- Su, J., et al. CoSENT: Consistent Sentence Embeddings. arXiv:2109.12684, 2021. [CoSENT training for text2vec]
