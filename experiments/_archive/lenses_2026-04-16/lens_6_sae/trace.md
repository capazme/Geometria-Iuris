# Trace: Lens VI — Sparse Autoencoder Decomposition

**Thesis chapter(s)**: §3.1.1 extension (mechanistic evidence for domain signal) /
§4.1 (what the experiments say together) — **index mapping TBD**
**Date**: 2026-04-13
**Status**: in progress

---

## Context

Lenses I-V establish that legal categories have geometric reality in embedding
spaces (domain signal, topology, neighborhoods, values), and that this structure
diverges systematically across legal traditions. But they treat the embedding
space as a black box: we measure distances without knowing *what* the model
encodes in each dimension.

Lens VI asks a mechanistic question: can the model's representation be decomposed
into interpretable features, and if so, do those features align with legal domain
categories? A positive answer would provide evidence that the geometric structure
documented in Lenses I-V is not an artifact of distance metrics but reflects
genuine internal organisation of legal meaning.

This connects to the epistemological argument of the thesis (→ §1.4): if the
instrument not only measures distances but can be shown to *encode* legally
meaningful distinctions, the measurability thesis gains a mechanistic footing.

**Risk note**: This is the most exploratory part of the thesis. SAEs on sentence
embeddings are a 2025 development with limited precedent. If the analysis does
not produce interpretable features, the thesis stands on Lenses I-V + β + δ.

---

## D1 — Architecture choice

### Options considered

#### Option A — Standard SAE (ReLU + L1 penalty)
- Architecture: z = ReLU(W_enc · x + b_enc); x̂ = W_dec · z + b_dec
- Sparsity: L1 penalty on z, coefficient λ tuned empirically
- Pro: simplest, most cited (Anthropic Monosemanticity 2023)
- Con: L1 coefficient λ is hard to tune; trades off reconstruction vs sparsity

#### Option B — TopK SAE (Gao et al. 2024)
- Architecture: same encoder/decoder, but activation = TopK(pre-activation)
- Sparsity: fixed L0 = k (exactly k features active per input)
- Pro: no λ tuning, clean sparsity control, BatchTopK variant used in
  "Decoding Dense Embeddings" (EMNLP 2025) on sentence embeddings
- Con: harder gradient flow (top-k is not smooth), needs auxiliary dead neuron loss

#### Option C — Gated SAE (Rajamanoharan et al. 2024)
- Architecture: separate gating network decides which features activate
- Pro: better reconstruction at same sparsity (DeepMind)
- Con: more parameters, more complex, less precedent on sentence embeddings

### Decision: Option B — TopK SAE

**Rationale**: The closest methodological precedent (arXiv:2506.00041, EMNLP 2025)
uses BatchTopK on sentence-level dense embeddings with strong interpretability
results. TopK eliminates the L1 coefficient search, which is critical given our
limited compute budget. The auxiliary dead neuron loss addresses gradient flow.

**Thesis text implication** → §2.4 / §4.1: "We adopt the TopK sparse autoencoder
architecture (Gao et al. 2024), following its successful application to sentence
encoder interpretability (arXiv:2506.00041)."

---

## D2 — Training data

### Options considered

#### Option A — Bare embeddings only (9,472 × 1024)
- Pro: cleanest signal, each vector = one legal term's bare representation
- Con: small N; 9,472 samples for thousands of features

#### Option B — Bare + all 8 context embeddings per term (~75K vectors)
- Would require re-encoding to store individual context vectors (currently
  only the mean is stored)
- Pro: 8× more data
- Con: context embeddings are noisy variants of the same term; SAE might
  learn "context variation" rather than "legal structure"

#### Option C — Bare + e-Legislation corpus embeddings
- Embed thousands of legal passages from HK e-Legislation
- Pro: massive data augmentation
- Con: changes the unit of analysis from "legal term" to "legal passage";
  the feature-to-domain mapping becomes less interpretable

### Decision: Option A — Bare embeddings only

**Rationale**: The thesis question is whether the *term-level* representation
encodes legal structure. Using bare embeddings keeps the unit of analysis clean:
each training sample is one legal concept, and feature activations can be directly
mapped to the 9-domain taxonomy. The 9,472 sample size is small but viable if we
constrain the expansion factor (see D3). If reconstruction quality is poor, we
revisit Option B as a sensitivity check.

**Thesis text implication** → §3.1.1 extension: "The SAE is trained on the bare
embedding vectors of the 9,472-term legal lexicon, preserving the one-to-one
correspondence between training samples and legal concepts."

---

## D3 — Expansion factor and sparsity level

### Options considered

| Config | Features | k | L0 ratio | Activations/feature (avg) |
|--------|----------|---|----------|--------------------------|
| 4× / k=32 | 4,096 | 32 | 0.78% | 74 |
| 8× / k=32 | 8,192 | 32 | 0.39% | 37 |
| 4× / k=16 | 4,096 | 16 | 0.39% | 37 |
| 16× / k=64 | 16,384 | 64 | 0.39% | 37 |

Reference: "Decoding Dense" uses 32× / k=32-128 with 8.8M samples.
Scaling heuristic: our data is ~1000× smaller, so ~30× less expansion.

### Decision: Primary config 4× / k=32, ablation sweep

**Rationale**: 4,096 features with k=32 gives ~74 activations per feature on
average — enough to detect domain-level patterns with 9 categories. The
activations/feature count is the binding constraint: below ~30, statistical
tests for domain enrichment lose power. We run an ablation sweep over
{2×, 4×, 8×} and k ∈ {16, 32, 64} to check sensitivity.

**Thesis text implication** → §2.4: "The expansion factor (4×) and sparsity
level (k=32) are calibrated to the lexicon size, ensuring sufficient
activations per feature for domain enrichment analysis."

---

## D4 — Loss function and training procedure

### Decision

**Loss**: MSE reconstruction + auxiliary dead neuron loss (Gao et al. 2024)

```
L = ||x - x̂||² + α · L_aux
```

where L_aux encourages dead neurons to reconstruct the input using the
top-2k dead neurons (prevents feature collapse). α = 1/32 following
arXiv:2506.00041.

**Training**:
- Optimizer: AdamW (lr=1e-3, weight_decay=0.0)
- Batch size: 256
- Epochs: 1000 (dataset is small → each epoch = 37 steps, total ~37K steps)
- Warmup: 100 epochs linear LR warmup
- Decoder weight normalization: unit norm after each step
- Dead neuron resampling: every 100 epochs
- Device: MPS (Apple Silicon)
- Estimated training time: ~5-10 min for primary config

**Rationale**: Higher LR than the retrieval paper (1e-3 vs 5e-5) because
our dataset is small and well-structured; we need fast convergence to avoid
overfitting. Weight decay is zero because decoder normalization already
regularizes. 1000 epochs sounds large but each epoch processes only 9,472
samples — total compute is modest.

**Thesis text implication** → §2.4: training procedure described as standard
TopK SAE with hyperparameters calibrated to the lexicon size.

---

## D5 — Analysis plan

### Feature → domain mapping

For each learned feature f_i (i = 1..4096):
1. **Top-activating terms**: rank all 9,472 terms by activation a_i(t)
2. **Domain enrichment**: for each domain d, compute:
   - Proportion of top-50 activating terms from domain d
   - Fisher's exact test (one-sided) for overrepresentation
   - Holm correction across all (feature × domain) tests
3. **Domain selectivity index (DSI)**: for each feature, the Gini impurity
   of its top-50 activations across domains. DSI = 1 means all top activators
   come from one domain (maximally selective); DSI = 1/9 = random.
4. **Feature coverage**: for each domain, how many features are significantly
   enriched? Are some domains "sharper" than others in feature space?

### Reconstruction quality

- Explained variance ratio (EVR): 1 - ||x - x̂||² / ||x||²
- Cosine similarity between input and reconstruction (mean ± std)
- These metrics validate that the SAE learns a faithful decomposition, not noise.

### Interpretability validation

- Manual inspection of top-20 features by DSI: are the top-activating
  terms semantically coherent to a legal expert?
- Comparison with Lens I domain signal: do SAE-discovered domains align
  with the a priori taxonomy?

**Thesis text implication** → §3.1.1 extension / §4.1: "The SAE reveals
[N] features with significant domain enrichment (Fisher's exact, p < 0.05,
Holm-corrected), of which [M] are selectively activated by terms from a
single legal domain."

---

## D6 — Thesis section mapping (OPEN)

Lens VI is not yet in the canonical index (`003_GeometriaIuris_Indice.docx`).
Options for placement:

1. **§3.1.1 extension**: as mechanistic evidence for domain signal
2. **New §3.4**: a fourth experiment chapter
3. **§4.1**: as part of "what the experiments say together" (synthesis)

Decision deferred until results are available. If features are strongly
domain-selective, §3.1.1 extension is strongest. If results are mixed,
§4.1 as interpretive supplement is safer.

---

## D7 — Results (2026-04-13)

### Training (primary config: 4× expansion, k=32)

| Metric | Value |
|--------|-------|
| Dict size | 4,096 features |
| Training time | 14.1 min (MPS) |
| MSE | 0.000022 |
| Explained Variance Ratio | 0.977 |
| Cosine similarity (mean ± std) | 0.989 ± 0.003 |
| L0 (features per sample) | 32.0 ± 0.0 |
| Active features | 4,085 / 4,096 (11 dead) |

Reconstruction is near-perfect: the SAE captures 97.7% of the variance in
BGE-EN-v1.5 embeddings using only 32 active features per term.

### Domain enrichment

27 features with significant domain enrichment (Fisher's exact, p < 0.05,
Holm-Bonferroni corrected across 28,672 tests). Distribution by domain:

| Domain | Significant features | Top feature example |
|--------|---------------------|---------------------|
| criminal | 7 | F2005: criminalisation, criminal law, convict (14/14) |
| labor_social | 7 | F3322: outworker, workplace, worker, sick leave (18/19) |
| international | 5 | F405: Geneva Convention, Genocide Convention (23/26) |
| administrative | 3 | F3178: salaries tax, taxation, taxable (12/12) |
| constitutional | 3 | F1122: voting, casting vote, election (6/6) |
| civil | 2 | F3032: tenant, tenancy, unexpired tenancy (17/17) |
| procedure | 0 | — |

### Interpretation (level 2)

The SAE reveals that BGE-EN-v1.5 internally organizes legal concepts along
domain boundaries. Features like F2005 (criminal law), F3322 (labor law),
and F405 (international treaties) activate almost exclusively for terms from
their respective domains. This is mechanistic evidence that the domain signal
documented in Lens I (RSA ρ = 0.23–0.30) corresponds to interpretable,
sparse internal representations — not merely to aggregate distance patterns.

The criminal and labor domains have the most dedicated features (7 each),
consistent with their distinct vocabularies. Civil law, the largest domain
(136 terms), has only 2 significant features after correction, suggesting
its semantic space overlaps with other domains — aligning with Lens V's
finding that civil law terms have the lowest neighborhood stability.

Procedure (53 terms) has zero significant features. This may reflect its
cross-cutting nature: procedural terms (e.g., "appeal", "jurisdiction")
activate in the context of any substantive domain.

### Limits (level 3)

1. **Small labeled set**: only 430 of 9,472 terms carry domain labels;
   the enrichment analysis is conditioned on this subset. Features that
   activate for unlabeled terms (9,042) are invisible to the domain analysis.
2. **Stringent correction**: Holm-Bonferroni across 28,672 tests is
   conservative. Many features with DSI = 1.0 and 5–11 labeled terms
   do not survive correction. The 27 significant features are a lower
   bound on domain-selective features.
3. ~~Single model~~ — **resolved by D8 cross-model analysis** (see below).
4. **Feature semantics are post-hoc**: domain labels are assigned by
   statistical enrichment, not by inspecting the decoder directions.
   A feature enriched for "criminal" may encode a linguistic pattern
   (e.g., Latinate vocabulary) rather than legal-domain membership.

---

## D8 — Cross-model SAE analysis (2026-04-13)

Trained identical TopK SAE (4x, k=32) on all 10 embedding models.
Total training time: 112.5 min (MPS).

### Reconstruction quality (all models)

| Model | Tradition | EVR | cos | Active |
|-------|-----------|-----|-----|--------|
| E5-large | WEIRD | 0.984 | 0.992 | 4073/4096 |
| BGE-M3-ZH | Bilingual | 0.983 | 0.992 | 4096/4096 |
| BGE-ZH-large | Sinic | 0.982 | 0.991 | 4096/4096 |
| Dmeta-ZH | Sinic | 0.981 | 0.991 | 3066/3072 |
| BGE-M3-EN | Bilingual | 0.979 | 0.989 | 4096/4096 |
| Qwen3-0.6B-ZH | Bilingual | 0.979 | 0.989 | 4096/4096 |
| Qwen3-0.6B-EN | Bilingual | 0.978 | 0.989 | 4096/4096 |
| BGE-EN-large | WEIRD | 0.977 | 0.989 | 4096/4096 |
| Text2vec-large-ZH | Sinic | 0.967 | 0.983 | 4096/4096 |
| FreeLaw-EN | WEIRD (legal) | 0.952 | 0.976 | 3072/3072 |

All models decompose well (EVR > 95%). FreeLaw-EN has the lowest EVR,
suggesting legal fine-tuning creates representations that resist sparse
decomposition (richer internal structure).

### Significant features per domain (Fisher exact, Holm p < 0.05)

| Domain | BGE-EN | E5 | FrLaw | BGE-ZH | Txt2v | Dmeta | M3-EN | M3-ZH | Qw-EN | Qw-ZH |
|--------|--------|-----|-------|--------|-------|-------|-------|-------|-------|-------|
| constitutional | 3 | 1 | **6** | 1 | 0 | 0 | 2 | 1 | **8** | 3 |
| criminal | **7** | 2 | 5 | 2 | 4 | 1 | 5 | 0 | **12** | 2 |
| civil | 2 | 2 | 1 | 1 | 1 | 1 | 3 | 1 | 3 | 2 |
| international | 5 | **7** | 3 | **9** | 4 | 3 | **7** | 5 | **9** | 5 |
| labor_social | **7** | 5 | 5 | 6 | **5** | **5** | 3 | 3 | 5 | 3 |
| administrative | 3 | 1 | 0 | 2 | 1 | 3 | 1 | 1 | 1 | 3 |
| procedure | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **TOTAL** | **27** | **18** | **20** | **21** | **15** | **13** | **22** | **11** | **38** | **18** |

### Interpretation (level 2)

1. **Universality**: all 10 models produce domain-selective SAE features
   (13-38 significant, Holm-corrected). The phenomenon is not specific to
   BGE-EN-v1.5 — it holds across architectures (BERT, decoder), languages
   (EN, ZH), and legal traditions (WEIRD, Sinic, bilingual).

2. **International and labor are universal signals**: these two domains
   have significant features in every model, likely because their
   vocabularies are highly distinctive (treaties, conventions / worker,
   employer, sick leave).

3. **Criminal is EN-dominant**: 5-12 features in EN models, 0-4 in ZH
   models. The Anglo-Saxon criminal vocabulary (theft, felony,
   manslaughter) is more specialized than its Chinese counterpart.

4. **Procedure is transparent**: 0 significant features in 9/10 models.
   Procedural terms activate across substantive domains — they have no
   dedicated features because they are domain-agnostic.

5. **FreeLaw constitutional boost**: the legally fine-tuned model has 6
   constitutional features (top domain), vs 0-3 in general-purpose models.
   Legal fine-tuning amplifies constitutional distinctions.

6. **Qwen3-0.6B-EN is richest**: 38 significant features. The decoder
   architecture may encode domain membership more granularly than
   encoder-only models.

7. **ZH systematically less selective**: BGE-M3 EN vs ZH (22 vs 11),
   Qwen3 EN vs ZH (38 vs 18). Consistent with Lens I RSA: within-Sinic
   coherence is lower than within-WEIRD.

### Limits (level 3)

1. Feature counts are not directly comparable across models with different
   input dimensions (768 vs 1024 → different dict sizes: 3072 vs 4096).
2. The enrichment test depends on the EN/ZH term labels, which were
   assigned based on English legal taxonomy. ZH models may organize
   concepts along different categorical boundaries.
3. No causal analysis: we observe correlation between features and domains,
   not that features *cause* domain membership in the representation.

---

## References

- Gao, L., et al. (2024). Scaling and evaluating sparse autoencoders.
  arXiv:2406.04093.
- Bussmann, B., et al. (2025). Decoding dense embeddings: Sparse autoencoders
  for interpreting and discretizing dense retrieval. EMNLP 2025.
  arXiv:2506.00041.
- Tehenan, M., et al. (2025). Mechanistic decomposition of sentence
  representations. arXiv:2506.04373.
- Bricken, T., et al. (2023). Towards monosemanticity: Decomposing language
  models with dictionary learning. Anthropic.
- Rajamanoharan, S., et al. (2024). Improving dictionary learning with
  gated sparse autoencoders. arXiv:2404.16014.
- Li, W., Michaud, E. J., & Tegmark, M. (2025). The geometry of concepts:
  Sparse autoencoder feature structure. arXiv:2410.19750.
- Sofroniew, N., et al. (2026). Emotion concepts in Claude Sonnet.
  arXiv:2604.07729.
