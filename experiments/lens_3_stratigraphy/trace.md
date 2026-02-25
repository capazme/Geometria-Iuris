# Trace: Lens III — Layer stratigraphy

**Thesis chapter(s)**: Ch.3 §3.1.3 — The depth of legal meaning
**Date**: 2026-02-22
**Status**: active

---

## Context

Lens III asks at what depth inside the transformer architecture legal meaning
crystallizes. The hypothesis (§3.1.3) is that different transformer layers encode
meaning at different levels of abstraction, mapping onto Tarello's interpretive
canons: literal (surface layers) → systematic → teleological → general principles
(deep layers). The analysis is **agnostic**: we measure first, interpret in Ch.4.

Two levels of analysis:
- **§3.1.3a** — Single-term behavior (drift + neighborhood change across layers)
- **§3.1.3b** — Global structure (domain signal emergence + RSA convergence)

All 6 models, each with its natural layer count.

---

## Decision log

### D1 — Layer extraction method
**Options considered**:
- Option A: Use SentenceTransformer `.encode()` with custom hooks (pro: minimal
  code; con: no direct access to hidden_states, fragile to library updates)
- Option B: Access `model[0].auto_model` with `output_hidden_states=True`,
  replicate native pooling (CLS or Mean) per model (pro: full control,
  reproducible; con: more code, must verify pooling fidelity)
- Option C: Use HuggingFace Transformers directly, bypass sentence-transformers
  (pro: cleanest API; con: must replicate tokenizer config and pooling exactly)

**Decision**: Option B
**Rationale**: sentence-transformers wraps a `Transformer` module at `model[0]`
whose `.auto_model` is a standard HuggingFace model. Setting
`auto_model.config.output_hidden_states = True` exposes the full hidden state
tuple (L+1 tensors including embedding layer). Native pooling mode is readable
from `model[1].pooling_mode_cls_token` (True → CLS; False → Mean). This
guarantees that `layers[:, -1, :]` matches the precomputed final-layer
embeddings from Lens I within float32 tolerance — a critical sanity check.

**Thesis text implication**: → §2.3 Models as cultural informants — the
extraction method is transparent and reproducible. → §3.1.3 The depth of legal
meaning — the method allows probing all L+1 representations (embedding layer
through final hidden state).

### D2 — Term set
**Options considered**:
- Option A: Core terms only (397) — same as Lens I for comparability
- Option B: Core + background (≈720) — richer signal but introduces noise from
  k-NN-assigned domains

**Decision**: Option A — Core terms only (397)
**Rationale**: Lens I used core terms for all structural analyses (RDM, RSA,
domain signal). Using the same set ensures that any difference between Lens I
(final layer) and Lens III (per-layer) is attributable to depth, not to a
different term pool. Background terms have k-NN-assigned domains with imperfect
confidence; mixing them in would confound the domain signal analysis.

**Thesis text implication**: → §3.1.3 — results are directly comparable with
§3.1 findings. The claim "domain signal emerges at layer L" is grounded in the
same 397 terms used to establish the signal exists at the final layer.

### D3 — Metrics (4 metrics, two per level)
**Options considered**:
- Drift only (cosine distance between consecutive layers)
- Jaccard only (k-NN overlap between consecutive layers)
- Both drift + Jaccard for §3.1.3a; domain signal r + RSA ρ for §3.1.3b
- CKA (centered kernel alignment) instead of RSA for §3.1.3b

**Decision**: 4 metrics across two sub-sections:
- **§3.1.3a** (single-term level):
  - *Drift*: `cosine_distance(vec[t, l], vec[t, l+1])` — measures how much a
    term's representation changes between consecutive layers
  - *Jaccard*: `1 - |kNN(t,l) ∩ kNN(t,l+1)| / |kNN(t,l) ∪ kNN(t,l+1)|`
    with k=7 — measures neighborhood instability between consecutive layers
- **§3.1.3b** (structural level):
  - *Domain signal r*: Mann-Whitney rank-biserial r from intra/inter-domain
    distance split at each layer — same metric as §3.1.1 but computed per layer
  - *RSA ρ*: `spearmanr(upper_tri(RDM_l), upper_tri(RDM_final))` — how similar
    is each layer's relational structure to the final output

**Rationale**: Drift and Jaccard capture complementary aspects of single-term
behavior (representational change vs neighborhood stability). Domain signal r
and RSA ρ capture complementary aspects of global structure (categorical
organization vs relational geometry). RSA ρ here is a simple Spearman
correlation (no Mantel test needed) because we compare representations within
the same model — the question is convergence, not cross-model agreement.

**Thesis text implication**: → §3.1.3 The depth of legal meaning — the four
metrics together answer: (a) where do individual terms undergo the most change?
(b) at what depth does legal-domain structure crystallize? The drift/Jaccard
pair maps onto the letterale→sistematico transition (§3.1.3a), while domain
signal r maps onto the emergence of categorical legal thinking (§3.1.3b).

### D4 — Falsification criterion
**Options considered**: N/A — this is a methodological requirement.

**Decision**: Three falsification conditions:
1. If domain signal r is flat across layers (no emergence pattern), legal
   meaning is not depth-dependent — the embedding layer already contains all
   categorical information.
2. If RSA ρ saturates at layer 1 (ρ ≈ 1.0 from the start), the relational
   structure is trivially determined by the input embeddings — depth adds
   nothing.
3. If drift is uniformly low across all layers (<0.01), the model barely
   transforms the input — the "layer stratigraphy" metaphor is vacuous.

**Rationale**: Each condition would invalidate a specific thesis claim. Together
they ensure the analysis is genuinely informative rather than confirming a
foregone conclusion.

**Thesis text implication**: → §3.1.3 — the falsification criteria are reported
in the methodology section. If any condition is met, the corresponding claim
is retracted and the section discusses *why* the hypothesis failed, which is
itself a meaningful finding.

---

## Model layer counts

| Model | Layers (L) | Hidden states (L+1) | Pooling | dim |
|---|---|---|---|---|
| BGE-EN-large | 24 | 25 | CLS | 1024 |
| E5-large | 24 | 25 | Mean | 1024 |
| FreeLaw-EN | 22 | 23 | Mean | 768 |
| BGE-ZH-large | 24 | 25 | CLS | 1024 |
| Text2vec-large-ZH | 24 | 25 | Mean | 1024 |
| Dmeta-ZH | 12 | 13 | CLS | 768 |

---

## Implementation notes

### MPS non-determinism (discovered 2026-02-22)

During implementation, we discovered that Apple MPS (Metal Performance Shaders)
produces non-deterministic results for BGE-ZH, Text2vec, and Dmeta models when
batch composition changes (batch_size=64 on 9472 terms vs batch_size=32 on 397
terms yields cos_sim as low as 0.23). The root cause is MPS's non-deterministic
handling of attention masking with padding.

**Resolution**: All precomputed embeddings and layer extractions are performed
on **CPU only** (deterministic to float32 precision). This adds ~2× wall time
but guarantees reproducibility across batch sizes, term sets, and runs.
The `--device cpu` flag is now the default.

→ §2.3 Models as cultural informants — the reproducibility guarantee is
important for scientific claims.

---

## Open questions

---

## References

- Ethayarajh, K. (2019). How contextual are contextualized word
  representations? Comparing the geometry of BERT, ELMo, and GPT-2
  representations. EMNLP 2019.
- Nili, H. et al. (2014). A toolbox for representational similarity analysis.
  PLoS Computational Biology, 10(4), e1003553.
- Tarello, G. (1980). L'interpretazione della legge. Giuffrè.
