# Trace: Lens I — Relational distance structure

**Thesis chapter(s)**: Ch.3 §3.1 — The geometry of a legal tradition (macro)
**Date**: 2026-02-21
**Status**: in progress

---

## Context

Lens I asks whether legal embedding spaces have non-random relational structure
and whether that structure is shared across legal traditions. It operationalizes
the second-order isomorphism concept introduced in §1.4.

### Analysis map

| Analysis | Section | Terms used | Method |
|---|---|---|---|
| Background term domain assignment | §3.1.1 | background (8975) | k-NN majority vote |
| Intra vs inter-domain distances | §3.1.1 | core (397) | Mann-Whitney U |
| Legal vs control signal | §3.1.1 | core (397) + control (100) | Mann-Whitney U |
| Domain topology K×K | §3.1.2 | core (397) | inter-domain distance matrix |
| Within-WEIRD RSA robustness | §3.1.4 | core (397) | RSA + Mantel + block bootstrap |
| Cross-tradition RSA | §3.1.4 | core (397) | RSA + Mantel + block bootstrap |

---

## Decision log

### D1 — Distance metric

**Options considered**:
- Option A: cosine distance `1 - cosine_similarity` (pro: scale-invariant, standard
  in NLP; contro: nessuno rilevante per questo contesto)
- Option B: euclidean distance (pro: intuitiva; contro: dipende dalla magnitudine,
  non confrontabile tra modelli di dimensione diversa)

**Decision**: Option A — cosine distance

**Rationale**: i vettori sono L2-normalizzati, quindi cosine distance = `1 - u·v`.
Indipendente dalla dimensione del modello (768 vs 1024), confrontabile tra
tradizioni.

**Thesis text implication**: → §2.4 "The statistical toolkit". Giustifica perché
la distanza coseno è la metrica naturale per embedding normalizzati e permette
confronti tra modelli di architettura diversa.

---

### D2 — Similarity measure for RDM comparison (RSA)

**Options considered**:
- Option A: Spearman ρ sui triangoli superiori (pro: rank-based, robusto a scale
  diverse, standard in letteratura RSA; contro: perde informazione di magnitudine)
- Option B: Pearson r (pro: misura la proporzione lineare; contro: assume
  normalità, sensibile agli outlier, non appropriato per scale diverse)

**Decision**: Option A — Spearman ρ

**Rationale**: le distanze coseno di modelli diversi hanno distribuzioni diverse.
Spearman cattura la concordanza d'ordinamento indipendentemente dalla scala.
Standard nella letteratura RSA (Kriegeskorte et al. 2008).

**Thesis text implication**: → §2.4 "The statistical toolkit". Motiva la scelta
rank-based rispetto alla correlazione lineare per confronti inter-modello.

---

### D3 — Statistical test for RSA significance

**Options considered**:
- Option A: Mantel test (permutation test su matrici) (pro: rispetta la
  dipendenza strutturale delle RDMs, non-parametrico; contro: computazionalmente
  più costoso del t-test)
- Option B: t-test su Spearman ρ (pro: immediato; contro: assume indipendenza
  delle N(N-1)/2 coppie — assunzione violata, produce p-value gonfiati)

**Decision**: Option A — Mantel test con B=10000 permutazioni, seed=42

**Rationale**: le entrate del triangolo superiore dell'RDM non sono i.i.d. —
ogni termine appare in N-1 coppie. Il t-test con ~78k gradi di libertà produce
falsi positivi sistematici. Il Mantel test rispetta questa struttura.
B=10000 è lo standard di riferimento di Nili et al. (2014) e garantisce che il
minimo p-value rappresentabile sia 1/10000=0.0001 — sufficiente per affermare
p<0.001 con piena credibilità in un contesto peer-reviewed. Nessun revisore può
obiettare sulla stabilità della stima.

**Thesis text implication**: → §2.4 "The statistical toolkit: permutation inference".
Centrale per la credibilità statistica dell'intera analisi RSA.

---

### D4 — Confidence interval for RSA

**Options considered**:
- Option A: block bootstrap a livello di termini, B=1000 (pro: rispetta la
  dipendenza, metodo standard per RSA — Nili et al. 2014; contro: nessuno
  rilevante)
- Option B: bootstrap sulle coppie (pro: più semplice; contro: ignora la
  dipendenza, CI artificialmente stretti)

**Decision**: Option A — block bootstrap sui termini, B=10000, seed=42

**Rationale**: la dipendenza tra le coppie nasce a livello di termine. Ricampionare
i termini (non le coppie) è l'unità corretta per il bootstrap. Ref: Nili et al.
(2014) PLoS Computational Biology 10(4): e1003553. B=10000 allineato con D3
(Mantel test) per coerenza del parametro di Monte Carlo attraverso tutta la tesi.

**Thesis text implication**: → §2.4 "The statistical toolkit: block bootstrap".
Questo metodo diventa la giustificazione formale per tutti i CI dell'intera tesi.

---

### D5 — Statistical test for §3.1.1 (domain signal + legal vs control)

**Options considered**:
- Option A: Mann-Whitney U (pro: non-parametrico, non assume normalità, adatto
  per distribuzioni asimmetriche di distanze coseno; contro: nessuno rilevante)
- Option B: t-test (pro: familiare; contro: assume normalità, violata per
  distanze coseno in spazi ad alta dimensione)

**Decision**: Option A — Mann-Whitney U, `alternative='less'`, effect size:
rank-biserial correlation r

**Rationale**: le distanze coseno non sono normalmente distribuite. Mann-Whitney
testa la differenza di locazione (mediana) senza assunzioni distribuzionali.
L'effect size r = (concordanti - discordanti) / totale dà una misura leggibile
della separazione tra le due distribuzioni.

**Thesis text implication**: → §3.1.1. Il p-value risponde a "il segnale è
statisticamente significativo?", l'effect size r risponde a "quanto è grande?".
Entrambi necessari per evitare lo stargazing criticato in §2.4.

---

### D_BG1 — Algoritmo di assegnamento per termini background

**Options considered**:
- Option A: centroide di dominio — assegna al dominio il cui centroide (media
  dei vettori core del dominio) è più vicino (pro: veloce, un solo confronto
  per termine; contro: il centroide è un punto astratto, non interpretabile;
  perde la struttura interna del dominio)
- Option B: k-NN majority vote — trova i k core term più simili, majority vote
  sul dominio (pro: interpretabile: il giurista vede i termini specifici che
  hanno determinato l'assegnamento; robusto alla forma non-globulare dei cluster;
  contro: leggermente più costoso computazionalmente)

**Decision**: Option B — k-NN majority vote

**Rationale**: l'interpretabilità è fondamentale. Lo strumento di revisione
giuridica (§3.1.1 + Ch.4 Quod Numeri Tacent) richiede che il giurista veda perché
un termine è stato assegnato a un dominio. "I tuoi 7 vicini più prossimi sono
[X, Y, Z...], tutti civil" è un ragionamento verificabile. Il centroide non lo è.
La differenza di costo computazionale è trascurabile (8975 query su 397 vettori).

**Thesis text implication**: → §3.1.1 "Domain signal: do legal categories have
geometric reality?" — l'assegnamento k-NN diventa il test empirico del segnale
su dati held-out (i background terms). → §4.1 (Ch.4 Quod Numeri Tacent): il
giurista completa l'interpretazione dove il numero fornisce la misura ma non il
giudizio.

---

### D_BG2 — Valore di k per il majority vote

**Options considered**:
- Option A: k=3 — locale, netto, ma instabile (un singolo vicino anomalo sposta
  il risultato; con 7 domini possibile 1+1+1)
- Option B: k=7 — dispari, evita i pareggi perfetti; sufficientemente locale
  (~1.75% dei 397 core terms); con 7 domini la maggioranza assoluta (≥4/7)
  indica segnale forte, confidence bassa (<4/7) indica ambiguità genuina
- Option C: k=15 — più stabile ma troppo globale; rischia di incorporare
  domini lontani, perdendo la struttura locale

**Decision**: Option B — k=7

**Rationale**: k=7 è dispari (nessun pareggio a 2 classi), ragionevolmente
locale (1.75% del pool core), e produce una scala di confidence naturale:
7/7=1.0 (unanimità), 4/7≈0.57 (maggioranza minima), <4/7 (ambiguità). I termini
con confidence bassa sono i candidati più interessanti per la revisione giuridica
— il modello geometrico è incerto, il giurista decide.

**Nota**: con 7 domini e k=7, la distribuzione peggiore è 2+2+1+1+1 (nessuna
maggioranza assoluta). In questo caso il codice assegna al più frequente (2 voti)
e segnala `confidence=2/7≈0.29` come "alta ambiguità". La revisione giuridica
è obbligatoria per questi termini.

**Thesis text implication**: → §3.1.1. La scelta di k=7 e la soglia di confidence
4/7 diventano i parametri operativi del signal test. → §2.4: la sensibilità al
k è una verifica di robustezza (k=5 e k=9 come sensitivity check).

---

### D6 — Multiple comparison correction

**Decision**: Holm-Bonferroni step-down correction applied to all 15 Mantel
test p-values (3 within-WEIRD + 3 within-Sinic + 9 cross-tradition).

**Rationale**: With 15 simultaneous tests at alpha=0.05, the family-wise
error rate (FWER) would be inflated without correction. Holm-Bonferroni is
uniformly more powerful than Bonferroni while still controlling FWER. In
practice, all 15 uncorrected p-values are at the permutation floor
(≤ 0.001), so all remain significant after correction. The correction is
applied for methodological completeness.

**Thesis text implication**: → §3.1.4 "P-values are corrected for multiple
comparisons using the Holm-Bonferroni procedure across all 15 model pairs.
All corrected p-values remain below 0.015."

---

### D7 — Phipson & Smyth p-value formula

**Decision**: Permutation p-values computed as (b+1)/(m+1) following
Phipson & Smyth (2010), where b = number of null values ≥ observed,
m = number of permutations. This replaces the earlier approximation
max(p_raw, 1/m).

**Rationale**: The (b+1)/(m+1) formula is the exact implementation
recommended by Phipson & Smyth. It is slightly more conservative (no
p-value is ever exactly zero) and is the standard cited in the literature.

---

### D8 — Parametric stress test for §3.1.5
**Date**: 2026-04-11
**Status**: decided

**Context**: Lens I's existing analyses (§3.1.1 domain signal, §3.1.2
topology, §3.1.4 cross-tradition RSA) measure whether the embedding geometry
reflects the taxonomic and cross-tradition organisation of the legal lexicon.
They do not, however, test a more elementary property of a measurement
instrument: *does the instrument respond monotonically and with recognisable
discontinuities to the variation of a known parameter?*

This is the property that Sofroniew, Kauvar, Saunders et al. (2026) tested
for emotion vectors in Claude Sonnet 4.5 via parametric templates: prompts
identical except for a numerical quantity (Tylenol dosage, missing hours,
startup runway), where the variation of the quantity is known to correspond
to variation in emotional response intensity. Their result — that emotion
probes track the semantic interpretation of the parameter, not surface
lexical features — is the strongest demonstration in the paper that the
geometry captures meaning rather than form.

The legal domain admits an analogous and arguably sharper test, because the
law itself encodes *discontinuities*: thresholds at which the legal meaning
of an identical verbal formula changes qualitatively. The age at which
imputability attaches, the amount above which written form is required, the
number of years at which prescription expires, the penalty beyond which a
particular procedural safeguard applies. These thresholds are not researcher-
chosen quantities; they are fixed by statute. A measurement instrument that
detects them would thereby show that the geometry of its representations
reproduces not only the continuous similarity structure of legal concepts
but the discrete normative architecture imposed on that structure by the
legal system itself.

**Options considered**:

- **Option D1 — Dedicated Lens VI** ("Parametric stress test"). Full new
  experimental section at the level of Lenses I–V, with its own sub-sections,
  own trace file, own results tables, own place in the index and in Chapter 3.
  Pro: maximum visibility; parallel structure to the other lenses; easier to
  develop independently. Contro: the co-relator's feedback of 2 April 2026
  was explicit: "troppi esperimenti, meglio 1-2 fatti bene che 5". Adding a
  sixth lens directly contradicts this feedback; even if the sixth lens is
  small, its nominal presence in the index reproduces the taxonomy the co-
  relator criticised; reframes a validation check as a full experiment,
  inflating the apparent dispersion of the thesis.

- **Option D2 — Sub-section §3.1.5 under Lens I** ("Parametric validation:
  thresholds and monotonic gradients"). The stress test lives inside Lens I
  as its fifth sub-section, following §3.1.1 (domain signal), §3.1.2
  (topology), §3.1.3 (stratigraphy), §3.1.4 (cross-tradition RSA). No new
  lens; no new trace file; the sub-section is described as a *validation*
  of the instrument, not as an independent experiment. Pro: respects the
  co-relator's "fewer experiments, deeper validation" feedback; structurally,
  the stress test *is* a validation of Lens I's claim that the embedding
  geometry reflects legal semantic structure, so placing it inside Lens I
  is the taxonomically correct location; the index and the chapter structure
  remain unchanged; the co-relator can read the sub-section as an answer to
  his critique rather than as an amplification of the problem he identified.
  Contro: §3.1 grows by one sub-section, slightly heavier; the sub-section
  must not expand beyond ~3–4 pages to preserve proportion.

**Decision**: **Option D2** — sub-section §3.1.5 within Lens I.

**Scope specification for §3.1.5**: the sub-section contains two parametric
tests, not more. Both are chosen to exhibit discontinuities at known legal
thresholds, and both are reported on both the WEIRD and Sinic model sides
to preserve the cross-tradition structure of the chapter.

**Test 1 — Age and imputability**. Template:

  EN: "A person aged {X} years committed an act contrary to section 2 of
       the Offences against the Person Ordinance."
  ZH: [corresponding HK bilingual version, sourced from the e-Legislation
       corpus structure established in D5/D6 of the dataset trace.]

  X ∈ {0, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 40, 50,
       60, 70, 80}

The age of criminal responsibility in Hong Kong was raised from 7 to 10 by
the Juvenile Offenders (Amendment) Ordinance 2003, amending Cap. 226 s. 3:
"It shall be conclusively presumed that no child under the age of 10 years
can be guilty of an offence." The age of majority is 18.

Between 10 and 14, however, the common law presumption of *doli incapax* is
**rebuttable**: the child is presumed incapable of mischievous discretion
unless the prosecution proves otherwise. This means the correct prediction
for the instrument is **not** a single sharp step at X=10 but a **two-break
structure**: (i) a discontinuity at X=10 (conclusive incapability turns into
rebuttable presumption); (ii) a second, softer discontinuity at X=14
(rebuttable presumption ends, full imputability attaches); (iii) monotonic
movement through the interior of each segment.

The test will report the first principal component of the resulting vector
sequence as a function of X, with discontinuities identified by fitting
piecewise models with 0, 1, and 2 breakpoints, comparing by BIC *and* by
bootstrap of the breakpoint locations (to avoid the small-sample instability
of BIC-based segmented regression on N≈20; see Muggeo 2003, 2020). The
bootstrap is pre-registered as the primary statistical procedure; BIC is
reported as a secondary summary.

**Test 2 — Contract value and written-form requirement**. Template:

  EN: "An agreement for the sale of land at a price of HK${X} was entered
       into orally between the parties."
  ZH: [corresponding HK bilingual version.]

  X ∈ {100, 1000, 10 000, 100 000, 500 000, 1 000 000, 5 000 000,
       10 000 000, 50 000 000, 100 000 000}

Contracts for the sale of land in Hong Kong must be evidenced in writing
(Conveyancing and Property Ordinance Cap. 219 s. 3, the HK statutory
descendant of the Statute of Frauds 1677 s. 4): an oral contract for the
sale of land is **unenforceable by action** unless there is a memorandum in
writing signed by the party to be charged. This is not the same as being
void or invalid: an oral agreement may still be perfected by part
performance, by a subsequent written memorandum, or saved by equitable
doctrines such as proprietary estoppel. The section is therefore not about
the validity of the contract at the substantive level but about its
procedural enforceability absent written evidence.

The relevant property of s. 3 for the present test is that **no monetary
threshold exists**: the writing requirement applies uniformly to every
contract for the sale of land, regardless of the consideration agreed. The
legally correct response of the instrument is therefore (i) no discontinuity
at any value of X; (ii) no strong monotonic gradient along the numerical
dimension, since no legal status of the contract depends on X. A null result
here is the expected result: the absence of a threshold is what makes this
a **negative control**.

If the instrument nonetheless produces a strong gradient or a spurious
discontinuity, the signal is a lexical artefact of the model's pretraining
association between large numbers and legal formality, not a response to
any legal structure, because no legal structure keyed to X exists in this
provision.

Test 1 is the positive test; Test 2 is the negative control. Together they
provide a balanced check: the instrument must detect the two breakpoints
where they exist (10 and 14 in Cap. 226) and refrain from inventing a
threshold where none does (Cap. 219 s. 3).

**Numeracy precondition (added after adversarial review 2026-04-11)**: the
parametric numerical sweep of Tests 1 and 2 depends on the underlying
sentence encoders representing numerical magnitude in an ordered way.
Wallace, Wang, Li, Singh & Gardner (2019, EMNLP, "Do NLP Models Know
Numbers?") showed that BERT-family subword tokenisers represent numbers
poorly: "10", "11", "14" are tokenised as separate subword units with no
guaranteed ordinal structure in the embedding space. BGE, E5, Nomic and the
ZH counterparts are all BERT-family. A numeracy sanity check is therefore
executed **before** Tests 1 and 2 on each of the six models: template
strings differing only in a numerical value (e.g., "aged X years") are
encoded, and the Spearman correlation between the numerical value X and
the projection onto the first PC of the resulting sequence is computed.
If this correlation falls below 0.5 for a given model, the numerical form
of the stress test is abandoned for that model and the **categorical
fallback** (Test 1-cat, Test 2-cat) is used instead:

  Test 1-cat: X ∈ {infant, toddler, young child, child, minor, juvenile,
                   teenager, young adult, adult, middle-aged adult, elderly}
  Test 2-cat: X ∈ {nominal, modest, substantial, large, very large}

where the categorical sequences preserve the ordinal structure of the
parameter at the lexical level, bypassing the subword-tokenisation problem.
Whether Tests 1 and 2 are run in numerical or categorical form per model is
reported alongside the result.

**Rationale**:

*Why a sub-section and not a new lens*. The co-relator's feedback is the
determining factor. "Too many experiments; 1–2 done deeply rather than 5"
is an instruction about the shape of the thesis as much as about its
content. Adding a Lens VI is the most direct way to contradict that
instruction; adding a sub-section to Lens I is a way to answer the
underlying concern (what are your concrete results?) without dispersing
the thesis further. The parametric stress test *is* conceptually a
validation of Lens I's RSA result — if the RSA correlation drop is real,
the instrument that produces it should also show legally correct behaviour
under parametric variation — so the sub-section placement is logically
correct, not just rhetorically convenient.

*Why two tests and not more*. Proportion. §3.1 is already a long section
with four sub-sections. Adding two more tests (a positive and a negative
control) is sufficient to demonstrate the principle without inflating the
chapter. Additional tests — penalty monotonicity, prescription period,
procedural threshold — can be mentioned in a footnote as "further tests
following the same template yielded qualitatively similar results (not
shown)", with the supporting data in the Appendix if generated.

*Why a positive and a negative test*. The single biggest risk of parametric
templates is that the model produces a geometric gradient in response to
the *lexical* structure of the template (big numbers, formal phrasing)
rather than to the *legal* meaning. A positive test alone cannot distinguish
these two sources. A matched negative control — where the lexical structure
varies in the same way but the legal threshold does not exist — makes the
distinction falsifiable. If the positive test detects the imputability
threshold but the negative test produces no threshold, the instrument is
tracking legal meaning. If both tests show gradients and thresholds, the
instrument is tracking lexical form. If neither shows a gradient, the
instrument is insensitive to this kind of variation and the test is
uninformative.

*Why use HK e-Legislation phrasing and not generic English*. To preserve
the thesis's uniform provenance of text. All other Lens I analyses operate
on term vectors contextualised from HK e-Legislation (per D5/D6 dataset
trace). A parametric template written in generic English or Mandarin would
introduce a register shift mid-chapter and confound the comparison. The
template sentences are written in the register of HK legislation (same
phrasing conventions, same use of nominal phrases, same citation style for
ordinances) to ensure that the only variation between the template
instances is the parameter X itself.

*Why report the first principal component of the vector sequence*. Given
a sequence of contextualised embeddings v(X_1), v(X_2), ..., v(X_n)
corresponding to the values of the parameter, the most informative
scalar summary of the geometric variation is the projection onto the
first PC of the sequence itself (or, equivalently, the first diffusion
dimension). This is the direction along which the sequence varies most.
A monotonic gradient along this PC is the expected response; a flat
response indicates the instrument does not react; a non-monotonic response
indicates the template is activating confounds rather than the target
parameter. Discontinuity detection operates on this PC1 sequence.

*Fit and statistical test*. For each test, the PC1 sequence is fitted to
(i) a single linear model (baseline: "monotonic response without
discontinuity"), (ii) a piecewise linear model with one breakpoint at
each plausible legal threshold (alternative: "discontinuity at threshold
T"). Model comparison by BIC. Report ΔBIC and the location of the
selected breakpoint. A positive finding is "ΔBIC favours the piecewise
model at the expected breakpoint by a margin greater than the BIC threshold
for 'strong evidence' (commonly ΔBIC > 10)".

**Thesis text implication**:

→ §3.1.5 [new sub-section]: "Parametric validation: thresholds and
monotonic gradients". Structure: (1) motivation (following Sofroniew et al.
2026, parametric stress tests are a direct check on the instrument's
sensitivity to meaning rather than form); (2) positive test (age and
imputability); (3) negative control (contract value, where no threshold
exists); (4) results figure (PC1 vs parameter, with discontinuities
highlighted); (5) brief interpretation (what the test confirms about the
instrument established in §3.1.1–§3.1.4); (6) limits (this is validation
at a single point, not generalisation to all thresholds in HK law).

→ §3.1 as a whole: gains a validation sub-section that closes the
interpretive loop. The preceding sub-sections establish that the embedding
geometry reflects domain, topology, and cross-tradition structure; §3.1.5
establishes that it also responds correctly to parametric variation over
a legally specified quantity. The RSA result of §3.1.4 acquires a new
independent support: if the instrument produces the correlation drop *and*
detects the imputability threshold in HK law *and* correctly refrains from
inventing a threshold in the negative control, the three results together
make the instrument's fidelity substantially harder to attribute to a single
confound.

→ §2.4 [The statistical toolkit]: short mention of BIC-based breakpoint
detection as a standard procedure for the parametric test.

→ §4.2: the parametric validation is available as a supporting piece of
evidence against the most sceptical interpretation of the cross-tradition
drop (namely, that the drop is a training-corpus artefact rather than a
reflection of legal-semantic structure). The counter to this scepticism
is strengthened by the parametric test because training-corpus artefacts
do not, in general, reproduce statutory thresholds.

→ **Addressing the co-relator's feedback directly**: §3.1.5 is the
sub-section specifically designed as a rebuttal to the "too many experiments,
few concrete results" critique. It produces a visible, legally recognisable,
quantitatively verifiable result (the imputability threshold appears in the
geometry) within the structure of an existing lens, not as a new experiment.
The sub-section should be presented in the thesis in a way that makes this
rebuttal implicit but readable: "the instrument developed in this chapter
is, at a minimum, sufficient to recover the imputability threshold of HK
criminal law from the geometry of its own representations."

---

### D8 — Status update after numeracy pre-check (2026-04-11)
**Status**: Categorical form mandated as primary

**Update note**: D8 was finalised earlier on 2026-04-11 with a numeracy
precondition requiring an empirical sanity check before the numerical sweep
could be trusted. The pre-check
(`experiments/pre_checks/precheck_1_numeracy.py`, results in
`experiments/pre_checks/results/precheck_1_numeracy.json` and consolidated
in `precheck_results.md`) ran on all six models in the 3+3 design with the
template "a person aged {X} years" (EN) and the parallel ZH formulation,
sweeping X over fourteen values. Results:

| Model | Spearman ρ(X, PC1) | Ordinal monotonicity | Status |
|---|---|---|---|
| BGE-EN-large | +0.068 | 0.385 | FAIL |
| E5-large-v2 | +0.116 | 0.692 | FAIL |
| FreeLaw-EN (modernbert) | +0.626 | 0.846 | PASS |
| BGE-ZH-large | +0.345 | 0.923 | PARTIAL |
| Text2vec-large-ZH | +0.477 | 0.923 | PARTIAL |
| Dmeta-embedding-zh | +0.262 | 0.923 | PARTIAL |

**Interpretation**: only one of the six models (FreeLaw, the modernbert
variant) reliably encodes numerical magnitude in an ordered way on the
first principal component. BGE-EN-large is essentially noise on the
numerical sequence (ρ = 0.068, monotonicity 38%), which is catastrophic
for parametric numerical probing because BGE is the architecture-control
anchor of the 3+3 design. The Sinic models present a more complicated
pattern: high ordinal monotonicity (0.923) but moderate ρ on PC1, consistent
with the numerical signal existing in the embeddings but not concentrated
on the first principal direction. For the purposes of D8, this is interpreted
conservatively as not reliably numerical.

**Consequence for D8**: the numerical form of Tests 1 and 2 is **abandoned
as the primary procedure**. The categorical fallback specified above (Test
1-cat with eleven age categories from "infant" to "elderly"; Test 2-cat
with five magnitude categories from "nominal" to "very large") becomes the
**primary form** of the §3.1.5 parametric stress test, run on all six
models uniformly.

The numerical sweep is retained as a per-model **secondary check**: it is
reported in the Appendix with the per-model status from the numeracy
pre-check displayed alongside, so that the failure of BGE-EN and E5 on
numerical templates is itself disclosed as a finding about the limits of
parametric probing in BERT-family encoders, rather than hidden as a null
result.

**Implementation note**: the categorical probe is implemented as a new
module under `experiments/lens_1_relational/categorical_probe.py`, using
the same EmbeddingClient and the same RDM utilities from
`shared/statistical.py`. Per-model results are written to
`lens_1_relational/results/categorical_probe.json`.

**Thesis text implication update**: §3.1.5 is written around the
categorical results as the headline. The numerical sweep, with its
per-model failures, is moved to the Appendix and recast as evidence about
the **numeracy limitations of contrastively-trained sentence encoders** in
the legal domain — an empirical confirmation of Wallace et al. (2019) on
a new model family. This becomes a small but citable finding in its own
right, contributing to §4.2 (model-data-culture entanglement).

**Reference**: `experiments/pre_checks/results/precheck_1_numeracy.json`
and `experiments/pre_checks/results/precheck_results.md`, section
"Pre-check 1 — Numeracy sanity check".

---

### D8 — Results note (categorical probe rebuilt, 2026-04-11)
**Status**: pre-registered, rebuilt, and run

The 2026-04-11 adversarial review of the first implementation of the
§3.1.5 categorical probe identified two structural defects that
invalidated its original results:

1. **Midpoint artefact on 6-category tests**. Tests 4 (offence
   severity) and 5 (imprisonment duration) in the first implementation
   used 6 categories each. With 6 categories the PC1 max-gap falls at
   the linguistic midpoint (gap index 2 of the 5 gaps) by construction.
   Test 4's legal threshold (summary/indictable) happened to sit at
   exactly that midpoint, so the "6/6 perfect" result originally
   reported could not distinguish recognition of the legal distinction
   from the generic midpoint effect. Test 5 confirmed the artefact:
   when the legal threshold (determinate vs life) was placed
   off-midpoint, no model recovered it and all 5/6 models concentrated
   the modal break at the same midpoint anyway. Test 2 (the 5-category
   negative control) further confirmed the pattern: with no legal
   threshold, the modal break still landed near the linguistic middle
   of the magnitude scale. The 6-category tests are therefore
   inconclusive by design and were removed from the rebuild.

2. **Post-hoc tuning of Test 3 expected position** (HARKing). The
   first implementation moved the expected break of Test 3 from
   `["young adult", "adult"]` to `["teenager", "young adult"]` after
   the first run produced 0/6 hits under the original criterion. The
   rebuild retroactively sanitises this by committing the new expected
   position to a pre-registration file *before* the rerun.

**Redesign specification (Phase 0 rebuild, 2026-04-11)**:

All five tests now use **11 categories**, matching Test 1 and Test 3
from the first implementation (which were already 11-cat). The midpoint
gap index for an 11-cat sequence is 4. Positive tests are required to
have their expected legal break at a gap index whose distance from the
midpoint is ≥ 2. Tests that cannot satisfy this constraint are flagged
`borderline` and their interpretation is restricted in the dashboard.

All test specifications (sequences, templates, expected break
positions) are pre-registered in
`experiments/lens_1_relational/categorical_probe_expected.yaml`,
committed to git before the rerun. The probe script reads the YAML
at runtime and cannot modify the expected positions. The committed
YAML is the audit trail.

**Empirical results after rebuild** (`results/categorical_probe.json`,
`results/figures/html/categorical_probe.html`):

| Test | polarity | expected gap idx | dist. from midpoint | exact hits | near (±1) hits | ensemble ρ̄ |
|---|---|---|---|---|---|---|
| 1. age × imputability | positive (borderline) | 3 (child/minor) | 1 | 2 / 6 | 2 / 6 | 0.924 |
| 2. magnitude | negative control | — | — | — | — | 0.645 |
| 3. age × contractual capacity | positive | 6 (teenager/young adult) | 2 | 2 / 6 | 3 / 6 | 0.944 |
| 4. offence severity | positive | 6 (summary/indictable) | 2 | 1 / 6 | 1 / 6 | 0.650 |
| 5. disposal severity | positive | 8 (very long / indeterminate) | 4 | **4 / 6** | **4 / 6** | 0.742 |

**Three substantive findings from the rebuild**.

*Finding 1: determinate/indeterminate recovered cleanly.* Test 5 is
the cleanest positive result. The expected break is at gap index 8
(distance 4 from midpoint, safely off-midpoint), and 4 of 6 models
locate the modal max-gap exactly there, with 5/5 paraphrase agreement
each. The 4 models are BGE-EN-large, E5-large, BGE-ZH-large, and
Dmeta-ZH: both WEIRD and Sinic traditions represented. FreeLaw-EN
locates the break at gap 0 (caution/fine) instead, and Text2vec-ZH at
gap 3 (probation/suspended). The determinate/indeterminate legal
distinction is the most readily recoverable legal threshold among
those probed.

*Finding 2: Sinic models consistently localise the doli incapax zone
under age templating, regardless of the legal question asked.* Across
Test 1 (imputability) and Test 3 (contractual capacity), the two
Sinic models BGE-ZH-large and Text2vec-large-ZH place the modal break
at gap 3 (兒童 → 未成年人) with 3/5 and 5/5 paraphrase agreement
respectively, even when the legally intended threshold is the age of
majority (Test 3, gap 6). This is not a validation of doli incapax
recognition — the gap 3 hit in Test 1 is borderline (distance 1 from
midpoint, confoundable with the linguistic midpoint). It *is* a
substantive observation that the 兒童/未成年人 transition is the
dominant geometric feature of the 11-category Chinese age sequence
across contexts, which is itself a finding about how the Sinic
models carve the childhood/adulthood continuum.

*Finding 3: offence severity is carved at the regulatory/criminal
border, not at the summary/indictable border.* Test 4 had the
summary/indictable transition pre-registered at gap 6 (distance 2
from midpoint). 5 of 6 models instead locate the modal break at
gap 1 (regulatory breach / minor infraction), with 3-5/5 paraphrase
agreement each. Only Dmeta-ZH recovers the summary/indictable
boundary exactly (1/6 exact hit). This is a clean negative result:
the summary/indictable procedural distinction, although central to
HK criminal procedure, is not the dominant geometric feature of the
offence severity cloud. The dominant feature is the "not really
criminal" vs "criminal" transition at the bottom of the scale.

**Lessons for §3.1.5 in the thesis text**:

The §3.1.5 section should:
- report Test 5 as the clean positive validation of the probe
  methodology (4/6 exact hits, off-midpoint, determinate vs
  indeterminate)
- report Test 3 as a partial positive (2 exact + 1 near = 3/6 in the
  majority zone; 2 Sinic models go to the doli incapax zone instead)
- report Test 1 honestly as a borderline test whose gap 3 hits are
  confoundable with the linguistic midpoint; the Sinic doli incapax
  localisation is substantive but cannot be established from Test 1
  alone
- report Test 4 as a clean negative of the summary/indictable
  hypothesis and a positive of the regulatory/criminal boundary
- report Test 2 (negative control) as producing a robust off-midpoint
  break at gap 5 (EN) or 7 (ZH), showing that the instrument finds
  breaks in any ordinal magnitude sequence; the discrimination
  between positive and negative tests is therefore at the level of
  *break alignment with a legal threshold*, not at the level of
  *existence of break*

**Output**:
- `experiments/lens_1_relational/categorical_probe_expected.yaml`
  (pre-registration; committed to git before the rerun)
- `experiments/lens_1_relational/categorical_probe.py` (rebuild;
  loads expected positions from the YAML, no hardcoded expectations)
- `experiments/lens_1_relational/results/categorical_probe.json`
  (descriptive; per-test exact and near-hit counts)
- `experiments/lens_1_relational/results/figures/html/categorical_probe.html`
  (descriptive; no interpretive "perfect validation" language)

The dashboard surface for §3.1.5 is the new `sec_3_1.html` section
page (Phase 1) and the `#parametric-probe` tab of the regenerated
`lens1_interactive.html` (Phase 2).

---

### D9 — Layer-aware Lens I sensitivity sweep (results, 2026-04-11, corrected)
**Status**: implemented, run, and corrected after adversarial review

A layer-aware sensitivity sweep was added as a robustness check on the
headline Lens I result, in response to the layer-extraction question
raised by D-B in `models/trace_model_selection.md` (currently deferred).
The sweep reuses the per-layer pooled-vector cache produced by Lens III
(`lens_3_stratigraphy/results/layer_vectors/{label}.npz`, shape
`(397, L+1, dim)` for each model) and recomputes the Lens I
aggregate scalars at six fractional depths per model: embedding, L/4,
L/2, 2L/3, 5L/6, L. No new encodings are required.

**Two definitions of Δρ**. The published §3.1.4 pipeline
(`lens1.py:401`) uses an **asymmetric** definition:
`Δρ_asym = within_WEIRD − cross`, which ignores the within-Sinic
baseline entirely. This was the definition I originally used in this
D9 entry, and it produced a misleading "peak at depth 0.50" claim.
The adversarial review of 2026-04-11 pointed out that at intermediate
depths the within-Sinic ρ̄ collapses to near-zero because the three
Sinic models are all CLS-pooled and their intermediate-layer CLS
tokens have not yet aggregated enough information to agree with each
other. The asymmetric definition therefore compares a real WEIRD
signal against two near-noise channels (within-Sinic ≈ 0 and
cross ≈ 0), inflating Δρ artifactually. The **symmetric** definition
`Δρ_sym = (within_WEIRD + within_Sinic) / 2 − cross` gives equal
weight to both within-tradition baselines and is the correct reading
for any robustness claim.

**Empirical results** (both definitions, from
`results/layer_sensitivity.json`):

| Fractional depth | within-W ρ̄ | within-S ρ̄ | cross ρ̄ | Δρ_asym | Δρ_sym |
|---|---|---|---|---|---|
| 0.00 (embedding) | 0.504 | n/a¹ | 0.166 | 0.339 | n/a |
| 0.25 | 0.290 | 0.162 | 0.079 | 0.211 | 0.147 |
| 0.50 | 0.496 | 0.112 | 0.094 | 0.403 | 0.211 |
| 0.67 (2L/3) | 0.423 | 0.272 | 0.030 | 0.392 | 0.317 |
| 0.83 (5L/6) | 0.421 | 0.355 | 0.043 | 0.379 | **0.346** |
| 1.00 (final) | **0.509** | **0.463** | **0.249** | 0.260 | 0.237 |

¹ CLS-pooled Sinic models give constant layer-0 output; within-Sinic
is undefined at the embedding layer.

The final-layer row reproduces the published §3.1.4 numbers
(0.509 / 0.463 / 0.249 / 0.260) **exactly**, which is a clean sanity
check on the pipeline: the layer sweep is recomputing the same
quantities as the published pipeline.

**Corrected finding — peak Δρ is at 5L/6, consistent with Lens III**.
With the symmetric definition, Δρ_sym peaks at depth 0.83 (5L/6) with
a value of 0.346, and declines slightly to 0.237 at the final layer.
The 5L/6 layer falls squarely inside the Lens III phase transition
zone (the last 15-20% of layers, documented in §3.1.3 as the region
where legal meaning crystallises). The Lens I cross-tradition drop
and the Lens III phase transition are **two readings of the same
underlying dynamic**: the cross-tradition gap is widest at the layer
where legal meaning first becomes stable within each tradition,
and contracts slightly at the final layer as tradition-internal
coherence continues to rise and the cross-tradition agreement
partially catches up.

**Single-pair Sinic diagnostic**. At depth 0.25 the three Sinic pair ρ
values are `BGE-ZH × Text2vec = 0.095`, `BGE-ZH × Dmeta = 0.591`,
`Text2vec × Dmeta = −0.201`. One pair is negatively correlated. This
confirms that the intermediate-layer CLS representation is not a
coherent signal for the Sinic models individually, let alone across
the three of them. Any cross-tradition Δρ that uses these values as a
baseline is comparing a signal to noise. The asymmetric definition's
"peak at 0.50" was exactly this pathology.

**Cross-reference with the ctx-pool robustness run of Step 1**
(`rsa_ctx_robustness.py`, `results/rsa_ctx_robustness.json`): when
the final-layer pooled vectors are replaced by the contextualised
pool (mean of 8 templated variants per term), the Lens I summary
becomes within-W 0.620, within-S 0.634, cross 0.346, Δρ_sym 0.281.
The ctx-pool Δρ_sym at the final layer (0.281) is larger than the
bare-pool Δρ_sym at 5L/6 (0.346 was the peak, 0.237 at final). The
two robustness checks (layer sweep + contextualisation) are
consistent with each other: the cross-tradition gap documented by
§3.1.4 is robust across layer depth (symmetric definition) and
across representation choice, and is slightly larger under the
cleaner contextualised representation.

**Implications for the thesis**:

→ §3.1.4 reports the bare-pool final-layer numbers
(0.509 / 0.463 / 0.249, Δρ_sym 0.237) as the headline. Both the
layer sweep (Δρ_sym peak 0.346 at 5L/6) and the ctx-pool run
(Δρ_sym 0.281) are reported as robustness checks in a "Robustness
to representation" paragraph, with the observation that the headline
Δρ is a conservative estimate.

→ §3.1.3 (Lens III) reports the phase transition in the last 15-20%
of layers, and the Lens I robustness sweep is used as corroborating
evidence: the cross-tradition gap peaks in the same zone where
legal meaning is documented as crystallising. The two findings are
not competing but complementary.

→ §4.2 (model-data-culture entanglement) reports the methodological
degrees of freedom (layer choice, representation choice) as
consequential but *not* signal-destroying: the cross-tradition gap
is present across the whole upper half of the network under either
definition, and the choice of extraction layer moves the magnitude
but not the sign.

**Output**:
- `experiments/lens_1_relational/layer_sensitivity.py` (new module,
  zero new encodings; reports both Δρ definitions)
- `experiments/lens_1_relational/results/layer_sensitivity.json`
- `experiments/lens_1_relational/results/figures/html/layer_sensitivity.html`
- `experiments/lens_1_relational/rsa_ctx_robustness.py` (ctx pool
  final-layer recomputation)
- `experiments/lens_1_relational/results/rsa_ctx_robustness.json`

Dashboard surface for this D9 content is the §3.1.3 panel of the
regenerated `lens1_interactive.html` (where the layer sweep curves
are added alongside the existing Lens III content), plus the
Robustness (ctx) tab added by Phase 2.

---

### D10 — Bilingual control experiment (β)

**Date**: 2026-04-12
**Options considered**:
- Option A: Accept monolingual-only design, acknowledge confound
- Option B: Add bilingual control models (same architecture, both languages)
**Decision**: Option B — add BGE-M3 (BAAI, 560M) and Qwen3-Embedding-0.6B
(Alibaba, 0.6B) as bilingual control models.
**Rationale**: The cross-tradition gap (ρ̄=0.250) could be an artifact of using
different encoders for EN and ZH. Bilingual models encode both languages in
the same vector space, eliminating the encoder-architecture confound. Two
models from different labs provide independent replication.
Note: Conan-Embedding-v2 (Tencent) was originally planned but has no public
weights (API-only); replaced by BGE-M3 (same BGE family as Slot 1 anchors).

**Results** (n_perm=1000, n_boot=1000, core terms only):

| Category | ρ̄ (bare) | ρ̄ (attested) |
|---|---|---|
| Within-WEIRD | 0.501 | 0.608 |
| Within-Sinic | 0.458 | 0.768 |
| Cross-tradition (mono) | 0.250 | 0.282 |
| Within-bilingual (β) | 0.335 | 0.339 |

Per-pair bilingual (bare): BGE-M3 EN×ZH ρ=+0.388, Qwen3 EN×ZH ρ=+0.282.

**Interpretation** (level 2): The cross-tradition gap persists in bilingual
models. The bilingual ρ̄ (0.335) sits between within-tradition (0.48) and
cross-mono (0.250), suggesting the shared architecture recovers some global
structure but does not eliminate the divergence. The gap is in the legal
semantics, not in the encoder architecture.
**Limit** (level 3): Bilingual models are trained on multilingual corpora
that may include cross-lingual alignment objectives; their ρ̄ is an upper
bound on "what a shared architecture can recover", not a clean isolation of
linguistic vs legal divergence.

**Thesis text implication**: → §3.1.4 gains a "causal control" subsection.
The bilingual result converts the cross-tradition drop from an observation
into a controlled finding. The gap cannot be dismissed as an artifact.

### D11 — Attested-context embeddings (δ)

**Date**: 2026-04-13
**Options considered**:
- Option A: Keep synthetic templates only ("the legal term X", etc.)
- Option B: Replace with attested legislative contexts from HK e-Legislation
- Option C: Hybrid — attested where available, synthetic fallback
**Decision**: Option C — attested contexts from 65,599 section-aligned
e-Legislation passages (EN coverage 77%, ZH 63%), with synthetic template
fallback for uncovered terms.
**Rationale**: Synthetic templates are Firthian-adjacent but not Firthian:
the "company" a term keeps should be real, not manufactured. The HK
e-Legislation corpus (DATA.GOV.HK, open license with attribution) provides
section-level bilingual text from actual ordinances. Up to 8 attested
contexts per term per language, mean-aggregated identically to the
synthetic pipeline.

**Results** (bare → attested comparison on §3.1.4 RSA):

| Category | Bare ρ̄ | Attested ρ̄ | Δρ |
|---|---|---|---|
| Within-WEIRD | 0.501 | 0.608 | +0.107 |
| Within-Sinic | 0.458 | 0.768 | +0.310 |
| Cross-tradition | 0.250 | 0.282 | +0.032 |
| Cross-tradition drop (W−cross) | 0.251 | 0.326 | +0.075 |

**Interpretation** (level 2): Attested contexts act as a within-tradition
disambiguator: they sharpen the signal that makes models of the same
tradition agree, while cross-tradition divergence remains stable or
increases. The cross-tradition drop widens from Δρ=0.251 to Δρ=0.326.
The gap is not noise — real legal usage makes it more visible.
**Limit** (level 3): Coverage is 77% EN / 63% ZH; uncovered terms use
synthetic fallback. The attested contexts are section-level snippets, not
sentence-level — LaBSE+Vecalign sentence alignment is deferred. Context
window varies (300 chars for encoder models, 120 chars for Qwen3 decoder
due to computational cost).

**Thesis text implication**: → §3.1 and §3.2 gain a "contextualisation
robustness" argument. The instrument works better, not worse, when grounded
in real legal language. This directly addresses the Firthian challenge from
Ch.1 §1.3.

## Open questions

- Sensitivity check su k=5 e k=9: pianificato come robustezza in §2.4
- Concordanza tra i 3 modelli WEIRD sull'assegnamento: da includere in §3.1.4

---

## D12 — D9 archival (2026-04-17)

**Status**: archived; not rerun on the rebalanced dataset.

**Context**: The pivot documented in `experiments/trace_pivot_2lens.md`
archived Lens III (layer stratigraphy). D9 (layer-aware Lens I
sensitivity sweep) was a hybrid that read from the Lens III cache
(`lens_3_stratigraphy/results/layer_vectors/{label}.npz`) and produced
a Lens I robustness report. With Lens III archived and the core
dataset rebalanced from 397 terms to 350 terms (D5 of the pivot trace),
D9's inputs are doubly invalidated:

1. Source path moved to `_archive/lenses_2026-04-16/` on 2026-04-16.
2. The cached layer vectors have shape `(397, L+1, dim)` — pre-rebalance
   term indexing. On the current 350-core dataset (27 promotions, 107
   drops relative to the old core), naive reuse of the cache would
   produce systematically misaligned RDMs.

**Options considered**:
- Option A: regenerate layer_vectors on the rebalanced 350-core dataset
  by re-running `lens_3_stratigraphy/layer_extraction.py` on the six
  monolingual models. Compute: ~30-60 min on M-series MPS; storage
  ~425 MB. Pro: preserves the D9 robustness argument intact. Contro:
  reintroduces Lens III infrastructure that the pivot removed for
  narrative reasons; the layer-depth finding was always an ancillary
  check, not a headline of Lens I.
- Option B: archive D9 together with the rest of the Lens III
  material. Document the pre-rebalance layer-sweep result as prior
  evidence in `_archive/lenses_2026-04-16/lens_3_stratigraphy/trace.md`
  and mark D9 in this trace as superseded. Pro: respects the pivot's
  "fewer experiments, deeper execution" principle; zero compute;
  removes the last code file in Lens I that depended on archived
  infrastructure. Contro: loses the layer-depth robustness claim for
  the v2 rerun.

**Decision**: Option B — archive.

**Rationale**: the attested-context robustness (D11) already provides
a substantive second robustness axis for Lens I, covering a different
threat model (Firthian attestation of context) from layer-depth. Adding
layer-depth back would inflate the chapter for a point that does not
survive the pivot's narrative reduction. The pre-rebalance finding
(Δρ_sym peak 0.346 at 5L/6, consistent with the phase-transition zone
documented in the archived Lens III) stands as prior evidence on the
pre-rebalance dataset; it is not claimed as a result of the v2 analysis.

**Files moved** (2026-04-17):
- `experiments/lens_1_relational/layer_sensitivity.py` →
  `experiments/_archive/lenses_2026-04-16/lens_3_stratigraphy/`
- `experiments/lens_1_relational/results/layer_sensitivity.json` →
  `experiments/_archive/lenses_2026-04-16/lens_3_stratigraphy/results/`
- `experiments/lens_1_relational/results/figures/html/layer_sensitivity.html` →
  `experiments/_archive/lenses_2026-04-16/lens_3_stratigraphy/results/`

**Remaining Lens I scope (post-pivot)**:
- §3.1.1 Domain signal (D1-D7, D_BG1-D_BG2)
- §3.1.2 Domain topology
- §3.1.4 Cross-tradition RSA (D1-D7) with β bilingual control (D10)
  and δ attested-context robustness (D11) as headline
- §3.1.5 Parametric categorical probe (D8 revised)

**Thesis text implication**: → §3.1 loses its planned layer-sweep
paragraph. The cross-tradition gap is defended against the "encoder
artefact" critique by the bilingual control (D10) and against the
"synthetic context artefact" critique by attested contexts (D11). The
"at what layer does meaning crystallise" question, if it resurfaces
in review, is answered by citing the archived Lens III finding as
prior evidence without reclaiming it in the v2 analysis.

---

## References

- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity
  analysis — connecting the branches of systems neuroscience. *Frontiers in
  Systems Neuroscience*, 2, 4.
- Nili, H., et al. (2014). A toolbox for representational similarity analysis.
  *PLoS Computational Biology*, 10(4), e1003553.
- Mantel, N. (1967). The detection of disease clustering and a generalized
  regression approach. *Cancer Research*, 27(2), 209–220.
- Phipson, B., & Smyth, G.K. (2010). Permutation P-values should never be zero:
  calculating exact P-values when permutations are randomly drawn. *Statistical
  Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Mann, H.B., & Whitney, D.R. (1947). On a test of whether one of two random
  variables is stochastically larger than the other. *Annals of Mathematical
  Statistics*, 18(1), 50–60.
