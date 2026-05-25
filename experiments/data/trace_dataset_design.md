# Trace: Dataset design

**Thesis chapter(s)**: Ch.2 §2.1 — The legal lexicon as scientific instrument
**Date**: 2026-02-21
**Status**: complete

---

## Context

The dataset is the primary scientific instrument of the thesis. Every downstream
experiment (Lens I–V) operates on this corpus. Decisions made here directly
determine what the thesis can and cannot claim. The dataset must satisfy two
requirements simultaneously: (1) it must be theoretically motivated as a
representative sample of legal language; (2) it must be practically aligned
across two languages (English and Chinese) to enable cross-tradition comparison.

---

## Decision log

*(Filled in as each decision is made)*

### D1 — Legal domains
**Options considered**:
- Alt. A: 5 macro-domains (public/private/criminal/international/procedure) — too coarse
- Alt. B: HK DOJ division structure — ruled out: DOJ divisions are institutional, not thematic
- Alt. C: 6 balanced domains (standard CLS) — solid but misses procedure
- Alt. D: Deontic organization — theoretically original but weak external validation
  and ~30% of terms are polysemically ambiguous across categories
- Alt. E: Previous 9 domains pruned to 7 — `jurisprudence` and `environmental_tech`
  have no coverage in any authoritative cross-tradition source

**Decision**: 7 subject-matter domains (Option β):
`constitutional`, `civil`, `criminal`, `administrative`,
`international`, `labor_social`, `procedure`

Key moves vs. previous 9 domains:
- `rights` merged into `constitutional`: in both WEIRD and Sinic traditions,
  rights (situazioni giuridiche soggettive) derive from constitutional sources
  as their supreme foundation (Windscheid, Jellinek, Crisafulli). The separation
  was WEIRD-centric and unsupported in Sinic taxonomy.
- `governance` renamed `administrative`: aligns with EuroVoc 1226 and NPC cat.3 (行政法)
- `jurisprudence` dropped: no coverage in EuroVoc or NPC; meta-legal, not a
  subject-matter domain. Terms to be redistributed or moved to background.
- `environmental_tech` dropped: anachronistic for both traditions; no authoritative
  cross-tradition classification; 19 terms (weakest domain by count).
- `procedure` added: EuroVoc 1221 (Justice) + NPC cat.7 (诉讼法); universal,
  well-delimited, cross-tradition symmetric.

**Authoritative sources (dual justification per domain)**:
| Domain        | EuroVoc                  | NPC 部门法              |
|---------------|--------------------------|-------------------------|
| constitutional| 1206 + 1236              | cat.1 (宪法相关法)       |
| civil         | 1211                     | cat.2 (民法商法)         |
| criminal      | 1216                     | cat.6 (刑法)             |
| administrative| 1226                     | cat.3 (行政法)           |
| international | 1231                     | academic curricula      |
| labor_social  | employment domain        | cat.5 (社会法)           |
| procedure     | 1221 (Justice)           | cat.7 (诉讼法)           |

Sources:
- Publications Office of the EU. EuroVoc Thesaurus v4.23.
- National People's Congress PRC. National Database of Laws and Regulations
  (国家法律法规数据库), launched 2021. flk.npc.gov.cn.

**Thesis text implication**:
§2.1 can justify the seven-domain structure by reference to two independent
authoritative sources — one from each legal tradition under comparison. This
prevents the charge that the domain taxonomy is an arbitrary researcher choice
or a WEIRD-centric projection onto Sinic law. The merger of `rights` into
`constitutional` is itself a substantive finding: it reflects a shared deep
structure in both traditions that the geometry may or may not confirm.

---

### D2 — Term structure (JSON schema)
**Options considered**:
- Alt. A: Flat schema with a single `zh` field — insufficient: DOJ entries routinely
  supply multiple Chinese equivalents from different institutional sources; a flat
  schema erases this multi-variant structure, forcing premature disambiguation.
- Alt. B: Rich schema with `flags`, `expected_divergence`, and `hk_specific` boolean —
  bakes analytical hypotheses into the dataset itself; violates the principle that
  the dataset is a neutral instrument. Interpretive metadata belongs in external
  documentation, not in the corpus that experiments read.
- Alt. C: Schema per tier (separate schemas for core / background / control) —
  needlessly complex; fields that do not apply to a tier can be empty lists or null
  without sacrificing schema uniformity or parseability.
- Alt. D: Uniform schema with structured multi-variant and provenance fields (chosen) —
  records all DOJ-supplied information in a lossless way; permits post-hoc analytical
  overlays without contaminating the base data.

**Decision**: Single uniform JSON schema across all tiers:

```json
{
  "en": "mens rea",
  "zh_canonical": "犯罪意圖",
  "domain": "criminal",
  "tier": "core",
  "zh_variants": ["犯罪意圖", "犯罪心態"],
  "zh_sources": ["The Glossary of Legal Terms for Criminal Proceedings"],
  "doj_divisions": ["PD"],
  "source": "HK DOJ"
}
```

Field specifications:
- `en` (str): English headword — lowercase, as it appears in the DOJ glossary.
- `zh_canonical` (str): canonical Traditional Chinese translation, selected via
  the source priority hierarchy (named glossary publication > LRC > PD > ILD >
  LPD > LDD); artifact corrections applied per `domain_mapping_rules.md` Rule 2.
- `domain` (str | null): one of the 7 legal domains (`constitutional`, `civil`,
  `criminal`, `administrative`, `international`, `labor_social`, `procedure`);
  `null` for background and control terms, which are not assigned to a domain.
- `tier` (str): `"core"` | `"background"` | `"control"` — see tier definitions below.
- `zh_variants` (list[str]): all ZH translations supplied by the DOJ for this entry,
  listed in source priority order; empty list for CC-CEDICT-sourced terms (control
  terms have no DOJ variants).
- `zh_sources` (list[str]): names of DOJ source publications from which the ZH
  variants were drawn; empty list for non-DOJ terms.
- `doj_divisions` (list[str]): DOJ division codes (CD, ILD, LDD, LPD, LRC, PD)
  that contributed translations; empty list for non-DOJ terms.
- `source` (str): `"HK DOJ"` | `"CC-CEDICT"` — provenance of the EN↔ZH pairing.

**Tier definitions**:
- `core`: Terms assigned to one of the 7 legal domains. These are the primary
  experimental units for all five lenses. Domain assignment is the result of
  curator judgment anchored to EuroVoc and NPC classifications (see D1).
- `background`: Terms from the DOJ corpus that are included in the embedding pool
  but not assigned to a domain. Subtypes: high-polysemy bare nouns, procedural
  nouns, commercial terms, and role nouns. Their function is to densify the
  k-NN neighbourhood space for the NDA experiment (Lens V), preventing the
  neighbourhood of core terms from being artificially inflated by spatial
  proximity arising from corpus sparsity.
- `control`: Items drawn from the Swadesh 100 basic vocabulary list (see D3 below).
  No legal content by construction. Domain = null. These serve as a semantic
  baseline: if EN and ZH models align well on Swadesh items but poorly on legal
  terms, divergence in the legal domain is attributable to legal-semantic
  structure, not to a general cross-lingual embedding gap.

**Analytical metadata excluded from schema**: No `flags`, `expected_divergence`,
`hk_specific`, or any other interpretive annotation is stored in the dataset JSON.
Such metadata lives in external files (`hk_specific_terms.md`,
`domain_mapping_rules.md`) and is applied during analysis, not at corpus
construction time. This ensures that the experiments are not pre-loaded with
the researcher's expectations.

**Thesis text implication**:
§2.1 introduces the three-tier organisation and defines each tier's function
within the experimental design, documents the full field specification and
justifies the exclusion of interpretive fields: the dataset is a neutral
instrument; analysis is performed on top of it, not embedded within it.
Appendix A presents the schema formally and lists the full inventory of terms
per tier and domain.

---

### D3 — Control terms
**Options considered**:
- Alt. A: 50 random CC-CEDICT nouns (concrete objects) — arbitrary selection with
  no principled criterion; not reproducible; no scientific precedent; the choice of
  "concrete nouns" already encodes a hypothesis about what counts as non-legal.
- Alt. B: Domain-stratified contrast set (concrete vs. abstract non-legal terms) —
  excluded: stratifying the control set by abstractness bakes in the expectation
  that abstractness correlates with cross-lingual divergence, which is precisely
  the kind of assumption the experiments are designed to test. The control set
  must be neutral with respect to all hypotheses.
- Alt. C: Swadesh 100-item basic vocabulary list (Swadesh 1952, revised 1955) —
  optimal: well-validated across more than 200 languages, specifically designed
  to capture stable universal vocabulary, zero legal content by construction,
  and a standard instrument in computational and historical linguistics.
- Alt. D: Leipzig-Jakarta list (Tadmor et al. 2009; 100 basic words, resistance
  to borrowing criterion) — valid scientific instrument but less established in
  NLP literature; optimised for historical borrowing resistance rather than
  semantic universality; Swadesh is the more widely recognised standard and
  has greater prior-art in cross-lingual semantic studies.

**Decision**: 100-item Swadesh basic vocabulary list (Morris Swadesh, 1952,
revised 1955) as the control set. Tier = `"control"`. Domain = `null`.

**Rationale**:
The Swadesh list was designed specifically to identify the stable, universal core
of vocabulary that persists across languages through time: body parts, pronouns,
basic natural phenomena, cardinal spatial relations, fundamental actions, and
elementary properties. This makes it the ideal control instrument for the thesis:

1. **Principled and authority-backed**: The list is the product of decades of
   cross-linguistic fieldwork and has been validated against more than 200
   language families. Its use in this thesis is not an arbitrary researcher
   choice — it is the established standard tool in computational and historical
   linguistics for testing cross-linguistic semantic alignment at what Swadesh
   called the "language bedrock" level.

2. **Zero legal content by construction**: No item on the Swadesh 100 list belongs
   to any legal domain. The list was specifically designed to exclude
   culturally-variable, domain-specific vocabulary. This property cannot be
   achieved by ad hoc selection (Alt. A) or by researcher judgment alone.

3. **Interpretive asymmetry as a diagnostic**: If EN and ZH sentence-embedding
   models produce high Swadesh alignment but low legal-term alignment, the
   divergence observed in Lens I–V is attributable to legal-semantic structure,
   not to a general cross-lingual gap in the model pair. This logic — using
   basic vocabulary alignment as a baseline against which domain-specific
   divergence is measured — is structurally analogous to the use of Swadesh
   items in phylogenetic computational linguistics (Pagel et al. 2007).

4. **Symmetry with the experimental units**: The Swadesh items are single-concept
   terms (nouns, pronouns, verbs, adjectives), structurally comparable to the
   core legal terms. This prevents confounds arising from comparing single-word
   legal terms against multi-word expressions or sentences.

**Practical implementation**: ZH translations derived from the standard Swadesh
list ZH (Mandarin/Simplified) translations, converted to Traditional Chinese via
OpenCC for embedding compatibility with the DOJ corpus ZH. The `zh_variants` and
`zh_sources` fields are empty lists (no DOJ provenance). The `source` field is
`"CC-CEDICT"`, used as a shorthand for the publicly available CEDICT-derived
translations that underpin the standard Swadesh ZH mapping. The full EN→ZH
Swadesh mapping is documented in §A.1 of the thesis.

**Thesis text implication**:
§2.1 presents the Swadesh list as the control instrument and explains why
competing alternatives were rejected. §2.4 uses Swadesh alignment scores as the
reference level against which legal-domain divergence is assessed: a finding of
low legal alignment on top of high Swadesh alignment is the critical interpretive
condition for attributing divergence to legal-semantic structure. Appendix A
provides the full 100-item Swadesh EN→ZH mapping with Traditional Chinese
equivalents and OpenCC conversion notes.

---

### D4 — Sources and alignment strategy
**Options considered**:
- Alt. A: Multiple sources (DOJ + CC-CEDICT legal terms + academic bilingual
  glossaries) — complexity without demonstrable benefit: the DOJ corpus already
  covers 9,387 terms across all major legal domains; additional sources introduce
  heterogeneous translation philosophies and inconsistent terminological choices
  that would require a secondary alignment step of its own to resolve.
- Alt. B: DOJ + mainland Chinese legal dictionary (e.g., 法律詞典, 商務印書館) —
  theoretically attractive for testing cross-tradition equivalence directly at
  source level, but no machine-readable edition is available; manual entry at
  scale is infeasible and introduces transcription error risk without adding
  systematic coverage beyond what the DOJ already provides.
- Alt. C: DOJ as single primary source for legal terms + Swadesh for control terms
  (chosen) — 97.5% retention after filtering demonstrates that the DOJ corpus is
  already curated and comprehensive; a single-source strategy avoids the need for
  cross-source harmonisation decisions that would constitute an additional layer
  of undocumented researcher choice.

**Decision**: HK DOJ Bilingual Legal Glossary as the single primary source for
all core and background terms. Swadesh 100 (via CC-CEDICT-derived ZH) for
control terms. No additional sources.

**Rationale**:

**Alignment is intrinsic to the source**: The HK DOJ Bilingual Legal Glossary is
itself a bilingual alignment instrument — every English headword has one or more
Chinese equivalents curated by DOJ legal linguists with institutional authority
over the terminology of Hong Kong law. This is not a dataset that requires
external alignment: the EN↔ZH pairing is authoritative by construction. The
thesis inherits this alignment rather than constructing it, which is the correct
epistemic stance: the question is not whether terms can be paired across languages,
but what the geometry of those pairs reveals about cross-tradition semantic structure.

**zh_canonical selection (source priority hierarchy)**: When the DOJ supplies
multiple ZH variants for a single EN headword (from different institutional
publications), the zh_canonical is selected according to a fixed source priority
hierarchy: named glossary publication > LRC > PD > ILD > LPD > LDD. This
hierarchy is not arbitrary — it reflects the relative specificity and legal
authority of each DOJ division's publication. The hierarchy is applied
mechanically and documented in `domain_mapping_rules.md`, making the selection
reproducible and auditable.

**zh_canonical correction (Rule 2)**: In a small number of cases, the
first-occurring ZH variant selected by the hierarchy is an artifact of DOJ
internal conventions (e.g., a transliteration rather than a semantic translation,
or a Hong Kong-specific usage that is opaque in Mainland legal discourse). For
these cases, documented individually in `domain_mapping_rules.md` Rule 2, the
curator selects the shortest or most widely recognised ZH equivalent among the
DOJ's own supplied variants. This is a correction, not an interpretive choice:
the zh_canonical must be the form that the embedding model will encounter as a
semantically meaningful input. No translation is invented; the correction selects
among variants already present in the DOJ record.

**No machine translation applied**: zh_canonical is always curator-selected from
among the DOJ's own ZH variants. No machine translation model is used at any
stage of dataset construction. This preserves the bilingual authority of the DOJ
source and prevents the introduction of a second model's semantic geometry into
the dataset before any experiment is run.

**Coverage gap for control terms**: The Swadesh list items are not in the DOJ
glossary — they are not legal terms and were never subject to legal-linguistic
curation. Their ZH equivalents are drawn from the standard Swadesh ZH translation
(Mandarin/Simplified) with OpenCC conversion to Traditional Chinese. The `source`
field for these entries is `"CC-CEDICT"`, used as shorthand for the publicly
available CEDICT-derived translations that the standard Swadesh ZH mapping relies
on. This is a controlled, documented exception to the single-source strategy, and
it is the only exception.

**Single-source limitation**: The use of a single source (HK DOJ) for all legal
terms means that the corpus reflects Hong Kong's common-law-influenced bilingual
legal tradition. Terms, translation choices, and domain emphases may differ from
those that would emerge from a Mainland Chinese or Taiwanese source. This
limitation is explicitly acknowledged in §4.2 and is, in fact, a feature of
the study's framing: the comparison is between EN and ZH legal semantics as
mediated through a common institutional context, which controls for one dimension
of cross-tradition variation. §4.2 discusses the implications for
generalisability and identifies extending the corpus to Mainland or Taiwanese
sources as future work.

**Thesis text implication**:
§2.1 introduces the HK DOJ Bilingual Legal Glossary and establishes its
institutional authority as the basis for adopting the DOJ's own bilingual
pairings without external alignment. It justifies the single-source strategy
and rejects multi-source alternatives on grounds of harmonisation complexity
and absence of additional coverage benefit. §2.2 (Hong Kong as natural
laboratory) articulates the key epistemological point: the thesis treats the
DOJ's EN↔ZH pairings as authoritative; the experiments test the geometry of
those pairings, not the validity of the pairings themselves. §4.2 (irreducible
entanglement) addresses the HK-specific character of the source and its
implications for claims about cross-tradition Chinese legal semantics more
broadly.

---

### D5 — Bilingual contextual corpus source
**Date**: 2026-04-11
**Status**: decided

**Context**: Decisions D1-D4 defined the dataset as a list of 397 core legal terms
with curator-selected English and Chinese canonical forms drawn from the HK DOJ
Bilingual Legal Glossary. Until now, the experiments have embedded each term as
an isolated lexical string (the headword and, optionally, its glossary definition).
Decision D6 below changes this: terms will be represented as the mean of their
embeddings across multiple contexts of authentic legal use. This change requires
a bilingual corpus of legal texts from which such contexts can be retrieved. D5
fixes the source of that corpus.

**Options considered**:
- Option 1: **HKLII case law** (`hklii.hk`) — Court of Final Appeal through
  District Court, decisions from 1946 onward. Pro: large volume, covers
  adjudicatory practice. Contro: bilingual coverage uneven (strong at CFA level,
  degrading down the hierarchy); EN-ZH alignment is inferred, not guaranteed;
  HTML scraping introduces noise and parser fragility.
- Option 2: **HKSAR Judiciary Legal Reference System** (`legalref.judiciary.hk`)
  — upstream source feeding HKLII. Pro: authoritative, explicit language tagging.
  Contro: same coverage asymmetry as HKLII; no public API; documents still exist
  as court decisions, not as the normative texts from which the DOJ glossary was
  constructed.
- Option 3: **HK e-Legislation** (`elegislation.gov.hk`) — all ordinances of the
  Hong Kong legal system in both languages, with sentence-level alignment
  guaranteed by statute. Pro: the EN and ZH versions are *both* authentic
  normative texts under the Official Languages Ordinance (Cap. 5) and the
  Interpretation and General Clauses Ordinance (Cap. 1 s. 10B), which means
  neither is a translation of the other; the alignment is official, not inferred;
  structured XML available for current ordinances; the DOJ glossary was itself
  constructed from the terminology of this same corpus, ensuring that the terms
  for which we seek contexts are exactly the terms that appear in it. Contro:
  legislative language rather than adjudicatory language; some terms drawn from
  the DOJ glossary may appear rarely or not at all in current legislation.
- Option 4: **Hybrid e-Legislation + HKLII fallback** — primary retrieval from
  e-Legislation, fallback to HKLII case law for terms with fewer than K_min
  occurrences in legislation. Pro: maximises coverage while preserving the
  alignment guarantee for the bulk of terms. Contro: mixed provenance requires
  per-term documentation; two parsers instead of one.

**Decision**: **Option 3 (e-Legislation)** as single primary source.

**Rationale**:

*Official alignment is the epistemological keystone*. Under the Interpretation
and General Clauses Ordinance (Cap. 1, s. 10B(1)), "the English language text
and the Chinese language text of an Ordinance shall be equally authentic, and
the Ordinance shall be construed accordingly". This is not a translation
relationship: it is a doubly-authored normative instrument. Every section of
every ordinance in the e-Legislation corpus exists in two versions that are,
by law, of equal legal force. This is the strongest possible form of bilingual
alignment available for any legal corpus, because it is the only one where the
alignment is constitutive of the text rather than a downstream annotation.

For the thesis's argument in §2.2 (Hong Kong as natural laboratory), this is
the decisive property. The thesis claims that HK provides a setting in which
common law is expressed in Chinese while preserving its institutional
continuity with the English common law tradition. The e-Legislation corpus
*materially instantiates* this claim: the same institutional act produces two
texts, each of which is the law.

*Corpus-glossary coherence*. The HK DOJ Bilingual Legal Glossary, adopted as
the primary source in D4, was itself constructed by DOJ legal linguists working
on the bilingual ordinances of Hong Kong. The terms that populate the glossary
are the terms that appear in legislation. Using e-Legislation as the source of
usage contexts therefore closes a natural loop: we extract contexts of use for
each term from the very corpus in which that term was identified as requiring
bilingual standardisation. Case law, by contrast, uses legal vocabulary as
*applied* rather than as *defined*; using it as the primary source would
introduce a shift between the universe in which the terms were selected (DOJ
glossary, tied to legislation) and the universe in which their usage is
observed (adjudicatory practice).

*Engineering quality*. Current HK ordinances are published as structured XML
and HTML with persistent section IDs, making retrieval a problem of
deterministic indexing rather than robust HTML scraping. The alignment between
EN and ZH is carried at the section (and in many cases subsection) level,
allowing a clean many-to-many mapping between linguistic and structural units.
The PDF form of each ordinance preserves the same alignment for human
verification.

*Coverage limitation and fallback policy*. Some terms in the DOJ glossary —
particularly procedural doctrinal terms or Latin phrases (e.g., *de son tort*,
*locus standi*) — may appear rarely in current legislation because they are
living in the case law, not the statute book. For these terms, D5 adopts a
conditional fallback: if a term has fewer than K_min = 5 occurrences in the
e-Legislation corpus after the retrieval step of D6, it is marked as
low-coverage and D6 will specify whether to (a) proceed with fewer contexts,
(b) retrieve from HKLII case law as a fallback, or (c) exclude the term from
the contextualised pipeline and retain only the lexical-isolation embedding.
The choice between (a), (b), (c) is a D6 decision, not a D5 decision.

**Thesis text implication**:
→ §2.1 [The legal lexicon as scientific instrument]: the dataset is redescribed
as a list of terms *plus* a retrieval index into the bilingual corpus of HK
ordinances. The term is no longer a string; it is a coordinate in the legal
textual universe of Hong Kong law.

→ §2.2 [Hong Kong as natural laboratory]: the argument is strengthened from
"HK translates common law into Chinese" (which is easy to misread as a
translation claim) to "HK produces common law in Chinese and English
simultaneously, with statutory equal authenticity". The e-Legislation corpus
is the material instantiation of this claim.

→ §2.3: unaffected.

→ §4.2 [model-data-culture entanglement]: the coverage asymmetry (terms more
frequent in case law than in legislation) is acknowledged as a limitation of
the primary source, and the fallback policy (D6) is cited as the mitigation.

---

### D6 — Contextualised term representation: from lexical isolation to usage mean
**Date**: 2026-04-11
**Status**: decided

**Context**: Until now, the representation of a legal term has been the
embedding of its canonical form as a string (sometimes augmented with its
DOJ-supplied definition, depending on the model's input format). This
representation is an artifact of convenience: sentence-transformer APIs accept
a string and return a vector, and the natural interpretation of "the embedding
of a legal term" becomes "the vector produced by encoding the term".

This decision reverses the convenience with a correction demanded by the
thesis's own theoretical foundations. Chapter 1 §1.3 argues, with Firth and
Wittgenstein, that the meaning of a term is constituted by its use: "you shall
know a word by the company it keeps". The embedding of an isolated lemma is
not the company the word keeps; it is the word extracted from its company.
A representation faithful to the distributional hypothesis must approximate
the average of the term across its contexts of use.

This move also aligns the pipeline with the methodology of Sofroniew, Kauvar,
Saunders et al. (2026), who extract emotion concept vectors in Claude Sonnet
4.5 not from isolated emotion words but from short stories in which characters
experience the specified emotion, averaging activations across token positions
from position 50 onward. Their approach is the computational operationalisation
of the distributional hypothesis: a concept is the mean of its contextualised
activations.

**Options considered**:
- Option A1: **LLM-generated scenarios**. For each term, generate K short
  scenarios (EN and ZH) in which the term appears in a legally plausible
  context, using an LLM. Pro: scalable to any term regardless of corpus
  frequency; full control over coverage and register; procedure matches
  Sofroniew et al. exactly. Contro: introduces a circularity in which the
  model being probed for its representation of a concept is fed text generated
  by another model's representation of the same concept; the resulting vectors
  reflect a second-order pairing (embedding model + generator model) rather
  than a direct observation of the embedding model's encoding of legal
  practice; the "use" being captured is simulated use, not attested use.
- Option A2: **Corpus-retrieved contexts** (from the e-Legislation corpus
  fixed in D5). For each term, retrieve K sentences (or sentence-equivalent
  units, e.g., clauses within long legislative sections) from the bilingual
  ordinances in which the term appears, in each language. Pro: the contexts
  are attested, authoritative legal use; the distributional hypothesis is
  applied to real practice rather than to simulation; the pipeline becomes a
  direct observation of the corpus from which the glossary itself was derived;
  epistemologically cleaner for a methodology thesis. Contro: coverage is
  bounded by corpus frequency; rare doctrinal terms may have few or no
  occurrences; requires a retrieval index and parser for the e-Legislation
  XML/HTML.
- Option A3: **Hybrid retrieval + LLM fallback**. Primary retrieval via A2;
  for terms with fewer than K_min occurrences, fall back to A1. Pro:
  universal coverage. Contro: mixed provenance requires per-term flagging;
  the corpus becomes inhomogeneous; opens the door to the criticism that
  low-coverage terms (often the most theoretically interesting ones:
  polysemic false friends, Latinate doctrinal terms) are precisely the terms
  represented by generated rather than attested contexts, exactly where the
  stakes of authenticity are highest.

**Decision**: **Option A2 (corpus-retrieved contexts from e-Legislation)** as
the sole source of contextualisation. No LLM-generated fallback.

Low-coverage policy: if a term has fewer than K_min = 5 attested occurrences
in the e-Legislation corpus (summed across EN and ZH), the term is retained
in the dataset but flagged `low_coverage = true`, and the experiments report
both (i) results computed on the full dataset including low-coverage terms,
with their contextualised vectors built from whatever occurrences exist down
to a minimum of 1, and (ii) results restricted to the high-coverage subset
(≥ K_min occurrences per language). Discrepancies between (i) and (ii) are
reported as evidence of sensitivity to coverage, not suppressed.

**K (contexts per term) policy**: the target is K = 12 contexts per language
per term, matching the Sofroniew et al. per-topic count. Where fewer than 12
occurrences exist, all available occurrences are used. Where more than 12
exist, sampling is reproducible (fixed seed, uniform over occurrences;
alternative stratified-by-ordinance sampling is noted as a sensitivity
analysis in the trace but not as the primary procedure).

**Rationale**:

*Philosophical coherence with §1.3*. This is the decisive reason. §1.3 of the
thesis argues that meaning is use, that use is observable as distributional
regularity, and that embedding models capture distributional regularity as
geometric structure. If meaning is use, then the representation of a term
should be constructed from instances of use, not from the isolated lemma. The
isolated-lemma representation was an inherited convenience of the
sentence-transformer API; it was never the representation the thesis's
theoretical chapter was arguing for. Option A2 corrects the incoherence.

*Authenticity over simulation*. A methodology thesis cannot rest on simulated
data where attested data is available. The e-Legislation corpus is the
authoritative record of the legal community whose semantic organisation the
thesis seeks to measure. Generating synthetic contexts would mean measuring
the generator's idea of legal use rather than legal use itself. This is the
same objection that §2.2 levels against machine-translation alignment: the
alignment would become a training artifact of a second model rather than an
empirical datum. The same objection applies here.

*Convergence with corpus of term provenance*. D4 and D5 together fix the
universe of reference: HK DOJ glossary terms + HK bilingual ordinances. These
two objects are natively coupled because the glossary was constructed from
the ordinances. Option A2 preserves that coupling at the instance level:
every contextualised vector is built from attested occurrences in the exact
corpus that motivated the term's inclusion in the glossary.

*Rejection of A1*. The epistemological cost of A1 is not that the generated
text is low-quality (it could be very high-quality) but that the generated
text cannot, in principle, falsify any hypothesis that the generator model
itself would not already reject. This is the circularity: the generator's
conception of "criminal intent in context" is the conception that will be
fed to the probe; if the two models disagree on what criminal intent means,
the disagreement is masked by the generator's framing of the scenario.
Only attested use can reveal disagreement, because only attested use is
independent of any model's preconceptions.

*Rejection of A3*. Hybrid provenance would mean that the most theoretically
interesting terms — low-coverage doctrinal terms, precisely the ones most
likely to manifest tradition-specific semantics — would be represented
exclusively by generated contexts, while the bulk of the dataset (ordinary
legislative vocabulary) would be represented by attested contexts. The
experimental signal at the theoretical core of the thesis would then come
from the generated half, and the reader would be right to question whether
the observed divergences are features of the legal traditions or artifacts
of the generator. A2 with the explicit low-coverage policy is better because
it preserves the provenance invariant at the cost of accepting smaller N for
the rare-term subset, which can be addressed statistically (wider CIs, lower
power, honest reporting).

*Relation to Sofroniew et al. (2026)*. Sofroniew et al. use generated stories
because their object of study is a general language model's representation of
emotion concepts, and no attested corpus of "200 stories per emotion" exists.
For the legal domain, the attested corpus does exist (e-Legislation), and
choosing A2 over A1 is the right adaptation of their method to the different
material conditions of the two domains. The thesis will cite Sofroniew et al.
as the methodological precedent and explicitly note the adaptation: where they
had to simulate use, the legal domain allows observation of use.

*Averaging procedure*. For each term t and each model m, the contextualised
representation is computed as:

  v(t, m) = mean_{c in contexts(t)} encode(m, c)

where `contexts(t)` is the set of K (or fewer) retrieved sentence-level units
containing t, `encode(m, c)` is the embedding of context c by model m at the
layer selected by D-B (see `models/trace_model_selection.md`), and the mean
is the arithmetic mean of the unit-normalised vectors. An ablation over mean
vs. weighted mean (by sentence length or TF-IDF) is noted as a sensitivity
analysis but not as the primary procedure.

**Thesis text implication**:

→ §1.3: the gap between the theoretical commitment ("meaning is use") and
the experimental implementation (isolated-lemma embeddings) is closed. The
text of §1.3 can now claim, without hedge, that the experimental pipeline is
a direct operationalisation of the distributional hypothesis: each legal term
is represented by the average of its embedded contexts of attested use in
the bilingual corpus of its own legal community.

→ §2.1: the dataset description is extended. The primary object is no longer
the list of 397 terms but the pairing of each term with its set of attested
contexts from the e-Legislation corpus. The JSON schema for each term gains
a `contexts` field per language, containing the ordinance section IDs from
which contexts were drawn.

→ §2.3: unaffected at the model-selection level, but the embedding extraction
procedure for each model is now "mean of contextualised encodings" rather
than "encoding of the lemma".

→ §3.1–§3.3: the RDMs, neighbourhoods, and axis projections of Lenses I, IV,
and V are all recomputed on the new term representations. The experimental
chapter will report the change in results between lexical-isolation and
contextualised representations as part of the sensitivity analysis.

→ §4.1: the synthesis across lenses is reinforced because the underlying
representation is now uniform across all lenses and philosophically coherent
with the theoretical argument.

→ §4.4 (Horizons): the use of attested-use contexts opens a line of future
research on term-level temporal dynamics (how the geometry of a term shifts
across decades of the same corpus), which is not in scope for this thesis
but is a natural continuation.

---

### D5 + D6 — Status update after adversarial pre-check (2026-04-11)
**Status**: Alpha-lite / deferred

**Update note**: D5 (HK e-Legislation as primary corpus) and D6
(contextualised term representation via mean pooling over retrieved contexts)
were both proposed earlier on 2026-04-11 in response to the methodological
parallel with Sofroniew et al. (2026). Later the same day, an adversarial
review of the proposal raised structural objections, and three empirical
pre-checks were run on the existing pipeline to test whether the proposed
revisions were empirically necessary or only theoretically attractive.

The polysemy / aggregation pre-check (`pre_checks/precheck_2_polysemy.py`,
results in `pre_checks/results/precheck_2_polysemy.json` and consolidated
in `pre_checks/results/precheck_results.md`) showed that the headline Lens I
result Δρ ≈ 0.260 is **robust to aggregation choice** across the three
meaningful aggregators tested (mean, medoid, bare term): the maximum
pairwise Δρ difference is 0.044, below the ROBUST threshold of 0.05. The
first-PC aggregator was found pathological at N=5 cloud size and excluded
from the meaningful comparison.

A counter-intuitive finding: mean pooling over template variants produces a
**larger** Δρ (+0.2827) than the bare-term representation (+0.2391). The
adversarial critique (Arora 2018; Chronis & Erk 2020) that mean-pooling
destroys polysemic information does not bite here, because the template
variants represent the same concept in different surface forms rather than
distinct senses of an ambiguous word. Mean pooling acts as a denoiser, not
as a sense-collapser.

**Consequence for D5 and D6**:

- The contextualised extraction (D6) is **downgraded from primary to
  Alpha-lite**: it remains methodologically valuable as a means of aligning
  the experimental pipeline with the philosophical commitment of §1.3
  (meaning is use), but it is no longer required to defend the headline
  Lens I result. When implemented, it is reported as a Firthian sensitivity
  analysis alongside the existing bare-term primary result, not as a
  replacement for it.

- The HK e-Legislation corpus (D5) is **deferred**: the substantial
  infrastructure required to parse, index, and retrieve from the bilingual
  ordinances is now a Horizons-tier task to be undertaken if and when D6 is
  promoted from sensitivity analysis to primary representation, or for
  future work on temporal dynamics of legal vocabulary. It is not a
  prerequisite for the thesis as currently framed.

- The current pipeline (bare-term embeddings, precomputed cache in
  `data/processed/embeddings/`) is empirically validated against the
  adversarial critique and **stands as the primary representation** for
  the experimental chapters.

**Thesis text implication update**: §1.3, §2.1, §2.2 do not require revision
on the basis of D5+D6: the primary results remain those computed on the
bare-term representation. §2.4 gains a new sub-section on adversarial
robustness pre-checks documenting this revision process. §4.4 (Horizons)
gains a paragraph noting that the contextualised pipeline and the e-Legislation
corpus are first-class candidates for the immediate continuation of the
research programme.

The original D5 and D6 entries above remain unchanged as the historical
record of the proposal that the empirical pre-check downgraded.

**Reference**: `experiments/pre_checks/results/precheck_results.md`,
sections "Pre-check 2 — Polysemy / aggregation robustness" and
"Consolidated verdict".

---

### D7 — Firthian-strict pivot (2026-05-01)

**Status**: supersedes D5 + D6 (and the 2026-04-11 pre-check addendum) for
the experimental rerun. The D5/D6 entries above remain unchanged as the
historical record of the previous design.

**Trigger**: a 2026-05-01 audit of the not-yet-headline-rerun attested pool
(`embeddings_ctx_attested/`, 10 model slots) revealed that:

1. 16% of 0-ZH-attested core terms (21 of 131) carry DOJ-glossary metadata
   in `zh_canonical` (`（香港）`, `※比較`, `☛參看`, `〔〕`, terminal `的`)
   that prevents corpus matching: they are present in the e-Legislation
   corpus under their base form but the matching script searches for the
   annotated form.
2. `build_attested_pool.py` pads terms with N<8 attested contexts using
   synthetic templates and mean-aggregates. Terms with 0 attested contexts
   produce vectors that are mean-of-8-synthetic, indistinguishable in
   downstream RDMs from genuine attested vectors. The "attested" pool as
   built is partially synthetic for ~190 of 350 core terms.
3. The bare embeddings encode `zh_canonical` *with* the markers; the
   bare-vs-attested delta therefore conflates the Firthian effect with
   spurious marker noise.
4. The three `index.json` files under `embeddings/`,
   `embeddings_contextualized/`, `embeddings_ctx_attested/` carry three
   different snapshots of the `tier` flag (350 / 397 / 430).

**Decision**: refound the dataset on a single declarable criterion:
*bilingual legal terms attested ≥4 times in both EN and ZH HK Cap.
e-Legislation, on glossary-cleaned forms*. Per-domain re-curation restores
50/domain balance.

**Pipeline changes**:

- New fields `en_clean` and `zh_clean` added to `legal_terms.json`:
  conservative EN cleanup (split on `;`), aggressive ZH cleanup
  (parentheses, marker truncation, terminal `的`, glossary brackets).
  Original `en` and `zh_canonical` preserved as catalogued forms.
- All 9472 × 10 model slots re-encoded for the *bare* representation on
  `*_clean` forms (~1h compute).
- Strict-filter computation: K≥4 in both EN and ZH on cleaned forms,
  yielding ~4033 candidates pool-wide, of which ~165 are in the current
  core and ~3835 in background (post k-NN domain assignment).
- Per-domain LLM curation (D5-style) selects 50 from {current core
  K≥4} ∪ {top attested-background candidates}. Hard-gate exceptions for
  doctrinally central K<4 terms allowed but explicitly justified per term.
- New `build_attested_pool.py`: mean-aggregate over available attested
  contexts only, no synthetic padding. Fails loudly on N<4 (D4 hard gate
  guarantees N≥4 for all final-core terms).
- New `data/sync_indexes.py`: regenerates the three `index.json` files
  from `legal_terms.json` (master), invoked after each dataset edit.

**Rationale**: a single pre-registrable criterion replaces two overlapping
ones (doctrinal selection of D5 + implicit "if no attestation, fake it" of
the padding logic). Honours Ch.1 §1.3 Firthian commitment exactly.

**Thesis text implication**: → §2.1 documents the cleanup rules and the
strict criterion verbatim, with worked examples for each cleanup rule.
→ §2.2 (HK as laboratory) declares the bias toward HK-Cap-statutorily-
covered subject matter and motivates the §4.2 horizons paragraph on
HKCFA/case-law enrichment as the complementary direction. → §2.4 adds the
no-padding aggregation rule and the K≥4 ≤ N ≤ 8 attested-mean specification.

**Full design**: `experiments/trace_firthian_pivot.md` (D1-D6 with
options, decision, rationale, thesis text implication for each step).

**Cross-reference**: `experiments/trace_pivot_2lens.md` D11.

---

## Open questions

---

## References
