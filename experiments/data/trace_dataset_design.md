# Trace: Dataset design

**Thesis chapter(s)**: Ch.6 — The legal cloud: corpus design and the logic of legal sampling
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
§6.2.1 (Domain stratification) can justify the seven-domain structure by
reference to two independent authoritative sources — one from each legal
tradition under comparison. This prevents the charge that the domain taxonomy
is an arbitrary researcher choice or a WEIRD-centric projection onto Sinic law.
The merger of `rights` into `constitutional` is itself a substantive finding:
it reflects a shared deep structure in both traditions that the geometry may or
may not confirm.

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
- `control`: Items drawn from the Swadesh 100 basic vocabulary list (see D3).
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
§6.1 (corpus structure) introduces the three-tier organisation and defines each
tier's function within the experimental design. §6.2.2 (schema transparency)
documents the full field specification and justifies the exclusion of interpretive
fields: the dataset is a neutral instrument; analysis is performed on top of it,
not embedded within it. §A.1 (data appendix) presents the schema formally and
lists the full inventory of terms per tier and domain.

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
§6.1.2 (control terms rationale) presents the Swadesh list as the control
instrument and explains why competing alternatives were rejected. §7.3
(baseline interpretation) uses Swadesh alignment scores as the reference level
against which legal-domain divergence is assessed: a finding of low legal
alignment on top of high Swadesh alignment is the critical interpretive condition
for attributing divergence to legal-semantic structure. §A.1 (data appendix)
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
limitation is explicitly acknowledged in §6.3.1 and is, in fact, a feature of
the study's framing: the comparison is between EN and ZH legal semantics as
mediated through a common institutional context, which controls for one dimension
of cross-tradition variation. §6.3.1 discusses the implications for
generalisability and identifies extending the corpus to Mainland or Taiwanese
sources as future work.

**Thesis text implication**:
§6.1 (source description) introduces the HK DOJ Bilingual Legal Glossary and
establishes its institutional authority as the basis for adopting the DOJ's own
bilingual pairings without external alignment. §6.1.1 (the selection problem)
justifies the single-source strategy and rejects multi-source alternatives on
grounds of harmonisation complexity and absence of additional coverage benefit.
§6.1.3 (alignment intrinsic to source) articulates the key epistemological point:
the thesis treats the DOJ's EN↔ZH pairings as authoritative; the experiments
test the geometry of those pairings, not the validity of the pairings themselves.
§6.3.1 (limitations) addresses the HK-specific character of the source and its
implications for claims about cross-tradition Chinese legal semantics more broadly.

---

## Open questions

---

## References
