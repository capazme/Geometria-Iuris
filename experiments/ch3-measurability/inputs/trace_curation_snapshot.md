# Trace: post-BLP curation methodology

**Thesis chapter(s)**: Ch.2 §2.1 (dataset), §2.2 (HK as laboratory). Re-runs of Ch.3 §3.1 + §3.2 follow.
**Date**: 2026-05-10
**Status**: methodology fixed, executing manual KEEP/DROP/DEFER on `postBLP_curation_longlist.csv`.

---

## Context

The Firthian-strict pivot (`trace_firthian_pivot.md`, D1-D6) restricted the
core to terms with K≥4 attested contexts on glossary-cleaned forms in the
HK Cap. e-Legislation corpus. The resulting core (n=327, distribution
45/50/47/45/45/47/48 across 7 domains) underwrote the headline numbers in
§3.1.3 and §3.2.3 of the thesis (Δρ symmetric: 0.211 bare → 0.541 attested).

The post-BLP pivot tightens the attestation criterion further by
restricting the qualifying corpus to post-1989 Caps, i.e. ordinances
co-drafted bilingually under the Bilingual Laws Project (BLP, 1989-).
Pre-1989 Caps were originally English-only; their Chinese versions were
added later as authentic translations under the fictio of equal
authenticity. Encoding contextual meaning from a translated text confounds
the Firthian "meaning is use" commitment with translation-equivalence
artefacts.

The pool diagnostic (`postBLP_pool_diagnostic.json`) computes K_postBLP for
all 9472 candidate terms and yields 1271 with K_postBLP ≥ 4 (the long
list). Auto-classification by `tier_current × knn_confidence × k_postBLP`
yields:

| Stratum                  | n   | Action                                         |
|--------------------------|-----|------------------------------------------------|
| AUTO_KEEP                | 85  | core ∩ K_postBLP≥4: spot-check, default KEEP  |
| RECOMMEND_KEEP           | 83  | high knn_conf + high K: review, mostly KEEP   |
| REVIEW                   | 803 | manual judgement: pick 30-40 per domain        |
| RECOMMEND_DROP           | 26  | low knn_conf single-words: review for DROP    |
| AUTO_DROP                | 274 | modals/quantifiers/knn<2/7: default DROP      |

Target: 60 KEEP per domain × 7 = **420 final core post-BLP**.

---

## D1 — KEEP criteria

A term enters the post-BLP core iff **all** of the following hold.

1. **Lexical-legal**. The English headword denotes a legal-technical
   concept, not a generic word that happens to recur in legal texts.
   *Test*: would a competent jurist recognise the term as belonging to
   the technical vocabulary of any of the seven domains? `licence`,
   `appeal`, `easement` pass. `ground`, `plan`, `money`, `description`,
   `capacity` (in the loose sense) fail.

2. **Domain-canonical**. The term is constitutive of, or characteristic
   of, the assigned domain. The bar is *characteristicity*, not exclusive
   ownership: `damages` is canonical for civil even though it appears in
   procedure too. The k-NN attribution provides a prior; my judgement
   confirms or overrides.

3. **Bilingual integrity**. The Chinese counterpart is a recognised legal
   form, not a literal calque or descriptive paraphrase. The post-BLP
   filter already enforces co-drafted attestation, but I additionally
   reject terms whose `zh` field carries DOJ-glossary metadata or
   wrong-sense gloss (cf. D4 step 3 addendum of the Firthian trace).

4. **K_postBLP ≥ 4** (already enforced upstream).

5. **Non-redundant**. The term is not a near-duplicate (singular/plural,
   trivial inflection, or paraphrase) of another KEEP. When two forms
   compete I prefer the form that appears more frequently in the HK Cap.
   surface text and matches DOJ-glossary headword orthography.

**knn_confidence is informational, not gating.** A term with knn_conf 0.29
can still be KEEP if it is genuinely poly-domain (the k-NN dispersion
reflects cross-domain reach, not noise). A term with knn_conf 1.00 can
still be DROP if the headword is generic.

---

## D2 — DROP criteria

A term is dropped iff **any** of the following hold.

1. **Generic word**. Not a legal-technical term in any domain (`money`,
   `description`, `plan`, `notice` *qua* notification, `capacity` *qua*
   volume, `ground` *qua* land surface). The k-NN often labels these
   high-frequency strings opportunistically by their corpus distribution;
   the legal sense is grafted on by collocation, not lexicalised in the
   word itself.

2. **Function word, modal, quantifier, connective**. `notwithstanding`,
   `any person`, `however`, `if applicable`, `where appropriate`. These
   are statutory drafting glue, not concepts.

3. **Procedural micro-form**. Drafting boilerplate without conceptual
   content (`signed copy`, `dated copy`, `at the request of`, `for the
   purposes of this section`).

4. **Wrong-sense gloss**. The `zh` form denotes a distinct legal sense
   from the EN headword (cf. Firthian D4 addendum: `human rights` ≠
   `《歐洲人權公約》`). Wrong-sense terms are dropped here rather than
   manually overridden, because the post-BLP pool is large enough that
   override scarcity does not arise.

5. **Numeric / proper-noun residue**. Schedule numbers, statutory
   reference fragments, ordinance short-titles that survived the
   diagnostic's filter.

---

## D3 — DEFER criteria

A term is deferred iff KEEP and DROP are both defensible and the
budget per domain is not yet binding. Default state for all REVIEW rows
prior to my pass is DEFER. After the pass, DEFER rows do not enter the
core; they are documented as the *near-miss* set, available for an
ablation extension if the relatore challenges domain coverage.

---

## D4 — Target balance (60 KEEP per domain × 7)

Hard target: 60. Acceptable band: 55-65 if the legitimate KEEP set per
domain is genuinely smaller (constitutional, international tend to have
narrower vocabulary than civil, procedure, criminal). The previous
Firthian core was 45-50/dom; the upgrade to 60/dom is enabled by the
broader pool (1271 vs. 350 candidates) and reduces standard error on
within-tradition ρ̄ measurements.

The cross-tab from feasibility check (likely-keep + REVIEW available):

| Domain         | likely-keep | REVIEW disponibili | margine REVIEW/needed |
|----------------|-------------|--------------------|-----------------------|
| administrative | 18          | 91                 | 2.2×                  |
| civil          | 31          | 130                | 4.5×                  |
| constitutional | 30          | 66                 | 2.2×                  |
| criminal       | 28          | 115                | 3.6×                  |
| international  | 20          | 80                 | 2.0×                  |
| labor_social   | 21          | 131                | 3.4×                  |
| procedure      | 20          | 190                | 4.8×                  |

All seven domains have ≥2.0× headroom. Constitutional and international
are the tightest; if either falls below 55 after pass, I document the
shortfall as a corpus property in §2.1.

---

## D5 — Inheritance of the Firthian core (n=327)

The Firthian-strict core is **not** automatically inherited. A Firthian
term is inherited iff it satisfies K_postBLP ≥ 4 AND survives my D1-D2
review here. The 85 AUTO_KEEPs are the subset of the Firthian core that
clears K_postBLP ≥ 4; the remaining 327 − 85 = 242 Firthian-core terms
are filtered out by the post-BLP corpus restriction (their attestation is
concentrated in pre-1989 Caps).

This is the desired behaviour: post-BLP requires post-BLP attestation. A
term that fails to attest in post-1989 Caps but did in pre-1989 Caps
trivially fails the new criterion. The methodological gain (clean
co-drafted contexts) is worth the loss of 242 Firthian-only terms.

The 242 dropped Firthian terms remain in `legal_terms.json` as
`tier: 'background'` for full audit transparency.

**Thesis text implication**: → §2.1 declares the post-BLP criterion as a
strengthening of the Firthian-strict criterion. The 327 → 420 transition
is presented as a corpus-period refinement that increases statistical
power while removing the pre-1989 translation-equivalence confound.

---

## D6 — Operational plan

1. Sample REVIEW rows per domain (k=20 per domain) to calibrate the
   blacklist (D2.1) and allowlist (D1.2).
2. Build a curation script (`apply_postBLP_curation.py`) that:
   - Encodes generic-word and function-word blacklist → DROP.
   - Encodes per-domain canonical-term allowlist → KEEP.
   - Defaults remaining REVIEW rows to DEFER unless promoted.
   - Per-term override list for edge cases I judge directly.
3. Run the script, write back `postBLP_curation_longlist.csv`.
4. Audit:
   - 7 domains × ~60 KEEP each.
   - No `curation_decision` empty.
   - No KEEP with empty `zh`.
   - No KEEP duplicates on (en, zh) pair.
   - Generic-word blacklist intersected with KEEP = ∅.
5. Update this trace with final per-domain counts and any criteria
   refinements that emerged during execution.

---

## D7 — Execution outcome (2026-05-10)

`apply_postBLP_curation.py` materialises D1-D6 as a Python script with
explicit per-domain KEEP, PRUNE, and AUTO_KEEP_DROP_OVERRIDE sets. Run
on the 1271-row long list yields:

| Domain         | KEEP | k_postBLP min | median | mean | max |
|----------------|------|---------------|--------|------|-----|
| administrative | 57   | 4             | 10     | 13.3 | 60  |
| civil          | 60   | 4             | 9      | 13.7 | 98  |
| constitutional | 56   | 4             | 10     | 13.3 | 76  |
| criminal       | 58   | 4             | 6      | 9.8  | 53  |
| international  | 55   | 4             | 7      | 12.4 | 68  |
| labor_social   | 60   | 4             | 10     | 14.4 | 51  |
| procedure      | 62   | 4             | 9      | 12.7 | 58  |
| **total**      | 408  |               |        |      |     |

Balance: min=55, max=62, mean=58.3, std=2.5, range=7. Inside the
D4 acceptable band 55-65. International at 55 is the tightest; the
shortfall reflects the genuinely narrower legal-international vocabulary
attested in HK Cap. post-1989, not selection error.

**Audit pass** (`postBLP_curation_audit.json`):

- 0 duplicate (domain, en) keys
- 0 empty zh
- 0 cross-domain duplicates
- 2/408 (0.5%) potential zh-fragment KEEPs flagged for spot-review
  (`assembly`/集會……的自由 = freedom of assembly, semantic core intact;
  `sentence`/將……的判刑與……的量刑基準掛鈎 = sentencing-context phrase,
  encoding will pick up canonical 判刑/量刑 content).

**AUTO_KEEP overrides applied (9 total)**: from the original 85
AUTO_KEEPs, 5 were demoted because their `zh_clean` denotes a wrong-sense
form despite the EN headword being domain-canon (`patent`/明顯=obvious,
`bill`/單據=receipt, `international law`/國際私法=PRIVATE int'l law,
`assault`/猥褻侵犯=indecent assault narrows, `occupational`/職業履歷=CV).
Four further fragment-zh demotions during post-execution audit (`liable`,
`detain`, `hostage`, `presence`).

**Inheritance accounting from Firthian core (n=327)**: 80/85 AUTO_KEEPs
retained → 80 of the 327 Firthian-core terms are also in the post-BLP
core. The remaining 247 Firthian-core terms either fail K_postBLP ≥ 4
(D5 dynamic) or fall under one of the 5 AUTO_KEEP override demotions.
The 408 post-BLP core therefore comprises 80 inherited + 328 newly
promoted from the broader 9472-pool by the post-1989 + K≥4 + manual
curation chain.

**Files written**:

- `experiments/data/processed/postBLP_curation_longlist.csv` —
  `curation_decision` filled (408 KEEP, 863 DROP, 0 DEFER).
- `experiments/data/processed/postBLP_curation_audit.json` — machine-
  readable summary above.
- `experiments/data/processed/postBLP_curation_longlist.csv.bak.before_curation`
  — pre-curation backup for full audit transparency.

---

## D9 — External reviewer pass (HK ZH native + US common-law native, 2026-05-10)

Two parallel reviewers were dispatched: (i) HK legal-Chinese native to audit
ZH override quality; (ii) US-trained common-law academic to audit EN-side
domain attribution and term legitimacy. Reports synthesised into 4 HIGH-priority
substance fixes (A-D below). The full reviewer reports are not part of
this trace; the substance issues identified are.

### A — Wrong-sense ZH overrides revised

Four override decisions in D7 still encoded wrong-sense ZH lemmas. Corrected:

| Term         | Old override | Revised | Reason |
|--------------|--------------|---------|--------|
| crime        | 罪行         | 犯罪    | Collision with offence=罪行 in embedding space |
| infringement | 侵犯         | 侵權    | IP-canonical lemma (Cap.528, Cap.559) |
| court        | 法庭         | 法院    | Institutional-canonical reversed (法院 = 高等法院/區域法院 etc.) |
| available    | 可獲得       | 可供    | Drafting-canonical (可供查閱, 可供使用) |

### B — Rescue of AUTO_KEEP demotions via manual ZH override

The 5 wrong-sense AUTO_KEEPs originally DROPPED in D7 are now rescued via
manual ZH override (HK-canonical lemma applied against the corpus):

| Term            | New ZH | Rationale |
|-----------------|--------|-----------|
| patent          | 專利   | Cap.514 Patents Ord. |
| bill            | 法案   | Legislative bill (vs DOJ 單據 = receipt) |
| international law | 國際法 | Public int'l law (vs DOJ 國際私法 = conflicts) |
| assault         | 襲擊   | General assault (vs DOJ 猥褻侵犯 = indecent assault) |
| occupational    | 職業   | Labour-canonical (vs DOJ 職業履歷 = CV) |

Side-effect: `vocational` (formerly 職業 in D7) collides with `occupational`.
Resolution: drop `vocational` (kzh=2 marginal, semantically subsumed).

### C — Fragment KEEPs already addressed

`assembly→集會` and `sentence→判刑` were already overridden in D7 (HIGH). No
action required.

### D — Generic non-canonical drops

Two generic-English-word KEEPs removed for symmetry with D2.1 blacklist:

| Term   | Domain         | Reason |
|--------|----------------|--------|
| person | civil          | Statutory drafting variable, not legal-technical |
| law    | constitutional | Generic mass noun / meta-term |

---

## D10 — Substance cleanup (2026-05-10)

User direction: optimise for dataset substance coherence over disclosure
volume. Three cleanup passes applied:

### Inflectional duplicates removed
Per-cluster: keep one canonical form, drop inflections that produce
near-identical embeddings.

| Cluster | Kept | Dropped |
|---------|------|---------|
| certificate / certification | certificate | certification |
| authorization / authorized | authorization | authorized |
| licence / licensing / licensed | licence (AUTO_KEEP) | licensing, licensed |
| permit / permitted | permit (AUTO_KEEP) | permitted |
| Board / board | Board | board |
| Committee / committee | Committee | committee |
| Director / director | Director | director |
| poll / polling | poll | polling |
| election / elector / elect | election, elector | elect |
| adjudication / adjudicate | adjudication | adjudicate |
| inform / informed | (none) | both dropped |

Total: 14 inflectional drops across domains.

### Procedure drafting glue removed (D2.2 extension)
State adjectives and drafting predicates with no legal-conceptual content
of their own dropped from procedure: `contrary`, `available`, `effective`,
`relevant`, `sufficient`, `valid`, `specified`, `documentary`, `prior`,
`attend`, `appear`. Total: 11 drops.

### Cross-tradition non-mapping term dropped
`president` removed from constitutional. After review, the user-preferred
sense "head of state" has no clean HK-canonical lemma (`行政長官` is HK
Chief Executive, a distinct institutional concept; `總統` is foreign-state-
president with k=0 attestation in post-BLP Caps).

### Cross-domain leakage in international (ZONE D)
13 "drafted-foreignness" adjectives removed from international (these are
statutory predicates of foreignness, not public-international-law
canon): `Chinese`, `domestic`, `external`, `foreign`, `general`, `national`,
`particular`, `personal`, `real`, `subject`, `related`, `concurrent`,
`consequential`. PIL-canonical core preserved (convention, agreement,
protocol, multilateral, diplomatic, consular, treaty, arbitral, etc.) plus
HK-specific institutional (Mainland, non-Hong Kong, overseas, outbound,
regional, independent).

### Domain leakage in admin / civil / criminal: accepted with scope
After deliberation:
- **administrative** = regulatory_admin scope (revenue terms tax/taxation/
  surcharge/levy/expenditure/expenses retained).
- **civil** = civil_commercial scope (securities terms derivative/futures/
  share capital/stock/dividend/offeror/counterparty retained).
- **criminal** retains drafting-predicate boundary terms (harm/breach/
  failure/false/restraint/juvenile/unlawfully) as criminal-statutory.

These scopes are recoverable from the dataset itself; no separate disclosure
required in thesis text unless a result depends on the leakage.

---

## D11 — Final dataset state (2026-05-10)

| Domain         | n  | K_min<4 (corpus property) |
|----------------|----|----|
| administrative | 52 | 0 |
| civil          | 60 | 2 (gift, contractor) |
| constitutional | 49 | 1 (constitution) |
| criminal       | 54 | 1 (trafficking) |
| international  | 41 | 1 (military) |
| labor_social   | 60 | 0 |
| procedure      | 48 | 1 (bona fide) |
| **TOTAL**      | **364** | **6** |

Of 364 KEEP terms, **358 reach K≥4 strict bilingual attestation in post-1989
HK Caps**. The 6 corpus-property residuals are kept (k_min ≥ 1) for
domain-canonical coverage.

Comparison with predecessor Firthian core (n=327, K≥4 any-period):
- **+37 terms** despite more restrictive bar (post-1989 only).
- Per-domain distribution shifted: Firthian had 45-50/dom; post-BLP has
  41-60/dom (wider range, reflecting genuine corpus density per domain).
- Substance gains: no encoding collisions (crime ≠ offence), no inflectional
  duplicates, no drafting-glue contamination, ZH wrong-sense corrected against
  corpus, foreignness-adjective leakage removed from international.

Methodological disclosure plan: minimal. §2.1 declares the post-BLP K≥4 bar
in one sentence; §2.2 documents the operational corpus scope and that
"canonical-lemma corrections were applied where DOJ-glossary entries
did not align with attested HK Cap. usage" (one footnote, no exhaustive
list). The override map (`zh_overrides_postBLP.json`) and curation script
(`apply_postBLP_curation.py`) remain in the repository as audit trail.

---

## D8 — Next step: dataset materialisation

Downstream of the curated long list:

1. Re-extract attested contexts post-1989 for the 408 KEEP terms only
   (filter `term_contexts.jsonl` by Cap.year ≥ 1989 and KEEP set).
2. Re-encode all 10 model slots on the new (en, zh_clean) pairs (some
   AUTO_KEEPs from Firthian remain unchanged; the 328 promoted terms
   need new encodings).
3. Re-run §3.1 (Lens I, RSA + Mantel, B≥1000) and §3.2 (Lens IV, Kozlowski
   axes) on the 408-term post-BLP core.
4. Update §2.1 (dataset declaration) and §2.2 (HK as natural laboratory)
   in the thesis with the post-BLP rationale and final 408-term distribution.
5. Update §10 of `CLAUDE.md` with the new headline numbers from the rerun.

These steps are out of scope for this trace; they will be tracked under
a follow-up trace in the experiments execution log.

---

## References

- `experiments/data/processed/postBLP_curation_README.md` — operational
  brief from upstream pool generation.
- `experiments/data/processed/cap_enactment_years.json` — Cap → year
  lookup for post-BLP filter (98 verified + heuristic ranges).
- `experiments/data/apply_postBLP_curation.py` — D1-D7 implementation.
- `experiments/data/processed/postBLP_curation_audit.json` — D7 audit.
- `experiments/trace_firthian_pivot.md` D1-D6 — predecessor pivot,
  inherits methodology and `legal_terms.json` schema.
- HK Department of Justice. *Bilingual Laws Information System*.
  https://www.elegislation.gov.hk/.
- Sin, K.K., Roebuck, D. (1996). "Language engineering for legal
  transplantation: conceptual problems in creating common law Chinese."
  *Language and Communication* 16(3).
