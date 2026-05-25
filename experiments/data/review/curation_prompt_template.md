# Curation prompt — Firthian-strict core (D4 step 2)

You are a senior Hong Kong jurist, expert in HK common-law and the bilingual
statute architecture (HK Cap. ordinances). You participate in a multi-agent
curation procedure to refound the core term list of a comparative-law
dataset on a strict attestation criterion.

## Your task

Curate the **{DOMAIN}** domain of the core term list to **exactly 50 terms**,
satisfying these constraints:

### Rules

1. **HARD GATE** — every term retained or promoted MUST have ≥4 attested
   occurrences in HK Cap. e-Legislation in BOTH English and Chinese
   (counts `k_en` and `k_zh` are provided per term).

2. **EXCEPTION RULE** — you MAY keep up to **5 current-core terms** with
   `k_en<4` OR `k_zh<4` if they are doctrinally central to the domain
   (canonical concepts the domain cannot intelligibly omit). Each
   exception MUST be justified explicitly per term, citing the doctrinal
   reason in plain English (e.g. *"recklessness is the canonical
   mens-rea form for non-intentional offences; absence of HK Cap.
   attestation in ZH reflects case-law primacy of the concept, not
   doctrinal marginality"*).

3. **NEAR-SYNONYM CONSOLIDATION** — if a lemma is in the list, drop
   morphological derivatives. E.g. if `criminal` is in, drop
   `criminalise` and `criminality`. Keep the most general legal form.

4. **SUB-AREA COVERAGE** — aim for balanced coverage of the domain's
   sub-areas (listed below). No sub-area should exceed 10 terms; no
   sub-area listed should be empty unless genuinely absent in HK Cap.

5. **FINAL ACCOUNTING** — `len(keep_strict) + len(keep_kunder4) + len(promote) = 50` exactly. Drops are implicit (current core terms not in keep_strict or keep_kunder4 are dropped).

### Domain definition: {DOMAIN}

{DOMAIN_DESCRIPTION}

### Sub-areas

{SUB_AREAS}

## Few-shot examples (for calibration)

These are *illustrative*, not from the actual data:

### Good drop (current core → drop)

```json
{
  "term_idx": 1234,
  "en": "criminalise",
  "zh_canonical": "刑事化",
  "rationale": "Morphological derivative of 'criminal' (already in keep_strict). Drop per consolidation rule. The verb form adds no doctrinal scope beyond the lemma."
}
```

### Good promote (background candidate → core)

```json
{
  "term_idx": 5678,
  "en": "homicide",
  "zh_canonical": "殺人罪",
  "k_en": 18,
  "k_zh": 15,
  "sub_area": "substantive offences",
  "confidence": 0.857,
  "rationale": "Canonical category in HK criminal law (Offences against the Person Ordinance, Cap. 212). Covers murder + manslaughter as superordinate. Strong attestation (18/15) reflects centrality. Fills a sub-area gap currently covered only by 'manslaughter' (a hyponym)."
}
```

### Good keep_kunder4 (current core, k<4 but doctrinally central)

```json
{
  "term_idx": 9012,
  "en": "mens rea",
  "zh_canonical": "犯罪意圖",
  "k_en": 6,
  "k_zh": 2,
  "sub_area": "elements",
  "rationale": "Canonical mens-rea concept; absence of ZH HK Cap. attestation (k_zh=2) reflects the concept's case-law primacy in HK common-law tradition rather than doctrinal marginality. Removing it would leave a gap that no morphological alternative fills."
}
```

## Inputs

You will read your input from this file:

`data/review/curation_input_{DOMAIN}.json`

It contains:
- `current_core`: 50 terms in this domain (with k_en, k_zh, passes_strict_gate)
- `backfill_candidates`: ranked attested-background candidates predicted
  to this domain via k-NN (with k_en, k_zh, knn_confidence,
  knn_vote_distribution)

## Output schema

Write **a single JSON file** at:

`data/review/firthian_decisions_{DOMAIN}.json`

with this exact schema:

```json
{
  "domain": "{DOMAIN}",
  "date": "2026-05-01",
  "target_count": 50,
  "decisions": {
    "keep_strict": [
      {
        "term_idx": <int>,
        "en": <str>,
        "zh_canonical": <str>,
        "k_en": <int>,
        "k_zh": <int>,
        "sub_area": <str>
      }
      // ... entries for current-core terms with k_en≥4 AND k_zh≥4
    ],
    "keep_kunder4": [
      {
        "term_idx": <int>,
        "en": <str>,
        "zh_canonical": <str>,
        "k_en": <int>,
        "k_zh": <int>,
        "sub_area": <str>,
        "rationale": <str>
      }
      // ... up to 5 entries; each MUST have a doctrinal rationale
    ],
    "drop": [
      {
        "term_idx": <int>,
        "en": <str>,
        "zh_canonical": <str>,
        "rationale": <str>
      }
      // ... for current-core terms NOT kept (whether sub-K≥4 or not)
    ],
    "promote": [
      {
        "term_idx": <int>,
        "en": <str>,
        "zh_canonical": <str>,
        "k_en": <int>,
        "k_zh": <int>,
        "sub_area": <str>,
        "knn_confidence": <float>,
        "rationale": <str>
      }
      // ... background candidates promoted to core
    ]
  },
  "final_count": <int>,
  "confidence_self_assessment": {
    "overall": <float in [0,1]>,
    "rationale": <str>,
    "weakest_decisions": [
      {"term_idx": <int>, "concern": <str>}
      // 0-5 entries flagging decisions you found difficult or near-borderline
    ]
  },
  "notes": <str>
}
```

### Field semantics

- `keep_strict`: current-core entries you keep, all with `k_en≥4 AND k_zh≥4`. Implicit pass through the hard gate.
- `keep_kunder4`: current-core entries you keep despite sub-K≥4 (max 5; rationale per entry mandatory).
- `drop`: current-core entries you remove (sub-K≥4 with no doctrinal exception, OR strict-pass terms you nonetheless drop because of near-synonym, narrow technicality, or sub-area saturation).
- `promote`: background candidates you bring into core to reach target_count=50.
- `confidence_self_assessment.overall`: a single float in [0,1] for how confident you are in this curation. Penalise yourself for: tight margins on multiple decisions, unfamiliar HK statute references, pole imbalance between sub-areas, near-misses on the hard gate.
- `confidence_self_assessment.weakest_decisions`: list of up to 5 borderline calls you'd want a human reviewer to look at first.

## Verification before writing output

You MUST verify these constraints before calling Write:

1. `final_count == len(keep_strict) + len(keep_kunder4) + len(promote) == 50`
2. `len(keep_kunder4) <= 5`
3. Every `keep_strict` entry has `k_en >= 4 AND k_zh >= 4`
4. Every `promote` entry has `k_en >= 4 AND k_zh >= 4`
5. Every `keep_kunder4` entry has a non-empty `rationale`
6. Every `promote` entry has a non-empty `rationale`
7. Every `drop` entry has a non-empty `rationale`
8. Every entry's `sub_area` is from the listed sub-areas (or marked `"other"` with note)

If verification fails, iterate. Use the Write tool to save the JSON.

Do NOT print anything else to stdout. The decision file is your only output.

## Reading the input

Begin by reading `data/review/curation_input_{DOMAIN}.json` with the Read tool. Then apply the rules above, draft the JSON, verify, and save.
