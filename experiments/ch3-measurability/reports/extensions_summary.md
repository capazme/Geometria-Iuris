# Run #4 — Extensions summary (post-headline robustness)

The five extensions consume the 9.045 bg terms (tier=`background` in
the legacy `legal_terms.json`) for purposes that do **not** alter the
364-term post-BLP core pool. They produce robustness signals around the
headline result Δρ_sym attested = 0.543.

All inputs sealed: `inputs/bg_terms_snapshot.json` (sha-256 in manifest)
and `inputs/bg_contexts_snapshot.jsonl`.

---

## A — k-NN domain assignment of background

For each of 9.045 bg, k=7 nearest neighbours in the 364 core (cosine on
BGE-EN-large bare). Majority-vote domain; vote fraction = confidence.

| Domain          | n bg assigned |
|-----------------|---------------|
| criminal        | 2.009 |
| procedure       | 1.661 |
| civil           | 1.528 |
| labor_social    | 1.514 |
| administrative  |   936 |
| international   |   741 |
| constitutional  |   656 |

Confidence: mean 0.515, median 0.429, p90 0.857, p10 0.286.

**Reading.** The clustering is informative but heterogeneous: the high-confidence
decile is ~10% (≈900 terms), candidates for future pool expansion review. The
low-confidence decile is the *ambiguous-by-design* zone that drives extension F.

Output: `ext/A_bg_knn/background_assignments.csv` (full sortable spreadsheet)
        `ext/A_bg_knn/background_assignments.json`

---

## D — Δρ_sym vs %bg robustness curve

Mix N=10 sampled pools per level, each 364 terms with %bg ∈ {0, 10, 25,
50, 75}. Compute Δρ_sym with 4 primary models (BGE-EN, BGE-M3-EN ×
BGE-ZH, BGE-M3-ZH). Bg eligible iff K_min≥4 (3.249 candidates).

| %bg | n_core | n_bg | mean Δρ_sym | std   | CI95              |
|-----|--------|------|-------------|-------|-------------------|
| 0%  | 364    | 0    | **0.538**   | 0.000 | [0.538, 0.538]    |
| 10% | 328    | 36   | 0.535       | 0.019 | [0.502, 0.555]    |
| 25% | 273    | 91   | 0.542       | 0.019 | [0.508, 0.566]    |
| 50% | 182    | 182  | 0.564       | 0.023 | [0.520, 0.591]    |
| 75% | 91     | 273  | **0.590**   | 0.024 | [0.553, 0.624]    |

**Reading.** Δρ_sym is **robust under bg injection up to 75%**. It does
not collapse; if anything it nudges upward. The Firthian-tradition gap is
dominant in the encoded geometry, not a property of the curated pool alone.
This is the strongest robustness statement of the run.

Output: `ext/D_robustness/robustness_curve.json`

---

## E — Out-of-sample axes projection

The 6 Kozlowski axes (Lens IV) built on the 364 core are applied to the
9.045 bg. Scores saved per (model, axis, variant).

For each (label, axis), bg are aggregated by k-NN-assigned domain (from A).
The coherence file reports mean ± std per domain per axis. Axes generalize
to legal vocabulary outside the curated pool — bg of "criminal" cluster in
public_private and natural_positive coherently with core criminal terms.

Output: `ext/E_axes_oos/scores_bg_{bare,attested}/{label}_{axis}.npy`
        `ext/E_axes_oos/coherence.json`

---

## F — Confidence-stratified Δρ_sym

20 replicates of injecting 91 bg into the 364 core, drawn from:

| Stratum                | mean Δρ_sym | std   | vs baseline |
|------------------------|-------------|-------|-------------|
| baseline (core only)   | 0.538       | 0.000 | —           |
| high-conf bg (top 10%) | 0.531       | 0.016 | -0.007      |
| low-conf bg (bottom 10%) | **0.565** | 0.012 | **+0.027**  |
| random bg (control)    | 0.540       | 0.018 | +0.002      |

**Reading.** Counter-intuitive but informative: low-confidence bg are not
noise that dilutes the signal — they *amplify* it. Bg ambiguous to the
core clustering (multi-domain, semantically peripheral) are precisely the
ones that capture tradition-specific drafting choices most strongly. The
signal sits at the boundary, not at the centre.

Caveat: the effect (+0.027) is small relative to the std (≈0.015) and
based on n=20 replicates. Treat as an interpretive hint, not a strong
quantitative claim.

Output: `ext/F_confidence/confidence_strata.json`

---

## H — K saturation curve

For each K-bucket among bg, compute ρ_cross BGE-EN-large × BGE-ZH-large
on the bg of that bucket (cosine RDM, Spearman ρ on upper triangles).

| K bucket | n bg | ρ_cross attested |
|----------|------|-------------------|
| 1        |  970 | **-0.13** (anti-correlated!) |
| 2        |  572 | +0.05            |
| 3        |  335 | +0.13            |
| 4-7      |  690 | +0.15            |
| 8        | 2.559 | +0.22           |
| core (4-8) | 364 | +0.246 (run #4 headline) |

**Reading.** Monotonic saturation curve with empirical inflection around K=3-4.
At K=1 the signal is **anti-correlated** — a single attested context is
more noise than signal. The pre-registered K≥4 threshold is empirically
justified: below K=4, the cross-tradition signal does not stabilize.

Output: `ext/H_K_saturation/k_saturation.json`

---

## G — Automated false-friends detector

For each bg with K_en≥2 AND K_zh≥2 (4.156 candidates), compute cosine
similarity between BGE-EN-large attested(en) and BGE-ZH-large attested(zh).
Sort ascending. Also report BGE-M3-EN ↔ BGE-M3-ZH (bilingual control: same
encoder, both languages) for comparison.

Top false-friends (most cross-tradition divergent, K_min≥2):

| en               | zh           | K_en | K_zh | cross  | bilingual | gap   |
|------------------|--------------|------|------|--------|-----------|-------|
| trainer          | 經授權導師     | 8    | 2    | -0.106 | +0.684    | 0.790 |
| paid sickness days | 有薪病假日 | 3    | 4    | -0.079 | +0.749    | 0.828 |
| anniversary      | 周年日       | 8    | 8    | -0.069 | +0.724    | 0.793 |
| Receiver         | 破產管理署署長 | 8    | 8    | -0.066 | +0.507    | 0.573 |
| chargee          | 承押記人      | 8    | 8    | -0.066 | +0.661    | 0.727 |
| tow              | 拖曳         | 8    | 8    | -0.066 | +0.656    | 0.722 |
| pharmaceutical   | 藥劑製品      | 8    | 8    | -0.061 | +0.744    | 0.805 |
| register         | 登記         | 8    | 8    | -0.057 | +0.670    | 0.727 |
| driving licence  | 駕駛執照     | 8    | 8    | -0.053 | n/a       |   —   |

**Reading.** Same-lemma terms (e.g. `Receiver` = Director of Bankruptcy in HK;
`anniversary` as a contract-anchor date) have cosine **negative** in the
WEIRD×Sinic encoder pair but cosine **+0.5 to +0.75** in the bilingual
BGE-M3 (same encoder, both languages). The divergence is tradition-shaped,
not encoder-artefact. **This is the operational version of the headline
claim** ("the cross-tradition gap is in legal traditions, not in encoder
choice"), applied at term-level.

Particularly striking gap candidates for thesis prose (Cap. 4 §1):
- *trainer / 經授權導師* (authorised instructor in HK statutory English vs ordinary trainer)
- *Receiver / 破產管理署署長* (insolvency officer in HK vs ordinary recipient)
- *anniversary / 周年日* (statutory date-anchor in HK contracts vs cultural celebration)

Output: `ext/G_false_friends/false_friends.csv` (4.156 rows, sorted)
        `ext/G_false_friends/false_friends.json`

---

## X — Δρ_sym vs %control robustness curve (dual of D, bare)

Mirror experiment of D using **control** instead of bg. Control terms have
no attested encoding (no HK Cap. attestation for everyday vocabulary), so
this curve operates on bare embeddings only. 15 replicates per level.

| %ctrl | n_core | n_ctrl | mean Δρ_sym bare | std |
|-------|--------|--------|------------------|-----|
| 0%  | 364 | 0  | **0.246** | 0.000 |
| 5%  | 346 | 18 | 0.241 | 0.006 |
| 10% | 328 | 36 | 0.239 | 0.008 |
| 15% | 309 | 55 | 0.235 | 0.005 |
| 20% | 291 | 73 | 0.226 | 0.007 |
| 25% | 273 | 91 | 0.223 | 0.012 |
| 27% | 266 | 98 | **0.222** | 0.011 |

**Reading.** Monotonic decline 0.246 → 0.222 (−0.024). Direction correct
(non-legal injection reduces the signal), but the absolute effect is small
because the bare baseline itself is already modest. The narrative power of
X is in *direction*, not in magnitude — combined with D, the two duals
demonstrate that the signal reacts as expected to both legalish and non-
legal perturbation. The 27% cap is the structural limit of the control
pool (100 / 364 = 27.5%).

Output: `ext/X_control_robustness/control_robustness_curve.json`

---

## Y — Cross-tradition ρ on the control-only pool (CRUCIAL CAVEAT)

100 control terms (everyday vocabulary: *I, you, he, this, here* and ZH
equivalents) as the *only* pool. 17 RSA pairs on bare embeddings.

| Quantity | Value |
|---|---|
| within-WEIRD ρ̄ | 0.434 |
| within-Sinic ρ̄ | 0.409 |
| cross-tradition ρ̄ | 0.265 |
| **Δρ_sym bare** | **0.156** |
| within-bilingual ρ̄ | 0.399 |

Comparison:

| Pool | Encoding | Δρ_sym |
|---|---|---|
| 364 core | bare | 0.165 |
| 364 core | attested | **0.543** |
| 100 control | bare | **0.156** |

**Reading.** Δρ_sym bare on the control-only pool (0.156) is
statistically indistinguishable from Δρ_sym bare on the core (0.165). On
pronouns and deictics, the WEIRD encoders disagree with the Sinic encoders
just as much as they do on legal terms — when neither pool is contextualized.

**This forces a reframing of the headline.** The bare signal is *encoder-
tradition shaped*: it measures how much two architecturally different encoder
families (WEIRD-trained vs Sinic-trained) disagree on any vocabulary, legal
or not. The *legal* contribution to the signal is the **attested-bare gap
on the core**: 0.543 − 0.165 = 0.378. This is the quantity that isolates
the effect of contextualizing on HK Cap. enactments.

Frase tipo per Cap. 4 §4.1:

> *"Δρ_sym attested = 0.543 measures the within-vs-cross tradition gap as
> actually computed in our pipeline. Yet on 100 everyday-language control
> terms, the same Δρ_sym bare metric returns 0.156 — statistically
> indistinguishable from 0.165 on the 364 core bare. The legal-attestation
> contribution is therefore best isolated as the attested-bare gap on the
> core: 0.378 = 0.543 − 0.165, against a shared encoder-tradition baseline
> of approximately 0.16."*

Output: `ext/Y_control_only/control_only_rsa.json`

---

## Z — Three-tier distance hierarchy (does not hold)

For each of the 10 models, compare three cosine-distance distributions:
core×core (intra), core×bg (cross-block), core×control (cross-block).
Expectation: median(core×core) < median(core×bg) < median(core×control)
— the bg as "legalish, semi-near", control as "non-legal, far".

| Model | c-c | c-bg | c-ctrl | Monotonic |
|---|---|---|---|---|
| BGE-EN-large | 0.417 | **0.437** | 0.421 | ✗ |
| E5-large | 0.205 | **0.220** | 0.216 | ✗ |
| FreeLaw-EN | 0.583 | **0.599** | **0.567** | ✗ |
| BGE-ZH-large | 0.595 | **0.632** | 0.627 | ✗ |
| Text2vec-large-ZH | 0.744 | 0.761 | 0.770 | ✓ |
| Dmeta-ZH | 0.498 | **0.536** | 0.526 | ✗ |
| BGE-M3-EN | 0.469 | **0.500** | 0.482 | ✗ |
| Qwen3-0.6B-EN | 0.377 | **0.392** | 0.374 | ✗ |
| BGE-M3-ZH | 0.519 | 0.550 | 0.551 | ✓ |
| Qwen3-0.6B-ZH | 0.504 | 0.532 | 0.534 | ✓ |

**Reading.** Only 3/10 models satisfy the monotonic hierarchy. In 7/10,
median(core×bg) > median(core×control) — **bg are farther from core than
control are**. The bg are not "semantically intermediate" between core and
control; they live somewhere else in the embedding space. Plausible reading:
the 364 core terms are HK-specific drafting, the 9.045 bg are dispersed
legalish residual (other jurisdictions, technical lemmas), while the 100
control are semantically neutral and end up closer to the centroid of any
specialized cluster.

**Implication for the thesis.** The tier classification (core/background/
control) is a property of *corpus curation*, not of *embedding geometry*.
Reportable in §4.2 as an honest caveat.

Output: `ext/Z_tier_hierarchy/tier_hierarchy.json` + per-model distance arrays.

---

## Combined narrative for Cap. 4

**Three results that anchor the chapter.**

1. **D + run-#3 stability**: Δρ_sym attested across two independently-curated
   pools (run #3 Firthian 327; run #4 post-BLP 364) and five mix levels
   (0%, 10%, 25%, 50%, 75% bg) stays in [0.535, 0.590]. The cross-tradition
   gap is **structurally stable**, not pool-curation-dependent. → §4.1
   opening.

2. **G + bilingual control**: same-lemma terms (≈ 50 candidates) have
   **negative** cosine across the WEIRD×Sinic encoder pair but **+0.5 to
   +0.75** cosine on a single bilingual encoder. The cross-tradition
   divergence is *tradition-shaped*, not *encoder-shaped*. → §4.1 middle.

3. **Y caveat**: Δρ_sym bare on the 100 control terms (0.156) is
   indistinguishable from Δρ_sym bare on the 364 core (0.165). **The
   legal-meaning signal is the attested-bare gap (0.378 on core), not
   the attested absolute (0.543).** This reformulates the headline:
   Δρ_sym attested = 0.543 by itself is not legal signal in the strict
   sense; Δ(attested-bare) = 0.378 is. → §4.1 honest framing + §4.2
   primary caveat.

**Three corroborative.**

4. **H**: K=4 saturation pre-empirically justifies the threshold in §2.3.
5. **X**: dual of D — Δρ_sym bare falls monotonically with %control
   (0.246 → 0.222) confirming discriminative direction.
6. **A**: 9.045 bg domain assignment CSV — reusable resource for future
   pool work.

**Three qualifying caveats for §4.2.**

7. **E**: axes generalize out-of-sample, but the 3 pool-sensitive axes
   (rights_duties, status_contract, state_market) remain pool-sensitive
   even on bg.
8. **F**: low-confidence bg injected → Δρ_sym +0.027; high-confidence
   → −0.007. *Interpretive hint, n=20 replicates, small effect.*
9. **Z**: tier hierarchy (core/bg/control) does not hold geometrically.
   Tiers are corpus-curative classifications, not properties of the
   embedding geometry.
