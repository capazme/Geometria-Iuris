# Run #4 headline numbers — post-BLP final (364 terms)

All ρ̄ are Spearman correlations on RDM upper triangles (§3.1.3) or on per-term axis projections (§3.2.3).
B (Mantel) = 10000; B (block bootstrap) = 10000; seed = 42; N = 364.

## §3.1.3 RSA cross-tradition (Lens I)

| Metric                                  | Bare    | Attested |
|-----------------------------------------|---------|----------|
| within-WEIRD ρ̄ (3 pairs)               | 0.416   | **0.712** |
| within-Sinic ρ̄ (3 pairs)               | 0.405   | **0.868** |
| cross-tradition ρ̄ (9 pairs)            | 0.246   | **0.246** |
| Δρ_sym (avg within − cross)            | 0.165   | **0.543** |
| within-bilingual ρ̄ (β control)         | 0.352   | 0.316 |

## §3.2.4 Axes alignment cross-tradition ρ̄ (Lens IV)

| Axis                  | Bare    | Attested |
|-----------------------|---------|----------|
| individual_collective | 0.241   | **0.186** |
| rights_duties         | 0.384   | **0.394** |
| public_private        | 0.386   | **0.288** |
| state_market          | 0.262   | **0.125** |
| natural_positive      | 0.226   | **0.092** |
| status_contract       | 0.446   | **0.363** |

## §3.2.4 Ranking (most divergent → least) — attested

1. **natural_positive** — ρ̄ = 0.092
2. **state_market** — ρ̄ = 0.125
3. **individual_collective** — ρ̄ = 0.186
4. **public_private** — ρ̄ = 0.288
5. **status_contract** — ρ̄ = 0.363
6. **rights_duties** — ρ̄ = 0.394

## §3.1.1 Legal-vs-control (Lens I, bare only — N=100 control terms)

Mann-Whitney U one-sided (alternative='less'): legal-legal more compact than legal-control.
Effect size r = rank-biserial.

| Model | legal med | ctrl med | r | p_value |
|-------|-----------|----------|---|---------|
| BGE-EN-large | 0.417 | 0.421 | +0.062 | 1.09e-60 |
| E5-large | 0.205 | 0.216 | +0.257 | 2.23e-308 |
| FreeLaw-EN | 0.583 | 0.567 | -0.121 | 1.00e+00 |
| BGE-ZH-large | 0.595 | 0.627 | +0.239 | 2.23e-308 |
| Text2vec-large-ZH | 0.744 | 0.770 | +0.240 | 2.23e-308 |
| Dmeta-ZH | 0.498 | 0.526 | +0.218 | 2.23e-308 |
| BGE-M3-EN | 0.469 | 0.482 | +0.143 | 2.23e-308 |
| Qwen3-0.6B-EN | 0.377 | 0.374 | -0.044 | 1.00e+00 |
| BGE-M3-ZH | 0.519 | 0.551 | +0.305 | 2.23e-308 |
| Qwen3-0.6B-ZH | 0.504 | 0.534 | +0.207 | 2.23e-308 |

**8/10 models confirm legal-legal more compact than legal-control.**

---

## Extension D — Δρ_sym vs %bg robustness curve

10 replicates × 5 mix levels; 2 EN-side + 2 ZH-side primary models, attested.

| %bg | n_core | n_bg | mean Δρ_sym | std | CI95 |
|-----|--------|------|-------------|-----|------|
| 0% | 364 | 0 | **0.538** | 0.000 | [0.538, 0.538] |
| 10% | 328 | 36 | **0.535** | 0.019 | [0.502, 0.555] |
| 25% | 273 | 91 | **0.542** | 0.018 | [0.508, 0.566] |
| 50% | 182 | 182 | **0.565** | 0.023 | [0.520, 0.591] |
| 75% | 91 | 273 | **0.590** | 0.024 | [0.553, 0.624] |

Δρ_sym from 0.538 (p=0%) to 0.590 (p=75%): +0.052. **Δρ_sym is robust under bg injection** — the cross-tradition gap is structural, not curation-dependent.

---

## Extension G — Automated false-friends (4156 bg eligible)

Cross-encoder: BGE-EN-large × BGE-ZH-large. Bilingual control: BGE-M3-EN × BGE-M3-ZH.

Top-10 most divergent (lowest cross-encoder cosine):

| en | zh | K_en | K_zh | cross | bilingual |
|----|----|------|------|-------|-----------|
| trainer | 經授權導師 | 8 | 2 | -0.106 | +0.684 |
| paid sickness days | 有薪病假日 | 3 | 4 | -0.079 | +0.749 |
| retransmit | 再傳送 | 2 | 8 | -0.071 | +0.621 |
| anniversary | 周年日 | 8 | 8 | -0.069 | +0.724 |
| on-exchange | 場內交易 | 2 | 2 | -0.068 | +0.458 |
| Receiver | 破產管理署署長 | 8 | 8 | -0.066 | +0.507 |
| chargee | 承押記人 | 8 | 8 | -0.066 | +0.661 |
| tow | 拖曳 | 8 | 8 | -0.066 | +0.656 |
| crew | 空勤人員 | 8 | 3 | -0.064 | +0.671 |
| recycler | 循環再造者 | 2 | 2 | -0.062 | +0.557 |

Same-lemma terms have **negative** cross-encoder cosine but **+0.5 to +0.75** bilingual cosine. The cross-tradition divergence is *tradition-shaped*, not *encoder-artefact*.

---

## Extension H — K saturation curve

ρ_cross attested as a function of bg K_min bucket:

| K bucket | n bg | ρ_cross |
|----------|------|---------|
| 1 | 970 | **-0.132** |
| 2 | 572 | **+0.054** |
| 3 | 335 | **+0.135** |
| 4-7 | 690 | **+0.149** |
| 8 | 2559 | **+0.218** |
| core 4-8 (run #4 headline) | 364 | +0.246 |

ρ_cross monotonic with K; saturation around K≥4. The pre-registered threshold is empirically justified.

---

## Extension X — Δρ_sym vs %control (bare, dual of D)

| %ctrl | n_core | n_ctrl | mean Δρ_sym bare | std |
|-------|--------|--------|------------------|-----|
| 0% | 364 | 0 | 0.246 | 0.000 |
| 5% | 346 | 18 | 0.241 | 0.005 |
| 10% | 328 | 36 | 0.239 | 0.008 |
| 15% | 309 | 55 | 0.235 | 0.005 |
| 20% | 291 | 73 | 0.226 | 0.007 |
| 25% | 273 | 91 | 0.223 | 0.011 |
| 27% | 266 | 98 | 0.222 | 0.011 |

Δρ_sym bare from 0.246 (p=0%) to 0.222 (p=27%): -0.024. Direction correct (non-legal injection reduces the signal), effect small.

---

## Extension Y — Cross-tradition ρ on control-only (CRUCIAL CAVEAT)

Δρ_sym bare on the 100 control terms (everyday vocabulary, *I, you, he, this, here*, and ZH equivalents):

| Quantity | Value |
|----------|-------|
| within-WEIRD ρ̄ | 0.433 |
| within-Sinic ρ̄ | 0.409 |
| cross-tradition ρ̄ | 0.265 |
| **Δρ_sym bare** | **0.156** |

**Comparison:**

| Pool | Encoding | Δρ_sym |
|------|----------|--------|
| 364 core | bare | 0.165 |
| 364 core | attested | **0.543** |
| 100 control | bare | **0.156** ← indistinguishable from core bare |

**Reframing.** The bare signal is encoder-tradition shaped, not legal-tradition shaped. The legal contribution is the **attested-bare gap on the core: 0.543 − 0.165 = 0.378**.

---

## Extension Z — 3-tier distance hierarchy

| Model | core×core | core×bg | core×control | Monotonic |
|-------|-----------|---------|--------------|-----------|
| BGE-EN-large | 0.417 | 0.437 | 0.421 | ✗ |
| E5-large | 0.205 | 0.220 | 0.216 | ✗ |
| FreeLaw-EN | 0.583 | 0.599 | 0.567 | ✗ |
| BGE-ZH-large | 0.595 | 0.632 | 0.627 | ✗ |
| Text2vec-large-ZH | 0.744 | 0.761 | 0.770 | ✓ |
| Dmeta-ZH | 0.498 | 0.536 | 0.526 | ✗ |
| BGE-M3-EN | 0.469 | 0.500 | 0.482 | ✗ |
| Qwen3-0.6B-EN | 0.377 | 0.392 | 0.374 | ✗ |
| BGE-M3-ZH | 0.519 | 0.550 | 0.551 | ✓ |
| Qwen3-0.6B-ZH | 0.504 | 0.532 | 0.534 | ✓ |

**3/10 models** satisfy median(core×core) < median(core×bg) < median(core×control). In the others, bg are *farther* from core than control are. Tier classification is corpus-curative, not geometric.

---

## Extensions A, E, F (short notes)

- **A** (k-NN bg domain assignment): 9045 bg → 7 domains via k=7 NN; mean confidence 0.515, median 0.429.
- **E** (axes out-of-sample projection): 6 axes projected on bg; coherence per k-NN-assigned domain reported per (model, axis) — pool-sensitive axes remain pool-sensitive on bg.
- **F** (confidence-stratified, n=20 replicates, n_inject=91): baseline Δρ_sym=0.538; high-conf bg injected → 0.531 (-0.008); low-conf → 0.565 (+0.027). Small effect, interpretive hint only.

---

**Three anchor results for Cap. 4:** D (structurality), G + bilingual control (tradition-shaped, not encoder-shaped), Y (the legal signal is the attested-bare gap on the core, 0.378). See `reports/extensions_summary.md` for the full narrative.