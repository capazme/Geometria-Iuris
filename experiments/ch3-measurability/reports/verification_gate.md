# Verification gate — run #4 post-BLP

Per PLAN.md §8. If any check is FAIL: stop and investigate.

| Check                                  | Status | Value vs threshold |
|----------------------------------------|--------|--------------------|
| ≥10 embeddings dirs with bare + attested | PASS   | 10/10 |
| ρ̄_cross attested in [run3 ±0.10]      | PASS   | 0.246 vs 0.259 |
| ρ̄_W attested in [run3 ±0.10]          | PASS   | 0.712 vs 0.760 |
| ρ̄_S attested in [run3 ±0.10]          | PASS   | 0.868 vs 0.845 |
| Δρ_sym attested ≥ 0.4                  | PASS   | 0.543 |
| Mantel p_max ≤ 0.001                   | PASS   | 0.000100 |
| Holm p_max ≤ 0.005                     | PASS   | 0.001700 |
| legal-vs-control: ≥8/10 models with r>0 and p<0.05 | PASS   | 8/10 |

**8/8 checks passed**.