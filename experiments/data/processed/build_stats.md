# Build Statistics — legal_terms.json

## 1. Input / output
| | Count |
|--|-------|
| Input (doj_filtered.json) | 9387 |
| Excluded (EXCLUDE list) | 15 |
| → Background (polysemy >50 variants) | 196 |
| → Background (no domain match) | 8779 |
| **Core (domain assigned)** | **397** |
| **Background** | **8975** |
| **Control (Swadesh)** | **100** |
| **Total output** | **9472** |
| zh_canonical corrections applied | 6 |

## 2. Core terms per domain
| Domain | Count |
|--------|-------|
| administrative | 12 |
| civil | 136 |
| constitutional | 48 |
| criminal | 66 |
| international | 52 |
| labor_social | 30 |
| procedure | 53 |

## 3. Domain assignment resolution
| Resolution type | Count |
|-----------------|-------|
| none | 8779 |
| unique | 397 |

## 4. Notes
- Core terms: assigned to one of 7 legal domains via keyword rules
- Background terms: in embedding pool for k-NN neighbourhood but not assigned a domain
- Control terms: Swadesh 100 basic vocabulary, semantic baseline
- `domain_review.json`: all terms where conflict_priority resolution was used
  (multiple domains matched; priority rule applied — may warrant manual review)
- Ref: domain_mapping_rules.md v1.1, hk_specific_terms.md v1.0