# HK-Specific Terms — Classification and Treatment

**Version**: 1.0
**Date**: 2026-02-21
**Author**: domain-mapper agent
**Scope**: Terms from the HK DOJ Bilingual Glossary that are specific to the
Hong Kong / UK common law system and may lack meaningful Sinic equivalents.

**Thesis reference**: §8.3.3 (Boundary objects: the concepts that resist),
§9.1.3 (What this laboratory makes visible — and what it cannot see)

---

## Why This Matters

Hong Kong is the thesis's "natural laboratory" precisely because it operates common law
in Chinese. However, some terms are so embedded in the UK/HK institutional infrastructure
that their Chinese translations in the DOJ glossary are transliterations or descriptive
glosses, not genuine conceptual equivalents in the Sinic legal tradition.

For such terms, the ZH embedding will anchor on the translated form (Traditional Chinese
as used in HK), which may sit at an idiosyncratic location in the ZH embedding space —
neither aligned with Mainland legal ZH nor with the WEIRD EN embedding's conceptual
neighborhood. This creates a **third-tradition artifact** that can contaminate the
EN↔ZH comparison.

Three treatment options per term:
- **EXCLUDE**: Remove from core corpus. Too HK-specific to generalize.
- **FLAG**: Keep in corpus but mark `hk_specific: true`. Include in analysis but
  interpret results with caution. Possibly discuss in §8.3.3.
- **BOUNDARY OBJECT**: Keep in core corpus and foreground as a scientifically interesting
  case. The concept's resistance to clean cross-tradition mapping is itself a finding.
  Report in §8.3.3 and potentially §9.5.3.

---

## Category A: Legal Profession Roles

These terms describe professional categories that exist in common law jurisdictions but
have no structural equivalent in the Sinic civil law tradition (which uses 律师 as a
single professional category without the barrister/solicitor split).

| Term | ZH (DOJ) | Treatment | Rationale |
|------|----------|-----------|-----------|
| `barrister` | 大律師 | EXCLUDE | The barrister/solicitor split is UK/HK specific. Mainland ZH uses 律师 for all lawyers. The DOJ ZH is a HK neologism (大律師 = "senior lawyer"), not a Sinic tradition concept. |
| `solicitor` | 律師 | FLAG | Maps to 律師 in ZH (same word used for all lawyers in Mainland). In HK context the meaning is narrowed to the non-barrister branch. This divergence in semantic coverage is itself a false friend candidate. |
| `King's Counsel / Queen's Counsel` | 御用大律師 | EXCLUDE | A British Crown appointment. The ZH translation is purely descriptive. No Sinic equivalent exists; the concept has no purchase in civil law tradition. |
| `Bencher` | 掌管四所大律師學院的資深會員 | EXCLUDE | Specific to the English Inns of Court institution. No ZH legal tradition equivalent. |
| `a consiliis` | 屬大律師身分的 | EXCLUDE | Latin designation for counsel rank in English courts. The ZH gloss is descriptive only. |
| `Official Solicitor` | 法定代表律師（香港）| EXCLUDE | A HK statutory office with no Mainland equivalent. |
| `instructing solicitor` | （代當事人向大律師）發出指示的律師 | EXCLUDE | Role-specific to the barrister/solicitor system. |

---

## Category B: HK/UK Institutional Structures

Court structures and institutions created by HK or UK statute with no cross-tradition
conceptual equivalent.

| Term | ZH (DOJ) | Treatment | Rationale |
|------|----------|-----------|-----------|
| `Judicial Committee of the Privy Council` | 樞密院司法委員會（英） | EXCLUDE | Final appellate body for UK territories. Replaced in HK by Court of Final Appeal post-1997. Deeply UK-constitutional, no Sinic equivalent. |
| `Court of Final Appeal` (HK) | 香港終審法院 | FLAG | Exists as HK-specific institution but the concept of "final appellate court" has equivalents in Sinic tradition (最高人民法院). Flag as HK-specific institution but the abstract concept is cross-traditional. |
| `Chancery Division` | 大法官法庭（英） | EXCLUDE | English court division inheriting equity jurisdiction. No structural equivalent in civil law or Sinic system (which has no equity/law split). |
| `Inns of Court` | 四所大律師學院（英） | EXCLUDE | English professional guild institution. No Sinic equivalent. |
| `Lands Tribunal` | 土地審裁處（香港） | EXCLUDE | HK-specific statutory tribunal for land disputes. Too institutional-specific. |
| `High Court` (HK) | 高等法院（香港）| FLAG | Generic label but HK-specific structure. The abstract "High Court" concept maps broadly but the HK institution is specific. |
| `District Court` (HK) | 區域法院（香港）| FLAG | Similar to above — the institutional label is HK-specific. |
| `Magistrate's Court` / `magistrate` (HK) | 裁判官 / 裁判法院 | FLAG | The magistrate as low-level judicial officer has partial equivalents (基层法院 in Mainland), but the HK institution is distinct. |
| `Legislative Council` | 立法會（香Kong）| FLAG | HK-specific legislature. The concept of "legislature" is cross-traditional; the institution is HK-specific. |
| `Executive Council` | 行政會議（香港）| FLAG | HK-specific advisory body. No direct Sinic equivalent but the executive council concept has approximate parallels. |
| `Chief Executive` (HK) | 行政長官（香港）| FLAG | HK-specific constitutional office. The term is also used generically (CEO). Flag as HK-specific in constitutional usage. |

---

## Category C: Common Law / Equity Distinction

The equity/common law split is constitutive of English legal history. The Sinic civil
law tradition has no equivalent binary. These terms embed this split structurally.

| Term | ZH (DOJ) | Treatment | Rationale |
|------|----------|-----------|-----------|
| `common law` | 不成文法 | BOUNDARY OBJECT | The ZH gloss (不成文法 = "unwritten law") captures one aspect but loses the historical/institutional dimension. This mistranslation IS the interesting finding: the concept of "common law" as a tradition has no equivalent, and the ZH model will anchor on "unwritten law" semantics. Scientifically valuable for §8.3.3. |
| `equity` (Chancery sense) | 衡平法 | BOUNDARY OBJECT | The conscience-based jurisdiction of equity has no Sinic equivalent. The ZH term 衡平法 is a coinage (衡平 = balance/equity). Its embedding will be isolated. This is a prime false friend candidate — see domain_mapping_rules.md §False Friend Candidates. |
| `at law and in equity` | 在普通法及衡平法上 | EXCLUDE | A compound phrase encoding the law/equity split. Too system-specific to stand as a core term. |
| `clog on the equity of redemption` | 贖回業權的障礙 | EXCLUDE | Highly technical equity doctrine, HK/UK specific. |

---

## Category D: Crown / Royal Prerogative Concepts

Terms grounded in the UK constitutional doctrine of Crown prerogative, which has no
equivalent in republican or Sinic political philosophy.

| Term | ZH (DOJ) | Treatment | Rationale |
|------|----------|-----------|-----------|
| `Crown lease` | 官契 | EXCLUDE | HK colonial land tenure system. No Sinic equivalent. |
| `Block Crown Lease` | 集體官契 | EXCLUDE | Same — HK colonial land system specific. |
| `abjuration of the realm` | 棄國宣誓 | EXCLUDE | Medieval English concept. No ZH embedding anchor. |
| `royalty` (Crown) | 使用費 | FLAG | The ZH gloss maps to "usage fee" (royalty as payment), missing the Crown dimension. Polysemy creates a false friend: EN "royalty" → Crown origin; ZH 使用費 → purely financial. Flag. |

---

## Category E: Latin Terms

Latin terms present a systematic challenge. The HK DOJ glossary includes many Latin
legal maxims and terms in current use in English common law. Some have standard Chinese
translations (which are themselves learned coinages in HK legal usage). Others are simply
transliterated or glossed descriptively.

### Treatment Policy for Latin Terms

**Decision**: Include Latin terms as core terms IF AND ONLY IF:
1. They have a standard ZH equivalent in the DOJ glossary (not a mere description), AND
2. They are in active use in both traditions (i.e., known in Sinic legal education as
   concepts, even if not used in daily ZH law practice).

**Exclusion criteria**: Exclude if the ZH entry is purely a descriptive gloss with no
recognised conceptual foothold in Sinic legal discourse.

### Latin Terms — Case-by-Case Analysis

| Latin term | ZH (DOJ) | Include? | Rationale |
|------------|----------|----------|-----------|
| `mens rea` | 犯罪意圖 / 犯罪心態 | INCLUDE | Standard in criminal law across both traditions. ZH criminal law has the concept 主观过错 (subjective fault). Good cross-traditional anchor. |
| `actus reus` | 犯罪行為 | INCLUDE | Paired with mens rea. The criminal act concept is universal. ZH criminal law: 客观行为 element. |
| `habeas corpus` | 人身保護令 | BOUNDARY OBJECT | Exists in HK common law. Mainland ZH has no equivalent writ — the concept of challenging detention via court writ is absent from Sinic tradition. Scientifically valuable as a concept that "resists" translation. |
| `bona fide` | 善意 | INCLUDE | 善意 (good faith) is a well-established concept in both traditions (ZH civil law: 善意第三人, bona fide purchaser equivalent). |
| `ab initio` | 從一開始 | FLAG | Procedural Latin, common in legal reasoning. ZH equivalent is functional (自始) but the term is more a logical operator than a legal concept. |
| `locus standi` | 申請資格；法律地位 | INCLUDE → `procedure` | Standing/locus standi is a core procedural concept. ZH procedural law has equivalent doctrine (原告资格). |
| `res judicata` | 已判事項 | INCLUDE → `procedure` | Issue preclusion doctrine exists in both traditions. ZH equivalent: 一事不再理. |
| `ex parte` | 單方面 | FLAG | Procedural term. Common in HK courts. ZH equivalent concept exists but terminology differs. Flag as HK-procedural. |
| `obiter dictum` | 附帶意見 | FLAG | Relevant to stare decisis doctrine, which doesn't apply in Sinic civil law. The concept of non-binding judicial observation is present but less developed. |
| `ratio decidendi` | 判決理由 | BOUNDARY OBJECT | Central to common law case method. Sinic tradition does not use binding precedent (stare decisis), so ratio decidendi has no full equivalent. The ZH embedding will anchor on "reasons for judgment" without the binding-precedent dimension. |
| `stare decisis` | 遵循先例 | BOUNDARY OBJECT | The doctrine itself defines the common law system. Sinic civil law does not recognize binding precedent (though guiding cases are used). The cross-traditional gap here IS the finding. |
| `ultra vires` | 越權 | INCLUDE → `administrative` | The doctrine of acting beyond one's powers is universal in administrative law. ZH equivalent: 超越职权. |
| `prima facie` | 表面上 | INCLUDE → `procedure` | Procedural concept (prima facie case). Used in ZH courts in evidentiary context (表面证据). |
| `de facto` | 事實上 | INCLUDE → background | Too general to be a domain concept; use as background term. |
| `de jure` | 按照法律的 | INCLUDE → background | Same as de facto — background. |
| `inter alia` | 其中 | EXCLUDE | Discourse marker, not a legal concept. |
| `a fortiori` | 何況 | EXCLUDE | Logical operator, not a legal concept. |
| `in camera` | 以非公開形式 | INCLUDE → `procedure` | Closed proceedings concept is cross-traditional. ZH equivalent: 不公开审理. |
| `inter vivos` | 生前 | INCLUDE → `civil` | Trust/gift concept used in succession law. ZH equivalent: 生前赠与. |
| `sui generis` | 獨特的 | EXCLUDE | Descriptive adjective, not a domain concept. |
| `nemo judex in causa sua` | 不可在自身案件中擔任法官 | FLAG | Foundational natural justice principle. In Sinic tradition, the recusal doctrine exists but is less explicitly theorized as a Latin maxim. |
| `audi alteram partem` | 聆聽另一方的陳詞 | FLAG | Natural justice principle (hear the other side). Exists in Sinic administrative law under different formulation. |
| `pacta sunt servanda` | 協議必須遵守 | INCLUDE → `international` | Core international law principle. Present in both traditions. ZH: 条约必须遵守. |
| `jus cogens` | 強制性規範 | INCLUDE → `international` | Peremptory norms of international law. Present in both traditions through UN Charter / international law education. |
| `erga omnes` | 對所有人具有效力的（義務）| INCLUDE → `international` | Obligations erga omnes concept is used in both traditions' international law scholarship. |

---

## Category F: HK-Unique Procedural/Institutional Terms

| Term | ZH (DOJ) | Treatment | Rationale |
|------|----------|-----------|-----------|
| `Rules of the High Court` | 高等法院規則（香港）| EXCLUDE | HK-specific procedural code. |
| `accusatorial procedure` | 舉證式司法程序 | FLAG | Adversarial vs. inquisitorial distinction is cross-traditional BUT the ZH gloss explains it as a common law concept. Flag as potentially asymmetric: WEIRD models will have this well-embedded; ZH models may anchor on the explanatory gloss. |
| `inquisitorial procedure` | 彈劾式司法程序 | FLAG | Counterpart to accusatorial. Same flag applies. |

---

## Summary Statistics

| Category | Total identified | Exclude | Flag | Boundary Object | Include |
|----------|-----------------|---------|------|-----------------|---------|
| A: Legal profession roles | 7 | 6 | 1 | 0 | 0 |
| B: HK/UK institutional structures | 11 | 3 | 7 | 0 | 1 |
| C: Common law / equity split | 4 | 2 | 0 | 2 | 0 |
| D: Crown / prerogative | 4 | 3 | 1 | 0 | 0 |
| E: Latin terms | 23 | 4 | 7 | 4 | 8 |
| F: HK procedural/institutional | 3 | 1 | 2 | 0 | 0 |
| **Total** | **52** | **19** | **18** | **6** | **9** |

---

## Boundary Objects: Proposed Thesis Treatment

The 6 boundary object terms warrant dedicated discussion in §8.3.3:

1. **`common law`** — The term that names the entire WEIRD tradition has no true Sinic
   equivalent. The ZH embedding anchors on "unwritten law," losing the historical/
   institutional dimension. This is the clearest case of a concept that "resists"
   cross-traditional mapping.

2. **`equity`** (Chancery sense) — The conscience-based equity jurisdiction is constitutive
   of English private law. No Sinic tradition equivalent. The ZH coinage 衡平法 will have
   an isolated, low-connectivity embedding.

3. **`habeas corpus`** — Liberty protection through adversarial writ. The Mainland Sinic
   tradition handles detention challenges through different mechanisms (检察院 oversight,
   administrative channels). The concept's absence from Sinic tradition means the ZH
   embedding anchors on the translated label without conceptual depth.

4. **`ratio decidendi`** — Central to common law case-based reasoning. The Sinic civil law
   tradition lacks a binding precedent doctrine (although guiding cases are used since 2010).
   The ZH embedding will sit close to "reasons for judgment" without the
   authoritative/binding dimension.

5. **`stare decisis`** — The doctrine of precedent itself. A structural difference between
   the two traditions. Expected to show among the highest neighborhood divergence scores
   (lowest Jaccard) in NDA Part A.

6. **`habeas corpus`** (see item 3 above — listed separately from `ratio decidendi` and
   `stare decisis` as it concerns fundamental liberty, not reasoning methodology).

---

## Curators' Note on HK as Natural Laboratory

The existence of these boundary objects does not undermine the thesis's use of HK as a
natural laboratory — it confirms it. HK provides:

1. **Genuine bilingually aligned terms** where both EN and ZH embeddings can be
   meaningfully compared (the majority of the corpus).
2. **Boundary objects** where the structural difference between legal traditions is
   visible in the very impossibility of alignment. These are the most scientifically
   interesting findings.

The HK glossary's institutional HK-specific terms (Category B) should be excluded
precisely to keep the analysis clean: we want to capture cross-*traditional* divergence,
not cross-*institutional* divergence caused by HK's particular constitutional settlement.
