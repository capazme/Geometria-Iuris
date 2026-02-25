# Domain Mapping Rules — 7 Legal Domains

**Version**: 1.1
**Date**: 2026-02-21
**Author**: domain-mapper agent
**Scope**: Rules for assigning terms from the HK DOJ Bilingual Legal Glossary to the 7
subject-matter domains used in the thesis *Geometria Iuris*.
**Validated against**: `experiments/data/processed/doj_filtered.json` (9,387 terms)

**Thesis reference**: §2.1 (selection problem), §2.1 (domain stratification),
§2.1 (edge cases)

---

## Rationale for 7 Domains

The taxonomy follows a **dual-justification** principle: every domain is grounded in
two independent authoritative sources, one from each legal tradition under comparison.

| Domain             | EuroVoc reference | NPC 部门法 category         |
| ------------------ | ----------------- | --------------------------- |
| `constitutional` | 1206 + 1236       | cat.1 宪法相关法            |
| `civil`          | 1211              | cat.2 民法商法              |
| `criminal`       | 1216              | cat.6 刑法                  |
| `administrative` | 1226              | cat.3 行政法                |
| `international`  | 1231              | academic curricula (国际法) |
| `labor_social`   | employment        | cat.5 社会法                |
| `procedure`      | 1221 (Justice)    | cat.7 诉讼法                |

**Key design decisions** (documented in `trace_dataset_design.md` §D1):

- `rights` merged into `constitutional`: subjective rights derive from constitutional
  foundations in both traditions (Windscheid, Jellinek, Crisafulli).
- `governance` renamed `administrative`: aligns with EuroVoc 1226 and NPC cat.3.
- `jurSwadesh listSwadesh listisprudence` dropped: meta-legal, no coverage in either authoritative taxonomy.
- `environmental_tech` dropped: insufficient cross-tradition symmetric coverage.
- `procedure` added: universal cross-tradition category, well-delimited in both systems.

---

## Domain 1: `constitutional`

### Positive Keywords

Terms whose headword contains (case-insensitive, root match):

```
constitution, constitut*, sovereign*, sovereignty, fundamental right*,
basic right*, human right*, civil liberty, civil liberties, separation of power*,
legislature, legislative, parliament*, congress, senate, referendum,
federalism, federaliz*, unitary state, republic, monarchy, citizenship,
nationality, naturalization, suffrage, election, electoral, franchise,
amendment, bill of rights, habeas corpus, due process, equal protection,
freedom of speech, freedom of assembly, freedom of religion, press freedom,
judicial review, constitutionality, unconstitutional, veto, prerogative,
emergency power*, martial law, state of emergency, basic law, head of state,
executive power, supremacy, separation, checks and balance*
```

### Negative Keywords (exclusion overrides)

Exclude despite positive signal:

```
- "constitutional court" as a procedural venue → keep as constitutional
  (the court type is constitutionally grounded, not merely procedural)
- "electoral fraud" → criminal
- "legislative drafting" → administrative (regulatory process)
- "legislative procedure" → procedure
```

### Ambiguity Resolution

| Border case                                                | Resolution                                                                                                                              |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Rights language in labor context (e.g., "right to strike") | →`labor_social` if the right is employment-specific; → `constitutional` only if the term is a general fundamental rights concept  |
| "Judicial review"                                          | →`constitutional` (grounds for challenge are constitutional); NOT `procedure` (it's a substantive doctrine, not a procedural tool) |
| "Due process"                                              | →`constitutional` (substantive constitutional principle across both traditions)                                                      |
| "Equal protection" vs "anti-discrimination"                | Equal protection →`constitutional`; workplace anti-discrimination → `labor_social`                                                |
| "Sovereignty" in international context                     | →`constitutional` (internal sovereignty); a term like "sovereign immunity" in international disputes → `international`            |
| "Habeas corpus"                                            | →`constitutional` (liberty guarantee), NOT `procedure` despite procedural form                                                     |
| "Emergency powers"                                         | →`constitutional`                                                                                                                    |
| Confucian concept of 仁政 ("benevolent government")        | → background corpus, not core; too philosophical for subject-matter taxonomy                                                           |

### Typical Terms (canonical examples)

```
constitution, sovereignty, citizenship, fundamental rights, human rights,
judicial review, separation of powers, legislature, referendum, amendment,
due process, equal protection, bill of rights, federalism, republic,
nationality, naturalization, suffrage, prerogative, habeas corpus,
freedom of expression, freedom of assembly, civil liberty, basic law,
electoral system, executive power, constitutional court, rule of law,
head of state, emergency powers
```

---

## Domain 2: `civil`

### Positive Keywords

```
contract*, agreement, obligation, breach, consideration, offer, acceptance,
damages, liability, tort*, negligence, trespass, nuisance, defamation,
property, ownership, title, easement, mortgage, pledge, lien, charge,
trust, beneficiary, trustee, fiduciary, equity, landlord, tenant, lease,
inheritance, succession, will, testament, intestate, probate, executor,
family, marriage, divorce, custody, adoption, maintenance, alimony,
consumer*, sale of goods, agency, principal, guarantor, surety,
unjust enrichment, restitution, rescission, frustration, misrepresentation,
fraud (civil), bailment, lien, assignment, novation, subrogation, indemnity
```

### Negative Keywords

```
- "criminal fraud" → criminal
- "equity" in equitable remedy sense → civil
- "equity" as "equity share" → background (commercial/financial)
- "trust" as "trust company" → background (commercial)
- "family court" as court → procedure
- "employment contract" → labor_social (the employment relationship governs)
```

### Ambiguity Resolution

| Border case               | Resolution                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Contract in employment    | →`labor_social` when the relationship governed is employer/employee; `civil` for commercial contracts generally                                                           |
| "Equity"                  | →`civil` when referring to the body of equitable doctrine (trusts, fiduciary duties); flag as HK-specific when referring to Chancery distinction (see hk_specific_terms.md) |
| Family law vs. social law | Family status, marriage, divorce →`civil`; social assistance, welfare benefits → `labor_social`                                                                          |
| Intellectual property     | IP rights (patent, copyright, trademark) →`civil` (private law); IP registration procedures → background                                                                   |
| Consumer protection       | Core concepts →`civil`; consumer protection regulation → `administrative`                                                                                                |
| "Damages"                 | →`civil` unless in a criminal sentencing context (then → `criminal`)                                                                                                     |
| "Restitution"             | →`civil` (unjust enrichment); "restorative justice" → `criminal`                                                                                                         |

### Typical Terms (canonical examples)

```
contract, tort, negligence, breach of contract, damages, property,
ownership, trust, mortgage, lease, inheritance, succession, will,
marriage, divorce, custody, fiduciary duty, consideration, offer,
acceptance, misrepresentation, unjust enrichment, easement, defamation,
nuisance, bailment, assignment, restitution, agency, indemnity,
frustration of contract, consumer rights, sale of goods, title
```

---

## Domain 3: `criminal`

### Positive Keywords

```
crime, criminal*, offence, offense, felony, misdemeanor, homicide, murder,
manslaughter, assault, battery, robbery, theft, larceny, burglary,
fraud (criminal), forgery, bribery, corruption, perjury, contempt of court,
extortion, blackmail, conspiracy, attempt, inchoate, aiding and abetting,
accessory, accomplice, accomplice liability, mens rea, actus reus,
guilty mind, intent, recklessness, negligence (criminal), strict liability,
sentence, sentencing, imprisonment, probation, parole, fine, conviction,
acquittal, plea, guilty plea, charge, arraignment, indictment, prosecution,
defendant, accused, victim, recidivism, criminal record, rehabilitation,
criminal law, penal code, penal*, punishment, penalty, deterrence,
capital punishment, death penalty, incarceration, remand, bail (criminal)
```

### Negative Keywords

```
- "civil fraud" → civil
- "criminal procedure" (as procedural term) → procedure
- "criminal jurisdiction" (as jurisdictional term) → procedure
- "extradition" → international (inter-state dimension dominates)
```

### Ambiguity Resolution

| Border case                                | Resolution                                                                                                                               |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| "Bail"                                     | →`criminal` (pretrial liberty in criminal proceedings); `civil` only if in civil debt context                                       |
| "Contempt of court"                        | →`criminal` if criminal contempt (disobeying court order with punitive sanction); `procedure` if civil contempt enforcement context |
| "Strict liability"                         | →`criminal` for strict liability offences; `civil` for strict liability torts; assign by context                                    |
| "Bribery" vs. "corruption"                 | Both →`criminal`; "public corruption" may also touch `administrative` but criminal is primary                                       |
| "Juvenile justice"                         | →`criminal` (remains criminal law, specialized subject)                                                                               |
| White-collar crime terms                   | →`criminal` unless purely regulatory (then → `administrative`)                                                                     |
| "Criminal negligence" vs. civil negligence | Distinguish by context: "criminal negligence" →`criminal`; general negligence → `civil`                                            |

### Typical Terms (canonical examples)

```
murder, manslaughter, assault, theft, robbery, fraud, bribery, perjury,
mens rea, actus reus, intent, recklessness, guilty plea, conviction,
acquittal, sentence, imprisonment, probation, parole, criminal record,
accomplice, conspiracy, attempt, strict liability, criminal negligence,
defendant, prosecution, charge, indictment, bail, remand,
capital punishment, rehabilitation, recidivism, penal code
```

---

## Domain 4: `administrative`

### Positive Keywords

```
administrative, administration, regulation, regulatory, licence, license,
permit, authorization, delegated legislation, statutory instrument,
subordinate legislation, by-law, ordinance (regulatory), rule-making,
government agency, statutory body, public authority, civil service,
public law, natural justice, audi alteram partem, nemo judex, bias,
ultra vires, procedural fairness, legitimate expectation, proportionality,
judicial review (administrative ground), reasonableness, Wednesbury,
public interest, expropriation, compulsory purchase, eminent domain,
planning, zoning, environmental regulation, tax, taxation, customs,
immigration (administrative), public procurement, state aid,
accountability, transparency, ombudsman, freedom of information
```

### Negative Keywords

```
- "administrative court" as court → procedure
- "tax fraud" → criminal
- "immigration crime" → criminal
- "public nuisance" → criminal
- "natural justice" (as constitutional due process) → constitutional
  Note: "natural justice" has a narrower administrative law sense in common law;
  in that narrow sense → administrative
```

### Ambiguity Resolution

| Border case       | Resolution                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "Judicial review" | →`constitutional` for rights-based review; `administrative` for legality/rationality review of administrative decisions. When ambiguous, prefer `constitutional`  |
| "Natural justice" | →`administrative` in the narrower procedural fairness sense (audi alteram partem, nemo judex); → `constitutional` when invoking broader constitutional due process |
| "Proportionality" | →`administrative` in administrative law context; → `constitutional` in rights limitation context                                                                   |
| "Taxation"        | Core tax law concepts →`administrative`; criminal tax evasion → `criminal`                                                                                         |
| "Immigration"     | Status rules →`administrative`; immigration offences → `criminal`                                                                                                  |
| "Regulation"      | Regulatory concepts →`administrative`; regulations as private law instruments → `civil`                                                                            |
| "Public official" | →`administrative`; bribery of official → `criminal`                                                                                                                |

### Typical Terms (canonical examples)

```
administrative law, regulation, licence, permit, ultra vires, judicial review,
natural justice, legitimate expectation, proportionality, Wednesbury reasonableness,
bias, audi alteram partem, nemo judex, statutory body, civil service,
delegated legislation, statutory instrument, compulsory purchase, planning permission,
expropriation, taxation, customs, ombudsman, freedom of information,
public procurement, accountability, transparency, public authority
```

---

## Domain 5: `international`

### Positive Keywords

```
international law, treaty, convention, covenant, protocol, charter,
bilateral, multilateral, ratification, accession, reservation, jus cogens,
erga omnes, pacta sunt servanda, sovereign immunity, diplomatic,
consular, ambassador, envoy, extraterritoriality, extradition,
asylum, refugee, stateless, nationality (international), human rights law,
humanitarian law, war crimes, crimes against humanity, genocide,
international tribunal, ICJ, ICC, arbitration (international), ICSID,
WTO, United Nations, UN, Security Council, General Assembly, WHO, ILO,
lex mercatoria, conflict of laws, private international law, choice of law,
forum non conveniens, comity, jurisdiction (international), recognition,
enforcement of foreign judgments, letters of request, mutual legal assistance
```

### Negative Keywords

```
- Domestic "arbitration" → background or civil
- "Consular" fees → background/administrative
- "UN" in organizational context without legal substance → background
```

### Ambiguity Resolution

| Border case                                    | Resolution                                                                                                                                     |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| "Extradition"                                  | →`international` (governs inter-state relations); the domestic extradition proceeding → `procedure`                                      |
| "Sovereign immunity"                           | →`international` (doctrine in inter-state relations); → `constitutional` only for domestic immunity of the Crown                         |
| "Arbitration"                                  | International commercial arbitration →`international`; domestic commercial arbitration → `civil`; labour arbitration → `labor_social` |
| "Human rights law"                             | Core international HR instruments (ICCPR, UDHR) →`international`; domestic constitutional rights → `constitutional`                      |
| "Conflict of laws / private international law" | →`international` (though private); separate tradition from public international law                                                         |
| "Recognition of foreign judgments"             | →`international` / `procedure` border; prefer `international` for the doctrine, `procedure` for the mechanism                         |

### Typical Terms (canonical examples)

```
treaty, convention, sovereign immunity, diplomatic privilege, extradition,
asylum, refugee, ratification, jus cogens, erga omnes, pacta sunt servanda,
arbitration (international), ICSID, ICJ, ICC, war crimes, genocide,
crimes against humanity, humanitarian law, UN Charter, General Assembly,
Security Council, conflict of laws, choice of law, forum non conveniens,
comity, letters of request, mutual legal assistance, lex mercatoria
```

---

## Domain 6: `labor_social`

### Positive Keywords

```
employment, employee, employer, worker, labour, labor, work*,
contract of employment, dismissal, redundancy, unfair dismissal,
wrongful dismissal, constructive dismissal, notice period,
discrimination (employment), harassment, sexual harassment,
equal opportunity, equal pay, minimum wage, overtime, working hours,
rest day, annual leave, maternity leave, paternity leave, sick leave,
pension, provident fund, social security, welfare, social insurance,
trade union, collective bargaining, strike, industrial action,
occupational safety, health and safety, disability, accessibility,
social protection, unemployment, worker's compensation, vicarious liability
```

### Negative Keywords

```
- "labour law" as a field label → keep as label for the domain
- "employment tribunal" → procedure (adjudicatory body)
- "discrimination" in constitutional context (equal protection) → constitutional
- "pension fund" as financial instrument → background/civil
```

### Ambiguity Resolution

| Border case             | Resolution                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| "Discrimination"        | →`labor_social` when employment-context; → `constitutional` when general equal protection principle           |
| "Vicarious liability"   | →`civil` for general tort context; → `labor_social` when specifically employer's liability for employee acts  |
| "Social security"       | →`labor_social`; not administrative despite government administration                                            |
| "Health and safety"     | Occupational H&S →`labor_social`; general public safety regulation → `administrative`                         |
| "Collective bargaining" | →`labor_social`; even though it also involves contract law                                                       |
| "Worker" vs. "employee" | Both →`labor_social`; the gig economy / independent contractor distinction may produce interesting false friends |

### Typical Terms (canonical examples)

```
employment contract, dismissal, redundancy, unfair dismissal, minimum wage,
discrimination, equal opportunity, trade union, collective bargaining,
strike, maternity leave, pension, social security, occupational safety,
wrongful dismissal, harassment, vicarious liability, worker's compensation,
annual leave, overtime, notice period, probationary period, social insurance,
disability rights, unemployment benefit, industrial action
```

---

## Domain 7: `procedure`

### Positive Keywords

```
jurisdiction, competence (court), standing, locus standi, ius standi,
appeal, appellate, cassation, first instance, final appeal,
evidence, admissibility, admissible, hearsay, privilege, burden of proof,
standard of proof, presumption, witness, examination, cross-examination,
affidavit, sworn statement, subpoena, summons, writ, pleading, claim,
counterclaim, interlocutory, injunction (procedural), discovery, disclosure,
limitation period, prescription, res judicata, issue estoppel, cause of action,
venue, forum, transfer, consolidation, class action, joinder,
judgment, enforcement, execution, contempt, stay of proceedings,
alternative dispute resolution, mediation, conciliation, arbitration (domestic),
court, tribunal, judge, magistrate (role), clerk, registrar,
in camera, ex parte, service of process, notice, leave (court)
```

### Negative Keywords

```
- "injunction" as a substantive remedy (in property/IP) → civil
- "habeas corpus" → constitutional (liberty guarantee, not mere procedure)
- "judicial review" → constitutional/administrative (not procedure)
- "arbitration" (international) → international
- "criminal procedure" terms → procedure (as a domain, not subcategory)
```

### Ambiguity Resolution

| Border case                        | Resolution                                                                                                                                    |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| "Limitation period / prescription" | →`procedure` (when the right is extinguished by time for procedural reasons); some jurisdictions treat as substantive — flag as ambiguous |
| "Res judicata"                     | →`procedure`                                                                                                                               |
| "Contempt of court"                | →`procedure` if civil enforcement mechanism; → `criminal` if criminal contempt with sanction                                            |
| "Injunction"                       | →`procedure` as a form of interlocutory relief; → `civil` when as final substantive remedy in property/tort                             |
| "Arbitration"                      | Domestic arbitration →`procedure` or `civil` depending on whether it is court-integrated; international → `international`             |
| "Mediation"                        | →`procedure` (dispute resolution mechanism)                                                                                                |
| "Magistrate"                       | →`procedure` (adjudicatory role); but as HK-specific institution → flag (see hk_specific_terms.md)                                        |
| "Forum non conveniens"             | →`international` (cross-border) or `procedure` (domestic); prefer `international`                                                      |

### Typical Terms (canonical examples)

```
jurisdiction, standing, locus standi, appeal, evidence, admissibility,
hearsay, burden of proof, standard of proof, presumption, affidavit,
subpoena, summons, writ, pleading, discovery, limitation period,
res judicata, issue estoppel, injunction, stay of proceedings,
in camera, ex parte, service of process, contempt of court,
class action, joinder, consolidation, mediation, arbitration (domestic),
judgment, enforcement, execution, tribunal
```

---

## Cross-Domain Disambiguation Summary

| Term / Concept    | Competing domains                                | Decision rule                                                                                                                                 |
| ----------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Judicial review   | constitutional / administrative / procedure      | Constitutional if rights-based; administrative if legality of administrative act; NOT procedure                                               |
| Natural justice   | constitutional / administrative                  | Administrative for narrow procedural fairness; constitutional for general due process                                                         |
| Arbitration       | civil / international / procedure / labor_social | By context: international →`international`; domestic commercial → `civil`; labour → `labor_social`; generic process → `procedure` |
| Discrimination    | constitutional / labor_social                    | Constitutional for general equality principle; labor_social for employment context                                                            |
| Extradition       | international / procedure                        | Doctrine and treaty →`international`; domestic proceeding → `procedure`                                                                 |
| Bail              | criminal / civil                                 | Criminal pretrial →`criminal`; civil debt context → `civil` (rare)                                                                      |
| Contempt of court | criminal / procedure                             | Criminal contempt →`criminal`; civil enforcement → `procedure`                                                                          |
| Proportionality   | constitutional / administrative                  | Rights limitation →`constitutional`; administrative decision review → `administrative`                                                  |
| Damages           | civil / criminal                                 | Civil remedy →`civil`; criminal fine/sentence → `criminal`                                                                              |
| Strict liability  | civil / criminal                                 | Strict liability tort →`civil`; strict liability offence → `criminal`                                                                   |

---

## False Friend Candidates

*(High scientific value — see §3.2 Normative decompositions)*

These are terms where the EN and ZH embedding neighborhoods are predicted to diverge
most significantly, revealing cross-traditional semantic differences.

| Term (EN)          | ZH equivalent | Predicted divergence                                                                    |
| ------------------ | ------------- | --------------------------------------------------------------------------------------- |
| `contract`       | 合同 / 契約   | EN: agreement → consideration → autonomy; ZH: relationship → trust → duty           |
| `law`            | 法 / 法律     | EN: rule → justice → legitimacy; ZH: order → state → sanction                       |
| `rights`         | 權利          | EN: individual → liberty → protection; ZH: collective → responsibility → harmony    |
| `equity`         | 衡平法        | EN: fairness → conscience → Chancery; ZH: may not have strong embedding anchor        |
| `person` (legal) | 人 / 法人     | EN: natural/legal person; ZH: 自然人/法人 distinction present but grounded differently  |
| `sovereignty`    | 主權          | EN: popular sovereignty → constitution; ZH: state sovereignty → territorial integrity |
| `punishment`     | 懲罰          | EN: deterrence / retribution; ZH: education / rehabilitation (Confucian tradition)      |
| `judge`          | 法官          | EN: independent adjudicator; ZH: state officer with loyalty duty                        |
| `property`       | 財產          | EN: absolute individual entitlement; ZH: social function of property (socialist roots)  |
| `mediation`      | 調解          | EN: alternative to litigation; ZH: preferred primary mechanism (Confucian harmony)      |
| `precedent`      | 先例          | EN: binding (stare decisis); ZH: non-binding (civil law tradition)                      |
| `due process`    | 正當程序      | EN: procedural + substantive; ZH: primarily procedural                                  |

---

## Corpus-Validated Exclusion Rules

*(Added v1.1 — derived from analysis of doj_filtered.json, 9,387 terms)*

### Rule 1: Polysemy Threshold

Terms with **>50 zh_variants** in doj_filtered.json are systematically polysemous across
the full HK legal corpus and must be excluded from the core domain corpus. They lack a
stable semantic anchor for embedding. Assign to background only.

Top high-polysemy terms confirmed in corpus (zh_variants count):

| Term         | ZH variants | Treatment                                                                                             |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------- |
| `order`    | 375         | background                                                                                            |
| `right`    | 352         | background                                                                                            |
| `law`      | 302         | background                                                                                            |
| `evidence` | 278         | background (but `evidence` as concept → procedure)                                                 |
| `interest` | 264         | background                                                                                            |
| `person`   | 261         | background (but `legal person` / `natural person` → core/civil)                                  |
| `notice`   | 237         | background                                                                                            |
| `public`   | 230         | background                                                                                            |
| `contract` | 195         | EXCEPTION: core/civil — use compound forms only (e.g.,`contract of sale`, `employment contract`) |
| `court`    | 188         | background (but specific court concepts → procedure)                                                 |
| `offence`  | 186         | background (but specific offence types → criminal)                                                   |
| `claim`    | 185         | background                                                                                            |
| `property` | 157         | EXCEPTION: core/civil — the bare noun is too broad; use in compound form                             |
| `charge`   | 153         | background (resolved per polysemy flags below)                                                        |
| `act`      | 168         | background                                                                                            |

**Exception rule**: A high-polysemy bare noun MAY appear in the core corpus as the
canonical label for a specific compound concept (e.g., "consideration" as the contract
law concept, NOT "consideration" as a bare noun). Curator must verify zh_canonical
maps to the intended sense.

### Rule 2: zh_canonical Artifact

The `zh_canonical` field in doj_filtered.json is the **first-occurring** ZH variant,
which is NOT necessarily the most semantically central translation. For embedding purposes,
the curator must select the most semantically central ZH variant, typically the shortest
and most widely recognized term in Mainland ZH legal discourse.

Examples of zh_canonical artifacts to correct:

| Term           | zh_canonical (DOJ)                         | Preferred ZH for embedding |
| -------------- | ------------------------------------------ | -------------------------- |
| `contract`   | 外判合約 (outsource contract)              | 合同                       |
| `law`        | 法律上的錯誤 (error of law)                | 法律 or 法                 |
| `punishment` | 不相稱的懲罰 (disproportionate punishment) | 懲罰                       |
| `judge`      | 到法官席前 (come before the judge)         | 法官                       |
| `rights`     | 供股 (rights issue, financial)             | 權利                       |

### Rule 3: Division Signals (Secondary Heuristic)

DOJ division codes provide a secondary signal for domain assignment (not sufficient alone,
but useful for disambiguation when headword keyword is ambiguous):

| Division                             | Domain signal                         |
| ------------------------------------ | ------------------------------------- |
| `CD` (Civil Division)              | → civil, procedure (civil)           |
| `PD` (Prosecutions Division)       | → criminal, procedure (criminal)     |
| `ILD` (International Law Division) | → international                      |
| `LDD` (Law Drafting Division)      | → broad; use keyword rules primarily |
| `LRC` (Law Reform Commission)      | → broad; use keyword rules primarily |
| `LPD` (Legal Policy Division)      | → administrative (secondary signal)  |

Use division code ONLY as a tiebreaker when keyword rules produce ambiguity.

---

## Polysemy Flags

Terms with multiple legal senses that must be disambiguated at curation time:

| Term           | Senses                                                                                              | Flag                                                  |
| -------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `equity`     | (1) Chancery/equitable law; (2) company shares                                                      | (1) → civil; (2) → background                       |
| `trust`      | (1) fiduciary trust; (2) commercial trust company                                                   | (1) → civil; (2) → background                       |
| `act`        | (1) statute/legislation; (2) actus reus element                                                     | (1) → constitutional/administrative; (2) → criminal |
| `charge`     | (1) criminal charge; (2) security interest on property                                              | (1) → criminal; (2) → civil                         |
| `assignment` | (1) contract law transfer; (2) procedural assignment                                                | (1) → civil; (2) → procedure                        |
| `order`      | (1) court order; (2) executive/administrative order                                                 | (1) → procedure; (2) → administrative               |
| `discharge`  | (1) discharge from contract; (2) discharge from custody                                             | (1) → civil; (2) → criminal                         |
| `notice`     | (1) statutory notice (administrative); (2) notice to quit (civil); (3) notice of motion (procedure) | Assign by context                                     |

---

## Background Terms (not core, but corpus-enriching)

Terms that provide semantic context for k-NN neighborhood analysis but should not be
assigned to a core domain:

- **High-polysemy bare nouns** (>50 zh_variants): `order`, `right`, `law`, `evidence`,
  `interest`, `person`, `notice`, `public`, `court`, `offence`, `claim`, `property`,
  `charge`, `act`, `agreement`, `action`, `legal`, `costs`, `payment`, `case`
  (see Corpus-Validated Exclusion Rules above)
- Legal roles without domain signal: `judge`, `advocate`, `solicitor`, `clerk`,
  `registrar` (as bare role nouns, not doctrine concepts)
- Procedural nouns without substantive content: `form`, `schedule`, `appendix`,
  `section`, `subsection`, `clause`, `subclause`
- Commercial/financial terms without clear legal tradition signal: `account`, `audit`,
  `accounting`, `securities`, `bond`, `debenture`
- Organisational terms: `committee`, `board`, `council`, `authority` (without specific
  legal doctrine attached)
