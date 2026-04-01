# Trace: Lens I — Relational distance structure

**Thesis chapter(s)**: Ch.3 §3.1 — The geometry of a legal tradition (macro)
**Date**: 2026-02-21
**Status**: in progress

---

## Context

Lens I asks whether legal embedding spaces have non-random relational structure
and whether that structure is shared across legal traditions. It operationalizes
the second-order isomorphism concept introduced in §1.4.

### Analysis map

| Analysis | Section | Terms used | Method |
|---|---|---|---|
| Background term domain assignment | §3.1.1 | background (8975) | k-NN majority vote |
| Intra vs inter-domain distances | §3.1.1 | core (397) | Mann-Whitney U |
| Legal vs control signal | §3.1.1 | core (397) + control (100) | Mann-Whitney U |
| Domain topology K×K | §3.1.2 | core (397) | inter-domain distance matrix |
| Within-WEIRD RSA robustness | §3.1.4 | core (397) | RSA + Mantel + block bootstrap |
| Cross-tradition RSA | §3.1.4 | core (397) | RSA + Mantel + block bootstrap |

---

## Decision log

### D1 — Distance metric

**Options considered**:
- Option A: cosine distance `1 - cosine_similarity` (pro: scale-invariant, standard
  in NLP; contro: nessuno rilevante per questo contesto)
- Option B: euclidean distance (pro: intuitiva; contro: dipende dalla magnitudine,
  non confrontabile tra modelli di dimensione diversa)

**Decision**: Option A — cosine distance

**Rationale**: i vettori sono L2-normalizzati, quindi cosine distance = `1 - u·v`.
Indipendente dalla dimensione del modello (768 vs 1024), confrontabile tra
tradizioni.

**Thesis text implication**: → §2.4 "The statistical toolkit". Giustifica perché
la distanza coseno è la metrica naturale per embedding normalizzati e permette
confronti tra modelli di architettura diversa.

---

### D2 — Similarity measure for RDM comparison (RSA)

**Options considered**:
- Option A: Spearman ρ sui triangoli superiori (pro: rank-based, robusto a scale
  diverse, standard in letteratura RSA; contro: perde informazione di magnitudine)
- Option B: Pearson r (pro: misura la proporzione lineare; contro: assume
  normalità, sensibile agli outlier, non appropriato per scale diverse)

**Decision**: Option A — Spearman ρ

**Rationale**: le distanze coseno di modelli diversi hanno distribuzioni diverse.
Spearman cattura la concordanza d'ordinamento indipendentemente dalla scala.
Standard nella letteratura RSA (Kriegeskorte et al. 2008).

**Thesis text implication**: → §2.4 "The statistical toolkit". Motiva la scelta
rank-based rispetto alla correlazione lineare per confronti inter-modello.

---

### D3 — Statistical test for RSA significance

**Options considered**:
- Option A: Mantel test (permutation test su matrici) (pro: rispetta la
  dipendenza strutturale delle RDMs, non-parametrico; contro: computazionalmente
  più costoso del t-test)
- Option B: t-test su Spearman ρ (pro: immediato; contro: assume indipendenza
  delle N(N-1)/2 coppie — assunzione violata, produce p-value gonfiati)

**Decision**: Option A — Mantel test con B=10000 permutazioni, seed=42

**Rationale**: le entrate del triangolo superiore dell'RDM non sono i.i.d. —
ogni termine appare in N-1 coppie. Il t-test con ~78k gradi di libertà produce
falsi positivi sistematici. Il Mantel test rispetta questa struttura.
B=10000 è lo standard di riferimento di Nili et al. (2014) e garantisce che il
minimo p-value rappresentabile sia 1/10000=0.0001 — sufficiente per affermare
p<0.001 con piena credibilità in un contesto peer-reviewed. Nessun revisore può
obiettare sulla stabilità della stima.

**Thesis text implication**: → §2.4 "The statistical toolkit: permutation inference".
Centrale per la credibilità statistica dell'intera analisi RSA.

---

### D4 — Confidence interval for RSA

**Options considered**:
- Option A: block bootstrap a livello di termini, B=1000 (pro: rispetta la
  dipendenza, metodo standard per RSA — Nili et al. 2014; contro: nessuno
  rilevante)
- Option B: bootstrap sulle coppie (pro: più semplice; contro: ignora la
  dipendenza, CI artificialmente stretti)

**Decision**: Option A — block bootstrap sui termini, B=10000, seed=42

**Rationale**: la dipendenza tra le coppie nasce a livello di termine. Ricampionare
i termini (non le coppie) è l'unità corretta per il bootstrap. Ref: Nili et al.
(2014) PLoS Computational Biology 10(4): e1003553. B=10000 allineato con D3
(Mantel test) per coerenza del parametro di Monte Carlo attraverso tutta la tesi.

**Thesis text implication**: → §2.4 "The statistical toolkit: block bootstrap".
Questo metodo diventa la giustificazione formale per tutti i CI dell'intera tesi.

---

### D5 — Statistical test for §3.1.1 (domain signal + legal vs control)

**Options considered**:
- Option A: Mann-Whitney U (pro: non-parametrico, non assume normalità, adatto
  per distribuzioni asimmetriche di distanze coseno; contro: nessuno rilevante)
- Option B: t-test (pro: familiare; contro: assume normalità, violata per
  distanze coseno in spazi ad alta dimensione)

**Decision**: Option A — Mann-Whitney U, `alternative='less'`, effect size:
rank-biserial correlation r

**Rationale**: le distanze coseno non sono normalmente distribuite. Mann-Whitney
testa la differenza di locazione (mediana) senza assunzioni distribuzionali.
L'effect size r = (concordanti - discordanti) / totale dà una misura leggibile
della separazione tra le due distribuzioni.

**Thesis text implication**: → §3.1.1. Il p-value risponde a "il segnale è
statisticamente significativo?", l'effect size r risponde a "quanto è grande?".
Entrambi necessari per evitare lo stargazing criticato in §2.4.

---

### D_BG1 — Algoritmo di assegnamento per termini background

**Options considered**:
- Option A: centroide di dominio — assegna al dominio il cui centroide (media
  dei vettori core del dominio) è più vicino (pro: veloce, un solo confronto
  per termine; contro: il centroide è un punto astratto, non interpretabile;
  perde la struttura interna del dominio)
- Option B: k-NN majority vote — trova i k core term più simili, majority vote
  sul dominio (pro: interpretabile: il giurista vede i termini specifici che
  hanno determinato l'assegnamento; robusto alla forma non-globulare dei cluster;
  contro: leggermente più costoso computazionalmente)

**Decision**: Option B — k-NN majority vote

**Rationale**: l'interpretabilità è fondamentale. Lo strumento di revisione
giuridica (§3.1.1 + Ch.4 Quod Numeri Tacent) richiede che il giurista veda perché
un termine è stato assegnato a un dominio. "I tuoi 7 vicini più prossimi sono
[X, Y, Z...], tutti civil" è un ragionamento verificabile. Il centroide non lo è.
La differenza di costo computazionale è trascurabile (8975 query su 397 vettori).

**Thesis text implication**: → §3.1.1 "Domain signal: do legal categories have
geometric reality?" — l'assegnamento k-NN diventa il test empirico del segnale
su dati held-out (i background terms). → §4.1 (Ch.4 Quod Numeri Tacent): il
giurista completa l'interpretazione dove il numero fornisce la misura ma non il
giudizio.

---

### D_BG2 — Valore di k per il majority vote

**Options considered**:
- Option A: k=3 — locale, netto, ma instabile (un singolo vicino anomalo sposta
  il risultato; con 7 domini possibile 1+1+1)
- Option B: k=7 — dispari, evita i pareggi perfetti; sufficientemente locale
  (~1.75% dei 397 core terms); con 7 domini la maggioranza assoluta (≥4/7)
  indica segnale forte, confidence bassa (<4/7) indica ambiguità genuina
- Option C: k=15 — più stabile ma troppo globale; rischia di incorporare
  domini lontani, perdendo la struttura locale

**Decision**: Option B — k=7

**Rationale**: k=7 è dispari (nessun pareggio a 2 classi), ragionevolmente
locale (1.75% del pool core), e produce una scala di confidence naturale:
7/7=1.0 (unanimità), 4/7≈0.57 (maggioranza minima), <4/7 (ambiguità). I termini
con confidence bassa sono i candidati più interessanti per la revisione giuridica
— il modello geometrico è incerto, il giurista decide.

**Nota**: con 7 domini e k=7, la distribuzione peggiore è 2+2+1+1+1 (nessuna
maggioranza assoluta). In questo caso il codice assegna al più frequente (2 voti)
e segnala `confidence=2/7≈0.29` come "alta ambiguità". La revisione giuridica
è obbligatoria per questi termini.

**Thesis text implication**: → §3.1.1. La scelta di k=7 e la soglia di confidence
4/7 diventano i parametri operativi del signal test. → §2.4: la sensibilità al
k è una verifica di robustezza (k=5 e k=9 come sensitivity check).

---

### D6 — Multiple comparison correction

**Decision**: Holm-Bonferroni step-down correction applied to all 15 Mantel
test p-values (3 within-WEIRD + 3 within-Sinic + 9 cross-tradition).

**Rationale**: With 15 simultaneous tests at alpha=0.05, the family-wise
error rate (FWER) would be inflated without correction. Holm-Bonferroni is
uniformly more powerful than Bonferroni while still controlling FWER. In
practice, all 15 uncorrected p-values are at the permutation floor
(≤ 0.001), so all remain significant after correction. The correction is
applied for methodological completeness.

**Thesis text implication**: → §3.1.4 "P-values are corrected for multiple
comparisons using the Holm-Bonferroni procedure across all 15 model pairs.
All corrected p-values remain below 0.015."

---

### D7 — Phipson & Smyth p-value formula

**Decision**: Permutation p-values computed as (b+1)/(m+1) following
Phipson & Smyth (2010), where b = number of null values ≥ observed,
m = number of permutations. This replaces the earlier approximation
max(p_raw, 1/m).

**Rationale**: The (b+1)/(m+1) formula is the exact implementation
recommended by Phipson & Smyth. It is slightly more conservative (no
p-value is ever exactly zero) and is the standard cited in the literature.

---

## Open questions

- Sensitivity check su k=5 e k=9: pianificato come robustezza in §2.4
- Concordanza tra i 3 modelli WEIRD sull'assegnamento: da includere in §3.1.4

---

## References

- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity
  analysis — connecting the branches of systems neuroscience. *Frontiers in
  Systems Neuroscience*, 2, 4.
- Nili, H., et al. (2014). A toolbox for representational similarity analysis.
  *PLoS Computational Biology*, 10(4), e1003553.
- Mantel, N. (1967). The detection of disease clustering and a generalized
  regression approach. *Cancer Research*, 27(2), 209–220.
- Phipson, B., & Smyth, G.K. (2010). Permutation P-values should never be zero:
  calculating exact P-values when permutations are randomly drawn. *Statistical
  Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Mann, H.B., & Whitney, D.R. (1947). On a test of whether one of two random
  variables is stochastically larger than the other. *Annals of Mathematical
  Statistics*, 18(1), 50–60.
