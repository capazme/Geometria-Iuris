# Geometria Iuris — Panoramica tecnica

*Documento preparato per il Prof. Ing. D'Acquisto — Correlatore*

## 1. Domanda di ricerca

La tesi si chiede se i modelli di sentence-embedding addestrati su comunità linguistiche diverse codifichino strutture geometriche misurabili e diverse nella rappresentazione dei concetti giuridici. La pretesa è epistemologica: gli spazi di embedding possono fungere da strumenti di misura per il significato giuridico, a condizione che la disciplina inferenziale sia rigorosa.

## 2. Design sperimentale

**Architettura:** design simmetrico 3 × 3. Tre modelli addestrati su corpora inglesi (tradizione "WEIRD") vengono confrontati con tre modelli addestrati su corpora cinesi (tradizione "Sinic"). Ogni modello è monolingue per scelta progettuale: non si utilizzano modelli multilingui, per evitare di imporre un allineamento cross-linguistico come obiettivo di addestramento.

| Slot | WEIRD (inglese) | Sinic (cinese) | Dim |
|------|----------------|-----------------|-----|
| Architettura condivisa | BAAI/bge-large-en-v1.5 | BAAI/bge-large-zh-v1.5 | 1024 |
| STS-oriented | intfloat/e5-large-v2 | GanymedeNil/text2vec-large-chinese | 1024 |
| Domain-specific | freelawproject/modernbert-embed-base | DMetaSoul/Dmeta-embedding-zh | 768 |

I tre slot controllano per effetti architetturali: lo slot 1 condivide la stessa architettura base (BGE) tra le due tradizioni; lo slot 2 impiega modelli STS sviluppati indipendentemente; lo slot 3 introduce il fine-tuning di dominio (FreeLaw per l'inglese; per il cinese non è disponibile un modello di embedding legale monolingue).

**Dataset:** 9.472 termini tratti dal Glossario Giuridico Bilingue del Dipartimento di Giustizia di Hong Kong. Nessuna traduzione automatica: il glossario fornisce gli abbinamenti bilingui ufficiali.

| Livello | Termini | Ruolo |
|---------|---------|-------|
| Core | 397 | Termini di analisi primaria, assegnati a 7 rami del diritto |
| Background | 8.975 | Vocabolario giuridico completo, pool di vicinato |
| Controllo | 100 | Swadesh-100, termini non giuridici di uso quotidiano (baseline) |

I 7 rami: penale (66 termini), civile (136), costituzionale (48), internazionale (52), processuale (53), lavoro/sociale (30), amministrativo (12).

**Struttura dei confronti:** 15 coppie di modelli totali: 3 within-WEIRD (EN vs EN), 3 within-Sinic (ZH vs ZH), 9 cross-tradition (EN vs ZH). Le coppie within-tradition fungono da baseline di affidabilità; le coppie cross-tradition sono l'oggetto primario di analisi.

## 3. Quattro analisi ("lenti")

Ogni lente esamina lo stesso dataset e lo stesso pannello di modelli da una prospettiva geometrica diversa.

### 3.1 Lente I — Struttura relazionale delle distanze (§3.1)

**Oggetto:** per ciascun modello si calcola una matrice 397×397 di dissimilarità relazionale (RDM). Ogni cella registra la distanza coseno (1 meno il prodotto scalare dei vettori L2-normalizzati) tra due termini giuridici.

**Metodi:**

| Test | Metodo | Cosa misura |
|------|--------|-------------|
| Intra vs. inter-dominio | Mann-Whitney U, unilaterale, r rank-biserial | Se le coppie di termini dello stesso ramo hanno distanze minori di quelle tra rami diversi |
| Legale vs. controllo | Mann-Whitney U, unilaterale, r rank-biserial | Se i termini giuridici sono più compatti dei termini non giuridici di controllo |
| RSA within-tradition | Spearman ρ sui triangoli superiori delle RDM (78.006 valori per coppia) | Accordo sulla struttura relazionale tra modelli della stessa tradizione |
| RSA cross-tradition | Spearman ρ + test di Mantel a permutazione | Accordo sulla struttura relazionale tra modelli di tradizioni diverse |

**Apparato statistico:**

- *r rank-biserial:* effect size per Mann-Whitney U. r = 1 − 2U/(n₁·n₂). Range [−1, +1]; r positivo = il primo gruppo ha valori sistematicamente minori.
- *Test di Mantel:* significatività basata su permutazione per correlazioni tra matrici (B=10.000). Le righe e colonne di una RDM vengono permutate congiuntamente; ρ è ricalcolato ad ogni permutazione. Il p-value è limitato inferiormente a 1/B secondo Phipson & Smyth (2010).
- *Block bootstrap CI:* intervalli di confidenza per Spearman ρ (B=1.000). I termini (non le coppie) sono ricampionati con rimpiazzo; le sotto-RDM vengono estratte e ρ ricalcolato. Questo rispetta la struttura di dipendenza intrinseca nelle matrici di distanza (Nili et al. 2014).

### 3.2 Lente III — Stratigrafia dei layer (§3.1.3)

**Oggetto:** i modelli transformer sono costituiti da layer successivi (da 12 a 24 a seconda dell'architettura). Per ciascun modello, gli hidden state a ogni layer intermedio vengono estratti. La RDM è ricalcolata ad ogni layer, consentendo un'analisi della struttura geometrica in funzione della profondità.

**Metriche:**

- *Drift coseno:* d(t,ℓ) = 1 − cos(h_t^ℓ, h_t^{ℓ+1}). Misura quanto la rappresentazione di ciascun termine cambia tra layer consecutivi.
- *Instabilità Jaccard del vicinato:* J(t,ℓ) = 1 − |kNN_ℓ ∩ kNN_{ℓ+1}| / |kNN_ℓ ∪ kNN_{ℓ+1}|, con k=7. Misura quanto cambia l'identità dei vicini più prossimi di un termine tra layer consecutivi.
- *Segnale di dominio per layer:* r rank-biserial r(ℓ) sulla RDM al layer ℓ, con la stessa partizione intra/inter-dominio della Lente I. Traccia quando la struttura di dominio compare nell'elaborazione del modello.
- *Convergenza RSA:* ρ(ℓ) = Spearman(utri(RDM_ℓ), utri(RDM_finale)). Misura quanto la struttura delle distanze al layer ℓ è simile all'output finale.
- *Neighbourhood Trajectory Analysis (NTA):* per una selezione di termini polisemici, i k=7 vicini esatti vengono tracciati attraverso layer campionati, registrando ingressi e uscite.

**Nota implementativa:** gli hidden state sono estratti tramite Hugging Face `output_hidden_states=True`, pooled (CLS o mean a seconda dell'architettura) e L2-normalizzati. Tutte le estrazioni girano su CPU per garantire il determinismo (Apple Silicon MPS produce risultati non deterministici per alcuni modelli cinesi quando cambia la composizione del batch).

### 3.3 Lente V — Vicinati semantici (§3.2)

**Oggetto:** per ciascuno dei 397 termini core, i k=15 vicini più prossimi sono identificati nell'intero pool di 9.472 termini tramite similarità coseno. L'indice di Jaccard J = |A ∩ B| / |A ∪ B| misura la sovrapposizione tra gli insiemi di vicini di due modelli.

**Metodi:**

- Jaccard per-termine J(t) per ciascuna delle 15 coppie di modelli; aggregazione per media.
- Mann-Whitney U per confrontare il Jaccard medio cross-tradition vs. within-tradition.
- Test di Kruskal-Wallis H: se l'appartenenza a un ramo del diritto è associata a diversi livelli di divergenza di vicinato.
- Mann-Whitney a coppie con correzione di Bonferroni per i confronti post-hoc.
- Ranking dei "false friends": termini ordinati per divergenza = 1 − J̄_cross.

### 3.4 Lente II — Tassonomia gerarchica (§4.4, Horizons)

**Oggetto:** i 397 termini core vengono raggruppati mediante clustering gerarchico agglomerativo sulla RDM di ciascun modello. I dendrogrammi risultanti rappresentano la tassonomia implicita del vocabolario giuridico secondo ciascun modello: quali concetti vengono uniti per primi (massima prossimità) e quali restano separati fino agli ultimi livelli della gerarchia.

**Metodi previsti:**

- *Clustering gerarchico agglomerativo:* a partire dalla RDM 397×397, con criterio di linkage da definire (Ward, average, complete). Il dendrogramma viene tagliato a k livelli crescenti.
- *Indice di Fowlkes-Mallows (FM):* per ogni livello di taglio k, FM(k) = TP / √((TP+FP)(TP+FN)) misura la concordanza tra due partizioni. FM=1 indica partizioni identiche; FM=0 indica concordanza casuale.
- *Confronto cross-tradition:* per ogni k, l'indice FM è calcolato tra il dendrogramma di un modello WEIRD e quello di un modello Sinic. La curva FM(k) mostra a quale granularità tassonomica le due tradizioni concordano e a quale divergono.
- *Test a permutazione:* per valutare se il FM osservato è significativamente superiore al caso, le etichette di cluster vengono permutate e FM è ricalcolato (B da definire).

**Stato:** pianificata nell'indice (§4.4 Horizons), non ancora implementata. L'analisi non è bloccante per i capitoli sperimentali principali (§3.1–§3.3) ed è collocata nella sezione "orizzonti" del capitolo conclusivo. La decisione di implementarla o differirla dipenderà dal tempo disponibile e dal valore aggiunto rispetto alle quattro lenti già completate.

### 3.5 Lente IV — Proiezione su assi valoriali (§3.3)

**Oggetto:** tre assi concettuali sono costruiti con il metodo del vettore-differenza di Kozlowski (2019): *individuale/collettivo*, *diritti/doveri*, *pubblico/privato*. Ogni asse è definito da 10 coppie di antonimi. I modelli WEIRD usano coppie inglesi; i modelli Sinic usano coppie cinesi (nessuna traduzione, costruzione indipendente). La direzione di ciascun asse = media L2-normalizzata dei 10 vettori-differenza.

I 397 termini core sono proiettati su ciascun asse tramite similarità coseno con il vettore-asse.

**Metodi:**

- *Sanity check:* per ogni coppia di antonimi, il membro positivo deve proiettarsi più in alto del membro negativo. Pass rate = corretti / (2 × n_coppie).
- *Ortogonalità inter-asse:* similarità coseno tra i tre vettori-asse. Valori prossimi a 0 indicano dimensioni indipendenti.
- *Allineamento cross-modello:* Spearman ρ tra i 397 punteggi di proiezione di due modelli su un dato asse.
- *Row-resample bootstrap CI:* B=10.000. I termini sono ricampionati con rimpiazzo; ρ è ricalcolato ad ogni estrazione. I percentili 2,5 e 97,5 formano l'intervallo di confidenza al 95%. Si usa il row-resample (non il block bootstrap) perché le osservazioni (punteggi per-termine) sono indipendenti, a differenza delle distanze pairwise della Lente I.
- *Kruskal-Wallis H:* confronta la distribuzione del ρ cross-tradition tra i tre assi.

## 4. Riepilogo dei metodi statistici

| Metodo | Usato in | Scopo | Riferimento |
|--------|----------|-------|-------------|
| Mann-Whitney U + r rank-biserial | Lenti I, III, IV, V | Confronto non parametrico tra due gruppi | scipy.stats.mannwhitneyu |
| Test di Mantel a permutazione | Lente I | Significatività della correlazione tra matrici di distanza | Implementazione custom; B=10.000 |
| Block bootstrap (per-termine) | Lente I | IC per correlazioni tra RDM, rispettando la dipendenza intra-termine | Nili et al., PLoS Comp. Bio., 2014 |
| Row-resample bootstrap | Lente IV | IC per Spearman ρ su osservazioni indipendenti | B=10.000 |
| Kruskal-Wallis H + Bonferroni | Lenti IV, V | Confronto multi-gruppo + post-hoc | scipy.stats.kruskal |
| Bound di Phipson & Smyth | Lenti I, V | Evita p-value nulli nei test a permutazione | Phipson & Smyth, Stat. Appl. Genet. Mol. Biol., 2010 |
| Vettore-differenza di Kozlowski | Lente IV | Costruzione di assi da coppie di antonimi | Kozlowski et al., Am. Sociol. Rev., 2019 |
| RSA (Kriegeskorte) | Lenti I, III | Confronto tra spazi tramite isomorfismo di secondo ordine | Kriegeskorte et al., Front. Syst. Neurosci., 2008 |
| Clustering gerarchico + Fowlkes-Mallows | Lente II | Concordanza tassonomica cross-tradition a granularità variabile | Fowlkes & Mallows, J. Am. Stat. Assoc., 1983 |

## 5. Stato attuale

| Componente | Stato |
|------------|-------|
| Costruzione dataset | Completo |
| Precomputo embedding (6 modelli × 9.472 termini) | Completo |
| Lente I — Struttura relazionale (§3.1) | Completo |
| Lente III — Stratigrafia dei layer (§3.1.3) | Completo |
| Lente IV — Assi valoriali (§3.3) | Completo |
| Lente V — Vicinati semantici (§3.2) | Completo |
| Lente II — Tassonomia gerarchica (§4.4 Horizons) | Pianificata, non ancora implementata |
| Capitolo 1 — Fondamenti teorici | Prima bozza completa (~10.000 parole) |
| Capitoli 2–4 — Redazione | Non ancora iniziata |

## 6. Domande aperte per la validazione metodologica

1. **Pseudo-replicazione nei confronti RSA.** I 9 valori ρ cross-tradition provengono da una griglia 3×3 dove ciascun modello compare in 3 coppie. La dimensione campionaria effettiva è inferiore a 9. Il confronto Mann-Whitney (within vs. cross) è difendibile, oppure sarebbe preferibile un approccio a effetti misti?

2. **Comparazioni multiple.** La correzione di Bonferroni è usata per i test post-hoc a coppie. È appropriata dato il numero ridotto di gruppi (7 domini, 3 assi), oppure sarebbe più adeguata la procedura FDR di Benjamini-Hochberg?

3. **Block bootstrap vs. row-resample bootstrap.** Il block bootstrap (Nili et al. 2014) è usato per le correlazioni tra RDM; il row-resample per le proiezioni per-termine. La distinzione è correttamente motivata dalla diversa struttura di dipendenza?

4. **Determinismo nell'estrazione dei layer.** MPS (Apple Silicon GPU) produce risultati non deterministici per alcuni modelli cinesi quando cambia la composizione del batch. Tutti i risultati sono stati calcolati su CPU. È sufficiente, oppure sarebbe opportuno documentare un protocollo formale di riproducibilità?

5. **Scelta di k.** k=7 per l'analisi di vicinato tra layer, k=15 per il confronto di vicinato cross-modello. I valori sono scelti in base al rapporto con la dimensione del pool (~2%). È necessaria un'analisi di sensitività al variare di k?

6. **Convenzioni per l'effect size.** I valori di r rank-biserial nel test del segnale di dominio oscillano tra +0,23 e +0,30. I benchmark convenzionali (Cohen) li classificano come piccoli-medi. Queste convenzioni sono appropriate per questo tipo di dati, oppure andrebbero sviluppati benchmark specifici per il dominio?

7. **Granularità del test a permutazione.** Tutti i 15 p-value RSA raggiungono il floor della permutazione (0,0001 a B=10.000). Sarebbe opportuno aumentare B per ottenere p-value più differenziati, oppure il floor è di per sé informativo?
