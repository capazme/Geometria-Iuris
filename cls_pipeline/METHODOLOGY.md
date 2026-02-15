# CLS Pipeline — Scelte Metodologiche e Conoscenze Consolidate

> Documento di riferimento per le decisioni metodologiche della pipeline.
> Ogni sezione riporta: scelta, motivazione, riferimento bibliografico.

---

## 1. Il problema del confound modello/cultura

### Diagnosi
Un esperimento che confronta UN modello WEIRD (es. BGE-EN) con UN modello Sinic
(es. BGE-ZH) non può distinguere tra:
- **Effetto culturale** (ciò che vogliamo misurare): differenze nelle strutture
  semantiche che riflettono tradizioni giuridiche diverse.
- **Effetto architetturale** (confound): differenze dovute all'architettura del
  modello, ai dati di training, alla dimensione dell'embedding, ecc.

### Soluzione: multi-model robustness
Si eseguono gli stessi esperimenti su **N coppie di modelli** (prodotto cartesiano
WEIRD × Sinic). Se l'effetto è consistente tra tutte le coppie, non può essere
attribuito a un singolo modello.

- **3 WEIRD**: E5-large-v2, BGE-EN-v1.5, Nomic-embed-v1.5
- **3 Sinic**: BGE-ZH-v1.5, Text2vec-chinese, SBERT-chinese-nli-v2
- **9 coppie** = prodotto cartesiano

### Cosa aggregare
Il multi-model **non è un nuovo test di ipotesi**. È una dimostrazione di
**robustezza**. Le metriche chiave sono:
- **Media** della statistica tra le 9 coppie
- **Deviazione standard** (bassa std = effetto robusto)
- **Range** [min, max] (il caso peggiore è ancora significativo?)
- **Coefficiente di variazione** (CV = std/media)

Non serve correzione per test multipli (Bonferroni/FDR) perché non stiamo testando
9 ipotesi diverse — stiamo verificando la stessa ipotesi su 9 realizzazioni
indipendenti.

Rif.: Liang et al. (2022) "Probing language model representations with
multi-model agreement", EMNLP.

---

## 2. Esperimento 1: RSA (Representational Similarity Analysis)

### Cosa misura
Confronto di **secondo ordine**: non allinea i vettori direttamente, ma confronta
le *geometrie interne* dei due spazi. Se due concetti sono "vicini" nello spazio
WEIRD, lo sono anche nello spazio Sinic?

### Pipeline
1. **RDM** (Representational Dissimilarity Matrix): matrice N×N di distanze coseno
   intra-spazio. Una per WEIRD, una per Sinic.
2. **Spearman r**: correlazione di rango tra i triangoli superiori delle due RDM.
   Spearman (non Pearson) perché non assume linearità — ci interessa la
   preservazione dell'*ordine* delle distanze.
3. **Mantel test**: test di permutazione per significatività. Permuta righe e
   colonne *simultaneamente* per preservare la simmetria della matrice.
4. **Block bootstrap CI**: intervallo di confidenza ricampionando *termini interi*
   (non coppie di distanze) per rispettare la dipendenza strutturale delle RDM.

### Scelte consolidate
- **Distanza coseno** (non euclidea): invariante alla norma dei vettori, che negli
  embedding può variare per motivi non semantici.
- **Spearman** (non Pearson): robusto a relazioni monotone non lineari.
- **Block bootstrap** (non pair bootstrap): ogni termine contribuisce a (N-1)
  distanze → le coppie non sono indipendenti. Il pair bootstrap sottostima la
  varianza e produce CI anti-conservativi.
- **Correzione Phipson-Smyth** (+1 a numeratore e denominatore del p-value):
  evita p=0 esatto e corregge il bias discreto.
- **n_permutations=10000** per Mantel, **n_bootstrap=1000** per CI.

### Statistica chiave per multi-model
`spearman_r` (scalare, range [-1, 1])

### Interpretazione
- r ≈ 0 → geometrie indipendenti (le tradizioni giuridiche organizzano i concetti
  in modi completamente diversi)
- r > 0 → parziale isomorfismo (alcune strutture concettuali sono condivise)
- r ≈ 1 → isomorfismo perfetto (gli spazi hanno la stessa geometria)
- r² = varianza condivisa tra le due geometrie

Rif.: Kriegeskorte, Mur & Bandettini (2008) Frontiers in Systems Neuroscience, 2, 4.
Rif.: Nili et al. (2014) PLoS Comp. Bio., 10(4), e1003553 (block bootstrap).
Rif.: Mantel (1967) Cancer Research, 27(2), 209-220.
Rif.: Phipson & Smyth (2010) Stat. Appl. Genet. Mol. Biol., 9(1), Art. 39.

---

## 3. Esperimento 2: Gromov-Wasserstein (GW)

### Cosa misura
**Distorsione strutturale via trasporto ottimale.** Quanto bisogna "deformare" uno
spazio per allinearlo all'altro? GW cerca il piano di trasporto che minimizza la
distorsione delle distanze interne.

### Differenza da RSA
RSA correla le distanze (secondo ordine, scalare). GW cerca l'*allineamento
ottimale* tra gli spazi (più informativo: produce un piano di trasporto, non solo
un coefficiente).

### Scelte consolidate
- **Regolarizzazione entropica** (epsilon=5e-3, Sinkhorn): rende il problema
  differenziabile e risolvibile in O(n² log n) anziché O(n³ log n). Epsilon
  piccolo = buona approssimazione.
- **Loss quadratica** ("square_loss"): amplifica le discrepanze strutturali.
- **Distribuzioni uniformi**: ogni termine ha lo stesso peso (nessun motivo per
  pesare diversamente i concetti).
- **Permutation test**: permuta righe/colonne della matrice di costo sinica.
  Sotto H0, non c'è corrispondenza strutturale.
- **p-value direzionale**: proporzione di distanze permutate ≤ osservata.
  Se la distanza osservata è significativamente bassa → isomorfismo reale.

### Statistica chiave per multi-model
`distance` (scalare, ≥ 0, più basso = più isomorfo)

### Interpretazione
- GW ≈ 0 → spazi quasi isomorfi
- GW alto → forte anisomorfismo strutturale
- Piano di trasporto: mostra *quali* concetti si corrispondono tra i due spazi

Rif.: Alvarez-Melis & Jaakkola (2018) EMNLP.
Rif.: Peyré & Cuturi (2019) Foundations and Trends in ML, 11(5-6).
Rif.: Cuturi (2013) "Sinkhorn Distances", NeurIPS.

---

## 4. Esperimento 3: Proiezione Assiologica (Kozlowski)

### Cosa misura
Le dimensioni culturali (individuo↔collettivo, formale↔sostanziale, ecc.)
sono codificate come *direzioni* nello spazio embedding. Si proiettano i termini
giuridici su queste direzioni e si confronta l'ordine risultante.

### Scelte consolidate
- **Coppie multiple di antonimi** per asse (non una sola coppia): riduce il rumore
  idiosincratico. L'asse è la media normalizzata delle differenze.
- **Assi indipendenti per modello**: l'asse WEIRD usa coppie inglesi, l'asse Sinic
  usa coppie cinesi. Non si impone che "freedom" e "自由" definiscano la stessa
  direzione.
- **Cosine similarity come proiezione**: score in [-1, 1], interpretabile.
- **Bootstrap CI** sullo Spearman (term-level resampling).

### Complicazione per multi-model
A differenza di RSA/GW, qui servono le **funzioni di embedding** (non solo le
matrici pre-calcolate) per costruire gli assi da zero per ogni coppia di modelli.
L'orchestratore generico non basta — serve un wrapper specifico.

### Statistica chiave per multi-model
`spearman_r` per ciascun asse (più statistiche, una per dimensione culturale).
Aggregazione: media/std del rho per-asse tra le 9 coppie.

### Interpretazione
- rho alto → la dimensione culturale ordina i concetti allo stesso modo in
  entrambe le tradizioni
- rho basso/negativo → la dimensione culturale è organizzata diversamente
- Outlier interessanti: termini con grande discrepanza di proiezione

Rif.: Kozlowski, Taddy & Evans (2019) Am. Soc. Rev., 84(5), 905-949.

---

## 5. Esperimento 4: Clustering Gerarchico + Fowlkes-Mallows

### Cosa misura
I due modelli producono la stessa **tassonomia** dei concetti giuridici? Si
costruiscono dendrogrammi indipendenti e si confrontano le partizioni.

### Scelte consolidate
- **Metodo Ward**: minimizza la varianza intra-cluster. Produce cluster compatti
  e di dimensione simile — adatto a domini giuridici con dimensioni comparabili.
- **Multi-k** ([3, 5, 7, 10]): un singolo k potrebbe essere artefatto. Si
  verifica la robustezza a granularità diverse.
- **FM index**: media geometrica di PPV e TPR sulle coppie co-assegnate.
  FM = sqrt(PPV × TPR). FM=1 = partizioni identiche.
- **Permutation test per-k**: permuta le etichette cluster dello spazio sinico.

### Problema per multi-model: quale scalare?
FM varia con k. Opzioni considerate:
1. FM a un k di riferimento (arbitrario)
2. Media FM su tutti i k (perde granularità)
3. **Report per-k su tutte le coppie** (scelta preferita: matrice k × 9 coppie)

**Decisione**: per l'aggregazione multi-model, usare la **media FM su k** come
scalare riassuntivo (`stat_key="mean_fm"`), ma conservare i valori per-k nei
risultati dettagliati per ispezione.

### Interpretazione
- FM ≈ 1/k (random) → tassonomie indipendenti
- FM >> 1/k → struttura tassonomica condivisa
- FM alto a tutti i k → robustezza della corrispondenza tassonomica

Rif.: Fowlkes & Mallows (1983) JASA, 78(383), 553-569.

---

## 6. Esperimento 5: Neighborhood Divergence Analysis (NDA)

### Part A: k-NN Jaccard
**Cosa misura**: per ogni concetto giuridico, il vicinato semantico è lo stesso
nei due spazi? "Falsi amici" = stessa parola, vicinati diversi.

### Scelte consolidate (Part A)
- **k=15**: circa 2% del pool totale (~770 termini). Bilancio tra stabilità
  statistica e sensibilità locale.
- **Pool completo**: core + background + control = 768 termini. I background
  terms forniscono contesto (il vicinato di "contract" include termini come
  "agreement", "obligation" — non solo i core terms).
- **Jaccard**: J = |W ∩ S| / |W ∪ S|. J=1 = vicinati identici, J=0 = nessun
  vicino in comune.
- **Permutation test**: permuta associazioni concetto↔embedding nello spazio
  sinico. Sotto H0, la Jaccard media è bassa.

### Part B: Decomposizioni normative
**Cosa misura**: aritmetica vettoriale (A - B → ?) per testare ipotesi
giurisprudenziali specifiche. Es.: "Legge - Stato = ?" rivela cosa resta del
concetto di legge quando si rimuove la componente statale.

### Complicazione per multi-model
Come gli Axes, NDA Part A richiede embedding del **corpus completo** per ogni
modello, e Part B richiede **funzioni di embedding**. Servono wrapper specifici.

### Statistica chiave per multi-model
- Part A: `mean_jaccard` (scalare, range [0, 1])
- Part B: qualitativo — più adatto a discussione narrativa che ad aggregazione
  numerica. Potremmo comunque aggregare la Jaccard media delle decomposizioni.

### Interpretazione
- Mean Jaccard alta → i vicinati sono concordi (effetto culturale limitato)
- Mean Jaccard bassa → divergenza sistematica dei vicinati
- Falsi amici specifici → casi di studio per analisi qualitativa

Rif.: Haemmerli et al. (2024) arXiv:2411.08687 (NNGS).
Rif.: Mikolov et al. (2013) NAACL-HLT (vector arithmetic).

---

## 7. Piano di implementazione multi-model

### Stato attuale
| Esperimento | Multi-model | Orchestratore generico? |
|-------------|-------------|-------------------------|
| RSA         | FATTO       | Si                      |
| GW          | DA FARE     | Si (firma compatibile)  |
| Clustering  | DA FARE     | Quasi (serve `mean_fm`) |
| NDA Part A  | DA FARE     | No (serve corpus full)  |
| Axes        | DA FARE     | No (serve embed_fn)     |
| NDA Part B  | DA FARE     | No (serve embed_fn)     |

### Strategia
Due opzioni per gli esperimenti incompatibili:

**Opzione A**: Estendere l'orchestratore per passare dati aggiuntivi.
- Pro: codice centralizzato
- Contro: l'interfaccia generica diventa specifica, perde eleganza

**Opzione B** (preferita): Creare wrapper per-esperimento che adattano la firma.
- Pro: ogni esperimento gestisce la propria complessità
- Contro: un po' di boilerplate

```python
# Esempio wrapper per Axes multi-model
def _run_axes_for_pair(model_w, model_s, client, texts_en, texts_zh, value_axes, ...):
    emb_w = client.get_embeddings_for_model(texts_en, model_w)
    emb_s = client.get_embeddings_for_model(texts_zh, model_s)

    def embed_weird(texts):
        return client.get_embeddings_for_model(texts, model_w)
    def embed_sinic(texts):
        return client.get_embeddings_for_model(texts, model_s)

    return run_axes_experiment(emb_w, emb_s, labels, value_axes,
                              embed_weird, embed_sinic, ...)
```

### Ordine di implementazione
1. GW (più semplice, firma compatibile)
2. Clustering (serve `mean_fm` come scalare)
3. Axes (serve wrapper con embed_fn)
4. NDA Part A (serve wrapper con corpus completo)
5. NDA Part B (serve wrapper con embed_fn, ma meno prioritario per aggregazione)

---

## 8. Scelte statistiche trasversali

### Permutation tests (non parametrici)
Gli embedding producono distribuzioni non normali con dipendenze strutturali. I
test parametrici (t-test, F-test) presuppongono normalità e indipendenza: entrambe
le assunzioni cadono. I permutation test generano una distribuzione nulla empirica.

Rif.: Good (2005) "Permutation, Parametric, and Bootstrap Tests", 3rd ed., Springer.

### Correzione Phipson-Smyth
p = (count + 1) / (n_perm + 1). Evita p=0 esatto. Garantisce p ≥ 1/(n_perm+1).

Rif.: Phipson & Smyth (2010) Stat. Appl. Genet. Mol. Biol., 9(1).

### Bootstrap di Efron
Ricampionamento con rimpiazzo dal campione originale. Metodo dei percentili per CI.

Rif.: Efron (1979) Annals of Statistics, 7(1), 1-26.

### Distanza del coseno (non euclidea)
Invariante alla norma. Negli embedding, la norma può variare per motivi non
semantici (frequenza token, batch di addestramento).

### Seed fisso (42)
Riproducibilità completa. Tutti i test usano lo stesso seed globale.

---

## 9. Dataset

### Composizione
- **394 core terms**: 9 domini giuridici (constitutional, rights, civil, criminal,
  governance, jurisprudence, international, labor_social, environmental_tech)
- **324 background terms**: termini giuridici generici che forniscono contesto
  al vicinato (NDA)
- **50 control terms**: termini concreti/quotidiani non giuridici (baseline)
- **Totale**: ~768 termini

### Fonte primaria
HK DOJ Bilingual Legal Glossary (34k+ voci). Traduzione ufficiale EN↔ZH del
governo di Hong Kong — la fonte più autorevole per terminologia giuridica bilingue.

### Conversione T2S
OpenCC Traditional→Simplified per compatibilità con BGE-ZH (addestrato su
cinese semplificato).

---

## 10. Visualizzazioni

### Principi
- **Okabe-Ito palette**: colorblind-safe, 8 colori distinguibili
- **Publication-ready PNG**: matplotlib/seaborn, DPI 300, dimensioni fisse
- **Interactive HTML**: Plotly con hover, zoom, dropdown
- **Light mode**: omette dati pesanti (RDM raw, transport plan) per file piccoli

### Per-esperimento
| Esperimento | PNG plots | HTML tabs |
|-------------|-----------|-----------|
| RSA | Heatmap, inter-domain, hexbin, null dist | Heatmap + scatter + metrics |
| GW | Transport histogram, top-K alignments | (in progress) |
| Axes | Forest plot, scatter per asse | (in progress) |
| Clustering | Dendrogram troncato, FM bar chart | (in progress) |
| NDA | Jaccard histogram, false friends network, decomposition bars | (in progress) |
| Multi-model | Consistency heatmap, forest plot, null dist overlay | Heatmap + forest + table |
