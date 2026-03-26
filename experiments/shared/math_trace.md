# Computational & Mathematical Trace — Geometria Iuris

Spiegazione funzione per funzione di ogni operazione computazionale e matematica
usata nel pipeline. Scritta per un lettore con formazione giuridica e cultura
scientifica generale (riferimento: liceo classico + derivate concettualmente).

**Convenzioni:**
- `codice inline` = nome di funzione, variabile, o valore concreto
- *corsivo* = termine tecnico introdotto per la prima volta
- Ogni sezione segue lo stesso schema fisso

---

## Indice

### Fondamenti
- [F1 — Cos'è un vettore](#f1--cosè-un-vettore)
- [F2 — Cos'è una matrice](#f2--cosè-una-matrice)
- [F3 — Cos'è un array NumPy](#f3--cosè-un-array-numpy)

### Caricamento dati
- [L1 — `load_precomputed()`](#l1--load_precomputed)
- [L2 — `np.load()`](#l2--npload)

### Costruzione dell'RDM
- [R1 — Moltiplicazione matriciale `@`](#r1--moltiplicazione-matriciale-)
- [R2 — `np.fill_diagonal()`](#r2--npfill_diagonal)
- [R3 — `np.triu_indices()`](#r3--nptriu_indices)

### Analisi §3.1 — Domain signal
- [A1 — `scipy.stats.mannwhitneyu()`](#a1--scipystatmannwhitneyu)
- [A2 — Effect size: rank-biserial correlation](#a2--effect-size-rank-biserial-correlation)
- [A3 — Matrice inter-dominio K×K](#a3--matrice-inter-dominio-k×k)

### Analisi §3.1.4 — RSA
- [S1 — `scipy.stats.spearmanr()`](#s1--scipystats-spearmanr)
- [S2 — Mantel test (permutation test su matrici)](#s2--mantel-test-permutation-test-su-matrici)
- [S3 — `np.random.default_rng()` e `rng.permutation()`](#s3--nprandomdefault_rng-e-rngpermutation)
- [S4 — `np.ix_()`](#s4--npix_)
- [S5 — Block bootstrap per confidence interval](#s5--block-bootstrap-per-confidence-interval)
- [S6 — `np.percentile()`](#s6--nppercentile)

### Estrazione per layer §3.1.3
- [E1 — `output_hidden_states` e hidden states di un transformer](#e1--output_hidden_states-e-hidden-states-di-un-transformer)
- [E2 — Pooling: CLS vs Mean](#e2--pooling-cls-vs-mean)
- [E3 — Normalizzazione L2 per layer](#e3--normalizzazione-l2-per-layer)
- [E4 — `np.savez_compressed()` e caching](#e4--npsavez_compressed-e-caching)
- [E5 — Sanity check: layer finale vs precomputed](#e5--sanity-check-layer-finale-vs-precomputed)

### Analisi §3.1.3a — Drift e Jaccard per layer
- [D1 — Cosine drift tra layer consecutivi](#d1--cosine-drift-tra-layer-consecutivi)
- [D2 — k-NN Jaccard instability tra layer consecutivi](#d2--k-nn-jaccard-instability-tra-layer-consecutivi)
- [D3 — Aggregazione per dominio](#d3--aggregazione-per-dominio)
- [D4 — Top-N terms per drift totale](#d4--top-n-terms-per-drift-totale)

### Analisi §3.1.3b — Domain signal emergence e RSA convergence
- [G1 — Domain signal per layer: curva r(l)](#g1--domain-signal-per-layer-curva-rl)
- [G2 — RSA convergence to final: curva ρ(l)](#g2--rsa-convergence-to-final-curva-ρl)
- [G3 — Threshold detection: soglie 50% e 90%](#g3--threshold-detection-soglie-50-e-90)

### Analisi §3.1.3c — Neighborhood Trajectory Analysis (NTA)
- [N1 — Pool construction: core + control](#n1--pool-construction-core--control)
- [N2 — k-NN retrieval per layer](#n2--k-nn-retrieval-per-layer)
- [N3 — Entry/exit detection](#n3--entryexit-detection)
- [N4 — Domain/tier composition tracking](#n4--domaintier-composition-tracking)

### Analisi §3.3 — Value axis projection
- [V1 — Kozlowski difference-vector: costruzione dell'asse](#v1--kozlowski-difference-vector-costruzione-dellasse)
- [V2 — Proiezione su un asse: cosine similarity come punteggio](#v2--proiezione-su-un-asse-cosine-similarity-come-punteggio)
- [V3 — Sanity check: orientamento dell'asse](#v3--sanity-check-orientamento-dellasse)
- [V4 — Cosine similarity inter-asse (diagnostica ortogonalità)](#v4--cosine-similarity-inter-asse-diagnostica-ortogonalità)
- [V5 — Spearman ρ tra vettori di punteggio](#v5--spearman-ρ-tra-vettori-di-punteggio)
- [V6 — Row-resample bootstrap per Spearman ρ](#v6--row-resample-bootstrap-per-spearman-ρ)
- [V7 — Mann-Whitney U su gruppi di ρ (cross vs within)](#v7--mann-whitney-u-su-gruppi-di-ρ-cross-vs-within)
- [V8 — Kruskal-Wallis H + post-hoc Bonferroni](#v8--kruskal-wallis-h--post-hoc-bonferroni)

---

## Fondamenti

---

### F1 — Cos'è un vettore

**Concetto**

Un *vettore* è una lista ordinata di numeri. In geometria, rappresenta una direzione
e una lunghezza in uno spazio a N dimensioni.

```
v = [0.12, -0.45, 0.87, ...]   ← lista di numeri reali
```

In questo progetto ogni termine giuridico — "habeas corpus", "contratto",
"nullità" — è rappresentato da un vettore di 768 o 1024 numeri. Non ha senso
guardare i singoli numeri: è la posizione del vettore *rispetto agli altri* che
conta.

**Intuizione per il giurista**

Immagina uno spazio in cui ogni concetto giuridico occupa una posizione.
Il contratto è vicino all'obbligazione. Il crimine è vicino alla sanzione.
La Costituzione è lontana dal reato penale. Il vettore è semplicemente
le coordinate di quel punto nello spazio.

**Notazione**: vettori in minuscolo grassetto **v** ∈ ℝᵈ (d = dimensione).

---

### F2 — Cos'è una matrice

**Concetto**

Una *matrice* è una griglia rettangolare di numeri, con righe e colonne.

```
M = [[1.0, 0.3, 0.7],
     [0.3, 1.0, 0.2],
     [0.7, 0.2, 1.0]]
```

Una matrice N×M ha N righe e M colonne. Una matrice N×N (uguale numero di
righe e colonne) si chiama *quadrata*.

Negli esperimenti usiamo soprattutto:
- Matrici N×d dove N = numero di termini, d = dimensione del vettore
- Matrici N×N dove ogni cella rappresenta la distanza tra due termini

**Notazione**: matrici in MAIUSCOLO grassetto **M**, elementi come M[i,j].

---

### F3 — Cos'è un array NumPy

**Contesto**: `import numpy as np`

*NumPy* è la libreria Python per il calcolo numerico. Il suo oggetto principale
è l'`ndarray` (n-dimensional array): un vettore, una matrice, o una struttura
di dimensioni superiori.

**Perché non usare le liste Python normali?**

Una lista Python `[1, 2, 3]` è flessibile ma lenta: ogni elemento è un oggetto
separato in memoria. Un array NumPy è un blocco di memoria contiguo di numeri
dello stesso tipo (es. `float32`): 10-100 volte più veloce per operazioni
matematiche.

**Proprietà chiave di un array:**

```python
arr.shape    # (N, d) — dimensioni
arr.dtype    # float32, float64, int64, ...
arr.ndim     # 1 (vettore), 2 (matrice), ...
```

In questo progetto tutti i vettori di embedding sono `float32` (numeri a 32 bit)
perché offrono sufficiente precisione con metà della memoria rispetto a `float64`.

---

## Caricamento dati

---

### L1 — `load_precomputed()`

**Source**: `shared/embeddings.py`
**Usata in**: ogni Lens, come prima operazione

**Firma**
```python
vectors, index = load_precomputed(model_label, embeddings_dir)
```

**Cosa fa**

Carica da disco i vettori pre-calcolati per un modello e il registro condiviso
dei termini.

**Parametri**
- `model_label` (str): nome breve del modello, es. `"BGE-EN-large"`
- `embeddings_dir` (str | Path): cartella prodotta da `shared/precompute.py`

**Restituisce**
- `vectors`: array NumPy di forma `(N, dim)`, dtype `float32`, normalizzati L2
- `index`: lista di N dizionari, ognuno con `{"en", "zh_canonical", "domain", "tier"}`

**Perché pre-calcolati?**

Ogni modello impiega 3–8 minuti per processare 9.472 termini. Eseguire
questa operazione ogni volta che si lancia un esperimento sarebbe insostenibile.
I vettori vengono calcolati una volta sola (`shared/precompute.py`) e salvati
come file binari `.npy`. Ogni Lens li legge in pochi secondi.

**Dipendenza critica**: l'ordine dei termini in `vectors[i]` corrisponde
esattamente a `index[i]`. Questa corrispondenza non va mai interrotta.

**Esempio**
```python
from shared.embeddings import load_precomputed

EMB_DIR = ROOT / "data" / "processed" / "embeddings"
vecs, index = load_precomputed("BGE-EN-large", EMB_DIR)

print(vecs.shape)          # (9472, 1024)
print(index[0])            # {"en": "habeas corpus", "zh_canonical": "...", ...}
print(index[0]["en"])      # "habeas corpus"  →  vecs[0] è il suo vettore
```

---

### L2 — `np.load()`

**Source**: NumPy
**Usata in**: `load_precomputed()` internamente

**Firma**
```python
arr = np.load(path)
```

**Cosa fa**

Legge un file `.npy` (formato binario NumPy) e restituisce l'array salvato,
con forma e tipo esattamente come al momento del salvataggio.

**Perché `.npy` e non `.csv`?**

Un file CSV di 9.472 × 1.024 numeri float occuperebbe ~200MB come testo e
richiederebbe parsing riga per riga. Il file `.npy` occupa ~39MB (float32
compresso) e si legge in un'unica operazione di lettura memoria. La differenza
è 10-50× in velocità e dimensione.

---

## Costruzione dell'RDM

La *Relational Dissimilarity Matrix* (RDM) è la struttura dati centrale di Lens I.
Le sezioni seguenti spiegano le tre operazioni per costruirla.

---

### R1 — Moltiplicazione matriciale `@`

**Source**: NumPy (operatore built-in Python 3.5+)
**Usata in**: `compute_rdm()` — calcolo delle similarità coseno

**Firma**
```python
result = A @ B
```

**Cosa fa**

Moltiplica due matrici secondo le regole dell'algebra lineare. Se `A` ha forma
`(M, K)` e `B` ha forma `(K, N)`, il risultato ha forma `(M, N)`.

**Il caso che ci interessa**: `vecs @ vecs.T`

`vecs` ha forma `(N, d)` — N termini, ciascuno un vettore di d componenti.
`vecs.T` è la *trasposta*: forma `(d, N)` — righe e colonne scambiate.

```
vecs:    (N, d)
vecs.T:  (d, N)
risultato: (N, N)
```

L'elemento `[i, j]` del risultato è il *prodotto scalare* tra la riga i di `vecs`
e la colonna j di `vecs.T`, cioè tra il vettore del termine i e il vettore del
termine j:

```
(vecs @ vecs.T)[i, j] = vecs[i] · vecs[j] = Σₖ vecs[i,k] × vecs[j,k]
```

**Perché questo è la similarità coseno?**

La similarità coseno tra due vettori è definita come:

```
cos(u, v) = (u · v) / (‖u‖ × ‖v‖)
```

I nostri vettori sono normalizzati L2: `‖u‖ = ‖v‖ = 1`. Quindi:

```
cos(u, v) = u · v   (con vettori normalizzati)
```

In una riga: `vecs @ vecs.T` produce l'intera matrice N×N di similarità coseno
con una sola operazione, senza alcun loop.

**Intuizione per il giurista**

Il prodotto scalare misura quanto due frecce "puntano nella stessa direzione".
Due vettori identici danno 1.0 (stessa direzione). Due vettori perpendicolari
danno 0.0 (nessuna relazione). Due vettori opposti danno -1.0 (raro negli
embedding semantici, indicherebbe significati "contrari").

**Complessità computazionale**

Con N=9.472 e d=1.024: la matrice ha 9.472² ≈ 89,7 milioni di celle.
NumPy esegue questa operazione in pochi secondi su CPU sfruttando le
istruzioni SIMD del processore (operazioni vettoriali hardware).

**Esempio**
```python
vecs = np.array([[1.0, 0.0],   # termine A
                 [0.0, 1.0],   # termine B
                 [0.707, 0.707]])  # termine C (a 45° tra A e B)

sim = vecs @ vecs.T
# sim[A,B] = 0.0  (perpendicolari → nessuna similarità)
# sim[A,C] = 0.707  (45°)
# sim[A,A] = 1.0  (identici → diagonale)
```

---

### R2 — `np.fill_diagonal()`

**Source**: NumPy
**Usata in**: `compute_rdm()` — azzeramento della diagonale

**Firma**
```python
np.fill_diagonal(matrix, value)
```

**Cosa fa**

Imposta tutti gli elementi sulla diagonale principale di `matrix` al valore `value`.
Opera *in-place*: modifica la matrice originale senza restituire nulla.

**Perché la diagonale è un problema?**

La diagonale di `sim = vecs @ vecs.T` è sempre 1.0 (ogni vettore è identico
a se stesso). La distanza coseno `1 - sim` avrebbe quindi 0.0 sulla diagonale
— il che è matematicamente corretto (distanza di un punto da se stesso = 0).

Ma in tutte le analisi successive vogliamo *escludere* i confronti di un termine
con se stesso. Avere 0 sulla diagonale non è un errore, ma azzerare esplicitamente
previene errori numerici da floating point e rende il codice auto-documentante.

**Esempio**
```python
rdm = 1.0 - (vecs @ vecs.T)
np.fill_diagonal(rdm, 0.0)   # sicurezza numerica
```

---

### R3 — `np.triu_indices()`

**Source**: NumPy
**Usata in**: `upper_tri()` — estrazione del triangolo superiore

**Firma**
```python
rows, cols = np.triu_indices(n, k=1)
```

**Cosa fa**

Restituisce gli *indici* di tutti gli elementi nel triangolo superiore
di una matrice n×n, escludendo la diagonale (parametro `k=1`).

**Perché serve?**

L'RDM è simmetrica: `RDM[i,j] = RDM[j,i]`. Confrontare entrambi i triangoli
significherebbe contare ogni coppia due volte. L'analisi statistica (Spearman,
Mann-Whitney) deve operare su coppie *uniche*: il triangolo superiore contiene
esattamente N(N-1)/2 valori distinti.

Per N=397 termini core: 397×396/2 = **78.606 coppie uniche**.

**Il parametro `k`**

- `k=0`: include la diagonale (N² elementi)
- `k=1`: esclude la diagonale (quello che vogliamo: N(N-1)/2 elementi)
- `k=2`: esclude anche la sovra-diagonale (raramente utile)

**Esempio**
```python
n = 4
rows, cols = np.triu_indices(n, k=1)
# rows = [0, 0, 0, 1, 1, 2]
# cols = [1, 2, 3, 2, 3, 3]
# → 6 coppie: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

rdm = np.array([[0, 0.3, 0.7, 0.5],
                [0.3, 0, 0.4, 0.6],
                [0.7, 0.4, 0, 0.2],
                [0.5, 0.6, 0.2, 0]])

tri = rdm[rows, cols]
# tri = [0.3, 0.7, 0.5, 0.4, 0.6, 0.2]
# → vettore di 6 valori, uno per coppia unica
```

---

## Analisi §3.1 — Domain signal

---

### A1 — `scipy.stats.mannwhitneyu()`

**Source**: SciPy
**Usata in**: §3.1.1 (intra vs inter-dominio), §3.1.1 (legal vs control)
**Implementata in**: `shared/statistical.py` → `mannwhitney_with_r()`

**Firma**
```python
result = scipy.stats.mannwhitneyu(x, y, alternative='less')
result.statistic   # statistica U = numero di coppie "discordanti"
result.pvalue      # p-value
```

**Cosa fa**

Testa se i valori in x tendono sistematicamente a essere più piccoli dei valori
in y, senza assumere nessuna distribuzione specifica.

**L'algoritmo (esempio numerico)**

```
Intra-dominio (x): [0.10, 0.25, 0.40, 0.55]   (4 valori)
Inter-dominio (y): [0.35, 0.60, 0.70, 0.80]   (4 valori)

Passo 1 — mescola e ordina tutto:
  Valore: 0.10  0.25  0.35  0.40  0.55  0.60  0.70  0.80
  Gruppo:   x     x     y     x     x     y     y     y
  Rango:    1     2     3     4     5     6     7     8

Passo 2 — somma dei ranghi di x:
  R_x = 1 + 2 + 4 + 5 = 12

Passo 3 — statistica U:
  U_x = R_x - n_x(n_x+1)/2  =  12 - 4×5/2  =  12 - 10  =  2

Interpretazione: U_x = 2 = numero di coppie (xᵢ, yⱼ) in cui xᵢ > yⱼ.
Solo 2 coppie su 16 totali sono "discordanti" → x quasi sempre < y.
```

**H₀ e H₁**

- H₀: nessuna differenza sistematica tra x e y
- H₁ (`alternative='less'`): x < y

p-value = probabilità di ottenere U ≤ 2 per puro caso sotto H₀.
Con 16 coppie totali e U=2, p è molto piccolo → rifiutiamo H₀.

**Perché Mann-Whitney e non t-test?**

Il t-test assume normalità. Le distanze coseno in embedding ad alta dimensione
non sono normali (distribuzione asimmetrica, dipendente dalla geometria).
Mann-Whitney lavora solo sui ranghi: non fa assunzioni sulla distribuzione.

**Intuizione per il giurista**

Metti in fila tutti i valori di x e y dal più piccolo al più grande. Se i valori
di x tendono ad occupare le posizioni in testa alla classifica (numeri piccoli),
c'è evidenza che x < y sistematicamente. U conta quante eccezioni ci sono.

**Implementazione nel progetto**
```python
# shared/statistical.py
res = mannwhitney_with_r(intra_dists, inter_dists, alternative="less")
# res.statistic = U
# res.p_value
# res.effect_r  = rank-biserial correlation
```

---

### A2 — Effect size: rank-biserial correlation r

**Source**: formula custom — `shared/statistical.py` → `mannwhitney_with_r()`
**Usata in**: §3.1.1, §3.1.1 — dimensione dell'effetto

**Il problema del p-value da solo**

Con decine di migliaia di coppie, qualsiasi differenza sarà statisticamente
significativa. Il p-value dice "esiste un effetto?". L'*effect size* dice
"quanto è grande?". Entrambi sono necessari (→ §2.4 "refusal of stargazing").

**La formula**

Partendo dall'esempio di A1 (n_x=4, n_y=4, U=2):

```
Coppie concordanti  (x < y): 16 - 2 = 14
Coppie discordanti  (x > y): 2
Totale coppie: 16

r = (concordanti - discordanti) / totale
  = (14 - 2) / 16
  = 0.75

Equivalente:  r = 1 - 2U/(n_x × n_y) = 1 - 2×2/16 = 0.75
```

**Lettura diretta**

```
r = +1.00  →  x SEMPRE < y  (separazione perfetta)
r = +0.75  →  (1+0.75)/2 = 87.5% delle coppie ha x < y
r = +0.50  →  effetto grande (convenzione Cohen)
r = +0.30  →  effetto medio
r = +0.10  →  effetto piccolo
r =  0.00  →  nessuna differenza sistematica
r = -1.00  →  x SEMPRE > y
```

Il collegamento `(1+r)/2 = probabilità che un valore casuale di x < un valore casuale di y`
è la lettura più intuitiva: r=0.75 → 87.5% di probabilità.

---

### A3 — Matrice inter-dominio K×K

**Source**: `lens_1_relational/lens1.py` → `_domain_topology()`
**Usata in**: §3.1.2 — topologia dei domini

**Cosa produce**

Con K domini giuridici, una matrice K×K dove la cella (d1, d2) è la distanza
coseno media tra tutti i termini del dominio d1 e tutti i termini del dominio d2.

È la "mappa geografica" dello spazio giuridico: leggi quanto sono vicine o
lontane tra loro le aree del diritto secondo il modello.

**Costruzione passo per passo**

Esempio con 3 domini semplificati (penale, civile, costituzionale):

```
Termini core e loro domini:
  idx 0: habeas_corpus   → penale
  idx 1: contratto       → civile
  idx 2: dolo            → penale
  idx 3: colpa           → penale
  idx 4: proprietà       → civile
  idx 5: sovranità       → costituzionale

Cella (penale, civile):
  idx_penale = [0, 2, 3]
  idx_civile = [1, 4]
  sottomatrice = rdm[[0,2,3], :][:, [1,4]]   ← np.ix_() fa questo
             = [[rdm[0,1], rdm[0,4]],
                [rdm[2,1], rdm[2,4]],
                [rdm[3,1], rdm[3,4]]]
  cella(penale,civile) = media di tutti e 6 i valori

Cella diagonale (penale, penale):
  idx_penale = [0, 2, 3]
  sottomatrice 3×3 → prendo solo triangolo superiore (3 coppie)
  cella(penale,penale) = media delle 3 distanze intra-penale
```

**Lettura del risultato**

Cella piccola → domini geometricamente vicini (concetti semanticamente affini).
Cella grande → domini lontani. La diagonale è la coesione interna del dominio.

---

## Analisi §3.1.4 — RSA e Mantel test

---

### S1 — `scipy.stats.spearmanr()`

**Source**: SciPy
**Usata in**: RSA — correlazione tra RDMs (§3.1.4)

**Firma**
```python
result = scipy.stats.spearmanr(x, y)
result.statistic   # ρ (rho), ∈ [-1, 1]
result.pvalue      # p-value (NON usato in RSA — usiamo il Mantel test)
```

**Cosa fa**

Calcola la *correlazione di Spearman* ρ tra due vettori x e y.

**Il concetto**

La correlazione di Spearman è la correlazione di Pearson applicata ai *ranghi*
anziché ai valori originali. In pratica:

1. Converti x nei suoi ranghi: il valore più piccolo → rango 1, il più grande → rango N
2. Converti y nei suoi ranghi
3. Calcola la correlazione di Pearson tra i due vettori di ranghi

```
ρ = Pearson(rank(x), rank(y))
```

La correlazione di Pearson tra due vettori u e v è:

```
r = Σᵢ(uᵢ - ū)(vᵢ - v̄) / (N × std(u) × std(v))
```

dove ū e v̄ sono le medie e std è la deviazione standard.

**Perché i ranghi e non i valori diretti?**

Pearson misura correlazione *lineare*: funziona bene se x grande implica y grande
in modo proporzionale. Spearman misura correlazione *monotona*: funziona bene
se x grande implica y grande (o piccolo), indipendentemente dalla proporzione.

Per RDMs di modelli diversi (dimensioni 768 vs 1024, scale di distanze diverse),
non ci aspettiamo che le distanze siano proporzionali — solo che i *ranghi*
siano concordanti. Spearman è quindi più appropriato.

**Intuizione per il giurista**

Due giudici devono classificare 10 sentenze dalla più severa alla più lieve.
Non importa che il primo giudice usi una scala 1-100 e il secondo una 1-10.
Importa che concordino sull'ordinamento. Spearman misura questa concordanza
d'ordinamento tra due RDMs.

**Valori**
```
ρ =  1.0  → perfetta concordanza d'ordinamento
ρ =  0.0  → nessuna relazione
ρ = -1.0  → ordinamento perfettamente inverso
```

**Attenzione**: il p-value restituito da `spearmanr()` assume indipendenza tra
le osservazioni — assunzione violata per le RDMs (vedi [F3](#f3--cosè-un-array-numpy)
e dipendenza tra coppie). In RSA usiamo sempre il **Mantel test** per la
significatività, non il p-value di `spearmanr()`.

**Esempio**
```python
from scipy.stats import spearmanr
import numpy as np

# Due vettori: triangoli superiori di due RDMs
tri_A = np.array([0.1, 0.5, 0.8, 0.3])
tri_B = np.array([0.2, 0.6, 0.9, 0.4])   # stesso ordine, scala diversa

result = spearmanr(tri_A, tri_B)
print(result.statistic)   # → 1.0 (stesso ordinamento)
```

---

### S2 — Mantel test (permutation test su matrici)

**Source**: implementazione custom in `shared/statistical.py`
**Usata in**: §3.1.4 — significatività statistica di RSA

**Concetto**

Il *Mantel test* (Mantel 1967, originariamente sviluppato per epidemiologia
geografica) testa se la correlazione tra due matrici di distanza è maggiore
di quanto ci si aspetterebbe per caso.

**Il problema della significatività per RDMs**

Non possiamo usare il p-value standard di Spearman: le N(N-1)/2 entrate
del triangolo superiore non sono indipendenti. Il termine i appare in N-1 coppie.
Con N=397 termini: 78.606 coppie, ma solo 397 "unità informative" reali.
Un t-test con 78.604 gradi di libertà produce p-value falsi.

**La soluzione: distribuzione nulla per permutazione**

L'idea: se H₀ è vera (nessuna correlazione reale tra le RDMs), possiamo
rimescolare le etichette dei termini senza cambiare la distribuzione di ρ.

Algoritmo:
```
1. Calcola ρ_obs = Spearman(tri(RDM_A), tri(RDM_B))

2. Ripeti B=1000 volte:
   a. Genera permutazione casuale π degli indici [0, ..., N-1]
   b. Permuta righe E colonne di RDM_B: RDM_B_perm = RDM_B[π, :][:, π]
   c. Calcola ρ_perm = Spearman(tri(RDM_A), tri(RDM_B_perm))
   d. Salva ρ_perm nella distribuzione nulla

3. p-value = #{ρ_perm ≥ ρ_obs} / B
```

**Perché permutare righe E colonne insieme?**

L'RDM è una matrice di distanze tra oggetti *etichettati*. L'oggetto i occupa
la riga i E la colonna i. Permutare solo le righe spezzerebbe questa simmetria.
Permutare π[i] significa "rinomina il termine i come π[i]" — va fatto
coerentemente su entrambi gli assi per mantenere la struttura di matrice di
distanza.

**Intuizione per il giurista**

Immagina un grafico in cui i punti sono i termini giuridici e le linee sono
le distanze. Osservi che la mappa A e la mappa B sembrano simili (alta ρ).
Il Mantel test chiede: se avessi casualmente rinominato tutti i punti in B
(scambiato "contratto" con "reato", "diritto" con "obbligo", ecc.),
quanto spesso otterresti una similarità così alta per puro caso?
Se quasi mai: l'osservazione è significativa.

---

### S3 — `np.random.default_rng()` e `rng.permutation()`

**Source**: NumPy
**Usata in**: `shared/statistical.py` — Mantel test, block bootstrap

**Firma**
```python
rng = np.random.default_rng(seed=42)      # crea generatore locale
pi  = rng.permutation(n)                  # permutazione casuale di [0,...,n-1]
idx = rng.choice(n, size=n, replace=True) # campionamento con rimpiazzo
```

**Il seed e la riproducibilità**

Un computer non produce numeri davvero casuali: usa un algoritmo deterministico
che parte da un valore iniziale (il *seed*) e genera una sequenza che appare
casuale. Stesso seed → stessa sequenza, sempre, su qualsiasi macchina.

Analogia: il seed è l'ordine iniziale di un mazzo di carte. Se "mescoli" sempre
partendo dallo stesso ordine, con lo stesso metodo, ottieni sempre le stesse carte.

Con `seed=42` fisso: il Mantel test e il bootstrap producono esattamente gli
stessi p-value e CI ad ogni esecuzione → riproducibilità garantita per peer-review.

**`rng.permutation(n)`** → array di n interi (0…n-1) in ordine casuale:
```python
rng = np.random.default_rng(42)
rng.permutation(5)  # → [3, 0, 4, 1, 2]  (sempre questo con seed=42)
```

**`rng.choice(n, size=n, replace=True)`** → n valori da {0…n-1} con rimpiazzo:
```python
rng.choice(5, size=5, replace=True)  # → [2, 2, 0, 4, 2]  (2 compare 3 volte)
```
"Con rimpiazzo" = dopo aver estratto un numero, lo rimetti nel pool. Alcuni indici
ripetono; altri mancano. È il meccanismo del bootstrap.

**Perché `default_rng()` e non la vecchia `np.random.seed()`?**

La vecchia API usa uno stato globale: due parti del codice che chiamano
`np.random` si "contaminano" a vicenda in modo imprevedibile. `default_rng()`
crea un generatore locale, isolato, thread-safe — best practice moderna.

---

### S4 — `np.ix_()`

**Source**: NumPy
**Usata in**: Mantel test (permutazione), §3.1.2 (matrice inter-dominio),
block bootstrap (sotto-RDM)

**Firma**
```python
grid = np.ix_(rows, cols)
submatrix = matrix[grid]      # equivalente a matrix[rows, :][:, cols]
```

**Cosa fa**

Costruisce una griglia di indici per selezionare una *sottomatrice* da una
matrice. È una scorciatoia per il broadcasting bidimensionale.

**Il problema senza `np.ix_()`**

Supponi di avere una matrice 5×5 e voglio estrarre la sottomatrice
formata dalle righe [0,2,4] e le colonne [0,2,4]:

```python
# ✗ Non funziona come ci si aspetta:
matrix[[0,2,4], [0,2,4]]   # → solo 3 elementi: (0,0), (2,2), (4,4)

# ✓ Funziona con np.ix_:
grid = np.ix_([0,2,4], [0,2,4])
matrix[grid]   # → matrice 3×3 come voluto
```

**Uso nel Mantel test**
```python
pi = rng.permutation(N)
rdm_perm = rdm_B[np.ix_(pi, pi)]   # permuta righe e colonne insieme
```

**Uso nel block bootstrap**
```python
idx = rng.choice(N, size=N, replace=True)
sub_A = rdm_A[np.ix_(idx, idx)]   # sotto-RDM con i termini ricampionati
sub_B = rdm_B[np.ix_(idx, idx)]   # stesso ricampionamento su entrambe
```

---

### S5 — Block bootstrap per confidence interval

**Source**: implementazione custom in `shared/statistical.py`
**Usata in**: §3.1.4 — intervallo di confidenza per ρ RSA
**Riferimento**: Nili et al. (2014), *PLoS Computational Biology* 10(4): e1003553

**Concetto**

Il *bootstrap* (Efron 1979) è un metodo per stimare l'incertezza di una
statistica campionando ripetutamente dai propri dati. Il *block bootstrap*
è la versione che rispetta la struttura di dipendenza dei dati.

**Perché non il bootstrap standard sulle coppie?**

Con N=397 termini e 78.606 coppie, potremmo ricampionare le coppie.
Ma le coppie non sono indipendenti: il termine i_0 appare in 396 coppie.
Ricampionare coppie come se fossero indipendenti produrrebbe CI artificialmente
stretti (falsa precisione).

**La soluzione: ricampionare i TERMINI, non le coppie**

Il termine è l'unità statistica naturale — è lì che nasce la dipendenza.

Algoritmo:
```
Dato RDM_A e RDM_B di dimensione N×N, seed fisso:

Per b = 1, ..., B (B=1000):
  1. Campiona con rimpiazzo N indici tra [0, N-1]:
     idx = rng.choice(N, size=N, replace=True)
     → alcuni termini compaiono più volte, altri sono assenti

  2. Estrai sotto-RDM di dimensione N×N (con ripetizioni):
     sub_A = RDM_A[np.ix_(idx, idx)]
     sub_B = RDM_B[np.ix_(idx, idx)]
     → stesso idx per entrambe: confrontiamo sempre le stesse coppie

  3. Calcola Spearman sul triangolo superiore della sotto-RDM:
     ρ_b = Spearman(tri(sub_A), tri(sub_B))

Distribuzione bootstrap = {ρ_1, ..., ρ_B}
CI al 95% = [percentile(2.5), percentile(97.5)]
```

**Perché stesso `idx` per entrambe le RDMs?**

Stiamo misurando la correlazione tra la struttura di due modelli *sugli stessi
termini*. Se ricampionassimo indici diversi per A e B, staremmo confrontando
distribuzioni di coppie diverse — non quello che vogliamo.

**Intuizione per il giurista**

Immagina di dover stimare quanto due periti concordano nel valutare sentenze.
Hai 397 sentenze. Il bootstrap chiede: se avessi avuto un insieme leggermente
diverso di sentenze (alcune mancanti, alcune duplicate per caso), quanto
cambierebbe la misura di concordanza? Il CI cattura questa incertezza campionaria.

**Interpretazione del CI**

CI al 95% = [0.41, 0.58] significa: con il 95% di confidenza, il vero valore
di ρ per la popolazione da cui proviene il nostro campione di termini è tra
0.41 e 0.58. Un CI stretto indica stima precisa; un CI largo indica che il
risultato è sensibile alla scelta specifica dei termini nel campione.

---

### S6 — `np.percentile()`

**Source**: NumPy
**Usata in**: block bootstrap — estrazione dei quantili del CI

**Firma**
```python
low, high = np.percentile(values, [2.5, 97.5])
```

**Cosa fa**

Calcola i *percentili* di un array. Il percentile p è il valore sotto il quale
cade il p% dei dati.

Per un CI al 95%:
- Percentile 2.5: il 2.5% dei valori bootstrap è sotto questa soglia
- Percentile 97.5: il 97.5% è sotto questa soglia

Il 95% dei valori bootstrap cade nell'intervallo `[low, high]`.

**Perché percentile 2.5 e 97.5?**

Un CI al 95% taglia il 5% rimanente in due parti uguali: 2.5% in coda bassa,
2.5% in coda alta. Questo è il *metodo dei percentili* (o *bootstrap percentile
interval*), il più semplice dei metodi bootstrap per CI. Con B=1000 campioni
è stabile e adeguato per il nostro scopo.

---

---

## Assegnamento per dominio (§3.1.1 — background terms)

---

### B1 — k-Nearest Neighbors (k-NN) majority vote

**Source**: implementazione custom in `lens_1_relational/domain_assignment.py`
**Usata in**: §3.1.1 — assegnamento dei termini background ai domini

**Concetto**

k-NN (k-Nearest Neighbors) è un algoritmo di classificazione per analogia:
per assegnare un'etichetta a un punto sconosciuto, guarda i k punti più simili
tra quelli già etichettati e prendi la classe di maggioranza.

Non c'è un modello da addestrare. Non ci sono parametri da ottimizzare.
L'unica operazione è una misura di distanza.

**Il caso concreto**

Hai 397 termini core con etichetta di dominio. Arriva il background term
"conveyancing fee" senza etichetta. Trovi i 7 core terms più simili nello
spazio di embedding e chiedi loro: "di che dominio sei?"

```
"conveyancing fee"  →  7 vicini più simili (cosine similarity):

  sim=0.91  property transfer   → civil     ← voto: civil
  sim=0.89  title deed          → civil     ← voto: civil
  sim=0.87  stamp duty          → civil     ← voto: civil
  sim=0.85  mortgage            → civil     ← voto: civil
  sim=0.83  conveyance          → civil     ← voto: civil
  sim=0.81  equity of redemption→ civil     ← voto: civil
  sim=0.72  criminal procedure  → criminal  ← voto: criminal

Conteggio voti:  civil=6,  criminal=1
→ assigned_domain: civil
→ confidence: 6/7 ≈ 0.86
```

**L'algoritmo passo per passo**

```
Input:
  - vecs_core:  (397, dim)  vettori core L2-normalizzati
  - labels:     (397,)      etichette di dominio dei core terms
  - vecs_bg:    (8975, dim) vettori background L2-normalizzati
  - k = 7

Per ogni background term b (indice i):
  1. Calcola similarità coseno con tutti i core terms:
     sim = vecs_bg[i] @ vecs_core.T   → vettore di 397 valori

  2. Trova gli indici dei k più alti:
     top_k_idx = np.argsort(sim)[-k:][::-1]   → 7 indici

  3. Prendi le etichette di quei k core terms:
     top_k_labels = labels[top_k_idx]

  4. Conta le occorrenze per dominio → majority vote:
     domain_counts = Counter(top_k_labels)
     assigned = domain_counts.most_common(1)[0][0]

  5. Calcola confidence:
     confidence = domain_counts[assigned] / k

Output per ogni term: assigned_domain, confidence, top_k_idx
```

**Perché k=7**

k=7 è dispari: elimina i pareggi perfetti tra due classi. Produce una scala
di confidence naturale:
- 7/7 = 1.00 → unanimità (segnale geometrico fortissimo)
- 4/7 ≈ 0.57 → maggioranza minima (segnale sufficiente)
- 3/7 ≈ 0.43 → minoranza relativa (ambiguità genuina → revisione giuridica obbligatoria)
- 2/7 ≈ 0.29 → massima dispersione (il modello non sa rispondere)

I termini con confidence < 4/7 non vengono scartati — sono i più interessanti:
il modello geometrico è incerto, il giurista decide. Questi candidati
corrispondono ai "boundary objects" di §4.1.

**Sensibilità al parametro k**

La scelta di k=7 viene verificata con k=5 e k=9 come robustness check (§2.4):
se le assegnazioni cambiano radicalmente al variare di k, il segnale è instabile.

**`np.argsort()`**

Restituisce gli indici che ordinerebbero l'array dal più piccolo al più grande.
`[-k:]` prende gli ultimi k (i più grandi). `[::-1]` li inverte (dal più grande
al più piccolo). Risultato: indici dei k vicini più simili, ordinati per similarità.

```python
sim = np.array([0.3, 0.9, 0.7, 0.85, 0.6])
np.argsort(sim)           # [0, 4, 2, 3, 1]  ← dal più piccolo
np.argsort(sim)[-3:]      # [2, 3, 1]  ← ultimi 3 (i più grandi)
np.argsort(sim)[-3:][::-1]  # [1, 3, 2]  ← dal più grande: indici 1(0.9), 3(0.85), 2(0.7)
```

**`collections.Counter`**

Conta le occorrenze in una lista.

```python
from collections import Counter

top_k_labels = ['civil', 'civil', 'criminal', 'civil', 'civil', 'civil', 'criminal']
counts = Counter(top_k_labels)
# Counter({'civil': 5, 'criminal': 2})
counts.most_common(1)
# [('civil', 5)]
counts.most_common(1)[0][0]
# 'civil'   ← il dominio assegnato
```

---

---

## Estrazione per layer §3.1.3

Le sezioni seguenti descrivono come si estraggono le rappresentazioni intermedie
di un termine a ogni layer del transformer, prerequisito per tutta l'analisi di
Lens III (§3.1.3).

**Notazione per tutta la sezione §3.1.3**

```
N   = numero di termini core = 397
L   = numero di transformer layer (24 per i modelli large, 12 per Dmeta-ZH)
d   = dimensione dell'embedding (1024 per BGE-EN/BGE-ZH, 768 per gli altri)
t   = indice del termine, t ∈ {0, ..., N-1}
l   = indice del layer, l ∈ {0, ..., L}   (L+1 stati totali)

h_t^l  ∈ ℝ^d     = vettore del termine t al layer l, dopo pooling e L2-norm
H^l    ∈ ℝ^{N×d} = matrice di tutti i termini al layer l (riga t = h_t^l)
```

La struttura dati fondamentale è il tensore tridimensionale:

```
layer_vecs ∈ ℝ^{N × (L+1) × d}

layer_vecs[t, l, :]  =  h_t^l   (il vettore del termine t al layer l)
layer_vecs[:, l, :]  =  H^l     (tutti i termini al layer l)
```

---

### E1 — `output_hidden_states` e hidden states di un transformer

**Source**: Hugging Face Transformers
**Usata in**: `lens_3_stratigraphy/layer_extraction.py` → `extract_per_layer()`

**Concetto**

Un modello transformer è una pila di *layer* (strati). Ogni layer riceve la
rappresentazione dal layer precedente, la trasforma, e la passa al successivo.
L'output "normale" di un modello è ciò che esce dall'ultimo layer. Ma ogni
layer intermedio produce anch'esso una rappresentazione — è questo che ci
interessa.

**La catena computazionale di un transformer encoder**

Formalmente, il modello è una composizione di funzioni:

```
Input testo:     "habeas corpus"
                     ↓
Tokenizer:       [101, 2457, 3982, 102]           → token IDs
                     ↓
Embedding:       h^0 = Embed(tokens)               → (seq_len, d)
                     ↓
Layer 1:         h^1 = TransformerBlock_1(h^0)      → (seq_len, d)
Layer 2:         h^2 = TransformerBlock_2(h^1)      → (seq_len, d)
  ...
Layer L:         h^L = TransformerBlock_L(h^{L-1})  → (seq_len, d)
```

Ogni `TransformerBlock` contiene:

```
TransformerBlock(x) = FFN(LayerNorm(x + SelfAttention(x)))
```

Dove:
- *Self-Attention*: ogni token "guarda" tutti gli altri token e calcola
  una media pesata delle loro rappresentazioni (i pesi = attention weights)
- *FFN* (Feed-Forward Network): due moltiplicazioni matriciali con una non-
  linearità (ReLU o GELU) — trasforma il vettore "localmente"
- *LayerNorm*: normalizza le attivazioni per stabilità numerica
- Il `+` è una *connessione residua*: il layer aggiunge alla rappresentazione
  precedente anziché sostituirla. Questo è cruciale per il drift: se un layer
  non "impara" nulla di utile, la connessione residua garantisce che il vettore
  passi quasi inalterato → drift ≈ 0.

**Analogia per il giurista**

Un testo legislativo passa attraverso fasi interpretative: letterale, sistematica,
teleologica, per principi generali (i canoni di Tarello). A ogni fase il significato
si arricchisce. I layer di un transformer funzionano in modo analogo: i primi
catturano informazioni superficiali (ortografia, struttura sintattica), i successivi
catturano informazioni semantiche e relazionali. La domanda di §3.1.3 è:
*a quale profondità emerge il significato giuridico?*

**Come si attivano**

```python
auto_model.config.output_hidden_states = True
outputs = auto_model(**encoded)

outputs.hidden_states   # tupla di L+1 tensori
# hidden_states[0]  = output dell'embedding layer (layer 0)      = h^0
# hidden_states[1]  = output del primo transformer layer         = h^1
# ...
# hidden_states[L]  = output dell'ultimo layer                   = h^L
```

Per un modello con L=24 layer (es. BGE-EN-large), si ottengono 25 tensori:
il layer di embedding (layer 0) + 24 layer transformer.

**Forma di ogni tensore**: `(batch_size, sequence_length, dim)`

Dove:
- `batch_size` = numero di termini processati insieme (nel nostro caso, lotti di 32)
- `sequence_length` = lunghezza in token (con padding alla lunghezza massima nel batch)
- `dim` = dimensione del vettore (768 o 1024)

**Perché L+1 e non L?**

Il layer 0 non è un "vero" transformer layer. È il *layer di embedding*:
la lookup table che converte ogni token nel suo vettore iniziale, prima che
qualsiasi self-attention operi su di esso. Ci interessa come punto di partenza:
il vettore h_t^0 contiene solo informazione lessicale (quale parola è), mentre
h_t^L contiene informazione semantica contestuale (cosa significa in relazione
agli altri concetti).

**I modelli e i loro layer**

| Modello | L | L+1 stati | dim |
|---|---|---|---|
| BGE-EN-large | 24 | 25 | 1024 |
| E5-large | 24 | 25 | 1024 |
| FreeLaw-EN | 22 | 23 | 768 |
| BGE-ZH-large | 24 | 25 | 1024 |
| Text2vec-large-ZH | 24 | 25 | 1024 |
| Dmeta-ZH | 12 | 13 | 768 |

---

### E2 — Pooling: CLS vs Mean

**Source**: `layer_extraction.py` → `_pool_hidden_state()`, `_detect_pooling()`
**Usata in**: trasformazione da (batch, seq_len, dim) a (batch, dim)

**Il problema**

Ogni layer produce un tensore `(batch, seq_len, dim)`: un vettore per *ogni
token* nella sequenza. Ma noi abbiamo bisogno di *un vettore per termine*.
Il passaggio da molti vettori (uno per token) a un singolo vettore si chiama
*pooling*.

**Due strategie principali**

**CLS pooling**: prendi il vettore del token speciale `[CLS]`, che i modelli
BERT-like inseriscono all'inizio di ogni sequenza. Durante il training, questo
token è addestrato a "riassumere" l'intera sequenza.

Notazione: dato il tensore hidden `H ∈ ℝ^{B × S × d}` (B = batch, S = seq_len):

```
CLS:   pool(H) = H[:, 0, :]    → ℝ^{B × d}
```

Si seleziona la posizione 0 (il token [CLS]) per ogni elemento del batch.

```
Sequenza:    [CLS]  habeas  corpus  [SEP]  [PAD]  [PAD]
Indice:         0      1       2      3      4      5
Vettori:       v₀     v₁      v₂     v₃    v₄     v₅
                ↑
         pool = v₀
```

```python
pooled = hidden[:, 0, :]    # → (batch, dim)
```

**Mean pooling**: media di tutti i vettori dei token reali (esclusi i token
di padding). Il razionale: il significato della sequenza è distribuito su
tutti i token, non concentrato in `[CLS]`.

Notazione formale. Sia `m_i ∈ {0, 1}` la *attention mask* del token i
(1 = token reale, 0 = padding):

```
Mean:  pool(H, m) = (Σ_{i: m_i=1} v_i) / (Σ_i m_i)
```

Esempio numerico con dim=3 per chiarezza:

```
Sequenza:    [CLS]  habeas  corpus  [SEP]  [PAD]  [PAD]
Maschera m:     1      1       1      1      0      0
Vettori:     [0.2]  [0.4]   [0.6]  [0.8]  [0.0]  [0.0]
             [0.1]  [0.3]   [0.5]  [0.7]  [0.0]  [0.0]
             [0.9]  [0.1]   [0.2]  [0.3]  [0.0]  [0.0]

Σ token reali:  [0.2+0.4+0.6+0.8, 0.1+0.3+0.5+0.7, 0.9+0.1+0.2+0.3]
              = [2.0, 1.6, 1.5]

Conteggio reali: 4

pool = [2.0/4, 1.6/4, 1.5/4] = [0.5, 0.4, 0.375]
```

**Implementazione PyTorch riga per riga**

```python
# hidden: (batch, seq_len, dim)
# attention_mask: (batch, seq_len), valori 0/1

mask = attention_mask.unsqueeze(-1).float()
# .unsqueeze(-1) aggiunge una dimensione: (batch, seq_len) → (batch, seq_len, 1)
# Questo permette il broadcasting con hidden (batch, seq_len, dim)
# Dopo: mask ha forma (batch, seq_len, 1)

summed = (hidden * mask).sum(dim=1)
# hidden * mask: multiplica ogni vettore token per 0 o 1.
# I PAD diventano vettori zero → non contribuiscono alla somma.
# .sum(dim=1): somma lungo la dimensione seq_len
# Risultato: (batch, dim)

counts = mask.sum(dim=1).clamp(min=1e-9)
# Numero di token reali per ogni sequenza nel batch
# .clamp(min=1e-9): se una sequenza fosse vuota (0 token reali, impossibile
#   ma prevenuto), la divisione per zero è impedita
# Risultato: (batch, 1) — un singolo numero per sequenza

pooled = summed / counts
# Media: somma dei vettori reali / numero di token reali
# Broadcasting: (batch, dim) / (batch, 1) → (batch, dim)
```

**Quale modello usa quale?**

| Modello | Pooling | L | dim | Note |
|---|---|---|---|---|
| BGE-EN-large | CLS | 24 | 1024 | |
| E5-large | Mean | 24 | 1024 | prefix: "query: " |
| FreeLaw-EN | Mean | 22 | 768 | |
| BGE-ZH-large | CLS | 24 | 1024 | |
| Text2vec-large-ZH | Mean | 24 | 1024 | |
| Dmeta-ZH | CLS | 12 | 768 | |

La scelta è determinata dal training del modello: usare Mean su un modello
addestrato con CLS (o viceversa) produce vettori qualitativamente diversi.
Per un'estrazione per-layer corretta, il pooling deve essere replicato
*identicamente* a ogni layer.

**Rilevamento automatico**

```python
def _detect_pooling(model):
    pooling_layer = model[1]   # SentenceTransformer è una Sequential:
                                # model[0] = Transformer, model[1] = Pooling
    if pooling_layer.pooling_mode_cls_token:
        return "cls"
    return "mean"
```

SentenceTransformer espone la strategia di pooling come attributo booleano:
la leggiamo e la replichiamo identicamente a ogni layer.

**Perché il pooling a ogni layer (e non solo al layer finale)?**

Normalmente i modelli applicano il pooling solo all'output del layer finale.
Noi lo applichiamo anche ai layer intermedi per avere un vettore h_t^l per
ogni (termine, layer). Senza pooling, avremmo un tensore per-token e non
potremmo calcolare distanze coseno tra *termini* a layer diversi.

Applicare lo stesso pooling a ogni layer è un'approssimazione: i layer
intermedi non sono stati ottimizzati per produrre vettori utili sotto quel
pooling. Ma è l'approssimazione standard in letteratura (Ethayarajh 2019,
Voita et al. 2019) e l'unica praticabile senza ri-addestrare il modello.

---

### E3 — Normalizzazione L2 per layer

**Source**: `layer_extraction.py` → `_pool_hidden_state()`
**Usata in**: dopo il pooling, a ogni layer

**La formula**

Dato un vettore v ∈ ℝ^d dopo il pooling:

```
‖v‖₂ = √(Σ_{j=1}^{d} v_j²)

v̂ = v / ‖v‖₂
```

Dopo la normalizzazione: ‖v̂‖₂ = 1 esattamente.

**Esempio numerico (d=4)**

```
v     = [3.0,  4.0,  0.0,  0.0]
‖v‖₂  = √(9 + 16 + 0 + 0) = √25 = 5.0
v̂     = [3/5,  4/5,  0/5,  0/5] = [0.6,  0.8,  0.0,  0.0]

Verifica: ‖v̂‖₂ = √(0.36 + 0.64 + 0 + 0) = √1.0 = 1.0  ✓
```

**Conseguenza fondamentale per la similarità coseno**

La similarità coseno tra due vettori è definita come:

```
cos(u, v) = (u · v) / (‖u‖₂ × ‖v‖₂)
```

Se entrambi i vettori sono L2-normalizzati (‖û‖ = ‖v̂‖ = 1):

```
cos(û, v̂) = (û · v̂) / (1 × 1) = û · v̂ = Σ_j û_j × v̂_j
```

La similarità coseno diventa un semplice *prodotto scalare* (dot product).
Questo ha due implicazioni:

1. L'intera matrice di similarità N×N si calcola con una sola moltiplicazione
   matriciale: `sim = H^l @ (H^l)^T` (vedi R1)
2. La distanza coseno è `1 - û · v̂` = `1 - sim[i,j]`

**Perché normalizzare a ogni layer (e non solo al layer finale)?**

Senza normalizzazione, le norme dei vettori crescono spostandosi dal layer 0
al layer L. Esempio misurato su BGE-EN-large:

```
‖h_t^0‖₂  ≈  4.2    (layer embedding)
‖h_t^12‖₂ ≈  8.7    (layer intermedio)
‖h_t^24‖₂ ≈ 15.3    (layer finale)
```

Se calcolassimo `1 - cos(h_t^{12}, h_t^{13})` con vettori non normalizzati,
il risultato dipenderebbe sia dalla *direzione* (che ci interessa) sia dalla
*norma* (che non ci interessa). La normalizzazione L2 isola la componente
direzionale, rendendo confronti tra layer validi.

Formalmente: la distanza coseno normalizzata è invariante alla scala dei vettori.
Senza normalizzazione, `1 - (u · v)/(‖u‖‖v‖)` è equivalente ma numericamente
meno stabile (divisione per prodotto di norme grandi → perdita di precisione float32).

**Implementazione PyTorch**

```python
norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
pooled = pooled / norms
```

Riga per riga:
- `pooled.norm(dim=1, keepdim=True)`: calcola ‖v‖₂ per ogni riga (termine),
  mantenendo la dimensione per broadcasting. Forma: (batch, 1)
- `.clamp(min=1e-9)`: previene divisione per zero (mai osservato in pratica,
  ma necessario per correttezza numerica)
- `pooled / norms`: broadcasting (batch, dim) / (batch, 1) → normalizzazione

**Tipo di dato**: il risultato è convertito a `float32` (`.astype(np.float32)`).
La precisione di float32 (23 bit di mantissa, ~7 cifre decimali) è sufficiente
per le operazioni di distanza coseno e Spearman. Float64 raddoppierebbe la
memoria senza beneficio misurabile.

---

### E4 — `np.savez_compressed()` e caching

**Source**: NumPy
**Usata in**: `layer_extraction.py` → `extract_per_layer()`

**Firma**
```python
np.savez_compressed(path, layers=result)
# ...
data = np.load(path)
layers = data["layers"]
```

**Cosa fa**

Salva un array NumPy su disco in formato compresso (ZIP + .npy interni).
A differenza di `np.save()` (che produce un `.npy` non compresso),
`savez_compressed` applica compressione deflate (zlib) a ciascun array.

**Dimensioni tipiche**

```
Per BGE-EN-large:
  N=397 × (L+1)=25 × d=1024 × 4 byte (float32)
  = 397 × 25 × 1024 × 4
  = 40,601,600 byte ≈ 38.7 MB non compresso

Per Dmeta-ZH:
  N=397 × (L+1)=13 × d=768 × 4 byte
  = 397 × 13 × 768 × 4
  = 15,851,520 byte ≈ 15.1 MB non compresso
```

La compressione riduce del ~35% grazie alla correlazione tra layer adiacenti
(connessione residua → valori simili → alta comprimibilità).

**Il ciclo cache**

```
extract_per_layer(label, terms):
  1. Cerca {label}.npz in results/layer_vectors/
  2. Se esiste E la prima dimensione == len(terms) → return (cache hit)
  3. Se non esiste → forward pass → salva .npz → return
```

Il check `layers.shape[0] == len(terms)` previene il caso in cui il cache
contenga dati per un numero diverso di termini (es. una run precedente con
termini background inclusi).

Il parametro `force=True` ignora il cache e forza la ri-estrazione.

---

### E5 — Sanity check: layer finale vs precomputed

**Source**: `layer_extraction.py` → `verify_final_layer()`
**Usata in**: dopo ogni estrazione, prima di qualsiasi analisi

**Concetto**

Il layer finale dell'estrazione per-layer DEVE coincidere con i vettori
pre-calcolati da `shared/precompute.py` (usati in Lens I). Se non coincidono,
c'è un errore nel pooling, nella tokenizzazione, o nel prefix di istruzione
— e tutti i risultati per-layer sarebbero inaffidabili.

La verifica avviene termine per termine su tutti i 397 core terms.

**Criterio primario: `np.allclose()`**

```python
np.allclose(final, vecs_core, atol=1e-4)
```

`allclose` implementa:

```
∀ t ∈ {0,...,N-1}, ∀ j ∈ {0,...,d-1}:
  |h_t^L[j] − precomputed_t[j]| ≤ atol

dove atol = 10⁻⁴ = 0.0001
```

Per N=397, d=1024: 397 × 1024 = 406.528 confronti individuali.
Se anche uno solo fallisce, `allclose` restituisce False.

**Perché atol=10⁻⁴ e non più stretto?**

Float32 ha ~7 cifre di precisione. Le operazioni nel forward pass (matmul,
softmax, layernorm) accumulano errore numerico. Su CPU l'accumulo resta
ampiamente sotto 10⁻⁴; su GPU/MPS può essere più grande a causa di
operazioni fused (es. flash attention) che sacrificano precisione per velocità.

**Criterio secondario: similarità coseno (fallback)**

Se `allclose` fallisce (differenze assolute > 10⁻⁴), si verifica che la
*direzione* dei vettori sia comunque allineata:

```python
cos_sims = np.sum(final * vecs_core, axis=1)   # (N,) — un coseno per termine
```

Questa riga calcola:

```
cos_t = Σ_{j=1}^{d} h_t^L[j] × precomputed_t[j]
```

Poiché entrambi i vettori sono L2-normalizzati (‖·‖ = 1), questo è il
prodotto scalare = la similarità coseno.

```python
min_cos = cos_sims.min()
if min_cos > 0.999:     # match accettabile
```

Il valore 0.999 corrisponde a un angolo tra i vettori di:

```
θ = arccos(0.999) ≈ 0.045 radianti ≈ 2.56°
```

Un angolo di 2.56° in uno spazio a 1024 dimensioni è una deviazione
trascurabile per le analisi di distanza e rank-order.

**Dove interviene nella pipeline**

```
Per ogni modello m ∈ {BGE-EN, E5, FreeLaw, BGE-ZH, Text2vec, Dmeta}:
  1. extract_per_layer(m, terms, device="cpu")  → layer_vecs  (N, L+1, d)
  2. verify_final_layer(m, layer_vecs, core_idx)
       → confronta layer_vecs[:, -1, :] con precomputed[core_idx]
       → se FAIL: abort con diagnostica (max_diff, mean_diff, min_cos)
  3. Solo se la verifica passa → procedi con drift, Jaccard, ecc.
```

**Nota sulla non-determinism MPS**

Il chip Apple MPS (Metal Performance Shaders) produce risultati non-deterministici
quando la composizione del batch cambia. Osservazione empirica:

```
encode(testi_397, batch_size=64)  ≠  encode(testi_397, batch_size=32)  su MPS
→ min cosine similarity = 0.23 per BGE-ZH-large (!)

encode(testi_397, batch_size=64)  ≈  encode(testi_397, batch_size=32)  su CPU
→ min cosine similarity = 0.999999
```

Tutti i vettori del progetto (precomputed e layer vectors) sono stati
rigenerati su CPU, dove il sanity check è esatto (atol=1e-4).

---

## Analisi §3.1.3a — Drift e Jaccard per layer

Le sezioni seguenti spiegano le due metriche che misurano il *comportamento
del singolo termine* man mano che attraversa i layer del transformer.

L'output di questa sezione è una matrice per metrica:

```
drift   ∈ ℝ^{N × L}     dove drift[t, l] = distanza tra layer l e layer l+1
jaccard ∈ ℝ^{N × L}     dove jaccard[t, l] = instabilità del vicinato tra l e l+1
```

Nota: L transizioni (non L+1): tra L+1 stati ci sono L transizioni consecutive.
Per BGE-EN-large: 397 × 24 = 9.528 valori per matrice.

---

### D1 — Cosine drift tra layer consecutivi

**Source**: `lens_3_stratigraphy/lens3.py` → `_compute_drift()`
**Usata in**: §3.1.3a — quantificare quanto cambia la rappresentazione di un
termine passando da un layer al successivo

**Il concetto**

Il *drift* misura quanto il vettore di un termine si sposta passando dal
layer l al layer l+1. Un drift alto significa che quel layer sta modificando
radicalmente la rappresentazione; un drift basso significa che il layer sta
facendo aggiustamenti minimi.

**La formula**

```
drift(t, l) = d_cos(h_t^l, h_t^{l+1}) = 1 − cos(h_t^l, h_t^{l+1})
```

Dove:
- `h_t^l` ∈ ℝ^d = vettore del termine t al layer l, L2-normalizzato (‖h_t^l‖ = 1)
- `cos(u, v)` = similarità coseno = `(u · v) / (‖u‖ × ‖v‖)`
- Per vettori normalizzati: `cos(u, v) = u · v = Σ_{j=1}^{d} u_j × v_j`
- `d_cos` = distanza coseno = `1 − cos` ∈ [0, 2]

**`scipy.spatial.distance.cosine`**

```python
from scipy.spatial.distance import cosine as cosine_distance
```

La funzione SciPy calcola esattamente:

```python
cosine_distance(u, v) = 1.0 - (u @ v) / (norm(u) * norm(v))
```

Per vettori L2-normalizzati (‖u‖ = ‖v‖ = 1) questo si riduce a:

```python
cosine_distance(u, v) = 1.0 - (u @ v)     # semplice: 1 − dot product
```

**Perché `scipy.spatial.distance.cosine` e non direttamente `1 - u @ v`?**

Il risultato è identico per vettori normalizzati. Si usa SciPy per:
1. Chiarezza: la funzione si chiama esplicitamente "cosine distance"
2. Robustezza: se per errore un vettore non fosse normalizzato, SciPy
   normalizza internamente → risultato corretto comunque
3. Evita errori di segno (`1 - sim` vs `sim - 1`)

**Esempio numerico (d=4)**

```
h_t^5 = [0.50, 0.50, 0.50, 0.50]    (layer 5, normalizzato: ‖·‖ = 1.0)
h_t^6 = [0.48, 0.52, 0.51, 0.49]    (layer 6, normalizzato: ‖·‖ = 1.0)

cos(h_t^5, h_t^6) = 0.50×0.48 + 0.50×0.52 + 0.50×0.51 + 0.50×0.49
                   = 0.240 + 0.260 + 0.255 + 0.245
                   = 1.000

drift(t, 5) = 1 − 1.000 = 0.000   (il vettore quasi non si è mosso)
```

Secondo esempio con drift significativo:

```
h_t^5 = [0.50, 0.50, 0.50, 0.50]
h_t^6 = [0.70, 0.30, 0.50, 0.40]    (normalizzato: ‖·‖ ≈ 1.0)

cos = 0.50×0.70 + 0.50×0.30 + 0.50×0.50 + 0.50×0.40
    = 0.350 + 0.150 + 0.250 + 0.200 = 0.950

drift(t, 5) = 1 − 0.950 = 0.050   (trasformazione significativa)
```

**Valori tipici osservati nel progetto**

```
drift ≈ 0.001  → layer quasi-identità (connessione residua dominante)
drift ≈ 0.01   → modifiche lievi (tipico dei primi layer: sintassi)
drift ≈ 0.05   → trasformazione significativa (tipico dei layer intermedi)
drift > 0.10   → rielaborazione radicale (il picco di drift, 1-2 layer)
drift > 0.50   → non osservato; indicherebbe un bug
```

**Intuizione per il giurista**

Il drift è il "passo" che un concetto compie a ogni fase di elaborazione.
Se "buona fede" fa un grande passo tra il layer 15 e il 16, quel layer sta
aggiungendo informazione semantica cruciale. La curva drift[t, :] è la
traccia del percorso di un singolo termine attraverso lo spazio vettoriale.

**Implementazione completa**

```python
def _compute_drift(layer_vecs):
    # layer_vecs: (N, L+1, dim), dtype float32, L2-normalized per layer
    n_terms, n_states, _ = layer_vecs.shape
    n_transitions = n_states - 1    # L transizioni tra L+1 stati
    drift = np.zeros((n_terms, n_transitions), dtype=np.float32)

    for t in range(n_terms):            # per ogni termine
        for l in range(n_transitions):  # per ogni transizione l → l+1
            drift[t, l] = cosine_distance(
                layer_vecs[t, l],       # h_t^l   ∈ ℝ^d
                layer_vecs[t, l + 1],   # h_t^{l+1} ∈ ℝ^d
            )
            # cosine_distance = 1 - (h_t^l · h_t^{l+1}) / (‖h_t^l‖ × ‖h_t^{l+1}‖)
            # = 1 - h_t^l · h_t^{l+1}     (perché entrambi L2-normalizzati)

    return drift   # (N, L) matrice, dtype float32
```

**Costo**: N × L chiamate a `cosine_distance`, ciascuna O(d).
Totale: O(N × L × d) = 397 × 24 × 1024 ≈ 9.8M operazioni. Trascurabile (< 1s).

**Output**: matrice (N, L) — una riga per termine, una colonna per
transizione. Per BGE-EN-large con L=24: matrice 397×24.

---

### D2 — k-NN Jaccard instability tra layer consecutivi

**Source**: `lens_3_stratigraphy/lens3.py` → `_compute_jaccard()`
**Usata in**: §3.1.3a — misurare la stabilità del vicinato semantico
attraverso i layer

**Il concetto**

Il drift (D1) misura quanto si muove il vettore di un termine nel punto.
Ma un termine potrebbe muoversi *tanto* restando comunque tra gli stessi
vicini (come chi corre sul posto). La Jaccard instability misura qualcosa
di diverso: *i vicini di un termine cambiano tra un layer e il successivo?*

È una metrica *topologica* (sensibile alla struttura locale) anziché
*metrica* (sensibile alla posizione).

**La formula**

L'indice di Jaccard misura la sovrapposizione tra due insiemi. Noi lo usiamo
come *distanza* (1 − overlap):

```
J(t, l) = 1 − |kNN(t,l) ∩ kNN(t,l+1)| / |kNN(t,l) ∪ kNN(t,l+1)|
```

Dove:
- `kNN(t,l)` = {i₁, i₂, ..., i_k} = insieme degli indici dei k termini più
  simili al termine t al layer l (escludendo t stesso)
- `∩` = intersezione: elementi presenti in *entrambi* gli insiemi
- `∪` = unione: elementi presenti in *almeno uno* dei due insiemi
- k = 7 (stesso parametro di Lens I, → B1)

**Proprietà matematiche dell'indice di Jaccard**

```
Siano A, B due insiemi finiti non vuoti.

J_sim(A, B) = |A ∩ B| / |A ∪ B|     ∈ [0, 1]     (similarità)
J_dist(A, B) = 1 − J_sim(A, B)       ∈ [0, 1]     (distanza)
```

Proprietà:
- J_dist è una *metrica* (soddisfa identità, simmetria, disuguaglianza triangolare)
- |A ∪ B| = |A| + |B| − |A ∩ B|
- Se |A| = |B| = k: |A ∪ B| varia tra k (se A = B) e 2k (se A ∩ B = ∅)

**Possibili valori con k=7**

```
|kNN_l ∩ kNN_{l+1}|    |kNN_l ∪ kNN_{l+1}|    J(t,l)
        7                      7                0.000  (vicinato identico)
        6                      8                0.250
        5                      9                0.444
        4                     10                0.600
        3                     11                0.727
        2                     12                0.833
        1                     13                0.923
        0                     14                1.000  (nessun vicino in comune)
```

Con k=7: solo 8 valori discreti possibili per termine per transizione.
La media su 397 termini produce una distribuzione quasi continua.

**Esempio numerico completo**

```
5 termini: contratto(0), obbligo(1), reato(2), dolo(3), proprietà(4)
k = 3 (per semplicità)

Similarità al layer l:
             contr   obbl    reato   dolo    prop
  contratto  [—      0.91    0.30    0.25    0.85]
  obbligo    [0.91   —       0.35    0.20    0.80]
  reato      [0.30   0.35    —       0.90    0.28]
  dolo       [0.25   0.20    0.90    —       0.22]
  proprietà  [0.85   0.80    0.28    0.22    —   ]

kNN("contratto", l) = {obbligo(0.91), proprietà(0.85), reato(0.30)} = {1, 4, 2}

Similarità al layer l+1 (dopo la trasformazione):
             contr   obbl    reato   dolo    prop
  contratto  [—      0.88    0.32    0.28    0.82]
  ...

kNN("contratto", l+1) = {obbligo(0.88), proprietà(0.82), dolo(0.28)} = {1, 4, 3}
     ↑ reato uscito, dolo entrato

Intersezione: {1, 4} ∩ {1, 4, 3} → no: {1, 4, 2} ∩ {1, 4, 3} = {1, 4} → 2
Unione: {1, 4, 2} ∪ {1, 4, 3} = {1, 2, 3, 4} → 4

J("contratto", l) = 1 − 2/4 = 0.500
```

**Implementazione riga per riga**

```python
def _compute_jaccard(layer_vecs, k=7):
    # layer_vecs: (N, L+1, dim), L2-normalized
    n_terms, n_states, _ = layer_vecs.shape
    n_transitions = n_states - 1
    jaccard = np.zeros((n_terms, n_transitions), dtype=np.float32)

    for l in range(n_transitions):
        # 1. Matrice di similarità coseno al layer l: (N, N)
        # Poiché i vettori sono L2-normalizzati, sim = dot product
        sim_l  = layer_vecs[:, l, :]   @ layer_vecs[:, l, :].T
        # layer_vecs[:, l, :] ha forma (N, d)
        # layer_vecs[:, l, :].T ha forma (d, N)
        # risultato: (N, N), dove sim_l[i,j] = cos(h_i^l, h_j^l)

        sim_l1 = layer_vecs[:, l+1, :] @ layer_vecs[:, l+1, :].T
        # stessa operazione per il layer l+1

        for t in range(n_terms):
            # 2. Escludi il termine t da se stesso
            s_l = sim_l[t].copy()      # vettore (N,): sim di t con tutti
            s_l[t] = -np.inf           # t non può essere vicino di se stesso
            s_l1 = sim_l1[t].copy()
            s_l1[t] = -np.inf

            # 3. Trova i k indici con similarità più alta
            nn_l  = set(np.argsort(s_l)[-k:])    # set di k indici
            nn_l1 = set(np.argsort(s_l1)[-k:])

            # np.argsort restituisce indici ordinati dal più piccolo al
            # più grande. [-k:] prende gli ultimi k (i più grandi).
            # -np.inf è il valore più piccolo possibile → l'indice t
            # non comparirà mai tra gli ultimi k.

            # 4. Calcola la distanza di Jaccard
            intersection = len(nn_l & nn_l1)    # operazione su set Python
            union = len(nn_l | nn_l1)
            jaccard[t, l] = 1.0 - intersection / union if union > 0 else 0.0
            # union > 0 è sempre vero con k ≥ 1, ma preveniamo div/0

    return jaccard   # (N, L) matrice, dtype float32
```

**Perché `set()` e non `np.intersect1d()`?**

Con k=7: gli insiemi hanno al massimo 7 elementi. Le operazioni su `set`
Python (`&`, `|`, `len`) sono O(k) e istantanee. `np.intersect1d` ha overhead
di conversione a array che dominerebbe il tempo per insiemi così piccoli.

**Perché `-np.inf` per l'auto-esclusione?**

La similarità coseno di un vettore L2-normalizzato con se stesso è esattamente
1.0 (il massimo possibile). Senza esclusione, il termine t sarebbe sempre il
proprio vicino più simile. Impostare `s_l[t] = -np.inf` garantisce che:
- `np.argsort` lo piazza all'indice 0 (il più piccolo)
- `[-k:]` non lo seleziona mai (a meno che k ≥ N, impossibile con k=7 e N=397)

**Differenza rispetto al drift (D1)**

| Proprietà | Drift (D1) | Jaccard (D2) |
|---|---|---|
| Formula | `1 − h_t^l · h_t^{l+1}` | `1 − |kNN_l ∩ kNN_{l+1}| / |kNN_l ∪ kNN_{l+1}|` |
| Tipo | Distanza metrica continua | Distanza topologica discreta |
| Dominio | [0, 2] | {0, 0.25, 0.44, ..., 1.0} (discreto con k=7) |
| Sensibilità | Alla direzione del punto t | Alla struttura del vicinato di t |
| Un termine si muove molto, stessi vicini | Alto | Basso |
| Vicinato si riorganizza, movimenti piccoli | Basso | Alto |

I due sono complementari: drift = velocità del singolo punto, Jaccard =
stabilità della struttura locale.

**Costo computazionale**

Per ogni transizione l:
- 2 moltiplicazioni matriciali (N×d) × (d×N) = O(N²d) ciascuna
- N sort di vettori lunghi N = O(N² log N)
- N operazioni su set di k elementi = O(Nk) — trascurabile

Totale per modello: O(L × N²d) = 24 × 397² × 1024 ≈ 3.9 × 10⁹ operazioni.
NumPy esegue le matmul con BLAS ottimizzato → ~10s per modello su CPU.

---

### D3 — Aggregazione per dominio: media condizionata

**Source**: `lens_3_stratigraphy/lens3.py` → `_aggregate_by_domain()`
**Usata in**: §3.1.3a — produrre curve per-dominio di drift e Jaccard

**Il concetto**

Le matrici drift e Jaccard hanno 397 righe (una per termine). Per il confronto
tra domini, aggreghiamo: per ogni dominio d, calcoliamo la media dei valori
dei termini che appartengono a quel dominio, layer per layer.

**La formula**

Sia D_d = {t : domain(t) = d} l'insieme dei termini del dominio d, e |D_d|
la sua cardinalità. Per una matrice M ∈ ℝ^{N × L} (drift o Jaccard):

```
M̄_d[l] = (1 / |D_d|) × Σ_{t ∈ D_d} M[t, l]     per l = 0, ..., L-1
```

Questo produce un vettore M̄_d ∈ ℝ^L per ogni dominio — la curva media
di quel dominio attraverso i layer.

**Esempio numerico (3 domini, 3 transizioni)**

```
Dominio "civil" (|D| = 2):
  contratto:  [0.020, 0.050, 0.030]   (drift per transizione 0→1, 1→2, 2→3)
  proprietà:  [0.030, 0.040, 0.020]

  M̄_civil = [(0.020+0.030)/2, (0.050+0.040)/2, (0.030+0.020)/2]
           = [0.025, 0.045, 0.025]

Dominio "criminal" (|D| = 2):
  reato:  [0.010, 0.080, 0.040]
  dolo:   [0.020, 0.070, 0.050]

  M̄_criminal = [0.015, 0.075, 0.045]

Dominio "constitutional" (|D| = 1):
  sovranità:  [0.040, 0.030, 0.010]

  M̄_constitutional = [0.040, 0.030, 0.010]   (un solo termine → nessuna media)
```

**Implementazione con NumPy boolean masking**

```python
def _aggregate_by_domain(matrix, domains):
    # matrix: (N, L), dtype float32
    # domains: lista di N stringhe
    dom_arr = np.array(domains)
    result = {}
    for d in sorted(set(domains)):
        mask = (dom_arr == d)
        # mask è un array booleano (N,):
        #   [True, False, False, True, ...] per il dominio d
        #
        # matrix[mask] seleziona solo le righe dove mask=True:
        #   se mask ha k True → sottomatrice (k, L)
        #
        # .mean(axis=0) media lungo le righe → vettore (L,)
        result[d] = matrix[mask].mean(axis=0).tolist()
    return result
    # output: {"civil": [0.025, 0.045, ...], "criminal": [...], ...}
```

**`np.array(domains)` — perché convertire?**

Le operazioni di confronto element-wise (`==`) su liste Python producono un
singolo booleano (l'intera lista è uguale?). Su array NumPy producono un
array booleano element-wise — esattamente il mask di cui abbiamo bisogno.

```python
domains = ["civil", "criminal", "civil", "criminal", "const"]

# Lista Python:
domains == "civil"       # → False (confronta l'intera lista con la stringa)

# Array NumPy:
np.array(domains) == "civil"  # → [True, False, True, False, False]
```

**Scopo**: le curve per-dominio permettono di verificare se domini diversi
hanno profili di drift diversi (es. il diritto penale "matura" a layer
diversi rispetto al diritto civile).

---

### D4 — Top-N terms per drift totale

**Source**: `lens_3_stratigraphy/lens3.py` → `_top_drift_terms()`
**Usata in**: §3.1.3a — identificare i termini con la maggiore trasformazione

**La formula**

Il drift totale di un termine t è la somma delle distanze coseno su tutte le
transizioni:

```
total_drift(t) = Σ_{l=0}^{L-1} drift(t, l) = Σ_{l=0}^{L-1} [1 − cos(h_t^l, h_t^{l+1})]
```

Geometricamente: la *lunghezza del percorso* del termine t nello spazio
vettoriale, dove il percorso è la catena di spostamenti layer-by-layer.

Non è la distanza diretta tra layer 0 e layer L (linea retta), ma la somma
dei singoli passi (percorso effettivo). La distinzione è la stessa tra
"chilometri percorsi" e "distanza in linea d'aria".

**Implementazione**

```python
def _top_drift_terms(drift, terms, n=10):
    # drift: (N, L) — matrice completa del drift
    total = drift.sum(axis=1)
    # .sum(axis=1): somma lungo le colonne (i layer) per ogni riga (termine)
    # Risultato: vettore (N,) — un drift totale per termine

    top_idx = np.argsort(total)[-n:][::-1]
    # np.argsort: indici che ordinano dal più piccolo al più grande
    # [-n:]: ultimi n (i più grandi)
    # [::-1]: invertiti (dal più grande al più piccolo)

    return [
        {
            "en": terms[i]["en"],
            "domain": terms[i]["domain"],
            "total_drift": round(float(total[i]), 4),
            "drift_curve": [round(float(d), 4) for d in drift[i]],
        }
        for i in top_idx
    ]
```

**Interpretazione**

Un drift totale alto significa che il termine ha subito una trasformazione
complessiva grande attraverso il transformer. Questi sono i termini la cui
rappresentazione è più lontana dalla lettura lessicale iniziale.

La `drift_curve` (il vettore drift[t, :]) mostra *dove* avviene la
trasformazione: un picco al layer 12 significa che quel layer è particolarmente
attivo per quel termine.

---

## Analisi §3.1.3b — Domain signal emergence e RSA convergence

Le sezioni seguenti descrivono le due metriche che misurano la *struttura
globale* dello spazio a ogni layer, producendo curve layer-by-layer che
mostrano *quando* emerge il significato giuridico.

L'output di questa sezione è, per ogni modello:

```
domain_signal_r  ∈ ℝ^{L+1}    r(l) = Mann-Whitney effect size al layer l
rsa_vs_final     ∈ ℝ^{L+1}    ρ(l) = Spearman(RDM_l, RDM_final)
```

---

### G1 — Domain signal per layer: curva r(l)

**Source**: `lens_3_stratigraphy/lens3.py` → `run_section_313b()`
**Usata in**: §3.1.3b — misurare a quale layer emerge la struttura per dominio

**Il concetto**

In Lens I (§3.1.1) abbiamo misurato il domain signal r al layer finale:
"le distanze intra-dominio sono sistematicamente minori delle inter-dominio?"
(→ A1-A2). Qui eseguiamo lo stesso test a *ogni layer*, producendo una curva
r(l) che traccia l'emergere del segnale giuridico attraverso la profondità.

**L'algoritmo passo per passo**

```
Input: layer_vecs ∈ ℝ^{N × (L+1) × d}, domains (N etichette)

Per ogni layer l = 0, 1, ..., L:

  Passo 1 — Costruisci la RDM al layer l
    H^l = layer_vecs[:, l, :]                  → (N, d)
    sim^l = H^l @ (H^l)^T                      → (N, N), similarità coseno
    RDM^l = 1 − sim^l                           → (N, N), distanza coseno
    np.fill_diagonal(RDM^l, 0.0)                → diagonale azzerata
    tri^l = RDM^l[triu_indices(N, k=1)]         → vettore di N(N-1)/2 valori
                                                   = 78.606 coppie

  Passo 2 — Split intra vs inter dominio
    Per ogni coppia (i, j) nel triangolo superiore:
      se domain(i) == domain(j): → intra_dists
      se domain(i) != domain(j): → inter_dists

  Passo 3 — Mann-Whitney U + rank-biserial r (identico a §3.1.1)
    U, p = mannwhitneyu(intra_dists, inter_dists, alternative="less")
    r(l) = 1 − 2U / (n_intra × n_inter)

  Salva r(l)

Risultato: vettore r = [r(0), r(1), ..., r(L)]   di L+1 valori
```

**La funzione `_intra_inter_split()` in dettaglio**

```python
def _intra_inter_split(rdm, domains):
    n = len(rdm)
    rows, cols = np.triu_indices(n, k=1)
    # rows, cols: indici delle coppie nel triangolo superiore
    # Esempio con N=4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) → 6 coppie

    dom_arr = np.array(domains)
    same = dom_arr[rows] == dom_arr[cols]
    # same: array booleano (78.606,) — True se i due termini hanno stesso dominio
    # dom_arr[rows]: dominio del primo termine in ogni coppia
    # dom_arr[cols]: dominio del secondo termine in ogni coppia
    # L'operatore == element-wise produce il mask

    tri = rdm[rows, cols]
    # tri: le 78.606 distanze coseno delle coppie uniche

    return tri[same], tri[~same]
    # tri[same]:  distanze intra-dominio (coppie dove i due termini sono nello
    #             stesso dominio)
    # tri[~same]: distanze inter-dominio (coppie con domini diversi)
```

**Dimensioni tipiche degli split (N=397, 9 domini)**

```
Coppie totali: 397 × 396 / 2 = 78.606
Coppie intra-dominio: Σ_d |D_d|×(|D_d|-1)/2 ≈ 6.500 (dipende dalla distribuzione)
Coppie inter-dominio: 78.606 − 6.500 ≈ 72.100
```

Il rapporto ~8:92 (intra:inter) riflette il fatto che con 9 domini, la
maggioranza delle coppie è inter-dominio.

**Cosa ci aspettiamo**

```
r(0) ≈ 0       → layer embedding: poca struttura semantica
r(L/4) ≈ 0.1   → inizio della specializzazione
r(L/2) ≈ 0.3   → struttura intermedia
r(3L/4) ≈ 0.5  → forte specializzazione
r(L) ≈ 0.6     → il massimo (corrisponde al valore di Lens I)
```

Se r fosse piatto (≈ costante a tutti i layer), il significato giuridico non
sarebbe *depth-dependent* — sarebbe già nella rappresentazione superficiale.
Questo è il criterio di falsificazione D4 della trace.

**Differenza rispetto a Lens I**

In Lens I: r calcolato una sola volta, al layer finale, con Mantel test per
significatività. Qui: r calcolato L+1 volte, senza Mantel test (troppo costoso
per L+1 ripetizioni, e la significatività per layer non è il focus — il focus
è la *forma della curva*).

---

### G2 — RSA convergence to final: curva ρ(l)

**Source**: `lens_3_stratigraphy/lens3.py` → `run_section_313b()`
**Usata in**: §3.1.3b — misurare a quale layer la struttura relazionale si
stabilizza

**Il concetto**

La curva r(l) misura il segnale di *dominio* (7 etichette discrete). Ma la
struttura dell'RDM contiene molto di più: le distanze tra *tutte* le 78.606
coppie, non solo la differenza media intra vs inter. La curva ρ(l) chiede una
domanda più forte: *a quale layer l'intera geometria relazionale diventa
"uguale" alla geometria finale?*

**La formula**

```
ρ(l) = Spearman(upper_tri(RDM^l), upper_tri(RDM^L))
```

Espanso:

```
Siano:
  x = upper_tri(RDM^l)     ∈ ℝ^{N(N-1)/2}     (78.606 distanze al layer l)
  y = upper_tri(RDM^L)     ∈ ℝ^{N(N-1)/2}     (78.606 distanze al layer finale)

  rx = rank(x)   ← converti in ranghi: il più piccolo → 1, il più grande → 78606
  ry = rank(y)

  ρ = Pearson(rx, ry) = [Σ_i (rx_i − r̄x)(ry_i − r̄y)] / [√(Σ_i(rx_i−r̄x)²) × √(Σ_i(ry_i−r̄y)²)]
```

In pratica: Spearman ρ è la correlazione di Pearson applicata ai ranghi (→ S1).

**Implementazione**

```python
# Calcolo una sola volta, fuori dal loop sui layer:
rdm_final = compute_rdm(layer_vecs[:, -1, :])     # RDM al layer L
tri_final = upper_tri(rdm_final)                    # vettore di 78.606 valori

# Nel loop per l = 0, ..., L:
rdm_l = compute_rdm(layer_vecs[:, l, :])
tri_l = upper_tri(rdm_l)

if l == n_states - 1:
    rho = 1.0    # identità: layer L vs layer L
else:
    rho = float(spearmanr(tri_l, tri_final).statistic)
```

**Il caso l = L**

Al layer finale, `tri_l` e `tri_final` sono lo stesso vettore. Spearman di
un vettore con se stesso è per definizione 1.0. Lo impostiamo direttamente
per evitare il calcolo (e il rischio di errori numerici su pareggi di rango
con 78.606 valori identici).

**Proprietà matematiche della curva ρ(l)**

```
ρ(L) = 1.0         per definizione
ρ(0) ∈ [0.3, 0.6]  tipicamente (embedding vs finale)
```

La curva è *generalmente* monotonicamente crescente, ma non necessariamente:
un layer potrebbe temporaneamente "destrutturare" la rappresentazione prima
di ristrutturarla (dip locale). Questo è stato osservato in letteratura per
layer di attenzione particolarmente aggressivi (Rogers et al. 2020).

**Perché non serve il Mantel test?**

In §3.1.4 (Lens I) confrontavamo RDMs di *modelli diversi addestrati
indipendentemente*: la correlazione potrebbe essere dovuta a proprietà
superficiali condivise (es. frequenza dei termini), non a struttura genuina.
Il Mantel test (permutazione) esclude questo.

Qui confrontiamo *layer dello stesso modello sugli stessi dati*. La
dipendenza non è un confound ma il fenomeno che studiamo: come la
trasformazione layer-by-layer fa convergere la struttura verso la
configurazione finale. Non c'è ipotesi nulla da testare — la curva ρ(l)
è puramente descrittiva.

**Intuizione per il giurista**

Immagina di confrontare una bozza di legge a revisioni successive con il
testo finale promulgato. ρ(l) dice: "a quale revisione la struttura del
testo (l'organizzazione degli articoli, le connessioni tra norme) è
diventata sostanzialmente uguale alla versione finale?" Un ρ alto presto
→ le revisioni successive sono cosmetiche; un ρ basso fino alla fine →
la struttura è stata riscritta radicalmente nelle ultime revisioni.

**Relazione tra r(l) e ρ(l)**

Le due curve non sono indipendenti: se il domain signal r cresce (i domini
si separano), la geometria dell'RDM necessariamente cambia → ρ cresce. Ma
ρ può crescere *senza* che r cresca (es. le distanze all'interno di ciascun
dominio si riorganizzano mantenendo la separazione media costante).

In generale: ρ ≥ f(r) per qualche funzione crescente f, ma la relazione
esatta dipende dalla distribuzione delle distanze.

---

### G3 — Threshold detection: soglie 50% e 90%

**Source**: `lens_3_stratigraphy/lens3.py` → `_find_threshold()`
**Usata in**: §3.1.3b — riassumere la curva ρ(l) in due numeri scalari

**Il problema**

I 6 modelli hanno diversi numeri di layer (L=12, 22, 24). Dire "ρ raggiunge
0.9 al layer 20" non è comparabile con "ρ raggiunge 0.9 al layer 10" senza
sapere quanti layer totali ha il modello. Servono numeri sintetici che
permettano il confronto.

**La formula**

```
τ(θ) = min{l ∈ {0, ..., L} : ρ(l) ≥ θ}

Dove θ è la soglia (0.5 o 0.9)
```

In parole: il primo layer dove la curva ρ(l) supera la soglia θ.

**Implementazione**

```python
def _find_threshold(values, threshold):
    """Linear scan: O(L) — L ≤ 25, trascurabile."""
    for i, v in enumerate(values):
        if v >= threshold:
            return i
    return None    # la soglia non è mai raggiunta (non osservato per θ ≤ 0.9)
```

**Le due soglie e il loro significato**

```
θ = 0.5:   τ(0.5) = primo layer dove la correlazione con la geometria
           finale supera il 50%. La struttura relazionale è "a metà
           strada" → il modello ha iniziato la specializzazione.

θ = 0.9:   τ(0.9) = primo layer dove la correlazione supera il 90%.
           La struttura è "essenzialmente stabilizzata" → i layer
           successivi fanno solo aggiustamenti fini.
```

**Esempio con valori osservati**

```
BGE-EN-large (L=24, 25 stati):
  ρ = [0.32, 0.38, 0.42, 0.48, 0.54, 0.63, 0.71, 0.79, 0.85, 0.90, 0.93, ...]
  τ(0.5) = 4    (primo ρ ≥ 0.5 al layer 4)
  τ(0.9) = 9    (primo ρ ≥ 0.9 al layer 9)
  → la struttura è al 90% al layer 9 su 24: il 62.5% dei layer (10–24) è
    "ridondante" per la struttura complessiva

Dmeta-ZH (L=12, 13 stati):
  ρ = [0.40, 0.55, 0.68, 0.79, 0.85, 0.90, ...]
  τ(0.5) = 1    (raggiunge il 50% già al layer 1)
  τ(0.9) = 5    (raggiunge il 90% al layer 5)
  → la struttura è al 90% al layer 5 su 12: il 58% dei layer (6–12) è
    "ridondante"
```

**Confronto normalizzato (opzionale)**

Per rendere il confronto ancora più diretto tra modelli di dimensioni diverse,
si può normalizzare: τ(θ)/L = "frazione del modello necessaria a raggiungere
la soglia". Per BGE-EN-large: τ(0.9)/24 = 9/24 = 0.375. Per Dmeta-ZH:
τ(0.9)/12 = 5/12 = 0.417. Questo non è calcolato nel codice ma è un'operazione
banale per la discussione nel testo della tesi.

---

## Analisi §3.1.3c — Neighborhood Trajectory Analysis (NTA)

La NTA è il complemento qualitativo ai due esperimenti quantitativi (§3.1.3a e
§3.1.3b). Mentre drift e Jaccard misurano *quanto* cambia un vicinato, la NTA
mostra *cosa* cambia — quali termini entrano ed escono dal k-NN di una parola
scelta, layer per layer. Il pool di vicini include sia termini giuridici (core)
che non giuridici (control), permettendo di osservare il momento in cui un
termine "diventa legale" (i suoi vicini non-legali vengono sostituiti da
termini di dominio).

---

### N1 — Pool construction: core + control

**Source**: `lens_3_stratigraphy/lens3.py` → `_load_pool_terms()`
**Usata in**: §3.1.3c — definire lo spazio in cui cercare i vicini

**Il problema**

Nella §3.1.3a, il k-NN di ogni termine è calcolato nel pool di soli 397
termini core (tutti giuridici). Questo permette di vedere *quale* dominio
legale domina il vicinato, ma non permette di vedere se il vicinato contiene
termini non-legali. Per la NTA, serve un pool che includa anche parole
quotidiane come baseline.

**Composizione del pool**

```
Pool P = Core ∪ Control

Core:    {t ∈ Index : tier(t) = "core" ∧ domain(t) ≠ ∅}  → 397 termini
Control: {t ∈ Index : tier(t) = "control"}                 → 100 termini

|P| = 497
```

I **control terms** sono 100 parole Swadesh-like (pronomi, verbi di base,
sostantivi concreti: "I", "sleep", "water", "stone", ...). Non hanno dominio
giuridico — sono il punto zero della specializzazione semantica.

**Ogni termine nel pool ha due attributi categorici:**

```
tier(t) ∈ {"core", "control"}
domain(t) ∈ {"civil", "criminal", "constitutional", ..., "control"}
```

Dove `domain = "control"` è assegnato ai termini con `tier = "control"` che
non hanno dominio giuridico nel dataset.

**Implementazione**

```python
def _load_pool_terms():
    _, index = load_precomputed("BGE-EN-large", EMB_DIR)
    pool_idx = [
        i for i, t in enumerate(index)
        if (t["tier"] == "core" and t["domain"]) or t["tier"] == "control"
    ]
    # pool_idx: lista di 497 indici nell'array completo
    pool_terms = [index[i] for i in pool_idx]
    return pool_terms, pool_idx
```

**Perché il pool è lo stesso per tutti i modelli?**

L'indice (`index.json`) contiene sia `en` che `zh_canonical` per ogni termine.
I modelli EN usano `t["en"]`, i modelli ZH usano `t["zh_canonical"]`. Il pool
è strutturalmente identico (stessi 497 termini, stesso ordine), ma il testo
passato al tokenizer cambia per lingua. I layer vectors vengono salvati con
`cache_label="{model}_pool"` per non confondersi con i vettori core-only
(397 termini) usati in §3.1.3a/b.

---

### N2 — k-NN retrieval per layer

**Source**: `lens_3_stratigraphy/lens3.py` → `run_nta()` (inner loop)
**Usata in**: §3.1.3c — trovare i k vicini più simili a un termine target

**La formula**

Per un termine target t, al layer l, il k-NN set è:

```
kNN(t, l) = argmax_k {sim(t, j, l) : j ∈ P \ {t}}

dove sim(t, j, l) = ĥ_t^(l) · ĥ_j^(l)    (dot product di vettori L2-normalizzati)
```

`argmax_k` significa: i k indici j con il valore più alto di sim.

**Perché il dot product è equivalente alla cosine similarity?**

Poiché tutti i vettori sono L2-normalizzati (→ E3):

```
cos(ĥ_t, ĥ_j) = (ĥ_t · ĥ_j) / (‖ĥ_t‖ · ‖ĥ_j‖) = ĥ_t · ĥ_j / (1 · 1) = ĥ_t · ĥ_j
```

Il dot product e la cosine similarity coincidono. Questo permette di calcolare
tutte le similitudini con una singola moltiplicazione matrice-vettore.

**Implementazione riga per riga**

```python
# layer_vecs: (497, L+1, dim) — vettori pool L2-normalizzati
# t_idx: indice del termine target nel pool
# l: indice del layer corrente

vec_t = layer_vecs[t_idx, l, :]       # (dim,) — vettore del target
sims = layer_vecs[:, l, :] @ vec_t    # (497,) — sim con ogni termine del pool
sims[t_idx] = -np.inf                 # esclude self (→ D2)
top_k_idx = np.argsort(sims)[-k:][::-1]  # k indici con sim più alta, decrescente
```

**`np.argsort(sims)[-k:][::-1]`** — come funziona?

1. `np.argsort(sims)` → restituisce gli indici che *ordinerebbero* sims in
   ordine crescente: `[idx_min, ..., idx_max]`
2. `[-k:]` → prende gli ultimi k (i più grandi)
3. `[::-1]` → inverte l'ordine → dal più grande al più piccolo

**Esempio con k=3 e 6 termini**

```
sims = [0.42, -inf, 0.71, 0.33, 0.89, 0.55]
                ↑ self escluso

argsort = [1, 3, 0, 5, 2, 4]    (dal più piccolo al più grande)
[-3:]   = [5, 2, 4]              (i 3 più grandi)
[::-1]  = [4, 2, 5]              (ordine decrescente)

Risultato: kNN = {indice 4 (sim=0.89), indice 2 (sim=0.71), indice 5 (sim=0.55)}
```

**Costo computazionale**

Per ogni termine target, ogni layer campionato:
- 1 matmul (497,dim) × (dim,) = O(497 × dim)
- 1 argsort di 497 elementi = O(497 × log 497)

Con 8 target terms, 7 layer campionati, 6 modelli:
8 × 7 × 6 = 336 operazioni — trascurabile (~1s totale).

---

### N3 — Entry/exit detection

**Source**: `lens_3_stratigraphy/lens3.py` → `run_nta()` (entry/exit logic)
**Usata in**: §3.1.3c — evidenziare quali vicini cambiano tra un layer e il successivo

**Il concetto**

Per ogni coppia di layer consecutivi campionati (l_prev, l_curr), calcoliamo:

```
Entered(t, l_curr) = kNN(t, l_curr) \ kNN(t, l_prev)
Exited(t, l_curr)  = kNN(t, l_prev) \ kNN(t, l_curr)
```

Dove `\` è la differenza insiemistica: A \ B = {x ∈ A : x ∉ B}.

**Entered**: termini che sono nel vicinato al layer corrente ma non erano nel
vicinato al layer precedente. Sono i "nuovi arrivati".

**Exited**: termini che erano nel vicinato al layer precedente ma non sono più
nel vicinato corrente. Sono stati "espulsi".

**Relazione con Jaccard (D2)**

La Jaccard distance tra due layer è:

```
J = 1 - |kNN_prev ∩ kNN_curr| / |kNN_prev ∪ kNN_curr|
```

Ma `|kNN_prev ∪ kNN_curr| = 2k - |kNN_prev ∩ kNN_curr|`, e il numero di
entered/exited è:

```
|Entered| = |Exited| = k - |kNN_prev ∩ kNN_curr|
```

Quindi Jaccard = 0 significa |Entered| = |Exited| = 0 (vicinato identico), e
Jaccard = 1 significa |Entered| = |Exited| = k (vicinato completamente diverso).

**La NTA rende Jaccard leggibile**: J = 0.57 per "strike" al layer 8 non dice
molto. Ma vedere che sono *entrati* "lockout", "arbitration", "collective
bargaining" ed è *uscito* "hit" (ctrl) racconta una storia precisa.

**Implementazione**

```python
# prev_nn_set: set[int] — indici del kNN al layer precedente
# nn_set: set[int] — indici del kNN al layer corrente

for ni in top_k_idx:
    entry = {"rank": rank, "en": name, "domain": dom, "tier": tier, "sim": s}
    if prev_nn_set is not None:
        if ni not in prev_nn_set:
            entry["status"] = "entered"      # ← nuovo arrivato

exited = []
if prev_nn_set is not None:
    for ni in prev_nn_set:
        if ni not in nn_set:
            exited.append({"en": name, "domain": dom, "tier": tier})
```

**Nota**: al layer 0 (embedding layer), `prev_nn_set = None` — non c'è un layer
precedente, quindi non si può parlare di entered/exited. Tutti i vicini sono
mostrati senza annotazione.

---

### N4 — Domain/tier composition tracking

**Source**: `lens_3_stratigraphy/lens3.py` → `run_nta()` (domain_evolution)
**Usata in**: §3.1.3c — quantificare la transizione legal/non-legal layer per layer

**Il concetto**

Per ogni termine target t, a ogni layer campionato l, contiamo:

```
n_legal(t, l) = |{j ∈ kNN(t, l) : tier(j) = "core"}|
n_control(t, l) = |{j ∈ kNN(t, l) : tier(j) = "control"}|

Con: n_legal + n_control = k = 7
```

Questa è la metrica che cattura il momento di "cristallizzazione legale":
quando n_control passa da >0 a 0, il termine ha perso tutti i vicini
non-giuridici — il suo vicinato è interamente composto da termini legali.

**Decomposizione per dominio**

Oltre al conteggio legal/control, contiamo anche per dominio specifico:

```
dom_counts(t, l) = {d: |{j ∈ kNN(t, l) : domain(j) = d}| for d in all_domains}
```

Esempio: per "corruption" al layer 24 di BGE-EN-large:

```
dom_counts = {criminal: 7}
n_legal = 7, n_control = 0
```

Il termine è completamente convergito al dominio criminale. Al layer 0 invece:

```
dom_counts = {civil: 5, control: 1, procedure: 1}
n_legal = 6, n_control = 1
```

Il vicinato all'embedding layer è dominato da termini civili (il dominio più
numeroso nel dataset — effetto di base rate), con un termine di controllo.

**Implementazione**

```python
domain_evolution = []
for ld in layers_data:
    dom_counts = {}       # {domain_name: count}
    n_control = 0
    n_legal = 0
    for nb in ld["neighbors"]:
        d = nb["domain"]
        dom_counts[d] = dom_counts.get(d, 0) + 1
        if nb["tier"] == "control":
            n_control += 1
        else:
            n_legal += 1
    domain_evolution.append({
        "layer": ld["layer"],
        "domains": dom_counts,
        "n_legal": n_legal,
        "n_control": n_control,
    })
```

**Pattern osservati (confermati su 6 modelli)**

```
1. Layer 0 — Embedding layer: vicinati quasi identici per tutti i target terms.
   L'embedding layer non ha ancora applicato nessun transformer block.
   Dominano i termini del dominio più numeroso (civil) per effetto base rate.

2. Layer 4–8 — Peak di control terms per termini polisemici:
   Termini con forte uso non-giuridico (strike, comity, disclosure) raggiungono
   il massimo di control neighbors (fino a 5–7 su 7) ai layer intermedi-bassi.
   Il modello sta ancora rappresentando il significato quotidiano.

3. Layer 12+ — Cristallizzazione:
   n_control → 0 per la maggior parte dei termini. Il vicinato diventa
   interamente legale e converge verso il dominio atteso.

4. Eccezioni sistematiche:
   - "strike" mantiene 1–5 ctrl al layer finale in 3/6 modelli
     → la polisemia militare/sportiva resiste al fine-tuning
   - "comity" mantiene ctrl al finale in BGE-ZH e Dmeta
     → il concetto di "comity" è meno grammaticalizzato in cinese
```

**Legame con la tesi**

La sequenza n_control = [1, 2, 5, 3, 0, 0, 0] per un termine polisemico
fornisce un dato concreto per → §3.1.3 "The depth of legal meaning": il
significato giuridico non è presente ab initio nell'embedding layer, ma
*emerge* attraverso i layer di attenzione. Il layer dove n_control → 0 è
il *punto di cristallizzazione* — il passaggio dal "letterale" al "sistematico"
nella terminologia di Tarello.

---

*Fine trace v2.2 — §3.1.3c NTA aggiunta con dettaglio matematico completo.*

---

## Analisi §3.3 — Value axis projection

Le sezioni seguenti spiegano le operazioni computazionali di Lens IV (§3.3),
che costruisce assi valoriali da coppie di antonimi e misura l'allineamento
delle proiezioni tra tradizioni giuridiche.

---

### V1 — Kozlowski difference-vector: costruzione dell'asse

**Source**: `lens_4_values/lens4.py` → `_build_axis()`
**Usata in**: §3.3.1 — axis construction

**Concetto**

Il metodo di Kozlowski et al. (2019) costruisce un *asse semantico* a partire
da coppie di antonimi. L'idea: se "individual" e "collective" occupano posizioni
diverse nello spazio, la direzione che va da "collective" a "individual" definisce
la dimensione "individuale–collettivo".

**Procedura**

Data una lista di P coppie di antonimi (positivo, negativo):

```
Per ogni coppia i:
    diff_i = embed(pos_i) - embed(neg_i)

asse = L2_normalize( mean(diff_1, diff_2, ..., diff_P) )
```

In formule:

```
â = (1/P) Σᵢ (eₚₒₛᵢ − eₙₑᵍᵢ)
asse = â / ‖â‖
```

dove `‖â‖` è la norma euclidea (lunghezza) del vettore medio.

**Perché la media di più coppie?**

Una singola coppia (es. "individual" vs "collective") è rumorosa: la direzione
individual→collective potrebbe catturare idiosincrasie di quelle due parole
specifiche. Con 10 coppie, il rumore idiosincratico si cancella e resta il
segnale condiviso — la direzione che *tutte* le coppie hanno in comune.

Kozlowski et al. (2019) raccomandano almeno 5 coppie. Il nostro pipeline usa
10 coppie per asse per lingua (EN e ZH indipendenti).

**Perché normalizzare L2?**

Dopo la media, il vettore ha una lunghezza arbitraria. La normalizzazione L2
(`â / ‖â‖`) lo porta a lunghezza 1, in modo che il prodotto scalare con
qualsiasi vettore target corrisponda direttamente alla *similarità coseno*.

**Codice**

```python
def _build_axis(pair_vectors):
    diffs = np.array([pos - neg for pos, neg in pair_vectors])
    mean_diff = diffs.mean(axis=0)
    norm = np.linalg.norm(mean_diff)
    if norm > 0:
        mean_diff /= norm
    return mean_diff
```

**Intuizione per il giurista**

Immagina di avere 10 giuristi che ti indicano dove si trova il concetto
"individuale" rispetto a "collettivo". Ogni giurista punta in una direzione
leggermente diversa. La media delle 10 indicazioni è la miglior stima
della direzione condivisa. La normalizzazione dice: "non ci interessa
quanto è forte il segnale, solo la direzione."

**Legame con la tesi**: → §3.3.1 "We construct three value axes following
Kozlowski et al. (2019), computing the mean difference vector of 10 antonym
pairs per axis and per language, then L2-normalizing."

---

### V2 — Proiezione su un asse: cosine similarity come punteggio

**Source**: `lens_4_values/lens4.py` → `_project_terms()`
**Usata in**: §3.3.2 — cross-tradition alignment

**Concetto**

Una volta costruito l'asse (un vettore unitario di direzione), *proiettiamo*
ogni termine giuridico su quell'asse. Il risultato è uno *score*: un singolo
numero che dice "quanto questo termine è vicino al polo positivo dell'asse."

**Operazione**

```
score(t) = embed(t) · asse = Σₖ embed(t)[k] × asse[k]
```

Poiché sia `embed(t)` sia `asse` sono normalizzati L2, il prodotto scalare
è la *similarità coseno* tra il termine e la direzione dell'asse.

- `score > 0`: il termine è più vicino al polo positivo (es. "individual")
- `score < 0`: il termine è più vicino al polo negativo (es. "collective")
- `score ≈ 0`: il termine è neutro rispetto a questa dimensione

**Codice**

```python
def _project_terms(vecs_core, axis_vec):
    return (vecs_core @ axis_vec).astype(np.float64)
```

`vecs_core` ha forma `(397, dim)`, `axis_vec` ha forma `(dim,)`. Il risultato
è un vettore di 397 score, uno per ogni termine core.

**Range dei valori**: con vettori normalizzati, lo score è in [-1, +1].
In pratica raramente si avvicina ai limiti: valori tipici sono nell'intervallo
[-0.3, +0.3], perché i termini giuridici sono distribuiti in molte dimensioni
e la proiezione su una singola direzione cattura solo una frazione della
varianza totale.

**Intuizione per il giurista**

Immagina l'asse individuale–collettivo come un righello. La proiezione
colloca ogni concetto giuridico su quel righello: "habeas corpus" più verso
l'individuale, "public interest" più verso il collettivo. Lo score è
semplicemente la posizione sul righello.

**Legame con la tesi**: → §3.3.2 "Each of the 397 core legal terms is
projected onto each value axis via cosine similarity, yielding a score
in [-1, +1] that represents the term's position along the value dimension."

---

### V3 — Sanity check: orientamento dell'asse

**Source**: `lens_4_values/lens4.py` → `main()` (inline in axis loop)
**Usata in**: §3.3.1 — axis construction quality

**Concetto**

Il sanity check verifica che l'asse funzioni correttamente: gli antonimi
positivi devono proiettare con score > 0 e i negativi con score < 0.
Se "individual" proiettasse a < 0 sull'asse individuale–collettivo,
l'asse sarebbe invertito o corrotto.

**Procedura**

Per ciascuna delle 10 coppie (pos, neg) usate per costruire l'asse:

```
positive_correct = count(embed(pos_i) · asse > 0)
negative_correct = count(embed(neg_i) · asse < 0)
sanity_pass = positive_correct + negative_correct
sanity_total = 2 × n_pairs
```

**Interpretazione**

- `20/20`: tutti gli antonimi proiettano nel verso corretto → asse robusto
- `18/20`: 1 coppia ambigua → accettabile (il metodo medio è resiliente)
- `<16/20`: 2+ coppie invertite → l'asse potrebbe catturare un'altra dimensione

**Limiti del sanity check**

Questo test verifica solo la *direzione* (segno), non il *potere discriminante*:
una proiezione di +0.001 passa il test tanto quanto +0.500. Non verifica neppure
che l'asse catturi la dimensione *intesa* piuttosto che un confondente correlato.
Per una validazione più forte, si potrebbe aggiungere la coerenza intra-asse
(coseno medio tra i singoli vettori-differenza e il vettore asse).

**Legame con la tesi**: → §3.3.1 "The sanity pass rate ranges from 18/20 to
20/20 across all 6 models and 3 axes, confirming that the axes capture the
intended polarity."

---

### V4 — Cosine similarity inter-asse (diagnostica ortogonalità)

**Source**: `lens_4_values/lens4.py` → `run_section_331()`
**Usata in**: §3.3.1 — orthogonality diagnostic

**Concetto**

Se tre assi misurano dimensioni *indipendenti*, i loro vettori-asse dovrebbero
essere ortogonali: il coseno tra qualsiasi coppia di assi dovrebbe essere ≈ 0.

**Operazione**

```
cos(asse_A, asse_B) = asse_A · asse_B
```

Poiché entrambi sono normalizzati L2, il prodotto scalare è direttamente
la similarità coseno.

**Risultati osservati**

| Coppia di assi | Range osservato | Significato |
|---|---|---|
| individual_collective vs rights_duties | +0.14 a +0.42 | Correlazione moderata |
| individual_collective vs public_private | -0.45 a -0.64 | Anti-correlazione forte |
| rights_duties vs public_private | -0.03 a -0.22 | Quasi ortogonali |

L'anti-correlazione tra individual_collective e public_private è consistente
su tutti i 6 modelli (WEIRD e Sinic). Questo non è un artefatto — riflette
la struttura concettuale: la dicotomia individuo/collettivo e la dicotomia
pubblico/privato sono facce complementari della stessa *summa divisio*
nella tradizione giuridica (Sacco 2019).

**Conseguenza per l'analisi**: le proiezioni sui tre assi non sono
statisticamente indipendenti. Il Kruskal-Wallis in §3.3.3 va interpretato
con cautela, perché confronta ρ su assi parzialmente sovrapposti.

**Legame con la tesi**: → §3.3.1 "The inter-axis cosine matrix reveals
that individual_collective and public_private are anti-correlated (cos ∈
[-0.45, -0.64]), consistent with the *summa divisio* interpretation in
comparative legal theory."

---

### V5 — Spearman ρ tra vettori di punteggio

**Source**: `scipy.stats.spearmanr()`
**Usata in**: §3.3.2 — cross-tradition alignment

**Concetto**

Per ciascuna coppia di modelli (15 coppie totali: 9 cross + 6 within) e
ciascun asse (3), calcoliamo la correlazione di Spearman tra i 397 score
di proiezione. Questo misura: "i due modelli mettono gli stessi termini
nello stesso ordine lungo questa dimensione valoriale?"

**Differenza chiave rispetto a Lens I (RSA)**

In Lens I, Spearman è calcolata tra i triangoli superiori di due RDM
(N(N-1)/2 = 78.606 coppie, non indipendenti). In Lens IV, Spearman è
calcolata tra due vettori di N=397 score (uno per termine, indipendenti).

Questa differenza è cruciale:
- Lens I: le 78.606 coppie *non sono* indipendenti (ogni termine appare
  in N-1 coppie) → serve il block bootstrap e il Mantel test
- Lens IV: i 397 score *sono* indipendenti (ogni termine produce uno
  score autonomo) → il bootstrap iid e la Spearman standard sono validi

**Operazione**

```
ρ = Spearman(scores_model_A, scores_model_B)
```

dove `scores_model_A` e `scores_model_B` sono vettori di 397 numeri.

La correlazione di Spearman è la correlazione di Pearson applicata ai *ranghi*
(per la formula completa → sezione S1). È robusta a distribuzioni non-normali
e misura correlazione *monotona*, non lineare.

**Costruzione indipendente degli assi**

I modelli WEIRD costruiscono l'asse da coppie EN; i modelli Sinic da coppie ZH.
Il ρ cross-tradizione misura quindi: "le dimensioni valoriali costruite
*endogenamente* da ciascuna tradizione linguistica producono ordinamenti simili
dei concetti giuridici?"

Un ρ alto = convergenza strutturale nonostante costruzione indipendente.
Un ρ basso = divergenza, che può riflettere sia diversità concettuale genuina
sia non-equivalenza degli assi. Il ρ è un *lower bound* sull'allineamento reale.

**Legame con la tesi**: → §3.3.2 "The Spearman ρ between independently
constructed axes measures rank alignment of 397 legal terms. It represents
a lower bound on true cultural alignment because it compounds genuine
divergence with axis construction noise."

---

### V6 — Row-resample bootstrap per Spearman ρ

**Source**: `lens_4_values/lens4.py` → `_spearman_bootstrap()` →
`shared/statistical.py` → `bootstrap_ci_generic()`
**Usata in**: §3.3.2 — confidence intervals

**Concetto**

Il *bootstrap* (Efron 1979) stima l'incertezza di una statistica ricampionando
i dati. Per Lens IV usiamo il *row-resample bootstrap*: ricampioniamo i
397 termini (righe), non le coppie di termini come in Lens I.

**Perché row-resample e non block bootstrap?**

In Lens I, ogni termine appare in N-1 coppie della RDM → le osservazioni non
sono indipendenti → serve il block bootstrap che ricampiona blocchi di termini,
preservando la struttura di dipendenza (→ sezione S5).

In Lens IV, ogni termine produce un singolo score di proiezione, indipendente
dagli score degli altri termini. Le 397 osservazioni sono iid (condizionatamente
al modello e all'asse). Il bootstrap iid sulle righe è quindi appropriato e
più efficiente del block bootstrap.

**Procedura**

```
data = column_stack([scores_A, scores_B])   # (397, 2)

Per b = 1, ..., B (B=10000):
    1. Campiona 397 indici con ripetizione
    2. Seleziona le righe corrispondenti: data_b = data[indices]
    3. Calcola ρ_b = Spearman(data_b[:, 0], data_b[:, 1])

CI_95% = [percentile_2.5(ρ_1..ρ_B), percentile_97.5(ρ_1..ρ_B)]
```

**Scelta di B=10000**

B=10000 allineato con Lens I per coerenza del parametro di Monte Carlo
attraverso la tesi. Con B=10000, ciascun endpoint del CI al 95% è basato su
250 valori (10000 × 0.025), con errore Monte Carlo SE ≈ 0.0016 — trascurabile
rispetto all'ampiezza tipica del CI (~0.18).

**Codice**

```python
def _spearman_bootstrap(scores_a, scores_b, n_boot=10000, seed=42):
    stacked = np.column_stack([scores_a, scores_b])
    def stat_fn(data):
        return float(spearmanr(data[:, 0], data[:, 1]).statistic)
    return bootstrap_ci_generic(stacked, stat_fn, n_boot=n_boot, seed=seed)
```

**Intuizione per il giurista**

Hai 397 termini. Il bootstrap chiede: "se avessi avuto 397 termini *diversi*
(ma campionati dallo stesso universo giuridico), avrei ottenuto un ρ simile?"
Ripetendo 10.000 volte, otteniamo la distribuzione della risposta. L'intervallo
che contiene il 95% delle risposte è il CI.

**Legame con la tesi**: → §3.3.2 "95% bootstrap percentile intervals (B=10,000,
row-resampling terms) quantify uncertainty. Row-resampling is appropriate because
each term yields one independent projection score, unlike RSA where observations
are structurally dependent pairs."

---

### V7 — Mann-Whitney U su gruppi di ρ (cross vs within)

**Source**: `shared/statistical.py` → `mannwhitney_with_r()`
**Usata in**: §3.3.2 — tradition separation test

**Concetto**

Per ciascun asse, abbiamo 9 valori ρ cross-tradizione e 6 valori ρ
within-tradizione. La domanda: "i ρ cross sono *sistematicamente* più bassi
dei ρ within?" Risponde il test di Mann-Whitney U (non-parametrico).

**Operazione**

```
H0: cross_rhos e within_rhos provengono dalla stessa distribuzione
H1: cross_rhos sono stocasticamente inferiori (alternative="less")
```

Il test calcola la statistica U e il p-value. L'effect size è il
rank-biserial correlation r (→ sezione A2):

```
r = 1 - 2U / (n₁ × n₂)
```

dove n₁=9 (cross), n₂=6 (within). r → 1 significa separazione completa.

**Campioni piccoli e pseudo-replicazione**

Con n₁=9 e n₂=6, il test ha potenza limitata. Inoltre, i 9 ρ cross provengono
da una griglia 3×3 di modelli: ogni modello EN appare in 3 coppie e ogni modello
ZH in 3 coppie. Le osservazioni non sono completamente indipendenti. I gradi
di libertà effettivi sono inferiori a 9 e 6.

Di conseguenza, i p-value del Mann-Whitney vanno interpretati come *indicativi*,
non come test formali. Gli effect size (r=0.78–1.00) sono invece interpretabili:
indicano separazione quasi completa tra i gruppi, indipendentemente dalla
significatività formale.

**Risultati osservati**

| Asse | cross ρ̄ | within ρ̄ | effect r | p |
|---|---|---|---|---|
| individual_collective | 0.292 | 0.538 | +1.00 | 0.0002 |
| rights_duties | 0.380 | 0.627 | +1.00 | 0.0002 |
| public_private | 0.402 | 0.581 | +0.78 | 0.006 |

**Legame con la tesi**: → §3.3.2 "The Mann-Whitney test is reported as
descriptive evidence of tradition separation. Effect sizes (r=0.78–1.00)
indicate near-complete separation, though the test's formal p-values
are approximate due to pseudo-replication in the 3×3 model grid."

---

### V8 — Kruskal-Wallis H + post-hoc Bonferroni

**Source**: `scipy.stats.kruskal()` + `mannwhitney_with_r()`
**Usata in**: §3.3.3 — which axes diverge most?

**Concetto**

Il Kruskal-Wallis H è l'analogo non-parametrico dell'ANOVA a una via.
Confronta le *distribuzioni* dei ρ cross-tradizione tra i 3 assi (9 valori
per asse, 27 totali).

```
H0: i 3 gruppi di ρ cross provengono dalla stessa distribuzione
H1: almeno un gruppo differisce
```

**Perché Kruskal-Wallis e non ANOVA?**

- Solo 9 osservazioni per gruppo → impossibile verificare normalità
- I ρ sono bounded (correlazioni), distribuzione potenzialmente asimmetrica
- Il test non-parametrico non richiede assunzioni distribuzionali

**Post-hoc**

Se H è significativo (p < 0.05), si eseguono confronti pairwise tra coppie di
assi con Mann-Whitney, correggendo per 3 confronti con Bonferroni:
`p_adjusted = min(p_raw × 3, 1.0)`.

**Power e limiti**

Con n=9 per gruppo e η²_H ≈ 0.16 (effect size medio-grande), la potenza
stimata è ~0.50 — insufficiente per raggiungere con sicurezza p < 0.05.
Il risultato KW (H=7.20, p=0.027) è significativo, ma i post-hoc pairwise
non sopravvivono alla correzione Bonferroni (p_adj > 0.05 per tutti e 3
i confronti). Questo è atteso con la potenza disponibile.

**Interpretazione corretta**

Il KW rileva una differenza globale tra assi (p=0.027), con individual_collective
(ρ̄=0.292) come asse più divergente. Ma il design non ha sufficiente potenza
per isolare *quale* coppia di assi differisce. Il ranking descrittivo
(individual_collective < rights_duties < public_private) è il dato riportabile.

**Risultati osservati**

```
KW: H=7.20, p=0.027 (significativo)

Post-hoc (Bonferroni, 3 confronti):
  ind_coll vs public_private:  r=+0.65  p_adj=0.065  (n.s.)
  ind_coll vs rights_duties:   r=+0.60  p_adj=0.102  (n.s.)
  public_private vs rights_duties: r=-0.21  p_adj=1.000  (n.s.)
```

**Legame con la tesi**: → §3.3.3 "The Kruskal-Wallis test detects a global
difference in cross-tradition alignment across axes (H=7.20, p=0.027), with
individual_collective as the most divergent axis. Post-hoc pairwise tests do
not survive Bonferroni correction, consistent with the limited power of
n=9 per group."

---

*Fine trace v2.3 — §3.3 Value axis projection aggiunta con 8 sezioni (V1-V8).*
