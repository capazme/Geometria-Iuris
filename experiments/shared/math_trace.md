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

### Analisi §8.2 — Struttura geometrica (WEIRD-only)
- [A1 — `scipy.stats.mannwhitneyu()`](#a1--scipystatmannwhitneyu)
- [A2 — Effect size: rank-biserial correlation](#a2--effect-size-rank-biserial-correlation)
- [A3 — Matrice inter-dominio K×K](#a3--matrice-inter-dominio-k×k)

### Analisi §8.5 / §9.2 — RSA e Mantel test
- [S1 — `scipy.stats.spearmanr()`](#s1--scipystats-spearmanr)
- [S2 — Mantel test (permutation test su matrici)](#s2--mantel-test-permutation-test-su-matrici)
- [S3 — `np.random.default_rng()` e `rng.permutation()`](#s3--nprandomdefault_rng-e-rngpermutation)
- [S4 — `np.ix_()`](#s4--npix_)
- [S5 — Block bootstrap per confidence interval](#s5--block-bootstrap-per-confidence-interval)
- [S6 — `np.percentile()`](#s6--nppercentile)

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

## Analisi §8.2 — Struttura geometrica (WEIRD-only)

---

### A1 — `scipy.stats.mannwhitneyu()`

**Source**: SciPy
**Usata in**: §8.2.1 (intra vs inter-dominio), §8.2.2 (legal vs control)
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
**Usata in**: §8.2.1, §8.2.2 — dimensione dell'effetto

**Il problema del p-value da solo**

Con decine di migliaia di coppie, qualsiasi differenza sarà statisticamente
significativa. Il p-value dice "esiste un effetto?". L'*effect size* dice
"quanto è grande?". Entrambi sono necessari (→ §7.2.3 "refusal of stargazing").

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
**Usata in**: §8.2.3 — topologia dei domini

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

## Analisi §8.5 / §9.2 — RSA e Mantel test

---

### S1 — `scipy.stats.spearmanr()`

**Source**: SciPy
**Usata in**: RSA — correlazione tra RDMs

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
**Usata in**: §8.5.1, §9.2 — significatività statistica di RSA

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
**Usata in**: Mantel test (permutazione), §8.2.3 (matrice inter-dominio),
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
**Usata in**: §8.5.1, §9.2 — intervallo di confidenza per ρ RSA
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

## Assegnamento per dominio (§8.1 — background terms)

---

### B1 — k-Nearest Neighbors (k-NN) majority vote

**Source**: implementazione custom in `lens_1_relational/domain_assignment.py`
**Usata in**: §8.1 — assegnamento dei termini background ai domini

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
corrispondono ai "boundary objects" di §8.3.3.

**Sensibilità al parametro k**

La scelta di k=7 viene verificata con k=5 e k=9 come robustness check (§7.3):
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

*Fine trace v1.1 — aggiornato con ogni nuova funzione introdotta nel pipeline.*
