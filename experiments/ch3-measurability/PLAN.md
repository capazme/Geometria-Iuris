# Run #4 — post-BLP final · PLAN

**Status**: planning complete, awaiting execution in a new session.
**Author**: planned 2026-05-11.
**Working directory at execution time**: `/Users/gpuzio/Desktop/CODE/THESIS`.

---

## 1. Context — perché esiste questa run

È la **quarta run completa** dei due esperimenti di Cap. 3. Probabilmente l'ultima.

### Storia delle run precedenti

| Run | Pool | Encoding | Quando | Esito |
|---|---|---|---|---|
| #1 | 394 core ad hoc + synthetic padding | bare only | inizio 2026 | superseded |
| #2 | 350 core + 100 control | bare only | pre-Firthian | superseded |
| #3 | **327 Firthian-strict** | bare + attested | 2026-05-02 | run di consegna intermedia |
| **#4** | **364 post-BLP** | **bare + attested-postBLP** | **2026-05-11/12** | **target: tesi finale** |

### Cosa cambia in #4

Il pool è ricurato per **eliminare il bias della fictio di equal authenticity**: i 364 termini hanno tutti K≥4 attestazioni in ordinanze HK Cap. **enacted post-1989** sotto il Bilingual Laws Project, cioè co-redatte bilingue. La fictio non si applica più strutturalmente. Curazione documentata in `experiments/data/trace_postBLP_curation.md` (D1-D11), 104 zh override + 9 DROP in `zh_overrides_postBLP.json`.

5 fronti di miglioramento qualità rispetto al baseline #3:
1. No encoding collisions (crime/offence distinti)
2. No inflectional duplicates (14 cluster ripuliti)
3. No drafting-glue procedure (-11 termini)
4. ZH wrong-sense corretto (4 override revised, 5 AUTO_KEEP rescued)
5. No foreignness-adjective leakage in international (13 drops)

Banda per-dominio 41-60, media 52.0 (vs rigido 45-50 della run #3).

---

## 2. Inputs frozen — la cassetta di partenza

Tutti i file in input sono **read-only** durante run #4 (sigillati prima dell'esecuzione). Da copiare in `experiments/ch3-measurability/inputs/` come snapshot:

| File sorgente | Snapshot in run4 |
|---|---|
| `experiments/data/processed/legal_terms.json` | `inputs/legal_terms_postBLP_snapshot.json` |
| `experiments/data/processed/zh_overrides_postBLP.json` | `inputs/zh_overrides_snapshot.json` |
| `experiments/data/processed/cap_enactment_years.json` | `inputs/cap_enactment_years_snapshot.json` |
| `experiments/data/processed/elegislation/term_contexts_postBLP.jsonl` | `inputs/term_contexts_bilingual_snapshot.jsonl` |
| `experiments/data/processed/elegislation/coverage_postBLP.json` | `inputs/context_coverage_snapshot.json` |
| `experiments/data/trace_postBLP_curation.md` | `inputs/trace_curation_snapshot.md` |
| `experiments/lens_4_values/value_axes.yaml` (unchanged: 6×2×10 coppie poli) | `inputs/value_axes_snapshot.yaml` |

Verifica integrità: ogni snapshot ha un hash SHA-256 in `inputs/manifest.json`.

---

## 3. Folder structure target

```
experiments/ch3-measurability/
├── PLAN.md                          ← questo file
├── README.md                        ← cosa è questa folder, come riprodurre
├── trace.md                         ← D1-D12 decisioni di parametro della run
├── config.yaml                      ← TUTTI i parametri (seed, B, modelli, …)
├── manifest.json                    ← hashes degli input + delle output finali
│
├── inputs/                          ← snapshot frozen (sopra)
│
├── embeddings/                      ← precomputed dopo step 2
│   ├── BGE-EN-large/
│   │   ├── vecs_bare.npy            (364, 1024) float32, L2-norm
│   │   ├── vecs_attested.npy        (364, 1024)
│   │   ├── coverage.json            (per termine: n_contexts usati per attested)
│   │   └── meta.json                (model id, dim, dtype, run_date)
│   └── … (10 modelli)
│
├── experiment_1_structure/                           ← §3.1 outputs
│   ├── results_bare/
│   │   ├── experiment_1_results.json       (section_311, section_31, section_314)
│   │   ├── rdms/{model}.npz
│   │   ├── distances/{model}.npz    (intra, inter, legal, control)
│   │   └── distributions/{pair}.npz (null 1000, bootstrap 10000)
│   ├── results_attested/             (stessa struttura)
│   └── categorical_probe.json
│
├── experiment_2_axes/                           ← §3.2 outputs
│   ├── results_bare/
│   │   ├── experiment_2_results.json       (section_331, 332, 333)
│   │   └── scores/{model}_{axis}.npy
│   └── results_attested/             (stessa struttura)
│
├── reports/                         ← human-readable
│   ├── numbers_headline.md          (i numeri da copiare in CLAUDE.md §10)
│   ├── changes_vs_run3.md           (delta rispetto al Firthian baseline)
│   └── diagnostic_plots/
│       ├── forest_rsa_attested.png
│       ├── forest_rsa_bare.png
│       ├── domain_topology_consensus.png
│       ├── axes_alignment_2x3.png
│       └── divergent_terms_scatter.png
│
└── notebooks/
    └── ch3-measurability_analysis.ipynb  ← narrazione end-to-end (vedi §6)
```

---

## 4. Pipeline — sette step con parametri espliciti

Ogni step è **idempotente**: rilanciarlo produce lo stesso output (stesso seed, stesse versioni). Tutti gli step usano `config.yaml` come single source of truth dei parametri.

### Step 0 — Sigillo degli input (5 min)

Script: `experiments/ch3-measurability/scripts/00_seal_inputs.py`

- Copia i file di §2 in `inputs/`
- Calcola SHA-256 di ciascuno, scrive `manifest.json`
- Verifica: 364 termini KEEP in `legal_terms_postBLP_snapshot.json`

**Verifica end-to-end**: `manifest.json` esiste, 7 hash diversi, 364 KEEP confermati.

### Step 1 — Build config (5 min)

Script: scrive `config.yaml` con:
```yaml
run_id: ch3-measurability
date: 2026-05-12
seed: 42
n_perm_mantel: 10000        # ↑ da 1000 (definitive run)
n_boot: 10000               # ↑ da 1000
k_nn_domain: 7
holm_correction: true
threshold_year_postBLP: 1989

models_weird:
  - BGE-EN-large
  - E5-large
  - FreeLaw-EN
models_sinic:
  - BGE-ZH-large
  - Text2vec-large-ZH
  - Dmeta-ZH
models_bilingual:
  - {label: BGE-M3-EN,    sibling: BGE-M3-ZH}
  - {label: Qwen3-0.6B-EN, sibling: Qwen3-0.6B-ZH}

attested_max_contexts: 8     # per term, take up to N post-1989 attestations
attested_min_contexts: 4     # require K≥4 (the strict Firthian on postBLP)

device: cpu
dtype: float32
```

### Step 2 — Re-encoding (5-10h)

Script: `experiments/ch3-measurability/scripts/02_encode_core.py`

Per ogni modello in `config.yaml`:
1. **Bare**: encode la stringa canonica (`en_clean` per WEIRD/bilingual-EN, `zh_clean` per Sinic/bilingual-ZH). Output: `vecs_bare.npy` shape `(364, dim)`, L2-normalized.
2. **Attested**: per ogni termine, prendi le sue ≤8 attestazioni post-1989 da `term_contexts_postBLP.jsonl`, encode ciascun contesto, mediana o media dei vettori (decisione D6 in `trace.md`: **media**, come in run #3), normalizza a L2. Output: `vecs_attested.npy`.
3. **Coverage diagnostic**: per ogni termine traccia `n_contexts` effettivamente usati (di solito 4-8); flag i termini con n_contexts < 4 (i 6 residual K<4).

**Output per modello**:
- `embeddings/{model}/vecs_bare.npy`
- `embeddings/{model}/vecs_attested.npy`
- `embeddings/{model}/coverage.json` con per term: en, n_contexts_bare=1, n_contexts_attested, list of caps
- `embeddings/{model}/meta.json` con id, dim, dtype, date, hash

**Verifica end-to-end**:
- 10 cartelle `embeddings/{model}/` esistono
- Ogni `vecs_*.npy` ha shape `(364, dim)` corretta
- Tutti i vettori L2-norm = 1 ± 1e-5
- `coverage.json` riporta 358 termini con n_contexts ≥ 4 e 6 con K<4

**Compute**: ~30-60 min per modello (la maggior parte è l'attested, che richiede 4-8 forward pass per termine). Stima: **~6-8h totali su CPU**, può girare overnight.

### Step 3 — Lens I (§3.1) bare (2-3h)

Script: `experiments/ch3-measurability/scripts/03_structure_bare.py`

Esegue le 4 sotto-sezioni:
- **§3.1.1** intra-vs-inter (Mann-Whitney + r) per i 3 WEIRD (computato dal RDM); legal-vs-control per i 10 modelli.
- **§3.1.2** domain topology 7×7 per ogni modello.
- **§3.1.3** RSA 17 coppie con:
  - Spearman ρ
  - Mantel test, **B=10000** (p-min rappresentabile = 0.0001)
  - Block bootstrap CI 95%, **B=10000**
  - Holm correction sulle 17 p-value
- **§3.1.4** categorical probe (5 test pre-registrati, 10 modelli).

Riusa il codice esistente in `experiments/lens_1_relational/lens1.py` con override `--emb-dir experiments/ch3-measurability/embeddings/`.

**Output**: tutto in `experiment_1_structure/results_bare/`. Inclusi gli RDM dump (per il dashboard) e le distribuzioni null/bootstrap per coppia.

### Step 4 — Lens I (§3.1) attested (2-3h)

Stesso, con `vecs_attested.npy` invece di `vecs_bare.npy`.

**Output**: `experiment_1_structure/results_attested/`.

### Step 5 — Lens IV (§3.2) bare + attested (2-3h)

Script: `experiments/ch3-measurability/scripts/05_axes_experiment.py`

- §3.2.1 axes construction (sanity leave-one-out per ogni axis × model)
- §3.2.2 orthogonality (6×6 cosine matrix per model)
- §3.2.3 alignment per axis (45 pairs × 6 axes con bootstrap CI)
- §3.2.4 ranking aggregato
- §3.2.5 divergent terms (W̄ vs S̄ scores)

Riusa `experiments/lens_4_values/lens4.py`.

**Output**: `experiment_2_axes/results_bare/` e `experiment_2_axes/results_attested/`. Score files (`scores/{model}_{axis}.npy`) sono 364 floats per termine, per modello, per asse.

### Step 6 — Reports + headline numbers (1h)

Script: `experiments/ch3-measurability/scripts/06_reports.py`

Estrae i numeri chiave da `experiment_1_structure/` e `experiment_2_axes/` e produce:
- **`reports/numbers_headline.md`**: i valori da copiare in CLAUDE.md §10 (ρ̄ per categoria, Δρ, axis ranking, ecc.)
- **`reports/changes_vs_run3.md`**: confronto numerico run #3 (Firthian 327) vs run #4 (postBLP 364) — sia per §3.1.3 che §3.2.x
- **`reports/diagnostic_plots/`**: 5 PNG chiave (forest plot, ridge plot, topology heatmap, axes 2×3, divergent scatter)

### Step 7 — Notebook narrativo (2-3h)

Script (manuale, jupyter): `notebooks/ch3-measurability_analysis.ipynb`

Vedi §6 sotto.

---

## 5. Compute budget totale

| Step | Tempo stimato | Tipo |
|---|---|---|
| 0. Seal inputs | 5 min | sequenziale |
| 1. Config | 5 min | sequenziale |
| 2. Re-encoding 10 modelli × bare+attested | 6-8h | sequenziale (CPU); parallelizzabile per modello |
| 3. Lens I bare | 2-3h | sequenziale, ~17 pairs × B=10000 |
| 4. Lens I attested | 2-3h | sequenziale |
| 5. Lens IV bare+attested | 2-3h | sequenziale |
| 6. Reports | 1h | sequenziale |
| 7. Notebook narrativo | 2-3h | manuale |
| **Totale** | **16-22h** | overnight + un giorno |

Strategia: lanciare step 2 (re-encoding) overnight venerdì → step 3-5 sabato mattina → step 6-7 sabato pomeriggio. Pronto domenica per consolidare CLAUDE.md e tesi.

---

## 6. Notebook structure (`ch3-measurability_analysis.ipynb`)

Cell-by-cell:

1. **Setup**: import, load config, load 364 KEEP, summary stats
2. **Pool description**: distribuzione per dominio, anno-distribution delle attestazioni post-1989, top 10 Cap (tutti post-1989)
3. **Encoding diagnostic**: per modello e per encoding, mean/std del modulo (deve essere 1.0), distribuzione delle distanze pair-wise (sanity check)
4. **§3.1.1 distances**: violin plot intra/inter per i 3 WEIRD bare+attested; bar legal-vs-control 10 modelli
5. **§3.1.2 topology**: heatmap 7×7 per BGE-EN come exemplar + consensus matrix media 10 modelli
6. **§3.1.3 RSA — il cuore**: forest plot 17 pairs (bare e attested affiancati), Δρ table, Mantel p, CI 95%
7. **§3.1.4 probe**: forest 5 test × 10 modelli + exact-hit table
8. **§3.2.1-2 axes**: 6 assi costruiti, sanity 10×6, orthogonality consensus 6×6
9. **§3.2.3 alignment**: forest 2×3 attested + bare, ρ̄ cross/within per axis
10. **§3.2.4 ranking** + **§3.2.5 divergent**: most-divergent per axis, top-20 terms
11. **Comparison vs run #3**: tabella before/after per §3.1.3 e §3.2.x (mostra che i numeri tengono o cambiano, e di quanto)
12. **Conclusioni per la tesi**: 3-5 frasi pronte per essere copiate in §3 di Cap. 4

Output: notebook eseguito senza errori, tutti i plot salvati in `reports/diagnostic_plots/`.

---

## 7. Trace document (`trace.md`)

12 decisioni di parametro da registrare **prima** dell'esecuzione, ognuna con: opzioni considerate, decisione, rationale, thesis implication (§ riferimento).

D1. Modello list: 10 modelli (3 WEIRD + 3 Sinic + 4 bilingual lati). NB Nomic non incluso (decisione di scope nella run #3, confermata).
D2. Encoding: bare + attested-postBLP (no all-attestations variant — il pool stesso è postBLP).
D3. Attestation averaging: media dei vettori delle ≤8 occorrenze, NOT median (continuità con run #3).
D4. Mantel B=10000 (definitive; p-floor 0.0001).
D5. Bootstrap B=10000 a livello di termine (Nili 2014).
D6. Holm-Bonferroni sulle 17 p-value (FWER control).
D7. Seed 42, CPU float32 deterministico.
D8. Per-domain reporting: 7 domini, banda 41-60 esplicita (no rounded balance).
D9. Categorical probe: stessi 5 test pre-registrati di run #3.
D10. Output destination: `ch3-measurability/`, NON sovrascrivere `lens_*/results*`.
D11. Dashboard update: rigenerare `dashboard_v3` letto da `ch3-measurability/` (loader update minimo).
D12. Riproducibilità: ogni script con `argparse --config config.yaml`, no hard-coded paths.

---

## 8. Verification gate

Run #4 è "good to go" se:

- [ ] 10 cartelle `embeddings/` con vec_bare + vec_attested
- [ ] Tutti i vettori L2-norm = 1 ± 1e-5
- [ ] `experiment_1_structure/results_*` con 17 ρ + CI + p_mantel + p_holm per ogni
- [ ] `experiment_2_axes/results_*` con sanity + ortho + alignment + ranking + divergent
- [ ] **ρ̄_cross attested in range plausibile** (run #3 era 0.259; ci aspettiamo simile, ±0.05)
- [ ] **ρ̄_W e ρ̄_S attested in range plausibile** (run #3: 0.760 e 0.845; ci aspettiamo simile)
- [ ] Δρ symmetric attested > 0.4 (era 0.541)
- [ ] Mantel p ≤ 0.0001 per tutte le 17 coppie (B=10000)
- [ ] Holm-corrected p ≤ 0.0017 (= 0.0001 × 17, conservative bound)
- [ ] Tutti i 5 probe test §3.1.4 producono ρ_ensemble nel range run #3 ±0.10
- [ ] 6 PNG diagnostic plots prodotti e visivamente corretti

Se uno di questi fallisce: STOP e indaga (non procedere al notebook e all'aggiornamento tesi).

---

## 9. Outputs finali → destinazioni

Dopo che `ch3-measurability/` è completo:

| Output | Destinazione | Formato |
|---|---|---|
| Numbers headline | `CLAUDE.md` §10 (sovrascrivi i numeri Firthian) | markdown table |
| Run #4 trace | resta in `ch3-measurability/trace.md` | reference |
| Notebook narrativo | resta in `ch3-measurability/notebooks/` | .ipynb + .html export |
| §2.1 (lessico) | tesi: `documenti/capitoli/ch2/sections/v1_ch2_2_1_*.md` | edit |
| §2.2 (HK fictio) | tesi: 1 footnote sui canonical-lemma corrections | edit |
| §3 (results chapter) | tesi: `documenti/capitoli/ch3/` | sostituisci numeri |
| Dashboard v3 | rigenera leggendo da `ch3-measurability/` | code change in `dashboard_v3/data/*.py` |

---

## 10. Rollback plan

Se qualcosa fallisce a metà:
- `ch3-measurability/` è isolato — nessuna sovrascrittura di `lens_*/results*` di run #3
- Run #3 (`experiments/lens_1_relational/results*/`, `experiments/lens_4_values/results*/`) resta intatta e disponibile
- Dashboard v3 con i numeri run #3 continua a funzionare fino a esplicito switch
- Si può tornare a run #3 in qualsiasi momento cambiando il path nei loader del dashboard

Non si modifica `legal_terms.json` durante run #4 — è già stato modificato dalla curazione post-BLP e quello è il dataset di riferimento.

---

## 11. Pre-execution checklist (per la nuova sessione)

Prima di lanciare lo step 0:

- [ ] Verifica che `experiments/data/processed/legal_terms.json` abbia 364 KEEP
- [ ] Verifica che `term_contexts_postBLP.jsonl` esista con i contesti reali
- [ ] Verifica che `coverage_postBLP.json` confermi che 358/364 termini hanno K≥4
- [ ] `experiments/ch3-measurability/` è vuoto (a parte questo PLAN.md, README.md, e le 6 subfolder skeleton)
- [ ] Almeno 50 GB di disco libero (embeddings + RDM dump + distribuzioni)
- [ ] Python env con: numpy, scipy, sentence-transformers, opencc-python-reimplemented, plotly, jupyter

---

## 12. Apertura della nuova sessione

Suggerimento per come kick-off:

> "Eseguiamo run #4 secondo `experiments/ch3-measurability/PLAN.md`. Parti dallo step 0."

A quel punto, in ordine:
1. Crea `scripts/00_seal_inputs.py`, esegui, verifica manifest
2. Scrivi `config.yaml` come da §4 step 1
3. Scrivi `scripts/02_encode_core.py`, lancia (background, overnight)
4. Quando l'encoding finisce: scrivi `scripts/03_structure_bare.py` + `04_structure_attested.py`, lancia
5. Scrivi `scripts/05_axes_experiment.py`, lancia
6. Scrivi `scripts/06_reports.py`, esegui
7. Apri il notebook, fai la narrazione cell-by-cell, esegui, esporta HTML
8. Update CLAUDE.md §10 e le sezioni tesi
9. Rigenera dashboard v3

---

## 13. Note di scope

- **Cosa NON si fa in run #4**: niente nuovo modello, niente nuovo asse Kozlowski, niente nuovo experimental design. Solo re-run pulito sul pool post-BLP curato.
- **Cosa SI fa**: B=10000 (più rigoroso), dataset post-BLP (no fictio bias), audit trail completo, notebook narrativo.
- **Cosa è opzionale ma raccomandato**: HTML export del notebook per inclusione nell'appendice di consegna tesi.
