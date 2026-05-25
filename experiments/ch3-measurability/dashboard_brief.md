# Dashboard brief — sessione di handoff

Documento di contesto per la nuova sessione di Claude Code dedicata alla
**dashboard definitiva** di Capitolo 3.

Leggere top-to-bottom. Tutti i pointer in fondo.

---

## 0. Obiettivo della sessione

Costruire (o aggiornare) la dashboard interattiva che accompagna Cap. 2 e
Cap. 3 della tesi *Geometria Iuris*, sostituendo la base dati run-#3 con
quella di `experiments/ch3-measurability/` (run #4 post-BLP + 9 estensioni
A-Z + caveat Y sul framing bare/attested).

Audience finale: **commissione di laurea LUISS** (relatore giurista +
co-relatore ingegnere + 1 prof. giurista). La prosa principale si legge
senza cliccare; l'apparato tecnico (formule, p-value, codice) resta
visibile in-pagina alla fine di ciascuna sotto-sezione.

---

## 1. Punto di partenza — cosa esiste già

### 1.1 Dati (in `experiments/ch3-measurability/`)

Tutti i numeri sono in JSON, hashati nel `manifest.json` (50 SHA-256 totali).

- **Experiment 1 — Distance Structure (§3.1)**
  - `experiment_1_structure/results_{bare,attested}/experiment_1_results.json` — 17 RSA pairs (Mantel B=10000, block-bootstrap CI B=10000, Holm correction), §3.1.1 intra-vs-inter Mann-Whitney, §3.1.2 7×7 domain topology per model
  - `experiment_1_structure/results_bare/legal_vs_control.json` — §3.1.1 legal-vs-control (8/10 models confirm; FreeLaw-EN e Qwen3-0.6B-EN no — informativo)
  - `experiment_1_structure/results_{bare,attested}/categorical_probe.json` — §3.1.4 (pool-independent, linked from run #3)

- **Experiment 2 — Value Axes (§3.2)**
  - `experiment_2_axes/results_{bare,attested}/experiment_2_results.json` — 6 axes (individual_collective, rights_duties, public_private, state_market, natural_positive, status_contract), 45 pairs × 6 axes alignment, ranking, divergent terms top-K

- **9 estensioni** in `ext/{A_bg_knn, D_robustness, E_axes_oos, F_confidence, G_false_friends, H_K_saturation, X_control_robustness, Y_control_only, Z_tier_hierarchy}/*.json`

- **Reports markdown** in `reports/`:
  - `numbers_headline.md` — tabelle pronte (auto-generato da `06_reports.py`)
  - `extensions_summary.md` — narrazione delle 9 estensioni
  - `changes_vs_run3.md` — delta vs run #3
  - `verification_gate.md` — 8/8 PASS

### 1.2 Dashboard v3 esistente

`experiments/dashboard_v3/` ha già pattern + 4 pagine HTML:

```
dashboard_v3/
├── build.py               entry point — rigenera tutte le pagine
├── README.md
├── shared_ui.py           CSS Okabe-Ito, helper HTML, Plotly defaults
├── apparatus.py           apparatus_block() — apparato tecnico L2
├── pages/
│   ├── index.py           strumenti (modelli, dataset, toolkit)
│   ├── come_funziona.py   come funzionano gli algoritmi
│   ├── exp31.py           §3.1 organizzazione del lessico
│   └── exp32.py           §3.2 proiezione sui valori
├── data/
│   ├── results_31.py      LOADER §3.1 — PUNTA A lens_1_relational/ (run #3)
│   └── results_32.py      LOADER §3.2 — PUNTA A lens_4_values/ (run #3)
├── figures/
│   ├── exp31.py           Plotly figures per §3.1
│   └── exp32.py           Plotly figures per §3.2
└── *.html (4 pagine, prodotte 2026-05-04)
```

**Pattern "concentrico" per ciascuna sotto-sezione di Cap. 3** (vedi
`dashboard_v3/README.md`):

```
1. scenario giuridico       (italico, 1-2 frasi, esempio concreto)
2. risultato in parole       (verdetto, 1 frase, no jargon)
3. grafico annotato          (Plotly, freccia sul dato saliente)
4. take-home                 (1 frase, link a Cap. 4 senza interpretarne)
5. apparato tecnico          (L2 full-width, formula + stats + codice)
```

Nessun glossario in coda; definizioni inline al primo uso.

---

## 2. Cosa la nuova sessione deve fare

### 2.1 Minimo indispensabile (riallineamento dati)

1. **Aggiornare i loader** `dashboard_v3/data/results_31.py` e
   `results_32.py` per puntare a `experiments/ch3-measurability/`:
   - Sostituire `lens_1_relational/results_{bare,attested}/lens1_results.json`
     → `experiments/ch3-measurability/experiment_1_structure/results_{bare,attested}/experiment_1_results.json`
   - Sostituire `lens_4_values/results_{bare,attested}/lens4_results.json`
     → `experiments/ch3-measurability/experiment_2_axes/results_{bare,attested}/experiment_2_results.json`
   - Aggiornare RDM/distributions paths (`/rdms/`, `/distributions/`,
     `/topology/`) di conseguenza.

2. **Rigenerare le 4 HTML** con `python3 build.py` e verificare che i numeri
   matchino `reports/numbers_headline.md`.

### 2.2 Estensione editoriale (le 9 estensioni A-Z)

Le 9 estensioni NON sono in dashboard_v3. Andrebbero integrate. Tre opzioni:

- **a) Una nuova pagina** `esperimento-3-3.html` "Robustness e caveat" con
  D + G + H come headline-strengthening e F + X + Y + Z come caveat
- **b) Integrazione in `esperimento-3-1.html`** (estendendo §3.1.3 con D, H,
  Y, e §3.1.1 con i numeri legal-vs-control già pronti)
- **c) Due pagine separate** `esperimento-3-1.html` + `esperimento-3-1-estensioni.html`

**La scelta (a) è la più pulita** ma allunga la dashboard a 5 pagine.

### 2.3 Reframing critico (Y caveat) — obbligatorio

Il headline Δρ_sym attested = 0.543 è metodologicamente ambiguo se citato
in isolamento. Y mostra che Δρ_sym bare sul pool dei 100 control (0.156) è
indistinguibile da Δρ_sym bare sul core (0.165): **il bare è encoder-tradition
shaped, il legal signal è il gap attested-bare = 0.378**.

La sezione §3.1.3 della dashboard deve presentare:
- Δρ_sym attested 0.543 (numero come computato)
- Δρ_sym bare 0.165 (baseline encoder-tradition)
- Δρ_sym bare on control 0.156 (= baseline)
- **gap legale = 0.378 = 0.543 − 0.165** (highlighted)

Frase canonica disponibile in `HANDOFF.md` §A-bis.

---

## 3. Decisioni da prendere all'inizio della prossima sessione

Tre domande, brevi:

1. **Scope dashboard**: a) minimo riallineamento dati (run #3 → run #4); b) +
   integrazione delle 9 estensioni; c) ridisegno completo (es. Streamlit
   interactive invece di HTML statico Plotly)?

2. **Lingua**: dashboard v3 esistente è in italiano (commissione). Tesi è in
   inglese. Confermare italiano?

3. **Pagina dedicata al Y caveat** (es. "Bare-vs-attested framing" come
   sezione finale di §3.1.3) oppure caveat inline nella sezione 3.1.3
   esistente?

---

## 4. Pointer ai file critici

**Da leggere prima di scrivere codice:**

1. `experiments/ch3-measurability/OVERVIEW.md` — mappa completa della release
2. `experiments/ch3-measurability/HANDOFF.md` — framing del thesis writer
3. `experiments/ch3-measurability/reports/numbers_headline.md` — tabelle pronte
4. `experiments/ch3-measurability/reports/extensions_summary.md` — narrazione 9 extensions
5. `experiments/ch3-measurability/notebooks/analysis.ipynb` — 16 sezioni / 45 celle, cell-by-cell replay (utile per Plotly chart inspiration)
6. `experiments/dashboard_v3/README.md` — pattern concentrico + audience
7. `experiments/dashboard_v3/pages/exp31.py` — esempio della struttura attuale §3.1
8. `experiments/dashboard_v3/data/results_31.py` — loader DA RISCRIVERE

**Convenzioni di citazione nella tesi** (path-relative):
- `experiments/ch3-measurability/reports/numbers_headline.md §3.1.3`
- `experiments/ch3-measurability/experiment_1_structure/results_attested/experiment_1_results.json`
- `experiments/ch3-measurability/ext/Y_control_only/control_only_rsa.json`

---

## 5. Frasi di apertura della nuova sessione

> "Eseguiamo la dashboard definitiva partendo da
> `experiments/ch3-measurability/dashboard_brief.md`. Leggi quel file
> top-to-bottom, poi confermami le tre decisioni di §3 prima di scrivere
> codice."

Oppure più concreto:

> "Aggiorna `experiments/dashboard_v3/data/results_31.py` e `results_32.py`
> per puntare a `experiments/ch3-measurability/` (run #4 post-BLP), poi
> rigenera le 4 pagine HTML e verifica che i numeri matchino
> `reports/numbers_headline.md`. Vedi `dashboard_brief.md` per il contesto
> completo."

---

## 6. Stato finale al momento del handoff

**Run #4 post-BLP completa** (2026-05-17):

- 364 core terms, 9.045 background, 100 control
- 10 encoder models, bare + attested (4 primary anche per bg)
- 9 estensioni A-Z eseguite
- Verification gate **8/8 PASS** (Δρ_sym attested = 0.543; Mantel p_max =
  1e-4; Holm p_max ≤ 1.7e-3)
- 50 SHA-256 nel manifest (7+2 input + 20 output + 30 embedding)
- Nomenclatura ripulita: `lens1/lens4` → `experiment_1_structure/`,
  `experiment_2_axes/`; `postBLP` filename → `bilingual`/curation;
  scripts con nomi descrittivi; `_lib.py` self-contained (802 righe).

**Tre risultati anchor per Cap. 4** (HANDOFF.md):

1. **D + run-#3 stability** → Δρ_sym attested strutturalmente stabile in
   [0.535, 0.590] su due pool indipendentemente curati + 5 mix levels
2. **G + bilingual control** → same-lemma terms hanno cosine cross-encoder
   negativo ma cosine bilingual +0.5/+0.75 → divergenza tradition-shaped,
   non encoder-artefact
3. **Y caveat** → il legal signal è il gap attested-bare (0.378), non
   l'attested absolute (0.543); il bare è encoder-tradition baseline

---

*Compatto e self-contained. La nuova sessione carica questo file + memory
`project_ch3-measurability.md` + CLAUDE.md §10 e ha tutto il contesto.*
