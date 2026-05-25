# Run #4 — post-BLP final

Quarta (e probabilmente ultima) esecuzione completa degli esperimenti di
Cap. 3 della tesi *Geometria Iuris*. Pool: 364 termini post-1989 curati
(banda 41-60 per dominio, 7 domini).

## Cosa c'è in questa folder

| File / folder | Cosa contiene |
|---|---|
| `PLAN.md` | il piano operativo dettagliato — leggere prima di iniziare |
| `README.md` | questo file, orientamento veloce |
| `inputs/` | snapshot frozen degli input (popolato in step 0) |
| `embeddings/` | precomputed bare + attested per i 10 modelli (popolato in step 2) |
| `experiment_1_structure/` | output §3.1 (RSA, Mantel, bootstrap) — popolato in step 3-4 |
| `experiment_2_axes/` | output §3.2 (Kozlowski axes) — popolato in step 5 |
| `reports/` | numeri headline + plot diagnostic — popolato in step 6 |
| `notebooks/` | analysis narrativa end-to-end — popolato in step 7 |
| `scripts/` | gli script Python di ciascuno step (creati al momento dell'esecuzione) |
| `config.yaml` | tutti i parametri della run (B=10000, seed=42, 10 modelli) |
| `manifest.json` | hash SHA-256 degli input + output per audit |
| `trace.md` | 12 decisioni di parametro (D1-D12) |

## Come si esegue

Dal repo root:

```bash
# Step 0 — seal inputs (5 min)
python3 experiments/ch3-measurability/scripts/00_seal_inputs.py

# Step 2 — re-encoding (6-8h, overnight)
python3 experiments/ch3-measurability/scripts/02_encode_core.py

# Step 3-5 — analisi
python3 experiments/ch3-measurability/scripts/03_structure_bare.py
python3 experiments/ch3-measurability/scripts/04_structure_attested.py
python3 experiments/ch3-measurability/scripts/05_axes_experiment.py

# Step 6 — reports
python3 experiments/ch3-measurability/scripts/06_reports.py

# Step 7 — notebook narrativo (manuale in jupyter)
jupyter notebook experiments/ch3-measurability/notebooks/ch3-measurability_analysis.ipynb
```

## Cosa NON è in questa folder

- Il dataset stesso (resta in `experiments/data/processed/legal_terms.json`)
- Il codice di calcolo riusabile (resta in `experiments/lens_1_relational/`,
  `experiments/lens_4_values/`, `experiments/shared/`). Gli script di
  questa folder sono solo orchestrazione + parametri.
- Le run precedenti (run #3 Firthian baseline è in
  `experiments/lens_*/results*/`, intatte).

## Isolamento da run #3

Run #4 produce output in `experiments/ch3-measurability/`, **non sovrascrive**
nessun file di run #3. Si può tornare a run #3 in qualsiasi momento.

Vedi `PLAN.md` §10 per il rollback plan.
