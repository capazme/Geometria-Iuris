# Dashboard finale · *Geometria Iuris*

Compagna del Capitolo 3 della tesi. Sostituisce `experiments/dashboard_v3/`.

## In due righe

- **Cosa**: 6 pagine HTML statiche autoconsistenti che presentano i due esperimenti di §3.1 e §3.2, le 9 estensioni A–Z, e il caveat Y che riformula il headline number.
- **Come si apre**: doppio clic su `output/home.html`. Niente server, niente internet, niente installazioni.

## Come si distribuisce

Una sola cartella, ~4.5 MB, zippabile e condivisibile:

```bash
cd experiments/dashboard_final
zip -r dashboard_final.zip output/
```

Chi riceve lo zip estrae e apre `output/home.html` con qualsiasi browser. Plotly è vendored in `output/assets/plotly.min.js` (4.4 MB) — l'unica dipendenza, già inclusa.

## Come si rigenera

Dalla radice del repo:

```bash
python3 experiments/dashboard_final/build.py
```

Lo script:

1. Carica i dati definitivi da `experiments/ch3-measurability/` (run #4 post-BLP, 2026-05-17, verification gate 8/8 PASS, 50 SHA-256 in `manifest.json`).
2. Costruisce le 6 pagine via i moduli in `pages/` (importano figure pure da `figures/` e loader da `data/`).
3. Scrive `output/{home, methodology, how_it_works, experiment_31, experiment_32, robustness_caveats}.html`.
4. Copia `assets/plotly.min.js` e `assets/style.css` in `output/assets/`.
5. Stampa un report con le dimensioni di ogni artefatto.

Tempo di build: ~1 s. Idempotente.

## Decisioni di stack

| Decisione | Scelta | Motivo |
|---|---|---|
| Rendering | HTML statico + Plotly vendored | Self-contained, niente runtime, niente CDN, niente account |
| Lingua | Inglese | Coerenza con la tesi; audience commissione LUISS |
| Layout | 6 pagine + sticky nav lineare | Lettura sequenziale per la commissione |
| Y caveat | Pagina dedicata "Robustness & caveats" | Anchor anchor, non inline in §3.1.3 |
| Formule | Unicode + HTML, niente KaTeX | Self-contained, niente font extra |
| Interattività | Hover/zoom Plotly nativi, niente JS custom | Sufficiente per la commissione, zero superficie di bug |

## Struttura del codice

```
dashboard_final/
├── build.py                          entry point
├── README.md                         this file
├── shared_ui.py                      palette, CSS, helpers HTML, Plotly defaults
├── apparatus.py                      apparatus_block() per L2 technical
├── pages/
│   ├── home.py                       gateway + verification gate badge + 3 anchor cards
│   ├── methodology.py                10 modelli + dataset + statistical toolkit
│   ├── how_it_works.py               pipeline narrative + bare-vs-attested
│   ├── experiment_31.py              §3.1.1–§3.1.4
│   ├── experiment_32.py              §3.2.1–§3.2.5
│   └── robustness_caveats.py         D + G + H + Y + F + X + Z + A + E
├── data/
│   ├── loader_31.py                  → ch3-measurability/experiment_1_structure/
│   ├── loader_32.py                  → ch3-measurability/experiment_2_axes/
│   └── loader_extensions.py          → ch3-measurability/ext/A..Z/
├── figures/
│   ├── exp31.py                      Plotly figure factories per §3.1
│   ├── exp32.py                      Plotly figure factories per §3.2
│   └── extensions.py                 Plotly figure factories per le 9 ext
├── assets/
│   ├── plotly.min.js                 vendored (Plotly 2.35.2, 4.3 MB)
│   └── style.css                     [auto-generato da build.py]
└── output/                           [auto-generato da build.py — 6 HTML + assets]
```

## Numeri canonici (spot check)

Tutti citati verbatim dai JSON in `experiments/ch3-measurability/`:

| Numero | Dove | File JSON di origine |
|---|---|---|
| Δρ_sym attested = **0.543** | Home anchor, §3.1.3, Robustness Y | `experiment_1_structure/results_attested/experiment_1_results.json` |
| Δρ_sym bare = **0.165** | §3.1.3, Robustness Y | `experiment_1_structure/results_bare/experiment_1_results.json` |
| Legal gap = **0.378** | Home anchor, Robustness Y (callout) | `ext/Y_control_only/control_only_rsa.json` (= 0.543 − 0.165) |
| natural_positive ρ̄_cross = **0.092** | §3.2.4 most divergent | `experiment_2_axes/results_attested/experiment_2_results.json` |
| Verification gate **8 / 8 PASS** | Home, Methodology | `reports/verification_gate.md` |

## Audience

Commissione di laurea LUISS:

- **Relatore (giurista)** — legge la prosa principale concentrica, ignora gli apparati tecnici collassati.
- **Co-relatore (ingegnere)** — apre gli `<details>` degli apparati tecnici, verifica formule + p-value + code-ref.
- **Prof. giurista esterno** — usa la sticky nav per saltare alla sezione di interesse; la Home + Robustness Y caveat sono la roadmap.

## Limiti noti

- **Niente accessibilità screen-reader formale**: le figure hanno alt-text inline nelle didascalie, ma non c'è un audit ARIA completo. Statico HTML, niente live region.
- **Niente UMAP/PCA 3D**: dashboard v3 aveva uno scatter 3D delle 327 termini; non riportato qui per economia di scope (i nuovi 364 termini hanno geometria simile; il punto di Cap. 3 non richiede la visualizzazione 3D per essere fatto).
- **§3.2.5 top-divergent terms**: nel `experiment_2_results.json` la sezione 325 può essere vuota; la pagina si auto-degrada a stub con rimando al testo della tesi.

## Source of truth

Tutti i dati provengono da `experiments/ch3-measurability/` (run #4, frozen 2026-05-17). Niente è hand-typed nei HTML. Per i singoli numeri vedere `experiments/ch3-measurability/reports/numbers_headline.md`.

---

*Versione: 2026-05-18. Distribuzione: cartella `output/` + `assets/plotly.min.js` zippabile (~4.5 MB).*
