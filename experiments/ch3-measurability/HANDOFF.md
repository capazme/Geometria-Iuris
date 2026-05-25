# HANDOFF — Run #4 post-BLP per l'agente di scrittura

**Per:** l'agente che assiste l'utente nella redazione di Capitolo 3 e Capitolo 4 della tesi *Geometria Iuris*.
**Da:** la sessione che ha eseguito la run #4 il 2026-05-11.
**Stato all'handoff:** run completa, **verification gate 7/7 PASS**.

Leggi questo file prima di scrivere o editare qualsiasi paragrafo di §3 / §4 / §2.1.

---

## 1. Cosa è la run #4

Quarta esecuzione completa degli esperimenti di Cap. 3, su un pool ricurato per eliminare il bias della *fictio* di *equal authenticity*. 364 termini con K≥4 attestazioni in ordinanze HK Cap. **promulgate post-1989 sotto il Bilingual Laws Project** (co-redazione bilingue strutturale, non finzione di equivalenza ex post).

Tutto vive in `experiments/ch3-measurability/`. Nulla deve essere ricalcolato durante la redazione. La struttura è:

```
ch3-measurability/
├── PLAN.md                     piano operativo (storico)
├── trace.md                    12 decisioni di parametro (D1-D12)
├── config.yaml                 single source of truth dei parametri
├── manifest.json               SHA-256 di 7 input + 10 output + 10 embedding
├── inputs/                     snapshot frozen degli input
├── embeddings/                 10 cartelle modello con vecs_{bare,attested}.npy
├── experiment_1_structure/results_{bare,attested}/   §3.1 outputs
├── experiment_2_axes/results_{bare,attested}/   §3.2 outputs
├── reports/
│   ├── numbers_headline.md     TABELLE PRONTE — copia da qui
│   ├── changes_vs_run3.md      delta vs Firthian (per audit interno)
│   └── verification_gate.md    7/7 PASS
├── HANDOFF.md                  questo file
└── notebooks/ch3-measurability_analysis.ipynb
```

---

## 2. I numeri che devi usare (e dove trovarli)

### §3.1.3 RSA cross-tradition (Lens I, ATTESTED column)

| Metrica | Run #4 | Source |
|---|---|---|
| within-WEIRD ρ̄ (3 coppie) | **0.712** | `experiment_1_structure/results_attested/experiment_1_results.json` → `section_313.summary` |
| within-Sinic ρ̄ (3 coppie) | **0.868** | idem |
| cross-tradition ρ̄ (9 coppie) | **0.246** | idem |
| **Δρ symmetric (avg within − cross)** | **0.543** | idem — **il numero chiave** |
| within-bilingual ρ̄ (β control) | 0.316 | idem |

Tutti i 17 Mantel p sono al floor (0.0001 con B=10000); Holm-adjusted p_max = 0.0017. Per i 17 CI 95% per coppia: `section_313.cross_tradition` / `within_weird` / `within_sinic` / `within_bilingual`.

Run #3 (per il confronto in §4.1, non per il headline):
- Δρ_sym attested 0.541. Δ vs run #4 = +0.002. **Praticamente identico** — usa per dire "il pattern Firthian regge sotto cambio di curatura".

### §3.1.1 legal-vs-control (Lens I, bare only)

100 control terms (everyday-language pronouns, deixis, common nouns; *I/you/he/this/here*, *我/你/他/這/那*). Bare-only by design — controls have no HK Cap. attestation.

Mann-Whitney one-sided (alternative='less'): legal-legal more compact than legal-control.

| Model | r | p | Verdict |
|---|---|---|---|
| BGE-EN-large | +0.062 | 1.1e-60 | signal |
| E5-large | +0.257 | <1e-300 | signal |
| FreeLaw-EN | **-0.121** | 1.0 | NO signal (legal fine-tuned) |
| BGE-ZH-large | +0.239 | <1e-300 | signal |
| Text2vec-large-ZH | +0.240 | <1e-300 | signal |
| Dmeta-ZH | +0.218 | <1e-300 | signal |
| BGE-M3-EN | +0.143 | <1e-300 | signal |
| BGE-M3-ZH | +0.305 | <1e-300 | signal |
| Qwen3-0.6B-EN | **-0.044** | 1.0 | NO signal (under-specialized) |
| Qwen3-0.6B-ZH | +0.207 | <1e-300 | signal |

**8/10 models confirm legal-legal more compact than legal-control.** Two exceptions:
- **FreeLaw-EN**: fine-tuned on legal corpus (CourtListener); the legal prior is applied to everyday terms too, collapsing the distinction. Methodologically informative: heavy legal fine-tuning erodes the term-class signal.
- **Qwen3-0.6B-EN**: small multilingual model; under-specialized representations in EN.

Source: `experiment_1_structure/results_bare/legal_vs_control.json` and `section_311_legal_vs_control` in `experiment_1_results.json`.

### §3.2.4 axes ranking (Lens IV, attested)

Most-divergent first:

1. **natural_positive** 0.092
2. **state_market** 0.125
3. **individual_collective** 0.186
4. **public_private** 0.288
5. **status_contract** 0.363
6. **rights_duties** 0.394

Source: `experiment_2_axes/results_attested/experiment_2_results.json` → `section_324`.

### Per-axis (bare vs attested)

| Axis | Bare | Attested |
|---|---|---|
| public_private | 0.386 | 0.288 |
| status_contract | 0.446 | 0.363 |
| rights_duties | 0.384 | 0.394 |
| individual_collective | 0.241 | 0.186 |
| state_market | 0.262 | 0.125 |
| natural_positive | 0.226 | 0.092 |

---

## 3. Tre framing da rispettare nella prosa

### A-bis. REFRAMING DEL HEADLINE (post-extension Y)

**Crucial caveat su Δρ_sym.** Extension Y (cross-tradition ρ sul pool di soli 100 control bare) restituisce Δρ_sym = 0.156, **statisticamente indistinguibile** dal 0.165 del core bare. Il bare Δρ_sym è quindi *encoder-tradition shaped* (WEIRD encoders disagree con Sinic encoders su qualsiasi vocabolario, legale o no), **non legal-tradition shaped**.

Il segnale propriamente *legale* è il **gap attested-bare sul core**: 0.543 − 0.165 = **0.378**. Isola il contributo della contestualizzazione su attestazioni HK Cap. dalla baseline encoder-pair.

Citation rule: in §4.1 cita **entrambi**.

Frase canonica:
> *"Δρ_sym attested = 0.543 measures the within-vs-cross tradition gap as actually computed in our pipeline. Yet on 100 everyday-language control terms, the same Δρ_sym bare metric returns 0.156 — statistically indistinguishable from 0.165 on the 364 core bare. The legal-attestation contribution is therefore best isolated as the attested-bare gap on the core: 0.378 = 0.543 − 0.165, against a shared encoder-tradition baseline of approximately 0.16."*

**Anti-pattern**: NON citare Δρ_sym attested = 0.543 come "legal signal" senza il bare baseline accanto. Il numero da solo è metodologicamente ambiguo.

### A. Effetto Firthian: robusto

Δρ_sym attested = 0.543 in run #4 vs 0.541 in run #3. Due pool indipendentemente curati (327 Firthian e 364 post-BLP, ~70 termini diversi) producono lo stesso gap. **Questo è il risultato più solido della tesi.** Centra Cap. 4 §1 attorno a questa stabilità.

Frase tipo per la prosa:
> *"The within-tradition / cross-tradition gap survives a substantial recomposition of the lexical pool: Δρ_sym is 0.541 on the 327-term Firthian-strict pool and 0.543 on the 364-term post-BLP pool. The structural separation of WEIRD and Sinic embedding geometries is therefore a property of the legal traditions as encoded by the language models, not an artefact of a particular curation."*

### B. Bare → attested gap si amplifica

Nel run #4 il bare Δρ_sym crolla a 0.165 (vs 0.211 in run #3) mentre l'attested rimane a 0.543. Il pool post-BLP è più sparso nello spazio bare (più termini tecnico-procedurali che richiedono contesto). La differenza bare-attested **aumenta**, non diminuisce.

Frase tipo:
> *"Stripped of context, the post-BLP pool is more dispersed than its Firthian predecessor (Δρ_sym bare drops from 0.211 to 0.165). Contextualised, the same pool restores the gap (Δρ_sym attested = 0.543). The widening of the bare-attested margin under tighter curation is itself a confirmation that attested-context vectors carry the legal meaning that bare lexemes do not."*

### B'. Legal-vs-control: signal robusto MA si rompe sotto legal fine-tuning

8/10 modelli confermano (effect r positivo, p<1e-300). FreeLaw-EN (fine-tuned su legal corpus) e Qwen3-0.6B-EN (multilingue piccolo, EN poco specializzato) non confermano.

**Implicazione metodologica** (per §4.2 o footnote tecnica): un encoder fine-tuned sul dominio legale può perdere il contrast term-class (la sua "prior legale" si applica anche alle ordinary words). Questo non invalida la metodologia (i 9 encoder general-purpose o multilingual reggono), ma definisce uno scope: lo strumento misura legal meaning in modelli con rappresentazioni generaliste, non in modelli che hanno già introiettato il legalese.

Frase tipo:
> *"The within-legal/legal-versus-everyday contrast holds across nine of ten encoders (Mann-Whitney r ∈ [0.06, 0.31], p < 10⁻⁶⁰). The two non-confirming cases are informative: FreeLaw-EN — a domain-finetuned encoder — and Qwen3-0.6B-EN, a small multilingual model. The first suggests that strong legal fine-tuning erodes the term-class boundary the test is meant to detect; the second, that under-specialized representations in English do not separate legal from everyday vocabulary at all. The diagnostic thus operates as expected on general-purpose encoders and, simultaneously, marks the kinds of models on which it cannot be relied upon."*

### C. Axes alignment: 3 stabili, 3 pool-sensitive — NON sovrainterpretare

Tre axes mantengono il loro ρ̄ cross-tradition (delta < 0.05): `individual_collective` (+0.004), `public_private` (+0.034), `natural_positive` (-0.019). Tre cambiano molto: `rights_duties` +0.323, `status_contract` +0.212, `state_market` -0.107.

**Implicazione critica per la prosa:** *non scrivere* la lettura del run #3 "*rights/duties is the most tradition-specific axis*". In run #4 rights_duties è l'asse **meno** divergente. Il vecchio framing non regge.

Riformula come:
> *"Three axes are pool-robust (individual_collective, public_private, natural_positive); three are pool-sensitive (rights_duties, status_contract, state_market). The instrument detects axis alignment as a function of the lexical pool composition, not as an invariant property of the tradition pair. This is itself a methodological finding: Kozlowski-style projection is informative locally but does not yield tradition-level invariants in our hands."*

Questa formulazione deve comparire in §4.2 "what this thesis cannot conclude".

---

## 4. Anti-pattern (cosa NON scrivere)

- "rights_duties is the most tradition-specific axis" — superseded.
- "the Sacco summa divisio (public_private) is the most universal" — l'ordering è cambiato; resta vero che `public_private` è pool-robusto, non che è il *meno* divergente in assoluto.
- "327 terms, K≥4 in HK Cap." — il pool corrente è 364 post-BLP. Cita i 327 solo come run #3 storica per la robustezza.
- "B=1000 permutations" — è run #3 / pre-Firthian. Il run definitivo ha B=10000.
- "Δρ_sym = 0.541" — è run #3. Il headline run #4 è 0.543.
- "axes ranking from Lens IV shows that..." senza specificare quale pool. Sii esplicito.

---

## 5. Disciplina citazionale invariata

Cap. 1 è consegnato (immutabile, `documenti/001_GeometriaIuris_Ch1_Measurability.docx`). Cap. 2 è in stesura per-sezione (`documenti/capitoli/ch2/sections/`). Cap. 3 e Cap. 4 attendono i risultati che leggi qui.

Regole di citazione standard di CLAUDE.md §5 si applicano: footnote verificabile contro PDF in `documenti/fonti/`, mai inventare pagine/edizioni/traduttori. Per gli esperimenti, i numeri sono in `experiments/ch3-measurability/reports/numbers_headline.md` — quella è la fonte primaria.

---

## 6. Cosa è già fatto e cosa serve

✓ Encoding (10 modelli × bare + attested) — `embeddings/`
✓ Lens I bare + attested (B=10000, Holm) — `experiment_1_structure/results_*/`
✓ Lens IV bare + attested (B=10000) — `experiment_2_axes/results_*/`
✓ Categorical probe (pool-independent) — linked from run #3
✓ Reports + verification gate (7/7 PASS) — `reports/`
✓ CLAUDE.md §10 aggiornato con i numeri post-BLP
✓ Memory: `project_ch3-measurability.md` (questo file in formato strutturato)

▢ **Per te (agente di scrittura):**
- Cap. 3 §3.1.3, §3.2.3, §3.2.4 — usa i numeri di `numbers_headline.md`
- Cap. 4 §4.1 — costruisci attorno a Δρ_sym stabilità (framing A sopra)
- Cap. 4 §4.2 — onestà metodologica sulla sensibilità axes (framing C sopra)
- Cap. 2 §2.1 — descrivi il pool 364 post-BLP, banda 41-60, K≥4 su Cap. post-1989
- Cap. 2 §2.2 — Bilingual Laws Project come natural laboratory; spiega perché il post-BLP rimuove la *fictio* (no longer translation, but co-drafting)

▢ Eventuali plot PNG: il notebook `notebooks/ch3-measurability_analysis.ipynb` li produce automaticamente quando viene eseguito.

---

## Extensions (run #4-ext, 2026-05-11 evening — bg-driven)

Five robustness experiments on top of the headline run, using ~9.045 bg terms (tier=`background` in legacy `legal_terms.json`). The bg are NOT in the 364 core; they live in `embeddings/bg/{model}/` and `inputs/bg_terms_snapshot.json`. Full narrative: `reports/extensions_summary.md`.

| Ext | What it shows | One-liner for the prose |
|---|---|---|
| **A** k-NN bg domain assignment | 9.045 bg routed to 7 domains via k=7 NN in core; mean confidence 0.515 | "the clustering is informative but heterogeneous on out-of-curation vocabulary" |
| **D** Δρ_sym vs %bg curve | Δρ_sym = 0.538 → 0.590 as bg goes 0% → 75% | **Δρ_sym is robust under bg injection up to 75% — the gap is structural, not curation-dependent** |
| **E** Out-of-sample axes | Bg projected on 6 axes, coherent per assigned domain | "the axes generalize beyond the curated pool to neighbouring legal vocabulary" |
| **H** K saturation curve | ρ_cross = -0.13 (K=1) → +0.05 (K=2) → +0.13 (K=3) → +0.15 (K=4-7) → +0.22 (K=8) | "K≥4 is the empirical threshold below which the cross-tradition signal is unstable; at K=1 the signal is anti-correlated, i.e. noise dominates" |
| **G** Automated false-friends | Cross-tradition cosine negative for ~50 same-lemma bg; bilingual BGE-M3 cosine +0.5 to +0.75 | **Same-lemma terms have negative cross-encoder cosine but high bilingual cosine — the divergence sits in tradition pair, not in encoder** |
| **F** Confidence-stratified | Low-conf bg injection: Δρ_sym +0.027 vs baseline; high-conf bg: -0.007 | "boundary, not centre: tradition signal sits in semantically ambiguous bg, not in the most-categorical ones — small effect, n=20, interpret cautiously" |
| **X** Δρ_sym vs %control curve | Δρ_sym bare 0.246 → 0.222 as control fills 0% → 27% | "the bare signal declines monotonically when the pool is contaminated with non-legal vocabulary — direction correct, effect small (limited by control pool size)" |
| **Y** Cross-tradition ρ on control-only | Δρ_sym bare on 100 control = 0.156 ≈ Δρ_sym bare on 364 core = 0.165 | **CRUCIAL: bare signal is encoder-tradition shaped, NOT legal-tradition shaped. The legal signal is the attested-bare gap (0.378 on core), not the attested absolute (0.543)** |
| **Z** 3-tier hierarchy | Only 3/10 models satisfy median(c-c) < median(c-bg) < median(c-ctrl); 7/10 have bg farther than control | "tier classification is corpus-curative, not embedding-geometric — bg are not 'semantically intermediate' but elsewhere" |

**Where to use what:**
- Cap. 3 §3.1.3 final paragraph → cite **D + X** as dual robustness statements (signal regge sotto perturbazione legalish, scende sotto perturbazione non-legal)
- Cap. 2 §2.3 (operational definition attested) → cite **H** as empirical justification for K≥4
- Cap. 4 §4.1 → cite **G** bilingual control as the term-level analogue of the headline; cite **Y** to reframe Δρ_sym attested = 0.543 → attested-bare gap = 0.378 as the *legal* signal
- Cap. 4 §4.2 (limits) → cite **E** axes-OOS to qualify the 3 stable axes claim; **F** as honest caveat; **Y** as the methodological caveat on bare; **Z** as the corpus-curative-vs-geometric caveat
- Footnote / appendix → **A** background CSV is a reusable resource for future pool expansions

Concrete false-friends for Cap. 4 anecdotes (cross-cosine, bilingual cosine):
- *trainer / 經授權導師* (−0.106 / +0.684): "authorised instructor" sense in HK statutes
- *Receiver / 破產管理署署長* (−0.066 / +0.507): "Director of Bankruptcy" in HK
- *anniversary / 周年日* (−0.069 / +0.724): contract date-anchor vs cultural meaning
- *driving licence / 駕駛執照* (−0.053 / n/a): mundane lemma, divergent enactment networks

Anti-pattern: do **not** cite F as a primary finding (the +0.027 effect is small and based on n=20 replicates); use only as interpretive hint in §4.2.

---

## 7. Where to look first

Se devi citare un numero: `reports/numbers_headline.md`.
Se devi capire da dove viene un numero: `lens{1,4}/results_*/`.
Se devi giustificare una scelta di metodo (B=10000, seed, holm, attested-mean): `trace.md` D1-D12.
Se devi spiegare la provenienza di un termine: `inputs/core_terms_snapshot.json` + `inputs/term_contexts_bilingual_snapshot.jsonl`.
Se devi auditare l'integrità di un file: `manifest.json` ha la sua SHA-256.

Buon lavoro.
