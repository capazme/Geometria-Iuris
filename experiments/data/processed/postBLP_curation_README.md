# Curazione post-BLP — istruzioni operative

## File chiave

- `cap_enactment_years.json` — Cap → year (top 50 verificati + heuristic)
- `postBLP_pool_diagnostic.json` — diagnostic per ogni termine (read-only)
- `postBLP_curation_longlist.csv` — la long list da curare a mano

## Stato della long list (1271 righe)

Le righe sono divise in 5 stati nella colonna `curation_decision` + `auto_recommendation`:

| Stato | Decisione default | Cosa fare |
|---|---|---|
| `AUTO_KEEP` (85) | tieni | spot-check facoltativo, niente azione obbligatoria |
| `RECOMMEND_KEEP` (~250) | tieni | rivedi: confidence k-NN ≥0.71 e k_postBLP alto. Quasi sempre OK. |
| `REVIEW` (~660) | da decidere | **questo è il lavoro vero**: scegli ~30-40 per dominio |
| `RECOMMEND_DROP` | scarta | rivedi solo se ti sembra dubbio |
| `AUTO_DROP` (274) | scarta | spot-check facoltativo |

## Target finale: 60 termini per dominio × 7 = 420 totali

Per ciascun dominio, dopo la curazione manuale:
- AUTO_KEEP + RECOMMEND_KEEP marcati come `KEEP` ≈ 30-50 per dominio (non saranno tutti tenuti, valuta)
- Aggiungi dai REVIEW finché arrivi a 60
- I rimanenti REVIEW e i RECOMMEND_DROP/AUTO_DROP restano fuori

## Come marcare la decisione finale

Nella colonna `curation_decision`, sostituisci il valore esistente con uno di:
- `KEEP` — entra nel nuovo core post-BLP
- `DROP` — fuori dal pool
- `DEFER` — non deciso (lascia così, non entra)

Eventuale spiegazione in `curation_notes`.

## Tempo stimato

- AUTO_KEEP: spot-check 5-10 min (85 righe)
- RECOMMEND_KEEP: revisione veloce 30-60 min (~250 righe)
- REVIEW: lavoro core 5-8h (~660 righe, ~95/dominio per evitare bias)
- AUTO_DROP / RECOMMEND_DROP: ignorabile

Totale: 6-10h di curazione manuale focalizzata.

## Quando hai finito

Salvami il CSV con la colonna `curation_decision` aggiornata. Da lì proseguo con:
1. Re-extract contesti post-1989 dai termini KEEP
2. Re-encoding 10 modelli
3. Re-run §3.1 + §3.2
4. Update §2.1, §2.2, §3.x della tesi + dashboard v3
