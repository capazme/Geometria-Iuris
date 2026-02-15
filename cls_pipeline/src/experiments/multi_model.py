"""
multi_model.py — Orchestratore per analisi multi-modello.

Esegue ogni esperimento su N coppie di modelli (prodotto cartesiano
weird × sinic) e aggrega i risultati, dimostrando che le differenze
trovate non dipendono dal singolo modello ma dalla tradizione culturale.

Questo approccio risolve il confound modello/cultura: se la correlazione
RSA (o GW, FM, Jaccard) è consistente tra 9 coppie di modelli diversi,
l'effetto è robusto alla scelta del modello specifico.

References
----------
Liang et al. (2022) "Probing language model representations with
multi-model agreement", EMNLP.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.config_loader import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class MultiModelResult:
    """Risultato aggregato su N coppie di modelli."""
    pair_results: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    model_pairs: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": len(self.pair_results),
            "model_pairs": [
                {"weird": w, "sinic": s} for w, s in self.model_pairs
            ],
            "pair_results": self.pair_results,
            "aggregate": self.aggregate,
        }


def run_experiment_multi_model(
    experiment_fn: Callable,
    model_pairs: list[tuple[ModelConfig, ModelConfig]],
    client: Any,
    texts_weird: list[str],
    texts_sinic: list[str],
    stat_key: str = "spearman_r",
    **experiment_kwargs,
) -> MultiModelResult:
    """
    Esegue un esperimento su N coppie di modelli e aggrega i risultati.

    Per ogni coppia (model_w, model_s):
    1. Genera embedding con client.get_embeddings_for_model()
    2. Esegue l'esperimento
    3. Raccoglie il risultato

    Poi aggrega: media, mediana, std, min, max della statistica chiave.

    Parameters
    ----------
    experiment_fn : callable
        Funzione esperimento con firma (emb_w, emb_s, labels, **kwargs) -> result.
        Il risultato deve avere to_dict() e l'attributo stat_key.
    model_pairs : list[tuple[ModelConfig, ModelConfig]]
        Coppie (weird, sinic) da testare.
    client : EmbeddingClient
        Client per generare embedding.
    texts_weird : list[str]
        Testi lato WEIRD (inglese).
    texts_sinic : list[str]
        Testi lato Sinic (cinese).
    stat_key : str
        Nome dell'attributo da aggregare (es. "spearman_r", "distance").
    **experiment_kwargs
        Parametri aggiuntivi passati a experiment_fn.

    Returns
    -------
    MultiModelResult
        Risultato con pair_results e statistiche aggregate.
    """
    pair_results = []
    model_pair_names = []
    stat_values = []

    for i, (model_w, model_s) in enumerate(model_pairs):
        logger.info(
            "Multi-model coppia %d/%d: %s × %s",
            i + 1, len(model_pairs), model_w.label, model_s.label,
        )

        # Genera embedding per questa coppia
        emb_w = client.get_embeddings_for_model(texts_weird, model_w)
        emb_s = client.get_embeddings_for_model(texts_sinic, model_s)

        # Esegui l'esperimento
        result = experiment_fn(emb_w, emb_s, **experiment_kwargs)

        # Raccogli risultato
        result_dict = result.to_dict()
        result_dict["model_weird"] = model_w.label
        result_dict["model_sinic"] = model_s.label
        pair_results.append(result_dict)
        model_pair_names.append((model_w.label, model_s.label))

        # Statistica chiave
        stat_val = getattr(result, stat_key, None)
        if stat_val is not None:
            stat_values.append(float(stat_val))
            logger.info(
                "  %s = %.4f (%s × %s)",
                stat_key, stat_val, model_w.label, model_s.label,
            )

    # Aggregazione
    aggregate = {}
    if stat_values:
        arr = np.array(stat_values)
        aggregate = {
            "stat_key": stat_key,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_pairs": len(stat_values),
            "values": stat_values,
        }
        logger.info(
            "Multi-model aggregato (%s): media=%.4f, std=%.4f, range=[%.4f, %.4f]",
            stat_key, aggregate["mean"], aggregate["std"],
            aggregate["min"], aggregate["max"],
        )

    return MultiModelResult(
        pair_results=pair_results,
        aggregate=aggregate,
        model_pairs=model_pair_names,
    )
