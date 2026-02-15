"""
test_analogy.py — Sanity check: l'aritmetica vettoriale funziona
con i sentence-transformers usati nella pipeline?

Test classico: king - man + woman ≈ queen? (Mikolov et al. 2013)
Se fallisce, NDA Part B (decomposizioni normative) va ripensata.

Esegui con:
    cd cls_pipeline
    python scripts/test_analogy.py
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configurazione ─────────────────────────────────────────────────
MODELS = {
    "E5-large-v2 (WEIRD)": {
        "name": "intfloat/e5-large-v2",
        "prefix": "query: ",
    },
    "BGE-ZH-v1.5 (Sinic)": {
        "name": "BAAI/bge-large-zh-v1.5",
        "prefix": "",
    },
}

# Analogie classiche (EN)
EN_ANALOGIES = [
    # (A, B, C, atteso D) → A - B + C ≈ D
    ("king", "man", "woman", "queen"),
    ("Paris", "France", "Germany", "Berlin"),
    ("walking", "walk", "swim", "swimming"),
    ("bigger", "big", "small", "smaller"),
]

# Analogie giuridiche (EN) — queste interessano per la tesi
EN_LEGAL_ANALOGIES = [
    ("law", "state", "nature", "natural law"),
    ("judge", "court", "legislature", "legislator"),
    ("crime", "punishment", "reward", "merit"),
    ("contract", "agreement", "dispute", "litigation"),
]

# Analogie cinesi (ZH) — per BGE-ZH
ZH_ANALOGIES = [
    ("国王", "男人", "女人", "王后"),     # king - man + woman = queen
    ("巴黎", "法国", "德国", "柏林"),     # Paris - France + Germany = Berlin
]

ZH_LEGAL_ANALOGIES = [
    ("法律", "国家", "自然", "自然法"),    # law - state + nature = natural law
    ("法官", "法院", "立法机关", "立法者"),  # judge - court + legislature = legislator
]

# Vocabolario per ricerca nearest neighbors
EN_VOCAB = [
    # Royalty / gender
    "king", "queen", "prince", "princess", "man", "woman", "boy", "girl",
    "monarch", "ruler", "emperor", "empress",
    # Geography
    "Paris", "France", "Germany", "Berlin", "London", "England", "Rome", "Italy",
    "Tokyo", "Japan", "Beijing", "China",
    # Verbs
    "walking", "walk", "swim", "swimming", "run", "running", "talk", "talking",
    # Adjectives
    "bigger", "big", "small", "smaller", "larger", "large", "tiny", "huge",
    # Legal
    "law", "state", "nature", "natural law", "justice", "right", "duty",
    "obligation", "contract", "agreement", "dispute", "litigation",
    "judge", "court", "legislature", "legislator", "lawyer", "attorney",
    "crime", "punishment", "reward", "merit", "penalty", "sentence",
    "constitution", "sovereignty", "democracy", "authority", "power",
    "freedom", "liberty", "rights", "equity", "fairness",
]

ZH_VOCAB = [
    "国王", "王后", "王子", "公主", "男人", "女人", "男孩", "女孩",
    "君主", "统治者", "皇帝", "皇后",
    "巴黎", "法国", "德国", "柏林", "伦敦", "英国", "罗马", "意大利",
    "东京", "日本", "北京", "中国",
    "法律", "国家", "自然", "自然法", "正义", "权利", "义务",
    "契约", "协议", "纠纷", "诉讼",
    "法官", "法院", "立法机关", "立法者", "律师",
    "犯罪", "惩罚", "奖励", "功绩", "刑罚", "判决",
    "宪法", "主权", "民主", "权威", "权力",
    "自由", "人权", "公平", "公正",
]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalizza a norma unitaria."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def embed_texts(model: SentenceTransformer, texts: list[str], prefix: str) -> np.ndarray:
    """Embed e normalizza."""
    encode_texts = [f"{prefix}{t}" for t in texts] if prefix else texts
    emb = model.encode(encode_texts, convert_to_numpy=True, show_progress_bar=False)
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return emb / norms


def test_analogy(
    model: SentenceTransformer,
    prefix: str,
    a: str, b: str, c: str, expected: str,
    vocab: list[str],
    vocab_emb: np.ndarray,
    vocab_lookup: dict[str, int],
) -> dict:
    """
    Testa A - B + C ≈ D?

    Restituisce il ranking di 'expected' e i top-5 vicini.
    """
    # Embed A, B, C
    emb = embed_texts(model, [a, b, c], prefix)
    vec_a, vec_b, vec_c = emb[0], emb[1], emb[2]

    # Aritmetica vettoriale
    residual = vec_a - vec_b + vec_c
    residual = l2_normalize(residual)

    # Cosine similarity col vocabolario
    sims = cosine_similarity(residual.reshape(1, -1), vocab_emb)[0]

    # Escludi A, B, C dai risultati
    exclude = {a.lower(), b.lower(), c.lower()}
    scored = [
        (vocab[i], float(sims[i]))
        for i in range(len(vocab))
        if vocab[i].lower() not in exclude
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Rank di 'expected'
    rank = None
    for i, (word, _) in enumerate(scored):
        if word.lower() == expected.lower():
            rank = i + 1
            break

    return {
        "formula": f"{a} - {b} + {c} = ?",
        "expected": expected,
        "rank": rank,
        "top5": scored[:5],
        "expected_sim": sims[vocab_lookup.get(expected, 0)] if expected in vocab_lookup else None,
    }


def test_subtraction(
    model: SentenceTransformer,
    prefix: str,
    a: str, b: str,
    vocab: list[str],
    vocab_emb: np.ndarray,
) -> dict:
    """
    Testa A - B → ? (decomposizione, come in NDA Part B).

    Restituisce i top-5 vicini del residuo.
    """
    emb = embed_texts(model, [a, b], prefix)
    residual = emb[0] - emb[1]
    residual = l2_normalize(residual)

    sims = cosine_similarity(residual.reshape(1, -1), vocab_emb)[0]

    exclude = {a.lower(), b.lower()}
    scored = [
        (vocab[i], float(sims[i]))
        for i in range(len(vocab))
        if vocab[i].lower() not in exclude
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        "formula": f"{a} - {b} = ?",
        "top5": scored[:5],
    }


def run_model_tests(model_label: str, model_config: dict, is_chinese: bool):
    """Esegue tutti i test per un modello."""
    print(f"\n{'=' * 70}")
    print(f"  {model_label}")
    print(f"{'=' * 70}")

    model_name = model_config["name"]
    prefix = model_config["prefix"]

    # Cache dir
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print(f"  Caricamento modello: {model_name}...")
    model = SentenceTransformer(model_name, cache_folder=str(models_dir), trust_remote_code=True)

    vocab = ZH_VOCAB if is_chinese else EN_VOCAB
    analogies = ZH_ANALOGIES if is_chinese else EN_ANALOGIES
    legal_analogies = ZH_LEGAL_ANALOGIES if is_chinese else EN_LEGAL_ANALOGIES

    print(f"  Embedding vocabolario ({len(vocab)} parole)...")
    vocab_emb = embed_texts(model, vocab, prefix)
    vocab_lookup = {w: i for i, w in enumerate(vocab)}

    # ── Test analogie classiche ────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  ANALOGIE CLASSICHE (A - B + C ≈ D?)")
    print(f"  {'─' * 50}")

    for a, b, c, expected in analogies:
        result = test_analogy(model, prefix, a, b, c, expected, vocab, vocab_emb, vocab_lookup)
        rank_str = f"rank #{result['rank']}" if result['rank'] else "NON TROVATO nel vocab"
        print(f"\n  {result['formula']}")
        print(f"  Atteso: {expected} → {rank_str}")
        print(f"  Top 5:")
        for word, sim in result['top5']:
            marker = " <<<" if word.lower() == expected.lower() else ""
            print(f"    {word:<25} sim={sim:.4f}{marker}")

    # ── Test analogie giuridiche ───────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  ANALOGIE GIURIDICHE (A - B + C ≈ D?)")
    print(f"  {'─' * 50}")

    for a, b, c, expected in legal_analogies:
        result = test_analogy(model, prefix, a, b, c, expected, vocab, vocab_emb, vocab_lookup)
        rank_str = f"rank #{result['rank']}" if result['rank'] else "NON TROVATO"
        print(f"\n  {result['formula']}")
        print(f"  Atteso: {expected} → {rank_str}")
        print(f"  Top 5:")
        for word, sim in result['top5']:
            marker = " <<<" if word.lower() == expected.lower() else ""
            print(f"    {word:<25} sim={sim:.4f}{marker}")

    # ── Test decomposizioni (A - B = ?) ────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  DECOMPOSIZIONI (A - B = ?) — stile NDA Part B")
    print(f"  {'─' * 50}")

    if is_chinese:
        subtractions = [
            ("法律", "国家"),      # law - state
            ("正义", "权力"),      # justice - power
            ("权利", "义务"),      # rights - duty
            ("宪法", "民主"),      # constitution - democracy
        ]
    else:
        subtractions = [
            ("law", "state"),
            ("justice", "power"),
            ("rights", "duty"),
            ("constitution", "democracy"),
        ]

    for a, b in subtractions:
        result = test_subtraction(model, prefix, a, b, vocab, vocab_emb)
        print(f"\n  {result['formula']}")
        print(f"  Top 5 vicini del residuo:")
        for word, sim in result['top5']:
            print(f"    {word:<25} sim={sim:.4f}")

    del model
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("  SANITY CHECK: Aritmetica Vettoriale con Sentence-Transformers")
    print("  Se king - man + woman ≠ queen, NDA Part B va ripensata.")
    print("=" * 70)

    # Test EN model
    run_model_tests(
        "E5-large-v2 (WEIRD)",
        MODELS["E5-large-v2 (WEIRD)"],
        is_chinese=False,
    )

    # Test ZH model
    run_model_tests(
        "BGE-ZH-v1.5 (Sinic)",
        MODELS["BGE-ZH-v1.5 (Sinic)"],
        is_chinese=True,
    )

    print(f"\n{'=' * 70}")
    print("  CONCLUSIONE")
    print(f"{'=' * 70}")
    print("  Se 'queen' appare nei top-5 per king-man+woman:")
    print("    → L'aritmetica vettoriale funziona → NDA Part B ha senso")
    print("  Se non appare (o appare lontano):")
    print("    → I sentence-transformers non preservano regolarità lineari")
    print("    → NDA Part B va ripensata (es. operazioni nello spazio di")
    print("      similarità, non nello spazio vettoriale diretto)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
