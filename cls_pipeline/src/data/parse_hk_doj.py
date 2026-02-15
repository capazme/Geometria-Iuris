"""
parse_hk_doj.py — Parser per il glossario bilingue HK DOJ (XML → JSON lookup).

Legge il file XML scaricato dal Department of Justice di Hong Kong,
estrae le coppie headword→chinese_def, converte il cinese da tradizionale
a semplificato (per coerenza con il modello BGE-ZH), e produce un file
JSON di lookup pronto per build_dataset.py.

Fonti:
  - HK DOJ Bilingual Legal Glossary: https://www.glossary.doj.gov.hk/
  - OpenCC: conversione Tradizionale→Semplificato (profilo t2s)

Usage:
    python -m src.data.parse_hk_doj
"""
# ─── Strategia di parsing ────────────────────────────────────────────
# Il glossario DOJ contiene 73.184 record XML. Molti termini multi-parola
# (es. "due process", "freedom of speech") sono sotto-voci di headword
# mono-parola (es. "due", "freedom"), con l'espressione completa nel
# campo english_def. Per catturare tutto:
#   1. Indice primario: headword → traduzioni
#   2. Indice secondario: english_def (pulito) → traduzione
# Il lookup finale unisce entrambi gli indici. Per ogni chiave:
#   - zh_options ordinate per lunghezza (termine breve prima)
#   - sources tracciate
#
# La conversione Trad→Simplified usa OpenCC con profilo "t2s".
# ─────────────────────────────────────────────────────────────────────

import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from opencc import OpenCC

logger = logging.getLogger(__name__)

# Profilo OpenCC: Traditional → Simplified
_cc = OpenCC("t2s")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def _clean_english_def(english_def: str) -> str:
    """
    Extract the clean term from english_def, stripping annotations.

    Examples:
        "due process" → "due process"
        "due process of law" → "due process of law"
        "due process (of law)" → "due process (of law)"
        "gift in contemplation of death [also donatio mortis causa]" → "gift in contemplation of death"
        "a consiliis [\"of counsel\"]" → "a consiliis"
    """
    # Rimuovi contenuto tra [...] (annotazioni, riferimenti)
    cleaned = re.sub(r"\s*\[.*?\]", "", english_def)
    # Rimuovi contenuto tra (...) solo se alla fine e contiene parole chiave
    # di annotazione (arch., also, HK, etc.) — ma NON "(of law)" che è parte del termine
    cleaned = re.sub(r"\s*\((?:arch\.|also |HK)[^)]*\)\s*$", "", cleaned)
    return cleaned.strip()


def parse_xml(xml_path: Path) -> dict[str, dict]:
    """
    Parse HK DOJ XML glossary into a lookup dictionary.

    Builds two indices merged together:
    1. By headword (e.g. "sovereignty" → 主权)
    2. By english_def (e.g. "due process" → 正当的法律程序)

    Parameters
    ----------
    xml_path : Path
        Path to the DOJ XML file.

    Returns
    -------
    dict[str, dict]
        Mapping from lowercase term to:
        {
            "zh_options": [str, ...],   # simplified Chinese translations
            "sources": [str, ...],       # DOJ source glossaries
            "headword": str,             # original headword
        }
    """
    # Il file XML ha un \r\n iniziale e contiene '&' non-escaped nelle
    # source (es. "Kong & Ors"). Si corregge prima del parsing.
    raw = xml_path.read_text(encoding="utf-8").lstrip()
    raw = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", raw)
    root = ET.fromstring(raw)

    lookup: dict[str, dict] = {}
    n_records = 0

    def _add_entry(key: str, zh_simplified: str, source: str, headword_raw: str):
        """Add a single entry to the lookup, deduplicating."""
        if key not in lookup:
            lookup[key] = {
                "zh_options": [],
                "sources": [],
                "headword": headword_raw,
            }
        if zh_simplified not in lookup[key]["zh_options"]:
            lookup[key]["zh_options"].append(zh_simplified)
        if source and source not in lookup[key]["sources"]:
            lookup[key]["sources"].append(source)

    for glossary in root.iter("glossary"):
        headword_el = glossary.find("headword")
        chinese_def_el = glossary.find("chinese_def")
        english_def_el = glossary.find("english_def")
        source_el = glossary.find("source")

        if headword_el is None or chinese_def_el is None:
            continue

        headword_raw = (headword_el.text or "").strip()
        chinese_raw = (chinese_def_el.text or "").strip()
        english_def_raw = (english_def_el.text or "").strip() if english_def_el is not None else ""
        source = (source_el.text or "").strip() if source_el is not None else ""

        if not headword_raw or not chinese_raw:
            continue

        n_records += 1
        zh_simplified = _cc.convert(chinese_raw)

        # 1) Indice per headword
        hw_key = headword_raw.lower().strip('"').strip("'")
        _add_entry(hw_key, zh_simplified, source, headword_raw)

        # 2) Indice per english_def (se diverso da headword)
        if english_def_raw:
            ed_clean = _clean_english_def(english_def_raw)
            ed_key = ed_clean.lower().strip('"').strip("'")
            if ed_key and ed_key != hw_key:
                _add_entry(ed_key, zh_simplified, source, headword_raw)

    # Ordina zh_options per lunghezza (termine breve prima)
    for entry in lookup.values():
        entry["zh_options"].sort(key=len)

    n_headwords = sum(1 for _ in root.iter("glossary"))
    logger.info("Parsed %d XML records → %d unique keys (headword + english_def)", n_records, len(lookup))
    return lookup


def save_lookup(lookup: dict[str, dict], output_path: Path) -> None:
    """Save lookup dictionary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)

    logger.info("Lookup saved to: %s (%d entries)", output_path, len(lookup))


def main():
    """Parse HK DOJ XML and generate lookup JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = get_project_root()
    xml_path = root / "data" / "raw" / "hk_doj" / "ec_open_data_2026_02_05.xml"
    output_path = root / "data" / "processed" / "hk_doj_lookup.json"

    if not xml_path.exists():
        logger.error("XML file not found: %s", xml_path)
        return 1

    logger.info("Parsing HK DOJ glossary: %s", xml_path)
    lookup = parse_xml(xml_path)
    save_lookup(lookup, output_path)

    # Stats
    n_single = sum(1 for v in lookup.values() if len(v["zh_options"]) == 1)
    n_multi = sum(1 for v in lookup.values() if len(v["zh_options"]) > 1)
    logger.info("  Single translation: %d", n_single)
    logger.info("  Multiple translations: %d", n_multi)

    return 0


if __name__ == "__main__":
    sys.exit(main())
