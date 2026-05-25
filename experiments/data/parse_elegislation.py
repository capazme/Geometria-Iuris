"""
Parse HK e-Legislation XML into section-level parallel EN/ZH corpus.

Reads the 8 ZIP files from DATA.GOV.HK (4 EN + 4 ZH-Hant), extracts text
from <section> elements, aligns EN↔ZH pairs by structural temporalId, and
outputs a JSONL corpus for downstream contextualised embedding retrieval.

Output
------
data/processed/elegislation/
    sections.jsonl          one JSON object per aligned section
    definitions.jsonl       bilingual term definitions from <def> elements
    stats.json              corpus statistics

Usage
-----
    python data/parse_elegislation.py [--raw-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "hk_elegislation"
OUT_DIR = ROOT / "data" / "processed" / "elegislation"

NS = {"hk": "http://www.xml.gov.hk/schemas/hklm/1.0"}

# ZIP file pairs: (EN zip, ZH zip) — matched by chapter range
ZIP_PAIRS = [
    ("hkel_c_leg_cap_1_cap_300_en.zip", "hkel_c_leg_cap_1_cap_300_zh-Hant.zip"),
    ("hkel_c_leg_cap_301_cap_600_en.zip", "hkel_c_leg_cap_301_cap_600_zh-Hant.zip"),
    ("hkel_c_leg_cap_601_cap_end_en.zip", "hkel_c_leg_cap_601_cap_end_zh-Hant.zip"),
    ("hkel_c_instruments_en.zip", "hkel_c_instruments_zh-Hant.zip"),
]

# Regex to extract cap number from filename
CAP_RE = re.compile(r"cap_(\d+[A-Z]*)")


def text_content(elem: ET.Element) -> str:
    """Recursively extract all text from an element, stripping XML tags."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(text_content(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def clean_text(raw: str) -> str:
    """Normalize whitespace and strip source notes artifacts."""
    t = re.sub(r"\s+", " ", raw).strip()
    # Remove trailing source note patterns like "(Added 89 of 1993 s. 2)"
    t = re.sub(r"\s*\((?:Added|Amended|Replaced|Repealed|由|增補|修訂|代替|廢除).*$", "", t)
    return t


def parse_xml_bytes(data: bytes) -> ET.Element:
    """Parse XML bytes into an ElementTree root."""
    return ET.fromstring(data)


def extract_sections(root: ET.Element) -> dict[str, dict]:
    """
    Extract section-level text keyed by temporalId.

    Returns dict[temporalId] = {"heading": str, "text": str, "defs": [...]}
    """
    sections: dict[str, dict] = {}

    for section in root.iter("{http://www.xml.gov.hk/schemas/hklm/1.0}section"):
        tid = section.get("temporalId", "")
        if not tid:
            continue

        heading_el = section.find("hk:heading", NS)
        heading = clean_text(text_content(heading_el)) if heading_el is not None else ""

        # Collect all content text within this section (including subsections)
        content_parts: list[str] = []
        for content_el in section.iter("{http://www.xml.gov.hk/schemas/hklm/1.0}content"):
            t = clean_text(text_content(content_el))
            if t:
                content_parts.append(t)

        # Collect definitions
        defs: list[dict] = []
        for def_el in section.iter("{http://www.xml.gov.hk/schemas/hklm/1.0}def"):
            def_name = def_el.get("name", "")
            terms: list[dict] = []
            for term_el in def_el.iter("{http://www.xml.gov.hk/schemas/hklm/1.0}term"):
                lang = term_el.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                term_text = clean_text(text_content(term_el))
                if term_text:
                    terms.append({"lang": lang, "text": term_text})
            def_text = clean_text(text_content(def_el))
            if terms:
                defs.append({"name": def_name, "terms": terms, "text": def_text})

        sections[tid] = {
            "heading": heading,
            "text": " ".join(content_parts),
            "defs": defs,
        }

    return sections


def cap_id_from_filename(fname: str) -> str:
    """Extract cap identifier from filename like 'cap_102A_en_c\\cap_102A_..._en_c.xml'."""
    m = CAP_RE.search(fname)
    return m.group(1) if m else fname


def process_zip_pair(
    en_zip_path: Path,
    zh_zip_path: Path,
    sections_out: list[dict],
    definitions_out: list[dict],
) -> dict[str, int]:
    """Process one pair of EN/ZH ZIP files, appending to output lists."""
    stats = {"en_files": 0, "zh_files": 0, "aligned_sections": 0, "definitions": 0}

    # Index EN files by cap_id
    en_data: dict[str, dict[str, dict]] = {}  # cap_id → {tid → section}
    with zipfile.ZipFile(en_zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            stats["en_files"] += 1
            cap_id = cap_id_from_filename(name)
            try:
                root = parse_xml_bytes(zf.read(name))
                en_data[cap_id] = extract_sections(root)
            except ET.ParseError as e:
                logger.warning("  XML parse error in %s: %s", name, e)

    # Process ZH files and align with EN
    with zipfile.ZipFile(zh_zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            stats["zh_files"] += 1
            cap_id = cap_id_from_filename(name)
            try:
                root = parse_xml_bytes(zf.read(name))
                zh_sections = extract_sections(root)
            except ET.ParseError as e:
                logger.warning("  XML parse error in %s: %s", name, e)
                continue

            en_sections = en_data.get(cap_id, {})

            # Align by temporalId
            all_tids = set(en_sections) | set(zh_sections)
            for tid in sorted(all_tids):
                en_sec = en_sections.get(tid, {})
                zh_sec = zh_sections.get(tid, {})

                en_text = en_sec.get("text", "")
                zh_text = zh_sec.get("text", "")

                if not en_text and not zh_text:
                    continue

                sections_out.append({
                    "cap": cap_id,
                    "section_id": tid,
                    "en_heading": en_sec.get("heading", ""),
                    "zh_heading": zh_sec.get("heading", ""),
                    "en_text": en_text,
                    "zh_text": zh_text,
                })
                stats["aligned_sections"] += 1

                # Collect definitions (prefer EN defs which have ZH inline)
                for d in en_sec.get("defs", []):
                    en_terms = [t["text"] for t in d["terms"] if not t["lang"] or t["lang"] == "en"]
                    zh_terms = [t["text"] for t in d["terms"] if "zh" in t.get("lang", "")]
                    if en_terms and zh_terms:
                        definitions_out.append({
                            "cap": cap_id,
                            "section_id": tid,
                            "name": d["name"],
                            "en_terms": en_terms,
                            "zh_terms": zh_terms,
                            "en_definition": d["text"],
                        })
                        stats["definitions"] += 1

    return stats


def main(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: list[dict] = []
    definitions: list[dict] = []
    total_stats: dict[str, int] = defaultdict(int)

    for en_fname, zh_fname in ZIP_PAIRS:
        en_path = raw_dir / en_fname
        zh_path = raw_dir / zh_fname

        if not en_path.exists() or not zh_path.exists():
            logger.error("Missing: %s or %s", en_path, zh_path)
            return 1

        logger.info("Processing %s + %s ...", en_fname, zh_fname)
        stats = process_zip_pair(en_path, zh_path, sections, definitions)
        for k, v in stats.items():
            total_stats[k] += v
        logger.info("  EN files: %d, ZH files: %d, aligned sections: %d, definitions: %d",
                     stats["en_files"], stats["zh_files"],
                     stats["aligned_sections"], stats["definitions"])

    # Write outputs
    sections_path = out_dir / "sections.jsonl"
    with sections_path.open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("\nSections: %s (%d records)", sections_path, len(sections))

    definitions_path = out_dir / "definitions.jsonl"
    with definitions_path.open("w", encoding="utf-8") as f:
        for d in definitions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Definitions: %s (%d records)", definitions_path, len(definitions))

    # Compute additional stats
    en_with_text = sum(1 for s in sections if s["en_text"])
    zh_with_text = sum(1 for s in sections if s["zh_text"])
    both_text = sum(1 for s in sections if s["en_text"] and s["zh_text"])
    en_chars = sum(len(s["en_text"]) for s in sections)
    zh_chars = sum(len(s["zh_text"]) for s in sections)

    stats_out = {
        "total_sections": len(sections),
        "en_with_text": en_with_text,
        "zh_with_text": zh_with_text,
        "both_languages": both_text,
        "total_definitions": len(definitions),
        "en_total_chars": en_chars,
        "zh_total_chars": zh_chars,
        **dict(total_stats),
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats_out, indent=2, ensure_ascii=False))
    logger.info("Stats: %s", stats_path)

    logger.info("\n=== Summary ===")
    logger.info("  Total sections: %d", len(sections))
    logger.info("  Both EN+ZH: %d (%.1f%%)", both_text, 100 * both_text / max(len(sections), 1))
    logger.info("  Total definitions: %d", len(definitions))
    logger.info("  EN corpus: %.1f M chars", en_chars / 1e6)
    logger.info("  ZH corpus: %.1f M chars", zh_chars / 1e6)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", help="Directory with ZIP files (default: data/raw/hk_elegislation/)")
    parser.add_argument("--out-dir", help="Output directory (default: data/processed/elegislation/)")
    sys.exit(main(parser.parse_args()))
