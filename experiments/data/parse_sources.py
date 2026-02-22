"""
Parser per il glossario HK DOJ English-Chinese.

Parsa glossary_ec.xml (file master) + file per-division per taggare ogni entry
con la division di appartenenza. Gestisce BOM, ampersand non escaped, e
duplicati headword con varianti ZH multiple.

Output:
  data/processed/doj_raw.json      - tutte le entries raggruppate per headword
  data/processed/doj_filtered.json - subset dopo filtri strutturali
  data/processed/corpus_stats.md   - statistiche descrittive
"""

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent / "raw" / "hk_doj"
OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = RAW_DIR / "glossary_ec.xml"

DIVISION_FILES = {
    "CD":  RAW_DIR / "glossary_ec_CD.xml",
    "ILD": RAW_DIR / "glossary_ec_ILD.xml",
    "LDD": RAW_DIR / "glossary_ec_LDD.xml",
    "LPD": RAW_DIR / "glossary_ec_LPD.xml",
    "LRC": RAW_DIR / "glossary_ec_LRC.xml",
    "PD":  RAW_DIR / "glossary_ec_PD.xml",
}

# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

_ENTITY_FIXES = [
    # Normalizza ampersand non escaped prima del parse
    (re.compile(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)'), '&amp;'),
]

_TAG_RE = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)


def _sanitize(content: str) -> str:
    for pattern, replacement in _ENTITY_FIXES:
        content = pattern.sub(replacement, content)
    return content


def _extract_field(block: str, tag: str) -> str:
    m = re.search(rf'<{tag}>(.*?)</{tag}>', block, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_xml_file(path: Path) -> list[dict]:
    """Parsa un file XML DOJ, restituisce lista di dict con headword/english_def/chinese_def/source."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    # Gestione BOM e \r\n
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = _sanitize(raw)

    entries = []
    for block in re.split(r'<glossary>', raw)[1:]:
        end = block.find('</glossary>')
        if end == -1:
            continue
        block = block[:end]

        headword   = _extract_field(block, "headword")
        english_def = _extract_field(block, "english_def")
        chinese_def = _extract_field(block, "chinese_def")
        source      = _extract_field(block, "source")

        if not headword:
            continue

        entries.append({
            "headword":    headword,
            "english_def": english_def,
            "chinese_def": chinese_def,
            "source":      source,
        })

    return entries


# ---------------------------------------------------------------------------
# Division index: headword → set of divisions
#                 headword → {zh_variant → set of divisions}
# ---------------------------------------------------------------------------

def build_division_index() -> tuple[dict[str, set], dict[str, dict[str, set]]]:
    """
    Restituisce:
      div_index:    headword → set[division]
      zh_div_index: headword → {chinese_def → set[division]}
    """
    div_index: dict[str, set] = defaultdict(set)
    zh_div_index: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for div, path in DIVISION_FILES.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping division {div}", file=sys.stderr)
            continue
        entries = parse_xml_file(path)
        for e in entries:
            hw = e["headword"]
            zh = e["chinese_def"]
            div_index[hw].add(div)
            if zh:
                zh_div_index[hw][zh].add(div)

    return div_index, zh_div_index


# ---------------------------------------------------------------------------
# Grouping: headword → aggregated record with zh_variants list
# ---------------------------------------------------------------------------

def group_entries(entries: list[dict], division_index: tuple[dict, dict]) -> list[dict]:
    """Raggruppa per headword, raccogliendo varianti ZH con source e divisions corrispondenti."""
    div_index, zh_div_index = division_index
    by_headword: dict[str, dict] = {}

    for e in entries:
        hw = e["headword"]
        zh = e["chinese_def"]
        en = e["english_def"]
        src = e["source"]

        if hw not in by_headword:
            by_headword[hw] = {
                "headword":    hw,
                "english_def": en,
                "source":      src,
                "zh_variants": [],
                "zh_sources":  [],
                "divisions":   sorted(div_index.get(hw, [])),
            }

        record = by_headword[hw]
        # Aggiunge variante ZH solo se non duplicata, con source corrispondente
        if zh and zh not in record["zh_variants"]:
            record["zh_variants"].append(zh)
            record["zh_sources"].append(src)

    return list(by_headword.values())


# ---------------------------------------------------------------------------
# ZH canonical selection
# ---------------------------------------------------------------------------

# Gerarchia division per zh_canonical (ordine = priorità decrescente)
_DIV_PRIORITY = ["LRC", "PD", "ILD", "LPD", "LDD"]

_NAMED_GLOSSARY_RE = re.compile(r'glossar', re.IGNORECASE)


def _is_named_glossary(source: str) -> bool:
    return bool(_NAMED_GLOSSARY_RE.search(source))


def select_zh_canonical(record: dict, zh_div_map: dict[str, set]) -> str:
    """
    Seleziona la variante ZH canonica secondo la gerarchia:
    1. Prima variante da pubblicazione nominata (source contiene "Glossary"/"Glossaries")
    2. Prima variante presente in LRC
    3. Prima variante presente in PD
    4. Prima variante presente in ILD
    5. Prima variante presente in LPD
    6. Prima variante presente in LDD
    7. zh_variants[0] come fallback

    zh_div_map: chinese_def → set[division] per questo headword
    """
    variants = record["zh_variants"]
    sources  = record["zh_sources"]

    if not variants:
        return ""

    # Regola 1: named glossary source
    for zh, src in zip(variants, sources):
        if _is_named_glossary(src):
            return zh

    # Regole 2-6: per division
    for div in _DIV_PRIORITY:
        for zh in variants:
            if div in zh_div_map.get(zh, set()):
                return zh

    # Fallback
    return variants[0]


# ---------------------------------------------------------------------------
# CJK check
# ---------------------------------------------------------------------------

_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), # CJK Compatibility Supplement
]


def has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

_LEGISLATIVE_PREFIXES = re.compile(r'^(Cap\.|s\.|Art\.|r\.)\s', re.IGNORECASE)
_DIGIT_IN_HW = re.compile(r'\d')
_ALLCAPS_ABBR = re.compile(r'^[A-Z]{1,4}$')


def apply_filters(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """
    Applica filtri sequenziali. Restituisce (survived, stats_per_filter).
    stats_per_filter: nome_filtro → quanti eliminated a quel passo.
    """
    stats: dict[str, int] = {}
    pool = records[:]

    def _filter(pool, name, predicate):
        before = len(pool)
        kept = [r for r in pool if predicate(r)]
        stats[name] = before - len(kept)
        return kept

    # ---- Filtri POSITIVI ----
    pool = _filter(pool, "pos_word_count",
        lambda r: len(r["headword"].split()) <= 5
    )
    pool = _filter(pool, "pos_has_cjk",
        lambda r: any(has_cjk(zh) for zh in r["zh_variants"]) if r["zh_variants"]
                  else has_cjk(r.get("chinese_def", ""))
    )
    pool = _filter(pool, "pos_not_trivial",
        lambda r: not (
            r["headword"].isdigit() or
            _ALLCAPS_ABBR.match(r["headword"]) or
            len(r["headword"]) == 1
        )
    )

    # ---- Filtri NEGATIVI ----
    pool = _filter(pool, "neg_legislative_ref",
        lambda r: not _LEGISLATIVE_PREFIXES.match(r["headword"])
    )
    pool = _filter(pool, "neg_parentheses",
        lambda r: "(" not in r["headword"] and ")" not in r["headword"]
    )
    pool = _filter(pool, "neg_long_def",
        lambda r: len(r["english_def"]) <= 300
    )
    pool = _filter(pool, "neg_digit_in_hw",
        lambda r: not _DIGIT_IN_HW.search(r["headword"])
    )

    return pool, stats


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def word_len_dist(records: list[dict]) -> dict[int, int]:
    c: dict[int, int] = Counter(len(r["headword"].split()) for r in records)
    return dict(sorted(c.items()))


def zh_variant_dist(records: list[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in records:
        n = len(r["zh_variants"])
        key = str(n) if n <= 4 else "5+"
        c[key] += 1
    return dict(sorted(c.items()))


def division_dist(records: list[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in records:
        for div in r["divisions"]:
            c[div] += 1
    # Aggiungi "none" per entries senza division tag
    no_div = sum(1 for r in records if not r["divisions"])
    if no_div:
        c["(none)"] = no_div
    return dict(sorted(c.items()))


def source_dist(records: list[dict]) -> dict[str, int]:
    c: Counter = Counter(r["source"] for r in records)
    return dict(c.most_common())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== HK DOJ Glossary Parser ===")

    # 1. Parsa file master
    print(f"\n[1/5] Parsing {MASTER_FILE.name} ...")
    raw_entries = parse_xml_file(MASTER_FILE)
    print(f"      Raw entries parsed: {len(raw_entries)}")

    # 2. Build division index dai file per-division
    print("\n[2/5] Building division index ...")
    div_index, zh_div_index = build_division_index()
    print(f"      Headwords with division tag: {len(div_index)}")

    # 3. Raggruppa per headword
    print("\n[3/5] Grouping by headword ...")
    grouped = group_entries(raw_entries, (div_index, zh_div_index))
    # Aggiunge zh_canonical a ogni record
    for record in grouped:
        hw = record["headword"]
        zh_div_map = zh_div_index.get(hw, {})
        record["zh_canonical"] = select_zh_canonical(record, zh_div_map)
    print(f"      Unique headwords: {len(grouped)}")

    # 4. Salva doj_raw.json
    raw_out = OUT_DIR / "doj_raw.json"
    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
    print(f"\n[4/5] Saved {raw_out} ({raw_out.stat().st_size // 1024} KB)")

    # 5. Applica filtri
    print("\n[5/5] Applying filters ...")
    filtered, filter_stats = apply_filters(grouped)
    print(f"      Survived: {len(filtered)} / {len(grouped)}")

    filtered_out = OUT_DIR / "doj_filtered.json"
    with open(filtered_out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"      Saved {filtered_out} ({filtered_out.stat().st_size // 1024} KB)")

    # 6. Statistiche
    _write_stats(raw_entries, grouped, filtered, filter_stats)

    print("\nDone.")


def _write_stats(raw_entries, grouped, filtered, filter_stats):
    # Calcola stats raw (entries non ancora filtrate)
    raw_word_len = word_len_dist(grouped)
    filt_word_len = word_len_dist(filtered)
    raw_div = division_dist(grouped)
    raw_src = source_dist(grouped)
    raw_zh_var = zh_variant_dist(grouped)
    filt_div = division_dist(filtered)

    lines = []
    lines.append("# Corpus Statistics — HK DOJ Glossary\n")

    lines.append("## 1. Totals")
    lines.append(f"- Raw entries parsed from master XML: {len(raw_entries)}")
    lines.append(f"- Unique headwords (grouped): {len(grouped)}")
    lines.append(f"- Survived after all filters: {len(filtered)}")
    lines.append(f"- Retention rate: {len(filtered)/len(grouped)*100:.1f}%\n")

    lines.append("## 2. Word-length distribution of headwords")
    lines.append("| Words | Raw count | Filtered count |")
    lines.append("|-------|-----------|----------------|")
    all_lens = sorted(set(raw_word_len) | set(filt_word_len))
    for n in all_lens:
        lines.append(f"| {n} | {raw_word_len.get(n, 0)} | {filt_word_len.get(n, 0)} |")
    lines.append("")

    lines.append("## 3. Entries per division (raw grouped)")
    lines.append("| Division | Raw headwords | Filtered headwords |")
    lines.append("|----------|---------------|--------------------|")
    all_divs = sorted(set(raw_div) | set(filt_div))
    for d in all_divs:
        lines.append(f"| {d} | {raw_div.get(d, 0)} | {filt_div.get(d, 0)} |")
    lines.append("")

    lines.append("## 4. ZH variant count per headword (raw)")
    lines.append("| Variants | Headwords |")
    lines.append("|----------|-----------|")
    for k, v in sorted(raw_zh_var.items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 5. Filter attrition (sequential)")
    lines.append("| Filter | Eliminated |")
    lines.append("|--------|-----------|")
    remaining = len(grouped)
    for fname, eliminated in filter_stats.items():
        remaining -= eliminated
        lines.append(f"| {fname} | {eliminated} (→ {remaining} remaining) |")
    lines.append("")

    lines.append("## 6. Top 10 source publications (raw grouped)")
    lines.append("| Source | Headwords |")
    lines.append("|--------|-----------|")
    for src, cnt in list(raw_src.items())[:10]:
        lines.append(f"| {src[:80]} | {cnt} |")
    lines.append("")

    stats_out = OUT_DIR / "corpus_stats.md"
    stats_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"      Saved {stats_out}")

    # Stampa summary a terminale
    print(f"\n--- SUMMARY ---")
    print(f"  Raw entries: {len(raw_entries)}")
    print(f"  Unique HW:   {len(grouped)}")
    print(f"  Filtered:    {len(filtered)}")
    for div in ["CD", "ILD", "LDD", "LPD", "LRC", "PD"]:
        print(f"  {div}: raw={raw_div.get(div,0)}, filtered={filt_div.get(div,0)}")


if __name__ == "__main__":
    main()
