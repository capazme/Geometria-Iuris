"""
build_dataset.py — assembla legal_terms.json da doj_filtered.json + swadesh_control.json.

Pipeline:
  1. Carica doj_filtered.json (9.387 termini filtrati)
  2. Applica EXCLUDE list (hk_specific_terms.md: 19 termini esclusi)
  3. Soglia polisemia: >50 zh_variants → tier=background
  4. Assegna dominio (7 domini) via regole keyword + tiebreaker division
  5. Risolve conflitti multi-dominio con ordine di priorità + division code
  6. Corregge zh_canonical per artifact documentati (Rule 2)
  7. Converte schema: headword→en, divisions→doj_divisions, source→"HK DOJ"
  8. Integra swadesh_control.json (100 control terms, tier=control)
  9. Salva legal_terms.json + build_stats.md + domain_review.json

Ref: trace_dataset_design.md (D2, D3, D4)
     domain_mapping_rules.md v1.1
     hk_specific_terms.md v1.0
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).parent
PROCESSED = DATA_DIR / "processed"

DOJ_FILTERED = PROCESSED / "doj_filtered.json"
SWADESH_FILE = DATA_DIR / "swadesh_control.json"
OUT_LEGAL    = PROCESSED / "legal_terms.json"
OUT_STATS    = PROCESSED / "build_stats.md"
OUT_REVIEW   = PROCESSED / "domain_review.json"

# ---------------------------------------------------------------------------
# EXCLUDE list
# Fonte: hk_specific_terms.md (categoria EXCLUDE di ogni categoria A-F)
# Confronto case-insensitive sul headword normalizzato.
# ---------------------------------------------------------------------------

EXCLUDE_TERMS: set[str] = {
    # Category A — ruoli professionali HK/UK specifici
    "barrister",
    "king's counsel",
    "queen's counsel",
    "bencher",
    "a consiliis",
    "official solicitor",
    "instructing solicitor",
    # Category B — istituzioni HK/UK specifiche
    "judicial committee of the privy council",
    "chancery division",
    "inns of court",
    "lands tribunal",
    # Category C — distinzione common law / equity
    "at law and in equity",
    "clog on the equity of redemption",
    # Category D — Crown / prerogativa regia
    "crown lease",
    "block crown lease",
    "abjuration of the realm",
    # Category E — Latin terms esplicitamente esclusi
    "inter alia",
    "a fortiori",
    "sui generis",
    # Category F — termini procedurali HK-specifici
    "rules of the high court",
}

# ---------------------------------------------------------------------------
# ZH CANONICAL CORRECTIONS (domain_mapping_rules.md Rule 2)
# zh_canonical da doj_filtered.json è il primo occorso, non necessariamente
# il più semanticamente centrale. Correzioni manuali documentate qui.
# ---------------------------------------------------------------------------

ZH_CANONICAL_CORRECTIONS: dict[str, str] = {
    "contract":   "合同",
    "law":        "法律",
    "punishment": "懲罰",
    "judge":      "法官",
    "rights":     "權利",
    "equality":   "平等",   # DOJ artifact: zh_canonical=平等機會 (equal opportunity)
}

# ---------------------------------------------------------------------------
# POLYSEMY THRESHOLD (domain_mapping_rules.md Rule 1)
# ---------------------------------------------------------------------------

POLYSEMY_THRESHOLD = 50

# ---------------------------------------------------------------------------
# DOMAIN RULES
# Ordine = priorità (constitutional > civil > criminal > ... > procedure).
# Per ogni dominio: 'pos' = pattern positivi (substring su headword.lower()),
#                   'neg' = pattern negativi (override exclusion),
#                   'div' = division codes che rafforzano l'assegnazione.
#
# Strategia: match positivo attivato se QUALSIASI pattern è substring del
# headword lowercase. Match negativo disattiva il dominio se QUALSIASI pattern
# è substring. In caso di multi-match, vince il dominio con priorità più alta.
# Division code usato come tiebreaker finale.
# ---------------------------------------------------------------------------

DOMAIN_RULES: dict[str, dict] = {
    "constitutional": {
        "pos": [
            "constitut", "sovereign", "fundamental right", "basic right",
            "human right", "civil liberty", "civil liberties",
            "separation of power", "legislature", "legislat",
            "parliament", "congress", "senate", "referendum",
            "federali", "unitary state", "republic", "monarch",
            "citizenship", "naturaliz", "naturalisation", "suffrage",
            "electoral", "franchise", "amendment", "bill of rights",
            "habeas corpus", "due process", "equal protection",
            "freedom of speech", "freedom of assembly", "freedom of religion",
            "press freedom", "unconstitutional", "constitutionali",
            "veto", "prerogative", "emergency power", "martial law",
            "state of emergency", "basic law", "head of state",
            "checks and balance", "rule of law",
            "freedom", "equality",
        ],
        "neg": [
            "electoral fraud",
            "legislative drafting",
            "legislative procedure",
            "european convention",  # ECHR → international (trattato internazionale HR)
        ],
        "div": [],
    },
    "civil": {
        "pos": [
            "tort", "trespass", "nuisance", "defamation",
            "easement", "mortgage", "pledge",
            "fiduciary", "trustee", "beneficiary",
            "landlord", "tenant", "lease",
            "inheritance", "succession", "intestat", "probate",
            "executor", "testat", "testamentar",
            "divorce", "custody", "adoption", "alimony",
            "unjust enrichment", "restitution", "rescission",
            "frustration of contract", "misrepresentation",
            "bailment", "novation", "subrogation", "indemnity",
            "inter vivos", "sale of goods", "guarantor", "surety",
            "copyright", "patent", "trademark", "intellectual property",
            "contractual", "breach of contract", "offer and acceptance",
            "consideration", "negligence",
        ],
        "neg": [
            "employment contract",
            "family court",
            "criminal negligence",
            "extort",             # extortion/extortionate → criminal, not civil
            "cooperation treaty", # Patent Cooperation Treaty → international
        ],
        "div": ["CD"],
    },
    "criminal": {
        "pos": [
            "criminal", "offence", "offense", "felony", "misdemeanor",
            "homicide", "murder", "manslaughter", "assault", "battery",
            "robbery", "theft", "larceny", "burglary", "forgery",
            "bribery", "corruption", "perjury",
            "extortion", "blackmail", "conspiracy", "inchoate",
            "aiding and abetting", "accessory", "accomplice",
            "mens rea", "actus reus", "guilty mind", "recklessness",
            "sentencing", "imprisonment", "probation", "parole",
            "conviction", "acquittal", "guilty plea", "arraignment",
            "indictment", "prosecution", "accused", "recidivi",
            "penal", "capital punishment", "death penalty",
            "incarceration", "remand",
        ],
        "neg": [
            "criminal procedure",
            "criminal jurisdiction",
            "criminal court",
        ],
        "div": ["PD"],
    },
    "administrative": {
        "pos": [
            "administrative", "ultra vires", "natural justice",
            "audi alteram", "nemo judex", "legitimate expectation",
            "wednesbury", "delegated legislation",
            "statutory instrument", "subordinate legislation", "by-law",
            "statutory body", "public authority", "civil service",
            "compulsory purchase", "eminent domain",
            "ombudsman", "freedom of information",
            "planning permission", "planning law",
            "regulatory authorit", "regulatory framework",
        ],
        "neg": [
            "tax fraud", "tax evasion",
            "immigration crime",
            "public nuisance",
        ],
        "div": ["LPD"],
    },
    "international": {
        "pos": [
            "international law", "treaty", "convention", "covenant",
            "protocol", "bilateral", "multilateral", "ratification",
            "accession", "jus cogens", "erga omnes", "pacta sunt servanda",
            "sovereign immunity", "diplomatic", "consular",
            "ambassador", "envoy", "extraterritorialit",
            "extradition", "asylum", "refugee", "stateless",
            "humanitarian law", "war crime", "crimes against humanity",
            "genocide", "international tribunal", "lex mercatoria",
            "conflict of laws", "private international law",
            "choice of law", "forum non conveniens", "comity",
            "mutual legal assistance", "letters of request",
        ],
        "neg": [
            "clinical",           # clinical dispute protocol → civil/procedure, not international
        ],
        "div": ["ILD"],
    },
    "labor_social": {
        "pos": [
            "employment", "employee", "employer", "labour", "labor",
            "dismissal", "redundanc", "unfair dismissal", "wrongful dismissal",
            "constructive dismissal", "notice period",
            "harassment", "sexual harassment", "equal opportunity", "equal pay",
            "minimum wage", "overtime", "working hours", "rest day",
            "annual leave", "maternity leave", "paternity leave", "sick leave",
            "provident fund", "social security", "social insurance",
            "trade union", "collective bargaining", "strike",
            "industrial action", "occupational safety", "health and safety",
            "social protection", "unemployment", "worker's compensation",
        ],
        "neg": [
            "employment tribunal",
        ],
        "div": [],
    },
    "procedure": {
        "pos": [
            "locus standi", "ius standi",
            "appellate", "cassation", "first instance",
            "admissibilit", "admissible", "hearsay",
            "burden of proof", "standard of proof",
            "cross-examination", "affidavit", "sworn statement",
            "subpoena", "summons", "pleading", "counterclaim",
            "interlocutory", "discovery", "disclosure",
            "limitation period", "res judicata", "issue estoppel",
            "cause of action", "class action", "joinder",
            "stay of proceedings", "mediation", "conciliation",
            "in camera", "ex parte", "service of process",
        ],
        "neg": [
            "habeas corpus",
        ],
        "div": [],
    },
}

# Ordine di priorità (indice = priorità, 0 = massima)
DOMAIN_PRIORITY: list[str] = list(DOMAIN_RULES.keys())

# ---------------------------------------------------------------------------
# Domain assignment
# ---------------------------------------------------------------------------

def assign_domain(headword: str, divisions: list[str]) -> str | None:
    """
    Restituisce il dominio assegnato o None (→ background).

    Algoritmo:
    1. Per ogni dominio, verifica se almeno un pattern positivo è substring
       del headword lowercase.
    2. Se un pattern negativo è presente, rimuove quel dominio dai candidati.
    3. Se un solo dominio → lo assegna.
    4. Se più domini → usa ordine di priorità; in caso di parità, usa division code.
    5. Se zero domini → background.
    """
    hw = headword.lower()
    candidates: list[str] = []

    for domain, rules in DOMAIN_RULES.items():
        # Match positivo
        if not any(p in hw for p in rules["pos"]):
            continue
        # Override negativo
        if any(p in hw for p in rules["neg"]):
            continue
        candidates.append(domain)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Conflitto: usa priorità + division tiebreaker
    # Prima: prova a usare division come tiebreaker
    for domain in DOMAIN_PRIORITY:
        if domain in candidates and any(d in divisions for d in DOMAIN_RULES[domain]["div"]):
            return domain

    # Fallback: prendi il dominio con priorità più alta nell'ordine DOMAIN_PRIORITY
    for domain in DOMAIN_PRIORITY:
        if domain in candidates:
            return domain

    return None  # non dovrebbe mai arrivare qui


def assign_domain_with_meta(headword: str, divisions: list[str]) -> tuple[str | None, str]:
    """
    Versione con metadati: restituisce (domain | None, resolution_type).
    resolution_type: 'unique' | 'conflict_div' | 'conflict_priority' | 'none'
    """
    hw = headword.lower()
    candidates: list[str] = []

    for domain, rules in DOMAIN_RULES.items():
        if not any(p in hw for p in rules["pos"]):
            continue
        if any(p in hw for p in rules["neg"]):
            continue
        candidates.append(domain)

    if not candidates:
        return None, "none"

    if len(candidates) == 1:
        return candidates[0], "unique"

    # Tiebreaker division
    for domain in DOMAIN_PRIORITY:
        if domain in candidates and any(d in divisions for d in DOMAIN_RULES[domain]["div"]):
            return domain, "conflict_div"

    # Fallback priorità
    for domain in DOMAIN_PRIORITY:
        if domain in candidates:
            return domain, "conflict_priority"

    return None, "none"


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_record(raw: dict) -> dict:
    """
    Converte un record doj_filtered.json nello schema target legal_terms.json.
    Applica zh_canonical correction se headword è in ZH_CANONICAL_CORRECTIONS.
    """
    headword = raw["headword"]
    zh_can = raw.get("zh_canonical", "")

    # Correzione artifact zh_canonical (Rule 2)
    if headword.lower() in ZH_CANONICAL_CORRECTIONS:
        zh_can = ZH_CANONICAL_CORRECTIONS[headword.lower()]

    return {
        "en":           headword,
        "zh_canonical": zh_can,
        "domain":       None,       # impostato dopo
        "tier":         None,       # impostato dopo
        "zh_variants":  raw.get("zh_variants", []),
        "zh_sources":   raw.get("zh_sources", []),
        "doj_divisions": raw.get("divisions", []),
        "source":       "HK DOJ",
    }


# ---------------------------------------------------------------------------
# Swadesh loader
# ---------------------------------------------------------------------------

def load_swadesh() -> list[dict]:
    """
    Carica swadesh_control.json e normalizza allo schema target
    (rimuove zh_simplified, zh_traditional che non fanno parte dello schema).
    """
    raw = json.loads(SWADESH_FILE.read_text(encoding="utf-8"))
    result = []
    for entry in raw:
        result.append({
            "en":            entry["en"],
            "zh_canonical":  entry["zh_canonical"],
            "domain":        None,
            "tier":          "control",
            "zh_variants":   entry.get("zh_variants", []),
            "zh_sources":    entry.get("zh_sources", []),
            "doj_divisions": entry.get("doj_divisions", []),
            "source":        "Swadesh-1955",
        })
    return result


# ---------------------------------------------------------------------------
# Stats writer
# ---------------------------------------------------------------------------

def write_stats(
    total_input: int,
    n_excluded: int,
    n_polysemy_bg: int,
    core_by_domain: dict[str, int],
    n_bg_no_match: int,
    n_corrections: int,
    n_control: int,
    conflict_counts: dict[str, int],
) -> None:
    n_core = sum(core_by_domain.values())
    n_bg   = n_polysemy_bg + n_bg_no_match
    total_out = n_core + n_bg + n_control

    lines = ["# Build Statistics — legal_terms.json", ""]
    lines += [
        "## 1. Input / output",
        f"| | Count |",
        f"|--|-------|",
        f"| Input (doj_filtered.json) | {total_input} |",
        f"| Excluded (EXCLUDE list) | {n_excluded} |",
        f"| → Background (polysemy >50 variants) | {n_polysemy_bg} |",
        f"| → Background (no domain match) | {n_bg_no_match} |",
        f"| **Core (domain assigned)** | **{n_core}** |",
        f"| **Background** | **{n_bg}** |",
        f"| **Control (Swadesh)** | **{n_control}** |",
        f"| **Total output** | **{total_out}** |",
        f"| zh_canonical corrections applied | {n_corrections} |",
        "",
    ]

    lines += ["## 2. Core terms per domain", "| Domain | Count |", "|--------|-------|"]
    for domain, count in sorted(core_by_domain.items()):
        lines.append(f"| {domain} | {count} |")
    lines.append("")

    lines += [
        "## 3. Domain assignment resolution",
        "| Resolution type | Count |",
        "|-----------------|-------|",
    ]
    for rtype, count in sorted(conflict_counts.items()):
        lines.append(f"| {rtype} | {count} |")
    lines.append("")

    lines += [
        "## 4. Notes",
        "- Core terms: assigned to one of 7 legal domains via keyword rules",
        "- Background terms: in embedding pool for k-NN neighbourhood but not assigned a domain",
        "- Control terms: Swadesh 100 basic vocabulary, semantic baseline",
        "- `domain_review.json`: all terms where conflict_priority resolution was used",
        "  (multiple domains matched; priority rule applied — may warrant manual review)",
        "- Ref: domain_mapping_rules.md v1.1, hk_specific_terms.md v1.0",
    ]

    OUT_STATS.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== build_dataset.py ===")

    # --- 1. Carica doj_filtered.json ---
    raw_data: list[dict] = json.loads(DOJ_FILTERED.read_text(encoding="utf-8"))
    print(f"[1/6] Loaded {len(raw_data)} terms from doj_filtered.json")
    total_input = len(raw_data)

    # --- 2. Applica EXCLUDE list ---
    excluded = [r for r in raw_data if r["headword"].lower() in EXCLUDE_TERMS]
    pool     = [r for r in raw_data if r["headword"].lower() not in EXCLUDE_TERMS]
    n_excluded = len(excluded)
    print(f"[2/6] Excluded {n_excluded} terms (EXCLUDE list) → {len(pool)} remaining")

    # --- 3. Classifica e assembla ---
    core_terms:       list[dict] = []
    background_terms: list[dict] = []
    review_cases:     list[dict] = []  # conflitti risolti per priorità (da revisionare)

    n_polysemy_bg = 0
    n_bg_no_match = 0
    n_corrections = 0
    core_by_domain: dict[str, int] = Counter()
    conflict_counts: dict[str, int] = Counter()

    for raw in pool:
        record = build_record(raw)

        # Traccia correzioni zh_canonical
        if raw["headword"].lower() in ZH_CANONICAL_CORRECTIONS:
            n_corrections += 1

        # Polisemia → background
        if len(raw.get("zh_variants", [])) > POLYSEMY_THRESHOLD:
            record["tier"]   = "background"
            record["domain"] = None
            background_terms.append(record)
            n_polysemy_bg += 1
            continue

        # Assegnazione dominio
        domain, resolution = assign_domain_with_meta(
            raw["headword"], raw.get("divisions", [])
        )
        conflict_counts[resolution] += 1

        if domain is None:
            record["tier"]   = "background"
            record["domain"] = None
            background_terms.append(record)
            n_bg_no_match += 1
        else:
            record["tier"]   = "core"
            record["domain"] = domain
            core_terms.append(record)
            core_by_domain[domain] += 1

            if resolution == "conflict_priority":
                review_cases.append({
                    "en":         raw["headword"],
                    "domain":     domain,
                    "resolution": resolution,
                    "divisions":  raw.get("divisions", []),
                })

    print(f"[3/6] Classified: {len(core_terms)} core, {len(background_terms)} background")

    # --- 4. Carica Swadesh (control) ---
    swadesh = load_swadesh()
    print(f"[4/6] Loaded {len(swadesh)} Swadesh control terms")

    # --- 5. Assembla output ---
    all_terms = core_terms + background_terms + swadesh
    output = {"terms": all_terms}

    OUT_LEGAL.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[5/6] Saved {OUT_LEGAL} ({OUT_LEGAL.stat().st_size // 1024} KB)")

    # --- 6. Salva stats + review ---
    write_stats(
        total_input   = total_input,
        n_excluded    = n_excluded,
        n_polysemy_bg = n_polysemy_bg,
        core_by_domain= core_by_domain,
        n_bg_no_match = n_bg_no_match,
        n_corrections = n_corrections,
        n_control     = len(swadesh),
        conflict_counts = conflict_counts,
    )

    OUT_REVIEW.write_text(
        json.dumps(review_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[6/6] Saved stats → {OUT_STATS}")
    print(f"      Saved review cases → {OUT_REVIEW} ({len(review_cases)} terms)")

    print("\n--- SUMMARY ---")
    print(f"  Core:       {len(core_terms)}")
    for d in DOMAIN_PRIORITY:
        print(f"    {d:<20} {core_by_domain.get(d, 0)}")
    print(f"  Background: {len(background_terms)}")
    print(f"    polysemy >50:       {n_polysemy_bg}")
    print(f"    no domain match:    {n_bg_no_match}")
    print(f"  Control:    {len(swadesh)}")
    print(f"  TOTAL:      {len(all_terms)}")
    print(f"\n  zh_canonical corrections: {n_corrections}")
    print(f"  conflict_priority cases (review): {len(review_cases)}")


if __name__ == "__main__":
    main()
