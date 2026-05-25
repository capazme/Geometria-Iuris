"""
Apply manual curation decisions to the post-BLP long list.

Reads `experiments/data/processed/postBLP_curation_longlist.csv` and writes
back the same path with the `curation_decision` column updated according to
the decisions encoded below. Methodology: see
`experiments/data/trace_postBLP_curation.md` D1-D6.

Per-domain KEEP set: terms I actively want in the post-BLP core. Anything
in `RECOMMEND_KEEP` / `REVIEW` / `RECOMMEND_DROP` not in this set defaults
to DROP. AUTO_KEEP defaults to KEEP unless flagged in `auto_keep_drop_override`
because zh is wrong-sense or fragment.

Run:
    python3 experiments/data/apply_postBLP_curation.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

CSV = Path("/Users/gpuzio/Desktop/CODE/THESIS/experiments/data/processed/postBLP_curation_longlist.csv")

# -----------------------------------------------------------------------------
# AUTO_KEEP overrides — Firthian core terms whose zh_clean is wrong-sense or
# fragmentary; under post-BLP D2.4 we drop rather than override.
# -----------------------------------------------------------------------------
AUTO_KEEP_DROP_OVERRIDE: dict[str, set[str]] = {
    # Reviewer 2 (US common-law native, 2026-05-10) flagged that demoting these 5
    # for wrong-sense ZH was a substantive loss of common-law canon (patent/bill/
    # international law/assault/occupational) treatable by manual ZH override
    # rather than dropping. Per recommendation, these are now rescued via
    # zh_overrides_postBLP.json with manually corrected ZH lemmas
    # (patent→專利, bill→法案, international law→國際法, assault→襲擊,
    # occupational→職業). The AUTO_KEEP_DROP_OVERRIDE set is therefore empty.
}

# -----------------------------------------------------------------------------
# Per-domain KEEP set (RECOMMEND_KEEP + REVIEW promotions).
# Strings are matched against the `en` column verbatim.
# Target: ~60 per domain (AUTO_KEEP + KEEPs below).
# -----------------------------------------------------------------------------
KEEP: dict[str, set[str]] = {
    "administrative": {
        # RECOMMEND_KEEP
        "code", "notification",
        # REVIEW promotions
        "register", "certificate", "practice", "application", "requirement",
        "accounting", "procedure", "address", "accreditation", "maintenance",
        "certification", "comply", "modification", "exemption", "entry",
        "levy", "approval", "authorization", "taxation", "discipline",
        "surcharge", "expenditure", "non-compliance", "recall", "registry",
        "implementation", "expenses", "licensing", "licensed", "certification",
        "Registrar", "controller", "permitted", "in force", "set aside",
        "duly", "authorized", "provisional", "revocation", "complaint",
        "governing",   # 規管文書 = governing instrument, admin canon
        "inspector",   # 視察主任 admin/regulatory canon
        "administer",  # 主持 verbal canon for administer
    },
    "civil": {
        # RECOMMEND_KEEP
        "provision", "property", "asset", "transfer", "transaction", "owner",
        "loan", "ownership", "estate", "premises", "lease", "lien", "gift",
        # REVIEW promotions
        "trade mark", "obligation", "grant", "entity", "settlement",
        "insurance", "prejudice", "stock", "land", "infringement", "deposit",
        "carriage", "undertaking", "family", "principle", "share capital",
        "futures", "guarantee", "holder", "client", "derivative", "contractor",
        "partnership", "proprietor", "design", "priority", "lot", "unincorporated",
        "relative", "bearer", "distribution", "proceeds", "insolvency",
        "vesting", "purchaser", "charitable", "domicile", "proprietary",
        "subscriber", "disposition", "good faith", "rental", "household",
        "matrimonial", "covenant", "dealer", "distributor", "offeror",
        "claimant", "customer", "dividend", "licensee", "stay", "will",
        # 'person' removed 2026-05-10 per Reviewer 2 (US common-law native):
        # generic statutory drafting variable, not legal-technical in any operational sense.
        # Inclusion would inflate within-civil ρ̄ artefactually.
    },
    "constitutional": {
        # RECOMMEND_KEEP
        "state", "Board", "Committee", "constituency", "voting", "board",
        "polling", "political", "majority", "poll", "Council",
        "representative", "elect",
        # REVIEW promotions
        "act", "system", "panel", "title", "representation", "director",
        "equal", "judicial", "territorial", "nomination", "candidate",
        "Director", "absolute", "Commissioner", "education", "society",
        "territory", "waters", "president", "quorum", "Presiding Officer",
        "third party", "assembly", "incorporation", "religion",
        "Heung Yee Kuk",
        "system",   # 對訟式刑事司法體系 narrowed but constitutional-system canon
        "equal",    # 平等 — constitutional equality
        "meeting",  # 會議 — legislative meeting
        "absolute", # 全權 — constitutional canon (absolute discretion)
        "delegate", # 獲轉授權力的人 — delegate of power, constitutional
        "organization", # 非屬法團的組織 — non-corporate org, constitutional
        # 'law' removed 2026-05-10 per Reviewer 2 (US common-law native):
        # generic mass noun / meta-term, not a constitutional concept.
        # Symmetric with D2.1 blacklist of 'money' / 'description' / 'plan'.
    },
    "criminal": {
        # RECOMMEND_KEEP
        "offence", "charge", "criminal", "unlawful", "false", "prosecution",
        "suspend", "wilfully", "imprisonment", "detention", "crime",
        "indecent", "offender", "murder", "obscene",
        # REVIEW promotions
        "sentence", "conduct", "judgment", "justice", "warrant", "debt",
        "disciplinary", "liable", "force", "Tribunal", "breach", "trial",
        "failure", "removal", "harm", "minor", "tribunal", "guilty",
        "armed", "forfeiture", "convict", "misconduct", "threat",
        "conceal", "contravention", "infringing", "restraint", "dangerous",
        "intimidate", "trafficking", "detain", "unlawfully", "attempt",
        "pardon", "drug", "police", "prohibited", "withdrawal", "defraud",
        "execution", "hostage", "confiscation", "counterfeit", "culpable",
        "encumbrance", "intimidation", "juvenile", "punish", "violence",
        "bankruptcy", "seizure", "bail", "death", "expulsion", "intention",
        "plea", "revoke", "judge",  # judge from REC_DROP → KEEP
    },
    "international": {
        # RECOMMEND_KEEP
        "consular", "defence", "United Nations", "multilateral", "independent",
        "diplomatic", "overseas", "general",
        # REVIEW promotions
        "convention", "agreement", "protection", "arrangement", "trade",
        "personal", "declaration", "standard", "national", "subject",
        "Mainland", "domestic", "mutual", "protocol", "particular",
        "association", "maritime", "marine", "Chinese",
        "non-Hong Kong", "aircraft", "intermediary", "boundary",
        "conference", "conflict", "particulars", "presence", "related",
        "material particular", "People's Republic of China", "tanker",
        "World Trade Organisation", "outbound", "sea-going", "arbitrator",
        "vicinity", "concurrent", "real", "regional",
    },
    "labor_social": {
        # RECOMMEND_KEEP
        "offer", "work", "construction", "delay", "escalator", "hour", "unit",
        "full-time",
        # REVIEW promotions
        "account", "financial", "business", "payment", "contract", "company",
        "fund", "discharge", "fee", "investment", "agent", "benefit",
        "amount", "care", "professional", "contribution", "ship",
        "Chinese medicine", "position", "exercise", "incapacity", "profit",
        "bank", "health", "management", "delivery", "child", "facility",
        "leave", "corporation", "interim", "lift", "post", "corporate",
        "current", "level", "income", "rate", "immunity", "sum",
        "manager", "renewal", "industry", "absence", "healthcare",
        "associate", "incapable", "payable", "relief", "support",
        "engineer", "pay", "age", "vacancy", "duration", "medical",
        "mental", "retirement", "carrier", "hire", "transport", "village",
        "emergency", "works", "building", "insurer", "staff", "unfit",
        "entitlement", "neglect", "industrial", "working order",
        "eligibility", "gratuity", "personnel", "practising", "sitting",
        "vocational", "winding up", "expiry", "firm", "termination",
        "trader",
        "office",  # REC_DROP override → KEEP (job/office tenure canon)
        "applicant",  # REC_DROP override → KEEP
    },
    "procedure": {
        # RECOMMEND_KEEP
        "evidence", "appeal", "statement", "contrary", "proof", "hear",
        "examine", "witness", "objection", "appear", "substantiate",
        "beyond reasonable doubt",
        # REVIEW promotions
        "proceeding", "claim", "court", "review", "report", "issue",
        "approved", "hearing", "lawful", "specified", "effective",
        "relevant", "sufficient", "available", "reason", "request",
        "exclusive", "recognized", "arbitration", "competent", "summary",
        "investigation", "surrender", "actual", "preliminary", "process",
        "bona fide", "examination", "satisfy", "certified", "inquiry",
        "prior", "dispute", "fine", "written", "adjudication", "admit",
        "deliver", "letter", "serve", "attend", "determine", "represent",
        "signature", "valid", "confer", "existing", "establish",
        "supplementary", "accept", "extend", "notifiable", "recognize",
        "limitation", "petition", "presumption", "proceed", "publish",
        "audit", "endorse", "in favour of", "initiate", "uphold", "prove",
        "verification", "admissible", "submit", "exhibit", "submission",
        "doubt", "eligible", "interested", "continue", "documentary",
        "in writing", "inference", "informed", "conformity", "immediately",
        "in good faith", "inform", "prepare", "validity", "adjudicate",
        "ascertain", "certified copy", "identify",
        "form",  # REC_DROP override → KEEP (formal pleading canon)
        "respondent",  # REC_DROP override → KEEP (procedural party)
    },
}

# -----------------------------------------------------------------------------
# Per-domain PRUNE set — terms initially in KEEP that I now demote to DROP
# to bring each domain to ~60. Reasons: zh fragment too noisy, redundant with
# stronger sibling, or genericity outweighs marginal canon status. Applied as
# an override on KEEP set.
# -----------------------------------------------------------------------------
PRUNE: dict[str, set[str]] = {
    "civil": {
        # 21 to drop from 81 → 60
        "owner",          # 登記車主 = registered vehicle owner, wrongly narrowed
        "guarantee",      # 以……為受惠人的保證書 fragmentary
        "good faith",     # 真誠 not standard 善意
        "customer",       # 客戶盡職審查措施 AML-narrow
        "claimant",       # 致申索人通知書 fragmentary
        "will",           # 恢復已撤銷的遺囑 fragmentary
        "household", "relative", "matrimonial",  # 婚姻 duplicates marriage
        "covenant", "dealer", "distributor",
        "subscriber", "bearer", "lot",
        "design",         # IP-narrow; civil keeps copyright/trade mark
        "principle",      # 原則 generic
        "domicile", "proprietor", "stay", "carriage",
    },
    "criminal": {
        # additional zh-fragment cleanups (post-audit)
        "liable",         # 可遭受…… minimal-content fragment
        "detain",         # 串謀將人強行帶走…取得贖金 kidnapping-narrow, redundant with detention
        "hostage",        # 《反對劫持人質國際公約》 treaty title (D2.5)
        # 24 to drop from 84 → 60
        "execution",      # 簽立 = signing of instrument, wrong sense
        "encumbrance",    # civil overlap
        "bankruptcy",     # civil insolvency overlap
        "expulsion",      # immigration not criminal
        "withdrawal", "rehabilitation", "death",
        "armed", "force", "drug",
        "infringing",     # IP fragment, civil overlap
        "Tribunal", "tribunal", "removal", "judge",
        "police", "prohibited", "trial", "minor",
        "conduct", "judgment", "warrant",
        "pardon", "intimidation",  # duplicates intimidate
        "intimidate",     # KEEP just one of pair
    } - {"intimidate"},   # actually keep intimidate, drop intimidation
    "labor_social": {
        # ~45 to drop from 105 → ~60
        # treaty titles & wrong-sense
        "industrial", "carrier", "transport", "agent", "incapacity",
        "personnel", "engineer", "absence", "retirement", "support",
        "relief",
        # commercial/financial overflow that doesn't belong in labor (better in civil)
        "trader", "firm", "renewal", "expiry", "termination", "discharge",
        "bank", "investment", "corporate", "corporation",
        "interim", "facility", "income", "rate", "associate", "immunity",
        "ship", "village", "post",
        # generic boilerplate / state words
        "eligibility", "neglect", "unfit", "vacancy", "construction",
        "duration", "mental", "incapable", "hire", "delay", "offer",
        "working order", "sitting", "practising",
        "audit", "prepare", "renewal",
        "lift",
    },
    "procedure": {
        # ~48 to drop from 108 → ~60. Keep procedural canons even if zh fragment.
        # severely fragmented zh — drop
        "hear",          # 不能以……為藉口
        # generic verbs / phrases — drop
        "extend", "establish", "confer", "existing",
        "in favour of", "in good faith", "immediately", "letter",
        "deliver", "dispute", "uphold", "publish", "represent",
        "prepare", "report", "doubt", "interested", "eligible",
        "ascertain", "identify", "satisfy",
        "endorse", "request", "reason", "supplementary",
        "summary", "approved", "accept", "actual",
        "form",          # 形式 generic
        "respondent",    # zh fragment
        "recognize", "recognized",
        "exclusive", "conformity", "inference", "continue",
        "notifiable", "proceed", "preliminary",
        "submission", "submit", "verification",
        "competent",
        "deliver", "letter",
        "validity",
        # 2026-05-10 post-reviewer cleanup: drafting glue (D2.2 extension)
        # State adjectives with no legal-conceptual content of their own
        "contrary", "available", "effective", "relevant", "sufficient",
        "valid", "specified", "documentary", "informed", "inform",
        "prior", "attend", "appear", "adjudicate",
    },
    "administrative": {
        # 2026-05-10 post-reviewer cleanup: inflectional duplicates
        "certification",  # cluster with `certificate` (k=55 dominant)
        "authorized",     # adj of `authorization`
        "licensed",       # adj of `licence` (AUTO_KEEP)
        "licensing",      # gerund of `licence`
        "permitted",      # adj of `permit` (AUTO_KEEP)
    },
    "constitutional": {
        # 2026-05-10 post-reviewer cleanup
        "board",          # case-duplicate of Board
        "committee",      # case-duplicate of Committee
        "director",       # case-duplicate of Director
        "polling",        # gerund, cluster with `poll`
        "elect",          # verb, `election` + `elector` already KEEP
        "president",      # user decision: drop (no clean cross-tradition lemma)
    },
    "international": {
        "presence",       # 在……在場下 fragment "in the presence of"
        # 2026-05-10 post-reviewer cleanup: drafted-foreignness adjectives
        # User decision: drop the most adjective-generic; keep PIL-canonical
        # plus HK-specific (Mainland, non-Hong Kong, regional, overseas, outbound, independent).
        "Chinese", "domestic", "external", "foreign", "general", "national",
        "particular", "personal", "real", "subject", "related",
        "concurrent", "consequential",
    },
}


# Sanity check: verify no string accidentally lives in two domains' KEEP sets.
_seen = {}
for dom, terms in KEEP.items():
    for t in terms:
        if t in _seen:
            raise ValueError(f"'{t}' is in both {_seen[t]} and {dom} KEEP sets")
        _seen[t] = dom


def main() -> int:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    print(f"Loaded {len(rows)} rows from {CSV.name}")

    decisions = Counter()
    domain_counts = defaultdict(Counter)
    unrecognised_keep = []  # KEEP entries that don't match any row in the CSV

    for r in rows:
        dom = r["domain"]
        en = r["en"]
        ar = r["auto_recommendation"]
        keep_set = KEEP.get(dom, set())
        drop_override = AUTO_KEEP_DROP_OVERRIDE.get(dom, set())
        prev = r["curation_decision"]

        if r["tier_current"] == "core" and ar == "":
            # AUTO_KEEP stratum
            if en in drop_override:
                new = "DROP"
                note = "wrong-sense zh: D2.4 override"
            elif en in PRUNE.get(dom, set()):
                new = "DROP"
                note = "PRUNE applied to AUTO_KEEP (cleanup duplicate / drafting glue)"
            else:
                new = "KEEP"
                note = "AUTO_KEEP: core ∩ K_postBLP≥4"
        elif ar == "" and r["tier_current"] == "background":
            # AUTO_DROP stratum
            new = "DROP"
            note = prev or "AUTO_DROP"
        elif ar == "RECOMMEND_KEEP":
            if en in keep_set and en not in PRUNE.get(dom, set()):
                new = "KEEP"
                note = "REC_KEEP confirmed"
            else:
                new = "DROP"
                note = "REC_KEEP rejected: zh fragment / non-canonical / pruned"
        elif ar == "REVIEW":
            if en in keep_set and en not in PRUNE.get(dom, set()):
                new = "KEEP"
                note = "REVIEW promoted"
            else:
                new = "DROP"
                note = "REVIEW not selected / pruned"
        elif ar == "RECOMMEND_DROP":
            if en in keep_set and en not in PRUNE.get(dom, set()):
                new = "KEEP"
                note = "REC_DROP override: legal canon"
            else:
                new = "DROP"
                note = "REC_DROP confirmed"
        else:
            new = "DEFER"
            note = f"unhandled: ar={ar!r}, tier={r['tier_current']!r}"

        r["curation_decision"] = new
        r["curation_notes"] = note
        decisions[new] += 1
        domain_counts[dom][new] += 1

    # Tally KEEP entries that were specified but didn't match
    for dom, terms in KEEP.items():
        en_in_dom = {r["en"] for r in rows if r["domain"] == dom}
        missing = terms - en_in_dom
        for m in sorted(missing):
            unrecognised_keep.append((dom, m))

    print("\nDecision tally:")
    for d, n in sorted(decisions.items()):
        print(f"  {d:6} {n}")

    print("\nPer-domain (KEEP / DROP / DEFER / total):")
    for dom in sorted(domain_counts):
        c = domain_counts[dom]
        tot = sum(c.values())
        print(f"  {dom:18} K={c.get('KEEP',0):3}  D={c.get('DROP',0):4}  DEF={c.get('DEFER',0):3}  total={tot:4}")

    if unrecognised_keep:
        print(f"\n[!] {len(unrecognised_keep)} KEEP entries did not match any CSV row:")
        for dom, en in unrecognised_keep:
            print(f"     {dom}: '{en}'")

    fieldnames = list(rows[0].keys())
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
