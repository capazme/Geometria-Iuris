"""
build_dataset.py — Construct data/processed/legal_terms.json from raw sources.

Assembles core terms (~500), background terms (~500), control terms (~50),
value axes, and normative decompositions. Primary source: HK DOJ Bilingual
Legal Glossary (34k+ entries). Translations taken as-is from DOJ lookup.

Usage:
    python -m src.data.build_dataset
"""
# ─── Fonti e criteri di selezione ────────────────────────────────────
# Fonte primaria: HK DOJ Bilingual Legal Glossary (2026-02-05), 73.184
# record XML, 34.000+ chiavi uniche dopo indicizzazione per headword +
# english_def. Traduzioni prese così come sono dal glossario ufficiale.
#
# Per i termini non presenti nel DOJ (concetti confuciani, termini di
# teoria del diritto, control terms non giuridici), si usa CC-CEDICT
# come fonte secondaria.
#
# I 500 termini core coprono 9 domini giuridici con selezione basata
# su rilevanza comparatistica WEIRD/Sinic. I 500 background forniscono
# contesto semantico per k-NN (Exp. 5A). I 50 control terms sono
# termini non giuridici per baseline.
#
# Rif.: Kozlowski, Taddy & Evans (2019) "The Geometry of Culture",
#        American Sociological Review, 84(5).
# ─────────────────────────────────────────────────────────────────────

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def _load_doj_lookup() -> dict[str, dict]:
    """Load the DOJ lookup JSON (built by parse_hk_doj.py)."""
    path = get_project_root() / "data" / "processed" / "hk_doj_lookup.json"
    if not path.exists():
        raise FileNotFoundError(
            f"DOJ lookup not found: {path}\n"
            f"Run 'python -m src.data.parse_hk_doj' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _doj(lookup: dict, en: str, zh_override: str | None = None) -> str:
    """
    Get the DOJ translation for an English term.

    Returns the shortest zh_option (first in sorted list) unless
    zh_override is provided. Falls back to zh_override if term
    not found in DOJ.
    """
    key = en.lower()
    if key in lookup:
        return lookup[key]["zh_options"][0]
    if zh_override:
        return zh_override
    raise KeyError(f"Term '{en}' not in DOJ lookup and no override provided")


def build_core_terms() -> list[dict]:
    """
    Build ~500 core legal terms with high cultural relevance.

    Selected from HK DOJ glossary across 9 legal domains, focusing on
    terms where WEIRD and Sinic legal systems may diverge in meaning
    or conceptual scope.
    """
    lookup = _load_doj_lookup()

    def t(id_: str, en: str, domain: str, zh_override: str | None = None,
          source: str = "HK DOJ") -> dict:
        zh = _doj(lookup, en, zh_override)
        return {"id": id_, "en": en, "zh": zh, "domain": domain, "source": source}

    # Per i termini non nel DOJ, override manuale con source CC-CEDICT
    def t_cc(id_: str, en: str, zh: str, domain: str) -> dict:
        return {"id": id_, "en": en, "zh": zh, "domain": domain, "source": "CC-CEDICT"}

    return [
        # =================================================================
        # CONSTITUTIONAL (~55 termini)
        # Struttura dello stato, poteri, organi, forme di governo
        # =================================================================
        t("sovereignty", "Sovereignty", "constitutional"),
        t("constitution", "Constitution", "constitutional"),
        t("democracy", "Democracy", "constitutional"),
        t("separation_of_powers", "Separation of Powers", "constitutional"),
        t("judicial_independence", "Judicial Independence", "constitutional"),
        t("judicial_review", "Judicial Review", "constitutional"),
        t("rule_of_law", "Rule of Law", "constitutional"),
        t("prerogative", "Prerogative", "constitutional"),
        t("autonomy", "Autonomy", "constitutional"),
        t("devolution", "Devolution", "constitutional"),
        t("suffrage", "Suffrage", "constitutional"),
        t("citizenship", "Citizenship", "constitutional"),
        t("nationality", "Nationality", "constitutional"),
        t("naturalization", "Naturalization", "constitutional"),
        t("immigration", "Immigration", "constitutional"),
        t("deportation", "Deportation", "constitutional"),
        t("state", "State", "constitutional"),
        t("government", "Government", "constitutional"),
        t("authority", "Authority", "constitutional"),
        t("legitimacy", "Legitimacy", "constitutional"),
        t("legislature", "Legislature", "constitutional"),
        t("executive", "Executive", "constitutional"),
        t("judicial", "Judicial", "constitutional"),
        t("veto", "Veto", "constitutional"),
        t("impeachment", "Impeachment", "constitutional"),
        t("martial_law", "Martial Law", "constitutional"),
        t("emergency", "Emergency", "constitutional"),
        t("amnesty", "Amnesty", "constitutional"),
        t("curfew", "Curfew", "constitutional"),
        t("constitutional", "Constitutional", "constitutional"),
        t("legislative_council", "Legislative Council", "constitutional"),
        t("executive_council", "Executive Council", "constitutional"),
        t("basic_law", "Basic Law", "constitutional"),
        t("one_country_two_systems", "One Country, Two Systems", "constitutional"),
        t("national_security", "National Security", "constitutional"),
        t("election", "Election", "constitutional"),
        t("franchise", "Franchise", "constitutional"),
        t_cc("plebiscite", "Plebiscite", "公民投票", "constitutional"),
        t("proclamation", "Proclamation", "constitutional"),
        t("gazette", "Gazette", "constitutional"),
        t("promulgation", "Promulgation", "constitutional"),
        t("repeal", "Repeal", "constitutional"),
        t("amendment", "Amendment", "constitutional"),
        t("ratification", "Ratification", "constitutional"),
        t_cc("federalism", "Federalism", "联邦制", "constitutional"),
        t_cc("republic", "Republic", "共和国", "constitutional"),
        t_cc("monarchy", "Monarchy", "君主制", "constitutional"),
        t_cc("parliament", "Parliament", "议会", "constitutional"),
        t_cc("referendum", "Referendum", "公投", "constitutional"),
        t_cc("cabinet", "Cabinet", "内阁", "constitutional"),
        t_cc("tyranny", "Tyranny", "暴政", "constitutional"),
        t_cc("dictatorship", "Dictatorship", "独裁", "constitutional"),

        # =================================================================
        # RIGHTS (~65 termini)
        # Diritti fondamentali, libertà, garanzie processuali
        # =================================================================
        t("human_rights", "Human Rights", "rights"),
        t("freedom", "Freedom", "rights"),
        t("liberty", "Liberty", "rights"),
        t("privacy", "Privacy", "rights"),
        t("dignity", "Dignity", "rights"),
        t("equality", "Equality", "rights"),
        t("freedom_of_speech", "Freedom of Speech", "rights"),
        t("freedom_of_the_press", "Freedom of the Press", "rights"),
        t("freedom_of_association", "Freedom of Association", "rights"),
        t("freedom_of_conscience", "Freedom of Conscience", "rights"),
        t("freedom_of_contract", "Freedom of Contract", "rights"),
        t("freedom_of_religious_belief", "Freedom of Religious Belief", "rights"),
        t("presumption_of_innocence", "Presumption of Innocence", "rights"),
        t("due_process", "Due Process", "rights"),
        t("due_process_of_law", "Due Process of Law", "rights"),
        t("habeas_corpus", "Habeas Corpus", "rights"),
        t("fair_hearing", "Fair Hearing", "rights"),
        t("legal_aid", "Legal Aid", "rights"),
        t("right_to_counsel", "Legal Representation", "rights"),
        t("torture", "Torture", "rights"),
        t("slavery", "Slavery", "rights"),
        t("assembly", "Assembly", "rights"),
        t("petition", "Petition", "rights"),
        t("protest", "Protest", "rights"),
        t("censorship", "Censorship", "rights"),
        t("surveillance", "Surveillance", "rights"),
        t("asylum", "Asylum", "rights"),
        t("refugee", "Refugee", "rights"),
        t("discrimination", "Discrimination", "rights"),
        t("minority", "Minority", "rights"),
        t("indigenous", "Indigenous", "rights"),
        t("disability", "Disability", "rights"),
        t("self_determination", "Self-determination", "rights"),
        t("fundamental_rights", "Fundamental Rights", "rights"),
        t("bill_of_rights", "Bill of Rights", "rights"),
        t("right_of_abode", "Right of Abode", "rights"),
        t("right_of_appeal", "Right of Appeal", "rights"),
        t("right_to_silence", "Right to Silence", "rights"),
        t("burden_of_proof", "Burden of Proof", "rights"),
        t("beyond_reasonable_doubt", "Beyond Reasonable Doubt", "rights"),
        t("balance_of_probabilities", "Balance of Probabilities", "rights"),
        t("standard_of_proof", "Standard of Proof", "rights"),
        t("presumption", "Presumption", "rights"),
        t("vote", "Vote", "rights"),
        t("expression", "Expression", "rights"),
        t("conscience", "Conscience", "rights"),
        t_cc("due_process_rights", "Due Process Rights", "正当程序权利", "rights"),
        t_cc("freedom_of_expression", "Freedom of Expression", "表达自由", "rights"),
        t_cc("freedom_of_religion", "Freedom of Religion", "宗教自由", "rights"),
        t_cc("fair_trial", "Fair Trial", "公正审判", "rights"),

        # =================================================================
        # CIVIL (~70 termini)
        # Proprietà, contratti, famiglia, commerciale, successioni
        # =================================================================
        t("property", "Property", "civil"),
        t("contract", "Contract", "civil"),
        t("tort", "Tort", "civil"),
        t("negligence", "Negligence", "civil"),
        t("liability", "Liability", "civil"),
        t("obligation", "Obligation", "civil"),
        t("intellectual_property", "Intellectual Property", "civil"),
        t("ownership", "Ownership", "civil"),
        t("possession", "Possession", "civil"),
        t("consideration", "Consideration", "civil"),
        t("estoppel", "Estoppel", "civil"),
        t("misrepresentation", "Misrepresentation", "civil"),
        t("duress", "Duress", "civil"),
        t("frustration", "Frustration", "civil"),
        t("breach", "Breach", "civil"),
        t("rescission", "Rescission", "civil"),
        t("rectification", "Rectification", "civil"),
        t("restitution", "Restitution", "civil"),
        t("fraud", "Fraud", "civil"),
        t("damages", "Damages", "civil"),
        t("compensation", "Compensation", "civil"),
        t("warranty", "Warranty", "civil"),
        t("indemnity", "Indemnity", "civil"),
        t("guarantee", "Guarantee", "civil"),
        t("lien", "Lien", "civil"),
        t("pledge", "Pledge", "civil"),
        t("mortgage", "Mortgage", "civil"),
        t("lease", "Lease", "civil"),
        t("tenancy", "Tenancy", "civil"),
        t("easement", "Easement", "civil"),
        t("conveyance", "Conveyance", "civil"),
        t("assignment", "Assignment", "civil"),
        t("novation", "Novation", "civil"),
        t("subrogation", "Subrogation", "civil"),
        t("trust", "Trust", "civil"),
        t("will", "Will", "civil"),
        t("inheritance", "Inheritance", "civil"),
        t("adoption", "Adoption", "civil"),
        t("marriage", "Marriage", "civil"),
        t("divorce", "Divorce", "civil"),
        t("custody", "Custody", "civil"),
        t("alimony", "Alimony", "civil"),
        t("guardianship", "Guardianship", "civil"),
        t("maintenance", "Maintenance", "civil"),
        t("bankruptcy", "Bankruptcy", "civil"),
        t("insolvency", "Insolvency", "civil"),
        t("liquidation", "Liquidation", "civil"),
        t("winding_up", "Winding Up", "civil"),
        t("merger", "Merger", "civil"),
        t("acquisition", "Acquisition", "civil"),
        t("corporation", "Corporation", "civil"),
        t("partnership", "Partnership", "civil"),
        t("shareholder", "Shareholder", "civil"),
        t("fiduciary", "Fiduciary", "civil"),
        t("agency", "Agency", "civil"),
        t("patent", "Patent", "civil"),
        t("copyright", "Copyright", "civil"),
        t("trade_mark", "Trade Mark", "civil"),
        t("bailment", "Bailment", "civil"),
        t("specific_performance", "Specific Performance", "civil"),
        t("waiver", "Waiver", "civil"),
        t("disclaimer", "Disclaimer", "civil"),
        t("limitation", "Limitation", "civil"),
        t("prescription", "Prescription", "civil"),
        t("good_faith", "Good Faith", "civil"),
        t("joint_venture", "Joint Venture", "civil"),
        t("debenture", "Debenture", "civil"),
        t("securities", "Securities", "civil"),
        t("insurance", "Insurance", "civil"),
        t("surety", "Surety", "civil"),

        # =================================================================
        # CRIMINAL (~60 termini)
        # Reati, procedura penale, pene, difese
        # =================================================================
        t("crime", "Crime", "criminal"),
        t("punishment", "Punishment", "criminal"),
        t("justice", "Justice", "criminal"),
        t("guilt", "Guilt", "criminal"),
        t("murder", "Murder", "criminal"),
        t("manslaughter", "Manslaughter", "criminal"),
        t("homicide", "Homicide", "criminal"),
        t("infanticide", "Infanticide", "criminal"),
        t("assault", "Assault", "criminal"),
        t("robbery", "Robbery", "criminal"),
        t("theft", "Theft", "criminal"),
        t("burglary", "Burglary", "criminal"),
        t("larceny", "Larceny", "criminal"),
        t("embezzlement", "Embezzlement", "criminal"),
        t("bribery", "Bribery", "criminal"),
        t("corruption", "Corruption", "criminal"),
        t("perjury", "Perjury", "criminal"),
        t("conspiracy", "Conspiracy", "criminal"),
        t("sedition", "Sedition", "criminal"),
        t("treason", "Treason", "criminal"),
        t("arson", "Arson", "criminal"),
        t("kidnapping", "Kidnapping", "criminal"),
        t("blackmail", "Blackmail", "criminal"),
        t("extortion", "Extortion", "criminal"),
        t("forgery", "Forgery", "criminal"),
        t("smuggling", "Smuggling", "criminal"),
        t("trafficking", "Trafficking", "criminal"),
        t("piracy", "Piracy", "criminal"),
        t("mens_rea", "Mens Rea", "criminal"),
        t("actus_reus", "Actus Reus", "criminal"),
        t("death_penalty", "Death Penalty", "criminal"),
        t("capital_punishment", "Capital Punishment", "criminal"),
        t("imprisonment", "Imprisonment", "criminal"),
        t("detention", "Detention", "criminal"),
        t("parole", "Parole", "criminal"),
        t("probation", "Probation", "criminal"),
        t("bail", "Bail", "criminal"),
        t("sentencing", "Sentencing", "criminal"),
        t("conviction", "Conviction", "criminal"),
        t("acquittal", "Acquittal", "criminal"),
        t("indictment", "Indictment", "criminal"),
        t("plea", "Plea", "criminal"),
        t("confession", "Confession", "criminal"),
        t("accomplice", "Accomplice", "criminal"),
        t("aiding_and_abetting", "Aiding and Abetting", "criminal"),
        t("money_laundering", "Money Laundering", "criminal"),
        t("cybercrime", "Cybercrime", "criminal"),
        t("criminal_law", "Criminal Law", "criminal"),
        t_cc("terrorism", "Terrorism", "恐怖主义", "criminal"),
        t_cc("counterfeiting", "Counterfeiting", "伪造", "criminal"),
        t_cc("war_crime", "War Crime", "战争罪", "criminal"),

        # =================================================================
        # GOVERNANCE (~60 termini)
        # PA, regolazione, fiscalità, polizia, politica pubblica
        # =================================================================
        t("legislation", "Legislation", "governance"),
        t("regulation", "Regulation", "governance"),
        t("compliance", "Compliance", "governance"),
        t("enforcement", "Enforcement", "governance"),
        t("sanction", "Sanction", "governance"),
        t("embargo", "Embargo", "governance"),
        t("public_interest", "Public Interest", "governance"),
        t("public_policy", "Public Policy", "governance"),
        t("public_order", "Public Order", "governance"),
        t("police", "Police", "governance"),
        t("prison", "Prison", "governance"),
        t("correctional", "Correctional", "governance"),
        t("rehabilitation", "Rehabilitation", "governance"),
        t("commutation", "Commutation", "governance"),
        t("clemency", "Clemency", "governance"),
        t("pardon", "Pardon", "governance"),
        t("reprieve", "Stay of Execution", "governance"),
        t("remission", "Remission", "governance"),
        t("delegation", "Delegation", "governance"),
        t("privatization", "Privatization", "governance"),
        t("nationalization", "Nationalization", "governance", zh_override="国有化"),
        t("expropriation", "Expropriation", "governance"),
        t("confiscation", "Confiscation", "governance"),
        t("compulsory_acquisition", "Compulsory Acquisition", "governance"),
        t("taxation", "Taxation", "governance"),
        t("tariff", "Tariff", "governance"),
        t("customs", "Customs", "governance"),
        t("excise", "Excise", "governance"),
        t("duty", "Duty", "governance"),
        t("revenue", "Revenue", "governance"),
        t("audit", "Audit", "governance"),
        t("license", "Licence", "governance"),
        t("permit", "Permit", "governance"),
        t("inspection", "Inspection", "governance"),
        t("quarantine", "Quarantine", "governance"),
        t("transparency", "Transparency", "governance"),
        t("accountability", "Accountability", "governance"),
        t("corruption_governance", "Corrupt", "governance"),
        t("ombudsman", "Ombudsman", "governance"),
        t("public_health", "Public Health", "governance"),
        t("planning_permission", "Building Plans", "governance"),
        t("zoning", "Zoning", "governance", zh_override="分区"),
        t("land_use", "Land Use", "governance", zh_override="土地用途"),
        t_cc("bureaucracy", "Bureaucracy", "官僚体制", "governance"),
        t_cc("subsidy", "Subsidy", "补贴", "governance"),
        t_cc("ministry", "Ministry", "部委", "governance"),
        t_cc("municipality", "Municipality", "市政府", "governance"),

        # =================================================================
        # JURISPRUDENCE (~55 termini)
        # Teoria del diritto, fonti, principi generali
        # =================================================================
        t("law", "Law", "jurisprudence"),
        t("rights_concept", "Rights", "jurisprudence"),
        t("equity", "Equity", "jurisprudence"),
        t("common_law", "Common Law", "jurisprudence"),
        t("civil_law", "Civil Law", "jurisprudence"),
        t("precedent", "Precedent", "jurisprudence"),
        t("custom", "Custom", "jurisprudence"),
        t("convention", "Convention", "jurisprudence"),
        t("doctrine", "Doctrine", "jurisprudence"),
        t("stare_decisis", "Stare Decisis", "jurisprudence"),
        t("res_judicata", "Res Judicata", "jurisprudence"),
        t("ultra_vires", "Ultra Vires", "jurisprudence"),
        t("intra_vires", "Intra Vires", "jurisprudence"),
        t("bona_fide", "Bona Fide", "jurisprudence"),
        t("mala_fide", "Mala Fide", "jurisprudence"),
        t("obiter", "Obiter", "jurisprudence"),
        t("ratio", "Ratio", "jurisprudence"),
        t("consent", "Consent", "jurisprudence"),
        t("laches", "Laches", "jurisprudence"),
        t("acquiescence", "Acquiescence", "jurisprudence"),
        t("codification", "Codification", "jurisprudence"),
        t("interpretation", "Interpretation", "jurisprudence"),
        t("adjudication", "Adjudication", "jurisprudence"),
        t("rules_of_natural_justice", "Rules of Natural Justice", "jurisprudence"),
        t("international_law", "International Law", "jurisprudence"),
        t("statute", "Statute", "jurisprudence"),
        t("ordinance", "Ordinance", "jurisprudence"),
        t("decree", "Decree", "jurisprudence"),
        t("morality", "Morality", "jurisprudence"),
        t("ethics", "Ethics", "jurisprudence"),
        t("order", "Order", "jurisprudence"),
        t("harmony", "Harmony", "jurisprudence"),
        t("legitimacy_juris", "Lawfulness", "jurisprudence"),
        t_cc("natural_law", "Natural Law", "自然法", "jurisprudence"),
        t_cc("positive_law", "Positive Law", "实证法", "jurisprudence"),
        t_cc("social_contract", "Social Contract", "社会契约", "jurisprudence"),
        t_cc("civil_society", "Civil Society", "公民社会", "jurisprudence"),
        t_cc("common_good", "Common Good", "公共善", "jurisprudence"),
        t_cc("rule_by_law", "Rule by Law", "以法治国", "jurisprudence"),

        # Concetti confuciani / tradizione giuridica cinese
        t_cc("li_ritual", "Ritual Propriety", "礼", "jurisprudence"),
        t_cc("ren_benevolence", "Benevolence", "仁", "jurisprudence"),
        t_cc("fa_law", "Legalism", "法", "jurisprudence"),
        t_cc("de_virtue", "Moral Virtue", "德", "jurisprudence"),
        t_cc("xiao_filial_piety", "Filial Piety", "孝", "jurisprudence"),
        t_cc("zhong_loyalty", "Loyalty to Ruler", "忠", "jurisprudence"),
        t_cc("yi_righteousness", "Righteousness", "义", "jurisprudence"),
        t_cc("gongzheng", "Fairness", "公正", "jurisprudence"),
        t_cc("minzhu_jizhongzhi", "Democratic Centralism", "民主集中制", "jurisprudence"),
        t_cc("fazhi_guojia", "Socialist Rule of Law", "社会主义法治", "jurisprudence"),

        # =================================================================
        # INTERNATIONAL (~45 termini)
        # Trattati, organizzazioni, diritto umanitario, commercio
        # =================================================================
        t("treaty", "Treaty", "international"),
        t("jurisdiction", "Jurisdiction", "international"),
        t("extradition", "Extradition", "international"),
        t("immunity", "Immunity", "international"),
        t("recognition", "Recognition", "international"),
        t("intervention", "Intervention", "international"),
        t("peacekeeping", "Peacekeeping", "international", zh_override="维和"),
        t("genocide", "Genocide", "international"),
        t("humanitarian", "Humanitarian", "international"),
        t("protocol", "Protocol", "international"),
        t("accession", "Accession", "international"),
        t("reservation", "Reservation", "international"),
        t("arbitration", "Arbitration", "international"),
        t("mediation", "Mediation", "international"),
        t("conciliation", "Conciliation", "international"),
        t("neutral", "Neutral", "international"),
        t("belligerent", "Belligerent", "international", zh_override="交战方"),
        t("diplomacy", "Diplomatic Relations", "international"),
        t("diplomatic", "Diplomatic", "international"),
        t("embargo_intl", "Blockade", "international", zh_override="封锁"),
        t("sanction_intl", "Economic Sanction", "international", zh_override="经济制裁"),
        t("reciprocity", "Reciprocity", "international"),
        t("sovereignty_intl", "Territorial Integrity", "international", zh_override="领土完整"),
        t("negotiation", "Negotiation", "international"),
        t("good_offices", "Good Offices", "international", zh_override="斡旋"),
        t("repatriation", "Repatriation", "international"),
        t("stateless", "Stateless", "international"),
        t("convention_intl", "International Convention", "international"),
        t_cc("self_determination_intl", "Right to Self-determination", "自决权", "international"),
        t_cc("humanitarian_law", "Humanitarian Law", "人道法", "international"),
        t_cc("crimes_against_humanity", "Crimes against Humanity", "反人类罪", "international"),

        # =================================================================
        # LABOR & SOCIAL (~45 termini)
        # Lavoro, sicurezza sociale, welfare, discriminazione
        # =================================================================
        t("employment", "Employment", "labor_social"),
        t("dismissal", "Dismissal", "labor_social"),
        t("severance", "Severance", "labor_social"),
        t("compensation_labor", "Employees' Compensation", "labor_social"),
        t("occupational", "Occupational", "labor_social"),
        t("safety", "Safety", "labor_social"),
        t("pension", "Pension", "labor_social"),
        t("welfare", "Welfare", "labor_social"),
        t("harassment", "Harassment", "labor_social"),
        t("maternity", "Maternity", "labor_social"),
        t("paternity", "Paternity", "labor_social"),
        t("overtime", "Overtime", "labor_social"),
        t("leave", "Leave", "labor_social"),
        t("strike", "Strike", "labor_social"),
        t("lockout", "Lockout", "labor_social"),
        t("trade_union", "Trade Union", "labor_social"),
        t("collective_bargaining", "Collective Bargaining", "labor_social", zh_override="集体谈判"),
        t("unfair_dismissal", "Unfair Dismissal", "labor_social"),
        t("redundancy", "Severance Payment", "labor_social"),
        t("minimum_wage", "Minimum Wage", "labor_social"),
        t("social_welfare", "Social Welfare", "labor_social"),
        t("personal_data", "Personal Data", "labor_social"),
        t_cc("labor_union", "Labor Union", "工会", "labor_social"),
        t_cc("social_security", "Social Security", "社会保障", "labor_social"),
        t("data_protection", "Data Protection", "labor_social", zh_override="资料保障"),

        # =================================================================
        # ENVIRONMENTAL & TECH (~45 termini)
        # Ambiente, tecnologia, dati, IP moderna
        # =================================================================
        t("pollution", "Pollution", "environmental_tech"),
        t("conservation", "Conservation", "environmental_tech"),
        t("sustainability", "Sustainability", "environmental_tech"),
        t("waste", "Waste", "environmental_tech"),
        t("emission", "Emission", "environmental_tech"),
        t("renewable", "Renewable", "environmental_tech"),
        t("nuclear", "Nuclear", "environmental_tech", zh_override="核"),
        t("e_commerce", "E-Commerce", "environmental_tech"),
        t("electronic", "Electronic", "environmental_tech"),
        t("digital", "Digital", "environmental_tech"),
        t("internet", "Internet", "environmental_tech"),
        t("telecommunications", "Telecommunications", "environmental_tech"),
        t("broadcasting", "Broadcasting", "environmental_tech"),
        t("spectrum", "Spectrum", "environmental_tech"),
        t("encryption", "Encryption", "environmental_tech"),
        t_cc("environmental_law", "Environmental Law", "环境法", "environmental_tech"),
        t_cc("climate_change", "Climate Change", "气候变化", "environmental_tech"),
        t_cc("biodiversity", "Biodiversity", "生物多样性", "environmental_tech"),
        t_cc("artificial_intelligence", "Artificial Intelligence", "人工智能", "environmental_tech"),
    ]


def build_background_terms() -> list[dict]:
    """
    Build ~500 background terms for k-NN neighborhood context.

    Procedurally neutral terms and general legal vocabulary from DOJ
    glossary, providing semantic backdrop for NDA (Exp. 5A).
    """
    lookup = _load_doj_lookup()

    def t(id_: str, en: str, zh_override: str | None = None,
          source: str = "HK DOJ") -> dict:
        zh = _doj(lookup, en, zh_override)
        return {"id": id_, "en": en, "zh": zh, "source": source}

    def t_cc(id_: str, en: str, zh: str) -> dict:
        return {"id": id_, "en": en, "zh": zh, "source": "CC-CEDICT"}

    return [
        # ─── Procedura civile ────────────────────────────────────────
        t("appeal", "Appeal"),
        t("plaintiff", "Plaintiff"),
        t("defendant", "Defendant"),
        t("verdict", "Verdict"),
        t("judgment", "Judgment"),
        t("evidence", "Evidence"),
        t("witness", "Witness"),
        t("testimony", "Testimony"),
        t("court", "Court"),
        t("trial", "Trial"),
        t("hearing", "Hearing"),
        t("motion", "Motion"),
        t("complaint", "Complaint"),
        t("summons", "Summons"),
        t("subpoena", "Subpoena"),
        t("affidavit", "Affidavit"),
        t("deposition", "Deposition"),
        t("discovery", "Discovery"),
        t("injunction", "Injunction"),
        t("writ", "Writ"),
        t("cross_examination", "Cross-examination"),
        t("examination_in_chief", "Examination in Chief"),
        t("pleading", "Pleading"),
        t("counterclaim", "Counterclaim"),
        t("interlocutory", "Interlocutory"),
        t("originating_summons", "Originating Summons"),
        t("statement_of_claim", "Statement of Claim"),
        t("defence_pleading", "Defence"),
        t("reply_pleading", "Reply"),
        t("rejoinder", "Rejoinder"),
        t("interrogatory", "Interrogatory"),
        t("costs", "Costs"),
        t("taxation_of_costs", "Taxation of Costs"),
        t("garnishee", "Garnishee"),
        t("interpleader", "Interpleader"),
        t("third_party_proceedings", "Third Party"),
        t("default_judgment", "Default Judgment"),
        t("summary_judgment", "Summary Judgment"),
        t("stay_of_proceedings", "Stay of Proceedings"),
        t("discontinuance", "Discontinuance"),
        t("settlement", "Settlement"),
        t("consent_order", "Consent Order"),

        # ─── Procedura penale ────────────────────────────────────────
        t("arrest", "Arrest"),
        t("charge", "Charge"),
        t("caution", "Caution"),
        t("remand", "Remand"),
        t("committal", "Committal"),
        t("arraignment", "Arraignment"),
        t("voir_dire", "Voir Dire"),
        t("verdict_criminal", "Not Guilty"),
        t("nolle_prosequi", "Nolle Prosequi"),
        t("mitigation", "Mitigation"),
        t("allocution", "Allocution", zh_override="陈述"),
        t("suspended_sentence", "Suspended Sentence"),
        t("community_service", "Community Service Order"),
        t("fine", "Fine"),
        t("forfeiture", "Forfeiture"),
        t("restitution_order", "Restitution Order", zh_override="赔偿令"),
        t("binding_over", "Bind Over"),
        t("conditional_discharge", "Conditional Discharge"),
        t("absolute_discharge", "Absolute Discharge"),

        # ─── Persone e ruoli ─────────────────────────────────────────
        t("judge", "Judge"),
        t("lawyer", "Lawyer"),
        t("prosecutor", "Prosecutor"),
        t("notary", "Notary"),
        t("arbitrator", "Arbitrator"),
        t("mediator", "Mediator"),
        t("guardian", "Guardian"),
        t("trustee", "Trustee"),
        t("beneficiary", "Beneficiary"),
        t("creditor", "Creditor"),
        t("debtor", "Debtor"),
        t("citizen", "Citizen"),
        t("alien", "Alien"),
        t("minor", "Minor"),
        t("executor", "Executor"),
        t("administrator_estate", "Administrator"),
        t("receiver", "Receiver"),
        t("barrister", "Barrister"),
        t("solicitor", "Solicitor"),
        t("magistrate", "Magistrate"),
        t("registrar", "Registrar"),
        t("coroner", "Coroner"),
        t("juror", "Juror"),
        t("litigant", "Litigant"),
        t("appellant", "Appellant"),
        t("respondent", "Respondent"),
        t("petitioner", "Petitioner"),
        t("claimant", "Claimant"),
        t("surety_person", "Bail Surety", zh_override="保释担保人"),
        t("informant", "Informant"),
        t("complainant", "Complainant"),
        t("accused", "Accused"),
        t("convict", "Convict"),
        t("inmate", "Inmate"),
        t("victim", "Victim"),
        t("donee", "Donee"),
        t("donor", "Donor"),
        t("assignee", "Assignee"),
        t("assignor", "Assignor"),
        t("mortgagor", "Mortgagor"),
        t("mortgagee", "Mortgagee"),
        t("lessor", "Lessor"),
        t("lessee", "Lessee"),
        t("vendor", "Vendor"),
        t("purchaser", "Purchaser"),
        t("seller", "Seller"),
        t("buyer", "Buyer"),
        t("guarantor", "Guarantor"),
        t("principal", "Principal"),
        t("agent_person", "Agent"),
        t("bailee", "Bailee"),
        t("bailor", "Bailor"),

        # ─── Documenti legali ────────────────────────────────────────
        t("deed", "Deed"),
        t("instrument", "Instrument"),
        t("certificate", "Certificate"),
        t("affirmation", "Affirmation"),
        t("oath", "Oath"),
        t("undertaking", "Undertaking"),
        t("bond_doc", "Bond"),
        t("debenture_doc", "Promissory Note"),
        t("power_of_attorney", "Power of Attorney"),
        t("proxy", "Proxy"),
        t("mandate", "Mandate"),
        t("authorization", "Authorization"),
        t("memorandum", "Memorandum"),
        t("articles_of_association", "Articles of Association"),
        t("prospectus", "Prospectus"),
        t("gazette_doc", "Gazette"),
        t("notice", "Notice"),
        t("summons_doc", "Writ of Summons"),

        # ─── Istituzioni ─────────────────────────────────────────────
        t("high_court", "High Court"),
        t("district_court", "District Court"),
        t("magistrates_court", "Magistrate's Court"),
        t("court_of_appeal", "Court of Appeal"),
        t("court_of_final_appeal", "Court of Final Appeal"),
        t("tribunal", "Tribunal"),
        t("coroners_court", "Coroner's Court", zh_override="死因裁判法庭"),
        t("small_claims_tribunal", "Small Claims Tribunal"),
        t("lands_tribunal", "Lands Tribunal"),
        t("labour_tribunal", "Labour Tribunal"),
        t("registry", "Registry"),
        t("legal_department", "Department of Justice"),

        # ─── Terminologia commerciale ────────────────────────────────
        t("share", "Share"),
        t("dividend", "Dividend"),
        t("profit", "Profit"),
        t("loss", "Loss"),
        t("asset", "Asset"),
        t("capital", "Capital"),
        t("investment", "Investment"),
        t("bond_fin", "Bond"),
        t("equity_fin", "Equity"),
        t("derivative", "Derivative"),
        t("interest", "Interest"),
        t("principal_fin", "Principal Sum", zh_override="本金"),
        t("collateral", "Collateral"),
        t("credit", "Credit"),
        t("debit", "Debit"),
        t("default", "Default"),
        t("arrears", "Arrears"),
        t("indemnification", "Indemnification"),
        t("underwriting", "Underwriting"),
        t("flotation", "Flotation", zh_override="上市发行"),
        t("listing", "Listing"),
        t("prospectus_fin", "Listing Prospectus", zh_override="上市招股书"),
        t("insider_dealing", "Insider Dealing"),
        t("market_misconduct", "Market Misconduct"),
        t("director", "Director"),
        t("company_secretary", "Company Secretary"),
        t("annual_return", "Annual Return"),
        t("resolution", "Resolution"),
        t("winding_up_bg", "Compulsory Winding Up", zh_override="强制清盘"),
        t("receivership", "Receivership"),
        t("scheme_of_arrangement", "Scheme of Arrangement"),

        # ─── Terminologia fiscale ────────────────────────────────────
        t("income_tax", "Income Tax"),
        t("profits_tax", "Profits Tax"),
        t("stamp_duty", "Stamp Duty"),
        t("estate_duty", "Estate Duty"),
        t("rates", "Rates"),
        t("assessment", "Assessment"),
        t("exemption", "Exemption"),
        t("allowance", "Allowance"),
        t("deduction", "Deduction"),
        t("rebate", "Rebate"),
        t("refund", "Refund"),
        t("levy", "Levy"),
        t("surcharge", "Surcharge"),

        # ─── Terminologia immobiliare ────────────────────────────────
        t("freehold", "Freehold"),
        t("leasehold", "Leasehold"),
        t("covenant", "Covenant"),
        t("encumbrance", "Encumbrance"),
        t("title", "Title"),
        t("deed_of_mutual_covenant", "Deed of Mutual Covenant"),
        t("sub_lease", "Sub-lease"),
        t("licence_to_assign", "Licence to Assign", zh_override="转让许可"),
        t("surrender", "Surrender"),
        t("forfeiture_lease", "Forfeiture of Lease", zh_override="租赁没收"),
        t("dilapidation", "Dilapidation", zh_override="失修"),
        t("fixtures", "Fixtures", zh_override="固定附着物"),
        t("chattels", "Chattels"),
        t("real_property", "Real Property"),
        t("personal_property", "Personal Property"),
        t("estate_property", "Estate"),

        # ─── Terminologia marittima ──────────────────────────────────
        t("admiralty", "Admiralty"),
        t("salvage", "Salvage"),
        t("general_average", "General Average"),
        t("charter_party", "Charter Party"),
        t("bill_of_lading", "Bill of Lading"),
        t("carriage_of_goods", "Carriage of Goods", zh_override="货物运输"),
        t("demurrage", "Demurrage"),
        t("bottomry", "Bottomry"),
        t("freight", "Freight"),
        t("maritime_lien", "Maritime Lien"),
        t("collision", "Collision"),

        # ─── Terminologia assicurativa ───────────────────────────────
        t("premium", "Premium"),
        t("policy", "Policy"),
        t("claim", "Claim"),
        t("subrogation_ins", "Right of Subrogation", zh_override="代位求偿权"),
        t("underwriter", "Underwriter"),
        t("reinsurance", "Reinsurance"),
        t("indemnity_ins", "Indemnity Policy", zh_override="赔偿保单"),
        t("insurable_interest", "Insurable Interest"),
        t("utmost_good_faith", "Utmost Good Faith"),
        t("contribution", "Contribution"),
        t("average", "Average"),

        # ─── Principi e massime latine ───────────────────────────────
        t("culpa", "Culpa"),
        t("dolus", "Dolus"),
        t("nexus", "Nexus"),
        t("cy_pres", "Cy-près"),
        t("nemo_judex", "Nemo Judex in Causa Sua"),
        t("audi_alteram_partem", "Audi Alteram Partem"),
        t("volenti", "Volenti Non Fit Injuria", zh_override="自愿者不构成侵害"),
        t("ex_parte", "Ex Parte"),
        t("in_camera", "In Camera"),
        t("prima_facie", "Prima Facie"),
        t("ab_initio", "Ab Initio"),
        t("ad_hoc", "Ad Hoc"),
        t("de_facto", "De Facto"),
        t("de_jure", "De Jure"),
        t("inter_alia", "Inter Alia"),
        t("pro_rata", "Pro Rata"),
        t("bona_vacantia", "Bona Vacantia"),
        t("quantum_meruit", "Quantum Meruit"),
        t("lis_pendens", "Lis Pendens"),
        t("sub_judice", "Sub Judice"),
        t("obiter_dictum", "Obiter Dictum"),
        t("ratio_decidendi", "Ratio Decidendi"),
        t("amicus_curiae", "Amicus Curiae"),

        # ─── Concetti astratti / filosofici ──────────────────────────
        t("virtue", "Virtue"),
        t("stability", "Stability"),
        t("reform", "Reform"),
        t("community", "Community"),
        t("individual", "Individual"),
        t("collective", "Collective"),
        t("nation", "Nation"),
        t("people", "People"),
        t("society", "Society"),
        t("family", "Family"),
        t("person", "Person"),
        t("obedience", "Obedience"),
        t("loyalty", "Loyalty"),
        t_cc("patriotism", "Patriotism", "爱国"),
        t_cc("tradition", "Tradition", "传统"),
        t_cc("modernity", "Modernity", "现代性"),
        t_cc("civilization", "Civilization", "文明"),
        t_cc("culture", "Culture", "文化"),
        t_cc("progress", "Progress", "进步"),
        t_cc("revolution", "Revolution", "革命"),

        # ─── Termini aggiuntivi procedurali ──────────────────────────
        t("deadline", "Deadline"),
        t("filing", "Filing"),
        t("registration", "Registration"),
        t("signature", "Signature"),
        t("seal", "Seal"),
        t("clause", "Clause"),
        t("provision", "Provision"),
        t("appeal_court_bg", "Court of Appeal"),
        t("limitation_period", "Limitation Period"),
        t("confirmation", "Confirmation"),
        t("revocation", "Revocation"),
        t("suspension", "Suspension"),
        t("abatement", "Abatement"),
        t("ademption", "Ademption", zh_override="遗赠物灭失"),
        t("advancement", "Advancement"),
        t("attestation", "Attestation"),
        t("codicil", "Codicil"),
        t("probate", "Probate"),
        t("letters_of_administration", "Letters of Administration"),
        t("caveat", "Caveat"),
        t("citation", "Citation"),
        t("renunciation", "Renunciation"),
        t("disclaimer_bg", "Disclaimer"),
        t("vesting", "Vesting"),
        t("severance_joint", "Severance of Joint Tenancy", zh_override="联权共有的分割"),
        t("partition", "Partition"),
        t("apportionment", "Apportionment"),

        # ─── Termini aggiuntivi sostanziali ──────────────────────────
        t("trespass", "Trespass"),
        t("nuisance", "Nuisance"),
        t("defamation", "Defamation"),
        t("libel", "Libel"),
        t("slander", "Slander"),
        t("malicious_prosecution", "Malicious Prosecution"),
        t("false_imprisonment", "False Imprisonment"),
        t("conversion", "Conversion"),
        t("detinue", "Detinue"),
        t("unjust_enrichment", "Unjust Enrichment"),
        t("constructive_trust", "Constructive Trust"),
        t("resulting_trust", "Resulting Trust"),
        t("express_trust", "Express Trust", zh_override="明示信托"),
        t("charitable_trust", "Charitable Trust"),
        t("undue_influence_bg", "Undue Influence", zh_override="不当影响"),
        t("unconscionable_bg", "Unconscionable"),
        t("champerty", "Champerty"),
        t("maintenance_bg", "Maintenance of Action", zh_override="包揽诉讼"),
        t("barratry", "Barratry"),

        # ─── Termini HK-specifici ────────────────────────────────────
        t("cap", "Cap"),
        t("subsidiary_legislation", "Subsidiary Legislation"),
        t("official_languages", "Official Languages"),
        t("chinese_customary_law", "Chinese Custom"),
        t("new_territories", "New Territories"),
        t("tso", "Tso"),
        t("tong", "Tong"),
        t("ding", "Ding", zh_override="丁"),
        t("heung_yee_kuk", "Heung Yee Kuk"),
        t("icac", "Independent Commission Against Corruption"),
    ]


def build_control_terms() -> list[dict]:
    """
    Build ~50 non-legal control terms for baseline comparison.

    Concrete, everyday terms with universally standard translations.
    Not from DOJ (not a general dictionary).
    Source: CC-CEDICT / standard translations.
    """
    def t(id_: str, en: str, zh: str, category: str) -> dict:
        return {"id": id_, "en": en, "zh": zh, "category": category, "source": "CC-CEDICT"}

    return [
        # Oggetti (12)
        t("table", "Table", "桌子", "objects"),
        t("chair", "Chair", "椅子", "objects"),
        t("car", "Car", "汽车", "objects"),
        t("book", "Book", "书", "objects"),
        t("phone", "Phone", "电话", "objects"),
        t("computer", "Computer", "电脑", "objects"),
        t("bicycle", "Bicycle", "自行车", "objects"),
        t("clock", "Clock", "钟", "objects"),
        t("mirror", "Mirror", "镜子", "objects"),
        t("umbrella", "Umbrella", "雨伞", "objects"),
        t("pen", "Pen", "笔", "objects"),
        t("cup", "Cup", "杯子", "objects"),

        # Animali (8)
        t("dog", "Dog", "狗", "animals"),
        t("cat", "Cat", "猫", "animals"),
        t("fish", "Fish", "鱼", "animals"),
        t("horse", "Horse", "马", "animals"),
        t("bird", "Bird", "鸟", "animals"),
        t("tiger", "Tiger", "虎", "animals"),
        t("elephant", "Elephant", "大象", "animals"),
        t("rabbit", "Rabbit", "兔子", "animals"),

        # Natura (8)
        t("river", "River", "河", "nature"),
        t("mountain", "Mountain", "山", "nature"),
        t("rain", "Rain", "雨", "nature"),
        t("sun", "Sun", "太阳", "nature"),
        t("tree", "Tree", "树", "nature"),
        t("flower", "Flower", "花", "nature"),
        t("ocean", "Ocean", "海洋", "nature"),
        t("forest", "Forest", "森林", "nature"),

        # Cibo (6)
        t("rice", "Rice", "米", "food"),
        t("bread", "Bread", "面包", "food"),
        t("tea", "Tea", "茶", "food"),
        t("water", "Water", "水", "food"),
        t("fruit", "Fruit", "水果", "food"),
        t("meat", "Meat", "肉", "food"),

        # Luoghi (6)
        t("hospital", "Hospital", "医院", "places"),
        t("school", "School", "学校", "places"),
        t("market", "Market", "市场", "places"),
        t("airport", "Airport", "机场", "places"),
        t("library", "Library", "图书馆", "places"),
        t("museum", "Museum", "博物馆", "places"),

        # Azioni (5)
        t("cooking", "Cooking", "烹饪", "actions"),
        t("singing", "Singing", "唱歌", "actions"),
        t("running", "Running", "跑步", "actions"),
        t("reading", "Reading", "阅读", "actions"),
        t("sleeping", "Sleeping", "睡觉", "actions"),

        # Astratti quotidiani (5)
        t("weather", "Weather", "天气", "abstract_daily"),
        t("color", "Color", "颜色", "abstract_daily"),
        t("time", "Time", "时间", "abstract_daily"),
        t("number", "Number", "数字", "abstract_daily"),
        t("language", "Language", "语言", "abstract_daily"),
    ]


def build_value_axes() -> dict:
    """
    Build 3 value axes with 5-10 antonym pairs each (Kozlowski method).

    Sources: Rokeach (1973), Schwartz (2012), Geometry of Culture
    (Kozlowski, Taddy & Evans 2019), CVC/C-VARC, HK DOJ.
    """
    # Metodo Kozlowski: ogni asse è definito da 10 coppie di antonimi.
    # Le coppie sono selezionate per catturare la stessa dimensione valoriale
    # da angolazioni diverse (lessicali, istituzionali, astratte).
    # Coppie in inglese (en_pairs) e cinese (zh_pairs) sono traduzioni
    # indipendenti, non allineamenti meccanici: "autonomy/conformity"
    # e "自主/顺从" catturano lo stesso asse culturale ma con sfumature
    # specifiche di ciascuna lingua.
    return {
        "individual_collective": {
            "en_pairs": [
                ["freedom", "obedience"],
                ["autonomy", "conformity"],
                ["independence", "submission"],
                ["liberty", "duty"],
                ["self-reliance", "community"],
                ["privacy", "surveillance"],
                ["dissent", "consensus"],
                ["individual", "collective"],
                ["rights", "obligations"],
                ["choice", "tradition"],
            ],
            "zh_pairs": [
                ["自由", "服从"],
                ["自主", "顺从"],
                ["独立", "屈服"],
                ["自由", "义务"],
                ["自力更生", "社区"],
                ["隐私", "监控"],
                ["异议", "共识"],
                ["个人", "集体"],
                ["权利", "义务"],
                ["选择", "传统"],
            ],
        },
        "rights_duties": {
            "en_pairs": [
                ["rights", "duties"],
                ["entitlement", "obligation"],
                ["liberty", "responsibility"],
                ["claim", "compliance"],
                ["freedom", "obedience"],
                ["privilege", "burden"],
                ["immunity", "liability"],
                ["autonomy", "accountability"],
                ["consent", "command"],
                ["empowerment", "submission"],
            ],
            "zh_pairs": [
                ["权利", "义务"],
                ["权益", "责任"],
                ["自由", "责任"],
                ["主张", "遵从"],
                ["自由", "服从"],
                ["特权", "负担"],
                ["豁免", "责任"],
                ["自主", "问责"],
                ["同意", "命令"],
                ["赋权", "服从"],
            ],
        },
        "public_private": {
            "en_pairs": [
                ["public", "private"],
                ["state", "individual"],
                ["collective", "personal"],
                ["government", "citizen"],
                ["official", "unofficial"],
                ["transparency", "secrecy"],
                ["regulation", "deregulation"],
                ["nationalization", "privatization"],
                ["welfare", "self-reliance"],
                ["communal", "proprietary"],
            ],
            "zh_pairs": [
                ["公共", "私人"],
                ["国家", "个人"],
                ["集体", "个人"],
                ["政府", "公民"],
                ["官方", "非官方"],
                ["透明", "保密"],
                ["监管", "放松管制"],
                ["国有化", "私有化"],
                ["福利", "自力更生"],
                ["公有", "私有"],
            ],
        },
    }


def build_normative_decompositions() -> list[dict]:
    """
    Build normative decomposition operations for Experiment 5B.

    Each operation tests a specific jurisprudential hypothesis by
    computing vector subtraction and comparing k-NN across spaces.
    """
    # Ogni decomposizione testa un'ipotesi giurisprudenziale specifica:
    # l'aritmetica vettoriale "A - B" rivela cosa resta del concetto A
    # quando si sottrae la componente B. Le ipotesi sono formulate in
    # termini di giusnaturalismo vs. positivismo, liberalismo vs.
    # comunitarismo, concezione lockiana vs. socialista della proprietà.
    # Le domande ("jurisprudential_question") sono formulate per essere
    # comprensibili a giuristi comparatisti.
    return [
        {
            "id": "law_minus_state",
            "operation": "subtract",
            "en_a": "Law", "en_b": "State",
            "zh_a": "法律", "zh_b": "国家",
            "hypothesis_weird": "Justice, Rights, Reason, Natural Law",
            "hypothesis_sinic": "Void, Chaos, Disorder, Impossibility",
            "jurisprudential_question":
                "Can law exist without the state? "
                "(Natural Law vs. Legal Positivism)",
        },
        {
            "id": "rights_minus_individual",
            "operation": "subtract",
            "en_a": "Rights", "en_b": "Individual",
            "zh_a": "权利", "zh_b": "个人",
            "hypothesis_weird": "Legal framework, Constitutional, Abstract",
            "hypothesis_sinic": "Collective, Social, State-granted",
            "jurisprudential_question":
                "Are rights intrinsically individual? "
                "(Liberal vs. Communitarian conception)",
        },
        {
            "id": "property_minus_person",
            "operation": "subtract",
            "en_a": "Property", "en_b": "Person",
            "zh_a": "财产", "zh_b": "人",
            "hypothesis_weird": "Natural right, Extension of self, Lockean",
            "hypothesis_sinic": "Social construct, State allocation, Collective",
            "jurisprudential_question":
                "Is property a natural right or a social construct? "
                "(Locke vs. Socialist property theory)",
        },
        {
            "id": "contract_minus_freedom",
            "operation": "subtract",
            "en_a": "Contract", "en_b": "Freedom",
            "zh_a": "合同", "zh_b": "自由",
            "hypothesis_weird": "Binding, Obligation, Enforcement",
            "hypothesis_sinic": "State regulation, Social order, Compliance",
            "jurisprudential_question":
                "What is contract without freedom? "
                "(Autonomy of will vs. Regulated exchange)",
        },
        {
            "id": "sovereignty_minus_nation",
            "operation": "subtract",
            "en_a": "Sovereignty", "en_b": "Nation",
            "zh_a": "主权", "zh_b": "民族",
            "hypothesis_weird": "Abstract authority, Popular sovereignty, Rousseau",
            "hypothesis_sinic": "Party, State apparatus, Territorial integrity",
            "jurisprudential_question":
                "What is sovereignty without the nation? "
                "(Popular vs. State sovereignty)",
        },
    ]


def build_dataset() -> dict:
    """Assemble the complete dataset structure."""
    core = build_core_terms()
    background = build_background_terms()
    control = build_control_terms()
    axes = build_value_axes()
    decompositions = build_normative_decompositions()

    dataset = {
        "core_terms": core,
        "background_terms": background,
        "control_terms": control,
        "value_axes": axes,
        "normative_decompositions": decompositions,
        "metadata": {
            "primary_source": "HK DOJ Bilingual Legal Glossary (2026-02-05)",
            "primary_source_url": "https://www.glossary.doj.gov.hk/",
            "n_core_terms": len(core),
            "n_background_terms": len(background),
            "n_control_terms": len(control),
            "n_total_terms": len(core) + len(background) + len(control),
            "n_value_axes": len(axes),
            "n_decompositions": len(decompositions),
            "chinese_script": "simplified",
            "conversion": "Traditional→Simplified via OpenCC (t2s)",
            "version": "3.0",
        },
    }

    return dataset


def main():
    """Build and save the structured dataset."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = get_project_root()
    output_path = root / "data" / "processed" / "legal_terms.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building structured dataset...")
    dataset = build_dataset()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("Dataset saved to: %s", output_path)
    logger.info("  Core terms: %d", len(dataset["core_terms"]))
    logger.info("  Background terms: %d", len(dataset["background_terms"]))
    logger.info("  Control terms: %d", len(dataset["control_terms"]))
    logger.info("  Total terms: %d", dataset["metadata"]["n_total_terms"])
    logger.info("  Value axes: %d", len(dataset["value_axes"]))
    logger.info("  Normative decompositions: %d", len(dataset["normative_decompositions"]))

    # Verifica duplicati
    all_ids = (
        [t["id"] for t in dataset["core_terms"]]
        + [t["id"] for t in dataset["background_terms"]]
        + [t["id"] for t in dataset["control_terms"]]
    )
    dupes = [id_ for id_ in all_ids if all_ids.count(id_) > 1]
    if dupes:
        logger.warning("DUPLICATE IDs found: %s", set(dupes))
    else:
        logger.info("  No duplicate IDs")

    # Conteggio fonti
    all_terms = dataset["core_terms"] + dataset["background_terms"] + dataset["control_terms"]
    sources = {}
    for t in all_terms:
        s = t.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    for s, n in sorted(sources.items()):
        logger.info("  Source '%s': %d terms", s, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
