"""Text normalisation for business names and addresses (country-agnostic core with
US / India / France aware dictionaries and a generic fallback for any other country).

Every record is reduced to a handful of canonical string fields that the blocking
and feature stages consume:

name fields
  nf  : full canonical name tokens (transliterated, accent-free, leet fixed, legal forms canonical)
  nc  : core name tokens (nf minus legal forms / honorifics / stop words / country qualifiers)
  na  : alias part (text before an a/k/a | dba | fka | formerly ... marker), core tokens
  nw  : web / domain stem if the name carries a domain or '| www.x.com'
  nl  : canonical legal-form tokens present in the name
  nflag: bit flags (1 alias marker, 2 domain-like, 4 indic script, 8 web suffix, 16 handle '@')
address fields
  at  : canonical address tokens (state names -> 's_xx' tokens, street types canonical, ...)
  an  : numeric tokens in order of appearance (leading zeros stripped)
  ast : canonical state token ('' if none)
"""
import json
import unicodedata

import regex as re
from unidecode import unidecode

try:
    from indic_transliteration import sanscript
    _SCRIPTS = [
        (0x0900, 0x097F, sanscript.DEVANAGARI), (0x0980, 0x09FF, sanscript.BENGALI),
        (0x0A00, 0x0A7F, sanscript.GURMUKHI), (0x0A80, 0x0AFF, sanscript.GUJARATI),
        (0x0B00, 0x0B7F, sanscript.ORIYA), (0x0B80, 0x0BFF, sanscript.TAMIL),
        (0x0C00, 0x0C7F, sanscript.TELUGU), (0x0C80, 0x0CFF, sanscript.KANNADA),
        (0x0D00, 0x0D7F, sanscript.MALAYALAM),
    ]
except Exception:  # pragma: no cover
    sanscript = None
    _SCRIPTS = []

ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
WORD = re.compile(r"[\p{L}\p{M}\p{N}]+")
INDIC = re.compile(r"[ऀ-෿]")
ALNUM = re.compile(r"[a-z0-9]+")
DIGALPHA = re.compile(r"\d+|[a-z]+")

# ----------------------------------------------------------------------------- names
ALIAS_RE = re.compile(
    r"\s(?:a\s?/\s?k\s?/\s?a|aka|d\s?/\s?b\s?/\s?a|dba|f\s?/\s?k\s?/\s?a|fka|formerly\s+known\s+as|formerly:?|"
    r"doing\s+business\s+as|n[ée]e|trading\s+as|t\s?/\s?a|also\s+known\s+as|previously)\s", re.I)
ID_RE = re.compile(r"[\(\[]\s*id\s*[:#]?\s*\d+\s*[\)\]]", re.I)
DOMAIN_FULL = re.compile(r"^\s*@?\s*(?:https?://)?(?:www\.)?([\p{L}\p{N}][\p{L}\p{N}.-]*?)\.(?:com|c0m|co\.in|in|net|org|fr|biz|info|co|io|us)\s*$", re.I)
WEB_IN = re.compile(r"(?:https?://)?(?:www\.)([\p{L}\p{N}-]+)\.[a-z0-9.]+", re.I)
DOTABBR = re.compile(r"(?<![\p{L}\p{N}])((?:\p{L}\.){1,}\p{L})\.?(?![\p{L}])")
SLASHABBR = re.compile(r"(?<![\p{L}])(m|c|s|w)\s?/\s?(s|o)(?![\p{L}])", re.I)

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
ORD_SUFFIX = ("st", "nd", "rd", "th")

NAME_CANON = {
    "incorporated": "inc", "incorporation": "inc", "lnc": "inc", "incorp": "inc",
    "corporation": "corp", "corpn": "corp", "corporations": "corp",
    "company": "co", "companies": "cos",
    "limited": "ltd", "ltda": "ltd", "limted": "ltd",
    "private": "pvt", "pte": "pvt", "pvte": "pvt",
    "centre": "center", "cntr": "center", "ctr": "center",
    "intl": "international", "int": "international",
    "mfg": "manufacturing", "bros": "brothers", "assoc": "associates", "assocs": "associates",
    "svcs": "services", "svc": "service", "mgmt": "management", "dept": "department",
    "et": "and", "und": "and",
    "st": "saint", "ste": "sainte",
    "etablissements": "ets", "etablissement": "ets",
}
LEGAL = {
    "inc", "corp", "co", "cos", "ltd", "pvt", "llc", "llp", "lp", "lllp", "pllc", "plc", "pc", "pa", "psc",
    "apc", "opc", "sarl", "sas", "sasu", "eurl", "sci", "sa", "ei", "snc", "scp", "selarl", "scm", "gie",
    "sca", "eirl", "scop", "ets", "gmbh", "ag", "bv", "nv", "srl", "spa", "ab", "oy", "as",
}
HONOR = {"the", "dr", "sri", "shri", "smt", "mr", "mrs", "ms", "m/s", "messrs", "mssrs", "le", "la", "les"}
STOP = {"and", "of", "the", "de", "du", "des", "la", "le", "les", "d", "l", "a", "an", "for", "in", "at", "on", "y", "en", "au", "aux"}
# Country words are kept in the core name as one shared token: an appended country word
# ("Acacias Club France SA" at another house number) marks a *different* (clone) entity,
# and a shared token lets what is learned on one country transfer to unseen ones.
COUNTRY_WORDS = {"india", "france", "usa", "america", "bharat"}
COUNTRY_TOKEN = "zzcountry"
QUAL_LEGACY = {"india", "france", "usa", "us", "america", "bharat"}  # country_token=False: dropped from core

# ----------------------------------------------------------------------------- addresses
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "washington dc": "dc", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr", "guam": "gu", "virgin islands": "vi",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "orissa": "od", "odisha": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": None, "nct of delhi": "dl",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk", "ladakh": "la", "puducherry": "py", "pondicherry": "py",
    "chandigarh": "ch", "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
IN_CODES = {"ap", "ar", "as", "br", "cg", "ct", "ga", "gj", "hr", "hp", "jh", "ka", "kl", "mp", "mh", "mn", "ml",
            "mz", "nl", "od", "or", "pb", "rj", "sk", "tn", "tg", "ts", "tr", "up", "uk", "ut", "wb", "dl", "jk",
            "la", "py", "ch", "an", "dn", "dd", "ld"}
IN_CODE_ALIAS = {"ts": "tg", "or": "od", "ct": "cg", "ut": "uk"}
FR_REGIONS = {
    "hauts-de-france": "hdf", "hauts de france": "hdf", "nord": "hdf", "pas-de-calais": "hdf", "pas de calais": "hdf",
    "nouvelle-aquitaine": "naq", "nouvelle aquitaine": "naq", "gironde": "naq",
    "pays de la loire": "pdl", "pays-de-la-loire": "pdl", "loire-atlantique": "pdl", "loire atlantique": "pdl",
    "ile-de-france": "idf", "ile de france": "idf", "paris": None,
    "auvergne-rhone-alpes": "ara", "provence-alpes-cote d'azur": "pac", "occitanie": "occ", "grand est": "ges",
    "bretagne": "bre", "normandie": "nor", "bourgogne-franche-comte": "bfc", "centre-val de loire": "cvl",
    "corse": "cor",
}

ADDR_CANON = {
    # street types (English)
    "street": "st", "str": "st", "strt": "st", "saint": "st", "avenue": "ave", "av": "ave", "aven": "ave",
    "avn": "ave", "avnue": "ave", "road": "rd", "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct",
    "circle": "cir", "circ": "cir", "place": "pl", "boulevard": "blvd", "boul": "blvd", "bd": "blvd",
    "boulv": "blvd", "parkway": "pkwy", "pkway": "pkwy", "pky": "pkwy", "highway": "hwy", "hiway": "hwy",
    "terrace": "ter", "terr": "ter", "trail": "trl", "square": "sq", "point": "pt", "route": "rte",
    "expressway": "expy", "freeway": "fwy", "crossing": "xing", "heights": "hts", "turnpike": "tpke",
    "mount": "mt", "mountain": "mtn", "plaza": "plz", "junction": "jct", "extension": "ext", "cove": "cv",
    "creek": "crk", "ridge": "rdg", "valley": "vly", "village": "vlg", "alley": "aly", "center": "ctr",
    "centre": "ctr", "fort": "ft", "harbor": "hbr", "lake": "lk", "island": "is", "station": "sta",
    "apartment": "apt", "apartments": "apt", "suite": "ste", "building": "bldg", "bldng": "bldg", "floor": "fl",
    "flr": "fl", "room": "rm", "department": "dept", "township": "twp", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # India
    "nr": "near", "opp": "opposite", "behind": "behind", "bh": "behind", "marg": "marg", "colony": "colony",
    "clny": "colony", "sec": "sector", "sect": "sector", "dist": "district", "distt": "district", "tq": "taluk",
    "tal": "taluk", "taluka": "taluk", "mandal": "mandal", "vill": "village", "vpo": "village", "po": "po",
    "ps": "ps", "gf": "ground", "ff": "first", "sf": "second", "blk": "block", "bldg": "bldg",
    "chs": "chs", "soc": "society", "stn": "sta", "hsg": "housing", "indl": "industrial", "ind": "industrial",
    "estt": "estate", "est": "estate", "extn": "ext", "cross": "cross", "crs": "cross", "main": "main",
    "layout": "layout", "lyt": "layout", "nagar": "nagar", "ngr": "nagar", "bombay": "mumbai",
    "bengaluru": "bangalore", "gurugram": "gurgaon", "calcutta": "kolkata", "madras": "chennai", "poona": "pune",
    # France
    "rue": "rue", "r": "rue", "allee": "allee", "all": "allee", "impasse": "impasse", "imp": "impasse",
    "chemin": "chemin", "ch": "chemin", "che": "chemin", "chem": "chemin", "rte": "rte", "residence": "residence",
    "res": "residence", "resid": "residence", "quai": "quai", "qu": "quai", "cours": "cours", "crs.": "cours",
    "faubourg": "fbg", "fg": "fbg", "fbg": "fbg", "passage": "pass", "pas": "pass", "lotissement": "lot",
    "lot": "lot", "hameau": "hameau", "ham": "hameau", "cite": "cite", "sente": "sente", "sentier": "sentier",
    "promenade": "prom", "prom": "prom", "esplanade": "espl", "espl": "espl", "domaine": "dom", "dom": "dom",
    "zone": "zone", "za": "za", "zi": "zi", "zac": "zac", "rpt": "rondpoint", "rond": "rondpoint",
    "ave.": "ave", "bis": "bis", "ter.": "ter", "grande": "grande", "gde": "grande", "gd": "grand",
    "docteur": "dr", "doct": "dr", "general": "gen", "gal": "gen", "marechal": "mal", "mal": "mal",
    "president": "pres", "pdt": "pres", "professeur": "prof", "pr": "prof",
    "ste.": "ste", "sainte": "ste",
}
ORD_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
    "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12", "thirteenth": "13",
    "fourteenth": "14", "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
    "nineteenth": "19", "twentieth": "20", "ist": "1", "premier": "1", "premiere": "1",
}
ADDR_DROP_BASE = {"no", "nos", "hno", "null", "none", "nil", "unit", "number", "num", "nbr", "and"}
# generic locality qualifiers the generator adds/drops ('Richmond City', 'City Of Madison',
# 'Mumbai Suburban', 'Kolkata Region', 'Greater Mumbai', 'Shalersville Twp')
LOCALITY_DROP = {"city", "county", "cnty", "of", "town", "twp", "township", "region", "suburban", "greater", "cdp"}
NULL_COMP = {"null", "<null>", "n/a", "none", "nil", "na", "-", "--", "nan", "unknown"}
DEG_RE = re.compile(r"(?i)\bn\s*[°º]\s*|[°º]")
NUM_MARK = re.compile(r"(?:\b(?:h\s*\.?\s*no|door\s*no|flat\s*no|plot\s*no|shop\s*no|house\s*no|no)\s*[.:#-]*\s*(?=\d))|#+|\bn\s*[°º]\s*|\bnº", re.I)


def _script_translit(tok):
    if sanscript is None:
        return unidecode(tok)
    cp = ord(tok[0])
    for lo, hi, sc in _SCRIPTS:
        if lo <= cp <= hi:
            try:
                return unidecode(sanscript.transliterate(tok, sc, sanscript.ITRANS)).lower()
            except Exception:
                return unidecode(tok).lower()
    return unidecode(tok).lower()


def _state_lookup(comp):
    """comp: lower-case ascii component -> canonical state token or None."""
    c = comp.strip(" .-")
    c2 = re.sub(r"[-\s]*\d+$", "", c).strip(" .-")  # 'west bengal-0741222'
    for cand in (c, c2):
        if not cand:
            continue
        if cand in US_STATES and US_STATES[cand]:
            return "s_us_" + US_STATES[cand]
        if cand in IN_STATES and IN_STATES[cand]:
            return "s_in_" + IN_STATES[cand]
        if cand in FR_REGIONS and FR_REGIONS[cand]:
            return "s_fr_" + FR_REGIONS[cand]
    return None


US_CODES = set(US_STATES.values())


class Normalizer:
    def __init__(self, translit_path=None, drop_locality=True, country_token=True):
        """Representations used by the pipeline:
        blocking: drop_locality=False, country_token=False; stage-0: True, False;
        matcher features: True, True (defaults)."""
        self.addr_drop = ADDR_DROP_BASE | (LOCALITY_DROP if drop_locality else set())
        self.country_token = country_token
        self.name_map, self.addr_comp, self.addr_tok = {}, {}, {}
        if translit_path:
            d = json.load(open(translit_path, encoding="utf-8"))
            self.name_map, self.addr_comp, self.addr_tok = d["name"], d["addr_comp"], d["addr_tok"]
        self._cache_tr = {}

    # ------------------------------------------------------------------ helpers
    def _tr_token(self, tok, table):
        """Transliterate one Indic token to ascii (dictionary first, script fallback)."""
        r = table.get(tok)
        if r is not None:
            return r
        key = (id(table), tok)
        r = self._cache_tr.get(key)
        if r is None:
            r = _script_translit(tok)
            if len(self._cache_tr) < 500000:
                self._cache_tr[key] = r
        return r

    def _deindic(self, s, table):
        if not INDIC.search(s):
            return s, False
        return WORD.sub(lambda m: self._tr_token(m.group(0), table) if INDIC.search(m.group(0)) else m.group(0), s), True

    @staticmethod
    def _leet(tok):
        # fix leetspeak only in tokens that are mostly letters with a few substituted digits
        nd = sum(c.isdigit() for c in tok)
        if nd == 0 or nd == len(tok):
            return tok
        if len(tok) - nd >= 2 and nd <= 2 and not (tok[:-2].isdigit() and tok[-2:] in ORD_SUFFIX):
            return tok.translate(LEET)
        return tok

    # ------------------------------------------------------------------ names
    def name(self, raw):
        s = unicodedata.normalize("NFKC", raw).translate(ZW)
        flags = 0
        web = ""
        if "|" in s:
            parts = s.split("|")
            s = parts[0]
            for p in parts[1:]:
                m = WEB_IN.search(p) or DOMAIN_FULL.match(p)
                if m:
                    web = m.group(1)
                    flags |= 8
                elif p.strip() and not s.strip():
                    s = p
        s = ID_RE.sub(" ", s)
        alias = ""
        m = ALIAS_RE.search(" " + s + " ")
        if m:
            st = max(m.start() - 1, 0)
            en = max(m.end() - 1, 0)
            pre, post = s[:st], s[en:]
            if post.strip():
                alias, s = pre, post
                flags |= 1
        s, ind = self._deindic(s, self.name_map)
        if ind:
            flags |= 4
        if alias:
            alias, _ = self._deindic(alias, self.name_map)
        md = DOMAIN_FULL.match(s)
        if md:
            web = web or md.group(1)
            s = md.group(1)
            flags |= 2
        if "@" in s:
            flags |= 16
        s = s.replace("&", " and ").replace("+", " and ")
        s = SLASHABBR.sub(lambda mm: mm.group(1) + mm.group(2), s)
        s = DOTABBR.sub(lambda mm: mm.group(1).replace(".", ""), s)
        s = unidecode(s).lower()
        s = re.sub(r"(^|\s)@+", " ", s)
        toks = [self._leet(t) for t in re.findall(r"[a-z0-9@$]+", s)]
        toks = [t.replace("@", "a").replace("$", "s") if any(c.isalpha() for c in t) else t for t in toks]
        toks = [NAME_CANON.get(t, t) for t in toks if t and t not in ("@", "$")]
        if self.country_token:
            toks = [COUNTRY_TOKEN if t in COUNTRY_WORDS else t for t in toks]
        legal = sorted({t for t in toks if t in LEGAL})
        qual = set() if self.country_token else QUAL_LEGACY
        core = [t for t in toks if t not in LEGAL and t not in HONOR and t not in STOP and t not in qual]
        if not core:
            core = [t for t in toks if t not in LEGAL] or toks
        acore = ""
        if alias:
            a = unidecode(alias).lower()
            at = [NAME_CANON.get(self._leet(t), self._leet(t)) for t in re.findall(r"[a-z0-9]+", a)]
            acore = " ".join(t for t in at if t not in LEGAL and t not in HONOR and t not in STOP)
        if web:
            web = re.sub(r"[^a-z0-9]", "", unidecode(web).lower().translate(LEET))
        return " ".join(toks), " ".join(core), acore, web, " ".join(legal), flags

    # ------------------------------------------------------------------ addresses
    def address(self, raw, country=""):
        s = unicodedata.normalize("NFKC", raw).translate(ZW)
        comps_out, toks, nums = [], [], []
        state = ""
        ind = False
        for comp in s.split(","):
            comp = comp.strip()
            if not comp:
                continue
            if INDIC.search(comp):
                ind = True
                rep = self.addr_comp.get(comp)
                if rep is None:
                    comp, _ = self._deindic(comp, self.addr_tok)
                else:
                    comp = rep
            comp = DEG_RE.sub(" ", comp)
            c = unidecode(comp).lower().strip()
            if c in NULL_COMP or c.strip("<>[]() .-") in NULL_COMP:
                continue
            st = _state_lookup(c)
            if st is None and len(c) == 2 and c.isalpha():
                # bare 2-letter state code, namespaced by the record's country
                if country == "US" and c in US_CODES:
                    st = "s_us_" + c
                elif country == "India" and c in IN_CODES:
                    st = "s_in_" + IN_CODE_ALIAS.get(c, c)
                elif country not in ("US", "India") and (c in US_CODES or c in IN_CODES):
                    st = "s_x_" + c
            if st is not None:
                state = state or st
                toks.append(st)
                continue
            c = NUM_MARK.sub(" ", c)
            ctoks = []
            for w in re.findall(r"[a-z0-9]+", c):
                parts = DIGALPHA.findall(w)
                if len(parts) > 1 and parts[0].isdigit() and parts[1] in ORD_SUFFIX and len(parts) == 2:
                    parts = [parts[0]]
                for p in parts:
                    if p.isdigit():
                        p = p.lstrip("0") or "0"
                        nums.append(p)
                        ctoks.append(p)
                    else:
                        p = ORD_WORDS.get(p, p)
                        if p.isdigit():
                            nums.append(p)
                            ctoks.append(p)
                            continue
                        p = ADDR_CANON.get(p, p)
                        if p in self.addr_drop:
                            continue
                        ctoks.append(p)
            if ctoks:
                toks.extend(ctoks)
                comps_out.append(" ".join(ctoks))
        return " ".join(toks), " ".join(nums), state, "|".join(comps_out), ind


def resolve_state_codes(at_tokens, country):
    """Map bare 2-letter code tokens 's_c_xx' to the country specific namespace."""
    if "s_c_" not in at_tokens:
        return at_tokens
    pre = {"US": "s_us_", "India": "s_in_"}.get(country)
    out = []
    for t in at_tokens.split(" "):
        if t.startswith("s_c_"):
            code = t[4:]
            if pre == "s_us_" and code in US_CODES:
                t = pre + code
            elif pre == "s_in_" and code in IN_CODES:
                t = pre + IN_CODE_ALIAS.get(code, code)
            else:
                t = "s_x_" + code
        out.append(t)
    return " ".join(out)
