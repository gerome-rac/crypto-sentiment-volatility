"""The rulebook: every word list, pattern and threshold the pipeline applies.

One module, so the rules are stated once and every platform uses the same
definitions. Each block names the decision that set it (MASTERPLAN_NOV15.md
section 7 and the three pipeline defect maps). Patterns are RE2 (DuckDB's
regexp engine): no look-arounds. They are emitted into SQL as dollar-quoted
strings, so backslashes need no escaping.

Text preparation, in this order, before any column is computed (step 2, the filter scripts):
  1. HTML codes decoded with Python's html.unescape (PB5, P18, PD1). EmTract later
     receives the ORIGINAL text.
  2. Links removed (PD4): a coin or context word inside a URL does not count;
     posts whose only coin word sits in a link are flagged coin_only_in_link.
  3. For tier-2/3 coin words only: cashtags masked, so the letters inside "$BTC"
     are not a second, tier-2 hit (PD5).
  4. For BTC and ETH only: fork names removed before matching (PJ6).
"""
from __future__ import annotations

import hashlib
import html
from typing import Dict, List, Optional, Sequence, Tuple

COINS: Tuple[str, ...] = ("BTC", "ETH", "BNB", "XRP", "DOGE")
TIERS: Tuple[str, ...] = ("t1", "t2", "t3")
RULES_VERSION = "rules-2026-09-26"

# ---------------------------------------------------------------------------
# Coin terms (planning/final_keyword_list.md section 2; unchanged since 18 Sep). Four departures from
# that list, decided 18 Sep: `doge` only in tier 3 (DOGE tier 2 is empty); `ether` in ETH tier 3; `merge`
# only as the annotation column TERM_MERGE (a common word, informative only around Sep 2022); `to the moon`
# assigns no coin (generic hype; `moon` is an ambiguous context word).
# t1/t2 count on their own, t3 only with crypto context (D12).
# ---------------------------------------------------------------------------
COIN_PATTERNS: Dict[Tuple[str, str], Optional[str]] = {
    ("BTC", "t1"): r"\$btc(\.x)?\b|#bitcoin\b|#btc\b|\bbitcoins?\b",
    ("BTC", "t2"): r"\bbtc\b|#xbt\b|\bxbt\b",
    ("BTC", "t3"): r"\bsatoshis?\b|\bsats\b|\borange coin\b",
    ("ETH", "t1"): r"\$eth(\.x)?\b|#ethereum\b|#eth\b|\bethereum\b",
    ("ETH", "t2"): r"\bvitalik\b",
    ("ETH", "t3"): r"\beth\b|\bether\b",
    ("BNB", "t1"): r"\$bnb(\.x)?\b|#bnb\b|#binancecoin\b|\bbinance ?coin\b",
    ("BNB", "t2"): r"\bbnb\b",
    ("BNB", "t3"): r"\bbinance\b",
    ("XRP", "t1"): r"\$xrp(\.x)?\b|#xrp\b|\bxrp\b",
    ("XRP", "t2"): r"\bripple ?labs\b|#ripple\b",
    ("XRP", "t3"): r"\bripple\b|\bxrparmy\b",
    ("DOGE", "t1"): r"\$doge(\.x)?\b|#dogecoin\b|#doge\b|\bdogecoins?\b",
    ("DOGE", "t2"): None,                       # deliberately empty tier (constant FALSE)
    ("DOGE", "t3"): r"\bdoge\b|\bshibe\b|\bmuch wow\b",
}

# Context lists of the inclusion pass, unchanged since 18 Sep. They decide only
# which rows enter the coarse tables (StockTwits, Twitter); the rules below use
# the split decided on 26 Sep.
INCLUSION_CTX_SPECIFIC = (
    r"\bcrypto\w*|\bblockchain\w*|\baltcoins?\b|\bhodl\w*|\bdefi\b"
    r"|\bcoinbase\b|\bbinance\b|\bsatoshis?\b|\bstaking\b"
    r"|\bmarket ?cap\b|\bbull ?run\b|\bbear ?market\b")
INCLUSION_CTX_AMBIGUOUS = (
    r"\btokens?\b|\bcoins?\b|\bwallets?\b|\bexchanges?\b|\bmining\b"
    r"|\bmoon\w*|\bpump\w*|\bdump\w*|\bath\b")

# Clearly crypto words (26 Sep 2026, Gérome): the context of the coin gate (D12)
# and the word route into the market measure, on every platform.
CTX_CRYPTO = (r"\bcrypto\w*|\bblockchain\w*|\baltcoins?\b|\bhodl\w*|\bdefi\b"
              r"|\bcoinbase\b|\bbinance\b|\bsatoshis?\b|\bstaking\b")
# The same list without a coin's own tier-3 words (D5/D12): only BTC overlaps.
CTX_CRYPTO_EXCL = {"BTC": (r"\bcrypto\w*|\bblockchain\w*|\baltcoins?\b|\bhodl\w*|\bdefi\b"
                           r"|\bcoinbase\b|\bbinance\b|\bstaking\b")}

# Ambiguous terms (26 Sep 2026): never enough on their own. Posts with two or
# more DIFFERENT terms get a separate flag (register A47); pump and dump are one
# term. Eleven term classes.
AMBIGUOUS_TERMS: Tuple[Tuple[str, str], ...] = (
    ("market_cap", r"\bmarket ?cap\b"), ("bull_run", r"\bbull ?run\b"), ("bear_market", r"\bbear ?market\b"),
    ("token", r"\btokens?\b"), ("coin", r"\bcoins?\b"), ("wallet", r"\bwallets?\b"),
    ("exchange", r"\bexchanges?\b"), ("mining", r"\bmining\b"), ("moon", r"\bmoon\w*"),
    ("pump_dump", r"\bpump\w*|\bdump\w*"), ("ath", r"\bath\b"),
)
AMBIGUOUS_FLAG_MIN_TERMS = 2

TERM_MERGE = r"\bmerge\w*"                     # annotation only (ETH Merge, Sep 2022)

# Fork names removed before matching BTC and ETH (PJ6, 23 Sep).
FORK_BTC = r"\bbitcoin ?(cash|sv|gold)\b|\bwrapped ?bitcoin\b"
FORK_ETH = r"\bethereum ?classic\b"

# BNB chain words (PJ4, 25 Sep): bare "bnb" stays in BNB; these rows are flagged (A39).
BSC_CONTEXT = r"\bbsc\b|\bsmart ?chain\b|\bbep-?20\b|\bpancake ?swap\b|\bbnb ?chain\b"

# Reddit r/btc (P15, 23/25 Sep): Bitcoin Cash posts and threads out, then BTC only by ticker.
RBTC_BITCOIN_CASH = r"\bbch\b|bitcoin ?cash|\$bch|\bbsv\b|bitcoin ?sv|\bbchn\b|\bbcha\b|\bxec\b"
RBTC_BTC_TICKER = r"\$btc(\.x)?\b|#btc\b|\bbtc\b|#xbt\b|\bxbt\b"
# "r/btc" itself is ignored before the ticker is looked for (23 Sep prototype, 43,378 rows).
RBTC_SELF = r"(^|[^a-z0-9_])/?r/btc\b"

# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
URL_RX = r"https?://[^\s]+"
HASHTAG_RX = r"#\w+"
MENTION_RX = r"@\w+|(^|[^\w/])/?u/[A-Za-z0-9_-]+"      # @name, and Reddit u/name (P18)
# Tickers (PG6): letters first (so money amounts never count), digits allowed,
# up to 20 characters, optional .X suffix; each ticker counted once per post.
CASHTAG_RX = r"\$[A-Za-z][A-Za-z0-9]{0,19}(\.[A-Za-z])?\b"
# StockTwits symbol entry (PD6): the same symbols as cash_crypto_any.
STOCKTWITS_CRYPTO_SYMBOL_RX = r"[A-Z0-9.\-]{1,20}\.X"

# ---------------------------------------------------------------------------
# Bot and spam pass (D21; every rule runs on all posts and writes its own flag)
# ---------------------------------------------------------------------------
SELF_ID_PHRASES = ("performed automatically", "i am a bot")      # P22, on own text
AUTOMODERATOR = "AutoModerator"
# Campaign rule (Chen & Subramanian 2018, arXiv 1804.05232; PG2, 25 Sep).
CAMPAIGN_MIN_ACCOUNTS = 20        # accounts posting one fingerprint on one UTC day
CAMPAIGN_WINDOW = 200             # an account's most recent posts up to that day
CAMPAIGN_ALPHA = 3                # a window fingerprint must occur in >= 3 group accounts' windows
CAMPAIGN_BETA = 0.6               # share of the window that must carry such fingerprints
# Template rule (Electronic Markets 2025, 35:64; PG2/P23, 25 Sep).
TEMPLATE_WINDOW = 200
TEMPLATE_RATIO = 0.7              # difflib SequenceMatcher ratio above this: same template
TEMPLATE_SHARE = 0.5              # more than this share of the window's pairs: bot that day
# Keyword rule (Kraaijeveld & De Smedt 2020 as adapted in Electronic Markets 2025; 26 Sep).
# As in the 26 Sep prototype whose counts were decided on (StockTwits 313, Reddit 107,
# Twitter 37,913 posts without the name criterion): the phrases as written, and the
# words with their -s/-ed/-ing forms.
KW_GIVEAWAY = r"give away|giving away"
KW_PUMP = r"\bpump(s|ed|ing)?\b"
KW_JOIN = r"\b(join|register)(s|ed|ing)?\b"
KW_MAX_HASHTAGS = 5               # more than 5 hashtags
KW_MAX_TICKERS = 5                # more than 5 crypto tickers
KW_MIN_CRITERIA = 2               # at least 2 criteria (StockTwits: 2 of 4, no user names)
# Invite links (Nizzoli et al. 2020): flagged, never excluded.
INVITE_RX = r"t\.me/|telegram\.me/|discord\.gg/|discord(app)?\.com/invite/"
# Fingerprint (D21): at least this many words, else no fingerprint.
FINGERPRINT_MIN_WORDS = 4


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def dq(pattern: str) -> str:
    """Dollar-quote a pattern for SQL."""
    if "$$" in pattern:
        raise ValueError(f"pattern contains the dollar-quote delimiter: {pattern!r}")
    return f"$${pattern}$$"


def match(expr: str, pattern: str, flags: str = "i") -> str:
    return f"regexp_matches({expr}, {dq(pattern)}, '{flags}')" if flags else f"regexp_matches({expr}, {dq(pattern)})"


def remove(expr: str, pattern: str, flags: str = "gi") -> str:
    return f"regexp_replace({expr}, {dq(pattern)}, ' ', '{flags}')"


def register_udfs(con) -> None:
    """py_html_unescape(VARCHAR): Python's html.unescape, vectorised over Arrow.

    Python's function is the reference; DuckDB's own html_unescape lives in the
    inet extension, which is downloaded on first use, so it is not relied on.
    """
    import duckdb
    import pyarrow as pa

    def _unescape(arr):
        vals = arr.to_pylist()
        return pa.array([None if v is None else (html.unescape(v) if "&" in v else v) for v in vals], type=pa.string())

    try:
        con.remove_function("py_html_unescape")
    except Exception:
        pass
    varchar = duckdb.sqltypes.VARCHAR if hasattr(duckdb, "sqltypes") else duckdb.typing.VARCHAR  # 1.5 / older
    con.create_function("py_html_unescape", _unescape, [varchar], varchar, type="arrow", null_handling="special")


def inclusion_predicate(raw_expr: str) -> str:
    """Rows kept by the coarse pass (StockTwits, Twitter): any coin term or either context list.

    Evaluated on the raw text, links included, exactly as since 18 Sep, so the
    coarse tables stay a superset of every rule below.
    """
    pats = [p for p in COIN_PATTERNS.values() if p is not None] + [INCLUSION_CTX_SPECIFIC, INCLUSION_CTX_AMBIGUOUS]
    return match(raw_expr, "|".join(f"(?:{p})" for p in pats))


# Prepared text, in three SELECT layers so each expensive expression runs once per row.
PREP_COLUMNS = ("_dec_full", "_dec_own", "_nl", "_links", "_nl_own", "_fp_text")


def layer1(full_expr: str, own_expr: Optional[str]) -> List[Tuple[str, str]]:
    """HTML-decoded texts. full = what coin and context evidence reads (Reddit: with
    quote lines; Twitter quotes: text plus quoted text); own = the author's own
    words (spam and structure). own_expr None: the same text."""
    cols = [("_dec_full", f"py_html_unescape(coalesce({full_expr}, ''))")]
    cols.append(("_dec_own", "_dec_full" if own_expr is None else f"py_html_unescape(coalesce({own_expr}, ''))"))
    return cols


def fingerprint_text(dec_own: str = "_dec_own") -> str:
    """D21 fingerprint text: lowercase, links and mentions removed, punctuation and
    emojis removed (letters of every alphabet, marks and digits kept), single spaces."""
    t = f"lower({dec_own})"
    t = remove(t, URL_RX, "g")
    t = remove(t, MENTION_RX, "g")
    t = rf"regexp_replace({t}, $$[^\p{{L}}\p{{M}}\p{{N}}\s]$$, '', 'g')"
    return rf"trim(regexp_replace({t}, $$\s+$$, ' ', 'g'))"


def layer2() -> List[Tuple[str, str]]:
    """Link-free texts (PD4), the links themselves, and the fingerprint text."""
    return [
        ("_nl", remove("_dec_full", URL_RX, "g")),
        ("_links", f"array_to_string(regexp_extract_all(_dec_full, {dq(URL_RX)}), ' ')"),
        ("_nl_own", remove("_dec_own", URL_RX, "g")),
        ("_fp_text", fingerprint_text("_dec_own")),
    ]


def masked(expr: str) -> str:
    """Cashtags masked, for tier-2/3 words only (PD5)."""
    return remove(expr, CASHTAG_RX, "g")


def _coin_hits(nl: str) -> List[Tuple[str, str]]:
    cols: List[Tuple[str, str]] = []
    base = {"BTC": remove(nl, FORK_BTC), "ETH": remove(nl, FORK_ETH)}     # PJ6
    for coin in COINS:
        t = base.get(coin, nl)
        for tier in TIERS:
            p = COIN_PATTERNS[(coin, tier)]
            name = f"hit_{coin.lower()}_{tier}"
            if p is None:
                cols.append((name, "FALSE"))
            else:
                cols.append((name, match(t if tier == "t1" else masked(t), p)))
    return cols


def evidence_columns(nl: str = "_nl", links: str = "_links") -> List[Tuple[str, str]]:
    """Coin and context evidence on decoded, link-free text (layer 3)."""
    cols = _coin_hits(nl)
    any_hit = " OR ".join(n for n, _ in cols)
    all_coin = "|".join(f"(?:{p})" for p in COIN_PATTERNS.values() if p is not None)
    cols.append(("coin_only_in_link", f"({match(links, all_coin)} AND NOT ({any_hit}))"))
    cols.append(("ctx_crypto", match(nl, CTX_CRYPTO)))
    cols.append(("ctx_crypto_excl_btc", match(nl, CTX_CRYPTO_EXCL["BTC"])))
    cols.append(("n_ambiguous_terms", " + ".join(f"CAST({match(nl, p)} AS INTEGER)" for _, p in AMBIGUOUS_TERMS)))
    cols.append(("term_merge", match(nl, TERM_MERGE)))
    cols.append(("bsc_context", match(nl, BSC_CONTEXT)))
    return cols


def own_coin_hit(nl_own: str = "_nl_own") -> str:
    """Any coin term in the author's own words (Twitter own_text_hit), same preparation."""
    return "(" + " OR ".join(c for _, c in _coin_hits(nl_own) if c != "FALSE") + ")"


def structure_columns() -> List[Tuple[str, str]]:
    """Per-post structure on the author's own decoded words (layer 3)."""
    return [
        ("n_chars", "length(_dec_own)"),
        ("n_tokens", r"CASE WHEN trim(_dec_own) = '' THEN 0 ELSE len(str_split_regex(trim(_dec_own), $$\s+$$)) END"),
        ("n_urls", f"len(regexp_extract_all(_dec_own, {dq(URL_RX)}))"),
        ("n_hashtags", f"len(regexp_extract_all(_nl_own, {dq(HASHTAG_RX)}))"),
        ("n_mentions", f"len(regexp_extract_all(_nl_own, {dq(MENTION_RX)}))"),
        ("cashtags", f"list_distinct(list_transform(regexp_extract_all(_nl_own, {dq(CASHTAG_RX)}), "
                     r"x -> upper(regexp_replace(x, '^\$|\.[A-Za-z]$', '', 'g'))))"),
        # is_url_only (restored 28 Sep 2026 to the definition of the first pass of 18 Sep): the post has at
        # least one link, and its own words, lowercased, with links and @mentions removed, contain no letter
        # a-z and no digit (letters of other alphabets do not count, as in the first pass). No rule reads it.
        ("is_url_only", f"(trim(regexp_replace(regexp_replace(regexp_replace(lower(_dec_own), {dq(URL_RX)}, '', 'g'), "
                        f"{dq(MENTION_RX)}, '', 'g'), $$[^a-z0-9 ]$$, '', 'g')) = '' "
                        f"AND len(regexp_extract_all(_dec_own, {dq(URL_RX)})) > 0)"),
        ("fingerprint", f"CASE WHEN _fp_text <> '' AND len(string_split(_fp_text, ' ')) >= {FINGERPRINT_MIN_WORDS} "
                        f"THEN md5(_fp_text) END"),
    ]


def annotate_sql(source_select: str, full_expr: str, own_expr: Optional[str],
                 extra_layer3: Sequence[Tuple[str, str]] = ()) -> str:
    """Wrap a source SELECT (which must expose the text columns) in the three layers.

    Returns a SELECT with the source columns, the evidence columns, the structure
    columns and any extra layer-3 columns; the helper columns are dropped.
    """
    l1 = ", ".join(f"{e} AS {n}" for n, e in layer1(full_expr, own_expr))
    l2 = ", ".join(f"{e} AS {n}" for n, e in layer2())
    l3 = ", ".join(f"{e} AS {n}" for n, e in list(evidence_columns()) + list(structure_columns()) + list(extra_layer3))
    return (f"SELECT * EXCLUDE ({', '.join(PREP_COLUMNS)}) FROM (\n"
            f"  SELECT *, {l3} FROM (\n"
            f"    SELECT *, {l2} FROM (\n"
            f"      SELECT *, {l1} FROM ({source_select}) _src) _l1) _l2) _l3")


def evidence_names() -> List[str]:
    return [n for n, _ in evidence_columns()]


def structure_names() -> List[str]:
    return [n for n, _ in structure_columns()]


def name_has_bot_sql(col: str) -> str:
    """SQL twin of name_has_bot (Twitter screen names); NULL when there is no name."""
    return f"CASE WHEN {col} IS NULL OR {col} = '' THEN NULL ELSE contains(lower({col}), 'bot') END"


# ---------------------------------------------------------------------------
# User names (never stored; only these flags)
# ---------------------------------------------------------------------------

def name_bot_token(name: Optional[str]) -> Optional[bool]:
    """'bot' as a separate token of the name (ends in _bot or -bot, or bot between separators); arXiv 2601.14423."""
    if not name:
        return None
    toks = [t for t in "".join(ch if ch.isalnum() else " " for ch in name.lower()).split() if t]
    return "bot" in toks


def name_has_bot(name: Optional[str]) -> Optional[bool]:
    """'bot' anywhere in the user name: criterion of the keyword rule (Electronic Markets 2025)."""
    if not name:
        return None
    return "bot" in name.lower()


# ---------------------------------------------------------------------------
# Assignment (step 4): the coin gate and the market rule
# ---------------------------------------------------------------------------

def coin_gate(coin: str, symbol_col: Optional[str] = None) -> str:
    """D12: t1 or t2 (or the platform symbol) counts; t3 only with a crypto word other than the coin's own t3 words; BNB t1/t2 only."""
    c = coin.lower()
    parts = [f"hit_{c}_t1", f"hit_{c}_t2"]
    if symbol_col:
        parts.append(symbol_col)
    if coin != "BNB":
        ctx = "ctx_crypto_excl_btc" if coin == "BTC" else "ctx_crypto"
        parts.append(f"(hit_{c}_t3 AND {ctx})")
    return "(" + " OR ".join(parts) + ")"


def market_rule(any_coin: str, symbol_any: Optional[str] = None) -> str:
    """Market measure (26 Sep 2026, Gérome; StockTwits and Twitter): the post counts for no coin and carries
    another crypto .X symbol (StockTwits) or at least one clearly crypto word. Generic market words and
    everyday words never count on their own."""
    route = "ctx_crypto" if symbol_any is None else f"({symbol_any} OR ctx_crypto)"
    return f"(NOT ({any_coin}) AND {route})"


def ambiguous_extra(any_series: str) -> str:
    """Several-ambiguous-words flag (26 Sep; robustness A47): two or more different ambiguous terms,
    on a post that is in no coin series and not in the market measure."""
    return f"(n_ambiguous_terms >= {AMBIGUOUS_FLAG_MIN_TERMS} AND NOT ({any_series}))"


def keyword_criteria(dec_own: str, n_hashtags: str, n_crypto_tickers: str, name_flag: Optional[str]) -> List[Tuple[str, str]]:
    nl = remove(dec_own, URL_RX, "g")
    cols = [
        ("kw_giveaway", match(nl, KW_GIVEAWAY)),
        ("kw_pump_join", f"({match(nl, KW_PUMP)} AND {match(nl, KW_JOIN)})"),
        ("kw_hashtags", f"({n_hashtags} > {KW_MAX_HASHTAGS})"),
        ("kw_tickers", f"({n_crypto_tickers} > {KW_MAX_TICKERS})"),
    ]
    if name_flag is not None:
        cols.append(("kw_name_bot", f"coalesce({name_flag}, FALSE)"))
    return cols


# ---------------------------------------------------------------------------
# Fingerprints: each step records a digest of exactly the rules it applies, so
# changing a bot threshold does not mark the coarse tables as stale.
# ---------------------------------------------------------------------------

SECTIONS = {
    "filter": ["COINS", "TIERS", "COIN_PATTERNS", "INCLUSION_CTX_SPECIFIC", "INCLUSION_CTX_AMBIGUOUS", "CTX_CRYPTO",
           "CTX_CRYPTO_EXCL", "AMBIGUOUS_TERMS", "AMBIGUOUS_FLAG_MIN_TERMS", "TERM_MERGE", "FORK_BTC", "FORK_ETH",
           "BSC_CONTEXT", "RBTC_BITCOIN_CASH", "RBTC_BTC_TICKER", "RBTC_SELF", "URL_RX", "HASHTAG_RX", "MENTION_RX",
           "CASHTAG_RX", "STOCKTWITS_CRYPTO_SYMBOL_RX", "FINGERPRINT_MIN_WORDS", "PREP_COLUMNS",
           "dq", "match", "remove", "register_udfs", "inclusion_predicate", "layer1", "fingerprint_text", "layer2",
           "masked", "_coin_hits", "evidence_columns", "own_coin_hit", "structure_columns", "annotate_sql",
           "evidence_names", "structure_names", "name_has_bot_sql", "name_bot_token", "name_has_bot"],
    "bot": ["SELF_ID_PHRASES", "AUTOMODERATOR", "CAMPAIGN_MIN_ACCOUNTS", "CAMPAIGN_WINDOW", "CAMPAIGN_ALPHA",
           "CAMPAIGN_BETA", "TEMPLATE_WINDOW", "TEMPLATE_RATIO", "TEMPLATE_SHARE", "KW_GIVEAWAY", "KW_PUMP", "KW_JOIN",
           "KW_MAX_HASHTAGS", "KW_MAX_TICKERS", "KW_MIN_CRITERIA", "INVITE_RX", "URL_RX", "dq", "match", "remove",
           "keyword_criteria"],
    "series": ["COINS", "coin_gate", "market_rule", "ambiguous_extra", "AMBIGUOUS_FLAG_MIN_TERMS", "COIN_PATTERNS",
           "URL_RX", "dq", "match"],
}


def section_sha(section: str) -> str:
    """sha256 over the source or value of every rule a step uses."""
    import inspect
    g = globals()
    parts = [RULES_VERSION]
    for name in SECTIONS[section]:
        obj = g[name]
        parts.append(f"{name}=" + (inspect.getsource(obj) if callable(obj) else repr(obj)))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def rules_sha() -> str:
    """Digest of the whole rulebook (all sections)."""
    return hashlib.sha256("|".join(section_sha(s) for s in sorted(SECTIONS)).encode()).hexdigest()[:16]
