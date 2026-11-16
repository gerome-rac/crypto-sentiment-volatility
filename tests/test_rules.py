"""Unit tests for pipeline/rules.py: every decided rule on small, readable cases.

Run from code/:  python tests/test_rules.py   (or: pytest tests/test_rules.py). Carries the trap cases of
the earlier tests/test_coarse.py plus one case per
decision made since 23 Sep 2026.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402

from pipeline import rules as R  # noqa: E402


def _con():
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    R.register_udfs(con)
    return con


def annotate(texts, own=None):
    """Annotate a list of texts (full text; own text = same unless given)."""
    con = _con()
    rows = []
    for i, t in enumerate(texts):
        o = t if own is None else own[i]
        rows.append(f"({i}, {R.dq(t) if t else chr(39) * 2}, {R.dq(o) if o else chr(39) * 2})")
    con.execute(f"CREATE TABLE s AS SELECT * FROM (VALUES {', '.join(rows)}) v(i, t, o)")
    df = con.execute(R.annotate_sql("SELECT i, t, o FROM s", "t", "o",
                                    extra_layer3=[("own_hit", R.own_coin_hit())]) + " ORDER BY i").df()
    return [r for _, r in df.iterrows()]


def gate(r, coin, symbol=False):
    """Evaluate the D12 gate in Python on an annotated row (mirror of rules.coin_gate)."""
    c = coin.lower()
    ok = bool(r[f"hit_{c}_t1"]) or bool(r[f"hit_{c}_t2"]) or symbol
    if coin != "BNB":
        ctx = r["ctx_crypto_excl_btc"] if coin == "BTC" else r["ctx_crypto"]
        ok = ok or (bool(r[f"hit_{c}_t3"]) and bool(ctx))
    return ok


def test_inclusion_unchanged():
    con = _con()
    cases = [("$BTC.X looking strong", True), ("ETH Zurich is a university", True), ("dodge the ball", False),
             ("he was crippled", False), ("the path to a bath", False), ("together whether method", False),
             ("earnings beat on $AAPL", False), ("blockchain settlement", True), ("to the moon", True),
             ("ripple labs sued again", True), ("https://bitcoin.org/faq", True), ("just had breakfast", False)]
    for text, want in cases:
        got = con.execute(f"SELECT {R.inclusion_predicate(R.dq(text))}").fetchone()[0]
        assert got == want, (text, got, want)


def test_tiers_cashtag_mask_pd5():
    a, b, c = annotate(["$BTC.X to the moon", "btc looks weak today", "$BTC.X and btc miners"])
    assert a.hit_btc_t1 and not a.hit_btc_t2, "a cashtag alone is tier 1 only"
    assert b.hit_btc_t2 and not b.hit_btc_t1
    assert c.hit_btc_t1 and c.hit_btc_t2


def test_gate_d12_context():
    rows = annotate(["ETH Zurich is a university", "ether and defi", "satoshi", "sats on coinbase",
                     "market cap and bull run on eth", "binance is down", "much wow doge"])
    zurich, ether, sat, sats_cb, generic, binance, doge = rows
    assert not gate(zurich, "ETH") and gate(ether, "ETH")
    assert not gate(sat, "BTC"), "satoshi needs another crypto word"
    assert gate(sats_cb, "BTC")
    assert not gate(generic, "ETH"), "market cap / bull run no longer count as context (26 Sep)"
    assert not gate(binance, "BNB") and binance.ctx_crypto, "bare binance: market, not BNB"
    assert not gate(doge, "DOGE"), "tier-3 doge without a crypto word stays out"


def test_forks_pj6():
    cash, real, wrapped, classic, mixed = annotate([
        "bitcoin cash to the moon", "bitcoin cash is the real bitcoin", "wrapped bitcoin on ethereum",
        "ethereum classic hard fork", "ethereum and dogecoin, not bitcoin sv"])
    assert not cash.hit_btc_t1 and real.hit_btc_t1
    assert not wrapped.hit_btc_t1 and wrapped.hit_eth_t1
    assert not classic.hit_eth_t1
    assert mixed.hit_eth_t1 and mixed.hit_doge_t1 and not mixed.hit_btc_t1, "only BTC is lost"


def test_links_pd4_and_html_pb5():
    link, html_row, apos = annotate(["Check https://coindesk.com/bitcoin-price today",
                                     "I &lt;3 bitcoin &amp; eth", "it&#39;s #btc"])
    assert not link.hit_btc_t1 and link.coin_only_in_link
    assert html_row.hit_btc_t1
    assert apos.n_hashtags == 1, "the escaped apostrophe is not a hashtag (PD1)"


def test_word_traps():
    rows = annotate(["dodge the ball", "he was crippled", "the path to a bath", "together whether"])
    for r in rows:
        assert not any(r[f"hit_{c.lower()}_{t}"] for c in R.COINS for t in R.TIERS), r.t
        assert r.n_ambiguous_terms == 0, r.t


def test_context_flags():
    bsc, amb2, pd, merge = annotate(["binance smart chain bnb", "market cap and bull run", "pump and dump",
                                     "the merge is coming"])
    assert bsc.hit_bnb_t2 and bsc.bsc_context
    assert amb2.n_ambiguous_terms == 2 and not amb2.ctx_crypto
    assert pd.n_ambiguous_terms == 1, "pump and dump are one term"
    assert merge.term_merge


def test_structure_pg6_p18():
    tick, ment, url, short, f1, f2 = annotate([
        "$BTC.X $eth $100 $5k $SHIB2 $BTC", "@alice and u/bob and /u/carol", "https://x.com/abc",
        "Buy $BTC now!!! https://x.co @bob", "Hello, World! This is great.", "hello world this is great"])
    assert sorted(tick.cashtags) == ["BTC", "ETH", "SHIB2"], "each ticker once, digits allowed, no amounts"
    assert ment.n_mentions == 3
    assert url.is_url_only
    assert short.fingerprint is None or short.fingerprint != short.fingerprint, "fewer than 4 words: none"
    assert f1.fingerprint == f2.fingerprint and len(f1.fingerprint) == 32


def test_own_text_separate():
    r, = annotate(["> quoted bitcoin line\nmy own words"], own=["my own words"])
    assert r.hit_btc_t1 and not r.own_hit, "coin evidence reads the full text, own_hit only the own text"


def test_names():
    cases = {"AutoModerator": (False, False), "tipbot": (False, True), "Tip_Bot": (True, True), "bot": (True, True),
             "robotics_guy": (False, True), "price-bot": (True, True), "bot123": (False, True), "alice": (False, False)}
    con = _con()
    for name, (tok, has) in cases.items():
        assert R.name_bot_token(name) == tok, name
        assert R.name_has_bot(name) == has, name
        assert con.execute(f"SELECT {R.name_has_bot_sql(R.dq(name))}").fetchone()[0] == has, name
    assert R.name_bot_token(None) is None and R.name_has_bot("") is None


def test_keyword_rule():
    con = _con()
    texts = ["giving away 1 BTC, join our pump", "we give away prizes", "#a #b #c #d #e #f", "joined the pumping group"]
    sel = ", ".join(f"{e} AS {n}" for n, e in R.keyword_criteria("t", "0", "0", None))
    for t, want in zip(texts, [(True, True), (True, False), (False, False), (False, True)]):
        got = con.execute(f"SELECT {sel} FROM (SELECT {R.dq(t)} AS t)").fetchone()
        assert (got[0], got[1]) == want, (t, got)
    assert con.execute(f"SELECT {R.keyword_criteria('t', '6', '6', None)[2][1]} FROM (SELECT 'x' AS t)").fetchone()[0]


def test_invite_and_self_id():
    con = _con()
    for t, want in [("join t.me/pumpgroup", True), ("discord.gg/abc", True), ("discordapp.com/invite/x", True),
                    ("telegram is an app", False)]:
        assert con.execute(f"SELECT {R.match(R.dq(t), R.INVITE_RX)}").fetchone()[0] == want, t
    assert all(p == p.lower() for p in R.SELF_ID_PHRASES)


def test_rbtc_rules():
    con = _con()
    for t, bch in [("bitcoin cash is fast", True), ("BCH to the moon", True), ("bitcoinsv", True), ("xec", True),
                   ("bitcoin is king", False)]:
        assert con.execute(f"SELECT {R.match(R.dq(t), R.RBTC_BITCOIN_CASH)}").fetchone()[0] == bch, t
    for t, tick in [("see r/btc for more", False), ("btc to 100k", True), ("$BTC pumps", True), ("xbt futures", True),
                    ("bitcoin only", False)]:
        expr = R.match(R.remove(R.dq(t.lower()), R.RBTC_SELF, "g"), R.RBTC_BTC_TICKER)
        assert con.execute(f"SELECT {expr}").fetchone()[0] == tick, t


def test_stocktwits_symbol_rx():
    con = _con()
    for s, want in [("BTC.X", True), ("BABYSPACEFLOKI.X", True), ("AAPL", False), ("BTC", False), ("SHIB-USD.X", True)]:
        assert con.execute(f"SELECT regexp_full_match({R.dq(s)}, {R.dq(R.STOCKTWITS_CRYPTO_SYMBOL_RX)})").fetchone()[0] == want, s


def test_gate_sql_matches_python():
    rows = annotate(["ether and defi", "satoshi", "bnb chain", "binance coin", "ripple effect", "xrp army"])
    con = _con()
    for r in rows:
        for coin in R.COINS:
            cols = {k: bool(r[k]) for k in r.index if k.startswith("hit_") or k.startswith("ctx_")}
            sel = " , ".join(f"{str(v).upper()} AS {k}" for k, v in cols.items())
            got = con.execute(f"SELECT {R.coin_gate(coin)} FROM (SELECT {sel})").fetchone()[0]
            assert got == gate(r, coin), (r.t, coin)


def test_market_rule_and_ambiguous_flag():
    """26 Sep: market = no coin and (another crypto .X symbol or a clearly crypto word); A47 flag."""
    rows = annotate(["$SOL.X to the moon", "market cap and bull run", "crypto is up today", "bitcoin up",
                     "$AAPL and crypto", "pump and dump to the moon", "coins and tokens everywhere"])
    symbol_any = [True, False, False, False, False, False, False]
    con = _con()
    got_market, got_amb = [], []
    for r, sym in zip(rows, symbol_any):
        cols = {k: bool(r[k]) for k in r.index if k.startswith("hit_") or k.startswith("ctx_")}
        sel = ", ".join(f"{str(v).upper()} AS {k}" for k, v in cols.items())
        any_coin = " OR ".join(R.coin_gate(c) for c in R.COINS)
        m = con.execute(f"SELECT {R.market_rule(any_coin, str(sym).upper())} FROM (SELECT {sel})").fetchone()[0]
        amb = con.execute(f"SELECT {R.ambiguous_extra(str(m).upper())} FROM (SELECT {int(r.n_ambiguous_terms)} AS n_ambiguous_terms)"
                          ).fetchone()[0]
        got_market.append(m)
        got_amb.append(amb and not any(gate(r, c) for c in R.COINS))
    assert got_market == [True, False, True, False, True, False, False], got_market
    assert got_amb == [False, True, False, False, False, True, True], got_amb


def test_url_only_first_pass():
    """is_url_only as in the first pass (restored 28 Sep 2026): a link, and besides links and @mentions no
    letter a-z and no digit; letters of other alphabets do not count."""
    link, mention, word, cyrillic, no_link = annotate(
        ["https://x.co/a", "@bob https://x.co/a", "moon https://x.co/a", "\u041f\u0440\u0438\u0432\u0435\u0442 https://x.co/a", "@bob only"])
    assert link.is_url_only and mention.is_url_only, "a link alone, or with a mention, is url-only"
    assert not word.is_url_only, "a word besides the link is not"
    assert cyrillic.is_url_only, "letters outside a-z do not count (first-pass definition)"
    assert not no_link.is_url_only, "without a link a post is never url-only"


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
