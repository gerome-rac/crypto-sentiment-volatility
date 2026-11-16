"""End-to-end test: small hand-made inputs for all three platforms through steps 02-05.

Run from code/:  python tests/test_pipeline_sample.py
Everything is written to a temporary folder (THESIS_PIPELINE_OUT, THESIS_SCRATCH),
never to code/data. Each case below is one decided rule, checked at the end of the
chain in the assignment tables and the daily counts. No real post is used.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parents[1]          # code/
UTC = dt.timezone.utc
DAY = dt.datetime(2021, 5, 10, 12, 0, tzinfo=UTC)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def reddit_extract(out: Path) -> None:
    ext = out / "reddit" / "extract"
    ext.mkdir(parents=True)
    cols = ["id", "ts", "kind", "subreddit", "text", "score", "author_hash", "is_automoderator", "name_bot_token",
            "name_has_bot", "author_deleted", "account_age_days", "distinguished", "stickied", "removed_by_category",
            "is_robot_indexable", "edited", "retrieved_at", "link_id", "parent_id", "is_dead", "dead_reason",
            "own_text", "quoted_text", "has_quote_lines", "coin_label"]

    def row(i, sub, kind, text, author, part, link=None, automod=False, tok=False, has=False, dead=False, hour=0):
        post = kind == "posts"
        return [f"r{i}", DAY + dt.timedelta(hours=hour, minutes=i), kind, sub, "" if dead else text, 1, author, automod,
                tok, has, False, 10.0, None, False, None, True if post else None, None, DAY, None if post else link,
                None if post else link, dead, "deleted" if dead else None, "" if dead else text, None, False, part]

    parts = {
        "BTC": [row(1, "Bitcoin", "posts", "bitcoin is king", "a1", "BTC"),
                row(2, "btc", "posts", "bitcoin cash is the real bitcoin", "a2", "BTC"),
                row(3, "btc", "comments", "btc to 100k", "a3", "BTC", link="t3_r2"),          # in a BCH thread
                row(4, "btc", "posts", "btc to 100k soon", "a4", "BTC"),                      # counts (ticker)
                row(5, "btc", "posts", "bitcoin only talk here", "a5", "BTC"),                # no ticker: out
                row(6, "Bitcoin", "comments", "", "a6", "BTC", link="t3_r1", dead=True)],
        "BNB": [row(10, "binance", "posts", "bnb to the moon", "b1", "BNB"),
                row(11, "binance", "posts", "is ethereum better on binance", "b2", "BNB"),     # market
                row(12, "binance", "posts", "my withdrawal is stuck help", "b3", "BNB")],      # support post
        "MARKET": [row(20, "CryptoCurrency", "posts", "altcoins are pumping", "m1", "MARKET"),
                   row(21, "CryptoCurrency", "comments", "I am a bot, and this action was performed automatically.",
                       "m2", "MARKET", link="t3_r20", automod=True),
                   row(22, "CryptoCurrency", "comments", "nice", "m3", "MARKET", link="t3_r20", tok=True, has=True)],
    }
    types = {"id": pa.string(), "ts": pa.timestamp("us", tz="UTC"), "kind": pa.string(), "subreddit": pa.string(),
             "text": pa.string(), "score": pa.int64(), "author_hash": pa.string(), "is_automoderator": pa.bool_(),
             "name_bot_token": pa.bool_(), "name_has_bot": pa.bool_(), "author_deleted": pa.bool_(),
             "account_age_days": pa.float64(), "distinguished": pa.string(), "stickied": pa.bool_(),
             "removed_by_category": pa.string(), "is_robot_indexable": pa.bool_(), "edited": pa.timestamp("us", tz="UTC"),
             "retrieved_at": pa.timestamp("us", tz="UTC"), "link_id": pa.string(), "parent_id": pa.string(),
             "is_dead": pa.bool_(), "dead_reason": pa.string(), "own_text": pa.string(), "quoted_text": pa.string(),
             "has_quote_lines": pa.bool_(), "coin_label": pa.string()}
    schema = pa.schema([(c, types[c]) for c in cols])      # the s0 schema, so empty columns keep their type
    for part, rows in parts.items():
        t = pa.table({c: [r[k] for r in rows] for k, c in enumerate(cols)}, schema=schema)
        p = ext / f"reddit_{part}.parquet"
        pq.write_table(t, p)
        (ext / f"reddit_{part}_stats.json").write_text(json.dumps({"output_md5": md5(p), "checks": {"rows_out": len(rows)}}))


def stocktwits_stage_a(root: Path) -> None:
    rows = []
    mid = [1000]

    def add(user, text, syms, minutes=0, reply_to=None, day=DAY):
        mid[0] += 1
        rows.append([mid[0], (day + dt.timedelta(minutes=minutes)).replace(tzinfo=None), user, text, None, reply_to,
                     reply_to, syms, len(text or ""), "[]"])
        return mid[0]

    btc_post = add(1, "$BTC.X looks strong", "['BTC.X']")
    add(2, "ETH Zurich is a university", "[]")                       # t3 without context: kept, no coin
    add(3, "$SOL.X to the moon", "['SOL.X']")                        # market via symbol
    add(4, "crypto is up today", "[]")                               # market via word
    add(5, "$AAPL earnings beat", "['AAPL']")                        # not admitted
    add(6, "$MARA mining bitcoin", "['MARA']")                        # BTC, equity_only (A38)
    add(7, "bnb on binance smart chain", "[]")                        # BNB, bsc_context (A39)
    add(8, "check https://coindesk.com/bitcoin-news", "[]")           # link only (A40)
    add(9, "it&#39;s #btc time", "[]")                                # HTML decoded, BTC
    add(10, "lol that coin is going down", "[]", reply_to=btc_post)   # coin-less reply (A42)
    add(11, "pump and dump to the moon", "[]")                        # A47 flag
    add(12, "giving away free $BTC.X join our pump group #a #b #c #d #e #f", "['BTC.X']")  # keyword rule
    add(13, "join t.me/cryptopump for $ETH.X signals", "['ETH.X']")   # invite link, flag only
    for u in range(100, 125):                                         # campaign: 25 accounts, one text
        add(u, "Big news for crypto holders today check this out", "[]", minutes=u)
    for k in range(30):                                               # template bot over three days
        add(200, f"Buy $DOGE.X now it will moon soon friends {k % 3}", "['DOGE.X']", minutes=k,
            day=DAY + dt.timedelta(days=k // 10))
    cols = ["message_id", "created_at", "user_id", "message_body", "sentiment", "parent_message_id",
            "in_reply_to_message_id", "symbol_list", "length", "important_words"]
    t = pa.table({c: [r[k] for r in rows] for k, c in enumerate(cols)})
    d = root / "month=2021-05"
    d.mkdir(parents=True)
    pq.write_table(t, d / "data_0.parquet")


def twitter_stage_a(root: Path) -> None:
    d = root / "2021" / "date=2021-05-10"
    d.mkdir(parents=True)
    ms = int(DAY.timestamp() * 1000)

    def tw(tid, typ, text, user, name, quoted=None, qid=None, rep=None, add_ms=0):
        return {"day": DAY.date(), "day_coverage_pct": 99.0, "tweet_id": tid, "created_at": "x", "timestamp_ms": ms + add_ms,
                "tweet_type": typ, "lang": "en", "text": text, "quoted_text": quoted, "quoted_id": qid, "reply_to_id": rep,
                "reply_to_user": "someone" if rep else None, "user_id": user, "user_screen_name": name,
                "user_followers": 10, "user_verified": False, "retweet_count": 0, "favorite_count": 0}

    tweets = [tw("1", "original", "bitcoin to the moon", "u1", "alice"),
              tw("1", "original", "bitcoin to the moon", "u1", "alice", add_ms=5),   # second capture
              tw("2", "quote", "look at this", "u2", "bob", quoted="$ETH breaking out", qid="99"),
              tw("3", "original", "dogecoin much wow", "u3", "pricebot"),
              tw("4", "original", "coins and tokens everywhere", "u4", "carol"),
              tw("5", "original", "ethereum rising", "u5", "dave")]
    pq.write_table(pa.Table.from_pylist(tweets), d / "tweets.parquet")

    def rt(rid, orig, text, who, add_ms):
        return {"day": DAY.date(), "day_coverage_pct": 99.0, "retweet_id": rid, "retweet_created_at": "x",
                "retweet_timestamp_ms": ms + add_ms, "retweeter_id": who, "retweeter_screen_name": "rt_" + who,
                "retweeter_followers": 1, "orig_tweet_id": orig, "orig_created_at": "Sun May 09 10:00:00 +0000 2021",
                "orig_author_id": "u9" if orig == "50" else "u5", "orig_author_screen_name": "erin",
                "orig_author_followers": 5, "lang": "en", "orig_text": text}

    rts = [rt("r1", "5", "ethereum rising fast today", "v1", 100),        # longer copy wins
           rt("r2", "50", "xrp lawsuit news", "v2", 200),                  # retweet-only, first seen here
           rt("r2", "50", "xrp lawsuit news", "v2", 200),                  # same retweet captured twice
           rt("r3", "50", "xrp lawsuit news", "v3", 300),
           rt("r5", "60", "doge news today B", "v5", 150),                 # tie: equally long copies, this one
           rt("r4", "60", "doge news today A", "v4", 400)]                 # captured first; A sorts first
    pq.write_table(pa.Table.from_pylist(rts), d / "retweets.parquet")


def twitter_ledgers(root: Path) -> None:
    """One capture ledger of step 1, as the coverage part of 02_filter_twitter.py reads it."""
    root.mkdir(parents=True)
    led = {"status": "ok", "stem": "twitter_stream_2021_05_10",
           "days_from_data": {"2021-05-10": {"minutes": 1440, "first": "0000", "last": "2359", "pct": 100.0}},
           "days": {"2021-05-10": {"pct": 100.0}}}
    (root / "twitter_stream_2021_05_10.stats.json").write_text(json.dumps(led))


def run(args, env) -> str:
    res = subprocess.run([sys.executable] + args, cwd=str(CODE), env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed:\n{res.stdout[-2000:]}\n{res.stderr[-3000:]}")
    return res.stdout + res.stderr


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out, scratch, st_src, tw_src = tmp / "out", tmp / "scratch", tmp / "st", tmp / "tw"
        ledgers = tmp / "ledgers"
        twitter_ledgers(ledgers)
        reddit_extract(out)
        stocktwits_stage_a(st_src)
        twitter_stage_a(tw_src)
        env = dict(os.environ, THESIS_PIPELINE_OUT=str(out), THESIS_SCRATCH=str(scratch), THESIS_TWITTER_LEDGERS=str(ledgers))
        steps = [["pipeline/02_filter_reddit.py", "--partitions", "BTC", "BNB", "MARKET"],
                 ["pipeline/02_filter_stocktwits.py", "--src", str(st_src)],
                 ["pipeline/02_filter_twitter.py", "coarse", "--src-mode", "local", "--src-root", str(tw_src)],
                 ["pipeline/02_filter_twitter.py", "units"],
                 ["pipeline/03_bot_filter.py", "stocktwits", "--workers", "2"],
                 ["pipeline/03_bot_filter.py", "reddit", "--workers", "2"],
                 ["pipeline/03_bot_filter.py", "twitter", "--workers", "2"],
                 ["pipeline/04_series_reddit.py"], ["pipeline/04_series_stocktwits.py"],
                 ["pipeline/04_series_twitter.py"], ["pipeline/05_daily_counts.py"]]
        for s in steps:
            run(s, env)
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")

        def one(platform, post_id, col):
            p = out / platform / "panel" / "assignment.parquet"
            r = con.execute(f"SELECT {col} FROM read_parquet('{p}') WHERE CAST(post_id AS VARCHAR) = ?", [str(post_id)]).fetchone()
            return None if r is None else r[0]

        fails = []

        def expect(cond, msg):
            if not cond:
                fails.append(msg)

        # Reddit (P15, P22)
        expect(one("reddit", "r1", "in_btc"), "r/Bitcoin post counts for BTC")
        expect(not one("reddit", "r2", "in_btc") and one("reddit", "r2", "rbtc_bitcoin_cash"), "r/btc Bitcoin Cash post out")
        expect(not one("reddit", "r3", "in_btc"), "comment in a Bitcoin Cash thread out")
        expect(one("reddit", "r4", "in_btc"), "r/btc post naming the BTC ticker counts")
        expect(not one("reddit", "r5", "in_btc") and one("reddit", "r5", "rbtc_no_ticker"), "r/btc post without ticker out")
        expect(one("reddit", "r6", "in_btc") is None, "dead rows are not in the assignment")
        expect(one("reddit", "r10", "in_bnb"), "r/binance post naming BNB counts for BNB")
        expect(one("reddit", "r11", "in_market") and not one("reddit", "r11", "in_eth"), "r/binance other coin: market only")
        expect(one("reddit", "r12", "rbinance_support") and not one("reddit", "r12", "in_market"), "r/binance support post")
        expect(one("reddit", "r21", "bot_automod") and one("reddit", "r21", "bot_self_id") and one("reddit", "r21", "bot_any"),
               "AutoModerator and self-ID flagged")
        expect(one("reddit", "r22", "bot_name"), "bot-token name flagged")
        # StockTwits (D12, market rule, A38-A42, A47, D21)
        expect(one("stocktwits", 1001, "in_btc"), "$BTC.X counts for BTC")
        expect(not one("stocktwits", 1002, "in_eth") and not one("stocktwits", 1002, "in_market"), "ETH Zurich: no coin, no market")
        expect(one("stocktwits", 1003, "in_market"), "other crypto symbol: market")
        expect(one("stocktwits", 1004, "in_market"), "crypto word: market")
        expect(one("stocktwits", 1005, "in_btc") is None, "pure equity post not admitted")
        expect(one("stocktwits", 1006, "in_btc") and one("stocktwits", 1006, "equity_only"), "equity-only BTC post flagged")
        expect(one("stocktwits", 1007, "in_bnb") and one("stocktwits", 1007, "bsc_context"), "bnb with chain words")
        expect(not one("stocktwits", 1008, "in_btc") and one("stocktwits", 1008, "link_btc"), "coin word only in a link")
        expect(one("stocktwits", 1009, "in_btc"), "HTML-decoded #btc")
        expect(one("stocktwits", 1010, "reply_parent_btc"), "coin-less reply gets its parent's coin (A42)")
        expect(one("stocktwits", 1011, "ambiguous_extra"), "several ambiguous words flagged (A47)")
        expect(one("stocktwits", 1012, "bot_keyword"), "keyword rule")
        expect(one("stocktwits", 1013, "invite_link") and not one("stocktwits", 1013, "bot_any"), "invite link: flag only")
        expect(all(one("stocktwits", 1014 + k, "bot_campaign") for k in range(25)), "campaign of 25 accounts flagged")
        expect(one("stocktwits", 1039 + 29, "bot_template"), "template bot flagged")
        # Twitter (step 2 units, D13)
        expect(one("twitter", "1", "in_btc"), "tweet counts for BTC once (two captures)")
        expect(one("twitter", "2", "in_eth"), "quote counts via the quoted text")
        expect(one("twitter", "50", "in_xrp") and not one("twitter", "50", "authored"), "retweet-only original counts (A46 flag)")
        expect(one("twitter", "4", "ambiguous_extra"), "A47 on Twitter")
        u = out / "twitter" / "units" / "units.parquet"
        n_units, ev, from_rt = con.execute(f"SELECT count(*), sum(n_as_retweet), count_if(text_from_retweet) FROM read_parquet('{u}')").fetchone()
        expect(n_units == 7, f"seven units (got {n_units})")
        expect(ev == 5, f"five distinct retweet events (got {ev})")
        expect(from_rt == 1, "the longer retweet copy wins")
        tie = con.execute(f"SELECT text FROM read_parquet('{u}') WHERE unit_id = '60'").fetchone()[0]
        expect(tie == "doge news today B", f"equally long copies: the earliest captured wins (got {tie!r})")
        cov = out / "twitter" / "units" / "twitter_coverage_utc.parquet"
        n_cov = con.execute(f"SELECT count(*) FROM read_parquet('{cov}')").fetchone()[0]
        expect(n_cov == 1280, f"coverage table on the full calendar (got {n_cov})")
        day50 = con.execute(f"SELECT CAST(panel_ts AS DATE) FROM read_parquet('{u}') WHERE unit_id = '50'").fetchone()[0]
        expect(str(day50) == "2021-05-10", f"retweet-only dated at its first sampled retweet (got {day50})")
        # resume (runner.is_current): a second run with unchanged inputs, code and rules repeats nothing
        again = [run(["pipeline/02_filter_stocktwits.py", "--src", str(st_src)], env),
                 run(["pipeline/03_bot_filter.py", "reddit", "--workers", "2"], env),
                 run(["pipeline/04_series_twitter.py"], env)]
        expect(all("current, skipped" in a for a in again), "second runs are skipped as current")
        dc = out / "panel" / "daily_counts.parquet"
        n_days = con.execute(f"SELECT count(DISTINCT day) FROM read_parquet('{dc}')").fetchone()[0]
        expect(n_days == 1280, "daily counts on the full calendar")
        n_dead = con.execute(f"SELECT n_dead FROM read_parquet('{dc}') WHERE platform = 'reddit' AND series = 'BTC' "
                             f"AND day = DATE '2021-05-10'").fetchone()[0]
        expect(n_dead == 1, f"A24: the dead r/Bitcoin comment is counted in n_dead (got {n_dead})")
        n_other = con.execute(f"SELECT count_if(n_dead IS NOT NULL) FROM read_parquet('{dc}') WHERE platform <> 'reddit'").fetchone()[0]
        expect(n_other == 0, "n_dead only on Reddit")
        if fails:
            print("FAIL:\n  " + "\n  ".join(fails))
            return 1
        print("end-to-end sample: all checks passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
