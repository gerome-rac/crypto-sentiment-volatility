#!/usr/bin/env python3
"""
04_series_reddit.py -- step 4 for Reddit: which series every post counts for.

Input    data/pipeline/reddit/coarse/coin=*/data_0.parquet (step 2) and
         data/pipeline/reddit/bot/post_flags.parquet (step 3).
Output   data/pipeline/reddit/panel/assignment.parquet: one row per live post or comment,
         with in_btc ... in_doge, in_market, the bot flags and the robustness columns.

The rule: a Reddit post counts for its subreddit's coin
-------------------------------------------------------
Dead rows (removed or deleted) never count. Otherwise the group of step 1 decides:
    in_eth, in_xrp, in_doge  every post of the ETH, XRP and DOGE groups;
    in_btc   every post of the BTC group, except in r/btc: there, posts and comments
             about Bitcoin Cash (the post names it, or it sits in the thread of a post that
             names it) are out, and the rest counts only when it names a BTC ticker
             (r/btc is also the Bitcoin Cash community);
    in_bnb   posts of r/binance that name BNB (tiers 1 and 2 only; r/binance is mostly
             about the exchange);
    in_market  every post of the MARKET group, plus r/binance posts that do not name BNB
             but name another coin or carry a clearly crypto word.
The remaining r/binance posts (naming neither) are support posts: in no series, kept as a
robustness column. A post never counts for a coin of another subreddit.

Robustness columns (the register's arms)
----------------------------------------
    textgate_<coin>   A23  the coin gate of the other platforms applied to the text
    n_coins           A28  number of coins the text names
    bnb_strict        A29  BNB named by a tier-1 term
    bsc_context       A39  Binance Smart Chain context
    coin_only_in_link      the only coin word sits inside a link
    rbinance_support  A41  r/binance posts naming neither BNB nor crypto
    rbtc_bitcoin_cash, rbtc_no_ticker   the two r/btc exclusions, for checking
    ambiguous_multi        two or more different ambiguous terms
    edited            A45  the post was edited
    text_key          A44  md5 of the text, for once-per-day counting of exact copies
    bot_*, invite_link     the flags of step 3 (A43 keeps bot_any posts)

Checks: one row per post; every post has its bot flags; no post is in a coin series and in
the market.

Decisions: Reddit defect map P15 (r/btc, r/binance); MASTERPLAN D12 (coin gate), D21 (panel rules).

Usage (from code/)
------------------
    python pipeline/04_series_reddit.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, runner, series  # noqa: E402
from pipeline import rules as R  # noqa: E402
from pipeline.runner import log  # noqa: E402

STEP = "04_series_reddit"
CODE_SHA = runner.code_fingerprint(__file__, series.__file__)
RULES_SHA = R.section_sha("series") + "-" + R.section_sha("filter")


def reddit_sql() -> str:
    src = config.out_path("reddit", "coarse", "coin=*", "data_0.parquet")
    flags = config.out_path("reddit", "bot", "post_flags.parquet")
    gates = {c: R.coin_gate(c) for c in R.COINS}
    other_coin = " OR ".join(gates[c] for c in R.COINS if c != "BNB")
    textgate = ", ".join(f"{gates[c]} AS textgate_{c.lower()}" for c in R.COINS)
    n_text = " + ".join(f"CAST({gates[c]} AS INTEGER)" for c in R.COINS)
    return f"""
    WITH b AS (
      SELECT r.*, lower(r.subreddit) AS sub,
             CASE WHEN r.kind = 'posts' THEN 't3_' || r.id ELSE r.link_id END AS thread
      FROM read_parquet('{src}') r WHERE NOT r.is_dead),
    bch_threads AS (SELECT DISTINCT 't3_' || id AS thread FROM b WHERE sub = 'btc' AND kind = 'posts' AND bch_mention),
    x AS (
      SELECT b.*, (b.sub = 'btc' AND (b.bch_mention OR b.thread IN (SELECT thread FROM bch_threads))) AS rbtc_bitcoin_cash,
             {gates['BNB']} AS names_bnb,
             ({other_coin} OR ctx_crypto) AS other_or_ctx
      FROM b)
    SELECT x.id AS post_id, x.ts, CAST(x.ts AS DATE) AS day, x.author_hash AS account, x.coin_label, x.subreddit, x.kind,
           (x.coin_label = 'BTC' AND (x.sub <> 'btc' OR (NOT x.rbtc_bitcoin_cash AND x.btc_ticker))) AS in_btc,
           (x.coin_label = 'ETH') AS in_eth,
           (x.coin_label = 'BNB' AND x.names_bnb) AS in_bnb,
           (x.coin_label = 'XRP') AS in_xrp,
           (x.coin_label = 'DOGE') AS in_doge,
           (x.coin_label = 'MARKET' OR (x.coin_label = 'BNB' AND NOT x.names_bnb AND x.other_or_ctx)) AS in_market,
           (x.sub = 'btc' AND x.rbtc_bitcoin_cash) AS rbtc_bitcoin_cash,
           (x.sub = 'btc' AND NOT x.rbtc_bitcoin_cash AND NOT x.btc_ticker) AS rbtc_no_ticker,
           (x.coin_label = 'BNB' AND NOT x.names_bnb AND NOT x.other_or_ctx) AS rbinance_support,
           {textgate}, ({n_text}) AS n_coins,
           (x.hit_bnb_t1) AS bnb_strict, x.bsc_context, x.coin_only_in_link,
           x.n_ambiguous_terms >= {R.AMBIGUOUS_FLAG_MIN_TERMS} AS ambiguous_multi,
           x.edited IS NOT NULL AS edited, md5(x.text) AS text_key,
           f.bot_any, f.bot_automod, f.bot_self_id, f.bot_name, f.bot_keyword, f.bot_campaign, f.bot_template,
           f.invite_link
    FROM x LEFT JOIN read_parquet('{flags}') f ON f.post_id = x.id"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    log(f"{STEP}: code {CODE_SHA}, rules {RULES_SHA}")
    return series.assign("reddit", reddit_sql(), STEP, CODE_SHA, RULES_SHA)


if __name__ == "__main__":
    raise SystemExit(main())
