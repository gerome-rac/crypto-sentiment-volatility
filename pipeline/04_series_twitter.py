#!/usr/bin/env python3
"""
04_series_twitter.py -- step 4 for Twitter: which series every tweet counts for.

Input    data/pipeline/twitter/units/units.parquet (step 2) and
         data/pipeline/twitter/bot/post_flags.parquet (step 3).
Output   data/pipeline/twitter/panel/assignment.parquet: one row per tweet (unit), with
         in_btc ... in_doge, in_market, the bot flags and the robustness columns.

The rules
---------
The same coin gate and market rule as StockTwits, without symbols, on the unit's text plus
its parent text (the quoted tweet):
    coin c: a tier-1 or tier-2 term of c, or a tier-3 term with a clearly crypto word other
    than c's own tier-3 words; BNB by tiers 1 and 2 only. A tweet counts for every coin it
    names.
    market: a tweet that counts for no coin and carries at least one clearly crypto word.
A tweet's day is the UTC day of panel_ts: its writing time for captured tweets, its first
sampled retweet for originals known only through retweets.

Robustness columns (the register's arms)
----------------------------------------
    n_coins                 A28  number of coins the tweet counts for
    authored                A46  FALSE for retweet-only originals (the arm drops them)
    authored_before_window       a retweet-only original written before 1 Aug 2019
    n_as_retweet                 retweet events of the tweet (A12 uses the daily table)
    bnb_strict              A29  a tier-1 BNB term
    bsc_context             A39  Binance Smart Chain context
    coin_only_in_link, link_<coin>  A40  coin words only inside links
    own_text_hit                 a coin term in the author's own words (not only in the quote)
    ambiguous_extra         A47  two or more different ambiguous terms, outside coins and market
    text_key                A44  md5 of text and parent text, for once-per-day counting
    bot_*, invite_link           the flags of step 3 (A43 keeps bot_any tweets)

Checks: one row per tweet; every tweet has its bot flags; no tweet is in a coin series and
in the market.

Decisions: MASTERPLAN D12, D13 (retweet-only dating), D21; market rule 26 Sep 2026.

Usage (from code/)
------------------
    python pipeline/04_series_twitter.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, runner, series  # noqa: E402
from pipeline import rules as R  # noqa: E402
from pipeline.series import link_evidence, links_of  # noqa: E402
from pipeline.runner import log  # noqa: E402

STEP = "04_series_twitter"
CODE_SHA = runner.code_fingerprint(__file__, series.__file__)
RULES_SHA = R.section_sha("series") + "-" + R.section_sha("filter")


def twitter_sql() -> str:
    src = config.out_path("twitter", "units", "units.parquet")
    flags = config.out_path("twitter", "bot", "post_flags.parquet")
    gates = {c: R.coin_gate(c) for c in R.COINS}
    coin_cols = ", ".join(f"{gates[c]} AS in_{c.lower()}" for c in R.COINS)
    any_coin = " OR ".join(f"in_{c.lower()}" for c in R.COINS)
    n_coins = " + ".join(f"CAST(in_{c.lower()} AS INTEGER)" for c in R.COINS)
    link_cols = ", ".join(f"(coin_only_in_link AND {link_evidence(c, '_links', 'ctx_crypto')}) AS link_{c.lower()}"
                          for c in R.COINS)
    return f"""
    WITH a0 AS (
      SELECT u.*, {coin_cols},
             CASE WHEN coin_only_in_link THEN {links_of("text || ' ' || coalesce(parent_text, '')")} ELSE '' END AS _links
      FROM read_parquet('{src}') u),
    a AS (SELECT a0.* EXCLUDE (_links), {R.market_rule(any_coin)} AS in_market, ({n_coins}) AS n_coins, {link_cols}
          FROM a0)
    SELECT a.unit_id AS post_id, a.panel_ts AS ts, CAST(a.panel_ts AS DATE) AS day, a.author_id AS account, a.role,
           a.in_btc, a.in_eth, a.in_bnb, a.in_xrp, a.in_doge, a.in_market, a.n_coins,
           a.role <> 'retweet_only' AS authored, a.authored_before_window, a.n_as_retweet,
           a.hit_bnb_t1 AS bnb_strict, a.bsc_context, a.coin_only_in_link,
           {', '.join('a.link_' + c.lower() for c in R.COINS)},
           a.own_text_hit,
           {R.ambiguous_extra('a.in_market OR ' + ' OR '.join('a.in_' + c.lower() for c in R.COINS)).replace('n_ambiguous_terms', 'a.n_ambiguous_terms')} AS ambiguous_extra,
           a.text_key,
           f.bot_any, f.bot_keyword, f.bot_campaign, f.bot_template, f.invite_link
    FROM a LEFT JOIN read_parquet('{flags}') f ON f.post_id = a.unit_id"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    log(f"{STEP}: code {CODE_SHA}, rules {RULES_SHA}")
    return series.assign("twitter", twitter_sql(), STEP, CODE_SHA, RULES_SHA)


if __name__ == "__main__":
    raise SystemExit(main())
