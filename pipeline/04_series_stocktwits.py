#!/usr/bin/env python3
"""
04_series_stocktwits.py -- step 4 for StockTwits: which series every message counts for.

Input    data/pipeline/stocktwits/coarse/month=*/data_0.parquet (step 2) and
         data/pipeline/stocktwits/bot/post_flags.parquet (step 3).
Output   data/pipeline/stocktwits/panel/assignment.parquet: one row per message, with
         in_btc ... in_doge, in_market, the bot flags and the robustness columns.

The rules
---------
Coin gate (rules.coin_gate), for each of the five coins:
    the message counts for coin c if the text holds a tier-1 or tier-2 term of c, or the
    message carries c's symbol (c.X; for BTC also the plain symbol BTC), or the text holds
    a tier-3 term of c together with a clearly crypto word other than c's own tier-3
    words. BNB counts by tier 1, tier 2 or its symbol only.
    A message counts for every coin it names.
Market (rules.market_rule):
    a message that counts for no coin enters the market series if it carries another
    crypto .X symbol or at least one clearly crypto word. Generic market words and
    everyday words never count on their own.

Robustness columns (the register's arms)
----------------------------------------
    n_coins             A28  number of coins the message counts for
    equity_only         A38  the message has symbols, none of them crypto
    bnb_strict          A29  $BNB.X or a tier-1 BNB term
    bsc_context         A39  Binance Smart Chain context
    link_<coin>         A40  the coin's words appear only inside links
    is_reply, reply_parent_<coin>  A42  a reply naming no coin, whose parent counts for the coin
    ambiguous_extra     A47  two or more different ambiguous terms, outside coins and market
    body_null                the message has no body
    text_key            A44  md5 of the body, for once-per-day counting of exact copies
    bot_*, invite_link       the flags of step 3 (A43 keeps bot_any messages)

Checks: one row per message; every message has its bot flags; no message is in a coin
series and in the market.

Decisions: MASTERPLAN D12 (coin gate), D21; StockTwits defect map PJ2, PJ3, PJ8; market
rule 26 Sep 2026.

Usage (from code/)
------------------
    python pipeline/04_series_stocktwits.py
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

STEP = "04_series_stocktwits"
CODE_SHA = runner.code_fingerprint(__file__, series.__file__)
RULES_SHA = R.section_sha("series") + "-" + R.section_sha("filter")


def stocktwits_sql() -> str:
    src = config.out_path("stocktwits", "coarse", "month=*", "data_0.parquet")
    flags = config.out_path("stocktwits", "bot", "post_flags.parquet")
    gates = {c: R.coin_gate(c, f"cash_{c.lower()}") for c in R.COINS}
    coin_cols = ", ".join(f"{gates[c]} AS in_{c.lower()}" for c in R.COINS)
    any_coin = " OR ".join(f"in_{c.lower()}" for c in R.COINS)
    n_coins = " + ".join(f"CAST(in_{c.lower()} AS INTEGER)" for c in R.COINS)
    link_cols = ", ".join(f"(coin_only_in_link AND {link_evidence(c, '_links', 'ctx_crypto')}) AS link_{c.lower()}"
                          for c in R.COINS)
    parent_cols = ", ".join(f"coalesce(p.in_{c.lower()}, FALSE) AND NOT ({' OR '.join('a.in_' + k.lower() for k in R.COINS)}) "
                            f"AS reply_parent_{c.lower()}" for c in R.COINS)
    return f"""
    WITH a0 AS (
      SELECT s.*, {coin_cols},
             CASE WHEN coin_only_in_link THEN {links_of('message_body')} ELSE '' END AS _links
      FROM read_parquet('{src}') s),
    a AS (
      SELECT a0.* EXCLUDE (_links),
             {R.market_rule(any_coin, 'cash_crypto_any')} AS in_market,
             ({n_coins}) AS n_coins, {link_cols}
      FROM a0)
    SELECT a.message_id AS post_id, a.created_at AS ts, CAST(a.created_at AS DATE) AS day,
           CAST(a.user_id AS VARCHAR) AS account,
           a.in_btc, a.in_eth, a.in_bnb, a.in_xrp, a.in_doge, a.in_market, a.n_coins,
           (a.n_symbols > 0 AND NOT a.cash_crypto_any AND NOT a.btc_plain_symbol) AS equity_only,
           (a.cash_bnb OR a.hit_bnb_t1) AS bnb_strict, a.bsc_context, a.coin_only_in_link,
           {', '.join('a.link_' + c.lower() for c in R.COINS)},
           a.in_reply_to_message_id IS NOT NULL AS is_reply, {parent_cols},
           {R.ambiguous_extra('a.in_market OR ' + ' OR '.join('a.in_' + c.lower() for c in R.COINS)).replace('n_ambiguous_terms', 'a.n_ambiguous_terms')} AS ambiguous_extra,
           a.body_null, md5(coalesce(a.message_body, '')) AS text_key,
           f.bot_any, f.bot_keyword, f.bot_campaign, f.bot_template, f.invite_link
    FROM a
    LEFT JOIN (SELECT message_id, in_btc, in_eth, in_bnb, in_xrp, in_doge FROM a) p
           ON p.message_id = a.in_reply_to_message_id
    LEFT JOIN read_parquet('{flags}') f ON f.post_id = a.message_id"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    log(f"{STEP}: code {CODE_SHA}, rules {RULES_SHA}")
    return series.assign("stocktwits", stocktwits_sql(), STEP, CODE_SHA, RULES_SHA)


if __name__ == "__main__":
    raise SystemExit(main())
