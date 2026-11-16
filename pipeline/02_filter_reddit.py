#!/usr/bin/env python3
"""
02_filter_reddit.py -- step 2 for Reddit: every post and comment with the computed columns.

Input    data/pipeline/reddit/extract/reddit_<P>.parquet (step 1), P = BTC, ETH, BNB, XRP,
         DOGE, MARKET. Each file must be the one step 1 recorded (same md5).
Output   data/pipeline/reddit/coarse/coin=<P>/data_0.parquet, one per group, with _stats.json.

What this step does
-------------------
Reddit needs no inclusion filter: a post's subreddit already says what it is about, so
EVERY row is kept (dead rows included; they carry no evidence). Which coin a post counts
for is decided in 04_series_reddit.py. This step adds, per row, the columns every
platform gets from the shared rulebook (rules.annotate_sql):
  evidence   hit_<coin>_<tier> (5 coins x 3 tiers), coin_only_in_link, ctx_crypto,
             ctx_crypto_excl_btc, n_ambiguous_terms, term_merge, bsc_context; read on
             the FULL text (text, quote lines included);
  structure  n_chars, n_tokens, n_urls, n_hashtags, n_mentions, cashtags, is_url_only,
             fingerprint; read on the author's OWN words (own_text, quote lines removed).
Before any column is computed the text is HTML-decoded, links are removed (a coin word
inside a URL does not count), cashtags are masked for tier-2/3 words, and fork names are
removed for BTC and ETH (rules.py explains each step). The stored text is never changed.
Two Reddit-only columns serve the r/btc rule of step 4:
  bch_mention  the text names Bitcoin Cash (RBTC_BITCOIN_CASH);
  btc_ticker   a BTC ticker other than the subreddit's own name (RBTC_BTC_TICKER).

Checks (a failed check writes nothing)
--------------------------------------
rows out = rows in = the rows step 1 wrote; ids unique and the id set identical; every
step-1 column passes through unchanged (checksum over all of them); no computed column is
NULL; dead rows carry no evidence and no fingerprint; coin_label equals the group.

Decisions: Reddit defect map P15 (r/btc), P18 (own words), PB5/PD4/PD5/PJ6 (text
preparation); MASTERPLAN D12 (coin tiers).

Usage (from code/)
------------------
    python pipeline/02_filter_reddit.py [--partitions BTC ETH ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, filtering, runner  # noqa: E402
from pipeline import rules as R  # noqa: E402
from pipeline.filtering import BOOL_EVIDENCE, PARQUET, evidence_counts, lit, null_evidence  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "02_filter_reddit"
CODE_SHA = runner.code_fingerprint(__file__, filtering.__file__)
RULES_SHA = R.section_sha("filter")

REDDIT_PARTITIONS = ("BNB", "BTC", "DOGE", "ETH", "MARKET", "XRP")


def run(args) -> int:
    src_dir = config.out_path("reddit", "extract")
    out_root = config.out_path("reddit", "coarse")
    con = runner.connect()
    R.register_udfs(con)
    total = 0
    for part in args.partitions or REDDIT_PARTITIONS:
        src = src_dir / f"reddit_{part}.parquet"
        extract_stats = json.loads((src_dir / f"reddit_{part}_stats.json").read_text())
        inputs = runner.describe_inputs([src])
        check(inputs[0]["md5"] == extract_stats["output_md5"], f"{src.name} is the file step 1 recorded")
        final = out_root / f"coin={part}"
        if runner.is_current(final, inputs, CODE_SHA, RULES_SHA):
            log(f"reddit {part}: current, skipped")
            continue
        t0 = time.time()
        tmp = runner.tmp_path(final)
        tmp.mkdir(parents=True)
        cols_in = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src}')").fetchall()]
        extra = [("bch_mention", R.match("_dec_full", R.RBTC_BITCOIN_CASH)),
                 ("btc_ticker", R.match(R.remove("lower(_nl)", R.RBTC_SELF, "g"), R.RBTC_BTC_TICKER, ""))]
        sql = R.annotate_sql(f"SELECT * FROM read_parquet('{src}')", "text", "own_text", extra_layer3=extra)
        con.execute(f"COPY ({sql}) TO '{tmp}/data_0.parquet' {PARQUET}")
        out = f"read_parquet('{tmp}/data_0.parquet')"
        checks: List[Dict[str, object]] = []
        col_list = ", ".join(cols_in)
        n_in, sum_in = con.execute(f"SELECT count(*), bit_xor(hash({col_list})) FROM read_parquet('{src}')").fetchone()
        n_out, n_ids, sum_out = con.execute(
            f"SELECT count(*), count(DISTINCT id), bit_xor(hash({col_list})) FROM {out}").fetchone()
        check(n_out == n_in == extract_stats["checks"]["rows_out"], f"rows out = rows in ({n_out:,})", checks)
        check(n_ids == n_out, "ids unique", checks)
        check(sum_in == sum_out, "every step-1 column passes through unchanged (checksum)", checks)
        miss = con.execute(f"SELECT count(*) FROM read_parquet('{src}') ANTI JOIN {out} USING (id)").fetchone()[0]
        check(miss == 0, "id set identical", checks)
        check(null_evidence(con, out) == 0, "no NULL in evidence or structure columns", checks)
        any_ev = " OR ".join(BOOL_EVIDENCE)
        dead_hits = con.execute(
            f"SELECT count_if(is_dead AND ({any_ev} OR fingerprint IS NOT NULL OR bch_mention OR btc_ticker)) FROM {out}"
        ).fetchone()[0]
        check(dead_hits == 0, "dead rows carry no evidence and no fingerprint", checks)
        bad_label = con.execute(f"SELECT count_if(coin_label IS DISTINCT FROM {lit(part)}) FROM {out}").fetchone()[0]
        check(bad_label == 0, "coin_label equals the partition", checks)
        live, dead, bch, tick = con.execute(
            f"SELECT count_if(NOT is_dead), count_if(is_dead), count_if(bch_mention AND NOT is_dead), "
            f"count_if(btc_ticker AND NOT is_dead) FROM {out}").fetchone()
        counts = {"rows": int(n_out), "live": int(live), "dead": int(dead),
                  "evidence_live": evidence_counts(con, out, "NOT is_dead"),
                  "bch_mention_live": int(bch), "btc_ticker_live": int(tick)}
        runner.commit(tmp, final)
        runner.write_stats(final, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=con,
                           counts=counts, checks=checks,
                           extra={"platform": "reddit", "partition": part, "seconds": round(time.time() - t0, 1)})
        total += n_out
        log(f"reddit {part}: {n_out:,} rows (live {live:,}) in {time.time() - t0:.0f} s")
    con.close()
    log(f"reddit done: {total:,} rows written this run")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partitions", nargs="*", default=None, help="groups to process (default: all six)")
    args = ap.parse_args(argv)
    log(f"{STEP}: code {CODE_SHA}, rules {RULES_SHA}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
