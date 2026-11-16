#!/usr/bin/env python3
"""
02_filter_stocktwits.py -- step 2 for StockTwits: the crypto-relevant messages with the computed columns.

Input    Stage A, month=YYYY-MM/*.parquet (step 1; kept on Google Drive as
         stocktwits_parquet), copied to a local folder (--src). A month that is still
         downloading (a *.partial file) is never read.
Output   data/pipeline/stocktwits/coarse/month=YYYY-MM/data_0.parquet, one per month, with _stats.json.

What this step does, per month
------------------------------
1. Keeps a message if EITHER
     a. its body matches the inclusion pattern (rules.inclusion_predicate): any coin term
        of tiers 1-3, or a word of the two context lists INCLUSION_CTX_SPECIFIC and
        INCLUSION_CTX_AMBIGUOUS; matched on the raw body, links included, so this table
        stays a superset of every later rule; OR
     b. its symbol list holds a crypto symbol: 1-20 capital letters, digits, dots or
        hyphens followed by .X (STOCKTWITS_CRYPTO_SYMBOL_RX), matched as a whole list
        entry. The plain symbol BTC (without .X) does not admit a message; it counts
        for BTC in step 4 once the message is admitted.
   Every other message is dropped here and never seen again.
2. Parses the author-typed symbols (symbol_list) into syms and crypto_symbols, and adds
   n_symbols, n_crypto_symbols, cash_crypto_any, cash_<coin> (the coin's .X symbol; for
   BTC also the plain symbol BTC) and btc_plain_symbol.
3. Adds the computed columns of the shared rulebook (rules.annotate_sql): evidence
   (hit_<coin>_<tier>, coin_only_in_link, ctx_crypto, n_ambiguous_terms, ...) and
   structure (n_chars, n_tokens, n_urls, n_hashtags, n_mentions, cashtags, is_url_only,
   fingerprint), both on the HTML-decoded body (StockTwits has no quote lines, so full
   text = own words). The stored body is never changed.
4. created_at becomes a UTC timestamp; length and important_words are dropped (not used).

Checks (a failed check writes nothing)
--------------------------------------
rows out = the rows an independent count admits; message_id unique; every row inside the
study window (UTC); no computed column is NULL. The stats file records how many messages
the text and the symbols admitted.

Decisions: StockTwits defect map PB3 (dropped columns), PD6 (symbol pattern, decided
23 Sep 2026), PJ2 (plain BTC symbol), PJ8 (symbols are author-typed); MASTERPLAN D12.

Usage (from code/)
------------------
    python pipeline/02_filter_stocktwits.py --src <local copy of stocktwits_parquet> [--months 2019-08 ...] [--delete-source]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, filtering, runner  # noqa: E402
from pipeline import rules as R  # noqa: E402
from pipeline.filtering import PARQUET, evidence_counts, lit, null_evidence  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "02_filter_stocktwits"
CODE_SHA = runner.code_fingerprint(__file__, filtering.__file__)
RULES_SHA = R.section_sha("filter")

# symbol_list is a Python-repr string, e.g. "['BTC.X', 'AAPL']": parsed by string
# surgery, as since 18 Sep (chr(39) is the single quote).
ST_SYMS = ("list_filter(str_split(replace(replace(replace(coalesce(symbol_list, ''), '[', ''), ']', ''), "
           "chr(39), ''), ', '), x -> x <> '')")
# Entry pattern on the raw string, with the quotes, so a whole symbol is matched (PD6).
ST_SYMBOL_ENTRY = "'" + R.STOCKTWITS_CRYPTO_SYMBOL_RX + "'"
ST_TEXT = "coalesce(message_body, '')"


def stocktwits_source(glob: str) -> str:
    crypto = f"list_filter(syms, x -> regexp_full_match(x, {R.dq(R.STOCKTWITS_CRYPTO_SYMBOL_RX)}))"
    cash = ", ".join(f"list_contains(crypto_symbols, '{c}.X') AS cash_{c.lower()}" for c in R.COINS if c != "BTC")
    return f"""
      SELECT message_id, created_at, user_id, message_body, sentiment, parent_message_id, in_reply_to_message_id,
             symbol_list, syms, body_null, text_any, symbol_any,
             len(syms) AS n_symbols, crypto_symbols, len(crypto_symbols) AS n_crypto_symbols,
             len(crypto_symbols) > 0 AS cash_crypto_any,
             (list_contains(crypto_symbols, 'BTC.X') OR list_contains(syms, 'BTC')) AS cash_btc,
             list_contains(syms, 'BTC') AS btc_plain_symbol, {cash}
      FROM (
        SELECT *, {crypto} AS crypto_symbols FROM (
          SELECT message_id, timezone('UTC', created_at) AS created_at, user_id, message_body, sentiment,
                 parent_message_id, in_reply_to_message_id, symbol_list, {ST_SYMS} AS syms,
                 message_body IS NULL AS body_null,
                 {R.inclusion_predicate(ST_TEXT)} AS text_any,
                 regexp_matches(coalesce(symbol_list, ''), {R.dq(ST_SYMBOL_ENTRY)}) AS symbol_any
          FROM read_parquet('{glob}')) _a
        WHERE text_any OR symbol_any) _b"""


def run(args) -> int:
    src_root = Path(args.src)
    out_root = config.out_path("stocktwits", "coarse")
    months = sorted(p.name.split("=", 1)[1] for p in src_root.glob("month=*") if p.is_dir())
    if args.months:
        months = [m for m in months if m in set(args.months)]
    check(len(months) > 0, f"month partitions found under {src_root}")
    con = runner.connect()
    R.register_udfs(con)
    total = 0
    skipped = []
    for month in months:
        files = sorted((src_root / f"month={month}").glob("*.parquet"))
        if not files or list((src_root / f"month={month}").glob("*.partial")):
            skipped.append(month)          # still downloading: never read a half-written month
            continue
        inputs = runner.describe_inputs(files)
        final = out_root / f"month={month}"
        if runner.is_current(final, inputs, CODE_SHA, RULES_SHA):
            log(f"stocktwits {month}: current, skipped")
            continue
        t0 = time.time()
        glob = str(src_root / f"month={month}" / "*.parquet")
        tmp = runner.tmp_path(final)
        tmp.mkdir(parents=True)
        sql = R.annotate_sql(stocktwits_source(glob), "message_body", None)
        con.execute(f"COPY ({sql}) TO '{tmp}/data_0.parquet' {PARQUET}")
        out = f"read_parquet('{tmp}/data_0.parquet')"
        checks: List[Dict[str, object]] = []
        n_in, n_admit_text, n_admit_sym, n_admit = con.execute(f"""
            SELECT count(*), count_if(t), count_if(s), count_if(t OR s) FROM (
              SELECT {R.inclusion_predicate(ST_TEXT)} AS t,
                     regexp_matches(coalesce(symbol_list, ''), {R.dq(ST_SYMBOL_ENTRY)}) AS s
              FROM read_parquet('{glob}'))""").fetchone()
        n_out, n_ids, n_month_bad, n_window_bad = con.execute(f"""
            SELECT count(*), count(DISTINCT message_id),
                   count_if(strftime(created_at, '%Y-%m') <> {lit(month)}),
                   count_if(CAST(created_at AS DATE) < DATE '{config.WINDOW_START}'
                            OR CAST(created_at AS DATE) > DATE '{config.WINDOW_END}')
            FROM {out}""").fetchone()
        check(n_out == n_admit, f"rows out = rows admitted by an independent count ({n_out:,} of {n_in:,})", checks)
        check(n_ids == n_out, "message_id unique", checks)
        check(n_window_bad == 0, "every row inside the study window (UTC)", checks)
        check(null_evidence(con, out) == 0, "no NULL in evidence or structure columns", checks)
        sym_names = ["cash_crypto_any", "btc_plain_symbol", "body_null"] + [f"cash_{c.lower()}" for c in R.COINS]
        sym_vals = con.execute("SELECT " + ", ".join(f"count_if({n})" for n in sym_names) + f" FROM {out}").fetchone()
        counts = {"rows_in": int(n_in), "rows_out": int(n_out), "admitted_by_text": int(n_admit_text),
                  "admitted_by_symbol": int(n_admit_sym), "rows_outside_partition_month": int(n_month_bad),
                  "evidence": evidence_counts(con, out), "symbols": dict(zip(sym_names, (int(v) for v in sym_vals)))}
        runner.commit(tmp, final)
        runner.write_stats(final, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA,
                           con=con, counts=counts, checks=checks,
                           extra={"platform": "stocktwits", "partition": month, "seconds": round(time.time() - t0, 1)})
        total += n_out
        log(f"stocktwits {month}: {n_out:,} of {n_in:,} rows kept in {time.time() - t0:.0f} s")
        if args.delete_source:
            for f in files:
                f.unlink()
            log(f"stocktwits {month}: local Stage A copy deleted (re-fetchable from {config.STOCKTWITS_STAGE_A_REMOTE})")
    con.close()
    if skipped:
        log(f"stocktwits: {len(skipped)} months not complete locally, not processed: {', '.join(skipped)}")
    log(f"stocktwits done: {total:,} rows written this run")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(config.SCRATCH_DIR / "stocktwits_parquet"),
                    help="local copy of Stage A (month=YYYY-MM/ folders)")
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--delete-source", action="store_true",
                    help="delete each local Stage A month after its output is committed")
    args = ap.parse_args(argv)
    log(f"{STEP}: code {CODE_SHA}, rules {RULES_SHA}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
