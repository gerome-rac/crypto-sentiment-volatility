#!/usr/bin/env python3
"""
04_series_dlt.py -- step 04 for the DLT-Tweets corpus (the D17 event arm): the rules of steps 02-04
applied to DLT, one row per in-window DLT row with its coin evidence, series membership and per-post
bot flags; also the corpus profile and the D17 pre-check. Added 30 Sep 2026 (then named dlt_prepare.py;
its `clean` subcommand is now `06_emotion_emtract.py dlt clean`). It changes no earlier step and
imports their rules:

  rules.py / filtering.py / series.py   coin evidence and structure (rules.annotate_sql, as step 02 for
                                         the Twitter units), the coin gate and the market rule (step 04),
                                         the per-post bot rules (step 03: keyword rule, invite links)
  runner.py                              DuckDB settings (UTC asserted, memory, threads, spill), safe
                                         writes (temporary file -> checks -> commit -> stats file)
  03_bot_filter.py                       the crypto ticker list (`tickers`)

Data: DLT-Tweets (ExponentialScience/DLT-Tweets on Hugging Face, revision
c7d979394488bbd94bd6f493af1775d6b6159176; DLT-Corpus, arXiv 2602.22045), its 14 Parquet shards
downloaded to data/raw/dlt/data/ (not in the repository).

Input   data/raw/dlt/data/train-000NN-of-00014.parquet (the Hugging Face DLT-Tweets shards; columns
        timestamp, tweet, language, total_tokens, sentiment_*, confidence_level, year)
Output  data/pipeline/dlt/ (see each subcommand); every table has a stats file.

Subcommands (run from code/)
----------------------------
  tickers   (DuckDB)   the crypto ticker list of step 03 -> dlt/inputs/crypto_tickers.json
  units     (DuckDB)   one row per in-window DLT row: row_id, timestamp fields, text, the step-02
                       evidence and structure columns, the step-04 series columns, the step-03 per-post
                       bot flags -> dlt/units/dlt_units.parquet
  profile   (DuckDB)   corpus profile over ALL raw rows: schema, timestamp classes, rows per month,
                       text properties that matter for EmTract, exact duplicates -> dlt/profile/
  precheck  (DuckDB)   D17 pre-check: daily rows per series, the IA gap windows, rows per month
                       -> dlt/precheck/

Row id (stable, documented): row_id = NN * 100,000,000 + r, where NN is the shard number in the file
name train-000NN-of-00014.parquet and r the 0-based physical row number inside that file (DuckDB's
file_row_number; every shard has fewer than 100,000,000 rows, checked). src_shard = NN, src_row = r.

Timestamps (profiled 30 Sep 2026): 'YYYY-MM-DD HH:MM:SS+00:00' (zone given), 'YYYY-MM-DD HH:MM:SS'
(no zone), 'YYYY-MM-DD HH:MM:SS.000' (always midnight) and 'YYYY-MM-DD' (date only). ts_class keeps the
form; has_time is TRUE for the two second-resolution forms (a bare midnight '.000' stamp counts as a
date); tz_explicit is TRUE when the string carries its offset. Strings without a zone are read as UTC
(an assumption the corpus card does not settle; reported). day_utc is the UTC day; ts_utc is NULL when
there is no time. The study window is 2019-08-01 .. 2023-01-31 (UTC days, config.py); rows outside it
are counted, not written, as step 02 reads only window days.

What is not applied (and why): DLT has no account ids or user names, so the account-based bot rules
of step 03 (campaign, template) and the user-name criterion of the keyword rule cannot run; the keyword
rule uses its four text criteria (at least 2 of 4), as on StockTwits, which has no names either. No
same-text rule, no de-duplication and no language filter (D10, D11, 26 Sep 2026). Nothing is deleted:
the coarse inclusion pattern of step 02 is stored as a flag (coarse_inclusion), not applied.

Usage (from code/; environment `thesis` (environment.yml), as steps 01-05: the units query needs
DuckDB 1.5; the emtract-ref environment's DuckDB 1.4.5 cannot bind it)
------------------------------------------------------------------------------------------------
    python pipeline/04_series_dlt.py tickers
    python pipeline/04_series_dlt.py units --ticker-list data/pipeline/dlt/inputs/crypto_tickers.json
    python pipeline/04_series_dlt.py profile
    python pipeline/04_series_dlt.py precheck
The full run (30 Sep 2026) ran units, profile and precheck on a 30-vCPU cloud VM in a copy of that
environment (DuckDB 1.5.5): units 3 min, profile 18 s, precheck 1 s (README.md); the ticker list was
written on the laptop. The pre-check output of that run is stored as dlt/d17/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from pipeline import config, runner  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "04_series_dlt"
RAW = Path(os.environ.get("THESIS_DLT_RAW", str(config.RAW_DIR / "dlt")))
OUT = config.OUT_DIR / "dlt"
SHARD_RX = re.compile(r"train-(\d{5})-of-(\d{5})\.parquet$")
ROW_ID_BASE = 100_000_000
N_SHARDS_EXPECTED = int(os.environ.get("THESIS_DLT_EXPECT_SHARDS", "14"))
GAP_WINDOWS = (                       # the three IA gap windows (brief of 30 Sep 2026)
    ("gap1_2021-01-07_2021-01-24", "2021-01-07", "2021-01-24"),
    ("gap2_2021-04-11_2021-04-26", "2021-04-11", "2021-04-26"),
    ("gap3_2022-11-23_2022-12-12", "2022-11-23", "2022-12-12"),
    # the corrected UTC boundaries of planning/events_calendar.md (days below 50% coverage), for comparison
    ("gap1_alt_2021-01-07_2021-01-25", "2021-01-07", "2021-01-25"),
    ("gap2_alt_2021-04-11_2021-04-27", "2021-04-11", "2021-04-27"),
)
SERIES = ("btc", "eth", "bnb", "xrp", "doge", "market")
EMOJI_RX = r"[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}\x{1F000}-\x{1F2FF}\x{2B00}-\x{2BFF}\x{FE0F}]"
HTML_ENTITY_RX = r"&(amp|lt|gt|quot|apos|nbsp|#[0-9]+|#x[0-9A-Fa-f]+);"
TICKERS_FILE = OUT / "inputs" / "crypto_tickers.json"     # written by `tickers`, read by `units --ticker-list`


# ==================================================================================================
# Raw shards, timestamps and the study window
# ==================================================================================================

def raw_shards() -> List[Tuple[int, Path]]:
    out = []
    for f in sorted(glob.glob(str(RAW / "data" / "train-*-of-*.parquet"))):
        m = SHARD_RX.search(f)
        if m:
            out.append((int(m.group(1)), Path(f)))
    return out


def classified(src: str, col: str = "timestamp") -> str:
    """Raw rows with ts_class, has_time, tz_explicit and ts_utc (independent of the session zone)."""
    zoned = rf"regexp_full_match({col}, '\d{{4}}-\d{{2}}-\d{{2}}[ T]\d{{2}}:\d{{2}}:\d{{2}}(\.\d+)?(Z|[+-]\d{{2}}:?\d{{2}})')"
    naive = rf"regexp_full_match({col}, '\d{{4}}-\d{{2}}-\d{{2}}[ T]\d{{2}}:\d{{2}}:\d{{2}}(\.\d+)?')"
    midnight = rf"regexp_full_match({col}, '\d{{4}}-\d{{2}}-\d{{2}}[ T]00:00:00\.0+')"
    date_only = rf"regexp_full_match({col}, '\d{{4}}-\d{{2}}-\d{{2}}')"
    cls = (f"CASE WHEN {col} IS NULL THEN 'null' WHEN {zoned} THEN 'sec_zoned' "
           f"WHEN {midnight} THEN 'ms_midnight' WHEN {naive} THEN 'sec_naive' "
           f"WHEN {date_only} THEN 'date' ELSE 'other' END")
    return f"""SELECT *,
        ts_class IN ('sec_zoned', 'sec_naive') AS has_time,
        ts_class = 'sec_zoned' AS tz_explicit,
        CASE WHEN ts_class = 'sec_zoned' THEN try_cast({col} AS TIMESTAMPTZ)
             WHEN ts_class = 'sec_naive' THEN timezone('UTC', try_cast({col} AS TIMESTAMP)) END AS ts_utc
      FROM (SELECT *, {cls} AS ts_class FROM {src}) _c"""


def with_day(src: str) -> str:
    return f"""SELECT *, CASE WHEN ts_utc IS NOT NULL THEN CAST(timezone('UTC', ts_utc) AS DATE)
                             WHEN ts_class IN ('date', 'ms_midnight') THEN try_cast(left(timestamp, 10) AS DATE) END AS day_utc
               FROM ({classified(src)}) _d"""


def in_window(day: str = "day_utc") -> str:
    return f"({day} BETWEEN DATE '{config.WINDOW_START}' AND DATE '{config.WINDOW_END}')"


def write_csv(con, sql: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT CSV, HEADER)")
    os.replace(tmp, path)
    return int(con.execute(f"SELECT count(*) FROM read_csv('{path}', header = true)").fetchone()[0])


def own_sha() -> str:
    return runner.code_fingerprint(Path(__file__).resolve())


# ==================================================================================================
# tickers: the crypto ticker list of step 03 (added 2 Oct 2026)
# ==================================================================================================

def tickers(args) -> int:
    """The crypto ticker list of step 03 (every StockTwits '.X' symbol, suffix removed;
    03_bot_filter.crypto_tickers) as JSON for `units --ticker-list`. The file of 30 Sep 2026 was written
    by a one-off command with the same function and the same fields; sha256_of_list is the sha256 of
    json.dumps(tickers)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bot_filter_03", str(CODE / "pipeline" / "03_bot_filter.py"))
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)
    con = runner.connect()
    try:
        tick = bot.crypto_tickers(con)
    finally:
        con.close()
    check(len(tick) > 0 and tick == sorted(tick) and len(set(tick)) == len(tick), "ticker list non-empty, sorted, distinct")
    rec = {"tickers": tick, "sha256_of_list": hashlib.sha256(json.dumps(tick).encode()).hexdigest(),
           "source": "03_bot_filter.crypto_tickers(con) over data/pipeline/stocktwits/coarse/month=*/data_0.parquet",
           "written_at": dt.date.today().isoformat(), "n": len(tick)}
    st = config.out_path("twitter", "bot", "post_flags.parquet.stats.json")
    if st.exists():                          # step 03 records the list's length for Twitter
        n03 = json.loads(st.read_text()).get("counts", {}).get("crypto_ticker_list")
        check(n03 == len(tick), f"{len(tick)} tickers = step 03's count for Twitter ({n03})")
        rec["twitter_step03_count"] = n03
    out = Path(args.out) if args.out else TICKERS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out)
    log(f"wrote {out}: {len(tick)} tickers, sha256_of_list {rec['sha256_of_list'][:12]}")
    return 0


# ==================================================================================================
# units: one row per in-window DLT row (steps 02-04 rules)
# ==================================================================================================

def units_sql(path: Path, shard_no: int) -> str:
    from pipeline import rules as R
    from pipeline.series import link_evidence, links_of
    src = f"read_parquet('{path}', file_row_number = true)"
    base = f"""SELECT CAST({shard_no} AS BIGINT) * {ROW_ID_BASE} + file_row_number AS row_id,
                      CAST({shard_no} AS SMALLINT) AS src_shard, CAST(file_row_number AS INTEGER) AS src_row,
                      timestamp AS ts_raw, ts_class, has_time, tz_explicit, ts_utc, day_utc,
                      tweet AS text, CAST(NULL AS VARCHAR) AS parent_text, language, year AS dlt_year,
                      total_tokens AS dlt_total_tokens
               FROM ({with_day(src)}) _w WHERE {in_window()}"""
    full = "text || ' ' || coalesce(parent_text, '')"          # as the Twitter units of step 02
    ann = R.annotate_sql(base, full, "text", extra_layer3=[("own_text_hit", R.own_coin_hit())])
    gates = {c: R.coin_gate(c) for c in R.COINS}
    coin_cols = ", ".join(f"{gates[c]} AS in_{c.lower()}" for c in R.COINS)
    any_coin = " OR ".join(f"in_{c.lower()}" for c in R.COINS)
    n_coins = " + ".join(f"CAST(in_{c.lower()} AS INTEGER)" for c in R.COINS)
    link_cols = ", ".join(f"(coin_only_in_link AND {link_evidence(c, '_links', 'ctx_crypto')}) AS link_{c.lower()}"
                          for c in R.COINS)
    amb = R.ambiguous_extra("a.in_market OR " + " OR ".join("a.in_" + c.lower() for c in R.COINS)).replace(
        "n_ambiguous_terms", "a.n_ambiguous_terms")
    kw = R.keyword_criteria("own_dec", "n_hashtags", "n_tickers", None)      # no user names: 4 criteria
    kw_sel = ", ".join(f"{e} AS {n}" for n, e in kw)
    kw_sum = " + ".join(f"CAST({n} AS INTEGER)" for n, _ in kw)
    return f"""
    WITH a0 AS (
      SELECT u.*, {R.inclusion_predicate(full)} AS coarse_inclusion, {coin_cols},
             CASE WHEN coin_only_in_link THEN {links_of(full)} ELSE '' END AS _links
      FROM ({ann}) u),
    a AS (SELECT a0.* EXCLUDE (_links), {R.market_rule(any_coin)} AS in_market, ({n_coins}) AS n_coins, {link_cols}
          FROM a0),
    b AS (SELECT a.*, a.hit_bnb_t1 AS bnb_strict, {amb} AS ambiguous_extra, md5(a.text) AS text_md5,
                 regexp_matches(a.text, '^\\s*RT\\b') AS rt_prefix,
                 py_html_unescape(coalesce(a.text, '')) AS own_dec,
                 len(list_intersect(a.cashtags, (SELECT list(t) FROM TK))) AS n_tickers
          FROM a a),
    c AS (SELECT b.*, {kw_sel}, {R.match('own_dec', R.INVITE_RX)} AS invite_link FROM b)
    SELECT c.* EXCLUDE (own_dec), ({kw_sum}) AS n_kw_criteria, ({kw_sum}) >= {R.KW_MIN_CRITERIA} AS bot_keyword,
           ({kw_sum}) >= {R.KW_MIN_CRITERIA} AS bot_any
    FROM c ORDER BY row_id"""


def units(args) -> int:
    import numpy as np
    import pyarrow.parquet as pq
    from pipeline import filtering, rules as R
    shards = raw_shards()
    check(len(shards) == N_SHARDS_EXPECTED, f"{N_SHARDS_EXPECTED} raw DLT shards found ({len(shards)})")
    tick_rec = json.loads(Path(args.ticker_list).read_text())
    tick = tick_rec["tickers"]
    check(hashlib.sha256(json.dumps(tick).encode()).hexdigest() == tick_rec["sha256_of_list"],
          "ticker list matches its recorded sha256")
    final = OUT / "units" / "dlt_units.parquet"
    inputs = runner.describe_inputs([p for _, p in shards] + [Path(args.ticker_list)])
    code_sha = own_sha()
    rules_sha = R.section_sha("filter") + "-" + R.section_sha("bot") + "-" + R.section_sha("series")
    if runner.is_current(final, inputs, code_sha, rules_sha):
        log("units current, skipped")
        return 0
    t0 = time.time()
    con = runner.connect()
    R.register_udfs(con)
    con.execute("CREATE TEMP TABLE TK AS SELECT unnest(?::VARCHAR[]) AS t", [tick])
    checks: List[Dict[str, object]] = []
    parts_dir = OUT / "units" / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    per_shard: Dict[str, Dict[str, int]] = {}
    parts: List[Path] = []
    for no, path in shards:
        n_meta = pq.ParquetFile(str(path)).metadata.num_rows
        check(n_meta < ROW_ID_BASE, f"{path.name}: fewer than {ROW_ID_BASE:,} rows")
        part = parts_dir / f"part-{no:05d}.parquet"
        con.execute(f"COPY ({units_sql(path, no)}) TO '{part}' {filtering.PARQUET}")
        src = f"read_parquet('{path}', file_row_number = true)"
        n_all, n_in, n_before, n_after, n_noday = con.execute(f"""
            SELECT count(*), count_if({in_window()}), count_if(day_utc < DATE '{config.WINDOW_START}'),
                   count_if(day_utc > DATE '{config.WINDOW_END}'), count_if(day_utc IS NULL) FROM ({with_day(src)})""").fetchone()
        n_part = con.execute(f"SELECT count(*) FROM read_parquet('{part}')").fetchone()[0]
        check(n_all == n_meta, f"{path.name}: rows read = parquet metadata ({n_meta:,})", checks)
        check(n_part == n_in, f"{path.name}: rows written = in-window rows ({n_in:,})", checks)
        check(n_in + n_before + n_after + n_noday == n_all, f"{path.name}: in + before + after + no day = all", checks)
        per_shard[path.name] = {"rows": int(n_all), "in_window": int(n_in), "before_window": int(n_before),
                                "after_window": int(n_after), "no_day": int(n_noday)}
        parts.append(part)
        log(f"units {path.name}: {n_in:,} of {n_all:,} rows in the window, {time.time() - t0:.0f} s")
    # One file, rows in row_id order: each part is sorted by row_id and the parts follow the shard
    # numbers, so the row groups are streamed in part order (no global sort needed; checked below).
    tmp = runner.tmp_path(final)
    writer = None
    try:
        for part in parts:
            pf = pq.ParquetFile(str(part))
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg)
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp), t.schema, compression="zstd")
                writer.write_table(t, row_group_size=t.num_rows)
    finally:
        if writer is not None:
            writer.close()
    o = f"read_parquet('{tmp}')"
    n_u, n_ids = con.execute(f"SELECT count(*), count(DISTINCT row_id) FROM {o}").fetchone()
    exp_in = sum(v["in_window"] for v in per_shard.values())
    check(n_u == n_ids == exp_in, f"one row per row_id, rows = in-window rows ({n_u:,})", checks)
    rid = pq.read_table(str(tmp), columns=["row_id"]).column("row_id").to_numpy()
    check(bool((np.diff(rid) > 0).all()) if len(rid) > 1 else True, "rows stored in increasing row_id order", checks)
    bad_day = con.execute(f"SELECT count_if(day_utc IS NULL OR NOT {in_window()}) FROM {o}").fetchone()[0]
    check(bad_day == 0, "every row's day_utc inside the window", checks)
    check(filtering.null_evidence(con, o) == 0, "no NULL in evidence or structure columns", checks)
    both = con.execute(f"SELECT count_if(in_market AND ({' OR '.join('in_' + c.lower() for c in R.COINS)})) FROM {o}").fetchone()[0]
    check(both == 0, "no row in a coin series and in the market", checks)
    nullflag = con.execute(f"SELECT count_if(bot_keyword IS NULL OR invite_link IS NULL OR bot_any IS NULL) FROM {o}").fetchone()[0]
    check(nullflag == 0, "no NULL bot flag", checks)
    counts = units_counts(con, o)
    counts["per_shard"] = per_shard
    counts["rows_total"] = sum(v["rows"] for v in per_shard.values())
    counts["rows_in_window"] = exp_in
    counts["ticker_list"] = {"n": len(tick), "sha256": tick_rec["sha256_of_list"]}
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP + " units", inputs=inputs, code_sha=code_sha, rules_sha=rules_sha, con=con,
                       counts=counts, checks=checks,
                       extra={"seconds": round(time.time() - t0, 1),
                              "row_id": f"NN * {ROW_ID_BASE:,} + file_row_number (NN from train-000NN-of-00014)",
                              "bot_rules_applied": ["keyword (4 text criteria, >= 2; as StockTwits)", "invite_link (flag only)"],
                              "bot_rules_not_applicable": ["campaign (needs account ids)", "template (needs account ids)",
                                                           "keyword criterion 'bot' in the user name (no user names)"],
                              "bot_any": "bot_keyword (the only exclusion rule computable without accounts)"})
    for p in parts:
        p.unlink()
    try:
        parts_dir.rmdir()
    except OSError:
        pass
    con.close()
    log(f"units: {n_u:,} rows in {time.time() - t0:.0f} s")
    return 0


def units_counts(con, o: str) -> Dict[str, object]:
    from pipeline import filtering, rules as R
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    out: Dict[str, object] = {}
    out["ts_class"] = {r[0]: int(r[1]) for r in q(f"SELECT ts_class, count(*) FROM {o} GROUP BY 1 ORDER BY 1")}
    out["has_time"] = int(q(f"SELECT count_if(has_time) FROM {o}")[0][0])
    out["tz_explicit"] = int(q(f"SELECT count_if(tz_explicit) FROM {o}")[0][0])
    out["language"] = {str(r[0]): int(r[1]) for r in q(f"SELECT language, count(*) FROM {o} GROUP BY 1 ORDER BY 2 DESC")}
    ser = ", ".join(f"count_if(in_{s}) AS {s}" for s in SERIES)
    ser_nb = ", ".join(f"count_if(in_{s} AND NOT bot_any) AS {s}" for s in SERIES)
    out["series"] = dict(zip(SERIES, (int(v) for v in q(f"SELECT {ser} FROM {o}")[0])))
    out["series_after_bot_pass"] = dict(zip(SERIES, (int(v) for v in q(f"SELECT {ser_nb} FROM {o}")[0])))
    out["in_no_series"] = int(q(f"SELECT count_if(NOT ({' OR '.join('in_' + s for s in SERIES)})) FROM {o}")[0][0])
    out["n_coins"] = {int(r[0]): int(r[1]) for r in q(f"SELECT n_coins, count(*) FROM {o} GROUP BY 1 ORDER BY 1")}
    out["coarse_inclusion"] = int(q(f"SELECT count_if(coarse_inclusion) FROM {o}")[0][0])
    flags = ["kw_giveaway", "kw_pump_join", "kw_hashtags", "kw_tickers", "bot_keyword", "invite_link", "bot_any",
             "bnb_strict", "bsc_context", "coin_only_in_link", "own_text_hit", "ambiguous_extra", "rt_prefix",
             "term_merge"] + [f"link_{c.lower()}" for c in R.COINS]
    out["flags"] = dict(zip(flags, (int(v) for v in q("SELECT " + ", ".join(f"count_if({f})" for f in flags) + f" FROM {o}")[0])))
    out["n_kw_criteria"] = {int(r[0]): int(r[1]) for r in q(f"SELECT n_kw_criteria, count(*) FROM {o} GROUP BY 1 ORDER BY 1")}
    out["evidence"] = filtering.evidence_counts(con, o)
    out["distinct_text_md5"] = int(q(f"SELECT count(DISTINCT text_md5) FROM {o}")[0][0])
    out["text_null_or_empty"] = int(q(f"SELECT count_if(text IS NULL OR trim(text) = '') FROM {o}")[0][0])
    return out


# ==================================================================================================
# profile: the corpus over all raw rows
# ==================================================================================================
PROFILE_FLAGS: Tuple[Tuple[str, str], ...] = (
    ("emoji", f"regexp_matches(tweet, '{EMOJI_RX}')"),
    ("url_http", r"regexp_matches(tweet, 'https?://[^\s]+')"),
    ("url_tco", r"regexp_matches(tweet, 't\.co/')"),
    ("upper_any", "regexp_matches(tweet, '[A-Z]')"),
    ("no_upper_has_lower", "NOT regexp_matches(tweet, '[A-Z]') AND regexp_matches(tweet, '[a-z]')"),
    ("at_any", "contains(tweet, '@')"),
    ("at_word", r"regexp_matches(tweet, '@\w')"),
    ("at_bare", r"regexp_matches(tweet, '@([^\w]|$)')"),
    ("starts_at_word", r"regexp_matches(tweet, '^\s*@\w')"),
    ("rt_start", r"regexp_matches(tweet, '^\s*RT\b')"),
    ("rt_at_start", r"regexp_matches(tweet, '^\s*RT @')"),
    ("rt_colon_start", r"regexp_matches(tweet, '^\s*RT\s*:')"),
    ("rt_anywhere", r"regexp_matches(tweet, '\bRT\b')"),
    ("via_at", r"regexp_matches(tweet, '\bvia @\w')"),
    ("html_entity", f"regexp_matches(tweet, '{HTML_ENTITY_RX}')"),
    ("hashtag", r"regexp_matches(tweet, '#\w+')"),
    ("cashtag", r"regexp_matches(tweet, '\$[A-Za-z][A-Za-z0-9]{0,19}(\.[A-Za-z])?\b')"),
    ("newline", r"regexp_matches(tweet, '[\n\r]')"),
    ("status_link", r"regexp_matches(tweet, 'twitter\.com/[^\s]+/status/')"),
    ("ends_ellipsis", r"regexp_matches(tweet, '(…|\.\.\.)\s*$')"),
    ("ellipsis_then_url_end", r"regexp_matches(tweet, '…\s*https?://[^\s]+\s*$')"),
    ("null_or_empty", "tweet IS NULL OR trim(tweet) = ''"),
    ("over_280_chars", "length(tweet) > 280"),
)


def profile(args) -> int:
    import pyarrow.parquet as pq
    shards = raw_shards()
    check(len(shards) == N_SHARDS_EXPECTED, f"{N_SHARDS_EXPECTED} raw DLT shards found ({len(shards)})")
    out_dir = OUT / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = runner.connect()
    files = [str(p) for _, p in shards]
    src = f"read_parquet({files!r})"
    con.execute(f"CREATE TEMP VIEW r AS SELECT *, coalesce({in_window()}, FALSE) AS in_window FROM ({with_day(src)})")
    res: Dict[str, object] = {"files": []}
    for _, p in shards:
        md = pq.ParquetFile(str(p)).metadata
        res["files"].append({"file": p.name, "rows": md.num_rows, "row_groups": md.num_row_groups,
                             "bytes": p.stat().st_size, "created_by": md.created_by})
    sch = pq.ParquetFile(files[0]).schema_arrow
    res["schema"] = [f"{f.name}: {f.type}" for f in sch]
    res["schema_equal_across_shards"] = all(pq.ParquetFile(f).schema_arrow.equals(sch) for f in files)
    n: Dict[str, int] = {}
    n["month"] = write_csv(con, """SELECT strftime(day_utc, '%Y-%m') AS month, count(*) AS rows_all, count_if(in_window) AS rows_in_window,
                   count_if(has_time) AS rows_has_time, count_if(tz_explicit) AS rows_tz_explicit,
                   count_if(ts_class = 'sec_naive') AS rows_sec_naive, count_if(ts_class IN ('date', 'ms_midnight')) AS rows_date_only
               FROM r GROUP BY 1 ORDER BY 1 NULLS FIRST""", out_dir / "dlt_profile_rows_by_month.csv")
    n["year"] = write_csv(con, """SELECT year(day_utc) AS year_utc, count(*) AS rows_all, count_if(in_window) AS rows_in_window,
                   count_if(year = CAST(year(day_utc) AS VARCHAR)) AS year_field_equal
               FROM r GROUP BY 1 ORDER BY 1 NULLS FIRST""", out_dir / "dlt_profile_rows_by_year.csv")
    n["ts_class"] = write_csv(con, """SELECT ts_class, count(*) AS rows_all, count_if(in_window) AS rows_in_window,
                   min(timestamp) AS min_value, max(timestamp) AS max_value,
                   count_if(ts_class IN ('sec_zoned', 'sec_naive') AND right(substr(timestamp, 1, 19), 2) = '00') AS seconds_00
               FROM r GROUP BY 1 ORDER BY 2 DESC""", out_dir / "dlt_profile_timestamp_classes.csv")
    n["offsets"] = write_csv(con, r"""SELECT regexp_extract(timestamp, '(Z|[+-]\d{2}:?\d{2})$') AS utc_offset, count(*) AS rows_all
               FROM r WHERE ts_class = 'sec_zoned' GROUP BY 1 ORDER BY 2 DESC""", out_dir / "dlt_profile_timestamp_offsets.csv")
    n["language"] = write_csv(con, "SELECT language, count(*) AS rows_all, count_if(in_window) AS rows_in_window FROM r GROUP BY 1 ORDER BY 2 DESC",
                              out_dir / "dlt_profile_language.csv")
    sel = ", ".join(f"count_if({e}) AS {k}" for k, e in PROFILE_FLAGS)
    rows = con.execute(f"SELECT in_window, count(*) AS n, {sel} FROM r GROUP BY 1 ORDER BY 1").fetchall()
    by = {bool(r[0]): r for r in rows}
    width = 2 + len(PROFILE_FLAGS)
    tot = [sum(int(by[k][i]) for k in by) for i in range(width)]
    n_all = tot[1]
    n_in = int(by[True][1]) if True in by else 0
    lines = ["property,rows_all,share_all_pct,rows_in_window,share_in_window_pct,pattern"]
    for i, (k, e) in enumerate(PROFILE_FLAGS):
        a = tot[2 + i]
        w = int(by[True][2 + i]) if True in by else 0
        pat = e.replace('"', '""')
        lines.append(f"{k},{a},{100 * a / n_all if n_all else 0:.4f},{w},{100 * w / n_in if n_in else 0:.4f},\"{pat}\"")
    p = out_dir / "dlt_profile_text_properties.csv"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, p)
    dup = con.execute("""SELECT count(*), count(DISTINCT md5_number(tweet)),
                                count_if(in_window), count(DISTINCT md5_number(tweet)) FILTER (WHERE in_window) FROM r""").fetchone()
    grp = con.execute("""SELECT count(*), sum(c), max(c) FROM
                         (SELECT md5_number(tweet) h, count(*) c FROM r WHERE in_window GROUP BY 1 HAVING count(*) > 1)""").fetchone()
    lens = con.execute("SELECT quantile_cont(length(tweet), [0.05, 0.5, 0.95, 0.99]), max(length(tweet)), "
                       "quantile_cont(total_tokens, [0.05, 0.5, 0.95]), max(total_tokens) FROM r WHERE in_window").fetchone()
    res.update({"rows_total": n_all, "rows_in_window": n_in,
                "duplicates": {"rows_all": int(dup[0]), "distinct_texts_all": int(dup[1]), "rows_in_window": int(dup[2]),
                               "distinct_texts_in_window": int(dup[3]), "in_window_groups_gt1": int(grp[0] or 0),
                               "in_window_rows_in_groups_gt1": int(grp[1] or 0), "in_window_max_group": int(grp[2] or 0),
                               "method": "md5_number(tweet) of the exact string"},
                "chars_in_window_p05_p50_p95_p99": lens[0], "chars_in_window_max": lens[1],
                "dlt_total_tokens_in_window_p05_p50_p95": lens[2], "dlt_total_tokens_in_window_max": lens[3],
                "csv_rows": n, "seconds": round(time.time() - t0, 1), "settings": runner.settings(con),
                "emoji_pattern": EMOJI_RX, "code_sha": own_sha()})
    runner.write_json_atomic(out_dir / "dlt_profile.json", res)
    con.close()
    log(f"profile: {n_all:,} rows ({n_in:,} in the window) in {time.time() - t0:.0f} s")
    return 0


# ==================================================================================================
# precheck: D17 day-level counts (facts only)
# ==================================================================================================

def precheck(args) -> int:
    out_dir = OUT / "precheck"
    src = OUT / "units" / "dlt_units.parquet"
    check(src.exists(), f"units table found: {src}")
    t0 = time.time()
    con = runner.connect()
    u = f"read_parquet('{src}')"
    days = (f"SELECT CAST(range AS DATE) AS day_utc FROM range(DATE '{config.WINDOW_START}', "
            f"DATE '{config.WINDOW_END}' + INTERVAL 1 DAY, INTERVAL 1 DAY)")
    agg = ", ".join([f"count_if(in_{s}) AS n_{s}" for s in SERIES] +
                    [f"count_if(in_{s} AND NOT bot_any) AS n_{s}_after_bot" for s in SERIES])
    names = ["n_rows", "n_rows_has_time", "n_rows_after_bot", "n_bot_flagged"] + [f"n_{s}" for s in SERIES] + \
            [f"n_{s}_after_bot" for s in SERIES]
    con.execute(f"""CREATE TEMP TABLE DAILY AS
        SELECT d.day_utc, {', '.join(f'CAST(coalesce(x.{c}, 0) AS BIGINT) AS {c}' for c in names)}
        FROM ({days}) d LEFT JOIN (
            SELECT day_utc, count(*) AS n_rows, count_if(has_time) AS n_rows_has_time,
                   count_if(NOT bot_any) AS n_rows_after_bot, count_if(bot_any) AS n_bot_flagged, {agg}
            FROM {u} GROUP BY 1) x USING (day_utc)""")
    n_days = write_csv(con, "SELECT * FROM DAILY ORDER BY day_utc", out_dir / "dlt_precheck_daily_counts.csv")
    check(n_days == config.N_DAYS, f"{config.N_DAYS} window days in the daily table ({n_days})")
    tot_rows = con.execute(f"SELECT count(*) FROM {u}").fetchone()[0]
    check(con.execute("SELECT sum(n_rows) FROM DAILY").fetchone()[0] == tot_rows, "daily rows add up to the units rows")
    parts = []
    for name, a, b in GAP_WINDOWS:
        for s in ("rows",) + SERIES:
            col, after = f"n_{s}", f"n_{s}_after_bot"
            parts.append(f"""SELECT '{name}' AS window, DATE '{a}' AS first_day, DATE '{b}' AS last_day, '{s}' AS series,
                   count(*) AS days, sum({col}) AS rows_total, sum({after}) AS rows_after_bot,
                   count_if({col} > 0) AS days_with_rows, min({col}) AS min_per_day,
                   quantile_disc({col}, 0.5) AS median_per_day, max({col}) AS max_per_day
                   FROM DAILY WHERE day_utc BETWEEN DATE '{a}' AND DATE '{b}'""")
    write_csv(con, " UNION ALL ".join(parts), out_dir / "dlt_precheck_gap_windows.csv")
    msel = ", ".join([f"sum({c}) AS {c}" for c in names] + ["count_if(n_rows = 0) AS days_without_rows"])
    write_csv(con, f"SELECT strftime(day_utc, '%Y-%m') AS month, count(*) AS days, {msel} FROM DAILY GROUP BY 1 ORDER BY 1",
              out_dir / "dlt_precheck_monthly_counts.csv")
    write_csv(con, "SELECT * FROM DAILY WHERE day_utc BETWEEN DATE '2022-06-01' AND DATE '2022-08-31' ORDER BY day_utc",
              out_dir / "dlt_precheck_daily_jun_aug_2022.csv")
    runner.write_json_atomic(out_dir / "dlt_precheck.json", {
        "step": STEP + " precheck", "units_rows": int(tot_rows), "days": n_days,
        "gap_windows": [list(g) for g in GAP_WINDOWS], "seconds": round(time.time() - t0, 1),
        "settings": runner.settings(con), "code_sha": own_sha(),
        "input": runner.describe_inputs([src], with_md5=False),
        "note": "counts only; a day without DLT rows has 0 (the corpus has no row that day)"})
    con.close()
    log(f"precheck written in {time.time() - t0:.0f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="04_series_dlt.py", description="step 04 for the DLT corpus (see the header)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tickers", help="the crypto ticker list of step 03 as JSON")
    t.add_argument("--out", default=None, help=f"default {TICKERS_FILE}")
    u = sub.add_parser("units")
    u.add_argument("--ticker-list", required=True, help="JSON with the crypto ticker list (`tickers`)")
    sub.add_parser("profile")
    sub.add_parser("precheck")
    args = ap.parse_args(argv)
    log(f"{STEP} {args.cmd}: code {own_sha()}, out {OUT}")
    return {"tickers": tickers, "units": units, "profile": profile, "precheck": precheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
