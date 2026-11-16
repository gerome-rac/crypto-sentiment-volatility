#!/usr/bin/env python3
"""
02_filter_twitter.py -- step 2 for Twitter: the crypto-relevant tweets, one row per tweet, and the capture coverage.

Three parts, run in this order:

1. coarse    Stage A day by day -> data/pipeline/twitter/coarse/date=YYYY-MM-DD/{tweets,retweets}.parquet
2. units     the coarse days -> data/pipeline/twitter/units/ (one row per tweet)
3. coverage  the capture ledgers of step 1 -> data/pipeline/twitter/units/twitter_coverage_utc.parquet
             (run after `units`, or alone as `coverage`; rebuilt whenever a ledger changes)

Part 1, coarse (per day)
------------------------
Input: Stage A, YYYY/date=YYYY-MM-DD/{tweets,retweets}.parquet (step 1; kept on Google
Drive as twitter_parquet). Each day is fetched, filtered, written and its local copy
deleted (--src-mode rclone), or read from a local folder (--src-mode local); every
fetched file must match the md5 Drive reports.
* A tweet is kept if its text plus its quoted text matches the inclusion pattern
  (rules.inclusion_predicate: any coin term of tiers 1-3, or a word of the two context
  lists; on the raw text, links included). A retweet is kept if the ORIGINAL's text
  matches. Every other row is dropped here.
* Columns are selected by name in a fixed order and type (older files lack some columns:
  they become typed NULLs). Screen names are replaced by flags (name_has_bot: "bot"
  anywhere in the name) and never written.
* ts = the tweet's time in UTC; for retweets also orig_ts, the original's writing time,
  decoded from its id (Twitter ids carry their creation time: ms = (id >> 22) +
  1288834974657; checked on all 13,178,398 stored tweets on 22 Sep 2026).
* The computed columns of the shared rulebook (rules.annotate_sql): evidence on text +
  quoted text (tweets) or the original's text (retweets); structure on the author's own
  text; own_text_hit = a coin term in the author's own words.
Checks: rows out = the rows an independent count admits; no computed column is NULL; no
screen-name column written.

Part 2, units: one row per tweet
--------------------------------
The 1% stream captures a tweet directly, as the original inside retweets, or both, and
often several times. This part merges the captures BY ID, never by text:
* direct captures of one tweet_id collapse to one row (T); retweet rows collapse to one
  row per retweet_id, its earliest capture (RT), and then to one row per original (RA:
  number of retweet events, retweeting accounts, first and last retweet);
* a unit's text is its longest captured copy (direct capture or retweet copy; a retweet
  copy wins only when strictly longer). Among equally long copies the earliest captured
  wins, then the first in sort order (decided 28 Sep 2026; before, the choice among equal
  lengths was arbitrary: 23 units had such a tie, 11 of them changed text);
* units are the captured tweets (original, reply, quote) plus the originals known only
  through retweets (role retweet_only). A retweet-only original is dated at its first
  sampled retweet (panel_ts = first_retweeted_at); its writing time is kept as authored_at
  (authored_before_window when written before 1 Aug 2019);
* the computed columns are recomputed on the chosen text (text + parent text for
  evidence, text for structure); text_md5 = md5(text), text_key = md5(text, parent text).
Also written: retweet_events_daily.parquet (one row per original and UTC day of its
retweets, for the retweet-event robustness series) and text_variants.parquet (every other
text a unit was captured with).
Checks: one row per tweet id, and ids = captured tweets + retweeted originals; retweet
events counted once each; every panel day inside the window; every unit has an author and
a writing time; every source text is a unit text, a parent text or a stored variant;
retweet events per day add up to the events per original.

Part 3, coverage: how much of each UTC day the stream captured
--------------------------------------------------------------
day_coverage_pct in Stage A is ARCHIVE-day coverage (the share of the 1,440 minute-files
present). Before 2021-08-23 an archive day is a US Mountain-time day, so it does not
describe the UTC day the panel uses. Each step-1 ledger records, for every UTC day it
touched, the distinct minutes that contain a captured post; a UTC day that straddles two
archives gets minutes from both. This part sums them per UTC day:
  minutes_utc (lower bound) and minutes_utc_hi (upper bound; they differ only where two
  archives overlap), coverage_pct_utc = 100 * minutes_utc / 1440, n_archives, archives,
  first_hhmm, last_hhmm, archive_day_pct (the archive-day coverage of the archive
  labelled with that day), pre_switch.
Overlaps occur in two known places (a one-minute spill into the previous UTC day after the
switch; the two archives covering 2021-08-23 00:00-05:59); overlapping contributions form
a cluster whose lower bound is its largest contribution and whose upper bound is min(sum,
span). Checks: minutes_utc <= 1,440 every day; minutes_utc_hi - minutes_utc <= 5 every
day; after the switch coverage_pct_utc agrees with archive_day_pct within 0.5 points on
at least 99% of days. The table has its own stats file (inputs: every ledger with bytes and md5)
and is rebuilt whenever a ledger or this part's code changes (28 Sep 2026: the ledger of the
2020-11-15 archive, missing since that archive was rebuilt on 18 Sep, was reconstructed then).

Decisions: MASTERPLAN D10 (md5, not hash), D13 (retweet-only dating, 26 Sep 2026); Twitter defect
map D14/D17 (safe writing), text choice, screen names; MASTERPLAN 28 Sep 2026 (tie-break; a direct
capture before an equally long retweet copy).

Usage (from code/)
------------------
    python pipeline/02_filter_twitter.py coarse [--src-mode rclone|local] [--src-root ...] [--days 2019-08-01 ...]
    python pipeline/02_filter_twitter.py units
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, filtering, runner  # noqa: E402
from pipeline import rules as R  # noqa: E402
from pipeline.filtering import PARQUET, evidence_counts, null_evidence  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "02_filter_twitter"
RULES_SHA = R.section_sha("filter")


# =====================================================================================
# Part 1, coarse: Stage A day by day -> coarse day partitions
# =====================================================================================

TWEETS_SCHEMA: List[Tuple[str, str]] = [
    ("day", "DATE"), ("day_coverage_pct", "DOUBLE"), ("tweet_id", "VARCHAR"),
    ("created_at", "VARCHAR"), ("timestamp_ms", "BIGINT"), ("tweet_type", "VARCHAR"),
    ("lang", "VARCHAR"), ("text", "VARCHAR"), ("quoted_text", "VARCHAR"),
    ("quoted_id", "VARCHAR"), ("reply_to_id", "VARCHAR"), ("reply_to_user", "VARCHAR"),
    ("user_id", "VARCHAR"), ("user_screen_name", "VARCHAR"),
    ("user_followers", "BIGINT"), ("user_verified", "BOOLEAN"),
    ("retweet_count", "BIGINT"), ("favorite_count", "BIGINT"),
]
RETWEETS_SCHEMA: List[Tuple[str, str]] = [
    ("day", "DATE"), ("day_coverage_pct", "DOUBLE"), ("retweet_id", "VARCHAR"),
    ("retweet_created_at", "VARCHAR"), ("retweet_timestamp_ms", "BIGINT"),
    ("retweeter_id", "VARCHAR"), ("retweeter_screen_name", "VARCHAR"),
    ("retweeter_followers", "BIGINT"), ("orig_tweet_id", "VARCHAR"),
    ("orig_created_at", "VARCHAR"), ("orig_author_id", "VARCHAR"),
    ("orig_author_screen_name", "VARCHAR"), ("orig_author_followers", "BIGINT"),
    ("lang", "VARCHAR"), ("orig_text", "VARCHAR"),
]
TW_SCHEMA = {"tweets": TWEETS_SCHEMA, "retweets": RETWEETS_SCHEMA}
# Screen names become flags and are never written (18 Sep decision); reply_to_user is a screen name too.
NAME_FLAGS = {"tweets": [("user_screen_name", "user_name_has_bot")],
              "retweets": [("retweeter_screen_name", "retweeter_name_has_bot"),
                           ("orig_author_screen_name", "orig_author_name_has_bot")]}
DROPPED = {"user_screen_name", "reply_to_user", "retweeter_screen_name", "orig_author_screen_name"}
TW_FULL = {"tweets": "coalesce(text, '') || ' ' || coalesce(quoted_text, '')", "retweets": "coalesce(orig_text, '')"}
TW_OWN = {"tweets": "text", "retweets": None}
TW_ID = {"tweets": "tweet_id", "retweets": "retweet_id"}
# The original's writing time is decoded from its tweet id (Twitter snowflake: ms = (id >> 22) + 1288834974657;
# verified against all 13,178,398 stored created_at values on 22 Sep), so the two string formats of orig_created_at
# (classic until 2022-11-22, ISO 8601 from 2022-12-13) do not matter; the strings are the fallback for non-numeric ids.
SNOWFLAKE_EPOCH_MS = 1288834974657
TW_TS = {"tweets": "timezone('UTC', epoch_ms(timestamp_ms)) AS ts",
         "retweets": "timezone('UTC', epoch_ms(retweet_timestamp_ms)) AS ts, "
                     "coalesce(CASE WHEN regexp_full_match(orig_tweet_id, '[0-9]+') THEN timezone('UTC', epoch_ms("
                     f"CAST((CAST(orig_tweet_id AS HUGEINT) >> 22) + {SNOWFLAKE_EPOCH_MS} AS BIGINT))) END, "
                     "try_strptime(orig_created_at, '%a %b %d %H:%M:%S %z %Y'), "
                     "try_strptime(orig_created_at, '%Y-%m-%dT%H:%M:%S.%gZ')) AS orig_ts"}


def projection(table: str, present: Sequence[str]) -> str:
    """Canonical order and types, selected by name; typed NULLs for columns a file lacks (as since 18 Sep)."""
    return ", ".join(f"CAST({n} AS {t}) AS {n}" if n in present else f"CAST(NULL AS {t}) AS {n}"
                     for n, t in TW_SCHEMA[table])


def twitter_source(path: str, table: str, present: Sequence[str]) -> str:
    keep = ", ".join(n for n, _ in TW_SCHEMA[table] if n not in DROPPED)
    flags = ", ".join(f"{R.name_has_bot_sql(src)} AS {dst}" for src, dst in NAME_FLAGS[table])
    return (f"SELECT {keep}, {flags}, {TW_TS[table]} FROM (SELECT {projection(table, present)} "
            f"FROM read_parquet('{path}')) _p WHERE {R.inclusion_predicate(TW_FULL[table])}")


def _lsjson(target: str, recursive: bool) -> list:
    cmd = [config.RCLONE_BIN, "lsjson", "--files-only", "--hash", "--hash-type", "md5", target]
    if recursive:
        cmd[2:2] = ["-R", "--fast-list"]
    last = ""
    for i in range(5):                       # the network drops connections now and then
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if res.returncode == 0:
            return json.loads(res.stdout)
        if "directory not found" in res.stderr.lower():
            return []
        last = res.stderr[-300:]
        time.sleep(15 * (i + 1))
    raise runner.CheckError(f"listing {target} failed: {last}")


def drive_listing(remote: str, days=None) -> Dict[str, Dict[str, object]]:
    """Path -> size and md5 of the Stage A objects (Drive keeps md5 checksums); only the given days if any."""
    out: Dict[str, Dict[str, object]] = {}
    if days:
        for d in days:
            rel_dir = f"{d.year}/date={d.isoformat()}"
            for o in _lsjson(f"{remote}/{rel_dir}", recursive=False):
                out[f"{rel_dir}/{o['Path']}"] = {"bytes": o["Size"], "md5": (o.get("Hashes") or {}).get("md5")}
        return out
    for o in _lsjson(remote, recursive=True):
        out[o["Path"]] = {"bytes": o["Size"], "md5": (o.get("Hashes") or {}).get("md5")}
    return out


def fetch(remote: str, rel: str, dst: Path, attempts: int = 4) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for i in range(attempts):
        res = subprocess.run([config.RCLONE_BIN, "copyto", "--retries", "3", "--low-level-retries", "20",
                              f"{remote}/{rel}", str(dst)], capture_output=True, text=True, timeout=7200)
        if res.returncode == 0 and dst.exists():
            return
        last = res.stderr[-400:]
        time.sleep(10 * (i + 1))
    raise runner.CheckError(f"fetch {rel} failed after {attempts} attempts: {last}")


def coarse(args) -> int:
    out_root = config.out_path("twitter", "coarse")
    start = date.fromisoformat(config.WINDOW_START)
    days = [start + timedelta(d) for d in range(config.N_DAYS)]
    if args.days:
        want = set(args.days)
        days = [d for d in days if d.isoformat() in want]
    if args.src_mode == "rclone":
        listing = drive_listing(args.src_root, days if args.days else None)
        config.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        (config.SCRATCH_DIR / "twitter_stage_a_listing.json").write_text(json.dumps(listing))
    else:
        root = Path(args.src_root)
        listing = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "md5": None} for p in root.rglob("*.parquet")}
    con = runner.connect()
    R.register_udfs(con)
    work = config.SCRATCH_DIR / "twitter_stage_a"
    n_done = n_absent = 0
    for d in days:
        rels = {t: f"{d.year}/date={d.isoformat()}/{t}.parquet" for t in ("tweets", "retweets")}
        present = {t: r for t, r in rels.items() if r in listing}
        final = out_root / f"date={d.isoformat()}"
        if not present:
            n_absent += 1
            continue
        inputs = [{"path": f"{args.src_root}/{r}", "bytes": listing[r]["bytes"], "md5": listing[r]["md5"]}
                  for _, r in sorted(present.items())]
        if runner.is_current(final, inputs, part_sha("coarse"), RULES_SHA):
            continue
        t0 = time.time()
        tmp = runner.tmp_path(final)
        tmp.mkdir(parents=True)
        checks: List[Dict[str, object]] = []
        counts: Dict[str, object] = {}
        local_files = []
        for table, rel in sorted(present.items()):
            if args.src_mode == "rclone":
                local = work / rel
                fetch(args.src_root, rel, local)
                local_files.append(local)
                if listing[rel]["md5"]:
                    check(runner.file_md5(local) == listing[rel]["md5"], f"{rel}: md5 equals the Drive checksum", checks)
            else:
                local = Path(args.src_root) / rel
            cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{local}')").fetchall()]
            extra = [("own_text_hit", R.own_coin_hit())] if table == "tweets" else []
            sql = R.annotate_sql(twitter_source(str(local), table, cols), TW_FULL[table], TW_OWN[table],
                                 extra_layer3=extra)
            outf = tmp / f"{table}.parquet"
            con.execute(f"COPY ({sql}) TO '{outf}' {PARQUET}")
            o = f"read_parquet('{outf}')"
            n_in, n_admit = con.execute(
                f"SELECT count(*), count_if({R.inclusion_predicate(TW_FULL[table])}) "
                f"FROM (SELECT {projection(table, cols)} FROM read_parquet('{local}'))").fetchone()
            n_out, n_ids, n_ts_null = con.execute(
                f"SELECT count(*), count(DISTINCT {TW_ID[table]}), count_if(ts IS NULL) FROM {o}").fetchone()
            check(n_out == n_admit, f"{table}: rows out = rows admitted by an independent count ({n_out:,} of {n_in:,})",
                  checks)
            check(null_evidence(con, o) == 0, f"{table}: no NULL in evidence or structure columns", checks)
            out_cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {o}").fetchall()]
            check(not (set(out_cols) & DROPPED), f"{table}: no screen-name column written", checks)
            counts[table] = {"rows_in": int(n_in), "rows_out": int(n_out), "distinct_ids": int(n_ids),
                             "ts_null": int(n_ts_null), "source_columns": len(cols),
                             "evidence": evidence_counts(con, o)}
        runner.commit(tmp, final)
        runner.write_stats(final, step=STEP + " coarse", inputs=inputs, code_sha=part_sha("coarse"), rules_sha=RULES_SHA,
                           con=con, counts=counts, checks=checks,
                           extra={"platform": "twitter", "partition": d.isoformat(), "seconds": round(time.time() - t0, 1)})
        for f in local_files:
            f.unlink()
        n_done += 1
        log(f"twitter {d}: " + ", ".join(f"{t} {c['rows_out']:,}/{c['rows_in']:,}" for t, c in counts.items())
            + f" in {time.time() - t0:.0f} s")
    con.close()
    log(f"twitter done: {n_done} days written this run, {n_absent} days without Stage A")
    return 0


# =====================================================================================
# Part 2, units: one row per tweet (the captures of a tweet merged by id)
# =====================================================================================

UNIT_COLS = ["unit_id", "role", "text", "parent_text", "parent_id", "parent_created_at", "author_id",
             "author_name_has_bot", "author_followers", "author_verified", "favorite_count", "lang", "authored_at",
             "panel_ts", "authored_before_window", "capture_day", "day_coverage_pct", "n_captures", "text_from_retweet",
             "n_text_variants", "n_as_retweet", "n_retweeting_accounts", "first_retweeted_at", "last_retweeted_at"]


def snowflake(col: str) -> str:
    """Creation time from a tweet id (exact to the millisecond; verified on all stored tweets, 22 Sep)."""
    return (f"CASE WHEN {col} IS NULL OR NOT regexp_matches({col}, '^[0-9]+$') THEN NULL "
            f"ELSE timezone('UTC', epoch_ms(CAST((CAST({col} AS HUGEINT) >> 22) + {SNOWFLAKE_EPOCH_MS} AS BIGINT))) END")


def units() -> int:
    """Part 2 (and part 3 at the end): see the module docstring."""
    src_root = config.out_path("twitter", "coarse")
    out_dir = config.out_path("twitter", "units")
    tw_files = sorted(glob.glob(str(src_root / "date=*" / "tweets.parquet")))
    rt_files = sorted(glob.glob(str(src_root / "date=*" / "retweets.parquet")))
    check(len(tw_files) > 0, "coarse Twitter day partitions found")
    inputs = runner.describe_inputs(tw_files + rt_files)
    final = out_dir / "units.parquet"
    if runner.is_current(final, inputs, part_sha("units"), RULES_SHA):
        log("units current, skipped")
        return 0
    t0 = time.time()
    # Intermediate tables live in a scratch database file, not in memory: the in-memory run needed more than
    # 9.3 GB for the full window (27 Sep); in a file DuckDB can move table blocks to disk. Same queries, same output.
    db = config.SCRATCH_DIR / "s2_twitter_units.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    for stale in (db, Path(str(db) + ".wal")):
        if stale.exists():
            stale.unlink()
    con = runner.connect(database=db)
    # Memory, not results: which row DISTINCT ON keeps is fixed by its ORDER BY, and every output is written with an
    # explicit ORDER BY, so the physical order of the intermediate tables need not be kept.
    con.execute("SET preserve_insertion_order = false")
    R.register_udfs(con)
    tw = f"read_parquet({tw_files!r})"
    rt = f"read_parquet({rt_files!r})"
    checks: List[Dict[str, object]] = []

    # ---- captures of one tweet collapse to one row (by id, never by text) ----------
    con.execute(f"""CREATE OR REPLACE TABLE T AS
        SELECT tweet_id, any_value(tweet_type) AS role, arg_max(text, length(text)) AS text,
               count(DISTINCT md5(text)) AS n_direct_texts, count(*) AS n_captures,
               min(ts) AS created_at, any_value(lang) AS lang, any_value(user_id) AS user_id,
               bool_or(user_name_has_bot) AS user_name_has_bot, any_value(user_followers) AS user_followers,
               any_value(user_verified) AS user_verified, any_value(favorite_count) AS favorite_count,
               min(day) AS capture_day, any_value(day_coverage_pct) AS day_coverage_pct,
               nullif(trim(any_value(quoted_text)), '') AS parent_text,
               coalesce(any_value(quoted_id), any_value(reply_to_id)) AS parent_id
        FROM {tw} WHERE text IS NOT NULL AND trim(text) <> '' GROUP BY tweet_id""")
    # Among equally long texts the earliest captured, then the first in sort order (decided 28 Sep 2026). Only a tweet
    # captured with more than one text can have such a tie, so the key is applied to those tweets only (a key that holds
    # the text a second time does not fit in memory over all of them); for every other tweet all copies are one text.
    con.execute(f"""UPDATE T SET text = x.text FROM (
        SELECT tweet_id, arg_min(text, (-length(text), ts, text)) AS text FROM {tw}
        WHERE text IS NOT NULL AND trim(text) <> '' AND tweet_id IN (SELECT tweet_id FROM T WHERE n_direct_texts > 1)
        GROUP BY tweet_id) x WHERE x.tweet_id = T.tweet_id""")
    # One row per retweet_id, the earliest capture (DISTINCT ON ... ORDER BY retweet_id, ts). Done in two parts because
    # a DISTINCT ON over all ~25M retweet rows with their texts does not fit in memory (27 Sep): ids captured more than
    # once are few; every other row passes unchanged, and DISTINCT ON runs on the few duplicated ids only.
    rt_cols = ("retweet_id, orig_tweet_id, retweeter_id, retweeter_name_has_bot, ts, orig_ts, orig_text, orig_author_id, "
               "orig_author_name_has_bot, lang")
    rt_rows = f"(SELECT {rt_cols} FROM {rt} WHERE orig_text IS NOT NULL AND trim(orig_text) <> '')"
    con.execute(f"""CREATE OR REPLACE TABLE RT_DUP AS
        SELECT retweet_id FROM {rt_rows} GROUP BY retweet_id HAVING count(*) > 1""")
    con.execute(f"""CREATE OR REPLACE TABLE RT AS
        SELECT r.* FROM {rt_rows} r ANTI JOIN RT_DUP d USING (retweet_id)
        UNION ALL
        SELECT * FROM (SELECT DISTINCT ON (r.retweet_id) r.* FROM {rt_rows} r SEMI JOIN RT_DUP d USING (retweet_id)
                       ORDER BY r.retweet_id, r.ts)""")

    con.execute("""CREATE OR REPLACE TABLE RA AS
        SELECT orig_tweet_id, count(*) AS n_as_retweet, count(DISTINCT retweeter_id) AS n_retweeting_accounts,
               min(orig_ts) AS orig_created_at, min(ts) AS first_retweeted_at, max(ts) AS last_retweeted_at,
               arg_max(orig_text, length(orig_text)) AS orig_text, count(DISTINCT md5(orig_text)) AS n_rt_texts,
               any_value(orig_author_id) AS orig_author_id, bool_or(orig_author_name_has_bot) AS orig_author_name_has_bot,
               any_value(lang) AS lang
        FROM RT GROUP BY orig_tweet_id""")
    con.execute("""UPDATE RA SET orig_text = x.orig_text FROM (
        SELECT orig_tweet_id, arg_min(orig_text, (-length(orig_text), ts, orig_text)) AS orig_text FROM RT
        WHERE orig_tweet_id IN (SELECT orig_tweet_id FROM RA WHERE n_rt_texts > 1) GROUP BY orig_tweet_id) x
        WHERE x.orig_tweet_id = RA.orig_tweet_id""")
    n_rt_rows, n_rt_distinct = con.execute(
        f"SELECT count(*), count(DISTINCT retweet_id) FROM {rt} WHERE orig_text IS NOT NULL AND trim(orig_text) <> ''").fetchone()
    log(f"tweets collapsed and retweets tallied in {time.time() - t0:.0f} s")

    # ---- units: authored tweets, then originals known only through retweets ---------
    start, end = config.WINDOW_START, config.WINDOW_END
    con.execute(f"""CREATE OR REPLACE TABLE U0 AS
        SELECT t.tweet_id AS unit_id, t.role,
               CASE WHEN length(r.orig_text) > length(t.text) THEN r.orig_text ELSE t.text END AS text,
               t.parent_text, t.parent_id, {snowflake('t.parent_id')} AS parent_created_at,
               t.user_id AS author_id, t.user_name_has_bot AS author_name_has_bot, t.user_followers AS author_followers,
               t.user_verified AS author_verified, t.favorite_count, t.lang,
               t.created_at AS authored_at, t.created_at AS panel_ts, FALSE AS authored_before_window,
               t.capture_day, t.day_coverage_pct, t.n_captures,
               coalesce(length(r.orig_text) > length(t.text), FALSE) AS text_from_retweet,
               CAST(NULL AS INTEGER) AS n_text_variants,
               coalesce(r.n_as_retweet, 0) AS n_as_retweet, coalesce(r.n_retweeting_accounts, 0) AS n_retweeting_accounts,
               r.first_retweeted_at, r.last_retweeted_at
        FROM T t LEFT JOIN RA r ON r.orig_tweet_id = t.tweet_id
        UNION ALL
        SELECT r.orig_tweet_id, 'retweet_only', r.orig_text, NULL, NULL, CAST(NULL AS TIMESTAMPTZ),
               r.orig_author_id, r.orig_author_name_has_bot, NULL, NULL, NULL, r.lang,
               r.orig_created_at, r.first_retweeted_at, CAST(r.orig_created_at AS DATE) < DATE '{start}',
               NULL, NULL, 0, FALSE, CAST(NULL AS INTEGER), r.n_as_retweet, r.n_retweeting_accounts,
               r.first_retweeted_at, r.last_retweeted_at
        FROM RA r WHERE NOT EXISTS (SELECT 1 FROM T t WHERE t.tweet_id = r.orig_tweet_id)""")

    # distinct texts observed per tweet (direct captures and retweet copies)
    con.execute(f"""CREATE OR REPLACE TABLE NV AS SELECT unit_id, CAST(count(DISTINCT h) AS INTEGER) AS n FROM (
        SELECT tweet_id AS unit_id, md5(text) AS h FROM {tw} WHERE text IS NOT NULL AND trim(text) <> ''
        UNION ALL SELECT orig_tweet_id, md5(orig_text) FROM RT) GROUP BY 1""")
    con.execute("UPDATE U0 SET n_text_variants = NV.n FROM NV WHERE NV.unit_id = U0.unit_id")

    # ---- evidence and structure on the final text (the rules of this step) ------------
    tmp = runner.tmp_path(final)
    extra = [("own_text_hit", R.own_coin_hit())]
    sql = R.annotate_sql("SELECT * FROM U0", "text || ' ' || coalesce(parent_text, '')", "text", extra_layer3=extra)
    con.execute(f"""COPY (SELECT *, text AS own_text, md5(text) AS text_md5,
                                md5(text || chr(31) || coalesce(parent_text, '')) AS text_key
                         FROM ({sql}) ORDER BY panel_ts, unit_id) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    o = f"read_parquet('{tmp}')"
    n_u, n_ids, n_ro, n_pre, ev = con.execute(
        f"SELECT count(*), count(DISTINCT unit_id), count_if(role = 'retweet_only'), count_if(authored_before_window), "
        f"sum(n_as_retweet) FROM {o}").fetchone()
    exp = con.execute("SELECT count(*) FROM (SELECT tweet_id FROM T UNION SELECT orig_tweet_id FROM RA)").fetchone()[0]
    check(n_u == n_ids == exp, f"one row per tweet id, ids = tweets and retweeted originals ({n_u:,})", checks)
    check(int(ev) == n_rt_distinct, f"retweet events counted once each ({n_rt_distinct:,} of {n_rt_rows:,} rows)", checks)
    out_of_window = con.execute(f"SELECT count_if(CAST(panel_ts AS DATE) NOT BETWEEN DATE '{start}' AND DATE '{end}') "
                                f"FROM {o}").fetchone()[0]
    check(out_of_window == 0, "every unit's panel day inside the window", checks)
    check(con.execute(f"SELECT count_if(author_id IS NULL) FROM {o}").fetchone()[0] == 0, "every unit has an author", checks)
    check(con.execute(f"SELECT count_if(authored_at IS NULL) FROM {o}").fetchone()[0] == 0,
          "every unit has its writing time (authored_at)", checks)

    # ---- text variants and retweet events -------------------------------------------
    tv = out_dir / "text_variants.parquet"
    tvt = runner.tmp_path(tv)
    con.execute(f"""COPY (
        WITH obs AS (SELECT tweet_id AS unit_id, text AS variant, 'tweet' AS source FROM {tw}
                     WHERE text IS NOT NULL AND trim(text) <> ''
                     UNION ALL SELECT orig_tweet_id, orig_text, 'retweet' FROM RT)
        SELECT o.unit_id, o.variant, list_distinct(list(o.source)) AS sources, count(*) AS n_captures
        FROM obs o JOIN {o} u ON u.unit_id = o.unit_id AND md5(u.text) <> md5(o.variant)
        GROUP BY 1, 2) TO '{tvt}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n_tv = con.execute(f"SELECT count(*) FROM read_parquet('{tvt}')").fetchone()[0]
    missing_texts = con.execute(f"""
        WITH src AS (SELECT DISTINCT md5(text) h FROM {tw} WHERE text IS NOT NULL AND trim(text) <> ''
                     UNION SELECT DISTINCT md5(orig_text) FROM RT
                     UNION SELECT DISTINCT md5(trim(quoted_text)) FROM {tw} WHERE quoted_text IS NOT NULL AND trim(quoted_text) <> ''),
             have AS (SELECT md5(text) h FROM {o} UNION SELECT md5(parent_text) FROM {o} WHERE parent_text IS NOT NULL
                      UNION SELECT md5(variant) FROM read_parquet('{tvt}'))
        SELECT count(*) FROM src ANTI JOIN have USING (h)""").fetchone()[0]
    check(missing_texts == 0, "every source text is a unit text, a parent text or a stored variant", checks)
    ed = out_dir / "retweet_events_daily.parquet"
    edt = runner.tmp_path(ed)
    con.execute(f"""COPY (
        SELECT orig_tweet_id AS unit_id, CAST(ts AS DATE) AS retweet_day, count(*) AS n_events,
               count(DISTINCT retweeter_id) AS n_accounts, CAST(count_if(retweeter_name_has_bot) AS BIGINT) AS n_events_retweeter_name_bot,
               min(ts) AS first_at, max(ts) AS last_at, min(orig_ts) AS authored_at
        FROM RT GROUP BY 1, 2 ORDER BY 2, 1) TO '{edt}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    ev_days = con.execute(f"SELECT sum(n_events) FROM read_parquet('{edt}')").fetchone()[0]
    check(int(ev_days) == n_rt_distinct, "retweet events per day add up to the distinct retweets", checks)
    bad = con.execute(f"""WITH e AS (SELECT unit_id, sum(n_events) n FROM read_parquet('{edt}') GROUP BY 1)
                          SELECT count(*) FROM e LEFT JOIN {o} u USING (unit_id) WHERE u.unit_id IS NULL OR e.n <> u.n_as_retweet"""
                      ).fetchone()[0]
    check(bad == 0, "per original, retweet events = n_as_retweet", checks)

    roles = {r: int(n) for r, n in con.execute(f"SELECT role, count(*) FROM {o} GROUP BY 1 ORDER BY 1").fetchall()}
    counts = {"units": int(n_u), "roles": roles, "retweet_only": int(n_ro), "authored_before_window": int(n_pre),
              "retweet_rows": int(n_rt_rows), "retweet_events": int(n_rt_distinct), "text_variants": int(n_tv),
              "text_from_retweet": int(con.execute(f"SELECT count_if(text_from_retweet) FROM {o}").fetchone()[0])}
    for p, t in ((tv, tvt), (ed, edt)):
        runner.commit(t, p)
        runner.write_stats(p, step=STEP + " units", inputs=inputs, code_sha=part_sha("units"), rules_sha=RULES_SHA, con=con,
                           counts={"rows": int(con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0])},
                           checks=[])
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP + " units", inputs=inputs, code_sha=part_sha("units"), rules_sha=RULES_SHA, con=con,
                       counts=counts, checks=checks, extra={"seconds": round(time.time() - t0, 1)})
    con.close()
    for scratch in (db, Path(str(db) + ".wal")):   # this run's scratch database
        if scratch.exists():
            scratch.unlink()
    log(f"units: {n_u:,} ({n_ro:,} retweet-only, {n_pre:,} of them written before the window) in {time.time() - t0:.0f} s")

    return 0


# =====================================================================================
# Part 3, coverage: captured minutes per UTC day, from the step-1 ledgers
# (carried over unchanged from build_twitter_utc_coverage.py, 22 Sep 2026)
# =====================================================================================

WINDOW = (dt.date(2019, 8, 1), dt.date(2023, 1, 31))
SWITCH = dt.date(2021, 8, 23)
MINUTES = 1440


def stem_date(stem: str) -> str:
    """twitter_stream_2020_04_15 / twitter-stream-2020-07-01 / twitter-stream-20210823 -> ISO date."""
    digits = "".join(ch for ch in stem if ch.isdigit())[:8]
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def hhmm_to_min(s: str) -> int:
    return int(s[:2]) * 60 + int(s[2:])


def coverage(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledgers", required=True, help="directory of *.stats.json ledgers")
    ap.add_argument("--out", required=True, help="output parquet path")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.ledgers, "*.stats.json")))
    if not files:
        raise SystemExit(f"no ledgers in {args.ledgers}")

    contrib = defaultdict(list)          # utc_day -> [(stem, minutes, first, last)]
    archive_day = {}                     # archive day label -> member-path pct
    n_ok = 0
    for f in files:
        led = json.load(open(f))
        if led.get("status") != "ok":
            continue
        n_ok += 1
        for d, v in (led.get("days_from_data") or {}).items():
            if v["minutes"]:
                contrib[d].append((led["stem"], v["minutes"], v["first"], v["last"]))
        # same rule as twitter_etl.py _coverage_for(): member-path coverage when the
        # layout has member paths, else the timestamp-derived coverage of the UTC day
        # equal to the archive's own date. The fallback fired for the Mar-Jun 2020
        # archives (unparseable member paths) and stamped the UTC slice 06:29-23:59
        # (~73%) onto a US-local partition that actually holds 1,411/1,440 minutes.
        if led.get("days"):
            for d, v in led["days"].items():
                archive_day.setdefault(d, v["pct"])
        else:
            own = stem_date(led["stem"])
            v = (led.get("days_from_data") or {}).get(own)
            if v:
                archive_day.setdefault(own, v["pct"])

    failures: List[str] = []
    rows = []
    day = WINDOW[0]
    while day <= WINDOW[1]:
        key = day.isoformat()
        c = contrib.get(key, [])
        # cluster contributions whose minute ranges overlap
        rng = sorted((hhmm_to_min(a), hhmm_to_min(b), m) for _, m, a, b in c)
        lo = hi = 0
        cl: list = []
        for a, b, m in rng + [(10**9, 10**9, 0)]:
            if cl and a > max(x[1] for x in cl):
                span = max(x[1] for x in cl) - min(x[0] for x in cl) + 1
                lo += max(x[2] for x in cl)
                hi += min(sum(x[2] for x in cl), span)
                cl = []
            cl.append((a, b, m))
        minutes, minutes_hi = lo, min(hi, MINUTES)
        if minutes > MINUTES:
            failures.append(f"{key}: {minutes} minutes > {MINUTES}")
        if minutes_hi - minutes > 5:
            failures.append(f"{key}: bounds {minutes}..{minutes_hi} differ by > 5 minutes")
        rows.append({
            "utc_day": day,
            "minutes_utc": minutes,
            "minutes_utc_hi": minutes_hi,
            "coverage_pct_utc": round(100 * minutes / MINUTES, 2),
            "n_archives": len(c),
            "archives": [s for s, _, _, _ in c],
            "first_hhmm": min((a for _, _, a, _ in c), default=None),
            "last_hhmm": max((b for _, _, _, b in c), default=None),
            "archive_day_pct": archive_day.get(key),
            "pre_switch": day < SWITCH,
        })
        day += dt.timedelta(days=1)

    post = [r for r in rows if not r["pre_switch"] and r["archive_day_pct"] is not None]
    agree = sum(abs(r["coverage_pct_utc"] - r["archive_day_pct"]) <= 0.5 for r in post)
    print(f"ledgers: {len(files)} files, {n_ok} ok")
    print(f"post-switch days: {agree:,}/{len(post):,} agree with archive-day coverage within 0.5 pp")
    if post and agree / len(post) < 0.99:
        failures.append(f"post-switch agreement {agree}/{len(post)} below 99%")
    gap = max(r["minutes_utc_hi"] - r["minutes_utc"] for r in rows)
    print(f"largest lower/upper bound gap: {gap} minutes")

    if failures:
        print("FAILED, output not written:\n  " + "\n  ".join(failures[:20]), file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
    os.replace(tmp, args.out)
    present = sum(r["minutes_utc"] > 0 for r in rows)
    print(f"{len(rows):,} UTC days, {present:,} with any capture -> {args.out}")
    return 0


def coverage_part() -> int:
    """Part 3 with a stats file (28 Sep 2026): the coverage table is rebuilt whenever a capture
    ledger or this part's code changes, and left as it is otherwise."""
    final = config.out_path("twitter", "units", "twitter_coverage_utc.parquet")
    ledgers = sorted(Path(config.TWITTER_LEDGER_DIR).glob("*.stats.json"))
    check(len(ledgers) > 0, f"capture ledgers found in {config.TWITTER_LEDGER_DIR}")
    inputs = runner.describe_inputs(ledgers)
    if runner.is_current(final, inputs, part_sha("coverage"), RULES_SHA):
        log("coverage: current, nothing to do")
        return 0
    t0 = time.time()
    tmp = runner.tmp_path(final)
    check(coverage(["--ledgers", str(config.TWITTER_LEDGER_DIR), "--out", str(tmp)]) == 0,
          "coverage table written (its own checks passed)")
    con = runner.connect()
    o = f"read_parquet('{tmp}')"
    checks: List[Dict[str, object]] = []
    n, n_days, lo, hi = con.execute(f"SELECT count(*), count(DISTINCT utc_day), min(utc_day), max(utc_day) FROM {o}").fetchone()
    check(n == n_days == config.N_DAYS, f"{config.N_DAYS} UTC days, one row each", checks)
    check((str(lo), str(hi)) == (str(config.WINDOW_START), str(config.WINDOW_END)), "exactly the study window", checks)
    bad = con.execute(f"SELECT count_if(minutes_utc < 0 OR minutes_utc > minutes_utc_hi OR minutes_utc_hi > {MINUTES}) "
                      f"FROM {o}").fetchone()[0]
    check(bad == 0, "0 <= minutes_utc <= minutes_utc_hi <= 1,440 on every day", checks)
    full, partial, none_, eff = con.execute(
        f"SELECT count_if(coverage_pct_utc >= 95), count_if(minutes_utc > 0 AND coverage_pct_utc < 95), "
        f"count_if(minutes_utc = 0), round(100.0 * sum(minutes_utc) / (count(*) * {MINUTES}), 2) FROM {o}").fetchone()
    n_ok = sum(json.loads(Path(f).read_text()).get("status") == "ok" for f in ledgers)
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP + " coverage", inputs=inputs, code_sha=part_sha("coverage"), rules_sha=RULES_SHA,
                       con=con, counts={"days": int(n), "ledgers": len(ledgers), "ledgers_ok": int(n_ok),
                                        "days_at_least_95pct": int(full), "days_partial": int(partial),
                                        "days_without_capture": int(none_), "effective_coverage_pct": float(eff)},
                       checks=checks, extra={"seconds": round(time.time() - t0, 1)})
    con.close()
    log(f"coverage: {n:,} UTC days, {full:,} at >= 95%, {partial:,} partial, {none_:,} without capture")
    return 0


# =====================================================================================
# Fingerprints and command line
# =====================================================================================

def part_sha(part: str) -> str:
    """Fingerprint of the code one part runs, so that an edit to the units part leaves the coarse days
    current (re-fetching them all from Drive takes hours)."""
    import hashlib
    import inspect
    own = {"coarse": [projection, twitter_source, fetch, coarse],
           "units": [snowflake, units],
           "coverage": [coverage, stem_date, hhmm_to_min, coverage_part]}[part]
    consts = {"coarse": [TWEETS_SCHEMA, RETWEETS_SCHEMA, NAME_FLAGS, sorted(DROPPED), TW_FULL, TW_OWN, TW_ID, TW_TS,
                         SNOWFLAKE_EPOCH_MS, PARQUET],
              "units": [UNIT_COLS, SNOWFLAKE_EPOCH_MS],
              "coverage": [WINDOW, SWITCH, MINUTES]}[part]
    parts = [runner.code_fingerprint(filtering.__file__)] + [inspect.getsource(f) for f in own] + [repr(c) for c in consts]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="part", required=True)
    c = sub.add_parser("coarse", help="part 1: Stage A day by day -> coarse day partitions")
    c.add_argument("--src-mode", choices=("rclone", "local"), default="rclone")
    c.add_argument("--src-root", default=config.TWITTER_STAGE_A_REMOTE)
    c.add_argument("--days", nargs="*", default=None)
    sub.add_parser("units", help="part 2, one row per tweet, retweet events, text variants; then part 3")
    sub.add_parser("coverage", help="part 3 alone: the coverage table, rebuilt when a capture ledger changes")
    args = ap.parse_args(argv)
    log(f"{STEP} {args.part}: code {part_sha(args.part)}, rules {RULES_SHA}")
    if args.part == "coarse":
        return coarse(args)
    if args.part == "units":
        return units() or coverage_part()
    return coverage_part()


if __name__ == "__main__":
    raise SystemExit(main())
