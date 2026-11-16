#!/usr/bin/env python3
"""
01_extract_stocktwits.py -- step 1 for StockTwits: the full-platform dump -> one Parquet file per month.

Input    the stocktwits-nyu dump: the platform history 2008-05-27 to 2024-01-02 as flat CSV
         files (749 files, 122.5 GB), not filtered by date or coin.
Output   month=YYYY-MM/data_0.parquet for the 42 months of the window ("Stage A"),
         plus _ledger.json with every count of the run. The filter step
         (02_filter_stocktwits.py) starts from these files.
Run      once, on 17 Sep 2026, on a VM, with a DuckDB memory limit of 93 GB (its ledger; the
         default below is 48 GB).
         Its output is kept on Google Drive (stocktwits_parquet); the raw dump is
         restricted and cannot be redistributed.

What this step does, in order
-----------------------------
1. sync_raw       copies the three folders it needs from the remote copy of the dump
                  and checks file count and bytes (never trusting a transfer that ends
                  without error). Three further folders are not needed: their columns
                  are all in feature_wo_messages (symbols/ is symbol_list with one row
                  per ticker; the two sentiment folders are subsets of tagged messages
                  with a date-only created_at), so leaving out their 22.8 GB loses no
                  information.
                      messages/             205 files  55.0 GB  message_id, message_body
                      feature_wo_messages/  248 files  30.8 GB  message_id, user_id, created_at,
                                                                sentiment, parent_message_id,
                                                                in_reply_to_message_id, symbol_list
                      msg_info/              66 files  14.0 GB  message_id, length, important_words
2. build_parquet  DuckDB, each folder read exactly once:
                  a. base table: one scan of feature_wo_messages, only rows with
                     created_at inside 2019-08-01 00:00:00 .. 2023-01-31 23:59:59 UTC
                     (a row-level filter: the files are not ordered by date), one row
                     per message_id (duplicates removed on this narrow table, never on text);
                  b. one COPY: base LEFT JOIN messages LEFT JOIN msg_info, split by month
                     (PARTITION_BY);
                  c. each month's row count is checked against base. More rows means
                     duplicate ids in messages or msg_info: only that month is rewritten
                     without them. Fewer rows cannot happen with a LEFT JOIN and stops
                     the run.
3. upload_verified  copies the months to the destination and checks every remote file
                  size against the local one before anything local is deleted.

What is deliberately NOT done here
----------------------------------
No coin or keyword decision is taken, and symbol_list is not resolved. symbol_list holds
the $tickers the author typed, as StockTwits parsed them, so it matches the text
(of 737,462 rows with symbols but no cashtag found by the regular expression, only 114
contain no "$"). A symbol shows that a post names a coin, not that it is about it; the
coin decision belongs to the later steps (02 and 04). symbol_list and important_words stay
raw stringified Python lists ("['BTC.X', 'AAPL']").

Output columns
--------------
    message_id, created_at, user_id, message_body, sentiment, parent_message_id,
    in_reply_to_message_id, symbol_list, length, important_words
(month is in the folder name). length and important_words are dropped in step 2.

Provenance note
---------------
The months on Drive were written on 17 Sep 2026 by a copy of this code on the VM with
the same data SQL. That copy uploaded month by month; this file uploads the whole tree
in one rclone call (faster, the same files). Remote names are placeholders: pass the real
rclone remote and path on the command line.

Decisions: StockTwits defect map PJ8 (symbols are author-typed); DOWNLOAD_STATUS and
VM_RUN_29AUG (source layout, measured 17 Sep 2026).

Usage
-----
    python pipeline/01_extract_stocktwits.py --raw-dir raw --out-dir stocktwits_parquet \
        --src-remote <remote>:<path>/stocktwits_nyu_full --upload-to <remote>:<path>/stocktwits_parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("01_extract_stocktwits")

WINDOW_START = "2019-08-01 00:00:00"
WINDOW_END = "2023-01-31 23:59:59"
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

DISCARD_FOLDERS = ("symbols", "sentiments", "symbol_sentiments")
KEEP_FOLDERS = ("feature_wo_messages", "messages", "msg_info")


@dataclass
class Config:
    raw_dir: str = "raw"
    out_dir: str = "stocktwits_parquet"
    work_dir: str = "duckdb_work"        # duckdb database file + spill; never uploaded
    # Placeholders: pass the real rclone remote/path on the command line.
    src_remote: str = "<remote>:<path>/stocktwits_nyu_full"
    upload_to: Optional[str] = "<remote>:<path>/stocktwits_parquet"
    threads: int = 0                     # 0 = all cores
    memory_limit: str = "48GB"
    # --drive-chunk-size 64M: the default 8 MiB means ~60 sequential chunk
    # requests for a 500 MB month. 64M matched the 31 MB/s the Twitter sync
    # sustained on this same account.
    rclone_extra: tuple = ("--transfers", "8", "--checkers", "8",
                            "--drive-chunk-size", "64M",
                            "--tpslimit", "8", "--tpslimit-burst", "10",
                            "--drive-pacer-min-sleep", "150ms",
                            "--drive-pacer-burst", "20",
                            "--retries", "5",
                            "--low-level-retries", "20", "--timeout", "120s",
                            "--contimeout", "30s", "--stats", "30s",
                            "--stats-one-line", "-v")


class RcloneError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------
def _rclone(args: list) -> None:
    # stderr is NOT captured: rclone's --stats lines must reach the run log, or
    # an hour-long download looks identical to a wedge (quirks §3).
    p = subprocess.run(["rclone", *args], stdout=subprocess.DEVNULL)
    if p.returncode != 0:
        raise RcloneError(f"rclone {' '.join(args[:3])} rc={p.returncode}")


def _remote_size(path: str) -> dict:
    p = subprocess.run(["rclone", "size", "--json", path], capture_output=True, text=True)
    if p.returncode != 0:
        raise RcloneError(f"rclone size {path} rc={p.returncode}: {p.stderr[-300:]}")
    return json.loads(p.stdout)


def _local_size(path: Path) -> dict:
    files = [f for f in path.rglob("*") if f.is_file()]
    return {"count": len(files), "bytes": sum(f.stat().st_size for f in files)}


def sync_raw(cfg: Config) -> dict:
    """Copy the three kept folders and verify count+bytes (quirks §7: never
    trust the exit code alone)."""
    out = {}
    for folder in KEEP_FOLDERS:
        dest = Path(cfg.raw_dir) / folder
        dest.mkdir(parents=True, exist_ok=True)
        src = f"{cfg.src_remote}/{folder}"
        log.info("rclone copy %s -> %s", src, dest)
        t0 = time.perf_counter()
        _rclone(["copy", src, str(dest), *cfg.rclone_extra])
        remote, local = _remote_size(src), _local_size(dest)
        if (remote["count"], remote["bytes"]) != (local["count"], local["bytes"]):
            raise RcloneError(f"{folder}: remote {remote['count']} files/{remote['bytes']} B "
                              f"!= local {local['count']}/{local['bytes']}")
        secs = time.perf_counter() - t0
        log.info("  %s verified: %d files, %.1f GB, %.0fs", folder, local["count"],
                 local["bytes"] / 1e9, secs)
        out[folder] = {**local, "seconds": round(secs)}
    return out


# ---------------------------------------------------------------------------
# DuckDB pipeline
# ---------------------------------------------------------------------------
def _month_files(out: Path, mo: str) -> list:
    return sorted((out / f"month={mo}").glob("*.parquet"))


def _pq_list(files: list) -> str:
    return "[" + ", ".join(f"'{f}'" for f in files) + "]"


def build_parquet(cfg: Config) -> dict:
    import duckdb

    out, work = Path(cfg.out_dir), Path(cfg.work_dir)
    if any(out.glob("month=*")):
        raise RuntimeError(f"{out} already holds month=* partitions; move them away before re-running")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "stocktwits.duckdb"
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    if cfg.threads:
        con.execute(f"SET threads={cfg.threads}")
    con.execute(f"SET memory_limit='{cfg.memory_limit}'")
    con.execute(f"SET temp_directory='{work / 'spill'}'")
    con.execute("SET preserve_insertion_order=false")   # lets COPY stream instead of buffering

    raw = Path(cfg.raw_dir)
    # Tested on real files (msg_200, msg_info_60, feature_wo_messages_200) against
    # Python's csv module as ground truth: identical row sets. Every option is load-bearing:
    #  - strict_mode=false: ~76 bodies per messages file contain a bare \r outside quotes;
    #    in strict mode the parser aborts the WHOLE file even with ignore_errors.
    #  - explicit quote/escape: with auto-detection, 3,724 bodies per file silently kept
    #    doubled quotes ("" instead of ").
    #  - new_line='\n': \r inside quoted bodies must not be read as a row break.
    csv_opts = ("header = true, ignore_errors = true, strict_mode = false, "
                "quote = '\"', escape = '\"', new_line = '\\n'")
    # EVERY column is read as VARCHAR. With a typed column, ignore_errors drops the
    # whole ROW on a cast failure, silently. Measured on feature_wo_messages_200:
    # declaring sentiment DOUBLE discarded all 655,351 'Bullish'/'Bearish'-tagged
    # rows (32% of the file). Casting happens below with TRY_CAST, and every
    # failure is counted.
    def view(name, folder, cols):
        spec = ", ".join(f"'{c}': 'VARCHAR'" for c in cols)
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv('{raw / folder / '*.csv'}', "
                    f"columns = {{{spec}}}, {csv_opts})")
    view("feature_raw", "feature_wo_messages", ["message_id", "user_id", "created_at", "sentiment",
                                               "parent_message_id", "in_reply_to_message_id", "symbol_list"])
    view("messages_str", "messages", ["message_id", "message_body"])
    view("msginfo_str", "msg_info", ["message_id", "length", "important_words"])
    con.execute("CREATE VIEW messages_raw AS SELECT TRY_CAST(message_id AS BIGINT) AS message_id, "
                "message_body FROM messages_str")
    con.execute("CREATE VIEW msginfo_raw AS SELECT TRY_CAST(message_id AS BIGINT) AS message_id, "
                "TRY_CAST(length AS INTEGER) AS length, important_words FROM msginfo_str")

    # --- 1. base table: the ONLY scan of feature_wo_messages ----------------
    # Every parseable in-window row plus every row with a cast problem is kept
    # one step longer, so the problems are counted without a second 30 GB scan.
    t0 = time.perf_counter()
    con.execute(f"""
        CREATE TABLE base_all AS
        SELECT TRY_CAST(message_id AS BIGINT)                          AS message_id,
               TRY_CAST(user_id AS BIGINT)                             AS user_id,
               try_strptime(created_at, '{TS_FMT}')                    AS created_at,
               sentiment,                                               -- 'Bullish' | 'Bearish' | NULL
               TRY_CAST(TRY_CAST(parent_message_id AS DOUBLE) AS BIGINT)      AS parent_message_id,
               TRY_CAST(TRY_CAST(in_reply_to_message_id AS DOUBLE) AS BIGINT) AS in_reply_to_message_id,
               symbol_list,
               (message_id IS NOT NULL AND TRY_CAST(message_id AS BIGINT) IS NULL)          AS bad_message_id,
               (user_id IS NOT NULL AND TRY_CAST(user_id AS BIGINT) IS NULL)                AS bad_user_id,
               (parent_message_id IS NOT NULL AND TRY_CAST(parent_message_id AS DOUBLE) IS NULL
                OR in_reply_to_message_id IS NOT NULL AND TRY_CAST(in_reply_to_message_id AS DOUBLE) IS NULL)
                                                                                            AS bad_reply_ids
        FROM feature_raw
        WHERE try_strptime(created_at, '{TS_FMT}') IS NULL
           OR TRY_CAST(message_id AS BIGINT) IS NULL
           OR try_strptime(created_at, '{TS_FMT}') BETWEEN TIMESTAMP '{WINDOW_START}'
                                                       AND TIMESTAMP '{WINDOW_END}'
    """)
    quality = dict(zip(
        ["unparseable_created_at", "bad_message_id", "bad_user_id", "bad_reply_ids",
         "sentiment_bullish", "sentiment_bearish", "sentiment_other"],
        con.execute("""
            SELECT count(*) FILTER (created_at IS NULL),
                   count(*) FILTER (bad_message_id OR message_id IS NULL),
                   count(*) FILTER (bad_user_id), count(*) FILTER (bad_reply_ids),
                   count(*) FILTER (valid AND sentiment = 'Bullish'),
                   count(*) FILTER (valid AND sentiment = 'Bearish'),
                   count(*) FILTER (valid AND sentiment IS NOT NULL AND sentiment NOT IN ('Bullish', 'Bearish'))
            FROM (SELECT *, created_at IS NOT NULL AND message_id IS NOT NULL AS valid FROM base_all)
        """).fetchone()))
    in_window = con.execute(
        "SELECT count(*) FROM base_all WHERE created_at IS NOT NULL AND message_id IS NOT NULL").fetchone()[0]
    con.execute("""
        CREATE TABLE base AS
        SELECT * EXCLUDE (bad_message_id, bad_user_id, bad_reply_ids) FROM base_all
        WHERE created_at IS NOT NULL AND message_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY message_id ORDER BY created_at) = 1
    """)
    con.execute("DROP TABLE base_all")
    base_rows = con.execute("SELECT count(*) FROM base").fetchone()[0]
    log.info("base: %s in-window rows, %s duplicate ids removed (%.0fs); quality %s",
             f"{in_window:,}", f"{in_window - base_rows:,}", time.perf_counter() - t0, quality)
    if quality["sentiment_other"]:
        log.warning("%s rows carry a sentiment value other than Bullish/Bearish — kept verbatim",
                    f"{quality['sentiment_other']:,}")
    expected = dict(con.execute(
        "SELECT strftime(created_at, '%Y-%m'), count(*) FROM base GROUP BY 1").fetchall())

    # --- 2. one join, one write ---------------------------------------------
    t0 = time.perf_counter()
    con.execute(f"""
        COPY (
            SELECT b.message_id, b.created_at, b.user_id, m.message_body,
                   b.sentiment, b.parent_message_id, b.in_reply_to_message_id,
                   b.symbol_list, mi.length, mi.important_words,
                   strftime(b.created_at, '%Y-%m') AS month
            FROM base b
            LEFT JOIN messages_raw m  USING (message_id)
            LEFT JOIN msginfo_raw  mi USING (message_id)
        ) TO '{out}' (FORMAT PARQUET, PARTITION_BY (month), COMPRESSION ZSTD)
    """)
    log.info("join + partitioned write done (%.0fs)", time.perf_counter() - t0)

    # --- 3. verify per month; repair only months with join fan-out ----------
    months, total_rows, total_unmatched, fanout_removed = {}, 0, 0, 0
    written = sorted(p.name.split("=", 1)[1] for p in out.glob("month=*") if p.is_dir())
    if set(written) != set(expected):
        raise RuntimeError(f"partition mismatch: written-only {set(written) - set(expected)}, "
                           f"expected-only {set(expected) - set(written)}")
    for mo in written:
        files = _month_files(out, mo)
        got = con.execute(f"SELECT count(*) FROM read_parquet({_pq_list(files)})").fetchone()[0]
        exp = expected[mo]
        if got < exp:
            raise RuntimeError(f"month={mo}: {got} rows < base {exp} — rows lost in the join")
        if got > exp:
            fixed = out / f"month={mo}" / "dedup.parquet.tmp"
            con.execute(f"""
                COPY (SELECT * FROM read_parquet({_pq_list(files)}, hive_partitioning = false)
                      QUALIFY row_number() OVER (PARTITION BY message_id ORDER BY message_body NULLS LAST) = 1)
                TO '{fixed}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
            for f in files:
                f.unlink()
            final = out / f"month={mo}" / "data_0.parquet"
            fixed.rename(final)
            files = [final]
            new = con.execute(f"SELECT count(*) FROM read_parquet('{final}')").fetchone()[0]
            if new != exp:
                raise RuntimeError(f"month={mo}: dedup left {new} rows, base has {exp}")
            log.warning("month=%s: removed %s join fan-out duplicates", mo, f"{got - exp:,}")
            fanout_removed += got - exp
        unmatched, d0, d1 = con.execute(f"""
            SELECT count(*) FILTER (message_body IS NULL), min(created_at)::DATE, max(created_at)::DATE
            FROM read_parquet({_pq_list(files)}, hive_partitioning = false)""").fetchone()
        size = sum(f.stat().st_size for f in files)
        months[mo] = {"rows": exp, "unmatched_text": unmatched, "first_day": str(d0),
                      "last_day": str(d1), "bytes": size, "files": [str(f) for f in files]}
        total_rows += exp
        total_unmatched += unmatched
        log.info("month=%s rows=%s unmatched_text=%s %s..%s %.2f GB",
                 mo, f"{exp:,}", f"{unmatched:,}", d0, d1, size / 1e9)
    con.close()

    if total_rows != base_rows:
        raise RuntimeError(f"row total {total_rows} != base {base_rows}")
    return {
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "discarded_folders": list(DISCARD_FOLDERS),
        "kept_folders": list(KEEP_FOLDERS),
        "feature_rows_in_window_before_dedup": in_window,
        "feature_duplicate_ids_removed": in_window - base_rows,
        "feature_quality": quality,
        "join_fanout_duplicates_removed": fanout_removed,
        "rows_written_total": total_rows,
        "rows_unmatched_text_total": total_unmatched,
        "months": months,
    }


# ---------------------------------------------------------------------------
# Upload + verify (like _upload_verified in 01_extract_twitter.py)
# ---------------------------------------------------------------------------
def upload_verified(ledger: dict, cfg: Config) -> None:
    """ONE rclone invocation for the whole tree, not one per month.

    Per-month copies measured 5.3 min/month against 35 MiB/s (whole tree in ~6
    min) for the bulk copy. Two reasons, both fatal to the per-month loop:
    Drive resolves paths by walking parent->child ids, so every cold rclone
    start re-walked <path>/stocktwits_parquet/month=... and created the
    leaf before moving a byte; and with one file per invocation --transfers had
    nothing to parallelise. Neither cost is visible in the byte counters, which
    is why the stall looked like a Drive-side throttle.
    """
    dest_root = cfg.upload_to.rstrip("/")
    _rclone(["copy", str(Path(cfg.out_dir)), dest_root,
             "--exclude", "_ledger.json", *cfg.rclone_extra])
    for mo, info in ledger["months"].items():
        for f in map(Path, info["files"]):
            p = subprocess.run(["rclone", "lsjson", f"{dest_root}/month={mo}/{f.name}"],
                               capture_output=True, text=True)
            try:
                remote_bytes = json.loads(p.stdout)[0]["Size"]
            except Exception:
                raise RcloneError(f"verify month={mo}/{f.name}: not found or unreadable lsjson")
            if remote_bytes != f.stat().st_size:
                raise RcloneError(f"verify month={mo}/{f.name}: remote {remote_bytes} != local {f.stat().st_size}")
    log.info("all %d months verified byte-exact on Drive", len(ledger["months"]))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="stocktwits-nyu -> per-month Parquet (Stage A)")
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out-dir", default="stocktwits_parquet")
    ap.add_argument("--work-dir", default="duckdb_work")
    ap.add_argument("--src-remote", default="<remote>:<path>/stocktwits_nyu_full",
                    help="rclone remote holding the stocktwits-nyu dump")
    ap.add_argument("--upload-to", default="<remote>:<path>/stocktwits_parquet",
                    help="rclone destination for the per-month parquet")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--skip-download", action="store_true",
                    help="raw/ already populated and verified (resume after an interruption)")
    ap.add_argument("--delete-local-raw", action="store_true",
                    help="delete raw CSVs once the upload is verified")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--memory-limit", default="48GB")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = Config(raw_dir=a.raw_dir, out_dir=a.out_dir, work_dir=a.work_dir,
                 src_remote=a.src_remote, upload_to=None if a.no_upload else a.upload_to,
                 threads=a.threads, memory_limit=a.memory_limit)
    t_start = time.perf_counter()
    download = {} if a.skip_download else sync_raw(cfg)
    ledger = build_parquet(cfg)
    ledger.update(download=download, config=asdict(cfg))

    ledger_path = Path(cfg.out_dir) / "_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, default=str))
    if cfg.upload_to:
        upload_verified(ledger, cfg)
        _rclone(["copy", str(ledger_path), f"{cfg.upload_to.rstrip('/')}/_ledger", *cfg.rclone_extra])
        shutil.rmtree(cfg.work_dir, ignore_errors=True)
        if a.delete_local_raw:
            shutil.rmtree(cfg.raw_dir, ignore_errors=True)

    log.info("finished in %.0f min", (time.perf_counter() - t_start) / 60)
    print(json.dumps({k: ledger[k] for k in ("rows_written_total", "rows_unmatched_text_total",
                                             "feature_duplicate_ids_removed",
                                             "join_fanout_duplicates_removed")}
                     | {"months": len(ledger["months"])}, indent=2))
