#!/usr/bin/env python3
"""
01_extract_twitter.py -- step 1 for Twitter: Internet Archive "Twitter Stream Grab" -> one Parquet file per day and table.

Input    the archives of the Twitter 1% sample stream kept by the Internet Archive
         (a mirror of 3.39 TB), one archive per day, each holding one file per minute.
Output   YYYY/date=YYYY-MM-DD/tweets.parquet and retweets.parquet ("Stage A"), plus one
         ledger per archive (_ledger/<archive>.stats.json) with every count of its run.
         The filter step (02_filter_twitter.py) starts from these files; the capture
         coverage table is built from the ledgers.
Run      once, 16-18 Sep 2026, on VMs; the output is kept on Google Drive (twitter_parquet).

What this step does with one archive, in order (process_archive_v1 / process_archive_v2)
----------------------------------------------------------------------------------------
1. acquire        copies the archive to local disk (zip files always: their table of
                  contents sits at the end, so they cannot be read from a stream), or
                  streams it; a transfer that ends with an error code is a failure,
                  never a short day.
2. iter_members   walks every file of the archive. parse_member reads date and minute
                  from the file's PATH, in the two layouts the mirror uses:
                      A  YYYY/MM/DD/HH/MM.json.bz2          (bzip2)
                      B  YYYYMMDD/YYYYMMDDHHMMSS.json.gz    (gzip)
3. prefilter      a byte pattern skips lines that cannot be English or undetermined
                  before any JSON parsing; it tolerates both spacings of the JSON
                  ({"lang":"en"} and {"lang": "en"}). validate_prefilter, a check run by
                  hand on sample archives (not part of a run), confirms that the pattern
                  keeps every line the parser keeps.
4. parse          keeps lang in ("en", "und"), splits content tweets (originals,
                  replies, quotes) from retweet events, takes the full text of long
                  tweets (get_full_text, v1), and writes each row to the file of its day
                  (DayWriters: one Parquet file per day and table, because one archive
                  is not one UTC day).
5. coverage       counts the minute-files seen per day (1,440 = complete; short days are
                  normal, e.g. 1,411 on 2020-12-14) and writes day_coverage_pct once
                  the whole archive has been read; the ledger also records the captured
                  minutes per UTC day (days_from_data), from which 02_filter_twitter.py
                  builds the UTC coverage table.
6. upload         copies every output and checks each remote size against the local
                  one before anything local is deleted.
No coin or relevance decision is taken here (enrich_row is an intentionally empty hook).
Screen names are written in Stage A; 02_filter_twitter.py turns them into name flags and
never writes them.

Two JSON formats, two parsers
-----------------------------
v1   Twitter API v1.1 JSON: every archive up to 2022-11-22. A tweet is one flat object
     (id_str, nested user); a retweet carries the full original (retweeted_status).
v2   Twitter API v2 JSON: the 49 archives from 2022-12-13 to 2023-01-30. Everything is
     wrapped in `data` (id, author_id, no inline user); a retweet points at an id
     that is resolved through includes.tweets / includes.users, and created_at is an
     ISO 8601 string (_iso_to_ms). The mirror holds no archive from 2022-11-23 to
     2022-12-12.
Both write the same columns, so the two eras union cleanly. The helpers both formats
share (archive names, member paths, decompression, remote transfer) appear once; the
functions that differ carry the suffix _v1 or _v2 (ConfigV1, DayWritersV1, ...).

Special cases of the archive, in the order the code meets them
--------------------------------------------------------------
* Three file-name conventions, not overlapping in date:
      twitter_stream_YYYY_MM_DD.tar   2019-08-01 .. 2020-06-30   (311 files)
      twitter-stream-YYYY-MM-DD.zip   2020-07-01 .. 2021-08-22   (382 files)
      twitter-stream-YYYYMMDD.tar     2021-08-23 .. 2023-01-30   (487 files)
  Until 2021-08-22 an archive day is a US Mountain-time day; from 2021-08-23 it is a
  UTC day. Output is partitioned by the date in the MEMBER PATH, never by the name.
* Names that lie: twitter-stream-20221212.tar.gz is one 47.5 GB bundle holding
  2022-12-13 to 2022-12-31; twitter_stream_2020_04_31 and _2020_06_31 are scripted
  dates without real content (IMPOSSIBLE_V1). Archives under 1 MiB are empty shells
  (10 in the window) and are recorded, not parsed.
* Truncated members: one archive, 2020-11-15 (9 of 1,294 minute-files cut short). The
  code keeps what was read and counts the member (truncated_members); the day has
  89.9% coverage.

Provenance note
---------------
Stage A was written by copies of this code on the VMs (16 Sep 2026). Two fixes were made
on 18 Sep and are in this file: the handler for truncated members, and reading back with
ParquetFile().read() instead of pq.read_table() (read_table can add a spurious `date`
column from the folder name on some pyarrow versions; not on the VM's 25.0.1). Neither
changes the output of any archive the VMs processed: the handler acts only on a truncated
member, and the one such archive, 2020-11-15, failed on the VMs and was rebuilt on
18 Sep with the handler (a run without a log). On the VMs, the large archives were split
into one archive per day before processing (parallelisation only), and a one-time
back-fill added day_coverage_pct, read from each archive's ledger, to files written
before the column existed; this file writes the column directly. Remote names are
placeholders: pass the real rclone remote and path on the command line.

Usage
-----
    python pipeline/01_extract_twitter.py v1 <archives or rclone remotes ...> --out <dir> --upload-to <remote>:<path>
    python pipeline/01_extract_twitter.py v2 <archives or rclone remotes ...> --out <dir> --upload-to <remote>:<path>
"""
from __future__ import annotations

import argparse
import bz2
import contextlib
import datetime as dt
import gzip
import io
import json
import logging
import os
import re
import subprocess
import tarfile
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson as _json
    JSON_LIB = "orjson"
except ImportError:                                   # pragma: no cover
    import json as _json
    JSON_LIB = "json"
_loads = _json.loads

log = logging.getLogger("01_extract_twitter")

EXPECTED_MEMBERS_PER_DAY = 1440
SHELL_BYTES = 1 << 20          # <1 MiB  => "empty shell" (10 such files in window)
# Archive names with a date that does not exist (scripted-date artifacts, no real content).
IMPOSSIBLE_V1 = {"twitter_stream_2020_04_31", "twitter_stream_2020_06_31",
              "twitter-stream-2017-09-31", "twitter-stream-2017-11-31"}
IMPOSSIBLE_V2: set = set()   # no known impossible-date filenames in the v2 era


# =====================================================================================
# Settings of one run (one class per JSON format; the same fields)
# =====================================================================================

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ConfigV1:
    out_dir: str = "processed_data_v2"
    work_dir: str = "tmp_archives"
    transport: str = "copy"        # "copy" (rclone copy -> local, seekable) | "cat" (stream)
    upload_to: Optional[str] = None      # e.g. "<remote>:<path>/twitter_processed_parquet"
    delete_local_parquet: bool = False   # only meaningful with upload_to
    delete_local_archive: bool = True    # applies to archives WE downloaded, never to user files
    lang_keep: Tuple[str, ...] = ("en", "und")
    use_prefilter: bool = True
    compression: str = "zstd"
    compression_level: int = 3
    batch_rows: int = 150_000
    sort_members: bool = True      # time-order rows for free when random access is available
    keep_screen_names: bool = True    # False saves ~33 MB/day (~39 GB total); ids are canonical
    keep_orig_text: bool = True       # False saves ~71 MB/day (~84 GB total) but needs coin flags first
    # --timeout guards the silent wedge documented in GDRIVE_TRANSFER_QUIRKS §3 and
    # observed in VM_RUN_29AUG.md: zero bytes for minutes, sockets open, no error,
    # transfer looks healthy. Without an idle timeout a stalled archive hangs a
    # worker indefinitely; with it the connection dies and --retries takes over.
    rclone_extra: Tuple[str, ...] = ("--tpslimit", "8", "--retries", "5",
                                     "--low-level-retries", "20", "--timeout", "120s",
                                     "--contimeout", "30s", "--stats", "30s")
    force: bool = False
    tail_minute_warn: int = 2350    # last member earlier than this => suspect truncation

    def lang_prefilter(self):
        """Compiled byte-regex used to skip parsing lines that cannot survive the
        language filter.

        MUST be whitespace-tolerant: the two eras are serialised differently.
          2020 zip/bz2 era : {"created_at":"...","lang":"ja",...}      (compact)
          2022 tar/gz era  : {"created_at": "...", "lang": "es", ...}  (spaced)
        A compact-only substring test matched ZERO lines in the 2022 era — i.e.
        it would have silently emitted nothing for 487 of 1181 archives while
        reporting them as processed. Measured: regex 2.4-3.2 us/line versus
        13-21 us for an orjson parse, and it removes ~65-70% of parses.
        """
        alt = b"|".join(re.escape(l.encode()) for l in self.lang_keep)
        return re.compile(rb'"lang"\s*:\s*"(?:' + alt + rb')"')


# ---------------------------------------------------------------------------
# Config  (the same fields as ConfigV1)
# ---------------------------------------------------------------------------
@dataclass
class ConfigV2:
    out_dir: str = "processed_data_v2_api"
    work_dir: str = "tmp_archives_v2"
    transport: str = "copy"
    upload_to: Optional[str] = None
    delete_local_parquet: bool = False
    delete_local_archive: bool = True
    lang_keep: Tuple[str, ...] = ("en", "und")
    use_prefilter: bool = True
    compression: str = "zstd"
    compression_level: int = 3
    batch_rows: int = 150_000
    sort_members: bool = True
    keep_screen_names: bool = True
    keep_orig_text: bool = True
    rclone_extra: Tuple[str, ...] = ("--tpslimit", "8", "--retries", "5",
                                     "--low-level-retries", "20", "--timeout", "120s",
                                     "--contimeout", "30s", "--stats", "30s")
    force: bool = False
    tail_minute_warn: int = 2350

    def lang_prefilter(self):
        """Same raw-byte prefilter as v1: v2 payloads still carry a top-level
        (well, top-level-of-`data`) `"lang":"xx"` field in the same textual
        shape, so the regex needs no change."""
        alt = b"|".join(re.escape(l.encode()) for l in self.lang_keep)
        return re.compile(rb'"lang"\s*:\s*"(?:' + alt + rb')"')


# Either format's settings: the shared helpers below read only fields both classes have.
Config = Union["ConfigV1", "ConfigV2"]


# =====================================================================================
# Output schemas (the two formats write the same columns)
# =====================================================================================

# ---------------------------------------------------------------------------
# Schemas  (FIX: nullable timestamp_ms, explicit `day`, optional columns)
# ---------------------------------------------------------------------------
def build_schemas_v1(cfg: ConfigV1) -> Tuple[pa.Schema, pa.Schema]:
    tweets = [
        ("day", pa.date32()),
        ("day_coverage_pct", pa.float64()),  # % of expected minute-files present for this day; filled in post-hoc once the full archive has been scanned
        ("tweet_id", pa.string()),
        ("created_at", pa.string()),
        ("timestamp_ms", pa.int64()),          # nullable; never coerced to 0
        ("tweet_type", pa.string()),
        ("lang", pa.string()),
        ("text", pa.string()),
        ("quoted_text", pa.string()),
        ("quoted_id", pa.string()),
        ("reply_to_id", pa.string()),
        ("reply_to_user", pa.string()),
        ("user_id", pa.string()),
        ("user_followers", pa.int64()),
        ("user_verified", pa.bool_()),
        ("retweet_count", pa.int64()),
        ("favorite_count", pa.int64()),
    ]
    retweets = [
        ("day", pa.date32()),
        ("day_coverage_pct", pa.float64()),  # % of expected minute-files present for this day; filled in post-hoc once the full archive has been scanned
        ("retweet_id", pa.string()),
        ("retweet_created_at", pa.string()),
        ("retweet_timestamp_ms", pa.int64()),
        ("retweeter_id", pa.string()),
        ("retweeter_followers", pa.int64()),
        ("orig_tweet_id", pa.string()),
        ("orig_created_at", pa.string()),
        ("orig_author_id", pa.string()),
        ("orig_author_followers", pa.int64()),
        ("lang", pa.string()),
    ]
    if cfg.keep_screen_names:
        tweets.insert(12, ("user_screen_name", pa.string()))
        retweets.insert(5, ("retweeter_screen_name", pa.string()))
        retweets.insert(10, ("orig_author_screen_name", pa.string()))
    if cfg.keep_orig_text:
        retweets.append(("orig_text", pa.string()))
    return pa.schema(tweets), pa.schema(retweets)


# ---------------------------------------------------------------------------
# Schemas — MUST match build_schemas_v1() so v1/v2 output unions
# ---------------------------------------------------------------------------
def build_schemas_v2(cfg: ConfigV2) -> Tuple[pa.Schema, pa.Schema]:
    tweets = [
        ("day", pa.date32()),
        ("day_coverage_pct", pa.float64()),
        ("tweet_id", pa.string()),
        ("created_at", pa.string()),
        ("timestamp_ms", pa.int64()),
        ("tweet_type", pa.string()),
        ("lang", pa.string()),
        ("text", pa.string()),
        ("quoted_text", pa.string()),
        ("quoted_id", pa.string()),
        ("reply_to_id", pa.string()),
        ("reply_to_user", pa.string()),
        ("user_id", pa.string()),
        ("user_followers", pa.int64()),
        ("user_verified", pa.bool_()),
        ("retweet_count", pa.int64()),
        ("favorite_count", pa.int64()),
    ]
    retweets = [
        ("day", pa.date32()),
        ("day_coverage_pct", pa.float64()),
        ("retweet_id", pa.string()),
        ("retweet_created_at", pa.string()),
        ("retweet_timestamp_ms", pa.int64()),
        ("retweeter_id", pa.string()),
        ("retweeter_followers", pa.int64()),
        ("orig_tweet_id", pa.string()),
        ("orig_created_at", pa.string()),
        ("orig_author_id", pa.string()),
        ("orig_author_followers", pa.int64()),
        ("lang", pa.string()),
    ]
    if cfg.keep_screen_names:
        tweets.insert(13, ("user_screen_name", pa.string()))
        retweets.insert(6, ("retweeter_screen_name", pa.string()))
        retweets.insert(11, ("orig_author_screen_name", pa.string()))
    if cfg.keep_orig_text:
        retweets.append(("orig_text", pa.string()))
    return pa.schema(tweets), pa.schema(retweets)


# =====================================================================================
# Archive names and member paths (shared)
# =====================================================================================

# ---------------------------------------------------------------------------
# Filename / member-path parsing
# ---------------------------------------------------------------------------
_ARCH_PATTERNS = [
    re.compile(r"^twitter_stream_(\d{4})_(\d{2})_(\d{2})\.tar$"),
    re.compile(r"^twitter-stream-(\d{4})-(\d{2})-(\d{2})\.(?:tar|zip)$"),
    re.compile(r"^twitter-stream-(\d{4})(\d{2})(\d{2})\.(?:tar\.gz|tar|zip)$"),
    re.compile(r"^twitter-(\d{4})-(\d{2})-(\d{2})\.tar$"),
]
_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".tar", ".zip")
# Layout B: YYYYMMDDHHMMSS.json.(gz|bz2)   -- flat dir, full-timestamp filename
_MEM_B = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.json(?:\.gz|\.bz2)?$")
# Layout A: .../YYYY/MM/DD/HH/MM.json.(bz2|gz)
# Layout C: DD/HH/MM.json.bz2  -- the 2019-08..2020-06 era stores members RELATIVE
#   to the month, so the year and month are absent from the member path entirely.
#   Measured on twitter_stream_2019_08_01.tar: first member "01/19/31.json.bz2".
# Hour/day dirs and the minute filename itself are NOT reliably zero-padded (seen
# on 2023-01 archives: ".../19/9.json.bz2" for hour 19 minute 9) -- \d{1,2} here,
# and every hh/mm component below is zfill(2)'d before use as an HHMM string, or
# max()/min() over a mix of "959" and "2350" picks "959" as bigger (string sort).
_MEM_A = re.compile(r"^(\d{1,2})\.json(?:\.gz|\.bz2)?$")


def archive_stem(src: str) -> str:
    name = Path(src.split(":")[-1]).name
    for ext in _EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def archive_date(src: str) -> Optional[dt.date]:
    """Date encoded in the archive FILENAME. Label only — never used for output
    partitioning (the .tar.gz bundle's label is not even inside it)."""
    name = Path(src.split(":")[-1]).name
    for pat in _ARCH_PATTERNS:
        m = pat.match(name)
        if m:
            try:
                return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None            # 2020_04_31 etc.
    return None


def parse_member(name: str, ym: Optional[Tuple[int, int]] = None
                 ) -> Tuple[Optional[dt.date], Optional[str], Optional[str]]:
    """-> (date, 'HHMM', layout) for the THREE mirror layouts, else (None, None, None).

    A  .../YYYY/MM/DD/HH/MM.json.bz2     absolute date in the path
    B  YYYYMMDD/YYYYMMDDHHMMSS.json.gz   absolute timestamp in the basename
    C  DD/HH/MM.json.bz2                 day-of-month only; `ym` must supply the
       year and month from the archive filename. Without `ym` this returns None
       rather than guessing. Only the twitter_stream_YYYY_MM_DD.tar convention
       uses layout C, and that filename's date is reliable (the filenames that
       lie -- the 19-day bundle and the impossible dates -- are other conventions
       and are handled before this point).
    """
    parts = [x for x in name.strip("/").split("/") if x not in ("", ".")]
    base = parts[-1]
    m = _MEM_B.match(base)
    if m:
        y, mo, d, hh, mi, _ = m.groups()
        try:
            return dt.date(int(y), int(mo), int(d)), hh + mi, "B"
        except ValueError:
            return None, None, None
    m = _MEM_A.match(base)
    if m:
        if len(parts) >= 5:
            y, mo, d, hh = parts[-5], parts[-4], parts[-3], parts[-2]
            if y.isdigit() and mo.isdigit() and d.isdigit() and hh.isdigit():
                try:
                    return dt.date(int(y), int(mo), int(d)), hh.zfill(2) + m.group(1).zfill(2), "A"
                except ValueError:
                    return None, None, None
        if len(parts) == 3 and ym and parts[0].isdigit() and parts[1].isdigit():
            try:
                return dt.date(ym[0], ym[1], int(parts[0])), parts[1].zfill(2) + m.group(1).zfill(2), "C"
            except ValueError:
                return None, None, None
    return None, None, None


def open_inner_stream(filename: str, raw):
    """Wrap a member's raw byte stream in the right decompressor.
    Both eras occur: bzip2 (layout A) and gzip (layout B)."""
    low = filename.lower()
    if low.endswith(".gz"):
        return gzip.GzipFile(fileobj=raw)
    if low.endswith(".bz2"):
        return bz2.BZ2File(raw)
    if low.endswith((".json", ".jsonl", ".txt")):
        return raw
    raise ValueError(f"unsupported member extension: {filename!r}")


# =====================================================================================
# Reading one tweet (v1: full text of long tweets; v2: ISO timestamps)
# =====================================================================================

def get_full_text(obj) -> str:
    """Un-truncated text. FIX: `extended_tweet` can be present and JSON null,
    which made the original `.get('extended_tweet', {}).get(...)` raise
    AttributeError and silently drop the row into schema_errors."""
    if not isinstance(obj, dict):
        return ""
    ext = obj.get("extended_tweet") or {}
    if isinstance(ext, dict):
        ft = ext.get("full_text")
        if ft:
            return ft
    return obj.get("text") or ""


def _iso_to_ms(s: Optional[str]) -> Optional[int]:
    """v2 has no native timestamp_ms field -- created_at is an ISO-8601
    string ("2023-01-01T12:30:00.000Z"); convert to epoch ms to keep the
    same column semantics as v1's timestamp_ms."""
    if not s:
        return None
    try:
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def enrich_row(row: dict, obj: dict) -> None:
    """EXTENSION POINT — relevance / coin assignment goes here later.

    Deliberately a no-op: the keyword tiers in planning/final_keyword_list.md
    are still being researched. When they are settled, set the flags on `row`
    here (and add the columns in build_schemas). Running the regex at this
    point costs microseconds against the ~3-6 minutes of decompression already
    paid per archive, and writing a filtered subset in the SAME pass avoids a
    second full read of ~300 GB.
    """
    return None


# =====================================================================================
# Transfer and archive members
# =====================================================================================

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
class RcloneError(RuntimeError):
    pass


def _is_remote(src: str) -> bool:
    return ":" in src and not Path(src).exists()


def remote_size(src: str, cfg: Config) -> Optional[int]:
    if not _is_remote(src):
        return Path(src).stat().st_size
    p = subprocess.run(["rclone", "lsjson", src], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)[0]["Size"]
    except Exception:
        return None


@contextlib.contextmanager
def acquire_v1(src: str, cfg: ConfigV1):
    """Yield ('path', Path) or ('stream', fileobj).

    FIX 1: zip members cannot be read from a pipe (the central directory is at
    the end of the file and zipfile needs seek), so zips are ALWAYS fetched to
    local disk. The original code raised ValueError for every remote *.zip,
    which would have failed all 382 zip-era archives — 2020-07-01..2021-08-22,
    including the Jan-2021 run-up and the Apr-2021 all-time high — while a
    2022-only test run showed no errors at all.

    FIX 2: `rclone cat` is checked for exit status. The original called
    terminate() without wait() and never looked at the return code, so a
    throttled or dropped transfer ended tarfile's 'r|*' iteration cleanly at a
    member boundary and produced a silently half-complete day.
    """
    if not _is_remote(src):
        yield "path", Path(src)
        return

    is_zip = src.lower().endswith(".zip")
    if cfg.transport == "copy" or is_zip:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        local = work / Path(src.split(":")[-1]).name
        cmd = ["rclone", "copy", src, str(work), *cfg.rclone_extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RcloneError(f"rclone copy rc={p.returncode}: {p.stderr[-500:]}")
        if not local.exists():
            raise RcloneError(f"rclone copy reported success but {local} is missing")
        try:
            yield "path", local
        finally:
            if cfg.delete_local_archive:
                local.unlink(missing_ok=True)
        return

    cmd = ["rclone", "cat", src, *cfg.rclone_extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=8 << 20)
    try:
        yield "stream", proc.stdout
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    else:
        try:
            proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
        err = proc.stderr.read().decode("utf-8", "replace")[-500:]
        if rc != 0:
            raise RcloneError(f"rclone cat rc={rc}: {err}")
    finally:
        try:
            proc.stderr.close()
        except Exception:
            pass


@contextlib.contextmanager
def acquire_v2(src: str, cfg: ConfigV2):
    if not _is_remote(src):
        yield "path", Path(src)
        return

    is_zip = src.lower().endswith(".zip")
    if cfg.transport == "copy" or is_zip:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        local = work / Path(src.split(":")[-1]).name
        cmd = ["rclone", "copy", src, str(work), *cfg.rclone_extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RcloneError(f"rclone copy rc={p.returncode}: {p.stderr[-500:]}")
        if not local.exists():
            raise RcloneError(f"rclone copy reported success but {local} is missing")
        try:
            yield "path", local
        finally:
            if cfg.delete_local_archive:
                local.unlink(missing_ok=True)
        return

    cmd = ["rclone", "cat", src, *cfg.rclone_extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=8 << 20)
    try:
        yield "stream", proc.stdout
    finally:
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read().decode(errors="replace")
            raise RcloneError(f"rclone cat rc={rc}: {err[-500:]}")


def iter_members(kind, handle, cfg: Config, ym=None) -> Iterator[Tuple[str, io.IOBase]]:
    """Yield (member_name, raw_binary_stream) for every regular file member."""
    if kind == "stream":
        with tarfile.open(fileobj=handle, mode="r|*") as tf:
            for m in tf:
                if m.isfile():
                    f = tf.extractfile(m)
                    if f is not None:
                        yield m.name, f
        return

    path = Path(handle)
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            if cfg.sort_members:
                infos.sort(key=lambda i: (parse_member(i.filename, ym)[0] or dt.date.min,
                                          parse_member(i.filename, ym)[1] or ""))
            for i in infos:
                with zf.open(i) as f:
                    yield i.filename, f
    else:
        with tarfile.open(path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            if cfg.sort_members:
                members.sort(key=lambda m: (parse_member(m.name, ym)[0] or dt.date.min,
                                            parse_member(m.name, ym)[1] or ""))
            for m in members:
                f = tf.extractfile(m)
                if f is not None:
                    yield m.name, f


# =====================================================================================
# Per-day writers
# =====================================================================================

# ---------------------------------------------------------------------------
# Per-day Parquet writers
# ---------------------------------------------------------------------------
class DayWritersV1:
    """One Parquet file per (day, table). Needed because one archive is not one
    day: twitter-stream-20221212.tar.gz holds 2022-12-13..12-31."""

    def __init__(self, cfg: ConfigV1, schemas, out_dir: Path):
        self.cfg, self.out = cfg, out_dir
        self.schema = {"tweets": schemas[0], "retweets": schemas[1]}
        self.writers, self.buf, self.rows = {}, {}, Counter()
        self.paths = {}

    def _buffer(self, table):
        return {n: [] for n in self.schema[table].names}

    def append(self, day, table, row):
        k = (day, table)
        if k not in self.buf:
            self.buf[k] = self._buffer(table)
        b = self.buf[k]
        for n in self.schema[table].names:
            b[n].append(row.get(n))
        if len(b["day"]) >= self.cfg.batch_rows:
            self.flush(k)

    def flush(self, k=None):
        keys = [k] if k is not None else list(self.buf)
        for key in keys:
            day, table = key
            b = self.buf.get(key)
            if not b or not b["day"]:
                continue
            schema = self.schema[table]
            tbl = pa.Table.from_arrays(
                [pa.array(b[n], type=schema.field(n).type) for n in schema.names],
                schema=schema)
            if key not in self.writers:
                d = self.out / f"date={day.isoformat()}"
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{table}.parquet"
                self.paths[key] = p
                self.writers[key] = pq.ParquetWriter(
                    p, schema, compression=self.cfg.compression,
                    compression_level=self.cfg.compression_level,
                    use_dictionary=True, write_statistics=True)
            self.writers[key].write_table(tbl)
            self.rows[key] += tbl.num_rows
            self.buf[key] = self._buffer(table)

    def close(self):
        self.flush()
        for w in self.writers.values():
            w.close()

    def verify(self):
        """Read each footer back and confirm the row count we believe we wrote.
        Cheap (footer only) and catches a truncated or unreadable Parquet
        before anything is uploaded or marked done."""
        out = {}
        for key, p in self.paths.items():
            got = pq.read_metadata(p).num_rows
            exp = self.rows[key]
            if got != exp:
                raise RuntimeError(f"{p}: parquet says {got} rows, writer wrote {exp}")
            out[f"{key[0].isoformat()}/{key[1]}"] = {"rows": got,
                                                     "bytes": p.stat().st_size,
                                                     "path": str(p)}
        return out


# ---------------------------------------------------------------------------
# Per-day Parquet writers — identical to twitter_etl.py
# ---------------------------------------------------------------------------
class DayWritersV2:
    def __init__(self, cfg: ConfigV2, schemas, out_dir: Path):
        self.cfg, self.out = cfg, out_dir
        self.schema = {"tweets": schemas[0], "retweets": schemas[1]}
        self.writers, self.buf, self.rows = {}, {}, Counter()
        self.paths = {}

    def _buffer(self, table):
        return {n: [] for n in self.schema[table].names}

    def append(self, day, table, row):
        k = (day, table)
        if k not in self.buf:
            self.buf[k] = self._buffer(table)
        b = self.buf[k]
        for n in self.schema[table].names:
            b[n].append(row.get(n))
        if len(b["day"]) >= self.cfg.batch_rows:
            self.flush(k)

    def flush(self, k=None):
        keys = [k] if k is not None else list(self.buf)
        for key in keys:
            day, table = key
            b = self.buf.get(key)
            if not b or not b["day"]:
                continue
            schema = self.schema[table]
            tbl = pa.Table.from_arrays(
                [pa.array(b[n], type=schema.field(n).type) for n in schema.names],
                schema=schema)
            if key not in self.writers:
                d = self.out / f"date={day.isoformat()}"
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{table}.parquet"
                self.paths[key] = p
                self.writers[key] = pq.ParquetWriter(
                    p, schema, compression=self.cfg.compression,
                    compression_level=self.cfg.compression_level,
                    use_dictionary=True, write_statistics=True)
            self.writers[key].write_table(tbl)
            self.rows[key] += tbl.num_rows
            self.buf[key] = self._buffer(table)

    def close(self):
        self.flush()
        for w in self.writers.values():
            w.close()

    def verify(self):
        out = {}
        for (day, table), p in self.paths.items():
            pf = pq.ParquetFile(p)
            n = pf.metadata.num_rows
            expect = self.rows[(day, table)]
            if n != expect:
                raise RuntimeError(f"{p}: wrote {expect} rows but footer says {n}")
            out[f"{day.isoformat()}/{table}"] = {"rows": n, "bytes": p.stat().st_size, "path": str(p)}
        return out


def _as_int_v1(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_int_v2(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# =====================================================================================
# Processing one archive
# =====================================================================================

# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------
def process_archive_v1(src: str, cfg: ConfigV1) -> dict:
    """Process one archive. Returns the ledger dict (also written as JSON)."""
    t0 = time.perf_counter()
    stem = archive_stem(src)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / f"{stem}.done.json"

    # FIX 3: resume keys on an atomic .done.json written only AFTER verification
    # and upload, not on the existence of a parquet file. Under the original
    # rule a half-written day, an empty shell, or a first-of-two successful
    # uploads was skipped forever.
    if done_path.exists() and not cfg.force:
        return {"stem": stem, "status": "skipped", "reason": "done marker present"}

    if stem in IMPOSSIBLE_V1:
        led = {"stem": stem, "status": "excluded", "reason": "impossible calendar date"}
        done_path.write_text(json.dumps(led, indent=2))
        return led

    size = remote_size(src, cfg)
    if size is not None and size < SHELL_BYTES:
        led = {"stem": stem, "status": "empty_shell", "archive_bytes": size,
               "reason": f"<{SHELL_BYTES} bytes; 10 such files exist in the window"}
        done_path.write_text(json.dumps(led, indent=2))
        return led

    schemas = build_schemas_v1(cfg)
    stats = Counter()
    members_by_date = defaultdict(set)
    layouts, first_member, last_member = Counter(), None, None
    prefilter = cfg.lang_prefilter()
    keep_lang = set(cfg.lang_keep)
    keep_sn, keep_ot = cfg.keep_screen_names, cfg.keep_orig_text
    writers = DayWritersV1(cfg, schemas, out_dir)
    fallback_day = archive_date(src)
    ym = (fallback_day.year, fallback_day.month) if fallback_day else None
    # Format-proof completeness, independent of the member-path layout: minute of
    # day derived from timestamp_ms with integer arithmetic (~50 ns/row). The
    # 2019-era layout C carries no year/month in the path, so a member-path-only
    # ledger was EMPTY for 311 of 1181 archives -- which also disabled the
    # truncation guard on that whole era.
    minutes_by_epochday = defaultdict(set)

    try:
        with acquire_v1(src, cfg) as (kind, handle):
            for name, raw in iter_members(kind, handle, cfg, ym):
                stats["members_seen"] += 1
                if first_member is None:
                    first_member = name
                last_member = name
                mday, hhmm, layout = parse_member(name, ym)
                if layout == "C" and fallback_day and mday != fallback_day:
                    stats["member_day_differs_from_filename"] += 1
                if mday is None:
                    stats["members_unparsed_path"] += 1
                    mday = fallback_day
                    if mday is None:
                        stats["members_skipped_no_date"] += 1
                        continue
                else:
                    layouts[layout] += 1
                    members_by_date[mday].add(hhmm)

                try:
                    stream = open_inner_stream(name, raw)
                except ValueError:
                    stats["members_unsupported_ext"] += 1
                    continue

                try:
                    for line in stream:
                        if not line.strip():
                            continue
                        stats["lines_read"] += 1

                        # Raw-byte language prefilter: ~100 ns against ~20-60 us
                        # for a full parse, and a strict SUPERSET of what we keep
                        # (it matches if either the outer object or a nested
                        # original carries an accepted lang). Validated in
                        # notebooks/02, test 4.
                        if cfg.use_prefilter and prefilter.search(line) is None:
                            stats["prefiltered_out"] += 1
                            continue

                        try:
                            obj = _loads(line)
                        except Exception:
                            stats["json_errors"] += 1
                            continue
                        if not (isinstance(obj, dict) and "id_str" in obj and "user" in obj):
                            stats["non_tweet_objects"] += 1
                            continue

                        try:
                            if "retweeted_status" in obj:
                                orig = obj.get("retweeted_status") or {}
                                lang = orig.get("lang") or obj.get("lang")
                                if lang not in keep_lang:
                                    stats["skipped_language"] += 1
                                    continue
                                rt = obj.get("user") or {}
                                oa = orig.get("user") or {}
                                row = {
                                    "day": mday,
                                    "retweet_id": obj["id_str"],
                                    "retweet_created_at": obj.get("created_at"),
                                    "retweet_timestamp_ms": _as_int_v1(obj.get("timestamp_ms")),
                                    "retweeter_id": rt.get("id_str"),
                                    "retweeter_followers": _as_int_v1(rt.get("followers_count")),
                                    "orig_tweet_id": orig.get("id_str"),
                                    "orig_created_at": orig.get("created_at"),
                                    "orig_author_id": oa.get("id_str"),
                                    "orig_author_followers": _as_int_v1(oa.get("followers_count")),
                                    "lang": lang,
                                }
                                if keep_sn:
                                    row["retweeter_screen_name"] = rt.get("screen_name")
                                    row["orig_author_screen_name"] = oa.get("screen_name")
                                if keep_ot:
                                    row["orig_text"] = get_full_text(orig)
                                enrich_row(row, obj)
                                writers.append(mday, "retweets", row)
                                stats["retweets_extracted"] += 1
                                _ts = row["retweet_timestamp_ms"]
                                if _ts:
                                    minutes_by_epochday[_ts // 86400000].add((_ts // 60000) % 1440)
                            else:
                                lang = obj.get("lang")
                                if lang not in keep_lang:
                                    stats["skipped_language"] += 1
                                    continue
                                if obj.get("in_reply_to_status_id") is not None:
                                    ttype = "reply"
                                elif obj.get("is_quote_status") and "quoted_status" in obj:
                                    ttype = "quote"
                                else:
                                    ttype = "original"
                                q = (obj.get("quoted_status") or {}) if ttype == "quote" else {}
                                user = obj.get("user") or {}
                                row = {
                                    "day": mday,
                                    "tweet_id": obj["id_str"],
                                    "created_at": obj.get("created_at"),
                                    "timestamp_ms": _as_int_v1(obj.get("timestamp_ms")),
                                    "tweet_type": ttype,
                                    "lang": lang,
                                    "text": get_full_text(obj),
                                    "quoted_text": get_full_text(q) if q else None,
                                    "quoted_id": q.get("id_str") if q else None,
                                    "reply_to_id": obj.get("in_reply_to_status_id_str"),
                                    "reply_to_user": obj.get("in_reply_to_screen_name"),
                                    "user_id": user.get("id_str"),
                                    "user_followers": _as_int_v1(user.get("followers_count")),
                                    "user_verified": bool(user.get("verified", False)),
                                    "retweet_count": _as_int_v1(obj.get("retweet_count")),
                                    "favorite_count": _as_int_v1(obj.get("favorite_count")),
                                }
                                if keep_sn:
                                    row["user_screen_name"] = user.get("screen_name")
                                enrich_row(row, obj)
                                writers.append(mday, "tweets", row)
                                stats["tweets_extracted"] += 1
                                _ts = row["timestamp_ms"]
                                if _ts:
                                    minutes_by_epochday[_ts // 86400000].add((_ts // 60000) % 1440)
                        except Exception as e:
                            stats["schema_errors"] += 1
                            log.error("%s/%s: schema error: %s", stem, name, e)
                except (EOFError, OSError, zlib.error) as e:
                    # A truncated inner member raises out of the `for line in
                    # stream` iteration, not from the row-building try above.
                    # Without this clause it propagates out of process_archive,
                    # run() marks the archive FAILED, no .done.json is written,
                    # and every retry fails identically -- which is exactly how
                    # 2020-11-15 ended up absent from the panel while its 1.78 GB
                    # zip sat intact (MD5-verified) on the mirror. Keep the lines
                    # already read from this member and move to the next one;
                    # prepare_reddit_crypto_v2.py takes the same position on
                    # truncated gzip members.
                    stats["truncated_members"] += 1
                    log.warning("%s/%s: truncated member (%s) -- keeping what was "
                                "read, continuing", stem, name, e)
                finally:
                    if stream is not raw:
                        try:
                            stream.close()
                        except Exception:
                            pass
    finally:
        writers.close()

    files = writers.verify()

    # Completeness: short days are normal on this mirror, so record rather than
    # assert. A missing TAIL, by contrast, is the signature of a dead transfer.
    per_day = {d.isoformat(): {"members": len(v),
                               "of_expected": EXPECTED_MEMBERS_PER_DAY,
                               "pct": round(100 * len(v) / EXPECTED_MEMBERS_PER_DAY, 2),
                               "first": min(v) if v else None,
                               "last": max(v) if v else None}
               for d, v in sorted(members_by_date.items())}
    def _hhmm(mod: int) -> str:
        return f"{mod // 60:02d}{mod % 60:02d}"

    per_day_data = {
        (dt.date(1970, 1, 1) + dt.timedelta(days=ed)).isoformat(): {
            "minutes": len(v),
            "of_expected": EXPECTED_MEMBERS_PER_DAY,
            "pct": round(100 * len(v) / EXPECTED_MEMBERS_PER_DAY, 2),
            "first": _hhmm(min(v)), "last": _hhmm(max(v))}
        for ed, v in sorted(minutes_by_epochday.items())}

    # Truncation guard. Prefer member-path coverage; fall back to the
    # timestamp-derived view so the guard is armed on every layout.
    if members_by_date:
        last_day = max(members_by_date)
        last_min = max(members_by_date[last_day]) if members_by_date[last_day] else "0000"
        guard_src = "member_path"
    elif per_day_data:
        last_day = max(per_day_data)
        last_min = per_day_data[last_day]["last"]
        guard_src = "timestamp_ms"
    else:
        last_day, last_min, guard_src = None, None, "none"
    if last_min is not None and int(last_min) < cfg.tail_minute_warn:
        log.warning(
            "%s: last minute of %s is %s (< %s, source=%s); day has gaps "
            "(possibly scattered through the day, not just the tail) — "
            "keeping it with day_coverage_pct recorded rather than discarding "
            "the whole day's real data",
            stem, last_day, last_min, cfg.tail_minute_warn, guard_src)
    if stats["tweets_extracted"] + stats["retweets_extracted"] > 0 and guard_src == "none":
        raise RuntimeError(f"{stem}: rows written but no completeness signal from "
                           "either the member paths or timestamp_ms — refusing to mark done")

    # Backfill the real day_coverage_pct into the already-written parquet files
    # now that the full archive has been scanned and per-day coverage is known.
    # (Written as NULL during the main pass since coverage for a day isn't final
    # until the whole archive has been read.)
    def _coverage_for(day_iso: str) -> Optional[float]:
        if day_iso in per_day:
            return per_day[day_iso]["pct"]
        if day_iso in per_day_data:
            return per_day_data[day_iso]["pct"]
        return None

    for (wday, wtable), wpath in writers.paths.items():
        pct = _coverage_for(wday.isoformat())
        if pct is None or not wpath.exists():
            continue
        # ParquetFile(...).read(), NOT pq.read_table() -- see process_archive_v2:
        # read_table() infers `date` from the .../date=YYYY-MM-DD/ parent directory
        # and write_table() then bakes that extra column into the output.
        tbl = pq.ParquetFile(wpath).read()
        col = pa.array([pct] * tbl.num_rows, type=pa.float64())
        idx = tbl.schema.get_field_index("day_coverage_pct")
        tbl = tbl.set_column(idx, "day_coverage_pct", col)
        pq.write_table(tbl, wpath, compression=cfg.compression,
                        compression_level=cfg.compression_level,
                        use_dictionary=True, write_statistics=True)

    ledger = {
        "stem": stem,
        "status": "ok",
        "source": src,
        "archive_bytes": size,
        "json_lib": JSON_LIB,
        "transport": "local" if not _is_remote(src) else cfg.transport,
        "duration_sec": round(time.perf_counter() - t0, 1),
        "layouts": dict(layouts),
        "first_member": first_member,
        "last_member": last_member,
        "days": per_day,
        "n_days": len(per_day),
        "days_from_data": per_day_data,
        "guard_source": guard_src,
        "stats": dict(stats),
        "files": files,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    # FIX 4: the stats Counter is persisted. It was collected and thrown away,
    # which is why the 2020-12-14 trial could not reveal that the archive holds
    # 1411 of 1440 minute-files. This file is the audit trail for the data
    # chapter and Appendix E.
    stats_path = out_dir / f"{stem}.stats.json"
    stats_path.write_text(json.dumps(ledger, indent=2))

    if cfg.upload_to:
        _upload_verified(ledger, stats_path, cfg)
        if cfg.delete_local_parquet:
            for f in files.values():
                Path(f["path"]).unlink(missing_ok=True)

    tmp = done_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: ledger[k] for k in
                               ("stem", "status", "n_days", "days", "days_from_data",
                                "guard_source", "stats", "finished_at")},
                              indent=2))
    os.replace(tmp, done_path)          # atomic
    return ledger


# ---------------------------------------------------------------------------
# Core per-archive processing — v2 SCHEMA SPECIFIC
# ---------------------------------------------------------------------------
def process_archive_v2(src: str, cfg: ConfigV2) -> dict:
    """Process one archive. Returns the ledger dict (also written as JSON)."""
    t0 = time.perf_counter()
    stem = archive_stem(src)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / f"{stem}.done.json"

    if done_path.exists() and not cfg.force:
        return {"stem": stem, "status": "skipped", "reason": "done marker present"}

    if stem in IMPOSSIBLE_V2:
        led = {"stem": stem, "status": "excluded", "reason": "impossible calendar date"}
        done_path.write_text(json.dumps(led, indent=2))
        return led

    size = remote_size(src, cfg)
    if size is not None and size < SHELL_BYTES:
        led = {"stem": stem, "status": "empty_shell", "archive_bytes": size,
               "reason": f"<{SHELL_BYTES} bytes; empty-shell archives exist in this window too"}
        done_path.write_text(json.dumps(led, indent=2))
        return led

    schemas = build_schemas_v2(cfg)
    stats = Counter()
    members_by_date = defaultdict(set)
    layouts, first_member, last_member = Counter(), None, None
    prefilter = cfg.lang_prefilter()
    keep_lang = set(cfg.lang_keep)
    keep_sn, keep_ot = cfg.keep_screen_names, cfg.keep_orig_text
    writers = DayWritersV2(cfg, schemas, out_dir)
    fallback_day = archive_date(src)
    ym = (fallback_day.year, fallback_day.month) if fallback_day else None
    minutes_by_epochday = defaultdict(set)

    try:
        with acquire_v2(src, cfg) as (kind, handle):
            for name, raw in iter_members(kind, handle, cfg, ym):
                stats["members_seen"] += 1
                if first_member is None:
                    first_member = name
                last_member = name
                mday, hhmm, layout = parse_member(name, ym)
                if layout == "C" and fallback_day and mday != fallback_day:
                    stats["member_day_differs_from_filename"] += 1
                if mday is None:
                    stats["members_unparsed_path"] += 1
                    mday = fallback_day
                    if mday is None:
                        stats["members_skipped_no_date"] += 1
                        continue
                else:
                    layouts[layout] += 1
                    members_by_date[mday].add(hhmm)

                try:
                    stream = open_inner_stream(name, raw)
                except ValueError:
                    stats["members_unsupported_ext"] += 1
                    continue

                try:
                    for line in stream:
                        if not line.strip():
                            continue
                        stats["lines_read"] += 1

                        if cfg.use_prefilter and prefilter.search(line) is None:
                            stats["prefiltered_out"] += 1
                            continue

                        try:
                            obj = _loads(line)
                        except Exception:
                            stats["json_errors"] += 1
                            continue

                        d = obj.get("data") if isinstance(obj, dict) else None
                        if not (isinstance(d, dict) and "id" in d and "author_id" in d):
                            stats["non_tweet_objects"] += 1
                            continue

                        try:
                            inc = obj.get("includes") or {}
                            users_by_id = {u["id"]: u for u in (inc.get("users") or []) if u.get("id")}
                            tweets_by_id = {t["id"]: t for t in (inc.get("tweets") or []) if t.get("id")}
                            refs = d.get("referenced_tweets") or []
                            rt_ref = next((r for r in refs if r.get("type") == "retweeted"), None)

                            if rt_ref:
                                orig = tweets_by_id.get(rt_ref.get("id")) or {}
                                lang = orig.get("lang") or d.get("lang")
                                if lang not in keep_lang:
                                    stats["skipped_language"] += 1
                                    continue
                                retweeter = users_by_id.get(d.get("author_id")) or {}
                                orig_author = users_by_id.get(orig.get("author_id")) or {}
                                row = {
                                    "day": mday,
                                    "retweet_id": d.get("id"),
                                    "retweet_created_at": d.get("created_at"),
                                    "retweet_timestamp_ms": _iso_to_ms(d.get("created_at")),
                                    "retweeter_id": d.get("author_id"),
                                    "retweeter_followers": _as_int_v2(
                                        (retweeter.get("public_metrics") or {}).get("followers_count")),
                                    "orig_tweet_id": orig.get("id"),
                                    "orig_created_at": orig.get("created_at"),
                                    "orig_author_id": orig.get("author_id"),
                                    "orig_author_followers": _as_int_v2(
                                        (orig_author.get("public_metrics") or {}).get("followers_count")),
                                    "lang": lang,
                                }
                                if keep_sn:
                                    row["retweeter_screen_name"] = retweeter.get("username")
                                    row["orig_author_screen_name"] = orig_author.get("username")
                                if keep_ot:
                                    row["orig_text"] = orig.get("text") or ""
                                enrich_row(row, obj)
                                writers.append(mday, "retweets", row)
                                stats["retweets_extracted"] += 1
                                _ts = row["retweet_timestamp_ms"]
                                if _ts:
                                    minutes_by_epochday[_ts // 86400000].add((_ts // 60000) % 1440)
                            else:
                                lang = d.get("lang")
                                if lang not in keep_lang:
                                    stats["skipped_language"] += 1
                                    continue
                                reply_ref = next((r for r in refs if r.get("type") == "replied_to"), None)
                                quote_ref = next((r for r in refs if r.get("type") == "quoted"), None)
                                if reply_ref:
                                    ttype = "reply"
                                elif quote_ref:
                                    ttype = "quote"
                                else:
                                    ttype = "original"
                                q = tweets_by_id.get(quote_ref.get("id")) if quote_ref else None
                                user = users_by_id.get(d.get("author_id")) or {}
                                pm = d.get("public_metrics") or {}
                                # v2 gives in_reply_to_user_id (an id), not a screen name;
                                # best-effort resolve via includes.users, fall back to the id.
                                reply_to_user = d.get("in_reply_to_user_id")
                                if reply_to_user and reply_to_user in users_by_id:
                                    reply_to_user = users_by_id[reply_to_user].get("username", reply_to_user)
                                row = {
                                    "day": mday,
                                    "tweet_id": d.get("id"),
                                    "created_at": d.get("created_at"),
                                    "timestamp_ms": _iso_to_ms(d.get("created_at")),
                                    "tweet_type": ttype,
                                    "lang": lang,
                                    "text": d.get("text") or "",
                                    "quoted_text": (q.get("text") if q else None),
                                    "quoted_id": (q.get("id") if q else None),
                                    "reply_to_id": (reply_ref.get("id") if reply_ref else None),
                                    "reply_to_user": reply_to_user,
                                    "user_id": d.get("author_id"),
                                    "user_followers": _as_int_v2(
                                        (user.get("public_metrics") or {}).get("followers_count")),
                                    "user_verified": bool(user.get("verified", False)),
                                    "retweet_count": _as_int_v2(pm.get("retweet_count")),
                                    "favorite_count": _as_int_v2(pm.get("like_count")),
                                }
                                if keep_sn:
                                    row["user_screen_name"] = user.get("username")
                                enrich_row(row, obj)
                                writers.append(mday, "tweets", row)
                                stats["tweets_extracted"] += 1
                                _ts = row["timestamp_ms"]
                                if _ts:
                                    minutes_by_epochday[_ts // 86400000].add((_ts // 60000) % 1440)
                        except Exception as e:
                            stats["schema_errors"] += 1
                            log.error("%s/%s: schema error: %s", stem, name, e)
                except (EOFError, OSError, zlib.error) as e:
                    # A truncated inner member raises out of the `for line in
                    # stream` iteration, not the row-building try above. Without
                    # this clause it propagates out of process_archive, run()
                    # marks the whole archive FAILED, no .done.json is written,
                    # and every retry fails identically -- which is how
                    # 2020-11-15 stayed absent from the panel while its
                    # MD5-verified 1.78 GB zip sat on the mirror. Only 9 of its
                    # 1,294 members were bad; keeping the rest recovers the day
                    # at 89.9% coverage. Same position prepare_reddit_crypto_v2.py
                    # takes on truncated gzip members.
                    stats["truncated_members"] += 1
                    log.warning("%s/%s: truncated member (%s) -- keeping what was "
                                "read, continuing", stem, name, e)
                finally:
                    if stream is not raw:
                        try:
                            stream.close()
                        except Exception:
                            pass
    finally:
        writers.close()

    files = writers.verify()

    per_day = {d.isoformat(): {"members": len(v),
                               "of_expected": EXPECTED_MEMBERS_PER_DAY,
                               "pct": round(100 * len(v) / EXPECTED_MEMBERS_PER_DAY, 2),
                               "first": min(v) if v else None,
                               "last": max(v) if v else None}
               for d, v in sorted(members_by_date.items())}

    def _hhmm(mod: int) -> str:
        return f"{mod // 60:02d}{mod % 60:02d}"

    per_day_data = {
        (dt.date(1970, 1, 1) + dt.timedelta(days=ed)).isoformat(): {
            "minutes": len(v),
            "of_expected": EXPECTED_MEMBERS_PER_DAY,
            "pct": round(100 * len(v) / EXPECTED_MEMBERS_PER_DAY, 2),
            "first": _hhmm(min(v)), "last": _hhmm(max(v))}
        for ed, v in sorted(minutes_by_epochday.items())}

    if members_by_date:
        last_day = max(members_by_date)
        last_min = max(members_by_date[last_day]) if members_by_date[last_day] else "0000"
        guard_src = "member_path"
    elif per_day_data:
        last_day = max(per_day_data)
        last_min = per_day_data[last_day]["last"]
        guard_src = "timestamp_ms"
    else:
        last_day, last_min, guard_src = None, None, "none"

    # Coverage guard: warn, don't discard. Same policy as twitter_etl.py --
    # a partial day still carries real tweets; day_coverage_pct records the
    # gap instead of us throwing the whole day away to avoid keeping it.
    if last_min is not None and int(last_min) < cfg.tail_minute_warn:
        log.warning(
            "%s: last minute of %s is %s (< %s, source=%s); day has gaps -- "
            "keeping it with day_coverage_pct recorded",
            stem, last_day, last_min, cfg.tail_minute_warn, guard_src)
    if stats["tweets_extracted"] + stats["retweets_extracted"] > 0 and guard_src == "none":
        raise RuntimeError(f"{stem}: rows written but no completeness signal from "
                           "either the member paths or timestamp_ms — refusing to mark done")

    def _coverage_for(day_iso: str) -> Optional[float]:
        if day_iso in per_day:
            return per_day[day_iso]["pct"]
        if day_iso in per_day_data:
            return per_day_data[day_iso]["pct"]
        return None

    for (wday, wtable), wpath in writers.paths.items():
        pct = _coverage_for(wday.isoformat())
        if pct is None or not wpath.exists():
            continue
        # ParquetFile(...).read(), NOT pq.read_table(): the output path is
        # .../date=YYYY-MM-DD/<table>.parquet, and read_table() routes through the
        # dataset API, which infers `date` from the parent directory and appends it
        # as a dictionary column. write_table() then bakes that spurious column into
        # the file. Whether it happens is pyarrow-version dependent (not observed on
        # the VM's 25.0.1, reproduced on 24.0.0), so a day rebuilt on a different
        # machine silently gains a 19th column the rest of the dataset lacks.
        tbl = pq.ParquetFile(wpath).read()
        col = pa.array([pct] * tbl.num_rows, type=pa.float64())
        idx = tbl.schema.get_field_index("day_coverage_pct")
        tbl = tbl.set_column(idx, "day_coverage_pct", col)
        pq.write_table(tbl, wpath, compression=cfg.compression,
                        compression_level=cfg.compression_level,
                        use_dictionary=True, write_statistics=True)

    ledger = {
        "stem": stem,
        "status": "ok",
        "schema_version": "v2",
        "source": src,
        "archive_bytes": size,
        "json_lib": JSON_LIB,
        "transport": "local" if not _is_remote(src) else cfg.transport,
        "duration_sec": round(time.perf_counter() - t0, 1),
        "layouts": dict(layouts),
        "first_member": first_member,
        "last_member": last_member,
        "days": per_day,
        "n_days": len(per_day),
        "days_from_data": per_day_data,
        "guard_source": guard_src,
        "stats": dict(stats),
        "files": files,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    stats_path = out_dir / f"{stem}.stats.json"
    stats_path.write_text(json.dumps(ledger, indent=2))

    if cfg.upload_to:
        _upload_verified(ledger, stats_path, cfg)
        if cfg.delete_local_parquet:
            for f in files.values():
                Path(f["path"]).unlink(missing_ok=True)

    tmp = done_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: ledger[k] for k in
                               ("stem", "status", "schema_version", "n_days", "days",
                                "days_from_data", "guard_source", "stats", "finished_at")},
                              indent=2))
    os.replace(tmp, done_path)
    return ledger


# =====================================================================================
# Upload, prefilter check, many archives
# =====================================================================================

def _upload_verified(ledger, stats_path, cfg: Config):
    """Upload every output, then confirm each remote size matches local before
    the caller writes the done marker."""
    dest_root = cfg.upload_to.rstrip("/")
    targets = [(Path(f["path"]), f"{dest_root}/date={k.split('/')[0]}")
               for k, f in ledger["files"].items()]
    targets.append((stats_path, f"{dest_root}/_ledger"))
    for local, dest in targets:
        p = subprocess.run(["rclone", "copy", str(local), dest, *cfg.rclone_extra],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RcloneError(f"upload {local.name} rc={p.returncode}: {p.stderr[-400:]}")
    for local, dest in targets:
        p = subprocess.run(["rclone", "lsjson", f"{dest}/{local.name}"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RcloneError(f"verify {local.name}: lsjson rc={p.returncode}")
        try:
            remote_bytes = json.loads(p.stdout)[0]["Size"]
        except Exception as e:
            raise RcloneError(f"verify {local.name}: unreadable lsjson ({e})")
        if remote_bytes != local.stat().st_size:
            raise RcloneError(f"verify {local.name}: remote {remote_bytes} != local "
                              f"{local.stat().st_size}")


# ---------------------------------------------------------------------------
# Validation helper used by the notebook
# ---------------------------------------------------------------------------
def validate_prefilter(src: str, cfg: ConfigV1, max_members: int = 3) -> dict:
    """Prove the raw-byte prefilter is a superset of the parsed decision.
    Any line the parser would keep MUST be matched by the prefilter."""
    prefilter = cfg.lang_prefilter()
    keep_lang = set(cfg.lang_keep)
    res = Counter()
    with acquire_v1(src, cfg) as (kind, handle):
        for i, (name, raw) in enumerate(iter_members(kind, handle, cfg)):
            if i >= max_members:
                break
            stream = open_inner_stream(name, raw)
            for line in stream:
                if not line.strip():
                    continue
                res["lines"] += 1
                hit = prefilter.search(line) is not None
                res["prefilter_hit"] += bool(hit)
                try:
                    obj = _loads(line)
                except Exception:
                    continue
                if not (isinstance(obj, dict) and "id_str" in obj and "user" in obj):
                    continue
                if "retweeted_status" in obj:
                    lang = (obj.get("retweeted_status") or {}).get("lang") or obj.get("lang")
                else:
                    lang = obj.get("lang")
                keep = lang in keep_lang
                res["parsed_keep"] += bool(keep)
                if keep and not hit:
                    res["FALSE_NEGATIVE"] += 1
    out = dict(res)
    out["prefilter_reduction_pct"] = round(
        100 * (1 - res["prefilter_hit"] / max(res["lines"], 1)), 1)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def list_remote(remote_dir: str) -> list:
    p = subprocess.run(["rclone", "lsf", remote_dir], capture_output=True, text=True)
    if p.returncode != 0:
        raise RcloneError(f"rclone lsf rc={p.returncode}: {p.stderr[-300:]}")
    names = [l.strip() for l in p.stdout.splitlines() if l.strip()]
    return [remote_dir.rstrip("/") + "/" + n for n in names
            if n.endswith((".tar", ".tar.gz", ".zip"))]


def run_v1(sources, cfg: ConfigV1, workers: int = 6, use_processes: bool = True) -> list:
    """Process many archives. Processes, not threads: json parsing holds the GIL,
    so a ThreadPoolExecutor serialises the parse across workers."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    Pool = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    out = []
    with Pool(max_workers=workers) as ex:
        futs = {ex.submit(process_archive_v1, s, cfg): s for s in sources}
        for fu in as_completed(futs):
            s = futs[fu]
            try:
                out.append(fu.result())
            except Exception as e:
                log.error("FAILED %s: %s", s, e)
                out.append({"stem": archive_stem(s), "status": "failed", "error": repr(e)})
    return out


def run_v2(sources, cfg: ConfigV2, workers: int = 6, use_processes: bool = True) -> list:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    Pool = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    out = []
    with Pool(max_workers=workers) as ex:
        futs = {ex.submit(process_archive_v2, s, cfg): s for s in sources}
        for fu in as_completed(futs):
            s = futs[fu]
            try:
                led = fu.result()
            except Exception as e:
                led = {"stem": archive_stem(s), "status": "error", "error": str(e)}
                log.error("FAILED %s: %s", s, e)
            out.append(led)
    return out


# =====================================================================================
# Command line
# =====================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Twitter Stream Grab -> Parquet (Stage A)")
    sub = ap.add_subparsers(dest="era", required=True)
    defaults = {"v1": ("processed_data_v2", "tmp_archives", "archives up to 2022-11-22 (Twitter API v1.1 JSON)"),
                "v2": ("processed_data_v2_api", "tmp_archives_v2", "archives from 2022-12-13 (Twitter API v2 JSON)")}
    for era, (out, work, what) in defaults.items():
        s = sub.add_parser(era, help=what)
        s.add_argument("sources", nargs="+", help="local paths or rclone remotes")
        s.add_argument("--out", default=out)
        s.add_argument("--work", default=work)
        s.add_argument("--transport", choices=["copy", "cat"], default="copy")
        s.add_argument("--upload-to", default=None)
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--no-prefilter", action="store_true")
        s.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config_cls, run = (ConfigV1, run_v1) if a.era == "v1" else (ConfigV2, run_v2)
    cfg = config_cls(out_dir=a.out, work_dir=a.work, transport=a.transport,
                     upload_to=a.upload_to, use_prefilter=not a.no_prefilter, force=a.force)
    res = run(a.sources, cfg, workers=a.workers)
    ok = sum(1 for r in res if r.get("status") == "ok")
    print(json.dumps({"archives": len(res), "ok": ok,
                      "tweets": sum(r.get("stats", {}).get("tweets_extracted", 0) for r in res),
                      "retweets": sum(r.get("stats", {}).get("retweets_extracted", 0) for r in res)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
