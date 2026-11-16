#!/usr/bin/env python3
"""
01_extract_reddit.py -- step 1 for Reddit: the raw Arctic Shift archive -> one table per subreddit group.

Input    the Arctic Shift dumps of the 14 study subreddits (posts and comments, .jsonl.gz),
         a local copy of the bucket folder reddit_arcticshift_crypto (chunks/ and out/).
Output   data/pipeline/reddit/extract/reddit_<P>.parquet for the six groups
         P = BTC, ETH, BNB, XRP, DOGE, MARKET, each with a _stats.json file.
Runtime  about 7 minutes on a laptop (6 workers).

Which subreddit feeds which group (SUB_TO_COIN)
------------------------------------------------
    BTC     r/bitcoin, r/btc, r/bitcoinmarkets        XRP     r/xrp, r/ripple
    ETH     r/ethereum, r/ethtrader, r/ethfinance     DOGE    r/dogecoin
    BNB     r/binance                                 MARKET  r/cryptocurrency, r/cryptomarkets,
                                                              r/satoshistreetbets, r/cryptocurrencytrading
The group is written as coin_label. Which posts of r/btc and r/binance count for a coin
is decided later, in 04_series_reddit.py.

The step runs in three phases; each refuses to run on unverified input
----------------------------------------------------------------------
1. verify    For every raw object: count, bytes and md5 against the bucket listing
             (crc32c for composite objects, which carry no md5), and `gzip -t` on every
             file, both the 11,897 chunks and the 21 sequential files. Writes
             raw_manifest.csv and raw_verify.json. Without access to the bucket,
             `--manifest` checks the local copy against a stored manifest instead.
2. parse     Streams every archive into parquet shards, one per task. Every line is
             accounted for: blank, no id, bad created_utc, outside the window, kept
             (live or dead). A decompression, decoding or JSON error stops the run and
             removes the task's partial shard. The md5 of the bytes actually read must
             equal the manifest.
3. finalize  DuckDB: one row per id. When the dumps hold several copies of a post, the
             copy is chosen by source file, then the latest retrieval, then the line
             number (a fixed order, so every run keeps the same copy); copies that
             disagree are counted, and the overlap where two dump files meet is counted
             apart. Each group's table is written to a temporary name, checked
             (check_output), and only then moved into place.

What a row holds
----------------
* Window: 2019-08-01 00:00:00 to 2023-01-31 23:59:59 UTC on created_utc (1,280 days).
* text: posts = title + newline + selftext, or the title alone when the selftext is
  empty or a placeholder such as [removed]; comments = body. The text is stored as
  written: no cleaning here (the emotion model cleans its own input).
* Dead rows (removed or deleted, text = '') are kept with is_dead = TRUE and dead_reason
  naming the placeholder, so is_dead <=> text = ''. They never enter a measure; the
  daily counts report them per group (n_dead). A post whose selftext was removed keeps
  its title; removed_by_category and is_robot_indexable record the removal (Reddit
  fills removed_by_category only from about November 2019; is_robot_indexable is FALSE
  on every removed post in every year).
* Author: a salted SHA-256 hash (author_hash), never the name. The salt file stays
  outside the repository; only its fingerprint enters the stats. '[deleted]' is an
  unknown author (hash NULL). Three flags are computed from the name before hashing:
  is_automoderator, name_bot_token ("bot" as a separate token of the name, the rule of
  arXiv 2601.14423) and name_has_bot ("bot" anywhere in the name, a criterion of the
  keyword rule in 03_bot_filter.py).
* Quote split: a quote line starts with '>' after leading spaces, except spoiler tags
  (>!...!<) and emoticons (>:( , >.<). own_text = the other lines, quoted_text = the
  quote lines verbatim; text itself is unchanged. The filter step reads own_text for
  the author's own words.
* Also kept: distinguished, stickied, edited, retrieved_at, account_age_days, and the
  public link_id and parent_id (needed to join replies to their thread).

DuckDB runs with TimeZone=UTC (asserted), 4 GB, 3 threads and a spill directory. No day
column is stored: days are derived in UTC from ts.

Decisions: MASTERPLAN D18 (re-parse); Reddit defect map P5-P8, P11-P14, P19, P22, P30.

Usage (from code/)
------------------
    python pipeline/01_extract_reddit.py verify   --raw <raw dir> --run-dir <run dir> [--manifest <raw_manifest.csv>]
    python pipeline/01_extract_reddit.py parse    --raw <raw dir> --run-dir <run dir> --workers 6
    python pipeline/01_extract_reddit.py finalize --run-dir <run dir> --out data/pipeline/reddit/extract

<raw dir> holds chunks/ and out/ exactly as in the bucket (default: config.REDDIT_RAW_DIR,
set with THESIS_REDDIT_RAW). run_all.py passes all three phases in this order.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import calendar
import csv
import gzip
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import rules as R  # noqa: E402  (user-name flags)

SPEC_VERSION = "reddit-extract-2026-09-26"

# The bucket the archive was copied from (only the default of `verify --bucket`).
BUCKET = os.environ.get("THESIS_REDDIT_BUCKET", "gs://<bucket>/reddit_arcticshift_crypto")

SUB_TO_COIN = {
    "bitcoin": "BTC", "btc": "BTC", "bitcoinmarkets": "BTC",
    "ethereum": "ETH", "ethtrader": "ETH", "ethfinance": "ETH",
    "binance": "BNB",
    "xrp": "XRP", "ripple": "XRP",
    "dogecoin": "DOGE",
    "cryptocurrency": "MARKET", "cryptomarkets": "MARKET",
    "satoshistreetbets": "MARKET", "cryptocurrencytrading": "MARKET",
}
COINS = ("BNB", "BTC", "DOGE", "ETH", "MARKET", "XRP")

# Study window: 2019-08-01 00:00:00 .. 2023-01-31 23:59:59 UTC
START_UTC, END_UTC = 1564617600, 1675209599
N_DAYS = 1280

# lowercased, stripped body/selftext -> dead_reason
DEAD_TEXTS = {
    "": "empty",
    "[removed]": "[removed]",
    "[deleted]": "[deleted]",
    "[removed by reddit]": "[removed by reddit]",
    "[deleted by user]": "[deleted by user]",
}

# August 2019 posts carry no removed_by_category key; the selftext placeholder
# is the only removal evidence there (P7). Used ONLY when the key is absent, and
# stored as the bracketed placeholder itself, so an inferred value can never be
# mistaken for one of Reddit's categories (moderator, reddit, deleted, ...).
RBC_FALLBACK = {
    "[removed]": "[removed]",
    "[deleted]": "[deleted]",
    "[removed by reddit]": "[removed by reddit]",
    "[deleted by user]": "[deleted by user]",
}

# Quote split. The quote test is `line.lstrip().startswith('>')`; these two
# patterns are the exceptions.
SPOILER_RX = r"^\s*>!.*!<"
EMOTICON_RX = (
    r"^\s*>(?:[:;=]['\-^]?[()\[\]{}|/\\*@$<>]"   # >:(  >:)  >;)  >=(  >:/  >:-(
    r"|[:;=]['\-^]?[DPpOo](?!\w)"                  # >:D  >:P  >:O  (not >:Dude)
    r"|[._~^\-]?<"                                 # ><   >.<  >_<  >~<
    r"|[._~^\-]>)"                                 # >.>  >_>   (never >> itself)
)
SPOILER = re.compile(SPOILER_RX)
EMOTICON = re.compile(EMOTICON_RX)

SALT_PATH = os.path.expanduser("~/.config/thesis/reddit_author_salt.hex")

DUCKDB_MEMORY = "4GB"
DUCKDB_THREADS = 3
TASK_BYTES = 150_000_000          # compressed input per parse task
FLUSH_ROWS = 100_000

SPEC = {
    "version": SPEC_VERSION,
    "window_utc": [START_UTC, END_UTC],
    "sub_to_coin": SUB_TO_COIN,
    "dead_texts": DEAD_TEXTS,
    "rbc_fallback_when_key_absent": RBC_FALLBACK,
    "is_robot_indexable": "raw boolean on posts (NULL when absent and on comments); added beyond the brief",
    "post_text": "(title + '\\n' + selftext).strip() if selftext.strip().lower() not a placeholder, else title.strip()",
    "comment_text": "body.strip(), '' if a placeholder",
    "is_dead": "text == '' (rows the first pass dropped)",
    "quote_line": "line.lstrip().startswith('>') and not spoiler and not emoticon",
    "spoiler_rx": SPOILER_RX,
    "emoticon_rx": EMOTICON_RX,
    "own_text": "'\\n'.join(non-quote lines).strip(); = text when no quote line",
    "quoted_text": "'\\n'.join(quote lines).strip(), verbatim incl. '>'; NULL when no quote line",
    "author_hash": "sha256(salt || author utf-8) hex; NULL for missing, '' and bracketed placeholders ([deleted])",
    "author_flags": "is_automoderator: author == 'AutoModerator'; name_bot_token: 'bot' is a separate token of the name, e.g. ends in _bot or -bot (arXiv 2601.14423; pipeline.rules.name_bot_token); name_has_bot: 'bot' anywhere in the name (keyword rule criterion; pipeline.rules.name_has_bot); author_deleted: author == '[deleted]'",
    "coin_label": "the subreddit's partition (SUB_TO_COIN), written here (formerly reddit_processed.py)",
    "account_age_days": "(created_utc - author_created_utc) / 86400",
    "retrieved_at": "retrieved_on if present, else retrieved_utc",
    "edited": "epoch seconds when a positive number, else NULL",
    "score": "int(score) if numeric else 0 (first-pass rule)",
    "dedup_order": "src_file ASC, retrieved_utc DESC NULLS LAST, src_line ASC",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def spec_sha() -> str:
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:16]


def code_sha() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def file_md5(path: str, block: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False, default=str)
    os.replace(tmp, path)


def classify(rel: str) -> Optional[Tuple[str, str, str]]:
    """(subreddit, kind, coin) from either naming convention; None if unmapped.

    Parallel chunk: chunks/<sub>__<YYYY-MM-DD>.jsonl.gz -> comments
    Sequential:     out/<sub>_<posts|comments>.jsonl.gz
    """
    base = os.path.basename(rel)
    if not base.endswith(".jsonl.gz"):
        return None
    base = base[: -len(".jsonl.gz")]
    if "__" in base:
        sub, kind = base.split("__", 1)[0], "comments"
    else:
        sub, _, kind = base.rpartition("_")
        if kind not in ("posts", "comments"):
            return None
    coin = SUB_TO_COIN.get(sub.lower())
    return None if coin is None else (sub, kind, coin)


def load_salt(create: bool) -> bytes:
    if not os.path.exists(SALT_PATH):
        if not create:
            sys.exit(f"salt file missing: {SALT_PATH}")
        os.makedirs(os.path.dirname(SALT_PATH), mode=0o700, exist_ok=True)
        fd = os.open(SALT_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_hex(32) + "\n")
        log(f"generated a new author salt at {SALT_PATH}")
    with open(SALT_PATH) as fh:
        salt = bytes.fromhex(fh.read().strip())
    if len(salt) != 32:
        sys.exit(f"salt file {SALT_PATH} does not hold 32 bytes")
    return salt


def salt_fingerprint(salt: bytes) -> str:
    """Identifies the salt across runs without revealing it."""
    return hashlib.sha256(b"fingerprint:" + salt).hexdigest()[:16]


# year boundaries for the field-list tally (UTC)
_YEAR_STARTS = [calendar.timegm((y, 1, 1, 0, 0, 0)) for y in range(2000, 2031)]


def utc_year(epoch: int) -> int:
    return 1999 + bisect.bisect_right(_YEAR_STARTS, epoch)


class HashingReader:
    """File wrapper that md5s exactly the bytes the gzip reader consumes."""

    def __init__(self, fh, hasher):
        self.fh, self.hasher, self.n = fh, hasher, 0

    def read(self, size=-1):
        b = self.fh.read(size)
        self.hasher.update(b)
        self.n += len(b)
        return b

    def readable(self):
        return True


class FatalParseError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def gcs_listing(bucket: str) -> Dict[str, dict]:
    """name (relative to the prefix) -> {bytes, md5_b64, crc32c_b64, components}."""
    out = subprocess.run(
        ["gcloud", "storage", "objects", "list", f"{bucket}/**",
         "--format=csv[no-heading](name,size,md5_hash,crc32c_hash,component_count)"],
        check=True, capture_output=True, text=True).stdout
    prefix = bucket.split("/", 3)[3] + "/"
    listing = {}
    for row in csv.reader(out.splitlines()):
        name, size, md5_b64, crc_b64, comps = row
        assert name.startswith(prefix), name
        listing[name[len(prefix):]] = {"bytes": int(size), "md5_b64": md5_b64,
                                       "crc32c_b64": crc_b64, "components": comps}
    return listing


def local_crc32c_b64(path: str) -> str:
    out = subprocess.run(["gcloud", "storage", "hash", path, "--skip-md5"],
                         check=True, capture_output=True, text=True).stdout
    m = re.search(r"crc32c_hash:\s*(\S+)", out)
    if not m:
        raise RuntimeError(f"no crc32c in `gcloud storage hash` output for {path}")
    return m.group(1)


def gzip_test(paths: List[str]) -> Tuple[List[str], str]:
    """`gzip -t` over a batch; returns (failed paths, stderr)."""
    r = subprocess.run(["gzip", "-t", *paths], capture_output=True, text=True)
    if r.returncode == 0:
        return [], ""
    failed = []
    for p in paths:   # re-test one by one to name the culprits
        if subprocess.run(["gzip", "-t", p], capture_output=True).returncode != 0:
            failed.append(p)
    return failed, r.stderr


def manifest_listing(path: str) -> Dict[str, dict]:
    """The stored manifest of an earlier verify, in the shape of gcs_listing().

    Every object must have passed the bucket check then (hash_ok = 1); the local
    files are compared with the md5 recorded for them.
    """
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["hash_ok"] != "1":
                raise SystemExit(f"{path}: {row['object']} did not pass the bucket check")
            out[row["object"]] = {"bytes": int(row["bytes"]), "md5_b64": row["gcs_md5_b64"],
                                  "crc32c_b64": row["gcs_crc32c_b64"], "md5_local": row["md5_local"]}
    return out


def cmd_verify(args) -> int:
    t0 = time.time()
    raw = os.path.abspath(args.raw)
    os.makedirs(args.run_dir, exist_ok=True)
    if args.manifest:
        listing = manifest_listing(args.manifest)
        args.bucket = f"manifest:{os.path.abspath(args.manifest)} (md5 {file_md5(args.manifest)})"
    else:
        listing = gcs_listing(args.bucket)
    log(f"verify: raw={raw}  source={args.bucket}")
    local = {}
    for root, _, files in os.walk(raw):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, raw)
            if f == ".DS_Store":
                continue
            local[rel] = os.path.getsize(full)

    only_gcs = sorted(set(listing) - set(local))
    only_local = sorted(set(local) - set(listing))
    size_diff = sorted(k for k in set(listing) & set(local) if listing[k]["bytes"] != local[k])
    gcs_bytes = sum(v["bytes"] for v in listing.values())
    local_bytes = sum(local.values())
    log(f"objects: gcs {len(listing):,} ({gcs_bytes:,} B)  local {len(local):,} ({local_bytes:,} B)")
    problems = []
    if only_gcs:
        problems.append(f"{len(only_gcs)} objects missing locally, e.g. {only_gcs[:3]}")
    if only_local:
        problems.append(f"{len(only_local)} local files not in the bucket, e.g. {only_local[:3]}")
    if size_diff:
        problems.append(f"{len(size_diff)} size mismatches, e.g. {size_diff[:3]}")

    names = sorted(set(listing) & set(local))

    def one(rel):
        full = os.path.join(raw, rel)
        md5_hex = file_md5(full)
        g = listing[rel]
        if "md5_local" in g:   # offline: the md5 recorded when the bucket check passed
            ok = g["md5_local"] == md5_hex
            how = "manifest_md5"
        elif g["md5_b64"]:
            ok = base64.b64decode(g["md5_b64"]).hex() == md5_hex
            how = "md5"
        else:  # composite upload: GCS keeps only crc32c
            ok = local_crc32c_b64(full) == g["crc32c_b64"]
            how = "crc32c"
        return rel, md5_hex, how, ok

    with ThreadPoolExecutor(max_workers=4) as ex:
        hashed = list(ex.map(one, names))
    bad_hash = [r for r, _, _, ok in hashed if not ok]
    if bad_hash:
        problems.append(f"{len(bad_hash)} hash mismatches, e.g. {bad_hash[:3]}")
    n_crc = sum(1 for _, _, how, _ in hashed if how == "crc32c")
    log(f"hashes: {len(hashed) - len(bad_hash):,} of {len(hashed):,} match "
        f"({n_crc} composite objects checked by crc32c)")

    gz = [os.path.join(raw, r) for r in names if r.endswith(".jsonl.gz")]
    n_seq = sum(1 for r in names if r.endswith(".jsonl.gz") and r.startswith("out/"))
    batches = [gz[i:i + 100] for i in range(0, len(gz), 100)]
    failed_gz = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for failed, err in ex.map(gzip_test, batches):
            failed_gz.extend(failed)
    if failed_gz:
        problems.append(f"gzip -t failed on {len(failed_gz)} files: {failed_gz[:5]}")
    log(f"gzip -t: {len(gz) - len(failed_gz):,} of {len(gz):,} pass "
        f"({n_seq} sequential, {len(gz) - n_seq:,} chunks)")

    manifest = os.path.join(args.run_dir, "raw_manifest.csv")
    tmp = manifest + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "bytes", "md5_local", "gcs_md5_b64", "gcs_crc32c_b64", "checked_by", "hash_ok"])
        for rel, md5_hex, how, ok in hashed:
            g = listing[rel]
            w.writerow([rel, local[rel], md5_hex, g["md5_b64"], g["crc32c_b64"], how, int(ok)])
    os.replace(tmp, manifest)

    verdict = {
        "finished_at": utcnow(), "seconds": round(time.time() - t0, 1),
        "bucket": args.bucket, "raw_dir": raw,
        "gcs_objects": len(listing), "gcs_bytes": gcs_bytes,
        "local_objects": len(local), "local_bytes": local_bytes,
        "hash_checked": len(hashed), "hash_by_crc32c": n_crc, "hash_mismatch": len(bad_hash),
        "gzip_tested": len(gz), "gzip_tested_sequential": n_seq, "gzip_failed": len(failed_gz),
        "manifest": manifest, "manifest_md5": file_md5(manifest),
        "problems": problems, "ok": not problems,
    }
    write_json_atomic(os.path.join(args.run_dir, "raw_verify.json"), verdict)
    if problems:
        for p in problems:
            log(f"VERIFY FAILED: {p}")
        return 3
    log(f"verify OK in {verdict['seconds']} s")
    return 0


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

SHARD_SCHEMA = pa.schema([
    ("id", pa.string()), ("created_utc", pa.int64()), ("kind", pa.string()),
    ("subreddit", pa.string()), ("coin", pa.string()), ("text", pa.string()),
    ("score", pa.int64()),
    ("author_hash", pa.string()), ("is_automoderator", pa.bool_()),
    ("name_bot_token", pa.bool_()), ("name_has_bot", pa.bool_()), ("author_deleted", pa.bool_()),
    ("account_age_days", pa.float64()),
    ("distinguished", pa.string()), ("stickied", pa.bool_()),
    ("removed_by_category", pa.string()), ("is_robot_indexable", pa.bool_()),
    ("edited_utc", pa.float64()), ("retrieved_utc", pa.int64()),
    ("link_id", pa.string()), ("parent_id", pa.string()),
    ("is_dead", pa.bool_()), ("dead_reason", pa.string()),
    ("own_text", pa.string()), ("quoted_text", pa.string()), ("has_quote_lines", pa.bool_()),
    ("src_file", pa.string()), ("src_line", pa.int64()),
])

FILE_COUNTERS = (
    "lines", "blank", "no_id", "bad_created", "oow_before", "oow_after", "kept",
    "kept_live", "kept_dead", "posts", "comments",
    "subreddit_mismatch", "kind_field_missing",
    "author_missing", "author_deleted", "author_other_placeholder", "automoderator", "name_bot_token", "name_has_bot",
    "author_created_missing", "account_age_negative",
    "distinguished_set", "stickied_true", "stickied_missing",
    "rbc_key_present", "rbc_key_absent", "rbc_fallback_used", "rbc_nonnull_raw", "rbc_on_comment",
    "robot_indexable_false", "robot_indexable_missing",
    "selftext_placeholder",
    "retrieved_on_only", "retrieved_utc_only", "retrieved_both", "retrieved_both_differ", "retrieved_none",
    "edited_time", "edited_true", "edited_other",
    "link_id_missing", "parent_id_missing",
    "score_non_numeric",
    "has_quote_lines", "own_text_empty_after_split", "quote_lines", "spoiler_lines", "emoticon_lines",
    "html_gt_line_rows",
)


def parse_task(task: dict) -> dict:
    """Worker: parse the files of one task into one shard. Raises on any error."""
    salt = bytes.fromhex(task["salt_hex"])
    shard_tmp = task["shard"] + ".tmp"
    writer = pq.ParquetWriter(shard_tmp, SHARD_SCHEMA, compression="zstd")
    cols = {f.name: [] for f in SHARD_SCHEMA}
    n_rows = 0
    per_file = []
    sha256 = hashlib.sha256
    dead_texts = DEAD_TEXTS
    spoiler_match, emoticon_match = SPOILER.match, EMOTICON.match

    def flush():
        nonlocal cols
        if not cols["id"]:
            return
        writer.write_table(pa.Table.from_pydict(cols, schema=SHARD_SCHEMA))
        cols = {f.name: [] for f in SHARD_SCHEMA}

    try:
        for rel, full, expect_bytes, expect_md5 in task["files"]:
            sub, kind, coin = classify(rel)
            is_post = kind == "posts"
            c = Counter()
            dead_by = Counter()
            keysets = Counter()
            hasher = hashlib.md5()
            line_no = 0
            with open(full, "rb") as fh:
                reader = HashingReader(fh, hasher)
                with gzip.GzipFile(fileobj=reader, mode="rb") as gz:
                    for raw_line in gz:
                        line_no += 1
                        if not raw_line.strip():
                            c["blank"] += 1
                            continue
                        try:
                            d = json.loads(raw_line)
                        except ValueError as err:   # JSONDecodeError and UnicodeDecodeError
                            raise FatalParseError(f"{rel}: line {line_no}: {type(err).__name__}: {err}") from None
                        if not isinstance(d, dict):
                            raise FatalParseError(f"{rel}: line {line_no}: JSON value is not an object")

                        cu = d.get("created_utc")
                        cu_int = None
                        if cu and not isinstance(cu, bool):
                            try:
                                cu_int = int(cu)
                            except (TypeError, ValueError):
                                cu_int = None
                        keysets[(utc_year(cu_int) if cu_int else None, tuple(d))] += 1

                        rid = d.get("id")
                        if not rid:
                            c["no_id"] += 1
                            continue
                        if cu_int is None:
                            c["bad_created"] += 1
                            continue
                        if cu_int < START_UTC:
                            c["oow_before"] += 1
                            continue
                        if cu_int > END_UTC:
                            c["oow_after"] += 1
                            continue
                        c["kept"] += 1

                        rs = d.get("subreddit")
                        if not isinstance(rs, str) or rs.lower() != sub.lower():
                            c["subreddit_mismatch"] += 1

                        # ---- text, dead state, removal ---------------------------
                        if is_post:
                            c["posts"] += 1
                            if "title" not in d:
                                c["kind_field_missing"] += 1
                            title = (d.get("title") or "").strip()
                            body = (d.get("selftext") or "").strip()
                            body_l = body.lower()
                            if body_l in dead_texts:
                                text = title
                                if body_l:
                                    c["selftext_placeholder"] += 1
                            else:
                                text = (title + "\n" + body).strip()
                            dead_reason = dead_texts[body_l] if not text else None
                            if "removed_by_category" in d:
                                c["rbc_key_present"] += 1
                                rbc = d["removed_by_category"]
                                rbc = rbc if isinstance(rbc, str) and rbc else None
                                if rbc is not None:
                                    c["rbc_nonnull_raw"] += 1
                            else:
                                c["rbc_key_absent"] += 1
                                rbc = RBC_FALLBACK.get(body_l)
                                if rbc is not None:
                                    c["rbc_fallback_used"] += 1
                            ri = d.get("is_robot_indexable")
                            if isinstance(ri, bool):
                                if not ri:
                                    c["robot_indexable_false"] += 1
                            else:
                                ri = None
                                c["robot_indexable_missing"] += 1
                            link_id = parent_id = None
                        else:
                            c["comments"] += 1
                            if "body" not in d:
                                c["kind_field_missing"] += 1
                            body = (d.get("body") or "").strip()
                            body_l = body.lower()
                            if body_l in dead_texts:
                                text = ""
                                dead_reason = dead_texts[body_l]
                            else:
                                text = body
                                dead_reason = None
                            if d.get("removed_by_category") is not None:
                                c["rbc_on_comment"] += 1
                            rbc = ri = None
                            link_id = d.get("link_id")
                            parent_id = d.get("parent_id")
                            link_id = link_id if isinstance(link_id, str) and link_id else None
                            parent_id = parent_id if isinstance(parent_id, str) and parent_id else None
                            if link_id is None:
                                c["link_id_missing"] += 1
                            if parent_id is None:
                                c["parent_id_missing"] += 1

                        is_dead = text == ""
                        if is_dead:
                            c["kept_dead"] += 1
                            dead_by[dead_reason] += 1
                        else:
                            c["kept_live"] += 1

                        # ---- quote split -----------------------------------------
                        own_text, quoted_text, has_quote = text, None, False
                        if "&gt;" in text and any(ln.lstrip().startswith("&gt;") for ln in text.split("\n")):
                            c["html_gt_line_rows"] += 1
                        if ">" in text:
                            own, quoted = [], []
                            for ln in text.split("\n"):
                                if ln.lstrip().startswith(">"):
                                    if spoiler_match(ln):
                                        c["spoiler_lines"] += 1
                                        own.append(ln)
                                    elif emoticon_match(ln):
                                        c["emoticon_lines"] += 1
                                        own.append(ln)
                                    else:
                                        quoted.append(ln)
                                else:
                                    own.append(ln)
                            if quoted:
                                has_quote = True
                                own_text = "\n".join(own).strip()
                                quoted_text = "\n".join(quoted).strip()
                                c["has_quote_lines"] += 1
                                c["quote_lines"] += len(quoted)
                                if not own_text:
                                    c["own_text_empty_after_split"] += 1

                        # ---- author ----------------------------------------------
                        author = d.get("author")
                        if not isinstance(author, str) or author == "":
                            c["author_missing"] += 1
                            author = None
                        author_deleted = author == "[deleted]"
                        is_automod = author == "AutoModerator"
                        name_tok = bool(R.name_bot_token(author)) if author is not None else False
                        name_has = bool(R.name_has_bot(author)) if author is not None else False
                        placeholder = author is not None and author.startswith("[") and author.endswith("]")
                        if author_deleted:
                            c["author_deleted"] += 1
                        elif placeholder:
                            c["author_other_placeholder"] += 1
                        if is_automod:
                            c["automoderator"] += 1
                        if name_tok:
                            c["name_bot_token"] += 1
                        if name_has:
                            c["name_has_bot"] += 1
                        author_hash = (None if author is None or placeholder
                                       else sha256(salt + author.encode("utf-8")).hexdigest())

                        ac = d.get("author_created_utc")
                        if isinstance(ac, (int, float)) and not isinstance(ac, bool):
                            age = (cu_int - ac) / 86400.0
                            if age < 0:
                                c["account_age_negative"] += 1
                        else:
                            age = None
                            c["author_created_missing"] += 1

                        # ---- moderation and timing fields ------------------------
                        dist = d.get("distinguished")
                        dist = dist if isinstance(dist, str) and dist else None
                        if dist is not None:
                            c["distinguished_set"] += 1
                        stk = d.get("stickied")
                        if isinstance(stk, bool):
                            if stk:
                                c["stickied_true"] += 1
                        else:
                            stk = None
                            c["stickied_missing"] += 1

                        ed = d.get("edited")
                        if isinstance(ed, bool) or ed is None:
                            if ed is True:
                                c["edited_true"] += 1
                            ed = None
                        elif isinstance(ed, (int, float)) and ed > 0:
                            ed = float(ed)
                            c["edited_time"] += 1
                        else:
                            c["edited_other"] += 1
                            ed = None

                        ro, ru = d.get("retrieved_on"), d.get("retrieved_utc")
                        ro = int(ro) if isinstance(ro, (int, float)) and not isinstance(ro, bool) else None
                        ru = int(ru) if isinstance(ru, (int, float)) and not isinstance(ru, bool) else None
                        if ro is not None and ru is not None:
                            c["retrieved_both"] += 1
                            if ro != ru:
                                c["retrieved_both_differ"] += 1
                        elif ro is not None:
                            c["retrieved_on_only"] += 1
                        elif ru is not None:
                            c["retrieved_utc_only"] += 1
                        else:
                            c["retrieved_none"] += 1
                        retrieved = ro if ro is not None else ru

                        sc = d.get("score")
                        if isinstance(sc, (int, float)) and not isinstance(sc, bool):
                            score = int(sc)
                        else:
                            score = 0
                            c["score_non_numeric"] += 1

                        cols["id"].append(str(rid))
                        cols["created_utc"].append(cu_int)
                        cols["kind"].append(kind)
                        cols["subreddit"].append(sub)
                        cols["coin"].append(coin)
                        cols["text"].append(text)
                        cols["score"].append(score)
                        cols["author_hash"].append(author_hash)
                        cols["is_automoderator"].append(is_automod)
                        cols["name_bot_token"].append(name_tok)
                        cols["name_has_bot"].append(name_has)
                        cols["author_deleted"].append(author_deleted)
                        cols["account_age_days"].append(age)
                        cols["distinguished"].append(dist)
                        cols["stickied"].append(stk)
                        cols["removed_by_category"].append(rbc)
                        cols["is_robot_indexable"].append(ri)
                        cols["edited_utc"].append(ed)
                        cols["retrieved_utc"].append(retrieved)
                        cols["link_id"].append(link_id)
                        cols["parent_id"].append(parent_id)
                        cols["is_dead"].append(is_dead)
                        cols["dead_reason"].append(dead_reason)
                        cols["own_text"].append(own_text)
                        cols["quoted_text"].append(quoted_text)
                        cols["has_quote_lines"].append(has_quote)
                        cols["src_file"].append(rel)
                        cols["src_line"].append(line_no)
                        n_rows += 1
                        if len(cols["id"]) >= FLUSH_ROWS:
                            flush()
                # drain: gzip stops at the end of the last member, make sure the
                # hash covers the whole file
                while reader.read(1 << 20):
                    pass

            c["lines"] = line_no
            got_md5 = hasher.hexdigest()
            if reader.n != expect_bytes or got_md5 != expect_md5:
                raise FatalParseError(f"{rel}: read {reader.n} B md5 {got_md5}, manifest "
                                      f"{expect_bytes} B md5 {expect_md5}")
            accounted = (c["blank"] + c["no_id"] + c["bad_created"] + c["oow_before"]
                         + c["oow_after"] + c["kept"])
            if accounted != c["lines"]:
                raise FatalParseError(f"{rel}: conservation broken, {c['lines']} lines vs {accounted} accounted")
            if c["kept"] != c["kept_live"] + c["kept_dead"] or c["kept"] != c["posts"] + c["comments"]:
                raise FatalParseError(f"{rel}: kept split does not add up")
            per_file.append({
                "file": rel, "bytes": expect_bytes, "md5": expect_md5,
                "subreddit": sub, "kind": kind, "coin": coin,
                "counts": {k: int(c[k]) for k in FILE_COUNTERS},
                "dead_by_reason": dict(dead_by),
                "keysets": [[y, list(ks), n] for (y, ks), n in keysets.items()],
            })
        flush()
        writer.close()
    except BaseException:
        writer.close()
        if os.path.exists(shard_tmp):
            os.remove(shard_tmp)
        raise

    os.replace(shard_tmp, task["shard"])
    # per-file counters and field lists go to a sidecar, so the ledger stays small
    files_json = task["shard"] + ".files.json"
    write_json_atomic(files_json, per_file)
    return {
        "task_id": task["task_id"], "coin": task["coin"], "shard": task["shard"],
        "shard_bytes": os.path.getsize(task["shard"]), "shard_md5": file_md5(task["shard"]),
        "files_json": files_json, "files_json_md5": file_md5(files_json),
        "rows": n_rows, "n_files": len(per_file),
        "kept": sum(f["counts"]["kept"] for f in per_file),
        "lines": sum(f["counts"]["lines"] for f in per_file),
    }


def read_manifest(run_dir: str) -> Dict[str, dict]:
    verdict_path = os.path.join(run_dir, "raw_verify.json")
    if not os.path.exists(verdict_path):
        sys.exit("run `verify` first: raw_verify.json missing")
    verdict = json.load(open(verdict_path))
    if not verdict.get("ok"):
        sys.exit("raw_verify.json says the raw copy did NOT verify; refusing to parse")
    manifest = os.path.join(run_dir, "raw_manifest.csv")
    if file_md5(manifest) != verdict["manifest_md5"]:
        sys.exit("raw_manifest.csv changed since verify; re-run verify")
    rows = {}
    with open(manifest) as fh:
        for r in csv.DictReader(fh):
            rows[r["object"]] = {"bytes": int(r["bytes"]), "md5": r["md5_local"], "ok": r["hash_ok"] == "1"}
    return rows


def plan_tasks(raw: str, shard_dir: str, manifest: Dict[str, dict]) -> List[dict]:
    data = sorted(k for k in manifest if k.endswith(".jsonl.gz"))
    unmapped = [k for k in data if classify(k) is None]
    if unmapped:
        sys.exit(f"{len(unmapped)} raw files map to no coin, e.g. {unmapped[:3]}")
    os.makedirs(shard_dir, exist_ok=True)
    tasks = []
    for coin in COINS:
        files = [k for k in data if classify(k)[2] == coin]
        group, size, n = [], 0, 0
        for k in files:
            b = manifest[k]["bytes"]
            if group and size + b > TASK_BYTES:
                tasks.append((coin, n, group, size))
                group, size, n = [], 0, n + 1
            group.append(k)
            size += b
        if group:
            tasks.append((coin, n, group, size))
    out = []
    for coin, n, group, size in tasks:
        tid = f"{coin}-{n:04d}"
        out.append({
            "task_id": tid, "coin": coin, "bytes": size,
            "shard": os.path.join(shard_dir, f"{tid}.parquet"),
            "files": [(k, os.path.join(raw, k), manifest[k]["bytes"], manifest[k]["md5"]) for k in group],
        })
    return out


def parse_code_sha() -> str:
    """Hash of the code that decides what goes into a shard. Editing the verify or
    finalize code leaves it unchanged; editing the parser invalidates every shard."""
    import inspect
    parts = [inspect.getsource(o) for o in (parse_task, classify, HashingReader, utc_year, FatalParseError,
                                            R.name_bot_token, R.name_has_bot)]
    parts += [repr(SHARD_SCHEMA), repr(FILE_COUNTERS), json.dumps(SPEC, sort_keys=True)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def task_key(task: dict) -> dict:
    return {"spec_sha": spec_sha(), "code_sha": parse_code_sha(),
            "inputs": [[k, b, m] for k, _, b, m in task["files"]]}


def shard_intact(r: dict) -> bool:
    return (os.path.exists(r["shard"]) and os.path.getsize(r["shard"]) == r["shard_bytes"]
            and file_md5(r["shard"]) == r["shard_md5"]
            and os.path.exists(r["files_json"]) and file_md5(r["files_json"]) == r["files_json_md5"])


def task_resumable(entry: Optional[dict], task: dict) -> bool:
    """Skip a task only if spec, code and every input (name, bytes, md5) match the
    ledger and the shard on disk is the one the ledger recorded -- never on the
    mere existence of a file."""
    return bool(entry) and entry.get("key") == task_key(task) and shard_intact(entry["result"])


def cmd_parse(args) -> int:
    t0 = time.time()
    raw = os.path.abspath(args.raw)
    manifest = read_manifest(args.run_dir)
    # the files on disk must still be the verified ones (bytes; md5 is re-checked while reading)
    for k, v in manifest.items():
        if os.path.getsize(os.path.join(raw, k)) != v["bytes"]:
            sys.exit(f"{k}: size on disk differs from the verified manifest")
    salt = load_salt(create=True)
    log(f"parse: spec {SPEC_VERSION} spec_sha={spec_sha()} parse_code_sha={parse_code_sha()} "
        f"salt at {SALT_PATH} (fingerprint {salt_fingerprint(salt)})")

    tasks = plan_tasks(raw, os.path.abspath(args.shard_dir or os.path.join(args.run_dir, "shards")), manifest)
    ledger_path = os.path.join(args.run_dir, "parse_ledger.json")
    ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {"tasks": {}}

    todo = [t for t in tasks if not task_resumable(ledger["tasks"].get(t["task_id"]), t)]
    log(f"{len(tasks)} tasks over {sum(len(t['files']) for t in tasks):,} files; "
        f"{len(tasks) - len(todo)} resumable from the ledger, {len(todo)} to run, workers={args.workers}")
    ledger["plan"] = [t["task_id"] for t in tasks]
    ledger["spec_sha"], ledger["parse_code_sha"] = spec_sha(), parse_code_sha()
    ledger["salt_fingerprint"] = salt_fingerprint(salt)
    write_json_atomic(ledger_path, ledger)

    for t in todo:
        t["salt_hex"] = salt.hex()
    todo.sort(key=lambda t: -t["bytes"])
    done_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(parse_task, t): t for t in todo}
        pending = set(futs)
        while pending:
            # FIRST_COMPLETED, not FIRST_EXCEPTION: the latter blocks until every
            # task is done, so the ledger would only be written at the very end
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                t = futs[f]
                err = f.exception()
                if err is not None:
                    for p in pending:
                        p.cancel()
                    log(f"FATAL in task {t['task_id']}: {err}")
                    ex.shutdown(wait=True, cancel_futures=True)
                    for tt in todo:   # no partial shard survives
                        if os.path.exists(tt["shard"] + ".tmp"):
                            os.remove(tt["shard"] + ".tmp")
                    return 4
                res = f.result()
                ledger["tasks"][t["task_id"]] = {"key": task_key(t), "result": res, "finished_at": utcnow()}
                write_json_atomic(ledger_path, ledger)
                done_rows += res["rows"]
                log(f"  {t['task_id']:12s} {len(t['files']):>5,} files {t['bytes'] / 1e6:7.1f} MB "
                    f"-> {res['rows']:>10,} rows   ({len(ledger['tasks'])}/{len(tasks)})")
    ledger["parse_seconds_last_run"] = round(time.time() - t0, 1)
    write_json_atomic(ledger_path, ledger)
    log(f"parse done in {time.time() - t0:,.0f} s; {done_rows:,} rows written this run")
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

FINAL_SELECT = """
    id,
    to_timestamp(created_utc) AS ts,
    kind, subreddit, text, score,
    author_hash, is_automoderator, name_bot_token, name_has_bot, author_deleted, account_age_days,
    distinguished, stickied, removed_by_category, is_robot_indexable,
    to_timestamp(edited_utc) AS edited,
    to_timestamp(retrieved_utc) AS retrieved_at,
    link_id, parent_id,
    is_dead, dead_reason, own_text, quoted_text, has_quote_lines,
    coin AS coin_label
"""


def duck(spill: str):
    import duckdb
    os.makedirs(spill, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{spill}'")
    con.execute("SET preserve_insertion_order=true")
    tz = con.execute("SELECT current_setting('TimeZone')").fetchone()[0]
    assert tz == "UTC", f"DuckDB session zone is {tz}, not UTC"
    return con


def one(con, sql: str):
    return con.execute(sql).fetchone()


def cmd_finalize(args) -> int:
    t0 = time.time()
    run_dir, out_dir = args.run_dir, args.out
    os.makedirs(out_dir, exist_ok=True)
    ledger = json.load(open(os.path.join(run_dir, "parse_ledger.json")))
    manifest = read_manifest(run_dir)
    raw_verify = json.load(open(os.path.join(run_dir, "raw_verify.json")))
    salt = load_salt(create=False)
    if ledger.get("salt_fingerprint") != salt_fingerprint(salt):
        sys.exit("the salt changed since parse; refusing to finalize")

    # every planned task done, under this spec and code, with an intact shard
    shards, files_stats = [], []
    for tid in ledger["plan"]:
        e = ledger["tasks"].get(tid)
        if e is None:
            sys.exit(f"task {tid} has not run; run `parse` first")
        if e["key"]["spec_sha"] != spec_sha() or e["key"]["code_sha"] != parse_code_sha():
            sys.exit(f"task {tid} was parsed under another spec/code; re-run `parse`")
        r = e["result"]
        if not shard_intact(r):
            sys.exit(f"shard {r['shard']} (or its sidecar) differs from the ledger")
        for k, b, m in e["key"]["inputs"]:
            if manifest[k]["bytes"] != b or manifest[k]["md5"] != m:
                sys.exit(f"{k}: input differs from the manifest used at parse time")
        shards.append(r["shard"])
        files_stats.extend(json.load(open(r["files_json"])))
    n_data_files = sum(1 for k in manifest if k.endswith(".jsonl.gz"))
    if len(files_stats) != n_data_files:
        sys.exit(f"ledger covers {len(files_stats)} files, manifest has {n_data_files}")

    tot = Counter()
    for f in files_stats:
        tot.update(f["counts"])
    log(f"finalize: {len(shards)} shards, {len(files_stats):,} files, {tot['lines']:,} raw lines, "
        f"{tot['kept']:,} kept rows")

    con = duck(args.spill or os.path.join(run_dir, "_duckdb_spill"))
    shard_list = "[" + ", ".join(f"'{s}'" for s in shards) + "]"
    con.execute(f"CREATE VIEW raw_rows AS SELECT * FROM read_parquet({shard_list})")
    n_rows = one(con, "SELECT count(*) FROM raw_rows")[0]
    assert n_rows == tot["kept"], f"shards hold {n_rows:,} rows, ledger says {tot['kept']:,}"

    # ---- duplicates ---------------------------------------------------------
    con.execute("CREATE TEMP TABLE dup_ids AS SELECT id FROM raw_rows GROUP BY id HAVING count(*) > 1")
    con.execute("""
        CREATE TEMP TABLE dup_rows AS
        SELECT r.id, r.coin, r.kind, r.src_file, r.src_line, r.retrieved_utc,
               md5(r.text) AS text_md5, r.score, r.is_dead,
               CASE WHEN r.src_file LIKE 'out/%' THEN 'sequential' ELSE 'chunk' END AS phase,
               row_number() OVER (PARTITION BY r.id
                                  ORDER BY r.src_file, r.retrieved_utc DESC NULLS LAST, r.src_line) AS copy_rank
        FROM raw_rows r SEMI JOIN dup_ids d ON r.id = d.id
    """)
    bad = one(con, """SELECT count(*) FROM (SELECT id FROM dup_rows GROUP BY id
                      HAVING count(DISTINCT coin) > 1 OR count(DISTINCT kind) > 1)""")[0]
    assert bad == 0, f"{bad} ids occur in more than one coin or kind"
    con.execute("""
        CREATE TEMP TABLE dup_summary AS
        SELECT id, any_value(coin) AS coin, any_value(kind) AS kind,
               count(*) AS n_copies, count(DISTINCT src_file) AS n_files,
               bool_or(phase = 'sequential') AND bool_or(phase = 'chunk') AS spans_phases,
               count(DISTINCT text_md5) = 1 AS text_agree,
               count(DISTINCT score) = 1 AS score_agree,
               count(DISTINCT is_dead) = 1 AS dead_agree,
               max(retrieved_utc) - min(retrieved_utc) AS retrieved_spread_s,
               arg_min(src_file, copy_rank) AS kept_src_file,
               CASE WHEN bool_or(phase = 'sequential') AND bool_or(phase = 'chunk') THEN 'seam'
                    WHEN count(DISTINCT src_file) = 1 THEN 'within_file'
                    ELSE 'cross_file_same_phase' END AS dup_class
        FROM dup_rows GROUP BY id
    """)
    dup_path = os.path.join(run_dir, "duplicates.parquet")
    con.execute(f"COPY (SELECT * FROM dup_summary ORDER BY coin, id) TO '{dup_path}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD)")
    os.replace(dup_path + ".tmp", dup_path)
    con.execute("CREATE TEMP TABLE losers AS SELECT src_file, src_line, coin FROM dup_rows WHERE copy_rank > 1")

    dup_by_coin = {r[0]: r[1:] for r in con.execute("""
        SELECT coin, count(*), sum(n_copies - 1),
               count_if(dup_class = 'seam'), sum(n_copies - 1) FILTER (WHERE dup_class = 'seam'),
               count_if(dup_class = 'within_file'), sum(n_copies - 1) FILTER (WHERE dup_class = 'within_file'),
               count_if(dup_class = 'cross_file_same_phase'),
               sum(n_copies - 1) FILTER (WHERE dup_class = 'cross_file_same_phase'),
               count_if(NOT text_agree), count_if(NOT score_agree), count_if(NOT dead_agree)
        FROM dup_summary GROUP BY coin
    """).fetchall()}
    log("duplicates: " + ", ".join(f"{c} {v[1]:,} rows" for c, v in sorted(dup_by_coin.items())))

    # ---- per-coin outputs ---------------------------------------------------
    grand_out = 0
    for coin in COINS:
        tc = time.time()
        final = os.path.join(out_dir, f"reddit_{coin}.parquet")
        tmp = final + ".tmp"
        rows_in = one(con, f"SELECT count(*) FROM raw_rows WHERE coin = '{coin}'")[0]
        n_losers = one(con, f"SELECT count(*) FROM losers WHERE coin = '{coin}'")[0]
        con.execute(f"""
            COPY (
              SELECT {FINAL_SELECT}
              FROM raw_rows r
              WHERE r.coin = '{coin}'
                AND NOT EXISTS (SELECT 1 FROM losers l
                                WHERE l.src_file = r.src_file AND l.src_line = r.src_line)
              ORDER BY created_utc, id
            ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """)
        try:
            checks = check_output(con, tmp, coin, rows_in, n_losers)
        except AssertionError:
            os.remove(tmp)
            raise
        os.replace(tmp, final)
        grand_out += checks["rows_out"]

        coin_files = [f for f in files_stats if f["coin"] == coin]
        fc = Counter()
        dead_by = Counter()
        for f in coin_files:
            fc.update(f["counts"])
            dead_by.update(f["dead_by_reason"])
        d = dup_by_coin.get(coin, (0,) * 12)
        stats = {
            "output": final, "output_bytes": os.path.getsize(final), "output_md5": file_md5(final),
            "spec_version": SPEC_VERSION, "spec_sha": spec_sha(), "code_sha": code_sha(),
            "parse_code_sha": parse_code_sha(),
            "salt_path": SALT_PATH, "salt_fingerprint": salt_fingerprint(salt),
            "source": raw_verify["bucket"], "raw_dir": raw_verify["raw_dir"],
            "raw_verify_ok": raw_verify["ok"], "manifest_md5": raw_verify["manifest_md5"],
            "inputs": [{"file": f["file"], "bytes": f["bytes"], "md5": f["md5"]} for f in coin_files],
            "counts": {
                **{k: int(fc[k]) for k in FILE_COUNTERS},
                "kept_dead_by_reason": dict(dead_by),
                "dup_ids": int(d[0]), "dup_rows_removed": int(d[1] or 0),
                "dup_seam_ids": int(d[2]), "dup_seam_rows_removed": int(d[3] or 0),
                "dup_within_file_ids": int(d[4]), "dup_within_file_rows_removed": int(d[5] or 0),
                "dup_cross_file_same_phase_ids": int(d[6]),
                "dup_cross_file_same_phase_rows_removed": int(d[7] or 0),
                "dup_text_disagree_ids": int(d[8]), "dup_score_disagree_ids": int(d[9]),
                "dup_dead_disagree_ids": int(d[10]),
            },
            "checks": checks,
            "conservation": {
                "lines = blank + no_id + bad_created + oow + kept":
                    fc["lines"] == fc["blank"] + fc["no_id"] + fc["bad_created"]
                    + fc["oow_before"] + fc["oow_after"] + fc["kept"],
                "kept = rows_out + dup_rows_removed": fc["kept"] == checks["rows_out"] + (d[1] or 0),
            },
            "seconds": round(time.time() - tc, 1), "finished_at": utcnow(),
        }
        assert all(stats["conservation"].values()), f"{coin}: conservation failed {stats['conservation']}"
        write_json_atomic(os.path.join(out_dir, f"reddit_{coin}_stats.json"), stats)
        log(f"  {coin:7s} {checks['rows_out']:>11,} rows (live {checks['rows_live']:,}, dead "
            f"{checks['rows_dead']:,})  days={checks['utc_days']}  {checks['ts_min']} .. {checks['ts_max']}  "
            f"{stats['output_bytes'] / 1e6:,.0f} MB  {stats['seconds']:.0f} s")

    n_distinct = one(con, "SELECT count(DISTINCT id) FROM raw_rows")[0]
    assert grand_out == n_distinct, f"outputs hold {grand_out:,} rows, raw has {n_distinct:,} distinct ids"
    summary = {
        "finished_at": utcnow(), "seconds": round(time.time() - t0, 1),
        "spec_sha": spec_sha(), "code_sha": code_sha(),
        "totals": {k: int(tot[k]) for k in FILE_COUNTERS},
        "distinct_ids": int(n_distinct), "rows_out": int(grand_out),
        "duplicates_file": dup_path,
    }
    write_json_atomic(os.path.join(run_dir, "finalize_summary.json"), summary)
    write_field_lists(run_dir, files_stats)
    write_per_file_counts(run_dir, files_stats)
    con.close()
    log(f"finalize done in {time.time() - t0:,.0f} s: {grand_out:,} rows = {n_distinct:,} distinct ids")
    return 0


def check_output(con, path: str, coin: str, rows_in: int, n_losers: int) -> dict:
    """Assert the invariants of one per-coin output; return the measured values."""
    src = f"read_parquet('{path}')"
    (rows_out, n_ids, n_live, n_dead, n_posts, n_comments, ts_min, ts_max, days,
     null_core) = one(con, f"""
        SELECT count(*), count(DISTINCT id), count_if(NOT is_dead), count_if(is_dead),
               count_if(kind = 'posts'), count_if(kind = 'comments'),
               min(ts), max(ts), count(DISTINCT CAST(ts AS DATE)),
               count_if(id IS NULL OR ts IS NULL OR text IS NULL OR kind IS NULL OR is_dead IS NULL)
        FROM {src}""")
    assert rows_out == rows_in - n_losers, f"{coin}: {rows_out} rows out, expected {rows_in} - {n_losers}"
    assert n_ids == rows_out, f"{coin}: ids not unique"
    assert null_core == 0, f"{coin}: NULL in id/ts/text/kind/is_dead"
    assert n_posts + n_comments == rows_out, f"{coin}: kind outside posts/comments"
    missing_ids = one(con, f"""SELECT count(*) FROM (SELECT DISTINCT id FROM raw_rows WHERE coin = '{coin}')
                              ANTI JOIN {src} USING (id)""")[0]
    extra_ids = one(con, f"""SELECT count(*) FROM {src} o
                            ANTI JOIN (SELECT DISTINCT id FROM raw_rows WHERE coin = '{coin}') r USING (id)""")[0]
    assert missing_ids == 0 and extra_ids == 0, f"{coin}: id set differs ({missing_ids} missing, {extra_ids} extra)"
    viol = one(con, f"""
        SELECT
          count_if(is_dead <> (text = '')),
          count_if(is_dead <> (dead_reason IS NOT NULL)),
          count_if(has_quote_lines AND quoted_text IS NULL),
          count_if(NOT has_quote_lines AND (quoted_text IS NOT NULL OR own_text <> text)),
          count_if(author_deleted AND author_hash IS NOT NULL),
          count_if(author_hash IS NOT NULL AND NOT regexp_matches(author_hash, '^[0-9a-f]{{64}}$')),
          count_if(kind = 'comments' AND (removed_by_category IS NOT NULL OR is_robot_indexable IS NOT NULL)),
          count_if(kind = 'posts' AND (link_id IS NOT NULL OR parent_id IS NOT NULL)),
          count_if(ts < TIMESTAMPTZ '2019-08-01 00:00:00+00' OR ts > TIMESTAMPTZ '2023-01-31 23:59:59+00'),
          count_if(name_bot_token IS NULL OR name_has_bot IS NULL OR (name_bot_token AND NOT name_has_bot)),
          count_if(coin_label IS DISTINCT FROM '{coin}')
        FROM {src}""")
    names = ["dead_vs_empty_text", "dead_vs_reason", "quote_without_quoted_text",
             "no_quote_but_split", "deleted_author_hashed", "bad_author_hash",
             "rbc_on_comment", "thread_ids_on_post", "ts_outside_window",
             "name_flags", "coin_label"]
    violations = dict(zip(names, (int(v) for v in viol)))
    assert not any(violations.values()), f"{coin}: invariant violations {violations}"
    assert days == N_DAYS, f"{coin}: {days} UTC days, expected {N_DAYS}"
    live_days = one(con, f"SELECT count(DISTINCT CAST(ts AS DATE)) FROM {src} WHERE NOT is_dead")[0]
    return {
        "rows_in": int(rows_in), "dup_losers": int(n_losers), "rows_out": int(rows_out),
        "rows_live": int(n_live), "rows_dead": int(n_dead),
        "posts": int(n_posts), "comments": int(n_comments),
        "ts_min": str(ts_min), "ts_max": str(ts_max), "utc_days": int(days), "utc_days_live": int(live_days),
        "id_set_missing": int(missing_ids), "id_set_extra": int(extra_ids),
        **{f"violations_{k}": v for k, v in violations.items()},
    }


def write_per_file_counts(run_dir: str, files_stats: List[dict]) -> None:
    """One row per raw file: every counter plus the dead rows by reason (P11)."""
    path = os.path.join(run_dir, "per_file_counts.csv")
    reasons = list(DEAD_TEXTS.values())
    with open(path + ".tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "bytes", "md5", "subreddit", "kind", "coin", *FILE_COUNTERS,
                    *[f"dead_{r}" for r in reasons]])
        for f in sorted(files_stats, key=lambda x: x["file"]):
            w.writerow([f["file"], f["bytes"], f["md5"], f["subreddit"], f["kind"], f["coin"],
                        *[f["counts"][k] for k in FILE_COUNTERS],
                        *[f["dead_by_reason"].get(r, 0) for r in reasons]])
    os.replace(path + ".tmp", path)


def write_field_lists(run_dir: str, files_stats: List[dict]) -> None:
    """Fields present per (subreddit, kind, year) and per (file, year), for the report."""
    per_group: Dict[tuple, Counter] = {}
    n_group: Counter = Counter()
    per_file = {}
    for f in files_stats:
        fy: Dict[int, Counter] = {}
        for year, keys, n in f["keysets"]:
            g = (f["subreddit"], f["kind"], "sequential" if f["file"].startswith("out/") else "chunk", year)
            per_group.setdefault(g, Counter())
            for k in keys:
                per_group[g][k] += n
            n_group[g] += n
            fy.setdefault(year, Counter())
            for k in keys:
                fy[year][k] += n
        per_file[f["file"]] = {str(y): dict(cn) for y, cn in fy.items()}
    groups = []
    for g, cn in sorted(per_group.items(), key=lambda x: tuple(str(v) for v in x[0])):
        groups.append({"subreddit": g[0], "kind": g[1], "phase": g[2], "year": g[3],
                       "records": n_group[g], "fields": {k: v for k, v in sorted(cn.items())}})
    all_keys = sorted({k for g in groups for k in g["fields"]})
    html_like = [k for k in all_keys if "html" in k.lower() or "quote" in k.lower()]
    write_json_atomic(os.path.join(run_dir, "fields_by_group_year.json"),
                      {"all_fields": all_keys, "html_or_quote_fields": html_like, "groups": groups})
    write_json_atomic(os.path.join(run_dir, "fields_by_file_year.json"), per_file)


# ---------------------------------------------------------------------------

def check_window_matches_config() -> None:
    """The window is restated here (it is part of the spec the shards were parsed under); it must equal config's."""
    from pipeline import config
    start = calendar.timegm(time.strptime(config.WINDOW_START, "%Y-%m-%d"))
    end = calendar.timegm(time.strptime(config.WINDOW_END, "%Y-%m-%d")) + 86399
    if (START_UTC, END_UTC, N_DAYS) != (start, end, config.N_DAYS):
        sys.exit(f"window {START_UTC}..{END_UTC} ({N_DAYS} days) differs from pipeline/config.py "
                 f"{config.WINDOW_START}..{config.WINDOW_END} ({config.N_DAYS} days)")


def main(argv=None) -> int:
    check_window_matches_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--raw", required=True)
    v.add_argument("--run-dir", required=True)
    v.add_argument("--bucket", default=BUCKET)
    v.add_argument("--manifest", default=None,
                   help="check against a stored raw_manifest.csv instead of the bucket listing")
    v.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("parse")
    p.add_argument("--raw", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--shard-dir", default=None, help="where the shards go (default <run dir>/shards)")
    f = sub.add_parser("finalize")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--spill", default=None, help="DuckDB spill directory (default <run dir>/_duckdb_spill)")
    args = ap.parse_args(argv)
    return {"verify": cmd_verify, "parse": cmd_parse, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
