"""Paths and fixed settings shared by every pipeline step.

Everything a step reads or writes is named here, except the Reddit author salt, which lives
outside the repository (SALT_PATH in 01_extract_reddit.py). Paths are derived from this file's
location (code/pipeline/), never from the working directory, so a step behaves the same
wherever it is started. Every path can be moved with an environment variable (THESIS_*);
the tests use this to run the whole pipeline in a temporary folder.

Layout under code/data/ (not part of the repository; see data/README.md)
    raw/        inputs from outside the pipeline: the local copy of the Reddit archive
                (needed only to re-run step 1), its checked manifest, and the capture
                ledgers of the Twitter extraction
    pipeline/   everything steps 1-5 write
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]          # code/
DATA_DIR = CODE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                              # inputs from outside the pipeline
OUT_DIR = Path(os.environ.get("THESIS_PIPELINE_OUT", DATA_DIR / "pipeline"))   # everything this pipeline writes
LOG_DIR = Path(os.environ.get("THESIS_LOG_DIR", CODE_DIR / "logs"))           # run logs (not in git)

# Scratch space for downloads and DuckDB spill files, outside the repository so temporary
# files never mix with results.
SCRATCH_DIR = Path(os.environ.get("THESIS_SCRATCH", "/tmp/thesis_pipeline"))
SPILL_DIR = SCRATCH_DIR / "duckdb_spill"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
# Reddit: the Arctic Shift dumps (bucket folder reddit_arcticshift_crypto, with chunks/ and
# out/), and the manifest of that folder (md5 per object, checked against the bucket on
# 23 Sep 2026), which `01_extract_reddit.py verify --manifest` checks the local copy against.
REDDIT_RAW_DIR = Path(os.environ.get("THESIS_REDDIT_RAW", RAW_DIR / "reddit_arcticshift_crypto"))
REDDIT_RAW_MANIFEST = Path(os.environ.get("THESIS_REDDIT_MANIFEST", RAW_DIR / "reddit_raw_manifest_20260923.csv"))
# StockTwits and Twitter Stage A (the outputs of step 1) are kept on Google Drive and read
# through rclone. Set THESIS_DRIVE_REMOTE to the name of the rclone remote that holds them,
# with its colon (for example "mydrive:"); remote names are private to a machine, so none is stored here.
RCLONE_BIN = os.environ.get("RCLONE_BIN") or shutil.which("rclone") or "rclone"
DRIVE_REMOTE = os.environ.get("THESIS_DRIVE_REMOTE", "<remote>:")
STOCKTWITS_STAGE_A_REMOTE = DRIVE_REMOTE + "stocktwits_parquet"      # month=YYYY-MM/
TWITTER_STAGE_A_REMOTE = DRIVE_REMOTE + "twitter_parquet"             # YYYY/date=YYYY-MM-DD/
# Twitter capture ledgers: one per archive, written by step 1 (_ledger/<archive>.stats.json);
# step 2 builds the UTC coverage table from them.
TWITTER_LEDGER_DIR = Path(os.environ.get("THESIS_TWITTER_LEDGERS", RAW_DIR / "twitter_capture_ledgers"))

# ---------------------------------------------------------------------------
# Sample window (UTC)
# ---------------------------------------------------------------------------
WINDOW_START = "2019-08-01"          # first UTC day, inclusive
WINDOW_END = "2023-01-31"            # last UTC day, inclusive
N_DAYS = 1280

COINS = ("BTC", "ETH", "BNB", "XRP", "DOGE")
PLATFORMS = ("reddit", "stocktwits", "twitter")

# ---------------------------------------------------------------------------
# DuckDB settings (decided 22-23 Sep 2026: 4 GB, 3 threads, spill, UTC)
# ---------------------------------------------------------------------------
DUCKDB_MEMORY = os.environ.get("THESIS_DUCKDB_MEMORY", "4GB")
DUCKDB_THREADS = int(os.environ.get("THESIS_DUCKDB_THREADS", "3"))


def out_path(platform: str, stage: str, *parts: str) -> Path:
    """code/data/pipeline/<platform>/<stage>/<parts...>"""
    return OUT_DIR.joinpath(platform, stage, *parts)
