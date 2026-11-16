#!/usr/bin/env python3
"""
06_emotion_emtract.py -- step 06: EmTract emotion scores (7 classes) for every text of the study:
the social-media posts (Reddit, StockTwits, Twitter), the DLT-Tweets event corpus and the RavenPack
news headlines (a control variable). One script; its subcommands run in the order below.

EmTract (Vamossy & Skog; MIT licence; model `emotion-stocktwits`, a DistilBERT classifier with the
classes neutral, happy, sad, anger, disgust, surprise, fear) is run as its three parts instead of
through its `Model.predict()`. The method -- settings, EmTract's own cleaning, the tokenizer, the
window plan for long texts, the model inputs, and why the parts run separately -- is in
emtract_method.py. This script holds the runs: it reads the tables of steps 01-04, writes new tables
next to them, and changes none of theirs. Every output is written through a temporary file, checked,
moved into place and described by a stats file (runner.py), and a stage skips an output whose stats
still match its inputs, code and rules.

Sections and subcommands (run from code/; [environment], see README.md "Environments")
---------------------------------------------------------------------------------------
  1  tokenize <reddit|stocktwits|twitter>        [emtract-ref, CPU]  stage 1: clean once, tokenize uncut,
                                                 plan the windows -> <platform>/emtract/tokens/
  2  pack <platform> --batch-id ID --max-inputs N  [thesis or emtract-ref]   model inputs of finished
                                                 shards -> _emtract_run/modal_inputs/<platform>/
     score --inputs FILE --out FILE --device cuda|mps|cpu   [thesis; GPU]   the model alone, fp32
     collect <platform> --batch-id ID            [thesis]   raw scores + map -> <platform>/emtract/scores/
  3  merge --reference-empty-score --out FILE    [emtract-ref]  EmTract's own predict('') (reference)
     merge --empty-score [--reference FILE]      [thesis]   the fixed score of an empty text (recorded only)
     merge <platform> [--partitions ...]         [thesis]   one row per (row_id, version): p, pc, pw
                                                 -> <platform>/emtract/emotion/<partition>.parquet
  4  text-key-index [twitter]                    [thesis]   one row per distinct cleaned Twitter text
                                                 -> twitter/emtract/text_key_index.parquet
     text-key-list                               [thesis]   its keys as 32-byte binary, sorted (input of
                                                 `dlt match`) -> twitter/emtract/text_key_list.parquet
  5  dlt clean | match | tokenize | verify       [emtract-ref; match: thesis]   the DLT corpus: rows whose
                                                 cleaned text was already scored on Twitter reuse that
                                                 score, the other distinct texts get stage 1 here
     pack / score / collect with platform 'dlt'  as section 2
     dlt completeness | merge-keys | build       [thesis]   -> dlt/emtract/emotion/dlt_emotion.parquet
     dlt refscore                                [emtract-ref]  EmTract's predict() on a sample
  6  news prepare | stage1 | score | refcheck    [emtract-ref; score: thesis on CPU or MPS]
                                                 -> ravenpack/emtract/news_headline_emotion.parquet

How the stored tables were made (29 Sep - 1 Oct 2026; README.md has hardware, times and costs)
-----------------------------------------------------------------------------------------------
    python pipeline/06_emotion_emtract.py tokenize reddit --workers 58 --start-method fork --shard-rows 20000
    python pipeline/06_emotion_emtract.py tokenize stocktwits --workers 58 --start-method fork --shard-rows 20000
    python pipeline/06_emotion_emtract.py tokenize twitter --workers 58 --start-method fork --shard-rows 20000
        (stage 1 ran on one 60-vCPU cloud VM: 5,363 shards, 100,281,908 texts, 121,637,153 model inputs)
    python pipeline/06_emotion_emtract.py pack reddit --batch-id reddit-0001 --max-inputs 5000000
        (repeated, platform by platform, until every finished shard was packed: 26 batches)
    python pipeline/06_emotion_emtract.py score --inputs <modal_inputs>/reddit/reddit-0001.parquet \\
        --out <modal_outputs>/reddit/reddit-0001.parquet --device cuda --model-dir <emtract_models>/stocktwits
        (one batch per rented NVIDIA L4 GPU container; the command is plain PyTorch and runs on any
        CUDA machine, or on the CPU / Apple MPS at lower speed)
    python pipeline/06_emotion_emtract.py collect reddit --batch-id reddit-0001
    python pipeline/06_emotion_emtract.py merge --reference-empty-score --out ref_empty.json   (emtract-ref)
    python pipeline/06_emotion_emtract.py merge --empty-score --reference ref_empty.json
    python pipeline/06_emotion_emtract.py merge reddit      (then stocktwits, twitter)
    python pipeline/06_emotion_emtract.py text-key-index
    python pipeline/06_emotion_emtract.py text-key-list
    python pipeline/04_series_dlt.py tickers ; ... units ; ... profile ; ... precheck     (DLT units)
    python pipeline/06_emotion_emtract.py dlt clean
    python pipeline/06_emotion_emtract.py dlt match --key-list data/pipeline/twitter/emtract/text_key_list.parquet
    python pipeline/06_emotion_emtract.py dlt tokenize ; dlt verify
    pack / score / collect for platform dlt (4 batches) ; dlt completeness ; dlt merge-keys ; dlt build
    python pipeline/06_emotion_emtract.py dlt refscore      (emtract-ref)
    python pipeline/06_emotion_emtract.py news prepare ; news stage1     (emtract-ref)
    python pipeline/06_emotion_emtract.py news score --device mps ; news refcheck --device cpu

The text-key list and the DLT ticker list were written by one-off commands on 30 Sep 2026; the
subcommands `text-key-list` and `04_series_dlt.py tickers` (added 2 Oct 2026) write the same files.
Tags such as D17, PI2 or "decision (4) of 29 Sep" refer to the thesis's decision log (README.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform as _platform
import random
import resource
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from pipeline import config  # noqa: E402
from pipeline import emtract_method as E  # noqa: E402  (standard library only at import)
from pipeline.emtract_method import CheckError, check, log  # noqa: E402

try:                                    # the safe-runner rules (runner.py imports DuckDB)
    from pipeline import runner  # noqa: E402
except ImportError:                     # a GPU machine that only runs `score` needs no DuckDB
    runner = None

STEP = "06_emotion_emtract"             # stats files name the subcommand: "06_emotion_emtract:pack", ...
STEP_TOKENIZE = STEP + ":tokenize"
STEP_MERGE = STEP + ":merge"
STEP_TKI = STEP + ":text-key-index"
STEP_DLT = STEP + ":dlt"
STEP_NEWS = STEP + ":news"

LABELS = E.LABELS
PROB_SUM_TOL = 1e-5                     # brief of 29 Sep 2026: probabilities sum to 1 within 1e-5
P_COLS = [f"p_{lab}" for lab in E.LABELS]
PC_COLS = [f"pc_{lab}" for lab in E.LABELS]
PW_COLS = [f"pw_{lab}" for lab in E.LABELS]
LOGIT_COLS = [f"logit_{lab}" for lab in E.LABELS]


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """sha256 over this script and emtract_method.py (runner.code_fingerprint): every stats file of
    this step records it, and a stage redoes an output whose recorded fingerprint differs."""
    return runner.code_fingerprint(Path(__file__).resolve(), Path(E.__file__).resolve())


def _np(col) -> np.ndarray:
    """numpy view of an arrow column (pyarrow 9 in emtract-ref has no zero_copy_only on ChunkedArray)."""
    if isinstance(col, np.ndarray):
        return col
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return col.to_numpy(zero_copy_only=False)


# ==================================================================================================
# 1. Stage 1 (`tokenize`): EmTract's clean_text once per text, tokens uncut, window plan
# ==================================================================================================
# Input    Reddit      data/pipeline/reddit/extract/reddit_<P>.parquet, P = BNB, BTC, DOGE, ETH,
#                      MARKET, XRP (step 1; columns id, text, own_text, quoted_text,
#                      has_quote_lines, is_dead)
#          StockTwits  data/pipeline/stocktwits/coarse/month=YYYY-MM/data_0.parquet (step 2;
#                      columns message_id, message_body)
#          Twitter     data/pipeline/twitter/units/units.parquet (step 2; columns unit_id, text,
#                      parent_text)
#          or, with --input-file, one prepared parquet with row_id (or the native id column) and
#          the platform's text columns, so that the same command runs on a machine
#          without the local tables.
# Output   data/pipeline/<platform>/emtract/tokens/<partition>/shard-NNNNN.parquet, each with a
#          stats file shard-NNNNN.parquet.stats.json (runner.write_stats). partition = the Reddit
#          coin label, the StockTwits month YYYY-MM, or 'units' for Twitter. Nothing is written
#          into the tables of steps 1-5. Also data/pipeline/_emtract_run/status_stage1.json
#          (this step is its only writer) and a lock file next to it.
#
# What stage 1 does (MASTERPLAN_NOV15.md section 7, decisions of 29 Sep 2026; EmTract plan,
# "Block A outcome", stage 1; the helpers are in emtract_method.py, which explains each rule)
# ------------------------------------------------------------------------------------------------
# 1. Splits each source file into shards: one shard per Parquet row group (the row groups hold
#    11k-252k rows), a row group above SHARD_MAX_ROWS split into equal pieces. A shard is the
#    unit of work, of checking and of resuming.
# 2. For every live source row of a shard, builds the text versions (own for every row; quoted
#    and both where a quoted text exists), cleans each version once with EmTract's own
#    clean_text, flags missing texts, texts empty after cleaning and cleaning errors (PI2),
#    tokenizes the cleaned text with tokenizers 0.10.3 without cutting, and plans the 62-wordpiece
#    windows of texts longer than 62 wordpieces. Reddit dead rows get no row (counted).
# 3. Writes one row per (source row, version), in source-row order, versions in the order own,
#    quoted, both, with the columns of the shared interface (item 1): row_id (the native id, type
#    as in the source), version, cleaned_text, is_missing_text, is_empty_after_clean,
#    is_clean_error, clean_error, n_tokens (int32), input_ids (list<uint16>, uncut, no special
#    tokens) and window_starts (list<int32>).
#
# Checks per shard (the safe-runner standard of runner.py: temporary file -> checks -> commit ->
# stats; a failed check commits nothing and stops the run)
# --------------------------------------------------------------------------------------------
# * rows in = live rows ('own' rows written) + dead rows skipped;
# * versions as expected: every live row has exactly one 'own' row; 'quoted' and 'both' exactly
#   where a quoted text exists; row order = source order;
# * n_tokens = len(input_ids) for every row; every id fits uint16 and survives the write
#   (read back: row count, total id count and id sum);
# * every window plan is valid (first start 0, last window ends at the last wordpiece, overlaps
#   >= 16, interior moves <= 8) and empty or missing texts have no window;
# * flags agree with the columns (missing -> cleaned_text NULL and no tokens; empty -> "" and no
#   tokens);
# * spot check: for 1% of the rows with tokens (at least one), and for up to 50 texts over 62
#   wordpieces, [CLS] + input_ids[:62] + [SEP] equals the tokenizer's own truncation
#   (truncation=True, max_length=64), i.e. exactly what predict() gives the model.
#
# Resuming, load and disk (plan step 7; decisions of 29 Sep 2026)
# ---------------------------------------------------------------
# * A shard is skipped when its file and stats file match the current inputs (path, bytes, md5 of
#   the source file), code and rules fingerprints and shard range (runner.is_current plus the
#   range). An interrupted run therefore restarts where it stopped; a shard written with
#   --limit-rows is redone by a full run because its range differs.
# * Work runs in long-lived worker processes (each loads the cleaning dictionaries and the
#   tokenizer once; the tokenizer takes about a minute to load), niced with os.nice(10) and with
#   TOKENIZERS_PARALLELISM=false, so each worker uses one core.
# * The number of workers comes from --workers (default DEFAULT_WORKERS = 4). CONTROL_FILE
#   (JSON: {"workers": N, "pause": true|false}) is read between shards: "pause" true lets the
#   running shards finish and starts no new one; "workers" changes the count without a restart.
#   A "workers" value already in the control file when the run starts is used only if --workers
#   is not given; later edits of the file always apply. Workers above the target stop after
#   their current shard.
# * While free disk is below MIN_FREE_DISK_GB (25 GB) no new shard starts; the check repeats
#   every minute.
# * STATUS_STAGE1_FILE is rewritten after every shard (and every minute while waiting): shards
#   done/total per partition, rows/s, ETA, worker RSS, load average, free disk, available memory
#   and swap in use.
# * Only one `tokenize` process runs at a time (a lock file next to the status file).
# * Speed, with identical output: EmTract's clean_text calls advertools.extract_emoji on the whole
#   string, which runs one regular-expression search per character; that search took 90 of 97 s of
#   cleaning in the profile of 29 Sep 2026. emtract_method.install_emoji_memo() memoizes that
#   per-character lookup exactly (same result for the same character); it is on unless
#   --no-emoji-memo is given, and each stats file records it. Output with and without the memo was
#   byte-identical on the pilot rows. --start-method fork loads the EmTract stack once in the
#   parent and shares it with the workers (less memory per worker); spawn (the default) loads it
#   in every worker.
#
# --out-root DIR moves the token shards, the status file, the control file and the lock to DIR
# (for pilots); the source tables are still read from their usual place. --limit-rows N takes only
# the first N rows of each partition and --shard-rows N cuts smaller shards (both for pilots; a
# full run redoes such shards, because their ranges differ).
#
# Usage (environment emtract-ref)
# -------------------------------
#     python pipeline/06_emotion_emtract.py tokenize reddit [--partitions BTC ETH] [--workers 4]
#     python pipeline/06_emotion_emtract.py tokenize stocktwits [--partitions 2019-08]
#     python pipeline/06_emotion_emtract.py tokenize twitter
#     python pipeline/06_emotion_emtract.py tokenize reddit --partitions MARKET --limit-rows 100000 --out-root /tmp/pilot
#     python pipeline/06_emotion_emtract.py tokenize twitter --input-file prepared.parquet --partition-name units --out-root out
#     python pipeline/06_emotion_emtract.py tokenize reddit --plan        (list the shards and exit)
SPOT_SHARE = 0.01                 # share of rows with tokens checked against the tokenizer's truncation
SPOT_LONG_MAX = 50                # extra spot checks among texts over 62 wordpieces
POLL_SECONDS = 2.0
DISK_RECHECK_SECONDS = 60.0


# --- Where things are ------------------------------------------------------------------------------

class Paths:
    """Output locations; --out-root moves all of them (tokens, status, control, lock)."""

    def __init__(self, platform: str, out_root: Optional[str]):
        self.platform = platform
        if out_root:
            root = Path(out_root).resolve()
            self.tokens = root / platform / "emtract" / "tokens"
            run_tokenize = root / "_emtract_run"
            self.control = run_tokenize / E.CONTROL_FILE.name
            self.status = run_tokenize / E.STATUS_STAGE1_FILE.name
        else:
            self.tokens = E.tokens_dir(platform)
            self.control = E.CONTROL_FILE
            self.status = E.STATUS_STAGE1_FILE
        self.lock = self.status.with_name(self.status.name + ".lock")


def id_column(platform: str, schema_names: List[str], input_file: bool) -> str:
    native = E.SOURCES[platform]["id"]
    if input_file and "row_id" in schema_names:
        return "row_id"
    if native in schema_names:
        return native
    raise SystemExit(f"source has neither 'row_id' nor '{native}'")


def shard_file(paths: Paths, spec: Dict) -> Path:
    return paths.tokens / spec["partition"] / f"shard-{spec['shard_index']:05d}.parquet"


def rules_fingerprint(tok_dir: Path) -> Tuple[str, Dict[str, str]]:
    """sha256 over what shapes the tokens besides this code: the vendored cleaning code and its
    dictionaries, the tokenizer files, the library versions and the window settings."""
    import advertools, emoji, symspellpy, tokenizers, transformers  # noqa: E401
    h = hashlib.sha256()
    files = sorted((E.EMTRACT_CODE_DIR / "emtract").rglob("*.py")) + \
        sorted((E.EMTRACT_CODE_DIR / "emtract" / "data" / "dictionaries").glob("*.csv"))
    tok_files = [tok_dir / n for n in ("tokenizer.json", "vocab.txt", "added_tokens.json",
                                       "special_tokens_map.json", "tokenizer_config.json")]
    for f in files + [f for f in tok_files if f.exists()]:
        h.update(f.name.encode()); h.update(f.read_bytes())
    versions = {"tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
                "emoji": emoji.__version__, "advertools": advertools.__version__,
                "symspellpy": getattr(symspellpy, "__version__", "?"), "python": sys.version.split()[0]}
    h.update(json.dumps(versions, sort_keys=True).encode())
    h.update(repr((E.MAX_LENGTH, E.MAX_TEXT_TOKENS, E.MIN_WINDOW_OVERLAP, E.MAX_MOVE, E.VERSIONS,
                   E.COMBINED_SEPARATOR, E.TOKENIZER_VARIANT)).encode())
    return h.hexdigest()[:16], versions


def shard_is_current(final: Path, spec: Dict, inputs: List[Dict], code_sha: str, rules_sha: str) -> bool:
    if not runner.is_current(final, inputs, code_sha, rules_sha):
        return False
    try:
        rec = json.loads(runner.stats_path(final).read_text())
    except (OSError, ValueError):
        return False
    keys = ("source", "row_group", "rg_offset", "rg_stop", "row_start", "row_stop")
    return all(rec.get("shard", {}).get(k) == spec[k] for k in keys)


# --- One shard (runs inside a worker) --------------------------------------------------------------

def process_shard(spec: Dict, ctx: Dict, tokens: "E.Tokens") -> Dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    platform = spec["platform"]
    final = Path(ctx["shard_file"])
    text_cols = list(E.SOURCES[platform]["text"])
    pf = pq.ParquetFile(spec["source"])
    idcol = id_column(platform, pf.schema_arrow.names, ctx["input_file"])
    tab = pf.read_row_group(spec["row_group"], columns=[idcol] + text_cols)
    tab = tab.slice(spec["rg_offset"], spec["rg_stop"] - spec["rg_offset"])
    n_in = tab.num_rows
    cols = {c: tab.column(c).to_pylist() for c in text_cols}
    checks: List[Dict[str, object]] = []
    if platform == "reddit":
        check(all(v is not None for v in cols["is_dead"]), "is_dead is never NULL", checks)

    out, counts = E.stage1_rows(platform, cols, tokens)
    src_idx = out["source_index"]
    n_out = len(src_idx)

    # --- versions as expected -------------------------------------------------------------
    check(counts["rows_in"] == n_in, "all source rows were read", checks)
    own_rows = counts["by_version"]["own"]
    check(n_in == own_rows + counts["rows_dead"], "rows in = rows written (own) + dead rows skipped", checks)
    if platform == "reddit":
        exp_quoted = sum(1 for d, q in zip(cols["is_dead"], cols["quoted_text"]) if not d and q is not None)
        exp_live = sum(1 for d in cols["is_dead"] if not d)
    elif platform == "twitter":
        exp_quoted = sum(1 for q in cols["parent_text"] if q is not None)
        exp_live = n_in
    else:
        exp_quoted, exp_live = 0, n_in
    check(own_rows == exp_live, "one 'own' row per live source row", checks)
    check(counts["by_version"]["quoted"] == exp_quoted and counts["by_version"]["both"] == exp_quoted,
          "'quoted' and 'both' rows exactly where a quoted text exists", checks)
    order_ok = all(src_idx[i] <= src_idx[i + 1] for i in range(n_out - 1))
    rank = {v: i for i, v in enumerate(E.VERSIONS)}
    ver = out["version"]
    order_ok = order_ok and all(src_idx[i] < src_idx[i + 1] or rank[ver[i]] < rank[ver[i + 1]]
                                for i in range(n_out - 1))
    check(order_ok, "rows in source order, versions own, quoted, both", checks)

    # --- tokens, flags, windows ----------------------------------------------------------
    ids, ntok, starts = out["input_ids"], out["n_tokens"], out["window_starts"]
    check(all(n == len(i) for n, i in zip(ntok, ids)), "n_tokens = len(input_ids)", checks)
    check(counts["max_token_id"] < 65536 and counts["max_token_id"] < tokens.n_vocab,
          "every token id fits uint16 and the vocabulary", checks)
    bad_plan = 0
    for n, s in zip(ntok, starts):
        try:
            E.check_window_plan(n, s)
        except ValueError:
            bad_plan += 1
    check(bad_plan == 0, "every window plan is valid (0 empty, [0] short, k windows long)", checks)
    ct, miss, emp, err = out["cleaned_text"], out["is_missing_text"], out["is_empty_after_clean"], out["is_clean_error"]
    flags_ok = all(
        (c is None) == m and (not m or (n == 0 and not e and not r)) and (not e or (c == "" and n == 0))
        and (not r or e) and ((c is None or c == "") == (m or e))
        for c, m, e, r, n in zip(ct, miss, emp, err, ntok))
    check(flags_ok, "flags agree with cleaned_text and n_tokens", checks)

    # --- spot check: slicing = the tokenizer's own truncation ----------------------------
    rng = random.Random(f"{platform}/{spec['partition']}/{spec['shard_index']}")
    with_tok = [i for i, n in enumerate(ntok) if n > 0]
    k = min(len(with_tok), max(1, math.ceil(SPOT_SHARE * len(with_tok)))) if with_tok else 0
    sample = set(rng.sample(with_tok, k)) if k else set()
    long_rows = [i for i in with_tok if ntok[i] > E.WINDOW]
    sample |= set(rng.sample(long_rows, min(SPOT_LONG_MAX, len(long_rows))))
    sample = sorted(sample)
    trunc = tokens.encode_truncated([ct[i] for i in sample]) if sample else []
    spot_ok = all(t == E.primary_input(ids[i], tokens.cls_id, tokens.sep_id) for i, t in zip(sample, trunc))
    check(spot_ok, f"[CLS]+ids[:62]+[SEP] = tokenizer truncation on {len(sample)} spot-checked rows", checks)

    # --- write, read back, commit --------------------------------------------------------
    row_id = tab.column(idcol).combine_chunks().take(pa.array(src_idx, type=pa.int64())) if n_out else \
        pa.array([], type=tab.schema.field(idcol).type)
    table = pa.table({
        "row_id": row_id,
        "version": pa.array(ver, type=pa.string()),
        "cleaned_text": pa.array(ct, type=pa.string()),
        "is_missing_text": pa.array(miss, type=pa.bool_()),
        "is_empty_after_clean": pa.array(emp, type=pa.bool_()),
        "is_clean_error": pa.array(err, type=pa.bool_()),
        "clean_error": pa.array(out["clean_error"], type=pa.string()),
        "n_tokens": pa.array(ntok, type=pa.int32()),
        "input_ids": pa.array(ids, type=pa.list_(pa.uint16())),
        "window_starts": pa.array(starts, type=pa.list_(pa.int32())),
    })
    tmp = final.parent / "_tmp" / final.stem / final.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    pq.write_table(table, str(tmp), compression="zstd")
    back = pq.read_table(str(tmp), columns=["row_id", "n_tokens", "input_ids"])
    flat = back.column("input_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    id_sum = int(sum(sum(i) for i in ids))
    check(back.num_rows == n_out, "written rows = rows built", checks)
    check(len(flat) == sum(ntok) and int(np.asarray(flat, dtype=np.int64).sum()) == id_sum,
          "written ids = built ids (count and sum)", checks)
    check(back.column("row_id").type == tab.schema.field(idcol).type, "row_id keeps the source type", checks)
    runner.commit(tmp, final)
    try:
        tmp.parent.rmdir()
    except OSError:
        pass
    secs = time.time() - t0
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        maxrss_mb = maxrss / 2**20 if sys.platform == "darwin" else maxrss / 1024
    except Exception:  # noqa: BLE001
        maxrss_mb = None
    counts.update({"rows_written": n_out, "spot_checked": len(sample), "seconds": round(secs, 2),
                   "source_rows_per_s": round(n_in / secs, 1) if secs else None,
                   "texts_per_s": round(n_out / secs, 1) if secs else None,
                   "bytes": final.stat().st_size})
    runner.write_stats(final, step=STEP_TOKENIZE, inputs=ctx["inputs"], code_sha=ctx["code_sha"], rules_sha=ctx["rules_sha"],
                       con=None, counts=counts, checks=checks,
                       extra={"platform": platform, "partition": spec["partition"],
                              "shard": {k: spec[k] for k in ("source", "row_group", "rg_offset", "rg_stop",
                                                             "row_start", "row_stop", "shard_index")},
                              "row_id_source_column": E.SOURCES[platform]["id"] if idcol == "row_id" else idcol,
                              "input_file_mode": ctx["input_file"], "emoji_memo": E.emoji_memo_active(),
                              "library_versions": ctx["versions"], "worker_maxrss_mb": maxrss_mb,
                              "special_ids": {"cls": tokens.cls_id, "sep": tokens.sep_id, "pad": tokens.pad_id}})
    return {"rows_in": n_in, "rows_written": n_out, "seconds": secs, "maxrss_mb": maxrss_mb, "shard_file": str(final),
            "bytes": counts["bytes"], "clean_error": counts["clean_error"], "empty": counts["empty_after_clean"]}


_PRELOADED: Optional["E.Tokens"] = None      # set in the parent with --start-method fork


def load_stack(tok_dir: str, emoji_memo: bool) -> "E.Tokens":
    E.load_cleaning()
    if emoji_memo:
        E.install_emoji_memo()
    return E.Tokens(Path(tok_dir))


def worker_main(conn, tok_dir: str, emoji_memo: bool = True) -> None:
    """Long-lived worker: load once (or inherit the parent's copy with fork), then process the
    shards the parent sends until None."""
    try:
        os.nice(10)
    except OSError:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        tokens = _PRELOADED if _PRELOADED is not None else load_stack(tok_dir, emoji_memo)
        if E.emoji_memo_active() != emoji_memo:
            raise RuntimeError("emoji memo state differs from the requested one")
        conn.send(("ready", os.getpid()))
    except Exception:  # noqa: BLE001
        conn.send(("error", None, traceback.format_exc()))
        return
    while True:
        msg = conn.recv()
        if msg is None:
            break
        spec, ctx = msg
        try:
            conn.send(("done", spec, process_shard(spec, ctx, tokens)))
        except Exception:  # noqa: BLE001
            conn.send(("error", spec, traceback.format_exc()))


# --- Parent: scheduling, control file, disk guard, status ------------------------------------------

def read_control(path: Path) -> Tuple[Optional[Dict], Optional[float]]:
    try:
        st = path.stat()
    except OSError:
        return None, None
    try:
        return json.loads(path.read_text()), st.st_mtime
    except (OSError, ValueError):
        log(f"control file is not valid JSON; ignored: {path}")
        return None, st.st_mtime


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(str(p)).free / 1e9


def run_tokenize(args) -> int:
    import psutil

    platform = args.platform
    paths = Paths(platform, args.out_root)
    if args.input_file:
        src = Path(args.input_file).resolve()
        part = args.partition_name or src.stem
        sources = [(part, src)]
    else:
        sources = E.source_files(platform, args.partitions)
    tok_dir = E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT
    code_sha = code_fingerprint()
    rules_sha, versions = rules_fingerprint(tok_dir)
    check(versions["tokenizers"] == E.TOKENIZERS_VERSION, f"tokenizers {E.TOKENIZERS_VERSION} (decision (6))")

    todo, per_part = [], {}
    for part, path in sources:
        shards = E.plan_shards(platform, part, path, args.limit_rows, args.shard_rows)
        per_part[part] = {"shards_total": len(shards), "shards_done": 0, "shards_skipped": 0,
                          "rows_total": sum(s["row_stop"] - s["row_start"] for s in shards), "rows_done": 0,
                          "shards_committed": 0, "rows_committed": 0}
        if args.plan:
            for s in shards:
                print(part, s["shard_index"], s["row_group"], s["row_start"], s["row_stop"])
            continue
        log(f"{part}: {len(shards)} shards; hashing {path.name}")
        inputs = runner.describe_inputs([path])
        for s in shards:
            final = shard_file(paths, s)
            if shard_is_current(final, s, inputs, code_sha, rules_sha):
                per_part[part]["shards_skipped"] += 1
                per_part[part]["shards_committed"] += 1
                per_part[part]["rows_committed"] += s["row_stop"] - s["row_start"]
                per_part[part]["shards_done"] += 1
                per_part[part]["rows_done"] += s["row_stop"] - s["row_start"]
                continue
            todo.append((s, {"shard_file": str(final), "inputs": inputs, "code_sha": code_sha,
                             "rules_sha": rules_sha, "versions": versions, "input_file": bool(args.input_file)}))
    if args.plan:
        return 0
    paths.status.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_fh = open(paths.lock, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another `tokenize` run holds {paths.lock}")
    log(f"{len(todo)} shards to do, {sum(p['shards_skipped'] for p in per_part.values())} current and skipped")

    ctl0, ctl_mtime0 = read_control(paths.control)
    target = args.workers or (ctl0 or {}).get("workers") or E.DEFAULT_WORKERS
    paused = bool((ctl0 or {}).get("pause", False))
    mctx = mp.get_context(args.start_method)
    if args.start_method == "fork":
        # load EmTract's dictionaries and the tokenizer once in the parent; forked workers share
        # these pages copy-on-write (gc.freeze keeps the collector from touching them)
        global _PRELOADED
        import gc
        log("loading the cleaning module and the tokenizer in the parent (fork mode)")
        _PRELOADED = load_stack(str(tok_dir), not args.no_emoji_memo)
        gc.collect()
        gc.freeze()
    workers: Dict[int, Dict] = {}        # wid -> {proc, conn, spec, ready, peak_rss_mb}
    next_wid = 0
    started = time.time()
    rows_this_run = 0
    errors: List[str] = []
    last_disk = 0.0
    disk_ok = True
    last_status = 0.0
    pending = list(todo)

    def write_status(note: str = "") -> None:
        elapsed = max(time.time() - started, 1e-9)
        rate = rows_this_run / elapsed
        remaining = sum(p["rows_total"] - p["rows_done"] for p in per_part.values())
        rss = {}
        for wid, w in workers.items():
            try:
                rss[str(w["proc"].pid)] = round(psutil.Process(w["proc"].pid).memory_info().rss / 2**20, 1)
            except (psutil.Error, AttributeError):
                pass
        runner.write_json_atomic(paths.status, {
            "step": STEP_TOKENIZE, "platform": platform, "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(), "workers_target": target, "workers_running": len(workers), "paused": paused,
            "waiting_for_disk": not disk_ok, "note": note,
            "running_shards": [f"{w['spec']['partition']}/{w['spec']['shard_index']:05d}" for w in workers.values() if w["spec"]],
            "partitions": per_part, "source_rows_this_run": rows_this_run, "elapsed_s": round(elapsed, 1),
            "source_rows_per_s": round(rate, 1), "eta_hours": round(remaining / rate / 3600, 2) if rate > 0 else None,
            "worker_rss_mb": rss, "worker_peak_rss_mb": {str(w["proc"].pid): w["peak"] for w in workers.values()},
            "load_average": list(os.getloadavg()), "free_disk_gb": round(free_gb(paths.tokens), 1),
            "mem_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
            "swap_used_gb": round(psutil.swap_memory().used / 1e9, 2), "start_method": args.start_method,
            "min_free_disk_gb": E.MIN_FREE_DISK_GB, "errors": errors})

    try:
        while pending or any(w["spec"] for w in workers.values()):
            # control file: a change after the start always applies
            ctl, mt = read_control(paths.control)
            if ctl is not None and (mt != ctl_mtime0 or not args.workers):
                if "workers" in ctl:
                    try:
                        target = max(0, min(int(ctl["workers"]), os.cpu_count() or 1))
                    except (TypeError, ValueError):
                        pass
                paused = bool(ctl.get("pause", False))
            if time.time() - last_disk >= (DISK_RECHECK_SECONDS if not disk_ok else 0):
                disk_ok = free_gb(paths.tokens) >= E.MIN_FREE_DISK_GB
                last_disk = time.time()
            # messages from workers
            for wid, w in list(workers.items()):
                try:
                    w["peak"] = max(w["peak"], round(psutil.Process(w["proc"].pid).memory_info().rss / 2**20, 1))
                except (psutil.Error, AttributeError):
                    pass
                while w["conn"].poll():
                    msg = w["conn"].recv()
                    if msg[0] == "ready":
                        w["ready"] = True
                    elif msg[0] == "done":
                        spec, res = msg[1], msg[2]
                        pp = per_part[spec["partition"]]
                        pp["shards_committed"] += 1
                        pp["rows_committed"] += res["rows_in"]
                        pp["shards_done"] += 1
                        pp["rows_done"] += res["rows_in"]
                        rows_this_run += res["rows_in"]
                        w["spec"] = None
                        log(f"{spec['partition']}/{spec['shard_index']:05d}: {res['rows_in']} rows -> "
                            f"{res['rows_written']} texts in {res['seconds']:.0f} s")
                        write_status()
                        last_status = time.time()
                    else:
                        where = f"{msg[1]['partition']}/{msg[1]['shard_index']:05d}" if msg[1] else "start-up"
                        errors.append(f"{where}: {msg[2].strip().splitlines()[-1]}")
                        log(f"worker error at {where}:\n{msg[2]}")
                        w["spec"] = None
                        pending.clear()       # fail fast: finish the running shards, start no new one
                if not w["proc"].is_alive() and not w["conn"].poll():
                    if w["spec"]:
                        errors.append(f"{w['spec']['partition']}/{w['spec']['shard_index']:05d}: worker died "
                                      f"(exit code {w['proc'].exitcode})")
                        pending.clear()
                    del workers[wid]
            # retire idle workers above the target, or all idle workers while paused / out of work
            idle = [wid for wid, w in workers.items() if w["ready"] and not w["spec"]]
            limit = 0 if (paused or not disk_ok or not pending) else target
            busy = sum(1 for w in workers.values() if w["spec"] or not w["ready"])
            while idle and busy + len(idle) > limit:
                wid = idle.pop()
                workers[wid]["conn"].send(None)
                workers[wid]["proc"].join(timeout=30)
                del workers[wid]
            # start workers and hand out shards
            if pending and not paused and disk_ok:
                for wid in [wid for wid, w in workers.items() if w["ready"] and not w["spec"]]:
                    if not pending:
                        break
                    spec, ctx = pending.pop(0)
                    workers[wid]["spec"] = spec
                    workers[wid]["conn"].send((spec, ctx))
                # start a worker only while the target allows it and a shard is waiting for it
                while len(workers) < target and sum(1 for w in workers.values() if not w["spec"]) < len(pending):
                    parent_conn, child_conn = mctx.Pipe()
                    proc = mctx.Process(target=worker_main, args=(child_conn, str(tok_dir), not args.no_emoji_memo), daemon=True)
                    proc.start()
                    workers[next_wid] = {"proc": proc, "conn": parent_conn, "spec": None, "ready": False, "peak": 0.0}
                    next_wid += 1
            if time.time() - last_status >= 60:
                write_status("paused" if paused else ("waiting for disk" if not disk_ok else ""))
                last_status = time.time()
            time.sleep(POLL_SECONDS)
    finally:
        for w in workers.values():
            try:
                w["conn"].send(None)
            except (OSError, BrokenPipeError):
                pass
        for w in workers.values():
            w["proc"].join(timeout=30)
            if w["proc"].is_alive():
                w["proc"].terminate()
        workers.clear()
        write_status("finished" if not errors else "stopped on error")
        for part, _ in sources:
            t = paths.tokens / part / "_tmp"
            if t.is_dir() and not any(t.iterdir()):
                t.rmdir()
        lock_fh.close()
    if errors:
        raise CheckError(f"{len(errors)} shard(s) failed: {errors[:3]}")
    log(f"done: {rows_this_run} source rows in {time.time() - started:.0f} s")
    return 0


def main_tokenize(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="06_emotion_emtract.py tokenize",
                                 description="EmTract stage 1: cleaning and tokens (section 1 of the header)")
    ap.add_argument("platform", choices=config.PLATFORMS)
    ap.add_argument("--partitions", nargs="*", help="Reddit coin labels, StockTwits months YYYY-MM, or 'units'")
    ap.add_argument("--limit-rows", type=int, help="only the first N source rows of each partition (pilots)")
    ap.add_argument("--out-root", help="write tokens, status, control and lock under this folder instead")
    ap.add_argument("--workers", type=int, help=f"worker processes (default {E.DEFAULT_WORKERS}; control file can change it)")
    ap.add_argument("--input-file", help="a prepared parquet with row_id (or the native id) and the text columns")
    ap.add_argument("--partition-name", help="partition name for --input-file (default: the file stem)")
    ap.add_argument("--shard-rows", type=int, help="split row groups into shards of at most N rows (pilots)")
    ap.add_argument("--start-method", choices=("spawn", "fork"), default="spawn",
                    help="how workers start: spawn loads the EmTract stack in every worker; fork loads it once "
                         "in the parent and shares it copy-on-write")
    ap.add_argument("--no-emoji-memo", action="store_true",
                    help="switch off the exact memo of advertools' per-character emoji lookup (slower, same output)")
    ap.add_argument("--plan", action="store_true", help="print the shard plan and exit")
    args = ap.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be at least 1")
    return run_tokenize(args)


# ==================================================================================================
# 2. Stage 2 (`pack`, `score`, `collect`): the stocktwits model alone on the stored token ids
# ==================================================================================================
# EmTract (Vamossy & Skog; MIT licence; upstream commit in emtract_method.EMTRACT_UPSTREAM_COMMIT)
# is run as its separate parts (MASTERPLAN_NOV15.md section 7, decisions of 29 Sep 2026, (5)).
# Stage 1 (`tokenize`, CPU, env emtract-ref) has already cleaned every text once with EmTract's own
# `clean_text`, turned the cleaned text into wordpiece ids with tokenizers 0.10.3 (uncut, no special
# tokens) and planned the 62-wordpiece windows of long texts. Stage 2 never sees a text: it moves
# token ids to the model and scores back. Three subcommands, run one after another:
#
#   pack <platform> --batch-id ID --max-inputs N            (CPU; env thesis or emtract-ref)
#       1. Lists the finished stage-1 shards, tokens_dir(platform)/<partition>/shard-NNNNN.parquet.
#          A shard is finished when its stats file exists, every check in it passed and the shard's
#          bytes and md5 equal the ones recorded there (pipeline/runner.py standard).
#       2. Skips every shard already named in the map stats of an earlier batch of this platform,
#          then takes whole shards, in partition and shard order, while the batch stays within N
#          model inputs. A shard is never split across batches.
#       3. Builds the model inputs of every row (shared interface, items 2-3):
#              primary   [CLS] + input_ids[:62] + [SEP]           exactly what EmTract's predict()
#                                                                  gives the model (truncation to 64)
#              window j  [CLS] + input_ids[s_j : s_j + 62] + [SEP] for the starts s_j planned by
#                                                                  stage 1; window 0 is the primary
#          Rows without a window (missing, empty or failed text) get no input. The ids of [CLS]
#          and [SEP] are read from the tokenizer files (vocab.txt, added_tokens.json and
#          special_tokens_map.json of the twitter folder, which EmTract's model.py loads for both
#          variants); they are never hard-coded.
#       4. Writes PACKED_INPUTS_DIR/<platform>/<ID>.parquet (_emtract_run/modal_inputs/) with only `seq`
#          (0..n-1) and `input_ids` (list<uint16>, with [CLS]/[SEP]). This is the only file that goes to
#          the GPU machine: it holds no post id, account id or text. The local map <ID>.map.parquet holds
#          seq, shard (path relative to tokens_dir(platform)), shard_row (int32), window (int16).
#       One writer: a lock file in the platform folder refuses a second pack running at the same time.
#
#   score --inputs FILE --out FILE --device cuda|mps|cpu     (a GPU machine; CPU or Apple MPS also work)
#       The model alone: DistilBertForSequenceClassification from the stocktwits files, eval(),
#       torch.no_grad(), float32 weights and activations, and on CUDA TF32 off
#       (torch.backends.cuda.matmul.allow_tf32 = False, torch.backends.cudnn.allow_tf32 = False).
#       EmTract's predict() runs in fp32; a published study finds fp32 reproducible across GPU types
#       and batch sizes, fp16/bf16 not (Yuan et al. 2025, arXiv 2506.09501). Inputs are sorted by
#       length (then seq) and padded to the smallest of PAD_BUCKETS = 16/32/48/64 that fits, with
#       attention mask 1 on real tokens and 0 on padding. Padding is masked, so bucketing changes a
#       result only at rounding level (measured locally: at most 2.3e-6,
#       emtract_reference_checks_20260929.md, check (f)). The softmax is taken on the CPU in float32
#       from the float32 logits, so the device changes only the logits. Writes seq, logit_<label>,
#       p_<label> (LABELS order, float32), ordered by seq, through a temporary file, and
#       <out>.stats.json with device, versions, timings and checks. Needs only torch, transformers,
#       numpy, pyarrow and this package's emtract_method.py / config.py (standard library only at
#       import), so the same command runs on a rented GPU and on a laptop.
#
#   collect <platform> --batch-id ID [--raw FILE]                                      (env thesis)
#       Joins the raw scores (default RAW_SCORES_DIR/<platform>/<ID>.parquet, i.e.
#       _emtract_run/modal_outputs/, copied back from the GPU machine with its .stats.json) with the local map and writes
#       scores_dir(platform)/<ID>.parquet with seq, shard, shard_row, window, logit_<label>,
#       p_<label> (interface item 5), through temp file -> checks -> commit -> stats (runner.py).
#
# Checks (each one asserted; a failure raises CheckError and nothing is committed)
#   pack     shard stats present, passed, bytes and md5 equal; per row len(input_ids) == n_tokens;
#            a row has windows iff n_tokens > 0; flagged rows (missing, empty, clean error) have no
#            tokens and every row without tokens is flagged; n <= 62 -> starts [0]; n > 62 ->
#            k = ceil((n - 16) / 46) starts, first 0, last n - 62, increasing, every overlap >= 16;
#            ids below the vocabulary size (so they fit uint16); the written files re-read: seq is
#            0..n-1, each input starts with [CLS], ends with [SEP] and has 3..64 ids, only primaries
#            are shorter than 64 ids, the map has one row per input; every 997th input and every
#            non-primary window rebuilt in plain Python equals the vectorised build.
#   score    seq is a permutation of 0..n-1; lengths 3..64; [CLS] first and [SEP] last; weights file
#            sha256 equals SHA256SUMS (when present); every parameter float32; TF32 off on CUDA;
#            logits and probabilities finite; every row of probabilities sums to 1 within 1e-5;
#            argmax of logits == argmax of probabilities; written file has one row per input.
#   collect  the score stats name the sha256 of this batch's inputs file (unchanged since pack) and
#            its input count; every score check passed; one score per input (seq equal to the
#            map's); probabilities sum to 1 within 1e-5; each shard still has the md5 recorded at
#            pack time; per shard row, the number of inputs equals its number of windows.
#
# Outputs
#   PACKED_INPUTS_DIR/<platform>/<ID>.parquet (+ .stats.json), <ID>.map.parquet (+ .stats.json)
#   scores_dir(platform)/<ID>.parquet (+ .stats.json)
#   `score` writes wherever --out points (<out> and <out>.stats.json).
#   Every raw scores file also carries its score stats inside the Parquet footer (key
#   emtract_score_stats), so a file that travels alone still says how it was made.
#
# Usage
# -----
#     python pipeline/06_emotion_emtract.py pack reddit --batch-id reddit-0001 --max-inputs 2000000
#     python pipeline/06_emotion_emtract.py score --inputs IN.parquet --out OUT.parquet --device cuda \
#         --model-dir data/raw/emtract_models/stocktwits
#     python pipeline/06_emotion_emtract.py collect reddit --batch-id reddit-0001
SPECIAL_KEYS = ("cls_token", "sep_token", "pad_token")
DEFAULT_BATCH_TOKENS = 65536          # 1,024 inputs x 64 tokens = EmTract's eval batch (model.py)
SAMPLE_EVERY = 997                    # pack: every 997th input is rebuilt in plain Python
EMBEDDED_STATS_KEY = b"emtract_score_stats"      # the score stats inside the raw scores' Parquet footer
PACK_RULES_SHA = f"emtract-{E.EMTRACT_UPSTREAM_COMMIT[:12]}-tok{E.TOKENIZERS_VERSION}"


def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats_path(p: Path) -> Path:
    return p.with_name(p.name + ".stats.json")


# --- Tokenizer files: special-token ids (never hard-coded) -----------------------------------------

def special_token_ids(tokenizer_dir) -> Dict[str, int]:
    """Ids of [CLS], [SEP] and [PAD] from special_tokens_map.json, vocab.txt and added_tokens.json."""
    d = Path(tokenizer_dir)
    smap = json.loads((d / "special_tokens_map.json").read_text(encoding="utf-8"))
    vocab: Dict[str, int] = {}
    with open(d / "vocab.txt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            vocab.setdefault(line.rstrip("\n"), i)
    added = json.loads((d / "added_tokens.json").read_text(encoding="utf-8")) if (d / "added_tokens.json").exists() else {}
    out: Dict[str, int] = {}
    for key in SPECIAL_KEYS:
        tok = smap[key]["content"] if isinstance(smap[key], dict) else smap[key]
        tid = added.get(tok, vocab.get(tok))
        if tid is None:
            raise CheckError(f"special token {tok!r} not in the tokenizer files")
        out[key.replace("_token", "")] = int(tid)
    out["vocab_size"] = max(list(vocab.values()) + list(added.values())) + 1
    return out


# --- Window plan checks (interface item 2); the plan itself is stage 1's ---------------------------

def expected_n_windows(n: int) -> int:
    """0 for an empty text, 1 up to 62 wordpieces, else k = ceil((n - 16) / 46)."""
    W, m = E.MAX_TEXT_TOKENS, E.MIN_WINDOW_OVERLAP
    if n <= 0:
        return 0
    if n <= W:
        return 1
    return math.ceil((n - m) / (W - m))


def check_window_starts(n: int, starts: Sequence[int]) -> Optional[str]:
    """None if `starts` is a valid plan for a text of n wordpieces, else the reason."""
    W, m = E.MAX_TEXT_TOKENS, E.MIN_WINDOW_OVERLAP
    k = expected_n_windows(n)
    if len(starts) != k:
        return f"{len(starts)} windows, expected {k}"
    if k == 0:
        return None
    if starts[0] != 0:
        return "first start is not 0"
    if k == 1:
        return None
    if starts[-1] != n - W:
        return f"last start {starts[-1]} != n - {W} = {n - W}"
    for a, b in zip(starts[:-1], starts[1:]):
        if not a < b:
            return "starts not increasing"
        if W - (b - a) < m:
            return f"overlap {W - (b - a)} < {m}"
    return None


# --- Building model inputs (vectorised) and a plain-Python reference for the checks ----------------

def _list_arrays(col) -> Tuple[np.ndarray, np.ndarray]:
    """(values, offsets) of a list column; a null list counts as empty."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if isinstance(arr, pa.ChunkedArray):
        arr = pa.concat_arrays(arr.chunks)
    offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    values = arr.values.to_numpy(zero_copy_only=False)
    base = int(offsets[0]) if len(offsets) else 0
    return values[base:int(offsets[-1])] if len(offsets) else values[:0], offsets - base


def build_inputs(values: np.ndarray, offsets: np.ndarray, rows: np.ndarray, starts: np.ndarray,
                 cls_id: int, sep_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Model inputs for (row, start) pairs, vectorised.

    values/offsets: the flattened uncut input_ids of a shard (offsets: n_rows + 1 entries);
    rows[i], starts[i]: shard row and window start of input i.
    Returns (flat uint16 ids with [CLS]/[SEP], int64 offsets of the inputs, n_inputs + 1 entries).
    """
    n_tok = (offsets[1:] - offsets[:-1])[rows]
    take = np.minimum(n_tok - starts, E.MAX_TEXT_TOKENS).astype(np.int64)
    if len(take) and take.min() < 1:
        raise CheckError("an input would hold no text token")
    out_off = np.zeros(len(take) + 1, dtype=np.int64)
    np.cumsum(take + 2, out=out_off[1:])
    out = np.empty(int(out_off[-1]), dtype=np.uint16)
    out[out_off[:-1]] = cls_id
    out[out_off[1:] - 1] = sep_id
    total = int(take.sum())
    rel = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(take) - take, take)
    src = np.repeat(offsets[rows].astype(np.int64) + starts, take) + rel
    dst = np.repeat(out_off[:-1] + 1, take) + rel
    out[dst] = values[src]
    return out, out_off


def reference_input(ids: Sequence[int], start: int, cls_id: int, sep_id: int) -> List[int]:
    """Plain-Python statement of interface item 3, used by the checks."""
    return [cls_id] + list(ids[start:start + E.MAX_TEXT_TOKENS]) + [sep_id]


def window_numbers(rows: np.ndarray) -> np.ndarray:
    """0, 1, 2, ... within each run of equal (sorted) row numbers."""
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.concatenate([[0], np.flatnonzero(rows[1:] != rows[:-1]) + 1])
    sizes = np.diff(np.append(first, len(rows)))
    return np.arange(len(rows)) - np.repeat(first, sizes)


# --- pack ------------------------------------------------------------------------------------------
SHARD_COLUMNS = ["n_tokens", "input_ids", "window_starts", "is_missing_text", "is_empty_after_clean",
                 "is_clean_error"]


def finished_shards(tokens_root: Path) -> List[Path]:
    """Shards whose stats file exists, passed every check and names the shard's current bytes and md5."""
    from pipeline import runner
    out = []
    for shard in sorted(tokens_root.glob("*/shard-*.parquet")):
        sp = _stats_path(shard)
        if not sp.exists():
            continue
        try:
            rec = json.loads(sp.read_text())
        except ValueError:
            continue
        if not rec.get("checks") or not all(c.get("passed") for c in rec["checks"]):
            continue
        outs = [o for o in rec.get("outputs", []) if Path(o["path"]).name == shard.name]
        if len(outs) != 1 or outs[0].get("bytes") != shard.stat().st_size:
            continue
        if outs[0].get("md5") != runner.file_md5(shard):
            continue
        out.append(shard)
    return out


def packed_shards(inputs_root: Path) -> Dict[str, str]:
    """Shards (path relative to tokens_dir -> batch id) already in an earlier batch of this platform."""
    done: Dict[str, str] = {}
    for mp in sorted(inputs_root.glob("*.map.parquet")):
        sp = _stats_path(mp)
        if not sp.exists():
            raise CheckError(f"map {mp.name} has no stats file: an earlier pack did not finish")
        rec = json.loads(sp.read_text())
        for s in rec.get("shards", []):
            done[s["shard"]] = rec.get("batch_id", mp.name)
    return done


def validate_shard(t: pa.Table, rel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-level checks of a stage-1 shard; returns (values, offsets, input rows, input starts)."""
    n_tokens = _np(t.column("n_tokens")).astype(np.int64)
    values, offsets = _list_arrays(t.column("input_ids"))
    ws_values, ws_offsets = _list_arrays(t.column("window_starts"))
    flags = np.zeros(t.num_rows, dtype=bool)
    for c in ("is_missing_text", "is_empty_after_clean", "is_clean_error"):
        flags |= _np(t.column(c)).astype(bool)
    n_ids = offsets[1:] - offsets[:-1]
    n_ws = ws_offsets[1:] - ws_offsets[:-1]
    check(bool((n_ids == n_tokens).all()), f"{rel}: len(input_ids) == n_tokens on every row")
    check(bool(((n_ws > 0) == (n_tokens > 0)).all()), f"{rel}: a row has windows iff n_tokens > 0")
    check(not bool((flags & (n_tokens > 0)).any()), f"{rel}: flagged rows have no tokens")
    check(bool(((n_tokens > 0) | flags).all()), f"{rel}: every row without tokens carries a flag")
    W, m = E.MAX_TEXT_TOKENS, E.MIN_WINDOW_OVERLAP
    k_exp = np.where(n_tokens <= 0, 0, np.where(n_tokens <= W, 1, -(-(n_tokens - m) // (W - m))))
    check(bool((n_ws == k_exp).all()), f"{rel}: window count 0 / 1 / ceil((n - 16) / 46)")
    if (n_ws > 0).any():
        check(bool((ws_values[ws_offsets[:-1][n_ws > 0]] == 0).all()), f"{rel}: first window start is 0")
    for r in np.flatnonzero(n_ws > 1):                # long texts: a few per cent of rows
        reason = check_window_starts(int(n_tokens[r]), ws_values[ws_offsets[r]:ws_offsets[r + 1]].tolist())
        if reason:
            raise CheckError(f"{rel}: row {r}: window plan invalid ({reason})")
    rows = np.repeat(np.arange(t.num_rows, dtype=np.int64), n_ws)
    return values, offsets, rows, ws_values.astype(np.int64)


def pack(platform: str, batch_id: str, max_inputs: int, tokenizer_dir: Optional[Path] = None) -> Dict[str, object]:
    from pipeline import runner
    tokens_root = E.tokens_dir(platform)
    inputs_root = E.PACKED_INPUTS_DIR / platform
    inputs_root.mkdir(parents=True, exist_ok=True)
    final_inputs = inputs_root / f"{batch_id}.parquet"
    final_map = inputs_root / f"{batch_id}.map.parquet"
    check(not final_inputs.exists() and not final_map.exists(), f"batch {batch_id} does not exist yet")
    lock = inputs_root / ".pack.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise CheckError(f"another pack holds {lock.name}; one writer per platform")
    try:
        os.write(fd, f"{os.getpid()} {batch_id}\n".encode())
        os.close(fd)
        sp = special_token_ids(tokenizer_dir or (E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT))
        cls_id, sep_id = sp["cls"], sp["sep"]
        done = packed_shards(inputs_root)
        todo = [s for s in finished_shards(tokens_root) if str(s.relative_to(tokens_root)) not in done]
        check(len(todo) > 0, f"{platform}: at least one finished, unpacked shard")
        chosen, n_total = [], 0
        for s in todo:
            ws = pq.read_table(s, columns=["window_starts"]).column("window_starts")
            n_in = int(pc.sum(pc.list_value_length(ws)).as_py() or 0)
            if n_in > max_inputs:
                if chosen:
                    break
                raise CheckError(f"shard {s.relative_to(tokens_root)} alone holds {n_in} inputs > --max-inputs")
            if n_total + n_in > max_inputs:
                break
            chosen.append((s, n_in))
            n_total += n_in
        checks: List[Dict[str, object]] = []
        id_parts, off_parts, map_parts, shard_recs = [], [], [], []
        seq0, flat0 = 0, 0
        for s, n_in in chosen:
            rel = str(s.relative_to(tokens_root))
            t = pq.read_table(s, columns=SHARD_COLUMNS)
            values, offsets, rows, starts = validate_shard(t, rel)
            check(len(rows) == n_in, f"{rel}: inputs == sum of window counts")
            if len(values):
                check(int(values.max()) < sp["vocab_size"], f"{rel}: ids below the vocabulary size")
            ids, off = build_inputs(values, offsets, rows, starts, cls_id, sep_id)
            wno = window_numbers(rows)
            probe = sorted(set(range(0, len(rows), SAMPLE_EVERY)) | set(np.flatnonzero(wno > 0).tolist()))
            for i in probe:
                r = int(rows[i])
                ref = reference_input(values[offsets[r]:offsets[r + 1]].tolist(), int(starts[i]), cls_id, sep_id)
                if ids[off[i]:off[i + 1]].tolist() != ref:
                    raise CheckError(f"{rel}: input {i} differs from the plain-Python build")
            id_parts.append(ids)
            off_parts.append(off[:-1] + flat0)
            flat0 += len(ids)
            map_parts.append(pa.table({
                "seq": pa.array(np.arange(seq0, seq0 + len(rows), dtype=np.int64)),
                "shard": pa.array([rel] * len(rows), pa.string()),
                "shard_row": pa.array(rows.astype(np.int32)),
                "window": pa.array(wno.astype(np.int16)),
            }))
            shard_recs.append({"shard": rel, "bytes": s.stat().st_size, "md5": runner.file_md5(s),
                               "rows": t.num_rows, "inputs": int(len(rows)),
                               "primary": int((wno == 0).sum()), "extra_windows": int((wno > 0).sum()),
                               "probed": len(probe)})
            seq0 += len(rows)
        flat = np.concatenate(id_parts) if id_parts else np.empty(0, np.uint16)
        offs = np.append(np.concatenate(off_parts) if off_parts else np.empty(0, np.int64), flat0)
        check(len(offs) - 1 == n_total, "inputs == planned total", checks)
        check(int(offs[-1]) <= np.iinfo(np.int32).max, "list offsets fit int32", checks)
        inputs_tbl = pa.table({
            "seq": pa.array(np.arange(n_total, dtype=np.int64)),
            "input_ids": pa.ListArray.from_arrays(pa.array(offs.astype(np.int32)), pa.array(flat, pa.uint16())),
        })
        tmp_in, tmp_map = runner.tmp_path(final_inputs), runner.tmp_path(final_map)
        pq.write_table(inputs_tbl, tmp_in, compression="zstd")
        pq.write_table(pa.concat_tables(map_parts), tmp_map, compression="zstd")
        # checks against the temporary files, re-read from disk
        rin, rmap = pq.read_table(tmp_in), pq.read_table(tmp_map)
        check(rin.column_names == ["seq", "input_ids"], "inputs file holds only seq and input_ids", checks)
        check(rin.schema.field("input_ids").type == pa.list_(pa.uint16()), "input_ids is list<uint16>", checks)
        seq = rin.column("seq").to_numpy()
        check(bool((seq == np.arange(n_total)).all()), "seq is 0..n-1 in order", checks)
        v, o = _list_arrays(rin.column("input_ids"))
        lens = np.diff(o)
        check(bool(((lens >= 3) & (lens <= E.MAX_LENGTH)).all()), "every input has 3..64 ids", checks)
        check(bool((v[o[:-1]] == cls_id).all()) and bool((v[o[1:] - 1] == sep_id).all()),
               "every input is [CLS] ... [SEP]", checks)
        check(rmap.num_rows == n_total and bool((rmap.column("seq").to_numpy() == seq).all()),
               "map has one row per input, same seq", checks)
        check(bool((rmap.column("window").to_numpy()[lens < E.MAX_LENGTH] == 0).all()),
               "only primaries are shorter than 64 ids", checks)
        check(sum(r["probed"] for r in shard_recs) > 0, "plain-Python rebuild probed at least one input", checks)
        runner.commit(tmp_in, final_inputs)
        runner.commit(tmp_map, final_map)
        n_primary = sum(r["primary"] for r in shard_recs)
        lower = (0,) + tuple(E.PAD_BUCKETS[:-1])
        counts = {"shards": len(shard_recs), "rows": sum(r["rows"] for r in shard_recs), "inputs": n_total,
                  "primary": n_primary, "extra_windows": n_total - n_primary,
                  "by_bucket": {str(b): int(((lens > lo) & (lens <= b)).sum()) for lo, b in zip(lower, E.PAD_BUCKETS)}}
        code_sha = code_fingerprint()
        runner.write_stats(final_map, step=STEP + ":pack", inputs=runner.describe_inputs([s for s, _ in chosen]),
                           code_sha=code_sha, rules_sha=PACK_RULES_SHA, con=None, counts=counts, checks=checks,
                           extra={"batch_id": batch_id, "platform": platform, "special_ids": sp,
                                  "shards": shard_recs, "inputs_sha256": sha256_file(final_inputs),
                                  "max_inputs": max_inputs})
        runner.write_stats(final_inputs, step=STEP + ":pack", inputs=[], code_sha=code_sha, rules_sha=PACK_RULES_SHA,
                           con=None, counts=counts, checks=checks,
                           extra={"batch_id": batch_id, "platform": platform, "special_ids": sp})
        log(f"pack {platform} {batch_id}: {len(shard_recs)} shards, {n_total:,} inputs "
             f"({n_total - n_primary:,} extra windows)")
    finally:
        if lock.exists():
            lock.unlink()
    out = {"inputs_file": str(final_inputs), "map_file": str(final_map), **counts}
    return out


# --- score (the same code on any machine: CUDA, Apple MPS or CPU) ----------------------------------

def bucket_of(lengths: np.ndarray) -> np.ndarray:
    """Smallest PAD_BUCKETS entry >= each length."""
    b = np.asarray(E.PAD_BUCKETS)
    idx = np.searchsorted(b, lengths, side="left")
    if len(lengths) and idx.max() >= len(b):
        raise CheckError("an input is longer than the largest pad bucket")
    return b[idx]


def plan_batches(lengths: np.ndarray, seq: np.ndarray, batch_tokens: int) -> List[Tuple[int, np.ndarray]]:
    """Inputs sorted by (length, seq), cut into batches of one bucket each: [(bucket, row indices)].
    A batch holds at most batch_tokens // bucket inputs."""
    order = np.lexsort((seq, lengths))
    buckets = bucket_of(lengths[order])
    out = []
    for b in E.PAD_BUCKETS:
        sel = order[buckets == b]
        bs = max(1, batch_tokens // b)
        for i in range(0, len(sel), bs):
            out.append((int(b), sel[i:i + bs]))
    return out


def pad_batch(values: np.ndarray, offsets: np.ndarray, idx: np.ndarray, bucket: int, pad_id: int):
    """(input_ids int64 [B, bucket], attention_mask int64 [B, bucket]) for the inputs idx."""
    lens = (offsets[idx + 1] - offsets[idx]).astype(np.int64)
    ids = np.full((len(idx), bucket), pad_id, dtype=np.int64)
    mask = np.zeros((len(idx), bucket), dtype=np.int64)
    col = np.arange(bucket)[None, :]
    m = col < lens[:, None]
    ids[m] = values[(offsets[idx][:, None] + col)[m]]
    mask[m] = 1
    return ids, mask


def softmax_f32(logits: np.ndarray) -> np.ndarray:
    import torch
    return torch.softmax(torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32)), dim=-1).numpy()


def load_model(model_dir: Path, device: str, attn: str = "default"):
    import torch
    import transformers
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    kw = {"num_labels": len(E.LABELS)}
    if attn != "default":
        kw["attn_implementation"] = attn
    model = transformers.DistilBertForSequenceClassification.from_pretrained(str(model_dir), **kw)
    return model.to(torch.float32).to(device).eval()


def _weights_file(model_dir: Path) -> Optional[Path]:
    for name in ("pytorch_model.bin", "model.safetensors"):
        if (model_dir / name).exists():
            return model_dir / name
    return None


def score(inputs: Path, out: Path, device: str, model_dir: Optional[Path] = None,
          batch_tokens: int = DEFAULT_BATCH_TOKENS, attn: str = "default", threads: Optional[int] = None) -> Dict[str, object]:
    import torch
    import transformers
    t_start = time.time()
    checks: List[Dict[str, object]] = []
    model_dir = Path(model_dir or (E.EMTRACT_MODEL_DIR / E.MODEL_VARIANT))
    if threads:
        torch.set_num_threads(int(threads))
    sp = special_token_ids(model_dir)
    tbl = pq.read_table(inputs)
    check(tbl.column_names == ["seq", "input_ids"], "inputs hold only seq and input_ids", checks)
    seq = tbl.column("seq").to_numpy().astype(np.int64)
    n = len(seq)
    check(bool((np.sort(seq) == np.arange(n)).all()), "seq is a permutation of 0..n-1", checks)
    values, offsets = _list_arrays(tbl.column("input_ids"))
    values = values.astype(np.int64)
    lengths = np.diff(offsets)
    if n:
        check(bool(((lengths >= 3) & (lengths <= E.MAX_LENGTH)).all()), "lengths 3..64", checks)
        check(bool((values[offsets[:-1]] == sp["cls"]).all() and (values[offsets[1:] - 1] == sp["sep"]).all()),
               "every input is [CLS] ... [SEP]", checks)
    wfile = _weights_file(model_dir)
    weights_sha = sha256_file(wfile) if wfile else None
    sums = model_dir / "SHA256SUMS"
    if sums.exists() and wfile is not None:
        want = {ln.split()[1].lstrip("*"): ln.split()[0] for ln in sums.read_text().splitlines() if ln.strip()}
        check(want.get(wfile.name) == weights_sha, "weights sha256 equals SHA256SUMS", checks)
    t0 = time.time()
    model = load_model(model_dir, device, attn)
    if device == "cuda":
        torch.cuda.synchronize()
    load_s = time.time() - t0
    check(all(p.dtype == torch.float32 for p in model.parameters()), "every parameter is float32", checks)
    if device == "cuda":
        check(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False,
               "TF32 off", checks)
    logits = np.empty((n, len(E.LABELS)), dtype=np.float32)
    batches = plan_batches(lengths, seq, batch_tokens)
    t1 = time.time()
    with torch.no_grad():
        for bucket, idx in batches:
            ids, mask = pad_batch(values, offsets, idx, bucket, sp["pad"])
            lo = model(input_ids=torch.from_numpy(ids).to(device),
                       attention_mask=torch.from_numpy(mask).to(device)).logits
            logits[seq[idx]] = lo.float().cpu().numpy()
    if device == "cuda":
        torch.cuda.synchronize()
    score_s = time.time() - t1
    probs = softmax_f32(logits)
    check(bool(np.isfinite(logits).all() and np.isfinite(probs).all()), "logits and probabilities finite", checks)
    check(bool((np.abs(probs.astype(np.float64).sum(axis=1) - 1) <= PROB_SUM_TOL).all()),
           "probabilities sum to 1 within 1e-5", checks)
    check(bool((logits.argmax(1) == probs.argmax(1)).all()), "argmax logits == argmax probabilities", checks)
    cols = {"seq": pa.array(np.arange(n, dtype=np.int64))}
    for j, lab in enumerate(E.LABELS):
        cols[f"logit_{lab}"] = pa.array(logits[:, j])
    for j, lab in enumerate(E.LABELS):
        cols[f"p_{lab}"] = pa.array(probs[:, j])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    rec = {
        "step": STEP + ":score", "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "host_machine": _platform.machine(), "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "pyarrow": pa.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "dtype": "float32", "batch_tokens": batch_tokens, "pad_buckets": list(E.PAD_BUCKETS),
        "torch_threads": torch.get_num_threads(), "inputs_file": Path(inputs).name,
        "inputs_sha256": sha256_file(inputs), "weights_file": wfile.name if wfile else None,
        "weights_sha256": weights_sha, "special_ids": sp, "labels": list(E.LABELS),
        "counts": {"inputs": n, "batches": len(batches),
                   "by_bucket": {str(b): int(sum(len(i) for bb, i in batches if bb == b)) for b in E.PAD_BUCKETS}},
        "timing_s": {"load": round(load_s, 3), "score": round(score_s, 3), "total": round(time.time() - t_start, 3)},
        "inputs_per_s_scoring": round(n / score_s, 1) if score_s > 0 else None,
        "checks": checks,
    }
    # the footer copy holds every check so far; the read-back check below is in the .stats.json
    # file and is repeated by collect ("one score per input")
    table = pa.table(cols).replace_schema_metadata({EMBEDDED_STATS_KEY: json.dumps(rec).encode()})
    pq.write_table(table, tmp, compression="zstd")
    check(pq.read_metadata(tmp).num_rows == n, "written scores: one row per input", checks)
    os.replace(tmp, out)
    Path(str(out) + ".stats.json").write_text(json.dumps(rec, indent=1))
    log(f"score {Path(inputs).name}: {n:,} inputs on {rec['device_name']} in {score_s:.1f} s "
         f"({rec['inputs_per_s_scoring']} inputs/s), load {load_s:.1f} s")
    return rec


# --- collect ---------------------------------------------------------------------------------------

def read_raw_stats(raw: Path) -> Tuple[Dict[str, object], str]:
    """The score stats of a raw scores file: its .stats.json file if present, else the copy in its
    Parquet footer (a raw scores file can travel without its stats file)."""
    side = Path(str(raw) + ".stats.json")
    if side.exists():
        return json.loads(side.read_text()), "stats_file"
    md = pq.read_schema(raw).metadata or {}
    if EMBEDDED_STATS_KEY not in md:
        raise CheckError(f"{raw.name}: no stats file and no score stats in the Parquet footer")
    return json.loads(md[EMBEDDED_STATS_KEY].decode()), "parquet_footer"


def collect(platform: str, batch_id: str, raw: Optional[Path] = None) -> Dict[str, object]:
    from pipeline import runner
    inputs_root = E.PACKED_INPUTS_DIR / platform
    final_inputs = inputs_root / f"{batch_id}.parquet"
    final_map = inputs_root / f"{batch_id}.map.parquet"
    raw = Path(raw or (E.RAW_SCORES_DIR / platform / f"{batch_id}.parquet"))
    final = E.scores_dir(platform) / f"{batch_id}.parquet"
    check(not final.exists(), f"{final.name} not collected yet (one writer; nothing is overwritten)")
    checks: List[Dict[str, object]] = []
    map_stats = json.loads(_stats_path(final_map).read_text())
    raw_stats, raw_stats_source = read_raw_stats(raw)
    check(raw_stats["inputs_sha256"] == sha256_file(final_inputs), "scores were computed on this batch's inputs", checks)
    check(raw_stats["inputs_sha256"] == map_stats["inputs_sha256"], "inputs file unchanged since pack", checks)
    check(bool(raw_stats.get("checks")) and all(c.get("passed") for c in raw_stats["checks"]),
           "every score check passed", checks)
    m = pq.read_table(final_map)
    s = pq.read_table(raw)
    n = m.num_rows
    check(raw_stats["counts"]["inputs"] == n and s.num_rows == n, "one score per input", checks)
    sseq = s.column("seq").to_numpy()
    check(bool((sseq == np.arange(n)).all()) and bool((m.column("seq").to_numpy() == sseq).all()),
           "score seq equals map seq (0..n-1)", checks)
    P = np.column_stack([s.column(f"p_{lab}").to_numpy() for lab in E.LABELS]).astype(np.float64)
    L = np.column_stack([s.column(f"logit_{lab}").to_numpy() for lab in E.LABELS])
    check(bool(np.isfinite(P).all() and np.isfinite(L).all()), "scores finite", checks)
    check(bool((np.abs(P.sum(1) - 1) <= PROB_SUM_TOL).all()), "probabilities sum to 1 within 1e-5", checks)
    tokens_root = E.tokens_dir(platform)
    shard_col = _np(m.column("shard"))
    row_col = m.column("shard_row").to_numpy().astype(np.int64)
    for rec in map_stats["shards"]:
        p = tokens_root / rec["shard"]
        check(runner.file_md5(p) == rec["md5"], f"{rec['shard']} unchanged since pack", checks)
        ws = pq.read_table(p, columns=["window_starts"]).column("window_starts")
        k = np.diff(_list_arrays(ws)[1])
        got = np.bincount(row_col[shard_col == rec["shard"]], minlength=len(k))
        check(len(got) == len(k) and bool((got == k).all()), f"{rec['shard']}: inputs per row == windows", checks)
    cols = {c: m.column(c) for c in ["seq", "shard", "shard_row", "window"]}
    for lab in E.LABELS:
        cols[f"logit_{lab}"] = s.column(f"logit_{lab}").cast(pa.float32())
    for lab in E.LABELS:
        cols[f"p_{lab}"] = s.column(f"p_{lab}").cast(pa.float32())
    tmp = runner.tmp_path(final)
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    check(pq.read_metadata(tmp).num_rows == n, "written table: one row per input", checks)
    runner.commit(tmp, final)
    counts = {"inputs": n, "primary": int((m.column("window").to_numpy() == 0).sum()),
              "shards": len(map_stats["shards"])}
    keep = ("device", "device_name", "python", "torch", "transformers", "cuda", "attn_implementation",
            "tf32_matmul", "tf32_cudnn", "dtype", "batch_tokens", "pad_buckets", "weights_sha256",
            "timing_s", "inputs_per_s_scoring", "written_at_utc")
    runner.write_stats(final, step=STEP + ":collect", inputs=runner.describe_inputs([final_map, final_inputs, raw]),
                       code_sha=code_fingerprint(), rules_sha=PACK_RULES_SHA,
                       con=None, counts=counts, checks=checks,
                       extra={"batch_id": batch_id, "platform": platform, "raw_stats_source": raw_stats_source,
                              "score": {k: raw_stats.get(k) for k in keep}})
    log(f"collect {platform} {batch_id}: {n:,} scores -> {final.name}")
    return counts


def main_stage2(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="06_emotion_emtract.py", description="EmTract stage 2: pack, score, collect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack", help="build model inputs from finished stage-1 shards")
    a.add_argument("platform")
    a.add_argument("--batch-id", required=True)
    a.add_argument("--max-inputs", type=int, required=True)
    a.add_argument("--tokenizer-dir", type=Path, default=None)
    b = sub.add_parser("score", help="run the model alone on an inputs file")
    b.add_argument("--inputs", type=Path, required=True, help="packed inputs file (pack)")
    b.add_argument("--out", type=Path, required=True, help="raw scores file to write")
    b.add_argument("--device", choices=["cuda", "mps", "cpu"], required=True)
    b.add_argument("--model-dir", type=Path, default=None)
    b.add_argument("--batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS)
    b.add_argument("--attn", choices=["default", "eager", "sdpa"], default="default")
    b.add_argument("--threads", type=int, default=None)
    c = sub.add_parser("collect", help="join raw scores with the map and write the scores table")
    c.add_argument("platform")
    c.add_argument("--batch-id", required=True)
    c.add_argument("--raw", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "pack":
        pack(args.platform, args.batch_id, args.max_inputs, args.tokenizer_dir)
    elif args.cmd == "score":
        score(args.inputs, args.out, args.device, args.model_dir, args.batch_tokens, args.attn, args.threads)
    else:
        collect(args.platform, args.batch_id, args.raw)
    return 0


# ==================================================================================================
# 3. Merge (`merge`): one EmTract score row per post and text version
# ==================================================================================================
# Input    stage 1 (`tokenize`): the token shards <platform>/emtract/tokens/<partition>/shard-NNNNN.parquet,
#          each with its stats file; stage 2 (`collect`): the score files
#          <platform>/emtract/scores/<batch_id>.parquet, each with its stats file, one row per model input
#          (seq, shard, shard_row, window, logit_<label>, p_<label>); the fixed score of an empty text
#          (_emtract_run/empty_text_score.json, written by `merge --empty-score`).
# Output   <platform>/emtract/emotion/<partition>.parquet with a stats file: one row per (row_id, version)
#          of the token shards, in their order (source-row order; versions own, quoted, both).
#
# Columns of the output
# ---------------------
# row_id, version, n_tokens, is_missing_text, is_empty_after_clean, is_clean_error   copied from stage 1
# n_windows      int16: 0 for an empty or missing text, 1 for a text of 1-62 wordpieces, k for a longer text
# p_<label>      the primary score: the model's 7 probabilities for [CLS] + the first 62 wordpieces + [SEP],
#                exactly what EmTract's predict() gives the model (decisions of 29 Sep 2026, (4)-(5))
# pc_<label>     the chunk score: the equal-weight mean of the k windows' probability vectors;
#                null if n_windows < 2 (a short text has no windows beyond the primary input)
# pw_<label>     the coverage-weighted mean (sensitivity check): token t is covered by c_t windows,
#                window j gets the weight w_j = sum over its tokens of 1/c_t, and pw = sum_j w_j p_j / sum_j w_j;
#                null if n_windows < 2
# Labels in EmTract's output order (emtract_method.LABELS). Empty or missing texts get null probabilities:
# they get no model call (decided 29 Sep 2026). EmTract's own predict() would give such a text one fixed
# score (that of [CLS][SEP]); that score is computed once, recorded in every stats file, and never copied
# into the rows.
#
# Aggregation rules and their sources
# -----------------------------------
# * The linear pool (mean of the softmax vectors) is the standard untrained aggregator of chunk scores
#   (Pappagari et al. 2019, "Average", Table 4), consistent with EmTract's own averaging of message
#   scores by firm-day (Vamossy & Skog, p. 3). Logit means and the maximum are not used
#   (long-text evidence, emtract_long_text_evidence.md, A4 and B8).
# * Equal weights are the primary chunk score; the coverage-weighted mean is the sensitivity check
#   (decided 29 Sep 2026: "averaged with equal weights; the coverage-weighted mean is a sensitivity
#   check"). The coverage weights have no published source; they are an own derivation recorded in
#   the evidence (A4: sum_j w_j = n, so pw is the mean over tokens of the average score of the
#   windows covering the token).
# * Window 0 is the primary input, so the primary score is not computed twice (evidence B3).
# * Means are computed in float64 from the stored float32 probabilities and stored as float32.
#
# Checks (every one raises CheckError and leaves the previous output untouched)
# -----------------------------------------------------------------------------
# Stage-1 side, per shard:
#   - the shard's stats file exists, its checks all passed, and its recorded size equals the file's;
#   - no nulls in row_id, version, the flags, n_tokens, input_ids, window_starts; versions are own,
#     quoted or both, in the order own [, quoted, both] per row_id;
#   - len(input_ids) = n_tokens; window_starts is [] exactly when n_tokens = 0; a text flagged missing
#     or empty has n_tokens = 0; every row with n_tokens = 0 is flagged missing or empty (no other
#     gaps); a clean error is also flagged empty (stage 1: an error text is what predict() would
#     have scored as "");
#   - the window plan: [0] for 1-62 wordpieces; for n > 62, k = ceil((n - 16)/46) starts, the first 0,
#     the last n - 62, increasing, every overlap >= 16, interior starts within 8 (+0.5 for rounding)
#     of their ideal position j(n - 62)/(k - 1) (interface item 2).
# Stage-2 side, per shard:
#   - every score file has a stats file whose checks all passed;
#   - the score rows of the shard are exactly the expected (shard_row, window) pairs: one per window
#     of every text with tokens, none for an empty or missing text, no duplicates, no orphans; the
#     rows read equal the rows the score index counted for the shard;
#   - probabilities finite, within [0, 1], summing to 1 within 1e-5; softmax(logits) equals the
#     stored probabilities within 1e-5;
#   - the coverage weights of every long text sum to n.
# Per platform: every shard named in a score file of a merged partition has a token shard; without
# --partitions, no score file names a partition that has no token shards.
# Output side, on the temporary file before it is moved into place:
#   - row count = token rows of the partition = merged rows; score rows = expected model inputs;
#   - null counts (parquet metadata): p_* null exactly for the rows with n_windows = 0,
#     pc_*/pw_* exactly for the rows with n_windows < 2, no other column null;
#   - (row_id, version) unique (DuckDB, runner.connect: UTC, 4 GB, 3 threads, spill);
#   - p, pc and pw sum to 1 within 1e-5 on every row where they are set.
#
# Stats file: inputs (every token shard and score file with bytes and md5), the counts (rows by
# version, missing, empty, clean errors, scored rows, long rows, model inputs, windows per text,
# long rows whose top emotion changes between p and pc / pw), the checks, the empty-text score, the
# stage-1 counts summed over the shards, and the source column of row_id as named by stage 1.
# A partition whose stats match its current inputs and code is skipped (runner.is_current).
#
# Memory: the partition is merged shard by shard (about 200,000 source rows each) and written as one
# row group per shard, so memory does not grow with the partition. The score files are indexed once
# per platform by reading only their shard column.
#
# Usage
# -----
#     python pipeline/06_emotion_emtract.py merge --reference-empty-score --out <file>   # env emtract-ref
#     python pipeline/06_emotion_emtract.py merge --empty-score --reference <file>       # env thesis: records both
#     python pipeline/06_emotion_emtract.py merge reddit [--partitions BTC ETH] [--force]
#     python pipeline/06_emotion_emtract.py merge stocktwits
#     python pipeline/06_emotion_emtract.py merge twitter
# The merge runs in environment `thesis` (DuckDB 1.5.5, pyarrow); the reference mode runs in
# `emtract-ref` (EmTract's exact versions) and calls EmTract's own Model('stocktwits').predict(['']).
W = E.MAX_TEXT_TOKENS            # 62 wordpieces per window
M = E.MIN_WINDOW_OVERLAP         # 16
SNAP_RADIUS = 8                  # interface item 2: interior starts move at most 8 tokens
SOFTMAX_TOL = 1e-5
EMPTY_REF_TOL = 1e-5             # empty-text score: torch 2.13 vs EmTract's versions (measured <= 3.5e-6 on 10,000 posts)
TOKEN_COLUMNS = ["row_id", "version", "is_missing_text", "is_empty_after_clean", "is_clean_error",
                 "n_tokens", "input_ids", "window_starts"]
SCORE_COLUMNS = ["shard", "shard_row", "window"] + LOGIT_COLS + P_COLS
VERSION_CODE = {v: i for i, v in enumerate(E.VERSIONS)}          # own 0, quoted 1, both 2
MERGE_RULES_SHA = hashlib.sha256(repr((E.LABELS, W, M, SNAP_RADIUS, PROB_SUM_TOL, SOFTMAX_TOL, E.VERSIONS,
                                 "p=window 0", "pc=equal-weight mean", "pw=coverage-weighted mean")
                                ).encode()).hexdigest()[:16]
# Stage 1 names the source column of row_id in its stats file; these are the keys looked for.
ROW_ID_SOURCE_KEYS = ("row_id_source", "row_id_column", "source_row_id_column", "row_id_source_column")


# --- Window plan (interface item 2) and coverage weights -------------------------------------------

def n_windows_for(n_tokens) -> np.ndarray:
    """k per text: 0 without tokens, 1 for 1-62 wordpieces, ceil((n - 16)/46) above."""
    n = np.asarray(n_tokens, dtype=np.int64)
    long_k = -(-(n - M) // (W - M))
    return np.where(n <= 0, 0, np.where(n <= W, 1, long_k)).astype(np.int64)


def window_plan_error(n: int, starts: Sequence[int]) -> Optional[str]:
    """None if `starts` is a valid plan for a text of n wordpieces, else the reason."""
    k = len(starts)
    if n <= 0:
        return None if k == 0 else "text without tokens has windows"
    if n <= W:
        return None if list(starts) == [0] else "text of 1-62 wordpieces must have window_starts [0]"
    want = int(n_windows_for([n])[0])
    if k != want:
        return f"{k} windows, expected ceil((n-16)/46) = {want}"
    if starts[0] != 0:
        return "first window does not start at 0"
    if starts[-1] != n - W:
        return "last window does not end at the last token"
    for j in range(k - 1):
        if starts[j + 1] <= starts[j]:
            return "window starts not increasing"
        if starts[j] + W - starts[j + 1] < M:
            return f"overlap below {M} between windows {j} and {j + 1}"
    for j in range(1, k - 1):
        ideal = j * (n - W) / (k - 1)
        if abs(starts[j] - ideal) > SNAP_RADIUS + 0.5:
            return f"window {j} starts more than {SNAP_RADIUS} tokens from its ideal position"
    return None


@lru_cache(maxsize=1 << 16)
def coverage_weights(n: int, starts: Tuple[int, ...]) -> np.ndarray:
    """w_j = sum over the tokens of window j of 1/c_t, c_t = number of windows covering token t."""
    diff = np.zeros(n + 1, dtype=np.int64)
    for s in starts:
        diff[s] += 1
        diff[s + W] -= 1
    cover = np.cumsum(diff[:-1])
    if not (cover >= 1).all():
        raise CheckError(f"window plan leaves tokens uncovered (n={n})")
    cum = np.concatenate([[0.0], np.cumsum(1.0 / cover)])
    s = np.asarray(starts, dtype=np.int64)
    w = cum[s + W] - cum[s]
    w.setflags(write=False)
    return w


def softmax64(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --- Stats files of the inputs ---------------------------------------------------------------------

def read_input_stats(path: Path, what: str) -> Dict[str, object]:
    """The stats file next to a stage-1 or stage-2 output: it must exist, pass and match the file's size."""
    sp = runner.stats_path(path)
    check(sp.exists(), f"{what} {path.name}: stats file missing (not committed by its step?)")
    rec = json.loads(sp.read_text())
    checks = rec.get("checks", [])
    check(all(c.get("passed") for c in checks), f"{what} {path.name}: its stats record a failed check")
    for o in rec.get("outputs", []):
        if Path(str(o.get("path", ""))).name == path.name:
            check(int(o["bytes"]) == path.stat().st_size, f"{what} {path.name}: size differs from its stats file")
    return rec


def row_id_source(rec: Dict[str, object]) -> Optional[str]:
    for where in (rec, rec.get("counts") or {}, rec.get("settings") or {}):
        for key in ROW_ID_SOURCE_KEYS:
            if isinstance(where, dict) and where.get(key):
                return str(where[key])
    return None


# --- Score index: which score file row groups hold which shard -------------------------------------

def score_files(platform: str) -> List[Path]:
    d = E.scores_dir(platform)
    return sorted(p for p in d.glob("*.parquet")) if d.exists() else []


def index_scores(platform: str) -> Tuple[Dict[str, List[Tuple[Path, int]]], Dict[str, int]]:
    """shard -> [(score file, row group)], shard -> score rows. Reads only the shard column."""
    where: Dict[str, List[Tuple[Path, int]]] = {}
    rows: Counter = Counter()
    for f in score_files(platform):
        read_input_stats(f, "score file")
        pf = pq.ParquetFile(f)
        missing = [c for c in SCORE_COLUMNS if c not in pf.schema_arrow.names]
        check(not missing, f"score file {f.name}: columns missing {missing}")
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=["shard"]).column(0).combine_chunks()
            check(col.null_count == 0, f"score file {f.name}: null shard")
            enc = col if pa.types.is_dictionary(col.type) else pc.dictionary_encode(col)
            counts = np.bincount(enc.indices.to_numpy(zero_copy_only=False), minlength=len(enc.dictionary))
            for value, cnt in zip(enc.dictionary.to_pylist(), counts.tolist()):
                if cnt:
                    where.setdefault(value, []).append((f, rg))
                    rows[value] += cnt
    return where, dict(rows)


def read_shard_scores(shard_rel: str, where: Dict[str, List[Tuple[Path, int]]]) -> Optional[pa.Table]:
    parts = []
    for f, rg in where.get(shard_rel, []):
        t = pq.ParquetFile(f).read_row_group(rg, columns=SCORE_COLUMNS)
        if pa.types.is_dictionary(t.schema.field("shard").type):
            t = t.set_column(t.schema.get_field_index("shard"), "shard", t.column("shard").cast(pa.string()))
        parts.append(t.filter(pc.equal(t.column("shard"), pa.scalar(shard_rel, pa.string()))))
    return pa.concat_tables(parts) if parts else None


# --- One shard -------------------------------------------------------------------------------------

def check_version_order(tok: pa.Table, where: str) -> np.ndarray:
    ver = tok.column("version").combine_chunks().cast(pa.string())
    bad = pc.sum(pc.invert(pc.is_in(ver, value_set=pa.array(list(E.VERSIONS))))).as_py() or 0
    check(bad == 0, f"{where}: {bad} rows with a version outside {E.VERSIONS}")
    codes = np.zeros(len(ver), dtype=np.int8)
    for v, c in VERSION_CODE.items():
        codes[_np(pc.equal(ver, v))] = c
    rid = tok.column("row_id").combine_chunks()
    n = len(codes)
    if n:
        same_prev = np.zeros(n, dtype=bool)
        if n > 1:
            same_prev[1:] = _np(pc.equal(rid.slice(1), rid.slice(0, n - 1)))
        prev = np.concatenate([[-1], codes[:-1]])
        nxt = np.concatenate([codes[1:], [-1]])
        ok_quoted = (codes != 1) | ((prev == 0) & same_prev & (nxt == 2))
        ok_both = (codes != 2) | ((prev == 1) & same_prev)
        ok_own = (codes != 0) | ~same_prev
        check(bool(ok_quoted.all() and ok_both.all() and ok_own.all()),
              f"{where}: versions not in the order own[, quoted, both] per row_id")
    return codes


def merge_shard(tok: pa.Table, scores: Optional[pa.Table], where: str, expected_score_rows: int,
                counts: Counter) -> pa.Table:
    """Checks one token shard against its scores and returns its merged rows."""
    missing_cols = [c for c in TOKEN_COLUMNS if c not in tok.column_names]
    check(not missing_cols, f"{where}: token columns missing {missing_cols}")
    for c in TOKEN_COLUMNS:
        check(tok.column(c).null_count == 0, f"{where}: nulls in {c}")
    n_rows = tok.num_rows
    codes = check_version_order(tok, where)
    ntok = _np(tok.column("n_tokens")).astype(np.int64)
    miss = _np(tok.column("is_missing_text")).astype(bool)
    empty = _np(tok.column("is_empty_after_clean")).astype(bool)
    cerr = _np(tok.column("is_clean_error")).astype(bool)
    ids = tok.column("input_ids").combine_chunks()
    ws = tok.column("window_starts").combine_chunks()
    ids_len = _np(pc.list_value_length(ids)).astype(np.int64)
    k = _np(pc.list_value_length(ws)).astype(np.int64)
    flat = _np(pc.list_flatten(ws)).astype(np.int64)

    check(bool((ntok >= 0).all()), f"{where}: negative n_tokens")
    check(bool((ids_len == ntok).all()), f"{where}: len(input_ids) != n_tokens on {int((ids_len != ntok).sum())} rows")
    check(bool(((ntok == 0) == (k == 0)).all()), f"{where}: window_starts is [] not exactly when n_tokens = 0")
    check(bool((ntok[miss | empty] == 0).all()), f"{where}: a text flagged missing or empty has tokens")
    gaps = (k == 0) & ~(miss | empty)
    check(not gaps.any(), f"{where}: {int(gaps.sum())} rows without tokens are neither missing nor empty "
                          f"({int((gaps & cerr).sum())} of them flagged is_clean_error)")
    check(bool((empty | ~cerr).all()), f"{where}: a clean error not flagged empty (stage 1 counts it as empty)")
    check(bool((k == n_windows_for(ntok)).all()), f"{where}: number of windows differs from ceil((n-16)/46)")
    seg = np.concatenate([[0], np.cumsum(k)[:-1]]).astype(np.int64)
    short = k == 1
    check(bool((flat[seg[short]] == 0).all()), f"{where}: a text of 1-62 wordpieces has a window start other than 0")
    long_rows = np.flatnonzero(k >= 2)
    for i in long_rows:
        err = window_plan_error(int(ntok[i]), flat[seg[i]:seg[i] + k[i]].tolist())
        check(err is None, f"{where}: shard row {int(i)}: {err}")

    # --- scores: exactly the expected (shard_row, window) pairs
    total = int(k.sum())
    exp_row = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    exp_win = np.arange(total, dtype=np.int64) - np.repeat(seg, k)
    got = scores.num_rows if scores is not None else 0
    check(got == expected_score_rows, f"{where}: read {got} score rows, the score index counted {expected_score_rows}")
    if scores is None or got == 0:
        sr = np.zeros(0, np.int64)
        win = np.zeros(0, np.int64)
    else:
        for c in SCORE_COLUMNS:
            check(scores.column(c).null_count == 0, f"{where}: nulls in score column {c}")
        sr = _np(scores.column("shard_row")).astype(np.int64)
        win = _np(scores.column("window")).astype(np.int64)
    order = np.lexsort((win, sr))
    sr_s, win_s = sr[order], win[order]
    if not (len(sr_s) == total and np.array_equal(sr_s, exp_row) and np.array_equal(win_s, exp_win)):
        key_exp = exp_row * 65536 + exp_win
        key_got = sr * 65536 + win
        uniq, cnt = np.unique(key_got, return_counts=True)
        dup = int((cnt > 1).sum())
        missing = int(np.setdiff1d(key_exp, uniq, assume_unique=True).size)
        orphan = int(np.setdiff1d(uniq, key_exp, assume_unique=True).size)
        raise CheckError(f"{where}: score rows do not match the expected inputs: expected {total}, got {len(sr)}; "
                         f"{missing} inputs without a score, {dup} inputs scored more than once, "
                         f"{orphan} scores for no expected input")

    L = len(E.LABELS)
    p_out = np.zeros((n_rows, L), dtype=np.float32)
    pc_out = np.zeros((n_rows, L), dtype=np.float32)
    pw_out = np.zeros((n_rows, L), dtype=np.float32)
    has = k >= 1
    long_mask = k >= 2
    if total:
        P = np.column_stack([_np(scores.column(c)) for c in P_COLS]).astype(np.float32)[order]
        LG = np.column_stack([_np(scores.column(c)) for c in LOGIT_COLS]).astype(np.float32)[order]
        check(bool(np.isfinite(P).all() and np.isfinite(LG).all()), f"{where}: non-finite scores")
        check(bool(((P >= 0) & (P <= 1)).all()), f"{where}: probabilities outside [0, 1]")
        P64 = P.astype(np.float64)
        dev = float(np.abs(P64.sum(axis=1) - 1.0).max())
        check(dev <= PROB_SUM_TOL, f"{where}: probabilities of an input sum to 1 +- {dev:.2e} > {PROB_SUM_TOL}")
        sm = float(np.abs(softmax64(LG) - P64).max())
        check(sm <= SOFTMAX_TOL, f"{where}: softmax(logits) differs from p by {sm:.2e} > {SOFTMAX_TOL}")
        counts["max_prob_sum_dev_e9"] = max(counts["max_prob_sum_dev_e9"], int(round(dev * 1e9)))
        counts["max_softmax_dev_e9"] = max(counts["max_softmax_dev_e9"], int(round(sm * 1e9)))

        p_out[has] = P[seg[has]]
        starts_has = seg[has]
        sums = np.add.reduceat(P64, starts_has, axis=0)
        pc_all = sums / k[has][:, None]
        w = np.ones(total, dtype=np.float64)
        for i in long_rows:
            s0, ki = int(seg[i]), int(k[i])
            wi = coverage_weights(int(ntok[i]), tuple(flat[s0:s0 + ki].tolist()))
            check(abs(float(wi.sum()) - float(ntok[i])) <= 1e-9 * max(1.0, float(ntok[i])),
                  f"{where}: coverage weights of shard row {int(i)} do not sum to n")
            w[s0:s0 + ki] = wi
        wsum = np.add.reduceat(P64 * w[:, None], starts_has, axis=0)
        wtot = np.add.reduceat(w, starts_has)
        pw_all = wsum / wtot[:, None]
        sel = long_mask[has]
        pc_out[long_mask] = pc_all[sel].astype(np.float32)
        pw_out[long_mask] = pw_all[sel].astype(np.float32)
        if long_mask.any():
            top_p = p_out[long_mask].argmax(axis=1)
            counts["long_top_changes_pc"] += int((pc_out[long_mask].argmax(axis=1) != top_p).sum())
            counts["long_top_changes_pw"] += int((pw_out[long_mask].argmax(axis=1) != top_p).sum())

    # --- counts
    counts["rows"] += n_rows
    for v, c in VERSION_CODE.items():
        counts[f"rows_{v}"] += int((codes == c).sum())
    counts["missing_text"] += int(miss.sum())
    counts["empty_after_clean"] += int(empty.sum())
    counts["clean_error"] += int(cerr.sum())
    counts["scored_rows"] += int(has.sum())
    counts["long_rows"] += int(long_mask.sum())
    counts["model_inputs"] += total
    if has.any():
        for kk, c in zip(*np.unique(np.minimum(k[has], 6), return_counts=True)):
            counts[f"windows_{int(kk)}{'_or_more' if kk == 6 else ''}"] += int(c)
    counts["max_windows"] = max(counts["max_windows"], int(k.max()) if n_rows else 0)

    cols = {
        "row_id": tok.column("row_id"),
        "version": tok.column("version").cast(pa.string()),
        "n_tokens": pa.array(ntok.astype(np.int32)),
        "is_missing_text": pa.array(miss),
        "is_empty_after_clean": pa.array(empty),
        "is_clean_error": pa.array(cerr),
        "n_windows": pa.array(k.astype(np.int16)),
    }
    for j, c in enumerate(P_COLS):
        cols[c] = pa.array(p_out[:, j], mask=~has)
    for j, c in enumerate(PC_COLS):
        cols[c] = pa.array(pc_out[:, j], mask=~long_mask)
    for j, c in enumerate(PW_COLS):
        cols[c] = pa.array(pw_out[:, j], mask=~long_mask)
    return pa.table(cols)


# --- Empty-text score ------------------------------------------------------------------------------

def _sums_entry(folder: Path, name: str) -> Optional[str]:
    sums = folder / "SHA256SUMS"
    if not sums.exists():
        return None
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*").split("/")[-1] == name:
            return parts[0]
    return None


def special_ids(tok_dir: Path) -> List[int]:
    """[CLS] and [SEP] ids from the tokenizer's own files (special_tokens_map.json names the tokens,
    vocab.txt gives their ids, as in every BERT wordpiece tokenizer). tokenizers >= 0.22 cannot load
    EmTract's tokenizer.json (DuplicatePatternError on its added emoji tokens), so env thesis reads
    the files; --reference checks the ids against EmTract's own tokenizer (tokenizers 0.10.3)."""
    names = json.loads((tok_dir / "special_tokens_map.json").read_text())
    vocab = {t: i for i, t in enumerate((tok_dir / "vocab.txt").read_text(encoding="utf-8").split("\n"))}
    return [int(vocab[names["cls_token"]]), int(vocab[names["sep_token"]])]


def compute_empty_score(reference: Optional[Path]) -> Dict[str, object]:
    """Env emtract, CPU, fp32: the stocktwits model on [CLS][SEP] (the input of an empty text)."""
    import torch
    import transformers

    model_dir = E.EMTRACT_MODEL_DIR / E.MODEL_VARIANT
    tok_dir = E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT
    sha = sha256_file(model_dir / "pytorch_model.bin")
    check(sha == _sums_entry(model_dir, "pytorch_model.bin"), "model weights differ from SHA256SUMS")
    ids = special_ids(tok_dir)
    model = transformers.DistilBertForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(E.LABELS))
    model = model.to("cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, 2, dtype=torch.long)).logits[0]
    p32 = torch.softmax(logits, dim=-1).numpy()
    lg = logits.numpy()
    rec: Dict[str, object] = {
        "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "written_by": STEP_MERGE + " --empty-score",
        "model_variant": E.MODEL_VARIANT,
        "tokenizer_variant": E.TOKENIZER_VARIANT,
        "model_weights_sha256": sha,
        "input": "[CLS] [SEP] (ids from the tokenizer files), attention mask 1 1, no padding",
        "input_ids": ids,
        "device": "cpu",
        "dtype": "float32",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "labels": list(E.LABELS),
        "logits": {lab: float(v) for lab, v in zip(E.LABELS, lg)},
        "p": {lab: float(v) for lab, v in zip(E.LABELS, p32)},
        "top_emotion": E.LABELS[int(np.argmax(p32))],
        "note": "EmTract's predict() gives every empty text this score; step 08 leaves empty and missing texts null",
    }
    if reference is not None:
        ref = json.loads(Path(reference).read_text())
        check(ref.get("input_ids_first_two") == ids, "special-token ids differ from EmTract's own tokenizer")
        diff = max(abs(float(ref["p"][lab]) - float(rec["p"][lab])) for lab in E.LABELS)
        check(diff <= EMPTY_REF_TOL, f"empty-text score differs from EmTract's predict('') by {diff:.2e}")
        ref["max_abs_diff_to_p"] = diff
        rec["reference_predict_empty"] = ref
    return rec


def reference_empty_score() -> Dict[str, object]:
    """Env emtract-ref: EmTract's own Model('stocktwits').predict(['']) with EmTract's exact versions.

    model.py loads './build/models/emotion-<variant>' and cleaning.py reads 'emtract/data/dictionaries/'
    relative to the working directory, so both are linked into a temporary folder; nothing is copied."""
    tmp = Path(tempfile.mkdtemp(prefix="emtract_ref_"))
    cwd = os.getcwd()
    try:
        (tmp / "build" / "models").mkdir(parents=True)
        for variant in ("twitter", "stocktwits"):
            os.symlink(E.EMTRACT_MODEL_DIR / variant, tmp / "build" / "models" / f"emotion-{variant}")
        os.symlink(E.EMTRACT_CODE_DIR / "emtract", tmp / "emtract")
        os.chdir(tmp)
        sys.path.insert(0, str(tmp))
        import torch
        import transformers
        from emtract import model as emtract_model  # EmTract's code, unchanged
        df = emtract_model.Model(E.MODEL_VARIANT).predict([""])
        enc = emtract_model.tokenize({"text": [""]})["input_ids"][0]
        return {
            "method": f"EmTract Model('{E.MODEL_VARIANT}').predict(['']) (upstream {E.EMTRACT_UPSTREAM_COMMIT[:7]})",
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "transformers": transformers.__version__},
            "input_ids_first_two": [int(x) for x in enc[:2]],
            "input_length": len(enc),
            "n_padding": int(sum(1 for x in enc[2:] if x == emtract_model.tokenizer.pad_token_id)),
            "p": {lab: float(df[lab].iloc[0]) for lab in E.LABELS},
            "top_emotion": E.LABELS[int(df["emotion"].iloc[0])],
        }
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def load_empty_score() -> Dict[str, object]:
    check(E.EMPTY_SCORE_FILE.exists(), "empty-text score missing: run `06_emotion_emtract.py merge --empty-score` first")
    rec = json.loads(E.EMPTY_SCORE_FILE.read_text())
    check(rec.get("model_variant") == E.MODEL_VARIANT, "empty-text score was computed with another model variant")
    check(set(rec.get("p", {})) == set(E.LABELS), "empty-text score does not hold the 7 labels")
    listed = _sums_entry(E.EMTRACT_MODEL_DIR / E.MODEL_VARIANT, "pytorch_model.bin")
    check(listed is not None and listed == rec.get("model_weights_sha256"),
          "empty-text score was computed with weights other than those in SHA256SUMS")
    return rec


# --- One partition ---------------------------------------------------------------------------------

def partitions_of(platform: str) -> List[str]:
    d = E.tokens_dir(platform)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))


def output_checks(tmp: Path, con, expect_rows: int, unscored: int, not_long: int, results: List[Dict]) -> None:
    md = pq.ParquetFile(tmp).metadata
    check(md.num_rows == expect_rows, f"output rows {md.num_rows} = token rows {expect_rows}", results)
    names = md.schema.to_arrow_schema().names
    nulls = {n: 0 for n in names}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        for ci in range(g.num_columns):
            col = g.column(ci)
            st = col.statistics
            check(st is not None and st.has_null_count, f"output has null counts for {col.path_in_schema}")
            nulls[col.path_in_schema] += st.null_count
    for n in names:
        want = unscored if n in P_COLS else not_long if n in PC_COLS + PW_COLS else 0
        check(nulls[n] == want, f"output nulls in {n}: {nulls[n]} = {want}", results)
    path = str(tmp).replace("'", "''")
    n, d = con.execute(f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT row_id, version "
                       f"FROM read_parquet('{path}'))) FROM read_parquet('{path}')").fetchone()
    check(n == d, f"(row_id, version) unique: {d} distinct of {n}", results)
    for cols, tag in ((P_COLS, "p"), (PC_COLS, "pc"), (PW_COLS, "pw")):
        expr = " + ".join(f"CAST({c} AS DOUBLE)" for c in cols)
        dev = con.execute(f"SELECT coalesce(max(abs({expr} - 1)), 0) FROM read_parquet('{path}') "
                          f"WHERE {cols[0]} IS NOT NULL").fetchone()[0]
        check(dev <= PROB_SUM_TOL, f"output {tag}_* sums to 1 within {PROB_SUM_TOL} (max dev {dev:.2e})", results)


def merge_partition(platform: str, partition: str, where: Dict[str, List[Tuple[Path, int]]],
                    score_rows: Dict[str, int], empty_score: Dict[str, object], force: bool,
                    out_dir: Optional[Path] = None) -> Optional[Dict]:
    """Merges one partition into <out_dir>/<partition>.parquet; out_dir defaults to
    emotion_dir(platform) (the DLT keys go to dlt/emtract/emotion_keys/, `dlt merge-keys`)."""
    tdir = E.tokens_dir(platform)
    shards = sorted((tdir / partition).glob("shard-*.parquet"))
    check(len(shards) > 0, f"{platform}/{partition}: no token shards")
    rels = [s.relative_to(tdir).as_posix() for s in shards]
    used_scores = sorted({f for r in rels for f, _ in where.get(r, [])})
    inputs = runner.describe_inputs(list(shards) + used_scores)
    final = (out_dir or E.emotion_dir(platform)) / f"{partition}.parquet"
    if not force and runner.is_current(final, inputs, code_fingerprint(), MERGE_RULES_SHA):
        log(f"{platform}/{partition}: current, skipped")
        return None
    results: List[Dict[str, object]] = []
    counts: Counter = Counter()
    for key in ("max_prob_sum_dev_e9", "max_softmax_dev_e9", "max_windows", "long_top_changes_pc", "long_top_changes_pw"):
        counts[key] = 0
    stage1_sums: Counter = Counter()
    sources = set()
    tmp = runner.tmp_path(final)
    writer = None
    token_rows = 0
    try:
        for shard, rel in zip(shards, rels):
            rec = read_input_stats(shard, "token shard")
            sources.add(row_id_source(rec))
            for kk, v in (rec.get("counts") or {}).items():
                if isinstance(v, int) and not isinstance(v, bool):
                    stage1_sums[kk] += v
            tok = pq.read_table(shard, columns=TOKEN_COLUMNS)
            token_rows += tok.num_rows
            out = merge_shard(tok, read_shard_scores(rel, where), f"{platform}/{rel}", score_rows.get(rel, 0), counts)
            if writer is None:
                writer = pq.ParquetWriter(tmp, out.schema, compression="zstd", write_statistics=True)
            writer.write_table(out, row_group_size=max(1, out.num_rows))
            log(f"{platform}/{rel}: {out.num_rows} rows merged")
    finally:
        if writer is not None:
            writer.close()
    check(len(sources) == 1, f"{platform}/{partition}: shards name one row_id source column "
                             f"({sorted(map(str, sources))})", results)
    check(counts["rows"] == token_rows, f"merged rows {counts['rows']} = token rows {token_rows}", results)
    n_scores = sum(score_rows.get(r, 0) for r in rels)
    check(counts["model_inputs"] == n_scores, f"score rows of the partition {n_scores} = expected inputs "
                                              f"{counts['model_inputs']}", results)
    results.append({"check": "per shard: stage-1 flags, window plans, one score per expected input, "
                             "probability sums and softmax (see header)", "passed": True})
    con = runner.connect()
    try:
        output_checks(tmp, con, token_rows, token_rows - counts["scored_rows"], token_rows - counts["long_rows"], results)
        runner.commit(tmp, final)
        runner.write_stats(final, step=STEP_MERGE, inputs=inputs, code_sha=code_fingerprint(), rules_sha=MERGE_RULES_SHA, con=con,
                           counts=dict(counts), checks=results, extra={
                               "platform": platform, "partition": partition,
                               "row_id_source": next(iter(sources)),
                               "labels": list(E.LABELS),
                               "aggregation": {"p": "window 0 = [CLS] + first 62 wordpieces + [SEP]",
                                               "pc": "equal-weight mean of the window probabilities (n_windows >= 2)",
                                               "pw": "coverage-weighted mean, w_j = sum_t in j 1/c_t (n_windows >= 2)",
                                               "empty_or_missing": "null (no model call)"},
                               "empty_text_score": empty_score,
                               "stage1_counts_summed": dict(stage1_sums),
                           })
    finally:
        con.close()
    log(f"{platform}/{partition}: {token_rows} rows, {counts['model_inputs']} inputs -> {final.name}")
    return dict(counts)


def merge(platform: str, partitions: Optional[Sequence[str]], force: bool) -> Dict[str, Dict]:
    check(platform in config.PLATFORMS, f"unknown platform {platform}")
    empty_score = load_empty_score()
    found = partitions_of(platform)
    chosen = list(partitions) if partitions else found
    unknown = [p for p in chosen if p not in found]
    check(not unknown, f"{platform}: no token shards for partitions {unknown}")
    where, score_rows = index_scores(platform)
    tdir = E.tokens_dir(platform)
    orphans = [s for s in score_rows if s.split("/", 1)[0] in chosen and not (tdir / s).exists()]
    check(not orphans, f"{platform}: {len(orphans)} shards in the score files have no token shard (first: {orphans[:3]})")
    if not partitions:
        other = sorted({s.split("/", 1)[0] for s in score_rows} - set(found))
        check(not other, f"{platform}: score files name partitions without token shards: {other}")
    out = {}
    for part in chosen:
        res = merge_partition(platform, part, where, score_rows, empty_score, force)
        if res is not None:
            out[part] = res
    return out


def main_merge(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="06_emotion_emtract.py merge",
                                 description="EmTract merge: one score row per post and text version (section 3)")
    ap.add_argument("platform", nargs="?", choices=config.PLATFORMS)
    ap.add_argument("--partitions", nargs="+")
    ap.add_argument("--force", action="store_true", help="merge even if the stats say the output is current")
    ap.add_argument("--empty-score", action="store_true", help="compute EmTract's empty-text score (env thesis)")
    ap.add_argument("--reference", type=Path, help="with --empty-score: the file written by --reference-empty-score")
    ap.add_argument("--reference-empty-score", action="store_true", help="EmTract's predict('') (env emtract-ref)")
    ap.add_argument("--out", type=Path, help="with --reference-empty-score: where to write the result")
    a = ap.parse_args(argv)
    if a.reference_empty_score:
        check(a.out is not None, "--reference-empty-score needs --out")
        runner.write_json_atomic(a.out, reference_empty_score())
        return 0
    if a.empty_score:
        rec = compute_empty_score(a.reference)
        runner.write_json_atomic(E.EMPTY_SCORE_FILE, rec)
        log(f"empty-text score recorded: top emotion {rec['top_emotion']}")
        return 0
    if not a.platform:
        ap.error("platform required")
    merge(a.platform, a.partitions, a.force)
    return 0


# ==================================================================================================
# 4. Text-key index of the Twitter scores (`text-key-index`, `text-key-list`)
# ==================================================================================================
# The text-key index of the EmTract-scored Twitter corpus (IA stream), so that an
# identical cleaned text in another corpus (the DLT event arm, MASTERPLAN_NOV15.md section 7, D17) can reuse
# the stored score instead of being scored again. New file, 30 Sep 2026; it changes no existing table.
#
# Input    twitter/emtract/tokens/units/shard-NNNNN.parquet (`tokenize`, each with its stats file) and
#          twitter/emtract/emotion/units.parquet (`merge`: one row group per token shard, in shard order,
#          rows in the shard's order -- checked here row by row).
# Output   twitter/emtract/text_key_index.parquet with a stats file (runner.write_stats): one row per distinct
#          text_key over the text versions 'own' and 'quoted', sorted by text_key. 'both' (own text + " " +
#          quoted text) is left out: it is a concatenation that a single post elsewhere cannot contain.
#          Written to <dir>/_tmp/ first, checked, then moved into place (runner.commit).
#
# The text key (shared rule for matching another corpus with this one, 30 Sep 2026)
# ---------------------------------------------------------------------------------
#     text_key = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
# on EmTract's cleaned text exactly as stored by stage 1, i.e. the string that tokenizers 0.10.3 turned into
# input_ids. Identical cleaned text -> identical token ids -> identical model inputs (same windows, same
# padding bucket) -> identical scores up to float noise from the batch an input was scored in. Texts that
# clean to "" (empty after cleaning; clean errors included) share the key of "" and have null scores (no
# model call), as in the emotion table. A row without cleaned text (is_missing_text; none on Twitter) gets
# no key; it is counted, not indexed.
#
# Columns
# -------
# text_key                sha256 hex (64 lower-case characters) of the cleaned text, UTF-8
# n_tokens                int32, wordpieces of the cleaned text (equal on every row sharing the key; checked)
# is_empty_after_clean    bool (equal on every row sharing the key; checked)
# n_rows                  int64, rows (row_id, version) of the versions own and quoted that share the key
# rep_row_id, rep_version the representative row: the smallest (row_id, version) sharing the key (a technical
#                         choice: the rows sharing a key hold the same score up to float noise)
# n_windows, p_*, pc_*, pw_*  the representative's values in the emotion table, unchanged (types as stored)
#
# Checks (runner standard: a failed check commits nothing)
# --------------------------------------------------------
# * every token shard's stats file exists and all its checks passed; its recorded bytes and md5 equal the
#   file's; each shard's md5 equals the one the emotion table's stats file recorded for it; the emotion
#   file's md5 equals the one in its own stats file, whose checks all passed;
# * per shard: the emotion row group has the shard's (row_id, version) sequence and the same n_tokens and
#   flags; n_tokens = len(input_ids); cleaned_text is NULL exactly for missing texts and "" exactly for
#   empty texts;
# * per key: n_tokens, n_windows, is_empty_after_clean and the token ids with their window plan (sha256 of
#   input_ids and window_starts) are equal on every row; every p, pc and pw column is null on all rows or on
#   none; p is null exactly for empty keys, pc and pw exactly when n_windows < 2;
# * the keys are 64 lower-case hex characters and strictly increasing through the file (hence distinct);
#   sum(n_rows) = rows indexed; all empty rows sit on the key of "", the only empty key; the file read back
#   has the counted rows and sum(n_rows);
# * spot check: SPOT_KEYS keys (half a reservoir sample of all keys, half of the keys shared by two or more
#   rows; DuckDB, one thread, fixed seeds): the cleaned text of up to SPOT_MEMBERS member rows per key is
#   read back from the token shards with DuckDB and hashed again; each key's member count equals n_rows;
#   the representative is the smallest member; its values in the emotion table equal the index row.
#
# Recorded, not checked (float noise, not an error): the largest |dp| between rows sharing a key, for p, pc
# and pw, and the number of keys above 0, 1e-6, 3e-6 and 1e-5.
#
# Memory: pass 1 reads one shard at a time and writes its rows into 16 bucket files by the key's first hex
# character (a scratch directory under config.SCRATCH_DIR, removed at the end unless --keep-work); pass 2
# aggregates one bucket at a time in DuckDB (runner.connect: UTC asserted, memory limit, threads, a spill
# directory inside the scratch directory).
#
# The key list (`text-key-list`): twitter/emtract/text_key_list.parquet, the index's text_key column as
# 32-byte binary digests (bytes.fromhex), ascending, zstd, row groups of 1,000,000, with a small JSON
# stats file. It is the only file the DLT match needs from the Twitter side (`dlt match --key-list`).
#
# Usage (environment thesis)
# ---------------------------
#     python pipeline/06_emotion_emtract.py text-key-index [twitter] [--force] [--keep-work]
#     python pipeline/06_emotion_emtract.py text-key-list
TKI_PARTITION = "units"                       # Twitter has one partition
KEY_VERSIONS = ("own", "quoted")          # 'both' left out (see header)
SCORE_COLS = ["n_windows"] + P_COLS + PC_COLS + PW_COLS
FLAG_COLS = ["is_missing_text", "is_empty_after_clean", "is_clean_error"]
TKI_TOKEN_COLUMNS = ["row_id", "version", "cleaned_text"] + FLAG_COLS + ["n_tokens", "input_ids", "window_starts"]
EMOTION_COLUMNS = ["row_id", "version", "n_tokens"] + FLAG_COLS + SCORE_COLS
N_BUCKETS = 16                            # first hex character of the key
FLUSH_ROWS = 100_000                      # bucket buffer size before a write
ROW_GROUP_ROWS = 250_000                  # output row groups (min/max of text_key per group)
SPOT_KEYS = 200
SPOT_MEMBERS = 50
SPOT_SEEDS = (20260930, 20261001)
DP_THRESHOLDS = (0.0, 1e-6, 3e-6, 1e-5)
EMPTY_KEY = hashlib.sha256(b"").hexdigest()
KEY_RULE = 'text_key = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()'
REP_RULE = "smallest (row_id, version) among the rows sharing the key"
TKI_RULES_SHA = hashlib.sha256(repr((KEY_RULE, KEY_VERSIONS, REP_RULE, E.LABELS, SCORE_COLS)).encode()).hexdigest()[:16]


def text_key(cleaned: Optional[str]) -> Optional[str]:
    """The shared text key: sha256 hex of the cleaned text's UTF-8 bytes (None for a missing text)."""
    return None if cleaned is None else hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def sql_str(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_list(paths: Sequence[object]) -> str:
    return "[" + ", ".join(sql_str(p) for p in paths) + "]"


# --- Inputs: stats files of the shards and of the emotion table ------------------------------------

def check_inputs(shards: List[Path], emo_file: Path, inputs: List[Dict[str, object]],
                 checks: List[Dict[str, object]]) -> Dict[str, object]:
    """Stats files of the shards and of the emotion table against the files as they are now."""
    md5 = {str(i["path"]): (i["bytes"], i["md5"]) for i in inputs}
    bad_stats = []
    for s in shards:
        sp = runner.stats_path(s)
        try:
            rec = json.loads(sp.read_text())
        except (OSError, ValueError):
            bad_stats.append(s.name)
            continue
        out = rec.get("outputs") or [{}]
        ok = (all(c.get("passed") for c in rec.get("checks", [])) and bool(rec.get("checks"))
              and (out[0].get("bytes"), out[0].get("md5")) == md5[str(s)])
        if not ok:
            bad_stats.append(s.name)
    check(not bad_stats, f"every token shard has a stats file whose checks passed and whose bytes and md5 "
          f"equal the file's ({len(shards)} shards; bad: {bad_stats[:3]})", checks)
    erec = json.loads(runner.stats_path(emo_file).read_text())
    eout = erec["outputs"][0]
    check(all(c.get("passed") for c in erec["checks"]) and (eout["bytes"], eout["md5"]) == md5[str(emo_file)],
          "the emotion table's stats file: all checks passed, bytes and md5 equal the file's", checks)
    recorded = {str(i["path"]): (i["bytes"], i["md5"]) for i in erec["inputs"] if "/tokens/" in str(i["path"])}
    check(len(recorded) == len(shards) and all(recorded.get(str(s)) == md5[str(s)] for s in shards),
          f"each token shard equals (bytes, md5) the input the merge recorded for it ({len(recorded)} recorded)",
          checks)
    return {"emotion_stats_written_at_utc": erec.get("written_at_utc"), "emotion_code_sha": erec.get("code_sha"),
            "emotion_rows": erec.get("counts", {}).get("rows"), "row_id_source": erec.get("row_id_source")}


# --- Pass 1: keys per row, into bucket files -------------------------------------------------------

def token_digest(ids: pa.ChunkedArray, starts: pa.ChunkedArray, n_tokens: np.ndarray, where: str) -> np.ndarray:
    """int64 of the first 8 bytes of sha256(input_ids as uint16 LE | window_starts as int32 LE), per row."""
    ids = ids.combine_chunks()
    starts = starts.combine_chunks()
    check(ids.null_count == 0 and starts.null_count == 0, f"{where}: input_ids and window_starts are never null")
    off, woff = ids.offsets.to_numpy(), starts.offsets.to_numpy()
    check(np.array_equal(np.diff(off), n_tokens), f"{where}: n_tokens = len(input_ids)")
    vb = ids.values.to_numpy(zero_copy_only=False).astype("<u2").tobytes()
    wb = starts.values.to_numpy(zero_copy_only=False).astype("<i4").tobytes()
    out = np.empty(len(off) - 1, dtype=np.int64)
    for j in range(len(out)):
        h = hashlib.sha256(vb[2 * off[j]:2 * off[j + 1]])
        h.update(b"|")
        h.update(wb[4 * woff[j]:4 * woff[j + 1]])
        out[j] = int.from_bytes(h.digest()[:8], "little", signed=True)
    return out


def pass1(shards: List[Path], emo_file: Path, work: Path, checks: List[Dict[str, object]]
          ) -> Tuple[List[Path], Counter, pa.Schema]:
    emo = pq.ParquetFile(emo_file)
    check(emo.metadata.num_row_groups == len(shards),
          f"emotion row groups {emo.metadata.num_row_groups} = token shards {len(shards)}", checks)
    emo_schema = emo.schema_arrow
    files = [work / f"bucket-{b:x}.parquet" for b in range(N_BUCKETS)]
    writers: List[Optional[pq.ParquetWriter]] = [None] * N_BUCKETS
    buffers: List[List[pa.Table]] = [[] for _ in range(N_BUCKETS)]
    buffered = [0] * N_BUCKETS
    c: Counter = Counter()
    for k in ("rows_read", "rows_key_versions", "empty_rows", "clean_error_rows", "missing_rows",
              "rows_without_key"):
        c[k] = 0

    def flush(b: int) -> None:
        if not buffers[b]:
            return
        t = pa.concat_tables(buffers[b])
        if writers[b] is None:
            writers[b] = pq.ParquetWriter(files[b], t.schema, compression="zstd")
        writers[b].write_table(t)
        c[f"bucket_rows_{b:x}"] += t.num_rows
        buffers[b], buffered[b] = [], 0

    t0 = time.time()
    for i, shard in enumerate(shards):
        where = f"{shard.parent.name}/{shard.name}"
        tok = pq.read_table(shard, columns=TKI_TOKEN_COLUMNS)
        emt = emo.read_row_group(i, columns=EMOTION_COLUMNS)
        check(tok.num_rows == emt.num_rows, f"{where}: emotion row group {i} has the shard's row count")
        for col in ["row_id", "version", "n_tokens"] + FLAG_COLS:
            check(tok.column(col).equals(emt.column(col)), f"{where}: emotion row group {i} equals the shard in {col}")
        ver = tok.column("version").to_pylist()
        ct = tok.column("cleaned_text").to_pylist()
        miss = tok.column("is_missing_text").to_pylist()
        emp = tok.column("is_empty_after_clean").to_pylist()
        err = tok.column("is_clean_error").to_pylist()
        ntok = tok.column("n_tokens").to_numpy()
        check(all((t is None) == m and (t == "") == e for t, m, e in zip(ct, miss, emp)),
              f"{where}: cleaned_text is NULL exactly for missing texts and '' exactly for empty texts")
        dig = token_digest(tok.column("input_ids"), tok.column("window_starts"), ntok, where)
        c["rows_read"] += tok.num_rows
        keep: List[int] = []
        keys: List[str] = []
        for j, v in enumerate(ver):
            c[f"rows_{v}"] += 1
            if v not in KEY_VERSIONS:
                continue
            c["rows_key_versions"] += 1
            c["empty_rows"] += bool(emp[j])
            c["clean_error_rows"] += bool(err[j])
            c["missing_rows"] += bool(miss[j])
            if ct[j] is None:
                c["rows_without_key"] += 1
                continue
            keep.append(j)
            keys.append(text_key(ct[j]))
        if not keep:
            continue
        idx = pa.array(keep, type=pa.int64())
        cols = {
            "text_key": pa.array(keys, type=pa.string()),
            "row_id": tok.column("row_id").take(idx),
            "version": tok.column("version").take(idx),
            "shard": pa.array(np.full(len(keep), i, dtype=np.int32)),
            "n_tokens": tok.column("n_tokens").take(idx),
            "is_empty_after_clean": tok.column("is_empty_after_clean").take(idx),
            "is_clean_error": tok.column("is_clean_error").take(idx),
            "tok_digest": pa.array(dig[np.asarray(keep)]),
        }
        cols.update({col: emt.column(col).take(idx) for col in SCORE_COLS})
        part = pa.table(cols)
        bucket = np.fromiter((int(k[0], 16) for k in keys), dtype=np.int8, count=len(keys))
        for b in range(N_BUCKETS):
            m = np.nonzero(bucket == b)[0]
            if len(m):
                buffers[b].append(part.take(pa.array(m, type=pa.int64())))
                buffered[b] += len(m)
                if buffered[b] >= FLUSH_ROWS:
                    flush(b)
        if (i + 1) % 100 == 0 or i + 1 == len(shards):
            el = time.time() - t0
            log(f"pass 1: {i + 1}/{len(shards)} shards, {c['rows_read']:,} rows, {c['rows_read'] / max(el, 1e-9):,.0f} rows/s")
    for b in range(N_BUCKETS):
        flush(b)
        if writers[b] is not None:
            writers[b].close()
    check(all(f.exists() for f in files), "every bucket file was written", checks)
    check(sum(c[f"bucket_rows_{b:x}"] for b in range(N_BUCKETS)) == c["rows_key_versions"] - c["rows_without_key"],
          "bucket rows = rows of the versions own and quoted with a cleaned text", checks)
    return files, c, emo_schema


# --- Pass 2: one row per key, bucket by bucket -----------------------------------------------------

def agg_sql(path: Path) -> str:
    all_score = P_COLS + PC_COLS + PW_COLS
    dp = {name: ", ".join(f"max({x}::DOUBLE) - min({x}::DOUBLE)" for x in cols_)
          for name, cols_ in (("dp", P_COLS), ("dpc", PC_COLS), ("dpw", PW_COLS))}
    nn = {name: ", ".join(f"count({x})" for x in cols_)
          for name, cols_ in (("p", P_COLS), ("pc", PC_COLS), ("pw", PW_COLS))}
    rep_cols = ", ".join(SCORE_COLS)
    return f"""
    WITH r AS (SELECT * FROM read_parquet({sql_str(path)})),
    agg AS (
        SELECT text_key,
               count(*) AS n_rows,
               count(*) FILTER (WHERE version = 'own') AS n_own,
               count(*) FILTER (WHERE version = 'quoted') AS n_quoted,
               count(*) FILTER (WHERE is_clean_error) AS n_clean_error,
               min(n_tokens) AS ntok_min, max(n_tokens) AS ntok_max,
               bool_and(is_empty_after_clean) AS emp_all, bool_or(is_empty_after_clean) AS emp_any,
               min(n_windows) AS nw_min, max(n_windows) AS nw_max,
               min(tok_digest) AS td_min, max(tok_digest) AS td_max,
               least({nn['p']}) AS n_p_min, greatest({nn['p']}) AS n_p_max,
               least({nn['pc']}) AS n_pc_min, greatest({nn['pc']}) AS n_pc_max,
               least({nn['pw']}) AS n_pw_min, greatest({nn['pw']}) AS n_pw_max,
               greatest({dp['dp']}) AS dp, greatest({dp['dpc']}) AS dpc, greatest({dp['dpw']}) AS dpw
        FROM r GROUP BY text_key),
    rep AS (
        SELECT * EXCLUDE (rn) FROM (
            SELECT text_key, row_id AS rep_row_id, version AS rep_version, {rep_cols},
                   row_number() OVER (PARTITION BY text_key ORDER BY row_id, version) AS rn
            FROM r)
        WHERE rn = 1)
    SELECT agg.*, rep.* EXCLUDE (text_key)
    FROM agg JOIN rep USING (text_key)
    ORDER BY text_key
    """


def out_schema(emo_schema: pa.Schema) -> pa.Schema:
    fields = [pa.field("text_key", pa.string()), pa.field("n_tokens", pa.int32()),
              pa.field("is_empty_after_clean", pa.bool_()), pa.field("n_rows", pa.int64()),
              pa.field("rep_row_id", emo_schema.field("row_id").type), pa.field("rep_version", pa.string())]
    fields += [pa.field(col, emo_schema.field(col).type) for col in SCORE_COLS]
    return pa.schema(fields)


def np_col(t: pa.Table, name: str) -> np.ndarray:
    col = t.column(name)
    check(col.null_count == 0, f"pass 2: {name} has no nulls")
    return col.to_numpy()


def pass2(con, files: List[Path], tmp: Path, schema: pa.Schema, c: Counter,
          checks: List[Dict[str, object]]) -> Dict[str, object]:
    writer = pq.ParquetWriter(tmp, schema, compression="zstd", write_statistics=True)
    prev_last: Optional[str] = None
    hist = Counter()
    dp_max = {"p": 0.0, "pc": 0.0, "pw": 0.0}
    dp_over = {k: Counter() for k in dp_max}
    top_nonempty: List[Tuple[int, int]] = []
    empty_keys: List[Tuple[str, int]] = []
    try:
        for b, f in enumerate(files):
            t0 = time.time()
            a = con.execute(agg_sql(f)).to_arrow_table()
            n = a.num_rows
            keys = a.column("text_key")
            where = f"bucket {b:x}"
            check(keys.null_count == 0 and pc.all(pc.match_substring_regex(keys, r"^[0-9a-f]{64}$")).as_py(),
                  f"{where}: keys are 64 lower-case hex characters")
            check(pc.all(pc.starts_with(keys, f"{b:x}")).as_py(), f"{where}: every key starts with {b:x}")
            if n > 1:
                check(pc.all(pc.less(keys.slice(0, n - 1), keys.slice(1))).as_py(),
                      f"{where}: keys strictly increasing")
            if n and prev_last is not None:
                check(prev_last < keys[0].as_py(), f"{where}: first key follows the previous bucket's last key")
            if n:
                prev_last = keys[n - 1].as_py()
            n_rows = np_col(a, "n_rows")
            check(np.array_equal(np_col(a, "ntok_min"), np_col(a, "ntok_max")), f"{where}: n_tokens equal within every key")
            nw_min, nw_max = np_col(a, "nw_min"), np_col(a, "nw_max")
            check(np.array_equal(nw_min, nw_max), f"{where}: n_windows equal within every key")
            emp_all, emp_any = np_col(a, "emp_all"), np_col(a, "emp_any")
            check(np.array_equal(emp_all, emp_any), f"{where}: is_empty_after_clean equal within every key")
            check(np.array_equal(np_col(a, "td_min"), np_col(a, "td_max")),
                  f"{where}: token ids and window plan equal within every key")
            for g in ("p", "pc", "pw"):
                lo, hi = np_col(a, f"n_{g}_min"), np_col(a, f"n_{g}_max")
                check(np.array_equal(lo, hi) and bool(np.all((hi == 0) | (hi == n_rows))),
                      f"{where}: every {g} column null on all rows of a key or on none")
            n_p = np_col(a, "n_p_max")
            check(bool(np.all((n_p == 0) == emp_all)), f"{where}: p null exactly for empty keys")
            n_pc, n_pw = np_col(a, "n_pc_max"), np_col(a, "n_pw_max")
            check(bool(np.all((n_pc > 0) == (nw_min >= 2)) and np.all((n_pw > 0) == (nw_min >= 2))),
                  f"{where}: pc and pw present exactly when n_windows >= 2")
            check(np.array_equal(np_col(a, "n_windows"), nw_min), f"{where}: the representative's n_windows = the key's")
            check(pc.all(pc.is_in(a.column("rep_version"), value_set=pa.array(KEY_VERSIONS))).as_py(),
                  f"{where}: representative version is own or quoted")
            check(bool(np.all(np_col(a, "n_own") + np_col(a, "n_quoted") == n_rows)), f"{where}: n_own + n_quoted = n_rows")
            # counts
            c["distinct_keys"] += n
            c["rows_covered"] += int(n_rows.sum())
            multi = n_rows >= 2
            c["keys_shared"] += int(multi.sum())
            c["rows_in_shared_keys"] += int(n_rows[multi].sum())
            c["keys_own_and_quoted"] += int(((np_col(a, "n_own") > 0) & (np_col(a, "n_quoted") > 0)).sum())
            c["keys_only_quoted"] += int((np_col(a, "n_own") == 0).sum())
            c["clean_error_rows_indexed"] += int(np_col(a, "n_clean_error").sum())
            c["keys_scored"] += int((n_p > 0).sum())
            c["keys_with_windows"] += int((nw_min >= 2).sum())
            c["rows_scored_indexed"] += int(n_rows[n_p > 0].sum())
            for lo_, hi_, lab in ((1, 1, "1"), (2, 2, "2"), (3, 9, "3-9"), (10, 99, "10-99"),
                                  (100, 999, "100-999"), (1000, None, ">=1000")):
                sel = (n_rows >= lo_) & ((n_rows <= hi_) if hi_ is not None else True)
                hist[lab] += int(sel.sum())
                hist[f"rows_{lab}"] += int(n_rows[sel].sum())
            ntok = np_col(a, "ntok_min")
            for j in np.nonzero(emp_all)[0]:
                empty_keys.append((keys[int(j)].as_py(), int(n_rows[j])))
            nonempty = ~emp_all
            if nonempty.any():
                order = np.argsort(-n_rows[nonempty], kind="stable")[:10]
                top_nonempty.extend(zip(n_rows[nonempty][order].tolist(), ntok[nonempty][order].tolist()))
                top_nonempty = sorted(top_nonempty, reverse=True)[:10]
            for g, col in (("p", "dp"), ("pc", "dpc"), ("pw", "dpw")):
                d = a.column(col).to_numpy(zero_copy_only=False)
                d = np.asarray(d, dtype=np.float64)
                d = d[multi & ~np.isnan(d)] if d.size else d
                if d.size:
                    dp_max[g] = max(dp_max[g], float(d.max()))
                    for thr in DP_THRESHOLDS:
                        dp_over[g][thr] += int((d > thr).sum())
                    dp_over[g]["keys_compared"] += int(d.size)
            out = pa.table({
                "text_key": keys, "n_tokens": a.column("ntok_min"), "is_empty_after_clean": a.column("emp_all"),
                "n_rows": a.column("n_rows"), "rep_row_id": a.column("rep_row_id"),
                "rep_version": a.column("rep_version"), **{col: a.column(col) for col in SCORE_COLS}})
            for fld in schema:
                check(out.schema.field(fld.name).type == fld.type,
                      f"{where}: column {fld.name} has type {out.schema.field(fld.name).type}, expected {fld.type}")
            out = out.select(schema.names).cast(schema)
            writer.write_table(out, row_group_size=ROW_GROUP_ROWS)
            log(f"pass 2: bucket {b:x}: {n:,} keys, {int(n_rows.sum()):,} rows, {time.time() - t0:.1f} s")
    finally:
        writer.close()
    return {"n_rows_histogram": dict(hist), "dp_max": dp_max,
            "dp_keys_above": {g: {str(k): v for k, v in cnt.items()} for g, cnt in dp_over.items()},
            "top_nonempty_keys_n_rows_n_tokens": top_nonempty, "empty_keys": empty_keys}


# --- Spot check and build --------------------------------------------------------------------------

def spot_check(con, tmp: Path, files: List[Path], shards: List[Path], emo_file: Path,
               checks: List[Dict[str, object]]) -> Dict[str, object]:
    half = SPOT_KEYS // 2
    con.execute("SET threads=1")          # reservoir samples are reproducible on one thread
    s1 = [r[0] for r in con.execute(
        f"SELECT text_key FROM read_parquet({sql_str(tmp)}) "
        f"USING SAMPLE reservoir({half} ROWS) REPEATABLE ({SPOT_SEEDS[0]}) ORDER BY text_key").fetchall()]
    con.register("spot_s1", pa.table({"text_key": pa.array(s1, type=pa.string())}))
    s2 = [r[0] for r in con.execute(
        f"SELECT text_key FROM (SELECT text_key FROM read_parquet({sql_str(tmp)}) WHERE n_rows >= 2 "
        f"AND text_key NOT IN (SELECT text_key FROM spot_s1)) "
        f"USING SAMPLE reservoir({SPOT_KEYS - half} ROWS) REPEATABLE ({SPOT_SEEDS[1]}) ORDER BY text_key").fetchall()]
    con.execute(f"SET threads={int(config.DUCKDB_THREADS)}")
    sample = sorted(set(s1) | set(s2))
    check(len(sample) == SPOT_KEYS, f"spot check: {len(sample)} distinct keys sampled = {SPOT_KEYS}", checks)
    con.register("spot_keys", pa.table({"text_key": pa.array(sample, type=pa.string())}))
    con.register("shard_paths", pa.table({"shard": pa.array(range(len(shards)), type=pa.int32()),
                                          "shard_path": pa.array([str(s) for s in shards], type=pa.string())}))
    members = con.execute(f"""
        SELECT m.text_key, m.row_id, m.version, p.shard_path, m.rn, m.n_members FROM (
            SELECT b.text_key, b.row_id, b.version, b.shard,
                   row_number() OVER (PARTITION BY b.text_key ORDER BY b.row_id, b.version) AS rn,
                   count(*) OVER (PARTITION BY b.text_key) AS n_members
            FROM read_parquet({sql_list(files)}) b JOIN spot_keys USING (text_key)) m
        JOIN shard_paths p USING (shard)
        WHERE m.rn <= {SPOT_MEMBERS}""").to_arrow_table()
    idx = con.execute(f"""
        SELECT i.* FROM read_parquet({sql_str(tmp)}) i JOIN spot_keys USING (text_key)""").to_arrow_table()
    check(idx.num_rows == SPOT_KEYS, "spot check: every sampled key is in the index once", checks)
    n_rows = dict(zip(idx.column("text_key").to_pylist(), idx.column("n_rows").to_pylist()))
    rep = dict(zip(idx.column("text_key").to_pylist(),
                   zip(idx.column("rep_row_id").to_pylist(), idx.column("rep_version").to_pylist())))
    mk = members.column("text_key").to_pylist()
    first = {(k): (r, v) for k, r, v, rn in zip(mk, members.column("row_id").to_pylist(),
                                                members.column("version").to_pylist(),
                                                members.column("rn").to_pylist()) if rn == 1}
    n_mem = dict(zip(mk, members.column("n_members").to_pylist()))
    check(all(n_mem.get(k) == n_rows[k] for k in sample),
          "spot check: each key's member count in the row map equals n_rows", checks)
    check(all(first.get(k) == rep[k] for k in sample),
          "spot check: the representative is the key's smallest (row_id, version)", checks)
    # read the cleaned text back from the token shards (DuckDB reader) and hash it again
    con.register("spot_members", members.select(["text_key", "row_id", "version", "shard_path"]))
    used = sorted(set(members.column("shard_path").to_pylist()))
    back = con.execute(f"""
        SELECT m.text_key, t.cleaned_text
        FROM spot_members m JOIN read_parquet({sql_list(used)}, filename = true) t
          ON t.filename = m.shard_path AND t.row_id = m.row_id AND t.version = m.version""").to_arrow_table()
    check(back.num_rows == members.num_rows,
          f"spot check: each of the {members.num_rows} member rows found once in its token shard", checks)
    mismatches = sum(text_key(t) != k for k, t in zip(back.column("text_key").to_pylist(),
                                                      back.column("cleaned_text").to_pylist()))
    check(mismatches == 0, f"spot check: sha256 of the cleaned text read back = text_key on all "
          f"{back.num_rows} member rows of {SPOT_KEYS} keys ({mismatches} mismatches)", checks)
    # the representative's stored values
    eq = " AND ".join(f"(e.{x} IS NOT DISTINCT FROM i.{x})" for x in SCORE_COLS + ["n_tokens", "is_empty_after_clean"])
    rows = con.execute(f"""
        SELECT i.text_key, {eq} AS same
        FROM (SELECT * FROM read_parquet({sql_str(tmp)}) JOIN spot_keys USING (text_key)) i
        JOIN read_parquet({sql_str(emo_file)}) e ON e.row_id = i.rep_row_id AND e.version = i.rep_version
        """).fetchall()
    check(len(rows) == SPOT_KEYS and all(r[1] for r in rows),
          f"spot check: the representative's values in the emotion table equal the index row "
          f"({sum(1 for r in rows if r[1])} of {len(rows)} joined rows)", checks)
    return {"keys": SPOT_KEYS, "from_all_keys": len(s1), "from_shared_keys": len(s2), "seeds": list(SPOT_SEEDS),
            "member_rows_rehashed": back.num_rows, "members_per_key_cap": SPOT_MEMBERS,
            "sampled_keys_shared": int(sum(1 for k in sample if n_rows[k] >= 2)),
            "sampled_keys_empty": int(sum(1 for k in sample if k == EMPTY_KEY)),
            "sha256_mismatches": mismatches, "representative_value_mismatches": sum(1 for r in rows if not r[1])}


def tki_build(platform: str, force: bool, keep_work: bool) -> int:
    tdir = E.tokens_dir(platform) / TKI_PARTITION
    emo_file = E.emotion_dir(platform) / f"{TKI_PARTITION}.parquet"
    final = E.emotion_dir(platform).parent / "text_key_index.parquet"
    shards = sorted(tdir.glob("shard-*.parquet"))
    check(len(shards) > 0 and emo_file.exists(), f"{platform}: token shards and the emotion table exist")
    log(f"{platform}: {len(shards)} token shards, emotion table {emo_file.name}; describing inputs (md5)")
    inputs = runner.describe_inputs(list(shards) + [emo_file])
    if not force and runner.is_current(final, inputs, code_fingerprint(), TKI_RULES_SHA):
        log(f"{final.name} is current; nothing to do (use --force)")
        return 0
    checks: List[Dict[str, object]] = []
    in_info = check_inputs(shards, emo_file, inputs, checks)
    config.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="text_key_index_", dir=config.SCRATCH_DIR))
    log(f"scratch directory {work}")
    try:
        files, c, emo_schema = pass1(shards, emo_file, work, checks)
        check(c["rows_read"] == in_info["emotion_rows"], f"rows read {c['rows_read']:,} = emotion table rows", checks)
        check(c["rows_without_key"] == c["missing_rows"], "rows without a key = missing texts", checks)
        check(True, "per shard: emotion rows equal the shard's (row_id, version, n_tokens, flags); n_tokens = "
              "len(input_ids); cleaned_text NULL exactly for missing and '' exactly for empty texts", checks)
        schema = out_schema(emo_schema)
        con = runner.connect(spill=work / "duckdb_spill")
        tmp = runner.tmp_path(final)
        info = pass2(con, files, tmp, schema, c, checks)
        check(True, "per key: n_tokens, n_windows, is_empty_after_clean, token ids and window plan equal on every "
              "row; p/pc/pw null on all rows or none; p null exactly for empty keys; pc/pw exactly for n_windows < 2",
              checks)
        check(True, "keys: 64 lower-case hex characters, strictly increasing through the file (distinct)", checks)
        check(c["rows_covered"] == c["rows_key_versions"] - c["rows_without_key"],
              f"sum(n_rows) {c['rows_covered']:,} = rows of own and quoted with a cleaned text", checks)
        empties = info["empty_keys"]
        if c["empty_rows"]:
            check(len(empties) == 1 and empties[0][0] == EMPTY_KEY and empties[0][1] == c["empty_rows"],
                  f"all {c['empty_rows']:,} empty rows sit on the key of '', the only empty key", checks)
        check(c["clean_error_rows_indexed"] == c["clean_error_rows"], "clean-error rows all indexed (on the empty key)",
              checks)
        md = pq.ParquetFile(tmp).metadata
        back = con.execute(f"SELECT count(*), sum(n_rows)::BIGINT FROM read_parquet({sql_str(tmp)})").fetchone()
        check(md.num_rows == c["distinct_keys"] and back[0] == c["distinct_keys"] and back[1] == c["rows_covered"],
              f"read back: {back[0]:,} rows = distinct keys, sum(n_rows) {back[1]:,} = rows covered", checks)
        check(pq.ParquetFile(tmp).schema_arrow.equals(schema), "read back: schema as specified", checks)
        spot = spot_check(con, tmp, files, shards, emo_file, checks)
        runner.commit(tmp, final)
        counts = {k: int(v) for k, v in sorted(c.items())}
        counts["rows_reusable"] = counts["rows_covered"] - counts["distinct_keys"]
        counts["max_abs_dp_within_key_p"] = info["dp_max"]["p"]
        counts["max_abs_dp_within_key_pc"] = info["dp_max"]["pc"]
        counts["max_abs_dp_within_key_pw"] = info["dp_max"]["pw"]
        runner.write_stats(final, step=STEP_TKI, inputs=inputs, code_sha=code_fingerprint(), rules_sha=TKI_RULES_SHA, con=con,
                           counts=counts, checks=checks, extra={
                               "platform": platform, "partition": TKI_PARTITION, "key_rule": KEY_RULE,
                               "key_versions": list(KEY_VERSIONS), "excluded_versions": ["both"],
                               "representative_rule": REP_RULE, "empty_key": EMPTY_KEY,
                               "columns": schema.names, "labels": list(E.LABELS), "inputs_info": in_info,
                               "n_rows_histogram": info["n_rows_histogram"],
                               "dp_keys_above": info["dp_keys_above"], "dp_thresholds": list(DP_THRESHOLDS),
                               "top_nonempty_keys_n_rows_n_tokens": info["top_nonempty_keys_n_rows_n_tokens"],
                               "empty_keys_n_rows": [e[1] for e in empties], "spot_check": spot})
        con.close()
        log(f"wrote {final} ({counts['distinct_keys']:,} keys, {counts['rows_covered']:,} rows) and its stats file")
    finally:
        if keep_work:
            log(f"scratch kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 0


def main_text_key_index(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="06_emotion_emtract.py text-key-index",
                                 description="text-key index of the Twitter scores (section 4)")
    ap.add_argument("platform", nargs="?", default="twitter", choices=["twitter"])
    ap.add_argument("--force", action="store_true", help="rebuild even if the stats say the output is current")
    ap.add_argument("--keep-work", action="store_true", help="keep the scratch directory (bucket files)")
    args = ap.parse_args(argv)
    try:
        return tki_build(args.platform, args.force, args.keep_work)
    except CheckError as exc:
        log(f"CHECK FAILED: {exc}")
        return 1


# --- Key list for the DLT match (added 2 Oct 2026; the file of 30 Sep was written by a one-off command) ---

TKI_KEY_LIST_ROW_GROUP = 1_000_000


def text_key_list(platform: str = "twitter") -> Dict[str, object]:
    """twitter/emtract/text_key_list.parquet from text_key_index.parquet: column text_key as the 32-byte
    sha256 digests (bytes.fromhex of the index's hex keys), strictly ascending, zstd, row groups of
    1,000,000; its JSON stats file records the source index (bytes, md5), the checks and the key rule."""
    index = E.emotion_dir(platform).parent / "text_key_index.parquet"
    final = index.with_name("text_key_list.parquet")
    check(index.exists(), f"text-key index found: {index}")
    hexkeys = pq.read_table(index, columns=["text_key"]).column("text_key").to_pylist()
    check(all(isinstance(k, str) and len(k) == 64 for k in hexkeys), "every index key is 64 hex characters")
    keys = [bytes.fromhex(k) for k in hexkeys]
    check(all(a < b for a, b in zip(keys[:-1], keys[1:])), "keys strictly ascending (hence distinct)")
    tmp = runner.tmp_path(final)
    pq.write_table(pa.table({"text_key": pa.array(keys, type=pa.binary())}), tmp, compression="zstd",
                   row_group_size=TKI_KEY_LIST_ROW_GROUP)
    back = pq.read_table(tmp).column("text_key").to_pylist()
    check(back == keys, "read back: the same keys in the same order")
    md = pq.ParquetFile(tmp).metadata
    runner.commit(tmp, final)
    idx_stats = json.loads(runner.stats_path(index).read_text()) if runner.stats_path(index).exists() else {}
    idx_md5 = next((o.get("md5") for o in idx_stats.get("outputs", []) if Path(str(o.get("path", ""))).name == index.name), None)
    rec = {"file": final.name, "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rows": len(keys), "bytes": final.stat().st_size, "md5": runner.file_md5(final), "column": "text_key",
           "arrow_type": "binary", "encoding": "32-byte sha256 digest, bytes.fromhex(text_key) of text_key_index.parquet",
           "sort": "ascending bytes (= ascending hex)", "compression": "zstd", "row_groups": md.num_row_groups,
           "source": {"path": str(index), "bytes": index.stat().st_size,
                      "md5": f"{idx_md5} (from its stats.json)" if idx_md5 else runner.file_md5(index)},
           "key_rule": KEY_RULE + "; versions own + quoted",
           "checks": {"all_len_32": all(len(k) == 32 for k in keys), "distinct_eq_rows": len(set(keys)) == len(keys),
                      "strictly_sorted": True, "set_equal_to_index": len(keys) == len(hexkeys)}}
    Path(str(final) + ".stats.json").write_text(json.dumps(rec, indent=1))
    log(f"wrote {final} ({len(keys):,} keys)")
    return rec


def main_text_key_list(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="06_emotion_emtract.py text-key-list",
                                 description="the text keys of the Twitter index as a sorted binary list (section 4)")
    ap.add_argument("platform", nargs="?", default="twitter", choices=["twitter"])
    args = ap.parse_args(argv)
    text_key_list(args.platform)
    return 0


# ==================================================================================================
# 5. The DLT corpus (`dlt ...`): reuse the Twitter scores of identical texts, score the rest
# ==================================================================================================
# The DLT-Tweets corpus (ExponentialScience/DLT-Tweets; the D17 event arm) goes through the same
# EmTract method as the main corpora. Its rows are prepared by 04_series_dlt.py (`units`: one row per
# in-window DLT row with row_id = NN * 100,000,000 + r, the step-02 coin evidence, the step-04 series
# columns and the step-03 per-post bot flags). Nothing is deleted, de-duplicated or language-filtered.
#
# Shared text key: text_key = sha256 hex of EmTract's cleaned_text encoded as UTF-8 (the same key as the
# Twitter index, section 4). Identical cleaned text gives identical model inputs and therefore identical
# scores, so a DLT row whose key occurs in the main-Twitter key list reuses the stored score and is not
# scored again; the other distinct texts are tokenized once each and scored with `pack`/`score`/`collect`.
#
# Subcommands (run from code/)
# ----------------------------
#   dlt clean   (emtract-ref env) EmTract's clean_text once on every in-window text (flags as stage 1),
#               text_key = sha256 of the cleaned text's UTF-8 bytes -> dlt/emtract/cleaned/shard-NNNNN.parquet
#               (shards of at most DLT_CLEAN_SHARD_ROWS rows of the units file; emoji memo as stage 1)
#   dlt match   (DuckDB) --key-list PATH  (text_key_list.parquet: one column text_key, 32-byte binary)
#           -> data/pipeline/dlt/emtract/map/dlt_text_key_map.parquet
#                one row per DLT units row: row_id, text_key (hex, NULL for a missing text),
#                in_main (TRUE when the key is in the main-Twitter key list), is_missing_text,
#                is_empty_after_clean, is_clean_error
#           -> data/pipeline/dlt/emtract/unmatched/unmatched_texts.parquet
#                one row per distinct text_key NOT in the main list: text_key, cleaned_text, the clean
#                flags, n_rows (DLT rows with this key), first_row_id; ordered by text_key
#   dlt tokenize  (emtract-ref env) --workers N
#           -> data/pipeline/dlt/emtract/tokens/own/shard-NNNNN.parquet
#                stage 1's token schema: row_id (= text_key, string), version ('own'), cleaned_text,
#                is_missing_text, is_empty_after_clean, is_clean_error, clean_error, n_tokens (int32),
#                input_ids (list<uint16>, uncut, no special tokens), window_starts (list<int32>);
#                stage 1's per-shard checks (n_tokens = len(ids), ids fit uint16 and the vocabulary,
#                valid window plans, flags agree, 1% + 50 long spot checks against the tokenizer's own
#                truncation, read-back of count and id sum)
#   dlt verify (emtract-ref env) --n N
#           exactness check: for N unmatched keys (seeded sample) the raw DLT text of the key's first
#           row goes through stage 1's own path (emtract_method.stage1_rows, platform twitter, no
#           parent text); cleaned_text, n_tokens, input_ids and window_starts must equal the token
#           shards, and sha256(cleaned_text) must equal the key.
#
# Inputs (all under data/pipeline/)
#   dlt/emtract/map/dlt_text_key_map.parquet   `dlt match`: one row per DLT in-window row
#                                              (row_id, text_key, in_main, is_missing_text, is_empty_after_clean,
#                                              is_clean_error)
#   dlt/emtract/tokens/own/shard-NNNNN.parquet `dlt tokenize`: one row per UNMATCHED distinct cleaned
#                                              text (row_id = text_key, version 'own'), stage 1's schema
#   dlt/emtract/scores/<batch>.parquet         `collect` (platform 'dlt'), one row per model input
#   twitter/emtract/text_key_index.parquet     `text-key-index`: one row per main-Twitter text key with
#                                              n_tokens, n_windows, p_*, pc_*, pw_*
#
# Subcommands (env thesis; `dlt refscore` in env emtract-ref)
#   dlt completeness   every finished DLT token shard packed in exactly one batch; per batch packed inputs = map rows
#                  = scored rows; every (shard, shard_row, window) scored exactly once, no orphan, per text the
#                  number of scores = k(n_tokens) with windows 0..k-1; probabilities sum to 1 within 1e-5.
#                  -> RUN_DIR/completeness_dlt.json
#   dlt merge-keys     the merge's merge_partition on dlt/own, with its output written to
#                  dlt/emtract/emotion_keys/own.parquet (one row per unmatched text key; row_id = text_key)
#   dlt build          dlt/emtract/emotion/dlt_emotion.parquet: one row per DLT in-window row_id with version 'own',
#                  text_key, in_main, the three flags, n_tokens, n_windows, p_*, pc_*, pw_* (names and types as
#                  the main emotion tables) and score_source:
#                      'reused_ia'   key in the main-Twitter list: the index row's values, unchanged
#                      'scored_dlt'  key not in the main list: the values of emotion_keys/own.parquet
#                      NULL          missing text or empty after cleaning (no model call; null scores, as `merge`)
#                  Checks on the temporary file (runner standard), then commit and stats; also writes a seeded
#                  sample of 300 non-empty row_ids for `dlt refscore` (row ids and stored p only, no text).
#   dlt refscore       env emtract-ref: EmTract's own Model('stocktwits').predict() on the ORIGINAL text of the
#                  sampled rows (dlt/units/dlt_units.parquet, column `text`), compared with the stored p_*.
#                  -> dlt/emtract/emotion/dlt_emotion_refscore.json
#
# Usage
# -----
#     python pipeline/06_emotion_emtract.py dlt clean [--workers N] [--start-method fork|spawn]   (emtract-ref)
#     python pipeline/06_emotion_emtract.py dlt match --key-list data/pipeline/twitter/emtract/text_key_list.parquet
#     python pipeline/06_emotion_emtract.py dlt tokenize --workers N                               (emtract-ref)
#     python pipeline/06_emotion_emtract.py dlt verify --n 20000                                   (emtract-ref)
#     python pipeline/06_emotion_emtract.py pack dlt --batch-id dlt-0001 --max-inputs 5000000   (then score, collect)
#     python pipeline/06_emotion_emtract.py dlt completeness
#     python pipeline/06_emotion_emtract.py dlt merge-keys
#     python pipeline/06_emotion_emtract.py dlt build
#     python pipeline/06_emotion_emtract.py dlt refscore --threads 4                               (emtract-ref)


# --- Paths and settings of the DLT arm -------------------------------------------------------------
DLT_CLEAN_SHARD_ROWS = 100_000            # clean shards: row groups of the units file cut into pieces <= this
DLT_DIR = config.OUT_DIR / "dlt"
DLT_CLEANED = DLT_DIR / "emtract" / "cleaned"
DLT_MAP = DLT_DIR / "emtract" / "map" / "dlt_text_key_map.parquet"
DLT_UNMATCHED = DLT_DIR / "emtract" / "unmatched" / "unmatched_texts.parquet"
DLT_TOKENS = DLT_DIR / "emtract" / "tokens" / "own"
DLT_UNITS = DLT_DIR / "units" / "dlt_units.parquet"
DLT_UNMATCHED_RG_ROWS = 50_000          # row group size of the unmatched table = one token shard
DLT_PLATFORM = "dlt"
DLT_PARTITION = "own"
DLT_KEYS_DIR = DLT_DIR / "emtract" / "emotion_keys"
DLT_KEYS = DLT_KEYS_DIR / f"{DLT_PARTITION}.parquet"
DLT_EMOTION_FILE = DLT_DIR / "emtract" / "emotion" / "dlt_emotion.parquet"
DLT_SAMPLE = DLT_DIR / "emtract" / "emotion" / "dlt_emotion_refsample.json"
DLT_REFSCORE = DLT_DIR / "emtract" / "emotion" / "dlt_emotion_refscore.json"
TEXT_KEY_INDEX = config.OUT_DIR / "twitter" / "emtract" / "text_key_index.parquet"
DLT_COMPLETENESS = E.RUN_DIR / "completeness_dlt.json"
DLT_N_SAMPLE = 300
DLT_CHUNK_ROWS = 2_000_000
DLT_SEED = 20260930


# --- dlt clean: EmTract's clean_text once per in-window text; text_key = sha256(cleaned UTF-8) -----

def _dlt_clean_shard(spec: Dict, ctx: Dict) -> Dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    final = Path(ctx["shard_file"])
    pf = pq.ParquetFile(spec["source"])
    tab = pf.read_row_group(spec["row_group"], columns=["row_id", "text"])
    tab = tab.slice(spec["rg_offset"], spec["rg_stop"] - spec["rg_offset"])
    rid = tab.column("row_id").to_pylist()
    txt = tab.column("text").to_pylist()
    checks: List[Dict[str, object]] = []
    cache: Dict[str, Tuple] = {}
    cleaned, miss, emp, err, emsg, keys = [], [], [], [], [], []
    err_types: Dict[str, int] = {}
    for t in txt:
        if t is None:
            r = (None, True, False, False, None)
        else:
            r = cache.get(t)
            if r is None:
                r = E.clean_with_flags(t)          # EmTract's clean_text, once per distinct text
                cache[t] = r
        cleaned.append(r[0]); miss.append(r[1]); emp.append(r[2]); err.append(r[3]); emsg.append(r[4])
        keys.append(text_key(r[0]))
        if r[3]:
            et = r[4].split(":", 1)[0]
            err_types[et] = err_types.get(et, 0) + 1
    n = len(rid)
    check(len(cleaned) == n, "one cleaned row per input row", checks)
    flags_ok = all((c is None) == m and (not e or c == "") and (not r_ or e) and ((c is None or c == "") == (m or e))
                   and ((k is None) == m) for c, m, e, r_, k in zip(cleaned, miss, emp, err, keys))
    check(flags_ok, "flags agree with cleaned_text and text_key", checks)
    table = pa.table({
        "row_id": pa.array(rid, type=pa.int64()),
        "cleaned_text": pa.array(cleaned, type=pa.string()),
        "is_missing_text": pa.array(miss, type=pa.bool_()),
        "is_empty_after_clean": pa.array(emp, type=pa.bool_()),
        "is_clean_error": pa.array(err, type=pa.bool_()),
        "clean_error": pa.array(emsg, type=pa.string()),
        "text_key": pa.array(keys, type=pa.string()),
    })
    tmp = final.parent / "_tmp" / final.stem / final.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    pq.write_table(table, str(tmp), compression="zstd")
    back = pq.read_table(str(tmp))
    check(back.num_rows == n and back.column("row_id").to_pylist() == rid, "written row ids = input row ids, in order", checks)
    ok_keys = all(text_key(c) == k for c, k in zip(back.column("cleaned_text").to_pylist(), back.column("text_key").to_pylist()))
    check(ok_keys, "text_key = sha256(cleaned_text) on the read-back file", checks)
    runner.commit(tmp, final)
    try:
        tmp.parent.rmdir()
    except OSError:
        pass
    secs = time.time() - t0
    try:
        import resource
        mr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        maxrss_mb = mr / 2**20 if sys.platform == "darwin" else mr / 1024
    except Exception:  # noqa: BLE001
        maxrss_mb = None
    counts = {"rows": n, "missing_text": sum(miss), "empty_after_clean": sum(emp), "clean_error": sum(err),
              "clean_error_types": err_types, "texts_cleaned_distinct": len(cache),
              "distinct_text_keys": len({k for k in keys if k is not None}), "seconds": round(secs, 2),
              "texts_per_s": round(n / secs, 1) if secs else None, "bytes": final.stat().st_size}
    runner.write_stats(final, step=STEP_DLT + " clean", inputs=ctx["inputs"], code_sha=ctx["code_sha"], rules_sha=ctx["rules_sha"],
                       con=None, counts=counts, checks=checks,
                       extra={"shard": {k: spec[k] for k in ("source", "row_group", "rg_offset", "rg_stop", "row_start",
                                                             "row_stop", "shard_index")},
                              "emoji_memo": E.emoji_memo_active(), "library_versions": ctx["versions"],
                              "worker_maxrss_mb": maxrss_mb, "text_key": "sha256 hex of cleaned_text.encode('utf-8')"})
    return {"rows": n, "seconds": secs, "shard_file": str(final), "empty": sum(emp), "clean_error": sum(err),
            "maxrss_mb": maxrss_mb}


def _dlt_clean_job(item):
    spec, ctx = item
    return spec["shard_index"], _dlt_clean_shard(spec, ctx)


def _dlt_clean_init(emoji_memo: bool) -> None:
    try:
        os.nice(10)
    except OSError:
        pass
    E.load_cleaning()
    if emoji_memo:
        E.install_emoji_memo()
    if E.emoji_memo_active() != emoji_memo:
        raise RuntimeError("emoji memo state differs from the requested one")


def dlt_clean(args) -> int:
    import multiprocessing as mp
    src = DLT_DIR / "units" / "dlt_units.parquet"
    check(src.exists(), f"units table found: {src}")
    out_dir = DLT_DIR / "emtract" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = E.plan_shards("twitter", "cleaned", src, args.limit_rows, DLT_CLEAN_SHARD_ROWS)
    tok_dir = E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT
    rules_sha, versions = rules_fingerprint(tok_dir)
    code_sha = code_fingerprint()
    log(f"clean: {len(specs)} shards planned; hashing {src.name}")
    inputs = runner.describe_inputs([src])
    todo = []
    for s in specs:
        final = out_dir / f"shard-{s['shard_index']:05d}.parquet"
        if shard_is_current(final, s, inputs, code_sha, rules_sha):
            continue
        todo.append((s, {"shard_file": str(final), "inputs": inputs, "code_sha": code_sha, "rules_sha": rules_sha,
                         "versions": versions}))
    log(f"clean: {len(todo)} to do, {len(specs) - len(todo)} current; {args.workers} workers ({args.start_method})")
    t0 = time.time()
    done_rows = 0
    total_rows = sum(s[0]["row_stop"] - s[0]["row_start"] for s in todo)
    status = out_dir / "_status.json"
    if todo:
        memo = not args.no_emoji_memo
        if args.start_method == "fork":
            import gc
            _dlt_clean_init(memo)                       # load once in the parent; workers share it copy-on-write
            gc.collect()
            gc.freeze()
            pool = mp.get_context("fork").Pool(args.workers)
        else:
            pool = mp.get_context("spawn").Pool(args.workers, initializer=_dlt_clean_init, initargs=(memo,))
        try:
            for i, (_, res) in enumerate(pool.imap_unordered(_dlt_clean_job, todo), 1):
                done_rows += res["rows"]
                el = time.time() - t0
                rate = done_rows / el if el else 0.0
                runner.write_json_atomic(status, {
                    "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "shards_done_this_run": i, "shards_todo": len(todo), "rows_done_this_run": done_rows,
                    "rows_todo": total_rows, "rows_per_s": round(rate, 1),
                    "eta_min": round((total_rows - done_rows) / rate / 60, 1) if rate else None,
                    "load_average": list(os.getloadavg()), "workers": args.workers})
                if i % 10 == 0 or i == len(todo):
                    log(f"clean {i}/{len(todo)} shards, {done_rows:,} rows, {rate:.0f} rows/s, "
                        f"eta {((total_rows - done_rows) / rate / 60) if rate else float('nan'):.0f} min")
        finally:
            pool.close()
            pool.join()
    tot = dlt_summarize_clean(specs, out_dir, src)
    log(f"clean done in {time.time() - t0:.0f} s: {tot['rows']:,} rows, {tot['empty_after_clean']:,} empty, "
        f"{tot['clean_error']:,} clean errors")
    return 0


def dlt_summarize_clean(specs: List[Dict], out_dir: Path, src: Path) -> Dict:
    """Totals over every clean shard and the coverage check: every planned units row exactly once."""
    import pyarrow.parquet as pq
    checks: List[Dict[str, object]] = []
    tot: Dict[str, object] = {"rows": 0, "missing_text": 0, "empty_after_clean": 0, "clean_error": 0, "shards": 0,
                              "bytes": 0, "seconds_sum": 0.0}
    err_types: Dict[str, int] = {}
    memo = set()
    for s in specs:
        f = out_dir / f"shard-{s['shard_index']:05d}.parquet"
        rec = json.loads(runner.stats_path(f).read_text())
        c = rec["counts"]
        check(all(x.get("passed") for x in rec["checks"]), f"{f.name}: all checks passed", checks)
        check(c["rows"] == s["row_stop"] - s["row_start"], f"{f.name}: rows = planned range", checks)
        for k in ("rows", "missing_text", "empty_after_clean", "clean_error", "bytes"):
            tot[k] += c[k]
        tot["seconds_sum"] += c["seconds"]
        tot["shards"] += 1
        for k, v in c["clean_error_types"].items():
            err_types[k] = err_types.get(k, 0) + v
        memo.add(rec.get("emoji_memo"))
    n_units = pq.ParquetFile(str(src)).metadata.num_rows
    planned = sum(s["row_stop"] - s["row_start"] for s in specs)
    check(tot["rows"] == planned, f"clean shards cover the planned rows ({tot['rows']:,} of {planned:,})", checks)
    contiguous = (not specs or specs[0]["row_start"] == 0) and all(a["row_stop"] == b["row_start"] for a, b in zip(specs[:-1], specs[1:]))
    check(contiguous, "shard ranges are contiguous from row 0", checks)
    tot.update({"clean_error_types": err_types, "units_rows": n_units, "covers_all_units_rows": tot["rows"] == n_units,
                "emoji_memo": sorted(str(m) for m in memo)})
    runner.write_json_atomic(out_dir / "_summary.json", {
        "step": STEP_DLT + " clean summary", "counts": tot, "checks": checks,
        "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return tot


# --- dlt match: DLT text keys against the main-Twitter key list ------------------------------------

def dlt_cleaned_files() -> List[str]:
    return sorted(glob.glob(str(DLT_CLEANED / "shard-*.parquet")))


def dlt_match(args) -> int:
    import pyarrow.parquet as pq
    files = dlt_cleaned_files()
    check(len(files) > 0, f"cleaned shards found in {DLT_CLEANED}")
    summ = json.loads((DLT_CLEANED / "_summary.json").read_text())
    check(summ["counts"]["covers_all_units_rows"], "cleaned shards cover every units row (dlt clean summary)")
    key_list = Path(args.key_list)
    kl_md5 = runner.file_md5(key_list)
    if args.key_list_md5:
        check(kl_md5 == args.key_list_md5, f"key list md5 = the given md5 ({kl_md5})")
    t0 = time.time()
    con = runner.connect()
    checks: List[Dict[str, object]] = []
    con.execute(f"CREATE TEMP VIEW C AS SELECT * FROM read_parquet({files!r})")
    con.execute(f"CREATE TEMP VIEW M AS SELECT text_key AS k FROM read_parquet('{key_list}')")
    n_m, n_m_d, bad_len = con.execute("SELECT count(*), count(DISTINCT k), count_if(octet_length(k) <> 32) FROM M").fetchone()
    check(bad_len == 0, "every main key is 32 bytes", checks)
    check(n_m == n_m_d, f"main keys distinct ({n_m:,})", checks)
    # distinct DLT keys with their clean flags (identical key => identical cleaned text => identical flags)
    con.execute("""CREATE TEMP TABLE K AS
        SELECT text_key, unhex(text_key) AS k, any_value(cleaned_text) AS cleaned_text,
               count(DISTINCT cleaned_text) AS n_texts, bool_or(is_empty_after_clean) AS is_empty_after_clean,
               bool_and(is_empty_after_clean) AS all_empty, bool_or(is_clean_error) AS is_clean_error,
               any_value(clean_error) AS clean_error, count(*) AS n_rows, min(row_id) AS first_row_id
        FROM C WHERE text_key IS NOT NULL GROUP BY text_key""")
    n_k, multi, mixed = con.execute("SELECT count(*), count_if(n_texts <> 1), count_if(is_empty_after_clean <> all_empty) FROM K").fetchone()
    check(multi == 0, "one cleaned text per key (no hash collision)", checks)
    check(mixed == 0, "the empty flag is constant within a key", checks)
    con.execute("CREATE TEMP TABLE KM AS SELECT K.*, (M.k IS NOT NULL) AS in_main FROM K LEFT JOIN M USING (k)")
    check(con.execute("SELECT count(*) FROM KM").fetchone()[0] == n_k, "the join keeps one row per DLT key", checks)
    # map: one row per units row
    DLT_MAP.parent.mkdir(parents=True, exist_ok=True)
    tmp = runner.tmp_path(DLT_MAP)
    con.execute(f"""COPY (SELECT C.row_id, C.text_key, coalesce(KM.in_main, FALSE) AS in_main, C.is_missing_text,
                              C.is_empty_after_clean, C.is_clean_error
                       FROM C LEFT JOIN KM USING (text_key) ORDER BY C.row_id)
                    TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)""")
    n_rows_c = con.execute("SELECT count(*) FROM C").fetchone()[0]
    n_map, n_map_ids = con.execute(f"SELECT count(*), count(DISTINCT row_id) FROM read_parquet('{tmp}')").fetchone()
    n_units = pq.ParquetFile(str(DLT_UNITS)).metadata.num_rows
    check(n_map == n_map_ids == n_rows_c == n_units, f"map: one row per units row ({n_map:,})", checks)
    runner.commit(tmp, DLT_MAP)
    # unmatched distinct texts
    DLT_UNMATCHED.parent.mkdir(parents=True, exist_ok=True)
    tmp2 = runner.tmp_path(DLT_UNMATCHED)
    con.execute(f"""COPY (SELECT text_key, cleaned_text, is_empty_after_clean, is_clean_error, clean_error,
                                n_rows, first_row_id FROM KM WHERE NOT in_main ORDER BY text_key)
                    TO '{tmp2}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {DLT_UNMATCHED_RG_ROWS})""")
    q = lambda s: con.execute(s).fetchone()  # noqa: E731
    n_un = q(f"SELECT count(*) FROM read_parquet('{tmp2}')")[0]
    counts = {
        "units_rows": int(n_units), "main_keys": int(n_m), "key_list_md5": kl_md5, "key_list_bytes": key_list.stat().st_size,
        "rows_missing_text": int(q("SELECT count_if(text_key IS NULL) FROM C")[0]),
        "rows_with_key": int(q("SELECT count_if(text_key IS NOT NULL) FROM C")[0]),
        "rows_empty_after_clean": int(q("SELECT count_if(is_empty_after_clean) FROM C")[0]),
        "rows_clean_error": int(q("SELECT count_if(is_clean_error) FROM C")[0]),
        "unique_text_keys": int(n_k),
        "unique_text_keys_nonempty": int(q("SELECT count_if(NOT is_empty_after_clean) FROM K")[0]),
        "matched_keys": int(q("SELECT count_if(in_main) FROM KM")[0]),
        "matched_keys_nonempty": int(q("SELECT count_if(in_main AND NOT is_empty_after_clean) FROM KM")[0]),
        "matched_rows": int(q("SELECT coalesce(sum(n_rows), 0) FROM KM WHERE in_main")[0]),
        "matched_rows_nonempty": int(q("SELECT coalesce(sum(n_rows), 0) FROM KM WHERE in_main AND NOT is_empty_after_clean")[0]),
        "unmatched_keys": int(n_un),
        "unmatched_keys_nonempty": int(q("SELECT count_if(NOT in_main AND NOT is_empty_after_clean) FROM KM")[0]),
        "unmatched_rows": int(q("SELECT coalesce(sum(n_rows), 0) FROM KM WHERE NOT in_main")[0]),
        "empty_key_in_main": bool(q(f"SELECT count(*) > 0 FROM M WHERE k = unhex('{hashlib.sha256(b'').hexdigest()}')")[0]),
    }
    check(counts["matched_keys"] + counts["unmatched_keys"] == n_k, "matched + unmatched keys = DLT keys", checks)
    check(counts["matched_rows"] + counts["unmatched_rows"] == counts["rows_with_key"], "matched + unmatched rows = rows with a key", checks)
    runner.commit(tmp2, DLT_UNMATCHED)
    inputs = runner.describe_inputs([key_list] + [Path(f) for f in files], with_md5=False)
    for f in (DLT_MAP, DLT_UNMATCHED):
        runner.write_stats(f, step=STEP_DLT + " match", inputs=inputs, code_sha=code_fingerprint(),
                           rules_sha="-", con=con, counts=counts, checks=checks,
                           extra={"seconds": round(time.time() - t0, 1),
                                  "text_key": "sha256 hex of cleaned_text.encode('utf-8'); main list stores bytes.fromhex(text_key)"})
    con.close()
    log(f"match: {counts['matched_keys']:,} of {n_k:,} keys in the main list ({counts['matched_rows']:,} rows); "
        f"{n_un:,} unmatched keys; {time.time() - t0:.0f} s")
    return 0


# --- dlt tokenize: stage 1's tokens and window plans for each unmatched distinct cleaned text ------

_TOK = None                             # per-process tokenizer (workers of `dlt tokenize` and `news stage1`)


def _dlt_load_tokens(tok_dir: str, memo: bool):
    E.load_cleaning()
    if memo:
        E.install_emoji_memo()
    return E.Tokens(Path(tok_dir))


def dlt_token_shard(spec: Dict, ctx: Dict) -> Dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    tokens = _TOK
    t0 = time.time()
    final = Path(ctx["shard_file"])
    tab = pq.ParquetFile(str(DLT_UNMATCHED)).read_row_group(spec["row_group"],
                                                         columns=["text_key", "cleaned_text", "is_empty_after_clean",
                                                                  "is_clean_error", "clean_error"])
    keys = tab.column("text_key").to_pylist()
    ct = tab.column("cleaned_text").to_pylist()
    emp = tab.column("is_empty_after_clean").to_pylist()
    err = tab.column("is_clean_error").to_pylist()
    emsg = tab.column("clean_error").to_pylist()
    n = len(keys)
    checks: List[Dict[str, object]] = []
    check(all(hashlib.sha256(c.encode("utf-8")).hexdigest() == k for c, k in zip(ct, keys)),
          "text_key = sha256(cleaned_text) for every row", checks)
    uniq = [c for c in ct if c]
    ids_of = dict(zip(uniq, tokens.encode_uncut(uniq)))
    ws = tokens.word_start
    ids, ntok, starts = [], [], []
    for c in ct:
        i = ids_of.get(c, []) if c else []
        ids.append(i)
        k = len(i)
        ntok.append(k)
        starts.append(E.window_starts(k, [ws[x] for x in i]) if k > E.WINDOW else ([0] if k else []))
    miss = [False] * n
    check(all(k == len(i) for k, i in zip(ntok, ids)), "n_tokens = len(input_ids)", checks)
    max_id = max((max(i) for i in ids if i), default=-1)
    check(max_id < 65536 and max_id < tokens.n_vocab, "every token id fits uint16 and the vocabulary", checks)
    bad_plan = 0
    for k, s in zip(ntok, starts):
        try:
            E.check_window_plan(k, s)
        except ValueError:
            bad_plan += 1
    check(bad_plan == 0, "every window plan is valid", checks)
    flags_ok = all((not e or (c == "" and k == 0)) and (not r or e) and ((c == "") == e)
                   for c, e, r, k in zip(ct, emp, err, ntok))
    check(flags_ok, "flags agree with cleaned_text and n_tokens", checks)
    rng = random.Random(f"dlt/own/{spec['shard_index']}")
    with_tok = [i for i, k in enumerate(ntok) if k > 0]
    kk = min(len(with_tok), max(1, math.ceil(SPOT_SHARE * len(with_tok)))) if with_tok else 0
    sample = set(rng.sample(with_tok, kk)) if kk else set()
    long_rows = [i for i in with_tok if ntok[i] > E.WINDOW]
    sample |= set(rng.sample(long_rows, min(SPOT_LONG_MAX, len(long_rows))))
    sample = sorted(sample)
    trunc = tokens.encode_truncated([ct[i] for i in sample]) if sample else []
    check(all(t == E.primary_input(ids[i], tokens.cls_id, tokens.sep_id) for i, t in zip(sample, trunc)),
          f"[CLS]+ids[:62]+[SEP] = tokenizer truncation on {len(sample)} spot-checked rows", checks)
    table = pa.table({
        "row_id": pa.array(keys, type=pa.string()),
        "version": pa.array(["own"] * n, type=pa.string()),
        "cleaned_text": pa.array(ct, type=pa.string()),
        "is_missing_text": pa.array(miss, type=pa.bool_()),
        "is_empty_after_clean": pa.array(emp, type=pa.bool_()),
        "is_clean_error": pa.array(err, type=pa.bool_()),
        "clean_error": pa.array(emsg, type=pa.string()),
        "n_tokens": pa.array(ntok, type=pa.int32()),
        "input_ids": pa.array(ids, type=pa.list_(pa.uint16())),
        "window_starts": pa.array(starts, type=pa.list_(pa.int32())),
    })
    tmp = final.parent / "_tmp" / final.stem / final.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    pq.write_table(table, str(tmp), compression="zstd")
    back = pq.read_table(str(tmp), columns=["row_id", "input_ids"])
    flat = back.column("input_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    check(back.num_rows == n and back.column("row_id").to_pylist() == keys, "written rows = rows built, in order", checks)
    check(len(flat) == sum(ntok) and int(np.asarray(flat, dtype=np.int64).sum()) == int(sum(sum(i) for i in ids)),
          "written ids = built ids (count and sum)", checks)
    runner.commit(tmp, final)
    try:
        tmp.parent.rmdir()
    except OSError:
        pass
    secs = time.time() - t0
    counts = {"rows": n, "empty_after_clean": sum(emp), "clean_error": sum(err),
              "over_62_tokens": sum(1 for k in ntok if k > E.WINDOW), "model_inputs": sum(len(s) for s in starts),
              "sum_n_tokens": sum(ntok), "max_n_tokens": max(ntok) if ntok else 0, "max_token_id": max_id,
              "spot_checked": len(sample), "seconds": round(secs, 2), "bytes": final.stat().st_size}
    runner.write_stats(final, step=STEP_DLT + " tokenize", inputs=ctx["inputs"], code_sha=ctx["code_sha"], rules_sha=ctx["rules_sha"],
                       con=None, counts=counts, checks=checks,
                       extra={"platform": "dlt", "partition": "own",
                              "shard": {"source": str(DLT_UNMATCHED), "row_group": spec["row_group"], "shard_index": spec["shard_index"],
                                        "row_start": spec["row_start"], "row_stop": spec["row_stop"]},
                              "row_id_is": "text_key (sha256 hex of cleaned_text)", "emoji_memo": E.emoji_memo_active(),
                              "library_versions": ctx["versions"],
                              "special_ids": {"cls": tokens.cls_id, "sep": tokens.sep_id, "pad": tokens.pad_id}})
    return counts


def _dlt_token_job(item):
    spec, ctx = item
    try:
        os.nice(10)
    except OSError:
        pass
    return spec["shard_index"], dlt_token_shard(spec, ctx)


def dlt_tokenize(args) -> int:
    global _TOK
    import multiprocessing as mp
    import gc
    import pyarrow.parquet as pq
    tok_dir = E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT
    rules_sha, versions = rules_fingerprint(tok_dir)
    code_sha = code_fingerprint()
    md = pq.ParquetFile(str(DLT_UNMATCHED)).metadata
    specs, g = [], 0
    for rg in range(md.num_row_groups):
        k = md.row_group(rg).num_rows
        specs.append({"shard_index": rg, "row_group": rg, "row_start": g, "row_stop": g + k})
        g += k
    inputs = runner.describe_inputs([DLT_UNMATCHED])
    DLT_TOKENS.mkdir(parents=True, exist_ok=True)
    todo = []
    for s in specs:
        final = DLT_TOKENS / f"shard-{s['shard_index']:05d}.parquet"
        if runner.is_current(final, inputs, code_sha, rules_sha):
            continue
        todo.append((s, {"shard_file": str(final), "inputs": inputs, "code_sha": code_sha, "rules_sha": rules_sha,
                         "versions": versions}))
    log(f"tokens: {len(specs)} shards, {len(todo)} to do, {args.workers} workers (fork)")
    t0 = time.time()
    _TOK = _dlt_load_tokens(str(tok_dir), True)
    gc.collect()
    gc.freeze()
    done = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (_, c) in enumerate(pool.imap_unordered(_dlt_token_job, todo), 1):
            done += c["rows"]
            if i % 20 == 0 or i == len(todo):
                log(f"tokens {i}/{len(todo)} shards, {done:,} texts, {done / (time.time() - t0):.0f} texts/s")
    # summary over every shard
    checks: List[Dict[str, object]] = []
    tot = {"shards": 0, "rows": 0, "empty_after_clean": 0, "clean_error": 0, "over_62_tokens": 0, "model_inputs": 0,
           "sum_n_tokens": 0, "bytes": 0}
    for s in specs:
        f = DLT_TOKENS / f"shard-{s['shard_index']:05d}.parquet"
        rec = json.loads(runner.stats_path(f).read_text())
        check(all(x.get("passed") for x in rec["checks"]), f"{f.name}: all checks passed", checks)
        check(rec["counts"]["rows"] == s["row_stop"] - s["row_start"], f"{f.name}: rows = planned", checks)
        tot["shards"] += 1
        for k in ("rows", "empty_after_clean", "clean_error", "over_62_tokens", "model_inputs", "sum_n_tokens", "bytes"):
            tot[k] += rec["counts"][k]
    check(tot["rows"] == md.num_rows, f"token shards cover every unmatched key ({tot['rows']:,})", checks)
    tot["rows_nonempty"] = tot["rows"] - tot["empty_after_clean"]
    runner.write_json_atomic(DLT_TOKENS.parent / "_tokens_summary.json", {
        "step": STEP_DLT + " tokenize summary", "counts": tot, "checks": checks, "rules_sha": rules_sha, "code_sha": code_sha,
        "library_versions": versions, "seconds": round(time.time() - t0, 1), "workers": args.workers,
        "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    log(f"tokens done: {tot['rows']:,} texts, {tot['model_inputs']:,} model inputs, {time.time() - t0:.0f} s")
    return 0


# --- dlt verify: the token shards equal stage 1's own path on the raw text -------------------------

def dlt_verify(args) -> int:
    t0 = time.time()
    tok = _dlt_load_tokens(str(E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT), True)
    con = runner.connect()
    files = sorted(glob.glob(str(DLT_TOKENS / "shard-*.parquet")))
    n_all = con.execute(f"SELECT count(*) FROM read_parquet('{DLT_UNMATCHED}')").fetchone()[0]
    rng = random.Random(20260930)
    pick = sorted(rng.sample(range(n_all), min(args.n, n_all)))
    con.execute("CREATE TEMP TABLE P AS SELECT unnest(?::BIGINT[]) AS i", [pick])
    rows = con.execute(f"""
        WITH u AS (SELECT row_number() OVER (ORDER BY text_key) - 1 AS i, text_key, first_row_id
                   FROM read_parquet('{DLT_UNMATCHED}')),
             s AS (SELECT u.text_key, u.first_row_id FROM u JOIN P USING (i))
        SELECT s.text_key, x.text, t.cleaned_text, t.n_tokens, t.input_ids, t.window_starts
        FROM s JOIN read_parquet('{DLT_UNITS}') x ON x.row_id = s.first_row_id
        JOIN read_parquet({files!r}) t ON t.row_id = s.text_key
        ORDER BY s.text_key""").fetchall()
    con.close()
    check(len(rows) == len(pick), f"sampled keys found in units and token shards ({len(rows):,})")
    out, _ = E.stage1_rows("twitter", {"text": [r[1] for r in rows], "parent_text": [None] * len(rows)}, tok)
    diff = {"cleaned_text": 0, "n_tokens": 0, "input_ids": 0, "window_starts": 0, "key": 0}
    for j, r in enumerate(rows):
        diff["cleaned_text"] += out["cleaned_text"][j] != r[2]
        diff["n_tokens"] += out["n_tokens"][j] != r[3]
        diff["input_ids"] += list(out["input_ids"][j]) != list(r[4])
        diff["window_starts"] += list(out["window_starts"][j]) != list(r[5])
        diff["key"] += hashlib.sha256((out["cleaned_text"][j] or "").encode("utf-8")).hexdigest() != r[0]
    res = {"step": STEP_DLT + " verify", "sampled": len(rows), "differences": diff, "seed": 20260930,
           "all_equal": all(v == 0 for v in diff.values()), "seconds": round(time.time() - t0, 1),
           "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    runner.write_json_atomic(DLT_TOKENS.parent / "_verify06.json", res)
    log(f"verify06: {res}")
    return 0 if res["all_equal"] else 1


# --- dlt completeness, merge-keys, build, refscore -------------------------------------------------

def _q(p) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def dlt_completeness(args) -> int:
    from pipeline import runner
    from pipeline.runner import check
    import pyarrow.parquet as pq
    t0 = time.time()
    checks: List[Dict[str, object]] = []
    tdir = E.tokens_dir(DLT_PLATFORM)
    shards = sorted((tdir / DLT_PARTITION).glob("shard-*.parquet"))
    rels = [s.relative_to(tdir).as_posix() for s in shards]
    finished = []
    for s, r in zip(shards, rels):
        st = s.with_name(s.name + ".stats.json")
        ok = st.exists()
        if ok:
            rec = json.loads(st.read_text())
            ok = all(c.get("passed") for c in rec.get("checks", []))
        finished.append(ok)
    check(all(finished), f"all {len(shards)} token shards finished (stats present, checks passed)", checks)
    inroot = E.PACKED_INPUTS_DIR / DLT_PLATFORM
    maps = sorted(inroot.glob("*.map.parquet"))
    batches = [m.name[:-len(".map.parquet")] for m in maps]
    per_batch = {}
    shard_batches: Dict[str, List[str]] = {}
    for b, m in zip(batches, maps):
        ms = json.loads((inroot / f"{b}.map.parquet.stats.json").read_text())
        n_map = pq.ParquetFile(m).metadata.num_rows
        n_in = pq.ParquetFile(inroot / f"{b}.parquet").metadata.num_rows
        sc = E.scores_dir(DLT_PLATFORM) / f"{b}.parquet"
        n_sc = pq.ParquetFile(sc).metadata.num_rows if sc.exists() else None
        scs = json.loads(sc.with_name(sc.name + ".stats.json").read_text()) if sc.exists() else {}
        sc_ok = bool(scs) and all(c.get("passed") for c in scs.get("checks", []))
        per_batch[b] = {"inputs": n_in, "map_rows": n_map, "scored_rows": n_sc, "collect_checks_passed": sc_ok}
        for r in pq.read_table(m, columns=["shard"]).column("shard").unique().to_pylist():
            shard_batches.setdefault(r, []).append(b)
        check(n_in == n_map == n_sc and sc_ok, f"batch {b}: inputs {n_in} = map {n_map} = scores {n_sc}, collect passed", checks)
    not_packed = [r for r in rels if r not in shard_batches]
    twice = [r for r, bs in shard_batches.items() if len(bs) > 1]
    orphan_shards = [r for r in shard_batches if r not in set(rels)]
    check(not not_packed, f"every token shard packed ({len(not_packed)} not packed)", checks)
    check(not twice, f"no shard packed twice ({len(twice)})", checks)
    check(not orphan_shards, f"no packed shard without token shard ({len(orphan_shards)})", checks)
    con = runner.connect(memory="4GB")
    try:
        tok_files = [str(s) for s in shards]
        sc_files = [str(E.scores_dir(DLT_PLATFORM) / f"{b}.parquet") for b in batches]
        con.execute(f"""CREATE TEMP TABLE T AS SELECT regexp_extract(filename, '(own/shard-[0-9]+\\.parquet)$', 1) AS shard,
                         CAST(file_row_number AS INTEGER) AS shard_row, n_tokens, is_missing_text, is_empty_after_clean,
                         len(window_starts) AS k, len(input_ids) AS n_ids
                         FROM read_parquet({tok_files!r}, filename=true, file_row_number=true)""")
        psum = " + ".join(f"CAST({c} AS DOUBLE)" for c in P_COLS)
        con.execute(f"""CREATE TEMP TABLE G AS SELECT shard, shard_row, count(*) AS n, count(DISTINCT "window") AS nd,
                         min("window") AS wmin, max("window") AS wmax, max(abs({psum} - 1)) AS dev
                         FROM read_parquet({sc_files!r}) GROUP BY shard, shard_row""")
        k_exp = "CASE WHEN n_tokens <= 0 THEN 0 WHEN n_tokens <= 62 THEN 1 ELSE CAST(ceil((n_tokens - 16) / 46.0) AS INTEGER) END"
        r = con.execute(f"""SELECT count(*) AS texts,
              count_if(T.k <> {k_exp}) AS bad_k,
              count_if(T.n_ids <> T.n_tokens) AS bad_len,
              count_if(T.k = 0 AND NOT (T.is_missing_text OR T.is_empty_after_clean)) AS flag_gap,
              count_if(T.k > 0 AND G.n IS NULL) AS missing_scores,
              count_if(T.k = 0 AND G.n IS NOT NULL) AS scores_for_flagged,
              count_if(G.n IS NOT NULL AND (G.n <> T.k OR G.nd <> T.k OR G.wmin <> 0 OR G.wmax <> T.k - 1)) AS wrong_window_sets,
              coalesce(sum(T.k), 0) AS expected_inputs,
              coalesce(max(G.dev), 0) AS max_dev,
              count_if(T.k = 0) AS flagged_no_call
              FROM T LEFT JOIN G USING (shard, shard_row)""").fetchone()
        names = ["texts", "bad_k", "bad_len", "flag_gap", "missing_scores", "scores_for_flagged", "wrong_window_sets",
                 "expected_inputs", "max_dev", "flagged_no_call"]
        res = dict(zip(names, r))
        orphans = con.execute("SELECT count(*) FROM G LEFT JOIN T USING (shard, shard_row) WHERE T.shard IS NULL").fetchone()[0]
        n_scores, n_distinct = con.execute(f"""SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT shard, shard_row, "window"
                                              FROM read_parquet({sc_files!r}))) FROM read_parquet({sc_files!r})""").fetchone()
    finally:
        con.close()
    res.update({"orphan_score_rows": int(orphans), "score_rows": int(n_scores), "distinct_score_keys": int(n_distinct)})
    for k in ("bad_k", "bad_len", "flag_gap", "missing_scores", "scores_for_flagged", "wrong_window_sets"):
        check(res[k] == 0, f"{k} = 0 ({res[k]})", checks)
    check(res["orphan_score_rows"] == 0, "no score for a (shard, row) without a token row", checks)
    check(n_scores == n_distinct == res["expected_inputs"] == sum(v["inputs"] for v in per_batch.values()),
          f"scored exactly once: rows {n_scores} = distinct {n_distinct} = expected {res['expected_inputs']} = packed", checks)
    check(res["max_dev"] <= PROB_SUM_TOL, f"probabilities sum to 1 within {PROB_SUM_TOL} (max {res['max_dev']:.2e})", checks)
    out = {"step": STEP_DLT + " completeness", "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "token_shards": len(shards), "batches": per_batch, "counts": res, "checks": checks,
           "ok": all(c["passed"] for c in checks), "seconds": round(time.time() - t0, 1)}
    runner.write_json_atomic(DLT_COMPLETENESS, out)
    print(json.dumps({k: out[k] for k in ("token_shards", "counts", "ok", "seconds")}, default=str))
    return 0


def dlt_merge_keys(args) -> int:
    empty = load_empty_score()
    where, score_rows = index_scores(DLT_PLATFORM)
    tdir = E.tokens_dir(DLT_PLATFORM)
    orphans = [s for s in score_rows if not (tdir / s).exists()]
    check(not orphans, f"dlt: {len(orphans)} shards in the score files have no token shard")
    res = merge_partition(DLT_PLATFORM, DLT_PARTITION, where, score_rows, empty, args.force, out_dir=DLT_KEYS_DIR)
    print(json.dumps(res, default=str))
    return 0


def dlt_build(args) -> int:
    from pipeline import runner
    from pipeline.runner import check, log
    import pyarrow.parquet as pq
    t0 = time.time()
    checks: List[Dict[str, object]] = []
    for p in (DLT_MAP, TEXT_KEY_INDEX, DLT_KEYS):
        check(p.exists(), f"input exists: {p}")
    ks = json.loads(DLT_KEYS.with_name(DLT_KEYS.name + ".stats.json").read_text())
    check(all(c.get("passed") for c in ks["checks"]), "emotion_keys stats: every check passed", checks)
    ixs = json.loads(TEXT_KEY_INDEX.with_name(TEXT_KEY_INDEX.name + ".stats.json").read_text())
    check(all(c.get("passed") for c in ixs["checks"]), "text_key_index stats: every check passed", checks)
    inputs = runner.describe_inputs([DLT_MAP, TEXT_KEY_INDEX, DLT_KEYS])
    con = runner.connect(memory="4GB")
    tmp = runner.tmp_path(DLT_EMOTION_FILE)
    try:
        con.execute(f"CREATE TEMP VIEW MP AS SELECT * FROM read_parquet({_q(DLT_MAP)})")
        con.execute(f"CREATE TEMP VIEW IX AS SELECT * FROM read_parquet({_q(TEXT_KEY_INDEX)})")
        con.execute(f"CREATE TEMP VIEW KY AS SELECT * FROM read_parquet({_q(DLT_KEYS)})")
        n_map, n_ids = con.execute("SELECT count(*), count(DISTINCT row_id) FROM MP").fetchone()
        check(n_map == n_ids, f"map row_id unique ({n_map:,})", checks)
        n_ky, n_ky_d, n_ky_ver = con.execute("SELECT count(*), count(DISTINCT row_id), count_if(version <> 'own') FROM KY").fetchone()
        check(n_ky == n_ky_d and n_ky_ver == 0, f"emotion_keys: one 'own' row per text key ({n_ky:,})", checks)
        def pick(c):
            return f"CASE WHEN m.in_main THEN i.{c} ELSE k.{c} END AS {c}"
        cols = ",\n".join(pick(c) for c in P_COLS + PC_COLS + PW_COLS)
        # The index restricted to the keys the DLT rows use (small), then the join in row_id chunks so that each
        # hash table is built on the chunk (a few million narrow rows) and the 16.7 M-row key table is only probed.
        con.execute("""CREATE TEMP TABLE IXS AS SELECT IX.* FROM IX
                       WHERE IX.text_key IN (SELECT DISTINCT text_key FROM MP WHERE in_main)""")
        lo, hi = con.execute("SELECT min(row_id), max(row_id) FROM MP").fetchone()
        bounds = [r[0] for r in con.execute(f"""SELECT row_id FROM (SELECT row_id, row_number() OVER (ORDER BY row_id) - 1 AS i
                                                FROM MP) WHERE i % {DLT_CHUNK_ROWS} = 0 ORDER BY row_id""").fetchall()]
        bounds.append(hi + 1)
        import pyarrow as pa
        writer = None
        try:
            for c0, c1 in zip(bounds[:-1], bounds[1:]):
                part = con.execute(f"""
                    SELECT m.row_id, 'own' AS version, m.text_key, m.in_main,
                       CAST(CASE WHEN m.in_main THEN i.n_tokens WHEN m.text_key IS NULL THEN 0 ELSE k.n_tokens END AS INTEGER) AS n_tokens,
                       m.is_missing_text, m.is_empty_after_clean, m.is_clean_error,
                       CAST(CASE WHEN m.in_main THEN i.n_windows WHEN m.text_key IS NULL THEN 0 ELSE k.n_windows END AS SMALLINT) AS n_windows,
                       {cols},
                       CASE WHEN m.is_missing_text OR m.is_empty_after_clean THEN NULL
                            WHEN m.in_main THEN 'reused_ia' ELSE 'scored_dlt' END AS score_source,
                       (CASE WHEN m.in_main THEN i.text_key ELSE k.row_id END) IS NOT NULL AS _found,
                       CASE WHEN m.in_main THEN i.is_empty_after_clean ELSE k.is_empty_after_clean END AS _src_empty,
                       (i.text_key IS NOT NULL AND k.row_id IS NOT NULL) AS _both
                    FROM (SELECT * FROM MP WHERE row_id >= {c0} AND row_id < {c1}) m
                    LEFT JOIN IXS i ON i.text_key = m.text_key
                    LEFT JOIN KY k ON k.row_id = m.text_key
                    ORDER BY m.row_id""").arrow()
                if hasattr(part, "read_all"):
                    part = part.read_all()
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp) + ".wide", part.schema, compression="zstd", write_statistics=True)
                writer.write_table(part, row_group_size=1_000_000)
                log(f"build: rows {c0}..{c1 - 1}: {part.num_rows:,}")
        finally:
            if writer is not None:
                writer.close()
        W = f"read_parquet({_q(str(tmp) + '.wide')})"
        k_exp = "CASE WHEN n_tokens <= 0 THEN 0 WHEN n_tokens <= 62 THEN 1 ELSE CAST(ceil((n_tokens - 16) / 46.0) AS INTEGER) END"
        noscore = "(is_missing_text OR is_empty_after_clean)"
        q = lambda s: con.execute(s).fetchone()  # noqa: E731
        n_w, n_wd = q(f"SELECT count(*), count(DISTINCT row_id) FROM {W}")
        check(n_w == n_map == n_wd, f"rows {n_w:,} = map rows {n_map:,}, row_id unique", checks)
        n_units = pq.ParquetFile(str(DLT_UNITS)).metadata.num_rows if DLT_UNITS.exists() else None
        if n_units is not None:
            check(n_w == n_units, f"rows = DLT in-window units rows ({n_units:,})", checks)
        r = q(f"""SELECT count_if(text_key IS NOT NULL AND NOT _found),
                         count_if(text_key IS NULL AND NOT is_missing_text),
                         count_if(text_key IS NOT NULL AND _src_empty <> is_empty_after_clean),
                         count_if(NOT {noscore} AND p_neutral IS NULL),
                         count_if({noscore} AND p_neutral IS NOT NULL),
                         count_if((n_windows >= 2) <> (pc_neutral IS NOT NULL) OR (n_windows >= 2) <> (pw_neutral IS NOT NULL)),
                         count_if(n_windows <> {k_exp}),
                         count_if(NOT {noscore} AND n_windows = 0),
                         count_if(is_clean_error AND NOT is_empty_after_clean),
                         count_if(_both),
                         count_if(in_main AND score_source = 'scored_dlt') + count_if(NOT in_main AND score_source = 'reused_ia')
                  FROM {W}""")
        names = ["key_not_found", "no_key_not_missing", "empty_flag_mismatch", "nonempty_without_score",
                 "empty_with_score", "pc_pw_null_mismatch", "n_windows_ne_k", "nonempty_zero_windows", "clean_error_not_empty",
                 "key_in_index_and_dlt_keys", "source_vs_in_main"]
        for nm, v in zip(names, r):
            check(v == 0, f"{nm} = 0 ({v})", checks)
        for colset, tag in ((P_COLS, "p"), (PC_COLS, "pc"), (PW_COLS, "pw")):
            expr = " + ".join(f"CAST({c} AS DOUBLE)" for c in colset)
            dev = q(f"SELECT coalesce(max(abs({expr} - 1)), 0) FROM {W} WHERE {colset[0]} IS NOT NULL")[0]
            check(dev <= PROB_SUM_TOL, f"{tag}_* sums to 1 within {PROB_SUM_TOL} (max {dev:.2e})", checks)
        # reuse check: 300 random reused rows equal the index exactly; 300 random scored rows equal emotion_keys
        allc = P_COLS + PC_COLS + PW_COLS + ["n_tokens", "n_windows"]
        eq = " AND ".join(f"(w.{c} IS NOT DISTINCT FROM s.{c})" for c in allc)
        n_re, ok_re = q(f"""SELECT count(*), count_if({eq}) FROM
                            (SELECT * FROM (SELECT * FROM {W} WHERE score_source = 'reused_ia') USING SAMPLE reservoir({DLT_N_SAMPLE} ROWS) REPEATABLE ({DLT_SEED % 100000})) w
                            JOIN IX s ON s.text_key = w.text_key""")
        n_sc, ok_sc = q(f"""SELECT count(*), count_if({eq}) FROM
                            (SELECT * FROM (SELECT * FROM {W} WHERE score_source = 'scored_dlt') USING SAMPLE reservoir({DLT_N_SAMPLE} ROWS) REPEATABLE ({DLT_SEED % 100000})) w
                            JOIN KY s ON s.row_id = w.text_key""")
        check(n_re == ok_re, f"reuse spot check: {ok_re} of {n_re} sampled reused rows equal the index", checks)
        check(n_sc == ok_sc, f"scored spot check: {ok_sc} of {n_sc} sampled scored rows equal emotion_keys", checks)
        counts = {}
        for nm, expr in (("rows", "count(*)"), ("rows_reused", "count_if(score_source = 'reused_ia')"),
                         ("rows_scored", "count_if(score_source = 'scored_dlt')"),
                         ("rows_empty_after_clean", "count_if(is_empty_after_clean)"),
                         ("rows_missing_text", "count_if(is_missing_text)"), ("rows_clean_error", "count_if(is_clean_error)"),
                         ("rows_in_main", "count_if(in_main)"), ("rows_no_score", f"count_if({noscore})"),
                         ("rows_long", "count_if(n_windows >= 2)"), ("distinct_keys", "count(DISTINCT text_key)"),
                         ("distinct_keys_reused", "count(DISTINCT CASE WHEN score_source = 'reused_ia' THEN text_key END)"),
                         ("distinct_keys_scored", "count(DISTINCT CASE WHEN score_source = 'scored_dlt' THEN text_key END)")):
            counts[nm] = int(q(f"SELECT {expr} FROM {W}")[0])
        check(counts["rows_reused"] + counts["rows_scored"] + counts["rows_no_score"] == counts["rows"],
              "reused + scored + no-score rows = rows", checks)
        n_ky_scored = q("SELECT count_if(NOT (is_missing_text OR is_empty_after_clean)) FROM KY")[0]
        check(counts["distinct_keys_scored"] == n_ky_scored,
              f"every non-empty emotion_keys key is used ({counts['distinct_keys_scored']:,} = {n_ky_scored:,})", checks)
        final_cols = ["row_id", "version", "text_key", "in_main", "n_tokens", "is_missing_text", "is_empty_after_clean",
                      "is_clean_error", "n_windows"] + P_COLS + PC_COLS + PW_COLS + ["score_source"]
        con.execute(f"COPY (SELECT {', '.join(final_cols)} FROM {W}) TO {_q(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)")
        n_f = pq.ParquetFile(str(tmp)).metadata.num_rows
        check(n_f == n_w, f"final file read back: {n_f:,} rows", checks)
        # types equal the main emotion table's
        main = config.OUT_DIR / "twitter" / "emtract" / "emotion" / "units.parquet"
        ms = pq.ParquetFile(str(main)).schema_arrow
        ds_ = pq.ParquetFile(str(tmp)).schema_arrow
        same = [c for c in ms.names if c in ds_.names and c != "row_id"]
        bad = [c for c in same if ms.field(c).type != ds_.field(c).type]
        check(not bad, f"column types equal the main emotion table's ({len(same)} shared columns; bad {bad})", checks)
        # sample for refscore: non-empty rows, half reused, half scored (row ids and stored p only)
        samp = con.execute(f"""SELECT row_id, score_source, {', '.join(P_COLS)} FROM (
              (SELECT * FROM (SELECT * FROM {W} WHERE score_source = 'reused_ia') USING SAMPLE reservoir({DLT_N_SAMPLE // 2} ROWS) REPEATABLE (7))
              UNION ALL
              (SELECT * FROM (SELECT * FROM {W} WHERE score_source = 'scored_dlt') USING SAMPLE reservoir({DLT_N_SAMPLE - DLT_N_SAMPLE // 2} ROWS) REPEATABLE (7)))
              ORDER BY row_id""").fetchall()
        per_coin = None
        coin_cols = []
        if DLT_UNITS.exists():
            ucols = pq.ParquetFile(str(DLT_UNITS)).schema_arrow.names
            coin_cols = [c for c in ("in_btc", "in_eth", "in_bnb", "in_xrp", "in_doge", "in_market") if c in ucols]
            if coin_cols:
                sel = ", ".join(f"count_if(u.{c}), count_if(u.{c} AND w.score_source = 'reused_ia'), "
                                f"count_if(u.{c} AND w.score_source = 'scored_dlt'), count_if(u.{c} AND w.score_source IS NULL)"
                                for c in coin_cols)
                vals = con.execute(f"SELECT {sel} FROM read_parquet({_q(DLT_UNITS)}) u JOIN {W} w USING (row_id)").fetchone()
                per_coin = {c: dict(zip(("rows", "reused", "scored", "no_score"), map(int, vals[4 * i:4 * i + 4])))
                            for i, c in enumerate(coin_cols)}
        os.remove(str(tmp) + ".wide")
        runner.commit(tmp, DLT_EMOTION_FILE)
        runner.write_stats(DLT_EMOTION_FILE, step=STEP_DLT + " build", inputs=inputs, code_sha=code_fingerprint(),
                           rules_sha="-", con=con, counts=counts, checks=checks, extra={
                               "seconds": round(time.time() - t0, 1), "labels": list(LABELS),
                               "score_source": {"reused_ia": "text_key in the main-Twitter list: text_key_index values unchanged",
                                                "scored_dlt": "unmatched key scored by pack/score/collect and merge-keys on dlt/own",
                                                "null": "missing text or empty after cleaning: no model call, null scores"},
                               "per_coin": per_coin, "per_coin_note": "units flags in_<coin> / in_market (overlapping series membership)"})
        runner.write_json_atomic(DLT_SAMPLE, {"seed": 7, "n": len(samp), "labels": list(LABELS),
                                          "rows": [{"row_id": s[0], "score_source": s[1], "p": list(s[2:])} for s in samp]})
    finally:
        con.close()
    log(f"build: {counts['rows']:,} rows ({counts['rows_reused']:,} reused, {counts['rows_scored']:,} scored, "
        f"{counts['rows_no_score']:,} without score) -> {DLT_EMOTION_FILE}; {time.time() - t0:.0f} s")
    print(json.dumps({"counts": counts, "per_coin": per_coin, "checks_passed": all(c["passed"] for c in checks)}, default=str))
    return 0


def dlt_refscore(args) -> int:
    import numpy as np
    import pyarrow.dataset as pads
    samp = json.loads(DLT_SAMPLE.read_text())
    ids = [r["row_id"] for r in samp["rows"]]
    tab = pads.dataset(str(DLT_UNITS)).to_table(columns=["row_id", "text"], filter=pads.field("row_id").isin(ids))
    text_of = dict(zip(tab.column("row_id").to_pylist(), tab.column("text").to_pylist()))
    assert len(text_of) == len(ids), f"found {len(text_of)} of {len(ids)} sampled rows in the units table"
    tmp = Path(tempfile.mkdtemp(prefix="emtract_ref_"))
    cwd = os.getcwd()
    try:
        (tmp / "build" / "models").mkdir(parents=True)
        for variant in ("twitter", "stocktwits"):
            os.symlink(E.EMTRACT_MODEL_DIR / variant, tmp / "build" / "models" / f"emotion-{variant}")
        os.symlink(E.EMTRACT_CODE_DIR / "emtract", tmp / "emtract")
        os.chdir(tmp)
        sys.path.insert(0, str(tmp))
        import torch
        import transformers
        torch.set_num_threads(args.threads)
        from emtract import model as emtract_model  # EmTract's code, unchanged
        t0 = time.time()
        df = emtract_model.Model(E.MODEL_VARIANT).predict([text_of[i] for i in ids])
        secs = time.time() - t0
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    ref = np.array([[float(df[lab].iloc[j]) for lab in LABELS] for j in range(len(ids))])
    got = np.array([r["p"] for r in samp["rows"]], dtype=float)
    d = np.abs(ref - got)
    src = np.array([r["score_source"] for r in samp["rows"]])
    top_eq = ref.argmax(1) == got.argmax(1)
    out = {"method": f"EmTract Model('{E.MODEL_VARIANT}').predict(original DLT text), upstream {E.EMTRACT_UPSTREAM_COMMIT[:7]}",
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__},
           "n": len(ids), "seconds": round(secs, 1), "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
           "top_equal": int(top_eq.sum()), "top_equal_all": bool(top_eq.all()),
           "by_source": {s: {"n": int((src == s).sum()), "max_abs_dp": float(d[src == s].max()) if (src == s).any() else None,
                             "top_equal": int(top_eq[src == s].sum())} for s in ("reused_ia", "scored_dlt")}}
    t = DLT_REFSCORE.with_name(DLT_REFSCORE.name + ".tmp")
    t.write_text(json.dumps(out, indent=1))
    os.replace(t, DLT_REFSCORE)
    print(json.dumps(out))
    return 0


def main_dlt(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="06_emotion_emtract.py dlt", description="the DLT corpus (section 5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clean", help="EmTract's clean_text once per in-window DLT text")
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    c.add_argument("--start-method", choices=("fork", "spawn"),
                   default="fork" if sys.platform.startswith("linux") else "spawn")
    c.add_argument("--no-emoji-memo", action="store_true")
    c.add_argument("--limit-rows", type=int, default=None, help="only the first N units rows (tests)")
    m = sub.add_parser("match", help="text keys against the main-Twitter key list")
    m.add_argument("--key-list", required=True)
    m.add_argument("--key-list-md5", default=None)
    t = sub.add_parser("tokenize", help="tokens of the unmatched distinct texts")
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    v = sub.add_parser("verify", help="token shards against stage 1's own path on the raw text")
    v.add_argument("--n", type=int, default=20000)
    sub.add_parser("completeness", help="every DLT input scored exactly once")
    k = sub.add_parser("merge-keys", help="the merge on dlt/own, written to emotion_keys/")
    k.add_argument("--force", action="store_true")
    sub.add_parser("build", help="one row per DLT row: reused or new scores")
    r = sub.add_parser("refscore", help="EmTract's predict() on a sample (env emtract-ref)")
    r.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    log(f"{STEP_DLT} {args.cmd}: out {DLT_DIR}")
    return {"clean": dlt_clean, "match": dlt_match, "tokenize": dlt_tokenize, "verify": dlt_verify,
            "completeness": dlt_completeness, "merge-keys": dlt_merge_keys, "build": dlt_build,
            "refscore": dlt_refscore}[args.cmd](args)


# ==================================================================================================
# 6. RavenPack news headlines (`news ...`): the same method on the news control variable
# ==================================================================================================
# EmTract scores for the RavenPack news headlines (control-variable input; added 30 Sep 2026). The
# headlines go through the functions of emtract_method.py (stage 1: stage1_rows via the 'stocktwits'
# text version, i.e. the text as it is, one 'own' version per text; clean_with_flags, the emoji memo,
# Tokens, window_starts, model_inputs), of section 2 (special_token_ids, plan_batches, pad_batch,
# load_model, softmax_f32, check_window_starts, sha256_file) and of section 3 (coverage_weights,
# n_windows_for, window_plan_error, softmax64).
#
# So each headline goes through exactly the social-media method (MASTERPLAN_NOV15 §7, decisions of
# 29 Sep 2026): EmTract's vendored clean_text once -> tokenizers 0.10.3 uncut -> stocktwits model,
# fp32. p_* = the primary score ([CLS] + first 62 wordpieces + [SEP], what predict() gives the model);
# for > 62 wordpieces, k = ceil((n-16)/46) windows of 62 with overlap >= 16, pc_* = equal-weight
# mean of the k window probability vectors, pw_* = coverage-weighted mean (sensitivity check, as in
# `merge`). Empty-after-clean texts (clean errors included) get no model call and null scores.
#
# Unit: one UNIQUE non-empty headline string, keyed by headline_md5 = md5 hex of its UTF-8 bytes
# (MASTERPLAN D10: md5/sha256, never DuckDB hash()). A mapping table carries every source row
# (row_idx = 0-based data-line order of the CSV) with its non-text columns and headline_md5.
#
# Subcommands (from code/, env emtract-ref):
#   news prepare   CSV -> news_row_headline_map.parquet, news_unique_headlines.parquet, profile json
#   news stage1    unique headlines -> news_headline_tokens.parquet   (<= --workers processes, spawn)
#   news score     tokens -> news_headline_emotion.parquet + .stats.json  (--device cpu|mps)
#   news refcheck  a random sample: predict()-equivalent path (clean_text -> tokenizer with truncation and
#             padding to 64 -> model -> scipy softmax) against the stored p_*
#
# Input    data/control_var/wrds_ravenpack_crypto_2019_2023.csv (WRDS RavenPack extract; licensed, not in
#          the repository)
# Output   data/pipeline/ravenpack/emtract/ (THESIS_NEWS_EMTRACT_OUT moves it): news_row_headline_map.parquet,
#          news_unique_headlines.parquet, news_headline_tokens.parquet, news_headline_emotion.parquet
#          (+ .stats.json), news_profile.json, news_refcheck.json
#
# Usage
# -----
#     python pipeline/06_emotion_emtract.py news prepare [--csv FILE]                     (emtract-ref)
#     python pipeline/06_emotion_emtract.py news stage1 --workers 6                       (emtract-ref)
#     python pipeline/06_emotion_emtract.py news score --device mps --threads 6           (thesis; cpu also works)
#     python pipeline/06_emotion_emtract.py news refcheck --n 2000 --device cpu           (emtract-ref)
NEWS_CSV_DEFAULT = E.CODE_DIR / "data" / "control_var" / "wrds_ravenpack_crypto_2019_2023.csv"
NEWS_DIR = Path(os.environ.get("THESIS_NEWS_EMTRACT_OUT", str(E.OUT_DIR / "ravenpack" / "emtract")))   # tests point this at a temp folder
NEWS_MAP_FILE = NEWS_DIR / "news_row_headline_map.parquet"
NEWS_UNIQ_FILE = NEWS_DIR / "news_unique_headlines.parquet"
NEWS_TOK_FILE = NEWS_DIR / "news_headline_tokens.parquet"
NEWS_EMO_FILE = NEWS_DIR / "news_headline_emotion.parquet"
NEWS_PROFILE_FILE = NEWS_DIR / "news_profile.json"
NEWS_REF_FILE = NEWS_DIR / "news_refcheck.json"
NEWS_NUMERIC = ("relevance", "event_sentiment_score", "event_relevance", "event_similarity_days", "css", "nip")
NEWS_CHUNK = 2000


# --- Helpers ---------------------------------------------------------------------------------------

def _write_parquet(table: pa.Table, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, final)


def _write_json(obj, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, final)


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def headline_key(text: Optional[str]) -> Optional[str]:
    """md5 hex of the exact UTF-8 bytes; None for a NULL or empty headline."""
    if text is None or text == "":
        return None
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == "darwin" else r / 1024


# --- news prepare: CSV -> row map and unique headlines ---------------------------------------------

def news_prepare(csv: Path) -> Dict:
    t0 = time.time()
    tbl = pacsv.read_csv(str(csv), convert_options=pacsv.ConvertOptions(
        column_types={}, strings_can_be_null=True, auto_dict_encode=False,
        include_columns=None), read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 24))
    # force every column to string first (no inference surprises), then type the numerics
    tbl = pa.table({c: tbl.column(c).cast(pa.string()) for c in tbl.column_names})
    n = tbl.num_rows
    heads = tbl.column("headline").to_pylist()
    keys = [headline_key(h) for h in heads]
    cols = {"row_idx": pa.array(np.arange(n, dtype=np.int64))}
    for c in tbl.column_names:
        if c == "headline":
            continue
        col = tbl.column(c)
        if c in NEWS_NUMERIC:
            col = pc.cast(col, pa.float64())
        cols[c] = col
    import pandas as pd
    ts = pd.to_datetime(tbl.column("timestamp_utc").to_pandas(), utc=True, errors="raise")
    cols["ts_utc"] = pa.array(ts.values.astype("datetime64[ms]"), type=pa.timestamp("ms", tz="UTC"))
    cols["headline_md5"] = pa.array(keys, type=pa.string())
    mp_tbl = pa.table(cols)
    _write_parquet(mp_tbl, NEWS_MAP_FILE)

    # unique headlines (sorted by key), with row and entity counts
    per_key: Dict[str, List] = {}
    ent = tbl.column("entity_name").to_pylist()
    for h, k, e in zip(heads, keys, ent):
        if k is None:
            continue
        r = per_key.get(k)
        if r is None:
            per_key[k] = [h, 1, {e}]
        else:
            if r[0] != h:
                raise RuntimeError("md5 collision between two different headlines")
            r[1] += 1
            r[2].add(e)
    uk = sorted(per_key)
    uniq = pa.table({"headline_md5": pa.array(uk), "headline": pa.array([per_key[k][0] for k in uk]),
                     "n_rows": pa.array([per_key[k][1] for k in uk], type=pa.int32()),
                     "n_entities": pa.array([len(per_key[k][2]) for k in uk], type=pa.int16()),
                     "entities": pa.array(["|".join(sorted(x or "" for x in per_key[k][2])) for k in uk])})
    _write_parquet(uniq, NEWS_UNIQ_FILE)

    # profile (no text)
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4; SET TimeZone='UTC'")
    con.register("m", mp_tbl)
    con.register("u", uniq)
    q = lambda s: con.execute(s).fetchall()
    prof = {
        "source_file": str(csv), "source_bytes": csv.stat().st_size, "source_md5": _md5_file(csv),
        "n_rows": n,
        "rows_null_or_empty_headline": int(sum(k is None for k in keys)),
        "n_unique_headlines": len(uk),
        "rows_per_entity": q("SELECT entity_name, rp_entity_id, count(*) FROM m GROUP BY 1,2 ORDER BY 3 DESC"),
        "rows_per_year": q("SELECT year(ts_utc), count(*) FROM m GROUP BY 1 ORDER BY 1"),
        "rows_per_entity_year": q("SELECT entity_name, year(ts_utc), count(*) FROM m GROUP BY 1,2 ORDER BY 1,2"),
        "news_type": q("SELECT news_type, count(*), count(DISTINCT headline_md5) FROM m GROUP BY 1 ORDER BY 2 DESC"),
        "unique_headlines_per_entity": q("SELECT entity_name, count(DISTINCT headline_md5) FROM m GROUP BY 1 ORDER BY 2 DESC"),
        "rows_per_headline_hist": q("SELECT CASE WHEN n_rows>=10 THEN '10+' ELSE CAST(n_rows AS VARCHAR) END b, count(*), sum(n_rows) FROM u GROUP BY 1 ORDER BY min(n_rows)"),
        "headlines_multi_entity": q("SELECT n_entities, count(*), sum(n_rows) FROM u GROUP BY 1 ORDER BY 1"),
        "max_rows_one_headline": q("SELECT max(n_rows) FROM u")[0][0],
        "rows_of_headlines_used_more_than_once": q("SELECT sum(n_rows) FROM u WHERE n_rows>1")[0][0],
        "headline_distinct_days": q("""SELECT CASE WHEN d>=10 THEN '10+' ELSE CAST(d AS VARCHAR) END, count(*) FROM
               (SELECT headline_md5, count(DISTINCT CAST(ts_utc AS DATE)) d FROM m WHERE headline_md5 IS NOT NULL GROUP BY 1)
               GROUP BY 1 ORDER BY min(d)"""),
        "same_headline_same_entity_rows": q("""SELECT count(*), sum(c) FROM (SELECT headline_md5, rp_entity_id, count(*) c FROM m
               WHERE headline_md5 IS NOT NULL GROUP BY 1,2 HAVING count(*)>1)"""),
        "coverage": q("""SELECT count(*) FILTER (WHERE event_sentiment_score IS NOT NULL),
               count(*) FILTER (WHERE category IS NOT NULL), count(*) FILTER (WHERE event_text IS NOT NULL),
               count(*) FILTER (WHERE css IS NOT NULL), count(*) FILTER (WHERE nip IS NOT NULL),
               count(DISTINCT event_text), count(DISTINCT category) FROM m"""),
        "relevance_hist": q("SELECT CASE WHEN relevance>=90 THEN '>=90' WHEN relevance>=75 THEN '75-89' WHEN relevance>=50 THEN '50-74' ELSE '<50' END, count(*) FROM m GROUP BY 1 ORDER BY 1"),
        "event_text_words": q("""SELECT quantile_cont(w,[0.5,0.9,0.99]), max(w) FROM (SELECT length(event_text)-length(replace(event_text,' ',''))+1 w FROM m WHERE event_text IS NOT NULL)"""),
        "headline_chars": q("SELECT quantile_cont(length(headline),[0.5,0.9,0.99]), max(length(headline)) FROM u"),
        "ess_values": q("SELECT count(DISTINCT event_sentiment_score), min(event_sentiment_score), max(event_sentiment_score) FROM m"),
        "css_values": q("SELECT count(DISTINCT css), min(css), max(css), avg(css), count(*) FILTER (WHERE css=0) FROM m"),
        "top_categories": q("SELECT category, count(*) FROM m WHERE category IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 15"),
        "top_sources": q("SELECT source_name, count(*) FROM m GROUP BY 1 ORDER BY 2 DESC LIMIT 15"),
        "seconds": round(time.time() - t0, 1), "maxrss_mb": round(_maxrss_mb(), 1),
    }
    _write_json(prof, NEWS_PROFILE_FILE)
    log(f"prepare: {n:,} rows, {len(uk):,} unique headlines, {time.time()-t0:.1f} s")
    return prof


# --- news stage1: the unique headlines through stage 1 ---------------------------------------------

def _news_worker_init(tok_dir: str, emoji_memo: bool) -> None:
    global _TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        os.nice(10)
    except OSError:
        pass
    E.load_cleaning()
    if emoji_memo:
        E.install_emoji_memo()
    _TOK = E.Tokens(Path(tok_dir))


def news_stage1_chunk(texts: Sequence[str], tokens=None) -> Dict[str, list]:
    """Stage 1 of the social-media run on a list of headlines: S.stage1_rows with the StockTwits
    text version (one 'own' version = the text unchanged), so cleaning, flags, uncut ids and the
    window plan are the same code."""
    tokens = tokens or _TOK
    cols, counts = E.stage1_rows("stocktwits", {"message_body": list(texts)}, tokens)
    if cols["source_index"] != list(range(len(texts))) or set(cols["version"]) - {"own"}:
        raise RuntimeError("stage1_rows did not return one 'own' version per headline")
    cols.pop("source_index"); cols.pop("version")
    cols["_counts"] = counts
    return cols


def _news_job(args):
    i, texts = args
    t = time.time()
    out = news_stage1_chunk(texts)
    out["_i"] = i
    out["_s"] = time.time() - t
    out["_rss"] = _maxrss_mb()
    return out


def news_stage1(workers: int, emoji_memo: bool = True, limit: Optional[int] = None) -> Dict:
    t0 = time.time()
    u = pq.read_table(NEWS_UNIQ_FILE, columns=["headline_md5", "headline"])
    keys = u.column("headline_md5").to_pylist()
    texts = u.column("headline").to_pylist()
    if limit:
        keys, texts = keys[:limit], texts[:limit]
    jobs = [(i, texts[a:a + NEWS_CHUNK]) for i, a in enumerate(range(0, len(texts), NEWS_CHUNK))]
    tok_dir = str(E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT)
    ctx = mp.get_context("spawn")
    res: Dict[int, Dict] = {}
    with ctx.Pool(workers, initializer=_news_worker_init, initargs=(tok_dir, emoji_memo)) as pool:
        for out in pool.imap_unordered(_news_job, jobs):
            res[out["_i"]] = out
            if len(res) % 50 == 0 or len(res) == len(jobs):
                el = time.time() - t0
                log(f"stage1 {len(res)}/{len(jobs)} chunks, {el/60:.1f} min, eta {el/len(res)*(len(jobs)-len(res))/60:.1f} min")
    names = ["cleaned_text", "is_missing_text", "is_empty_after_clean", "is_clean_error", "clean_error",
             "n_tokens", "input_ids", "window_starts"]
    full = {c: [] for c in names}
    agg = Counter()
    err_types = Counter()
    rss = []
    for i in range(len(jobs)):
        r = res[i]
        for c in names:
            full[c].extend(r[c])
        for k, v in r["_counts"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("max_n_tokens", "max_token_id"):
                agg[k] += v
        agg["max_n_tokens"] = max(agg["max_n_tokens"], r["_counts"]["max_n_tokens"])
        agg["max_token_id"] = max(agg["max_token_id"], r["_counts"]["max_token_id"])
        err_types.update(r["_counts"]["clean_error_types"])
        rss.append(r["_rss"])
    if len(full["n_tokens"]) != len(keys):
        raise RuntimeError("stage 1 lost rows")
    tbl = pa.table({
        "headline_md5": pa.array(keys), "headline": pa.array(texts),
        "cleaned_text": pa.array(full["cleaned_text"], type=pa.string()),
        "is_empty_after_clean": pa.array(full["is_empty_after_clean"]),
        "is_clean_error": pa.array(full["is_clean_error"]),
        "clean_error": pa.array(full["clean_error"], type=pa.string()),
        "n_tokens": pa.array(full["n_tokens"], type=pa.int32()),
        "input_ids": pa.array(full["input_ids"], type=pa.list_(pa.uint16())),
        "window_starts": pa.array(full["window_starts"], type=pa.list_(pa.int32())),
    })
    if any(full["is_missing_text"]):
        raise RuntimeError("a unique headline is missing (NULLs were excluded in prepare)")
    _write_parquet(tbl, NEWS_TOK_FILE)
    rec = {"step": STEP_NEWS + ":stage1", "workers": workers, "start_method": "spawn", "emoji_memo": emoji_memo,
           "chunk": NEWS_CHUNK, "n_headlines": len(keys), "counts": dict(agg), "clean_error_types": dict(err_types),
           "seconds": round(time.time() - t0, 1), "worker_maxrss_mb_max": round(max(rss), 1) if rss else None,
           "versions": {"tokenizers": __import__("tokenizers").__version__, "transformers": __import__("transformers").__version__},
           "tokenizer_dir": tok_dir, "tokens_file_md5": _md5_file(NEWS_TOK_FILE)}
    _write_json(rec, Path(str(NEWS_TOK_FILE) + ".stats.json"))
    log(f"stage1 done: {len(keys):,} headlines in {rec['seconds']/60:.1f} min")
    return rec


# --- news score: the stocktwits model on the headline inputs; p, pc, pw per headline ---------------

def news_score(device: str, threads: int, batch_tokens: int, limit: Optional[int] = None,
          out_file: Optional[Path] = None) -> Dict:
    import torch
    import transformers
    t0 = time.time()
    checks: List[Dict] = []

    def check(cond, msg):
        checks.append({"check": msg, "passed": bool(cond)})
        if not cond:
            raise RuntimeError("check failed: " + msg)

    tok = pq.read_table(NEWS_TOK_FILE, columns=["headline_md5", "is_empty_after_clean", "is_clean_error",
                                            "n_tokens", "input_ids", "window_starts"])
    if limit:
        tok = tok.slice(0, limit)
    n = tok.num_rows
    ntok = tok.column("n_tokens").to_numpy()
    ids_l = tok.column("input_ids").to_pylist()
    ws_l = tok.column("window_starts").to_pylist()
    empty = np.asarray(tok.column("is_empty_after_clean").to_pylist(), dtype=bool)
    model_dir = E.EMTRACT_MODEL_DIR / E.MODEL_VARIANT
    sp = special_token_ids(E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT)
    sp_m = special_token_ids(model_dir)
    check(sp == sp_m, "special ids equal in the twitter and stocktwits folders")
    # window plans (the checks of sections 2 and 3) and model inputs (emtract_method.model_inputs)
    k = n_windows_for(ntok)
    check(bool(((ntok == 0) == empty).all()), "n_tokens = 0 exactly for the empty-after-clean headlines")
    bad = 0
    inputs: List[List[int]] = []
    owner: List[int] = []
    for i in range(n):
        if len(ws_l[i]) != k[i] or window_plan_error(int(ntok[i]), ws_l[i]) is not None or \
                (ntok[i] > 0 and check_window_starts(int(ntok[i]), ws_l[i]) is not None):
            bad += 1
            continue
        for x in E.model_inputs(ids_l[i], ws_l[i], sp["cls"], sp["sep"]):
            inputs.append(x); owner.append(i)
    check(bad == 0, "every window plan valid (the checks of sections 2 and 3)")
    owner = np.asarray(owner, dtype=np.int64)
    n_in = len(inputs)
    check(n_in == int(k.sum()), "model inputs = sum of n_windows")
    lengths = np.fromiter((len(x) for x in inputs), dtype=np.int64, count=n_in)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    values = np.fromiter((t for x in inputs for t in x), dtype=np.int64, count=int(offsets[-1]))
    check(bool(((lengths >= 3) & (lengths <= E.MAX_LENGTH)).all()), "input lengths 3..64")
    del inputs, ids_l
    wfile = _weights_file(model_dir)
    wsha = sha256_file(wfile)
    want = {ln.split()[1].lstrip("*"): ln.split()[0] for ln in (model_dir / "SHA256SUMS").read_text().splitlines() if ln.strip()}
    check(want.get(wfile.name) == wsha, "weights sha256 equals SHA256SUMS")
    if threads:
        torch.set_num_threads(threads)
    model = load_model(model_dir, device)
    check(all(p.dtype == torch.float32 for p in model.parameters()), "every parameter is float32")
    seq = np.arange(n_in, dtype=np.int64)
    batches = plan_batches(lengths, seq, batch_tokens)
    logits = np.empty((n_in, len(LABELS)), dtype=np.float32)
    t1 = time.time()
    done = 0
    with torch.no_grad():
        for bi, (bucket, idx) in enumerate(batches):
            ids, mask = pad_batch(values, offsets, idx, bucket, sp["pad"])
            lo = model(input_ids=torch.from_numpy(ids).to(device), attention_mask=torch.from_numpy(mask).to(device)).logits
            logits[idx] = lo.float().cpu().numpy()
            done += len(idx)
            if bi % 100 == 0:
                el = time.time() - t1
                log(f"score {done:,}/{n_in:,} inputs, {el/60:.1f} min, {done/max(el,1e-9):.0f}/s")
    score_s = time.time() - t1
    probs = softmax_f32(logits)
    check(bool(np.isfinite(logits).all() and np.isfinite(probs).all()), "logits and probabilities finite")
    dev = float(np.abs(probs.astype(np.float64).sum(1) - 1).max()) if n_in else 0.0
    check(dev <= PROB_SUM_TOL, "probabilities sum to 1 within 1e-5")
    smd = float(np.abs(softmax64(logits) - probs.astype(np.float64)).max()) if n_in else 0.0
    check(smd <= 1e-5, "softmax64(logits) equals p within 1e-5")
    # aggregation (as `merge`): p = window 0, pc = equal-weight mean, pw = coverage-weighted mean
    L = len(LABELS)
    has = k >= 1
    long_mask = k >= 2
    seg = np.zeros(n, dtype=np.int64)
    seg[1:] = np.cumsum(k)[:-1]
    check(bool((owner[seg[has]] == np.nonzero(has)[0]).all()), "window 0 of every scored headline is its first input")
    P64 = probs.astype(np.float64)
    p_out = np.zeros((n, L), np.float32); pc_out = np.zeros((n, L), np.float32); pw_out = np.zeros((n, L), np.float32)
    p_out[has] = probs[seg[has]]
    if has.any():
        sums = np.add.reduceat(P64, seg[has], axis=0)
        pc_all = sums / k[has][:, None]
        w = np.ones(n_in, dtype=np.float64)
        for i in np.nonzero(long_mask)[0]:
            wi = coverage_weights(int(ntok[i]), tuple(ws_l[i]))
            check(abs(float(wi.sum()) - float(ntok[i])) <= 1e-9 * max(1.0, float(ntok[i])), "coverage weights sum to n")
            w[seg[i]:seg[i] + k[i]] = wi
        wsum = np.add.reduceat(P64 * w[:, None], seg[has], axis=0)
        wtot = np.add.reduceat(w, seg[has])
        pw_all = wsum / wtot[:, None]
        sel = long_mask[has]
        pc_out[long_mask] = pc_all[sel].astype(np.float32)
        pw_out[long_mask] = pw_all[sel].astype(np.float32)
    cols = {"headline_md5": tok.column("headline_md5"),
            "n_tokens": pa.array(ntok.astype(np.int32)),
            "is_empty_after_clean": tok.column("is_empty_after_clean"),
            "is_clean_error": tok.column("is_clean_error"),
            "n_windows": pa.array(k.astype(np.int16))}
    for j, lab in enumerate(LABELS):
        cols[f"p_{lab}"] = pa.array(p_out[:, j], mask=~has)
    for j, lab in enumerate(LABELS):
        cols[f"pc_{lab}"] = pa.array(pc_out[:, j], mask=~long_mask)
    for j, lab in enumerate(LABELS):
        cols[f"pw_{lab}"] = pa.array(pw_out[:, j], mask=~long_mask)
    uniq = pq.read_table(NEWS_UNIQ_FILE, columns=["headline_md5", "n_rows"])
    if limit:
        uniq = uniq.slice(0, limit)
    check(uniq.column("headline_md5").equals(tok.column("headline_md5")), "tokens and unique headlines in the same key order")
    cols["n_source_rows"] = uniq.column("n_rows")
    out = pa.table(cols)
    # output checks
    for c in out.column_names:
        nulls = out.column(c).null_count
        if c.startswith("p_"):
            check(nulls == int((~has).sum()), f"{c} null exactly for the unscored headlines")
        elif c.startswith(("pc_", "pw_")):
            check(nulls == int((~long_mask).sum()), f"{c} null exactly for n_windows < 2")
    check(len(set(out.column("headline_md5").to_pylist())) == n, "headline_md5 unique")
    for pref, m in (("p", has), ("pc", long_mask), ("pw", long_mask)):
        arr = {"p": p_out, "pc": pc_out, "pw": pw_out}[pref][m].astype(np.float64)
        if len(arr):
            check(float(np.abs(arr.sum(1) - 1).max()) <= PROB_SUM_TOL, f"{pref} sums to 1 within 1e-5")
    final = Path(out_file) if out_file else (NEWS_EMO_FILE if not limit else NEWS_EMO_FILE.with_name(f"_trial_{limit}_{device}.parquet"))
    _write_parquet(out, final)
    top_p = p_out[long_mask].argmax(1)
    empty_score = None
    rec = {
        "step": STEP_NEWS + ":score", "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device, "torch": torch.__version__, "transformers": transformers.__version__,
        "python": sys.version.split()[0], "numpy": np.__version__, "pyarrow": pa.__version__,
        "torch_threads": torch.get_num_threads(), "dtype": "float32", "batch_tokens": batch_tokens,
        "pad_buckets": list(E.PAD_BUCKETS), "weights_file": wfile.name, "weights_sha256": wsha,
        "model_variant": E.MODEL_VARIANT, "tokenizer_variant": E.TOKENIZER_VARIANT, "labels": list(LABELS),
        "special_ids": sp,
        "counts": {"headlines": n, "scored": int(has.sum()), "empty_after_clean": int(empty.sum()),
                   "clean_error": int(np.asarray(tok.column("is_clean_error").to_pylist()).sum()),
                   "long_over_62": int(long_mask.sum()), "model_inputs": n_in,
                   "windows_hist": {str(int(a)): int(b) for a, b in zip(*np.unique(k, return_counts=True))},
                   "max_n_tokens": int(ntok.max()) if n else 0,
                   "long_top_changes_pc": int((pc_out[long_mask].argmax(1) != top_p).sum()),
                   "long_top_changes_pw": int((pw_out[long_mask].argmax(1) != top_p).sum()),
                   "top_emotion_p": {LABELS[a]: int(b) for a, b in zip(*np.unique(p_out[has].argmax(1), return_counts=True))},
                   "mean_p": {lab: float(p_out[has][:, j].mean()) for j, lab in enumerate(LABELS)} if has.any() else None,
                   "by_bucket": {str(b): int(sum(len(i) for bb, i in batches if bb == b)) for b in E.PAD_BUCKETS}},
        "max_prob_sum_dev": dev, "max_softmax_dev": smd,
        "timing_s": {"score": round(score_s, 1), "total": round(time.time() - t0, 1)},
        "inputs_per_s": round(n_in / score_s, 1) if score_s > 0 else None,
        "maxrss_mb": round(_maxrss_mb(), 1),
        "inputs_files": {"tokens": {"path": str(NEWS_TOK_FILE), "md5": _md5_file(NEWS_TOK_FILE)},
                         "unique": {"path": str(NEWS_UNIQ_FILE), "md5": _md5_file(NEWS_UNIQ_FILE)}},
        "checks": checks,
    }
    rec["output_md5"] = _md5_file(final)
    _write_json(rec, Path(str(final) + ".stats.json"))
    log(f"score done: {n_in:,} inputs in {score_s/60:.1f} min ({rec['inputs_per_s']}/s) on {device}")
    return rec


# --- news refcheck: the predict()-equivalent path on a random sample -------------------------------

def news_refcheck(n_sample: int, seed: int, device: str, threads: int) -> Dict:
    """clean_text (EmTract's catch-all) -> the twitter tokenizer with truncation=True,
    padding='max_length', max_length=64 (EmTract's tokenize) -> the stocktwits model ->
    scipy.special.softmax, as Model.predict/batch_predict do (without Trainer/datasets), on a
    random sample of scored headlines; compared with the stored p_*."""
    import torch
    import transformers
    from scipy.special import softmax
    emo = pq.read_table(NEWS_EMO_FILE)
    uniq = pq.read_table(NEWS_UNIQ_FILE, columns=["headline_md5", "headline"])
    check_keys = emo.column("headline_md5").equals(uniq.column("headline_md5"))
    has = np.asarray(emo.column("n_windows").to_numpy() >= 1)
    rng = np.random.default_rng(seed)
    idx_scored = np.nonzero(has)[0]
    long_idx = np.nonzero(emo.column("n_windows").to_numpy() >= 2)[0]
    pick = np.unique(np.concatenate([rng.choice(idx_scored, min(n_sample, len(idx_scored)), replace=False),
                                     rng.choice(long_idx, min(200, len(long_idx)), replace=False) if len(long_idx) else np.array([], int)]))
    texts = [uniq.column("headline")[int(i)].as_py() for i in pick]
    E.load_cleaning()          # the real advertools re (no memo) for the reference path
    cleaned = [E.clean_text(t) for t in texts]
    tokd = transformers.DistilBertTokenizerFast.from_pretrained(str(E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT), model_max_length=64)
    if threads:
        torch.set_num_threads(threads)
    model = load_model(E.EMTRACT_MODEL_DIR / E.MODEL_VARIANT, device)
    outs = []
    with torch.no_grad():
        for a in range(0, len(cleaned), 256):
            enc = tokd(cleaned[a:a + 256], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            outs.append(model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits.float().cpu().numpy())
    pr = softmax(np.concatenate(outs), axis=1)
    stored = np.column_stack([emo.column(f"p_{lab}").combine_chunks().to_numpy(zero_copy_only=False)[pick] for lab in LABELS])
    d = np.abs(pr - stored)
    # (b) the split path on the same sample: stored stage-1 ids -> every window, unpadded-equivalent
    # batches (section 2's plan_batches/pad_batch, buckets 16/32/48/64) -> p = window 0, pc = mean
    tokt = pq.read_table(NEWS_TOK_FILE, columns=["headline_md5", "input_ids", "window_starts"])
    check_tok = tokt.column("headline_md5").equals(emo.column("headline_md5"))
    sp = special_token_ids(E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT)
    ins, own = [], []
    for j, i in enumerate(pick):
        for x in E.model_inputs(tokt.column("input_ids")[int(i)].as_py(), tokt.column("window_starts")[int(i)].as_py(),
                                sp["cls"], sp["sep"]):
            ins.append(x); own.append(j)
    own = np.asarray(own)
    lens = np.asarray([len(x) for x in ins]); offs = np.concatenate([[0], np.cumsum(lens)])
    vals = np.asarray([t for x in ins for t in x], dtype=np.int64)
    lg = np.empty((len(ins), len(LABELS)), dtype=np.float32)
    with torch.no_grad():
        for bucket, idx in plan_batches(lens, np.arange(len(ins)), 16384):
            ii, mm = pad_batch(vals, offs, idx, bucket, sp["pad"])
            lg[idx] = model(input_ids=torch.from_numpy(ii).to(device), attention_mask=torch.from_numpy(mm).to(device)).logits.float().cpu().numpy()
    P = softmax_f32(lg).astype(np.float64)
    first = np.searchsorted(own, np.arange(len(pick)))
    p_split = P[first]
    kk = np.bincount(own, minlength=len(pick))
    pc_split = np.add.reduceat(P, first, axis=0) / kk[:, None]
    lm = kk >= 2
    stored_pc = np.column_stack([emo.column(f"pc_{lab}").combine_chunks().to_numpy(zero_copy_only=False)[pick] for lab in LABELS])
    ds = np.abs(p_split - stored)
    dpc = np.abs(pc_split[lm] - stored_pc[lm]) if lm.any() else np.zeros((0, len(LABELS)))
    import scipy
    rec = {"n_checked": int(len(pick)), "n_long_in_sample": int(np.isin(pick, long_idx).sum()), "device": device,
           "keys_aligned": bool(check_keys and check_tok), "seed": seed,
           "reference_stack": {"python": sys.version.split()[0], "torch": torch.__version__,
                               "transformers": transformers.__version__, "scipy": scipy.__version__},
           "stored_scores_from": json.loads(Path(str(NEWS_EMO_FILE) + ".stats.json").read_text()).get("device"),
           "predict_path": {"path": "clean_text -> tokenizer(truncation, padding=max_length, 64) -> model fp32 -> scipy softmax",
                            "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
                            "top_emotion_agree": float((pr.argmax(1) == stored.argmax(1)).mean())},
           "split_path": {"path": "stored ids -> windows -> plan_batches/pad_batch -> model fp32 -> softmax_f32",
                          "n_inputs": int(len(ins)), "max_abs_dp": float(ds.max()), "mean_abs_dp": float(ds.mean()),
                          "top_emotion_agree_p": float((p_split.argmax(1) == stored.argmax(1)).mean()),
                          "n_long": int(lm.sum()), "max_abs_dpc": float(dpc.max()) if lm.any() else None,
                          "top_emotion_agree_pc": float((pc_split[lm].argmax(1) == stored_pc[lm].argmax(1)).mean()) if lm.any() else None},
           "max_abs_diff": float(max(d.max(), ds.max(), dpc.max() if lm.any() else 0.0))}
    _write_json(rec, NEWS_REF_FILE)
    log(f"refcheck: {rec}")
    return rec


def main_news(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="06_emotion_emtract.py news",
                                 description="EmTract scores for the RavenPack news headlines (section 6)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--csv", type=Path, default=NEWS_CSV_DEFAULT)
    b = sub.add_parser("stage1"); b.add_argument("--workers", type=int, default=6); b.add_argument("--limit", type=int)
    b.add_argument("--no-emoji-memo", action="store_true")
    c = sub.add_parser("score"); c.add_argument("--device", default="cpu"); c.add_argument("--threads", type=int, default=6)
    c.add_argument("--batch-tokens", type=int, default=16384); c.add_argument("--limit", type=int)
    c.add_argument("--out", type=Path, help="output file (default news_headline_emotion.parquet)")
    d = sub.add_parser("refcheck"); d.add_argument("--n", type=int, default=2000); d.add_argument("--seed", type=int, default=20260930)
    d.add_argument("--device", default="cpu"); d.add_argument("--threads", type=int, default=6)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        news_prepare(args.csv)
    elif args.cmd == "stage1":
        news_stage1(args.workers, not args.no_emoji_memo, args.limit)
    elif args.cmd == "score":
        news_score(args.device, args.threads, args.batch_tokens, args.limit, args.out)
    else:
        news_refcheck(args.n, args.seed, args.device, args.threads)
    return 0


# ==================================================================================================
# Command line: one subcommand per stage (the table at the top of this file)
# ==================================================================================================

COMMANDS = {
    # subcommand: (function, receives the subcommand word too?, one-line help)
    "tokenize": (main_tokenize, False, "section 1: stage 1, clean and tokenize a platform's texts"),
    "pack": (main_stage2, True, "section 2: model inputs from finished stage-1 shards"),
    "score": (main_stage2, True, "section 2: the stocktwits model alone on a packed inputs file"),
    "collect": (main_stage2, True, "section 2: raw scores + map -> the scores table"),
    "merge": (main_merge, False, "section 3: one row per (row_id, version); also --empty-score"),
    "text-key-index": (main_text_key_index, False, "section 4: one row per distinct cleaned Twitter text"),
    "text-key-list": (main_text_key_list, False, "section 4: the index keys as a sorted binary list"),
    "dlt": (main_dlt, False, "section 5: clean | match | tokenize | verify | completeness | merge-keys | build | refscore"),
    "news": (main_news, False, "section 6: prepare | stage1 | score | refcheck"),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: python pipeline/06_emotion_emtract.py <subcommand> [options]   (<subcommand> -h: its options)\n")
        for name, (_, _, text) in COMMANDS.items():
            print(f"  {name:15s} {text}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    fn, whole, _ = COMMANDS[argv[0]]
    return fn(argv if whole else argv[1:])


if __name__ == "__main__":
    sys.exit(main())
