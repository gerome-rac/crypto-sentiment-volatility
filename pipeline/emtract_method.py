"""
emtract_method.py -- EmTract's method as this study runs it: settings, paths, the stage-1 shard plan,
the text versions, EmTract's own cleaning with its flags, the tokenizer, the long-text window plan and
the model inputs. Shared by 06_emotion_emtract.py (all its subcommands) and 07_polarity_finbert.py
(the rows, the shard plan and the text versions).

EmTract (Vamossy & Skog; MIT licence; code vendored in third_party/emtract/, upstream commit
EMTRACT_UPSTREAM_COMMIT) is run as its three parts, one after another, instead of through its
`Model.predict()` (MASTERPLAN_NOV15.md section 7, decisions of 29 Sep 2026, (4)-(6)):

    stage 1 (CPU)   EmTract's own `clean_text` cleans the original text once, and tokenizers 0.10.3
                    (EmTract's training version) turns the cleaned text into wordpiece ids, uncut.
                    `06_emotion_emtract.py tokenize` -> <platform>/emtract/tokens/
    stage 2 (GPU)   the stocktwits model alone scores the stored ids. The primary input is
                    [CLS] + the first 62 wordpieces + [SEP], exactly what predict() gives the model;
                    longer texts also get windows.
                    `06_emotion_emtract.py pack | score | collect` -> <platform>/emtract/scores/
    merge           one row per post and text version: the primary score, the equal-weight window
                    mean and the coverage-weighted mean.
                    `06_emotion_emtract.py merge` -> <platform>/emtract/emotion/

Why the parts run separately: EmTract's `Model.predict()` (upstream emtract/model.py) does
`clean_text` on every text, then the DistilBERT fast tokenizer loaded from the twitter model folder with
`truncation=True, padding="max_length", max_length=64`, then the model. Stage 1 does the first two parts
once per text and stores the result, so that the model (stage 2) can run on another machine with a
current torch. On EmTract's 10,000 tagged posts this split equals `predict()` bit for bit (reference
checks of 29 Sep 2026, with EmTract's own library versions). Nothing here changes the tables written by
steps 01-05; every EmTract output is a new file keyed by the id of its row.

Apart from `plan_shards` (which reads a Parquet footer) nothing here reads or writes a table; every
function works on values handed to it. Importing this module needs only the standard library, so the
scoring command runs on a GPU machine that has torch, transformers, numpy and pyarrow but no DuckDB
(the check and log helpers then come from the fallback at the top instead of runner.py).

The stage-1 rules, with their sources
-------------------------------------
1. Text versions (decision (3) of 29 Sep; interface item 1 of the EmTract plan):
       platform    own                  quoted                          both
       reddit      own_text             quoted_text (with its '>'       text (the original body,
                                        markers), where                 quote lines where the
                                        has_quote_lines                 author put them)
       stocktwits  message_body         --                              --
       twitter     text                 parent_text, where not NULL     text + " " + parent_text
   Every live source row gets an 'own' version, also when that text is empty (the 16,402 Reddit
   rows without own text); 'quoted' and 'both' exist only where a quoted text exists. Reddit dead
   rows (is_dead) get no version at all; the caller counts them. A NULL source text is flagged
   `is_missing_text` and not cleaned (the 293 StockTwits NULL bodies; EmTract's catch-all would
   silently turn None into "").
2. Cleaning: EmTract's own `clean_text`, copied unchanged into code/third_party/emtract/ (one
   documented change: the SymSpell word list is read from the bundled CSV instead of GitHub;
   SOURCE.md there). Its dictionaries are read at import time through a path relative to that
   folder, so the import runs with EMTRACT_CODE_DIR as the working directory (`load_cleaning`);
   after the import no function of the module reads a file. Every text is cleaned exactly once
   (never a piece of a text, never twice: EmTract scoring rule of 29 Sep 2026). Identical
   original texts within one call share one cleaning result (a cache): `clean_text` is a pure
   function of its input (two passes over EmTract's 10,000 tagged posts were identical, reference
   checks of 29 Sep), so the cache changes no output, only the time.
3. Cleaning errors (StockTwits defect map PI2): `clean_text` wraps all its steps in a bare
   `except:` that returns "", so an error and a truly empty text look the same. For the texts
   that clean to "" only, `clean_text_uncaught` runs the same steps in the same order without the
   catch-all: if it raises, the text is flagged `is_clean_error` and the error is kept as
   'ExceptionType: message'; if it returns "", the text is truly empty. `clean_text_uncaught` is
   a line-by-line copy of `clean_text`'s body (third_party/emtract/emtract/processors/cleaning.py,
   lines 356-382); `load_cleaning` refuses to run if that file's sha256 is not the vendored one,
   so the copy cannot drift from the original unnoticed. An error text also counts as empty
   (is_empty_after_clean): it is what predict() would have scored as "".
4. Tokens: DistilBertTokenizerFast from the twitter model folder (as in EmTract's model.py, for
   both model variants), tokenizers 0.10.3 exactly (decision (6) of 29 Sep: 0.22.2 changes the ids
   of 1.24% of EmTract's tagged posts), `add_special_tokens=False`, no truncation, no padding.
   The stored ids are the full wordpiece sequence of the cleaned text; the model input that
   `predict()` builds is [CLS] + the first 62 of them + [SEP] (stage 1 checks this on a sample
   against the tokenizer's own truncation).
5. Windows for long texts (decision (4) of 29 Sep; interface item 2): a text with n > 62
   wordpieces gets k = ceil((n - 16) / 46) windows of 62 wordpieces, so that consecutive windows
   overlap by at least 16 wordpieces (a quarter window; Dai et al. 2022, p. 6). The ideal starts
   are spread evenly from 0 to n - 62; the first start is 0 and the last window ends at the last
   wordpiece. Each interior start moves to the nearest word start (a wordpiece that does not
   begin with '##') within 8 wordpieces of its ideal position, if every overlap stays >= 16;
   otherwise it stays. Window 0 is the primary input. Example: n = 100 gives [0, 38].
   Two details the interface leaves open are fixed here (and reported): an ideal position
   j*(n-62)/(k-1) is rounded half up to an integer; between two word starts at the same distance
   the earlier one is taken. A start that is already a word start does not move.
6. Model inputs (interface item 3): window j = [CLS] + ids[s_j : s_j + 62] + [SEP]; the
   special-token ids come from the tokenizer, never hard-coded.

Environment: `emtract-ref` for stage 1 (environment-emtract-ref.yml: Python 3.9, tokenizers 0.10.3,
transformers 4.10.3, emoji 1.7.0, advertools 0.13.1, symspellpy 6.7.0: EmTract's versions). The pure
functions (versions, window plan, model inputs, shard plan) need none of these and import anywhere.
Tags such as D17, PI2, R1 refer to the thesis's decision log and platform defect maps (README.md).
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import config
from .config import CODE_DIR, OUT_DIR, RAW_DIR, out_path

try:                                    # the pipeline's safe-runner helpers (runner.py imports DuckDB)
    from .runner import CheckError, check, log
except ImportError:                     # a GPU machine that only runs `score` has no DuckDB: same helpers
    import datetime as _dt

    class CheckError(RuntimeError):
        """A conservation or consistency check failed; the output was not committed."""

    def log(msg: str) -> None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC] {msg}", flush=True)

    def check(condition: bool, message: str, results: Optional[List[Dict[str, object]]] = None) -> None:
        """Record a check and stop the step if it fails."""
        if results is not None:
            results.append({"check": message, "passed": bool(condition)})
        if not condition:
            raise CheckError(message)


# ==================================================================================================
# 1. Settings: EmTract's code and model files, inputs and windows, output folders
# ==================================================================================================
# EmTract's cleaning code, copied unchanged except for one documented line (SOURCE.md there).
# cleaning.py reads its dictionaries through a path relative to this folder, so code that
# imports it must run with this folder as the working directory.
EMTRACT_CODE_DIR = CODE_DIR / "third_party" / "emtract"
EMTRACT_UPSTREAM_COMMIT = "4212881cbccd12c4cca08954a3bd37123812a26e"

# Model files of both variants with SHA256SUMS (downloaded 29 Sep 2026 from EmTract's public
# model store; not in git). EmTract loads one tokenizer from the twitter folder for both
# variants; its files give the same ids as the stocktwits folder's.
EMTRACT_MODEL_DIR = Path(os.environ.get("THESIS_EMTRACT_MODELS", RAW_DIR / "emtract_models"))
MODEL_VARIANT = "stocktwits"                    # decided 29 Sep 2026, (1)
TOKENIZER_VARIANT = "twitter"                   # as in EmTract's model.py
TOKENIZERS_VERSION = "0.10.3"                   # decided 29 Sep 2026, (6): never another version

# Label order of the model's outputs, by index (EmTract model.py; the config holds only LABEL_0-6).
LABELS = ("neutral", "happy", "sad", "anger", "disgust", "surprise", "fear")

# --- Inputs and windows (decisions of 29 Sep 2026, (4)) --------------------------------------------
MAX_LENGTH = 64                                 # EmTract's max_length, including [CLS] and [SEP]
MAX_TEXT_TOKENS = MAX_LENGTH - 2                # 62 wordpieces of cleaned text per input
MIN_WINDOW_OVERLAP = 16                         # a quarter window (Dai et al. 2022, p. 6)
PAD_BUCKETS = (16, 32, 48, 64)                  # stage 2 pads each input to the smallest that fits

# Text versions scored per row (decisions of 29 Sep 2026, (3)).
#   own     every row: Reddit own_text, StockTwits message_body, Twitter text
#   quoted  rows with quoted text: Reddit quoted_text (with its '>' markers), Twitter parent_text
#   both    the same rows: Reddit the original body; Twitter text + " " + parent_text
VERSIONS = ("own", "quoted", "both")
COMBINED_SEPARATOR = " "

# --- Output folders (data/pipeline/<platform>/emtract/...) -----------------------------------------

def tokens_dir(platform: str) -> Path:
    """Stage 1: <platform>/emtract/tokens/<partition>/shard-NNNNN.parquet (+ .stats.json)."""
    return out_path(platform, "emtract", "tokens")


def scores_dir(platform: str) -> Path:
    """Stage 2: <platform>/emtract/scores/<batch>.parquet (+ .stats.json), one row per model input."""
    return out_path(platform, "emtract", "scores")


def emotion_dir(platform: str) -> Path:
    """Merge: <platform>/emtract/emotion/<partition>.parquet, one row per (row_id, version)."""
    return out_path(platform, "emtract", "emotion")


# --- Run bookkeeping (data/pipeline/_emtract_run/; one writer per file) ----------------------------
# control.json              written by hand; read by `tokenize` between shards (workers, pause)
# status_stage1.json        written by `tokenize` only (progress, rates, memory, disk)
# empty_text_score.json     written by `merge --empty-score` only (EmTract's fixed score for an empty text)
# modal_inputs/<platform>/  written by `pack` only: packed model inputs and their maps
# modal_outputs/<platform>/ raw scores written by `score`, waiting for `collect`
# The code does not depend on any provider; the stored batches and maps are found under these names.
RUN_DIR = Path(os.environ.get("THESIS_EMTRACT_RUN", OUT_DIR / "_emtract_run"))
CONTROL_FILE = RUN_DIR / "control.json"
STATUS_STAGE1_FILE = RUN_DIR / "status_stage1.json"
EMPTY_SCORE_FILE = RUN_DIR / "empty_text_score.json"
PACKED_INPUTS_DIR = RUN_DIR / "modal_inputs"     # packed stage-2 inputs (ids and sequence numbers only)
RAW_SCORES_DIR = Path(os.environ.get("THESIS_EMTRACT_RAW_SCORES", RUN_DIR / "modal_outputs"))   # raw stage-2 scores waiting for collect

# Guards of stage 1
MIN_FREE_DISK_GB = float(os.environ.get("THESIS_EMTRACT_MIN_FREE_GB", "25"))
DEFAULT_WORKERS = int(os.environ.get("THESIS_EMTRACT_WORKERS", "4"))


# ==================================================================================================
# 2. Source tables and the stage-1 shard plan (also used by 07_polarity_finbert.py)
# ==================================================================================================
# Source columns per platform (step-1 scope audit of 29 Sep: R1/R7, S1/S4, T3/T5). The id column
# is the platform's native row id; `tokenize --input-file` also accepts it as 'row_id'.
SOURCES = {
    "reddit": {"id": "id", "text": ("text", "own_text", "quoted_text", "has_quote_lines", "is_dead")},
    "stocktwits": {"id": "message_id", "text": ("message_body",)},
    "twitter": {"id": "unit_id", "text": ("text", "parent_text")},
}

REDDIT_PARTITIONS = ("BNB", "BTC", "DOGE", "ETH", "MARKET", "XRP")
SHARD_MAX_ROWS = 300_000          # a larger row group is split into pieces of about SHARD_TARGET_ROWS
SHARD_TARGET_ROWS = 200_000


def source_files(platform: str, partitions: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """(partition, file) pairs of the local source tables, in a stable order."""
    if platform == "reddit":
        parts = partitions or list(REDDIT_PARTITIONS)
        out = [(p, config.out_path("reddit", "extract", f"reddit_{p}.parquet")) for p in parts]
    elif platform == "stocktwits":
        found = sorted(glob.glob(str(config.out_path("stocktwits", "coarse", "month=*", "data_0.parquet"))))
        months = {Path(f).parent.name.split("=", 1)[1]: Path(f) for f in found}
        parts = partitions or sorted(months)
        missing = [p for p in parts if p not in months]
        if missing:
            raise SystemExit(f"no StockTwits table for months {missing}")
        out = [(p, months[p]) for p in parts]
    elif platform == "twitter":
        if partitions and partitions != ["units"]:
            raise SystemExit("Twitter has one partition, 'units'")
        out = [("units", config.out_path("twitter", "units", "units.parquet"))]
    else:
        raise SystemExit(f"unknown platform {platform!r}")
    for p, f in out:
        if not f.exists():
            raise SystemExit(f"missing source table for partition {p}: {f}")
    return out


def plan_shards(platform: str, partition: str, path: Path, limit_rows: Optional[int],
                shard_rows: Optional[int] = None) -> List[Dict]:
    """One shard per row group; a row group above SHARD_MAX_ROWS (or above --shard-rows, for
    pilots) is split into equal pieces."""
    import pyarrow.parquet as pq
    md = pq.ParquetFile(str(path)).metadata
    shards, g = [], 0
    for rg in range(md.num_row_groups):
        n = md.row_group(rg).num_rows
        if shard_rows:
            pieces = max(1, math.ceil(n / shard_rows))
        else:
            pieces = 1 if n <= SHARD_MAX_ROWS else math.ceil(n / SHARD_TARGET_ROWS)
        bounds = [round(i * n / pieces) for i in range(pieces + 1)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if limit_rows is not None and g >= limit_rows:
                return shards
            if limit_rows is not None:
                b = min(b, a + (limit_rows - g))
            shards.append({"platform": platform, "partition": partition, "source": str(path),
                           "shard_index": len(shards), "row_group": rg, "rg_offset": a, "rg_stop": b,
                           "row_start": g, "row_stop": g + (b - a)})
            g += b - a
    return shards


# ==================================================================================================
# 3. Text versions
# ==================================================================================================

def build_versions(platform: str, cols: Dict[str, Sequence]) -> Tuple[List[int], List[str], List[Optional[str]], Dict[str, int]]:
    """Text versions of a batch of source rows, in source-row order, versions own, quoted, both.

    `cols` maps the platform's text columns (SOURCES[platform]['text']) to equal-length sequences.
    Returns (source_index, version, text, counts): one entry per (live row, version); text None
    means the source text is missing (NULL). counts: rows_in, rows_dead, rows_live, rows_quoted.
    """
    if platform not in SOURCES:
        raise ValueError(f"unknown platform {platform!r}")
    names = SOURCES[platform]["text"]
    n = len(cols[names[0]])
    for c in names:
        if len(cols[c]) != n:
            raise ValueError(f"column {c} has {len(cols[c])} values, expected {n}")
    idx: List[int] = []
    ver: List[str] = []
    txt: List[Optional[str]] = []
    dead = quoted = 0
    if platform == "reddit":
        text, own, quo, hq, isd = (cols[c] for c in names)
        for i in range(n):
            if isd[i]:
                dead += 1
                continue
            has_q = quo[i] is not None
            if bool(hq[i]) != has_q:
                # 01_extract_reddit asserts quoted_text IS NULL <=> NOT has_quote_lines; re-checked here.
                raise ValueError(f"row {i}: has_quote_lines={hq[i]!r} but quoted_text is "
                                 f"{'set' if has_q else 'NULL'}")
            idx.append(i); ver.append("own"); txt.append(own[i])
            if has_q:
                quoted += 1
                idx += [i, i]; ver += ["quoted", "both"]; txt += [quo[i], text[i]]
    elif platform == "stocktwits":
        (body,) = (cols[c] for c in names)
        for i in range(n):
            idx.append(i); ver.append("own"); txt.append(body[i])
    else:  # twitter
        text, par = (cols[c] for c in names)
        for i in range(n):
            idx.append(i); ver.append("own"); txt.append(text[i])
            if par[i] is not None:
                quoted += 1
                both = None if text[i] is None else text[i] + COMBINED_SEPARATOR + par[i]
                idx += [i, i]; ver += ["quoted", "both"]; txt += [par[i], both]
    counts = {"rows_in": n, "rows_dead": dead, "rows_live": n - dead, "rows_quoted": quoted}
    return idx, ver, txt, counts


# ==================================================================================================
# 4. Cleaning: EmTract's clean_text once per text, with the missing / empty / error flags
# ==================================================================================================
# sha256 of the vendored cleaning.py (third_party/emtract/SOURCE.md). clean_text_uncaught below
# copies that file's clean_text; a different file means the copy must be re-checked.
VENDORED_CLEANING_SHA256 = "db775342e77f221a127eb194d845173df9e56fe8e762d89265517c0711842931"


_CLEANING = None


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_cleaning():
    """Import EmTract's vendored cleaning module once (working directory = EMTRACT_CODE_DIR).

    Bytecode writing is switched off for the import, so that nothing is written into
    third_party/ (read-only for the pipeline)."""
    global _CLEANING
    if _CLEANING is not None:
        return _CLEANING
    src = EMTRACT_CODE_DIR / "emtract" / "processors" / "cleaning.py"
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if sha != VENDORED_CLEANING_SHA256:
        raise RuntimeError(f"{src.name} sha256 {sha} is not the vendored {VENDORED_CLEANING_SHA256}; "
                           "re-check clean_text_uncaught against it before scoring")
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if str(EMTRACT_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(EMTRACT_CODE_DIR))
        with _cwd(EMTRACT_CODE_DIR):
            from emtract.processors import cleaning
    finally:
        sys.dont_write_bytecode = old_flag
    if os.path.realpath(cleaning.__file__) != os.path.realpath(src):
        raise RuntimeError("the imported cleaning module is not the vendored file")
    _CLEANING = cleaning
    return cleaning


class _MemoRe:
    """Stand-in for the `re` module inside advertools.emoji: `findall` on advertools' fixed EMOJI
    pattern is memoized by its input string; every other call goes to `re` unchanged.

    Why (profile of 29 Sep 2026, 5,000 Reddit rows): clean_text calls
    advertools.extract_emoji(text) with a string, so advertools runs
    re.findall(EMOJI, character.lower()) once per character. Each call re-hashes the compiled
    138,453-character pattern inside re's cache lookup, which took 90 of the 97 seconds of
    cleaning. The result depends only on that one-character string, so a dict keyed by it returns
    exactly what re.findall returns (a fresh list each time); nothing in EmTract's or advertools'
    code is changed, and exceptions (for example extract_emoji's ZeroDivisionError on an empty
    string) happen exactly as before."""

    def __init__(self, real_re, pattern):
        self._re = real_re
        self._pattern = pattern
        self._memo: Dict[str, List[str]] = {}

    def findall(self, pattern, string, flags=0):
        if pattern is self._pattern and flags == 0 and isinstance(string, str):
            hit = self._memo.get(string)
            if hit is None:
                hit = self._re.findall(pattern, string)
                if len(string) <= 8:
                    self._memo[string] = hit
            return list(hit)
        return self._re.findall(pattern, string, flags)

    def __getattr__(self, name):
        return getattr(self._re, name)


def install_emoji_memo() -> bool:
    """Install the exact memo of _MemoRe into advertools.emoji (idempotent). Returns True if active."""
    load_cleaning()
    import re as _re
    import advertools.emoji as adv_emoji
    if isinstance(adv_emoji.re, _MemoRe):
        return True
    if adv_emoji.re is not _re or not hasattr(adv_emoji, "EMOJI"):
        raise RuntimeError("advertools.emoji does not look like advertools 0.13.1; memo not installed")
    adv_emoji.re = _MemoRe(_re, adv_emoji.EMOJI)
    return True


def uninstall_emoji_memo() -> None:
    """Put the real `re` back into advertools.emoji (for comparisons and tests)."""
    import advertools.emoji as adv_emoji
    if isinstance(adv_emoji.re, _MemoRe):
        adv_emoji.re = adv_emoji.re._re


def emoji_memo_active() -> bool:
    import advertools.emoji as adv_emoji
    return isinstance(adv_emoji.re, _MemoRe)


def clean_text(text: str) -> str:
    """EmTract's own clean_text, unchanged (vendored copy)."""
    return load_cleaning().clean_text(text)


def clean_text_uncaught(text, reduce_len=True, segment_words=True) -> str:
    """clean_text's steps in clean_text's order, without its bare `except:` (PI2).

    Copy of third_party/emtract/emtract/processors/cleaning.py lines 356-382 (the body of
    clean_text inside its try block). Raises whatever a step raises."""
    c = load_cleaning()
    # Fix HTML character entities:
    text = c._replace_html_entities(text)
    # Fix retext at, replace username with <user> placeholder
    text = c.remove_tags(text)
    # Remove links, dates, email addresses, images
    text = c.remove_links(text)
    # Split emojis and emoticons from words
    text = c.split_emoji_from_text(text)
    text = c.split_emoticon_from_text(text)
    # Remove company names and tickers
    text = c.remove_tickers_and_companies(text)
    # Fix contractions
    text = c.replace_contractions(text)
    if segment_words:
        text = c.correct_segments(text)
    # Normalize word lengthening
    if reduce_len:
        text = c.reduce_lengthening(text)
    check_text = text.strip().split()
    if len(check_text) < 1 or set(check_text).issubset(c.PLACEHOLDERS):
        text = ""
    return text


def clean_with_flags(text: Optional[str], cleaner: Optional[Callable[[str], str]] = None,
                     uncaught: Optional[Callable[[str], str]] = None
                     ) -> Tuple[Optional[str], bool, bool, bool, Optional[str]]:
    """Clean one text once. Returns (cleaned_text, is_missing_text, is_empty_after_clean,
    is_clean_error, clean_error).

    A missing (None) text is not cleaned: (None, True, False, False, None). A text that cleans to
    "" is re-run through `uncaught`: an exception there marks a cleaning error; "" means truly
    empty; anything else means clean_text and its copy disagree, which stops the run."""
    cleaner = cleaner or clean_text
    uncaught = uncaught or clean_text_uncaught
    if text is None:
        return None, True, False, False, None
    out = cleaner(text)
    if out != "":
        return out, False, False, False, None
    try:
        again = uncaught(text)
    except Exception as exc:  # noqa: BLE001 -- recording the error is the purpose
        return "", False, True, True, f"{type(exc).__name__}: {exc}"
    if again != "":
        raise RuntimeError("clean_text returned '' but its uncaught copy returned a non-empty text")
    return "", False, True, False, None


# ==================================================================================================
# 5. Tokens: tokenizers 0.10.3, the twitter tokenizer folder, no cutting
# ==================================================================================================
TOKENIZE_BATCH = 1024                          # texts per tokenizer call (does not change ids)


class Tokens:
    """tokenizers 0.10.3 fast tokenizer from the twitter model folder, as in EmTract's model.py."""

    def __init__(self, model_dir=None):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # one core per worker process
        import tokenizers
        import transformers
        if tokenizers.__version__ != TOKENIZERS_VERSION:
            raise RuntimeError(f"tokenizers {tokenizers.__version__}, expected {TOKENIZERS_VERSION} "
                               "(decision (6) of 29 Sep 2026)")
        transformers.logging.set_verbosity_error()   # the 'longer than 64' warning is expected: we do not cut
        self.dir = model_dir or (EMTRACT_MODEL_DIR / TOKENIZER_VARIANT)
        self.tok = transformers.DistilBertTokenizerFast.from_pretrained(str(self.dir), model_max_length=MAX_LENGTH)
        self.cls_id = int(self.tok.cls_token_id)
        self.sep_id = int(self.tok.sep_token_id)
        self.pad_id = int(self.tok.pad_token_id)
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        # word_start[id]: True unless the wordpiece continues a word ('##...').
        self.word_start = [not t.startswith("##") for t in vocab]
        self.n_vocab = len(vocab)
        self.versions = {"tokenizers": tokenizers.__version__, "transformers": transformers.__version__}

    def encode_uncut(self, texts: List[str]) -> List[List[int]]:
        out: List[List[int]] = []
        for a in range(0, len(texts), TOKENIZE_BATCH):
            enc = self.tok(texts[a:a + TOKENIZE_BATCH], add_special_tokens=False, truncation=False, padding=False)
            out.extend(enc["input_ids"])
        return out

    def encode_truncated(self, texts: List[str]) -> List[List[int]]:
        """The tokenizer's own truncation, as predict() calls it (without the padding)."""
        return self.tok(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]


# ==================================================================================================
# 6. Window plan for texts over 62 wordpieces
# ==================================================================================================
WINDOW = MAX_TEXT_TOKENS                    # 62 wordpieces per model input

OVERLAP = MIN_WINDOW_OVERLAP                # 16

MAX_STEP = WINDOW - OVERLAP                    # 46: largest distance between consecutive starts

MAX_MOVE = 8                                   # an interior start moves at most 8 wordpieces


def n_windows(n: int) -> int:
    """Number of model inputs for a text of n wordpieces (0 for an empty text)."""
    if n <= 0:
        return 0
    if n <= WINDOW:
        return 1
    return math.ceil((n - OVERLAP) / MAX_STEP)


def ideal_starts(n: int) -> List[int]:
    """Evenly spread starts from 0 to n - 62, each rounded half up (integer arithmetic)."""
    k = n_windows(n)
    if k <= 1:
        return [0] if k == 1 else []
    last = n - WINDOW
    return [(2 * j * last + (k - 1)) // (2 * (k - 1)) for j in range(k)]


def window_starts(n: int, word_start: Optional[Sequence[bool]] = None) -> List[int]:
    """Starts of the 62-wordpiece windows of a text of n wordpieces (interface item 2).

    word_start[t] tells whether wordpiece t begins a word; None switches the moves off."""
    starts = ideal_starts(n)
    k = len(starts)
    if k <= 2 or word_start is None:
        return starts
    if len(word_start) != n:
        raise ValueError(f"word_start has {len(word_start)} entries, expected {n}")
    last = n - WINDOW
    ideal = list(starts)
    for j in range(1, k - 1):
        s0 = ideal[j]
        best = None
        for d in range(0, MAX_MOVE + 1):
            for cand in ((s0,) if d == 0 else (s0 - d, s0 + d)):     # earlier candidate first
                if 0 < cand < last and word_start[cand]:
                    best = cand
                    break
            if best is not None:
                break
        if best is None or best == s0:
            continue
        prev, nxt = starts[j - 1], ideal[j + 1]          # the next start has not moved yet
        if prev < best < nxt and best - prev <= MAX_STEP and nxt - best <= MAX_STEP:
            starts[j] = best
    return starts


def check_window_plan(n: int, starts: Sequence[int]) -> None:
    """Raise ValueError unless `starts` is a valid plan for n wordpieces."""
    k = n_windows(n)
    if len(starts) != k:
        raise ValueError(f"n={n}: {len(starts)} windows, expected {k}")
    if k == 0:
        return
    if starts[0] != 0:
        raise ValueError(f"n={n}: first start {starts[0]}")
    if k == 1:
        return
    if starts[-1] != n - WINDOW:
        raise ValueError(f"n={n}: last start {starts[-1]}, expected {n - WINDOW}")
    ideal = ideal_starts(n)
    for j in range(1, k):
        if not (0 < starts[j] - starts[j - 1] <= MAX_STEP):
            raise ValueError(f"n={n}: starts {list(starts)} give an overlap below {OVERLAP} or no progress")
        if abs(starts[j] - ideal[j]) > MAX_MOVE:
            raise ValueError(f"n={n}: start {starts[j]} is more than {MAX_MOVE} from its ideal {ideal[j]}")


# ==================================================================================================
# 7. Model inputs (used by `pack` and the news scoring)
# ==================================================================================================

def model_inputs(input_ids: Sequence[int], starts: Sequence[int], cls_id: int, sep_id: int) -> List[List[int]]:
    """[CLS] + ids[s : s + 62] + [SEP] for every window start; [] for an empty or missing text.
    Window 0 is the primary input, identical to what predict() gives the model."""
    ids = list(input_ids)
    return [[cls_id] + ids[s:s + WINDOW] + [sep_id] for s in starts]


def primary_input(input_ids: Sequence[int], cls_id: int, sep_id: int) -> List[int]:
    """[CLS] + ids[:62] + [SEP] (EmTract's truncation to max_length 64)."""
    return [cls_id] + list(input_ids)[:WINDOW] + [sep_id]


# ==================================================================================================
# 8. Stage 1 for a batch of source rows (used by `tokenize`, the DLT and the news texts)
# ==================================================================================================

def stage1_rows(platform: str, cols: Dict[str, Sequence], tokens: "Tokens",
                cleaner: Optional[Callable[[str], str]] = None,
                uncaught: Optional[Callable[[str], str]] = None) -> Tuple[Dict[str, list], Dict[str, object]]:
    """Versions, cleaning, flags, uncut ids and window plans for a batch of source rows.

    Returns (columns, counts). columns holds, one entry per (live row, version): source_index,
    version, cleaned_text, is_missing_text, is_empty_after_clean, is_clean_error, clean_error,
    n_tokens, input_ids, window_starts. The caller adds row_id from source_index.
    counts never holds text: error messages are counted by exception type only."""
    idx, ver, txt, counts = build_versions(platform, cols)
    cache: Dict[str, Tuple] = {}
    cleaned: List[Optional[str]] = []
    miss: List[bool] = []
    empty: List[bool] = []
    err: List[bool] = []
    errmsg: List[Optional[str]] = []
    err_types: Dict[str, int] = {}
    for t in txt:
        if t is None:
            r = (None, True, False, False, None)
        else:
            r = cache.get(t)
            if r is None:
                r = clean_with_flags(t, cleaner, uncaught)
                cache[t] = r
        cleaned.append(r[0]); miss.append(r[1]); empty.append(r[2]); err.append(r[3]); errmsg.append(r[4])
        if r[3]:
            et = r[4].split(":", 1)[0]
            err_types[et] = err_types.get(et, 0) + 1
    uniq = sorted({c for c in cleaned if c})
    ids_of = dict(zip(uniq, tokens.encode_uncut(uniq)))
    ws = tokens.word_start
    input_ids: List[List[int]] = []
    starts: List[List[int]] = []
    n_tok: List[int] = []
    for c in cleaned:
        ids = ids_of.get(c, []) if c else []
        input_ids.append(ids)
        n = len(ids)
        n_tok.append(n)
        starts.append(window_starts(n, [ws[i] for i in ids]) if n > WINDOW else ([0] if n else []))
    by_ver = {v: ver.count(v) for v in VERSIONS}
    counts.update({
        "versions_out": len(ver), "by_version": by_ver,
        "missing_text": sum(miss), "empty_after_clean": sum(empty), "clean_error": sum(err),
        "clean_error_types": err_types,
        "nonempty_clean_zero_tokens": sum(1 for c, n in zip(cleaned, n_tok) if c and n == 0),
        "texts_cleaned_distinct": len(cache), "texts_tokenized_distinct": len(uniq),
        "over_62_tokens": sum(1 for n in n_tok if n > WINDOW),
        "model_inputs": sum(len(s) for s in starts),
        "max_n_tokens": max(n_tok) if n_tok else 0, "sum_n_tokens": sum(n_tok),
        "max_token_id": max((max(i) for i in input_ids if i), default=-1),
    })
    cols_out = {"source_index": idx, "version": ver, "cleaned_text": cleaned, "is_missing_text": miss,
                "is_empty_after_clean": empty, "is_clean_error": err, "clean_error": errmsg,
                "n_tokens": n_tok, "input_ids": input_ids, "window_starts": starts}
    return cols_out, counts

