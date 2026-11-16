"""Tests for step 06 (pipeline/06_emotion_emtract.py) and pipeline/emtract_method.py, on hand-made
texts and tiny stand-in models only (no real post is used).

Run from code/ in environment emtract-ref (EmTract's library versions):
    python tests/test_emotion_emtract.py
In environment thesis the groups that need EmTract's own stack (tokenizers 0.10.3) are skipped.
Everything is written under one temporary folder, never under code/data; the real EmTract tokenizer
and model files (data/raw/emtract_models/, see README.md) are read, never written.

Groups (each test names the rule it checks):
  A  stage 1 (`tokenize`, emtract_method): text versions per platform, the flags with a stand-in
     cleaner, the window plan (n = 0, 1, 62, 63, 100, 200, 500, word-start moves, the overlap limit,
     ties, a random property check), the model-input builder; with EmTract's stack: clean_text_uncaught
     equals clean_text whenever no error occurs, the empty / missing / error flags, the emoji memo
     changes no output, slicing the uncut ids equals the tokenizer's own truncation, and `tokenize` on a
     temporary folder (shards with the interface columns, a second run skips every shard, a changed
     shard is no longer current).
  B  stage 2 (`pack`, `score`, `collect`) on a tiny random DistilBERT: special-token ids from the
     tokenizer files, the window-plan checks, the vectorised input builder against a plain-Python
     statement, pack's batches and map, refused shards, padding buckets and order, score's order and
     values against a direct model call, collect and its checks.
  C  merge: the window plan and coverage weights, merged values on hand-made shards and scores, and
     every check that must refuse a bad input.
  D  news headlines: the headline key, the news stage 1 equals stage 1's own rows, and the four news
     subcommands end to end on a tiny CSV (with the real EmTract model, CPU).
Ported 2 Oct 2026 from tests/test_emtract_scoring.py, test_emtract_score.py, test_emtract_merge.py
and test_score_news_emtract.py (their tests of the archived bucket, Drive and monitor code were not
ported).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]          # code/
REAL_MODELS = CODE / "data" / "raw" / "emtract_models"   # EmTract's tokenizer and model files (read only)
TMP = Path(tempfile.mkdtemp(prefix="emotion_emtract_test_"))
RUN = TMP / "run"
MODELS = TMP / "models"                             # tiny stand-in tokenizer files and model (groups B, C)
os.environ.update({
    "THESIS_PIPELINE_OUT": str(TMP / "pipeline"),
    "THESIS_EMTRACT_RUN": str(RUN),
    "THESIS_EMTRACT_MODELS": str(MODELS),
    "THESIS_EMTRACT_RAW_SCORES": str(RUN / "modal_outputs"),
    "THESIS_SCRATCH": str(TMP / "scratch"),
})
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from pipeline import config, runner  # noqa: E402
from pipeline import emtract_method as E  # noqa: E402
from pipeline.emtract_method import CheckError  # noqa: E402

SCRIPT = CODE / "pipeline" / "06_emotion_emtract.py"
_spec = importlib.util.spec_from_file_location("emotion06", SCRIPT)
M06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M06)
PIPELINE_OUT = config.OUT_DIR                       # reset before every test (group C moves it)


class Skip(Exception):
    pass


# ==================================================================================================
# A. Stage 1: text versions, flags, window plan, model inputs, EmTract's stack, `tokenize`
# ==================================================================================================

def _stack():
    """(tokens) if EmTract's stack is available here, else raise Skip."""
    try:
        import tokenizers
    except ImportError:
        raise Skip("tokenizers not installed")
    if tokenizers.__version__ != E.TOKENIZERS_VERSION:
        raise Skip(f"tokenizers {tokenizers.__version__} (needs {E.TOKENIZERS_VERSION}: run in emtract-ref)")
    if not (REAL_MODELS / E.TOKENIZER_VARIANT / "tokenizer.json").exists():
        raise Skip("tokenizer files of the twitter model folder not found")
    global _TOKENS
    if _TOKENS is None:
        E.load_cleaning()
        _TOKENS = E.Tokens(REAL_MODELS / E.TOKENIZER_VARIANT)
    return _TOKENS


_TOKENS = None


# ---------------------------------------------------------------------------
# Text versions
# ---------------------------------------------------------------------------

def test_versions_reddit():
    cols = {
        "text": ["plain words", "", "> quoted bit\nmy reply", "> only a quote"],
        "own_text": ["plain words", "", "my reply", ""],
        "quoted_text": [None, None, "> quoted bit", "> only a quote"],
        "has_quote_lines": [False, False, True, True],
        "is_dead": [False, True, False, False],
    }
    idx, ver, txt, c = E.build_versions("reddit", cols)
    assert idx == [0, 2, 2, 2, 3, 3, 3], idx
    assert ver == ["own", "own", "quoted", "both", "own", "quoted", "both"], ver
    assert txt[1:4] == ["my reply", "> quoted bit", "> quoted bit\nmy reply"], "quoted keeps '>', both = original body"
    assert txt[4] == "", "a row without own text still gets an 'own' version (empty)"
    assert c == {"rows_in": 4, "rows_dead": 1, "rows_live": 3, "rows_quoted": 2}, c
    bad = dict(cols, has_quote_lines=[False, False, False, True])
    try:
        E.build_versions("reddit", bad)
    except ValueError:
        pass
    else:
        raise AssertionError("has_quote_lines disagreeing with quoted_text must stop")


def test_versions_stocktwits():
    idx, ver, txt, c = E.build_versions("stocktwits", {"message_body": ["up we go", None, ""]})
    assert idx == [0, 1, 2] and ver == ["own"] * 3 and txt == ["up we go", None, ""]
    assert c["rows_quoted"] == 0 and c["rows_dead"] == 0


def test_versions_twitter():
    cols = {"text": ["my take", "no quote here", None], "parent_text": ["their post", None, "a parent"]}
    idx, ver, txt, c = E.build_versions("twitter", cols)
    assert idx == [0, 0, 0, 1, 2, 2, 2] and ver == ["own", "quoted", "both", "own", "own", "quoted", "both"]
    assert txt[2] == "my take their post", "combined = own + single space + quoted"
    assert txt[4] is None and txt[6] is None, "a missing own text leaves own and both missing"
    assert c["rows_quoted"] == 2


# ---------------------------------------------------------------------------
# Flags with a stand-in cleaner (pure)
# ---------------------------------------------------------------------------

def test_flags_standin():
    def cleaner(t):
        return "" if t in ("gone", "boom") else t.lower()

    def uncaught(t):
        if t == "boom":
            raise KeyError("x")
        return ""
    assert E.clean_with_flags(None, cleaner, uncaught) == (None, True, False, False, None)
    assert E.clean_with_flags("Hi", cleaner, uncaught) == ("hi", False, False, False, None)
    assert E.clean_with_flags("gone", cleaner, uncaught) == ("", False, True, False, None)
    assert E.clean_with_flags("boom", cleaner, uncaught) == ("", False, True, True, "KeyError: 'x'")
    try:
        E.clean_with_flags("gone", cleaner, lambda t: "not empty")
    except RuntimeError:
        pass
    else:
        raise AssertionError("clean_text '' against a non-empty copy must stop")


# ---------------------------------------------------------------------------
# Window plan
# ---------------------------------------------------------------------------

def test_window_plan_sizes():
    assert E.window_starts(0) == [] and E.n_windows(0) == 0
    assert E.window_starts(1) == [0]
    assert E.window_starts(62) == [0], "62 wordpieces fit the primary input"
    assert E.window_starts(63) == [0, 1], "63: two windows, the last ends at the last wordpiece"
    assert E.window_starts(100) == [0, 38], "the interface example"
    assert E.window_starts(200) == [0, 46, 92, 138], E.window_starts(200)
    s500 = E.window_starts(500)
    assert len(s500) == 11 and s500[0] == 0 and s500[-1] == 438, s500
    for n in (63, 100, 108, 109, 200, 500, 1000, 5000):
        E.check_window_plan(n, E.window_starts(n))
        k = E.n_windows(n)
        assert k == -(-(n - 16) // 46), (n, k)
        assert E.n_windows(n) == 1 or all(b - a <= 46 for a, b in zip(E.window_starts(n), E.window_starts(n)[1:]))


def test_window_plan_word_starts():
    # n = 500: ideal starts 0, 44, 88, 131, ... (438 * j / 10, rounded half up)
    ideal = E.ideal_starts(500)
    assert ideal[:4] == [0, 44, 88, 131], ideal
    ws = [False] * 500
    ws[42] = True                              # 2 before the ideal 44 (88 - 42 = 46: overlap 16 kept)
    ws[91] = ws[85] = True                     # 3 after and 3 before 88: a tie, the earlier wins
    ws[131] = True                             # already a word start: stays
    ws[131 + 9 + 44] = True                    # beyond 8 from ideal[4] = 175: no move
    s = E.window_starts(500, ws)
    assert s[1] == 42 and s[2] == 85 and s[3] == 131 and s[4] == ideal[4], s[:5]
    E.check_window_plan(500, s)
    # n = 200: ideal [0, 46, 92, 138], spacing exactly 46, so any move would cut an overlap below 16:
    # the word starts at 47 (47 - 0 > 46) and 90 (138 - 90 > 46) are not used
    ws = [False] * 200
    ws[47] = True
    ws[90] = True
    s = E.window_starts(200, ws)
    assert s == [0, 46, 92, 138], s
    # n = 500 again: a nearest word start that would break the next overlap is not used (stays)
    ws = [False] * 500
    ws[41] = True                              # 88 - 41 = 47 > 46
    assert E.window_starts(500, ws)[1] == 44
    E.check_window_plan(200, s)
    # two windows: no interior start, nothing moves
    assert E.window_starts(100, [True] * 100) == [0, 38]


def test_window_plan_random_property():
    rng = random.Random(7)
    for _ in range(2000):
        n = rng.randint(63, 3000)
        ws = [rng.random() < 0.7 for _ in range(n)]
        E.check_window_plan(n, E.window_starts(n, ws))


def test_model_inputs():
    ids = list(range(1000, 1100))              # 100 made-up wordpiece ids
    starts = E.window_starts(100)
    inp = E.model_inputs(ids, starts, cls_id=7, sep_id=9)
    assert len(inp) == 2 and all(len(x) == 64 for x in inp)
    assert inp[0] == E.primary_input(ids, 7, 9) == [7] + ids[:62] + [9], "window 0 is the primary input"
    assert inp[1] == [7] + ids[38:100] + [9], "the last window ends at the last wordpiece"
    assert E.model_inputs([5, 6], [0], 7, 9) == [[7, 5, 6, 9]]
    assert E.model_inputs([], [], 7, 9) == [], "no model input for an empty text"


# ---------------------------------------------------------------------------
# EmTract's stack: cleaning copy, flags, slicing = truncation
# ---------------------------------------------------------------------------

HAND_MADE = [
    "Hello @someone, $BTC to the moon!!! https://example.com/x",
    "I don't think so... 100k by Friday? #crypto",
    "&gt; a quoted line\nmy own reply &amp; more",
    "soooooo good \U0001F680\U0001F680",
    "[removed]", "", "   ", "http://only.a/link", "@user", "$BTC $ETH",
    "wow!!!!! wow wow wow wow", "email me at someone@example.com please", "12:30 meeting",
    "caf\u00e9 na\u00efve \u2019quotes\u2019 \u2014 dash", "u/someone said :) and :(",
]


def test_clean_uncaught_equals_clean_text():
    _stack()
    for t in HAND_MADE:
        assert E.clean_text_uncaught(t) == E.clean_text(t), "the copy equals clean_text when no error occurs"


def test_clean_flags_real():
    _stack()
    assert E.clean_with_flags("[removed]")[:4] == ("", False, True, False), "a bracketed placeholder is truly empty"
    assert E.clean_with_flags("http://only.a/link")[:4] == ("", False, True, False)
    assert E.clean_with_flags("")[:4] == ("", False, True, False), "an empty string is empty, not an error"
    assert E.clean_with_flags(None) == (None, True, False, False, None), "a NULL text is missing, not cleaned"
    # an image file name alone: the image rule leaves "", and advertools' extract_emoji then divides
    # by zero; clean_text's catch-all hides this, the re-check records it (seen on real rows, 29 Sep)
    c, m, e, r, msg = E.clean_with_flags("photo.png")
    assert (c, e, r) == ("", True, True) and msg.startswith("ZeroDivisionError"), msg
    c, m, e, r, msg = E.clean_with_flags(12345)            # non-text input: clean_text's catch-all fires
    assert (c, e, r) == ("", True, True) and msg.startswith("TypeError:"), msg
    c = E.clean_with_flags(HAND_MADE[0])[0]
    assert c and "<ticker>" in c and "<user>" in c, "EmTract's placeholders"


def test_emoji_memo_identical():
    _stack()
    texts = HAND_MADE + ["\U0001F600\U0001F600 ok \u2764\ufe0f", "a\U0001F680b", "\U0001F468\u200d\U0001F469\u200d\U0001F467 family",
                         "photo.png", "\U0001F44D\U0001F3FD thumbs", ":) :( ;) <3"]
    E.uninstall_emoji_memo()
    base = [E.clean_with_flags(t) for t in texts]
    E.install_emoji_memo()
    try:
        assert E.emoji_memo_active()
        assert [E.clean_with_flags(t) for t in texts] == base, "the memo changes no cleaned text and no flag"
        assert [E.clean_with_flags(t) for t in texts] == base, "also when the memo is warm"
    finally:
        E.uninstall_emoji_memo()
    assert not E.emoji_memo_active()


def test_slicing_equals_truncation():
    tok = _stack()
    rng = random.Random(3)
    words = ["moon", "hodl", "unbelievably", "tokenomics", "the", "a", "price", "!", "dip", "\U0001F680",
             "rekt", "whales", "bullish", "because", "interoperability"]
    texts = [" ".join(rng.choice(words) for _ in range(rng.randint(1, 120))) for _ in range(200)]
    cleaned = [E.clean_text(t) for t in texts]
    cleaned = [c for c in cleaned if c]
    uncut = tok.encode_uncut(cleaned)
    trunc = tok.encode_truncated(cleaned)
    assert any(len(u) > 62 for u in uncut), "the sample has texts over 62 wordpieces"
    for u, t in zip(uncut, trunc):
        assert E.primary_input(u, tok.cls_id, tok.sep_id) == t
    assert tok.cls_id != tok.sep_id and tok.cls_id == tok.tok.convert_tokens_to_ids("[CLS]")


def test_stage1_rows():
    tok = _stack()
    cols = {"text": ["my take on it", "the moon and back " * 20, "[removed]"],
            "parent_text": ["their words here", None, None]}
    out, c = E.stage1_rows("twitter", cols, tok)
    assert out["version"] == ["own", "quoted", "both", "own", "own"]
    assert out["is_empty_after_clean"] == [False, False, False, False, True]
    assert out["n_tokens"][4] == 0 and out["window_starts"][4] == []
    assert out["n_tokens"][3] > 62 and len(out["window_starts"][3]) == E.n_windows(out["n_tokens"][3])
    assert all(n == len(i) for n, i in zip(out["n_tokens"], out["input_ids"]))
    assert c["by_version"] == {"own": 3, "quoted": 1, "both": 1} and c["empty_after_clean"] == 1


# ---------------------------------------------------------------------------
# Step 06 on a temporary folder: write, resume, skip
# ---------------------------------------------------------------------------

def test_step06_resume():
    _stack()
    import pyarrow as pa
    import pyarrow.parquet as pq
    step = M06
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n = 45
        own = [f"post number {i} says hodl" if i % 7 else "" for i in range(n)]
        quoted = [f"> earlier post {i}" if i % 5 == 0 else None for i in range(n)]
        text = [(q + "\n" + o) if q else o for q, o in zip(quoted, own)]
        dead = [i % 11 == 3 for i in range(n)]
        text = ["" if d else t for d, t in zip(dead, text)]
        own = ["" if d else t for d, t in zip(dead, own)]
        quoted = [None if d else q for d, q in zip(dead, quoted)]
        tab = pa.table({"row_id": [f"r{i}" for i in range(n)], "text": text, "own_text": own,
                        "quoted_text": quoted, "has_quote_lines": [q is not None for q in quoted], "is_dead": dead})
        src = tmp / "handmade.parquet"
        pq.write_table(tab, src, row_group_size=20)             # 3 row groups: 20, 20, 5
        out = tmp / "out"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", THESIS_EMTRACT_MODELS=str(REAL_MODELS))
        cmd = [sys.executable, str(SCRIPT), "tokenize", "reddit", "--input-file", str(src),
               "--partition-name", "TEST", "--out-root", str(out), "--workers", "2"]
        r = subprocess.run(cmd, cwd=CODE, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        shards = sorted((out / "reddit" / "emtract" / "tokens" / "TEST").glob("shard-*.parquet"))
        assert [p.name for p in shards] == ["shard-00000.parquet", "shard-00001.parquet", "shard-00002.parquet"]
        t = pa.concat_tables([pq.read_table(p) for p in shards])
        assert t.column_names == ["row_id", "version", "cleaned_text", "is_missing_text", "is_empty_after_clean",
                                  "is_clean_error", "clean_error", "n_tokens", "input_ids", "window_starts"]
        assert str(t.schema.field("input_ids").type) == "list<item: uint16>"
        assert str(t.schema.field("window_starts").type) == "list<item: int32>"
        assert str(t.schema.field("n_tokens").type) == "int32"
        n_dead, n_quote = sum(dead), sum(q is not None for q in quoted)
        assert t.num_rows == (n - n_dead) + 2 * n_quote, t.num_rows
        rows = t.to_pydict()
        assert "r3" not in rows["row_id"], "dead rows get no row"
        assert all(e for e, rid, v in zip(rows["is_empty_after_clean"], rows["row_id"], rows["version"])
                   if v == "own" and int(rid[1:]) % 7 == 0), "rows without own text are flagged empty"
        st = json.loads((shards[0].with_name(shards[0].name + ".stats.json")).read_text())
        assert st["row_id_source_column"] == "id" and all(c["passed"] for c in st["checks"])
        assert st["counts"]["rows_in"] == 20 and st["counts"]["rows_in"] == st["counts"]["rows_live"] + st["counts"]["rows_dead"]
        status = json.loads((out / "_emtract_run" / "status_stage1.json").read_text())
        assert status["partitions"]["TEST"]["shards_done"] == 3
        mtimes = [p.stat().st_mtime_ns for p in shards]
        # second run: every shard is current and skipped, nothing is rewritten
        r2 = subprocess.run(cmd, cwd=CODE, env=env, capture_output=True, text=True)
        assert r2.returncode == 0 and "0 shards to do, 3 current and skipped" in r2.stdout, r2.stdout[-1500:]
        assert [p.stat().st_mtime_ns for p in shards] == mtimes
        # a changed shard file is no longer current; the others still are
        inputs = step.runner.describe_inputs([src])
        code_sha = step.code_fingerprint()
        rules_sha, _ = step.rules_fingerprint(REAL_MODELS / E.TOKENIZER_VARIANT)
        paths = step.Paths("reddit", str(out))
        specs = E.plan_shards("reddit", "TEST", src.resolve(), None)
        assert all(step.shard_is_current(step.shard_file(paths, s), s, inputs, code_sha, rules_sha) for s in specs)
        with open(shards[1], "ab") as fh:
            fh.write(b"x")
        cur = [step.shard_is_current(step.shard_file(paths, s), s, inputs, code_sha, rules_sha) for s in specs]
        assert cur == [True, False, True], cur
        # a different range (a pilot's --limit-rows) is not current either
        s0 = dict(specs[0], rg_stop=10, row_stop=10)
        assert not step.shard_is_current(step.shard_file(paths, s0), s0, inputs, code_sha, rules_sha)


# ==================================================================================================
# B. Stage 2: pack, score, collect on a tiny random model
# ==================================================================================================

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(5, 100)]
ADDED = {"<emoji-a>": 100}              # stands in for EmTract's added (emoji) tokens
CLS, SEP, PAD = 2, 3, 0


def _tokenizer_files(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    (d / "added_tokens.json").write_text(json.dumps(ADDED), encoding="utf-8")
    (d / "special_tokens_map.json").write_text(json.dumps(
        {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
         "mask_token": "[MASK]"}), encoding="utf-8")


def _tiny_model(d: Path) -> None:
    import torch
    import transformers
    _tokenizer_files(d)
    cfg = transformers.DistilBertConfig(vocab_size=101, dim=32, n_layers=2, n_heads=2, hidden_dim=64,
                                        max_position_embeddings=64, num_labels=7, pad_token_id=PAD)
    torch.manual_seed(0)
    transformers.DistilBertForSequenceClassification(cfg).save_pretrained(str(d))


def s2_plan(n: int):
    """A valid window plan (even spread, no word-start snapping) for the hand-made shards."""
    k = M06.expected_n_windows(n)
    if k <= 1:
        return [0] * k
    return [round(j * (n - 62) / (k - 1)) for j in range(k)]


def s2_write_shard(platform: str, partition: str, name: str, lengths, seed: int = 0, with_stats: bool = True):
    """A stage-1 shard with the interface columns; length None = missing text, 0 = empty after cleaning."""
    rng = np.random.default_rng(seed)
    rows = {"row_id": [], "version": [], "cleaned_text": [], "is_missing_text": [], "is_empty_after_clean": [],
            "is_clean_error": [], "clean_error": [], "n_tokens": [], "input_ids": [], "window_starts": []}
    for i, n in enumerate(lengths):
        ids = [] if not n else rng.integers(5, 101, size=n).tolist()
        rows["row_id"].append(1000 + i)
        rows["version"].append("own")
        rows["cleaned_text"].append(None if n is None else "hand made")
        rows["is_missing_text"].append(n is None)
        rows["is_empty_after_clean"].append(n == 0)
        rows["is_clean_error"].append(False)
        rows["clean_error"].append(None)
        rows["n_tokens"].append(int(n or 0))
        rows["input_ids"].append(ids)
        rows["window_starts"].append(s2_plan(int(n or 0)))
    t = pa.table({**{k: v for k, v in rows.items() if k not in ("n_tokens", "input_ids", "window_starts")},
                  "n_tokens": pa.array(rows["n_tokens"], pa.int32()),
                  "input_ids": pa.array(rows["input_ids"], pa.list_(pa.uint16())),
                  "window_starts": pa.array(rows["window_starts"], pa.list_(pa.int32()))})
    p = E.tokens_dir(platform) / partition / name
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, p)
    if with_stats:
        runner.write_stats(p, step="06_test", inputs=[], code_sha="x", rules_sha="x", con=None,
                           counts={"rows": t.num_rows}, checks=[{"check": "test", "passed": True}])
    return p, t


def expect_error(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except M06.CheckError:
        return
    raise AssertionError(f"{fn.__name__} did not raise CheckError")


# ---------------------------------------------------------------------------

def test_special_ids_from_files():
    d = TMP / "tok_only"
    _tokenizer_files(d)
    sp = M06.special_token_ids(d)
    assert (sp["cls"], sp["sep"], sp["pad"]) == (CLS, SEP, PAD), sp
    assert sp["vocab_size"] == 101
    # moving [CLS] in the vocabulary moves the id: nothing is hard-coded
    v = VOCAB.copy()
    v[2], v[7] = v[7], v[2]
    (d / "vocab.txt").write_text("\n".join(v) + "\n", encoding="utf-8")
    assert M06.special_token_ids(d)["cls"] == 7


def test_window_plan_checks():
    assert [M06.expected_n_windows(n) for n in (0, 1, 62, 63, 100, 108, 109, 150)] == [0, 1, 1, 2, 2, 2, 3, 3]
    assert M06.check_window_starts(100, [0, 38]) is None          # the interface's example
    assert M06.check_window_starts(150, s2_plan(150)) is None
    assert M06.check_window_starts(62, [0]) is None
    assert M06.check_window_starts(100, [0]) is not None           # too few windows
    assert M06.check_window_starts(100, [0, 37]) is not None       # last window does not end at n
    assert M06.check_window_starts(150, [0, 60, 88]) is not None   # overlap 2 < 16
    assert M06.check_window_starts(150, [1, 44, 88]) is not None   # first start not 0


def test_build_inputs_matches_reference():
    rng = np.random.default_rng(1)
    lens = [1, 5, 62, 63, 100, 150, 200]
    seqs = [rng.integers(5, 101, size=n).tolist() for n in lens]
    offsets = np.zeros(len(lens) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    values = np.array(sum(seqs, []), np.uint16)
    rows, starts = [], []
    for r, n in enumerate(lens):
        for s in s2_plan(n):
            rows.append(r)
            starts.append(s)
    ids, off = M06.build_inputs(values, offsets, np.array(rows), np.array(starts), CLS, SEP)
    for i, (r, s) in enumerate(zip(rows, starts)):
        assert ids[off[i]:off[i + 1]].tolist() == M06.reference_input(seqs[r], s, CLS, SEP), (r, s)
    assert M06.window_numbers(np.array(rows)).tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 3]


def test_pack_and_map():
    _tokenizer_files(E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT)
    lens = [None, 0, 1, 5, 30, 62, 63, 100, 150]
    p1, t1 = s2_write_shard("reddit", "BTC", "shard-00000.parquet", lens, seed=2)
    s2_write_shard("reddit", "BTC", "shard-00001.parquet", [10, 20], seed=3, with_stats=False)   # unfinished
    res = M06.pack("reddit", "b1", max_inputs=1000)
    n_expect = sum(len(s2_plan(int(n or 0))) for n in lens)
    assert res["inputs_file"].endswith("b1.parquet") and res["shards"] == 1 and res["inputs"] == n_expect
    inp = pq.read_table(E.PACKED_INPUTS_DIR / "reddit" / "b1.parquet")
    mp = pq.read_table(E.PACKED_INPUTS_DIR / "reddit" / "b1.map.parquet")
    assert inp.column_names == ["seq", "input_ids"]                     # nothing else leaves the Mac
    assert inp.schema.field("input_ids").type == pa.list_(pa.uint16())
    assert mp.column_names == ["seq", "shard", "shard_row", "window"]
    assert mp.schema.field("shard_row").type == pa.int32() and mp.schema.field("window").type == pa.int16()
    assert inp.num_rows == mp.num_rows == n_expect
    assert inp.column("seq").to_pylist() == list(range(n_expect))
    got = inp.column("input_ids").to_pylist()
    ids = t1.column("input_ids").to_pylist()
    for s, r, w, x in zip(mp.column("seq").to_pylist(), mp.column("shard_row").to_pylist(),
                          mp.column("window").to_pylist(), got):
        assert x == M06.reference_input(ids[r], s2_plan(len(ids[r]))[w], CLS, SEP)
    assert set(mp.column("shard_row").to_pylist()) == set(range(2, len(lens)))     # rows 0, 1 (missing, empty): no input
    assert mp.column("shard").to_pylist()[0] == "BTC/shard-00000.parquet"
    st = json.loads((E.PACKED_INPUTS_DIR / "reddit" / "b1.map.parquet.stats.json").read_text())
    assert st["shards"][0]["md5"] == runner.file_md5(p1) and all(c["passed"] for c in st["checks"])
    # nothing left to pack: the only finished shard is packed, the other has no stats
    expect_error(M06.pack, "reddit", "b2", max_inputs=1000)
    # a shard whose md5 no longer matches its stats is not finished
    p3, _ = s2_write_shard("reddit", "ETH", "shard-00000.parquet", [4, 70], seed=4)
    with open(p3, "ab") as fh:
        fh.write(b"x")
    expect_error(M06.pack, "reddit", "b2", max_inputs=1000)
    # a new finished shard is packed alone; one that exceeds --max-inputs is refused
    s2_write_shard("reddit", "XRP", "shard-00000.parquet", [4, 70, 200], seed=5)
    expect_error(M06.pack, "reddit", "b2", max_inputs=3)
    res2 = M06.pack("reddit", "b2", max_inputs=7)
    assert res2["shards"] == 1 and res2["inputs"] == 7 and pq.read_metadata(E.PACKED_INPUTS_DIR / "reddit" / "b2.parquet").num_rows == 1 + 2 + 4
    # the same batch id is never written twice; a held lock refuses a second writer
    expect_error(M06.pack, "reddit", "b2", max_inputs=7)
    s2_write_shard("reddit", "DOGE", "shard-00000.parquet", [4], seed=6)
    lock = E.PACKED_INPUTS_DIR / "reddit" / ".pack.lock"
    lock.write_text("held")
    expect_error(M06.pack, "reddit", "b3", max_inputs=100)
    lock.unlink()


def test_invalid_shards_refused():
    _tokenizer_files(E.EMTRACT_MODEL_DIR / E.TOKENIZER_VARIANT)
    p, t = s2_write_shard("stocktwits", "2020-01", "shard-00000.parquet", [100], seed=7)
    bad = t.set_column(t.column_names.index("window_starts"), "window_starts",
                       pa.array([[0]], pa.list_(pa.int32())))          # a 100-token text needs 2 windows
    pq.write_table(bad, p)
    runner.write_stats(p, step="06_test", inputs=[], code_sha="x", rules_sha="x", con=None,
                       counts={}, checks=[{"check": "test", "passed": True}])
    expect_error(M06.pack, "stocktwits", "s1", max_inputs=100)
    bad2 = t.set_column(t.column_names.index("n_tokens"), "n_tokens", pa.array([99], pa.int32()))
    pq.write_table(bad2, p)
    runner.write_stats(p, step="06_test", inputs=[], code_sha="x", rules_sha="x", con=None,
                       counts={}, checks=[{"check": "test", "passed": True}])
    expect_error(M06.pack, "stocktwits", "s1", max_inputs=100)
    assert not (E.PACKED_INPUTS_DIR / "stocktwits" / "s1.parquet").exists()


def test_buckets_padding_order():
    assert M06.bucket_of(np.array([3, 16, 17, 32, 33, 48, 49, 64])).tolist() == [16, 16, 32, 32, 48, 48, 64, 64]
    expect_error(M06.bucket_of, np.array([65]))
    lengths = np.array([40, 3, 64, 3, 17, 16])
    seq = np.array([0, 1, 2, 3, 4, 5])
    batches = M06.plan_batches(lengths, seq, batch_tokens=32)          # 2 per batch at 16, 1 at 32 or more
    flat = [(b, i.tolist()) for b, i in batches]
    assert flat == [(16, [1, 3]), (16, [5]), (32, [4]), (48, [0]), (64, [2])], flat
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    values = np.arange(int(offsets[-1])) % 90 + 5
    ids, mask = M06.pad_batch(values, offsets, np.array([4, 1]), 32, PAD)
    assert ids.shape == mask.shape == (2, 32)
    assert mask[0].sum() == 17 and mask[1].sum() == 3 and (ids[1, 3:] == PAD).all()
    assert ids[0, :17].tolist() == values[offsets[4]:offsets[5]].tolist()


def test_score_order_and_values():
    import torch
    import transformers
    mdir = E.EMTRACT_MODEL_DIR / E.MODEL_VARIANT
    _tiny_model(mdir)
    inp = E.PACKED_INPUTS_DIR / "reddit" / "b1.parquet"
    out = TMP / "raw" / "b1.parquet"
    rec = M06.score(inp, out, "cpu", model_dir=mdir, batch_tokens=128)
    # save_pretrained writes model.safetensors under transformers 5.x and pytorch_model.bin under
    # 4.10.3 (env emtract-ref); the record must name the one file the installed version wrote
    saved = [f for f in ("pytorch_model.bin", "model.safetensors") if (mdir / f).exists()]
    assert len(saved) == 1, saved
    assert all(c["passed"] for c in rec["checks"]) and rec["weights_file"] == saved[0], (rec["weights_file"], saved)
    sc = pq.read_table(out)
    n = sc.num_rows
    assert sc.column("seq").to_pylist() == list(range(n))
    assert sc.column_names == ["seq"] + [f"logit_{l}" for l in E.LABELS] + [f"p_{l}" for l in E.LABELS]
    assert all(sc.schema.field(c).type == pa.float32() for c in sc.column_names[1:])
    # one input at a time, unpadded, must give the same numbers at the same seq
    model = transformers.DistilBertForSequenceClassification.from_pretrained(str(mdir)).eval()
    xs = pq.read_table(inp).column("input_ids").to_pylist()
    L = np.column_stack([sc.column(f"logit_{l}").to_numpy() for l in E.LABELS])
    with torch.no_grad():
        for i, x in enumerate(xs):
            ref = model(input_ids=torch.tensor([x])).logits.numpy()[0]
            assert np.abs(ref - L[i]).max() < 1e-5, (i, np.abs(ref - L[i]).max())
    # the row order of the inputs file does not matter: output follows seq
    t = pq.read_table(inp)
    perm = np.random.default_rng(8).permutation(t.num_rows)
    shuffled = TMP / "raw" / "shuffled.parquet"
    pq.write_table(t.take(pa.array(perm)), shuffled)
    M06.score(shuffled, TMP / "raw" / "shuffled_out.parquet", "cpu", model_dir=mdir, batch_tokens=128)
    sc2 = pq.read_table(TMP / "raw" / "shuffled_out.parquet")
    assert sc2.column("seq").to_pylist() == list(range(n))
    L2 = np.column_stack([sc2.column(f"logit_{l}").to_numpy() for l in E.LABELS])
    assert np.abs(L - L2).max() < 1e-5


def test_collect_and_its_checks():
    mdir = E.EMTRACT_MODEL_DIR / E.MODEL_VARIANT
    raw_dir = E.RAW_SCORES_DIR / "reddit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    src = TMP / "raw" / "b1.parquet"
    shutil.copy(src, raw_dir / "b1.parquet")
    shutil.copy(str(src) + ".stats.json", raw_dir / "b1.parquet.stats.json")
    counts = M06.collect("reddit", "b1")
    final = E.scores_dir("reddit") / "b1.parquet"
    t = pq.read_table(final)
    assert t.column_names[:4] == ["seq", "shard", "shard_row", "window"] and counts["inputs"] == t.num_rows
    assert json.loads((E.scores_dir("reddit") / "b1.parquet.stats.json").read_text())["score"]["dtype"] == "float32"
    expect_error(M06.collect, "reddit", "b1")                               # never written twice
    # batch b2: raw scores with one row missing, then with a wrong probability sum
    M06.score(E.PACKED_INPUTS_DIR / "reddit" / "b2.parquet", raw_dir / "b2.parquet", "cpu", model_dir=mdir)
    good = pq.read_table(raw_dir / "b2.parquet")
    pq.write_table(good.slice(0, good.num_rows - 1), raw_dir / "b2.parquet")
    expect_error(M06.collect, "reddit", "b2")
    col = good.column_names.index("p_happy")
    pq.write_table(good.set_column(col, "p_happy", pc_add(good.column("p_happy"), 0.01)), raw_dir / "b2.parquet")
    expect_error(M06.collect, "reddit", "b2")
    # scores computed on other inputs
    pq.write_table(good, raw_dir / "b2.parquet")
    st = json.loads((raw_dir / "b2.parquet.stats.json").read_text())
    st_bad = dict(st, inputs_sha256="0" * 64)
    (raw_dir / "b2.parquet.stats.json").write_text(json.dumps(st_bad))
    expect_error(M06.collect, "reddit", "b2")
    # a shard changed after packing
    (raw_dir / "b2.parquet.stats.json").write_text(json.dumps(st))
    shard = E.tokens_dir("reddit") / "XRP" / "shard-00000.parquet"
    keep = shard.read_bytes()
    shard.write_bytes(keep + b"x")
    expect_error(M06.collect, "reddit", "b2")
    shard.write_bytes(keep)
    assert not (E.scores_dir("reddit") / "b2.parquet").exists()
    M06.collect("reddit", "b2")
    assert pq.read_metadata(E.scores_dir("reddit") / "b2.parquet").num_rows == good.num_rows


def pc_add(col, x):
    import pyarrow.compute as pc
    return pc.cast(pc.add(col, pa.scalar(x, pa.float32())), pa.float32())


# ---------------------------------------------------------------------------
# Bucket route (no network)
# ---------------------------------------------------------------------------


# ==================================================================================================
# C. Merge: window plan, coverage weights, merged values, refusals
# ==================================================================================================

LAB = E.LABELS
W = E.MAX_TEXT_TOKENS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def merge_setup() -> None:
    """A stand-in SHA256SUMS and empty-text score for the merge (load_empty_score checks both)."""
    d = MODELS / E.MODEL_VARIANT
    d.mkdir(parents=True, exist_ok=True)
    (d / "SHA256SUMS").write_text("ab" * 32 + "  pytorch_model.bin\n")
    RUN.mkdir(parents=True, exist_ok=True)
    runner.write_json_atomic(E.EMPTY_SCORE_FILE, {
        "model_variant": E.MODEL_VARIANT, "model_weights_sha256": "ab" * 32,
        "p": {lab: 1.0 / 7 for lab in LAB}, "top_emotion": "neutral", "note": "hand-made for the test"})


def fresh_out(name: str) -> Path:
    out = TMP / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    config.OUT_DIR = out
    return out


def mg_plan(n: int):
    if n <= 0:
        return []
    if n <= W:
        return [0]
    k = int(M06.n_windows_for([n])[0])
    return [int(math.floor(j * (n - W) / (k - 1) + 0.5)) for j in range(k)]


def token_row(rid, version="own", n=5, missing=False, empty=False, cerr=False, starts=None):
    return {"row_id": rid, "version": version, "cleaned_text": None if missing else ("" if n == 0 else "x"),
            "is_missing_text": missing, "is_empty_after_clean": empty, "is_clean_error": cerr,
            "clean_error": "ValueError: made up" if cerr else None, "n_tokens": n,
            "input_ids": [1000 + (i % 50) for i in range(n)], "window_starts": mg_plan(n) if starts is None else starts}


TOKEN_SCHEMA = pa.schema([("row_id", pa.string()), ("version", pa.string()), ("cleaned_text", pa.string()),
                          ("is_missing_text", pa.bool_()), ("is_empty_after_clean", pa.bool_()),
                          ("is_clean_error", pa.bool_()), ("clean_error", pa.string()), ("n_tokens", pa.int32()),
                          ("input_ids", pa.list_(pa.uint16())), ("window_starts", pa.list_(pa.int32()))])


def mg_write_shard(platform, partition, idx, rows, stats=True):
    f = E.tokens_dir(platform) / partition / f"shard-{idx:05d}.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TOKEN_SCHEMA), f)
    if stats:
        runner.write_stats(f, step="06_tokenize_emtract", inputs=[], code_sha="c", rules_sha="r", con=None,
                           counts={"rows_in": len({r["row_id"] for r in rows}), "rows_dead": 1,
                                   "rows_written": len(rows)},
                           checks=[{"check": "made up", "passed": True}], extra={"row_id_source_column": "id"})
    return f


def probs(main: int, hi=0.4):
    p = [(1 - hi) / 6] * 7
    p[main] = hi
    return p


def score_rows(platform, shard_path, prob_of=None):
    """One score row per expected input of a shard; prob_of(shard_row, window) -> 7 probabilities."""
    t = pq.read_table(shard_path)
    rel = shard_path.relative_to(E.tokens_dir(platform)).as_posix()
    rows = []
    for i, ws in enumerate(t.column("window_starts").to_pylist()):
        for j, _ in enumerate(ws):
            p = prob_of(i, j) if prob_of else probs((i + j) % 7)
            rows.append({"shard": rel, "shard_row": i, "window": j, "p": p})
    return rows


def write_scores(platform, batch, rows, stats=True, perturb=0.0):
    cols = {"seq": list(range(len(rows))), "shard": [r["shard"] for r in rows],
            "shard_row": pa.array([r["shard_row"] for r in rows], pa.int32()),
            "window": pa.array([r["window"] for r in rows], pa.int16())}
    P = np.array([r["p"] for r in rows], dtype=np.float64).reshape(len(rows), 7)
    for j, lab in enumerate(LAB):
        cols[f"logit_{lab}"] = pa.array(np.log(P[:, j]).astype(np.float32))
    for j, lab in enumerate(LAB):
        cols[f"p_{lab}"] = pa.array((P[:, j] + (perturb if j == 0 else 0.0)).astype(np.float32))
    f = E.scores_dir(platform) / f"{batch}.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), f)
    if stats:
        runner.write_stats(f, step="07_score_emtract", inputs=[], code_sha="c", rules_sha="r", con=None,
                           counts={"rows": len(rows)}, checks=[{"check": "made up", "passed": True}])
    return f


def expect_fail(fn, text):
    try:
        fn()
    except CheckError as exc:
        assert text in str(exc), f"expected '{text}' in: {exc}"
        return
    raise AssertionError(f"no CheckError (expected '{text}')")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
HAND = {  # (shard, shard row) -> probabilities per window (0/2: n = 100, 0/4: n = 124, 1/0: n = 63)
    (0, 2): [probs(0), probs(1)],
    (0, 4): [probs(0), probs(1), probs(2)],
    (1, 0): [probs(3), probs(4)],
}


def build_happy(platform="reddit"):
    s0 = mg_write_shard(platform, "BTC", 0, [
        token_row("a1", n=5),
        token_row("a2", n=0, empty=True),
        token_row("a3", "own", n=100), token_row("a3", "quoted", n=10), token_row("a3", "both", n=124),
        token_row("a4", n=0, missing=True),
        token_row("a5", n=0, empty=True, cerr=True),
    ])
    s1 = mg_write_shard(platform, "BTC", 1, [token_row("a6", n=63), token_row("a7", n=62)])
    pick = lambda shard: (lambda i, j: HAND[(shard, i)][j] if (shard, i) in HAND else probs(i % 7, 0.7))
    r0, r1 = score_rows(platform, s0, pick(0)), score_rows(platform, s1, pick(1))
    write_scores(platform, "batch-00000", r0 + r1[:1])       # shard 1 split over two score files
    write_scores(platform, "batch-00001", r1[1:])
    return s0, s1


def test_window_plan_and_weights():
    merge_setup()
    assert [int(x) for x in M06.n_windows_for([0, 1, 62, 63, 100, 108, 109, 124, 200])] == [0, 1, 1, 2, 2, 2, 3, 3, 4]
    assert mg_plan(100) == [0, 38]
    for n, starts, want in [(63, (0, 1), [31.5, 31.5]), (100, (0, 38), [50, 50]), (108, (0, 46), [54, 54]),
                            (124, (0, 31, 62), [46.5, 31, 46.5]), (200, (0, 46, 92, 138), [54, 46, 46, 54])]:
        assert np.allclose(M06.coverage_weights(n, starts), want), (n, M06.coverage_weights(n, starts))
    w300 = M06.coverage_weights(300, (0, 40, 79, 119, 159, 198, 238))
    assert abs(w300[0] - 51) < 1e-9 and abs(w300[-1] - 51) < 1e-9 and abs(w300.sum() - 300) < 1e-9
    assert M06.window_plan_error(124, [0, 31, 62]) is None
    assert M06.window_plan_error(124, [0, 38, 62]) is None                       # moved by 7: allowed
    assert "ideal" in M06.window_plan_error(124, [0, 42, 62])                    # moved 11 from 31
    assert "overlap" in M06.window_plan_error(124, [0, 31, 90]) or "end" in M06.window_plan_error(124, [0, 31, 90])
    assert M06.window_plan_error(100, [0, 38, 38]) is not None
    assert M06.window_plan_error(50, [0, 10]) is not None


def test_merge_values():
    merge_setup()
    fresh_out("m_happy")
    build_happy()
    res = M06.merge("reddit", None, False)
    t = pq.read_table(E.emotion_dir("reddit") / "BTC.parquet").to_pylist()
    assert [r["row_id"] for r in t] == ["a1", "a2", "a3", "a3", "a3", "a4", "a5", "a6", "a7"]
    assert [r["version"] for r in t][2:5] == ["own", "quoted", "both"]
    assert [r["n_windows"] for r in t] == [1, 0, 2, 1, 3, 0, 0, 2, 1]
    for r in t:
        if r["n_windows"] == 0:
            assert all(r[f"{x}_{lab}"] is None for x in ("p", "pc", "pw") for lab in LAB)
        elif r["n_windows"] == 1:
            assert r["p_neutral"] is not None and all(r[f"pc_{lab}"] is None and r[f"pw_{lab}"] is None for lab in LAB)
    a1 = t[0]
    assert abs(a1["p_neutral"] - 0.7) < 1e-6                                      # row 0: probs(0, 0.7)
    own, both, a6 = t[2], t[4], t[7]
    # n = 100, windows probs(0), probs(1): p = window 0; pc = pw = mean (weights 50, 50)
    assert abs(own["p_neutral"] - 0.4) < 1e-6 and abs(own["p_happy"] - 0.1) < 1e-6
    for x in ("pc", "pw"):
        assert abs(own[f"{x}_neutral"] - 0.25) < 1e-6 and abs(own[f"{x}_happy"] - 0.25) < 1e-6
        assert abs(own[f"{x}_sad"] - 0.1) < 1e-6
    # n = 124, windows probs(0), probs(1), probs(2); weights 46.5, 31, 46.5
    assert abs(both["pc_neutral"] - 0.2) < 1e-6 and abs(both["pc_happy"] - 0.2) < 1e-6 and abs(both["pc_sad"] - 0.2) < 1e-6
    assert abs(both["pw_neutral"] - 26.35 / 124) < 1e-6 and abs(both["pw_happy"] - 21.7 / 124) < 1e-6
    assert abs(both["pw_sad"] - 26.35 / 124) < 1e-6 and abs(both["pw_anger"] - 0.1) < 1e-6
    # n = 63, windows probs(3), probs(4); weights 31.5, 31.5
    assert abs(a6["p_anger"] - 0.4) < 1e-6 and abs(a6["pc_anger"] - 0.25) < 1e-6 and abs(a6["pw_disgust"] - 0.25) < 1e-6
    st = json.loads(runner.stats_path(E.emotion_dir("reddit") / "BTC.parquet").read_text())
    c = st["counts"]
    assert c["rows"] == 9 and c["scored_rows"] == 6 and c["long_rows"] == 3 and c["model_inputs"] == 10
    assert c["rows_own"] == 7 and c["rows_quoted"] == 1 and c["rows_both"] == 1
    assert c["empty_after_clean"] == 2 and c["missing_text"] == 1 and c["clean_error"] == 1
    assert c["windows_1"] == 3 and c["windows_2"] == 2 and c["windows_3"] == 1
    assert st["empty_text_score"]["model_variant"] == E.MODEL_VARIANT and len(st["empty_text_score"]["p"]) == 7
    assert st["row_id_source"] == "id" and st["stage1_counts_summed"]["rows_written"] == 9
    assert all(x["passed"] for x in st["checks"]) and len(st["inputs"]) == 4
    assert res["BTC"]["rows"] == 9
    assert M06.merge("reddit", None, False) == {}                                  # current: skipped
    assert M06.merge("reddit", ["BTC"], True)["BTC"]["rows"] == 9                  # --force


def _variant(name, mutate_scores=None, token_rows=None, stats=True, perturb=0.0, extra_scores=None):
    fresh_out(name)
    s0 = mg_write_shard("reddit", "BTC", 0, token_rows or [token_row("b1", n=5), token_row("b2", n=100),
                                                       token_row("b3", n=0, empty=True)], stats=stats)
    rows = score_rows("reddit", s0)
    if mutate_scores:
        rows = mutate_scores(rows)
    write_scores("reddit", "batch-00000", rows + (extra_scores or []), perturb=perturb)
    return lambda: M06.merge("reddit", None, False)


def test_merge_checks():
    merge_setup()
    expect_fail(_variant("m_missing", lambda r: r[:-1]), "inputs without a score")
    expect_fail(_variant("m_dup", lambda r: r + r[-1:]), "scored more than once")
    expect_fail(_variant("m_orphan", lambda r: r + [dict(r[0], shard_row=2)]), "scores for no expected input")
    expect_fail(_variant("m_sum", perturb=1e-3), "sum to 1")
    expect_fail(_variant("m_gap", token_rows=[token_row("c1", n=5), token_row("c2", n=0, cerr=True)]),
                "neither missing nor empty")
    expect_fail(_variant("m_cerr", token_rows=[token_row("c1", n=0, missing=True, cerr=True)]), "clean error not flagged")
    expect_fail(_variant("m_plan", token_rows=[token_row("c1", n=124, starts=[0, 20, 62])]), "shard row 0")
    expect_fail(_variant("m_nostats", stats=False), "stats file missing")
    expect_fail(_variant("m_order", token_rows=[token_row("c1", "quoted", n=5)]), "order own")
    expect_fail(_variant("m_unknown", extra_scores=[{"shard": "BTC/shard-00009.parquet", "shard_row": 0, "window": 0,
                                                     "p": probs(0)}]), "have no token shard")
    # a failed check leaves no output behind
    assert not (E.emotion_dir("reddit") / "BTC.parquet").exists()
    fresh_out("m_noempty")
    build_happy()
    saved = E.EMPTY_SCORE_FILE.read_text()
    E.EMPTY_SCORE_FILE.unlink()
    try:
        expect_fail(lambda: M06.merge("reddit", None, False), "empty-text score missing")
    finally:
        E.EMPTY_SCORE_FILE.write_text(saved)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


# ==================================================================================================
# D. News headlines: key, stage 1, the four subcommands end to end
# ==================================================================================================

LONG = " ".join(f"Bitcoin miners report output number {i} as hashrate climbs again" for i in range(12))
ROWS = [  # entity, headline
    ("Bitcoin", "Bitcoin soars to record high as investors pile in"),
    ("Ethereum", "Bitcoin soars to record high as investors pile in"),      # same headline, other entity
    ("Bitcoin", "Crypto crash wipes out billions; traders fear more losses"),
    ("Dogecoin", "$DOGE"),                                                    # cleans to empty (ticker only)
    ("Bitcoin", LONG),                                                        # > 62 wordpieces
    ("Gold", "Gold steady ahead of Fed decision"),
]
HEADER = ["rp_entity_id", "rpa_date_utc", "rpa_time_utc", "timestamp_utc", "entity_name", "relevance",
          "event_sentiment_score", "event_relevance", "event_similarity_days", "category", "event_text",
          "news_type", "source_name", "css", "nip", "headline"]


def test_headline_key():
    m = M06
    assert m.headline_key("Bitcoin €") == hashlib.md5("Bitcoin €".encode("utf-8")).hexdigest()
    assert m.headline_key("") is None and m.headline_key(None) is None


def _news_stack():
    """(emtract_method, tokens) with the real tokenizer, or raise Skip where EmTract's stack is missing."""
    try:
        import tokenizers
        if tokenizers.__version__ != E.TOKENIZERS_VERSION:
            raise Skip(f"tokenizers {tokenizers.__version__} (needs {E.TOKENIZERS_VERSION}: run in emtract-ref)")
        E.load_cleaning(); E.install_emoji_memo()
        return E, E.Tokens(REAL_MODELS / E.TOKENIZER_VARIANT)
    except Skip:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Skip(f"EmTract stack not available: {exc}")


def test_stage1_chunk_equals_stage1_rows():
    S, tok = _news_stack()
    m = M06
    texts = [h for _, h in ROWS]
    got = m.news_stage1_chunk(texts, tok)
    want, _ = S.stage1_rows("stocktwits", {"message_body": texts}, tok)
    for c in ("cleaned_text", "is_empty_after_clean", "is_clean_error", "n_tokens", "input_ids", "window_starts"):
        assert got[c] == want[c], c
    assert got["n_tokens"][3] == 0 and got["is_empty_after_clean"][3]
    assert got["n_tokens"][4] > 62 and len(got["window_starts"][4]) >= 2


def test_end_to_end_tiny_csv():
    S, tok = _news_stack()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        f = d / "rp.csv"
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            for i, (ent, h) in enumerate(ROWS):
                w.writerow(["X", "2021-01-01", "10:00:00", f"2021-01-01 10:00:0{i}.5", ent, "99", "", "",
                            "", "", "", "FULL-ARTICLE", "Src", "0.1", "-0.2", h])
        env = dict(os.environ, THESIS_NEWS_EMTRACT_OUT=str(d / "out"), THESIS_EMTRACT_MODELS=str(REAL_MODELS))
        run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), "news", *a], cwd=str(CODE), env=env, check=True,
                                        capture_output=True, text=True)
        run("prepare", "--csv", str(f))
        run("stage1", "--workers", "2")
        run("score", "--device", "cpu", "--threads", "2")
        run("refcheck", "--n", "10", "--device", "cpu", "--threads", "2")
        out = d / "out"
        mp_ = pq.read_table(out / "news_row_headline_map.parquet").to_pylist()
        assert len(mp_) == len(ROWS) and [r["row_idx"] for r in mp_] == list(range(len(ROWS)))
        assert mp_[0]["headline_md5"] == mp_[1]["headline_md5"] and mp_[0]["headline_md5"] != mp_[2]["headline_md5"]
        emo = pq.read_table(out / "news_headline_emotion.parquet").to_pylist()
        assert len(emo) == 5
        by = {r["headline_md5"]: r for r in emo}
        assert by[mp_[0]["headline_md5"]]["n_source_rows"] == 2
        e = by[mp_[3]["headline_md5"]]
        assert e["is_empty_after_clean"] and e["n_windows"] == 0 and e["p_neutral"] is None
        s = by[mp_[0]["headline_md5"]]
        assert s["n_windows"] == 1 and s["pc_neutral"] is None
        assert abs(sum(s[f"p_{l}"] for l in E.LABELS) - 1) < 1e-5
        # the long headline: p and pc against a direct model call on its windows
        lg = by[mp_[4]["headline_md5"]]
        cols = m_cols = m = None
        m = M06
        st = m.news_stage1_chunk([LONG], tok)
        ins = S.model_inputs(st["input_ids"][0], st["window_starts"][0], tok.cls_id, tok.sep_id)
        assert lg["n_windows"] == len(ins) >= 2
        import torch
        model = M06.load_model(REAL_MODELS / E.MODEL_VARIANT, "cpu")
        with torch.no_grad():
            pr = torch.softmax(model(input_ids=torch.tensor(ins)).logits, -1).numpy().astype(np.float64)
        labs = E.LABELS
        assert max(abs(lg[f"p_{l}"] - pr[0, j]) for j, l in enumerate(labs)) < 1e-5
        assert max(abs(lg[f"pc_{l}"] - pr[:, j].mean()) for j, l in enumerate(labs)) < 1e-5
        import json
        ref = json.loads((out / "news_refcheck.json").read_text())
        assert ref["keys_aligned"] and ref["max_abs_diff"] < 1e-5
        stats = json.loads((out / "news_headline_emotion.parquet.stats.json").read_text())
        assert all(c["passed"] for c in stats["checks"])


# ==================================================================================================
# Runner (also works with pytest)
# ==================================================================================================

GROUPS = {
    "A": [test_versions_reddit, test_versions_stocktwits, test_versions_twitter, test_flags_standin,
          test_window_plan_sizes, test_window_plan_word_starts, test_window_plan_random_property, test_model_inputs,
          test_clean_uncaught_equals_clean_text, test_clean_flags_real, test_emoji_memo_identical,
          test_slicing_equals_truncation, test_stage1_rows, test_step06_resume],
    "B": [test_special_ids_from_files, test_window_plan_checks, test_build_inputs_matches_reference,
          test_pack_and_map, test_invalid_shards_refused, test_buckets_padding_order,
          test_score_order_and_values, test_collect_and_its_checks],
    "C": [test_window_plan_and_weights, test_merge_values, test_merge_checks],
    "D": [test_headline_key, test_stage1_chunk_equals_stage1_rows, test_end_to_end_tiny_csv],
}


def main() -> int:
    n = failed = skipped = 0
    try:
        for group, tests in GROUPS.items():
            for t in tests:
                n += 1
                config.OUT_DIR = PIPELINE_OUT
                t0 = time.time()
                try:
                    t()
                    print(f"ok    {group} {t.__name__} ({time.time() - t0:.1f} s)", flush=True)
                except Skip as exc:
                    skipped += 1
                    print(f"skip  {group} {t.__name__}: {exc}", flush=True)
                except Exception as exc:  # noqa: BLE001  (report every test, then fail)
                    failed += 1
                    print(f"FAIL  {group} {t.__name__}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"{n - failed - skipped} of {n} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
