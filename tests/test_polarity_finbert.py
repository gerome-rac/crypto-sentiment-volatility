"""Tests for step 07, the polarity scoring (pipeline/07_polarity_finbert.py, pipeline/polarity_method.py).

Run from code/:  python tests/test_polarity_finbert.py   (env thesis; or: pytest tests/test_polarity_finbert.py)
Nothing outside a temporary folder is written: every THESIS_* path points there. No corpus data is
read: the source tables are hand-made from pipeline/polarity_ref/selfcheck_texts.json (hand-written
texts) in the stage-1 layouts of all five platforms.

Unit tests use two tiny randomly initialised mock models registered at run time: 'mockbert' (a
WordPiece BERT, 2 layers, labels positive/negative/neutral) and 'mockroberta' (a byte-level BPE
RoBERTa trained on the self-check texts, labels bearish/neutral/bullish). The last test is a light
real-model smoke test on CPU (ProsusAI/finbert and ElKulako/cryptobert at their pinned commits, read
from code/data/raw/polarity_models/; skipped when those files are absent): the fixed-sample id
equality against the stored slow-tokenizer ids, and pipeline scores == the plain Hugging Face call on
the 212 hand-written texts.

What is tested
  windows      the generalised rule equals EmTract's stage-1 rule for W = 62 (starts, k, validity) and
               the coverage weights of EmTract's merge; valid plans for W = 126
  rows         own-version rows == EmTract's build_versions (Reddit dead rows, StockTwits NULLs);
               DLT/RavenPack own only
  tokenizer    prefix/suffix/pad from the files; uncut ids; primary == own truncation; word starts
               (WordPiece: not '##'); whitespace-only text has 0 tokens for WordPiece
  tokenize     all five platforms; schema, flags, ids, plans; a rerun writes nothing; --text-key
  selfcheck    refuses a reference that differs in one id
  pack         only seq + input_ids leave (list<uint16>); map; windows; --max-inputs; one writer;
               refuses a changed shard at collect
  score        fp32 equals one-at-a-time unpadded scoring within 1e-5, independent of row order;
               autocast + fp32 cross-check: >= 1 checked input per batch, x_logit set exactly there,
               the max |dp| guard
  merge        keys and order of the token rows; p NULL exactly for missing/empty; polarity =
               p_pos - p_neg; EmTract key-set check (passes, and refuses a missing row); windows
               (pc/pw); --dedup equals per-row scoring; completeness refuses an unpacked shard
  real models  (smoke) see above

Ported 2 Oct 2026 from tests/test_polarity.py (same tests; names of the merged step).
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
REAL_MODELS = CODE / "data" / "raw" / "polarity_models"
TMP = Path(tempfile.mkdtemp(prefix="polarity_test_"))
os.environ["THESIS_PIPELINE_OUT"] = str(TMP / "pipeline")
os.environ["THESIS_POLARITY_RUN"] = str(TMP / "run")
os.environ["THESIS_POLARITY_MODELS"] = str(TMP / "models")
os.environ["THESIS_POLARITY_REF_IDS"] = str(TMP / "ref")
os.environ["THESIS_SCRATCH"] = str(TMP / "scratch")
os.environ["THESIS_EMTRACT_RUN"] = str(TMP / "emtract_run")
(TMP / "ref").mkdir(parents=True)
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from pipeline import config, runner  # noqa: E402
from pipeline import emtract_method as EM  # noqa: E402
from pipeline import polarity_method as PM  # noqa: E402


def _imp(name, fname):
    spec = importlib.util.spec_from_file_location(name, CODE / "pipeline" / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P07 = _imp("polarity07", "07_polarity_finbert.py")
TEXTS = json.loads((PM.REF_DIR / "selfcheck_texts.json").read_text(encoding="utf-8"))["texts"]


# ---------------------------------------------------------------------------
# mock models
# ---------------------------------------------------------------------------

def _register(slug, d, labels, pos, neg, weights, default_len):
    files = tuple(sorted(p.name for p in d.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SOURCE.json")))
    sums = {n: PM._sha256(d / n) for n in files}
    (d / "SHA256SUMS").write_text("".join(f"{sums[n]}  {n}\n" for n in files))
    PM.MODELS[slug] = PM.ModelSpec(slug=slug, repo=f"mock/{slug}", revision="0" * 40, files=files, weights=weights,
                                 weights_sha256=sums[weights], labels=labels, pos_label=pos, neg_label=neg,
                                 default_max_len=default_len, max_positions=512)


def _mock_reference(slug, max_len=64):
    tok = PM.PolarityTokenizer(PM.model_dir(slug), max_len)
    ids, _ = tok.encode(TEXTS)
    import hashlib
    rec = {"model": slug, "repo": PM.MODELS[slug].repo, "revision": PM.MODELS[slug].revision,
           "built_with": {"transformers": "mock", "class": "mock"},
           "texts_sha256": hashlib.sha256((PM.REF_DIR / "selfcheck_texts.json").read_bytes()).hexdigest(),
           "n_texts": len(TEXTS), "special_ids": {"prefix": tok.prefix_id, "suffix": tok.suffix_id, "pad": tok.pad_id},
           "ids": ids}
    (TMP / "ref" / f"{slug}_reference_ids.json").write_text(json.dumps(rec))


def make_mocks():
    if "mockbert" in PM.MODELS:
        return
    import torch
    import transformers
    from tokenizers import ByteLevelBPETokenizer
    torch.manual_seed(0)
    # WordPiece BERT
    d = PM.model_dir("mockbert"); d.mkdir(parents=True, exist_ok=True)
    words = sorted({w.lower().strip(".,!?$#@") for t in TEXTS[:167] for w in t.split()} - {""})[:300]
    chars = [chr(c) for c in range(ord("a"), ord("z") + 1)] + list("0123456789")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(".,!?$#@:;-'\"()/%&*+=<>[]_~`") + chars + \
            ["##" + c for c in chars] + [w for w in words if w.isascii() and w.isalnum() and w not in chars]
    vocab = list(dict.fromkeys(vocab))
    (d / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab=str(d / "vocab.txt"), do_lower_case=True)
    assert len(tok) == len(vocab)
    tok.save_pretrained(str(d))
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=64, max_position_embeddings=512, num_labels=3,
                                  id2label={0: "positive", 1: "negative", 2: "neutral"},
                                  label2id={"positive": 0, "negative": 1, "neutral": 2}, pad_token_id=0)
    transformers.BertForSequenceClassification(cfg).save_pretrained(str(d))
    _register("mockbert", d, ("positive", "negative", "neutral"), "positive", "negative", "model.safetensors", 64)
    # byte-level BPE RoBERTa
    d = PM.model_dir("mockroberta"); d.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(TEXTS, vocab_size=600, min_frequency=2,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(d))
    rt = transformers.RobertaTokenizerFast(vocab=str(d / "vocab.json"), merges=str(d / "merges.txt"))
    assert len(rt) > 300
    rt.save_pretrained(str(d))
    cfg = transformers.RobertaConfig(vocab_size=len(rt), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                     intermediate_size=64, max_position_embeddings=514, type_vocab_size=1, num_labels=3,
                                     id2label={0: "Bearish", 1: "Neutral", 2: "Bullish"},
                                     label2id={"Bearish": 0, "Neutral": 1, "Bullish": 2}, pad_token_id=1,
                                     bos_token_id=0, eos_token_id=2)
    transformers.RobertaForSequenceClassification(cfg).save_pretrained(str(d))
    _register("mockroberta", d, ("bearish", "neutral", "bullish"), "bullish", "bearish", "model.safetensors", 64)
    _mock_reference("mockbert")
    _mock_reference("mockroberta")


# ---------------------------------------------------------------------------
# hand-made source tables in the five stage-1 layouts
# ---------------------------------------------------------------------------

def _write(path: Path, table: pa.Table, rg: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=rg or None)


def make_sources():
    if (TMP / "sources_done").exists():
        return
    rnd = random.Random(1)
    base = TEXTS
    for p in EM.REDDIT_PARTITIONS:
        n = 60
        own = [base[(i * 7 + len(p)) % len(base)] for i in range(n)]
        own[3] = ""
        own[4] = " \n\t "
        own[5] = own[6]                                  # duplicate text
        quoted = [("> " + base[i % 50]) if i % 9 == 0 else None for i in range(n)]
        text = [(q + "\n" + o) if q else o for q, o in zip(quoted, own)]
        dead = [i % 13 == 0 and i > 0 for i in range(n)]
        _write(config.out_path("reddit", "extract", f"reddit_{p}.parquet"), pa.table({
            "id": [f"{p.lower()}_{i}" for i in range(n)], "text": text, "own_text": own, "quoted_text": quoted,
            "has_quote_lines": [q is not None for q in quoted], "is_dead": dead}), rg=25)
    for m in ("2019-08", "2019-09"):
        n = 50
        body = [base[(i * 11 + int(m[-1])) % len(base)] for i in range(n)]
        body[2] = None
        body[7] = body[8]
        _write(config.out_path("stocktwits", "coarse", f"month={m}", "data_0.parquet"), pa.table({
            "message_id": pa.array([int(m[-1]) * 1000 + i for i in range(n)], pa.int64()), "message_body": body}))
    n = 120
    text = [base[(i * 5) % len(base)] for i in range(n)]
    for i in range(0, n, 10):
        text[i + 1] = text[i]                            # many duplicates for --dedup
    parent = [base[(i * 3) % len(base)] if i % 4 == 0 else None for i in range(n)]
    _write(config.out_path("twitter", "units", "units.parquet"),
           pa.table({"unit_id": [f"t{i}" for i in range(n)], "text": text, "parent_text": parent}), rg=40)
    dl = list(base) + [None]
    _write(config.out_path("dlt", "units", "dlt_units.parquet"),
           pa.table({"row_id": pa.array(list(range(len(dl))), pa.int64()), "text": dl}))
    hl = [t for t in base[:150] if t.strip()]
    import hashlib
    _write(config.out_path("ravenpack", "emtract", "news_unique_headlines.parquet"),
           pa.table({"headline_md5": [hashlib.md5(h.encode()).hexdigest() for h in hl], "headline": hl}))
    # EmTract emotion tables (keys only) for the key-set check: Twitter with all three versions
    t = pq.read_table(config.out_path("twitter", "units", "units.parquet"))
    idx, ver, _, _ = EM.build_versions("twitter", {"text": t.column("text").to_pylist(), "parent_text": t.column("parent_text").to_pylist()})
    ids = t.column("unit_id").to_pylist()
    _write(config.out_path("twitter", "emtract", "emotion", "units.parquet"),
           pa.table({"row_id": [ids[i] for i in idx], "version": ver}))
    (TMP / "sources_done").write_text("1")


def _reset(platform, tag):
    for d in (config.OUT_DIR / platform / "polarity" / tag, PM.run_dir(tag) / "modal_inputs" / platform,
              PM.run_dir(tag) / "modal_outputs" / platform):
        shutil.rmtree(d, ignore_errors=True)
    kf = PM.keys_file(platform, tag)
    for f in (kf, runner.stats_path(kf)):
        if f.exists():
            f.unlink()


def run_platform(platform, slug="mockbert", max_len=64, windows=False, dedup=False, precision="fp32",
                 max_inputs=10 ** 6, partitions=None, mdir=None, xcheck_every=8):
    tag = PM.tag(slug, max_len)
    _reset(platform, tag)
    P07.tokenize(platform, slug, max_len, ["own"], dedup, partitions, None, None, 1, mdir)
    if dedup:
        P07.keys(platform, slug, max_len)
    b = 0
    while True:
        try:
            P07.pack(platform, slug, max_len, f"b{b:02d}", max_inputs, windows, dedup)
        except P07.CheckError as e:
            if "unpacked shard" in str(e):
                break
            raise
        inp = PM.inputs_dir(platform, tag) / f"b{b:02d}.parquet"
        raw = PM.raw_dir(platform, tag) / f"b{b:02d}.parquet"
        P07.score(inp, raw, slug, max_len, "cpu", mdir or PM.model_dir(slug), precision, xcheck_every=xcheck_every)
        P07.collect(platform, slug, max_len, f"b{b:02d}")
        b += 1
    return P07.merge(platform, slug, max_len, None, force=True), tag


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_window_rule_equals_emtract():
    rule = PM.WindowRule(64)
    rnd = random.Random(7)
    m06 = _imp("emotion06", "06_emotion_emtract.py")  # EmTract's merge (section 3)
    for n in list(range(0, 400)) + [rnd.randint(400, 3000) for _ in range(300)]:
        ws = [rnd.random() < 0.7 for _ in range(n)]
        assert rule.n_windows(n) == EM.n_windows(n)
        assert rule.ideal_starts(n) == EM.ideal_starts(n)
        a, b = rule.starts(n, ws), EM.window_starts(n, ws)
        assert a == b, (n, a, b)
        EM.check_window_plan(n, a)
        assert rule.plan_error(n, a) is None
        if n > 62:
            assert np.allclose(rule.coverage_weights(n, a), m06.coverage_weights(n, tuple(a)))
    r128 = PM.WindowRule(128)
    for n in range(1, 2000, 7):
        ws = [rnd.random() < 0.6 for _ in range(n)]
        st = r128.starts(n, ws)
        assert r128.plan_error(n, st) is None, (n, st)
        assert all(st[j] + 126 - st[j + 1] >= 16 for j in range(len(st) - 1))
    assert r128.plan_error(300, [0, 200]) is not None


def test_rows_equal_emtract_versions():
    cols = {"text": ["a\n> q", "b", "c", None], "own_text": ["a", "b", "", None],
            "quoted_text": ["> q", None, None, None], "has_quote_lines": [True, False, False, False],
            "is_dead": [False, False, False, True]}
    idx, ver, txt, c = PM.build_rows("reddit", cols)
    assert (idx, ver, txt) == ([0, 1, 2], ["own"] * 3, ["a", "b", ""]) and c["rows_dead"] == 1
    idx, ver, txt, _ = PM.build_rows("reddit", cols, ("own", "quoted", "both"))
    i2, v2, t2, _ = EM.build_versions("reddit", cols)
    assert (idx, ver, txt) == (i2, v2, t2)
    idx, ver, txt, _ = PM.build_rows("stocktwits", {"message_body": ["x", None]})
    assert txt == ["x", None]
    idx, ver, txt, _ = PM.build_rows("dlt", {"text": ["x", None, ""]})
    assert idx == [0, 1, 2] and ver == ["own"] * 3
    try:
        PM.build_rows("ravenpack", {"headline": ["x"]}, ("own", "quoted"))
        raise AssertionError("ravenpack quoted accepted")
    except ValueError:
        pass


def test_tokenizer_wrapper():
    make_mocks()
    for slug in ("mockbert", "mockroberta"):
        tok = PM.PolarityTokenizer(PM.model_dir(slug), 64)
        sp = tok.specials()
        assert (sp["prefix"], sp["suffix"], sp["pad"]) == ((2, 3, 0) if slug == "mockbert" else (0, 2, 1)), sp
        assert sp["vocab_size"] > 100
        ids, ws = tok.encode(TEXTS, want_word_starts_above=62)
        tr = tok.truncated(TEXTS)
        assert all(tr[i] == tok.primary(ids[i]) for i in range(len(TEXTS)))
        assert ids[-2] == []                                   # empty text
        if slug == "mockbert":
            assert ids[-1] == []                               # whitespace only: 0 WordPiece tokens
            vocab = tok.tok.convert_ids_to_tokens(list(range(tok.vocab_size)))
            for i, w in zip(ids, ws):
                if w is not None:
                    assert w == [not vocab[t].startswith("##") for t in i]
        assert tok.id_dtype == "uint16"


def test_selfcheck_and_refusal():
    make_mocks()
    rec = P07.selfcheck("mockbert", 64)
    assert all(c["passed"] for c in rec["checks"])
    f = TMP / "ref" / "mockbert_reference_ids.json"
    good = f.read_text()
    bad = json.loads(good)
    bad["ids"][10] = bad["ids"][10][:-1] + [bad["ids"][10][-1] + 1]
    f.write_text(json.dumps(bad))
    try:
        P07.selfcheck("mockbert", 64)
        raise AssertionError("selfcheck accepted a changed id")
    except runner.CheckError:
        pass
    finally:
        f.write_text(good)


def test_tokenize_all_platforms():
    make_mocks(); make_sources()
    for platform, parts in (("reddit", ["BTC", "MARKET"]), ("stocktwits", None), ("twitter", None), ("dlt", None),
                            ("ravenpack", None)):
        tag = PM.tag("mockbert", 64)
        _reset(platform, tag)
        rec = P07.tokenize(platform, "mockbert", 64, ["own"], True, parts, None, None, 1)
        assert rec["shards_written"] >= 1
        again = P07.tokenize(platform, "mockbert", 64, ["own"], True, parts, None, None, 1)
        assert again["shards_written"] == 0, "rerun rewrote current shards"
        shards = sorted(PM.tokens_dir(platform, tag).glob("*/shard-*.parquet"))
        t = pa.concat_tables([pq.read_table(s) for s in shards])
        assert t.column_names == ["row_id", "version", "is_missing_text", "is_empty_text", "n_tokens", "input_ids",
                                  "window_starts", "text_key"], t.column_names
        assert t.schema.field("input_ids").type == pa.list_(pa.uint16())
        n = np.asarray(t.column("n_tokens"))
        miss, emp = np.asarray(t.column("is_missing_text")), np.asarray(t.column("is_empty_text"))
        assert ((n == 0) == (miss | emp)).all()
        if platform == "stocktwits":
            assert miss.sum() == 2
        if platform == "reddit":
            src = [pq.read_table(config.out_path("reddit", "extract", f"reddit_{p}.parquet")) for p in parts]
            want = sum(int((~np.asarray(s.column("is_dead"))).sum()) for s in src)
            assert t.num_rows == want and emp.sum() >= 4          # '' and whitespace in each partition
        assert t.column("text_key").null_count == int(miss.sum())


def test_pack_score_collect_merge_mockbert():
    make_mocks(); make_sources()
    out, tag = run_platform("twitter", max_inputs=60)
    maps = sorted(PM.inputs_dir("twitter", tag).glob("*.map.parquet"))
    assert len(maps) >= 2, "--max-inputs should give several batches"
    for f in sorted(PM.inputs_dir("twitter", tag).glob("b*.parquet")):
        if f.name.endswith(".map.parquet"):
            continue
        t = pq.read_table(f)
        assert t.column_names == ["seq", "input_ids"] and t.schema.field("input_ids").type == pa.list_(pa.uint16())
    pol = pq.read_table(PM.polarity_dir("twitter", tag) / "units.parquet")
    tok = pa.concat_tables([pq.read_table(s, columns=["row_id", "version", "n_tokens"])
                            for s in sorted(PM.tokens_dir("twitter", tag).glob("*/shard-*.parquet"))])
    assert pol.column("row_id").to_pylist() == tok.column("row_id").to_pylist()
    n = np.asarray(pol.column("n_tokens"))
    p = np.asarray(pol.column("p_positive").to_numpy(zero_copy_only=False), dtype=float)
    assert (np.isnan(p) == (n == 0)).all()
    d = (np.asarray(pol.column("p_positive").to_numpy(zero_copy_only=False), float)
         - np.asarray(pol.column("p_negative").to_numpy(zero_copy_only=False), float))
    pv = np.asarray(pol.column("polarity").to_numpy(zero_copy_only=False), float)
    ok = ~np.isnan(pv)
    assert np.allclose(pv[ok], d[ok], atol=1e-6)
    assert "emtract_table_checked" in out["partitions"]["units"]
    # the EmTract key-set check refuses a missing row
    em = config.out_path("twitter", "emtract", "emotion", "units.parquet")
    keep = pq.read_table(em)
    pq.write_table(keep.slice(1), em)
    try:
        P07.merge("twitter", "mockbert", 64, None, force=True)
        raise AssertionError("merge accepted a row missing from the EmTract table")
    except runner.CheckError:
        pass
    finally:
        pq.write_table(keep, em)


def test_score_reference_and_order():
    make_mocks(); make_sources()
    import torch
    _, tag = run_platform("dlt", max_inputs=10 ** 6)
    inp = PM.inputs_dir("dlt", tag) / "b00.parquet"
    t = pq.read_table(inp)
    ids = t.column("input_ids").to_pylist()
    model, labels = P07.load_model("mockbert", PM.model_dir("mockbert"), "cpu")
    with torch.no_grad():
        ref = np.stack([torch.softmax(model(input_ids=torch.tensor([x])).logits[0], -1).numpy() for x in ids])
    raw = pq.read_table(PM.raw_dir("dlt", tag) / "b00.parquet")
    got = np.stack([raw.column(f"p_{l}").to_numpy() for l in labels], 1)
    assert np.abs(got - ref).max() < 1e-5, np.abs(got - ref).max()
    perm = np.random.default_rng(0).permutation(t.num_rows)
    pq.write_table(t.take(pa.array(perm)), TMP / "perm.parquet")
    P07.score(TMP / "perm.parquet", TMP / "perm_out.parquet", "mockbert", 64, "cpu", PM.model_dir("mockbert"), "fp32")
    r2 = pq.read_table(TMP / "perm_out.parquet")
    g2 = np.stack([r2.column(f"p_{l}").to_numpy() for l in labels], 1)
    assert np.abs(g2 - got).max() < 1e-5


def _autocast_precision():
    import torch
    try:
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            torch.nn.Linear(4, 4)(torch.ones(2, 4))
        return "fp16"
    except Exception:                                    # noqa: BLE001
        return "bf16"


def test_autocast_xcheck():
    make_mocks(); make_sources()
    prec = _autocast_precision()
    _, tag = run_platform("dlt", precision=prec, xcheck_every=8)
    rec = json.loads((PM.raw_dir("dlt", tag) / "b00.parquet.stats.json").read_text())
    xc = rec["xcheck"]
    assert xc and len(xc["per_batch"]) == rec["counts"]["batches"]
    assert all(b[2] == -(-b[1] // 8) for b in xc["per_batch"])
    raw = pq.read_table(PM.raw_dir("dlt", tag) / "b00.parquet")
    xm = np.asarray(raw.column("xcheck"))
    assert xm.sum() == xc["inputs_checked"]
    assert raw.column("x_logit_positive").null_count == raw.num_rows - xm.sum()
    try:
        P07.score(PM.inputs_dir("dlt", tag) / "b00.parquet", TMP / "g.parquet", "mockbert", 64, "cpu",
                  PM.model_dir("mockbert"), prec, xcheck_every=1, xcheck_max_dp=0.0)
        if xc["max_abs_dp"] > 0:
            raise AssertionError("max |dp| guard did not fire")
    except P07.CheckError:
        pass


def test_windows_mockroberta():
    make_mocks(); make_sources()
    out, tag = run_platform("ravenpack", slug="mockroberta", max_len=32, windows=True)
    pol = pq.read_table(PM.polarity_dir("ravenpack", tag) / "headlines.parquet")
    k = np.asarray(pol.column("n_windows"))
    pc = pol.column("pc_bullish")
    assert (k >= 2).sum() > 0 and pc.null_count == int((k < 2).sum())
    a = np.asarray(pol.column("polarity_c").to_numpy(zero_copy_only=False), float)
    b = (np.asarray(pol.column("pc_bullish").to_numpy(zero_copy_only=False), float)
         - np.asarray(pol.column("pc_bearish").to_numpy(zero_copy_only=False), float))
    ok = ~np.isnan(a)
    assert np.allclose(a[ok], b[ok], atol=1e-6)
    assert "emtract_table_checked" not in out["partitions"]["headlines"]  # no EmTract news table here


def test_dedup_equals_per_row():
    make_mocks(); make_sources()
    _, tag = run_platform("twitter")
    base = pq.read_table(PM.polarity_dir("twitter", tag) / "units.parquet")
    out, _ = run_platform("twitter", dedup=True)
    dd = pq.read_table(PM.polarity_dir("twitter", tag) / "units.parquet")
    assert base.column("row_id").to_pylist() == dd.column("row_id").to_pylist()
    a = np.asarray(base.column("p_positive").to_numpy(zero_copy_only=False), float)
    b = np.asarray(dd.column("p_positive").to_numpy(zero_copy_only=False), float)
    assert (np.isnan(a) == np.isnan(b)).all() and np.nanmax(np.abs(a - b)) < 1e-5
    c = out["partitions"]["units"]
    assert c["scored_by_key"] > 0 and c["scored_self"] + c["scored_by_key"] == c["rows_with_tokens"]


def test_collect_refuses_changed_shard_and_incomplete_run():
    make_mocks(); make_sources()
    tag = PM.tag("mockbert", 64)
    _reset("stocktwits", tag)
    P07.tokenize("stocktwits", "mockbert", 64, ["own"], False, None, None, None, 1)
    P07.pack("stocktwits", "mockbert", 64, "s00", 60, False, False)       # one month only
    comp = P07.completeness("stocktwits", tag, "mockbert")
    assert not comp["complete"] and comp["n_shards_not_packed"] >= 1
    try:
        P07.merge("stocktwits", "mockbert", 64)
        raise AssertionError("merge ran on an incomplete run")
    except runner.CheckError:
        pass
    inp = PM.inputs_dir("stocktwits", tag) / "s00.parquet"
    P07.score(inp, PM.raw_dir("stocktwits", tag) / "s00.parquet", "mockbert", 64, "cpu", PM.model_dir("mockbert"), "fp32")
    rec = json.loads(runner.stats_path(PM.inputs_dir("stocktwits", tag) / "s00.map.parquet").read_text())
    shard = PM.tokens_dir("stocktwits", tag).resolve() / rec["shards"][0]["shard"]
    keep = shard.read_bytes()
    t = pq.read_table(shard)
    pq.write_table(t, shard, compression="snappy")                         # same rows, other bytes
    try:
        P07.collect("stocktwits", "mockbert", 64, "s00")
        raise AssertionError("collect accepted a changed shard")
    except P07.CheckError:
        pass
    finally:
        shard.write_bytes(keep)
    # one writer
    lock = PM.inputs_dir("stocktwits", tag) / ".pack.lock"
    lock.write_text("x")
    try:
        P07.pack("stocktwits", "mockbert", 64, "s01", 60, False, False)
        raise AssertionError("second pack ran under a held lock")
    except P07.CheckError:
        pass
    finally:
        lock.unlink()


def test_real_models_smoke():
    """Light real-model check on CPU (skipped without the pinned model files)."""
    import torch
    import transformers
    for slug, L in (("finbert", 64), ("cryptobert", 128)):
        real = REAL_MODELS / slug
        if not (real / "SHA256SUMS").exists():
            print(f"  SKIP {slug}: no model files in {real}")
            continue
        rec = P07.selfcheck(slug, L, real, write=False)        # stored slow-reference ids (committed file)
        assert all(c["passed"] for c in rec["checks"])
        make_sources()
        out, tag = run_platform("dlt", slug=slug, max_len=L, mdir=real)
        pol = pq.read_table(PM.polarity_dir("dlt", tag) / "units.parquet")
        tok = transformers.AutoTokenizer.from_pretrained(str(real), local_files_only=True)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(real), local_files_only=True).eval()
        labels = PM.MODELS[slug].labels
        texts = pq.read_table(config.out_path("dlt", "units", "dlt_units.parquet")).column("text").to_pylist()
        rows = [i for i, t in enumerate(texts) if t is not None and tok(t, add_special_tokens=False)["input_ids"]]
        probs = []
        with torch.no_grad():
            for a in range(0, len(rows), 32):
                enc = tok([texts[i] for i in rows[a:a + 32]], truncation=True, max_length=L, padding=True, return_tensors="pt")
                probs.append(torch.softmax(model(**enc).logits, -1).numpy())
        ref = np.concatenate(probs)
        got = np.stack([pol.column(f"p_{l}").to_numpy(zero_copy_only=False) for l in labels], 1)[rows].astype(float)
        diff = float(np.abs(got - ref).max())
        assert diff < 1e-5, (slug, diff)
        assert (ref.argmax(1) == got.argmax(1)).all()
        pv = np.asarray(pol.column("polarity").to_numpy(zero_copy_only=False), float)
        print(f"  {slug} L{L}: {len(rows)} texts scored, max |p - HF| {diff:.2e}; "
              f"polarity of 'BTC just broke 20k...' {pv[0]:+.3f}, 'Bitcoin is going to zero...' {pv[1]:+.3f}")


TESTS = [test_window_rule_equals_emtract, test_rows_equal_emtract_versions, test_tokenizer_wrapper,
         test_selfcheck_and_refusal, test_tokenize_all_platforms, test_pack_score_collect_merge_mockbert,
         test_score_reference_and_order, test_autocast_xcheck, test_windows_mockroberta, test_dedup_equals_per_row,
         test_collect_refuses_changed_shard_and_incomplete_run, test_real_models_smoke]

if __name__ == "__main__":
    failed = 0
    try:
        for t in TESTS:
            try:
                t()
                print(f"PASS {t.__name__}")
            except Exception as exc:                    # report every test, then fail
                import traceback
                failed += 1
                print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"{len(TESTS) - failed} of {len(TESTS)} passed")
    sys.exit(1 if failed else 0)
