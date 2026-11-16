#!/usr/bin/env python3
"""
07_polarity_finbert.py -- step 07: FinBERT polarity, P(positive) - P(negative), for every text that
step 06 gave an EmTract score: the social-media posts (Reddit, StockTwits, Twitter), the DLT-Tweets
event corpus and the RavenPack news headlines. One script; its subcommands run in the order below.

FinBERT (ProsusAI/finbert at commit 4556d13015211d73dccd3fdd39d39232506f3e43; Araci 2019,
arXiv 1908.10063; a BERT-base classifier with the classes positive, negative, neutral) scores the
ORIGINAL text of the same rows and text versions as EmTract's stage 1 -- no EmTract cleaning, no other
preprocessing -- with the model's own tokenizer, uncut, primary input [CLS] + the first 62 tokens +
[SEP] (max_len 64, FinBERT's training length), windows for longer texts. The models, paths and text
rules are in polarity_method.py (decision D19; CryptoBERT stays selectable with --model cryptobert
for the comparison that preceded the decision). The method follows step 06 stage by stage: stage 1
on the CPU, the model alone on a GPU on stored token ids, the merge on the CPU; the same temp-file,
check, commit and stats rules (runner.py).

Sections and subcommands (run from code/; environment `thesis`, environment.yml)
--------------------------------------------------------------------------------
  1  build-reference                  the slow-tokenizer reference ids of the 212 self-check texts
                                      (needs a transformers 4.x environment; stored in polarity_ref/)
     selfcheck                        current fast tokenizer == stored reference ids, before any shard
     tokenize <platform>              stage 1 -> <platform>/polarity/finbert-L64/tokens/<partition>/
     distinct <platform>, keys <platform>   optional: distinct-text counts, text keys for --dedup
  2  fetch-model                      the pinned model files from huggingface.co, sha256-checked
     pack <platform> --batch-id ID --max-inputs N [--windows]   model inputs of finished shards
     score --inputs FILE --out FILE --device cuda|mps|cpu        the model alone (fp16 + fp32 cross-check)
     collect <platform> --batch-id ID                           raw scores + map -> .../scores/
  3  merge <platform>                 one row per (row_id, version) -> .../polarity/<partition>.parquet
  (--model finbert is the default everywhere; <platform> is reddit, stocktwits, twitter, dlt or ravenpack)

How the stored tables were made (1-2 Oct 2026; README.md has hardware, times and costs)
--------------------------------------------------------------------------------------
    python pipeline/07_polarity_finbert.py selfcheck
    python pipeline/07_polarity_finbert.py tokenize reddit --workers 58   (stocktwits, twitter, dlt, ravenpack)
        (stage 1 ran on one 60-vCPU cloud VM: 16.8 min of machine time for the five sources)
    python pipeline/07_polarity_finbert.py pack reddit --batch-id reddit-0001 --max-inputs 5000000 --windows
        (repeated per platform until every finished shard was packed: 34 batches, 153,002,526 inputs)
    python pipeline/07_polarity_finbert.py score --inputs <modal_inputs>/reddit/reddit-0001.parquet \\
        --out <modal_outputs>/reddit/reddit-0001.parquet --model-dir <polarity_models>/finbert \\
        --device cuda --precision fp16 --xcheck-every 64
        (one batch per rented NVIDIA L4 GPU container; plain PyTorch, any CUDA machine works)
    python pipeline/07_polarity_finbert.py collect reddit --batch-id reddit-0001
    python pipeline/07_polarity_finbert.py merge reddit      (stocktwits, twitter, dlt, ravenpack)
Tags such as D19 refer to the thesis's decision log (README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform as _platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from pipeline import config  # noqa: E402
from pipeline import emtract_method as EM  # noqa: E402  (source tables and the shard plan, as step 06)
from pipeline import polarity_method as PM  # noqa: E402  (standard library only at import)
from pipeline.polarity_method import CheckError, check, log  # noqa: E402

try:                                    # the safe-runner rules (runner.py imports DuckDB)
    from pipeline import runner  # noqa: E402
except ImportError:                     # a GPU machine that only runs `score` needs no DuckDB
    runner = None

STEP = "07_polarity_finbert"            # stats files name the subcommand: "07_polarity_finbert:pack", ...
STEP_MERGE = STEP + ":merge"
DEFAULT_MODEL = "finbert"               # decision D19 (1 Oct 2026)


def code_fingerprint() -> str:
    """sha256 over this script and polarity_method.py (runner.code_fingerprint): every stats file of
    this step records it, and a stage redoes an output whose recorded fingerprint differs."""
    here = Path(__file__).resolve()
    return runner.code_fingerprint(here, here.with_name("polarity_method.py"))


# ==================================================================================================
# 1. Stage 1 (`selfcheck`, `tokenize`): the original text, the model's own tokenizer, uncut
# ==================================================================================================
# Rules (D19 facts §10; the thesis's data and pipeline rules; MASTERPLAN §7)
#   * Same rows and the same strings as EmTract stage 1 (emtract_method.build_versions for Reddit,
#     StockTwits, Twitter; DLT units.text and the RavenPack headline for the two other arms), the same
#     row ids (Reddit id, StockTwits message_id, Twitter unit_id, DLT row_id, RavenPack headline_md5)
#     and the same versions (default own only; --versions own quoted both as EmTract).
#   * No EmTract cleaning, no other preprocessing: the model's own tokenizer on the original text.
#   * Missing text (NULL) -> is_missing_text; a text with 0 content tokens -> is_empty_text; neither gets
#     a model input (NULL score in `merge`).
#   * The tokenizer is the current fast tokenizer of the pinned files; before any shard, `selfcheck`
#     asserts that it gives exactly the ids stored in polarity_ref/<model>_reference_ids.json, which
#     `build-reference` writes with the Python ("slow") reference tokenizer of transformers 4.x
#     (a separate transformers 4 environment), on the 212 hand-written texts of polarity_ref/selfcheck_texts.json; and that
#     prefix + ids[:max_len-2] + suffix equals the tokenizer's own truncation on those texts.
#     The library versions are recorded in every shard's stats and in the run record.
#
# Subcommands (from code/, env thesis: transformers 5.17.0, tokenizers 0.23.1, as the GPU environment)
#   build-reference --model M            (transformers 4.x only; writes polarity_ref/M_reference_ids.json)
#   selfcheck --model M [--max-len L]    the fixed-sample id-equality check; writes run_dir/selfcheck.json
#   tokenize <platform> --model M [--max-len L] [--versions own] [--text-key] [--partitions ...]
#            [--limit-rows N] [--shard-rows N] [--workers N] [--plan]
#       -> <platform>/polarity/<M>-L<L>/tokens/<partition>/shard-NNNNN.parquet (+ .stats.json)
#          columns: row_id (source type), version, is_missing_text, is_empty_text, n_tokens (int32),
#          input_ids (list<uint16>, uncut, no special tokens; uint32 if the vocabulary needs it),
#          window_starts (list<int32>: [] without tokens, [0] for 1..L-2 tokens, the window plan above),
#          text_key (binary(32), sha256 of the original text; only with --text-key, needed by --dedup)
#       One shard per source row group (as EmTract stage 1; above 300,000 rows split into ~200,000).
#       Checks per shard (CheckError, nothing committed): len(input_ids) = n_tokens; n_tokens = 0 exactly
#       for flagged rows; missing and empty never both; valid window plan for every row; ids below the
#       vocabulary size; 1% of rows + up to 50 rows above L-2 tokens: prefix + ids[:L-2] + suffix equals
#       the tokenizer's own truncation=True, max_length=L; written file re-read (rows, id sum).
#       A shard whose stats match its source range, code and rules is skipped.
#   distinct <platform> [--share 5] [--seed 20261001]
#       DuckDB (UTC, 4 GB, 3 threads, spill): rows, non-NULL texts and distinct original texts (sha256)
#       in a Bernoulli sample of the own texts; reads text, never writes it. -> RUN_DIR/distinct/<p>.json
#   keys <platform> --model M [--max-len L]
#       For --dedup: one row per distinct text_key with tokens: text_key, rep_shard, rep_row (the
#       smallest (shard, shard_row) holding it), n_rows. -> RUN_DIR/<tag>/keys/<platform>.parquet
#
# Usage
# -----
#     python pipeline/07_polarity_finbert.py selfcheck
#     python pipeline/07_polarity_finbert.py tokenize reddit --workers 8
#     python pipeline/07_polarity_finbert.py tokenize twitter --model cryptobert --max-len 128 --text-key
#     python pipeline/07_polarity_finbert.py keys twitter --model cryptobert --max-len 128
SPOT_EVERY = 100                     # 1% of rows (as step 06's SPOT_SHARE)
SPOT_LONG_MAX = 50                   # plus up to 50 long rows (as step 06)
REF_TEXTS = PM.REF_DIR / "selfcheck_texts.json"


# --- Tokenizer, its self-check and the stage-1 fingerprint -----------------------------------------

def ref_ids_file(slug: str) -> Path:
    """polarity_ref/<slug>_reference_ids.json; THESIS_POLARITY_REF_IDS (a folder) is looked at first (tests)."""
    alt = os.environ.get("THESIS_POLARITY_REF_IDS")
    if alt and (Path(alt) / f"{slug}_reference_ids.json").exists():
        return Path(alt) / f"{slug}_reference_ids.json"
    return PM.REF_DIR / f"{slug}_reference_ids.json"


def load_tokenizer(slug: str, max_len: int, mdir: Optional[Path] = None) -> PM.PolarityTokenizer:
    mdir = Path(mdir or PM.model_dir(slug))
    tok = PM.PolarityTokenizer(mdir, max_len)
    return tok


def rules_fingerprint(slug: str, max_len: int, versions: Sequence[str], with_key: bool, mdir: Path,
                      tok_versions: Dict[str, str]) -> str:
    spec = PM.MODELS[slug]
    sums = PM.read_sums(mdir / "SHA256SUMS")
    tok_files = {n: sums.get(n) for n in spec.files if n != spec.weights}
    return hashlib.sha256(repr((spec.repo, spec.revision, sorted(tok_files.items()), tok_versions["transformers"],
                                tok_versions["tokenizers"], int(max_len), PM.MIN_WINDOW_OVERLAP, PM.MAX_MOVE,
                                tuple(versions), bool(with_key))).encode()).hexdigest()[:16]


def ref_texts() -> List[str]:
    return json.loads(REF_TEXTS.read_text(encoding="utf-8"))["texts"]


def build_reference(slug: str, mdir: Optional[Path] = None, out: Optional[Path] = None) -> Dict:
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        raise SystemExit("build-reference needs transformers 4.x (a separate environment): in 5.x the 'slow' classes are Rust-backed")
    mdir = Path(mdir or PM.model_dir(slug))
    PM.check_model_files(slug, mdir)
    tok = transformers.AutoTokenizer.from_pretrained(str(mdir), use_fast=False, local_files_only=True)
    if getattr(tok, "is_fast", True):
        raise SystemExit(f"{type(tok).__name__} is not the Python reference tokenizer")
    texts = ref_texts()
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    both = tok("", add_special_tokens=True)["input_ids"]
    rec = {"model": slug, "repo": PM.MODELS[slug].repo, "revision": PM.MODELS[slug].revision,
           "built_with": {"transformers": transformers.__version__, "class": type(tok).__name__},
           "texts_sha256": hashlib.sha256(REF_TEXTS.read_bytes()).hexdigest(), "n_texts": len(texts),
           "special_ids": {"prefix": both[0], "suffix": both[-1], "pad": tok.pad_token_id},
           "ids": ids}
    alt = os.environ.get("THESIS_POLARITY_REF_IDS")
    out = Path(out or (Path(alt) / f"{slug}_reference_ids.json" if alt else PM.REF_DIR / f"{slug}_reference_ids.json"))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out)
    return {k: v for k, v in rec.items() if k != "ids"}


def selfcheck(slug: str, max_len: int, mdir: Optional[Path] = None, ref: Optional[Path] = None,
              write: bool = True) -> Dict:
    """Fixed-sample id equality: current fast tokenizer == stored slow-reference ids; primary input ==
    the tokenizer's own truncation. Raises CheckError on any difference."""
    mdir = Path(mdir or PM.model_dir(slug))
    checks: List[Dict] = []
    files = PM.check_model_files(slug, mdir)
    check(True, "model files equal SHA256SUMS", checks)
    tok = load_tokenizer(slug, max_len, mdir)
    ref = json.loads(Path(ref or ref_ids_file(slug)).read_text())
    texts = ref_texts()
    check(ref["texts_sha256"] == hashlib.sha256(REF_TEXTS.read_bytes()).hexdigest(),
          "reference ids were built on the current self-check texts", checks)
    check(ref["revision"] == PM.MODELS[slug].revision, "reference built at the pinned revision", checks)
    ids, _ = tok.encode(texts)
    diff = [i for i, (a, b) in enumerate(zip(ids, ref["ids"])) if a != b]
    check(len(ids) == len(ref["ids"]) and not diff,
          f"fast ids == slow reference ids on {len(texts)} texts (differences at {diff[:5]})", checks)
    sp = ref["special_ids"]
    check((tok.prefix_id, tok.suffix_id, tok.pad_id) == (sp["prefix"], sp["suffix"], sp["pad"]),
          "prefix/suffix/pad ids equal the reference", checks)
    trunc = tok.truncated(texts)
    bad = [i for i, t in enumerate(texts) if trunc[i] != tok.primary(ids[i])]
    check(not bad, f"prefix + ids[:{max_len - 2}] + suffix == own truncation at {max_len} (differences {bad[:5]})", checks)
    n_long = sum(1 for x in ids if len(x) > max_len - 2)
    check(n_long >= 3, f"the self-check texts include texts above {max_len - 2} tokens ({n_long})", checks)
    rec = {"step": STEP + ":selfcheck", "model": slug, "revision": PM.MODELS[slug].revision, "max_len": max_len,
           "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "versions": tok.versions,
           "reference_built_with": ref["built_with"], "n_texts": len(texts), "n_above_window": n_long,
           "n_empty": sum(1 for x in ids if not x), "special_ids": tok.specials(), "id_dtype": tok.id_dtype,
           "tokenizer_files_sha256": {k: v for k, v in files.items() if k != PM.MODELS[slug].weights},
           "checks": checks}
    if write:
        runner.write_json_atomic(PM.run_dir(PM.tag(slug, max_len)) / "selfcheck.json", rec)
    return rec


# --- Sources and shards ----------------------------------------------------------------------------

def source_files(platform: str, partitions: Optional[List[str]]) -> List[Tuple[str, Path]]:
    if platform in ("reddit", "stocktwits", "twitter"):
        return EM.source_files(platform, partitions)
    if platform == "dlt":
        out = [("units", config.out_path("dlt", "units", "dlt_units.parquet"))]
    elif platform == "ravenpack":
        out = [("headlines", config.out_path("ravenpack", "emtract", "news_unique_headlines.parquet"))]
    else:
        raise SystemExit(f"unknown platform {platform!r}")
    if partitions and partitions != [out[0][0]]:
        raise SystemExit(f"{platform} has one partition, {out[0][0]!r}")
    for p, f in out:
        if not f.exists():
            raise SystemExit(f"missing source table for partition {p}: {f}")
    return out


def plan(platform: str, partitions, limit_rows, shard_rows) -> List[Dict]:
    specs = []
    for part, path in source_files(platform, partitions):
        specs += EM.plan_shards(platform, part, path, limit_rows, shard_rows)
    return specs


def shard_file(tok_root: Path, spec: Dict) -> Path:
    return tok_root / spec["partition"] / f"shard-{spec['shard_index']:05d}.parquet"


def shard_is_current(final: Path, spec: Dict, inputs: List[Dict], code_sha: str, rules_sha: str) -> bool:
    if not runner.is_current(final, inputs, code_sha, rules_sha):
        return False
    rec = json.loads(runner.stats_path(final).read_text())
    keys = ("source", "row_group", "rg_offset", "rg_stop", "row_start", "row_stop")
    return all(rec.get("shard", {}).get(k) == spec[k] for k in keys)


# --- One shard (runs inside a worker) and the tokenize run -----------------------------------------
_TOK: Optional[PM.PolarityTokenizer] = None

def _worker_init(slug: str, max_len: int, mdir: str) -> None:
    global _TOK
    _TOK = load_tokenizer(slug, max_len, Path(mdir))


def process_shard(spec: Dict, ctx: Dict, tok: Optional[PM.PolarityTokenizer] = None) -> Dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    tok = tok or _TOK
    t0 = time.time()
    platform = spec["platform"]
    final = Path(ctx["shard_file"])
    rule = PM.WindowRule(ctx["max_len"])
    idcol = PM.SOURCES[platform]["id"]
    text_cols = list(PM.SOURCES[platform]["text"])
    pf = pq.ParquetFile(spec["source"])
    tab = pf.read_row_group(spec["row_group"], columns=[idcol] + text_cols)
    tab = tab.slice(spec["rg_offset"], spec["rg_stop"] - spec["rg_offset"])
    cols = {c: tab.column(c).to_pylist() for c in text_cols}
    out, counts = PM.stage1_rows(platform, cols, tok, rule, ctx["versions"], ctx["with_key"])
    checks: List[Dict] = []
    n = len(out["version"])
    src_idx = np.asarray(out["source_index"], dtype=np.int64)
    ids_col = tab.column(idcol).combine_chunks().take(pa.array(src_idx, pa.int64())) if n else tab.column(idcol).combine_chunks().slice(0, 0)
    check(ids_col.null_count == 0, "row ids not null", checks)
    n_tok = np.asarray(out["n_tokens"], dtype=np.int64)
    check(all(len(i) == k for i, k in zip(out["input_ids"], out["n_tokens"])), "len(input_ids) == n_tokens", checks)
    miss = np.asarray(out["is_missing_text"], dtype=bool)
    empt = np.asarray(out["is_empty_text"], dtype=bool)
    check(not bool((miss & empt).any()), "missing and empty never both", checks)
    check(bool(((n_tok == 0) == (miss | empt)).all()), "n_tokens = 0 exactly for missing or empty texts", checks)
    errs = [e for e in (rule.plan_error(k, s) for k, s in zip(out["n_tokens"], out["window_starts"])) if e]
    check(not errs, f"valid window plan for every row ({errs[:2]})", checks)
    max_id = counts["max_token_id"]
    check(max_id < tok.vocab_size, "ids below the vocabulary size", checks)
    id_type = pa.uint16() if tok.id_dtype == "uint16" else pa.uint32()
    # spot checks against the tokenizer's own truncation (needs the text; re-built from the source)
    idx_v, ver_v, txt_v, _ = PM.build_rows(platform, cols, ctx["versions"])
    check(list(idx_v) == list(out["source_index"]) and list(ver_v) == list(out["version"]), "row order stable", checks)
    probe = set(range(0, n, SPOT_EVERY))
    longs = [i for i in range(n) if n_tok[i] > rule.W][:SPOT_LONG_MAX]
    probe = sorted((probe | set(longs)) - {i for i in range(n) if n_tok[i] == 0})
    if probe:
        tr = tok.truncated([txt_v[i] for i in probe])
        bad = [i for i, t in zip(probe, tr) if t != tok.primary(out["input_ids"][i])]
        check(not bad, f"primary input == own truncation on {len(probe)} spot rows", checks)
    cols_out = {
        "row_id": ids_col,
        "version": pa.array(out["version"], pa.string()),
        "is_missing_text": pa.array(miss),
        "is_empty_text": pa.array(empt),
        "n_tokens": pa.array(n_tok.astype(np.int32)),
        "input_ids": pa.array(out["input_ids"], pa.list_(id_type)),
        "window_starts": pa.array(out["window_starts"], pa.list_(pa.int32())),
    }
    if ctx["with_key"]:
        cols_out["text_key"] = pa.array(out["text_key"], pa.binary(32))
    table = pa.table(cols_out)
    # own temp name per shard and process (runner.commit would remove a _tmp folder another worker uses)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        back = pq.read_table(tmp)
        check(back.num_rows == n, "written shard: one row per (row, version)", checks)
        flat = back.column("input_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        check(int(flat.astype(np.int64).sum()) == int(sum(sum(i) for i in out["input_ids"])), "written shard: id sum", checks)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    counts["spot_checked"] = len(probe)
    counts["n_windows_total"] = int(sum(len(s) for s in out["window_starts"]))
    runner.write_stats(final, step=STEP + ":tokenize", inputs=ctx["inputs"], code_sha=ctx["code_sha"], rules_sha=ctx["rules_sha"],
                       con=None, counts=counts, checks=checks,
                       extra={"shard": {k: spec[k] for k in ("platform", "partition", "source", "shard_index", "row_group",
                                                            "rg_offset", "rg_stop", "row_start", "row_stop")},
                              "model": ctx["model"], "max_len": ctx["max_len"], "versions_kept": list(ctx["versions"]),
                              "library_versions": tok.versions, "special_ids": tok.specials(),
                              "row_id_source": idcol, "seconds": round(time.time() - t0, 2)})
    return {"shard": str(final), "rows": n, **{k: counts[k] for k in ("missing_text", "empty_text", "over_W_tokens",
                                                                        "model_inputs_primary", "n_windows_total",
                                                                        "texts_tokenized_distinct")}}


def _job(item):
    spec, ctx = item
    return process_shard(spec, ctx)


def tokenize(platform: str, slug: str, max_len: int, versions: Sequence[str], with_key: bool,
             partitions=None, limit_rows=None, shard_rows=None, workers: int = 1, mdir: Optional[Path] = None,
             only_plan: bool = False) -> Dict:
    if platform in ("dlt", "ravenpack") and tuple(versions) != ("own",):
        raise SystemExit(f"{platform} has only the version 'own'")
    if not 4 <= max_len <= PM.MODELS[slug].max_positions:
        raise SystemExit(f"--max-len {max_len} outside 4..{PM.MODELS[slug].max_positions}")
    mdir = Path(mdir or PM.model_dir(slug))
    run_tag = PM.tag(slug, max_len)
    specs = plan(platform, partitions, limit_rows, shard_rows)
    tok_root = PM.tokens_dir(platform, run_tag)
    if only_plan:
        return {"shards": len(specs), "rows": sum(s["row_stop"] - s["row_start"] for s in specs)}
    sc = selfcheck(slug, max_len, mdir)                       # refuses to start on any id difference
    tok = load_tokenizer(slug, max_len, mdir)
    code_sha = code_fingerprint()
    rules_sha = rules_fingerprint(slug, max_len, versions, with_key, mdir, tok.versions)
    src_inputs = {}
    todo = []
    for spec in specs:
        src = spec["source"]
        if src not in src_inputs:
            src_inputs[src] = runner.describe_inputs([src], with_md5=False)
        final = shard_file(tok_root, spec)
        ctx = {"shard_file": str(final), "max_len": max_len, "versions": tuple(versions), "with_key": with_key,
               "inputs": src_inputs[src], "code_sha": code_sha, "rules_sha": rules_sha, "model": slug}
        if shard_is_current(final, spec, src_inputs[src], code_sha, rules_sha):
            continue
        todo.append((spec, ctx))
    log(f"tokenize {platform} {run_tag}: {len(specs)} shards, {len(todo)} to do, {workers} workers")
    t0 = time.time()
    results = []
    if workers <= 1:
        for spec, ctx in todo:
            results.append(process_shard(spec, ctx, tok))
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(slug, max_len, str(mdir))) as pool:
            for r in pool.imap_unordered(_job, todo):
                results.append(r)
                log(f"  {Path(r['shard']).parent.name}/{Path(r['shard']).name}: {r['rows']:,} rows")
    rec = {"step": STEP + ":tokenize", "platform": platform, "model": slug, "max_len": max_len, "run_tag": run_tag,
           "versions_kept": list(versions), "text_key": with_key, "shards": len(specs), "shards_written": len(results),
           "rows_written": sum(r["rows"] for r in results), "seconds": round(time.time() - t0, 1),
           "library_versions": tok.versions, "selfcheck_passed": all(c["passed"] for c in sc["checks"]),
           "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    runner.write_json_atomic(PM.run_dir(run_tag) / f"stage1_{platform}.json", rec)
    return rec


# --- Optional: distinct texts and text keys (--dedup) ----------------------------------------------
OWN_SQL = {   # the own text as EmTract's stage 1 reads it, with the same row filter
    "reddit": ("own_text", "NOT is_dead"),
    "stocktwits": ("message_body", "TRUE"),
    "twitter": ("text", "TRUE"),
    "dlt": ("text", "TRUE"),
    "ravenpack": ("headline", "TRUE"),
}

def distinct(platform: str, share: float, seed: int, con=None) -> Dict:
    files = [str(f) for _, f in source_files(platform, None)]
    col, where = OWN_SQL[platform]
    con = con or runner.connect()
    flist = "[" + ", ".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    t0 = time.time()
    q = (f"SELECT count(*) AS rows, count({col}) AS texts, count(DISTINCT sha256({col})) AS distinct_texts, "
         f"count(*) FILTER (WHERE {col} IS NOT NULL AND trim({col}) = '') AS blank_texts "
         f"FROM (SELECT {col} FROM read_parquet({flist}) WHERE {where}) "
         f"USING SAMPLE {float(share)}% (bernoulli, {int(seed)})")
    rows, texts, dist, blank = con.execute(q).fetchone()
    full = con.execute(f"SELECT count(*) FROM read_parquet({flist}) WHERE {where}").fetchone()[0]
    rec = {"step": STEP + ":distinct", "platform": platform, "sample": f"bernoulli {share}% seed {seed}",
           "rows_eligible_full": int(full), "rows_sampled": int(rows), "texts_sampled": int(texts),
           "distinct_texts_sampled": int(dist), "blank_texts_sampled": int(blank),
           "distinct_share_sampled": round(dist / texts, 6) if texts else None,
           "note": "within-sample share; the full-corpus share of distinct texts is lower (duplicates "
                   "across the sample boundary are not seen). Provisional.",
           "seconds": round(time.time() - t0, 1), "duckdb": runner.settings(con),
           "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    runner.write_json_atomic(PM.RUN_DIR / "distinct" / f"{platform}.json", rec)
    return rec


def finished_shards_by_bytes(tok_root: Path) -> List[Path]:
    """Token shards whose stats passed and name the shard's bytes (used by `keys`; `pack` and
    `merge` use finished_shards, which also compares the md5)."""
    out = []
    for f in sorted(tok_root.glob("*/shard-*.parquet")):
        sp = runner.stats_path(f)
        if not sp.exists():
            continue
        rec = json.loads(sp.read_text())
        o = rec.get("outputs", [{}])[0]
        if all(c.get("passed") for c in rec.get("checks", [])) and o.get("bytes") == f.stat().st_size:
            out.append(f)
    return out


def keys(platform: str, slug: str, max_len: int) -> Dict:
    import pyarrow.parquet as pq
    run_tag = PM.tag(slug, max_len)
    tok_root = PM.tokens_dir(platform, run_tag).resolve()
    shards = finished_shards_by_bytes(tok_root)
    check(len(shards) > 0, f"{platform}: finished token shards exist")
    check(all("text_key" in pq.read_schema(s).names for s in shards), "every shard has text_key (tokenize --text-key)")
    final = PM.keys_file(platform, run_tag)
    tmp = runner.tmp_path(final)
    con = runner.connect()
    root = str(tok_root) + "/"
    flist = "[" + ", ".join("'" + str(s).replace("'", "''") + "'" for s in shards) + "]"
    con.execute(f"""
        COPY (
          SELECT text_key, min(struct_pack(s := shard, r := shard_row)) AS rep, count(*)::BIGINT AS n_rows
          FROM (SELECT text_key, replace(filename, '{root}', '') AS shard, (file_row_number)::INTEGER AS shard_row
                FROM read_parquet({flist}, filename = true, file_row_number = true) WHERE n_tokens > 0)
          GROUP BY text_key
        ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"COPY (SELECT text_key, rep.s AS rep_shard, rep.r AS rep_row, n_rows FROM read_parquet('{tmp}') "
                f"ORDER BY rep_shard, rep_row) TO '{tmp}.2' (FORMAT parquet, COMPRESSION zstd)")
    os.replace(f"{tmp}.2", tmp)
    checks: List[Dict] = []
    n_keys, n_rows, n_dup = con.execute(f"SELECT count(*), sum(n_rows), count(*) - count(DISTINCT text_key) FROM read_parquet('{tmp}')").fetchone()
    want = con.execute(f"SELECT count(*) FROM read_parquet({flist}) WHERE n_tokens > 0").fetchone()[0]
    check(n_dup == 0, "one row per text_key", checks)
    check(int(n_rows or 0) == int(want), "n_rows sums to the rows with tokens", checks)
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP + ":keys", inputs=runner.describe_inputs(shards), code_sha=code_fingerprint(),
                       rules_sha=run_tag, con=con, counts={"keys": int(n_keys), "rows_with_tokens": int(want)}, checks=checks,
                       extra={"platform": platform, "run_tag": run_tag})
    return {"keys": int(n_keys), "rows_with_tokens": int(want), "file": str(final)}


def main_tokenize(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="07_polarity_finbert.py", description="polarity stage 1 (section 1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PM.MODELS), help="polarity model (default finbert, decision D19; cryptobert for the comparison)")
        p.add_argument("--max-len", type=int, default=None, help="primary length incl. specials (default: the model's)")
        p.add_argument("--model-dir", type=Path, default=None)

    p = sub.add_parser("build-reference"); model_args(p)
    p = sub.add_parser("selfcheck"); model_args(p)
    p = sub.add_parser("tokenize"); model_args(p)
    p.add_argument("platform", choices=PM.PLATFORMS)
    p.add_argument("--versions", nargs="+", default=list(PM.DEFAULT_VERSIONS), choices=PM.VERSIONS)
    p.add_argument("--text-key", action="store_true", help="store sha256(original text) for --dedup")
    p.add_argument("--partitions", nargs="+")
    p.add_argument("--limit-rows", type=int)
    p.add_argument("--shard-rows", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--plan", action="store_true")
    p = sub.add_parser("distinct")
    p.add_argument("platform", choices=PM.PLATFORMS)
    p.add_argument("--share", type=float, default=5.0, help="Bernoulli sample, percent")
    p.add_argument("--seed", type=int, default=20261001)
    p = sub.add_parser("keys"); model_args(p)
    p.add_argument("platform", choices=PM.PLATFORMS)
    a = ap.parse_args(argv)
    ml = None
    if hasattr(a, "model"):
        ml = a.max_len or PM.MODELS[a.model].default_max_len
    if a.cmd == "build-reference":
        print(json.dumps(build_reference(a.model, a.model_dir), indent=1))
    elif a.cmd == "selfcheck":
        rec = selfcheck(a.model, ml, a.model_dir)
        print(json.dumps({k: rec[k] for k in ("model", "max_len", "versions", "n_texts", "n_above_window")}, indent=1))
    elif a.cmd == "tokenize":
        print(json.dumps(tokenize(a.platform, a.model, ml, a.versions, a.text_key, a.partitions, a.limit_rows,
                                  a.shard_rows, a.workers, a.model_dir, a.plan), indent=1))
    elif a.cmd == "distinct":
        print(json.dumps(distinct(a.platform, a.share, a.seed), indent=1))
    elif a.cmd == "keys":
        print(json.dumps(keys(a.platform, a.model, ml), indent=1))
    return 0


# ==================================================================================================
# 2. Stage 2 (`fetch-model`, `pack`, `score`, `collect`): token ids to the model, scores back
# ==================================================================================================
#   fetch-model --model M [--dest DIR] [--sums FILE]        (once locally; and on every GPU machine)
#       Downloads the pinned files of M from huggingface.co at the pinned commit, checks the weights'
#       sha256 against the Hub's LFS oid (and every file against --sums when given), writes SHA256SUMS
#       and SOURCE.json. Standard library only.
#
#   pack <platform> --model M [--max-len L] --batch-id ID --max-inputs N [--windows] [--dedup]
#       Takes finished token shards of stage 1 (stats present, checks passed, bytes and md5 equal),
#       skips shards packed in an earlier batch of this platform and tag, and takes whole shards while
#       the batch stays within N model inputs (a shard is never split across batches). Inputs per row
#       with tokens: the primary prefix + ids[:L-2] + suffix; with --windows also every further window
#       prefix + ids[s:s+L-2] + suffix of stage 1's plan (window 0 is the primary). With --dedup only
#       the representative row of each text_key (keys file of `10 keys`, which must cover exactly the
#       current finished shards) gets inputs; `merge` maps the score back to every row of the key.
#       Writes run_dir/modal_inputs/<platform>/<ID>.parquet with ONLY seq and input_ids
#       (list<uint16>, or uint32 when the vocabulary needs it) -- the only file that goes to the GPU --
#       and the local map <ID>.map.parquet (seq, shard, shard_row, window).
#
#   score --inputs FILE --out FILE --model M [--max-len L] --model-dir D --device cuda|cpu|mps
#         [--precision fp16|bf16|fp32] [--xcheck-every 64] [--batch-tokens 65536]
#       AutoModelForSequenceClassification from the pinned files (weights sha256 = SHA256SUMS), fp32
#       weights, eval, no_grad, TF32 off on CUDA. Inputs sorted by (length, seq), padded to the
#       smallest multiple of 16 <= L that fits (attention mask 0 on padding), at most --batch-tokens
#       padded tokens per batch. --precision fp16 (default, as the brief of 1 Oct 2026) runs the forward
#       pass under torch.autocast(float16); every --xcheck-every-th input of every batch (first one
#       included, so >= 1 per batch) is scored again in plain fp32 in the same bucket: max |dp| and
#       top-label agreement are recorded per batch and in total, the fp32 logits are kept in the output
#       (x_logit_<label>, NULL elsewhere), and the job fails if max |dp| > --xcheck-max-dp (0.05) or
#       agreement < --xcheck-min-agree (0.99). Softmax on the CPU in float32 from the float32-cast
#       logits. Output: seq, logit_<label>, p_<label> (the model's own label order, lower-cased),
#       xcheck, x_logit_<label>; the score stats ride in the Parquet footer and in <out>.stats.json.
#       Needs torch, transformers, numpy, pyarrow and polarity_method.py / emtract_method.py /
#       config.py only, so the same command runs on a rented GPU and on a laptop.
#
#   collect <platform> --model M [--max-len L] --batch-id ID [--raw FILE]
#       Joins the raw scores (default run_dir/modal_outputs/<platform>/<ID>.parquet) with the map:
#       <platform>/polarity/<tag>/scores/<ID>.parquet with seq, shard, shard_row, window, the logits,
#       probabilities and cross-check columns. Checks: the score stats name this batch's inputs
#       sha256, model, revision, labels and length; every score check passed; one score per input;
#       probabilities sum to 1 within 1e-5; every packed shard still has its pack-time md5.
#
# Usage
# -----
#     python pipeline/07_polarity_finbert.py fetch-model
#     python pipeline/07_polarity_finbert.py pack reddit --batch-id reddit-0001 --max-inputs 5000000 --windows
#     python pipeline/07_polarity_finbert.py score --inputs IN.parquet --out OUT.parquet \
#         --model-dir data/raw/polarity_models/finbert --device cuda --precision fp16 --xcheck-every 64
#     python pipeline/07_polarity_finbert.py collect reddit --batch-id reddit-0001
SAMPLE_EVERY = 997                  # pack: every 997th input is rebuilt in plain Python (as step 06's pack)
WINDOW_PROBE_EVERY = 97             # ... and every 97th non-primary window
EMBEDDED_STATS_KEY = b"polarity_score_stats"
XCHECK_EVERY = 64
XCHECK_MAX_DP = 0.05
XCHECK_MIN_AGREE = 0.99


def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _list_arrays(col) -> Tuple[np.ndarray, np.ndarray]:
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    if isinstance(arr, pa.ChunkedArray):
        arr = pa.concat_arrays(arr.chunks)
    offsets = arr.offsets.to_numpy().astype(np.int64)
    values = arr.values.to_numpy(zero_copy_only=False)
    base = offsets[0]
    return values[base:offsets[-1]] if base else values[:offsets[-1]], offsets - base


# --- pack ------------------------------------------------------------------------------------------

def build_inputs(values: np.ndarray, offsets: np.ndarray, rows: np.ndarray, starts: np.ndarray, W: int,
                 prefix_id: int, suffix_id: int, dtype) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised prefix + values[off[r]+s : off[r]+s+W] + suffix for every (row, start); (flat, offsets)."""
    n_row = (offsets[rows + 1] - offsets[rows]).astype(np.int64)
    seg = np.minimum(W, n_row - starts).astype(np.int64)
    if len(seg) and (seg <= 0).any():
        raise CheckError("an input has no content tokens")
    lens = seg + 2
    out_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    total = int(out_off[-1])
    pos = np.arange(total, dtype=np.int64) - np.repeat(out_off[:-1], lens)
    which = np.repeat(np.arange(len(lens)), lens)
    flat = np.empty(total, dtype=dtype)
    first = pos == 0
    last = pos == np.repeat(lens - 1, lens)
    mid = ~(first | last)
    src = (offsets[rows] + starts)[which[mid]] + pos[mid] - 1
    flat[mid] = values[src]
    flat[first] = prefix_id
    flat[last] = suffix_id
    return flat, out_off


def finished_shards(tok_root: Path, run_tag: str) -> List[Path]:
    from pipeline import runner
    out = []
    slug, ml = run_tag.rsplit("-L", 1)
    for f in sorted(tok_root.glob("*/shard-*.parquet")):
        sp = runner.stats_path(f)
        if not sp.exists():
            continue
        rec = json.loads(sp.read_text())
        o = (rec.get("outputs") or [{}])[0]
        if (all(c.get("passed") for c in rec.get("checks", [])) and rec.get("model") == slug
                and int(rec.get("max_len", -1)) == int(ml) and o.get("bytes") == f.stat().st_size
                and o.get("md5") == runner.file_md5(f)):
            out.append(f)
    return out


def packed_shards(inputs_root: Path) -> Dict[str, str]:
    done = {}
    for sp in sorted(inputs_root.glob("*.map.parquet.stats.json")):
        rec = json.loads(sp.read_text())
        for s in rec.get("shards", []):
            done[s["shard"]] = rec.get("batch_id")
    return done


def _shard_meta(path: Path) -> Dict:
    from pipeline import runner
    return json.loads(runner.stats_path(path).read_text())


def pack(platform: str, slug: str, max_len: int, batch_id: str, max_inputs: int, windows: bool = False,
         dedup: bool = False) -> Dict:
    from pipeline import runner
    run_tag = PM.tag(slug, max_len)
    W = max_len - 2
    tok_root = PM.tokens_dir(platform, run_tag).resolve()
    inputs_root = PM.inputs_dir(platform, run_tag)
    inputs_root.mkdir(parents=True, exist_ok=True)
    final_in, final_map = inputs_root / f"{batch_id}.parquet", inputs_root / f"{batch_id}.map.parquet"
    check(not final_in.exists() and not final_map.exists(), f"batch {batch_id} does not exist yet")
    lock = inputs_root / ".pack.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise CheckError(f"another pack holds {lock.name}; one writer per platform")
    try:
        os.write(fd, f"{os.getpid()} {batch_id}\n".encode())
        os.close(fd)
        shards = finished_shards(tok_root, run_tag)
        check(len(shards) > 0, f"{platform} {run_tag}: finished token shards exist")
        metas = {str(s.relative_to(tok_root)): _shard_meta(s) for s in shards}
        sp_set = {json.dumps(m["special_ids"], sort_keys=True) for m in metas.values()}
        check(len(sp_set) == 1, "every shard records the same special ids")
        sp = json.loads(sp_set.pop())
        prefix_id, suffix_id = int(sp["prefix"]), int(sp["suffix"])
        dtype = np.uint16 if sp["vocab_size"] <= 65535 else np.uint32
        earlier = packed_shards(inputs_root)
        modes = {json.loads(p.read_text()).get("mode") for p in inputs_root.glob("*.map.parquet.stats.json")}
        mode = {"windows": bool(windows), "dedup": bool(dedup)}
        check(not modes or modes == {json.dumps(mode, sort_keys=True)}, "every batch of a platform uses the same --windows/--dedup")
        reps = None
        if dedup:
            kf = PM.keys_file(platform, run_tag)
            check(kf.exists(), f"keys file exists ({kf.name}; run `10 keys`)")
            krec = json.loads(runner.stats_path(kf).read_text())
            want = [(i["path"], i["bytes"], i.get("md5")) for i in runner.describe_inputs(shards)]
            check([(i["path"], i["bytes"], i.get("md5")) for i in krec["inputs"]] == want,
                   "keys file covers exactly the current finished shards")
            reps = kf
        todo = [s for s in shards if str(s.relative_to(tok_root)) not in earlier]
        check(len(todo) > 0, f"{platform}: at least one finished, unpacked shard")
        chosen, n_total, plans = [], 0, []
        for s in todo:
            rel = str(s.relative_to(tok_root))
            t = pq.read_table(s, columns=["n_tokens", "window_starts"])
            n_tok = t.column("n_tokens").to_numpy()
            ws_v, ws_o = _list_arrays(t.column("window_starts"))
            k = np.diff(ws_o)
            keep = n_tok > 0
            if reps is not None:
                rr = pq.read_table(reps, columns=["rep_row"], filters=[("rep_shard", "=", rel)]).column("rep_row").to_numpy()
                m = np.zeros(len(n_tok), dtype=bool)
                m[rr] = True
                check(bool((~m | keep).all()), f"{rel}: representatives have tokens")
                keep &= m
            rows_idx = np.flatnonzero(keep)
            if windows:
                cnt = k[rows_idx]
                rows = np.repeat(rows_idx, cnt)
                wno = (np.arange(len(rows)) - np.repeat(np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)).astype(np.int64)
                starts = ws_v[ws_o[rows] + wno].astype(np.int64)
            else:
                rows, wno, starts = rows_idx, np.zeros(len(rows_idx), np.int64), np.zeros(len(rows_idx), np.int64)
            n_in = len(rows)
            if n_in > max_inputs:
                if chosen:
                    break
                raise CheckError(f"shard {rel} alone holds {n_in} inputs > --max-inputs")
            if n_total + n_in > max_inputs:
                break
            chosen.append(s)
            plans.append((rel, rows, wno, starts, len(n_tok)))
            n_total += n_in
        checks: List[Dict] = []
        id_parts, off_parts, map_parts, recs = [], [], [], []
        seq0, flat0 = 0, 0
        for s, (rel, rows, wno, starts, n_rows) in zip(chosen, plans):
            t = pq.read_table(s, columns=["n_tokens", "input_ids"])
            values, offsets = _list_arrays(t.column("input_ids"))
            check(bool((np.diff(offsets) == t.column("n_tokens").to_numpy()).all()), f"{rel}: len(input_ids) == n_tokens")
            if len(values):
                check(int(values.max()) < sp["vocab_size"], f"{rel}: ids below the vocabulary size")
            flat, off = build_inputs(values, offsets, rows, starts, W, prefix_id, suffix_id, dtype)
            probe = sorted(set(range(0, len(rows), SAMPLE_EVERY)) | set(np.flatnonzero(wno > 0)[::WINDOW_PROBE_EVERY].tolist()))
            for i in probe:
                r = int(rows[i])
                ref = [prefix_id] + values[offsets[r] + starts[i]: min(offsets[r + 1], offsets[r] + starts[i] + W)].tolist() + [suffix_id]
                check(flat[off[i]:off[i + 1]].tolist() == ref, f"{rel}: input {i} equals the plain-Python build")
            id_parts.append(flat)
            off_parts.append(off[:-1] + flat0)
            flat0 += len(flat)
            map_parts.append(pa.table({"seq": pa.array(np.arange(seq0, seq0 + len(rows), dtype=np.int64)),
                                       "shard": pa.array([rel] * len(rows), pa.string()),
                                       "shard_row": pa.array(rows.astype(np.int32)),
                                       "window": pa.array(wno.astype(np.int16))}))
            recs.append({"shard": rel, "bytes": s.stat().st_size, "md5": runner.file_md5(s), "rows": n_rows,
                         "inputs": int(len(rows)), "primary": int((wno == 0).sum()),
                         "extra_windows": int((wno > 0).sum()), "probed": len(probe)})
            seq0 += len(rows)
        flat = np.concatenate(id_parts) if id_parts else np.empty(0, dtype)
        offs = np.append(np.concatenate(off_parts) if off_parts else np.empty(0, np.int64), flat0)
        check(len(offs) - 1 == n_total, "inputs == planned total", checks)
        check(int(offs[-1]) <= np.iinfo(np.int32).max, "list offsets fit int32", checks)
        id_type = pa.uint16() if dtype == np.uint16 else pa.uint32()
        tbl = pa.table({"seq": pa.array(np.arange(n_total, dtype=np.int64)),
                        "input_ids": pa.ListArray.from_arrays(pa.array(offs.astype(np.int32)), pa.array(flat, id_type))})
        tmp_in, tmp_map = runner.tmp_path(final_in), runner.tmp_path(final_map)
        pq.write_table(tbl, tmp_in, compression="zstd")
        pq.write_table(pa.concat_tables(map_parts) if map_parts else pa.table({"seq": pa.array([], pa.int64())}), tmp_map, compression="zstd")
        rin, rmap = pq.read_table(tmp_in), pq.read_table(tmp_map)
        check(rin.column_names == ["seq", "input_ids"], "inputs file holds only seq and input_ids", checks)
        check(rin.schema.field("input_ids").type == pa.list_(id_type), f"input_ids is list<{id_type}>", checks)
        seq = rin.column("seq").to_numpy()
        check(bool((seq == np.arange(n_total)).all()), "seq is 0..n-1 in order", checks)
        v, o = _list_arrays(rin.column("input_ids"))
        lens = np.diff(o)
        check(bool(((lens >= 3) & (lens <= max_len)).all()), f"every input has 3..{max_len} ids", checks)
        check(bool((v[o[:-1]] == prefix_id).all()) and bool((v[o[1:] - 1] == suffix_id).all()), "every input is prefix ... suffix", checks)
        check(rmap.num_rows == n_total and bool((rmap.column("seq").to_numpy() == seq).all()), "map: one row per input, same seq", checks)
        check(bool((rmap.column("window").to_numpy()[lens < max_len] == 0).all()) if windows else True,
               "only primaries are shorter than max_len", checks)
        check(sum(r["probed"] for r in recs) > 0, "plain-Python rebuild probed at least one input", checks)
        runner.commit(tmp_in, final_in)
        runner.commit(tmp_map, final_map)
        buckets = PM.pad_buckets(max_len)
        lower = (0,) + buckets[:-1]
        counts = {"shards": len(recs), "rows": sum(r["rows"] for r in recs), "inputs": n_total,
                  "primary": sum(r["primary"] for r in recs), "extra_windows": sum(r["extra_windows"] for r in recs),
                  "by_bucket": {str(b): int(((lens > lo) & (lens <= b)).sum()) for lo, b in zip(lower, buckets)}}
        code_sha = code_fingerprint()
        extra = {"batch_id": batch_id, "platform": platform, "model": slug, "revision": PM.MODELS[slug].revision,
                 "max_len": max_len, "run_tag": run_tag, "special_ids": sp, "mode": json.dumps(mode, sort_keys=True),
                 "shards": recs, "inputs_sha256": sha256_file(final_in), "max_inputs": max_inputs}
        runner.write_stats(final_map, step=STEP + ":pack", inputs=runner.describe_inputs(chosen), code_sha=code_sha,
                           rules_sha=run_tag, con=None, counts=counts, checks=checks, extra=extra)
        runner.write_stats(final_in, step=STEP + ":pack", inputs=[], code_sha=code_sha, rules_sha=run_tag, con=None,
                           counts=counts, checks=checks, extra={k: extra[k] for k in ("batch_id", "platform", "model", "max_len", "special_ids")})
        log(f"pack {platform} {batch_id}: {len(recs)} shards, {n_total:,} inputs ({counts['extra_windows']:,} extra windows)")
    finally:
        if lock.exists():
            lock.unlink()
    return {"inputs_file": str(final_in), "map_file": str(final_map), **counts}


# --- score (the same code on any machine: CUDA, Apple MPS or CPU) ----------------------------------

def plan_batches(lengths: np.ndarray, seq: np.ndarray, batch_tokens: int, buckets: Sequence[int]) -> List[Tuple[int, np.ndarray]]:
    order = np.lexsort((seq, lengths))
    b = np.asarray(buckets)
    idx = np.searchsorted(b, lengths[order], side="left")
    if len(idx) and idx.max() >= len(b):
        raise CheckError("an input is longer than the largest pad bucket")
    which = b[idx]
    out = []
    for bk in buckets:
        sel = order[which == bk]
        bs = max(1, batch_tokens // bk)
        for i in range(0, len(sel), bs):
            out.append((int(bk), sel[i:i + bs]))
    return out


def pad_batch(values: np.ndarray, offsets: np.ndarray, idx: np.ndarray, bucket: int, pad_id: int):
    lens = (offsets[idx + 1] - offsets[idx]).astype(np.int64)
    ids = np.full((len(idx), bucket), pad_id, dtype=np.int64)
    mask = np.zeros((len(idx), bucket), dtype=np.int64)
    col = np.arange(bucket)[None, :]
    m = col < lens[:, None]
    ids[m] = values[(offsets[idx][:, None] + col)[m]]
    mask[m] = 1
    return ids, mask


def softmax_f32(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float32) - logits.astype(np.float32).max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def load_model(slug: str, mdir: Path, device: str):
    import torch
    import transformers
    transformers.logging.set_verbosity_error()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = transformers.AutoModelForSequenceClassification.from_pretrained(str(mdir), local_files_only=True)
    labels = tuple(str(model.config.id2label[i]).lower() for i in range(model.config.num_labels))
    if labels != PM.MODELS[slug].labels:
        raise CheckError(f"{slug}: config labels {labels} != registry {PM.MODELS[slug].labels}")
    return model.to(torch.float32).to(device).eval(), labels


def score(inputs: Path, out: Path, slug: str, max_len: int, device: str, mdir: Path, precision: str = "fp16",
          xcheck_every: int = XCHECK_EVERY, xcheck_max_dp: float = XCHECK_MAX_DP,
          xcheck_min_agree: float = XCHECK_MIN_AGREE, batch_tokens: int = PM.DEFAULT_BATCH_TOKENS,
          threads: Optional[int] = None) -> Dict:
    import torch
    import transformers
    t_start = time.time()
    checks: List[Dict] = []
    mdir = Path(mdir)
    if threads:
        torch.set_num_threads(int(threads))
    files = PM.check_model_files(slug, mdir)
    check(True, "model files (weights included) equal SHA256SUMS", checks)
    tok = PM.PolarityTokenizer(mdir, max_len)
    tbl = pq.read_table(inputs)
    check(tbl.column_names == ["seq", "input_ids"], "inputs hold only seq and input_ids", checks)
    seq = tbl.column("seq").to_numpy().astype(np.int64)
    n = len(seq)
    check(bool((np.sort(seq) == np.arange(n)).all()), "seq is a permutation of 0..n-1", checks)
    values, offsets = _list_arrays(tbl.column("input_ids"))
    values = values.astype(np.int64)
    lengths = np.diff(offsets)
    if n:
        check(bool(((lengths >= 3) & (lengths <= max_len)).all()), f"lengths 3..{max_len}", checks)
        check(bool((values[offsets[:-1]] == tok.prefix_id).all() and (values[offsets[1:] - 1] == tok.suffix_id).all()),
               "every input is prefix ... suffix", checks)
        check(int(values.max()) < tok.vocab_size, "ids below the vocabulary size", checks)
    t0 = time.time()
    model, labels = load_model(slug, mdir, device)
    check(True, f"config labels equal the registry {labels}", checks)
    load_s = time.time() - t0
    check(all(p.dtype == torch.float32 for p in model.parameters()), "every parameter is float32", checks)
    if device == "cuda":
        check(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False, "TF32 off", checks)
    ac_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[precision]
    dev_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")
    L = len(labels)
    logits = np.empty((n, L), dtype=np.float32)
    xlog = np.full((n, L), np.nan, dtype=np.float32)
    xmask = np.zeros(n, dtype=bool)
    batches = plan_batches(lengths, seq, batch_tokens, PM.pad_buckets(max_len))
    per_batch = []
    t1 = time.time()
    with torch.no_grad():
        for bucket, idx in batches:
            ids, mask = pad_batch(values, offsets, idx, bucket, tok.pad_id)
            ti, tm = torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)
            if ac_dtype is None:
                lo = model(input_ids=ti, attention_mask=tm).logits
            else:
                with torch.autocast(device_type=dev_type, dtype=ac_dtype):
                    lo = model(input_ids=ti, attention_mask=tm).logits
            lo = lo.float().cpu().numpy()
            logits[seq[idx]] = lo
            if ac_dtype is not None and xcheck_every > 0:
                sub = np.arange(0, len(idx), xcheck_every)
                lo32 = model(input_ids=ti[sub], attention_mask=tm[sub]).logits.float().cpu().numpy()
                rows = seq[idx[sub]]
                xlog[rows] = lo32
                xmask[rows] = True
                dp = np.abs(softmax_f32(lo[sub]) - softmax_f32(lo32))
                per_batch.append([bucket, int(len(idx)), int(len(sub)), float(dp.max()),
                                  int((lo[sub].argmax(1) == lo32.argmax(1)).sum())])
    if device == "cuda":
        torch.cuda.synchronize()
    score_s = time.time() - t1
    probs = softmax_f32(logits)
    check(bool(np.isfinite(logits).all() and np.isfinite(probs).all()), "logits and probabilities finite", checks)
    check(bool((np.abs(probs.astype(np.float64).sum(axis=1) - 1) <= PM.PROB_SUM_TOL).all()), "probabilities sum to 1 within 1e-5", checks)
    check(bool((logits.argmax(1) == probs.argmax(1)).all()) if n else True, "argmax logits == argmax probabilities", checks)
    xc = None
    if per_batch:
        n_x = sum(b[2] for b in per_batch)
        agree = sum(b[4] for b in per_batch)
        xc = {"every": xcheck_every, "inputs_checked": n_x, "share_checked": round(n_x / max(n, 1), 5),
              "max_abs_dp": max(b[3] for b in per_batch), "label_agree": agree,
              "label_agree_share": round(agree / max(n_x, 1), 6), "label_flips": n_x - agree,
              "per_batch_columns": ["bucket", "inputs", "checked", "max_abs_dp", "agree"], "per_batch": per_batch}
        check(len(per_batch) == len(batches), "every batch has a fp32 cross-check subset", checks)
        check(xc["max_abs_dp"] <= xcheck_max_dp, f"fp32 cross-check: max |dp| {xc['max_abs_dp']:.2e} <= {xcheck_max_dp}", checks)
        check(xc["label_agree_share"] >= xcheck_min_agree, f"fp32 cross-check: label agreement {xc['label_agree_share']} >= {xcheck_min_agree}", checks)
    cols = {"seq": pa.array(np.arange(n, dtype=np.int64))}
    for j, lab in enumerate(labels):
        cols[f"logit_{lab}"] = pa.array(logits[:, j])
    for j, lab in enumerate(labels):
        cols[f"p_{lab}"] = pa.array(probs[:, j])
    cols["xcheck"] = pa.array(xmask)
    for j, lab in enumerate(labels):
        cols[f"x_logit_{lab}"] = pa.array(xlog[:, j], mask=~xmask)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    rec = {"step": STEP + ":score", "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "model": slug, "repo": PM.MODELS[slug].repo, "revision": PM.MODELS[slug].revision, "max_len": max_len,
           "labels": list(labels), "device": device,
           "device_name": torch.cuda.get_device_name(0) if device == "cuda" else device,
           "host_machine": _platform.machine(), "python": sys.version.split()[0], "torch": torch.__version__,
           "transformers": transformers.__version__, "numpy": np.__version__, "pyarrow": pa.__version__,
           "cuda": getattr(torch.version, "cuda", None), "precision": precision,
           "autocast": None if ac_dtype is None else str(ac_dtype),
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
           "batch_tokens": batch_tokens, "pad_buckets": list(PM.pad_buckets(max_len)), "torch_threads": torch.get_num_threads(),
           "inputs_file": Path(inputs).name, "inputs_sha256": sha256_file(inputs), "model_files_sha256": files,
           "special_ids": tok.specials(),
           "counts": {"inputs": n, "batches": len(batches)},
           "xcheck": xc,
           "timing_s": {"load": round(load_s, 3), "score": round(score_s, 3), "total": round(time.time() - t_start, 3)},
           "inputs_per_s_scoring": round(n / score_s, 1) if score_s > 0 else None, "checks": checks}
    table = pa.table(cols).replace_schema_metadata({EMBEDDED_STATS_KEY: json.dumps(rec).encode()})
    pq.write_table(table, tmp, compression="zstd")
    check(pq.read_metadata(tmp).num_rows == n, "written scores: one row per input", checks)
    os.replace(tmp, out)
    Path(str(out) + ".stats.json").write_text(json.dumps(rec, indent=1))
    log(f"score {Path(inputs).name}: {n:,} inputs, {precision}, {rec['device_name']}, {score_s:.1f} s"
         + (f", xcheck max|dp| {xc['max_abs_dp']:.2e}, flips {xc['label_flips']}" if xc else ""))
    return rec


# --- collect ---------------------------------------------------------------------------------------

def read_raw_stats(raw: Path) -> Dict:
    side = Path(str(raw) + ".stats.json")
    if side.exists():
        return json.loads(side.read_text())
    md = pq.read_schema(raw).metadata or {}
    if EMBEDDED_STATS_KEY not in md:
        raise CheckError(f"{raw.name}: no stats file and no score stats in the Parquet footer")
    return json.loads(md[EMBEDDED_STATS_KEY].decode())


def collect(platform: str, slug: str, max_len: int, batch_id: str, raw: Optional[Path] = None) -> Dict:
    from pipeline import runner
    run_tag = PM.tag(slug, max_len)
    inputs_root = PM.inputs_dir(platform, run_tag)
    tok_root = PM.tokens_dir(platform, run_tag).resolve()
    final_in, final_map = inputs_root / f"{batch_id}.parquet", inputs_root / f"{batch_id}.map.parquet"
    raw = Path(raw or (PM.raw_dir(platform, run_tag) / f"{batch_id}.parquet"))
    final = PM.scores_dir(platform, run_tag) / f"{batch_id}.parquet"
    check(not final.exists(), f"scores of {batch_id} not collected yet")
    checks: List[Dict] = []
    mrec = json.loads(runner.stats_path(final_map).read_text())
    srec = read_raw_stats(raw)
    check(all(c.get("passed") for c in srec.get("checks", [])), "every score check passed", checks)
    check(srec.get("inputs_sha256") == mrec["inputs_sha256"] == sha256_file(final_in),
           "score stats name this batch's inputs (sha256 unchanged since pack)", checks)
    check((srec.get("model"), srec.get("revision"), int(srec.get("max_len", -1))) == (slug, PM.MODELS[slug].revision, max_len),
           "scored with the pinned model, revision and length", checks)
    labels = tuple(srec["labels"])
    check(labels == PM.MODELS[slug].labels, "label order equals the registry", checks)
    for s in mrec["shards"]:
        p = tok_root / s["shard"]
        check(p.exists() and runner.file_md5(p) == s["md5"], f"{s['shard']}: unchanged since pack", checks)
    sc = pq.read_table(raw)
    mp_ = pq.read_table(final_map)
    check(sc.num_rows == mp_.num_rows == mrec["counts"]["inputs"] == srec["counts"]["inputs"], "one score per input", checks)
    check(bool((sc.column("seq").to_numpy() == mp_.column("seq").to_numpy()).all()), "score seq == map seq", checks)
    P = np.stack([sc.column(f"p_{lab}").to_numpy() for lab in labels], axis=1).astype(np.float64)
    check(bool((np.abs(P.sum(1) - 1) <= PM.PROB_SUM_TOL).all()), "probabilities sum to 1 within 1e-5", checks)
    out = mp_
    for name in sc.column_names[1:]:
        out = out.append_column(name, sc.column(name))
    tmp = runner.tmp_path(final)
    pq.write_table(out, tmp, compression="zstd")
    check(pq.read_metadata(tmp).num_rows == out.num_rows, "written: one row per input", checks)
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP + ":collect", inputs=runner.describe_inputs([final_map, raw]),
                       code_sha=code_fingerprint(), rules_sha=run_tag, con=None,
                       counts={"inputs": out.num_rows, "shards": len(mrec["shards"])}, checks=checks,
                       extra={"batch_id": batch_id, "platform": platform, "model": slug, "max_len": max_len,
                              "labels": list(labels), "shards": mrec["shards"], "mode": mrec["mode"],
                              "score_stats": {k: srec.get(k) for k in ("device_name", "precision", "torch", "transformers",
                                                                       "cuda", "timing_s", "inputs_per_s_scoring")},
                              "xcheck": {k: v for k, v in (srec.get("xcheck") or {}).items() if k != "per_batch"}})
    log(f"collect {platform} {batch_id}: {out.num_rows:,} scores")
    return {"scores_file": str(final), "inputs": out.num_rows}


def main_stage2(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="07_polarity_finbert.py", description="polarity stage 2 (section 2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def margs(p):
        p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PM.MODELS), help="polarity model (default finbert, decision D19; cryptobert for the comparison)")
        p.add_argument("--max-len", type=int, default=None)

    p = sub.add_parser("fetch-model")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PM.MODELS), help="polarity model (default finbert, decision D19; cryptobert for the comparison)")
    p.add_argument("--dest", type=Path)
    p.add_argument("--sums", type=Path)
    p = sub.add_parser("pack"); margs(p)
    p.add_argument("platform", choices=PM.PLATFORMS)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--max-inputs", type=int, required=True)
    p.add_argument("--windows", action="store_true")
    p.add_argument("--dedup", action="store_true")
    p = sub.add_parser("score"); margs(p)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--xcheck-every", type=int, default=XCHECK_EVERY)
    p.add_argument("--xcheck-max-dp", type=float, default=XCHECK_MAX_DP)
    p.add_argument("--xcheck-min-agree", type=float, default=XCHECK_MIN_AGREE)
    p.add_argument("--batch-tokens", type=int, default=PM.DEFAULT_BATCH_TOKENS)
    p.add_argument("--threads", type=int)
    p = sub.add_parser("collect"); margs(p)
    p.add_argument("platform", choices=PM.PLATFORMS)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--raw", type=Path)
    a = ap.parse_args(argv)
    ml = (a.max_len or PM.MODELS[a.model].default_max_len) if hasattr(a, "max_len") else None
    try:
        if a.cmd == "fetch-model":
            r = PM.fetch_model(a.model, a.dest, a.sums)
            print(json.dumps({k: v["sha256"][:16] for k, v in r["files"].items()}, indent=1))
        elif a.cmd == "pack":
            print(json.dumps(pack(a.platform, a.model, ml, a.batch_id, a.max_inputs, a.windows, a.dedup), indent=1))
        elif a.cmd == "score":
            rec = score(a.inputs, a.out, a.model, ml, a.device, a.model_dir or PM.model_dir(a.model), a.precision,
                        a.xcheck_every, a.xcheck_max_dp, a.xcheck_min_agree, a.batch_tokens, a.threads)
            print(json.dumps({k: rec[k] for k in ("counts", "timing_s", "inputs_per_s_scoring")}, indent=1))
        elif a.cmd == "collect":
            print(json.dumps(collect(a.platform, a.model, ml, a.batch_id, a.raw), indent=1))
    except CheckError as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


# ==================================================================================================
# 3. Merge (`merge`): one polarity row per (row_id, version), keyed like the EmTract tables
# ==================================================================================================
# Input   stage 1: <platform>/polarity/<tag>/tokens/<partition>/shard-NNNNN.parquet (+ stats)
#         stage 2: <platform>/polarity/<tag>/scores/<batch>.parquet (+ stats), one row per model input
#         with --dedup batches: run_dir/keys/<platform>.parquet (text_key -> representative row)
# Output  <platform>/polarity/<tag>/unit_scores/<batch>.parquet   one row per scored (shard, shard_row)
#         <platform>/polarity/<tag>/polarity/<partition>.parquet  (+ stats) one row per (row_id, version)
#         run_dir/completeness_<platform>.json
#
# Columns of the polarity table
#   row_id, version, n_tokens, is_missing_text, is_empty_text   from stage 1
#   n_windows        windows of stage 1's plan (0 without tokens, 1 for 1..L-2 tokens, k above)
#   logit_<label>, p_<label>   primary input (prefix + first L-2 tokens + suffix), the model's label order
#   polarity         P(pos_label) - P(neg_label) of the primary (FinBERT P(positive) - P(negative), the
#                    repo's own sentiment_score; CryptoBERT P(bullish) - P(bearish)); D19 facts §10, rule 5
#   pc_<label>, polarity_c   equal-weight mean of the k windows' probabilities (only for batches packed
#                    with --windows; NULL when k < 2): the linear pool, as EmTract's merge (Pappagari et al. 2019)
#   pw_<label>, polarity_w   coverage-weighted mean (EmTract's sensitivity check; own derivation there)
#   score_source     'scored' (the row's own input) or 'key' (the score of its text_key's representative;
#                    only with --dedup); NULL without a score
#   All probabilities, logits and polarity NULL exactly for rows with n_tokens = 0 (missing or empty).
#   Means in float64 from the stored float32 probabilities, stored as float32.
#
# Checks (CheckError; nothing committed)
#   unit scores (per batch): one primary per scored row; with windows, exactly the planned windows
#     (0..k-1) per row; softmax(logits) = p within 1e-5; p sums to 1 within 1e-5; coverage weights
#     sum to n_tokens.
#   completeness (per platform): every finished token shard is packed in exactly one batch and every
#     batch is collected; every row with tokens has a score (own or by key) -- written to
#     completeness_<platform>.json with the counts, also when it fails.
#   output (per partition, on the temporary file): rows = token rows; (row_id, version) unique; p NULL
#     exactly for n_tokens = 0; polarity = p_pos - p_neg within 1e-6 and in [-1, 1]; p sums to 1;
#     pc/pw NULL exactly where k < 2 or windows were not scored; and, where the EmTract table exists,
#     the same (row_id, version) set as EmTract's emotion table for the versions kept (anti-join both ways).
#
# Usage
# -----
#     python pipeline/07_polarity_finbert.py merge reddit [--max-len 64] [--partitions BTC] [--force]
#     python pipeline/07_polarity_finbert.py merge twitter --model cryptobert --max-len 128 --check-only
SOFTMAX_TOL = 1e-5


def emtract_table(platform: str, partition: str) -> Optional[Path]:
    if platform in ("reddit", "stocktwits", "twitter"):
        p = config.out_path(platform, "emtract", "emotion", f"{partition}.parquet")
    elif platform == "dlt":
        p = config.out_path("dlt", "emtract", "emotion", "dlt_emotion.parquet")
    else:
        p = config.out_path("ravenpack", "emtract", "news_headline_emotion.parquet")
    return p if p.exists() else None


def _np(col) -> np.ndarray:
    return col.to_numpy(zero_copy_only=False) if hasattr(col, "to_numpy") else np.asarray(col)


def softmax64(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --- Unit scores and completeness ------------------------------------------------------------------

def unit_scores(platform: str, run_tag: str, batch: Path, labels: Sequence[str], windows: bool) -> Path:
    final = PM.units_dir(platform, run_tag) / batch.name
    inputs = runner.describe_inputs([batch])
    if runner.is_current(final, inputs, code_fingerprint(), run_tag):
        return final
    max_len = int(run_tag.rsplit("-L", 1)[1])
    rule = PM.WindowRule(max_len)
    tok_root = PM.tokens_dir(platform, run_tag)
    checks: List[Dict] = []
    t = pq.read_table(batch)
    L = list(labels)
    P = np.stack([_np(t.column(f"p_{l}")) for l in L], 1).astype(np.float64)
    Z = np.stack([_np(t.column(f"logit_{l}")) for l in L], 1)
    check(bool((np.abs(softmax64(Z) - P) <= SOFTMAX_TOL).all()), "softmax(logits) == p within 1e-5", checks)
    check(bool((np.abs(P.sum(1) - 1) <= PM.PROB_SUM_TOL).all()), "p sums to 1 within 1e-5", checks)
    shard = _np(t.column("shard")).astype(object)
    srow = _np(t.column("shard_row")).astype(np.int64)
    win = _np(t.column("window")).astype(np.int64)
    parts = []
    for s in sorted(set(shard.tolist())):
        sel = np.flatnonzero(shard == s)
        tk = pq.read_table(tok_root / s, columns=["n_tokens", "window_starts"])
        n_tok = _np(tk.column("n_tokens")).astype(np.int64)
        ws = tk.column("window_starts").to_pylist()
        order = sel[np.lexsort((win[sel], srow[sel]))]
        r, w = srow[order], win[order]
        uniq, first, cnt = np.unique(r, return_index=True, return_counts=True)
        check(bool((w[first] == 0).all()), f"{s}: one primary (window 0) per scored row", checks)
        check(bool((n_tok[uniq] > 0).all()), f"{s}: scored rows have tokens", checks)
        k_plan = np.array([len(ws[i]) for i in uniq], dtype=np.int64)
        if windows:
            check(bool((cnt == k_plan).all()), f"{s}: every planned window scored once", checks)
            exp = np.concatenate([np.arange(c) for c in cnt]) if len(cnt) else np.empty(0, np.int64)
            check(bool((w == exp).all()), f"{s}: windows 0..k-1 per row", checks)
        else:
            check(bool((cnt == 1).all()), f"{s}: one input per row (primary only)", checks)
        cols = {"shard": pa.array([s] * len(uniq), pa.string()), "shard_row": pa.array(uniq.astype(np.int32)),
                "n_inputs": pa.array(cnt.astype(np.int16))}
        prim = order[first]
        for j, l in enumerate(L):
            cols[f"logit_{l}"] = pa.array(Z[prim, j].astype(np.float32))
        for j, l in enumerate(L):
            cols[f"p_{l}"] = pa.array(P[prim, j].astype(np.float32))
        pc = np.full((len(uniq), len(L)), np.nan)
        pw = np.full((len(uniq), len(L)), np.nan)
        if windows:
            bounds = np.concatenate([[0], np.cumsum(cnt)])
            for u in np.flatnonzero(cnt >= 2):
                rows_u = order[bounds[u]:bounds[u + 1]]
                Pu = P[rows_u]
                pc[u] = Pu.mean(0)
                wts = rule.coverage_weights(int(n_tok[uniq[u]]), ws[uniq[u]])
                check(abs(float(wts.sum()) - float(n_tok[uniq[u]])) < 1e-6, f"{s}: coverage weights sum to n", checks)
                pw[u] = (wts[:, None] * Pu).sum(0) / wts.sum()
        has = ~np.isnan(pc[:, 0])
        for j, l in enumerate(L):
            cols[f"pc_{l}"] = pa.array(pc[:, j].astype(np.float32), mask=~has)
        for j, l in enumerate(L):
            cols[f"pw_{l}"] = pa.array(pw[:, j].astype(np.float32), mask=~has)
        parts.append(pa.table(cols))
    out = pa.concat_tables(parts)
    tmp = runner.tmp_path(final)
    pq.write_table(out, tmp, compression="zstd")
    check(pq.read_metadata(tmp).num_rows == out.num_rows, "unit scores written", checks)
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP_MERGE + ":units", inputs=inputs, code_sha=code_fingerprint(), rules_sha=run_tag, con=None,
                       counts={"units": out.num_rows, "inputs": t.num_rows}, checks=checks,
                       extra={"labels": L, "windows": windows})
    return final


def completeness(platform: str, run_tag: str, slug: str) -> Dict:
    tok_root = PM.tokens_dir(platform, run_tag).resolve()
    finished = [str(p.relative_to(tok_root)) for p in finished_shards(tok_root, run_tag)]
    all_shards = sorted(str(p.relative_to(tok_root)) for p in tok_root.glob("*/shard-*.parquet"))
    maps = sorted(PM.inputs_dir(platform, run_tag).glob("*.map.parquet.stats.json"))
    packed: Counter = Counter()
    batches, modes = {}, set()
    for m in maps:
        rec = json.loads(m.read_text())
        modes.add(rec.get("mode"))
        batches[rec["batch_id"]] = {"inputs": rec["counts"]["inputs"], "shards": [s["shard"] for s in rec["shards"]]}
        for s in rec["shards"]:
            packed[s["shard"]] += 1
    collected = {}
    for b in batches:
        sp = runner.stats_path(PM.scores_dir(platform, run_tag) / f"{b}.parquet")
        if sp.exists():
            r = json.loads(sp.read_text())
            collected[b] = all(c.get("passed") for c in r.get("checks", [])) and r["counts"]["inputs"] == batches[b]["inputs"]
    rec = {"platform": platform, "run_tag": run_tag, "model": slug,
           "token_shards": len(all_shards), "finished_shards": len(finished),
           "unfinished_shards": sorted(set(all_shards) - set(finished))[:20],
           "batches": len(batches), "batches_collected": sum(1 for v in collected.values() if v),
           "batches_not_collected": sorted(b for b in batches if not collected.get(b)),
           "shards_not_packed": sorted(set(finished) - set(packed))[:20],
           "n_shards_not_packed": len(set(finished) - set(packed)),
           "shards_packed_twice": sorted(s for s, c in packed.items() if c > 1),
           "packed_shards_not_finished": sorted(set(packed) - set(finished))[:20],
           "modes": sorted(m for m in modes if m), "inputs_packed": sum(v["inputs"] for v in batches.values())}
    rec["complete"] = (len(all_shards) == len(finished) > 0 and not rec["n_shards_not_packed"]
                       and not rec["shards_packed_twice"] and not rec["packed_shards_not_finished"]
                       and not rec["batches_not_collected"] and len(modes) == 1)
    runner.write_json_atomic(PM.run_dir(run_tag) / f"completeness_{platform}.json", rec)
    return rec


# --- One partition ---------------------------------------------------------------------------------

def _sql_list(paths) -> str:
    return "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def merge_partition(platform: str, slug: str, max_len: int, partition: str, unit_files: List[Path],
                    windows: bool, dedup: bool, force: bool = False) -> Dict:
    run_tag = PM.tag(slug, max_len)
    spec = PM.MODELS[slug]
    L = spec.labels
    tok_root = PM.tokens_dir(platform, run_tag).resolve()
    shards = sorted((tok_root / partition).glob("shard-*.parquet"))
    check(len(shards) > 0, f"{platform}/{partition}: token shards exist")
    final = PM.polarity_dir(platform, run_tag) / f"{partition}.parquet"
    extra_in = [PM.keys_file(platform, run_tag)] if dedup else []
    inputs = runner.describe_inputs(list(shards) + list(unit_files) + extra_in)
    if not force and runner.is_current(final, inputs, code_fingerprint(), run_tag):
        log(f"{platform}/{partition}: current, skipped")
        return json.loads(runner.stats_path(final).read_text())["counts"]
    con = runner.connect()
    checks: List[Dict] = []
    root = str(tok_root) + "/"
    has_key = "text_key" in pq.read_schema(shards[0]).names
    con.execute(f"""CREATE TEMP VIEW tok AS SELECT row_id, version, n_tokens, is_missing_text, is_empty_text,
                    len(window_starts)::SMALLINT AS n_windows, {'text_key,' if has_key else ''}
                    replace(filename, '{root}', '') AS shard, file_row_number::INTEGER AS shard_row
                    FROM read_parquet({_sql_list(shards)}, filename = true, file_row_number = true)""")
    con.execute(f"CREATE TEMP VIEW u AS SELECT * FROM read_parquet({_sql_list(unit_files)})")
    pos, neg = spec.pos_label, spec.neg_label
    pcols = ", ".join([f"u.logit_{l}" for l in L] + [f"u.p_{l}" for l in L])
    win_cols = ""
    if windows:
        win_cols = (", " + ", ".join(f"u.pc_{l}" for l in L) + f", (u.pc_{pos}::DOUBLE - u.pc_{neg})::FLOAT AS polarity_c, "
                    + ", ".join(f"u.pw_{l}" for l in L) + f", (u.pw_{pos}::DOUBLE - u.pw_{neg})::FLOAT AS polarity_w")
    if dedup:
        check(has_key, "token shards have text_key (needed by --dedup)", checks)
        kf = PM.keys_file(platform, run_tag)
        join = (f"LEFT JOIN read_parquet('{kf}') k ON k.text_key = t.text_key AND t.n_tokens > 0 "
                f"LEFT JOIN u ON u.shard = k.rep_shard AND u.shard_row = k.rep_row")
        src = ("CASE WHEN u.shard IS NULL THEN NULL WHEN u.shard = t.shard AND u.shard_row = t.shard_row "
               "THEN 'scored' ELSE 'key' END")
    else:
        join = "LEFT JOIN u ON u.shard = t.shard AND u.shard_row = t.shard_row"
        src = "CASE WHEN u.shard IS NULL THEN NULL ELSE 'scored' END"
    tmp = runner.tmp_path(final)
    con.execute(f"""COPY (SELECT t.row_id, t.version, t.n_tokens, t.is_missing_text, t.is_empty_text, t.n_windows,
                    {pcols}, (u.p_{pos}::DOUBLE - u.p_{neg})::FLOAT AS polarity {win_cols}, {src} AS score_source
                    FROM tok t {join} ORDER BY t.shard, t.shard_row) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""")
    o = f"read_parquet('{tmp}')"
    n_tok_rows = con.execute("SELECT count(*) FROM tok").fetchone()[0]
    n_out, n_keys = con.execute(f"SELECT count(*), count(DISTINCT (row_id, version)) FROM {o}").fetchone()
    check(n_out == n_tok_rows, f"rows = token rows ({n_out} vs {n_tok_rows})", checks)
    check(n_keys == n_out, "(row_id, version) unique", checks)
    unscored, stray = con.execute(f"SELECT count(*) FILTER (WHERE n_tokens > 0 AND p_{L[0]} IS NULL), "
                                  f"count(*) FILTER (WHERE n_tokens = 0 AND p_{L[0]} IS NOT NULL) FROM {o}").fetchone()
    check(unscored == 0, f"every row with tokens has a score ({unscored} without)", checks)
    check(stray == 0, "no score for rows without tokens", checks)
    psum = " + ".join(f"p_{l}::DOUBLE" for l in L)
    bad_sum, bad_pol = con.execute(f"SELECT count(*) FILTER (WHERE abs({psum} - 1) > {PM.PROB_SUM_TOL}), "
                                   f"count(*) FILTER (WHERE abs(polarity - (p_{pos}::DOUBLE - p_{neg})) > 1e-6 OR abs(polarity) > 1) "
                                   f"FROM {o} WHERE p_{L[0]} IS NOT NULL").fetchone()
    check(bad_sum == 0, "p sums to 1 within 1e-5", checks)
    check(bad_pol == 0, "polarity = p_pos - p_neg, within [-1, 1]", checks)
    if windows:
        bad_pc = con.execute(f"SELECT count(*) FROM {o} WHERE (pc_{L[0]} IS NULL) <> (n_windows < 2)").fetchone()[0]
        check(bad_pc == 0, "pc/pw NULL exactly where n_windows < 2", checks)
    counts = dict(zip(["rows", "rows_with_tokens", "missing", "empty", "long", "scored_self", "scored_by_key"],
                      [int(x or 0) for x in con.execute(
                          f"SELECT count(*), count(*) FILTER (WHERE n_tokens > 0), count(*) FILTER (WHERE is_missing_text), "
                          f"count(*) FILTER (WHERE is_empty_text), count(*) FILTER (WHERE n_windows > 1), "
                          f"count(*) FILTER (WHERE score_source = 'scored'), count(*) FILTER (WHERE score_source = 'key') "
                          f"FROM {o}").fetchone()]))
    counts["by_version"] = dict(con.execute(f"SELECT version, count(*) FROM {o} GROUP BY 1 ORDER BY 1").fetchall())
    em = emtract_table(platform, partition)
    if em is not None:
        if platform == "ravenpack":
            q = (f"SELECT (SELECT count(*) FROM {o} a ANTI JOIN read_parquet('{em}') b ON a.row_id = b.headline_md5), "
                 f"(SELECT count(*) FROM read_parquet('{em}') b ANTI JOIN {o} a ON a.row_id = b.headline_md5)")
        else:
            vers = "(" + ", ".join(f"'{v}'" for v in counts["by_version"]) + ")"
            q = (f"SELECT (SELECT count(*) FROM {o} a ANTI JOIN (SELECT * FROM read_parquet('{em}') WHERE version IN {vers}) b "
                 f"ON a.row_id = b.row_id AND a.version = b.version), "
                 f"(SELECT count(*) FROM (SELECT * FROM read_parquet('{em}') WHERE version IN {vers}) b ANTI JOIN {o} a "
                 f"ON a.row_id = b.row_id AND a.version = b.version)")
        only_pol, only_em = con.execute(q).fetchone()
        check(only_pol == 0 and only_em == 0,
              f"same (row_id, version) set as the EmTract table {em.name} ({only_pol} only here, {only_em} only there)", checks)
        counts["emtract_table_checked"] = str(em)
    runner.commit(tmp, final)
    runner.write_stats(final, step=STEP_MERGE, inputs=inputs, code_sha=code_fingerprint(), rules_sha=run_tag, con=con, counts=counts,
                       checks=checks, extra={"platform": platform, "partition": partition, "model": slug,
                                             "repo": spec.repo, "revision": spec.revision, "max_len": max_len,
                                             "labels": list(L), "polarity": f"p_{pos} - p_{neg}",
                                             "windows": windows, "dedup": dedup})
    log(f"{platform}/{partition}: {counts['rows']:,} rows, {counts['rows_with_tokens']:,} scored")
    return counts


def merge(platform: str, slug: str, max_len: int, partitions: Optional[Sequence[str]] = None,
          force: bool = False, check_only: bool = False) -> Dict:
    run_tag = PM.tag(slug, max_len)
    comp = completeness(platform, run_tag, slug)
    if check_only:
        return comp
    check(comp["complete"], f"{platform} {run_tag}: run complete (see completeness_{platform}.json)")
    mode = json.loads(comp["modes"][0])
    labels = PM.MODELS[slug].labels
    units = []
    for f in sorted(PM.scores_dir(platform, run_tag).glob("*.parquet")):
        units.append(unit_scores(platform, run_tag, f, labels, mode["windows"]))
    tok_root = PM.tokens_dir(platform, run_tag)
    parts = sorted(p.name for p in tok_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if partitions:
        parts = [p for p in parts if p in set(partitions)]
    unit_parts = {u: {s.split("/", 1)[0] for s in set(pq.read_table(u, columns=["shard"]).column("shard").to_pylist())}
                  for u in units}
    out = {}
    for p in parts:
        uf = units if mode["dedup"] else [u for u in units if p in unit_parts[u]]
        out[p] = merge_partition(platform, slug, max_len, p, uf, mode["windows"], mode["dedup"], force)
    return {"completeness": comp, "partitions": out}


def main_merge(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="07_polarity_finbert.py merge", description="polarity merge (section 3)")
    ap.add_argument("platform", choices=PM.PLATFORMS)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PM.MODELS), help="polarity model (default finbert, decision D19; cryptobert for the comparison)")
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--partitions", nargs="+")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="completeness only")
    a = ap.parse_args(argv)
    ml = a.max_len or PM.MODELS[a.model].default_max_len
    try:
        print(json.dumps(merge(a.platform, a.model, ml, a.partitions, a.force, a.check_only), indent=1, default=str))
    except CheckError as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


# ==================================================================================================
# Command line: one subcommand per stage (the table at the top of this file)
# ==================================================================================================

COMMANDS = {
    # subcommand: (function, receives the subcommand word too?, one-line help)
    "build-reference": (main_tokenize, True, "section 1: slow-tokenizer reference ids (transformers 4.x)"),
    "selfcheck": (main_tokenize, True, "section 1: fast tokenizer == stored reference ids"),
    "tokenize": (main_tokenize, True, "section 1: stage 1 of a platform"),
    "distinct": (main_tokenize, True, "section 1: distinct-text counts in a sample (optional)"),
    "keys": (main_tokenize, True, "section 1: text keys for --dedup (optional)"),
    "fetch-model": (main_stage2, True, "section 2: the pinned model files, sha256-checked"),
    "pack": (main_stage2, True, "section 2: model inputs from finished stage-1 shards"),
    "score": (main_stage2, True, "section 2: the model alone on a packed inputs file"),
    "collect": (main_stage2, True, "section 2: raw scores + map -> the scores table"),
    "merge": (main_merge, False, "section 3: one row per (row_id, version)"),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: python pipeline/07_polarity_finbert.py <subcommand> [options]   (<subcommand> -h: its options)\n")
        for name, (_, _, text) in COMMANDS.items():
            print(f"  {name:16s} {text}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    fn, whole, _ = COMMANDS[argv[0]]
    return fn(argv if whole else argv[1:])


if __name__ == "__main__":
    sys.exit(main())
