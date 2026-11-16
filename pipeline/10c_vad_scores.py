#!/usr/bin/env python3
"""
10c_vad_scores.py -- NRC-VAD lexicon scores per post and per day (test 2.9, construct validity).

Written 7 Oct 2026, after the first real model run of A00 (09:17 UTC). Test 2.9 of
planning/final_test_plan.md is pre-registered (CORE); this script only adds new files and changes
no earlier step or output.

Inputs (local, read only)
-------------------------
    data/raw/nrc_vad/v1/NRC-VAD-Lexicon.txt                    NRC-VAD v1 (Mohammad 2018), 0..1, primary
    data/raw/nrc_vad/v2.1/unigrams-NRC-VAD-Lexicon-v2.1.txt    NRC-VAD v2.1 (Mohammad 2025), -1..1, robustness
    own text of every post, read as EmTract and FinBERT read it (OWN_SQL of 07_polarity_finbert.py):
        reddit      data/pipeline/reddit/extract/reddit_<P>.parquet   own_text, NOT is_dead, key id
        stocktwits  data/pipeline/stocktwits/coarse/month=*/data_0.parquet   message_body, key message_id
        twitter     data/pipeline/twitter/units/units.parquet   text, key unit_id
    post set: posts_<platform> of the text-series scratch database (step 10, read only)
    data/pipeline/twitter/units/twitter_coverage_utc.parquet   zero-capture days (D14)

Outputs
-------
    data/pipeline/<platform>/vad/vad_posts.parquet   one row per scored text: row_id, n_tokens, matches, scores
    data/pipeline/panel/vad_daily.parquet            series x platform (incl. POOLED) x UTC day
    (each with a .stats.json next to it)

Method (standard lexicon scoring; every choice is a named constant below)
---------------------------------------------------------------------------
* Text: lowercase; URLs, @mentions and $cashtags removed; apostrophes dropped ("don't" -> "dont");
  tokens = runs of the letters a-z; exact lookup, no stemming, no negation handling, no stop-word list.
  v1 is matched on its unigram entries (its 132 multi-word entries cannot match single tokens).
* Post score per dimension = mean over the matched tokens (NaN = no match, read as NULL in the daily step); polar subset (README of the lexicon):
  only entries <= 0.3333 or >= 0.6667 on that dimension.
* Posts: the MAIN variant of step 10 -- posts after the bot pass (NOT bot_any), own text, a post counts
  for each series it names, and only posts with both an EmTract and a FinBERT own-text score, so the
  VAD means average over the same posts as the EmTract means; a VAD score needs >= 1 match.
* Daily: plain mean and sample SD per platform x series x UTC day (no Twitter hour post-stratification,
  i.e. comparable to text_daily variant a20_unstratified); Twitter zero-capture days are missing.
  POOLED = equal-weighted mean of the available platform means (D15); pooled SD = mean of platform SDs.

Usage (from code/):  python pipeline/10c_vad_scores.py [--skip-scoring]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, runner  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "10c_vad_scores"
CODE_SHA = runner.code_fingerprint(__file__)
SERIES = ["BTC", "ETH", "BNB", "XRP", "DOGE", "MARKET"]
PLATFORMS = ["reddit", "stocktwits", "twitter"]
LEX_V1 = config.RAW_DIR / "nrc_vad" / "v1" / "NRC-VAD-Lexicon.txt"
LEX_V2 = config.RAW_DIR / "nrc_vad" / "v2.1" / "unigrams-NRC-VAD-Lexicon-v2.1.txt"
POLAR_LO, POLAR_HI = 0.3333, 0.6667
CLEAN_RE = r"https?://\S+|www\.\S+|@\w+|\$[a-z0-9_]+"
APOS_RE = "['\u2019]"
SPLIT_RE = r"[^a-z]+"
BATCH = 500_000
SCRATCH_DB = config.SCRATCH_DIR / "text_series" / "text_series.duckdb"
SOURCES = {
    "reddit": ("id", "own_text", "NOT is_dead"),
    "stocktwits": ("message_id", "message_body", None),
    "twitter": ("unit_id", "text", None),
}
DIMS = ["v", "a", "d"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_lexicons():
    l1 = pd.read_csv(LEX_V1, sep="\t", header=None, names=["term", "v", "a", "d"], keep_default_na=False, quoting=3)
    l2 = pd.read_csv(LEX_V2, sep="\t", keep_default_na=False, quoting=3).rename(columns={"term": "term", "valence": "v", "arousal": "a", "dominance": "d"})
    check(len(l1) == 19971 and len(l2) == 44728, f"lexicon sizes v1 {len(l1)}, v2.1 {len(l2)}")
    check(l1.term.is_unique and l2.term.is_unique, "lexicon terms unique")
    check(l1[DIMS].min().min() >= 0 and l1[DIMS].max().max() <= 1, "v1 scores in [0, 1]")
    check(l2[DIMS].min().min() >= -1 and l2[DIMS].max().max() <= 1, "v2.1 scores in [-1, 1]")
    l1u = l1[~l1.term.str.contains(" ")]
    lex = {"v1": (pa.array(l1u.term.tolist()), l1u[DIMS].to_numpy(np.float64)),
           "v2": (pa.array(l2.term.tolist()), l2[DIMS].to_numpy(np.float64))}
    meta = {"v1": {"path": str(LEX_V1), "md5": md5(LEX_V1), "entries": len(l1), "unigram_entries": len(l1u)},
            "v2.1": {"path": str(LEX_V2), "md5": md5(LEX_V2), "entries": len(l2)}}
    return lex, meta


def score_batch(texts: pa.Array, lex, unmatched_counter=None):
    n = len(texts)
    low = pc.utf8_lower(pc.fill_null(texts, ""))
    low = pc.replace_substring_regex(low, pattern=CLEAN_RE, replacement=" ")
    low = pc.replace_substring_regex(low, pattern=APOS_RE, replacement="")
    toks = pc.split_pattern_regex(low, pattern=SPLIT_RE)
    flat = pc.list_flatten(toks)
    par = pc.list_parent_indices(toks).to_numpy()
    nonempty = pc.greater(pc.utf8_length(flat), 0).to_numpy(zero_copy_only=False)
    out = {"n_tokens": np.bincount(par[nonempty], minlength=n).astype(np.int32)}
    for ver, (terms, vals) in lex.items():
        idx_arr = pc.index_in(flat, value_set=terms)
        idx = idx_arr.fill_null(-1).to_numpy().astype(np.int64)
        m = idx >= 0
        if ver == "v1" and unmatched_counter is not None:
            um = pc.filter(flat, pa.array(~m & nonempty))
            vc = pc.value_counts(um)
            for s in vc:
                unmatched_counter[s["values"].as_py()] = unmatched_counter.get(s["values"].as_py(), 0) + s["counts"].as_py()
        pm, vv = par[m], vals[idx[m]]
        nm = np.bincount(pm, minlength=n)
        out[f"n_matched_{ver}"] = nm.astype(np.int32)
        for j, dim in enumerate(DIMS):
            s = np.bincount(pm, weights=vv[:, j], minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{dim}_{ver}"] = np.where(nm > 0, s / nm, np.nan).astype(np.float32)
            if ver == "v1":
                pol = (vv[:, j] <= POLAR_LO) | (vv[:, j] >= POLAR_HI)
                npol = np.bincount(pm[pol], minlength=n)
                sp = np.bincount(pm[pol], weights=vv[pol, j], minlength=n)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{dim}_v1polar"] = np.where(npol > 0, sp / npol, np.nan).astype(np.float32)
    return out


def score_platform(platform: str, lex, checks):
    key, col, filt = SOURCES[platform]
    if platform == "reddit":
        files = sorted(glob.glob(str(config.out_path("reddit", "extract", "reddit_*.parquet"))))
    elif platform == "stocktwits":
        files = sorted(glob.glob(str(config.out_path("stocktwits", "coarse", "month=*", "data_0.parquet"))))
    else:
        files = [str(config.out_path("twitter", "units", "units.parquet"))]
    outp = config.out_path(platform, "vad", "vad_posts.parquet")
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".parquet.tmp")
    writer, n_rows, n_tok, n_m1, n_m2, n_any1 = None, 0, 0, 0, 0, 0
    unmatched = {}
    t0 = time.time()
    for fi, f in enumerate(files):
        cols = [key, col] + (["is_dead"] if filt else [])
        pf = pq.ParquetFile(f)
        for bi, rb in enumerate(pf.iter_batches(batch_size=BATCH, columns=cols)):
            tb = pa.Table.from_batches([rb])
            if filt:
                tb = tb.filter(pc.invert(pc.fill_null(tb.column("is_dead"), False)))
            if tb.num_rows == 0:
                continue
            sc = score_batch(tb.column(col).combine_chunks(), lex, unmatched if (fi == 0 and bi < 2) else None)
            arrays = {"row_id": pc.cast(tb.column(key), pa.string())}
            arrays.update({k: pa.array(v) for k, v in sc.items()})
            out = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(tmp, out.schema, compression="zstd")
            writer.write_table(out)
            n_rows += out.num_rows
            n_tok += int(sc["n_tokens"].sum()); n_m1 += int(sc["n_matched_v1"].sum()); n_m2 += int(sc["n_matched_v2"].sum())
            n_any1 += int((sc["n_matched_v1"] > 0).sum())
        log(f"{platform}: {Path(f).parent.name}/{Path(f).name} done, {n_rows:,} rows, {time.time() - t0:.0f}s")
    writer.close()
    tmp.replace(outp)
    top_unmatched = sorted(unmatched.items(), key=lambda kv: -kv[1])[:40]
    rec = {"rows": n_rows, "tokens": n_tok, "token_match_share_v1": n_m1 / n_tok, "token_match_share_v2": n_m2 / n_tok,
           "posts_with_match_share_v1": n_any1 / n_rows, "seconds": round(time.time() - t0, 1),
           "top_unmatched_tokens_sample": top_unmatched}
    con = runner.connect(memory="4GB", threads=6)
    nr, nd = con.execute(f"SELECT count(*), count(DISTINCT row_id) FROM read_parquet('{outp}')").fetchone()
    con.close()
    rec["distinct_row_ids"] = int(nd)
    check(nr == n_rows, f"{platform}: rows written {nr:,}", checks)
    rec["duplicate_row_ids"] = int(nr - nd)
    runner.write_stats(outp, step=STEP + ":score", inputs=[{"path": f} for f in files], code_sha=CODE_SHA,
                       rules_sha="test-2.9", con=None, counts=rec, checks=[c for c in checks if c["check"].startswith(platform)])
    return rec


MEASURES = [f"{d}_{v}" for v in ("v1", "v1polar", "v2") for d in DIMS]


def daily(checks):
    con = runner.connect(memory="4GB", threads=6)
    con.execute(f"ATTACH '{SCRATCH_DB}' AS ts (READ_ONLY)")
    cov = config.out_path("twitter", "units", "twitter_coverage_utc.parquet")
    parts = []
    for p in PLATFORMS:
        vp = config.out_path(p, "vad", "vad_posts.parquet")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE m AS
            SELECT p.post_id, p.day, p.in_btc, p.in_eth, p.in_bnb, p.in_xrp, p.in_doge, p.in_market,
                   v.row_id IS NOT NULL AS has_vad, v.n_tokens, v.n_matched_v1, v.n_matched_v2,
                   {', '.join(f'CASE WHEN isnan(v.{c}) THEN NULL ELSE CAST(v.{c} AS DOUBLE) END AS {c}' for c in MEASURES)}
            FROM ts.posts_{p} p
            LEFT JOIN (SELECT * FROM read_parquet('{vp}') QUALIFY row_number() OVER (PARTITION BY row_id) = 1) v
              ON v.row_id = p.post_id
            WHERE NOT p.bot_any AND p.p_happy IS NOT NULL AND p.pol IS NOT NULL""")
        n, n_has = con.execute("SELECT count(*), count_if(has_vad) FROM m").fetchone()
        check(n_has == n, f"{p}: every MAIN post has a VAD row ({n_has:,} of {n:,})", checks)
        aggs = ["count(*) AS n_main", "count(v_v1) AS n_vad_v1", "count(v_v2) AS n_vad_v2",
                "sum(n_tokens) AS n_tokens", "sum(n_matched_v1) AS n_matched_v1"]
        for c in MEASURES:
            aggs += [f"avg({c}) AS mean_{c}", f"stddev_samp({c}) AS sd_{c}", f"count({c}) AS n_{c}"]
        for s in SERIES:
            parts.append(con.execute(f"SELECT '{s}' AS series, '{p}' AS platform, day, {', '.join(aggs)} "
                                     f"FROM m WHERE in_{s.lower()} GROUP BY day").df())
        log(f"daily: {p} done")
    d = pd.concat(parts, ignore_index=True)
    grid = pd.MultiIndex.from_product([SERIES, PLATFORMS, pd.date_range(config.WINDOW_START, config.WINDOW_END, freq="D").date],
                                      names=["series", "platform", "day"]).to_frame(index=False)
    d["day"] = pd.to_datetime(d["day"]).dt.date
    d = grid.merge(d, on=["series", "platform", "day"], how="left")
    absent = set(pd.to_datetime(pd.read_parquet(cov).query("minutes_utc == 0")["utc_day"]).dt.date)
    mcols = [c for c in d.columns if c.startswith(("mean_", "sd_"))]
    tw_abs = (d.platform == "twitter") & d.day.isin(absent)
    d.loc[tw_abs, mcols] = np.nan
    d["twitter_absent"] = d.day.isin(absent)
    pooled = (d[~tw_abs].groupby(["series", "day"])[[c for c in mcols]].mean().reset_index())
    npl = d[~tw_abs & d["mean_d_v1"].notna()].groupby(["series", "day"]).size().rename("n_platforms_v1").reset_index()
    pooled = pooled.merge(npl, on=["series", "day"], how="left")
    pooled["platform"] = "POOLED"
    sums = d.groupby(["series", "day"])[["n_main", "n_vad_v1", "n_vad_v2", "n_tokens", "n_matched_v1"]].sum(min_count=1).reset_index()
    pooled = pooled.merge(sums, on=["series", "day"], how="left")
    pooled["twitter_absent"] = pooled.day.isin(absent)
    out = pd.concat([d, pooled], ignore_index=True).sort_values(["series", "platform", "day"]).reset_index(drop=True)
    check(len(out) == 6 * 4 * 1280, f"vad_daily: {len(out):,} rows = 6 series x 4 platforms x 1280 days", checks)
    check(not out.duplicated(["series", "platform", "day"]).any(), "vad_daily: one row per series, platform, day", checks)
    v = out["mean_d_v1"].dropna()
    check(v.between(0, 1).all(), "vad_daily: v1 dominance means within [0, 1]", checks)
    dest = config.OUT_DIR / "panel" / "vad_daily.parquet"
    tmp = dest.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(dest)
    runner.write_stats(dest, step=STEP + ":daily", inputs=[{"path": str(SCRATCH_DB)}, {"path": str(cov)}],
                       code_sha=CODE_SHA, rules_sha="test-2.9", con=None,
                       counts={"rows": len(out), "twitter_absent_days": len(absent)}, checks=checks)
    con.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scoring", action="store_true")
    a = ap.parse_args(argv)
    checks = []
    lex, meta = load_lexicons()
    recs = {}
    if not a.skip_scoring:
        for p in PLATFORMS:
            recs[p] = score_platform(p, lex, checks)
            log(f"{p}: {json.dumps({k: v for k, v in recs[p].items() if k != 'top_unmatched_tokens_sample'})}")
    daily(checks)
    summary = {"lexicons": meta, "scoring": recs, "checks_failed": [c for c in checks if not c["passed"]]}
    (config.OUT_DIR / "panel" / "reports").mkdir(parents=True, exist_ok=True)
    with open(config.OUT_DIR / "panel" / "reports" / "vad_run_summary_20261007.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    log("done")
    return 0 if not summary["checks_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
