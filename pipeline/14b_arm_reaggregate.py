#!/usr/bin/env python3
"""
14b_arm_reaggregate.py -- step 14b: daily platform rows of the register arms whose series are not in the stored
daily tables and need post-level re-aggregation: A40 (coin words inside links), A42 (StockTwits coin-less
replies inherit the parent's coin) and A47 (market measure plus the several-ambiguous-words flag).

Written 28 Oct 2026 (robustness tooling; no frozen rule changed; no earlier step or output changed).

What it does, per platform (StockTwits, Twitter), in one DuckDB session (TimeZone UTC asserted, 4 GB, 3 threads,
spill directory set):
  1. posts = the step-04 assignment (series flags, bot_any, the arm flags link_<coin>, reply_parent_<coin>,
     ambiguous_extra) left-joined on row_id = post_id to the own-text EmTract scores (p_*) and the own-text FinBERT
     polarity, exactly as step 10 (10_text_series.py build_posts); day = CAST(ts AS DATE) in UTC, hour = hour(ts);
     posts flagged by the bot pass are left out (NOT bot_any).
  2. hourly sums per (membership set, series, day, hour) with step 10's score expressions: a post is in the means
     only if both EmTract and FinBERT scored it; empty-after-clean and missing texts count as posts (3 Oct (2)).
  3. daily rows with step 10's daily SQL (raw mean, within-day sample SD for n >= 2, hour post-stratified mean with
     the fixed pre-forecast corpus profile of text_calibration_preforecast.json); Twitter: stratified mean on the
     partial days (0 < coverage < 95%), NaN on the 76 zero-capture days; R(n) = rel_<v> from the fixed sigma2.
Membership sets: 'main' (in_<series>) = the switch off; A40: coin c = in_c OR link_c (StockTwits, Twitter);
A42: coin c = in_c OR reply_parent_c (StockTwits); A47: MARKET = in_market OR ambiguous_extra (StockTwits,
Twitter). Reddit is not touched (no per-coin link flag; A42 and A47 are StockTwits/Twitter rules).

Check (asserted): with the switch off, the recomputed rows reproduce text_daily.parquet (variant 'main') on every
platform x series x day the arms touch: counts exactly, means and SDs within 1e-12 (DuckDB's parallel sums may
differ from the stored run in the last bits).

Output: data/pipeline/panel/arms/reagg_20261028/<ARM>_platform_daily.parquet (+ .stats.json): the arm's platform
rows in text_daily's format, read by 14_arm_panels.py. Never overwrites an existing file.

    python pipeline/14b_arm_reaggregate.py [--out-dir ...]     (from code/)
"""
from __future__ import annotations

import argparse
import re
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, runner  # noqa: E402

STEP = "14b_arm_reaggregate"
COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]
SERIES = COINS + ["MARKET"]
EMO = ["neutral", "happy", "sad", "anger", "disgust", "surprise", "fear"]
VARS = EMO + ["valence", "polarity"]
FULL_DAY_PCT = 95.0                                   # as step 10 ('fully observed' Twitter day)
PANEL = config.OUT_DIR / "panel"
CAL = PANEL / "text_calibration_preforecast.json"
TEXT = PANEL / "text_daily.parquet"
COV = config.out_path("twitter", "units", "twitter_coverage_utc.parquet")
ARM_SETS = {   # arm: {platform: {series: membership SQL}}
    "A40": {p: {c: f"(in_{c.lower()} OR link_{c.lower()})" for c in COINS} for p in ("stocktwits", "twitter")},
    "A42": {"stocktwits": {c: f"(in_{c.lower()} OR reply_parent_{c.lower()})" for c in COINS}},
    "A47": {p: {"MARKET": "(in_market OR ambiguous_extra)"} for p in ("stocktwits", "twitter")},
}
EXTRA = {"stocktwits": [f"link_{c.lower()}" for c in COINS] + [f"reply_parent_{c.lower()}" for c in COINS] + ["ambiguous_extra"],
         "twitter": [f"link_{c.lower()}" for c in COINS] + ["ambiguous_extra"]}


def files(platform, kind):
    if kind == "assign":
        return [str(config.out_path(platform, "panel", "assignment.parquet"))]
    if kind == "emo":
        return sorted(glob.glob(str(config.out_path(platform, "emtract", "emotion", "*.parquet"))))
    return sorted(glob.glob(str(config.out_path(platform, "polarity", "finbert-L64", "polarity", "*.parquet"))))


def rp(fs):
    return "read_parquet([" + ", ".join(f"'{f}'" for f in fs) + "])"


MAIN_OK = "(p_happy IS NOT NULL AND pol IS NOT NULL)"


def scores():
    """Step 10's score expressions for variant 'main' (10_text_series.scores, else-branch)."""
    out = {x: f"CASE WHEN {MAIN_OK} THEN p_{x} END" for x in EMO}
    out["polarity"] = f"CASE WHEN {MAIN_OK} THEN pol END"
    out["valence"] = f"({out['happy']}) - ({out['sad']}) - ({out['anger']}) - ({out['disgust']}) - ({out['fear']})"
    return out


def posts_sql(platform):
    a, e, f = rp(files(platform, "assign")), rp(files(platform, "emo")), rp(files(platform, "pol"))
    sc = ", ".join(f"a.in_{s.lower()}" for s in SERIES)
    ex = ", ".join(f"a.{c}" for c in EXTRA[platform])
    pc = ", ".join(f"e.p_{x}" for x in EMO)
    return f"""SELECT a.post_id, CAST(a.ts AS DATE) AS day, CAST(hour(a.ts) AS TINYINT) AS hr, {sc}, a.bot_any, {ex},
               coalesce(e.is_empty_after_clean, FALSE) AS is_empty, coalesce(e.is_missing_text, FALSE) AS is_missing,
               {pc}, f.polarity AS pol
        FROM {a} a
        LEFT JOIN (SELECT * FROM {e} WHERE version = 'own') e ON e.row_id = a.post_id
        LEFT JOIN (SELECT row_id, polarity FROM {f} WHERE version = 'own') f ON f.row_id = a.post_id
        WHERE CAST(a.ts AS DATE) BETWEEN DATE '{config.WINDOW_START}' AND DATE '{config.WINDOW_END}'"""


def hourly_sql(platform, sets):
    """sets: list of (set_name, series, membership SQL)."""
    sc = scores()
    aggs = ["CAST(count(*) AS BIGINT) AS n_posts", "CAST(count_if(is_empty) AS BIGINT) AS n_empty",
            "CAST(count_if(is_missing) AS BIGINT) AS n_missing", f"CAST(count({sc['happy']}) AS BIGINT) AS n_emo",
            f"CAST(count({sc['polarity']}) AS BIGINT) AS n_pol"]
    for x in VARS:
        aggs.append(f"sum(CAST({sc[x]} AS DOUBLE)) AS s_{x}")
        aggs.append(f"sum(CAST({sc[x]} AS DOUBLE) * CAST({sc[x]} AS DOUBLE)) AS q_{x}")
    parts = [f"SELECT '{nm}' AS set_name, '{s}' AS series, * FROM posts_{platform} WHERE {mem} AND NOT bot_any"
             for nm, s, mem in sets]
    return (f"SELECT set_name, '{platform}' AS platform, series, day, hr, {', '.join(aggs)} "
            f"FROM ({' UNION ALL '.join(parts)}) GROUP BY set_name, series, day, hr")


def daily_sql(profile):
    """Step 10's daily SQL (10_text_series.daily_sql), grouped by set_name instead of variant."""
    prof = "(VALUES " + ", ".join(f"({h}, {w!r})" for h, w in sorted(profile.items())) + ") pr(hr, pi)"
    cols = ["sum(n_posts) AS n_posts", "sum(n_empty) AS n_empty", "sum(n_missing) AS n_missing",
            "sum(n_emo) AS n_emo", "sum(n_pol) AS n_pol", "CAST(count_if(n_emo > 0) AS INTEGER) AS n_hours_emo"]
    for x in VARS:
        n = "n_pol" if x == "polarity" else "n_emo"
        cols.append(f"sum(s_{x}) / nullif(sum({n}), 0) AS raw_mean_{x}")
        cols.append(f"CASE WHEN sum({n}) >= 2 THEN sqrt(greatest((sum(q_{x}) - sum(s_{x}) * sum(s_{x}) / sum({n}))"
                    f" / (sum({n}) - 1), 0)) END AS sd_{x}")
        cols.append(f"sum(CASE WHEN {n} > 0 THEN pi * s_{x} / {n} END) / nullif(sum(CASE WHEN {n} > 0 THEN pi END), 0)"
                    f" AS strat_mean_{x}")
    return (f"SELECT set_name, platform, series, day, {', '.join(cols)} FROM hourly h JOIN {prof} USING (hr) "
            f"GROUP BY set_name, platform, series, day")


def finish(daily, cov, cal):
    """Complete calendar, Twitter stratification and zero-capture days, R(n) from the fixed sigma2 (as step 10)."""
    days = pd.date_range(config.WINDOW_START, config.WINDOW_END, freq="D")
    keys = daily[["set_name", "platform", "series"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"day": days}), how="cross")
    full = grid.merge(daily, on=["set_name", "platform", "series", "day"], how="left")
    for c in ["n_posts", "n_empty", "n_missing", "n_emo", "n_pol", "n_hours_emo"]:
        full[c] = full[c].fillna(0).astype("int64")
    full = full.merge(cov, on="day", how="left")
    tw = full.platform == "twitter"
    full["twitter_absent"] = full.minutes_utc == 0
    full["twitter_stratified"] = tw & (full.minutes_utc > 0) & (full.coverage_pct_utc < FULL_DAY_PCT)
    for x in VARS:
        full[f"mean_{x}"] = np.where(full.twitter_stratified, full[f"strat_mean_{x}"], full[f"raw_mean_{x}"])
        full.loc[tw & full.twitter_absent, [f"mean_{x}", f"sd_{x}"]] = np.nan
    d16 = cal["d16_reliability"]
    key = full.platform + "|" + full.series
    for x in VARS:
        ncol = "n_pol" if x == "polarity" else "n_emo"
        sw = key.map({f"{p}|{s}": d16[p][s][x]["sigma2_w"] for p in d16 for s in d16[p]}).astype(float)
        sb = key.map({f"{p}|{s}": d16[p][s][x]["sigma2_b"] for p in d16 for s in d16[p]}).astype(float)
        n = full[ncol].astype(float).where(full[ncol] > 0)
        n = n.where(~(tw & full.twitter_absent))
        full[f"rel_{x}"] = sb / (sb + sw / n)
    return full


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()



def _setting_bytes(v: str) -> float:
    """DuckDB size setting ('3.7 GiB', '4.0 GB', ...) in bytes."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]i?B)\s*", str(v))
    assert m, f"unreadable DuckDB size setting {v!r}"
    unit = {"KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "KiB": 2 ** 10, "MiB": 2 ** 20, "GiB": 2 ** 30, "TiB": 2 ** 40}
    return float(m.group(1)) * unit[m.group(2)]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(PANEL / "arms" / "reagg_20261028"))
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    t0 = time.time()
    checks = []
    cal = json.loads(CAL.read_text())
    profile = {int(h): float(w) for h, w in cal["twitter_hour_profile"]["preforecast"]["corpus"].items()}
    runner.check(cal["twitter_hour_profile"]["applied"] == "corpus" and len(profile) == 24, "fixed corpus hour profile, 24 hours", checks)
    scratch = config.SCRATCH_DIR / "arm_reagg_14b"
    scratch.mkdir(parents=True, exist_ok=True)
    db = scratch / "arm_reagg.duckdb"
    if db.exists():
        db.unlink()                      # own scratch file of this step, outside the repository
    con = runner.connect(spill=scratch / "duckdb_spill", database=db)
    runner.check(con.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC", "session TimeZone is UTC", checks)
    mem = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    runner.check(0 < _setting_bytes(mem) <= 4.0e9, f"DuckDB memory limit {mem} (at most 4 GB)", checks)
    thr = int(con.execute("SELECT current_setting('threads')").fetchone()[0])
    runner.check(1 <= thr <= 3, f"DuckDB threads {thr} (at most 3)", checks)
    settings = runner.settings(con)
    plats = sorted({p for s in ARM_SETS.values() for p in s})
    inputs = [CAL, TEXT, COV] + [Path(f) for p in plats for k in ("assign", "emo", "pol") for f in files(p, k)]
    runner.log("md5 of the inputs ...")
    inp = runner.describe_inputs(inputs)
    con.execute("DROP TABLE IF EXISTS hourly")
    first, timing = True, {}
    for p in plats:
        t1 = time.time()
        con.execute(f"DROP TABLE IF EXISTS posts_{p}")
        con.execute(f"CREATE TABLE posts_{p} AS {posts_sql(p)}")
        n_assign = con.execute(f"SELECT count(*) FROM {rp(files(p, 'assign'))}").fetchone()[0]
        n, n_ids = con.execute(f"SELECT count(*), count(DISTINCT post_id) FROM posts_{p}").fetchone()
        runner.check(n == n_assign and n_ids == n, f"{p}: every assignment row joined once ({n:,})", checks)
        sets = []
        for arm, d in ARM_SETS.items():
            for s, mem in d.get(p, {}).items():
                sets.append((arm, s, mem))
                if ("main", s) not in [(x[0], x[1]) for x in sets]:
                    sets.append(("main", s, f"in_{s.lower()}"))
        con.execute(("CREATE TABLE hourly AS " if first else "INSERT INTO hourly ") + hourly_sql(p, sets))
        first = False
        timing[p] = round(time.time() - t1, 1)
        runner.log(f"{p}: hourly sums for {len(sets)} (set, series) pairs in {timing[p]} s")
    daily = con.execute(daily_sql(profile)).df()
    daily["day"] = pd.to_datetime(daily["day"])
    cov = con.execute(f"SELECT utc_day AS day, coverage_pct_utc, minutes_utc FROM read_parquet('{COV}')").df()
    cov["day"] = pd.to_datetime(cov["day"])
    full = finish(daily, cov, cal)
    # switch off: 'main' reproduces the stored text_daily main rows
    td = pd.read_parquet(TEXT)
    td = td[td.variant == "main"].copy()
    td["day"] = pd.to_datetime(td["day"].astype(str))
    m = full[full.set_name == "main"].merge(td, on=["platform", "series", "day"], how="left", suffixes=("", "_st"))
    runner.check(len(m) == int((full.set_name == "main").sum()) and m["n_posts_st"].notna().all(), "switch off: every main row has a stored row", checks)
    for c in ["n_posts", "n_empty", "n_missing", "n_emo", "n_pol"]:
        runner.check(bool((m[c] == m[c + "_st"]).all()), f"switch off: {c} equals text_daily main on all {len(m):,} rows", checks)
    devs = {}
    for c in [f"mean_{x}" for x in VARS] + [f"sd_{x}" for x in VARS] + [f"rel_{x}" for x in VARS]:
        a_, b_ = m[c].astype(float), m[c + "_st"].astype(float)
        same_na = bool((a_.isna() == b_.isna()).all())
        dev = float(np.nanmax(np.abs(a_ - b_))) if a_.notna().any() else 0.0
        devs[c] = dev
        runner.check(same_na and dev <= 1e-12, f"switch off: {c} reproduces text_daily main (same missing days, max |diff| {dev:.1e})", checks)
    # outputs: the arm rows for the (platform, series) each arm changes
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = (["series", "platform", "day", "n_posts", "n_empty", "n_missing", "n_emo", "n_pol"] + [f"mean_{x}" for x in VARS]
            + [f"sd_{x}" for x in VARS] + [f"rel_{x}" for x in VARS] + ["twitter_absent", "twitter_stratified", "n_hours_emo"])
    written = {}
    for arm in ARM_SETS:
        f = full[full.set_name == arm][keep].copy()
        f.insert(0, "variant", arm.lower())
        f["variant_applies"] = True
        f = f.sort_values(["platform", "series", "day"]).reset_index(drop=True)
        mm = f.merge(td[["platform", "series", "day", "n_posts"]], on=["platform", "series", "day"], suffixes=("", "_main"))
        added = mm.groupby(["platform", "series"]).apply(lambda g: int((g.n_posts - g.n_posts_main).sum()), include_groups=False)
        runner.check(bool((mm.n_posts >= mm.n_posts_main).all()), f"{arm}: the arm only adds posts to the series it changes", checks)
        final = out_dir / f"{arm}_platform_daily.parquet"
        tmp = final.with_name(final.name + ".tmp")
        f.to_parquet(tmp, index=False)
        back = pd.read_parquet(tmp)
        runner.check(len(back) == len(f), f"{arm}: written table reads back ({len(f):,} rows)", checks)
        if final.exists():
            if md5(final) != md5(tmp):
                os.remove(tmp)
                raise FileExistsError(f"{final} exists with different content; not overwritten")
            os.remove(tmp)
        else:
            os.replace(tmp, final)
        written[arm] = {"path": str(final), "md5": md5(final), "rows": int(len(f)),
                        "posts_added_after_bot_pass": {f"{k[0]}|{k[1]}": int(v) for k, v in added.items()}}
    stats = {"step": STEP, "written_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
             "code_md5": md5(Path(__file__).resolve()), "settings": settings, "inputs": inp, "outputs": written,
             "membership_sets": ARM_SETS, "seconds": round(time.time() - t0, 1), "timing_s": timing,
             "switch_off_max_abs_diff": devs, "checks": checks}
    for arm, w in written.items():
        st = dict(stats, outputs={arm: w}, checks=checks)
        sp = Path(w["path"] + ".stats.json")
        if not sp.exists():
            runner.write_json_atomic(sp, st)
    runner.log(f"done in {time.time() - t0:.0f} s; {json.dumps({k: v['posts_added_after_bot_pass'] for k, v in written.items()})}")
    con.close()
    db.unlink()                          # scratch database of this run only
    return 0


if __name__ == "__main__":
    sys.exit(main())
