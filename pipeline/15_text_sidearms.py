#!/usr/bin/env python3
"""
15_text_sidearms.py -- step 15: daily text measures for the side arms raised on 12 Nov 2026
(aggregate emotion dispersion; valenced emotions). It changes no earlier step and no earlier output.

Inputs (all local, read only)
-----------------------------
    data/pipeline/<platform>/panel/assignment.parquet                   step 04: series flags, bot flag
    data/pipeline/<platform>/emtract/emotion/*.parquet                  step 06: p_* (own text)
    data/pipeline/<platform>/polarity/finbert-L64/polarity/*.parquet    step 07: p_positive, p_negative, polarity
    data/pipeline/twitter/units/twitter_coverage_utc.parquet            captured minutes per UTC day
    data/pipeline/panel/text_daily.parquet                              step 10, for the reproduction checks
    data/pipeline/panel/text_precision_daily.parquet                    step 10b, for the reproduction checks
    data/pipeline/panel/text_calibration_preforecast.json               step 10: Twitter hour profile (fixed)

Output
------
    data/pipeline/panel/text_sidearms_daily.parquet (+ .stats.json): series x platform (three platforms and
    POOLED) x UTC day.

Sample: exactly the main sample of step 10 -- posts after the bot pass whose own text has both an EmTract and a
FinBERT score; a post counts for each series it names; day = UTC day of `ts` (session TimeZone UTC, asserted).

Measures
--------
A. Emotion dispersion, per platform, series and day with at least two posts (as the within-day SDs of step 10:
   unstratified, not precision-weighted), from the seven EmTract probabilities p_jk of post j:

       gini_total   = 1 - sum_k m_k^2                 m_k = the day's mean of p_jk
       gini_within  = mean_j (1 - sum_k p_jk^2)
       gini_between = sqrt(sum_k sd_k^2)              sd_k = within-day sample SD of p_jk  (side-arm regressor)
       gini_between_pop = gini_total - gini_within = ((n - 1) / n) * sum_k sd_k^2   (identity, asserted)
       ent_total    = - sum_k m_k ln m_k
       ent_within   = mean_j ( - sum_k p_jk ln p_jk )
       ent_between  = ent_total - ent_within          generalised Jensen-Shannon divergence of the day's posts

   gini_between is the standard deviation of the message signal across messages -- the disagreement measure of
   Antweiler and Frank (2004, JF 59(3), eq. 7-8) and Cookson and Niessner (2020, JF 75(1), eq. 2) -- taken over the
   seven-class probability vector instead of a binary signal: the root of the trace of the within-day covariance
   matrix. Total = within + between is the analysis of variance for categorical data (Gini's measure; Light and
   Margolin 1971, JASA 66) and, for the entropy, the Jensen-Shannon decomposition (Lin 1991, IEEE Trans. Inf.
   Theory 37). DEPARTURE FROM THE SOURCES, flagged for Gerome: the finance sources use binary hard labels; the
   step to seven classes and to probabilities is not taken from a finance paper (task report, open point 1).

B. Valenced emotions, formed per post and then averaged: for each of the six emotions k,

       <k>_pos = mean_j p_jk * P_j(positive)        <k>_neg = mean_j p_jk * P_j(negative)

   with FinBERT's class probabilities of the same post. They pass through the aggregation of the emotion means:
   hour post-stratified mean on partial Twitter days (step 10), D16 precision weighting with the reliability
   R(n) estimated on the pre-forecast year (steps 10 and 10b), pooled as the equal-weighted mean of the
   available platforms (D15). Source for the sign split of surprise: Liu et al. (2024, p. 16311). The split by
   class probabilities rather than by the sign of the score, and the extension to the other five emotions, are
   not taken from a source (task report, open points 2 and 3).

Checks (asserted before the output is moved into place): the code path reproduces step 10's daily means, SDs,
post counts and reliabilities and step 10b's precision-weighted pooled means of the six emotions and polarity;
the Gini identity; ent_between >= 0; <k>_pos + <k>_neg <= mean of p_k.

Usage (from code/):  python pipeline/15_text_sidearms.py
"""
from __future__ import annotations

import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, runner  # noqa: E402
from pipeline.runner import check, log  # noqa: E402

STEP = "15_text_sidearms"
CODE_SHA = runner.code_fingerprint(__file__)
RULES_SHA = "side arms of 12 Nov 2026 (MASTERPLAN section 7, decisions of 12 Nov 2026)"

SERIES = ["BTC", "ETH", "BNB", "XRP", "DOGE", "MARKET"]
PLATFORMS = ["reddit", "stocktwits", "twitter"]
EMO = ["neutral", "happy", "sad", "anger", "disgust", "surprise", "fear"]
EMO6 = EMO[1:]
VAL = [f"{k}_{s}" for k in EMO6 for s in ("pos", "neg")]      # valenced emotions (post-level products)
POST = ["gw", "hw"]                                            # within-post Gini and entropy
VARS = EMO + ["polarity"] + POST + VAL                         # every post-level variable that is summed
PREFORECAST = ("2019-08-30", "2020-08-29")
RELIABILITY_TARGET = 0.7
FULL_DAY_PCT = 95.0
LEVEL_DAYS = 30
TOL = 1e-9

PANEL = config.OUT_DIR / "panel"
OUT = PANEL / "text_sidearms_daily.parquet"
SCRATCH = config.SCRATCH_DIR / "text_sidearms"


def files(platform: str, kind: str) -> List[str]:
    if kind == "emo":
        return sorted(glob.glob(str(config.out_path(platform, "emtract", "emotion", "*.parquet"))))
    return sorted(glob.glob(str(config.out_path(platform, "polarity", "finbert-L64", "polarity", "*.parquet"))))


def rp(fs: List[str]) -> str:
    return "read_parquet([" + ", ".join(f"'{f}'" for f in fs) + "])"


def post_exprs() -> Dict[str, str]:
    ex = {x: f"p_{x}" for x in EMO}
    ex["polarity"] = "pol"
    ex["gw"] = "1 - (" + " + ".join(f"p_{x} * p_{x}" for x in EMO) + ")"
    ex["hw"] = "-(" + " + ".join(f"CASE WHEN p_{x} > 0 THEN p_{x} * ln(p_{x}) ELSE 0 END" for x in EMO) + ")"
    for k in EMO6:
        ex[f"{k}_pos"] = f"p_{k} * fpos"
        ex[f"{k}_neg"] = f"p_{k} * fneg"
    return ex


def hourly_sql(assign: str, emo: str, pol: str, cond: str) -> str:
    """Sums and sums of squares per series, UTC day and hour over the main-sample posts of one platform part."""
    flags = ", ".join(f"a.in_{s.lower()}" for s in SERIES)
    ps = ", ".join(f"CAST(e.p_{x} AS DOUBLE) AS p_{x}" for x in EMO)
    ex = post_exprs()
    xs = ", ".join(f"{ex[v]} AS x_{v}" for v in VARS)
    sums = ", ".join(f"sum(x_{v}) AS s_{v}, sum(x_{v} * x_{v}) AS q_{v}" for v in VARS)
    member = "CASE s.series " + " ".join(f"WHEN '{s}' THEN in_{s.lower()}" for s in SERIES) + " END"
    return f"""
    WITH j AS (
        SELECT CAST(a.ts AS DATE) AS day, CAST(hour(a.ts) AS TINYINT) AS hr, {flags}, {ps},
               CAST(f.p_positive AS DOUBLE) AS fpos, CAST(f.p_negative AS DOUBLE) AS fneg,
               CAST(f.polarity AS DOUBLE) AS pol
        FROM {assign} a
        JOIN (SELECT * FROM {emo} WHERE version = 'own' AND p_happy IS NOT NULL) e ON e.row_id = a.post_id
        JOIN (SELECT row_id, p_positive, p_negative, polarity FROM {pol}
              WHERE version = 'own' AND polarity IS NOT NULL) f ON f.row_id = a.post_id
        WHERE NOT a.bot_any AND {cond}
          AND CAST(a.ts AS DATE) BETWEEN DATE '{config.WINDOW_START}' AND DATE '{config.WINDOW_END}'),
    x AS (SELECT day, hr, {', '.join(f'in_{s.lower()}' for s in SERIES)}, {xs} FROM j)
    SELECT s.series, x.day, x.hr, CAST(count(*) AS BIGINT) AS n, {sums}
    FROM x, (VALUES {', '.join(f"('{s}')" for s in SERIES)}) s(series)
    WHERE {member}
    GROUP BY s.series, x.day, x.hr"""


def build_hourly(con) -> pd.DataFrame:
    parts = []
    for p in PLATFORMS:
        a = rp([str(config.out_path(p, "panel", "assignment.parquet"))])
        if p == "reddit":      # score tables partitioned by the assignment's coin_label (as step 10)
            jobs = [(Path(e).stem, f"a.coin_label = '{Path(e).stem}'", rp([e]),
                     rp([str(config.out_path(p, "polarity", "finbert-L64", "polarity", Path(e).name))]))
                    for e in files(p, "emo")]
        else:
            jobs = [("all", "TRUE", rp(files(p, "emo")), rp(files(p, "pol")))]
        for name, cond, e, f in jobs:
            t = time.time()
            d = con.execute(hourly_sql(a, e, f, cond)).df()
            d.insert(0, "platform", p)
            parts.append(d)
            log(f"{p}: hourly sums, part {name}: {len(d):,} rows, {int(d.n.sum()):,} post-series, {time.time() - t:.0f} s")
    h = pd.concat(parts, ignore_index=True)
    h["day"] = pd.to_datetime(h["day"])
    h["hr"] = h["hr"].astype(int)
    num = [c for c in h.columns if c == "n" or c.startswith(("s_", "q_"))]
    return h.groupby(["platform", "series", "day", "hr"], as_index=False)[num].sum()


def daily_from_hourly(h: pd.DataFrame, profile: Dict[int, float]) -> pd.DataFrame:
    num = [c for c in h.columns if c == "n" or c.startswith(("s_", "q_"))]
    key = ["platform", "series", "day"]
    d = h.groupby(key, as_index=False)[num].sum()
    n = d["n"].astype(float)
    for v in VARS:
        d[f"raw_mean_{v}"] = d[f"s_{v}"] / n
        var = (d[f"q_{v}"] - d[f"s_{v}"] ** 2 / n) / (n - 1).where(n >= 2)
        d[f"sd_{v}"] = np.sqrt(var.clip(lower=0))
    # hour post-stratified mean (step 10): sum_h pi_h * mean_h / sum_h pi_h over the hours with posts
    hh = h[key + ["hr", "n"] + [f"s_{v}" for v in VARS]].copy()
    hh["pi"] = hh["hr"].map(profile).astype(float)
    cols = {}
    for v in VARS:
        cols[f"w_{v}"] = hh["pi"] * hh[f"s_{v}"] / hh["n"]
    hh = pd.concat([hh[key + ["pi"]], pd.DataFrame(cols)], axis=1)
    st = hh.groupby(key, as_index=False).sum()
    for v in VARS:
        st[f"strat_mean_{v}"] = st[f"w_{v}"] / st["pi"]
    d = d.merge(st[key + [f"strat_mean_{v}" for v in VARS]], on=key, how="left")
    return d.drop(columns=[c for c in d.columns if c.startswith(("s_", "q_"))])


def decompose(g: pd.DataFrame, v: str) -> Dict[str, float]:
    """One-way random-effects decomposition of step 10 (ANOVA estimator)."""
    d = g[g["n"] > 0]
    n = d["n"].astype(float).values
    m = d[f"raw_mean_{v}"].values
    sd = d[f"sd_{v}"].fillna(0.0).values
    N, D = n.sum(), len(n)
    if D < 2 or N - D <= 0:
        return {"sigma2_w": np.nan, "sigma2_b": np.nan, "n_star": np.nan}
    grand = (n * m).sum() / N
    msw = ((n - 1) * sd ** 2).sum() / (N - D)
    msb = (n * (m - grand) ** 2).sum() / (D - 1)
    n0 = (N - (n ** 2).sum() / N) / (D - 1)
    s2b = max((msb - msw) / n0, 0.0)
    R = RELIABILITY_TARGET
    return {"sigma2_w": float(msw), "sigma2_b": float(s2b),
            "n_star": float((R / (1 - R)) * msw / s2b) if s2b > 0 else math.inf}


def run() -> int:
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    con = runner.connect(database=SCRATCH / "text_sidearms.duckdb")     # 4 GB, 3 threads, spill directory, UTC
    check(con.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC", "session TimeZone is UTC")
    checks: List[Dict] = []
    counts: Dict[str, object] = {}

    cov_file = config.out_path("twitter", "units", "twitter_coverage_utc.parquet")
    in_files = [cov_file, PANEL / "text_daily.parquet", PANEL / "text_precision_daily.parquet",
                PANEL / "text_calibration_preforecast.json"]
    for p in PLATFORMS:
        in_files += [config.out_path(p, "panel", "assignment.parquet")] + [Path(x) for x in files(p, "emo") + files(p, "pol")]
    log("md5 of the inputs ...")
    inputs = runner.describe_inputs(in_files)

    cal = json.loads((PANEL / "text_calibration_preforecast.json").read_text())
    profile = {int(k): float(v) for k, v in cal["twitter_hour_profile"]["preforecast"]["corpus"].items()}
    check(len(profile) == 24 and abs(sum(profile.values()) - 1) < 1e-12, "stored Twitter hour profile: 24 hours, sums to 1", checks)

    # 1-2. hourly sums, daily means and SDs
    h = build_hourly(con)
    d = daily_from_hourly(h, profile)
    cov = con.execute(f"SELECT utc_day AS day, coverage_pct_utc, minutes_utc FROM read_parquet('{cov_file}')").df()
    cov["day"] = pd.to_datetime(cov["day"])
    days = pd.date_range(config.WINDOW_START, config.WINDOW_END, freq="D")
    check(len(days) == config.N_DAYS and len(cov) == config.N_DAYS, "1,280 UTC days in the calendar and the coverage table", checks)
    grid = pd.MultiIndex.from_product([PLATFORMS, SERIES, days], names=["platform", "series", "day"]).to_frame(index=False)
    full = grid.merge(d, on=["platform", "series", "day"], how="left").merge(cov, on="day", how="left")
    full["n"] = full["n"].fillna(0).astype("int64")
    tw = full.platform == "twitter"
    full["twitter_absent"] = full.minutes_utc == 0
    full["twitter_stratified"] = tw & (full.minutes_utc > 0) & (full.coverage_pct_utc < FULL_DAY_PCT)
    absent = tw & full.twitter_absent
    check(int(full.loc[absent, "n"].sum()) == 0, "no Twitter post on a zero-capture day", checks)
    new = {}
    for v in VARS:
        m = pd.Series(np.where(full.twitter_stratified, full[f"strat_mean_{v}"], full[f"raw_mean_{v}"]), index=full.index)
        new[f"mean_{v}"] = m.where(~absent)
    full = pd.concat([full, pd.DataFrame(new)], axis=1)
    full.loc[absent, [f"sd_{v}" for v in VARS] + [f"raw_mean_{v}" for v in VARS]] = np.nan

    # 3. reproduction of step 10 (means, SDs, post counts) on the six emotions, neutral and polarity
    td = pd.read_parquet(PANEL / "text_daily.parquet")
    td = td[td.variant == "main"].copy()
    td["day"] = pd.to_datetime(td["day"].astype(str))
    tdp = td[td.platform != "POOLED"]
    cmp = full.merge(tdp, on=["platform", "series", "day"], how="outer", suffixes=("", "_td"), indicator=True)
    check(len(cmp) == len(full) and bool((cmp._merge == "both").all()), "same platform x series x day grid as step 10", checks)
    check(bool((cmp["n"] == cmp["n_emo"]).all() and (cmp["n"] == cmp["n_pol"]).all()),
          f"post counts equal step 10's n_emo and n_pol on all {len(cmp):,} cells", checks)
    worst = {}
    for v in EMO + ["polarity"]:
        for a in ("mean", "sd"):
            mine, ref = cmp[f"{a}_{v}"], cmp[f"{a}_{v}_td"]
            check(bool((mine.isna() == ref.isna()).all()), f"{a}_{v}: missing on the same cells as step 10", checks)
            worst[f"{a}_{v}"] = float((mine - ref).abs().max())
    check(max(worst.values()) < TOL, f"daily means and SDs of the emotions and polarity equal step 10 (max abs diff {max(worst.values()):.2e})", checks)
    counts["max_abs_diff_vs_step10"] = worst

    # 4. D16 reliability (pre-forecast year, as step 10) for every variable; reproduction on the existing ones
    lo, hi = pd.Timestamp(PREFORECAST[0]), pd.Timestamp(PREFORECAST[1])
    pre = full[(full.day >= lo) & (full.day <= hi) & ~absent]
    dec = {(p, s): {v: decompose(g, v) for v in VARS} for (p, s), g in pre.groupby(["platform", "series"])}
    nn = full["n"].astype(float).where((full["n"] > 0) & ~absent)
    rel = {}
    for v in VARS:
        sw = pd.Series([dec[(p, s)][v]["sigma2_w"] for p, s in zip(full.platform, full.series)], index=full.index)
        sb = pd.Series([dec[(p, s)][v]["sigma2_b"] for p, s in zip(full.platform, full.series)], index=full.index)
        rel[f"rel_{v}"] = sb / (sb + sw / nn)
    full = pd.concat([full, pd.DataFrame(rel)], axis=1)
    cmp = full.merge(tdp, on=["platform", "series", "day"], suffixes=("", "_td"))
    wr = {v: float((cmp[f"rel_{v}"] - cmp[f"rel_{v}_td"]).abs().max()) for v in EMO + ["polarity"]}
    ok_na = all(bool((cmp[f"rel_{v}"].isna() == cmp[f"rel_{v}_td"].isna()).all()) for v in EMO + ["polarity"])
    check(ok_na and max(wr.values()) < TOL, f"reliabilities R(n) equal step 10's rel_* (max abs diff {max(wr.values()):.2e})", checks)
    counts["d16_valenced"] = {f"{p}|{s}": {v: dec[(p, s)][v] for v in VAL} for (p, s) in dec}

    # 5. precision-weighted means (step 10b) and pooled index (D15) for the means; reproduction on the existing ones
    mean_vars = EMO6 + ["polarity"] + VAL
    pw_cols = {}
    for v in mean_vars:
        pw = pd.Series(np.nan, index=full.index)
        for (s, p), g in full.groupby(["series", "platform"]):
            g = g.sort_values("day")
            raw = g[f"mean_{v}"].astype(float)
            R = g[f"rel_{v}"].astype(float)
            level = raw.shift(1).rolling(LEVEL_DAYS, min_periods=1).mean()
            val = np.where(level.notna(), level + R * (raw - level), raw)
            pw.loc[g.index] = pd.Series(val, index=g.index).where(raw.notna())
        pw_cols[f"pw_{v}"] = pw
    full = pd.concat([full, pd.DataFrame(pw_cols)], axis=1)

    # 6. dispersion measures per platform, series and day (n >= 2)
    two = (full["n"] >= 2) & ~absent
    m = full[[f"raw_mean_{x}" for x in EMO]].to_numpy()
    sd = full[[f"sd_{x}" for x in EMO]].to_numpy()
    n = full["n"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gt = 1 - (m ** 2).sum(axis=1)
        et = -(np.where(m > 0, m * np.log(np.where(m > 0, m, 1.0)), 0.0)).sum(axis=1)
        ssd = (sd ** 2).sum(axis=1)
        gb_pop = gt - full["raw_mean_gw"].to_numpy()
        ident = np.abs(gb_pop - (n - 1) / n * ssd)
    disp = pd.DataFrame({
        "gini_total": gt, "gini_within": full["raw_mean_gw"].to_numpy(), "gini_between": np.sqrt(ssd),
        "gini_between_pop": gb_pop, "ent_total": et, "ent_within": full["raw_mean_hw"].to_numpy(),
        "ent_between": et - full["raw_mean_hw"].to_numpy()}, index=full.index)
    disp.loc[~two.to_numpy(), :] = np.nan
    check(float(np.nanmax(ident[two.to_numpy()])) < TOL,
          f"Gini identity total - within = ((n-1)/n) * sum of the seven variances (max abs diff {float(np.nanmax(ident[two.to_numpy()])):.2e})", checks)
    check(float(disp["ent_between"].min()) > -TOL and float(disp["gini_between_pop"].min()) > -TOL,
          f"between-post parts are not negative (min entropy part {float(disp['ent_between'].min()):.2e})", checks)
    disp["ent_between"] = disp["ent_between"].clip(lower=0)
    disp["gini_between_pop"] = disp["gini_between_pop"].clip(lower=0)
    DISP = list(disp.columns)
    full = pd.concat([full, disp.add_prefix("disp_")], axis=1)
    for k in EMO6:
        gap = (full[f"raw_mean_{k}"] - full[f"raw_mean_{k}_pos"] - full[f"raw_mean_{k}_neg"]).min()
        check(float(gap) > -TOL, f"{k}: positive and negative parts do not exceed the emotion mean", checks)

    # 7. pooled rows: equal-weighted mean of the available platforms
    pk = ["series", "day"]
    agg = ({f"pw_{v}": "mean" for v in mean_vars} | {f"mean_{v}": "mean" for v in mean_vars}
           | {f"disp_{c}": "mean" for c in DISP} | {"n": "sum"})
    pooled = full.groupby(pk).agg(agg).reset_index()
    cnt = full.groupby(pk).agg(n_platforms_val=("mean_surprise_pos", "count"),
                               n_platforms_disp=("disp_gini_between", "count")).reset_index()
    pooled = pooled.merge(cnt, on=pk)
    pooled["platform"] = "POOLED"
    tp = pd.read_parquet(PANEL / "text_precision_daily.parquet")
    tp["day"] = pd.to_datetime(tp["day"].astype(str))
    c2 = pooled.merge(tp, on=pk, how="left")
    wp = {v: float((c2[f"pw_{v}"] - c2[f"mean_{v}_pw"]).abs().max()) for v in EMO6 + ["polarity"]}
    ok_na = all(bool((c2[f"pw_{v}"].isna() == c2[f"mean_{v}_pw"].isna()).all()) for v in EMO6 + ["polarity"])
    check(ok_na and max(wp.values()) < 1e-10,
          f"precision-weighted pooled means of the six emotions and polarity equal step 10b (max abs diff {max(wp.values()):.2e})", checks)
    counts["max_abs_diff_vs_step10b"] = wp
    # the pooled SD of step 10 from the same code path (mean of the platform SDs)
    psd = full.groupby(pk)[[f"sd_{x}" for x in EMO6 + ["polarity"]]].mean().reset_index()
    tdo = td[td.platform == "POOLED"]
    c3 = psd.merge(tdo, on=pk, suffixes=("", "_td"))
    ws = max(float((c3[f"sd_{x}"] - c3[f"sd_{x}_td"]).abs().max()) for x in EMO6 + ["polarity"])
    check(len(c3) == len(psd) and ws < TOL, f"pooled SDs equal step 10's POOLED rows (max abs diff {ws:.2e})", checks)

    # 8. output
    full["n_platforms_val"] = full["mean_surprise_pos"].notna().astype(int)
    full["n_platforms_disp"] = full["disp_gini_between"].notna().astype(int)
    keep = (["series", "platform", "day", "n", "n_platforms_val", "n_platforms_disp"] + [f"disp_{c}" for c in DISP]
            + [f"pw_{v}" for v in VAL] + [f"mean_{v}" for v in VAL])
    out = pd.concat([full[keep], pooled[keep]], ignore_index=True).sort_values(["series", "platform", "day"]).reset_index(drop=True)
    out = out.rename(columns={f"pw_{v}": f"emo_{v}" for v in VAL} | {f"mean_{v}": f"emo_{v}_raw" for v in VAL})
    out["n"] = out["n"].astype("int64")
    out["day"] = out["day"].dt.date
    exp = len(SERIES) * (len(PLATFORMS) + 1) * config.N_DAYS
    check(len(out) == exp, f"complete grid: {exp:,} rows", checks)
    po = out[out.platform == "POOLED"]
    counts["pooled_missing_days"] = {c: int(po[c].isna().sum()) for c in out.columns if c.startswith(("disp_", "emo_"))}
    counts["pooled_summary"] = {c: {"mean": float(po[c].mean()), "sd": float(po[c].std()), "min": float(po[c].min()),
                                    "max": float(po[c].max())} for c in out.columns if c.startswith(("disp_", "emo_")) and not c.endswith("_raw")}
    counts["rows"] = int(len(out))
    counts["posts_series_main"] = {p: int(full.loc[full.platform == p, "n"].sum()) for p in PLATFORMS}

    tmp = runner.tmp_path(OUT)
    out.to_parquet(tmp, index=False, compression="zstd")
    back = pd.read_parquet(tmp)
    check(len(back) == len(out) and list(back.columns) == list(out.columns), "written table reads back with all rows and columns", checks)
    runner.commit(tmp, OUT)
    runner.write_stats(OUT, step=STEP, inputs=inputs, code_sha=CODE_SHA, rules_sha=RULES_SHA, con=con, counts=counts,
                       checks=checks, extra={"seconds": round(time.time() - t0, 1)})
    con.close()
    log(f"{OUT}: {len(out):,} rows, {len(checks)} checks passed, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    log(f"{STEP}: code {CODE_SHA}")
    raise SystemExit(run())
